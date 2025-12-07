//! # Incremental DiskANN Index
//!
//! This module provides `IncrementalDiskANN`, a wrapper around `DiskANN` that supports:
//! - **Adding vectors** without rebuilding the entire index
//! - **Deleting vectors** via tombstones (lazy deletion)
//! - **Compaction** to merge deltas and remove tombstones
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                  IncrementalDiskANN                     │
//! ├─────────────────────────────────────────────────────────┤
//! │  ┌─────────────────┐  ┌─────────────────────────────┐   │
//! │  │   Base Index    │  │        Delta Layer          │   │
//! │  │   (DiskANN)     │  │   (in-memory vectors +      │   │
//! │  │   - immutable   │  │    small Vamana graph)      │   │
//! │  │   - mmap'd      │  │   - mutable                 │   │
//! │  └─────────────────┘  └─────────────────────────────┘   │
//! │                                                         │
//! │  ┌─────────────────────────────────────────────────┐    │
//! │  │              Tombstone Set                      │    │
//! │  │   (deleted IDs from base, excluded at search)   │    │
//! │  └─────────────────────────────────────────────────┘    │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```no_run
//! use anndists::dist::DistL2;
//! use diskann_rs::{IncrementalDiskANN, DiskAnnParams};
//!
//! // Build initial index
//! let vectors = vec![vec![0.0; 128]; 1000];
//! let mut index = IncrementalDiskANN::<DistL2>::build_default(&vectors, "index.db").unwrap();
//!
//! // Add new vectors incrementally
//! let new_vectors = vec![vec![1.0; 128]; 100];
//! let new_ids = index.add_vectors(&new_vectors).unwrap();
//!
//! // Delete vectors (lazy - marks as tombstone)
//! index.delete_vectors(&[0, 5, 10]).unwrap();
//!
//! // Search (automatically excludes tombstones, includes delta)
//! let results = index.search(&vec![0.5; 128], 10, 64);
//!
//! // Compact when delta gets large (rebuilds everything)
//! if index.should_compact() {
//!     index.compact("index_v2.db").unwrap();
//! }
//! ```

use crate::{DiskANN, DiskAnnError, DiskAnnParams};
use anndists::prelude::Distance;
use rayon::prelude::*;
use std::collections::{BinaryHeap, HashSet};
use std::cmp::{Ordering, Reverse};
use std::sync::RwLock;

/// Configuration for the incremental index behavior
#[derive(Clone, Copy, Debug)]
pub struct IncrementalConfig {
    /// Maximum vectors in delta before suggesting compaction
    pub delta_threshold: usize,
    /// Maximum tombstone ratio before suggesting compaction (0.0-1.0)
    pub tombstone_ratio_threshold: f32,
    /// Parameters for delta graph construction
    pub delta_params: DiskAnnParams,
}

impl Default for IncrementalConfig {
    fn default() -> Self {
        Self {
            delta_threshold: 10_000,
            tombstone_ratio_threshold: 0.1,
            delta_params: DiskAnnParams {
                max_degree: 32,        // Smaller for delta
                build_beam_width: 64,
                alpha: 1.2,
            },
        }
    }
}

/// Internal candidate for search merging
#[derive(Clone, Copy)]
struct Candidate {
    dist: f32,
    id: u64,  // Global ID (base or delta)
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.id == other.id
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Delta layer: in-memory vectors with a small navigation graph
struct DeltaLayer {
    /// Vectors added after base index was built
    vectors: Vec<Vec<f32>>,
    /// Small Vamana-style adjacency graph for delta vectors
    /// graph[i] contains neighbor indices (local to delta)
    graph: Vec<Vec<u32>>,
    /// Entry point for delta searches (local index)
    entry_point: Option<u32>,
    /// Max degree for delta graph
    max_degree: usize,
}

impl DeltaLayer {
    fn new(max_degree: usize) -> Self {
        Self {
            vectors: Vec::new(),
            graph: Vec::new(),
            entry_point: None,
            max_degree,
        }
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Add vectors to the delta layer and update the graph
    fn add_vectors<D: Distance<f32> + Copy + Sync>(
        &mut self,
        vectors: &[Vec<f32>],
        dist: D,
    ) -> Vec<u64> {
        let start_idx = self.vectors.len();
        let mut new_ids = Vec::with_capacity(vectors.len());

        for (i, v) in vectors.iter().enumerate() {
            let local_idx = start_idx + i;
            // Global ID: base_offset + local_idx (we use u64::MAX/2 as delta offset)
            let global_id = DELTA_ID_OFFSET + local_idx as u64;
            new_ids.push(global_id);

            self.vectors.push(v.clone());
            self.graph.push(Vec::new());

            // Connect to existing delta vectors using greedy search + prune
            if local_idx > 0 {
                let neighbors = self.find_and_prune_neighbors(local_idx, dist);
                self.graph[local_idx] = neighbors.clone();

                // Reverse edges (make graph bidirectional-ish)
                for &nb in &neighbors {
                    let nb_idx = nb as usize;
                    if !self.graph[nb_idx].contains(&(local_idx as u32))
                        && self.graph[nb_idx].len() < self.max_degree
                    {
                        self.graph[nb_idx].push(local_idx as u32);
                    }
                }
            }

            // Update entry point to be closest to centroid (simplified: just use first)
            if self.entry_point.is_none() {
                self.entry_point = Some(0);
            }
        }

        // Recompute entry point as approximate medoid
        if self.vectors.len() > 1 {
            self.entry_point = Some(self.compute_medoid(dist));
        }

        new_ids
    }

    fn compute_medoid<D: Distance<f32> + Copy + Sync>(&self, dist: D) -> u32 {
        if self.vectors.is_empty() {
            return 0;
        }

        // Compute centroid
        let dim = self.vectors[0].len();
        let mut centroid = vec![0.0f32; dim];
        for v in &self.vectors {
            for (i, &val) in v.iter().enumerate() {
                centroid[i] += val;
            }
        }
        for val in &mut centroid {
            *val /= self.vectors.len() as f32;
        }

        // Find closest to centroid
        let (best_idx, _) = self.vectors
            .iter()
            .enumerate()
            .map(|(idx, v)| (idx, dist.eval(&centroid, v)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap_or((0, f32::MAX));

        best_idx as u32
    }

    fn find_and_prune_neighbors<D: Distance<f32> + Copy>(
        &self,
        node_idx: usize,
        dist: D,
    ) -> Vec<u32> {
        let query = &self.vectors[node_idx];
        let beam_width = (self.max_degree * 2).max(16);

        // Greedy search from entry point
        let candidates = if let Some(entry) = self.entry_point {
            self.greedy_search_internal(query, entry as usize, beam_width, dist)
        } else {
            // No entry point yet, just compute distances to all
            self.vectors.iter()
                .enumerate()
                .filter(|(i, _)| *i != node_idx)
                .map(|(i, v)| (i as u32, dist.eval(query, v)))
                .collect()
        };

        // Alpha-prune
        self.prune_neighbors(node_idx, &candidates, dist)
    }

    fn greedy_search_internal<D: Distance<f32> + Copy>(
        &self,
        query: &[f32],
        start: usize,
        beam_width: usize,
        dist: D,
    ) -> Vec<(u32, f32)> {
        if self.vectors.is_empty() || start >= self.vectors.len() {
            return Vec::new();
        }

        let mut visited = HashSet::new();
        let mut frontier: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();
        let mut results: BinaryHeap<Candidate> = BinaryHeap::new();

        let start_dist = dist.eval(query, &self.vectors[start]);
        let start_cand = Candidate { dist: start_dist, id: start as u64 };
        frontier.push(Reverse(start_cand));
        results.push(start_cand);
        visited.insert(start);

        while let Some(Reverse(best)) = frontier.peek().copied() {
            if results.len() >= beam_width {
                if let Some(worst) = results.peek() {
                    if best.dist >= worst.dist {
                        break;
                    }
                }
            }
            let Reverse(current) = frontier.pop().unwrap();
            let cur_idx = current.id as usize;

            if cur_idx < self.graph.len() {
                for &nb in &self.graph[cur_idx] {
                    let nb_idx = nb as usize;
                    if !visited.insert(nb_idx) {
                        continue;
                    }
                    if nb_idx >= self.vectors.len() {
                        continue;
                    }

                    let d = dist.eval(query, &self.vectors[nb_idx]);
                    let cand = Candidate { dist: d, id: nb as u64 };

                    if results.len() < beam_width {
                        results.push(cand);
                        frontier.push(Reverse(cand));
                    } else if d < results.peek().unwrap().dist {
                        results.pop();
                        results.push(cand);
                        frontier.push(Reverse(cand));
                    }
                }
            }
        }

        results.into_vec()
            .into_iter()
            .map(|c| (c.id as u32, c.dist))
            .collect()
    }

    fn prune_neighbors<D: Distance<f32> + Copy>(
        &self,
        node_idx: usize,
        candidates: &[(u32, f32)],
        dist: D,
    ) -> Vec<u32> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let alpha = 1.2f32;
        let mut sorted = candidates.to_vec();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut pruned = Vec::new();

        for &(cand_id, cand_dist) in &sorted {
            if cand_id as usize == node_idx {
                continue;
            }

            let mut ok = true;
            for &sel in &pruned {
                let d = dist.eval(
                    &self.vectors[cand_id as usize],
                    &self.vectors[sel as usize],
                );
                if d < alpha * cand_dist {
                    ok = false;
                    break;
                }
            }

            if ok {
                pruned.push(cand_id);
                if pruned.len() >= self.max_degree {
                    break;
                }
            }
        }

        pruned
    }

    fn search<D: Distance<f32> + Copy>(
        &self,
        query: &[f32],
        k: usize,
        beam_width: usize,
        dist: D,
    ) -> Vec<(u64, f32)> {
        if self.vectors.is_empty() {
            return Vec::new();
        }

        let entry = self.entry_point.unwrap_or(0) as usize;
        let mut results = self.greedy_search_internal(query, entry, beam_width, dist);
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);

        // Convert local IDs to global delta IDs
        results.into_iter()
            .map(|(local_id, d)| (DELTA_ID_OFFSET + local_id as u64, d))
            .collect()
    }

    fn get_vector(&self, local_idx: usize) -> Option<&Vec<f32>> {
        self.vectors.get(local_idx)
    }
}

/// Offset for delta vector IDs to distinguish from base IDs
const DELTA_ID_OFFSET: u64 = 1u64 << 48;

/// Check if an ID is from the delta layer
#[inline]
pub fn is_delta_id(id: u64) -> bool {
    id >= DELTA_ID_OFFSET
}

/// Convert a delta global ID to local delta index
#[inline]
pub fn delta_local_idx(id: u64) -> usize {
    (id - DELTA_ID_OFFSET) as usize
}

/// An incremental DiskANN index supporting add/delete without full rebuild
pub struct IncrementalDiskANN<D>
where
    D: Distance<f32> + Send + Sync + Copy + Clone + 'static,
{
    /// The base immutable index (memory-mapped)
    base: Option<DiskANN<D>>,
    /// Delta layer for newly added vectors
    delta: RwLock<DeltaLayer>,
    /// Set of deleted vector IDs (tombstones)
    tombstones: RwLock<HashSet<u64>>,
    /// Distance metric
    dist: D,
    /// Configuration
    config: IncrementalConfig,
    /// Path to the base index file
    base_path: Option<String>,
    /// Dimensionality
    dim: usize,
}

impl<D> IncrementalDiskANN<D>
where
    D: Distance<f32> + Send + Sync + Copy + Clone + Default + 'static,
{
    /// Build a new incremental index with default parameters
    pub fn build_default(
        vectors: &[Vec<f32>],
        file_path: &str,
    ) -> Result<Self, DiskAnnError> {
        Self::build_with_config(vectors, file_path, IncrementalConfig::default())
    }

    /// Open an existing index for incremental updates
    pub fn open(path: &str) -> Result<Self, DiskAnnError> {
        Self::open_with_config(path, IncrementalConfig::default())
    }
}

impl<D> IncrementalDiskANN<D>
where
    D: Distance<f32> + Send + Sync + Copy + Clone + 'static,
{
    /// Build a new incremental index with custom configuration
    pub fn build_with_config(
        vectors: &[Vec<f32>],
        file_path: &str,
        config: IncrementalConfig,
    ) -> Result<Self, DiskAnnError>
    where
        D: Default,
    {
        let dist = D::default();
        let dim = vectors.first().map(|v| v.len()).unwrap_or(0);

        let base = DiskANN::<D>::build_index_default(vectors, dist, file_path)?;

        Ok(Self {
            base: Some(base),
            delta: RwLock::new(DeltaLayer::new(config.delta_params.max_degree)),
            tombstones: RwLock::new(HashSet::new()),
            dist,
            config,
            base_path: Some(file_path.to_string()),
            dim,
        })
    }

    /// Open an existing index with custom configuration
    pub fn open_with_config(path: &str, config: IncrementalConfig) -> Result<Self, DiskAnnError>
    where
        D: Default,
    {
        let dist = D::default();
        let base = DiskANN::<D>::open_index_default_metric(path)?;
        let dim = base.dim;

        Ok(Self {
            base: Some(base),
            delta: RwLock::new(DeltaLayer::new(config.delta_params.max_degree)),
            tombstones: RwLock::new(HashSet::new()),
            dist,
            config,
            base_path: Some(path.to_string()),
            dim,
        })
    }

    /// Create an empty incremental index (no base, delta-only)
    pub fn new_empty(dim: usize, dist: D, config: IncrementalConfig) -> Self {
        Self {
            base: None,
            delta: RwLock::new(DeltaLayer::new(config.delta_params.max_degree)),
            tombstones: RwLock::new(HashSet::new()),
            dist,
            config,
            base_path: None,
            dim,
        }
    }

    /// Add new vectors to the index. Returns their assigned IDs.
    pub fn add_vectors(&self, vectors: &[Vec<f32>]) -> Result<Vec<u64>, DiskAnnError> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        // Validate dimensions
        for (i, v) in vectors.iter().enumerate() {
            if v.len() != self.dim {
                return Err(DiskAnnError::IndexError(format!(
                    "Vector {} has dimension {} but index expects {}",
                    i, v.len(), self.dim
                )));
            }
        }

        let mut delta = self.delta.write().unwrap();
        let ids = delta.add_vectors(vectors, self.dist);
        Ok(ids)
    }

    /// Delete vectors by their IDs (lazy deletion via tombstones)
    pub fn delete_vectors(&self, ids: &[u64]) -> Result<(), DiskAnnError> {
        let mut tombstones = self.tombstones.write().unwrap();
        for &id in ids {
            tombstones.insert(id);
        }
        Ok(())
    }

    /// Check if a vector ID has been deleted
    pub fn is_deleted(&self, id: u64) -> bool {
        self.tombstones.read().unwrap().contains(&id)
    }

    /// Search the index, merging results from base and delta, excluding tombstones
    pub fn search(&self, query: &[f32], k: usize, beam_width: usize) -> Vec<u64> {
        self.search_with_dists(query, k, beam_width)
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    /// Search returning (id, distance) pairs
    pub fn search_with_dists(&self, query: &[f32], k: usize, beam_width: usize) -> Vec<(u64, f32)> {
        let tombstones = self.tombstones.read().unwrap();
        let delta = self.delta.read().unwrap();

        // Collect candidates from both layers
        let mut all_candidates: Vec<(u64, f32)> = Vec::with_capacity(k * 2);

        // Search base index
        if let Some(ref base) = self.base {
            let base_results = base.search_with_dists(query, k + tombstones.len(), beam_width);
            for (id, dist) in base_results {
                let global_id = id as u64;
                if !tombstones.contains(&global_id) {
                    all_candidates.push((global_id, dist));
                }
            }
        }

        // Search delta layer
        if !delta.is_empty() {
            let delta_results = delta.search(query, k, beam_width, self.dist);
            for (id, dist) in delta_results {
                if !tombstones.contains(&id) {
                    all_candidates.push((id, dist));
                }
            }
        }

        // Merge and return top-k
        all_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        all_candidates.truncate(k);
        all_candidates
    }

    /// Parallel batch search
    pub fn search_batch(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        beam_width: usize,
    ) -> Vec<Vec<u64>> {
        queries
            .par_iter()
            .map(|q| self.search(q, k, beam_width))
            .collect()
    }

    /// Get a vector by its ID (works for both base and delta)
    pub fn get_vector(&self, id: u64) -> Option<Vec<f32>> {
        if is_delta_id(id) {
            let delta = self.delta.read().unwrap();
            delta.get_vector(delta_local_idx(id)).cloned()
        } else if let Some(ref base) = self.base {
            let idx = id as usize;
            if idx < base.num_vectors {
                Some(base.get_vector(idx))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Check if compaction is recommended
    pub fn should_compact(&self) -> bool {
        let delta = self.delta.read().unwrap();
        let tombstones = self.tombstones.read().unwrap();

        let base_size = self.base.as_ref().map(|b| b.num_vectors).unwrap_or(0);
        let total_size = base_size + delta.len();

        // Check delta threshold
        if delta.len() >= self.config.delta_threshold {
            return true;
        }

        // Check tombstone ratio
        if total_size > 0 {
            let tombstone_ratio = tombstones.len() as f32 / total_size as f32;
            if tombstone_ratio >= self.config.tombstone_ratio_threshold {
                return true;
            }
        }

        false
    }

    /// Compact the index: merge base + delta, remove tombstones, write new file
    pub fn compact(&mut self, new_path: &str) -> Result<(), DiskAnnError>
    where
        D: Default,
    {
        let tombstones = self.tombstones.read().unwrap().clone();
        let delta = self.delta.read().unwrap();

        // Collect all live vectors
        let mut all_vectors: Vec<Vec<f32>> = Vec::new();

        // Add base vectors (excluding tombstones)
        if let Some(ref base) = self.base {
            for i in 0..base.num_vectors {
                if !tombstones.contains(&(i as u64)) {
                    all_vectors.push(base.get_vector(i));
                }
            }
        }

        // Add delta vectors (excluding tombstones)
        for (i, v) in delta.vectors.iter().enumerate() {
            let global_id = DELTA_ID_OFFSET + i as u64;
            if !tombstones.contains(&global_id) {
                all_vectors.push(v.clone());
            }
        }

        drop(delta);
        drop(tombstones);

        if all_vectors.is_empty() {
            return Err(DiskAnnError::IndexError(
                "Cannot compact: no vectors remaining after removing tombstones".to_string()
            ));
        }

        // Build new index
        let new_base = DiskANN::<D>::build_index_default(&all_vectors, self.dist, new_path)?;

        // Replace state
        self.base = Some(new_base);
        self.delta = RwLock::new(DeltaLayer::new(self.config.delta_params.max_degree));
        self.tombstones = RwLock::new(HashSet::new());
        self.base_path = Some(new_path.to_string());

        Ok(())
    }

    /// Get statistics about the index
    pub fn stats(&self) -> IncrementalStats {
        let delta = self.delta.read().unwrap();
        let tombstones = self.tombstones.read().unwrap();
        let base_count = self.base.as_ref().map(|b| b.num_vectors).unwrap_or(0);

        IncrementalStats {
            base_vectors: base_count,
            delta_vectors: delta.len(),
            tombstones: tombstones.len(),
            total_live: base_count + delta.len() - tombstones.len(),
            dim: self.dim,
        }
    }

    /// Get the dimensionality of vectors in this index
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Statistics about an incremental index
#[derive(Debug, Clone)]
pub struct IncrementalStats {
    pub base_vectors: usize,
    pub delta_vectors: usize,
    pub tombstones: usize,
    pub total_live: usize,
    pub dim: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use anndists::dist::DistL2;
    use std::fs;

    fn euclid(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
    }

    #[test]
    fn test_incremental_basic() {
        let path = "test_incremental_basic.db";
        let _ = fs::remove_file(path);

        // Build initial index
        let vectors = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];

        let index = IncrementalDiskANN::<DistL2>::build_default(&vectors, path).unwrap();

        // Search should work
        let results = index.search(&[0.1, 0.1], 2, 8);
        assert_eq!(results.len(), 2);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_incremental_add() {
        let path = "test_incremental_add.db";
        let _ = fs::remove_file(path);

        let vectors = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
        ];

        let index = IncrementalDiskANN::<DistL2>::build_default(&vectors, path).unwrap();

        // Add new vectors
        let new_vecs = vec![vec![0.5, 0.5], vec![2.0, 2.0]];
        let new_ids = index.add_vectors(&new_vecs).unwrap();
        assert_eq!(new_ids.len(), 2);
        assert!(is_delta_id(new_ids[0]));

        // Search should find the new vector
        let results = index.search_with_dists(&[0.5, 0.5], 1, 8);
        assert!(!results.is_empty());

        // The closest should be the one we just added at [0.5, 0.5]
        let (_best_id, best_dist) = results[0];
        assert!(best_dist < 0.01, "Expected to find [0.5, 0.5], got dist {}", best_dist);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_incremental_delete() {
        let path = "test_incremental_delete.db";
        let _ = fs::remove_file(path);

        let vectors = vec![
            vec![0.0, 0.0],  // id 0
            vec![1.0, 0.0],  // id 1
            vec![0.0, 1.0],  // id 2
        ];

        let index = IncrementalDiskANN::<DistL2>::build_default(&vectors, path).unwrap();

        // Delete vector 0
        index.delete_vectors(&[0]).unwrap();
        assert!(index.is_deleted(0));

        // Search near [0,0] should not return id 0
        let results = index.search(&[0.0, 0.0], 3, 8);
        assert!(!results.contains(&0), "Deleted vector should not appear in results");

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_incremental_compact() {
        let path1 = "test_compact_v1.db";
        let path2 = "test_compact_v2.db";
        let _ = fs::remove_file(path1);
        let _ = fs::remove_file(path2);

        let vectors = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];

        let mut index = IncrementalDiskANN::<DistL2>::build_default(&vectors, path1).unwrap();

        // Add some vectors
        index.add_vectors(&[vec![2.0, 2.0], vec![3.0, 3.0]]).unwrap();

        // Delete some
        index.delete_vectors(&[0, 1]).unwrap();

        let stats_before = index.stats();
        assert_eq!(stats_before.base_vectors, 4);
        assert_eq!(stats_before.delta_vectors, 2);
        assert_eq!(stats_before.tombstones, 2);

        // Compact
        index.compact(path2).unwrap();

        let stats_after = index.stats();
        assert_eq!(stats_after.base_vectors, 4); // 4 base - 2 deleted + 2 delta = 4
        assert_eq!(stats_after.delta_vectors, 0);
        assert_eq!(stats_after.tombstones, 0);

        // Search should still work
        let results = index.search(&[2.0, 2.0], 1, 8);
        assert!(!results.is_empty());

        let _ = fs::remove_file(path1);
        let _ = fs::remove_file(path2);
    }

    #[test]
    fn test_delta_only() {
        // Test index with no base, only delta
        let config = IncrementalConfig::default();
        let index = IncrementalDiskANN::<DistL2>::new_empty(2, DistL2 {}, config);

        // Add vectors
        let vecs = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
        ];
        index.add_vectors(&vecs).unwrap();

        // Search
        let results = index.search_with_dists(&[0.5, 0.5], 3, 8);
        assert_eq!(results.len(), 3);

        // Closest should be [0.5, 0.5]
        let best_vec = index.get_vector(results[0].0).unwrap();
        let dist = euclid(&best_vec, &[0.5, 0.5]);
        assert!(dist < 0.01);
    }
}
