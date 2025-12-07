//! # Filtered DiskANN Search
//!
//! Enables searching with metadata predicates, e.g.:
//! "Find 10 nearest neighbors WHERE category = 'electronics' AND price < 100"
//!
//! ## Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────┐
//! │                    FilteredDiskANN                         │
//! ├────────────────────────────────────────────────────────────┤
//! │  ┌──────────────────┐    ┌─────────────────────────────┐   │
//! │  │   DiskANN Index  │    │     Metadata Store          │   │
//! │  │   (vectors +     │    │   (labels per vector)       │   │
//! │  │    graph)        │    │   - numeric fields          │   │
//! │  │                  │    │   - string labels           │   │
//! │  └──────────────────┘    └─────────────────────────────┘   │
//! └────────────────────────────────────────────────────────────┘
//!
//! Search: expand beam, but skip candidates that don't match filter
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! use anndists::dist::DistL2;
//! use diskann_rs::{FilteredDiskANN, Filter};
//!
//! // Build index with metadata
//! let vectors = vec![vec![0.0; 128]; 1000];
//! let labels: Vec<Vec<u64>> = (0..1000).map(|i| vec![i % 10]).collect(); // 10 categories
//!
//! let index = FilteredDiskANN::<DistL2>::build(
//!     &vectors,
//!     &labels,
//!     "filtered.db"
//! ).unwrap();
//!
//! // Search with filter: only category 5
//! let query = vec![0.0f32; 128];
//! let filter = Filter::label_eq(0, 5); // field 0 == 5
//! let results = index.search_filtered(&query, 10, 128, &filter);
//! ```

use crate::{DiskANN, DiskAnnError, DiskAnnParams};
use anndists::prelude::Distance;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashSet};
use std::cmp::{Ordering, Reverse};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};

/// A single filter condition
#[derive(Clone, Debug)]
pub enum Filter {
    /// Label at field index equals value
    LabelEq { field: usize, value: u64 },
    /// Label at field index is in set
    LabelIn { field: usize, values: HashSet<u64> },
    /// Label at field index less than value
    LabelLt { field: usize, value: u64 },
    /// Label at field index greater than value
    LabelGt { field: usize, value: u64 },
    /// Label at field index in range [min, max]
    LabelRange { field: usize, min: u64, max: u64 },
    /// Logical AND of filters
    And(Vec<Filter>),
    /// Logical OR of filters
    Or(Vec<Filter>),
    /// No filter (match all)
    None,
}

impl Filter {
    /// Create equality filter
    pub fn label_eq(field: usize, value: u64) -> Self {
        Filter::LabelEq { field, value }
    }

    /// Create "in set" filter
    pub fn label_in(field: usize, values: impl IntoIterator<Item = u64>) -> Self {
        Filter::LabelIn {
            field,
            values: values.into_iter().collect(),
        }
    }

    /// Create less-than filter
    pub fn label_lt(field: usize, value: u64) -> Self {
        Filter::LabelLt { field, value }
    }

    /// Create greater-than filter
    pub fn label_gt(field: usize, value: u64) -> Self {
        Filter::LabelGt { field, value }
    }

    /// Create range filter [min, max] inclusive
    pub fn label_range(field: usize, min: u64, max: u64) -> Self {
        Filter::LabelRange { field, min, max }
    }

    /// Combine filters with AND
    pub fn and(filters: Vec<Filter>) -> Self {
        Filter::And(filters)
    }

    /// Combine filters with OR
    pub fn or(filters: Vec<Filter>) -> Self {
        Filter::Or(filters)
    }

    /// Check if a label vector matches this filter
    pub fn matches(&self, labels: &[u64]) -> bool {
        match self {
            Filter::None => true,
            Filter::LabelEq { field, value } => {
                labels.get(*field).map_or(false, |v| v == value)
            }
            Filter::LabelIn { field, values } => {
                labels.get(*field).map_or(false, |v| values.contains(v))
            }
            Filter::LabelLt { field, value } => {
                labels.get(*field).map_or(false, |v| v < value)
            }
            Filter::LabelGt { field, value } => {
                labels.get(*field).map_or(false, |v| v > value)
            }
            Filter::LabelRange { field, min, max } => {
                labels.get(*field).map_or(false, |v| v >= min && v <= max)
            }
            Filter::And(filters) => filters.iter().all(|f| f.matches(labels)),
            Filter::Or(filters) => filters.iter().any(|f| f.matches(labels)),
        }
    }
}

/// Metadata for filtered index
#[derive(Serialize, Deserialize, Debug)]
struct FilteredMetadata {
    num_vectors: usize,
    num_fields: usize,
}

/// Candidate for filtered search
#[derive(Clone, Copy)]
struct Candidate {
    dist: f32,
    id: u32,
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

/// DiskANN index with metadata filtering support
pub struct FilteredDiskANN<D>
where
    D: Distance<f32> + Send + Sync + Copy + Clone + 'static,
{
    /// The underlying vector index
    index: DiskANN<D>,
    /// Labels for each vector: labels[vector_id] = [field0, field1, ...]
    labels: Vec<Vec<u64>>,
    /// Number of label fields per vector
    num_fields: usize,
    /// Path to labels file (kept for potential future persistence)
    #[allow(dead_code)]
    labels_path: String,
}

impl<D> FilteredDiskANN<D>
where
    D: Distance<f32> + Send + Sync + Copy + Clone + Default + 'static,
{
    /// Build a new filtered index
    pub fn build(
        vectors: &[Vec<f32>],
        labels: &[Vec<u64>],
        base_path: &str,
    ) -> Result<Self, DiskAnnError> {
        Self::build_with_params(vectors, labels, base_path, DiskAnnParams::default())
    }

    /// Build with custom parameters
    pub fn build_with_params(
        vectors: &[Vec<f32>],
        labels: &[Vec<u64>],
        base_path: &str,
        params: DiskAnnParams,
    ) -> Result<Self, DiskAnnError> {
        if vectors.len() != labels.len() {
            return Err(DiskAnnError::IndexError(format!(
                "vectors.len() ({}) != labels.len() ({})",
                vectors.len(),
                labels.len()
            )));
        }

        let num_fields = labels.first().map(|l| l.len()).unwrap_or(0);
        for (i, l) in labels.iter().enumerate() {
            if l.len() != num_fields {
                return Err(DiskAnnError::IndexError(format!(
                    "Label {} has {} fields, expected {}",
                    i,
                    l.len(),
                    num_fields
                )));
            }
        }

        // Build the vector index
        let index_path = format!("{}.idx", base_path);
        let index = DiskANN::<D>::build_index_with_params(
            vectors,
            D::default(),
            &index_path,
            params,
        )?;

        // Save labels
        let labels_path = format!("{}.labels", base_path);
        Self::save_labels(&labels_path, labels, num_fields)?;

        Ok(Self {
            index,
            labels: labels.to_vec(),
            num_fields,
            labels_path,
        })
    }

    /// Open an existing filtered index
    pub fn open(base_path: &str) -> Result<Self, DiskAnnError> {
        let index_path = format!("{}.idx", base_path);
        let labels_path = format!("{}.labels", base_path);

        let index = DiskANN::<D>::open_index_default_metric(&index_path)?;
        let (labels, num_fields) = Self::load_labels(&labels_path)?;

        if labels.len() != index.num_vectors {
            return Err(DiskAnnError::IndexError(format!(
                "Labels count ({}) != index vectors ({})",
                labels.len(),
                index.num_vectors
            )));
        }

        Ok(Self {
            index,
            labels,
            num_fields,
            labels_path,
        })
    }

    fn save_labels(path: &str, labels: &[Vec<u64>], num_fields: usize) -> Result<(), DiskAnnError> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        let mut writer = BufWriter::new(file);

        let meta = FilteredMetadata {
            num_vectors: labels.len(),
            num_fields,
        };
        let meta_bytes = bincode::serialize(&meta)?;
        writer.write_all(&(meta_bytes.len() as u64).to_le_bytes())?;
        writer.write_all(&meta_bytes)?;

        // Write labels as flat array
        for label_vec in labels {
            for &val in label_vec {
                writer.write_all(&val.to_le_bytes())?;
            }
        }

        writer.flush()?;
        Ok(())
    }

    fn load_labels(path: &str) -> Result<(Vec<Vec<u64>>, usize), DiskAnnError> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read metadata
        let mut len_buf = [0u8; 8];
        reader.read_exact(&mut len_buf)?;
        let meta_len = u64::from_le_bytes(len_buf) as usize;

        let mut meta_bytes = vec![0u8; meta_len];
        reader.read_exact(&mut meta_bytes)?;
        let meta: FilteredMetadata = bincode::deserialize(&meta_bytes)?;

        // Read labels
        let mut labels = Vec::with_capacity(meta.num_vectors);
        let mut val_buf = [0u8; 8];

        for _ in 0..meta.num_vectors {
            let mut label_vec = Vec::with_capacity(meta.num_fields);
            for _ in 0..meta.num_fields {
                reader.read_exact(&mut val_buf)?;
                label_vec.push(u64::from_le_bytes(val_buf));
            }
            labels.push(label_vec);
        }

        Ok((labels, meta.num_fields))
    }
}

impl<D> FilteredDiskANN<D>
where
    D: Distance<f32> + Send + Sync + Copy + Clone + 'static,
{
    /// Search with a filter predicate
    ///
    /// Uses an expanded beam search that skips non-matching candidates
    /// but continues exploring the graph to find matches.
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        beam_width: usize,
        filter: &Filter,
    ) -> Vec<u32> {
        self.search_filtered_with_dists(query, k, beam_width, filter)
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    /// Search with filter, returning distances
    pub fn search_filtered_with_dists(
        &self,
        query: &[f32],
        k: usize,
        beam_width: usize,
        filter: &Filter,
    ) -> Vec<(u32, f32)> {
        // For unfiltered search, use the fast path
        if matches!(filter, Filter::None) {
            return self.index.search_with_dists(query, k, beam_width);
        }

        // Filtered search: we need to explore more of the graph
        // Use larger internal beam to find enough matching candidates
        let expanded_beam = (beam_width * 4).max(k * 10);

        let mut visited = HashSet::new();
        let mut frontier: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();
        let mut working_set: BinaryHeap<Candidate> = BinaryHeap::new();
        let mut results: Vec<(u32, f32)> = Vec::with_capacity(k);

        // Seed from medoid
        let start_dist = self.distance_to(query, self.index.medoid_id as usize);
        let start = Candidate {
            dist: start_dist,
            id: self.index.medoid_id,
        };
        frontier.push(Reverse(start));
        working_set.push(start);
        visited.insert(self.index.medoid_id);

        // Check if medoid matches filter
        if filter.matches(&self.labels[self.index.medoid_id as usize]) {
            results.push((self.index.medoid_id, start_dist));
        }

        // Expand until we have k results or exhausted search
        let mut iterations = 0;
        let max_iterations = expanded_beam * 2;

        while let Some(Reverse(best)) = frontier.peek().copied() {
            iterations += 1;
            if iterations > max_iterations {
                break;
            }

            // Early termination if we have enough results and best candidate
            // can't improve our worst result
            if results.len() >= k {
                if let Some((_, worst_dist)) = results.last() {
                    if best.dist > *worst_dist * 1.5 {
                        break;
                    }
                }
            }

            if working_set.len() >= expanded_beam {
                if let Some(worst) = working_set.peek() {
                    if best.dist >= worst.dist {
                        break;
                    }
                }
            }

            let Reverse(current) = frontier.pop().unwrap();

            // Explore neighbors
            for &nb in self.get_neighbors(current.id) {
                if nb == u32::MAX {
                    continue;
                }
                if !visited.insert(nb) {
                    continue;
                }

                let d = self.distance_to(query, nb as usize);
                let cand = Candidate { dist: d, id: nb };

                // Always add to working set for graph exploration
                if working_set.len() < expanded_beam {
                    working_set.push(cand);
                    frontier.push(Reverse(cand));
                } else if d < working_set.peek().unwrap().dist {
                    working_set.pop();
                    working_set.push(cand);
                    frontier.push(Reverse(cand));
                }

                // Check filter for results
                if filter.matches(&self.labels[nb as usize]) {
                    // Insert into results maintaining sorted order
                    let pos = results
                        .iter()
                        .position(|(_, dist)| d < *dist)
                        .unwrap_or(results.len());

                    if pos < k {
                        results.insert(pos, (nb, d));
                        if results.len() > k {
                            results.pop();
                        }
                    }
                }
            }
        }

        results
    }

    /// Parallel batch filtered search
    pub fn search_filtered_batch(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        beam_width: usize,
        filter: &Filter,
    ) -> Vec<Vec<u32>> {
        queries
            .par_iter()
            .map(|q| self.search_filtered(q, k, beam_width, filter))
            .collect()
    }

    /// Unfiltered search (delegates to base index)
    pub fn search(&self, query: &[f32], k: usize, beam_width: usize) -> Vec<u32> {
        self.index.search(query, k, beam_width)
    }

    /// Get labels for a vector
    pub fn get_labels(&self, id: usize) -> Option<&[u64]> {
        self.labels.get(id).map(|v| v.as_slice())
    }

    /// Get the underlying index
    pub fn inner(&self) -> &DiskANN<D> {
        &self.index
    }

    /// Number of vectors in the index
    pub fn num_vectors(&self) -> usize {
        self.index.num_vectors
    }

    /// Number of label fields per vector
    pub fn num_fields(&self) -> usize {
        self.num_fields
    }

    /// Count vectors matching a filter (useful for selectivity estimation)
    pub fn count_matching(&self, filter: &Filter) -> usize {
        self.labels.iter().filter(|l| filter.matches(l)).count()
    }

    fn get_neighbors(&self, node_id: u32) -> &[u32] {
        // Access internal neighbors through the index
        let offset = self.index.adjacency_offset
            + (node_id as u64 * self.index.max_degree as u64 * 4);
        let start = offset as usize;
        let end = start + (self.index.max_degree * 4);
        let bytes = &self.index.mmap[start..end];
        bytemuck::cast_slice(bytes)
    }

    fn distance_to(&self, query: &[f32], idx: usize) -> f32 {
        let offset = self.index.vectors_offset + (idx as u64 * self.index.dim as u64 * 4);
        let start = offset as usize;
        let end = start + (self.index.dim * 4);
        let bytes = &self.index.mmap[start..end];
        let vector: &[f32] = bytemuck::cast_slice(bytes);
        self.index.dist.eval(query, vector)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anndists::dist::DistL2;
    use std::fs;

    #[test]
    fn test_filter_eq() {
        let filter = Filter::label_eq(0, 5);
        assert!(filter.matches(&[5, 10]));
        assert!(!filter.matches(&[4, 10]));
        assert!(!filter.matches(&[]));
    }

    #[test]
    fn test_filter_in() {
        let filter = Filter::label_in(0, vec![1, 3, 5]);
        assert!(filter.matches(&[1]));
        assert!(filter.matches(&[3]));
        assert!(filter.matches(&[5]));
        assert!(!filter.matches(&[2]));
    }

    #[test]
    fn test_filter_range() {
        let filter = Filter::label_range(0, 10, 20);
        assert!(filter.matches(&[10]));
        assert!(filter.matches(&[15]));
        assert!(filter.matches(&[20]));
        assert!(!filter.matches(&[9]));
        assert!(!filter.matches(&[21]));
    }

    #[test]
    fn test_filter_and() {
        let filter = Filter::and(vec![
            Filter::label_eq(0, 5),
            Filter::label_gt(1, 10),
        ]);
        assert!(filter.matches(&[5, 15]));
        assert!(!filter.matches(&[5, 5]));
        assert!(!filter.matches(&[4, 15]));
    }

    #[test]
    fn test_filter_or() {
        let filter = Filter::or(vec![
            Filter::label_eq(0, 5),
            Filter::label_eq(0, 10),
        ]);
        assert!(filter.matches(&[5]));
        assert!(filter.matches(&[10]));
        assert!(!filter.matches(&[7]));
    }

    #[test]
    fn test_filtered_search_basic() {
        let base_path = "test_filtered";
        let _ = fs::remove_file(format!("{}.idx", base_path));
        let _ = fs::remove_file(format!("{}.labels", base_path));

        // Create vectors with categories
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|i| vec![i as f32, (i * 2) as f32])
            .collect();

        // Labels: [category] where category = i % 5
        let labels: Vec<Vec<u64>> = (0..100)
            .map(|i| vec![i % 5])
            .collect();

        let index = FilteredDiskANN::<DistL2>::build(&vectors, &labels, base_path).unwrap();

        // Search without filter
        let results = index.search(&[50.0, 100.0], 5, 32);
        assert_eq!(results.len(), 5);

        // Search with filter: only category 0
        let filter = Filter::label_eq(0, 0);
        let results = index.search_filtered(&[50.0, 100.0], 5, 32, &filter);

        // All results should be category 0
        for id in &results {
            assert_eq!(labels[*id as usize][0], 0);
        }

        let _ = fs::remove_file(format!("{}.idx", base_path));
        let _ = fs::remove_file(format!("{}.labels", base_path));
    }

    #[test]
    fn test_filtered_search_selectivity() {
        let base_path = "test_filtered_sel";
        let _ = fs::remove_file(format!("{}.idx", base_path));
        let _ = fs::remove_file(format!("{}.labels", base_path));

        // 1000 vectors, 10 categories
        let vectors: Vec<Vec<f32>> = (0..1000)
            .map(|i| vec![(i % 100) as f32, ((i / 100) * 10) as f32])
            .collect();

        let labels: Vec<Vec<u64>> = (0..1000)
            .map(|i| vec![i % 10]) // ~100 per category
            .collect();

        let index = FilteredDiskANN::<DistL2>::build(&vectors, &labels, base_path).unwrap();

        // Verify count
        let filter = Filter::label_eq(0, 3);
        assert_eq!(index.count_matching(&filter), 100);

        // Search for category 3
        let results = index.search_filtered(&[50.0, 50.0], 10, 64, &filter);
        assert!(results.len() <= 10);

        for id in &results {
            assert_eq!(labels[*id as usize][0], 3);
        }

        let _ = fs::remove_file(format!("{}.idx", base_path));
        let _ = fs::remove_file(format!("{}.labels", base_path));
    }

    #[test]
    fn test_filtered_persistence() {
        let base_path = "test_filtered_persist";
        let _ = fs::remove_file(format!("{}.idx", base_path));
        let _ = fs::remove_file(format!("{}.labels", base_path));

        let vectors: Vec<Vec<f32>> = (0..50)
            .map(|i| vec![i as f32, i as f32])
            .collect();
        let labels: Vec<Vec<u64>> = (0..50).map(|i| vec![i % 3, i]).collect();

        {
            let _index = FilteredDiskANN::<DistL2>::build(&vectors, &labels, base_path).unwrap();
        }

        // Reopen
        let index = FilteredDiskANN::<DistL2>::open(base_path).unwrap();
        assert_eq!(index.num_vectors(), 50);
        assert_eq!(index.num_fields(), 2);

        let filter = Filter::label_eq(0, 1);
        let results = index.search_filtered(&[25.0, 25.0], 5, 32, &filter);
        for id in &results {
            assert_eq!(index.get_labels(*id as usize).unwrap()[0], 1);
        }

        let _ = fs::remove_file(format!("{}.idx", base_path));
        let _ = fs::remove_file(format!("{}.labels", base_path));
    }
}
