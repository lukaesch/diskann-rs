//! # DiskAnn (generic over `anndists::Distance<f32>`)
//!
//! A minimal, on-disk, DiskANN-like library that:
//! - Builds a Vamana-style graph (greedy + α-pruning) in memory
//! - Writes vectors + fixed-degree adjacency to a single file
//! - Memory-maps the file for low-overhead reads
//! - Is **generic over any `Distance<f32>`** from `anndists` (L2, Cosine, Hamming, Dot, …)
//! - Supports **incremental updates** (add/delete vectors without full rebuild)
//!
//! ## Example
//! ```no_run
//! use anndists::dist::{DistL2, DistCosine};
//! use diskann_rs::{DiskANN, DiskAnnParams};
//!
//! // Build a new index from vectors, using L2 and default params
//! let vectors = vec![vec![0.0; 128]; 1000];
//! let index = DiskANN::<DistL2>::build_index_default(&vectors, DistL2{}, "index.db").unwrap();
//!
//! // Or with custom params
//! let index2 = DiskANN::<DistCosine>::build_index_with_params(
//!     &vectors,
//!     DistCosine{},
//!     "index_cos.db",
//!     DiskAnnParams { max_degree: 48, ..Default::default() },
//! ).unwrap();
//!
//! // Search the index
//! let query = vec![0.0; 128];
//! let neighbors = index.search(&query, 10, 64);
//!
//! // Open later (provide the same distance type)
//! let _reopened = DiskANN::<DistL2>::open_index_default_metric("index.db").unwrap();
//! ```
//!
//! ## Incremental Updates
//! ```no_run
//! use anndists::dist::DistL2;
//! use diskann_rs::IncrementalDiskANN;
//!
//! // Build initial index
//! let vectors = vec![vec![0.0; 128]; 1000];
//! let mut index = IncrementalDiskANN::<DistL2>::build_default(&vectors, "index.db").unwrap();
//!
//! // Add vectors without rebuilding
//! let new_ids = index.add_vectors(&[vec![1.0; 128]]).unwrap();
//!
//! // Delete vectors (lazy tombstoning)
//! index.delete_vectors(&[0, 1, 2]).unwrap();
//!
//! // Compact when needed
//! if index.should_compact() {
//!     index.compact("index_v2.db").unwrap();
//! }
//! ```
//!
//! ## File Layout
//! [ metadata_len:u64 ][ metadata (bincode) ][ padding up to vectors_offset ]
//! [ vectors (num * dim * f32) ][ adjacency (num * max_degree * u32) ]
//!
//! `vectors_offset` is a fixed 1 MiB gap by default.

mod incremental;
mod filtered;
pub mod simd;
pub mod pq;

pub use incremental::{
    IncrementalDiskANN, IncrementalConfig, IncrementalStats,
    is_delta_id, delta_local_idx,
};

pub use filtered::{FilteredDiskANN, Filter};

pub use simd::{SimdL2, SimdDot, SimdCosine, simd_info};

pub use pq::{ProductQuantizer, PQConfig, PQStats};

use anndists::prelude::Distance;
use bytemuck;
use memmap2::Mmap;
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use thiserror::Error;

/// Padding sentinel for adjacency slots (avoid colliding with node 0).
const PAD_U32: u32 = u32::MAX;

/// Defaults for in-memory DiskANN builds
pub const DISKANN_DEFAULT_MAX_DEGREE: usize = 64;
pub const DISKANN_DEFAULT_BUILD_BEAM: usize = 128;
pub const DISKANN_DEFAULT_ALPHA: f32 = 1.2;

/// Optional bag of knobs if you want to override just a few.
#[derive(Clone, Copy, Debug)]
pub struct DiskAnnParams {
    pub max_degree: usize,
    pub build_beam_width: usize,
    pub alpha: f32,
}
impl Default for DiskAnnParams {
    fn default() -> Self {
        Self {
            max_degree: DISKANN_DEFAULT_MAX_DEGREE,
            build_beam_width: DISKANN_DEFAULT_BUILD_BEAM,
            alpha: DISKANN_DEFAULT_ALPHA,
        }
    }
}

/// Custom error type for DiskAnnRS operations
#[derive(Debug, Error)]
pub enum DiskAnnError {
    /// Represents I/O errors during file operations
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Represents serialization/deserialization errors
    #[error("Serialization error: {0}")]
    Bincode(#[from] bincode::Error),

    /// Represents index-specific errors
    #[error("Index error: {0}")]
    IndexError(String),
}

/// Internal metadata structure stored in the index file
#[derive(Serialize, Deserialize, Debug)]
struct Metadata {
    dim: usize,
    num_vectors: usize,
    max_degree: usize,
    medoid_id: u32,
    vectors_offset: u64,
    adjacency_offset: u64,
    distance_name: String,
}

/// Candidate for search/frontier queues
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
        // Natural order by distance: smaller is "less".
        self.dist.partial_cmp(&other.dist)
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Main struct representing a DiskANN index (generic over distance)
pub struct DiskANN<D>
where
    D: Distance<f32> + Send + Sync + Copy + Clone + 'static,
{
    /// Dimensionality of vectors in the index
    pub dim: usize,
    /// Number of vectors in the index
    pub num_vectors: usize,
    /// Maximum number of edges per node
    pub max_degree: usize,
    /// Informational: type name of the distance (from metadata)
    pub distance_name: String,

    /// ID of the medoid (used as entry point)
    pub(crate) medoid_id: u32,
    // Offsets
    pub(crate) vectors_offset: u64,
    pub(crate) adjacency_offset: u64,

    /// Memory-mapped file
    pub(crate) mmap: Mmap,

    /// The distance strategy
    pub(crate) dist: D,
}

// constructors

impl<D> DiskANN<D>
where
    D: Distance<f32> + Send + Sync + Copy + Clone + 'static,
{
    /// Build with default parameters: (M=32, L=256, alpha=1.2).
    pub fn build_index_default(
        vectors: &[Vec<f32>],
        dist: D,
        file_path: &str,
    ) -> Result<Self, DiskAnnError> {
        Self::build_index(
            vectors,
            DISKANN_DEFAULT_MAX_DEGREE,
            DISKANN_DEFAULT_BUILD_BEAM,
            DISKANN_DEFAULT_ALPHA,
            dist,
            file_path,
        )
    }

    /// Build with a `DiskAnnParams` bundle.
    pub fn build_index_with_params(
        vectors: &[Vec<f32>],
        dist: D,
        file_path: &str,
        p: DiskAnnParams,
    ) -> Result<Self, DiskAnnError> {
        Self::build_index(
            vectors,
            p.max_degree,
            p.build_beam_width,
            p.alpha,
            dist,
            file_path,
        )
    }
}

/// Extra sugar when your distance type implements `Default` (most unit-struct metrics do).
impl<D> DiskANN<D>
where
    D: Distance<f32> + Default + Send + Sync + Copy + Clone + 'static,
{
    /// Build with default params **and** `D::default()` metric.
    pub fn build_index_default_metric(
        vectors: &[Vec<f32>],
        file_path: &str,
    ) -> Result<Self, DiskAnnError> {
        Self::build_index_default(vectors, D::default(), file_path)
    }

    /// Open an index using `D::default()` as the distance (matches what you built with).
    pub fn open_index_default_metric(path: &str) -> Result<Self, DiskAnnError> {
        Self::open_index_with(path, D::default())
    }
}

impl<D> DiskANN<D>
where
    D: Distance<f32> + Send + Sync + Copy + Clone + 'static,
{
    /// Builds a new index from provided vectors
    ///
    /// # Arguments
    /// * `vectors` - The vectors to index (slice of `Vec<f32>`)
    /// * `max_degree` - Maximum edges per node (M ~ 24-64)
    /// * `build_beam_width` - Construction L (e.g., 128-400)
    /// * `alpha` - Pruning parameter (1.2–2.0)
    /// * `dist` - Any `anndists::Distance<f32>` (e.g., `DistL2`)
    /// * `file_path` - Path of index file
    pub fn build_index(
        vectors: &[Vec<f32>],
        max_degree: usize,
        build_beam_width: usize,
        alpha: f32,
        dist: D,
        file_path: &str,
    ) -> Result<Self, DiskAnnError> {
        if vectors.is_empty() {
            return Err(DiskAnnError::IndexError("No vectors provided".to_string()));
        }

        let num_vectors = vectors.len();
        let dim = vectors[0].len();
        for (i, v) in vectors.iter().enumerate() {
            if v.len() != dim {
                return Err(DiskAnnError::IndexError(format!(
                    "Vector {} has dimension {} but expected {}",
                    i,
                    v.len(),
                    dim
                )));
            }
        }

        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .truncate(true)
            .open(file_path)?;

        // Reserve space for metadata (we'll write it after data)
        let vectors_offset = 1024 * 1024;
        let total_vector_bytes = (num_vectors as u64) * (dim as u64) * 4;

        // Write vectors contiguous (sequential I/O is fastest)
        file.seek(SeekFrom::Start(vectors_offset))?;
        for vector in vectors {
            let bytes = bytemuck::cast_slice(vector);
            file.write_all(bytes)?;
        }

        // Compute medoid using provided distance (parallelized distance eval)
        let medoid_id = calculate_medoid(vectors, dist);

        // Build Vamana-like graph (stronger refinement, parallel inner loops)
        let adjacency_offset = vectors_offset as u64 + total_vector_bytes;
        let graph = build_vamana_graph(
            vectors,
            max_degree,
            build_beam_width,
            alpha,
            dist,
            medoid_id as u32,
        );

        // Write adjacency lists (fixed max_degree, pad with PAD_U32)
        file.seek(SeekFrom::Start(adjacency_offset))?;
        for neighbors in &graph {
            let mut padded = neighbors.clone();
            padded.resize(max_degree, PAD_U32);
            let bytes = bytemuck::cast_slice(&padded);
            file.write_all(bytes)?;
        }

        // Write metadata
        let metadata = Metadata {
            dim,
            num_vectors,
            max_degree,
            medoid_id: medoid_id as u32,
            vectors_offset: vectors_offset as u64,
            adjacency_offset,
            distance_name: std::any::type_name::<D>().to_string(),
        };

        let md_bytes = bincode::serialize(&metadata)?;
        file.seek(SeekFrom::Start(0))?;
        let md_len = md_bytes.len() as u64;
        file.write_all(&md_len.to_le_bytes())?;
        file.write_all(&md_bytes)?;
        file.sync_all()?;

        // Memory map the file
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        Ok(Self {
            dim,
            num_vectors,
            max_degree,
            distance_name: metadata.distance_name,
            medoid_id: metadata.medoid_id,
            vectors_offset: metadata.vectors_offset,
            adjacency_offset: metadata.adjacency_offset,
            mmap,
            dist,
        })
    }

    /// Opens an existing index file, supplying the distance strategy explicitly.
    pub fn open_index_with(path: &str, dist: D) -> Result<Self, DiskAnnError> {
        let mut file = OpenOptions::new().read(true).write(false).open(path)?;

        // Read metadata length
        let mut buf8 = [0u8; 8];
        file.seek(SeekFrom::Start(0))?;
        file.read_exact(&mut buf8)?;
        let md_len = u64::from_le_bytes(buf8);

        // Read metadata
        let mut md_bytes = vec![0u8; md_len as usize];
        file.read_exact(&mut md_bytes)?;
        let metadata: Metadata = bincode::deserialize(&md_bytes)?;

        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        // Optional sanity/logging: warn if type differs from recorded name
        let expected = std::any::type_name::<D>();
        if metadata.distance_name != expected {
            eprintln!(
                "Warning: index recorded distance `{}` but you opened with `{}`",
                metadata.distance_name, expected
            );
        }

        Ok(Self {
            dim: metadata.dim,
            num_vectors: metadata.num_vectors,
            max_degree: metadata.max_degree,
            distance_name: metadata.distance_name,
            medoid_id: metadata.medoid_id,
            vectors_offset: metadata.vectors_offset,
            adjacency_offset: metadata.adjacency_offset,
            mmap,
            dist,
        })
    }

    /// Searches the index for nearest neighbors using a best-first beam search.
    /// Termination rule: continue while the best frontier can still improve the worst in working set.
    /// Like `search` but also returns the distance for each neighbor.
    /// Distances are exactly the ones computed during the beam search.
    pub fn search_with_dists(&self, query: &[f32], k: usize, beam_width: usize) -> Vec<(u32, f32)> {
        assert_eq!(
            query.len(),
            self.dim,
            "Query dim {} != index dim {}",
            query.len(),
            self.dim
        );

        #[derive(Clone, Copy)]
        struct Candidate {
            dist: f32,
            id: u32,
        }
        impl PartialEq for Candidate {
            fn eq(&self, o: &Self) -> bool {
                self.dist == o.dist && self.id == o.id
            }
        }
        impl Eq for Candidate {}
        impl PartialOrd for Candidate {
            fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
                self.dist.partial_cmp(&o.dist)
            }
        }
        impl Ord for Candidate {
            fn cmp(&self, o: &Self) -> Ordering {
                self.partial_cmp(o).unwrap_or(Ordering::Equal)
            }
        }

        let mut visited = HashSet::new();
        let mut frontier: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new(); // best-first by dist
        let mut w: BinaryHeap<Candidate> = BinaryHeap::new(); // working set, max-heap by dist

        // seed from medoid
        let start_dist = self.distance_to(query, self.medoid_id as usize);
        let start = Candidate {
            dist: start_dist,
            id: self.medoid_id,
        };
        frontier.push(Reverse(start));
        w.push(start);
        visited.insert(self.medoid_id);

        // expand while best frontier can still improve worst in working set
        while let Some(Reverse(best)) = frontier.peek().copied() {
            if w.len() >= beam_width {
                if let Some(worst) = w.peek() {
                    if best.dist >= worst.dist {
                        break;
                    }
                }
            }
            let Reverse(current) = frontier.pop().unwrap();

            for &nb in self.get_neighbors(current.id) {
                if nb == PAD_U32 {
                    continue;
                }
                if !visited.insert(nb) {
                    continue;
                }

                let d = self.distance_to(query, nb as usize);
                let cand = Candidate { dist: d, id: nb };

                if w.len() < beam_width {
                    w.push(cand);
                    frontier.push(Reverse(cand));
                } else if d < w.peek().unwrap().dist {
                    w.pop();
                    w.push(cand);
                    frontier.push(Reverse(cand));
                }
            }
        }

        // top-k by distance, keep distances
        let mut results: Vec<_> = w.into_vec();
        results.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        results.truncate(k);
        results.into_iter().map(|c| (c.id, c.dist)).collect()
    }
    /// search but only return neighbor ids
    pub fn search(&self, query: &[f32], k: usize, beam_width: usize) -> Vec<u32> {
        self.search_with_dists(query, k, beam_width)
            .into_iter()
            .map(|(id, _dist)| id)
            .collect()
    }

    /// Gets the neighbors of a node from the (fixed-degree) adjacency region
    fn get_neighbors(&self, node_id: u32) -> &[u32] {
        let offset = self.adjacency_offset + (node_id as u64 * self.max_degree as u64 * 4);
        let start = offset as usize;
        let end = start + (self.max_degree * 4);
        let bytes = &self.mmap[start..end];
        bytemuck::cast_slice(bytes)
    }

    /// Computes distance between `query` and vector `idx`
    fn distance_to(&self, query: &[f32], idx: usize) -> f32 {
        let offset = self.vectors_offset + (idx as u64 * self.dim as u64 * 4);
        let start = offset as usize;
        let end = start + (self.dim * 4);
        let bytes = &self.mmap[start..end];
        let vector: &[f32] = bytemuck::cast_slice(bytes);
        self.dist.eval(query, vector)
    }

    /// Gets a vector from the index (useful for tests)
    pub fn get_vector(&self, idx: usize) -> Vec<f32> {
        let offset = self.vectors_offset + (idx as u64 * self.dim as u64 * 4);
        let start = offset as usize;
        let end = start + (self.dim * 4);
        let bytes = &self.mmap[start..end];
        let vector: &[f32] = bytemuck::cast_slice(bytes);
        vector.to_vec()
    }
}

/// Calculates the medoid (vector closest to the centroid) using distance `D`
/// Parallelizes the per-vector distance evaluations.
fn calculate_medoid<D: Distance<f32> + Copy + Sync>(vectors: &[Vec<f32>], dist: D) -> usize {
    let dim = vectors[0].len();
    let mut centroid = vec![0.0f32; dim];

    for v in vectors {
        for (i, &val) in v.iter().enumerate() {
            centroid[i] += val;
        }
    }
    for val in &mut centroid {
        *val /= vectors.len() as f32;
    }

    let (best_idx, _best_dist) = vectors
        .par_iter()
        .enumerate()
        .map(|(idx, v)| (idx, dist.eval(&centroid, v)))
        .reduce(|| (0usize, f32::MAX), |a, b| if a.1 <= b.1 { a } else { b });

    best_idx
}

/// Builds a strengthened Vamana-like graph using multi-pass refinement.
/// - Multi-seed candidate gathering (medoid + random seeds)
/// - Union with current adjacency before α-prune
/// - 2 refinement passes with symmetrization after each pass
fn build_vamana_graph<D: Distance<f32> + Copy + Sync>(
    vectors: &[Vec<f32>],
    max_degree: usize,
    build_beam_width: usize,
    alpha: f32,
    dist: D,
    medoid_id: u32,
) -> Vec<Vec<u32>> {
    let n = vectors.len();
    let mut graph = vec![Vec::<u32>::new(); n];

    // Light random bootstrap to avoid disconnected starts
    {
        let mut rng = thread_rng();
        for i in 0..n {
            let mut s = HashSet::new();
            let target = (max_degree / 2).max(2).min(n.saturating_sub(1));
            while s.len() < target {
                let nb = rng.gen_range(0..n);
                if nb != i {
                    s.insert(nb as u32);
                }
            }
            graph[i] = s.into_iter().collect();
        }
    }

    // Refinement passes
    const PASSES: usize = 2;
    const EXTRA_SEEDS: usize = 2;

    let mut rng = thread_rng();
    for _pass in 0..PASSES {
        // Shuffle visit order each pass
        let mut order: Vec<usize> = (0..n).collect();
        order.shuffle(&mut rng);

        // Snapshot read of graph for parallel candidate building
        let snapshot = &graph;

        // Build new neighbor proposals in parallel
        let new_graph: Vec<Vec<u32>> = order
            .par_iter()
            .map(|&u| {
                let mut candidates: Vec<(u32, f32)> =
                    Vec::with_capacity(build_beam_width * (2 + EXTRA_SEEDS));

                // Include current adjacency with distances
                for &nb in &snapshot[u] {
                    let d = dist.eval(&vectors[u], &vectors[nb as usize]);
                    candidates.push((nb, d));
                }

                // Seeds: always medoid + some random starts
                let mut seeds = Vec::with_capacity(1 + EXTRA_SEEDS);
                seeds.push(medoid_id as usize);
                let mut trng = thread_rng();
                for _ in 0..EXTRA_SEEDS {
                    seeds.push(trng.gen_range(0..n));
                }

                // Gather candidates from greedy searches
                for start in seeds {
                    let mut part = greedy_search(
                        &vectors[u],
                        vectors,
                        snapshot,
                        start,
                        build_beam_width,
                        dist,
                    );
                    candidates.append(&mut part);
                }

                // Deduplicate by id keeping best distance
                candidates.sort_by(|a, b| a.0.cmp(&b.0));
                candidates.dedup_by(|a, b| {
                    if a.0 == b.0 {
                        if a.1 < b.1 {
                            *b = *a;
                        }
                        true
                    } else {
                        false
                    }
                });

                // α-prune around u
                prune_neighbors(u, &candidates, vectors, max_degree, alpha, dist)
            })
            .collect();

        // Symmetrize: union incoming + outgoing, then α-prune again (parallel)
        // Build inverse map: node-id -> position in `order`
        let mut pos_of = vec![0usize; n];
        for (pos, &u) in order.iter().enumerate() {
            pos_of[u] = pos;
        }

        // Build incoming as CSR
        let (incoming_flat, incoming_off) = build_incoming_csr(&order, &new_graph, n);

        // Union + prune in parallel
        graph = (0..n)
            .into_par_iter()
            .map(|u| {
                let ng = &new_graph[pos_of[u]]; // outgoing from this pass
                let inc = &incoming_flat[incoming_off[u]..incoming_off[u + 1]]; // incoming to u

                // pool = union(outgoing ∪ incoming) with tiny, cache-friendly ops
                let mut pool_ids: Vec<u32> = Vec::with_capacity(ng.len() + inc.len());
                pool_ids.extend_from_slice(ng);
                pool_ids.extend_from_slice(inc);
                pool_ids.sort_unstable();
                pool_ids.dedup();

                // compute distances once, then α-prune
                let pool: Vec<(u32, f32)> = pool_ids
                    .into_iter()
                    .filter(|&id| id as usize != u)
                    .map(|id| (id, dist.eval(&vectors[u], &vectors[id as usize])))
                    .collect();

                prune_neighbors(u, &pool, vectors, max_degree, alpha, dist)
            })
            .collect();
    }

    // Final cleanup (ensure <= max_degree everywhere)
    graph
        .into_par_iter()
        .enumerate()
        .map(|(u, neigh)| {
            if neigh.len() <= max_degree {
                return neigh;
            }
            let pool: Vec<(u32, f32)> = neigh
                .iter()
                .map(|&id| (id, dist.eval(&vectors[u], &vectors[id as usize])))
                .collect();
            prune_neighbors(u, &pool, vectors, max_degree, alpha, dist)
        })
        .collect()
}

/// Greedy search used during construction (read-only on `graph`)
/// Same termination rule as query-time search.
fn greedy_search<D: Distance<f32> + Copy>(
    query: &[f32],
    vectors: &[Vec<f32>],
    graph: &[Vec<u32>],
    start_id: usize,
    beam_width: usize,
    dist: D,
) -> Vec<(u32, f32)> {
    let mut visited = HashSet::new();
    let mut frontier: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new(); // min-heap by dist
    let mut w: BinaryHeap<Candidate> = BinaryHeap::new(); // max-heap by dist

    let start_dist = dist.eval(query, &vectors[start_id]);
    let start = Candidate {
        dist: start_dist,
        id: start_id as u32,
    };
    frontier.push(Reverse(start));
    w.push(start);
    visited.insert(start_id as u32);

    while let Some(Reverse(best)) = frontier.peek().copied() {
        if w.len() >= beam_width {
            if let Some(worst) = w.peek() {
                if best.dist >= worst.dist {
                    break;
                }
            }
        }
        let Reverse(cur) = frontier.pop().unwrap();

        for &nb in &graph[cur.id as usize] {
            if !visited.insert(nb) {
                continue;
            }
            let d = dist.eval(query, &vectors[nb as usize]);
            let cand = Candidate { dist: d, id: nb };

            if w.len() < beam_width {
                w.push(cand);
                frontier.push(Reverse(cand));
            } else if d < w.peek().unwrap().dist {
                w.pop();
                w.push(cand);
                frontier.push(Reverse(cand));
            }
        }
    }

    let mut v = w.into_vec();
    v.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
    v.into_iter().map(|c| (c.id, c.dist)).collect()
}

/// α-pruning from DiskANN/Vamana
fn prune_neighbors<D: Distance<f32> + Copy>(
    node_id: usize,
    candidates: &[(u32, f32)],
    vectors: &[Vec<f32>],
    max_degree: usize,
    alpha: f32,
    dist: D,
) -> Vec<u32> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let mut sorted = candidates.to_vec();
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut pruned = Vec::<u32>::new();

    for &(cand_id, cand_dist) in &sorted {
        if cand_id as usize == node_id {
            continue;
        }
        let mut ok = true;
        for &sel in &pruned {
            let d = dist.eval(&vectors[cand_id as usize], &vectors[sel as usize]);
            if d < alpha * cand_dist {
                ok = false;
                break;
            }
        }
        if ok {
            pruned.push(cand_id);
            if pruned.len() >= max_degree {
                break;
            }
        }
    }

    // fill with closest if still not full
    for &(cand_id, _) in &sorted {
        if cand_id as usize == node_id {
            continue;
        }
        if !pruned.contains(&cand_id) {
            pruned.push(cand_id);
            if pruned.len() >= max_degree {
                break;
            }
        }
    }

    pruned
}

fn build_incoming_csr(order: &[usize], new_graph: &[Vec<u32>], n: usize) -> (Vec<u32>, Vec<usize>) {
    // 1) count in-degree per node
    let mut indeg = vec![0usize; n];
    for (pos, _u) in order.iter().enumerate() {
        for &v in &new_graph[pos] {
            indeg[v as usize] += 1;
        }
    }
    // 2) prefix sums → offsets
    let mut off = vec![0usize; n + 1];
    for i in 0..n {
        off[i + 1] = off[i] + indeg[i];
    }
    // 3) fill flat incoming list
    let mut cur = off.clone();
    let mut incoming_flat = vec![0u32; off[n]];
    for (pos, &u) in order.iter().enumerate() {
        for &v in &new_graph[pos] {
            let idx = cur[v as usize];
            incoming_flat[idx] = u as u32;
            cur[v as usize] += 1;
        }
    }
    (incoming_flat, off)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anndists::dist::{DistCosine, DistL2};
    use rand::Rng;
    use std::fs;

    fn euclid(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    #[test]
    fn test_small_index_l2() {
        let path = "test_small_l2.db";
        let _ = fs::remove_file(path);

        let vectors = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
        ];

        let index = DiskANN::<DistL2>::build_index_default(&vectors, DistL2 {}, path).unwrap();

        let q = vec![0.1, 0.1];
        let nns = index.search(&q, 3, 8);
        assert_eq!(nns.len(), 3);

        // Verify the first neighbor is quite close in L2
        let v = index.get_vector(nns[0] as usize);
        assert!(euclid(&q, &v) < 1.0);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_cosine() {
        let path = "test_cosine.db";
        let _ = fs::remove_file(path);

        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
            vec![1.0, 0.0, 1.0],
        ];

        let index =
            DiskANN::<DistCosine>::build_index_default(&vectors, DistCosine {}, path).unwrap();

        let q = vec![2.0, 0.0, 0.0]; // parallel to [1,0,0]
        let nns = index.search(&q, 2, 8);
        assert_eq!(nns.len(), 2);

        // Top neighbor should have high cosine similarity (close direction)
        let v = index.get_vector(nns[0] as usize);
        let dot = v.iter().zip(&q).map(|(a, b)| a * b).sum::<f32>();
        let n1 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let n2 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos = dot / (n1 * n2);
        assert!(cos > 0.7);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_persistence_and_open() {
        let path = "test_persist.db";
        let _ = fs::remove_file(path);

        let vectors = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];

        {
            let _idx = DiskANN::<DistL2>::build_index_default(&vectors, DistL2 {}, path).unwrap();
        }

        let idx2 = DiskANN::<DistL2>::open_index_default_metric(path).unwrap();
        assert_eq!(idx2.num_vectors, 4);
        assert_eq!(idx2.dim, 2);

        let q = vec![0.9, 0.9];
        let res = idx2.search(&q, 2, 8);
        // [1,1] should be best
        assert_eq!(res[0], 3);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_grid_connectivity() {
        let path = "test_grid.db";
        let _ = fs::remove_file(path);

        // 5x5 grid
        let mut vectors = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                vectors.push(vec![i as f32, j as f32]);
            }
        }

        let index = DiskANN::<DistL2>::build_index_with_params(
            &vectors,
            DistL2 {},
            path,
            DiskAnnParams {
                max_degree: 4,
                build_beam_width: 64,
                alpha: 1.5,
            },
        )
        .unwrap();

        for target in 0..vectors.len() {
            let q = &vectors[target];
            let nns = index.search(q, 10, 32);
            if !nns.contains(&(target as u32)) {
                let v = index.get_vector(nns[0] as usize);
                assert!(euclid(q, &v) < 2.0);
            }
            for &nb in nns.iter().take(5) {
                let v = index.get_vector(nb as usize);
                assert!(euclid(q, &v) < 5.0);
            }
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_medium_random() {
        let path = "test_medium.db";
        let _ = fs::remove_file(path);

        let n = 200usize;
        let d = 32usize;
        let mut rng = rand::thread_rng();
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..d).map(|_| rng.r#gen::<f32>()).collect())
            .collect();

        let index = DiskANN::<DistL2>::build_index_with_params(
            &vectors,
            DistL2 {},
            path,
            DiskAnnParams {
                max_degree: 32,
                build_beam_width: 128,
                alpha: 1.2,
            },
        )
        .unwrap();

        let q: Vec<f32> = (0..d).map(|_| rng.r#gen::<f32>()).collect();
        let res = index.search(&q, 10, 64);
        assert_eq!(res.len(), 10);

        // Ensure distances are nondecreasing
        let dists: Vec<f32> = res
            .iter()
            .map(|&id| {
                let v = index.get_vector(id as usize);
                euclid(&q, &v)
            })
            .collect();
        let mut sorted = dists.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(dists, sorted);

        let _ = fs::remove_file(path);
    }
}
