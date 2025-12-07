//! # Product Quantization (PQ) for Vector Compression
//!
//! Product Quantization compresses high-dimensional vectors by:
//! 1. Dividing each vector into M subspaces (segments)
//! 2. Training K centroids per subspace (codebook)
//! 3. Representing each segment by its nearest centroid ID (1 byte for K=256)
//!
//! ## Compression Ratio
//!
//! For 128-dim float vectors (512 bytes) with M=8 subspaces:
//! - Original: 512 bytes
//! - Compressed: 8 bytes (one centroid ID per subspace)
//! - Compression: 64x
//!
//! ## Usage
//!
//! ```ignore
//! use diskann_rs::pq::{ProductQuantizer, PQConfig};
//!
//! // Train a quantizer on sample vectors
//! let vectors: Vec<Vec<f32>> = load_your_training_data();
//! let config = PQConfig::default(); // 8 subspaces, 256 centroids each
//! let pq = ProductQuantizer::train(&vectors, config).unwrap();
//!
//! // Encode vectors (each becomes M bytes)
//! let codes: Vec<Vec<u8>> = vectors.iter().map(|v| pq.encode(v)).collect();
//!
//! // Compute asymmetric distance (query vs quantized database vector)
//! let query = vec![0.0f32; 128];
//! let dist = pq.asymmetric_distance(&query, &codes[0]);
//! ```
//!
//! ## Asymmetric Distance Computation (ADC)
//!
//! For search, we compute exact query-to-centroid distances once,
//! then use a lookup table for fast distance approximation.

use crate::DiskAnnError;
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};

/// Configuration for Product Quantization
#[derive(Clone, Copy, Debug)]
pub struct PQConfig {
    /// Number of subspaces (M). Vector is divided into M segments.
    /// Typical values: 4, 8, 16, 32
    pub num_subspaces: usize,
    /// Number of centroids per subspace (K). Typically 256 (fits in u8).
    pub num_centroids: usize,
    /// Number of k-means iterations for training
    pub kmeans_iterations: usize,
    /// Sample size for training (if 0, use all vectors)
    pub training_sample_size: usize,
}

impl Default for PQConfig {
    fn default() -> Self {
        Self {
            num_subspaces: 8,
            num_centroids: 256,
            kmeans_iterations: 20,
            training_sample_size: 50_000,
        }
    }
}

/// Trained Product Quantizer
#[derive(Serialize, Deserialize, Clone)]
pub struct ProductQuantizer {
    /// Dimension of original vectors
    dim: usize,
    /// Number of subspaces
    num_subspaces: usize,
    /// Number of centroids per subspace
    num_centroids: usize,
    /// Dimension of each subspace
    subspace_dim: usize,
    /// Codebooks: [num_subspaces][num_centroids][subspace_dim]
    /// Flattened for cache efficiency
    codebooks: Vec<f32>,
}

impl ProductQuantizer {
    /// Train a product quantizer on a set of vectors
    pub fn train(vectors: &[Vec<f32>], config: PQConfig) -> Result<Self, DiskAnnError> {
        if vectors.is_empty() {
            return Err(DiskAnnError::IndexError("No vectors to train on".into()));
        }

        let dim = vectors[0].len();
        if dim % config.num_subspaces != 0 {
            return Err(DiskAnnError::IndexError(format!(
                "Dimension {} not divisible by num_subspaces {}",
                dim, config.num_subspaces
            )));
        }

        let subspace_dim = dim / config.num_subspaces;

        // Sample training vectors if needed
        let training_vectors: Vec<&Vec<f32>> = if config.training_sample_size > 0
            && vectors.len() > config.training_sample_size
        {
            let mut rng = thread_rng();
            vectors
                .choose_multiple(&mut rng, config.training_sample_size)
                .collect()
        } else {
            vectors.iter().collect()
        };

        // Train codebook for each subspace (parallel)
        let codebooks_per_subspace: Vec<Vec<f32>> = (0..config.num_subspaces)
            .into_par_iter()
            .map(|m| {
                // Extract subspace vectors
                let start = m * subspace_dim;
                let end = start + subspace_dim;

                let subspace_vectors: Vec<Vec<f32>> = training_vectors
                    .iter()
                    .map(|v| v[start..end].to_vec())
                    .collect();

                // Run k-means
                kmeans(
                    &subspace_vectors,
                    config.num_centroids,
                    config.kmeans_iterations,
                )
            })
            .collect();

        // Flatten codebooks for cache efficiency
        let mut codebooks =
            Vec::with_capacity(config.num_subspaces * config.num_centroids * subspace_dim);
        for cb in &codebooks_per_subspace {
            codebooks.extend_from_slice(cb);
        }

        Ok(Self {
            dim,
            num_subspaces: config.num_subspaces,
            num_centroids: config.num_centroids,
            subspace_dim,
            codebooks,
        })
    }

    /// Encode a vector into PQ codes (M bytes)
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        assert_eq!(vector.len(), self.dim, "Vector dimension mismatch");

        let mut codes = Vec::with_capacity(self.num_subspaces);

        for m in 0..self.num_subspaces {
            let start = m * self.subspace_dim;
            let end = start + self.subspace_dim;
            let subvec = &vector[start..end];

            // Find nearest centroid
            let mut best_centroid = 0u8;
            let mut best_dist = f32::MAX;

            for k in 0..self.num_centroids {
                let centroid = self.get_centroid(m, k);
                let dist = l2_distance(subvec, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_centroid = k as u8;
                }
            }

            codes.push(best_centroid);
        }

        codes
    }

    /// Batch encode vectors (parallel)
    pub fn encode_batch(&self, vectors: &[Vec<f32>]) -> Vec<Vec<u8>> {
        vectors.par_iter().map(|v| self.encode(v)).collect()
    }

    /// Compute asymmetric distance between query and quantized vector
    /// This uses precomputed distance tables for efficiency
    pub fn asymmetric_distance(&self, query: &[f32], codes: &[u8]) -> f32 {
        assert_eq!(query.len(), self.dim, "Query dimension mismatch");
        assert_eq!(codes.len(), self.num_subspaces, "Code length mismatch");

        let mut total_dist = 0.0f32;

        for m in 0..self.num_subspaces {
            let start = m * self.subspace_dim;
            let end = start + self.subspace_dim;
            let query_sub = &query[start..end];

            let centroid_id = codes[m] as usize;
            let centroid = self.get_centroid(m, centroid_id);

            total_dist += l2_distance(query_sub, centroid);
        }

        total_dist
    }

    /// Create a distance lookup table for a query (for fast batch queries)
    ///
    /// Returns: `[num_subspaces][num_centroids]` distance table
    pub fn create_distance_table(&self, query: &[f32]) -> Vec<f32> {
        assert_eq!(query.len(), self.dim);

        let mut table = Vec::with_capacity(self.num_subspaces * self.num_centroids);

        for m in 0..self.num_subspaces {
            let start = m * self.subspace_dim;
            let end = start + self.subspace_dim;
            let query_sub = &query[start..end];

            for k in 0..self.num_centroids {
                let centroid = self.get_centroid(m, k);
                table.push(l2_distance(query_sub, centroid));
            }
        }

        table
    }

    /// Compute distance using precomputed table (very fast)
    #[inline]
    pub fn distance_with_table(&self, table: &[f32], codes: &[u8]) -> f32 {
        let mut dist = 0.0f32;
        for (m, &code) in codes.iter().enumerate() {
            let idx = m * self.num_centroids + code as usize;
            dist += table[idx];
        }
        dist
    }

    /// Decode PQ codes back to approximate vector
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        assert_eq!(codes.len(), self.num_subspaces);

        let mut vector = Vec::with_capacity(self.dim);

        for (m, &code) in codes.iter().enumerate() {
            let centroid = self.get_centroid(m, code as usize);
            vector.extend_from_slice(centroid);
        }

        vector
    }

    /// Get centroid for subspace m, centroid k
    #[inline]
    fn get_centroid(&self, m: usize, k: usize) -> &[f32] {
        let offset = (m * self.num_centroids + k) * self.subspace_dim;
        &self.codebooks[offset..offset + self.subspace_dim]
    }

    /// Save quantizer to file
    pub fn save(&self, path: &str) -> Result<(), DiskAnnError> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self)?;
        Ok(())
    }

    /// Load quantizer from file
    pub fn load(path: &str) -> Result<Self, DiskAnnError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let pq: Self = bincode::deserialize_from(reader)?;
        Ok(pq)
    }

    /// Get stats about the quantizer
    pub fn stats(&self) -> PQStats {
        PQStats {
            dim: self.dim,
            num_subspaces: self.num_subspaces,
            num_centroids: self.num_centroids,
            subspace_dim: self.subspace_dim,
            codebook_size_bytes: self.codebooks.len() * 4,
            code_size_bytes: self.num_subspaces,
            compression_ratio: (self.dim * 4) as f32 / self.num_subspaces as f32,
        }
    }
}

/// Statistics about a ProductQuantizer
#[derive(Debug, Clone)]
pub struct PQStats {
    pub dim: usize,
    pub num_subspaces: usize,
    pub num_centroids: usize,
    pub subspace_dim: usize,
    pub codebook_size_bytes: usize,
    pub code_size_bytes: usize,
    pub compression_ratio: f32,
}

impl std::fmt::Display for PQStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Product Quantizer Stats:")?;
        writeln!(f, "  Original dimension: {}", self.dim)?;
        writeln!(f, "  Subspaces (M): {}", self.num_subspaces)?;
        writeln!(f, "  Centroids per subspace (K): {}", self.num_centroids)?;
        writeln!(f, "  Subspace dimension: {}", self.subspace_dim)?;
        writeln!(f, "  Codebook size: {} bytes", self.codebook_size_bytes)?;
        writeln!(f, "  Compressed code size: {} bytes", self.code_size_bytes)?;
        writeln!(f, "  Compression ratio: {:.1}x", self.compression_ratio)
    }
}

/// Simple k-means clustering
/// Returns flattened centroids of shape [k * dim]
/// Note: Always returns exactly `k` centroids (replicates if n < k)
fn kmeans(vectors: &[Vec<f32>], k: usize, iterations: usize) -> Vec<f32> {
    if vectors.is_empty() {
        return vec![0.0; k * 1]; // shouldn't happen, but safe fallback
    }

    let dim = vectors[0].len();
    let n = vectors.len();
    let effective_k = k.min(n); // For clustering, use available vectors

    // Initialize centroids with k-means++ style
    let mut centroids = Vec::with_capacity(k * dim);
    let mut rng = thread_rng();

    // First centroid: random vector
    let first = rng.gen_range(0..n);
    centroids.extend_from_slice(&vectors[first]);

    // Remaining centroids: weighted by distance to nearest existing centroid (k-means++)
    // Only do this for effective_k-1 more centroids (based on actual unique vectors)
    for _ in 1..effective_k {
        let num_current = centroids.len() / dim;
        let distances: Vec<f32> = vectors
            .iter()
            .map(|v| {
                let mut min_dist = f32::MAX;
                for c in 0..num_current {
                    let centroid = &centroids[c * dim..(c + 1) * dim];
                    let d = l2_distance(v, centroid);
                    min_dist = min_dist.min(d);
                }
                min_dist
            })
            .collect();

        // Sample weighted by distance
        let total: f32 = distances.iter().sum();
        if total == 0.0 {
            // All points are at centroids, pick random
            let idx = rng.gen_range(0..n);
            centroids.extend_from_slice(&vectors[idx]);
        } else {
            let threshold = rng.r#gen::<f32>() * total;
            let mut cumsum = 0.0;
            let mut picked = false;
            for (i, &d) in distances.iter().enumerate() {
                cumsum += d;
                if cumsum >= threshold {
                    centroids.extend_from_slice(&vectors[i]);
                    picked = true;
                    break;
                }
            }
            // Fallback if we didn't pick one (can happen with float precision)
            if !picked {
                centroids.extend_from_slice(&vectors[n - 1]);
            }
        }
    }

    // If k > n, replicate existing centroids to reach k
    while centroids.len() < k * dim {
        // Cycle through existing centroids
        let idx = (centroids.len() / dim) % effective_k;
        let centroid = centroids[idx * dim..(idx + 1) * dim].to_vec();
        centroids.extend_from_slice(&centroid);
    }
    centroids.truncate(k * dim);

    // Lloyd's algorithm iterations
    let mut assignments: Vec<usize>;

    for _ in 0..iterations {
        // Assignment step (parallel)
        assignments = vectors
            .par_iter()
            .map(|v| {
                let mut best_c = 0;
                let mut best_dist = f32::MAX;
                for c in 0..k {
                    let centroid = &centroids[c * dim..(c + 1) * dim];
                    let d = l2_distance(v, centroid);
                    if d < best_dist {
                        best_dist = d;
                        best_c = c;
                    }
                }
                best_c
            })
            .collect();

        // Update step
        let mut new_centroids = vec![0.0f32; k * dim];
        let mut counts = vec![0usize; k];

        for (i, &c) in assignments.iter().enumerate() {
            counts[c] += 1;
            for (j, &val) in vectors[i].iter().enumerate() {
                new_centroids[c * dim + j] += val;
            }
        }

        // Average
        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..dim {
                    new_centroids[c * dim + j] /= counts[c] as f32;
                }
            } else {
                // Empty cluster: reinitialize from random point
                let idx = rng.gen_range(0..n);
                for j in 0..dim {
                    new_centroids[c * dim + j] = vectors[idx][j];
                }
            }
        }

        centroids = new_centroids;
    }

    centroids
}

/// L2 squared distance
#[inline]
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.r#gen::<f32>()).collect())
            .collect()
    }

    #[test]
    fn test_pq_encode_decode() {
        let vectors = random_vectors(1000, 64, 42);
        let config = PQConfig {
            num_subspaces: 8,
            num_centroids: 256,
            kmeans_iterations: 10,
            training_sample_size: 0,
        };

        let pq = ProductQuantizer::train(&vectors, config).unwrap();

        // Encode and decode
        let original = &vectors[0];
        let codes = pq.encode(original);
        let decoded = pq.decode(&codes);

        // Should have same dimension
        assert_eq!(decoded.len(), original.len());

        // Decoded should be somewhat close to original (lossy compression)
        let dist = l2_distance(original, &decoded);
        assert!(
            dist < original.len() as f32 * 0.1,
            "Reconstruction error too high: {dist}"
        );
    }

    #[test]
    fn test_pq_asymmetric_distance() {
        let vectors = random_vectors(500, 32, 123);
        let config = PQConfig {
            num_subspaces: 4,
            num_centroids: 64,
            kmeans_iterations: 10,
            training_sample_size: 0,
        };

        let pq = ProductQuantizer::train(&vectors, config).unwrap();

        let query = &vectors[0];
        let target = &vectors[100];

        let codes = pq.encode(target);

        // Asymmetric distance should be similar to distance to decoded
        let asym_dist = pq.asymmetric_distance(query, &codes);
        let decoded = pq.decode(&codes);
        let exact_dist = l2_distance(query, &decoded);

        // Should be very close (asymmetric uses same centroids)
        assert!(
            (asym_dist - exact_dist).abs() < 1e-5,
            "asym={asym_dist}, exact={exact_dist}"
        );
    }

    #[test]
    fn test_pq_distance_table() {
        let vectors = random_vectors(500, 32, 456);
        let config = PQConfig {
            num_subspaces: 4,
            num_centroids: 64,
            kmeans_iterations: 10,
            training_sample_size: 0,
        };

        let pq = ProductQuantizer::train(&vectors, config).unwrap();

        let query = &vectors[0];
        let table = pq.create_distance_table(query);

        // Compare table-based vs direct asymmetric distance
        for target in vectors.iter().take(10) {
            let codes = pq.encode(target);
            let direct = pq.asymmetric_distance(query, &codes);
            let table_dist = pq.distance_with_table(&table, &codes);

            assert!(
                (direct - table_dist).abs() < 1e-5,
                "direct={direct}, table={table_dist}"
            );
        }
    }

    #[test]
    fn test_pq_batch_encode() {
        let vectors = random_vectors(100, 64, 789);
        let config = PQConfig::default();

        let pq = ProductQuantizer::train(&vectors, config).unwrap();
        let codes = pq.encode_batch(&vectors);

        assert_eq!(codes.len(), vectors.len());
        for code in &codes {
            assert_eq!(code.len(), config.num_subspaces);
        }
    }

    #[test]
    fn test_pq_save_load() {
        let vectors = random_vectors(200, 64, 111);
        let config = PQConfig {
            num_subspaces: 8,
            num_centroids: 128,
            kmeans_iterations: 5,
            training_sample_size: 0,
        };

        let pq = ProductQuantizer::train(&vectors, config).unwrap();
        let codes_before = pq.encode(&vectors[0]);

        let path = "test_pq.bin";
        pq.save(path).unwrap();

        let pq_loaded = ProductQuantizer::load(path).unwrap();
        let codes_after = pq_loaded.encode(&vectors[0]);

        assert_eq!(codes_before, codes_after);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_pq_stats() {
        let vectors = random_vectors(100, 128, 222);
        let config = PQConfig {
            num_subspaces: 8,
            num_centroids: 256,
            kmeans_iterations: 5,
            training_sample_size: 0,
        };

        let pq = ProductQuantizer::train(&vectors, config).unwrap();
        let stats = pq.stats();

        assert_eq!(stats.dim, 128);
        assert_eq!(stats.num_subspaces, 8);
        assert_eq!(stats.num_centroids, 256);
        assert_eq!(stats.subspace_dim, 16);
        assert_eq!(stats.code_size_bytes, 8);
        assert!(stats.compression_ratio > 50.0); // 128*4 / 8 = 64x

        println!("{}", stats);
    }

    #[test]
    fn test_pq_preserves_ordering() {
        let vectors = random_vectors(500, 64, 333);
        let config = PQConfig {
            num_subspaces: 8,
            num_centroids: 256,
            kmeans_iterations: 15,
            training_sample_size: 0,
        };

        let pq = ProductQuantizer::train(&vectors, config).unwrap();

        let query = &vectors[0];

        // Compute true distances
        let mut true_dists: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, v)| (i, l2_distance(query, v)))
            .collect();
        true_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Compute PQ distances
        let table = pq.create_distance_table(query);
        let codes: Vec<Vec<u8>> = vectors.iter().map(|v| pq.encode(v)).collect();

        let mut pq_dists: Vec<(usize, f32)> = codes
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, c)| (i, pq.distance_with_table(&table, c)))
            .collect();
        pq_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Check recall@10: how many of true top-10 appear in PQ top-10
        let true_top10: std::collections::HashSet<_> =
            true_dists.iter().take(10).map(|(i, _)| *i).collect();
        let pq_top10: std::collections::HashSet<_> =
            pq_dists.iter().take(10).map(|(i, _)| *i).collect();

        let recall: f32 = true_top10.intersection(&pq_top10).count() as f32 / 10.0;
        assert!(
            recall >= 0.5,
            "PQ recall@10 too low: {recall}. Expected >= 0.5"
        );
    }
}
