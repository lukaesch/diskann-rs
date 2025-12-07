//! Professional ANN Benchmark Suite for diskann-rs
//!
//! This benchmark measures:
//! - Index build time and throughput
//! - Query latency (p50, p95, p99)
//! - Recall@k at various k values
//! - Queries per second (QPS)
//! - Memory efficiency
//! - Incremental update performance
//!
//! Run with: cargo bench --bench benchmark

use anndists::dist::DistL2;
use diskann_rs::{DiskANN, DiskAnnParams, IncrementalDiskANN};
use rand::prelude::*;
use rayon::prelude::*;
use std::fs;
use std::time::Instant;

/// Benchmark configuration
struct BenchConfig {
    name: &'static str,
    num_vectors: usize,
    num_queries: usize,
    dim: usize,
    k: usize,
    beam_widths: Vec<usize>,
}

/// Results from a single benchmark run
#[derive(Debug, Clone)]
struct BenchResult {
    name: String,
    num_vectors: usize,
    dim: usize,
    k: usize,
    beam_width: usize,
    build_time_ms: f64,
    build_throughput: f64,  // vectors/sec
    query_time_us_p50: f64,
    query_time_us_p95: f64,
    query_time_us_p99: f64,
    qps: f64,
    recall: f64,
    index_size_mb: f64,
}

impl BenchResult {
    fn print_header() {
        println!("{}", "=".repeat(120));
        println!(
            "{:<20} {:>8} {:>6} {:>4} {:>6} {:>10} {:>12} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8}",
            "Benchmark", "Vectors", "Dim", "K", "Beam", "Build(ms)", "Build(vec/s)",
            "P50(μs)", "P95(μs)", "P99(μs)", "QPS", "Recall", "Size(MB)"
        );
        println!("{}", "-".repeat(120));
    }

    fn print(&self) {
        println!(
            "{:<20} {:>8} {:>6} {:>4} {:>6} {:>10.1} {:>12.0} {:>10.1} {:>10.1} {:>10.1} {:>10.0} {:>8.4} {:>8.2}",
            self.name, self.num_vectors, self.dim, self.k, self.beam_width,
            self.build_time_ms, self.build_throughput,
            self.query_time_us_p50, self.query_time_us_p95, self.query_time_us_p99,
            self.qps, self.recall, self.index_size_mb
        );
    }
}

/// Generate random vectors for benchmarking
fn generate_vectors(num: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..num)
        .map(|_| (0..dim).map(|_| rng.r#gen::<f32>()).collect())
        .collect()
}

/// Compute ground truth nearest neighbors using brute force
fn compute_ground_truth(
    queries: &[Vec<f32>],
    data: &[Vec<f32>],
    k: usize,
) -> Vec<Vec<usize>> {
    queries
        .par_iter()
        .map(|q| {
            let mut dists: Vec<(usize, f32)> = data
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let d: f32 = q.iter().zip(v).map(|(a, b)| (a - b).powi(2)).sum();
                    (i, d)
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            dists.iter().take(k).map(|(i, _)| *i).collect()
        })
        .collect()
}

/// Calculate recall given results and ground truth
fn calculate_recall(results: &[Vec<u32>], ground_truth: &[Vec<usize>], k: usize) -> f64 {
    let mut total_correct = 0usize;
    let total = results.len() * k;

    for (res, gt) in results.iter().zip(ground_truth.iter()) {
        let gt_set: std::collections::HashSet<_> = gt.iter().take(k).collect();
        for id in res.iter().take(k) {
            if gt_set.contains(&(*id as usize)) {
                total_correct += 1;
            }
        }
    }

    total_correct as f64 / total as f64
}

/// Calculate percentiles from sorted latencies
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Run a single benchmark configuration
fn run_benchmark(config: &BenchConfig, params: DiskAnnParams) -> Vec<BenchResult> {
    let index_path = format!("bench_{}.db", config.name);
    let _ = fs::remove_file(&index_path);

    println!("\n[{}] Generating {} vectors of dim {}...",
             config.name, config.num_vectors, config.dim);

    // Generate data
    let data = generate_vectors(config.num_vectors, config.dim, 42);
    let queries = generate_vectors(config.num_queries, config.dim, 123);

    // Compute ground truth
    println!("[{}] Computing ground truth (brute force)...", config.name);
    let ground_truth = compute_ground_truth(&queries, &data, config.k);

    // Build index
    println!("[{}] Building index with M={}, L={}, alpha={}...",
             config.name, params.max_degree, params.build_beam_width, params.alpha);

    let build_start = Instant::now();
    let index = DiskANN::<DistL2>::build_index_with_params(
        &data,
        DistL2 {},
        &index_path,
        params,
    )
    .expect("Failed to build index");
    let build_time = build_start.elapsed();

    let index_size = fs::metadata(&index_path)
        .map(|m| m.len() as f64 / (1024.0 * 1024.0))
        .unwrap_or(0.0);

    let build_throughput = config.num_vectors as f64 / build_time.as_secs_f64();

    let mut results = Vec::new();

    // Run searches at different beam widths
    for &beam_width in &config.beam_widths {
        println!("[{}] Benchmarking search with beam_width={}...", config.name, beam_width);

        // Warm-up
        for q in queries.iter().take(10) {
            let _ = index.search(q, config.k, beam_width);
        }

        // Measure individual query latencies
        let mut latencies_us = Vec::with_capacity(config.num_queries);
        let total_start = Instant::now();

        for q in &queries {
            let q_start = Instant::now();
            let _ = index.search(q, config.k, beam_width);
            latencies_us.push(q_start.elapsed().as_secs_f64() * 1_000_000.0);
        }

        let total_time = total_start.elapsed();

        // Collect results for recall calculation
        let search_results: Vec<Vec<u32>> = queries
            .iter()
            .map(|q| index.search(q, config.k, beam_width))
            .collect();

        let recall = calculate_recall(&search_results, &ground_truth, config.k);

        // Calculate percentiles
        latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = percentile(&latencies_us, 50.0);
        let p95 = percentile(&latencies_us, 95.0);
        let p99 = percentile(&latencies_us, 99.0);
        let qps = config.num_queries as f64 / total_time.as_secs_f64();

        results.push(BenchResult {
            name: config.name.to_string(),
            num_vectors: config.num_vectors,
            dim: config.dim,
            k: config.k,
            beam_width,
            build_time_ms: build_time.as_secs_f64() * 1000.0,
            build_throughput,
            query_time_us_p50: p50,
            query_time_us_p95: p95,
            query_time_us_p99: p99,
            qps,
            recall,
            index_size_mb: index_size,
        });
    }

    let _ = fs::remove_file(&index_path);
    results
}

/// Benchmark incremental operations
fn run_incremental_benchmark() {
    println!("\n{}", "=".repeat(80));
    println!("INCREMENTAL INDEX BENCHMARK");
    println!("{}", "=".repeat(80));

    let index_path = "bench_incremental.db";
    let _ = fs::remove_file(index_path);

    let dim = 128;
    let initial_size = 10_000;
    let batch_size = 1_000;
    let num_batches = 5;

    println!("\nConfiguration:");
    println!("  Initial vectors: {}", initial_size);
    println!("  Batch size: {}", batch_size);
    println!("  Number of batches: {}", num_batches);
    println!("  Dimensions: {}", dim);

    // Generate initial data
    let initial_data = generate_vectors(initial_size, dim, 42);

    // Build initial index
    let build_start = Instant::now();
    let index = IncrementalDiskANN::<DistL2>::build_default(&initial_data, index_path)
        .expect("Failed to build index");
    let build_time = build_start.elapsed();

    println!("\nInitial build: {:?} ({:.0} vec/s)",
             build_time, initial_size as f64 / build_time.as_secs_f64());

    // Measure incremental adds
    println!("\nIncremental add performance:");
    println!("{:>10} {:>12} {:>12} {:>12}", "Batch", "Add Time", "Vec/s", "Total Vecs");
    println!("{}", "-".repeat(50));

    for batch_num in 0..num_batches {
        let batch_data = generate_vectors(batch_size, dim, 100 + batch_num as u64);

        let add_start = Instant::now();
        index.add_vectors(&batch_data).expect("Failed to add vectors");
        let add_time = add_start.elapsed();

        let stats = index.stats();
        println!(
            "{:>10} {:>12.2?} {:>12.0} {:>12}",
            batch_num + 1,
            add_time,
            batch_size as f64 / add_time.as_secs_f64(),
            stats.total_live
        );
    }

    // Measure search performance with mixed base+delta
    let queries = generate_vectors(1000, dim, 999);
    let search_start = Instant::now();
    for q in &queries {
        let _ = index.search(q, 10, 64);
    }
    let search_time = search_start.elapsed();

    println!("\nSearch with delta layer:");
    println!("  1000 queries in {:?}", search_time);
    println!("  QPS: {:.0}", 1000.0 / search_time.as_secs_f64());

    // Measure delete performance
    let delete_ids: Vec<u64> = (0..100).map(|x| x as u64).collect();
    let delete_start = Instant::now();
    index.delete_vectors(&delete_ids).expect("Failed to delete");
    let delete_time = delete_start.elapsed();

    println!("\nDelete performance:");
    println!("  100 deletes in {:?}", delete_time);

    let _ = fs::remove_file(index_path);
}

/// Compare with a naive brute-force baseline
fn run_baseline_comparison() {
    println!("\n{}", "=".repeat(80));
    println!("BASELINE COMPARISON (DiskANN vs Brute Force)");
    println!("{}", "=".repeat(80));

    let configs = vec![
        (10_000, 128),
        (50_000, 128),
        (100_000, 128),
    ];

    println!("\n{:>10} {:>6} {:>12} {:>12} {:>10}",
             "Vectors", "Dim", "DiskANN(ms)", "BruteF(ms)", "Speedup");
    println!("{}", "-".repeat(60));

    for (num_vectors, dim) in configs {
        let data = generate_vectors(num_vectors, dim, 42);
        let queries = generate_vectors(100, dim, 123);
        let k = 10;
        let beam_width = 64;

        // Build DiskANN index
        let index_path = format!("bench_baseline_{}.db", num_vectors);
        let _ = fs::remove_file(&index_path);

        let index = DiskANN::<DistL2>::build_index_default(&data, DistL2 {}, &index_path)
            .expect("Failed to build");

        // Benchmark DiskANN
        let diskann_start = Instant::now();
        for q in &queries {
            let _ = index.search(q, k, beam_width);
        }
        let diskann_time = diskann_start.elapsed();

        // Benchmark brute force
        let brute_start = Instant::now();
        for q in &queries {
            let mut dists: Vec<(usize, f32)> = data
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let d: f32 = q.iter().zip(v).map(|(a, b)| (a - b).powi(2)).sum();
                    (i, d)
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let _: Vec<_> = dists.iter().take(k).collect();
        }
        let brute_time = brute_start.elapsed();

        let speedup = brute_time.as_secs_f64() / diskann_time.as_secs_f64();

        println!(
            "{:>10} {:>6} {:>12.2} {:>12.2} {:>10.1}x",
            num_vectors, dim,
            diskann_time.as_secs_f64() * 1000.0,
            brute_time.as_secs_f64() * 1000.0,
            speedup
        );

        let _ = fs::remove_file(&index_path);
    }
}

fn main() {
    println!("{}", "=".repeat(80));
    println!("           diskann-rs Professional Benchmark Suite");
    println!("{}", "=".repeat(80));
    println!("\nSystem: {} cores available", rayon::current_num_threads());

    // Define benchmark configurations
    // Use DISKANN_BENCH_LARGE=1 env var for full benchmarks
    let large_bench = std::env::var("DISKANN_BENCH_LARGE").is_ok();

    let configs = if large_bench {
        vec![
            BenchConfig {
                name: "small",
                num_vectors: 10_000,
                num_queries: 1_000,
                dim: 128,
                k: 10,
                beam_widths: vec![32, 64, 128],
            },
            BenchConfig {
                name: "medium",
                num_vectors: 100_000,
                num_queries: 1_000,
                dim: 128,
                k: 10,
                beam_widths: vec![64, 128, 256],
            },
            BenchConfig {
                name: "large",
                num_vectors: 500_000,
                num_queries: 500,
                dim: 128,
                k: 10,
                beam_widths: vec![128, 256],
            },
            BenchConfig {
                name: "high-dim",
                num_vectors: 50_000,
                num_queries: 500,
                dim: 768,
                k: 10,
                beam_widths: vec![64, 128],
            },
        ]
    } else {
        // Quick benchmarks (default) - runs in ~1 minute
        vec![
            BenchConfig {
                name: "small",
                num_vectors: 10_000,
                num_queries: 500,
                dim: 128,
                k: 10,
                beam_widths: vec![32, 64],
            },
            BenchConfig {
                name: "medium",
                num_vectors: 50_000,
                num_queries: 500,
                dim: 128,
                k: 10,
                beam_widths: vec![64, 128],
            },
        ]
    };

    let params = DiskAnnParams {
        max_degree: 64,
        build_beam_width: 128,
        alpha: 1.2,
    };

    // Run main benchmarks
    let mut all_results = Vec::new();
    for config in &configs {
        let results = run_benchmark(config, params);
        all_results.extend(results);
    }

    // Print summary table
    println!("\n{}", "=".repeat(120));
    println!("                              BENCHMARK RESULTS SUMMARY");
    BenchResult::print_header();
    for result in &all_results {
        result.print();
    }
    println!("{}", "=".repeat(120));

    // Run incremental benchmark
    run_incremental_benchmark();

    // Run baseline comparison
    run_baseline_comparison();

    println!("\n{}", "=".repeat(80));
    println!("Benchmarks complete!");
    println!("{}", "=".repeat(80));
}
