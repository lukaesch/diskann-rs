//! Comprehensive Benchmark Suite: diskann-rs vs hnsw_rs
//!
//! Generates SVG charts comparing:
//! 1. Build time at different scales
//! 2. Recall@10 vs QPS trade-off
//! 3. Memory usage during search
//! 4. Incremental update performance
//!
//! Run with: cargo bench --release --bench comparison

use anndists::dist::DistL2;
use diskann_rs::{DiskANN, DiskAnnParams, IncrementalDiskANN};
use hnsw_rs::prelude::*;
use plotters::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs;
use std::time::Instant;
use sysinfo::System;

const DIM: usize = 128;
const CHART_DIR: &str = "docs/charts";

// ============================================================================
// Data Generation
// ============================================================================

fn generate_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.r#gen::<f32>()).collect())
        .collect()
}

fn compute_ground_truth(queries: &[Vec<f32>], data: &[Vec<f32>], k: usize) -> Vec<Vec<usize>> {
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

fn calculate_recall(results: &[Vec<usize>], ground_truth: &[Vec<usize>], k: usize) -> f64 {
    let mut correct = 0usize;
    let total = results.len() * k;
    for (res, gt) in results.iter().zip(ground_truth.iter()) {
        let gt_set: HashSet<_> = gt.iter().take(k).collect();
        for id in res.iter().take(k) {
            if gt_set.contains(id) {
                correct += 1;
            }
        }
    }
    correct as f64 / total as f64
}

// ============================================================================
// Benchmark 1: Build Time
// ============================================================================

#[derive(Debug, Clone)]
struct BuildResult {
    library: String,
    num_vectors: usize,
    build_time_s: f64,
    throughput: f64,
}

fn benchmark_build_time() -> Vec<BuildResult> {
    println!("\n{}", "=".repeat(60));
    println!("BENCHMARK 1: Build Time Comparison");
    println!("{}", "=".repeat(60));

    let sizes = vec![10_000, 50_000, 100_000];
    let mut results = Vec::new();

    for &n in &sizes {
        println!("\n--- {} vectors, dim={} ---", n, DIM);
        let data = generate_vectors(n, DIM, 42);

        // diskann-rs
        {
            let path = format!("/tmp/bench_diskann_{}.db", n);
            let _ = fs::remove_file(&path);

            let start = Instant::now();
            let _index = DiskANN::<DistL2>::build_index_with_params(
                &data,
                DistL2 {},
                &path,
                DiskAnnParams {
                    max_degree: 32,
                    build_beam_width: 64,
                    alpha: 1.2,
                },
            )
            .unwrap();
            let elapsed = start.elapsed().as_secs_f64();

            println!("  diskann-rs: {:.2}s ({:.0} vec/s)", elapsed, n as f64 / elapsed);
            results.push(BuildResult {
                library: "diskann-rs".into(),
                num_vectors: n,
                build_time_s: elapsed,
                throughput: n as f64 / elapsed,
            });

            let _ = fs::remove_file(&path);
        }

        // hnsw_rs
        {
            let start = Instant::now();
            let hnsw = Hnsw::<f32, DistL2>::new(32, n, 16, 64, DistL2 {});

            let data_refs: Vec<(&Vec<f32>, usize)> = data.iter().zip(0..n).collect();
            hnsw.parallel_insert(&data_refs);

            let elapsed = start.elapsed().as_secs_f64();

            println!("  hnsw_rs:    {:.2}s ({:.0} vec/s)", elapsed, n as f64 / elapsed);
            results.push(BuildResult {
                library: "hnsw_rs".into(),
                num_vectors: n,
                build_time_s: elapsed,
                throughput: n as f64 / elapsed,
            });
        }
    }

    results
}

fn plot_build_time(results: &[BuildResult]) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(CHART_DIR)?;
    let path = format!("{}/build_time.svg", CHART_DIR);

    let root = SVGBackend::new(&path, (800, 500)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_time = results.iter().map(|r| r.build_time_s).fold(0.0, f64::max) * 1.1;

    let mut chart = ChartBuilder::on(&root)
        .caption("Build Time Comparison", ("sans-serif", 28).into_font())
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(0usize..110_000usize, 0f64..max_time)?;

    chart
        .configure_mesh()
        .x_desc("Number of Vectors")
        .y_desc("Build Time (seconds)")
        .x_label_formatter(&|x| format!("{}K", x / 1000))
        .draw()?;

    // diskann-rs line
    let diskann_data: Vec<(usize, f64)> = results
        .iter()
        .filter(|r| r.library == "diskann-rs")
        .map(|r| (r.num_vectors, r.build_time_s))
        .collect();

    chart
        .draw_series(LineSeries::new(diskann_data.clone(), &BLUE))?
        .label("diskann-rs")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    chart.draw_series(diskann_data.iter().map(|(x, y)| Circle::new((*x, *y), 5, BLUE.filled())))?;

    // hnsw_rs line
    let hnsw_data: Vec<(usize, f64)> = results
        .iter()
        .filter(|r| r.library == "hnsw_rs")
        .map(|r| (r.num_vectors, r.build_time_s))
        .collect();

    chart
        .draw_series(LineSeries::new(hnsw_data.clone(), &RED))?
        .label("hnsw_rs")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    chart.draw_series(hnsw_data.iter().map(|(x, y)| Circle::new((*x, *y), 5, RED.filled())))?;

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .position(SeriesLabelPosition::UpperLeft)
        .draw()?;

    root.present()?;
    println!("\nChart saved: {}", path);
    Ok(())
}

// ============================================================================
// Benchmark 2: Recall vs QPS
// ============================================================================

#[derive(Debug, Clone)]
struct RecallQpsResult {
    library: String,
    param: usize,
    recall: f64,
    qps: f64,
}

fn benchmark_recall_qps() -> Vec<RecallQpsResult> {
    println!("\n{}", "=".repeat(60));
    println!("BENCHMARK 2: Recall@10 vs QPS Trade-off");
    println!("{}", "=".repeat(60));

    let n = 50_000;
    let n_queries = 1000;
    let k = 10;

    println!("\nGenerating {} vectors, {} queries...", n, n_queries);
    let data = generate_vectors(n, DIM, 42);
    let queries = generate_vectors(n_queries, DIM, 123);

    println!("Computing ground truth...");
    let ground_truth = compute_ground_truth(&queries, &data, k);

    let mut results = Vec::new();

    let diskann_path = "/tmp/bench_recall_diskann.db";
    let _ = fs::remove_file(diskann_path);

    println!("\nBuilding diskann-rs index...");
    let diskann_index = DiskANN::<DistL2>::build_index_with_params(
        &data,
        DistL2 {},
        diskann_path,
        DiskAnnParams {
            max_degree: 64,
            build_beam_width: 128,
            alpha: 1.2,
        },
    )
    .unwrap();

    println!("Building hnsw_rs index...");
    // Hnsw::new(max_nb_conn M, capacity, max_layer, ef_construction, distance)
    let mut hnsw = Hnsw::<f32, DistL2>::new(64, n, 16, 128, DistL2 {});
    let data_refs: Vec<(&Vec<f32>, usize)> = data.iter().zip(0..n).collect();
    hnsw.parallel_insert(&data_refs);

    let params = vec![16, 32, 64, 128, 256, 512];

    println!("\n{:>10} {:>10} {:>10} {:>10}", "Library", "Param", "Recall", "QPS");
    println!("{}", "-".repeat(45));

    for &param in &params {
        // diskann-rs
        {
            let start = Instant::now();
            let search_results: Vec<Vec<usize>> = queries
                .iter()
                .map(|q| {
                    diskann_index
                        .search(q, k, param)
                        .into_iter()
                        .map(|id| id as usize)
                        .collect()
                })
                .collect();
            let elapsed = start.elapsed().as_secs_f64();

            let recall = calculate_recall(&search_results, &ground_truth, k);
            let qps = n_queries as f64 / elapsed;

            println!("{:>10} {:>10} {:>10.4} {:>10.0}", "diskann-rs", param, recall, qps);
            results.push(RecallQpsResult {
                library: "diskann-rs".into(),
                param,
                recall,
                qps,
            });
        }

        // hnsw_rs
        {
            hnsw.set_searching_mode(true);

            let start = Instant::now();
            let search_results: Vec<Vec<usize>> = queries
                .iter()
                .map(|q| {
                    let res = hnsw.search(q, k, param);
                    res.into_iter().map(|n| n.d_id).collect()
                })
                .collect();
            let elapsed = start.elapsed().as_secs_f64();

            let recall = calculate_recall(&search_results, &ground_truth, k);
            let qps = n_queries as f64 / elapsed;

            println!("{:>10} {:>10} {:>10.4} {:>10.0}", "hnsw_rs", param, recall, qps);
            results.push(RecallQpsResult {
                library: "hnsw_rs".into(),
                param,
                recall,
                qps,
            });
        }
    }

    let _ = fs::remove_file(diskann_path);
    results
}

fn plot_recall_qps(results: &[RecallQpsResult]) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{}/recall_vs_qps.svg", CHART_DIR);

    let root = SVGBackend::new(&path, (800, 500)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_qps = results.iter().map(|r| r.qps).fold(0.0, f64::max) * 1.1;

    let mut chart = ChartBuilder::on(&root)
        .caption("Recall@10 vs QPS Trade-off", ("sans-serif", 28).into_font())
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d(0.5f64..1.0f64, 0f64..max_qps)?;

    chart
        .configure_mesh()
        .x_desc("Recall@10")
        .y_desc("Queries Per Second (QPS)")
        .x_label_formatter(&|x| format!("{:.0}%", x * 100.0))
        .draw()?;

    // diskann-rs
    let diskann_data: Vec<(f64, f64)> = results
        .iter()
        .filter(|r| r.library == "diskann-rs")
        .map(|r| (r.recall, r.qps))
        .collect();

    chart
        .draw_series(LineSeries::new(diskann_data.clone(), &BLUE))?
        .label("diskann-rs")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    chart.draw_series(diskann_data.iter().map(|(x, y)| Circle::new((*x, *y), 5, BLUE.filled())))?;

    // hnsw_rs
    let hnsw_data: Vec<(f64, f64)> = results
        .iter()
        .filter(|r| r.library == "hnsw_rs")
        .map(|r| (r.recall, r.qps))
        .collect();

    chart
        .draw_series(LineSeries::new(hnsw_data.clone(), &RED))?
        .label("hnsw_rs")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    chart.draw_series(hnsw_data.iter().map(|(x, y)| Circle::new((*x, *y), 5, RED.filled())))?;

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;

    root.present()?;
    println!("Chart saved: {}", path);
    Ok(())
}

// ============================================================================
// Benchmark 3: Memory Usage Under Different Search Scenarios
// ============================================================================

#[derive(Debug, Clone)]
struct MemoryResult {
    scenario: String,
    library: String,
    beam_width: usize,
    num_queries: usize,
    ram_used_mb: f64,
}

fn measure_process_rss() -> f64 {
    // Use ps to get RSS (Resident Set Size) - actual physical memory used
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok();

    if let Some(out) = output {
        let rss_kb: f64 = String::from_utf8_lossy(&out.stdout)
            .trim()
            .parse()
            .unwrap_or(0.0);
        rss_kb / 1024.0 // Convert to MB
    } else {
        0.0
    }
}

fn benchmark_memory() -> Vec<MemoryResult> {
    println!("\n{}", "=".repeat(60));
    println!("BENCHMARK 3: Memory Usage Under Different Scenarios");
    println!("{}", "=".repeat(60));

    let n = 200_000; // Large enough to see memory effects
    let k = 10;

    println!("\nGenerating {} vectors (dim={})...", n, DIM);
    let data = generate_vectors(n, DIM, 42);

    let mut results = Vec::new();

    // Scenario definitions: (name, num_queries, beam_width)
    let scenarios = vec![
        ("Light (10q, beam=32)", 10, 32),
        ("Medium (100q, beam=64)", 100, 64),
        ("Heavy (1000q, beam=128)", 1000, 128),
        ("Stress (5000q, beam=256)", 5000, 256),
    ];

    // Build diskann-rs index
    let diskann_path = "/tmp/bench_mem_diskann.db";
    let _ = fs::remove_file(diskann_path);

    println!("Building diskann-rs index...");
    {
        let _diskann_index = DiskANN::<DistL2>::build_index_default(&data, DistL2 {}, diskann_path).unwrap();
        // Index built and dropped, file remains on disk
    }
    let index_file_mb = fs::metadata(diskann_path)
        .map(|m| m.len() as f64 / 1024.0 / 1024.0)
        .unwrap_or(0.0);

    // Build hnsw_rs index and measure its RAM requirement
    println!("Building hnsw_rs index...");
    let hnsw_before = measure_process_rss();
    let mut hnsw = Hnsw::<f32, DistL2>::new(64, n, 16, 128, DistL2 {});
    let data_refs: Vec<(&Vec<f32>, usize)> = data.iter().zip(0..n).collect();
    hnsw.parallel_insert(&data_refs);
    hnsw.set_searching_mode(true);
    let hnsw_after = measure_process_rss();
    let hnsw_base_ram = (hnsw_after - hnsw_before).max(50.0);

    println!("\nIndex file size: {:.1} MB", index_file_mb);
    println!("hnsw_rs base RAM: {:.1} MB (must hold full index)", hnsw_base_ram);

    println!("\n{:<30} {:>15} {:>15}", "Scenario", "diskann-rs", "hnsw_rs");
    println!("{}", "-".repeat(62));

    for (scenario_name, num_queries, beam_width) in &scenarios {
        let queries = generate_vectors(*num_queries, DIM, 999);

        // Purge disk cache if possible (best effort)
        #[cfg(target_os = "macos")]
        {
            let _ = std::process::Command::new("purge").output();
        }

        // Small sleep to let OS settle
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Open fresh index (no cached pages)
        let diskann_index = DiskANN::<DistL2>::open_index_with(diskann_path, DistL2 {}).unwrap();

        // Measure diskann-rs RSS before and after search
        let diskann_before = measure_process_rss();

        for q in &queries {
            let _ = diskann_index.search(q, k, *beam_width);
        }

        let diskann_after = measure_process_rss();
        let diskann_ram = (diskann_after - diskann_before).abs().max(1.0);

        // hnsw_rs: RAM is constant (full index always in memory)
        // Run search just to be fair, but RAM won't change
        for q in &queries {
            let _ = hnsw.search(q, k, *beam_width);
        }

        println!(
            "{:<30} {:>12.1} MB {:>12.1} MB",
            scenario_name, diskann_ram, hnsw_base_ram
        );

        results.push(MemoryResult {
            scenario: scenario_name.to_string(),
            library: "diskann-rs".into(),
            beam_width: *beam_width,
            num_queries: *num_queries,
            ram_used_mb: diskann_ram,
        });

        results.push(MemoryResult {
            scenario: scenario_name.to_string(),
            library: "hnsw_rs".into(),
            beam_width: *beam_width,
            num_queries: *num_queries,
            ram_used_mb: hnsw_base_ram,
        });

        // Drop index to release mmap
        drop(diskann_index);
    }

    println!("\nKey observations:");
    println!("  - diskann-rs RAM varies with workload (pages loaded on-demand)");
    println!("  - hnsw_rs requires {:.0} MB RAM constantly (full index)", hnsw_base_ram);
    println!("  - Under memory pressure, OS evicts diskann-rs pages automatically");

    let _ = fs::remove_file(diskann_path);

    results
}

fn plot_memory(results: &[MemoryResult]) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{}/memory_usage.svg", CHART_DIR);

    let root = SVGBackend::new(&path, (800, 450)).into_drawing_area();
    root.fill(&WHITE)?;

    // Extract data
    let scenario_labels = ["Light", "Medium", "Heavy", "Stress"];
    let scenario_subtitles = ["10 queries", "100 queries", "1K queries", "5K queries"];

    let diskann_data: Vec<f64> = results
        .iter()
        .filter(|r| r.library == "diskann-rs")
        .map(|r| r.ram_used_mb)
        .collect();

    let hnsw_data: Vec<f64> = results
        .iter()
        .filter(|r| r.library == "hnsw_rs")
        .map(|r| r.ram_used_mb)
        .collect();

    let max_ram = results.iter().map(|r| r.ram_used_mb).fold(0.0, f64::max) * 1.2;

    // Split root into chart area (top) and label area (bottom)
    let (chart_area, label_area) = root.split_vertically(390);

    let mut chart = ChartBuilder::on(&chart_area)
        .caption("RAM Usage: Memory-Mapped vs In-Memory (200K vectors)", ("sans-serif", 20).into_font())
        .margin(20)
        .x_label_area_size(30) // Reduced since we draw labels ourselves
        .y_label_area_size(70)
        .build_cartesian_2d(0f64..4.5f64, 0f64..max_ram)?;

    chart
        .configure_mesh()
        .y_desc("RAM Usage (MB)")
        .y_label_formatter(&|y| format!("{:.0}", y))
        .disable_x_mesh()
        .disable_x_axis()  // We'll draw our own x-axis labels
        .draw()?;

    let bar_width = 0.35;
    let group_width = 1.0;

    // Draw bars for each scenario
    for i in 0..4 {
        let base_x = i as f64 * group_width + 0.15;
        let _group_center = base_x + bar_width + 0.025; // Center between the two bars (used for calculating x-axis label positions below)

        // hnsw_rs bar (red) - left
        if i < hnsw_data.len() {
            chart.draw_series(std::iter::once(Rectangle::new(
                [(base_x, 0.0), (base_x + bar_width, hnsw_data[i])],
                RED.mix(0.8).filled(),
            )))?;
        }

        // diskann-rs bar (blue) - right
        if i < diskann_data.len() {
            let x = base_x + bar_width + 0.05;
            chart.draw_series(std::iter::once(Rectangle::new(
                [(x, 0.0), (x + bar_width, diskann_data[i])],
                BLUE.mix(0.8).filled(),
            )))?;

            // Add value label on diskann-rs bar
            chart.draw_series(std::iter::once(Text::new(
                format!("{:.0} MB", diskann_data[i]),
                (x + bar_width / 2.0, diskann_data[i] + max_ram * 0.02),
                ("sans-serif", 10).into_font().color(&BLACK),
            )))?;
        }

    }

    // Draw x-axis labels in the label area (below the chart)
    // Chart starts at x=90 (margin 20 + y_label_area 70)
    let chart_left = 90.0;
    let chart_width = 689.0; // 779 - 90

    for i in 0..4 {
        let group_center = chart_left + (i as f64 * 1.0 + 0.5) * (chart_width / 4.5);

        // Draw main label (scenario name) - y=15 in label area (which starts at y=390 in full canvas)
        label_area.draw(&plotters::prelude::Text::new(
            scenario_labels[i],
            (group_center as i32, 15),
            ("sans-serif", 12).into_font().color(&BLACK),
        ))?;

        // Draw subtitle (query count) - y=32 in label area
        label_area.draw(&plotters::prelude::Text::new(
            scenario_subtitles[i],
            (group_center as i32, 32),
            ("sans-serif", 9).into_font().color(&BLACK.mix(0.6)),
        ))?;
    }

    // Add hnsw_rs label (only once, since it's constant)
    if !hnsw_data.is_empty() {
        chart.draw_series(std::iter::once(Text::new(
            format!("{:.0} MB", hnsw_data[0]),
            (0.15 + bar_width / 2.0, hnsw_data[0] + max_ram * 0.02),
            ("sans-serif", 10).into_font().color(&BLACK),
        )))?;
    }

    // Add legend
    chart
        .draw_series(std::iter::once(Rectangle::new(
            [(0.0, 0.0), (0.0, 0.0)],
            RED.filled(),
        )))?
        .label("hnsw_rs (full index in RAM)")
        .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 15, y + 5)], RED.filled()));

    chart
        .draw_series(std::iter::once(Rectangle::new(
            [(0.0, 0.0), (0.0, 0.0)],
            BLUE.filled(),
        )))?
        .label("diskann-rs (mmap, on-demand)")
        .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 15, y + 5)], BLUE.filled()));

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.9))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    println!("Chart saved: {}", path);
    Ok(())
}

// ============================================================================
// Benchmark 4: Incremental Updates
// ============================================================================

#[derive(Debug, Clone)]
struct IncrementalResult {
    library: String,
    operation: String,
    vectors_per_sec: f64,
}

fn benchmark_incremental() -> Vec<IncrementalResult> {
    println!("\n{}", "=".repeat(60));
    println!("BENCHMARK 4: Incremental Updates");
    println!("{}", "=".repeat(60));

    let initial_n = 10_000;
    let add_n = 1_000;
    let delete_n = 100;

    let initial_data = generate_vectors(initial_n, DIM, 42);
    let new_data = generate_vectors(add_n, DIM, 123);

    let mut results = Vec::new();

    // diskann-rs (has incremental support)
    {
        let path = "/tmp/bench_incr_diskann.db";
        let _ = fs::remove_file(path);

        let index = IncrementalDiskANN::<DistL2>::build_default(&initial_data, path).unwrap();

        // Benchmark add
        let start = Instant::now();
        index.add_vectors(&new_data).unwrap();
        let add_elapsed = start.elapsed().as_secs_f64();
        let add_rate = add_n as f64 / add_elapsed;

        println!("diskann-rs ADD:    {:.0} vectors/sec", add_rate);
        results.push(IncrementalResult {
            library: "diskann-rs".into(),
            operation: "add".into(),
            vectors_per_sec: add_rate,
        });

        // Benchmark delete
        let delete_ids: Vec<u64> = (0..delete_n as u64).collect();
        let start = Instant::now();
        index.delete_vectors(&delete_ids).unwrap();
        let del_elapsed = start.elapsed().as_secs_f64();
        let del_rate = delete_n as f64 / del_elapsed;

        println!("diskann-rs DELETE: {:.0} vectors/sec", del_rate);
        results.push(IncrementalResult {
            library: "diskann-rs".into(),
            operation: "delete".into(),
            vectors_per_sec: del_rate,
        });

        let _ = fs::remove_file(path);
    }

    // hnsw_rs - can add but cannot delete
    {
        let hnsw = Hnsw::<f32, DistL2>::new(32, initial_n + add_n, 16, 64, DistL2 {});
        let data_refs: Vec<(&Vec<f32>, usize)> = initial_data.iter().zip(0..initial_n).collect();
        hnsw.parallel_insert(&data_refs);

        // Benchmark add (serial insertion)
        let start = Instant::now();
        for (i, v) in new_data.iter().enumerate() {
            hnsw.insert((v, initial_n + i));
        }
        let add_elapsed = start.elapsed().as_secs_f64();
        let add_rate = add_n as f64 / add_elapsed;

        println!("hnsw_rs ADD:       {:.0} vectors/sec", add_rate);
        results.push(IncrementalResult {
            library: "hnsw_rs".into(),
            operation: "add".into(),
            vectors_per_sec: add_rate,
        });

        // hnsw_rs has no delete
        println!("hnsw_rs DELETE:    N/A (requires full rebuild)");
        results.push(IncrementalResult {
            library: "hnsw_rs".into(),
            operation: "delete".into(),
            vectors_per_sec: 0.0,
        });
    }

    results
}

fn plot_incremental(results: &[IncrementalResult]) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{}/incremental_updates.svg", CHART_DIR);

    let root = SVGBackend::new(&path, (600, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    // Only show ADD comparison - DELETE is not comparable (instant tombstone vs full rebuild)
    let diskann_add = results
        .iter()
        .find(|r| r.library == "diskann-rs" && r.operation == "add")
        .map(|r| r.vectors_per_sec)
        .unwrap_or(0.0);

    let hnsw_add = results
        .iter()
        .find(|r| r.library == "hnsw_rs" && r.operation == "add")
        .map(|r| r.vectors_per_sec)
        .unwrap_or(0.0);

    let max_rate = diskann_add.max(hnsw_add) * 1.3;

    let mut chart = ChartBuilder::on(&root)
        .caption("Incremental Add Performance", ("sans-serif", 22).into_font())
        .margin(20)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(0f64..3f64, 0f64..max_rate)?;

    chart
        .configure_mesh()
        .y_desc("Vectors / Second")
        .y_label_formatter(&|y| {
            if *y >= 1_000.0 {
                format!("{:.0}K", y / 1_000.0)
            } else {
                format!("{:.0}", y)
            }
        })
        .disable_x_mesh()
        .x_labels(2)
        .x_label_formatter(&|x| {
            if *x < 1.0 {
                "diskann-rs".to_string()
            } else if *x < 2.0 {
                "hnsw_rs".to_string()
            } else {
                "".to_string()
            }
        })
        .draw()?;

    // diskann-rs bar (blue)
    chart.draw_series(std::iter::once(Rectangle::new(
        [(0.2, 0.0), (0.8, diskann_add)],
        BLUE.mix(0.8).filled(),
    )))?;

    // hnsw_rs bar (red)
    chart.draw_series(std::iter::once(Rectangle::new(
        [(1.2, 0.0), (1.8, hnsw_add)],
        RED.mix(0.8).filled(),
    )))?;

    // Add value labels on bars
    let diskann_label = format!("{:.0}K/s", diskann_add / 1000.0);
    let hnsw_label = format!("{:.0}K/s", hnsw_add / 1000.0);

    chart.draw_series(std::iter::once(Text::new(
        diskann_label,
        (0.5, diskann_add + max_rate * 0.03),
        ("sans-serif", 14).into_font().color(&BLACK),
    )))?;

    chart.draw_series(std::iter::once(Text::new(
        hnsw_label,
        (1.5, hnsw_add + max_rate * 0.03),
        ("sans-serif", 14).into_font().color(&BLACK),
    )))?;

    // Add speedup annotation
    let speedup = diskann_add / hnsw_add;
    chart.draw_series(std::iter::once(Text::new(
        format!("{:.0}x faster", speedup),
        (0.5, diskann_add * 0.5),
        ("sans-serif", 16).into_font().color(&WHITE),
    )))?;

    root.present()?;
    println!("Chart saved: {}", path);

    // Also create a separate delete comparison note in console
    let diskann_del = results
        .iter()
        .find(|r| r.library == "diskann-rs" && r.operation == "delete")
        .map(|r| r.vectors_per_sec)
        .unwrap_or(0.0);

    if diskann_del > 0.0 {
        println!(
            "Note: diskann-rs DELETE: {:.0} vec/s (instant tombstone), hnsw_rs: requires full rebuild",
            diskann_del
        );
    }

    Ok(())
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     diskann-rs vs hnsw_rs Benchmark Suite                  ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!("\nSystem: {} cores", rayon::current_num_threads());

    fs::create_dir_all(CHART_DIR).unwrap();

    // Run benchmarks
    let build_results = benchmark_build_time();
    plot_build_time(&build_results).unwrap();

    let recall_qps_results = benchmark_recall_qps();
    plot_recall_qps(&recall_qps_results).unwrap();

    let memory_results = benchmark_memory();
    plot_memory(&memory_results).unwrap();

    let incremental_results = benchmark_incremental();
    plot_incremental(&incremental_results).unwrap();

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║     All benchmarks complete!                               ║");
    println!("║     Charts saved to: docs/charts/                          ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!("\nGenerated charts:");
    println!("  - {}/build_time.svg", CHART_DIR);
    println!("  - {}/recall_vs_qps.svg", CHART_DIR);
    println!("  - {}/memory_usage.svg", CHART_DIR);
    println!("  - {}/incremental_updates.svg", CHART_DIR);
}
