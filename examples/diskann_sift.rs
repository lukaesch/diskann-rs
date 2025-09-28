// examples/diskann_sift1m.rs
#![allow(clippy::needless_range_loop)]

use anndists::dist::DistL2;
use cpu_time::ProcessTime;
use diskann_rs::{DiskANN, DiskAnnError, DiskAnnParams};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

mod utils;
use utils::*;

fn euclid(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for j in 0..a.len() {
        let d = a[j] - b[j];
        s += d * d;
    }
    s.sqrt()
}

fn run_search(
    index: &Arc<DiskANN<DistL2>>,
    anndata: &annhdf5::AnnBenchmarkData,
    k: usize,
    beam_width: usize,
) {
    let nb_search = anndata.test_data.len();

    println!(
        "\nSearching {} queries with k={}, beam_width={} …",
        nb_search, k, beam_width
    );

    let start_cpu = ProcessTime::now();
    let start_wall = SystemTime::now();

    // Run DiskANN in parallel; collect distances of returned neighbors
    let results_dists: Vec<Vec<f32>> = anndata
        .test_data
        .par_iter()
        .map(|q| {
            let ids = index.search(q, k, beam_width);
            let mut ds = Vec::with_capacity(ids.len());
            for &id in &ids {
                let v = index.get_vector(id as usize);
                ds.push(euclid(q, &v));
            }
            ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
            ds
        })
        .collect();

    let cpu_time = start_cpu.elapsed();
    let wall_time = start_wall.elapsed().unwrap();

    // Compute recall
    let mut recalls: Vec<usize> = Vec::with_capacity(nb_search);
    let mut nb_returned: Vec<usize> = Vec::with_capacity(nb_search);
    let mut last_distances_ratio: Vec<f32> = Vec::with_capacity(nb_search);

    for i in 0..nb_search {
        let gt_row = anndata.test_distances.row(i);
        let true_k = k.min(gt_row.len());
        let gt_kth = gt_row[true_k - 1];

        let dists = &results_dists[i];
        nb_returned.push(dists.len());

        let recall = dists.iter().filter(|x| **x <= gt_kth).count();
        recalls.push(recall);

        let ratio = if !dists.is_empty() {
            dists[dists.len() - 1] / gt_kth
        } else {
            0.0
        };
        last_distances_ratio.push(ratio);
    }

    let mean_recall = (recalls.iter().sum::<usize>() as f32) / ((k * recalls.len()) as f32);
    let mean_frac_returned =
        (nb_returned.iter().sum::<usize>() as f32) / ((nb_returned.len() * k) as f32);
    let mean_last_ratio =
        last_distances_ratio.iter().sum::<f32>() / (last_distances_ratio.len() as f32);

    let search_sys_time_us = wall_time.as_micros() as f32;
    let req_per_s = (nb_search as f32) * 1.0e6_f32 / search_sys_time_us;

    println!(
        "\n mean fraction nb returned by search {:?}",
        mean_frac_returned
    );
    println!("\n last distances ratio {:?}", mean_last_ratio);
    println!(
        "\n recall rate for {:?} is {:?} , nb req /s {:?}",
        anndata.fname, mean_recall, req_per_s
    );
    println!(
        " total cpu time for search requests {:?} , system time {:?}",
        cpu_time, wall_time
    );
}

fn main() -> Result<(), DiskAnnError> {
    // SIFT1M (L2) HDF5 path
    // wget http://ann-benchmarks.com/sift-128-euclidean.hdf5
    let fname = String::from("./sift-128-euclidean.hdf5");
    println!("\n\nDiskANN benchmark on {:?}", fname);

    // Make this mutable so we can clear fields to free memory.
    let mut anndata =
        annhdf5::AnnBenchmarkData::new(fname.clone()).expect("Failed to load SIFT1M HDF5 file");

    let knbn_max = anndata.test_distances.dim().1;
    let nb_elem = anndata.train_data.len();
    let nb_search = anndata.test_data.len();

    println!("Train size : {}", nb_elem);
    println!("Test size  : {}", nb_search);
    println!("Ground-truth k per query in file: {}", knbn_max);

    // Build/open parameters
    let max_degree = 64;
    let build_beam_width = 128;
    let alpha = 1.2;

    let index_path = "diskann_sift1m.db";
    let index = if !std::path::Path::new(index_path).exists() {
        println!(
            "\nBuilding DiskANN index: n={}, dim={}, max_degree={}, build_beam={}, alpha={}",
            nb_elem,
            anndata.train_data[0].0.len(),
            max_degree,
            build_beam_width,
            alpha
        );

        // Clone the training matrix ONLY when we actually need to build.
        let train_vectors: Vec<Vec<f32>> = anndata
            .train_data
            .iter()
            .map(|pair| pair.0.clone())
            .collect();

        let params = DiskAnnParams {
            max_degree,
            build_beam_width,
            alpha,
        };

        let start_cpu = ProcessTime::now();
        let start_wall = SystemTime::now();

        let idx =
            DiskANN::<DistL2>::build_index_with_params(&train_vectors, DistL2 {}, index_path, params)?;

        let cpu_time: Duration = start_cpu.elapsed();
        let wall_time = start_wall.elapsed().unwrap();
        println!(
            "Build complete. CPU time: {:?}, wall time: {:?}",
            cpu_time, wall_time
        );

        // FREE HERE #1: we no longer need the cloned training matrix
        drop(train_vectors);

        // FREE HERE #2: we also no longer need the loader’s train_data
        anndata.train_data.clear();
        anndata.train_data.shrink_to_fit(); // may or may not return to OS immediately

        idx
    } else {
        println!("\nIndex file {} exists, opening…", index_path);
        let start_wall = SystemTime::now();
        let idx = DiskANN::<DistL2>::open_index_with(index_path, DistL2 {})?;
        let wall_time = start_wall.elapsed().unwrap();
        println!(
            "Opened index: {} vectors, dim={}, metric={} in {:?}",
            idx.num_vectors, idx.dim, idx.distance_name, wall_time
        );

        // FREE HERE #3: when not building, you never need train_data at all
        anndata.train_data.clear();
        anndata.train_data.shrink_to_fit();

        idx
    };

    // OPTIONAL BIG WIN:
    // If you want to free *everything* from anndata except what's needed for recall:
    // let test_data = std::mem::take(&mut anndata.test_data);           // moves Vec out
    // let test_distances = anndata.test_distances.clone();              // keep GT (or take if movable)
    // let fname_label = anndata.fname.clone();
    // drop(anndata);                                                    // FREE HERE #4: drop the whole loader
    // Then change run_search signature to accept (&test_data, &test_distances, &fname_label).

    let index = Arc::new(index);

    // If per-thread scratch is heavy, limit threads (helps RSS):
    // std::env::set_var("RAYON_NUM_THREADS", "8");

    // Evaluate at k=10, beam 256
    let k = 10.min(knbn_max);
    run_search(&index, &anndata, k, 512);

    Ok(())
}