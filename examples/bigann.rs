//! examples/bigann.rs
//!
//! BigANN for diskann-rs.
//! - expects these files in the repository root:
//!     bigann_base.bvecs, bigann_query.bvecs, dis_100M.fvecs, idx_100M.ivecs
//! - builds (or reuses) a DiskANN index at "big_diskann_index.db"
//! - runs recall@10 and recall@100 on the first 100k queries
//!
//! Notes:
//! - This example converts u8 BVECs to f32 and builds an *in-memory* index
//!   (then mmaps it). Building on the full dataset requires a lot of RAM.
//!   Adjust NB_DATA_POINTS to a subset if needed.

use anndists::dist::DistL2;
use byteorder::{LittleEndian, ReadBytesExt};
use diskann_rs::{DiskANN, DiskAnnParams};
use rayon::prelude::*;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, Read};
use std::path::Path;
use std::time::Instant;

const DIM: usize = 128;

// How many database vectors to ingest from bigann_base.bvecs.
// Set this to the size of your local shard (e.g., 1_000_000 or 10_000_000).
const NB_DATA_POINTS: usize = 10_000_000;

// Number of queries to evaluate.
const NB_QUERY: usize = 10_000;

// Disk path for the index (auto-reused if it exists).
const INDEX_PATH: &str = "big_diskann_index.db";

// DiskANN build/search knobs (feel free to tweak).
const DISKANN_PARAMS: DiskAnnParams = DiskAnnParams {
    max_degree: 48,
    build_beam_width: 200,
    alpha: 1.2,
};
const BEAM_SEARCH: usize = 512;

fn read_bvecs_block<const SIZE: usize>(
    r: &mut BufReader<File>,
    max_points: usize,
) -> io::Result<Vec<Vec<u8>>> {
    let mut out = Vec::with_capacity(max_points);
    let mut buf = [0u8; SIZE];
    for _ in 0..max_points {
        let dim = match r.read_u32::<LittleEndian>() {
            Ok(v) => v as usize,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        };
        if dim != SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("bvecs dim {} != {}", dim, SIZE),
            ));
        }
        r.read_exact(&mut buf)?;
        out.push(buf.to_vec());
    }
    Ok(out)
}

fn read_all_bvecs_prefix<const SIZE: usize>(
    path: &str,
    n_points: usize,
    block: usize,
) -> io::Result<Vec<Vec<u8>>> {
    let f = OpenOptions::new().read(true).open(path)?;
    let mut br = BufReader::new(f);
    let mut all = Vec::with_capacity(n_points.min(1_000_000)); // pre-guess
    let mut read_total = 0usize;
    while read_total < n_points {
        let want = block.min(n_points - read_total);
        let mut chunk = read_bvecs_block::<SIZE>(&mut br, want)?;
        if chunk.is_empty() {
            break;
        }
        read_total += chunk.len();
        all.append(&mut chunk);
    }
    Ok(all)
}

fn read_query_bvecs<const SIZE: usize>(path: &str, n_queries: usize) -> io::Result<Vec<Vec<u8>>> {
    let f = OpenOptions::new().read(true).open(path)?;
    let mut br = BufReader::new(f);
    let mut out = Vec::with_capacity(n_queries);
    let mut buf = [0u8; SIZE];
    for _ in 0..n_queries {
        let dim = br.read_u32::<LittleEndian>()? as usize;
        if dim != SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("query bvecs dim {} != {}", dim, SIZE),
            ));
        }
        br.read_exact(&mut buf)?;
        out.push(buf.to_vec());
    }
    Ok(out)
}

fn read_f32_block(r: &mut BufReader<File>) -> io::Result<Vec<f32>> {
    let dim = r.read_u32::<LittleEndian>()? as usize;
    let mut v = vec![0f32; dim];
    for x in &mut v {
        *x = r.read_f32::<LittleEndian>()?;
    }
    Ok(v)
}

fn read_u32_block(r: &mut BufReader<File>) -> io::Result<Vec<u32>> {
    let dim = r.read_u32::<LittleEndian>()? as usize;
    let mut v = vec![0u32; dim];
    for x in &mut v {
        *x = r.read_u32::<LittleEndian>()?;
    }
    Ok(v)
}

/// Returns ground truth as Vec[query] -> Vec[(id, sqdist)].
/// BigANN dis_*.fvecs contains **squared** L2; we `sqrt` later when comparing.
fn read_ground_truth(
    i_path: &str,
    f_path: &str,
    n_queries: usize,
) -> io::Result<Vec<Vec<(u32, f32)>>> {
    let fi = OpenOptions::new().read(true).open(i_path)?;
    let ff = OpenOptions::new().read(true).open(f_path)?;
    let mut ri = BufReader::new(fi);
    let mut rf = BufReader::new(ff);

    let mut ids = Vec::with_capacity(n_queries);
    let mut dists = Vec::with_capacity(n_queries);
    for _ in 0..n_queries {
        ids.push(read_u32_block(&mut ri)?);
        dists.push(read_f32_block(&mut rf)?);
    }
    let kn = ids[0].len();
    let mut gt = Vec::with_capacity(n_queries);
    for q in 0..n_queries {
        let mut row = Vec::with_capacity(kn);
        for k in 0..kn {
            row.push((ids[q][k], dists[q][k])); // squared L2 here
        }
        gt.push(row);
    }
    Ok(gt)
}

#[inline]
fn u8s_to_f32(v: &[u8]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

#[inline]
fn euclid(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s.sqrt()
}

fn build_or_load_index(base_path: &str, index_path: &str, n_points: usize) -> DiskANN<DistL2> {
    if Path::new(index_path).exists() {
        println!("Opening existing index at '{}'", index_path);
        return DiskANN::<DistL2>::open_index_default_metric(index_path)
            .expect("open_index_default_metric failed");
    }

    println!(
        "Building index from '{}' (first {} points)…",
        base_path, n_points
    );
    let t0 = Instant::now();

    // Read a prefix of the base BVECs
    let block = 50_000;
    let base_u8 = read_all_bvecs_prefix::<DIM>(base_path, n_points, block)
        .expect("failed reading base bvecs");
    assert!(!base_u8.is_empty(), "no base vectors read");

    // Convert to f32 (no normalization)
    let vectors: Vec<Vec<f32>> = base_u8.iter().map(|v| u8s_to_f32(v)).collect();

    println!(
        "Loaded {} vectors in {:.1}s; building DiskANN…",
        vectors.len(),
        t0.elapsed().as_secs_f32()
    );

    let t1 = Instant::now();
    let index =
        DiskANN::<DistL2>::build_index_with_params(&vectors, DistL2 {}, index_path, DISKANN_PARAMS)
            .expect("build_index_with_params failed");

    println!(
        "Build + write done in {:.1}s, {}",
        t1.elapsed().as_secs_f32(),
        index_path
    );

    index
}

fn eval_recall(
    index: &DiskANN<DistL2>,
    queries_f32: &[Vec<f32>],
    gt: &[Vec<(u32, f32)>], // (id, sqdist)
    k: usize,
    beam: usize,
) {
    let t0 = Instant::now();

    // Parallel map-reduce over queries
    let correct: usize = queries_f32
        .par_iter()
        .enumerate()
        .map(|(qi, q)| {
            let nns = index.search(q, k, beam);
            let kth = gt[qi][k - 1].1.sqrt();

            // Count how many returned are within GT@k radius
            let mut local_correct = 0usize;
            for &id in &nns {
                // If you later add a zero-copy accessor, prefer:
                // let v = index.get_vector_slice(id as usize);
                // let d = euclid(q, v);
                let v = index.get_vector(id as usize);
                let d = euclid(q, &v);
                if d <= kth {
                    local_correct += 1;
                }
            }
            local_correct
        })
        .sum();

    let secs = t0.elapsed().as_secs_f32();
    let recall = (correct as f32) / ((k * queries_f32.len()) as f32);
    let qps = (queries_f32.len() as f32) / secs;

    println!(
        "k={k:>3}  recall={:.4}  qps={:.1}  time={:.1}s  (beam={})",
        recall, qps, secs, beam
    );
}

// Single-threaded
fn eval_recall_single(
    index: &DiskANN<DistL2>,
    queries_f32: &[Vec<f32>],
    gt: &[Vec<(u32, f32)>], // (id, sqdist)
    k: usize,
    beam: usize,
) {
    assert_eq!(queries_f32.len(), gt.len());
    let t0 = Instant::now();

    let mut correct = 0usize;
    for (qi, q) in queries_f32.iter().enumerate() {
        let nns = index.search(q, k, beam);
        let kth = gt[qi][k - 1].1.sqrt();

        for &id in &nns {
            let v = index.get_vector(id as usize);
            let d = euclid(q, &v);
            if d <= kth {
                correct += 1;
            }
        }
    }

    let secs = t0.elapsed().as_secs_f32();
    let recall = (correct as f32) / ((k * queries_f32.len()) as f32);
    let qps = (queries_f32.len() as f32) / secs;

    println!(
        "k={k:>3}  recall={:.4}  qps={:.1}  time={:.1}s  (beam={})",
        recall, qps, secs, beam
    );
}

fn main() {
    // Toggle this to choose evaluation mode
    const PARALLEL: bool = true;

    // Filenames in repo root
    // download all data here: http://corpus-texmex.irisa.fr (ANN_SIFT1B)
    let base_path = "bigann_base.bvecs";
    let query_path = "bigann_query.bvecs";
    let gt_i_path = "idx_10M.ivecs";
    let gt_f_path = "dis_10M.fvecs";

    // Build or open index
    let index = build_or_load_index(base_path, INDEX_PATH, NB_DATA_POINTS);

    // Read queries
    println!("Reading first {} queries from {}…", NB_QUERY, query_path);
    let queries_u8 = read_query_bvecs::<DIM>(query_path, NB_QUERY).expect("failed reading queries");
    let queries_f32: Vec<Vec<f32>> = queries_u8.iter().map(|v| u8s_to_f32(v)).collect();

    // Read ground truth (10M set)
    println!("Reading ground truth from {}, {}…", gt_i_path, gt_f_path);
    let gt =
        read_ground_truth(gt_i_path, gt_f_path, NB_QUERY).expect("failed reading ground truth");
    let kn = gt[0].len();
    println!("GT loaded: {} queries, GT@{} per query", gt.len(), kn);

    // Evaluate
    if PARALLEL {
        eval_recall(&index, &queries_f32, &gt, 10, BEAM_SEARCH);
        eval_recall(&index, &queries_f32, &gt, 100, BEAM_SEARCH);
    } else {
        eval_recall_single(&index, &queries_f32, &gt, 10, BEAM_SEARCH);
        eval_recall_single(&index, &queries_f32, &gt, 100, BEAM_SEARCH);
    }
}
