//! usearch-vs-turbovec ANN backend bench-off (ADR-0004 Phase 2, issue #76).
//!
//! This is the data that decides which ANN backend the retrieval path adopts.
//! It lives as an `#[ignore]`d unit test *inside* the crate for two reasons:
//!
//!   1. The crate is `crate-type = ["cdylib"]` (a PyO3 extension). On macOS any
//!      `cargo build` / `cargo bench` / `cargo run` emits the cdylib and fails
//!      to link (no libpython). Only `cargo test` links — it builds a test
//!      harness and never the cdylib — so the bench-off must be a test.
//!   2. It needs crate-internal APIs (`crate::data::triples`,
//!      `crate::models::two_tower`, `crate::ann::*`) that aren't part of the
//!      public surface.
//!
//! Run it explicitly (release; debug `burn` is impractically slow):
//! ```text
//! cargo test --release --features "ann ml-models" \
//!     -- --ignored ann_backend_comparison --nocapture
//! ```
//! The `#[ignore]` keeps it out of the normal (fast) `cargo test` suite.
//!
//! ANN recall is only meaningful on *structured* embeddings, so we don't feed
//! random noise: we synthesize CLUSTERED interactions (items belong to a
//! handful of latent clusters; each synthetic user draws its positives
//! predominantly from one cluster), train a real Two-Tower, and bench against
//! the trained item embeddings — which acquire genuine neighborhood structure.
//!
//! ## Scale runs (issue #82)
//!
//! The bench is parameterized via env vars so CI Linux can run it at the
//! scale where ANN earns its keep (default values keep the local run fast):
//!
//! - `ANN_BENCH_N_ITEMS`  (default 3000)
//! - `ANN_BENCH_DIM`      (default 64; must be a multiple of 8 — turbovec)
//! - `ANN_BENCH_K`        (default 10)
//! - `ANN_BENCH_MODE`     (`trained` | `synthetic`, default `trained`)
//! - `ANN_BENCH_CLUSTERS` (default 12; scale runs should scale this with N —
//!   a fixed 12 at 1M items means ~83k near-duplicate vectors per cluster,
//!   a pathologically hard geometry for graph indexes)
//!
//! `synthetic` skips Two-Tower training and synthesizes clustered unit
//! vectors directly. Training at 100k–1M items is impractical on hosted CI
//! (burn-ndarray CPU); the recall *decision* was made on real trained
//! embeddings at 3000 items (#76), and at scale the latency/memory ratios
//! this follow-up measures are properties of catalog size and geometry, which
//! synthesized clusters preserve.

use crate::ann::exact::{exact_top_k, recall_at_k};
use crate::ann::turbovec_backend::TurbovecBackend;
use crate::ann::usearch_backend::UsearchBackend;
use crate::ann::{AnnBackend, ExactBackend};
use crate::data::triples::{FeatureTable, Triple, TripleData};
use crate::models::two_tower::{TrainParams, train};
use ahash::AHashMap;

/// A tiny deterministic LCG so the synthetic data is reproducible without
/// pulling in (or seeding) a heavier RNG. Numerical-Recipes constants.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    /// Uniform in `[0, n)`.
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Build CLUSTERED `(user, item)` positive triples directly (no file I/O):
/// `n_items` items split evenly across `n_clusters` latent clusters, and
/// `n_users` synthetic users each tied to one cluster, drawing `pos_per_user`
/// positives — 85% from their own cluster, 15% leaked uniformly across the
/// catalog. The cluster signal gives the trained item embeddings real
/// neighborhood structure; the leak keeps it from collapsing to disjoint
/// blocks. `TripleData`'s fields are public, so we populate them directly and
/// skip the Polars loader.
fn clustered_triples(
    n_items: usize,
    n_clusters: usize,
    n_users: usize,
    pos_per_user: usize,
    seed: u64,
) -> TripleData {
    let mut rng = Lcg::new(seed);

    // Items: contiguous 0-based index space (no reserved item row). Cluster of
    // item `i` is `i % n_clusters` so clusters are evenly sized and interleaved.
    let idx_to_item: Vec<String> = (0..n_items).map(|i| format!("item_{i}")).collect();
    let item_to_idx: AHashMap<String, usize> = idx_to_item
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i))
        .collect();
    let cluster_of = |item: usize| item % n_clusters;
    // Precompute the item members of each cluster for in-cluster sampling.
    let mut cluster_members: Vec<Vec<usize>> = vec![Vec::new(); n_clusters];
    for item in 0..n_items {
        cluster_members[cluster_of(item)].push(item);
    }

    // Users: index 0 is the reserved cold-start prior (TripleData invariant);
    // real users intern at 1..=n_users.
    let mut idx_to_user: Vec<String> = vec![crate::data::triples::COLD_START_USER_ID.to_string()];
    let mut user_to_idx: AHashMap<String, usize> = AHashMap::new();
    let mut triples: Vec<Triple> = Vec::with_capacity(n_users * pos_per_user);

    for u in 0..n_users {
        let user_id = format!("user_{u}");
        let user_idx = idx_to_user.len();
        idx_to_user.push(user_id.clone());
        user_to_idx.insert(user_id, user_idx);

        let cluster = u % n_clusters;
        let members = &cluster_members[cluster];
        for _ in 0..pos_per_user {
            // 85% in-cluster, 15% catalog-wide leak.
            let item_idx = if rng.below(100) < 85 {
                members[rng.below(members.len())]
            } else {
                rng.below(n_items)
            };
            triples.push(Triple { user_idx, item_idx });
        }
    }

    TripleData {
        triples,
        user_to_idx,
        idx_to_user,
        item_to_idx,
        idx_to_item,
    }
}

/// Synthesize clustered L2-normalized embeddings directly (no training):
/// `n_clusters` random unit centers; each item is its cluster's center plus
/// small uniform jitter, re-normalized. Preserves the neighborhood structure
/// ANN recall depends on, at scales where training is impractical (#82).
fn synthetic_clustered_embeddings(
    n_items: usize,
    n_clusters: usize,
    dim: usize,
    seed: u64,
) -> Vec<Vec<f32>> {
    let mut rng = Lcg::new(seed);
    // Uniform in [-1, 1).
    let unit = |rng: &mut Lcg| (rng.next_u64() >> 40) as f32 / (1u64 << 23) as f32 - 1.0;
    let normalize = |v: &mut Vec<f32>| {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        v.iter_mut().for_each(|x| *x /= norm);
    };
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| {
            let mut c: Vec<f32> = (0..dim).map(|_| unit(&mut rng)).collect();
            normalize(&mut c);
            c
        })
        .collect();
    (0..n_items)
        .map(|i| {
            let center = &centers[i % n_clusters];
            // Jitter magnitude 0.15 keeps clusters tight but overlapping enough
            // that top-K neighborhoods aren't trivially disjoint blocks.
            let mut v: Vec<f32> = center.iter().map(|&c| c + 0.15 * unit(&mut rng)).collect();
            normalize(&mut v);
            v
        })
        .collect()
}

/// Resident-set size in bytes via `/proc/self/statm` (Linux only). `None`
/// where /proc doesn't exist (macOS) — callers print "n/a".
fn resident_bytes() -> Option<usize> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let resident_pages: usize = statm.split_whitespace().nth(1)?.parse().ok()?;
    Some(resident_pages * 4096)
}

/// Build a backend and measure the RSS delta across the build. The delta is
/// approximate (allocator caching, other test threads) but resolves the
/// measured-usearch vs analytic-turbovec asymmetry from the Phase 2 run.
fn build_with_rss<B: AnnBackend>(items: &[Vec<f32>]) -> (B, Option<usize>, f64) {
    let before = resident_bytes();
    let start = std::time::Instant::now();
    let backend = B::build(items);
    let build_secs = start.elapsed().as_secs_f64();
    let delta = match (before, resident_bytes()) {
        (Some(b), Some(a)) => Some(a.saturating_sub(b)),
        _ => None,
    };
    (backend, delta, build_secs)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|v| {
            v.parse()
                .unwrap_or_else(|_| panic!("{name} must be a positive integer, got {v:?}"))
        })
        .unwrap_or(default)
}

/// Average recall@k of `backend` against exact ground truth over the query set.
/// For each query item `qi`, ground truth is the exact top-k *excluding qi*,
/// and the backend is asked for top-k *also excluding qi*, so the two are
/// directly comparable.
fn mean_recall<B: AnnBackend>(backend: &B, items: &[Vec<f32>], queries: &[usize], k: usize) -> f32 {
    let mut total = 0.0f32;
    for &qi in queries {
        let approx = backend.search(&items[qi], k, &[qi]);
        // exact_top_k over the full catalog, then drop qi and take k.
        let mut exact = exact_top_k(&items[qi], items, k + 1);
        exact.retain(|(i, _)| *i != qi);
        exact.truncate(k);
        total += recall_at_k(&approx, &exact);
    }
    total / queries.len() as f32
}

/// Sorted per-query wall-clock of `backend.search(q, k, &[qi])` over the
/// query set, in microseconds. Percentiles (p50/p99) are read off the sorted
/// vec so a one-off allocation hiccup doesn't dominate the headline number.
fn latencies_us<B: AnnBackend>(
    backend: &B,
    items: &[Vec<f32>],
    queries: &[usize],
    k: usize,
) -> Vec<f64> {
    let mut times_us: Vec<f64> = Vec::with_capacity(queries.len());
    for &qi in queries {
        let start = std::time::Instant::now();
        let _ = backend.search(&items[qi], k, &[qi]);
        times_us.push(start.elapsed().as_secs_f64() * 1e6);
    }
    times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times_us
}

/// Nearest-rank percentile of an already-sorted sample (`p` in [0,100]).
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

/// Per-backend measurements for one bench run.
struct BenchRow {
    name: &'static str,
    recall: f32,
    p50_us: f64,
    p99_us: f64,
    index_bytes: usize,
    rss_delta: Option<usize>,
    build_secs: f64,
}

fn bench_backend<B: AnnBackend>(
    name: &'static str,
    items: &[Vec<f32>],
    queries: &[usize],
    k: usize,
) -> BenchRow {
    let (backend, rss_delta, build_secs) = build_with_rss::<B>(items);
    let recall = mean_recall(&backend, items, queries, k);
    let lats = latencies_us(&backend, items, queries, k);
    BenchRow {
        name,
        recall,
        p50_us: percentile(&lats, 50.0),
        p99_us: percentile(&lats, 99.0),
        index_bytes: backend.index_bytes(),
        rss_delta,
        build_secs,
    }
}

#[test]
#[ignore = "slow bench-off; run with --ignored --nocapture (env: ANN_BENCH_N_ITEMS/DIM/K/MODE)"]
fn ann_backend_comparison() {
    let n_items = env_usize("ANN_BENCH_N_ITEMS", 3000);
    let dim = env_usize("ANN_BENCH_DIM", 64);
    let k = env_usize("ANN_BENCH_K", 10);
    let mode = std::env::var("ANN_BENCH_MODE").unwrap_or_else(|_| "trained".into());
    let n_clusters = env_usize("ANN_BENCH_CLUSTERS", 12);
    assert!(
        dim % 8 == 0,
        "ANN_BENCH_DIM must be a multiple of 8 (turbovec invariant)"
    );

    const N_USERS: usize = 4000;
    const POS_PER_USER: usize = 12;
    const SEED: u64 = 0xA11CE;

    // 1) Item embeddings: real trained Two-Tower (the decision-grade path, #76)
    //    or directly-synthesized clusters (the scale path, #82 — training at
    //    100k+ items is impractical on hosted CI).
    let items: Vec<Vec<f32>> = match mode.as_str() {
        "trained" => {
            let data = clustered_triples(n_items, n_clusters, N_USERS, POS_PER_USER, SEED);
            let uft = FeatureTable::empty(data.num_users());
            let ift = FeatureTable::empty(data.num_items());
            let model = train(
                &data,
                &uft,
                &ift,
                TrainParams {
                    embedding_dim: dim,
                    epochs: 15,
                    ..Default::default()
                },
            )
            .expect("train");
            model.item_embeddings()
        }
        "synthetic" => synthetic_clustered_embeddings(n_items, n_clusters, dim, SEED),
        other => panic!("ANN_BENCH_MODE must be 'trained' or 'synthetic', got {other:?}"),
    };
    assert_eq!(items.len(), n_items);
    assert!(items.iter().all(|r| r.len() == dim));

    // 2) Query set: ~100 items sampled evenly across the catalog.
    let stride = (items.len() / 100).max(1);
    let queries: Vec<usize> = (0..items.len()).step_by(stride).collect();

    // 3) Build + measure each backend (build RSS delta, recall@K, p50/p99).
    let rows = [
        bench_backend::<ExactBackend>("exact", &items, &queries, k),
        bench_backend::<UsearchBackend>("usearch", &items, &queries, k),
        bench_backend::<TurbovecBackend>("turbovec", &items, &queries, k),
    ];

    // 4) Print the decision table (visible under --nocapture).
    let mb = |b: usize| b as f64 / (1024.0 * 1024.0);
    eprintln!();
    eprintln!(
        "ANN backend bench-off (ADR-0004 Phase 2 / #82): N_ITEMS={n_items}, dim={dim}, K={k}, \
         mode={mode}, clusters={n_clusters}, queries={}",
        queries.len()
    );
    eprintln!(
        "{:<10} | {:>9} | {:>10} | {:>10} | {:>12} | {:>12} | {:>9}",
        "backend", "recall@K", "p50 (us)", "p99 (us)", "index (MB)", "RSS Δ (MB)", "build (s)"
    );
    eprintln!(
        "{:-<10}-+-{:-<9}-+-{:-<10}-+-{:-<10}-+-{:-<12}-+-{:-<12}-+-{:-<9}",
        "", "", "", "", "", "", ""
    );
    for r in &rows {
        let rss = r
            .rss_delta
            .map(|b| format!("{:.3}", mb(b)))
            .unwrap_or_else(|| "n/a".into());
        eprintln!(
            "{:<10} | {:>9.4} | {:>10.2} | {:>10.2} | {:>12.3} | {:>12} | {:>9.2}",
            r.name,
            r.recall,
            r.p50_us,
            r.p99_us,
            mb(r.index_bytes),
            rss,
            r.build_secs
        );
    }
    eprintln!();
    eprintln!(
        "note: index (MB) = backend-reported index_bytes (usearch: measured memory_usage(); \
         turbovec: analytic packed-code estimate). RSS Δ = resident-set delta measured across \
         the index build (/proc/self/statm; Linux only) — the apples-to-apples column."
    );
    eprintln!();

    // 5) Real assertions (not just prints).
    // Exact backend is the recall==1.0 control.
    let exact_recall = rows[0].recall;
    assert!(
        (exact_recall - 1.0).abs() < 1e-6,
        "ExactBackend recall must be 1.0, got {exact_recall}"
    );
    // usearch on decision-grade trained embeddings should clear 0.80 easily.
    // This is a real floor: if it misses, that's a finding worth surfacing,
    // not a knob to lower. Scale runs (synthetic mode) REPORT recall instead
    // of asserting it — the 2026-07 CI run measured usearch at 0.568 @ 1M on
    // hard clustered geometry (default HNSW params), which is exactly the
    // kind of scale finding the run exists to record, not a build failure.
    if mode == "trained" {
        let usearch_recall = rows[1].recall;
        assert!(
            usearch_recall >= 0.80,
            "UsearchBackend recall@{k} below floor: {usearch_recall} < 0.80"
        );
    }
    // No turbovec floor: quantization may legitimately trade recall — reported,
    // not asserted.
}
