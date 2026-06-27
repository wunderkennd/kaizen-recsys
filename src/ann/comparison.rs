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

/// Median wall-clock of a single `backend.search(q, k, &[qi])` over the query
/// set, in microseconds. Median (not mean) so a one-off allocation hiccup
/// doesn't dominate.
fn median_latency_us<B: AnnBackend>(
    backend: &B,
    items: &[Vec<f32>],
    queries: &[usize],
    k: usize,
) -> f64 {
    let mut times_us: Vec<f64> = Vec::with_capacity(queries.len());
    for &qi in queries {
        let start = std::time::Instant::now();
        let _ = backend.search(&items[qi], k, &[qi]);
        times_us.push(start.elapsed().as_secs_f64() * 1e6);
    }
    if times_us.is_empty() {
        return 0.0;
    }
    times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = times_us.len() / 2;
    if times_us.len() % 2 == 1 {
        times_us[mid]
    } else {
        (times_us[mid - 1] + times_us[mid]) / 2.0
    }
}

#[test]
#[ignore = "slow bench-off (trains a Two-Tower); run with --ignored --nocapture"]
fn ann_backend_comparison() {
    const N_ITEMS: usize = 3000;
    const N_CLUSTERS: usize = 12;
    const N_USERS: usize = 4000;
    const POS_PER_USER: usize = 12;
    const DIM: usize = 64; // must be a multiple of 8 (turbovec invariant).
    const K: usize = 10;
    const SEED: u64 = 0xA11CE;

    // 1) Synthetic clustered interactions + empty feature tables.
    let data = clustered_triples(N_ITEMS, N_CLUSTERS, N_USERS, POS_PER_USER, SEED);
    let uft = FeatureTable::empty(data.num_users());
    let ift = FeatureTable::empty(data.num_items());

    // 2) Train a real Two-Tower. embedding_dim=64 (mult. of 8); modest epochs
    //    so the bench finishes in a couple of minutes in release.
    let model = train(
        &data,
        &uft,
        &ift,
        TrainParams {
            embedding_dim: DIM,
            epochs: 15,
            ..Default::default()
        },
    )
    .expect("train");

    // 3) Real L2-normalized item embeddings, shape (num_items, DIM).
    let items = model.item_embeddings();
    assert_eq!(items.len(), data.num_items());
    assert!(items.iter().all(|r| r.len() == DIM));

    // 4) Query set: ~100 items sampled evenly across the catalog.
    let stride = (items.len() / 100).max(1);
    let queries: Vec<usize> = (0..items.len()).step_by(stride).collect();

    // 5) Build each backend and measure recall@K, median latency, footprint.
    let exact_be = ExactBackend::build(&items);
    let usearch_be = UsearchBackend::build(&items);
    let turbovec_be = TurbovecBackend::build(&items);

    let exact_recall = mean_recall(&exact_be, &items, &queries, K);
    let usearch_recall = mean_recall(&usearch_be, &items, &queries, K);
    let turbovec_recall = mean_recall(&turbovec_be, &items, &queries, K);

    let exact_lat = median_latency_us(&exact_be, &items, &queries, K);
    let usearch_lat = median_latency_us(&usearch_be, &items, &queries, K);
    let turbovec_lat = median_latency_us(&turbovec_be, &items, &queries, K);

    let exact_bytes = exact_be.index_bytes();
    let usearch_bytes = usearch_be.index_bytes();
    let turbovec_bytes = turbovec_be.index_bytes();

    let mb = |b: usize| b as f64 / (1024.0 * 1024.0);

    // 6) Print the decision table (visible under --nocapture).
    eprintln!();
    eprintln!(
        "ANN backend bench-off (ADR-0004 Phase 2): N_ITEMS={N_ITEMS}, dim={DIM}, K={K}, \
         clusters={N_CLUSTERS}, queries={}",
        queries.len()
    );
    eprintln!(
        "{:<10} | {:>9} | {:>14} | {:>14} | {:>10}",
        "backend", "recall@10", "median lat (us)", "index_bytes", "index (MB)"
    );
    eprintln!(
        "{:-<10}-+-{:-<9}-+-{:-<14}-+-{:-<14}-+-{:-<10}",
        "", "", "", "", ""
    );
    let row = |name: &str, recall: f32, lat: f64, bytes: usize| {
        eprintln!(
            "{name:<10} | {recall:>9.4} | {lat:>14.2} | {bytes:>14} | {:>10.3}",
            mb(bytes)
        );
    };
    row("exact", exact_recall, exact_lat, exact_bytes);
    row("usearch", usearch_recall, usearch_lat, usearch_bytes);
    row("turbovec", turbovec_recall, turbovec_lat, turbovec_bytes);
    eprintln!();
    eprintln!(
        "note: usearch index_bytes = measured resident HNSW footprint (memory_usage()); \
         turbovec = analytic packed-code estimate (len*dim*bits/8 + scales). Not directly comparable."
    );
    eprintln!();

    // 7) Real assertions (not just prints).
    // Exact backend is the recall==1.0 control.
    assert!(
        (exact_recall - 1.0).abs() < 1e-6,
        "ExactBackend recall must be 1.0, got {exact_recall}"
    );
    // usearch on structured embeddings should clear 0.80 easily. This is a real
    // floor: if it misses, that's a finding worth surfacing, not a knob to lower.
    assert!(
        usearch_recall >= 0.80,
        "UsearchBackend recall@{K} below floor: {usearch_recall} < 0.80"
    );
    // No turbovec floor: quantization may legitimately trade recall — reported,
    // not asserted.
}
