//! ANN retrieval-index benchmark harness (issue #76 / ADR-0004 Phase 2).
//!
//! This is **library-agnostic scaffolding**. It establishes the three things
//! every ANN-backend comparison needs, independent of which backend wins:
//!
//!   1. A reproducible synthetic item-embedding matrix (`gen_embeddings`) —
//!      stand-in for a trained Two-Tower `(num_items, dim)` matrix
//!      (`catalog_matrix` in `src/models/two_tower.rs`). Swap in real
//!      exported embeddings to measure the *decision* recall numbers; for
//!      harness shape, deterministic synthetic vectors suffice.
//!   2. The **exact** maximum-inner-product top-K baseline (`exact_top_k`) —
//!      both the latency floor ANN must beat and the ground truth recall@K
//!      is measured against.
//!   3. `recall_at_k` — the metric the ADR's CI gate is defined on.
//!
//! The ANN backends (usearch / TurboVec) are deliberately NOT wired in yet —
//! that is the open decision in #76. Each gets a `bench_function` next to the
//! exact baseline once the backend is chosen, fed the *same* embeddings and
//! scored against the *same* `exact_top_k` ground truth.
//!
//! Run: `cargo bench --bench ann_retrieval`

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Catalog sizes to sweep. ANN's win over the exact O(n·dim) baseline only
/// appears at large catalogs, so the sweep brackets the crossover.
const CATALOG_SIZES: &[usize] = &[10_000, 100_000, 1_000_000];
const DIM: usize = 64;
const TOP_K: usize = 20;
/// Fixed seed: the bench must be reproducible run-to-run (no `Math.random`).
const SEED: u64 = 0xA11CE;

/// Generate `n` L2-normalized `dim`-dimensional embeddings from a fixed seed.
/// L2-normalized so inner product ≡ cosine, matching the Two-Tower towers.
fn gen_embeddings(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            v.iter().map(|x| x / norm).collect()
        })
        .collect()
}

/// Exact maximum-inner-product top-K: score every item, partial-select top-K.
/// Mirrors the dense serving path (`filter_sort_top_k`) and is the ground
/// truth recall@K is computed against.
fn exact_top_k(query: &[f32], items: &[Vec<f32>], k: usize) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = items
        .iter()
        .enumerate()
        .map(|(i, it)| {
            let dot = it.iter().zip(query).map(|(a, b)| a * b).sum::<f32>();
            (i, dot)
        })
        .collect();
    let k = k.min(scored.len());
    if k == 0 {
        return Vec::new();
    }
    // Descending by score, ascending index tie-break — total order, matching
    // src/serving.rs::filter_sort_top_k.
    let cmp = |a: &(usize, f32), b: &(usize, f32)| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    };
    if k < scored.len() {
        scored.select_nth_unstable_by(k - 1, cmp);
        scored.truncate(k);
    }
    scored.sort_unstable_by(cmp);
    scored
}

/// recall@k of an approximate result against the exact ground truth: the
/// fraction of exact top-k item ids the approximate set also returned. This
/// is the metric the ADR's CI gate floor is defined on.
#[allow(dead_code)]
pub fn recall_at_k(approx: &[(usize, f32)], exact: &[(usize, f32)]) -> f32 {
    if exact.is_empty() {
        return 1.0;
    }
    let truth: std::collections::HashSet<usize> = exact.iter().map(|(i, _)| *i).collect();
    let hits = approx.iter().filter(|(i, _)| truth.contains(i)).count();
    hits as f32 / exact.len() as f32
}

fn bench_exact_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("retrieval_top_k");
    let query = gen_embeddings(1, DIM, SEED ^ 0xFFFF).pop().unwrap();

    for &n in CATALOG_SIZES {
        let items = gen_embeddings(n, DIM, SEED);
        group.throughput(criterion::Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("exact", n), &items, |b, items| {
            b.iter(|| exact_top_k(black_box(&query), black_box(items), TOP_K));
        });
        // ANN backends slot in here, same `items` + `query`, then their
        // results are scored with `recall_at_k(.., &exact_top_k(..))`.
    }
    group.finish();
}

criterion_group!(benches, bench_exact_baseline);
criterion_main!(benches);
