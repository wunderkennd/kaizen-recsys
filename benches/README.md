# ANN retrieval backend bench-off (ADR-0004 Phase 2 · issue #76)

Chooses the approximate-nearest-neighbor backend the `RetrievalIndex` seam
(ADR-0004 Phase 1, #75) routes through for embedding models, by benchmarking
candidates against the exact maximum-inner-product baseline on real Two-Tower
embeddings.

## How it runs (and why it's a test, not a criterion bench)

The crate is a PyO3 `extension-module` (`crate-type = ["cdylib"]`). On macOS any
`cargo build` / `cargo bench` / `cargo run` emits the cdylib and fails to link
(no libpython); only `cargo test` links (it builds a test harness, never the
cdylib). So the bench-off lives as a feature-gated, `#[ignore]`d **in-crate
test** — `ann_backend_comparison` in `src/ann/comparison.rs` — which links *and*
reaches the crate-internal Two-Tower training + `ann` backends:

```bash
cargo test --release --features "ann ml-models" -- --ignored ann_backend_comparison --nocapture
```

It trains a small Two-Tower on synthetic *clustered* interactions (so the
embeddings have real neighborhood structure), takes the real `item_embeddings()`
matrix, builds each backend, and reports recall@K (vs `exact_top_k` ground
truth), median single-query latency, and `index_bytes`.

The `benches/ann_retrieval.rs` criterion bench remains as a self-contained
synthetic-data latency micro-bench (it can't import the crate, per the above).

## Results (N_ITEMS=3000, dim=64, K=10, 12 clusters, 100 queries)

Stable across runs; real trained Two-Tower embeddings.

| backend  | recall@10 | median latency | index size |
|----------|-----------|----------------|------------|
| exact    | 1.00      | ~150 µs        | 0.73 MB    |
| usearch  | 1.00      | ~18 µs         | 16.0 MB (measured HNSW resident) |
| turbovec | ~0.88     | ~8 µs          | 0.10 MB (analytic packed-code estimate) |

`exact` is the brute-force baseline ANN must beat. **Memory is not strictly
apples-to-apples**: usearch reports measured resident size (incl. the HNSW
graph); turbovec reports a packed-code lower bound. The order-of-magnitude
footprint gap (4-bit quantization) is real regardless.

## Decision

**Default backend: turbovec.** It leads on the axes that motivated adding an ANN
stage at all — ~155× smaller index and ~2× lower latency than usearch — at the
cost of ~12 points of recall@10 (0.88 vs 1.00), a deliberate trade on the
assumption a downstream reranker absorbs the retrieval recall loss (cf. the
OverArch pattern in `research/silver_torch_research.md`).

**usearch is kept as the exact-recall alternative**, selected when retrieval
recall must be exact or when memory is not the binding constraint. Both implement
`AnnBackend` behind the default-off `ann` feature, so the choice is serving
config, not a fork — and turbovec's pre-1.0 maturity risk is mitigated by the
seam (swappable to usearch without touching callers).

### Caveats / follow-ups
- turbovec requires `dim % 8 == 0` and uses positional ids (vs usearch's
  arbitrary dims + explicit u64 keys) — see each backend's module docs.

## Scale results (#82 · CI Linux · dim=128, K=10, synthetic clusters, ~1k items/cluster)

Run on ubuntu-latest via `.github/workflows/ann_bench.yml` (2026-07-02).
Embeddings are synthesized clusters (`ANN_BENCH_MODE=synthetic`) — training a
real Two-Tower at these sizes is impractical on hosted CI; the recall
*decision* was made on trained embeddings above, these runs firm up the
latency / memory / build-time ratios at the scale where ANN earns its keep.
**RSS Δ** is the resident-set delta measured across the index build — the
apples-to-apples memory column (the "index (MB)" column mixes usearch's
measured `memory_usage()` with turbovec's analytic packed-code estimate,
which flatters turbovec ~10×).

**N = 100k (100 clusters):**

| backend  | recall@10 | p50 (µs) | p99 (µs) | index (MB) | RSS Δ (MB) | build (s) |
|----------|-----------|----------|----------|------------|------------|-----------|
| exact    | 1.000     | 18,632   | 18,915   | 48.8       | 52.7       | 0.05      |
| usearch  | 0.983     | 68       | 188      | 81.3       | 70.4       | 13.7      |
| turbovec | 0.824     | 942      | 974      | 6.5        | 66.8       | 0.41      |

**N = 1M (1000 clusters):**

| backend  | recall@10 | p50 (µs) | p99 (µs) | index (MB) | RSS Δ (MB) | build (s) |
|----------|-----------|----------|----------|------------|------------|-----------|
| exact    | 1.000     | 209,333  | 218,123  | 488        | 526        | 0.48      |
| usearch  | 0.904     | 155      | 420      | 763        | 678        | 236       |
| turbovec | 0.833     | 9,445    | 9,641    | 64.9       | 265        | 2.9       |

### What the scale runs changed about the picture

- **The turbovec-default decision stands**, but with corrected magnitudes: the
  real memory advantage is **~2.6× RSS at 1M** (265 vs 678 MB), not the ~155×
  the analytic estimate suggested at 3k items. Build time is where turbovec
  dominates: **2.9 s vs 236 s at 1M** (80×) — which is also why index
  persistence (#77) was closed as not-justified for the default backend.
- **usearch's role sharpens to "the latency/recall alternative"**: 60× lower
  query latency than turbovec at 1M (155 µs vs 9.4 ms p50) and recall
  0.90–0.98 on realistic geometry, at the cost of the largest memory
  footprint and a minutes-scale build.
- **Geometry matters more than scale for HNSW recall**: with the original 12
  fixed clusters at 1M (~83k near-duplicates per cluster), usearch recall
  collapsed to 0.568 while turbovec held 0.727. With ~1k items/cluster it
  recovered to 0.904. turbovec's brute-force-over-quantized-codes recall is
  geometry-insensitive by construction (0.72–0.85 across every run). The
  usearch 0.80 recall floor therefore asserts only in `trained` mode; scale
  runs report instead of failing.
- turbovec p50 grows linearly with N (it scans all codes): 0.94 ms @ 100k →
  9.4 ms @ 1M. Still 22× faster than exact f32 scoring, but latency-sensitive
  1M+ catalogs should weigh usearch despite the memory/build cost.

## Components

- `src/ann/exact.rs` — `exact_top_k` (ground truth) + `recall_at_k` (the metric), unit-tested.
- `src/ann/mod.rs` — `AnnBackend` trait + `ExactBackend` control.
- `src/ann/usearch_backend.rs`, `src/ann/turbovec_backend.rs` — the two real backends.
- `src/ann/comparison.rs` — the `#[ignore]`d bench-off test above.
- `TrainedTwoTower::item_embeddings()` (`src/models/two_tower.rs`) — real embedding source.
