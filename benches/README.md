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
- Measured at 3000 items. The recall gap (a property of quantization) transfers;
  the absolute latency/memory *ratios* may shift at 1M+ items where ANN earns its
  keep. A scale bench on CI Linux is tracked as a follow-up issue.
- turbovec requires `dim % 8 == 0` and uses positional ids (vs usearch's
  arbitrary dims + explicit u64 keys) — see each backend's module docs.

## Components

- `src/ann/exact.rs` — `exact_top_k` (ground truth) + `recall_at_k` (the metric), unit-tested.
- `src/ann/mod.rs` — `AnnBackend` trait + `ExactBackend` control.
- `src/ann/usearch_backend.rs`, `src/ann/turbovec_backend.rs` — the two real backends.
- `src/ann/comparison.rs` — the `#[ignore]`d bench-off test above.
- `TrainedTwoTower::item_embeddings()` (`src/models/two_tower.rs`) — real embedding source.
