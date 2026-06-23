# ANN retrieval benchmark (ADR-0004 Phase 2 · issue #76)

Harness for choosing and validating the approximate-nearest-neighbor backend
that the `RetrievalIndex` seam (ADR-0004 Phase 1, #75) routes through for
embedding models.

## What this measures

For each catalog size (10k / 100k / 1M items, dim 64) the bench compares an ANN
backend against the **exact** maximum-inner-product top-K baseline on three axes:

| Axis | How | Gate |
|------|-----|------|
| **recall@K** vs exact | `recall_at_k(approx, exact_top_k(..))` | ADR CI floor (TBD, e.g. ≥0.95) |
| **latency** p50/p99 | criterion `bench_function` per backend | must beat exact at ≥100k |
| **resident memory** | index size vs full f32 matrix | quantization win (TurboVec's edge) |

The exact baseline mirrors serving's dense path (`filter_sort_top_k`): partial
quickselect + index tie-break, so recall ground truth matches production order.

## Status — scaffolding only

Landed: reproducible synthetic embeddings, exact baseline, `recall_at_k`, and a
criterion group benchmarking the exact path. The ANN backends are **not** wired
in yet — that's the open decision below. Each plugs in as a `bench_function`
beside `exact`, fed the same embeddings and scored against the same ground truth.

## Open decisions (gate the next increment)

1. **Backend: usearch vs TurboVec.** usearch — mature Rust crate, filtered
   search, HNSW. TurboVec — quantization-first (4–16× memory), newer /
   single-author. Both go behind a default-off `ann` Cargo feature (composes
   with `ml-models`). The bench picks the winner on the table above; the seam
   stays library-agnostic so the loser is swappable.
2. **Embedding source for the *decision* run.** Synthetic vectors validate the
   harness shape, but the recall numbers that decide the backend should come
   from a **real Two-Tower** `item_matrix` (`catalog_matrix`,
   `src/models/two_tower.rs`). Options: export from a Two-Tower trained on
   synthetic interactions, or a real catalog. Needs a small exporter
   (`item_matrix` → flat `f32` file the bench loads).

## Run

```bash
cargo bench --bench ann_retrieval     # latency sweep over catalog sizes
```

The exact baseline mirrors the already-tested `filter_sort_top_k`
(`src/serving.rs`); when `exact_top_k` / `recall_at_k` are promoted to a
feature-gated `src/ann` module alongside the real backend, they get unit tests
there (criterion benches with `harness = false` can't host libtest `#[test]`s).
