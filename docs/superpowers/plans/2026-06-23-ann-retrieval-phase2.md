# ANN Retrieval Backend Bench-Off Implementation Plan (ADR-0004 Phase 2)

> **Status:** Executed and complete (PR #79); progress was tracked in issue #76, per the repo's work-tracking policy. This file is the historical design record — see the revision note below for where the final shape diverged.

**Goal:** Bench usearch and TurboVec against the exact top-K baseline on real Two-Tower embeddings, behind a default-off `ann` Cargo feature, to choose the `RetrievalIndex` backend per ADR-0004 — without changing default builds, eval, or EASE.

**Architecture:** A new feature-gated `src/ann` module holds the exact baseline + recall metric (promoted from the bench, now unit-tested) and a small `AnnBackend` trait with one impl per ANN crate. A Two-Tower embedding exporter (under `ml-models`) dumps a real `item_matrix` to a flat `f32` file. The criterion bench (`benches/ann_retrieval.rs`) loads either synthetic or exported embeddings, runs each backend beside `exact`, and reports recall@K against the exact ground truth. The serving `RetrievalIndex` impl for the *winning* backend is a follow-up, out of scope here.

**Tech Stack:** Rust, criterion (bench), `usearch = 2.25` and `turbovec = 0.9` (both crates.io, optional/`ann`-gated), burn (`ml-models`, exporter only), the existing `src/serving.rs::filter_sort_top_k` (baseline reference).

**Builds on:** #79 (scaffold harness). Tracks #76. Resolves the tech debt logged in #76 (promote + unit-test bench utilities; dedupe exact baseline).

> **Revision (2026-06-26):** Tasks 7-9 were pivoted from "exporter bin + criterion
> bench" to a single in-crate `#[ignore]`d test (`src/ann/comparison.rs`). Reason:
> the crate is a PyO3 cdylib extension module, so bins/benches can't link locally
> (only `cargo test` does). See `benches/README.md` and issue #76 for the final
> shape and the backend decision (turbovec default). The tasks below are the
> original plan, retained as the historical record.

---

## File Structure

- `Cargo.toml` — add `ann` feature + optional `usearch`/`turbovec` deps; the bench gains no new always-on deps.
- `src/ann/mod.rs` — `ann` module root: re-exports, `AnnBackend` trait. **One responsibility:** the vector-level ANN abstraction used by the bench.
- `src/ann/exact.rs` — `exact_top_k`, `recall_at_k` (promoted from the bench), unit-tested. **Single source of truth** for ground truth + metric.
- `src/ann/usearch_backend.rs` — `UsearchBackend` impl of `AnnBackend`.
- `src/ann/turbovec_backend.rs` — `TurbovecBackend` impl of `AnnBackend`.
- `src/models/two_tower.rs` — add `pub fn item_embeddings()` accessor to `TrainedTwoTower` (modify; `ml-models`).
- `src/bin/export_tt_embeddings.rs` — exporter binary (`ml-models`): train Two-Tower on synthetic interactions → write `item_matrix` to disk.
- `benches/ann_retrieval.rs` — modify: import from `src/ann`, load real embeddings, add usearch/turbovec bench functions + recall output.

---

## Task 1: Add the `ann` Cargo feature and optional deps

**Files:**
- Modify: `Cargo.toml`

**Step 1: Add the feature and optional deps**

In `[features]` add:
```toml
# Approximate-nearest-neighbor retrieval backends for the Two-Tower serving
# path (ADR-0004 Phase 2). Default off; composes with ml-models. Pulls in
# whichever ANN crate(s) are being benched/used.
ann = ["dep:usearch", "dep:turbovec"]
```
In `[dependencies]` add:
```toml
usearch = { version = "2.25", optional = true }
turbovec = { version = "0.9", optional = true }
```

**Step 2: Verify default build is unchanged**

Run: `cargo build`
Expected: PASS, and `cargo tree -e features | grep -E "usearch|turbovec"` prints nothing (not in the default graph).

**Step 3: Verify the feature resolves**

Run: `cargo build --features ann`
Expected: PASS (downloads + compiles usearch and turbovec).

**Step 4: Commit**

```bash
git add Cargo.toml Cargo.lock
git commit -m "build(ann): add default-off ann feature with usearch + turbovec deps"
```

---

## Task 2: Promote the exact baseline + recall metric into `src/ann/exact.rs` (TDD)

Resolves the #76 tech debt: bench utilities become unit-tested library code.

**Files:**
- Create: `src/ann/mod.rs`
- Create: `src/ann/exact.rs`
- Modify: `src/lib.rs` (add `mod ann;` behind the feature)

**Step 1: Register the module**

In `src/lib.rs`, alongside the other `mod` declarations:
```rust
#[cfg(feature = "ann")]
mod ann;
```

**Step 2: Write `src/ann/mod.rs` with the failing-test wiring**

```rust
//! Approximate-nearest-neighbor retrieval backends (ADR-0004 Phase 2).
//! Feature-gated (`ann`); not in default or EASE-only builds.
pub mod exact;
```

**Step 3: Write the failing tests in `src/ann/exact.rs`**

```rust
//! Exact maximum-inner-product top-K and the recall@K metric — the ground
//! truth the ANN backends are scored against. Mirrors the dense serving
//! path (`crate::serving::filter_sort_top_k`); kept separate so the bench
//! and backends share one tested implementation.

/// Exact MIPS top-K: descending score, ascending-index tie-break (total order,
/// matching `filter_sort_top_k`).
pub fn exact_top_k(query: &[f32], items: &[Vec<f32>], k: usize) -> Vec<(usize, f32)> {
    unimplemented!()
}

/// Fraction of the exact top-k item ids that `approx` also returned.
pub fn recall_at_k(approx: &[(usize, f32)], exact: &[(usize, f32)]) -> f32 {
    unimplemented!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_top_k_descending_with_index_tiebreak() {
        let items = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.9, 0.1]];
        let top = exact_top_k(&[1.0, 0.0], &items, 2);
        assert_eq!(top.iter().map(|(i, _)| *i).collect::<Vec<_>>(), vec![0, 2]);
    }

    #[test]
    fn exact_top_k_ties_break_by_lower_index() {
        // items 0 and 1 tie on score against this query; lower index wins.
        let items = vec![vec![1.0], vec![1.0], vec![0.5]];
        let top = exact_top_k(&[1.0], &items, 2);
        assert_eq!(top[0].0, 0);
        assert_eq!(top[1].0, 1);
    }

    #[test]
    fn recall_full_and_partial() {
        let exact = vec![(1, 0.9), (2, 0.5), (3, 0.4)];
        assert_eq!(recall_at_k(&exact, &exact), 1.0);
        let approx = vec![(1, 0.9), (2, 0.5), (9, 0.3)];
        assert!((recall_at_k(&approx, &exact) - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn recall_of_empty_truth_is_one() {
        assert_eq!(recall_at_k(&[], &[]), 1.0);
    }
}
```

**Step 4: Run the tests, verify they fail**

Run: `cargo test --features ann --lib ann::exact`
Expected: FAIL with `not implemented` panics.

**Step 5: Implement to pass**

Replace the two `unimplemented!()` bodies:
```rust
pub fn exact_top_k(query: &[f32], items: &[Vec<f32>], k: usize) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = items
        .iter()
        .enumerate()
        .map(|(i, it)| (i, it.iter().zip(query).map(|(a, b)| a * b).sum::<f32>()))
        .collect();
    let k = k.min(scored.len());
    if k == 0 {
        return Vec::new();
    }
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

pub fn recall_at_k(approx: &[(usize, f32)], exact: &[(usize, f32)]) -> f32 {
    if exact.is_empty() {
        return 1.0;
    }
    let truth: std::collections::HashSet<usize> = exact.iter().map(|(i, _)| *i).collect();
    let hits = approx.iter().filter(|(i, _)| truth.contains(i)).count();
    hits as f32 / exact.len() as f32
}
```

**Step 6: Run tests, verify pass**

Run: `cargo test --features ann --lib ann::exact`
Expected: PASS (4 tests).

**Step 7: Commit**

```bash
git add src/lib.rs src/ann/mod.rs src/ann/exact.rs
git commit -m "feat(ann): exact MIPS baseline + recall metric as tested library code"
```

---

## Task 3: Define the `AnnBackend` trait (TDD with a trivial exact impl)

**Files:**
- Modify: `src/ann/mod.rs`

**Step 1: Write the trait + a failing contract test**

Append to `src/ann/mod.rs`:
```rust
/// A vector-level ANN index built once over an item-embedding matrix, then
/// queried for approximate top-K. The bench builds one per backend and
/// scores its results against `exact::exact_top_k`.
pub trait AnnBackend {
    /// Build the index from `(num_items, dim)` row-major embeddings.
    fn build(items: &[Vec<f32>]) -> Self
    where
        Self: Sized;

    /// Approximate top-K `(item_idx, score)` for `query`, excluding `exclude`.
    fn search(&self, query: &[f32], k: usize, exclude: &[usize]) -> Vec<(usize, f32)>;

    /// Resident index size in bytes (for the memory axis).
    fn index_bytes(&self) -> usize;
}

/// Reference backend: exact search wearing the `AnnBackend` hat. Lets the
/// trait + bench plumbing be tested before any real ANN crate is wired in,
/// and is a recall==1.0 control in the bench.
pub struct ExactBackend {
    items: Vec<Vec<f32>>,
}

impl AnnBackend for ExactBackend {
    fn build(items: &[Vec<f32>]) -> Self {
        Self { items: items.to_vec() }
    }
    fn search(&self, query: &[f32], k: usize, exclude: &[usize]) -> Vec<(usize, f32)> {
        let ex: std::collections::HashSet<usize> = exclude.iter().copied().collect();
        let full = exact::exact_top_k(query, &self.items, self.items.len());
        full.into_iter().filter(|(i, _)| !ex.contains(i)).take(k).collect()
    }
    fn index_bytes(&self) -> usize {
        self.items.iter().map(|v| v.len() * std::mem::size_of::<f32>()).sum()
    }
}

#[cfg(test)]
mod backend_tests {
    use super::*;

    #[test]
    fn exact_backend_matches_exact_top_k_and_excludes() {
        let items = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.9, 0.1]];
        let be = ExactBackend::build(&items);
        // Without exclusion, top-1 is item 0.
        assert_eq!(be.search(&[1.0, 0.0], 1, &[])[0].0, 0);
        // Excluding item 0, top-1 becomes item 2 ([0.9,0.1]).
        assert_eq!(be.search(&[1.0, 0.0], 1, &[0])[0].0, 2);
    }
}
```

**Step 2: Run, verify pass** (trait + impl are concrete, no RED needed beyond compile)

Run: `cargo test --features ann --lib ann::backend_tests`
Expected: PASS.

**Step 3: Commit**

```bash
git add src/ann/mod.rs
git commit -m "feat(ann): AnnBackend trait + ExactBackend reference impl"
```

---

## Task 4: usearch backend (TDD)

**Files:**
- Create: `src/ann/usearch_backend.rs`
- Modify: `src/ann/mod.rs` (`pub mod usearch_backend;`)

**Step 1: Confirm the usearch Rust API**

Read https://docs.rs/usearch/2.25 — confirm `usearch::{Index, IndexOptions, MetricKind, ScalarKind}`, `Index::new(&options)`, `index.reserve(n)`, `index.add(key, &vec)`, `index.search(&query, k) -> Matches { keys, distances }`. Note: usearch keys are `u64`; we use the item index as the key. Inner-product metric = `MetricKind::IP`.

**Step 2: Write the failing test**

`src/ann/usearch_backend.rs`:
```rust
use crate::ann::AnnBackend;

pub struct UsearchBackend {
    index: usearch::Index,
    dim: usize,
}

impl AnnBackend for UsearchBackend {
    fn build(items: &[Vec<f32>]) -> Self { unimplemented!() }
    fn search(&self, query: &[f32], k: usize, exclude: &[usize]) -> Vec<(usize, f32)> { unimplemented!() }
    fn index_bytes(&self) -> usize { unimplemented!() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ann::exact::exact_top_k;

    // Structured data: 3 tight clusters. ANN should recover exact top-1.
    fn clustered(n_per: usize) -> Vec<Vec<f32>> {
        let centers = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut v = Vec::new();
        for (c, ctr) in centers.iter().enumerate() {
            for j in 0..n_per {
                let jitter = (j as f32) * 1e-3;
                v.push(vec![ctr[0] + jitter, ctr[1] + jitter, ctr[2] + jitter]);
            }
        }
        let _ = c_unused(); v
    }
    fn c_unused() {}

    #[test]
    fn recovers_exact_top1_on_clustered_data() {
        let items = clustered(20);
        let be = UsearchBackend::build(&items);
        let query = vec![1.0, 0.0, 0.0]; // cluster 0
        let approx = be.search(&query, 1, &[]);
        let exact = exact_top_k(&query, &items, 1);
        assert_eq!(approx[0].0, exact[0].0, "usearch must recover exact NN on well-separated clusters");
    }
}
```
(Simplify `clustered` to drop the `c_unused` scaffolding when implementing — shown only to keep the test self-contained.)

**Step 3: Run, verify it fails**

Run: `cargo test --features ann --lib ann::usearch_backend`
Expected: FAIL (`not implemented`).

**Step 4: Implement against the confirmed API**

```rust
fn build(items: &[Vec<f32>]) -> Self {
    let dim = items.first().map(|v| v.len()).unwrap_or(0);
    let options = usearch::IndexOptions {
        dimensions: dim,
        metric: usearch::MetricKind::IP,
        quantization: usearch::ScalarKind::F32,
        ..Default::default()
    };
    let index = usearch::Index::new(&options).expect("usearch index");
    index.reserve(items.len()).expect("reserve");
    for (i, v) in items.iter().enumerate() {
        index.add(i as u64, v).expect("add");
    }
    Self { index, dim }
}

fn search(&self, query: &[f32], k: usize, exclude: &[usize]) -> Vec<(usize, f32)> {
    let ex: std::collections::HashSet<usize> = exclude.iter().copied().collect();
    // over-fetch to survive post-filtering of excluded ids.
    let m = self.index.search(query, k + exclude.len()).expect("search");
    m.keys.iter().zip(m.distances.iter())
        .map(|(&key, &dist)| (key as usize, dist))
        .filter(|(i, _)| !ex.contains(i))
        .take(k)
        .collect()
}

fn index_bytes(&self) -> usize {
    self.index.memory_usage()
}
```
Note: usearch IP returns *similarity-like* distances; confirm sign/order against docs.rs and flip if the bench's recall against `exact_top_k` (which sorts by descending dot) comes out near zero.

**Step 5: Run, verify pass**

Run: `cargo test --features ann --lib ann::usearch_backend`
Expected: PASS.

**Step 6: Commit**

```bash
git add src/ann/mod.rs src/ann/usearch_backend.rs
git commit -m "feat(ann): usearch AnnBackend impl (IP metric)"
```

---

## Task 5: turbovec backend (TDD)

**Files:**
- Create: `src/ann/turbovec_backend.rs`
- Modify: `src/ann/mod.rs` (`pub mod turbovec_backend;`)

**Step 1: Confirm the turbovec Rust API**

Read https://docs.rs/turbovec/0.9 — confirm the index type, constructor, and `add` / `add_with_ids` / `search(query, k, allowlist)` signatures (the README advertises these; the exact Rust types must be read from docs.rs, as 0.9 is pre-1.0). Map `allowlist` onto the exclude set (invert: allowlist = all ids minus excluded), or post-filter if an allowlist isn't ergonomic. Record the actual signatures in a top-of-file comment.

**Step 2: Write the failing test**

Mirror Task 4's structure exactly (`recovers_exact_top1_on_clustered_data`), with `TurbovecBackend` and `unimplemented!()` bodies. Note: TurboVec quantizes (2/4-bit), so on clustered data assert top-1 recovery but allow that recall@K at large K may be < 1.0 — the test asserts only top-1 cluster recovery, which quantization should preserve on well-separated clusters.

```rust
#[test]
fn recovers_exact_top1_on_clustered_data() {
    let items = clustered(20); // same helper as Task 4
    let be = TurbovecBackend::build(&items);
    let query = vec![1.0, 0.0, 0.0];
    assert_eq!(be.search(&query, 1, &[])[0].0,
               crate::ann::exact::exact_top_k(&query, &items, 1)[0].0);
}
```

**Step 3: Run, verify it fails**

Run: `cargo test --features ann --lib ann::turbovec_backend`
Expected: FAIL.

**Step 4: Implement against the confirmed API**

Fill `build` / `search` / `index_bytes` using the signatures recorded in Step 1. `index_bytes` is turbovec's headline axis — use its reported quantized footprint if exposed, else estimate from `num_items * dim * bits / 8`.

**Step 5: Run, verify pass**

Run: `cargo test --features ann --lib ann::turbovec_backend`
Expected: PASS. If quantization drops even top-1 on this easy data, lower the quantization (4-bit) and record it; if still failing, flag turbovec as not meeting the recall bar and note in #76.

**Step 6: Commit**

```bash
git add src/ann/mod.rs src/ann/turbovec_backend.rs
git commit -m "feat(ann): turbovec AnnBackend impl (quantized)"
```

---

## Task 6: Two-Tower `item_embeddings()` accessor (TDD, `ml-models`)

`TrainedTwoTower.item_matrix` is private; the exporter needs a public accessor.

**Files:**
- Modify: `src/models/two_tower.rs`

**Step 1: Write the failing test**

In the `#[cfg(test)] mod tests` of `src/models/two_tower.rs`, add:
```rust
#[test]
fn item_embeddings_shape_matches_catalog() {
    let (data, uft, ift) = tiny_data(); // existing helper
    let model = train(&data, &uft, &ift, TrainParams { embedding_dim: 16, epochs: 5, ..Default::default() }).unwrap();
    let emb = model.item_embeddings();
    assert_eq!(emb.len(), model.num_items());        // one row per item
    assert!(emb.iter().all(|r| r.len() == 16));      // dim matches
}
```

**Step 2: Run, verify it fails**

Run: `cargo test --features ml-models --lib two_tower::tests::item_embeddings_shape_matches_catalog`
Expected: FAIL (`no method named item_embeddings`).

**Step 3: Implement the accessor**

On `impl TrainedTwoTower` (near `predict_similar_items`, which already reads `self.item_matrix`):
```rust
/// The full `(num_items, dim)` L2-normalized item-embedding matrix as
/// host rows. Used by the ANN bench exporter (ADR-0004 Phase 2).
pub fn item_embeddings(&self) -> Vec<Vec<f32>> {
    let dims = self.item_matrix.dims(); // [num_items, dim]
    let flat: Vec<f32> = self.item_matrix.clone().into_data().convert::<f32>().to_vec().unwrap();
    flat.chunks(dims[1]).map(|c| c.to_vec()).collect()
}
```
(Confirm the `into_data().to_vec()` form against the existing extraction at `two_tower.rs:964-969`, which already pulls rows out of `item_matrix`; match that idiom exactly.)

**Step 4: Run, verify pass**

Run: `cargo test --features ml-models --lib two_tower::tests::item_embeddings_shape_matches_catalog`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/models/two_tower.rs
git commit -m "feat(two-tower): public item_embeddings() accessor for ANN export"
```

---

## Task 7: Embedding exporter binary (`ml-models`)

**Files:**
- Create: `src/bin/export_tt_embeddings.rs`

**Step 1: Write the exporter**

```rust
//! Train a small Two-Tower on synthetic interactions and dump its item
//! embedding matrix to a flat little-endian f32 file:
//!   [u32 num_items][u32 dim][f32 * num_items * dim]
//! Consumed by `benches/ann_retrieval.rs` for the recall decision run.
//! Run: cargo run --release --features ml-models --bin export_tt_embeddings -- out.f32 50000 64

use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args.get(1).cloned().unwrap_or_else(|| "tt_embeddings.f32".into());
    let n_items: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(50_000);
    let dim: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);

    // Build synthetic TripleData with cluster structure so embeddings are
    // realistic (not uniform noise). Reuse the crate's data/triples types.
    let model = rust_fease_recommender::ann_export::train_synthetic(n_items, dim);
    let emb = model.item_embeddings();

    let mut f = std::io::BufWriter::new(std::fs::File::create(&out).unwrap());
    f.write_all(&(emb.len() as u32).to_le_bytes()).unwrap();
    f.write_all(&(dim as u32).to_le_bytes()).unwrap();
    for row in &emb {
        for &x in row {
            f.write_all(&x.to_le_bytes()).unwrap();
        }
    }
    eprintln!("wrote {} items x {} dim to {}", emb.len(), dim, out);
}
```

**Step 2: Add `train_synthetic` helper behind `ml-models`**

Create `src/ann_export.rs` (gated `#[cfg(feature = "ml-models")]`, declared in `lib.rs`) with `pub fn train_synthetic(n_items: usize, dim: usize) -> crate::models::two_tower::TrainedTwoTower`. It builds `TripleData` with `n_items` items and a few thousand synthetic users whose positives are drawn from a handful of latent clusters (so item embeddings acquire structure), `FeatureTable::empty(..)` for both sides, and calls `two_tower::train(.., TrainParams { embedding_dim: dim, epochs: 20, ..Default::default() })`. Confirm `TripleData` / `FeatureTable` constructors against `src/data/triples.rs`.

**Step 3: Build + run on a small size, verify output**

Run: `cargo run --release --features ml-models --bin export_tt_embeddings -- /tmp/tt.f32 2000 32`
Expected: stderr `wrote 2000 items x 32 dim`, and `/tmp/tt.f32` is `8 + 2000*32*4` bytes.

**Step 4: Commit**

```bash
git add src/bin/export_tt_embeddings.rs src/ann_export.rs src/lib.rs
git commit -m "feat(ann): Two-Tower embedding exporter for the recall decision run"
```

---

## Task 8: Wire backends + real embeddings into the bench

**Files:**
- Modify: `benches/ann_retrieval.rs`

**Step 1: Replace the bench's local `exact_top_k`/`recall_at_k` with the library ones**

The bench is `harness = false`; it can `use rust_fease_recommender::ann::{...}` only when built with `--features ann`. Gate the bench body so `cargo bench --features ann` exercises the backends and `cargo bench` (no feature) still runs the synthetic exact baseline using a local fallback. Delete the now-duplicated local `exact_top_k`/`recall_at_k` definitions (dedupe — resolves the second #76 debt item) under the `ann` path.

**Step 2: Add an embedding loader**

```rust
fn load_embeddings(path: &str) -> Vec<Vec<f32>> {
    let bytes = std::fs::read(path).expect("embedding file");
    let n = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    bytes[8..].chunks(dim * 4)
        .take(n)
        .map(|row| row.chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
        .collect()
}
```
Embeddings come from `$ANN_EMB` env var if set (real export), else `gen_embeddings` (synthetic).

**Step 3: Bench each backend + print recall**

For each backend (`ExactBackend`, `UsearchBackend`, `TurbovecBackend`), under `#[cfg(feature = "ann")]`:
```rust
let be = Backend::build(&items);
group.bench_function(BenchmarkId::new("usearch", n), |b| {
    b.iter(|| be.search(black_box(&query), TOP_K, &[]));
});
// recall is data-quality, not timing — print once, outside the timing loop:
let approx = be.search(&query, TOP_K, &[]);
let exact = exact_top_k(&query, &items, TOP_K);
eprintln!("recall@{TOP_K} usearch n={n}: {:.4}  index_bytes={}", recall_at_k(&approx, &exact), be.index_bytes());
```

**Step 4: Run synthetic (wiring check)**

Run: `cargo bench --features ann --bench ann_retrieval -- --sample-size 10`
Expected: latencies for exact/usearch/turbovec; recall lines printed. (Synthetic recall is expected low — that's the documented limitation.)

**Step 5: Run real (decision)**

Run:
```bash
cargo run --release --features ml-models --bin export_tt_embeddings -- /tmp/tt.f32 200000 64
ANN_EMB=/tmp/tt.f32 cargo bench --features ann --bench ann_retrieval
```
Expected: recall@20 for usearch and turbovec on real geometry + latency + index_bytes.

**Step 6: Commit**

```bash
git add benches/ann_retrieval.rs
git commit -m "bench(ann): wire usearch + turbovec backends and real-embedding loader"
```

---

## Task 9: Record the decision

**Files:**
- Modify: `benches/README.md` (results table)

**Step 1: Fill the results table** in `benches/README.md` with measured recall@20 / p50 / p99 / index_bytes for usearch and turbovec at 100k + 1M on the real embeddings.
**Step 2: Post the decision** as a comment on #76: which backend wins on the agreed criteria (recall floor first, then latency, then memory), and whether turbovec's memory win justifies its recall cost.
**Step 3: Commit + mark PR #79 ready for review** (un-draft) once the decision is recorded.

```bash
git add benches/README.md
git commit -m "docs(ann): record backend bench-off results and decision (#76)"
```

The follow-up — implementing `RetrievalIndex` for Two-Tower with the winning backend in the serving path, behind `ann` — is a **separate** plan/PR (the seam from #75 is ready for it).

---

## Self-Review

**Spec coverage** (against #76 + the two decisions):
- "Add `ann` Cargo feature" → Task 1. ✓
- "RetrievalIndex impl for Two-Tower over a benched backend" → backends in Tasks 4–5; serving impl explicitly deferred to a follow-up (only the winner gets wired into serving — avoids building two throwaway integrations). ✓ (scoped)
- "bench TurboVec vs usearch: recall@K, p50/p99, memory" → Tasks 8–9. ✓
- "real Two-Tower embeddings for the decision" → Tasks 6–7. ✓
- "promote + unit-test bench utilities; dedupe exact baseline" (#76 debt) → Tasks 2 and 8 Step 1. ✓
- "exclude-list ↔ native filtered search" → usearch over-fetch+filter (Task 4); turbovec allowlist (Task 5 Step 1). ✓
- "CI recall gate; default/ml-models builds unchanged" → Task 1 Steps 2–3 (build invariance); the CI recall-gate wiring is part of the serving follow-up, noted here, not in scope.

**Placeholder scan:** The usearch/turbovec `build`/`search` bodies depend on each crate's exact Rust signatures, which Tasks 4–5 Step 1 require reading from docs.rs before implementing — this is a deliberate verify-then-write step, not a placeholder; the surrounding structure, tests, and expected behavior are fully specified. The `clustered` test helper carries scaffolding noted for cleanup.

**Type consistency:** `AnnBackend::{build, search, index_bytes}` signatures are identical across `ExactBackend`, `UsearchBackend`, `TurbovecBackend` and the bench call sites. `exact_top_k`/`recall_at_k` signatures match between `src/ann/exact.rs`, the backends, and the bench. `item_embeddings() -> Vec<Vec<f32>>` matches the exporter's consumption and the `load_embeddings` round-trip format.
