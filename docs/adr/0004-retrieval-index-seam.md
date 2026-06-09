# ADR-0004: Optional retrieval-index seam for embedding-based models

- **Status**: Proposed
- **Date**: 2026-06-09
- **Deciders**: project maintainers
- **Supersedes**: —
- **Related**: ADR-0001 (multi-model architecture, `RecModel` trait), ADR-0002
  (opt-in Cargo-feature precedent for `fast-blas`)

## Context

Every model returns a **dense, full-catalog score vector**. The `RecModel`
contract in `src/models/mod.rs` is:

```rust
fn predict_scores(&self, input: ModelInput<'_>) -> Result<Vec<f32>>; // len == num_items
```

and serving turns that vector into a ranking in `src/serving.rs::filter_sort_top_k`.
This is exhaustive ("brute-force") retrieval: score all `num_items`, then select
the top-K. As of the change that immediately precedes this ADR, the *selection*
step is an O(n) quickselect partition + O(K log K) sort rather than a full
O(n log n) sort — but the *scoring* step is still O(num_items) work, and for the
embedding models it still materializes the full `(num_items, dim)` item matrix in
memory.

For **EASE** — the default and only-published model — this is the right design and
there is nothing to improve here: EASE has no query embedding and no item vector
space. Its score is `interactions · S` (sparse × S-matrix); "retrieval" is
inseparable from the linear algebra. EASE is out of scope for everything below.

For the `ml-models`-gated embedding models the picture differs:

- **Two-Tower** produces an L2-normalized user vector and an L2-normalized
  `(num_items, dim)` item matrix; `score_all` (`src/models/two_tower.rs`) is a
  single matmul over that matrix. This is *exactly* a maximum-inner-product /
  cosine nearest-neighbor query — the canonical problem approximate-nearest-
  neighbor (ANN) indexes solve in sublinear time and (with quantization)
  a fraction of the memory.
- **`predict_similar_items`** for Two-Tower and SASRec is item-item embedding
  similarity — the same NN query against the item matrix.

Two scaling walls appear as catalogs grow into the hundreds of thousands to
millions of items, *for embedding models only*:

1. **Latency**: every prediction is O(num_items · dim) regardless of K.
2. **Serving memory**: the full f32 item matrix must be resident.

ANN libraries (FAISS, `usearch`, `hora`, and the quantization-focused
[TurboVec](https://github.com/RyanCodrai/turbovec)) address both: sublinear
query time and 4–16× memory compression via scalar/product quantization. They
trade *exactness* for those gains — results are approximate.

That trade-off collides with two project invariants:

- **Determinism / byte-identical outputs** is a stated guarantee (CLAUDE.md). ANN
  results are approximate and, depending on the index, non-deterministic.
- **Evaluation and tuning** optimize NDCG@k over `evaluate_model` / the search
  harnesses. Approximate retrieval inside the metric loop would conflate "the
  model is worse" with "retrieval recalled fewer true positives," corrupting
  comparisons.

The question this ADR settles is *where ANN may live* such that we get the
serving wins without paying for them in EASE builds, in eval correctness, or in
determinism guarantees we've made — and *whether to commit to a specific library*
now.

## Decision

We will introduce an **optional `RetrievalIndex` capability** that embedding
models may expose, consumed only by the serving layer, behind a new `ann` Cargo
feature (**default off**). We will **not** put ANN inside `predict_scores`, and we
will **not** touch the EASE or eval/tuning paths.

Concretely:

1. **A capability trait, not a change to `predict_scores`.** `predict_scores`
   keeps its contract — "score every item, exactly" — and stays the path eval,
   tuning, and `predict_batch` use. We add an *optional* sibling capability:

   ```rust
   /// Approximate top-K retrieval without scoring the full catalog.
   /// Implemented by embedding models; EASE does not implement it.
   pub trait RetrievalIndex: Send + Sync {
       /// Top-K (item_idx, score) for the given input, excluding `exclude`
       /// (already-interacted items). The model computes its own query
       /// embedding internally (e.g. Two-Tower runs the user tower forward),
       /// then queries the ANN structure.
       fn retrieve(
           &self,
           input: ModelInput<'_>,
           top_k: usize,
           exclude: &[usize],
       ) -> Result<Vec<(usize, f32)>>;
   }

   pub trait RecModel: Send + Sync {
       // ... existing methods unchanged ...

       /// Returns `Some` if this model supports approximate retrieval.
       /// Default `None` → callers fall back to dense scoring + selection.
       fn retrieval_index(&self) -> Option<&dyn RetrievalIndex> { None }
   }
   ```

   Passing the whole `ModelInput` (rather than a raw query vector) keeps the
   user-tower forward pass *inside the model* and keeps serving model-agnostic —
   the same property ADR-0001's `ModelInput` enum was designed to preserve.

2. **Serving routes through the capability when present; otherwise unchanged.**
   The `predict_top_k_*` methods become:

   ```rust
   match model.retrieval_index() {
       Some(ix) => ix.retrieve(input, top_k, &exclude),                  // ANN path
       None      => Ok(filter_sort_top_k(model.predict_scores(input)?,   // exact path
                                         &exclude, top_k)),
   }
   ```

   With the `ann` feature off, `retrieval_index()` is the default `None` and every
   model takes the exact path — bit-for-bit today's behavior.

3. **Library-agnostic seam; pick the backend by benchmark, not by ADR.** The
   trait names no library. The first concrete `RetrievalIndex` impl is a
   thin adapter over a chosen ANN crate, selected by a recall/latency/memory
   bench (Phase 2 below). `usearch` (mature, Rust-native, filtered search) and
   `TurboVec` (newer, quantization-first, strongest memory compression) are the
   leading candidates; the seam lets us swap or offer both without touching
   serving or the models.

4. **Exclude-list maps onto native filtered search.** The "drop already-interacted
   items" filter that `filter_sort_top_k` does today corresponds directly to the
   `allowlist`/filter parameter both candidate libraries expose, so the exact and
   approximate paths stay semantically aligned.

Explicit non-decisions (deferred or rejected):

5. **EASE** — permanently out of scope. No embedding space; `retrieval_index()`
   returns `None` and EASE serving is untouched.

6. **Eval and tuning** — stay on the exact dense path, always. `evaluate_model`
   and the search harnesses must call `predict_scores`, never `retrieve`, so NDCG
   reflects model quality, not index recall. This is a hard rule, not a default.

7. **Index persistence** (serialize the built ANN index alongside the model file)
   — deferred. Phase 1/2 build the index in-memory at load time from the already-
   persisted item embeddings. Persisting the quantized index (file-format bump,
   per-library format) is a later phase with its own consumer.

8. **Committing to TurboVec specifically, now** — rejected as premature (see
   Alternative D). We commit to the *seam*; the library is a benched, swappable
   detail.

## Alternatives considered

### A. Do nothing — keep exhaustive dense scoring everywhere
- ✅ Zero new code, deps, or feature flags. Exact and deterministic for all
  models. The preceding quickselect change already removed the sort bottleneck.
- ❌ Two-Tower serving stays O(num_items · dim) per query and holds the full f32
  item matrix resident. At million-item catalogs that is a latency and memory
  wall with no escape hatch.

**Rejected for large embedding-model catalogs** — but this *is* the right answer
for EASE and for small catalogs, which is why ANN is opt-in, not default.

### B. Put ANN inside `predict_scores`
Have the embedding models return an approximate full score vector (or a sparse
top-K) directly from `predict_scores`.

- ✅ One method; serving needs no routing.
- ❌ Violates the trait contract ("score *every* item"). `predict_scores` is what
  eval and tuning call — approximation would silently leak into NDCG measurement.
- ❌ Forces a sparse/partial return type onto a method whose every other
  implementor (and EASE) returns a dense exact vector.

**Rejected** — conflates the exact-scoring contract with an approximate serving
optimization. The capability trait keeps them separate.

### C. Do ANN in the Python layer
Keep Rust producing embeddings; build/query the index in Python (e.g. the
`turbovec` or `usearch` pip package) inside `inference.py` / `serving`.

- ✅ No Rust dep; uses each library's Python bindings directly.
- ❌ Inverts the project's "Rust core, thin Python" architecture (ADR-0001 §A):
  retrieval is core model behavior, not glue.
- ❌ Splits the item-embedding source of truth across the language boundary and
  duplicates the exclude-list logic that already lives in Rust serving.
- ❌ The Rust `ModelRegistry` serving path (`predict_top_k_*`) would gain no
  benefit at all.

**Rejected** — wrong layer; the same reasoning that rejected Python-level model
abstraction in ADR-0001.

### D. Commit to TurboVec now as the single ANN backend
Adopt TurboVec specifically, wire it directly into serving.

- ✅ Best-in-class memory compression (4–16×) and competitive-or-faster-than-FAISS
  query latency; Rust + Python, no managed services.
- ❌ Newer, single-maintainer project — production-dependency maturity risk
  relative to FAISS/`usearch`. ADR-0001 already flags `burn` pre-1.0 churn as a
  real cost; adding a second young core dependency without a bench is unforced.
- ❌ Hard-wiring any library into serving is exactly the coupling the capability
  trait avoids.

**Rejected as a *now* commitment** — TurboVec is a strong *candidate* to bench in
Phase 2. The seam (Decision #3) is what we commit to; the library is data-driven
and swappable. If TurboVec's memory win holds up under our recall bar, it wins on
its merits — through the seam, not around it.

### E. Exact GPU matmul instead of approximate ANN
Keep exact scoring but move the Two-Tower `score_all` matmul to GPU via a `burn`
GPU backend.

- ✅ Stays exact and deterministic; reuses ADR-0001's backend-parameterized seam.
- ❌ Doesn't touch serving *memory* — the full f32 item matrix still has to be
  resident (now in VRAM, which is scarcer).
- ❌ Adds a GPU serving dependency for what is fundamentally still O(num_items)
  work per query; ANN changes the complexity class, GPU only changes the constant.

**Rejected as a substitute** — orthogonal. A GPU backend could complement an exact
path for mid-size catalogs, but it is not an answer to the latency-*and*-memory
scaling wall that motivates this ADR. Tracked separately if a consumer appears.

## Consequences

### Positive
- **Sublinear serving retrieval** for embedding models at large catalogs, plus
  4–16× item-embedding memory compression (quantizing backend) — both opt-in via
  `--features ann`, both invisible to EASE-only and default builds.
- **Exact paths untouched.** Eval, tuning, `predict_batch`, EASE serving, and
  every default-feature build keep byte-identical behavior. The determinism
  guarantee is preserved where it is actually promised.
- **Library-swappable.** The bench picks the backend; switching `usearch` ↔
  TurboVec (or offering both behind sub-features) is an adapter change, not a
  serving or model change.
- **Clean seam reuse.** `ModelInput`-in / `(idx, score)`-out mirrors the existing
  trait shapes; the exclude-list reuses native filtered search. No new public
  Python surface in Phase 1.

### Negative / costs
- **Approximate results in the ANN serving path.** Recommendations from
  `retrieve` can differ from exact top-K. This is acceptable *only* in serving
  and *only* opt-in; it never reaches eval. We will publish a recall@K vs. exact
  number from the Phase 2 bench so users opting in know the trade.
- **A new opt-in dependency** (the chosen ANN crate) with its own build/version
  story — same class of cost as `fast-blas` (ADR-0002) and gated the same way.
- **Index build cost at load time** (until persistence lands, non-decision #7):
  constructing the index is O(num_items) work and transient memory on model load.
- **Two retrieval code paths to test.** Serving now has exact and approximate
  branches; both need coverage, and a regression test must assert the ANN path's
  recall stays above an agreed floor versus the exact path.

### Risks
- **Recall regression**: a mis-tuned index silently degrades recommendation
  quality. Mitigation: a recall@K-vs-exact gate in CI for the `ann` feature, with
  a documented floor; the exact path is always one feature-flag away.
- **Backend maturity** (esp. TurboVec): mitigated by the seam — the project never
  hard-depends on one library, and the bench is the gate.
- **Determinism in the ANN path**: some indexes are construction- or thread-order
  dependent. Mitigation: prefer a deterministic-construction configuration; if
  unavailable, document the ANN serving path as explicitly non-deterministic
  (the exact path remains the deterministic guarantee).
- **Feature-combination surface**: `ann` composes with `ml-models` (it is
  meaningless without it). CI must build `ml-models` and `ml-models,ann`;
  `retrieval_index()`'s default `None` keeps `ml-models`-without-`ann` correct.

## Phased rollout

| Phase | Scope | Gate |
|-------|-------|------|
| **1** | Define `RetrievalIndex` trait + `RecModel::retrieval_index()` default `None`; route `predict_top_k_*` through it with the dense fallback. **No ANN dependency, no behavior change** — every model still returns `None`. | All existing tests green; serving output byte-identical (default `None` path); new unit test asserts the routing falls back to `filter_sort_top_k`. |
| **2** | Add the `ann` Cargo feature and the first `RetrievalIndex` impl for Two-Tower over a benched backend (`usearch` vs. TurboVec on a representative catalog: recall@K, p50/p99 latency, resident memory). Wire `predict_similar_items` for Two-Tower/SASRec through the same index. | `cargo build` (no features) and `--features ml-models` unchanged; `--features ml-models,ann` builds and runs; bench report committed; CI recall@K-vs-exact gate above the agreed floor. |
| **3** | Persist the built index in the model file (format bump, per-backend) so load-time index construction is skipped. | Round-trip save/load test; load-time memory/latency measurably lower than Phase 2 rebuild-on-load. |

Phase 1 is a single dependency-free PR that establishes the seam and is safe to
land independently of any library choice. Phase 2 is where the TurboVec-vs-
alternatives question is actually answered, by measurement. Phase 3 is gated on a
real serving consumer feeling the load-time rebuild cost.

## References

- TurboVec (quantization-first ANN, Rust + Python): https://github.com/RyanCodrai/turbovec
- TurboQuant (the underlying data-oblivious quantizer): Google Research
- usearch (mature Rust-native ANN with filtered search): https://github.com/unum-cloud/usearch
- FAISS (the baseline both benchmark against): https://github.com/facebookresearch/faiss
- ADR-0001 §A (Rust-core abstraction) and the `ModelInput` enum design
- ADR-0002 (opt-in Cargo-feature pattern; `fast-blas` precedent)
- `src/serving.rs::filter_sort_top_k` (the exact selection path this seam falls back to)
