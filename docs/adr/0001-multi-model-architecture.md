# ADR-0001: Multi-model architecture for FEASE

- **Status**: Proposed
- **Date**: 2026-05-10
- **Deciders**: project maintainers
- **Supersedes**: —
- **Related**: design doc `docs/design/phase-1-recmodel-trait.md`

## Context

FEASE today ships exactly one recommender model — the closed-form linear EASE
variant in `src/model.rs` (`RustFeaseModel`). The training surface, evaluation
harness (`src/evaluation.rs`), tuning harness (`src/tuning.rs`), serialization
format (`src/serialization.rs`), and PyO3 bridge (`src/lib.rs`) all reference
`RustFeaseModel` concretely.

We want to add two more models alongside EASE:

1. **SASRec** — causal self-attention sequence model trained by SGD.
2. **Two-Tower** — separate user/item embedding networks, trained with
   in-batch sampled softmax. Must support categorical *and* dense numerical
   features.

Both new models require:
- automatic differentiation,
- mini-batch SGD with an optimizer (Adam),
- learned tensor parameters that need to be saved and loaded,
- input shapes that don't fit the existing `(CsMat<f64>, CsMat<f64>, CsMat<f64>)`
  long-format pipeline (sequences, triples, dense vectors).

The current code has no abstraction over "what is a recommender model" —
there is exactly one type, and every consumer references it directly.
Adding two more concrete types without a shared interface would force every
consumer (eval, tuning, registry, PyO3) into per-type branches.

## Decision

We will introduce a `RecModel` trait in a new `src/models/` module that all
three models implement. The existing `RustFeaseModel` is wrapped behind it
without algorithmic changes. Concretely:

1. **Abstraction lives in Rust**, not Python. The trait is defined in
   `src/models/mod.rs`, and the PyO3 layer exposes one `#[pyclass]` per model
   (`FeaseModel`, `SASRecModel`, `TwoTowerModel`) — all duck-typed to the
   same Python protocol.

2. **`burn` is the autodiff/tensor framework** for the new models. Backend is
   parameterized; we pin `burn-ndarray` (CPU) for the initial cut, with the
   seam in place to swap to `burn-tch` or `burn-wgpu` later without changing
   model code.

3. **Inputs are an enum, not a generic associated type**:
   ```rust
   pub enum ModelInput<'a> {
       Sparse { interactions: &'a [(usize, f64)],
                user_features: &'a [(usize, f64)] },
       Sequence { history: &'a [usize] },
       TowerUser { user_idx: Option<usize>,
                   cat_features: &'a [usize],
                   dense_features: &'a [f32] },
   }
   ```
   This keeps `&dyn RecModel` viable, which the registry and eval harness
   both rely on. Each model branches on the variant and errors loudly on
   shapes it doesn't accept.

4. **Per-model serialization**: keep `FEAS` magic for EASE (back-compat).
   Add `FSAS` for SASRec, `FTWO` for two-tower. The format header is
   `magic[4] || version[u32] || metadata[bincode] || params_blob`. EASE's
   `params_blob` stays nalgebra-serialized; SASRec/two-tower use burn's
   `Recorder` API.

5. **Generalization order**: scaffolding first (Phase 1), then add new
   models, then generalize the cross-cutting consumers. See the phased
   plan below.

6. **Cargo feature gate for ML models**: the `burn` dependency and the
   `SASRec` / `TwoTower` implementations live behind a `ml-models` Cargo
   feature, **default off**. The `RecModel` trait, the `ModelInput` enum,
   and the EASE adapter compile unconditionally. Consumers who only need
   EASE see no change in their build — same dependency tree, same wheel
   size, same compile time. Opting in is `cargo build --features
   ml-models` (or the equivalent maturin flag). The PyO3 layer
   conditionally compiles `SASRecModel` and `TwoTowerModel` under the
   same feature; Python callers either get the slim module (EASE only)
   or the full one. Whether to ship one vs. two wheels on PyPI is
   deferred to Phase 2, once we can measure real binary sizes.

## Alternatives considered

### A. Python-level abstraction with PyTorch
Define the `Model` protocol in Python; new models in PyTorch; EASE keeps
its Rust implementation and is wrapped on the Python side.

- ✅ Largest community + reference implementations for SASRec/two-tower.
- ✅ Free GPU support.
- ❌ Adds ~2GB PyTorch dependency to every install; CPU/GPU wheel split.
- ❌ Splits the codebase across two languages for "the same kind of thing"
  (a recommender model). Harder to reason about, harder to maintain a
  shared eval harness.
- ❌ User explicitly chose to stay Rust-native.

**Rejected** — explicit user direction.

### B. Generic associated type on the trait
```rust
trait RecModel { type Input<'a>; fn predict(&self, x: Self::Input<'_>) -> ...; }
```

- ✅ Type-safe per-model inputs.
- ❌ Breaks `&dyn RecModel`. The registry and eval harness become
  type-erased manually or duplicated per model. Significant complexity tax
  for a property we don't need (the inputs come from Python at runtime
  anyway).

**Rejected** — `dyn` ergonomics matter more than input type safety here.

### C. Separate trait per model family
`SparseRecModel`, `SequenceRecModel`, `TowerRecModel`. Eval/tuning code
takes a different trait per family.

- ✅ Cleaner per-family input contracts.
- ❌ Triplicates the eval and tuning harnesses. Defeats the goal of
  model-agnostic infrastructure.

**Rejected** — too much duplication for small ergonomic gain.

### D. `candle` instead of `burn`
HuggingFace's tensor library, used in production for inference workloads.

- ✅ Stable, smaller surface, less API churn.
- ❌ Inference-leaning; autodiff and training utilities are far less
  developed than burn's `burn-train`. We need training, not just inference.

**Rejected** — wrong tool for the job.

### E. Hand-rolled NumPy-style autodiff in pure Rust (`ndarray` + manual gradients)
Skip ML frameworks entirely; implement attention and SGD by hand.

- ✅ Zero new dependencies.
- ❌ Implementing transformer backprop correctly is weeks of work and a
  permanent maintenance burden. No community to crib from.

**Rejected** — false economy.

## Consequences

### Positive
- One trait → one eval harness, one tuning harness, one registry. Adding a
  fourth model later (e.g. NCF, GRU4Rec) is a single-file change plus a
  PyO3 wrapper.
- EASE behavior unchanged — the wrapper is a no-op semantically. Existing
  `FeaseModel` Python class and `build_and_train` entrypoint keep their
  signatures.
- `burn`'s backend abstraction means GPU support is a future config flag,
  not a rewrite.
- Per-model magic bytes in the file format mean `load_model(path)` can
  auto-detect type — Python callers don't need to know which model wrote a
  file.
- Cargo feature gate means EASE-only users see no change in their build
  or wheel: the new code is opt-in at compile time, not at runtime.

### Negative / costs
- New dependency: `burn` (pre-1.0, expect breaking changes between minor
  versions). Pinning required; bump cost is real.
- ~30s clean-build time increase from burn + transitive deps.
- CPU-only training is slow for non-toy datasets. Documented as a known
  limitation; GPU backends remain a follow-up.
- SASRec hard-requires the `days_ago` column for sequence ordering. EASE
  treats it as optional. We will fail loudly at SASRec training time with
  a clear error rather than silently using row order.
- Cold-start gap: pure SASRec needs a non-empty interaction history. Users
  with zero interactions get a popularity prior. Two-tower handles this
  natively via features. Documented in user-facing docs.

### Risks
- **`burn` API churn**: pinned version + a single update PR per bump
  contains the blast radius. If a bump becomes prohibitive, the backend-
  parameterized seam means we could swap frameworks without rewriting
  consumers.
- **Trait shape lock-in**: the `ModelInput` enum is an explicit extension
  point. Adding a new variant is a breaking change to the trait but not
  to the public Python API. Acceptable given it's an internal Rust
  abstraction.

### Backout / baseline

Commit `812ddbe` on `main` (the merge of PR #19, immediately before this
architectural work begins) is tagged `pre-multi-model` as an immortal
reference point. If the multi-model direction has to be unwound, that
tag is the recovery target. We are explicitly **not** maintaining a
long-lived parallel "simple EASE" branch — the Cargo feature gate gives
EASE-only users a slim build on `main`, and the tag covers the "preserve
a snapshot of the simpler codebase" need. Maintaining a parallel branch
would force every EASE bugfix to be applied twice without buying
anything the feature gate doesn't already provide.

## Phased rollout

| Phase | Scope | Gate |
|-------|-------|------|
| **1** | `RecModel` trait + `models/ease.rs` adapter wrapping `RustFeaseModel`. No behavior change. | All existing tests green; `FeaseModel` PyO3 surface byte-identical. |
| 2 | Add `burn` dep as an **optional dependency** gated on the `ml-models` feature; minimal SASRec forward pass compiles. | `cargo build` (no features) builds without pulling burn; `cargo build --features ml-models` succeeds; default wheel size unchanged. |
| 3 | SASRec training loop + `data/sequences.rs` + Python smoke test. | Overfit-on-tiny-data Rust test passes; Python `train → predict → save → load` roundtrip green. |
| 4 | Generalize `evaluation::evaluate_model` to `&dyn RecModel`; add `SASRecModel` PyO3 class. | Existing EASE eval still produces identical numbers; SASRec eval runs. |
| 5 | Two-Tower model + `data/triples.rs` + dense feature loader. | Same gates as Phase 3, plus dense-feature loader unit-tested. |
| 6 | Generalize `tuning` + `serving` registry. Per-model PyO3 search entrypoints. | Existing EASE grid search still produces identical results; SASRec/Two-Tower searches run end-to-end. |
| 7 | Docs pass: update `CLAUDE.md`, `README.md`, and add a model-comparison guide. | Docs reviewed; PR merges. |

This ADR covers the shape of the decision; Phase 1 is detailed in the
companion design doc and is what the accompanying PR proposes to land
first.

## References
- Steck, "Embarrassingly Shallow Autoencoders" (EASE), 2019.
- Kang & McAuley, "Self-Attentive Sequential Recommendation" (SASRec), 2018.
- Yi et al., "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item
  Recommendations", RecSys 2019 (two-tower with sampled softmax).
- `burn` framework: https://burn.dev/
