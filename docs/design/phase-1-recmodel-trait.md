# Phase 1 design: `RecModel` trait + EASE adapter

- **Status**: Proposed
- **Date**: 2026-05-10
- **Implements**: ADR-0001 Phase 1
- **Scope**: scaffolding only — no behavior change, no new dependencies, no
  algorithmic edits.

## Goal

Land the abstraction without changing what FEASE does. After this phase:

- `RustFeaseModel` is reachable both as today's concrete type *and* through
  a new `&dyn RecModel` interface via `models::ease::EaseAdapter`.
- The PyO3 `FeaseModel` class is byte-identical externally — same methods,
  same signatures, same outputs.
- All existing tests pass without modification.
- The trait is in place for Phases 2–6 to plug new models into.

This is deliberately the smallest non-trivial change that proves the
abstraction works. We do not move evaluation, tuning, serialization, or the
registry to use the trait yet — those generalizations land in Phases 4 and
6, after the second model exists to validate the trait shape under real
pressure.

## Non-goals

- No `burn` dependency in this phase (Phase 2).
- No new models (Phases 3, 5).
- No changes to `evaluation.rs`, `tuning.rs`, `serialization.rs`, or
  `serving.rs` consumers of `RustFeaseModel`. Those still take the concrete
  type. Generalizing them prematurely is the kind of speculative
  abstraction that bites — wait for the second model.
- No file format changes. EASE save/load remains identical.
- No public Python API changes.

## Module structure

New directory:

```
src/models/
  mod.rs            # trait + ModelInput + ModelKind + ValidationReport re-export
  ease.rs           # EaseAdapter wrapping RustFeaseModel
```

Untouched in this phase:

```
src/lib.rs                  # FeaseModel PyO3 class still uses RustFeaseModel directly
src/model.rs                # RustFeaseModel — no changes
src/data_pipeline.rs        # build_matrices — no changes
src/evaluation.rs           # still &RustFeaseModel
src/tuning.rs               # still concrete
src/serialization.rs        # FEAS magic, no changes
src/serving.rs              # ModelRegistry — no changes
src/{metrics,weighting,data_validation}.rs  # no changes
```

## The trait

```rust
// src/models/mod.rs

use crate::data_pipeline::Mappings;
use crate::model::ValidationReport;
use anyhow::Result;
use std::path::Path;

pub mod ease;

/// What kind of model this is. Used by callers that need to construct
/// model-appropriate input shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    Ease,
    SasRec,    // reserved for Phase 3
    TwoTower,  // reserved for Phase 5
}

/// Input passed to `predict_scores`. Each variant maps to a model family.
/// A model that receives an unsupported variant returns
/// `Err(UnsupportedInput)`.
#[derive(Debug)]
pub enum ModelInput<'a> {
    /// EASE: sparse interaction values + sparse user-feature values.
    /// `interactions: &[(item_idx, value)]`
    /// `user_features: &[(feature_idx, value)]`
    Sparse {
        interactions: &'a [(usize, f64)],
        user_features: &'a [(usize, f64)],
    },
    /// SASRec: a chronologically-ordered list of item indices (oldest first,
    /// most recent last). Reserved for Phase 3.
    Sequence { history: &'a [usize] },
    /// Two-Tower: user-side input. Reserved for Phase 5.
    TowerUser {
        user_idx: Option<usize>,
        cat_features: &'a [usize],
        dense_features: &'a [f32],
    },
}

/// Common interface for all recommender models.
///
/// Implementors are expected to be `Send + Sync` so that
/// `Arc<dyn RecModel>` can be used freely in serving paths.
pub trait RecModel: Send + Sync {
    fn kind(&self) -> ModelKind;

    /// Number of items in the model's catalog. Score vectors returned by
    /// `predict_scores` have this length.
    fn num_items(&self) -> usize;

    /// Mappings from string IDs to indices and back. Shared by all models
    /// because the data pipeline builds them once for the dataset.
    fn item_mapping(&self) -> &Mappings;

    /// Score every item in the catalog for the given input. Returns a
    /// `Vec<f32>` of length `num_items()`.
    ///
    /// Returns `Err` if the input variant is not supported by this model.
    fn predict_scores(&self, input: ModelInput<'_>) -> Result<Vec<f32>>;

    /// Top-K items most similar to a given item, by the model's notion of
    /// item similarity. EASE uses the item-item block of S; sequence and
    /// tower models will use embedding cosine.
    fn predict_similar_items(
        &self,
        item_idx: usize,
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>>;

    /// Self-check the model state. Same shape as the existing
    /// `RustFeaseModel::validate`.
    fn validate(&self) -> ValidationReport;

    /// Persist the model to disk. Phase 1 routes this to the existing
    /// EASE serializer; Phases 3 and 5 add per-model magic bytes.
    fn save(&self, path: &Path) -> Result<()>;
}
```

### Why `Vec<f32>` and not `Vec<f64>`?

EASE today returns `Vec<f64>` from `predict()`. Burn defaults to `f32` for
tensor params. Standardizing the trait on `f32` avoids per-call casting in
the hot path for the SGD models. The EASE adapter does a single
`as f32` cast per element on output — measured cost is negligible and
ranking quality is unchanged at f32 precision.

This is a conscious deviation from the existing `RustFeaseModel::predict`
return type. The concrete `RustFeaseModel::predict` keeps its `Vec<f64>`
signature for callers that use it directly.

### Why an enum, not generics?

Detailed in ADR-0001 §"Alternatives B". TL;DR: `&dyn RecModel` matters for
the registry and for a future model-agnostic eval harness; generic
associated types break dyn dispatch.

### Why is `save` on the trait but `load` is not?

`load` needs to *construct* the right concrete type from a file. That
inverts the polymorphism — it's a free function `load_model(path)` that
peeks the magic bytes and dispatches. Lives in `serialization.rs`.

## EASE adapter

```rust
// src/models/ease.rs

use super::{ModelInput, ModelKind, RecModel};
use crate::data_pipeline::Mappings;
use crate::model::{RustFeaseModel, ValidationReport};
use crate::serialization;
use anyhow::{Result, anyhow};
use std::path::Path;

/// Wraps `RustFeaseModel` to expose it via the `RecModel` trait.
/// Owns the model by value — same ownership pattern as
/// `lib.rs::FeaseModel`.
pub struct EaseAdapter {
    pub inner: RustFeaseModel,
}

impl EaseAdapter {
    pub fn new(inner: RustFeaseModel) -> Self {
        Self { inner }
    }
}

impl RecModel for EaseAdapter {
    fn kind(&self) -> ModelKind { ModelKind::Ease }

    fn num_items(&self) -> usize { self.inner.num_items }

    fn item_mapping(&self) -> &Mappings { &self.inner.mappings }

    fn predict_scores(&self, input: ModelInput<'_>) -> Result<Vec<f32>> {
        match input {
            ModelInput::Sparse { interactions, user_features } => {
                let scores_f64 = self.inner.predict(
                    interactions,
                    user_features,
                    self.inner.beta,
                );
                Ok(scores_f64.into_iter().map(|x| x as f32).collect())
            }
            other => Err(anyhow!(
                "EASE does not support {:?} input; expected ModelInput::Sparse",
                std::mem::discriminant(&other),
            )),
        }
    }

    fn predict_similar_items(
        &self,
        item_idx: usize,
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        Ok(self.inner.predict_similar_items(item_idx, top_k)
            .into_iter()
            .map(|(i, s)| (i, s as f32))
            .collect())
    }

    fn validate(&self) -> ValidationReport { self.inner.validate() }

    fn save(&self, path: &Path) -> Result<()> {
        serialization::save_model(&self.inner, path)
            .map_err(|e| anyhow!(e))
    }
}
```

This is the entire adapter. ~50 lines. No new logic — every method
delegates to the existing `RustFeaseModel`. The `as f32` casts are the
only computation introduced.

## `lib.rs` integration

`src/lib.rs` gains exactly one line at the top:

```rust
mod models;
```

The existing `FeaseModel` PyO3 class is **not** modified. It continues to
hold and call `RustFeaseModel` directly. This is intentional — the trait is
available, but routing the PyO3 layer through it doesn't pay off until we
have at least one second implementation. Adding an indirection that has
exactly one impl is the kind of speculative abstraction we're trying to
avoid.

The trait is exercised in this phase only by tests (next section). Real
PyO3 routing through `&dyn RecModel` lands in Phase 4 alongside
`SASRecModel`.

## Tests

New file: `src/models/mod.rs` `#[cfg(test)] mod tests`. Contents:

1. **Adapter is a no-op semantically**:
   - Build a tiny `RustFeaseModel` (4 items, 2 user features) by training
     on synthetic data via `data_pipeline::build_matrices` + `train`.
   - Call `model.predict(...)` directly to get baseline scores.
   - Wrap in `EaseAdapter`, call `predict_scores(ModelInput::Sparse {...})`.
   - Assert the f32 result, when cast back to f64, matches baseline within
     `1e-6`.

2. **Wrong-input rejection**:
   - `EaseAdapter::predict_scores(ModelInput::Sequence { history: &[] })`
     returns `Err`.

3. **Similar items roundtrip**:
   - `EaseAdapter::predict_similar_items(0, 5)` returns the same item
     indices in the same order as `RustFeaseModel::predict_similar_items`.

4. **`Send + Sync`**:
   - One-line const assertion: `const _: fn() = || { fn assert_send_sync<T: Send + Sync>() {} assert_send_sync::<EaseAdapter>(); };`

Existing tests in `src/data_pipeline.rs`, `src/model.rs` (if any), the
Python integration tests in `tests/`, and the eval/tuning paths must all
continue passing without modification. That is the regression bar for this
phase.

## Compatibility

- **Rust API**: additive. New `models` module, no breaking changes.
- **Python API**: zero changes.
- **File format**: zero changes.
- **`Cargo.toml`**: zero changes.
- **MSRV**: unchanged.
- **Wheel size**: unchanged.

## Out-of-band review checklist

- [ ] Trait imports compile when `models::RecModel` is used as `&dyn RecModel`
      (verifies object-safety; the GAT alternative would fail this).
- [ ] `cargo test` passes.
- [ ] `.venv/bin/python -m pytest tests/ -v` passes.
- [ ] `.venv/bin/maturin develop` produces a working wheel that imports
      cleanly into Python.
- [ ] No new TODO/FIXME comments introduced.

## What this PR contains

The PR that accompanies this design doc contains **only the docs** — the
ADR and this design doc. Implementation lands in a follow-up PR after
review of the architectural direction. Splitting the docs from the code
keeps the architectural review separate from the implementation review,
and lets the research subagent's findings (burn API specifics) inform
Phase 2 without delaying Phase 1.
