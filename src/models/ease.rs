//! `EaseAdapter` / `EaseAdapterRef` — expose `RustFeaseModel` via `RecModel`.
//!
//! Every method delegates to the existing concrete model. The only
//! computation introduced is the `as f32` cast on output scores; the
//! trait standardizes on f32 to match the burn-based models in later
//! phases.
//!
//! Two adapters share one set of delegation helpers:
//! - [`EaseAdapter`] owns a `RustFeaseModel` (used where the caller has
//!   a model by value).
//! - [`EaseAdapterRef`] borrows `&RustFeaseModel`, so paths that already
//!   hold the model (e.g. the PyO3 `evaluate`) avoid a deep clone of the
//!   multi-GB S matrix.

use super::{ModelInput, ModelKind, RecModel};
use crate::data_pipeline::Mappings;
use crate::model::{RustFeaseModel, ValidationReport};
use crate::serialization;
use anyhow::{Result, anyhow};
use std::path::Path;

// Shared delegation so the owning and borrowing adapters cannot drift.

fn rec_num_items(m: &RustFeaseModel) -> usize {
    m.num_items
}

fn rec_item_mapping(m: &RustFeaseModel) -> &Mappings {
    &m.mappings
}

fn rec_predict_scores(m: &RustFeaseModel, input: ModelInput<'_>) -> Result<Vec<f32>> {
    match input {
        ModelInput::Sparse {
            interactions,
            user_features,
        } => {
            let scores = m.predict(interactions, user_features, m.beta);
            Ok(scores.into_iter().map(|x| x as f32).collect())
        }
        ModelInput::Sequence { .. } => Err(anyhow!(
            "EASE does not support ModelInput::Sequence; expected ModelInput::Sparse"
        )),
        ModelInput::TowerUser { .. } => Err(anyhow!(
            "EASE does not support ModelInput::TowerUser; expected ModelInput::Sparse"
        )),
    }
}

fn rec_predict_similar_items(
    m: &RustFeaseModel,
    item_idx: usize,
    top_k: usize,
) -> Result<Vec<(usize, f32)>> {
    Ok(m.predict_similar_items(item_idx, top_k)
        .into_iter()
        .map(|(i, s)| (i, s as f32))
        .collect())
}

fn rec_save(m: &RustFeaseModel, path: &Path) -> Result<()> {
    serialization::save_model(m, path)
}

pub struct EaseAdapter {
    pub inner: RustFeaseModel,
}

impl EaseAdapter {
    pub fn new(inner: RustFeaseModel) -> Self {
        Self { inner }
    }
}

impl RecModel for EaseAdapter {
    fn kind(&self) -> ModelKind {
        ModelKind::Ease
    }

    fn num_items(&self) -> usize {
        rec_num_items(&self.inner)
    }

    fn item_mapping(&self) -> &Mappings {
        rec_item_mapping(&self.inner)
    }

    fn predict_scores(&self, input: ModelInput<'_>) -> Result<Vec<f32>> {
        rec_predict_scores(&self.inner, input)
    }

    fn predict_similar_items(&self, item_idx: usize, top_k: usize) -> Result<Vec<(usize, f32)>> {
        rec_predict_similar_items(&self.inner, item_idx, top_k)
    }

    fn validate(&self) -> ValidationReport {
        self.inner.validate()
    }

    fn save(&self, path: &Path) -> Result<()> {
        rec_save(&self.inner, path)
    }
}

/// Borrowing adapter: holds `&RustFeaseModel`, so callers that already
/// own the model expose it as `&dyn RecModel` with zero copy. `&T` is
/// `Send + Sync` because `RustFeaseModel` is `Sync`, satisfying
/// `RecModel: Send + Sync`.
pub struct EaseAdapterRef<'a> {
    pub inner: &'a RustFeaseModel,
}

impl<'a> EaseAdapterRef<'a> {
    pub fn new(inner: &'a RustFeaseModel) -> Self {
        Self { inner }
    }
}

impl RecModel for EaseAdapterRef<'_> {
    fn kind(&self) -> ModelKind {
        ModelKind::Ease
    }

    fn num_items(&self) -> usize {
        rec_num_items(self.inner)
    }

    fn item_mapping(&self) -> &Mappings {
        rec_item_mapping(self.inner)
    }

    fn predict_scores(&self, input: ModelInput<'_>) -> Result<Vec<f32>> {
        rec_predict_scores(self.inner, input)
    }

    fn predict_similar_items(&self, item_idx: usize, top_k: usize) -> Result<Vec<(usize, f32)>> {
        rec_predict_similar_items(self.inner, item_idx, top_k)
    }

    fn validate(&self) -> ValidationReport {
        self.inner.validate()
    }

    fn save(&self, path: &Path) -> Result<()> {
        rec_save(self.inner, path)
    }
}
