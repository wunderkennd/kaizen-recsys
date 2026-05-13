//! `EaseAdapter` — wraps `RustFeaseModel` to expose it via `RecModel`.
//!
//! Every method delegates to the existing concrete model. The only
//! computation introduced is the `as f32` cast on output scores; the
//! trait standardizes on f32 to match the burn-based models in later
//! phases.

use super::{ModelInput, ModelKind, RecModel};
use crate::data_pipeline::Mappings;
use crate::model::{RustFeaseModel, ValidationReport};
use crate::serialization;
use anyhow::{Result, anyhow};
use std::path::Path;

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
        self.inner.num_items
    }

    fn item_mapping(&self) -> &Mappings {
        &self.inner.mappings
    }

    fn predict_scores(&self, input: ModelInput<'_>) -> Result<Vec<f32>> {
        match input {
            ModelInput::Sparse {
                interactions,
                user_features,
            } => {
                let scores =
                    self.inner
                        .predict(interactions, user_features, self.inner.beta);
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

    fn predict_similar_items(
        &self,
        item_idx: usize,
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        Ok(self
            .inner
            .predict_similar_items(item_idx, top_k)
            .into_iter()
            .map(|(i, s)| (i, s as f32))
            .collect())
    }

    fn validate(&self) -> ValidationReport {
        self.inner.validate()
    }

    fn save(&self, path: &Path) -> Result<()> {
        serialization::save_model(&self.inner, path)
    }
}
