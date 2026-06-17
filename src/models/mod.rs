//! Common interface for recommender models.
//!
//! Phase 1 of ADR-0001: introduces the `RecModel` trait and an
//! `EaseAdapter` that wraps the existing `RustFeaseModel`. No behavior
//! change — every adapter method delegates to the concrete model.
//! Phases 2–6 plug new models (SASRec, Two-Tower) in behind the same
//! trait, and generalize the eval/tuning/serving consumers.

// Phase 1 is scaffolding: lib.rs, evaluation, tuning, and serving still
// reference `RustFeaseModel` concretely. Several trait methods and the
// `Sequence` / `TowerUser` variants only get consumers in Phases 3–6.
// This allow lives here so the scaffolding compiles cleanly until those
// phases land; it gets removed when the consumers do.
#![allow(dead_code, unused_imports)]

use crate::data_pipeline::Mappings;
use crate::model::ValidationReport;
use anyhow::Result;
use std::path::Path;

pub mod ease;

#[cfg(feature = "ml-models")]
pub mod sasrec;

#[cfg(feature = "ml-models")]
pub mod two_tower;

pub use ease::{EaseAdapter, EaseAdapterRef};

#[cfg(feature = "ml-models")]
pub use sasrec::TrainedSasRec;

/// What kind of model this is. Used by callers that need to construct
/// model-appropriate input shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    Ease,
    SasRec,
    TwoTower,
}

/// Input passed to `predict_scores`. Each variant maps to a model family.
/// A model that receives an unsupported variant returns `Err`.
#[derive(Debug)]
pub enum ModelInput<'a> {
    /// EASE: sparse interaction values + sparse user-feature values.
    /// `interactions: &[(item_idx, value)]`,
    /// `user_features: &[(feature_idx, value)]`.
    Sparse {
        interactions: &'a [(usize, f64)],
        user_features: &'a [(usize, f64)],
    },
    /// SASRec (Phase 3): chronologically-ordered item indices, oldest first.
    Sequence { history: &'a [usize] },
    /// Two-Tower (Phase 5): user-side input with categorical and dense features.
    TowerUser {
        user_idx: Option<usize>,
        cat_features: &'a [usize],
        dense_features: &'a [f32],
    },
}

/// Common interface for all recommender models.
///
/// Implementors are `Send + Sync` so `Arc<dyn RecModel>` works in serving
/// paths. Score vectors are `Vec<f32>` — burn-based models will produce
/// f32 natively, and EASE pays a single `as f32` cast per element on output.
pub trait RecModel: Send + Sync {
    fn kind(&self) -> ModelKind;

    /// Number of items in the catalog. Score vectors returned by
    /// `predict_scores` have this length.
    fn num_items(&self) -> usize;

    /// String-id ↔ index mappings built once by the data pipeline.
    fn item_mapping(&self) -> &Mappings;

    /// Score every item in the catalog for the given input.
    ///
    /// Returns `Err` if the input variant is not supported by this model.
    fn predict_scores(&self, input: ModelInput<'_>) -> Result<Vec<f32>>;

    /// Top-K items most similar to a given item, by the model's notion of
    /// item similarity. EASE uses the item-item block of S; sequence and
    /// tower models will use embedding similarity.
    fn predict_similar_items(&self, item_idx: usize, top_k: usize) -> Result<Vec<(usize, f32)>>;

    /// Self-check the model state.
    fn validate(&self) -> ValidationReport;

    /// Persist the model to disk. Phase 1 routes to the existing EASE
    /// serializer; Phases 3 and 5 add per-model magic bytes.
    fn save(&self, path: &Path) -> Result<()>;

    /// Translate a `feature_name → value` map into the integer
    /// `(cat_features, dense_features)` pair the model expects in
    /// `ModelInput::TowerUser` (#55 / #56).
    ///
    /// Default returns the empty pair — only Two-Tower currently uses
    /// the feature embedding tables at predict time. Lets the
    /// `ModelRegistry` route `predict_top_k_two_tower` through
    /// `&dyn RecModel` without downcasting to a concrete type.
    fn resolve_user_features(
        &self,
        _features: &ahash::AHashMap<String, f64>,
    ) -> (Vec<usize>, Vec<f32>) {
        (Vec::new(), Vec::new())
    }

    /// Returns `Some` if this model can retrieve top-K items without
    /// scoring the full catalog — i.e. it exposes an approximate-nearest-
    /// neighbor index (ADR-0004). Default `None` → serving falls back to
    /// dense `predict_scores` + `filter_sort_top_k`.
    ///
    /// A model that owns an ANN index over its item embeddings returns
    /// `Some`; EASE has no embedding space and always returns `None`.
    fn retrieval_index(&self) -> Option<&dyn RetrievalIndex> {
        None
    }
}

/// Optional capability: top-K retrieval without scoring the full catalog
/// (ADR-0004). A query embedding (e.g. a Two-Tower user vector) is matched
/// against an approximate-nearest-neighbor index in sublinear time.
///
/// Object-safe and `Send + Sync` so serving can hold `&dyn RetrievalIndex`.
/// The model computes its own query embedding from `input` internally, then
/// queries the index — keeping serving model-agnostic, the same property the
/// [`ModelInput`] enum preserves for `predict_scores`.
pub trait RetrievalIndex: Send + Sync {
    /// Top-K `(item_idx, score)` for `input`, excluding the item indices in
    /// `exclude` (e.g. items the user has already interacted with).
    fn retrieve(
        &self,
        input: ModelInput<'_>,
        top_k: usize,
        exclude: &[usize],
    ) -> Result<Vec<(usize, f32)>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::RustFeaseModel;
    use nalgebra::DMatrix;

    fn dummy_mappings() -> Mappings {
        Mappings {
            user_to_idx: Default::default(),
            idx_to_user: Default::default(),
            item_to_idx: Default::default(),
            idx_to_item: Default::default(),
            user_feature_to_idx: Default::default(),
            idx_to_user_feature: Default::default(),
            item_feature_to_idx: Default::default(),
            idx_to_item_feature: Default::default(),
        }
    }

    fn tiny_warm_model() -> RustFeaseModel {
        let n_items = 4;
        let n_user_features = 2;
        let total_dim = n_items + n_user_features;

        let mut s_mat = DMatrix::<f64>::zeros(total_dim, total_dim);
        s_mat[(0, 1)] = 0.5;
        s_mat[(1, 0)] = 0.5;
        s_mat[(4, 5)] = 0.8;
        s_mat[(5, 4)] = 0.8;
        s_mat[(2, 4)] = 1.0;
        s_mat[(4, 2)] = 1.0;

        RustFeaseModel {
            s_matrix: s_mat,
            num_items: n_items,
            num_user_features: n_user_features,
            num_item_features: 0,
            alpha: 1.0,
            beta: 1.0,
            lambda_: 100.0,
            meta_weight: 0.0,
            mappings: dummy_mappings(),
            weighting_config: None,
        }
    }

    #[test]
    fn adapter_predict_scores_matches_concrete() {
        let model = tiny_warm_model();
        let interactions = vec![(0_usize, 1.0_f64)];
        let user_features = vec![(1_usize, 1.0_f64)];

        let baseline = model.predict(&interactions, &user_features, model.beta);

        let adapter = EaseAdapter::new(model);
        let via_trait = adapter
            .predict_scores(ModelInput::Sparse {
                interactions: &interactions,
                user_features: &user_features,
            })
            .expect("Sparse input must be supported by EASE");

        assert_eq!(via_trait.len(), baseline.len());
        for (i, (a, b)) in via_trait.iter().zip(baseline.iter()).enumerate() {
            assert!(
                ((*a as f64) - *b).abs() < 1e-6,
                "score mismatch at item {i}: adapter={a} baseline={b}",
            );
        }
    }

    #[test]
    fn adapter_rejects_unsupported_input() {
        let adapter = EaseAdapter::new(tiny_warm_model());
        let history: [usize; 0] = [];
        let result = adapter.predict_scores(ModelInput::Sequence { history: &history });
        assert!(result.is_err(), "EASE must reject Sequence input");
    }

    #[test]
    fn adapter_similar_items_roundtrip() {
        let model = tiny_warm_model();
        let baseline = model.predict_similar_items(0, 5);

        let adapter = EaseAdapter::new(model);
        let via_trait = adapter
            .predict_similar_items(0, 5)
            .expect("similar items query must succeed");

        assert_eq!(via_trait.len(), baseline.len());
        for ((idx_a, score_a), (idx_b, score_b)) in via_trait.iter().zip(baseline.iter()) {
            assert_eq!(*idx_a, *idx_b, "item index order must match");
            assert!(
                ((*score_a as f64) - *score_b).abs() < 1e-6,
                "similarity mismatch: adapter={score_a} baseline={score_b}",
            );
        }
    }

    const _: fn() = || {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<EaseAdapter>();
    };
}
