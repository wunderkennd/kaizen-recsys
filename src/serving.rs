//! Serving module for territory-aware multi-model support and batch predictions.
//!
//! Ports the Python `RFYTerritoryGroupServing` pattern: maintain a dictionary of
//! trained models keyed by territory/region, and route predictions to the
//! appropriate model based on a user's territory.
//!
//! The registry is generalized over `Box<dyn RecModel>` (issue #39): a
//! territory may be served by EASE, SASRec, or Two-Tower. Routing and
//! prediction go through the [`RecModel`] trait, so the registry is
//! model-agnostic. EASE callers register a `RustFeaseModel` via
//! [`ModelRegistry::register`] (it is wrapped in an `EaseAdapter`);
//! other models register through [`ModelRegistry::register_model`].
//!
//! Also provides batch prediction for efficiently scoring multiple users at once.

use crate::model::RustFeaseModel;
use crate::models::{EaseAdapter, ModelInput, ModelKind, RecModel};
use ahash::AHashMap;
use anyhow::{Result, anyhow, bail};
use rayon::prelude::*;

/// A registry of recommender models keyed by territory/region name.
///
/// This is the Rust equivalent of Python's `RFYTerritoryGroupServing` — it
/// holds one trained model per territory group and routes predictions to
/// the correct model. Models are stored as `Box<dyn RecModel>` so a
/// territory can be served by any model family.
pub struct ModelRegistry {
    /// Map from territory name (e.g., "UNITED_STATES") to trained model.
    models: AHashMap<String, Box<dyn RecModel>>,
    /// Optional fallback territory name for unknown territories.
    fallback_territory: Option<String>,
}

impl ModelRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self {
            models: AHashMap::new(),
            fallback_territory: None,
        }
    }

    /// Creates a registry with a fallback territory.
    ///
    /// When `predict` is called with an unknown territory, the fallback model
    /// is used instead of returning an error.
    pub fn with_fallback(fallback_territory: String) -> Self {
        Self {
            models: AHashMap::new(),
            fallback_territory: Some(fallback_territory),
        }
    }

    /// Registers a trained EASE model for a given territory.
    ///
    /// The concrete `RustFeaseModel` is wrapped in an `EaseAdapter` so it
    /// is stored uniformly as `Box<dyn RecModel>`. Kept for source
    /// compatibility with existing EASE call sites.
    pub fn register(&mut self, territory: String, model: RustFeaseModel) {
        log::info!(
            "Registered EASE model for territory '{}' ({}x{} S matrix, {} items)",
            territory,
            model.s_matrix.nrows(),
            model.s_matrix.ncols(),
            model.num_items,
        );
        self.models
            .insert(territory, Box::new(EaseAdapter::new(model)));
    }

    /// Registers any trained model (EASE, SASRec, Two-Tower) for a
    /// territory via the [`RecModel`] trait object.
    ///
    /// The EASE PyO3 surface registers via [`Self::register`]; this
    /// trait-object entry point is the seam any `RecModel` (EASE, SASRec,
    /// Two-Tower) is registered through for territory routing.
    #[allow(dead_code)]
    pub fn register_model(&mut self, territory: String, model: Box<dyn RecModel>) {
        log::info!(
            "Registered {:?} model for territory '{}' ({} items)",
            model.kind(),
            territory,
            model.num_items(),
        );
        self.models.insert(territory, model);
    }

    /// Removes and returns the model for a given territory.
    pub fn unregister(&mut self, territory: &str) -> Option<Box<dyn RecModel>> {
        self.models.remove(territory)
    }

    /// Returns the model for a given territory (or the fallback if set).
    pub fn get_model(&self, territory: &str) -> Option<&dyn RecModel> {
        self.models
            .get(territory)
            .or_else(|| {
                self.fallback_territory
                    .as_ref()
                    .and_then(|fb| self.models.get(fb.as_str()))
            })
            .map(|b| b.as_ref())
    }

    /// Lists all registered territory names.
    pub fn territories(&self) -> Vec<&str> {
        self.models.keys().map(|k| k.as_str()).collect()
    }

    /// Number of registered models.
    pub fn len(&self) -> usize {
        self.models.len()
    }

    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// Predicts scores for a user in a specific territory.
    ///
    /// Routes to the correct territory model and scores it through the
    /// [`RecModel`] trait. The `RecModel` contract is f32; EASE pays a
    /// single `as f32` cast per element on output (unchanged math).
    /// Returns an error if the territory is unknown and no fallback is set,
    /// or if the routed model does not support sparse input.
    pub fn predict(
        &self,
        territory: &str,
        user_interactions: &[(usize, f64)],
        user_features: &[(usize, f64)],
    ) -> Result<Vec<f32>> {
        let model = self.route(territory)?;
        model.predict_scores(ModelInput::Sparse {
            interactions: user_interactions,
            user_features,
        })
    }

    /// Predicts the top-K items for a user in a territory, excluding the
    /// items the user has already interacted with.
    pub fn predict_top_k(
        &self,
        territory: &str,
        user_interactions: &[(usize, f64)],
        user_features: &[(usize, f64)],
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        let scores = self.predict(territory, user_interactions, user_features)?;
        let interacted: Vec<usize> = user_interactions.iter().map(|(idx, _)| *idx).collect();
        Ok(filter_sort_top_k(scores, &interacted, top_k))
    }

    /// Predicts similar items in a specific territory.
    pub fn predict_similar_items(
        &self,
        territory: &str,
        item_idx: usize,
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        let model = self.route(territory)?;
        model.predict_similar_items(item_idx, top_k)
    }

    /// String-id-native EASE top-K (#56). Mirrors `FeaseModel.predict`'s
    /// surface: callers pass `dict[str, float]` interactions and features
    /// and get back `(item_id, score)` rows. Errors if the registered
    /// model for `territory` isn't EASE.
    pub fn predict_top_k_ease(
        &self,
        territory: &str,
        interactions: &AHashMap<String, f64>,
        features: &AHashMap<String, f64>,
        top_k: usize,
    ) -> Result<Vec<(String, f32)>> {
        let model = self.route(territory)?;
        if model.kind() != ModelKind::Ease {
            bail!(
                "predict_top_k_ease called on territory '{territory}', but the registered \
                 model is {:?}. Use predict_top_k_{} instead.",
                model.kind(),
                kind_method_suffix(model.kind())
            );
        }
        let mappings = model.item_mapping();
        let interactions_idx: Vec<(usize, f64)> = interactions
            .iter()
            .filter_map(|(name, &v)| mappings.item_to_idx.get(name).map(|&i| (i, v)))
            .collect();
        let features_idx: Vec<(usize, f64)> = features
            .iter()
            .filter_map(|(name, &v)| mappings.user_feature_to_idx.get(name).map(|&i| (i, v)))
            .collect();
        let scores = model.predict_scores(ModelInput::Sparse {
            interactions: &interactions_idx,
            user_features: &features_idx,
        })?;
        let interacted: Vec<usize> = interactions_idx.iter().map(|(idx, _)| *idx).collect();
        let ranked = filter_sort_top_k(scores, &interacted, top_k);
        Ok(idx_scores_to_str(&ranked, mappings))
    }

    /// String-id-native SASRec top-K (#56). `history` is a chronological
    /// item-id list (oldest first), mirroring `SASRecModel.predict`.
    /// Unknown item ids are skipped silently. Errors if the registered
    /// model for `territory` isn't SASRec.
    #[cfg(feature = "ml-models")]
    pub fn predict_top_k_sasrec(
        &self,
        territory: &str,
        history: &[String],
        top_k: usize,
    ) -> Result<Vec<(String, f32)>> {
        let model = self.route(territory)?;
        if model.kind() != ModelKind::SasRec {
            bail!(
                "predict_top_k_sasrec called on territory '{territory}', but the registered \
                 model is {:?}. Use predict_top_k_{} instead.",
                model.kind(),
                kind_method_suffix(model.kind())
            );
        }
        let mappings = model.item_mapping();
        let history_idx: Vec<usize> = history
            .iter()
            .filter_map(|id| mappings.item_to_idx.get(id).copied())
            .collect();
        let scores = model.predict_scores(ModelInput::Sequence {
            history: &history_idx,
        })?;
        // SASRec ranking excludes items already in the user's history,
        // matching SASRecModel.predict's behaviour.
        let ranked = filter_sort_top_k(scores, &history_idx, top_k);
        Ok(idx_scores_to_str(&ranked, mappings))
    }

    /// String-id-native Two-Tower top-K (#56). Warm users use their
    /// learned id-row embedding; unknown users fall back to the
    /// reserved cold-start row. When `features` is non-empty, the
    /// trained model's `resolve_user_features` map translates names
    /// into category indices and dense columns (#55). Errors if the
    /// registered model for `territory` isn't Two-Tower.
    #[cfg(feature = "ml-models")]
    pub fn predict_top_k_two_tower(
        &self,
        territory: &str,
        user_id: &str,
        features: &AHashMap<String, f64>,
        top_k: usize,
    ) -> Result<Vec<(String, f32)>> {
        let model = self.route(territory)?;
        if model.kind() != ModelKind::TwoTower {
            bail!(
                "predict_top_k_two_tower called on territory '{territory}', but the registered \
                 model is {:?}. Use predict_top_k_{} instead.",
                model.kind(),
                kind_method_suffix(model.kind())
            );
        }
        let mappings = model.item_mapping();
        let user_idx = mappings.user_to_idx.get(user_id).copied();
        let (cat_features, dense_features) = model.resolve_user_features(features);
        let scores = model.predict_scores(ModelInput::TowerUser {
            user_idx,
            cat_features: &cat_features,
            dense_features: &dense_features,
        })?;
        // Two-Tower has no per-request history to exclude — matches
        // `TwoTowerModel.predict`'s behaviour.
        let ranked = filter_sort_top_k(scores, &[], top_k);
        Ok(idx_scores_to_str(&ranked, mappings))
    }

    /// Routes a territory to its model (or the fallback), erroring with a
    /// consistent message when neither exists.
    fn route(&self, territory: &str) -> Result<&dyn RecModel> {
        self.get_model(territory).ok_or_else(|| {
            anyhow!(
                "No model registered for territory '{}' (and no fallback available)",
                territory
            )
        })
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Method-name suffix matching a model kind — used in error messages
/// to point callers at the correct `predict_top_k_*` variant when they
/// invoke the wrong one.
fn kind_method_suffix(kind: ModelKind) -> &'static str {
    match kind {
        ModelKind::Ease => "ease",
        ModelKind::SasRec => "sasrec",
        ModelKind::TwoTower => "two_tower",
    }
}

/// Translate index-keyed `(item_idx, score)` rows back to string ids
/// via the model's catalog mapping. Skips indices that aren't present
/// in `idx_to_item` (shouldn't happen in practice — scores are indexed
/// over the catalog — but keep it total).
fn idx_scores_to_str(
    ranked: &[(usize, f32)],
    mappings: &crate::data_pipeline::Mappings,
) -> Vec<(String, f32)> {
    ranked
        .iter()
        .filter_map(|(idx, score)| mappings.idx_to_item.get(*idx).map(|s| (s.clone(), *score)))
        .collect()
}

/// A single user's input data for batch prediction.
#[derive(Debug, Clone)]
pub struct UserInput {
    /// (item_index, interaction_value) pairs.
    pub interactions: Vec<(usize, f64)>,
    /// (feature_index, feature_value) pairs.
    pub features: Vec<(usize, f64)>,
}

/// Predicts scores for multiple users in a single call, over any
/// [`RecModel`].
///
/// More efficient than calling `predict()` in a loop: rayon scores users
/// in parallel. Each user is fed as `ModelInput::Sparse`, so EASE,
/// SASRec, and Two-Tower all work behind the same call. Returns one score
/// vector per user, in the same order as `users`. Propagates the first
/// error if the routed model rejects sparse input.
///
/// The PyO3 batch path uses [`predict_batch_top_k`]; this score-only
/// variant is kept as a public building block and exercised by tests.
#[allow(dead_code)]
pub fn predict_batch(model: &dyn RecModel, users: &[UserInput]) -> Result<Vec<Vec<f32>>> {
    log::info!("Batch prediction for {} users", users.len());

    users
        .par_iter()
        .map(|user| {
            model.predict_scores(ModelInput::Sparse {
                interactions: &user.interactions,
                user_features: &user.features,
            })
        })
        .collect()
}

/// Filters scores by removing interacted items, selects the top-K by
/// descending score, and returns them sorted.
///
/// Generic over the score scalar (`f32` from the `RecModel` path, `f64`
/// from the concrete EASE path) so both registry/batch prediction and the
/// existing concrete EASE serving call site share one implementation.
///
/// Uses a quickselect partition (`select_nth_unstable_by`, O(n) average)
/// to isolate the top-K candidates, then sorts only that K-element slice
/// (O(K log K)) — cheaper than the old full O(n log n) sort when
/// `top_k << num_items`, which is the common serving case. Ties are broken
/// by ascending item index, making the order a *total* order: the result is
/// byte-identical to the previous stable `sort_by`, and deterministic
/// despite `select_nth_unstable`/`sort_unstable` not preserving input order.
pub fn filter_sort_top_k<T: PartialOrd + Copy>(
    scores: Vec<T>,
    interacted: &[usize],
    top_k: usize,
) -> Vec<(usize, T)> {
    let interacted_set: ahash::AHashSet<usize> = interacted.iter().copied().collect();

    let mut ranked: Vec<(usize, T)> = scores
        .into_iter()
        .enumerate()
        .filter(|(idx, _)| !interacted_set.contains(idx))
        .collect();

    // Descending by score; NaN/unorderable treated as equal (matching the
    // prior behavior), then ascending by index as a deterministic tie-break.
    let cmp = |a: &(usize, T), b: &(usize, T)| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    };

    if top_k == 0 {
        return Vec::new();
    }

    if top_k < ranked.len() {
        // Partition so the first `top_k` elements are the top-K (unordered
        // among themselves), then sort just that slice.
        ranked.select_nth_unstable_by(top_k - 1, cmp);
        ranked.truncate(top_k);
    }
    ranked.sort_unstable_by(cmp);
    ranked
}

/// Batch prediction that also returns top-K results per user, over any
/// [`RecModel`].
///
/// Returns a Vec of Vec<(item_index, score)>, sorted descending by score,
/// truncated to `top_k` items per user. Excludes items the user has
/// already interacted with.
pub fn predict_batch_top_k(
    model: &dyn RecModel,
    users: &[UserInput],
    top_k: usize,
) -> Result<Vec<Vec<(usize, f32)>>> {
    log::info!("Batch top-{} prediction for {} users", top_k, users.len());

    users
        .par_iter()
        .map(|user| {
            let scores = model.predict_scores(ModelInput::Sparse {
                interactions: &user.interactions,
                user_features: &user.features,
            })?;
            let interacted_indices: Vec<usize> =
                user.interactions.iter().map(|(idx, _)| *idx).collect();
            Ok(filter_sort_top_k(scores, &interacted_indices, top_k))
        })
        .collect()
}

// --- Unit Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_pipeline::Mappings;
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

    fn make_test_model(scale: f64) -> RustFeaseModel {
        let n_items = 4;
        let n_user_features = 2;
        let total_dim = n_items + n_user_features;

        let mut s = DMatrix::<f64>::zeros(total_dim, total_dim);
        s[(0, 1)] = 0.5 * scale;
        s[(1, 0)] = 0.5 * scale;
        s[(0, 2)] = 0.3 * scale;
        s[(2, 0)] = 0.3 * scale;

        RustFeaseModel {
            s_matrix: s,
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
    fn test_registry_basic() {
        let mut registry = ModelRegistry::new();
        assert!(registry.is_empty());

        registry.register("US".to_string(), make_test_model(1.0));
        registry.register("BR".to_string(), make_test_model(2.0));

        assert_eq!(registry.len(), 2);
        assert!(registry.get_model("US").is_some());
        assert!(registry.get_model("BR").is_some());
        assert!(registry.get_model("JP").is_none());
    }

    #[test]
    fn test_registry_fallback() {
        let mut registry = ModelRegistry::with_fallback("US".to_string());
        registry.register("US".to_string(), make_test_model(1.0));
        registry.register("BR".to_string(), make_test_model(2.0));

        // Unknown territory falls back to US
        let model = registry.get_model("JP").unwrap();
        assert_eq!(model.num_items(), 4);
    }

    #[test]
    fn test_registry_predict() {
        let mut registry = ModelRegistry::new();
        registry.register("US".to_string(), make_test_model(1.0));

        let scores = registry
            .predict("US", &[(0, 1.0)], &[])
            .expect("Prediction should work");
        assert_eq!(scores.len(), 4);

        // Unknown territory without fallback
        let result = registry.predict("JP", &[(0, 1.0)], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_registry_territory_isolation() {
        let mut registry = ModelRegistry::new();
        registry.register("US".to_string(), make_test_model(1.0));
        registry.register("BR".to_string(), make_test_model(2.0));

        let scores_us = registry.predict("US", &[(0, 1.0)], &[]).unwrap();
        let scores_br = registry.predict("BR", &[(0, 1.0)], &[]).unwrap();

        // BR model has 2x scale, so scores should differ
        let us_sum: f32 = scores_us.iter().sum();
        let br_sum: f32 = scores_br.iter().sum();
        assert!(
            (br_sum - 2.0 * us_sum).abs() < 1e-5,
            "BR scores should be 2x US scores"
        );
    }

    #[test]
    fn test_batch_predict() {
        let adapter = EaseAdapter::new(make_test_model(1.0));
        let users = vec![
            UserInput {
                interactions: vec![(0, 1.0)],
                features: vec![],
            },
            UserInput {
                interactions: vec![(1, 1.0)],
                features: vec![],
            },
            UserInput {
                interactions: vec![],
                features: vec![(0, 1.0)],
            },
        ];

        let results = predict_batch(&adapter, &users).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].len(), 4); // num_items
        assert_eq!(results[1].len(), 4);
        assert_eq!(results[2].len(), 4);
    }

    #[test]
    fn test_batch_matches_sequential() {
        let model = make_test_model(1.0);
        let users = vec![
            UserInput {
                interactions: vec![(0, 1.0)],
                features: vec![(1, 0.5)],
            },
            UserInput {
                interactions: vec![(1, 1.0), (2, 0.5)],
                features: vec![],
            },
        ];

        // Batch through the generalized `&dyn RecModel` path.
        let adapter = EaseAdapter::new(model.clone());
        let batch_results = predict_batch(&adapter, &users).unwrap();

        // Sequential, concrete f64 path. The `RecModel` contract is f32,
        // so the comparison tolerance is the single `as f32` cast.
        for (i, user) in users.iter().enumerate() {
            let sequential = model.predict(&user.interactions, &user.features, model.beta);
            assert_eq!(batch_results[i].len(), sequential.len());
            for (a, b) in batch_results[i].iter().zip(sequential.iter()) {
                assert!(
                    ((*a as f64) - b).abs() < 1e-6,
                    "Batch vs sequential mismatch for user {}: {} vs {}",
                    i,
                    a,
                    b
                );
            }
        }
    }

    #[test]
    fn test_batch_top_k() {
        let adapter = EaseAdapter::new(make_test_model(1.0));
        let users = vec![UserInput {
            interactions: vec![(0, 1.0)],
            features: vec![],
        }];

        let results = predict_batch_top_k(&adapter, &users, 2).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].len() <= 2);

        // Should not contain item 0 (interacted)
        for (idx, _) in &results[0] {
            assert_ne!(*idx, 0, "Should not recommend interacted item");
        }

        // Should be sorted descending
        if results[0].len() >= 2 {
            assert!(results[0][0].1 >= results[0][1].1);
        }
    }

    #[test]
    fn test_batch_empty() {
        let adapter = EaseAdapter::new(make_test_model(1.0));
        let results = predict_batch(&adapter, &[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_registry_unregister() {
        let mut registry = ModelRegistry::new();
        registry.register("US".to_string(), make_test_model(1.0));
        assert_eq!(registry.len(), 1);

        let removed = registry.unregister("US");
        assert!(removed.is_some());
        assert!(registry.is_empty());
    }

    /// #56: predict_top_k_ease translates string ids via the registered
    /// model's mappings and produces (item_id, score) rows.
    #[test]
    fn test_predict_top_k_ease_string_ids_route_through_mappings() {
        use crate::data_pipeline::Mappings;
        let mut model = make_test_model(1.0);
        let mut mappings = Mappings {
            user_to_idx: Default::default(),
            idx_to_user: Default::default(),
            item_to_idx: Default::default(),
            idx_to_item: vec!["a".into(), "b".into(), "c".into(), "d".into()],
            user_feature_to_idx: Default::default(),
            idx_to_user_feature: Default::default(),
            item_feature_to_idx: Default::default(),
            idx_to_item_feature: Default::default(),
        };
        for (i, id) in mappings.idx_to_item.iter().enumerate() {
            mappings.item_to_idx.insert(id.clone(), i);
        }
        model.mappings = mappings;

        let mut registry = ModelRegistry::new();
        registry.register("US".to_string(), model);

        let mut interactions = AHashMap::new();
        interactions.insert("a".to_string(), 1.0);
        let features = AHashMap::new();
        let ranked = registry
            .predict_top_k_ease("US", &interactions, &features, 3)
            .expect("predict_top_k_ease should succeed");
        // Ranked items are string ids, not indices.
        assert!(!ranked.is_empty());
        for (item_id, _) in &ranked {
            assert!(["a", "b", "c", "d"].contains(&item_id.as_str()));
            // The user already interacted with "a" — must be excluded.
            assert_ne!(item_id, "a");
        }
    }

    /// #56: predict_top_k_sasrec on a territory holding an EASE model
    /// errors loudly with a useful message instead of silently
    /// invoking the wrong predict path.
    #[cfg(feature = "ml-models")]
    #[test]
    fn test_predict_top_k_sasrec_rejects_ease_territory() {
        let mut registry = ModelRegistry::new();
        registry.register("US".to_string(), make_test_model(1.0));
        let err = registry
            .predict_top_k_sasrec("US", &["a".to_string()], 3)
            .expect_err("should reject mismatched model kind");
        let msg = format!("{err}");
        assert!(msg.contains("predict_top_k_sasrec"));
        assert!(msg.contains("predict_top_k_ease"));
    }

    /// #56: predict_top_k_two_tower on a territory holding an EASE model
    /// errors loudly with a useful message.
    #[cfg(feature = "ml-models")]
    #[test]
    fn test_predict_top_k_two_tower_rejects_ease_territory() {
        let mut registry = ModelRegistry::new();
        registry.register("US".to_string(), make_test_model(1.0));
        let features = AHashMap::new();
        let err = registry
            .predict_top_k_two_tower("US", "user0", &features, 3)
            .expect_err("should reject mismatched model kind");
        let msg = format!("{err}");
        assert!(msg.contains("predict_top_k_two_tower"));
        assert!(msg.contains("predict_top_k_ease"));
    }

    #[test]
    fn test_filter_sort_top_k() {
        let scores: Vec<f64> = vec![0.1, 0.9, 0.5, 0.3];
        let interacted = vec![0]; // exclude item 0

        let result = filter_sort_top_k(scores, &interacted, 2);

        assert_eq!(result.len(), 2);
        // Item 0 should be excluded
        for (idx, _) in &result {
            assert_ne!(*idx, 0);
        }
        // Should be sorted descending: item 1 (0.9), item 2 (0.5)
        assert_eq!(result[0].0, 1);
        assert!((result[0].1 - 0.9).abs() < 1e-12);
        assert_eq!(result[1].0, 2);
        assert!((result[1].1 - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_filter_sort_top_k_empty_interacted() {
        let scores: Vec<f64> = vec![0.3, 0.1, 0.9];
        let interacted: Vec<usize> = vec![];

        let result = filter_sort_top_k(scores, &interacted, 2);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, 2); // 0.9
        assert_eq!(result[1].0, 0); // 0.3
    }

    /// Tied scores must break by ascending item index, matching the prior
    /// stable sort. This is what keeps the quickselect-based top-K
    /// byte-identical despite the underlying algorithm being unstable.
    #[test]
    fn test_filter_sort_top_k_breaks_ties_by_index() {
        // Items 1, 3, 4 all tie at 0.5; item 2 is the clear top.
        let scores: Vec<f64> = vec![0.1, 0.5, 0.9, 0.5, 0.5];
        let result = filter_sort_top_k(scores, &[], 3);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].0, 2); // 0.9 first
        // Among the 0.5 ties, lowest index wins the remaining slot.
        assert_eq!(result[1].0, 1);
        assert_eq!(result[2].0, 3);
    }

    /// `top_k >= len` must skip the partition and just sort everything,
    /// returning every (non-interacted) item.
    #[test]
    fn test_filter_sort_top_k_k_exceeds_len() {
        let scores: Vec<f64> = vec![0.3, 0.9, 0.1];
        let result = filter_sort_top_k(scores, &[], 10);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].0, 1); // 0.9
        assert_eq!(result[1].0, 0); // 0.3
        assert_eq!(result[2].0, 2); // 0.1
    }

    /// `top_k == 0` returns nothing without touching the data.
    #[test]
    fn test_filter_sort_top_k_zero_k() {
        let scores: Vec<f64> = vec![0.3, 0.9, 0.1];
        let result = filter_sort_top_k(scores, &[], 0);
        assert!(result.is_empty());
    }
}
