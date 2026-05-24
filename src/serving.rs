//! Serving module for territory-aware multi-model support and batch predictions.
//!
//! Ports the Python `RFYTerritoryGroupServing` pattern: maintain a dictionary of
//! trained FEASE models keyed by territory/region, and route predictions to the
//! appropriate model based on a user's territory.
//!
//! Also provides batch prediction for efficiently scoring multiple users at once.

use crate::model::RustFeaseModel;
use ahash::AHashMap;
use anyhow::{Result, anyhow};
use rayon::prelude::*;

/// A registry of FEASE models keyed by territory/region name.
///
/// This is the Rust equivalent of Python's `RFYTerritoryGroupServing` — it holds
/// one trained model per territory group and routes predictions to the correct model.
pub struct FeaseModelRegistry {
    /// Map from territory name (e.g., "UNITED_STATES") to trained model.
    models: AHashMap<String, RustFeaseModel>,
    /// Optional fallback territory name for unknown territories.
    fallback_territory: Option<String>,
}

impl FeaseModelRegistry {
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

    /// Registers a trained model for a given territory.
    pub fn register(&mut self, territory: String, model: RustFeaseModel) {
        log::info!(
            "Registered model for territory '{}' ({}x{} S matrix, {} items)",
            territory,
            model.s_matrix.nrows(),
            model.s_matrix.ncols(),
            model.num_items,
        );
        self.models.insert(territory, model);
    }

    /// Removes and returns the model for a given territory.
    pub fn unregister(&mut self, territory: &str) -> Option<RustFeaseModel> {
        self.models.remove(territory)
    }

    /// Returns the model for a given territory (or the fallback if set).
    pub fn get_model(&self, territory: &str) -> Option<&RustFeaseModel> {
        self.models.get(territory).or_else(|| {
            self.fallback_territory
                .as_ref()
                .and_then(|fb| self.models.get(fb.as_str()))
        })
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
    /// Routes to the correct territory model. Returns an error if the territory
    /// is unknown and no fallback is set.
    pub fn predict(
        &self,
        territory: &str,
        user_interactions: &[(usize, f64)],
        user_features: &[(usize, f64)],
    ) -> Result<Vec<f64>> {
        let model = self.get_model(territory).ok_or_else(|| {
            anyhow!(
                "No model registered for territory '{}' (and no fallback available)",
                territory
            )
        })?;

        Ok(model.predict(user_interactions, user_features, model.beta))
    }

    /// Predicts similar items in a specific territory.
    pub fn predict_similar_items(
        &self,
        territory: &str,
        item_idx: usize,
        top_k: usize,
    ) -> Result<Vec<(usize, f64)>> {
        let model = self.get_model(territory).ok_or_else(|| {
            anyhow!(
                "No model registered for territory '{}' (and no fallback available)",
                territory
            )
        })?;

        Ok(model.predict_similar_items(item_idx, top_k))
    }
}

impl Default for FeaseModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// A single user's input data for batch prediction.
#[derive(Debug, Clone)]
pub struct UserInput {
    /// (item_index, interaction_value) pairs.
    pub interactions: Vec<(usize, f64)>,
    /// (feature_index, feature_value) pairs.
    pub features: Vec<(usize, f64)>,
}

/// Predicts scores for multiple users in a single call.
///
/// This is more efficient than calling `predict()` in a loop because it avoids
/// repeated function-call overhead and allows for potential SIMD/parallelism
/// in the future.
///
/// Returns a Vec of score vectors, one per user, in the same order as `users`.
#[allow(dead_code)]
pub fn predict_batch(model: &RustFeaseModel, users: &[UserInput]) -> Vec<Vec<f64>> {
    log::info!("Batch prediction for {} users", users.len());

    users
        .par_iter()
        .map(|user| model.predict(&user.interactions, &user.features, model.beta))
        .collect()
}

/// Filters scores by removing interacted items, sorts descending, and truncates to top-K.
///
/// Shared logic used by both batch prediction and single-user registry prediction.
pub fn filter_sort_top_k(
    scores: Vec<f64>,
    interacted: &[usize],
    top_k: usize,
) -> Vec<(usize, f64)> {
    let interacted_set: ahash::AHashSet<usize> = interacted.iter().copied().collect();

    let mut ranked: Vec<(usize, f64)> = scores
        .into_iter()
        .enumerate()
        .filter(|(idx, _)| !interacted_set.contains(idx))
        .collect();

    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked.truncate(top_k);
    ranked
}

/// Batch prediction that also returns top-K results per user.
///
/// Returns a Vec of Vec<(item_index, score)>, sorted descending by score,
/// truncated to `top_k` items per user. Excludes items the user has already
/// interacted with.
pub fn predict_batch_top_k(
    model: &RustFeaseModel,
    users: &[UserInput],
    top_k: usize,
) -> Vec<Vec<(usize, f64)>> {
    log::info!("Batch top-{} prediction for {} users", top_k, users.len());

    users
        .par_iter()
        .map(|user| {
            let scores = model.predict(&user.interactions, &user.features, model.beta);
            let interacted_indices: Vec<usize> =
                user.interactions.iter().map(|(idx, _)| *idx).collect();
            filter_sort_top_k(scores, &interacted_indices, top_k)
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
            transformation_schema: None,
        }
    }

    #[test]
    fn test_registry_basic() {
        let mut registry = FeaseModelRegistry::new();
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
        let mut registry = FeaseModelRegistry::with_fallback("US".to_string());
        registry.register("US".to_string(), make_test_model(1.0));
        registry.register("BR".to_string(), make_test_model(2.0));

        // Unknown territory falls back to US
        let model = registry.get_model("JP").unwrap();
        assert_eq!(model.num_items, 4);
    }

    #[test]
    fn test_registry_predict() {
        let mut registry = FeaseModelRegistry::new();
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
        let mut registry = FeaseModelRegistry::new();
        registry.register("US".to_string(), make_test_model(1.0));
        registry.register("BR".to_string(), make_test_model(2.0));

        let scores_us = registry.predict("US", &[(0, 1.0)], &[]).unwrap();
        let scores_br = registry.predict("BR", &[(0, 1.0)], &[]).unwrap();

        // BR model has 2x scale, so scores should differ
        let us_sum: f64 = scores_us.iter().sum();
        let br_sum: f64 = scores_br.iter().sum();
        assert!(
            (br_sum - 2.0 * us_sum).abs() < 1e-6,
            "BR scores should be 2x US scores"
        );
    }

    #[test]
    fn test_batch_predict() {
        let model = make_test_model(1.0);
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

        let results = predict_batch(&model, &users);
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

        // Batch
        let batch_results = predict_batch(&model, &users);

        // Sequential
        for (i, user) in users.iter().enumerate() {
            let sequential = model.predict(&user.interactions, &user.features, model.beta);
            assert_eq!(batch_results[i].len(), sequential.len());
            for (a, b) in batch_results[i].iter().zip(sequential.iter()) {
                assert!(
                    (a - b).abs() < 1e-12,
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
        let model = make_test_model(1.0);
        let users = vec![UserInput {
            interactions: vec![(0, 1.0)],
            features: vec![],
        }];

        let results = predict_batch_top_k(&model, &users, 2);
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
        let model = make_test_model(1.0);
        let results = predict_batch(&model, &[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_registry_unregister() {
        let mut registry = FeaseModelRegistry::new();
        registry.register("US".to_string(), make_test_model(1.0));
        assert_eq!(registry.len(), 1);

        let removed = registry.unregister("US");
        assert!(removed.is_some());
        assert!(registry.is_empty());
    }

    #[test]
    fn test_filter_sort_top_k() {
        let scores = vec![0.1, 0.9, 0.5, 0.3];
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
        let scores = vec![0.3, 0.1, 0.9];
        let interacted: Vec<usize> = vec![];

        let result = filter_sort_top_k(scores, &interacted, 2);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, 2); // 0.9
        assert_eq!(result[1].0, 0); // 0.3
    }
}
