//! This file is the main entrypoint for the Python library.
//! It uses PyO3 to define the Python-callable functions and the `FeaseModel` class.
//! This is the "bridge" between Python and Rust.

use model::RustFeaseModel; // The internal Rust struct
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyList, PyString};
use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

mod data_pipeline;
mod data_validation;
pub mod metrics;
mod model;
mod schemas;
mod serialization;
mod serving;
mod weighting;

/// A Python-accessible class that holds the trained FEASE model.
///
/// This struct is a thin wrapper around the internal `RustFeaseModel`,
/// adding Python methods like `predict` and properties like `num_items`.
///
/// We use `#[pyo3(get)]` to expose the Rust fields as read-only
/// Python properties.
#[pyclass]
struct FeaseModel {
    /// The internal Rust model, which holds the S-matrix and mappings.
    model: RustFeaseModel,

    /// Number of items the model was trained on.
    #[pyo3(get)]
    num_items: usize,

    /// Number of user features the model was trained on.
    #[pyo3(get)]
    num_user_features: usize,

    /// Number of item features the model was trained on.
    #[pyo3(get)]
    num_item_features: usize,
}

#[pymethods]
impl FeaseModel {
    /// Predicts recommendation scores for a user.
    ///
    /// Args:
    ///     interactions (dict[str, float]):
    ///         A dictionary mapping item_guids to interaction values
    ///         (e.g., log_watch_time) for the user.
    ///     features (dict[str, float]):
    ///         A dictionary mapping user_feature_names to their values
    ///         (e.g., {"plan_Premium": 1.0, "tenure_30d": 1.0}).
    ///     top_k (int):
    ///         The number of top recommendations to return.
    ///
    /// Returns:
    ///     list[tuple[str, float]]:
    ///         A list of (item_guid, score) tuples, sorted descending by score.
    #[pyo3(signature = (interactions, features, top_k=100))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        interactions: &Bound<'_, PyDict>,
        features: &Bound<'_, PyDict>,
        top_k: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        // --- 1. Convert Python inputs to Rust vectors ---
        let mut user_interactions: Vec<(usize, f64)> = Vec::with_capacity(interactions.len());
        let mut user_features: Vec<(usize, f64)> = Vec::with_capacity(features.len());

        // Convert interactions (item_guid -> item_idx)
        for (key, val) in interactions.iter() {
            let item_guid: &str = key.extract()?;
            let value: f64 = val.extract()?;
            // Silently ignore items not seen in training
            if let Some(&item_idx) = self.model.mappings.item_to_idx.get(item_guid) {
                user_interactions.push((item_idx, value));
            }
        }

        // Convert features (feature_name -> feature_idx)
        for (key, val) in features.iter() {
            let feature_name: &str = key.extract()?;
            let value: f64 = val.extract()?;
            // Silently ignore features not seen in training
            if let Some(&feature_idx) = self.model.mappings.user_feature_to_idx.get(feature_name) {
                user_features.push((feature_idx, value));
            }
        }

        // --- 2. Call the internal Rust prediction function ---
        let scores: Vec<f64> =
            self.model
                .predict(&user_interactions, &user_features, self.model.beta);

        // --- 3. Process and sort results ---
        // Build a set of interacted items for O(1) lookup
        let interacted: HashSet<usize> = user_interactions.iter().map(|(idx, _)| *idx).collect();

        let mut results_with_idx: Vec<(usize, f64)> = scores
            .into_iter()
            .enumerate()
            // Don't recommend items the user has already interacted with
            .filter(|(idx, _score)| !interacted.contains(idx))
            .collect();

        // Sort by score, descending.
        results_with_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // --- 4. Convert top-K results back to Python types ---
        let py_results = PyList::empty(py);
        for (idx, score) in results_with_idx.into_iter().take(top_k) {
            // Map item_idx back to item_guid
            if let Some(guid) = self.model.mappings.idx_to_item.get(idx) {
                let py_guid = PyString::new(py, guid);
                let py_score = PyFloat::new(py, score);
                py_results.append((py_guid, py_score))?;
            }
        }

        Ok(py_results)
    }

    /// Predicts recommendation scores for multiple users at once (batch mode).
    ///
    /// This is more efficient than calling `predict()` in a loop from Python
    /// because it avoids repeated Python/Rust boundary crossings.
    ///
    /// Args:
    ///     users (list[dict]): A list of user dicts, each with keys:
    ///         - "interactions": dict[str, float] of item_guid -> value
    ///         - "features": dict[str, float] of feature_name -> value
    ///     top_k (int): The number of top recommendations per user.
    ///
    /// Returns:
    ///     list[list[tuple[str, float]]]:
    ///         For each user, a list of (item_guid, score) tuples sorted descending.
    #[pyo3(signature = (users, top_k=100))]
    fn predict_batch<'py>(
        &self,
        py: Python<'py>,
        users: &Bound<'_, PyList>,
        top_k: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let mut batch_inputs = Vec::with_capacity(users.len());

        // Convert all users' Python dicts to Rust UserInput structs
        for user_obj in users.iter() {
            let user_dict: &Bound<'_, PyDict> = user_obj
                .cast()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(e.to_string()))?;

            let mut interactions = Vec::new();
            if let Some(inter_obj) = user_dict.get_item("interactions")? {
                let inter_dict: &Bound<'_, PyDict> = inter_obj
                    .cast()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(e.to_string()))?;
                for (key, val) in inter_dict.iter() {
                    let guid: &str = key.extract()?;
                    let value: f64 = val.extract()?;
                    if let Some(&idx) = self.model.mappings.item_to_idx.get(guid) {
                        interactions.push((idx, value));
                    }
                }
            }

            let mut features = Vec::new();
            if let Some(feat_obj) = user_dict.get_item("features")? {
                let feat_dict: &Bound<'_, PyDict> = feat_obj
                    .cast()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(e.to_string()))?;
                for (key, val) in feat_dict.iter() {
                    let name: &str = key.extract()?;
                    let value: f64 = val.extract()?;
                    if let Some(&idx) = self.model.mappings.user_feature_to_idx.get(name) {
                        features.push((idx, value));
                    }
                }
            }

            batch_inputs.push(serving::UserInput {
                interactions,
                features,
            });
        }

        // Run batch prediction with top-K filtering.
        // Release the GIL so other Python threads can proceed during rayon parallel work.
        let batch_results =
            py.detach(|| serving::predict_batch_top_k(&self.model, &batch_inputs, top_k));

        // Convert results back to Python
        let py_outer = PyList::empty(py);
        for user_results in batch_results {
            let py_inner = PyList::empty(py);
            for (idx, score) in user_results {
                if let Some(guid) = self.model.mappings.idx_to_item.get(idx) {
                    let py_guid = PyString::new(py, guid);
                    let py_score = PyFloat::new(py, score);
                    py_inner.append((py_guid, py_score))?;
                }
            }
            py_outer.append(py_inner)?;
        }

        Ok(py_outer)
    }

    /// Predicts similar items for a given item (More-Like-This / MLT).
    ///
    /// Uses the item-item block of the S matrix to find the most similar items
    /// to a given source item. This leverages the same learned weight matrix
    /// used for user recommendations.
    ///
    /// Args:
    ///     item_guid (str): The item GUID to find similar items for.
    ///     top_k (int): The number of similar items to return.
    ///
    /// Returns:
    ///     list[tuple[str, float]]:
    ///         A list of (item_guid, score) tuples, sorted descending by similarity.
    ///         Returns an empty list if the item_guid is unknown.
    #[pyo3(signature = (item_guid, top_k=20))]
    fn predict_similar_items<'py>(
        &self,
        py: Python<'py>,
        item_guid: &str,
        top_k: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let py_results = PyList::empty(py);

        // Look up item index
        let item_idx = match self.model.mappings.item_to_idx.get(item_guid) {
            Some(&idx) => idx,
            None => return Ok(py_results), // Unknown item -> empty list
        };

        let similar = self.model.predict_similar_items(item_idx, top_k);

        for (idx, score) in similar {
            if let Some(guid) = self.model.mappings.idx_to_item.get(idx) {
                let py_guid = PyString::new(py, guid);
                let py_score = PyFloat::new(py, score);
                py_results.append((py_guid, py_score))?;
            }
        }

        Ok(py_results)
    }

    /// Validates the trained model, checking for common issues.
    ///
    /// Returns:
    ///     tuple[bool, list[str]]:
    ///         A tuple of (passed, messages) where passed is True if all checks
    ///         passed and messages is a list of diagnostic strings.
    fn validate<'py>(&self, py: Python<'py>) -> PyResult<(bool, Bound<'py, PyList>)> {
        let report = self.model.validate();
        let py_messages = PyList::new(py, &report.messages)?;
        Ok((report.passed, py_messages))
    }

    /// Saves the trained model to a binary file.
    ///
    /// Args:
    ///     path (str): File path to save the model to (e.g., "model.fease").
    fn save(&self, path: String) -> PyResult<()> {
        serialization::save_model(&self.model, Path::new(&path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
    }
}

/// Loads a previously saved FEASE model from disk.
///
/// Args:
///     path (str): File path to load the model from.
///
/// Returns:
///     FeaseModel: The loaded model, ready for predictions.
#[pyfunction]
fn load_model(path: String) -> PyResult<FeaseModel> {
    let model = serialization::load_model(Path::new(&path))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    // Validate loaded model (same as build_and_train does after training)
    let report = model.validate();
    if !report.passed {
        let msg = format!(
            "Loaded model failed validation:\n{}",
            report.messages.join("\n")
        );
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg));
    }
    for msg in &report.messages {
        log::info!("[Rust] Loaded model validation: {}", msg);
    }

    Ok(FeaseModel {
        num_items: model.num_items,
        num_user_features: model.num_user_features,
        num_item_features: model.num_item_features,
        model,
    })
}

/// Runs data quality checks against historical baselines.
///
/// Uses the GaussianAnomalyDetector pattern from the Python EASE QA pipeline:
/// computes confidence intervals from historical data and checks if current
/// values fall within the expected range.
///
/// Args:
///     historical_users (list[float]): Historical distinct-user counts (e.g., per day).
///     historical_items (list[float]): Historical distinct-item counts.
///     historical_interactions (list[float]): Historical total-interaction counts.
///     current_users (float): Current run's distinct user count.
///     current_items (float): Current run's distinct item count.
///     current_interactions (float): Current run's total interaction count.
///     config (dict, optional): Override default std multipliers. Keys:
///         - "distinct_users_multiplier" (float, default 5.0)
///         - "distinct_items_multiplier" (float, default 5.0)
///         - "interactions_multiplier" (float, default 5.0)
///
/// Returns:
///     tuple[bool, list[str]]:
///         (all_passed, messages) where messages describe each check result.
#[pyfunction(signature = (
    historical_users,
    historical_items,
    historical_interactions,
    current_users,
    current_items,
    current_interactions,
    config = None
))]
#[allow(clippy::too_many_arguments)]
fn validate_data<'py>(
    py: Python<'py>,
    historical_users: Vec<f64>,
    historical_items: Vec<f64>,
    historical_interactions: Vec<f64>,
    current_users: f64,
    current_items: f64,
    current_interactions: f64,
    config: Option<&Bound<'_, PyDict>>,
) -> PyResult<(bool, Bound<'py, PyList>)> {
    let mut validation_config = data_validation::DataValidationConfig::default();

    if let Some(cfg) = config {
        if let Some(val) = cfg.get_item("distinct_users_multiplier")? {
            validation_config.distinct_users_multiplier = val.extract()?;
        }
        if let Some(val) = cfg.get_item("distinct_items_multiplier")? {
            validation_config.distinct_items_multiplier = val.extract()?;
        }
        if let Some(val) = cfg.get_item("interactions_multiplier")? {
            validation_config.interactions_multiplier = val.extract()?;
        }
    }

    let report = data_validation::validate_data_counts(
        &historical_users,
        &historical_items,
        &historical_interactions,
        current_users,
        current_items,
        current_interactions,
        &validation_config,
    );

    let messages: Vec<String> = report.results.iter().map(|r| r.to_string()).collect();
    let py_messages = PyList::new(py, &messages)?;

    Ok((report.all_passed(), py_messages))
}

/// A Python-callable function to build and train the model from file paths.
///
/// This is the main "factory" function for creating a FeaseModel instance
/// from your Python/Databricks environment.
///
/// Args:
///     interactions_path (str):
///         Local file path to the interactions Parquet/CSV file.
///     user_features_path (str):
///         Local file path to the user features Parquet/CSV file.
///     item_features_path (str):
///         Local file path to the item features Parquet/CSV file.
///     alpha (float): Weight for item features.
///     beta (float): Weight for user features.
///     lambda_ (float): L2 regularization term.
///     meta_weight (float): Weight for metadata rows in the Gram matrix.
///         When 0 or 1, metadata is weighted equally with interactions.
///         Values > 1 increase metadata influence; values < 1 decrease it.
#[pyfunction(signature = (
    interactions_path,
    user_features_path,
    item_features_path,
    alpha = 1.0,
    beta = 1.0,
    lambda_ = 100.0,
    meta_weight = 0.0,
    decay_rate = 0.0,
    ips_alpha = 0.0,
    sparsity_threshold = 0.0,
    event_weights = None
))]
#[allow(clippy::too_many_arguments)]
fn build_and_train(
    interactions_path: String,
    user_features_path: String,
    item_features_path: String,
    alpha: f64,
    beta: f64,
    lambda_: f64,
    meta_weight: f64,
    decay_rate: f64,
    ips_alpha: f64,
    sparsity_threshold: f64,
    event_weights: Option<&Bound<'_, PyDict>>,
) -> PyResult<FeaseModel> {
    log::info!("--- [Rust] Starting Model Training ---");
    let start_time = Instant::now();

    // --- Input validation ---
    if decay_rate < 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "decay_rate must be >= 0.0",
        ));
    }
    if ips_alpha < 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "ips_alpha must be >= 0.0",
        ));
    }
    if sparsity_threshold < 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "sparsity_threshold must be >= 0.0",
        ));
    }

    // --- Build WeightingConfig from Python args ---
    let weighting_config = if decay_rate > 0.0
        || ips_alpha > 0.0
        || sparsity_threshold > 0.0
        || event_weights.is_some()
    {
        let ew = match event_weights {
            Some(dict) => {
                let mut map = std::collections::HashMap::new();
                for (key, val) in dict.iter() {
                    let k: String = key.extract()?;
                    let v: f64 = val.extract()?;
                    map.insert(k, v);
                }
                Some(map)
            }
            None => None,
        };
        Some(weighting::WeightingConfig {
            event_weights: ew,
            decay_rate,
            ips_alpha,
            sparsity_threshold,
        })
    } else {
        None
    };

    // --- 1. Build Matrices ---
    let (x_mat, u_mat, t_mat, mappings) = match data_pipeline::build_matrices(
        &interactions_path,
        &user_features_path,
        &item_features_path,
        weighting_config.as_ref(),
    ) {
        Ok(data) => data,
        Err(e) => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                e.to_string(),
            ));
        }
    };
    log::info!(
        "[Rust] Matrix build complete in {:.2}s",
        start_time.elapsed().as_secs_f32()
    );

    let num_items = x_mat.cols();
    let num_user_features = u_mat.cols();
    let num_item_features = t_mat.rows();

    // --- 2. Train Model ---
    let train_start = Instant::now();
    let mut rust_model = RustFeaseModel::new(
        num_items,
        num_user_features,
        num_item_features,
        alpha,
        beta,
        lambda_,
        meta_weight,
        mappings,
    );

    // This is the main compute-heavy step
    if let Err(e) = rust_model.train(&x_mat, &u_mat, &t_mat) {
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            e.to_string(),
        ));
    }
    log::info!(
        "[Rust] Core model training complete in {:.2}s",
        train_start.elapsed().as_secs_f32()
    );

    rust_model.weighting_config = weighting_config;

    // --- 2b. Apply sparsity pruning ---
    if sparsity_threshold > 0.0 {
        rust_model.prune_sparse(sparsity_threshold);
    }

    // --- 3. Validate Model ---
    let report = rust_model.validate();
    if !report.passed {
        let msg = format!("Model validation failed:\n{}", report.messages.join("\n"));
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg));
    }
    for msg in &report.messages {
        log::info!("[Rust] Validation: {}", msg);
    }

    // --- 4. Wrap in Python Class ---
    let model = FeaseModel {
        model: rust_model,
        num_items,
        num_user_features,
        num_item_features,
    };

    log::info!(
        "[Rust] Total training time: {:.2}s",
        start_time.elapsed().as_secs_f32()
    );

    Ok(model)
}

/// A Python-accessible registry for territory-based multi-model routing.
///
/// This wraps the Rust `FeaseModelRegistry`, allowing Python callers to register
/// multiple trained `FeaseModel` instances (one per territory/region) and route
/// predictions to the correct model based on a territory key.
///
/// Example:
///     >>> registry = FeaseRegistry(fallback_territory="US")
///     >>> registry.register("US", us_model)
///     >>> registry.register("BR", br_model)
///     >>> scores = registry.predict("US", [(0, 1.0)], [(0, 1.0)])
///     >>> top_recs = registry.predict_top_k("JP", [(0, 1.0)], 10)  # falls back to US
#[pyclass]
struct FeaseRegistry {
    inner: serving::FeaseModelRegistry,
}

#[pymethods]
impl FeaseRegistry {
    /// Creates a new, empty registry.
    ///
    /// Args:
    ///     fallback_territory (str, optional): If set, unknown territories will
    ///         fall back to this territory's model instead of raising an error.
    #[new]
    #[pyo3(signature = (fallback_territory=None))]
    fn new(fallback_territory: Option<String>) -> Self {
        let inner = match fallback_territory {
            Some(fb) => serving::FeaseModelRegistry::with_fallback(fb),
            None => serving::FeaseModelRegistry::new(),
        };
        FeaseRegistry { inner }
    }

    /// Registers a trained FeaseModel for a territory.
    ///
    /// The model is cloned into the registry, so the original FeaseModel remains
    /// usable independently.
    ///
    /// Args:
    ///     territory (str): Territory key (e.g., "US", "BR", "EMEA").
    ///     model (FeaseModel): A trained model to associate with this territory.
    fn register(&mut self, territory: String, model: &FeaseModel) {
        self.inner.register(territory, model.model.clone());
    }

    /// Removes and returns True if a model was registered for the given territory.
    ///
    /// Args:
    ///     territory (str): Territory key to remove.
    ///
    /// Returns:
    ///     bool: True if a model was removed, False if no model was registered.
    fn unregister(&mut self, territory: &str) -> bool {
        self.inner.unregister(territory).is_some()
    }

    /// Lists all registered territory names.
    ///
    /// Returns:
    ///     list[str]: Territory keys in arbitrary order.
    fn territories(&self) -> Vec<String> {
        self.inner
            .territories()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Returns the number of registered models.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Returns True if any models are registered.
    fn __bool__(&self) -> bool {
        !self.inner.is_empty()
    }

    /// Predicts scores for a user in a specific territory (index-based).
    ///
    /// This is the low-level API using numeric indices. For string-based predictions,
    /// retrieve the model and call its `predict` method directly.
    ///
    /// Args:
    ///     territory (str): Territory key to route to.
    ///     user_interactions (list[tuple[int, float]]): (item_index, value) pairs.
    ///     user_features (list[tuple[int, float]], optional): (feature_index, value) pairs.
    ///
    /// Returns:
    ///     list[float]: Predicted scores for all items (length = num_items).
    ///
    /// Raises:
    ///     ValueError: If the territory is unknown and no fallback is set.
    #[pyo3(signature = (territory, user_interactions, user_features=None))]
    fn predict(
        &self,
        territory: &str,
        user_interactions: Vec<(usize, f64)>,
        user_features: Option<Vec<(usize, f64)>>,
    ) -> PyResult<Vec<f64>> {
        let features = user_features.unwrap_or_default();
        self.inner
            .predict(territory, &user_interactions, &features)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// Predicts top-K items for a user in a territory, excluding interacted items.
    ///
    /// Args:
    ///     territory (str): Territory key to route to.
    ///     user_interactions (list[tuple[int, float]]): (item_index, value) pairs.
    ///     top_k (int): Number of top items to return.
    ///     user_features (list[tuple[int, float]], optional): (feature_index, value) pairs.
    ///
    /// Returns:
    ///     list[tuple[int, float]]: Top-K (item_index, score) pairs, sorted descending.
    ///
    /// Raises:
    ///     ValueError: If the territory is unknown and no fallback is set.
    #[pyo3(signature = (territory, user_interactions, top_k, user_features=None))]
    fn predict_top_k(
        &self,
        territory: &str,
        user_interactions: Vec<(usize, f64)>,
        top_k: usize,
        user_features: Option<Vec<(usize, f64)>>,
    ) -> PyResult<Vec<(usize, f64)>> {
        let features = user_features.unwrap_or_default();
        let model = self.inner.get_model(territory).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "No model registered for territory '{}' (and no fallback available)",
                territory
            ))
        })?;

        let scores = model.predict(&user_interactions, &features, model.beta);
        let interacted_indices: Vec<usize> =
            user_interactions.iter().map(|(idx, _)| *idx).collect();

        Ok(serving::filter_sort_top_k(
            scores,
            &interacted_indices,
            top_k,
        ))
    }

    /// Predicts similar items for a given item index in a specific territory.
    ///
    /// Args:
    ///     territory (str): Territory key to route to.
    ///     item_idx (int): The item index to find similar items for.
    ///     top_k (int): Number of similar items to return.
    ///
    /// Returns:
    ///     list[tuple[int, float]]: (item_index, score) pairs, sorted descending.
    ///
    /// Raises:
    ///     ValueError: If the territory is unknown and no fallback is set.
    #[pyo3(signature = (territory, item_idx, top_k=20))]
    fn predict_similar_items(
        &self,
        territory: &str,
        item_idx: usize,
        top_k: usize,
    ) -> PyResult<Vec<(usize, f64)>> {
        self.inner
            .predict_similar_items(territory, item_idx, top_k)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
}

// --- Ranking Evaluation Metrics (PyO3 wrappers) ---

/// Precision@K: fraction of top-K recommendations that are relevant.
#[pyfunction]
#[pyo3(signature = (recommended, relevant, k))]
fn precision_at_k(recommended: Vec<usize>, relevant: HashSet<usize>, k: usize) -> f64 {
    metrics::precision_at_k(&recommended, &relevant, k)
}

/// Recall@K: fraction of relevant items captured in the top-K recommendations.
#[pyfunction]
#[pyo3(signature = (recommended, relevant, k))]
fn recall_at_k(recommended: Vec<usize>, relevant: HashSet<usize>, k: usize) -> f64 {
    metrics::recall_at_k(&recommended, &relevant, k)
}

/// NDCG@K: Normalized Discounted Cumulative Gain at K (binary relevance).
#[pyfunction]
#[pyo3(signature = (recommended, relevant, k))]
fn ndcg_at_k(recommended: Vec<usize>, relevant: HashSet<usize>, k: usize) -> f64 {
    metrics::ndcg_at_k(&recommended, &relevant, k)
}

/// Mean Average Precision over the full recommendation list.
#[pyfunction]
#[pyo3(signature = (recommended, relevant))]
fn mean_average_precision(recommended: Vec<usize>, relevant: HashSet<usize>) -> f64 {
    metrics::mean_average_precision(&recommended, &relevant)
}

/// Coverage: fraction of the item catalog recommended across all users.
#[pyfunction]
#[pyo3(signature = (all_recommendations, num_total_items))]
fn coverage(all_recommendations: Vec<Vec<usize>>, num_total_items: usize) -> f64 {
    metrics::coverage(&all_recommendations, num_total_items)
}

/// Hit Rate@K: 1.0 if any item in top-K is relevant, else 0.0.
#[pyfunction]
#[pyo3(signature = (recommended, relevant, k))]
fn hit_rate_at_k(recommended: Vec<usize>, relevant: HashSet<usize>, k: usize) -> f64 {
    metrics::hit_rate_at_k(&recommended, &relevant, k)
}

/// Defines the Python module.
/// This function is called when Python runs `import rust_fease_recommender`.
#[pymodule]
fn rust_fease_recommender(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_and_train, m)?)?;
    m.add_function(wrap_pyfunction!(load_model, m)?)?;
    m.add_function(wrap_pyfunction!(validate_data, m)?)?;
    m.add_function(wrap_pyfunction!(precision_at_k, m)?)?;
    m.add_function(wrap_pyfunction!(recall_at_k, m)?)?;
    m.add_function(wrap_pyfunction!(ndcg_at_k, m)?)?;
    m.add_function(wrap_pyfunction!(mean_average_precision, m)?)?;
    m.add_function(wrap_pyfunction!(coverage, m)?)?;
    m.add_function(wrap_pyfunction!(hit_rate_at_k, m)?)?;
    m.add_class::<FeaseModel>()?;
    m.add_class::<FeaseRegistry>()?;
    Ok(())
}
