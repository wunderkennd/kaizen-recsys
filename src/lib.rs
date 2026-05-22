//! This file is the main entrypoint for the Python library.
//! It uses PyO3 to define the Python-callable functions and the `FeaseModel` class.
//! This is the "bridge" between Python and Rust.

// burn's deeply nested associated types exceed Rust's default recursion
// limit of 128 when a backend-generic model is instantiated. Bumping it
// here (rather than in the ml-models module) avoids a confusing build
// failure when Phase 2b adds the first SASRec model. Harmless when the
// feature is off. See issue #24 research findings.
#![recursion_limit = "256"]

use model::RustFeaseModel; // The internal Rust struct
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyList, PyString};
use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

mod data;
mod data_pipeline;
mod data_validation;
pub mod evaluation;
pub mod metrics;
mod model;
mod models;
mod serialization;
mod serving;
pub mod tuning;
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
            let item_guid: String = key.extract()?;
            let value: f64 = val.extract()?;
            // Silently ignore items not seen in training
            if let Some(&item_idx) = self.model.mappings.item_to_idx.get(&item_guid) {
                user_interactions.push((item_idx, value));
            }
        }

        // Convert features (feature_name -> feature_idx)
        for (key, val) in features.iter() {
            let feature_name: String = key.extract()?;
            let value: f64 = val.extract()?;
            // Silently ignore features not seen in training
            if let Some(&feature_idx) = self.model.mappings.user_feature_to_idx.get(&feature_name) {
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
                    let guid: String = key.extract()?;
                    let value: f64 = val.extract()?;
                    if let Some(&idx) = self.model.mappings.item_to_idx.get(&guid) {
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
                    let name: String = key.extract()?;
                    let value: f64 = val.extract()?;
                    if let Some(&idx) = self.model.mappings.user_feature_to_idx.get(&name) {
                        features.push((idx, value));
                    }
                }
            }

            batch_inputs.push(serving::UserInput {
                interactions,
                features,
            });
        }

        // Run batch prediction with top-K filtering through the
        // generalized `&dyn RecModel` path. `EaseAdapterRef` borrows the
        // model (no deep clone of the S matrix).
        // Release the GIL so other Python threads can proceed during rayon parallel work.
        let adapter = crate::models::EaseAdapterRef::new(&self.model);
        let batch_results = py
            .detach(|| serving::predict_batch_top_k(&adapter, &batch_inputs, top_k))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Convert results back to Python
        let py_outer = PyList::empty(py);
        for user_results in batch_results {
            let py_inner = PyList::empty(py);
            for (idx, score) in user_results {
                if let Some(guid) = self.model.mappings.idx_to_item.get(idx) {
                    let py_guid = PyString::new(py, guid);
                    let py_score = PyFloat::new(py, score as f64);
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

    /// Evaluates this model against test data, computing recommendation metrics.
    ///
    /// Args:
    ///     test_interactions_path (str): Path to test interactions Parquet/CSV.
    ///     train_interactions_path (str): Path to train interactions Parquet/CSV.
    ///     k_values (list[int], optional): K values to evaluate at.
    ///         Defaults to [5, 10, 20, 50].
    ///
    /// Returns:
    ///     dict: An evaluation report with keys:
    ///         - "num_users" (int)
    ///         - "num_interactions" (int)
    ///         - "coverage" (float)
    ///         - "metrics" (list[dict]): Per-K metrics, each with keys:
    ///           "k", "precision", "recall", "ndcg", "map", "hit_rate"
    #[pyo3(signature = (test_interactions_path, train_interactions_path, user_features_path=None, k_values=None))]
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        test_interactions_path: &str,
        train_interactions_path: &str,
        user_features_path: Option<&str>,
        k_values: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let config = evaluation::EvalConfig {
            k_values: k_values.unwrap_or_else(|| vec![5, 10, 20, 50]),
        };

        // The evaluation harness is generalized over `&dyn RecModel`
        // (Phase 4a, issue #30). Wrap the concrete EASE model in a
        // borrowing adapter; the math is identical so PyO3 outputs stay
        // byte-identical (only a single `as f32` score round-trip), and
        // borrowing avoids deep-cloning the S matrix on every call.
        let adapter = crate::models::EaseAdapterRef::new(&self.model);
        let report = evaluation::evaluate_model(
            &adapter,
            test_interactions_path,
            train_interactions_path,
            user_features_path,
            &config,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let result = PyDict::new(py);
        result.set_item("num_users", report.num_users)?;
        result.set_item("num_interactions", report.num_interactions)?;
        result.set_item("coverage", report.coverage)?;

        let metrics_list = PyList::empty(py);
        for m in &report.metrics_at_k {
            let m_dict = PyDict::new(py);
            m_dict.set_item("k", m.k)?;
            m_dict.set_item("precision", m.precision)?;
            m_dict.set_item("recall", m.recall)?;
            m_dict.set_item("ndcg", m.ndcg)?;
            m_dict.set_item("map", m.map)?;
            m_dict.set_item("hit_rate", m.hit_rate)?;
            metrics_list.append(m_dict)?;
        }
        result.set_item("metrics", metrics_list)?;

        Ok(result)
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

/// Splits interaction data randomly per user into train and test sets.
///
/// Args:
///     interactions_path (str): Path to interactions Parquet/CSV file.
///     train_output (str): Output path for the train split.
///     test_output (str): Output path for the test split.
///     test_ratio (float): Fraction of each user's interactions to hold out (0.0-1.0).
///     seed (int): Random seed for reproducibility.
///
/// Returns:
///     tuple[int, int, int, int]: (train_interactions, test_interactions, train_users, test_users)
#[pyfunction]
#[pyo3(signature = (interactions_path, train_output, test_output, test_ratio=0.2, seed=42))]
fn random_split(
    interactions_path: &str,
    train_output: &str,
    test_output: &str,
    test_ratio: f64,
    seed: u64,
) -> PyResult<(usize, usize, usize, usize)> {
    let stats = evaluation::random_split(
        interactions_path,
        train_output,
        test_output,
        test_ratio,
        seed,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok((
        stats.train_interactions,
        stats.test_interactions,
        stats.train_users,
        stats.test_users,
    ))
}

/// Splits interaction data by time: recent interactions (days_ago <= cutoff) go to test.
///
/// Args:
///     interactions_path (str): Path to interactions Parquet/CSV file (must have `days_ago` column).
///     train_output (str): Output path for the train split.
///     test_output (str): Output path for the test split.
///     days_ago_cutoff (float): Interactions with days_ago <= this go to test.
///
/// Returns:
///     tuple[int, int, int, int]: (train_interactions, test_interactions, train_users, test_users)
#[pyfunction]
#[pyo3(signature = (interactions_path, train_output, test_output, days_ago_cutoff))]
fn temporal_split(
    interactions_path: &str,
    train_output: &str,
    test_output: &str,
    days_ago_cutoff: f64,
) -> PyResult<(usize, usize, usize, usize)> {
    let stats = evaluation::temporal_split(
        interactions_path,
        train_output,
        test_output,
        days_ago_cutoff,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok((
        stats.train_interactions,
        stats.test_interactions,
        stats.train_users,
        stats.test_users,
    ))
}

/// Leave-K-Out split: holds out exactly k random interactions per user for test.
///
/// Users with fewer than k+1 interactions go entirely to train.
///
/// Args:
///     interactions_path (str): Path to interactions Parquet/CSV file.
///     train_output (str): Output path for the train split.
///     test_output (str): Output path for the test split.
///     k (int): Number of interactions to hold out per user.
///     seed (int): Random seed for reproducibility.
///
/// Returns:
///     tuple[int, int, int, int]: (train_interactions, test_interactions, train_users, test_users)
#[pyfunction]
#[pyo3(signature = (interactions_path, train_output, test_output, k=1, seed=42))]
fn leave_k_out_split(
    interactions_path: &str,
    train_output: &str,
    test_output: &str,
    k: usize,
    seed: u64,
) -> PyResult<(usize, usize, usize, usize)> {
    let stats =
        evaluation::leave_k_out_split(interactions_path, train_output, test_output, k, seed)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok((
        stats.train_interactions,
        stats.test_interactions,
        stats.train_users,
        stats.test_users,
    ))
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
/// This wraps the Rust `ModelRegistry`, allowing Python callers to register
/// trained EASE / SASRec / Two-Tower models (one per territory/region) and route
/// predictions to the correct model based on a territory key (#56).
///
/// Note on `fallback_territory` + per-model predict (#56): the fallback
/// is resolved purely by territory key, not by model kind. If the
/// registered fallback's kind doesn't match the `predict_top_k_*`
/// method being called, the call errors with the same kind-mismatch
/// message you'd get on a directly-registered mismatch (e.g. calling
/// `predict_top_k_sasrec("JP", ...)` when "JP" is unknown and the
/// fallback "US" holds an EASE model). The error message points at the
/// correct method for the actual model kind, so callers can recover.
///
/// Example:
///     >>> registry = ModelRegistry(fallback_territory="US")
///     >>> registry.register("US", us_model)
///     >>> registry.register("BR", br_model)
///     >>> scores = registry.predict("US", [(0, 1.0)], [(0, 1.0)])
///     >>> top_recs = registry.predict_top_k("JP", [(0, 1.0)], 10)  # falls back to US
#[pyclass]
struct ModelRegistry {
    inner: serving::ModelRegistry,
}

#[pymethods]
impl ModelRegistry {
    /// Creates a new, empty registry.
    ///
    /// Args:
    ///     fallback_territory (str, optional): If set, unknown territories will
    ///         fall back to this territory's model instead of raising an error.
    #[new]
    #[pyo3(signature = (fallback_territory=None))]
    fn new(fallback_territory: Option<String>) -> Self {
        let inner = match fallback_territory {
            Some(fb) => serving::ModelRegistry::with_fallback(fb),
            None => serving::ModelRegistry::new(),
        };
        ModelRegistry { inner }
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
            .map(|scores| scores.into_iter().map(|s| s as f64).collect())
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
        self.inner
            .predict_top_k(territory, &user_interactions, &features, top_k)
            .map(|ranked| ranked.into_iter().map(|(i, s)| (i, s as f64)).collect())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
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
            .map(|ranked| ranked.into_iter().map(|(i, s)| (i, s as f64)).collect())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// Registers a trained SASRec model for a territory (#56). Mirrors
    /// `register` (EASE) but for `SASRecModel`. The underlying trained
    /// model is cloned into the registry; the source `SASRecModel`
    /// instance remains usable. Available only when the extension is
    /// built with `--features ml-models`.
    #[cfg(feature = "ml-models")]
    fn register_sasrec(&mut self, territory: String, model: &sasrec_py::SASRecModel) {
        self.inner
            .register_model(territory, Box::new(model.model.clone()));
    }

    /// Registers a trained Two-Tower model for a territory (#56).
    /// Mirrors `register` (EASE) but for `TwoTowerModel`. Available
    /// only when the extension is built with `--features ml-models`.
    #[cfg(feature = "ml-models")]
    fn register_two_tower(&mut self, territory: String, model: &two_tower_py::TwoTowerModel) {
        self.inner
            .register_model(territory, Box::new(model.model.clone()));
    }

    /// Predicts top-K items for an EASE territory using string ids
    /// (#56). Mirrors `FeaseModel.predict`'s API surface so callers no
    /// longer have to map `dict[str, float]` to integer indices by
    /// hand. The existing index-based `predict_top_k` is preserved for
    /// back-compat.
    ///
    /// Scores are routed through the `RecModel` trait, whose contract is
    /// f32 — so output values differ from a direct `FeaseModel.predict`
    /// call (which keeps EASE math in f64) by sub-ulp rounding (~1e-7).
    /// Ranking is unaffected; never compare scores bit-exact across the
    /// two surfaces.
    ///
    /// Errors:
    ///     ValueError: territory is unknown, or the registered model
    ///         is not EASE.
    #[pyo3(signature = (territory, interactions, features=None, top_k=100))]
    fn predict_top_k_ease(
        &self,
        territory: &str,
        interactions: &Bound<'_, PyDict>,
        features: Option<&Bound<'_, PyDict>>,
        top_k: usize,
    ) -> PyResult<Vec<(String, f64)>> {
        let interactions_map = pydict_to_str_f64(interactions)?;
        let features_map = match features {
            Some(d) => pydict_to_str_f64(d)?,
            None => ahash::AHashMap::new(),
        };
        self.inner
            .predict_top_k_ease(territory, &interactions_map, &features_map, top_k)
            .map(|ranked| ranked.into_iter().map(|(i, s)| (i, s as f64)).collect())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// Predicts top-K items for a SASRec territory using string ids
    /// (#56). `history` is the user's chronological item-id list,
    /// oldest first — same convention as `SASRecModel.predict`.
    /// Unknown item ids are silently skipped.
    ///
    /// Errors:
    ///     ValueError: territory is unknown, or the registered model
    ///         is not SASRec.
    #[cfg(feature = "ml-models")]
    #[pyo3(signature = (territory, history, top_k=100))]
    fn predict_top_k_sasrec(
        &self,
        territory: &str,
        history: Vec<String>,
        top_k: usize,
    ) -> PyResult<Vec<(String, f64)>> {
        self.inner
            .predict_top_k_sasrec(territory, &history, top_k)
            .map(|ranked| ranked.into_iter().map(|(i, s)| (i, s as f64)).collect())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// Predicts top-K items for a Two-Tower territory using string ids
    /// (#56). Warm users use their learned id row; unknown users fall
    /// back to the reserved cold-start row. Optional `features`
    /// follows the same routing rules as `TwoTowerModel.predict`
    /// (one-hot → cat embedding, numeric → dense column, unknown
    /// silently skipped — see #55).
    ///
    /// Errors:
    ///     ValueError: territory is unknown, or the registered model
    ///         is not Two-Tower.
    #[cfg(feature = "ml-models")]
    #[pyo3(signature = (territory, user_id, features=None, top_k=100))]
    fn predict_top_k_two_tower(
        &self,
        territory: &str,
        user_id: &str,
        features: Option<&Bound<'_, PyDict>>,
        top_k: usize,
    ) -> PyResult<Vec<(String, f64)>> {
        let features_map = match features {
            Some(d) => pydict_to_str_f64(d)?,
            None => ahash::AHashMap::new(),
        };
        self.inner
            .predict_top_k_two_tower(territory, user_id, &features_map, top_k)
            .map(|ranked| ranked.into_iter().map(|(i, s)| (i, s as f64)).collect())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
}

/// Convert a `dict[str, float]` PyDict into the
/// `ahash::AHashMap<String, f64>` the Rust registry methods consume.
fn pydict_to_str_f64(d: &Bound<'_, PyDict>) -> PyResult<ahash::AHashMap<String, f64>> {
    let mut map = ahash::AHashMap::with_capacity(d.len());
    for (k, v) in d.iter() {
        let key: String = k.extract()?;
        let val: f64 = v.extract()?;
        map.insert(key, val);
    }
    Ok(map)
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

/// Grid search over hyperparameters with k-fold cross-validation.
///
/// Args:
///     interactions_path (str): Path to interactions Parquet/CSV file.
///     user_features_path (str): Path to user features Parquet/CSV file.
///     item_features_path (str): Path to item features Parquet/CSV file.
///     param_grid (dict): Dict of parameter name -> list of values to try.
///         Keys: "alpha", "beta", "lambda_", "meta_weight", "decay_rate",
///         "ips_alpha", "sparsity_threshold". Missing keys get default single-element lists.
///     n_folds (int): Number of cross-validation folds (default 3).
///     eval_k (int): k for NDCG@k evaluation metric (default 10).
///     seed (int): Random seed for fold generation (default 42).
///
/// Returns:
///     dict: {"best_params": {...}, "best_score": float, "metric": str,
///            "trials": [{"params": {...}, "mean_score": float, "fold_scores": [...]}, ...]}
#[pyfunction]
#[pyo3(signature = (
    interactions_path,
    user_features_path,
    item_features_path,
    param_grid,
    n_folds = 3,
    eval_k = 10,
    seed = 42
))]
#[allow(clippy::too_many_arguments)]
fn grid_search_py(
    py: Python<'_>,
    interactions_path: &str,
    user_features_path: &str,
    item_features_path: &str,
    param_grid: &Bound<'_, PyDict>,
    n_folds: usize,
    eval_k: usize,
    seed: u64,
) -> PyResult<Py<PyAny>> {
    let grid = parse_param_grid(param_grid)?;
    // Release the GIL: the search is rayon-parallelized pure Rust work that
    // never touches Python objects, so holding the GIL would needlessly
    // block other Python threads for the whole (now longer, parallel) run.
    // Mirrors the `predict_batch` pattern above.
    let result = py
        .detach(|| {
            tuning::grid_search(
                interactions_path,
                user_features_path,
                item_features_path,
                &grid,
                n_folds,
                eval_k,
                seed,
            )
        })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    search_result_to_py(py, &result)
}

/// Random search over hyperparameters with k-fold cross-validation.
///
/// Args:
///     interactions_path (str): Path to interactions Parquet/CSV file.
///     user_features_path (str): Path to user features Parquet/CSV file.
///     item_features_path (str): Path to item features Parquet/CSV file.
///     param_grid (dict): Dict of parameter name -> list of values to sample from.
///     n_trials (int): Number of random configurations to evaluate (default 10).
///     n_folds (int): Number of cross-validation folds (default 3).
///     eval_k (int): k for NDCG@k evaluation metric (default 10).
///     seed (int): Random seed for fold generation and sampling (default 42).
///
/// Returns:
///     dict: Same structure as grid_search.
#[pyfunction]
#[pyo3(signature = (
    interactions_path,
    user_features_path,
    item_features_path,
    param_grid,
    n_trials = 10,
    n_folds = 3,
    eval_k = 10,
    seed = 42
))]
#[allow(clippy::too_many_arguments)]
fn random_search_py(
    py: Python<'_>,
    interactions_path: &str,
    user_features_path: &str,
    item_features_path: &str,
    param_grid: &Bound<'_, PyDict>,
    n_trials: usize,
    n_folds: usize,
    eval_k: usize,
    seed: u64,
) -> PyResult<Py<PyAny>> {
    let grid = parse_param_grid(param_grid)?;
    // Release the GIL during the rayon-parallelized search (see grid_search_py).
    let result = py
        .detach(|| {
            tuning::random_search(
                interactions_path,
                user_features_path,
                item_features_path,
                &grid,
                n_trials,
                n_folds,
                eval_k,
                seed,
            )
        })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    search_result_to_py(py, &result)
}

// ---------------------------------------------------------------------------
// Per-model search entrypoints
// ---------------------------------------------------------------------------
//
// The search machinery (`tuning::grid_search_with` / `random_search_with`)
// is generic over `tuning::FoldEvaluator<P>` and `tuning::SearchSpace`, so
// each model family gets its own PyO3 entrypoint over its own param schema:
// EASE uses the `(alpha, beta, lambda_, ...)` grid; SASRec uses
// `(embedding_dim, num_heads, num_layers, dropout, learning_rate,
// num_epochs)`; Two-Tower uses `(embedding_dim, temperature,
// learning_rate, id_dropout)`. All three run end-to-end through the shared
// k-fold + rayon runner and return the same result shape. The
// burn-backed SASRec / Two-Tower searches require the `ml-models` build
// feature; an EASE-only build reports that as a build-config error.

/// EASE grid search over hyperparameters with k-fold CV.
///
/// Identical behavior and result shape to `grid_search`; named explicitly
/// per model so SASRec/Two-Tower entrypoints can mirror it.
#[pyfunction]
#[pyo3(signature = (
    interactions_path,
    user_features_path,
    item_features_path,
    param_grid,
    n_folds = 3,
    eval_k = 10,
    seed = 42
))]
#[allow(clippy::too_many_arguments)]
fn grid_search_ease(
    py: Python<'_>,
    interactions_path: &str,
    user_features_path: &str,
    item_features_path: &str,
    param_grid: &Bound<'_, PyDict>,
    n_folds: usize,
    eval_k: usize,
    seed: u64,
) -> PyResult<Py<PyAny>> {
    grid_search_py(
        py,
        interactions_path,
        user_features_path,
        item_features_path,
        param_grid,
        n_folds,
        eval_k,
        seed,
    )
}

/// EASE random search over hyperparameters with k-fold CV.
#[pyfunction]
#[pyo3(signature = (
    interactions_path,
    user_features_path,
    item_features_path,
    param_grid,
    n_trials = 10,
    n_folds = 3,
    eval_k = 10,
    seed = 42
))]
#[allow(clippy::too_many_arguments)]
fn random_search_ease(
    py: Python<'_>,
    interactions_path: &str,
    user_features_path: &str,
    item_features_path: &str,
    param_grid: &Bound<'_, PyDict>,
    n_trials: usize,
    n_folds: usize,
    eval_k: usize,
    seed: u64,
) -> PyResult<Py<PyAny>> {
    random_search_py(
        py,
        interactions_path,
        user_features_path,
        item_features_path,
        param_grid,
        n_trials,
        n_folds,
        eval_k,
        seed,
    )
}

/// SASRec grid search over its architecture/optimizer hyperparameters
/// with k-fold CV.
///
/// `param_grid` keys: "embedding_dim", "num_heads", "num_layers"
/// (integers), "dropout", "learning_rate" (floats), "num_epochs"
/// (integers). Missing keys fall back to a single sensible default. The
/// interactions file must carry a numeric `days_ago` column (SASRec is
/// order-sensitive). `user_features_path` / `item_features_path` are
/// accepted for surface parity with EASE but unused: SASRec is a
/// sequence model. Result shape matches `grid_search_ease`. Requires the
/// `ml-models` build feature.
#[pyfunction]
#[pyo3(signature = (
    interactions_path,
    user_features_path,
    item_features_path,
    param_grid,
    n_folds = 3,
    eval_k = 10,
    seed = 42
))]
#[allow(clippy::too_many_arguments, unused_variables)]
fn grid_search_sasrec(
    py: Python<'_>,
    interactions_path: &str,
    user_features_path: &str,
    item_features_path: &str,
    param_grid: &Bound<'_, PyDict>,
    n_folds: usize,
    eval_k: usize,
    seed: u64,
) -> PyResult<Py<PyAny>> {
    #[cfg(feature = "ml-models")]
    {
        let grid = parse_sasrec_grid(param_grid)?;
        let result = py
            .detach(|| {
                let evaluator = tuning::SasRecFoldEvaluator::default();
                tuning::grid_search_with(
                    &evaluator,
                    interactions_path,
                    &grid,
                    n_folds,
                    eval_k,
                    seed,
                )
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        sasrec_search_result_to_py(py, &result)
    }
    #[cfg(not(feature = "ml-models"))]
    {
        Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "SASRec hyperparameter search requires the `ml-models` build feature \
             (the burn-backed SASRec model is not compiled into this EASE-only build).",
        ))
    }
}

/// SASRec random search over its architecture/optimizer hyperparameters
/// with k-fold CV. Same `param_grid` keys and result shape as
/// `grid_search_sasrec`. Requires the `ml-models` build feature.
#[pyfunction]
#[pyo3(signature = (
    interactions_path,
    user_features_path,
    item_features_path,
    param_grid,
    n_trials = 10,
    n_folds = 3,
    eval_k = 10,
    seed = 42
))]
#[allow(clippy::too_many_arguments, unused_variables)]
fn random_search_sasrec(
    py: Python<'_>,
    interactions_path: &str,
    user_features_path: &str,
    item_features_path: &str,
    param_grid: &Bound<'_, PyDict>,
    n_trials: usize,
    n_folds: usize,
    eval_k: usize,
    seed: u64,
) -> PyResult<Py<PyAny>> {
    #[cfg(feature = "ml-models")]
    {
        let grid = parse_sasrec_grid(param_grid)?;
        let result = py
            .detach(|| {
                let evaluator = tuning::SasRecFoldEvaluator::default();
                tuning::random_search_with(
                    &evaluator,
                    interactions_path,
                    &grid,
                    n_trials,
                    n_folds,
                    eval_k,
                    seed,
                )
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        sasrec_search_result_to_py(py, &result)
    }
    #[cfg(not(feature = "ml-models"))]
    {
        Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "SASRec hyperparameter search requires the `ml-models` build feature \
             (the burn-backed SASRec model is not compiled into this EASE-only build).",
        ))
    }
}

/// Two-Tower grid search over its hyperparameters with k-fold CV.
///
/// `param_grid` keys: "embedding_dim" (integers), "temperature",
/// "learning_rate", "id_dropout" (floats). Missing keys fall back to a
/// single sensible default. The model trains id-only from the
/// interactions file (the tuning surface takes no side-feature files);
/// `user_features_path` / `item_features_path` are accepted for surface
/// parity but unused. Result shape matches `grid_search_ease`. Requires
/// the `ml-models` build feature.
#[pyfunction]
#[pyo3(signature = (
    interactions_path,
    user_features_path,
    item_features_path,
    param_grid,
    n_folds = 3,
    eval_k = 10,
    seed = 42
))]
#[allow(clippy::too_many_arguments, unused_variables)]
fn grid_search_two_tower(
    py: Python<'_>,
    interactions_path: &str,
    user_features_path: &str,
    item_features_path: &str,
    param_grid: &Bound<'_, PyDict>,
    n_folds: usize,
    eval_k: usize,
    seed: u64,
) -> PyResult<Py<PyAny>> {
    #[cfg(feature = "ml-models")]
    {
        let grid = parse_two_tower_grid(param_grid)?;
        let result = py
            .detach(|| {
                let evaluator = tuning::TwoTowerFoldEvaluator::default();
                tuning::grid_search_with(
                    &evaluator,
                    interactions_path,
                    &grid,
                    n_folds,
                    eval_k,
                    seed,
                )
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        two_tower_search_result_to_py(py, &result)
    }
    #[cfg(not(feature = "ml-models"))]
    {
        Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "Two-Tower hyperparameter search requires the `ml-models` build feature \
             (the burn-backed Two-Tower model is not compiled into this EASE-only build).",
        ))
    }
}

/// Two-Tower random search over its hyperparameters with k-fold CV. Same
/// `param_grid` keys and result shape as `grid_search_two_tower`.
/// Requires the `ml-models` build feature.
#[pyfunction]
#[pyo3(signature = (
    interactions_path,
    user_features_path,
    item_features_path,
    param_grid,
    n_trials = 10,
    n_folds = 3,
    eval_k = 10,
    seed = 42
))]
#[allow(clippy::too_many_arguments, unused_variables)]
fn random_search_two_tower(
    py: Python<'_>,
    interactions_path: &str,
    user_features_path: &str,
    item_features_path: &str,
    param_grid: &Bound<'_, PyDict>,
    n_trials: usize,
    n_folds: usize,
    eval_k: usize,
    seed: u64,
) -> PyResult<Py<PyAny>> {
    #[cfg(feature = "ml-models")]
    {
        let grid = parse_two_tower_grid(param_grid)?;
        let result = py
            .detach(|| {
                let evaluator = tuning::TwoTowerFoldEvaluator::default();
                tuning::random_search_with(
                    &evaluator,
                    interactions_path,
                    &grid,
                    n_trials,
                    n_folds,
                    eval_k,
                    seed,
                )
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        two_tower_search_result_to_py(py, &result)
    }
    #[cfg(not(feature = "ml-models"))]
    {
        Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "Two-Tower hyperparameter search requires the `ml-models` build feature \
             (the burn-backed Two-Tower model is not compiled into this EASE-only build).",
        ))
    }
}

/// Parses a Python dict into a ParamGrid, using defaults for missing keys.
fn parse_param_grid(dict: &Bound<'_, PyDict>) -> PyResult<tuning::ParamGrid> {
    fn extract_vec(dict: &Bound<'_, PyDict>, key: &str, default: f64) -> PyResult<Vec<f64>> {
        match dict.get_item(key)? {
            Some(val) => {
                let list: Vec<f64> = val.extract()?;
                if list.is_empty() {
                    Ok(vec![default])
                } else {
                    Ok(list)
                }
            }
            None => Ok(vec![default]),
        }
    }

    Ok(tuning::ParamGrid {
        alpha: extract_vec(dict, "alpha", 1.0)?,
        beta: extract_vec(dict, "beta", 1.0)?,
        lambda_: extract_vec(dict, "lambda_", 100.0)?,
        meta_weight: extract_vec(dict, "meta_weight", 0.0)?,
        decay_rate: extract_vec(dict, "decay_rate", 0.0)?,
        ips_alpha: extract_vec(dict, "ips_alpha", 0.0)?,
        sparsity_threshold: extract_vec(dict, "sparsity_threshold", 0.0)?,
    })
}

/// Converts a SearchResult into a Python dict.
fn search_result_to_py(py: Python<'_>, result: &tuning::SearchResult) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);

    // best_params
    let bp = PyDict::new(py);
    bp.set_item("alpha", result.best_params.alpha)?;
    bp.set_item("beta", result.best_params.beta)?;
    bp.set_item("lambda_", result.best_params.lambda_)?;
    bp.set_item("meta_weight", result.best_params.meta_weight)?;
    bp.set_item("decay_rate", result.best_params.decay_rate)?;
    bp.set_item("ips_alpha", result.best_params.ips_alpha)?;
    bp.set_item("sparsity_threshold", result.best_params.sparsity_threshold)?;
    dict.set_item("best_params", bp)?;

    dict.set_item("best_score", result.best_score)?;
    dict.set_item("metric", &result.metric_name)?;

    // trials
    let trials_list = PyList::empty(py);
    for trial in &result.all_trials {
        let trial_dict = PyDict::new(py);

        let params_dict = PyDict::new(py);
        params_dict.set_item("alpha", trial.params.alpha)?;
        params_dict.set_item("beta", trial.params.beta)?;
        params_dict.set_item("lambda_", trial.params.lambda_)?;
        params_dict.set_item("meta_weight", trial.params.meta_weight)?;
        params_dict.set_item("decay_rate", trial.params.decay_rate)?;
        params_dict.set_item("ips_alpha", trial.params.ips_alpha)?;
        params_dict.set_item("sparsity_threshold", trial.params.sparsity_threshold)?;

        trial_dict.set_item("params", params_dict)?;
        trial_dict.set_item("mean_score", trial.mean_score)?;
        trial_dict.set_item("fold_scores", trial.fold_scores.clone())?;
        trials_list.append(trial_dict)?;
    }
    dict.set_item("trials", trials_list)?;

    Ok(dict.into())
}

// --- SASRec / Two-Tower search param parsing + result serialization ------
//
// ml-models-gated: these only have call sites under the same feature, and
// their grid types live behind the burn-backed model code.

#[cfg(feature = "ml-models")]
fn extract_f64_vec(dict: &Bound<'_, PyDict>, key: &str, default: f64) -> PyResult<Vec<f64>> {
    match dict.get_item(key)? {
        Some(v) => {
            let list: Vec<f64> = v.extract()?;
            Ok(if list.is_empty() { vec![default] } else { list })
        }
        None => Ok(vec![default]),
    }
}

#[cfg(feature = "ml-models")]
fn extract_usize_vec(dict: &Bound<'_, PyDict>, key: &str, default: usize) -> PyResult<Vec<usize>> {
    match dict.get_item(key)? {
        Some(v) => {
            let list: Vec<usize> = v.extract()?;
            Ok(if list.is_empty() { vec![default] } else { list })
        }
        None => Ok(vec![default]),
    }
}

/// Parse a Python dict into a [`tuning::SasRecParamGrid`], defaulting any
/// missing axis to a single sensible value.
#[cfg(feature = "ml-models")]
fn parse_sasrec_grid(dict: &Bound<'_, PyDict>) -> PyResult<tuning::SasRecParamGrid> {
    Ok(tuning::SasRecParamGrid {
        embedding_dim: extract_usize_vec(dict, "embedding_dim", 64)?,
        num_heads: extract_usize_vec(dict, "num_heads", 2)?,
        num_layers: extract_usize_vec(dict, "num_layers", 2)?,
        dropout: extract_f64_vec(dict, "dropout", 0.2)?,
        learning_rate: extract_f64_vec(dict, "learning_rate", 1e-3)?,
        num_epochs: extract_usize_vec(dict, "num_epochs", 50)?,
    })
}

/// Parse a Python dict into a [`tuning::TwoTowerParamGrid`], defaulting
/// any missing axis to a single sensible value.
#[cfg(feature = "ml-models")]
fn parse_two_tower_grid(dict: &Bound<'_, PyDict>) -> PyResult<tuning::TwoTowerParamGrid> {
    Ok(tuning::TwoTowerParamGrid {
        embedding_dim: extract_usize_vec(dict, "embedding_dim", 32)?,
        temperature: extract_f64_vec(dict, "temperature", 0.05)?,
        learning_rate: extract_f64_vec(dict, "learning_rate", 0.01)?,
        id_dropout: extract_f64_vec(dict, "id_dropout", 0.1)?,
    })
}

/// Serialize a SASRec [`tuning::SearchResult`] into the same dict shape as
/// EASE (`best_params` / `best_score` / `metric` / `trials`), with
/// SASRec's param keys.
#[cfg(feature = "ml-models")]
fn sasrec_search_result_to_py(
    py: Python<'_>,
    result: &tuning::SearchResult<tuning::SasRecParams>,
) -> PyResult<Py<PyAny>> {
    fn params_dict<'py>(py: Python<'py>, p: &tuning::SasRecParams) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("embedding_dim", p.embedding_dim)?;
        d.set_item("num_heads", p.num_heads)?;
        d.set_item("num_layers", p.num_layers)?;
        d.set_item("dropout", p.dropout)?;
        d.set_item("learning_rate", p.learning_rate)?;
        d.set_item("num_epochs", p.num_epochs)?;
        Ok(d)
    }

    let dict = PyDict::new(py);
    dict.set_item("best_params", params_dict(py, &result.best_params)?)?;
    dict.set_item("best_score", result.best_score)?;
    dict.set_item("metric", &result.metric_name)?;
    let trials_list = PyList::empty(py);
    for trial in &result.all_trials {
        let trial_dict = PyDict::new(py);
        trial_dict.set_item("params", params_dict(py, &trial.params)?)?;
        trial_dict.set_item("mean_score", trial.mean_score)?;
        trial_dict.set_item("fold_scores", trial.fold_scores.clone())?;
        trials_list.append(trial_dict)?;
    }
    dict.set_item("trials", trials_list)?;
    Ok(dict.into())
}

/// Serialize a Two-Tower [`tuning::SearchResult`] into the same dict shape
/// as EASE, with Two-Tower's param keys.
#[cfg(feature = "ml-models")]
fn two_tower_search_result_to_py(
    py: Python<'_>,
    result: &tuning::SearchResult<tuning::TwoTowerParams>,
) -> PyResult<Py<PyAny>> {
    fn params_dict<'py>(
        py: Python<'py>,
        p: &tuning::TwoTowerParams,
    ) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("embedding_dim", p.embedding_dim)?;
        d.set_item("temperature", p.temperature)?;
        d.set_item("learning_rate", p.learning_rate)?;
        d.set_item("id_dropout", p.id_dropout)?;
        Ok(d)
    }

    let dict = PyDict::new(py);
    dict.set_item("best_params", params_dict(py, &result.best_params)?)?;
    dict.set_item("best_score", result.best_score)?;
    dict.set_item("metric", &result.metric_name)?;
    let trials_list = PyList::empty(py);
    for trial in &result.all_trials {
        let trial_dict = PyDict::new(py);
        trial_dict.set_item("params", params_dict(py, &trial.params)?)?;
        trial_dict.set_item("mean_score", trial.mean_score)?;
        trial_dict.set_item("fold_scores", trial.fold_scores.clone())?;
        trials_list.append(trial_dict)?;
    }
    dict.set_item("trials", trials_list)?;
    Ok(dict.into())
}

// --- SASRec PyO3 surface (ml-models feature) -----------------------------
//
// Mirrors the `FeaseModel` class (predict / evaluate / save / load) for
// the burn-based SASRec sequence recommender. Gated on `ml-models` so the
// default EASE-only build neither pulls burn nor exposes these symbols —
// the Python package guards the import accordingly.

#[cfg(feature = "ml-models")]
mod sasrec_py {
    use super::*;
    use crate::data::sequences::build_sequences;
    use crate::models::sasrec::{SasRecConfig, SasRecTrainingConfig, TrainedSasRec, train_sasrec};
    use crate::models::{ModelInput, RecModel};
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::{Autodiff, NdArray};

    /// A trained SASRec model, callable from Python.
    #[pyclass]
    pub struct SASRecModel {
        // `pub(crate)` so `ModelRegistry::register_sasrec` (sibling
        // module in lib.rs) can clone the underlying TrainedSasRec
        // into the trait-object registry (#56).
        pub(crate) model: TrainedSasRec,
        #[pyo3(get)]
        num_items: usize,
        #[pyo3(get)]
        max_seq_len: usize,
    }

    #[pymethods]
    impl SASRecModel {
        /// Score recommendations from a chronologically-ordered history.
        ///
        /// Args:
        ///     history (list[str]): Item ids, oldest first. Unknown ids
        ///         are skipped.
        ///     top_k (int): Number of recommendations to return.
        ///
        /// Returns:
        ///     list[tuple[str, float]]: (item_id, score), descending,
        ///     excluding items already in `history`.
        #[pyo3(signature = (history, top_k=100))]
        fn predict<'py>(
            &self,
            py: Python<'py>,
            history: &Bound<'_, PyList>,
            top_k: usize,
        ) -> PyResult<Bound<'py, PyList>> {
            let mut hist_idx: Vec<usize> = Vec::with_capacity(history.len());
            for obj in history.iter() {
                let id: String = obj.extract()?;
                if let Some(&idx) = self.model.item_mapping().item_to_idx.get(&id) {
                    hist_idx.push(idx);
                }
            }

            let scores = self
                .model
                .predict_scores(ModelInput::Sequence { history: &hist_idx })
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            let seen: HashSet<usize> = hist_idx.iter().copied().collect();
            let mut ranked: Vec<(usize, f32)> = scores
                .into_iter()
                .enumerate()
                .filter(|(idx, _)| !seen.contains(idx))
                .collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let out = PyList::empty(py);
            let map = self.model.item_mapping();
            for (idx, score) in ranked.into_iter().take(top_k) {
                if let Some(guid) = map.idx_to_item.get(idx) {
                    out.append((PyString::new(py, guid), PyFloat::new(py, score as f64)))?;
                }
            }
            Ok(out)
        }

        /// Top-K items most similar to `item_id` by item-embedding cosine.
        #[pyo3(signature = (item_id, top_k=20))]
        fn predict_similar_items<'py>(
            &self,
            py: Python<'py>,
            item_id: &str,
            top_k: usize,
        ) -> PyResult<Bound<'py, PyList>> {
            let out = PyList::empty(py);
            let map = self.model.item_mapping();
            let Some(&idx) = map.item_to_idx.get(item_id) else {
                return Ok(out);
            };
            let sim = self
                .model
                .predict_similar_items(idx, top_k)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            for (i, score) in sim {
                if let Some(guid) = map.idx_to_item.get(i) {
                    out.append((PyString::new(py, guid), PyFloat::new(py, score as f64)))?;
                }
            }
            Ok(out)
        }

        /// Self-check the model state. Returns `(passed, messages)`.
        fn validate<'py>(&self, py: Python<'py>) -> PyResult<(bool, Bound<'py, PyList>)> {
            let report = self.model.validate();
            Ok((report.passed, PyList::new(py, &report.messages)?))
        }

        /// Evaluate against test interactions via the generalized
        /// `&dyn RecModel` harness (same metrics dict as `FeaseModel`).
        #[pyo3(signature = (test_interactions_path, train_interactions_path, k_values=None))]
        fn evaluate<'py>(
            &self,
            py: Python<'py>,
            test_interactions_path: &str,
            train_interactions_path: &str,
            k_values: Option<Vec<usize>>,
        ) -> PyResult<Bound<'py, PyDict>> {
            let config = evaluation::EvalConfig {
                k_values: k_values.unwrap_or_else(|| vec![5, 10, 20, 50]),
            };
            let report = evaluation::evaluate_model(
                &self.model,
                test_interactions_path,
                train_interactions_path,
                None,
                &config,
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            let result = PyDict::new(py);
            result.set_item("num_users", report.num_users)?;
            result.set_item("num_interactions", report.num_interactions)?;
            result.set_item("coverage", report.coverage)?;
            let metrics_list = PyList::empty(py);
            for m in &report.metrics_at_k {
                let d = PyDict::new(py);
                d.set_item("k", m.k)?;
                d.set_item("precision", m.precision)?;
                d.set_item("recall", m.recall)?;
                d.set_item("ndcg", m.ndcg)?;
                d.set_item("map", m.map)?;
                d.set_item("hit_rate", m.hit_rate)?;
                metrics_list.append(d)?;
            }
            result.set_item("metrics", metrics_list)?;
            Ok(result)
        }

        /// Persist the model to `path` (framed `FSAS` format).
        fn save(&self, path: String) -> PyResult<()> {
            self.model
                .save(Path::new(&path))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
        }
    }

    /// Train a SASRec model from a long-format interactions file.
    ///
    /// The interactions file must carry a numeric `days_ago` column so
    /// each user's history can be ordered chronologically (SASRec is
    /// order-sensitive; the data path fails loudly otherwise).
    ///
    /// Args:
    ///     interactions_path (str): Parquet/CSV with `user_id`,
    ///         `item_id`, `value`, `days_ago`.
    ///     embedding_dim, num_heads, num_layers, dropout: architecture.
    ///     max_seq_len: history length (also the positional-embedding cap).
    ///     num_epochs, batch_size, learning_rate, patience, seed: training.
    ///
    /// Returns:
    ///     SASRecModel
    #[pyfunction]
    #[pyo3(signature = (
        interactions_path,
        embedding_dim = 64,
        max_seq_len = 50,
        num_heads = 2,
        num_layers = 2,
        dropout = 0.2,
        num_epochs = 50,
        batch_size = 16,
        learning_rate = 1e-3,
        patience = 5,
        seed = 42
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn build_and_train_sasrec(
        interactions_path: String,
        embedding_dim: usize,
        max_seq_len: usize,
        num_heads: usize,
        num_layers: usize,
        dropout: f64,
        num_epochs: usize,
        batch_size: usize,
        learning_rate: f64,
        patience: usize,
        seed: u64,
    ) -> PyResult<SASRecModel> {
        let mappings = data_pipeline::build_interaction_mappings(&interactions_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        let dataset = build_sequences(&interactions_path, &mappings, max_seq_len)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        // vocab = catalog items + reserved pad token (see data::sequences).
        let vocab_size = mappings.idx_to_item.len() + 1;
        let model_config = SasRecConfig::new(
            vocab_size,
            embedding_dim,
            max_seq_len,
            num_heads,
            num_layers,
        )
        .with_dropout(dropout);
        let train_config = SasRecTrainingConfig::new()
            .with_num_epochs(num_epochs)
            .with_batch_size(batch_size)
            .with_learning_rate(learning_rate)
            .with_patience(patience)
            .with_seed(seed);

        let device = NdArrayDevice::default();
        let fitted =
            train_sasrec::<Autodiff<NdArray<f32>>>(&model_config, &train_config, &dataset, &device)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let trained = TrainedSasRec::new(fitted, model_config, mappings);
        let report = trained.validate();
        if !report.passed {
            let msg = format!(
                "SASRec model validation failed:\n{}",
                report.messages.join("\n")
            );
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg));
        }
        let num_items = RecModel::num_items(&trained);
        Ok(SASRecModel {
            model: trained,
            num_items,
            max_seq_len,
        })
    }

    /// Load a SASRec model previously written by `SASRecModel.save`.
    #[pyfunction]
    pub fn load_sasrec_model(path: String) -> PyResult<SASRecModel> {
        let model = TrainedSasRec::load_from(Path::new(&path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let report = model.validate();
        if !report.passed {
            let msg = format!(
                "Loaded SASRec model failed validation:\n{}",
                report.messages.join("\n")
            );
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg));
        }
        let num_items = RecModel::num_items(&model);
        let max_seq_len = model.config_max_seq_len();
        Ok(SASRecModel {
            model,
            num_items,
            max_seq_len,
        })
    }
}

/// Two-Tower Python wrapper. Compiled only with the `ml-models` feature
/// because the underlying model depends on `burn`.
///
/// Training inputs are the same long-format Parquet/CSV files as EASE
/// (`user_id`, `item_id`, `value` for interactions; long-format user /
/// item feature tables). Predict-time inputs are limited to a string
/// user id: warm users use their learned id embedding, unknown users
/// fall back to the reserved cold-start row that id-dropout trains as a
/// learned average-user prior (ADR-0001, PR #46). Predict-time arbitrary
/// user features are not currently supported — the trained model does
/// not persist a feature-name → category-index map.
#[cfg(feature = "ml-models")]
mod two_tower_py {
    use super::*;
    use crate::data::triples::{FeatureTable, load_features, load_triples};
    use crate::models::two_tower::{TrainParams, TrainedTwoTower, train};
    use crate::models::{ModelInput, RecModel};

    /// A trained Two-Tower model, callable from Python.
    #[pyclass]
    pub struct TwoTowerModel {
        // `pub(crate)` so `ModelRegistry::register_two_tower` (sibling
        // module in lib.rs) can clone the underlying TrainedTwoTower
        // into the trait-object registry (#56).
        pub(crate) model: TrainedTwoTower,
        #[pyo3(get)]
        num_items: usize,
        #[pyo3(get)]
        num_users: usize,
    }

    #[pymethods]
    impl TwoTowerModel {
        /// Score recommendations for `user_id`.
        ///
        /// Warm users (id seen in training) use their learned id-row
        /// embedding. Unknown users transparently fall back to the
        /// reserved cold-start row. Items the user already interacted
        /// with in training are *not* excluded — Two-Tower has no
        /// access to a user's full history at predict time.
        ///
        /// When `features` is supplied, each entry is matched against the
        /// user-feature maps captured at training time (#55):
        /// - One-hot/categorical features (`{"plan_free": 1.0}`) add the
        ///   slot's categorical-embedding to the user vector.
        /// - Dense numeric features (`{"tenure_days": 42.0}`) write into
        ///   the matching dense column.
        /// - Feature names unknown to the model are silently skipped.
        ///
        /// For cold-start users (unknown `user_id`), this is how to
        /// combine the learned cold-start prior with the new user's
        /// side info instead of falling back to the bare cold-start row.
        ///
        /// Args:
        ///     user_id (str): String user id.
        ///     features (dict[str, float], optional): Predict-time
        ///         user features. See above for routing rules.
        ///     top_k (int): Number of recommendations to return.
        ///
        /// Returns:
        ///     list[tuple[str, float]]: (item_id, score), descending.
        #[pyo3(signature = (user_id, features=None, top_k=100))]
        fn predict<'py>(
            &self,
            py: Python<'py>,
            user_id: &str,
            features: Option<&Bound<'_, PyDict>>,
            top_k: usize,
        ) -> PyResult<Bound<'py, PyList>> {
            let user_idx = self.model.item_mapping().user_to_idx.get(user_id).copied();

            // Translate predict-time string features through the
            // user-feature maps persisted on the trained model.
            let (cat_features, dense_features) = if let Some(d) = features {
                let mut map: ahash::AHashMap<String, f64> = ahash::AHashMap::new();
                for (k, v) in d.iter() {
                    let name: String = k.extract()?;
                    let value: f64 = v.extract()?;
                    map.insert(name, value);
                }
                self.model.resolve_user_features(&map)
            } else {
                (Vec::new(), Vec::new())
            };

            let scores = self
                .model
                .predict_scores(ModelInput::TowerUser {
                    user_idx,
                    cat_features: &cat_features,
                    dense_features: &dense_features,
                })
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            let mut ranked: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let out = PyList::empty(py);
            let map = self.model.item_mapping();
            for (idx, score) in ranked.into_iter().take(top_k) {
                if let Some(guid) = map.idx_to_item.get(idx) {
                    out.append((PyString::new(py, guid), PyFloat::new(py, score as f64)))?;
                }
            }
            Ok(out)
        }

        /// Top-K items most similar to `item_id` by item-embedding score.
        #[pyo3(signature = (item_id, top_k=20))]
        fn predict_similar_items<'py>(
            &self,
            py: Python<'py>,
            item_id: &str,
            top_k: usize,
        ) -> PyResult<Bound<'py, PyList>> {
            let out = PyList::empty(py);
            let map = self.model.item_mapping();
            let Some(&idx) = map.item_to_idx.get(item_id) else {
                return Ok(out);
            };
            let sim = self
                .model
                .predict_similar_items(idx, top_k)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            for (i, score) in sim {
                if let Some(guid) = map.idx_to_item.get(i) {
                    out.append((PyString::new(py, guid), PyFloat::new(py, score as f64)))?;
                }
            }
            Ok(out)
        }

        /// Self-check the model state. Returns `(passed, messages)`.
        fn validate<'py>(&self, py: Python<'py>) -> PyResult<(bool, Bound<'py, PyList>)> {
            let report = self.model.validate();
            Ok((report.passed, PyList::new(py, &report.messages)?))
        }

        /// Evaluate against test interactions via the `&dyn RecModel`
        /// harness routed through `TwoTowerEvalAdapter` (same metrics
        /// dict shape as `FeaseModel.evaluate`).
        #[pyo3(signature = (test_interactions_path, train_interactions_path, k_values=None))]
        fn evaluate<'py>(
            &self,
            py: Python<'py>,
            test_interactions_path: &str,
            train_interactions_path: &str,
            k_values: Option<Vec<usize>>,
        ) -> PyResult<Bound<'py, PyDict>> {
            let config = evaluation::EvalConfig {
                k_values: k_values.unwrap_or_else(|| vec![5, 10, 20, 50]),
            };
            let report = evaluation::evaluate_model(
                &self.model,
                test_interactions_path,
                train_interactions_path,
                None,
                &config,
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            let result = PyDict::new(py);
            result.set_item("num_users", report.num_users)?;
            result.set_item("num_interactions", report.num_interactions)?;
            result.set_item("coverage", report.coverage)?;
            let metrics_list = PyList::empty(py);
            for m in &report.metrics_at_k {
                let d = PyDict::new(py);
                d.set_item("k", m.k)?;
                d.set_item("precision", m.precision)?;
                d.set_item("recall", m.recall)?;
                d.set_item("ndcg", m.ndcg)?;
                d.set_item("map", m.map)?;
                d.set_item("hit_rate", m.hit_rate)?;
                metrics_list.append(d)?;
            }
            result.set_item("metrics", metrics_list)?;
            Ok(result)
        }

        /// Persist the model to `path` (framed `FTWO` format).
        fn save(&self, path: String) -> PyResult<()> {
            self.model
                .save_to(Path::new(&path))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
        }
    }

    /// Train a Two-Tower model from a long-format interactions file plus
    /// optional user/item feature files.
    ///
    /// Args:
    ///     interactions_path (str): Parquet/CSV with `user_id`, `item_id`,
    ///         optional `value` (rows with `value <= 0` are skipped).
    ///     user_features_path (str, optional): Long-format user features
    ///         (`user_id`, `feature_name`, `value`). One-hot-only feature
    ///         names route through the categorical embedding; others
    ///         become a dense vector.
    ///     item_features_path (str, optional): Same shape, with `item_id`.
    ///     embedding_dim, temperature, learning_rate, epochs, batch_size,
    ///         id_dropout, seed: training hyperparameters.
    ///
    /// Returns:
    ///     TwoTowerModel
    #[pyfunction]
    #[pyo3(signature = (
        interactions_path,
        user_features_path = None,
        item_features_path = None,
        embedding_dim = 32,
        temperature = 0.05,
        learning_rate = 0.01,
        epochs = 50,
        batch_size = 256,
        id_dropout = 0.1,
        seed = 0,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn build_and_train_two_tower(
        interactions_path: String,
        user_features_path: Option<String>,
        item_features_path: Option<String>,
        embedding_dim: usize,
        temperature: f64,
        learning_rate: f64,
        epochs: usize,
        batch_size: usize,
        id_dropout: f64,
        seed: u64,
    ) -> PyResult<TwoTowerModel> {
        let data = load_triples(&interactions_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        let user_ft = match user_features_path {
            Some(p) => load_features(&p, "user_id", &data.user_to_idx, data.num_users())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?,
            None => FeatureTable::empty(data.num_users()),
        };
        let item_ft = match item_features_path {
            Some(p) => load_features(&p, "item_id", &data.item_to_idx, data.num_items())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?,
            None => FeatureTable::empty(data.num_items()),
        };

        let params = TrainParams {
            embedding_dim,
            temperature,
            learning_rate,
            epochs,
            batch_size,
            id_dropout,
            seed,
        };

        let trained = train(&data, &user_ft, &item_ft, params)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let report = trained.validate();
        if !report.passed {
            let msg = format!(
                "Two-Tower model validation failed:\n{}",
                report.messages.join("\n")
            );
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg));
        }

        let num_items = RecModel::num_items(&trained);
        let num_users = data.num_users();
        Ok(TwoTowerModel {
            model: trained,
            num_items,
            num_users,
        })
    }

    /// Load a Two-Tower model previously written by `TwoTowerModel.save`.
    #[pyfunction]
    pub fn load_two_tower_model(path: String) -> PyResult<TwoTowerModel> {
        let model = TrainedTwoTower::load_from(Path::new(&path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let report = model.validate();
        if !report.passed {
            let msg = format!(
                "Loaded Two-Tower model failed validation:\n{}",
                report.messages.join("\n")
            );
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg));
        }
        let num_items = RecModel::num_items(&model);
        let num_users = model.item_mapping().idx_to_user.len();
        Ok(TwoTowerModel {
            model,
            num_items,
            num_users,
        })
    }
}

/// Defines the Python module.
/// This function is called when Python runs `import kzn_recsys._native`.
#[pymodule]
fn _native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_and_train, m)?)?;
    m.add_function(wrap_pyfunction!(load_model, m)?)?;
    m.add_function(wrap_pyfunction!(validate_data, m)?)?;
    m.add_function(wrap_pyfunction!(precision_at_k, m)?)?;
    m.add_function(wrap_pyfunction!(recall_at_k, m)?)?;
    m.add_function(wrap_pyfunction!(ndcg_at_k, m)?)?;
    m.add_function(wrap_pyfunction!(mean_average_precision, m)?)?;
    m.add_function(wrap_pyfunction!(coverage, m)?)?;
    m.add_function(wrap_pyfunction!(hit_rate_at_k, m)?)?;
    m.add_function(wrap_pyfunction!(random_split, m)?)?;
    m.add_function(wrap_pyfunction!(temporal_split, m)?)?;
    m.add_function(wrap_pyfunction!(leave_k_out_split, m)?)?;
    m.add_function(wrap_pyfunction!(grid_search_py, m)?)?;
    m.add_function(wrap_pyfunction!(random_search_py, m)?)?;
    m.add_function(wrap_pyfunction!(grid_search_ease, m)?)?;
    m.add_function(wrap_pyfunction!(random_search_ease, m)?)?;
    m.add_function(wrap_pyfunction!(grid_search_sasrec, m)?)?;
    m.add_function(wrap_pyfunction!(random_search_sasrec, m)?)?;
    m.add_function(wrap_pyfunction!(grid_search_two_tower, m)?)?;
    m.add_function(wrap_pyfunction!(random_search_two_tower, m)?)?;
    m.add_class::<FeaseModel>()?;
    m.add_class::<ModelRegistry>()?;

    #[cfg(feature = "ml-models")]
    {
        m.add_function(wrap_pyfunction!(sasrec_py::build_and_train_sasrec, m)?)?;
        m.add_function(wrap_pyfunction!(sasrec_py::load_sasrec_model, m)?)?;
        m.add_class::<sasrec_py::SASRecModel>()?;
        m.add_function(wrap_pyfunction!(
            two_tower_py::build_and_train_two_tower,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(two_tower_py::load_two_tower_model, m)?)?;
        m.add_class::<two_tower_py::TwoTowerModel>()?;
    }

    Ok(())
}
