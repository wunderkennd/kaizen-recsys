//! This file is the main entrypoint for the Python library.
//! It uses PyO3 to define the Python-callable functions and the `FeaseModel` class.
//! This is the "bridge" between Python and Rust.

use model::RustFeaseModel; // The internal Rust struct
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyList, PyString};
use std::time::Instant;

mod data_pipeline;
mod model;
mod schemas;

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
        let mut user_interactions: Vec<(usize, f64)> =
            Vec::with_capacity(interactions.len());
        let mut user_features: Vec<(usize, f64)> =
            Vec::with_capacity(features.len());

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
            if let Some(&feature_idx) =
                self.model.mappings.user_feature_to_idx.get(feature_name)
            {
                user_features.push((feature_idx, value));
            }
        }

        // --- 2. Call the internal Rust prediction function ---
        let scores: Vec<f64> = self
            .model
            .predict(&user_interactions, &user_features, self.model.beta);

        // --- 3. Process and sort results ---
        let mut results_with_idx: Vec<(usize, f64)> = scores
            .into_iter()
            .enumerate()
            // Don't recommend items the user has already interacted with
            .filter(|(idx, _score)| {
                !user_interactions
                    .iter()
                    .any(|(interacted_idx, _)| *interacted_idx == *idx)
            })
            .collect();

        // Sort by score, descending.
        results_with_idx.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

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
            None => return Ok(py_results), // Unknown item → empty list
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
    fn validate<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(bool, Bound<'py, PyList>)> {
        let report = self.model.validate();
        let py_messages = PyList::new(py, &report.messages)?;
        Ok((report.passed, py_messages))
    }
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
    meta_weight = 0.0
))]
fn build_and_train(
    interactions_path: String,
    user_features_path: String,
    item_features_path: String,
    alpha: f64,
    beta: f64,
    lambda_: f64,
    meta_weight: f64,
) -> PyResult<FeaseModel> {
    println!("--- [Rust] Starting Model Training ---");
    let start_time = Instant::now();

    // --- 1. Build Matrices ---
    let (x_mat, u_mat, t_mat, mappings) =
        match data_pipeline::build_matrices(
            &interactions_path,
            &user_features_path,
            &item_features_path,
        ) {
            Ok(data) => data,
            Err(e) => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    e.to_string(),
                ));
            }
        };
    println!(
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
    println!(
        "[Rust] Core model training complete in {:.2}s",
        train_start.elapsed().as_secs_f32()
    );

    // --- 3. Validate Model ---
    let report = rust_model.validate();
    if !report.passed {
        let msg = format!(
            "Model validation failed:\n{}",
            report.messages.join("\n")
        );
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg));
    }
    for msg in &report.messages {
        println!("[Rust] Validation: {}", msg);
    }

    // --- 4. Wrap in Python Class ---
    let model = FeaseModel {
        model: rust_model,
        num_items,
        num_user_features,
        num_item_features,
    };

    println!(
        "[Rust] Total training time: {:.2}s",
        start_time.elapsed().as_secs_f32()
    );

    Ok(model)
}

/// Defines the Python module.
/// This function is called when Python runs `import rust_fease_recommender`.
#[pymodule]
fn rust_fease_recommender(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_and_train, m)?)?;
    m.add_class::<FeaseModel>()?;
    Ok(())
}
