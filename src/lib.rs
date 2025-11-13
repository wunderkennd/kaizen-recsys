//! This file is the main entrypoint for the Python library.
//! It uses PyO3 to define the Python-callable functions and the `FeaseModel` class.
//! This is the "bridge" between Python and Rust.

use ahash::AHashMap;
use anyhow::Result;
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
    /// `#[pyo3(get)]` requires `RustFeaseModel` to be `Clone`.
    #[pyo3(get)]
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
    #[pyfn(signature = (interactions, features, top_k=100))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        interactions: &Bound<'_, PyDict>,
        features: &Bound<'_, PyDict>,
        top_k: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        // --- 1. Convert Python inputs to Rust vectors ---
        // We use `with_capacity` for a small optimization.
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
        // This is the high-performance part.
        let scores: Vec<f64> = self
            .model
            .predict(&user_interactions, &user_features, self.model.beta);

        // --- 3. Process and sort results ---
        // This part is a bit slow (O(M log M)) but fine for a single user.
        // For batch prediction, this should be optimized.
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
        // We use `partial_cmp` because f64 is not `Ord`.
        results_with_idx.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal) // FIX: Corrected typo `std.cmp`
        });

        // --- 4. Convert top-K results back to Python types ---
        // We use `new_bound` for `pyo3 = "0.21.0"`
        let py_results = PyList::empty_bound(py);
        for (idx, score) in results_with_idx.into_iter().take(top_k) {
            // Map item_idx back to item_guid
            if let Some(guid) = self.model.mappings.idx_to_item.get(idx) {
                let py_guid = PyString::new_bound(py, guid);
                let py_score = PyFloat::new_bound(py, score);
                py_results.append((py_guid, py_score))?;
            }
        }

        Ok(py_results)
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
#[pyfunction(signature = (
    interactions_path,
    user_features_path,
    item_features_path,
    alpha = 1.0,
    beta = 1.0,
    lambda_ = 100.0
))]
fn build_and_train(
    interactions_path: String,
    user_features_path: String,
    item_features_path: String,
    alpha: f64,
    beta: f64,
    lambda_: f64,
) -> PyResult<FeaseModel> {
    println!("--- [Rust] Starting Model Training ---");
    let start_time = Instant::now();

    // --- 1. Build Matrices ---
    // This calls the data pipeline, which uses Polars to read files
    // and `sprs` to build the matrices.
    let (x_mat, u_mat, t_mat, mappings) =
        match data_pipeline::build_matrices(
            &interactions_path,
            &user_features_path,
            &item_features_path,
        ) {
            Ok(data) => data,
            Err(e) => {
                // Convert the Rust `anyhow::Error` into a Python `PyErr`
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

    // --- 3. Wrap in Python Class ---
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
fn rust_fease_recommender(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_and_train, m)?)?;
    m.add_class::<FeaseModel>()?;
    Ok(())
}