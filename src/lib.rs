use pyo3::prelude::*;
use std::collections::HashMap;

mod data_pipeline;
mod model;
mod schemas;

use data_pipeline::{Mappings, build_matrices};
use model::{RustFeaseModel, create_sparse_matrix};
use nalgebra::DVector;

/// A trained FEASE model, ready for predictions.
///
/// This class is created by the `build_and_train` function.
/// It holds the trained model weights and the necessary ID/feature
/// mappings to make predictions.
#[pyclass]
struct FeaseModel {
    /// The internal Rust model containing the S-matrix
    model: RustFeaseModel,
    /// Mappings from string IDs/features to integer indices
    mappings: Mappings,
    /// Mapping from integer item_idx back to string item_guid
    idx_to_item: HashMap<usize, String>,
    /// The beta parameter used at training time
    beta: f64,
}

#[pymethods]
impl FeaseModel {
    /// Predicts item scores for a user.
    ///
    /// Args:
    ///     user_interactions (dict[str, float]): A dictionary mapping
    ///         `view_media_id` to its score (e.g., log_watch_time).
    ///         Use an empty dict `{}` for a cold-start user.
    ///     user_features (dict[str, float]): A dictionary mapping
    ///         feature names (e.g., "device_Mobile", "tenure_31-90d") to 1.0.
    ///     top_k (int): The number of recommendations to return.
    ///
    /// Returns:
    ///     list[tuple[str, float]]: A list of (item_guid, score) tuples,
    ///     sorted from highest score to lowest.
    #[pyo3(signature = (user_interactions, user_features, top_k=100))]
    fn predict(
        &self,
        user_interactions: HashMap<String, f64>,
        user_features: HashMap<String, f64>,
        top_k: usize,
    ) -> PyResult<Vec<(String, f64)>> {
        // 1. Convert Python HashMaps to Rust sparse vectors (1-row CsMat)
        let mut x_triplets = Vec::new();
        for (item_guid, val) in user_interactions {
            if let Some(item_idx) = self.mappings.item_mapping.get(&item_guid) {
                x_triplets.push((0, *item_idx, val));
            }
        }
        let x_sparse = create_sparse_matrix(1, self.model.num_items, x_triplets);

        let mut u_triplets = Vec::new();
        for (feat_name, val) in user_features {
            if let Some(feat_idx) = self.mappings.user_feature_mapping.get(&feat_name) {
                u_triplets.push((0, *feat_idx, val));
            }
        }
        let u_sparse = create_sparse_matrix(1, self.model.num_user_features, u_triplets);

        // 2. Call the internal Rust model's predict function
        let scores_vec: DVector<f64> = self.model.predict(&x_sparse, &u_sparse, self.beta);

        // 3. Map results back to item GUIs and sort
        let mut results: Vec<(String, f64)> = scores_vec
            .iter()
            .enumerate()
            .filter_map(|(item_idx, &score)| {
                // Map the index back to the original string ID
                self.idx_to_item
                    .get(&item_idx)
                    .map(|item_guid| (item_guid.clone(), score))
            })
            .collect();

        // Sort by score, descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 4. Take top_k and return
        results.truncate(top_k);
        Ok(results)
    }

    /// Gets the number of items the model was trained on.
    #[getter]
    fn num_items(&self) -> PyResult<usize> {
        Ok(self.model.num_items)
    }

    /// Gets the number of user features the model was trained on.
    #[getter]
    fn num_user_features(&self) -> PyResult<usize> {
        Ok(self.model.num_user_features)
    }
}

/// Builds the feature matrices from Parquet files, trains the FEASE model,
/// and returns a trained, ready-to-use model object.
///
/// Args:
///     engagement_path (str): Path to the `Engagement` Parquet file.
///     metadata_path (str): Path to the `Content Metadata` Parquet file.
///     alpha (float): Weight for item features (α).
///     beta (float): Weight for user features (β).
///     lambda_ (float): L2 regularization strength (λ). Note the underscore
///         to avoid conflict with Python's `lambda` keyword.
///
/// Returns:
///     FeaseModel: A trained, ready-to-use model.
#[pyfunction]
#[pyo3(signature = (engagement_path, metadata_path, alpha = 1.0, beta = 1.0, lambda_ = 100.0))]
fn build_and_train(
    engagement_path: String,
    metadata_path: String,
    alpha: f64,
    beta: f64,
    lambda_: f64,
) -> PyResult<FeaseModel> {
    // 1. Build the matrices from the data pipeline
    println!("Building sparse matrices from data...");
    let (x_mat, u_mat, t_mat, mappings) = build_matrices(&engagement_path, &metadata_path)
        .map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to build matrices: {}", e))
        })?;

    let num_items = x_mat.cols();
    let num_user_features = u_mat.cols();

    // 2. Train the model
    println!("Training model...");
    let mut model = RustFeaseModel::new(num_items, num_user_features);
    model.train(&x_mat, &u_mat, &t_mat, alpha, beta, lambda_);
    println!("Training complete.");

    // 3. Create the reverse item mapping for predictions
    let mut idx_to_item = HashMap::new();
    for (guid, &idx) in &mappings.item_mapping {
        idx_to_item.insert(idx, guid.clone());
    }

    // 4. Wrap and return the Python object
    Ok(FeaseModel {
        model,
        mappings,
        idx_to_item,
        beta,
    })
}

/// A Python module implementing the FEASE recommender in Rust.
#[pymodule]
fn rust_fease_recommender(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_and_train, m)?)?;
    m.add_class::<FeaseModel>()?;
    Ok(())
}
