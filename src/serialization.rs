//! Model serialization module for saving and loading trained FEASE models.
//!
//! Uses `serde` + `bincode` for efficient binary serialization of the S matrix
//! and all model metadata. This allows trained models to be persisted to disk
//! and reloaded without retraining.

use crate::data_pipeline::Mappings;
use crate::model::RustFeaseModel;
use anyhow::{Context, Result};
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Current version of the serialization format.
/// Increment when making breaking changes to `SerializedModel`.
const FORMAT_VERSION: u32 = 1;

/// Magic bytes to identify FEASE model files.
const MAGIC: &[u8; 4] = b"FEAS";

/// Serializable representation of a trained FEASE model.
///
/// This is a flat struct that captures everything needed to reconstruct
/// a `RustFeaseModel` without retraining. The S matrix is stored as a
/// flat Vec<f64> in column-major order (nalgebra's native layout).
#[derive(Serialize, Deserialize)]
struct SerializedModel {
    /// Format version for forward compatibility.
    version: u32,
    /// S matrix dimensions (rows == cols == num_items + num_user_features).
    s_nrows: usize,
    s_ncols: usize,
    /// S matrix data in column-major order.
    s_data: Vec<f64>,
    /// Model hyperparameters.
    num_items: usize,
    num_user_features: usize,
    num_item_features: usize,
    alpha: f64,
    beta: f64,
    lambda_: f64,
    meta_weight: f64,
    /// Mappings (string ID <-> index).
    user_to_idx: Vec<(String, usize)>,
    idx_to_user: Vec<String>,
    item_to_idx: Vec<(String, usize)>,
    idx_to_item: Vec<String>,
    user_feature_to_idx: Vec<(String, usize)>,
    idx_to_user_feature: Vec<String>,
    item_feature_to_idx: Vec<(String, usize)>,
    idx_to_item_feature: Vec<String>,
}

impl SerializedModel {
    fn from_model(model: &RustFeaseModel) -> Self {
        Self {
            version: FORMAT_VERSION,
            s_nrows: model.s_matrix.nrows(),
            s_ncols: model.s_matrix.ncols(),
            s_data: model.s_matrix.as_slice().to_vec(),
            num_items: model.num_items,
            num_user_features: model.num_user_features,
            num_item_features: model.num_item_features,
            alpha: model.alpha,
            beta: model.beta,
            lambda_: model.lambda_,
            meta_weight: model.meta_weight,
            user_to_idx: model.mappings.user_to_idx.iter().map(|(k, v)| (k.clone(), *v)).collect(),
            idx_to_user: model.mappings.idx_to_user.clone(),
            item_to_idx: model.mappings.item_to_idx.iter().map(|(k, v)| (k.clone(), *v)).collect(),
            idx_to_item: model.mappings.idx_to_item.clone(),
            user_feature_to_idx: model
                .mappings
                .user_feature_to_idx
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
            idx_to_user_feature: model.mappings.idx_to_user_feature.clone(),
            item_feature_to_idx: model
                .mappings
                .item_feature_to_idx
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
            idx_to_item_feature: model.mappings.idx_to_item_feature.clone(),
        }
    }

    fn into_model(self) -> Result<RustFeaseModel> {
        if self.version != FORMAT_VERSION {
            anyhow::bail!(
                "Unsupported model format version: {} (expected {})",
                self.version,
                FORMAT_VERSION
            );
        }

        let expected_len = self.s_nrows * self.s_ncols;
        if self.s_data.len() != expected_len {
            anyhow::bail!(
                "S matrix data length {} doesn't match dimensions {}x{} (expected {})",
                self.s_data.len(), self.s_nrows, self.s_ncols, expected_len
            );
        }
        let s_matrix = DMatrix::from_vec(self.s_nrows, self.s_ncols, self.s_data);

        let mappings = Mappings {
            user_to_idx: self.user_to_idx.into_iter().collect(),
            idx_to_user: self.idx_to_user,
            item_to_idx: self.item_to_idx.into_iter().collect(),
            idx_to_item: self.idx_to_item,
            user_feature_to_idx: self.user_feature_to_idx.into_iter().collect(),
            idx_to_user_feature: self.idx_to_user_feature,
            item_feature_to_idx: self.item_feature_to_idx.into_iter().collect(),
            idx_to_item_feature: self.idx_to_item_feature,
        };

        Ok(RustFeaseModel {
            s_matrix,
            num_items: self.num_items,
            num_user_features: self.num_user_features,
            num_item_features: self.num_item_features,
            alpha: self.alpha,
            beta: self.beta,
            lambda_: self.lambda_,
            meta_weight: self.meta_weight,
            mappings,
        })
    }
}

/// Saves a trained model to a binary file.
///
/// The file format is: `FEAS` magic (4 bytes) + bincode-encoded `SerializedModel`.
pub fn save_model(model: &RustFeaseModel, path: &Path) -> Result<()> {
    let serialized = SerializedModel::from_model(model);
    let encoded = bincode::serialize(&serialized)
        .context("Failed to serialize model")?;

    let mut data = Vec::with_capacity(MAGIC.len() + encoded.len());
    data.extend_from_slice(MAGIC);
    data.extend(encoded);

    fs::write(path, &data)
        .with_context(|| format!("Failed to write model to {}", path.display()))?;

    log::info!(
        "Model saved to {} ({:.2} MB)",
        path.display(),
        data.len() as f64 / (1024.0 * 1024.0)
    );
    Ok(())
}

/// Loads a trained model from a binary file.
///
/// Verifies the magic bytes and format version before deserializing.
pub fn load_model(path: &Path) -> Result<RustFeaseModel> {
    let data = fs::read(path)
        .with_context(|| format!("Failed to read model from {}", path.display()))?;

    if data.len() < MAGIC.len() {
        anyhow::bail!("File too small to be a valid FEASE model: {}", path.display());
    }

    if &data[..MAGIC.len()] != MAGIC {
        anyhow::bail!(
            "Invalid magic bytes in {}. Expected FEAS header.",
            path.display()
        );
    }

    let serialized: SerializedModel = bincode::deserialize(&data[MAGIC.len()..])
        .context("Failed to deserialize model data")?;

    let model = serialized.into_model()?;

    log::info!(
        "Model loaded from {} (S matrix: {}x{}, {} items, {} user features)",
        path.display(),
        model.s_matrix.nrows(),
        model.s_matrix.ncols(),
        model.num_items,
        model.num_user_features,
    );

    Ok(model)
}

// --- Unit Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use ahash::AHashMap;

    fn make_test_model() -> RustFeaseModel {
        let n_items = 3;
        let n_user_features = 2;
        let total_dim = n_items + n_user_features;

        let mut s = DMatrix::<f64>::zeros(total_dim, total_dim);
        s[(0, 1)] = 0.5;
        s[(1, 0)] = 0.5;
        s[(0, 2)] = 0.3;
        s[(2, 0)] = 0.3;

        let mut item_to_idx = AHashMap::new();
        item_to_idx.insert("item_a".to_string(), 0);
        item_to_idx.insert("item_b".to_string(), 1);
        item_to_idx.insert("item_c".to_string(), 2);

        let mut user_feature_to_idx = AHashMap::new();
        user_feature_to_idx.insert("feat_x".to_string(), 0);
        user_feature_to_idx.insert("feat_y".to_string(), 1);

        RustFeaseModel {
            s_matrix: s,
            num_items: n_items,
            num_user_features: n_user_features,
            num_item_features: 0,
            alpha: 1.0,
            beta: 0.5,
            lambda_: 100.0,
            meta_weight: 0.0,
            mappings: Mappings {
                user_to_idx: AHashMap::new(),
                idx_to_user: vec![],
                item_to_idx,
                idx_to_item: vec![
                    "item_a".to_string(),
                    "item_b".to_string(),
                    "item_c".to_string(),
                ],
                user_feature_to_idx,
                idx_to_user_feature: vec!["feat_x".to_string(), "feat_y".to_string()],
                item_feature_to_idx: AHashMap::new(),
                idx_to_item_feature: vec![],
            },
        }
    }

    #[test]
    fn test_save_load_roundtrip() {
        let model = make_test_model();
        let path = Path::new("./test_model_roundtrip.fease");

        // Save
        save_model(&model, path).expect("Failed to save model");
        assert!(path.exists());

        // Load
        let loaded = load_model(path).expect("Failed to load model");

        // Verify S matrix
        assert_eq!(loaded.s_matrix.nrows(), model.s_matrix.nrows());
        assert_eq!(loaded.s_matrix.ncols(), model.s_matrix.ncols());
        for i in 0..model.s_matrix.nrows() {
            for j in 0..model.s_matrix.ncols() {
                assert!(
                    (loaded.s_matrix[(i, j)] - model.s_matrix[(i, j)]).abs() < 1e-12,
                    "S[{},{}] mismatch: {} vs {}",
                    i,
                    j,
                    loaded.s_matrix[(i, j)],
                    model.s_matrix[(i, j)]
                );
            }
        }

        // Verify hyperparameters
        assert_eq!(loaded.num_items, model.num_items);
        assert_eq!(loaded.num_user_features, model.num_user_features);
        assert_eq!(loaded.num_item_features, model.num_item_features);
        assert!((loaded.alpha - model.alpha).abs() < 1e-12);
        assert!((loaded.beta - model.beta).abs() < 1e-12);
        assert!((loaded.lambda_ - model.lambda_).abs() < 1e-12);
        assert!((loaded.meta_weight - model.meta_weight).abs() < 1e-12);

        // Verify mappings
        assert_eq!(
            loaded.mappings.item_to_idx.get("item_a"),
            model.mappings.item_to_idx.get("item_a")
        );
        assert_eq!(
            loaded.mappings.item_to_idx.get("item_b"),
            model.mappings.item_to_idx.get("item_b")
        );
        assert_eq!(loaded.mappings.idx_to_item, model.mappings.idx_to_item);
        assert_eq!(
            loaded.mappings.idx_to_user_feature,
            model.mappings.idx_to_user_feature
        );

        // Cleanup
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_load_invalid_magic() {
        let path = Path::new("./test_bad_magic.fease");
        fs::write(path, b"BADDATA").unwrap();

        let result = load_model(path);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid magic bytes"));

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_load_too_small() {
        let path = Path::new("./test_too_small.fease");
        fs::write(path, b"FE").unwrap();

        let result = load_model(path);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("too small"));

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_predictions_after_roundtrip() {
        let model = make_test_model();
        let path = Path::new("./test_predict_roundtrip.fease");

        save_model(&model, path).expect("Failed to save");
        let loaded = load_model(path).expect("Failed to load");

        // Predict with both and compare
        let interactions = vec![(0, 1.0)];
        let features = vec![(0, 1.0)];

        let scores_original = model.predict(&interactions, &features, model.beta);
        let scores_loaded = loaded.predict(&interactions, &features, loaded.beta);

        assert_eq!(scores_original.len(), scores_loaded.len());
        for (a, b) in scores_original.iter().zip(scores_loaded.iter()) {
            assert!(
                (a - b).abs() < 1e-12,
                "Prediction mismatch after roundtrip: {} vs {}",
                a,
                b
            );
        }

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_load_corrupted_dimensions() {
        // Create a valid model file, then corrupt the s_data length
        let model = make_test_model();
        let path = Path::new("./test_corrupted_dims.fease");
        save_model(&model, path).expect("Failed to save");

        // Read the file, corrupt it by truncating some bytes from the end
        let mut data = fs::read(path).unwrap();
        data.truncate(data.len() - 50); // Remove some data bytes
        fs::write(path, &data).unwrap();

        let result = load_model(path);
        assert!(result.is_err(), "Should fail on corrupted model file");

        std::fs::remove_file(path).ok();
    }
}
