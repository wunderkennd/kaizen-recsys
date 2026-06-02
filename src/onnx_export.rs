//! Pure-Rust helper for the ONNX export seam.
//!
//! Extracts the EASE `S_items` sub-block (the first `M` rows of `S`) in the
//! exact byte layout the Python ONNX authoring layer expects: row-major,
//! little-endian `f64`. nalgebra stores `s_matrix` column-major, so we walk it
//! row-by-row; the Python side then does
//! `np.frombuffer(bytes, dtype="<f8").reshape(M, M + K)` with no transpose.

use crate::model::RustFeaseModel;

// allow(dead_code): the caller, FeaseModel::export_payload, lands in the next task (Task 2)
#[allow(dead_code)]
/// Returns `(bytes, rows, cols)` where `bytes` is the row-major little-endian
/// `f64` encoding of `S[0..M, 0..M+K]`, `rows == num_items` (M) and
/// `cols == num_items + num_user_features` (M + K).
pub fn s_items_row_major_le_bytes(model: &RustFeaseModel) -> (Vec<u8>, usize, usize) {
    let rows = model.num_items;
    let cols = model.num_items + model.num_user_features;
    let mut bytes = Vec::with_capacity(rows * cols * 8);
    for r in 0..rows {
        for c in 0..cols {
            bytes.extend_from_slice(&model.s_matrix[(r, c)].to_le_bytes());
        }
    }
    (bytes, rows, cols)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_pipeline::Mappings;
    use ahash::AHashMap;
    use nalgebra::DMatrix;

    fn dummy_mappings() -> Mappings {
        Mappings {
            user_to_idx: AHashMap::default(),
            idx_to_user: Default::default(),
            item_to_idx: AHashMap::default(),
            idx_to_item: Default::default(),
            user_feature_to_idx: AHashMap::default(),
            idx_to_user_feature: Default::default(),
            item_feature_to_idx: AHashMap::default(),
            idx_to_item_feature: Default::default(),
        }
    }

    #[test]
    fn s_items_bytes_are_row_major_le_and_subset() {
        // M = 2 items, K = 1 user feature → S is 3x3, S_items is the first 2 rows (2x3).
        let m = 2;
        let k = 1;
        let total = m + k;
        let mut s = DMatrix::<f64>::zeros(total, total);
        // Distinct values so row-major order is observable.
        for r in 0..total {
            for c in 0..total {
                s[(r, c)] = (r * 10 + c) as f64;
            }
        }
        let model = RustFeaseModel {
            s_matrix: s,
            num_items: m,
            num_user_features: k,
            num_item_features: 0,
            alpha: 1.0,
            beta: 1.0,
            lambda_: 10.0,
            meta_weight: 0.0,
            mappings: dummy_mappings(),
            weighting_config: None,
        };

        let (bytes, rows, cols) = s_items_row_major_le_bytes(&model);
        assert_eq!((rows, cols), (2, 3));
        assert_eq!(bytes.len(), 2 * 3 * 8);

        // Decode and check row-major order: [00,01,02, 10,11,12]
        let decoded: Vec<f64> = bytes
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(decoded, vec![0.0, 1.0, 2.0, 10.0, 11.0, 12.0]);
    }
}
