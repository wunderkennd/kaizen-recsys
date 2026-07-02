//! This file contains the core Rust logic for the FEASE model.
//! It uses `nalgebra` for dense linear algebra (inversion, etc.)
//! and `sprs` for sparse matrix operations (multiplication, storage).
//!
//! This file is "pure Rust" and has no Python (PyO3) dependencies.
//! It's called by `src/lib.rs` to perform the actual math.

use crate::data_pipeline::Mappings;
use crate::weighting::WeightingConfig;
use anyhow::{Result, anyhow};
use nalgebra::{DMatrix, DVector};
use sprs::{CsMat, TriMat};

/// Validation results returned by `validate()`.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Whether the model passed all checks.
    pub passed: bool,
    /// Human-readable messages for each check (errors only when `passed` is false).
    pub messages: Vec<String>,
}

/// The internal Rust struct that holds the trained model and mappings.
///
/// We add `#[derive(Clone, Debug)]` so this struct can be cloned, which is
/// required by `PyO3` to expose it as a `#[pyo3(get)]` property
/// on the Python `FeaseModel` class.
#[derive(Clone, Debug)]
pub struct RustFeaseModel {
    /// The learned weight matrix S.
    /// Dimensions: (items + user_features) x (items + user_features)
    pub s_matrix: DMatrix<f64>,
    pub num_items: usize,
    pub num_user_features: usize,
    #[allow(dead_code)]
    pub num_item_features: usize,
    pub alpha: f64,
    pub beta: f64,
    pub lambda_: f64,
    /// Optional metadata weight (analogous to Python's WeightedEASE `meta_weight`).
    /// When > 0, scales the contribution of metadata rows in the Gram matrix.
    /// A value of 1.0 weights metadata equally with interactions.
    pub meta_weight: f64,
    /// Mappings to convert between string IDs and numeric indices
    pub mappings: Mappings,
    pub weighting_config: Option<WeightingConfig>,
    /// Declarative raw-feature transformation for `predict_raw` (#71 theme A).
    /// Runtime-only until format V3 persists it (theme B); loading a V1/V2
    /// file leaves it `None`.
    pub transformation_schema: Option<crate::transform::FeatureTransformationSchema>,
}

impl RustFeaseModel {
    /// Creates a new, untrained model container.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_items: usize,
        num_user_features: usize,
        num_item_features: usize,
        alpha: f64,
        beta: f64,
        lambda_: f64,
        meta_weight: f64,
        mappings: Mappings,
    ) -> Self {
        let total_dim = num_items + num_user_features;
        // Initialize s_matrix as an empty matrix; it will be populated by train()
        RustFeaseModel {
            s_matrix: DMatrix::zeros(total_dim, total_dim),
            num_items,
            num_user_features,
            num_item_features,
            alpha,
            beta,
            lambda_,
            meta_weight,
            mappings,
            weighting_config: None,
            transformation_schema: None,
        }
    }

    /// Trains the FEASE model.
    ///
    /// This function computes the Gram matrix `G`, inverts it to get `P`,
    /// and then computes the final weight matrix `S` using the efficient
    /// closed-form formula from the EASE paper: `S_ij = -P_ij / P_jj`.
    ///
    /// When `meta_weight` > 0, a diagonal weight matrix W is applied to
    /// scale the metadata contributions (analogous to Python's WeightedEASE).
    /// Interaction rows get weight 1.0, metadata rows get weight `meta_weight`.
    pub fn train(
        &mut self,
        x_mat: &CsMat<f64>, // N x M (Users x Items)
        u_mat: &CsMat<f64>, // N x K (Users x UserFeatures)
        t_mat: &CsMat<f64>, // L x M (ItemFeatures x Items)
    ) -> Result<()> {
        log::info!("Starting FEASE model training...");
        let m = self.num_items;
        let k = self.num_user_features;
        let total_dim = m + k;

        log::info!("M (Items): {}, K (User Features): {}", m, k);
        log::info!(
            "Total dimension of Gram matrix: {}x{}",
            total_dim,
            total_dim
        );

        // ---
        // 1. Compute the Gram Matrix G = Z^T * W * Z in blocks (using sprs)
        // ---
        // Z = [ X  | βU ]   (N x M) | (N x K)
        //     [ αT |  0  ]   (L x M) | (L x K)
        //
        // Without W (meta_weight == 0 or 1):
        //   G_11 = X^T*X + α² * T^T*T
        //   G_12 = β * X^T*U
        //   G_21 = β * U^T*X
        //   G_22 = β² * U^T*U
        //
        // With W (diagonal weight matrix, interaction rows = 1.0, metadata rows = w):
        //   G_11 = X^T*X + w*α² * T^T*T
        //   G_12 = β * X^T*U              (metadata rows have 0 in user-feature cols)
        //   G_21 = β * U^T*X              (same reason)
        //   G_22 = β² * U^T*U
        // ---

        let w = if self.meta_weight > 0.0 {
            self.meta_weight
        } else {
            1.0
        };

        log::info!("Calculating G_11 = X^T*X + {}*α²*T^T*T...", w);
        let xtx = sparse_transpose_self_multiply(x_mat); // M x M
        let ttt = sparse_transpose_self_multiply(t_mat); // M x M
        let g_11 = &xtx + &(&ttt * (w * self.alpha * self.alpha));

        log::info!("Calculating G_12 = β * X^T*U...");
        let x_t = x_mat.transpose_view();
        let g_12 = &(&x_t * u_mat) * self.beta; // (M x N) * (N x K) -> M x K

        log::info!("Calculating G_21 = β * U^T*X...");
        let u_t = u_mat.transpose_view();
        let g_21 = &(&u_t * x_mat) * self.beta; // (K x N) * (N x M) -> K x M

        log::info!("Calculating G_22 = β² * U^T*U...");
        let utu = sparse_transpose_self_multiply(u_mat); // K x K
        let g_22 = &utu * (self.beta * self.beta);

        // ---
        // 2. Assemble the dense Gram matrix G in nalgebra
        // ---
        log::info!("Assembling dense Gram matrix G...");
        let mut g = DMatrix::<f64>::zeros(total_dim, total_dim);

        // Convert from ndarray (returned by .to_dense()) to nalgebra::DMatrix
        // We use `from_row_slice` because `sprs`'s `.to_dense()` creates a
        // row-major `ndarray`, while `nalgebra::DMatrix` is column-major.
        let g_11_dense = g_11.to_dense();
        let g_11_nalgebra = DMatrix::from_row_slice(
            g_11_dense.nrows(),
            g_11_dense.ncols(),
            g_11_dense.as_slice().unwrap(),
        );

        let g_12_dense = g_12.to_dense();
        let g_12_nalgebra = DMatrix::from_row_slice(
            g_12_dense.nrows(),
            g_12_dense.ncols(),
            g_12_dense.as_slice().unwrap(),
        );

        let g_21_dense = g_21.to_dense();
        let g_21_nalgebra = DMatrix::from_row_slice(
            g_21_dense.nrows(),
            g_21_dense.ncols(),
            g_21_dense.as_slice().unwrap(),
        );

        let g_22_dense = g_22.to_dense();
        let g_22_nalgebra = DMatrix::from_row_slice(
            g_22_dense.nrows(),
            g_22_dense.ncols(),
            g_22_dense.as_slice().unwrap(),
        );

        set_block(&mut g, &g_11_nalgebra, 0, 0);
        set_block(&mut g, &g_12_nalgebra, 0, m);
        set_block(&mut g, &g_21_nalgebra, m, 0);
        set_block(&mut g, &g_22_nalgebra, m, m);

        // ---
        // 3. Compute P = (G + λI)^-1
        // ---
        log::info!("Calculating P = (G + λI)^-1...");
        let mut g_reg = g;
        // Add λ to the diagonal (G + λI)
        for i in 0..total_dim {
            g_reg[(i, i)] += self.lambda_;
        }

        let p = invert_gram(g_reg)
            .ok_or_else(|| anyhow!("Failed to invert Gram matrix G. It may be singular."))?;

        // ---
        // 4. Compute S using the efficient closed-form EASE formula
        //    S_ij = -P_ij / P_jj  for i != j
        //    S_jj = 0             (zero-diagonal constraint)
        //
        // This is equivalent to the Python formula: B = P / (-diag(P)); fill_diagonal(B, 0)
        // and replaces the previous multi-step: S_unconstrained = P*G, D, S = S_un - P*D
        // Savings: eliminates 2 full matrix multiplications and 1 allocation.
        // ---
        log::info!("Computing S matrix with efficient B-matrix formula...");
        let mut s = DMatrix::<f64>::zeros(total_dim, total_dim);
        for j in 0..total_dim {
            let p_jj = p[(j, j)];
            // Guard against division by zero (shouldn't happen with λ > 0)
            let inv_p_jj = if p_jj.abs() > 1e-12 { -1.0 / p_jj } else { 0.0 };
            for i in 0..total_dim {
                if i != j {
                    s[(i, j)] = p[(i, j)] * inv_p_jj;
                }
                // s[(j, j)] remains 0.0 (zero-diagonal constraint)
            }
        }
        self.s_matrix = s;

        log::info!("Training complete!");
        Ok(())
    }

    /// Prunes small entries from the S matrix, setting values with
    /// |value| < threshold to zero. This increases sparsity and can
    /// reduce noise from near-zero weights.
    pub fn prune_sparse(&mut self, threshold: f64) {
        let mut pruned = 0usize;
        for val in self.s_matrix.iter_mut() {
            if val.abs() < threshold {
                *val = 0.0;
                pruned += 1;
            }
        }
        let total = self.s_matrix.nrows() * self.s_matrix.ncols();
        log::info!(
            "Sparsity pruning: zeroed {}/{} entries (threshold={})",
            pruned,
            total,
            threshold
        );
    }

    /// Validates the trained model, checking for common issues.
    ///
    /// Returns a `ValidationReport` with pass/fail status and diagnostic messages.
    /// Checks:
    /// - S matrix dimensions match expected (M+K)²
    /// - Diagonal entries are near-zero
    /// - No NaN or Inf values in S
    /// - S matrix is not all zeros (model actually learned something)
    pub fn validate(&self) -> ValidationReport {
        let mut messages = Vec::new();
        let mut passed = true;
        let total_dim = self.num_items + self.num_user_features;

        // Check 1: Dimensions
        if self.s_matrix.nrows() != total_dim || self.s_matrix.ncols() != total_dim {
            passed = false;
            messages.push(format!(
                "FAIL: S matrix dimensions {}x{} don't match expected {}x{}",
                self.s_matrix.nrows(),
                self.s_matrix.ncols(),
                total_dim,
                total_dim
            ));
        } else {
            messages.push(format!(
                "OK: S matrix dimensions {}x{}",
                total_dim, total_dim
            ));
        }

        // Check 2: Zero diagonal
        let mut max_diag = 0.0_f64;
        for i in 0..self.s_matrix.nrows().min(self.s_matrix.ncols()) {
            let val = self.s_matrix[(i, i)].abs();
            if val > max_diag {
                max_diag = val;
            }
        }
        if max_diag > 1e-6 {
            passed = false;
            messages.push(format!(
                "FAIL: Diagonal not near-zero (max |diag| = {:.2e})",
                max_diag
            ));
        } else {
            messages.push(format!(
                "OK: Diagonal near-zero (max |diag| = {:.2e})",
                max_diag
            ));
        }

        // Check 3: NaN/Inf
        let mut nan_count = 0usize;
        let mut inf_count = 0usize;
        for val in self.s_matrix.iter() {
            if val.is_nan() {
                nan_count += 1;
            }
            if val.is_infinite() {
                inf_count += 1;
            }
        }
        if nan_count > 0 || inf_count > 0 {
            passed = false;
            messages.push(format!(
                "FAIL: S matrix contains {} NaN and {} Inf values",
                nan_count, inf_count
            ));
        } else {
            messages.push("OK: No NaN or Inf values in S matrix".to_string());
        }

        // Check 4: Non-trivial (not all zeros)
        let max_abs = self
            .s_matrix
            .iter()
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        if max_abs < 1e-12 {
            passed = false;
            messages.push(
                "FAIL: S matrix is effectively all zeros — model may not have learned".to_string(),
            );
        } else {
            messages.push(format!("OK: S matrix max |value| = {:.4e}", max_abs));
        }

        ValidationReport { passed, messages }
    }

    /// Predicts scores for a given user.
    /// Takes slices of (index, value) tuples for interactions and features.
    pub fn predict(
        &self,
        user_interactions: &[(usize, f64)], // (item_idx, value)
        user_features: &[(usize, f64)],     // (feature_idx, value)
        beta: f64,
    ) -> Vec<f64> {
        let total_dim = self.num_items + self.num_user_features;

        // 1. Construct the user's input vector z = [x | β*u]
        let mut z_vec = vec![0.0; total_dim];

        // Fill item interactions
        for (item_idx, val) in user_interactions.iter() {
            if *item_idx < self.num_items {
                z_vec[*item_idx] = *val;
            }
        }

        // Fill user features
        for (feat_idx, val) in user_features.iter() {
            if *feat_idx < self.num_user_features {
                z_vec[self.num_items + *feat_idx] = *val * beta;
            }
        }

        // 2. Convert to nalgebra DVector
        let z = DVector::from_vec(z_vec);

        // 3. Predict scores: p = S * z
        // Row i of S dotted with z gives item i's score. S is not symmetric in
        // general (S[i,j] = -P[i,j]/P[j,j]); we use S directly, so that's fine.
        let p_full = &self.s_matrix * &z;

        // 4. We only want the item scores, which are the first M entries
        let p_items = p_full.rows(0, self.num_items);

        // Convert to a standard Vec<f64> to return to Python
        p_items.iter().cloned().collect()
    }

    /// Predicts scores for a user from string interaction keys and *raw*
    /// polymorphic user features (zero-drift online inference, #71 theme A).
    ///
    /// Raw features are routed through the embedded [`transformation_schema`]
    /// when present, so the exact train-time engineering runs at serve time.
    /// Without a schema, features are used as-is when their values are
    /// numeric (or numeric strings) — i.e. the caller already engineered them.
    /// Unknown item GUIDs and feature keys are skipped, matching the string-id
    /// tolerance of the serving layer.
    ///
    /// [`transformation_schema`]: RustFeaseModel::transformation_schema
    pub fn predict_raw(
        &self,
        interactions: &std::collections::HashMap<String, f64>,
        raw_user_features: &std::collections::HashMap<String, serde_json::Value>,
        beta: f64,
    ) -> Vec<f64> {
        // 1. Transform raw user features via the embedded schema if available.
        let transformed_features = if let Some(ref schema) = self.transformation_schema {
            crate::transform::transform_features(raw_user_features, schema)
        } else {
            let mut fm = std::collections::HashMap::new();
            for (k, v) in raw_user_features {
                if let Some(f) = v.as_f64() {
                    fm.insert(k.clone(), f);
                } else if let Some(f) = v.as_str().and_then(|s| s.parse::<f64>().ok()) {
                    fm.insert(k.clone(), f);
                }
            }
            fm
        };

        // 2. String keys → internal indices (unknowns skipped).
        let user_interactions: Vec<(usize, f64)> = interactions
            .iter()
            .filter_map(|(guid, &value)| {
                self.mappings.item_to_idx.get(guid).map(|&idx| (idx, value))
            })
            .collect();
        let user_features: Vec<(usize, f64)> = transformed_features
            .iter()
            .filter_map(|(name, &value)| {
                self.mappings
                    .user_feature_to_idx
                    .get(name)
                    .map(|&idx| (idx, value))
            })
            .collect();

        // 3. Delegate to the index-based prediction engine.
        self.predict(&user_interactions, &user_features, beta)
    }

    /// Predicts similar items for a given item (More-Like-This / MLT).
    ///
    /// Uses the item-item block of the S matrix to find the most similar items
    /// to a given source item. This is the same B matrix used for user recommendations,
    /// but queried with a single-item input vector.
    ///
    /// Returns a Vec of (item_index, score) pairs sorted by descending score,
    /// excluding the source item itself. Caller is responsible for mapping indices
    /// back to item GUIDs.
    pub fn predict_similar_items(&self, item_idx: usize, top_k: usize) -> Vec<(usize, f64)> {
        if item_idx >= self.num_items {
            return Vec::new();
        }

        // The item-item similarity is simply column `item_idx` of the S matrix
        // (restricted to the item rows 0..M).
        let mut scores: Vec<(usize, f64)> = (0..self.num_items)
            .filter(|&i| i != item_idx)
            .map(|i| (i, self.s_matrix[(i, item_idx)]))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scores.truncate(top_k);
        scores
    }
}

// --- Helper Functions ---

/// Helper function to compute B = A^T * A for a sparse matrix A
fn sparse_transpose_self_multiply(a: &CsMat<f64>) -> CsMat<f64> {
    let a_t = a.transpose_view();
    &a_t * a
}

/// Helper function to copy a dense block into a larger dense matrix.
fn set_block(mat: &mut DMatrix<f64>, block: &DMatrix<f64>, r_offset: usize, c_offset: usize) {
    mat.view_mut((r_offset, c_offset), (block.nrows(), block.ncols()))
        .copy_from(block);
}

/// Invert the regularized Gram matrix `(G + λI)`.
///
/// Default build: nalgebra's pure-Rust dense LU (`DMatrix::try_inverse`),
/// single-threaded, no system dependency. With the opt-in `fast-blas` Cargo
/// feature (ADR-0002 §"Decision" #2) this delegates to a system BLAS/LAPACK
/// via `nalgebra-lapack`'s LU (OpenBLAS on Linux/Windows, Apple Accelerate on
/// macOS), which is multi-threaded for large catalogs.
///
/// Both paths solve the same linear algebra and return `None` on a singular
/// matrix. Results differ only by BLAS-vs-pure-Rust floating-point ordering
/// (ADR-0002 §"Risks"); callers and tests must compare with tolerance, never
/// bit-exact.
fn invert_gram(g_reg: DMatrix<f64>) -> Option<DMatrix<f64>> {
    #[cfg(not(feature = "fast-blas"))]
    {
        g_reg.try_inverse()
    }
    #[cfg(feature = "fast-blas")]
    {
        nalgebra_lapack::LU::new(g_reg).inverse()
    }
}

/// Helper function to create a sparse matrix from (row, col, data) triplets.
/// Useful for loading data and creating test matrices.
#[allow(dead_code)]
pub fn create_sparse_matrix(
    rows: usize,
    cols: usize,
    triplets: Vec<(usize, usize, f64)>,
) -> CsMat<f64> {
    let mut mat = TriMat::new((rows, cols));
    for (r, c, v) in triplets {
        mat.add_triplet(r, c, v);
    }
    mat.to_csr()
}

// --- Unit Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;
    use sprs::CsMat;

    // Helper to create a sparse matrix for testing
    fn simple_csmat(rows: usize, cols: usize, triplets: Vec<(usize, usize, f64)>) -> CsMat<f64> {
        let mut mat = TriMat::new((rows, cols));
        for (r, c, v) in triplets {
            mat.add_triplet(r, c, v);
        }
        mat.to_csr()
    }

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

    #[test]
    fn test_fease_model_train() -> Result<()> {
        let n_users = 5;
        let n_items = 4;
        let n_user_features = 3;
        let n_item_features = 2;

        let alpha = 1.0;
        let beta = 1.0;
        let lambda = 100.0;

        // X: N x M (5x4)
        let x_mat = simple_csmat(
            n_users,
            n_items,
            vec![(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0), (3, 3, 1.0)],
        );
        // U: N x K (5x3)
        let u_mat = simple_csmat(
            n_users,
            n_user_features,
            vec![(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0), (4, 1, 1.0)],
        );
        // T: L x M (2x4)
        let t_mat = simple_csmat(
            n_item_features,
            n_items,
            vec![(0, 0, 1.0), (0, 1, 1.0), (1, 2, 1.0), (1, 3, 1.0)],
        );

        let mut model = RustFeaseModel::new(
            n_items,
            n_user_features,
            n_item_features,
            alpha,
            beta,
            lambda,
            0.0, // no meta_weight
            dummy_mappings(),
        );

        let train_result = model.train(&x_mat, &u_mat, &t_mat);
        assert!(train_result.is_ok());

        // Check dimensions of S
        let total_dim = n_items + n_user_features; // 4 + 3 = 7
        assert_eq!(model.s_matrix.nrows(), total_dim);
        assert_eq!(model.s_matrix.ncols(), total_dim);

        // Check that the diagonal is (close to) zero
        for i in 0..total_dim {
            assert!(model.s_matrix[(i, i)].abs() < 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_fease_model_train_with_meta_weight() -> Result<()> {
        let n_users = 5;
        let n_items = 4;
        let n_user_features = 3;
        let n_item_features = 2;

        let x_mat = simple_csmat(
            n_users,
            n_items,
            vec![(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0), (3, 3, 1.0)],
        );
        let u_mat = simple_csmat(
            n_users,
            n_user_features,
            vec![(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0), (4, 1, 1.0)],
        );
        let t_mat = simple_csmat(
            n_item_features,
            n_items,
            vec![(0, 0, 1.0), (0, 1, 1.0), (1, 2, 1.0), (1, 3, 1.0)],
        );

        // Train without meta_weight
        let mut model_no_w = RustFeaseModel::new(
            n_items,
            n_user_features,
            n_item_features,
            1.0,
            1.0,
            100.0,
            0.0, // no weight
            dummy_mappings(),
        );
        model_no_w.train(&x_mat, &u_mat, &t_mat)?;

        // Train with meta_weight = 2.0
        let mut model_w = RustFeaseModel::new(
            n_items,
            n_user_features,
            n_item_features,
            1.0,
            1.0,
            100.0,
            2.0, // meta_weight = 2.0
            dummy_mappings(),
        );
        model_w.train(&x_mat, &u_mat, &t_mat)?;

        // The S matrices should differ (meta_weight affects G_11)
        let diff = &model_w.s_matrix - &model_no_w.s_matrix;
        let max_diff = diff.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(max_diff > 1e-10, "Meta weight should change the S matrix");

        // Both should still have zero diagonals
        let total_dim = n_items + n_user_features;
        for i in 0..total_dim {
            assert!(model_w.s_matrix[(i, i)].abs() < 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_fease_model_predict_warm_and_cold() -> Result<()> {
        let n_items = 4;
        let n_user_features = 3;
        let total_dim = n_items + n_user_features;

        // Create a dummy, hand-calculated S matrix for predictable results
        let mut s_mat = DMatrix::<f64>::zeros(total_dim, total_dim);

        // S_11 (Item-Item): Item 0 and 1 are similar
        s_mat[(0, 1)] = 0.5;
        s_mat[(1, 0)] = 0.5;
        // S_22 (Feat-Feat): Feat 0 and 1 are similar
        s_mat[(4, 5)] = 0.8;
        s_mat[(5, 4)] = 0.8;
        // S_12/S_21 (Item-Feat): Feat 0 is associated with Item 2
        s_mat[(2, 4)] = 1.0;
        s_mat[(4, 2)] = 1.0;

        let model = RustFeaseModel {
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
            transformation_schema: None,
        };

        // --- 1. Warm User: interacts with Item 0, has Feature 1 ---
        let warm_interactions = vec![(0, 1.0)];
        let warm_features = vec![(1, 1.0)];
        let scores_warm = model.predict(&warm_interactions, &warm_features, 1.0);

        assert_eq!(scores_warm.len(), n_items);
        assert!((scores_warm[0] - 0.0).abs() < 1e-6); // Item 0
        assert!((scores_warm[1] - 0.5).abs() < 1e-6); // Item 1 (recommended)
        assert!((scores_warm[2] - 0.0).abs() < 1e-6); // Item 2
        assert!((scores_warm[3] - 0.0).abs() < 1e-6); // Item 3

        // --- 2. Cold User: NO interactions, has Feature 0 ---
        let cold_interactions = vec![];
        let cold_features = vec![(0, 1.0)];
        let scores_cold = model.predict(&cold_interactions, &cold_features, 1.0);

        assert_eq!(scores_cold.len(), n_items);
        assert!((scores_cold[0] - 0.0).abs() < 1e-6); // Item 0
        assert!((scores_cold[1] - 0.0).abs() < 1e-6); // Item 1
        assert!((scores_cold[2] - 1.0).abs() < 1e-6); // Item 2 (recommended)
        assert!((scores_cold[3] - 0.0).abs() < 1e-6); // Item 3

        Ok(())
    }

    #[test]
    fn test_predict_raw_routes_through_schema() -> Result<()> {
        use crate::transform::FeatureTransformationSchema;
        use std::collections::HashMap;

        let n_items = 4;
        let n_user_features = 3;
        let total_dim = n_items + n_user_features;

        // Same hand-built S as the warm/cold test: feature 0 ↔ item 2.
        let mut s_mat = DMatrix::<f64>::zeros(total_dim, total_dim);
        s_mat[(2, 4)] = 1.0;
        s_mat[(4, 2)] = 1.0;

        let mut mappings = dummy_mappings();
        mappings.item_to_idx = (0..n_items).map(|i| (format!("item_{i}"), i)).collect();
        mappings.user_feature_to_idx = [("plan_Premium".to_string(), 0)].into_iter().collect();

        let mut schema = FeatureTransformationSchema::new();
        schema.add_categorical("plan".to_string(), "plan".to_string());

        let model = RustFeaseModel {
            s_matrix: s_mat,
            num_items: n_items,
            num_user_features: n_user_features,
            num_item_features: 0,
            alpha: 1.0,
            beta: 1.0,
            lambda_: 100.0,
            meta_weight: 0.0,
            mappings,
            weighting_config: None,
            transformation_schema: Some(schema),
        };

        // Raw "plan": "Premium" must reach feature 0 via the schema and score
        // item 2; the unknown item GUID is skipped, not an error.
        let mut raw = HashMap::new();
        raw.insert("plan".to_string(), serde_json::json!("Premium"));
        let mut interactions = HashMap::new();
        interactions.insert("item_nope".to_string(), 1.0);
        let scores = model.predict_raw(&interactions, &raw, 1.0);
        assert_eq!(scores.len(), n_items);
        assert!((scores[2] - 1.0).abs() < 1e-6);

        // Without a schema, already-engineered numeric features pass through.
        let mut model_no_schema = model.clone();
        model_no_schema.transformation_schema = None;
        let mut engineered = HashMap::new();
        engineered.insert("plan_Premium".to_string(), serde_json::json!(1.0));
        let scores = model_no_schema.predict_raw(&HashMap::new(), &engineered, 1.0);
        assert!((scores[2] - 1.0).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_predict_similar_items() -> Result<()> {
        let n_items = 4;
        let n_user_features = 2;
        let total_dim = n_items + n_user_features;

        let mut s_mat = DMatrix::<f64>::zeros(total_dim, total_dim);
        // Items 0 and 1 are very similar
        s_mat[(0, 1)] = 0.9;
        s_mat[(1, 0)] = 0.9;
        // Items 0 and 2 are somewhat similar
        s_mat[(0, 2)] = 0.3;
        s_mat[(2, 0)] = 0.3;
        // Items 1 and 3 are somewhat similar
        s_mat[(1, 3)] = 0.4;
        s_mat[(3, 1)] = 0.4;

        let model = RustFeaseModel {
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
            transformation_schema: None,
        };

        // Similar to Item 0
        let similar = model.predict_similar_items(0, 3);
        assert_eq!(similar.len(), 3); // 3 other items
        assert_eq!(similar[0].0, 1); // Item 1 most similar (0.9)
        assert!((similar[0].1 - 0.9).abs() < 1e-6);
        assert_eq!(similar[1].0, 2); // Item 2 next (0.3)
        assert!((similar[1].1 - 0.3).abs() < 1e-6);

        // top_k = 1
        let similar_1 = model.predict_similar_items(0, 1);
        assert_eq!(similar_1.len(), 1);
        assert_eq!(similar_1[0].0, 1);

        // Out of range item
        let similar_oob = model.predict_similar_items(100, 5);
        assert!(similar_oob.is_empty());

        Ok(())
    }

    #[test]
    fn test_validate_passing_model() -> Result<()> {
        let n_items = 4;
        let n_user_features = 3;
        let n_item_features = 2;

        let x_mat = simple_csmat(
            5,
            n_items,
            vec![(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0), (3, 3, 1.0)],
        );
        let u_mat = simple_csmat(
            5,
            n_user_features,
            vec![(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)],
        );
        let t_mat = simple_csmat(
            n_item_features,
            n_items,
            vec![(0, 0, 1.0), (0, 1, 1.0), (1, 2, 1.0), (1, 3, 1.0)],
        );

        let mut model = RustFeaseModel::new(
            n_items,
            n_user_features,
            n_item_features,
            1.0,
            1.0,
            100.0,
            0.0,
            dummy_mappings(),
        );
        model.train(&x_mat, &u_mat, &t_mat)?;

        let report = model.validate();
        assert!(
            report.passed,
            "Validation should pass: {:?}",
            report.messages
        );

        Ok(())
    }

    #[test]
    fn test_validate_catches_nan() {
        let total_dim = 3;
        let mut s = DMatrix::<f64>::zeros(total_dim, total_dim);
        s[(0, 1)] = f64::NAN;

        let model = RustFeaseModel {
            s_matrix: s,
            num_items: 2,
            num_user_features: 1,
            num_item_features: 0,
            alpha: 1.0,
            beta: 1.0,
            lambda_: 100.0,
            meta_weight: 0.0,
            mappings: dummy_mappings(),
            weighting_config: None,
            transformation_schema: None,
        };

        let report = model.validate();
        assert!(!report.passed);
        assert!(report.messages.iter().any(|m| m.contains("NaN")));
    }

    #[test]
    fn test_validate_catches_all_zeros() {
        let model = RustFeaseModel {
            s_matrix: DMatrix::zeros(3, 3),
            num_items: 2,
            num_user_features: 1,
            num_item_features: 0,
            alpha: 1.0,
            beta: 1.0,
            lambda_: 100.0,
            meta_weight: 0.0,
            mappings: dummy_mappings(),
            weighting_config: None,
            transformation_schema: None,
        };

        let report = model.validate();
        assert!(!report.passed);
        assert!(report.messages.iter().any(|m| m.contains("all zeros")));
    }
}
