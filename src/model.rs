//! This file contains the core Rust logic for the FEASE model.
//! It uses `nalgebra` for dense linear algebra (inversion, etc.)
//! and `sprs` for sparse matrix operations (multiplication, storage).
//!
//! This file is "pure Rust" and has no Python (PyO3) dependencies.
//! It's called by `src/lib.rs` to perform the actual math.

use crate::data_pipeline::Mappings;
use anyhow::Result;
use nalgebra::{DMatrix, DVector};
use sprs::{CsMat, TriMat};

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
    pub num_item_features: usize,
    pub alpha: f64,
    pub beta: f64,
    pub lambda_: f64,
    /// Mappings to convert between string IDs and numeric indices
    pub mappings: Mappings,
}

impl RustFeaseModel {
    /// Creates a new, untrained model container.
    pub fn new(
        num_items: usize,
        num_user_features: usize,
        num_item_features: usize,
        alpha: f64,
        beta: f64,
        lambda_: f64,
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
            mappings,
        }
    }

    /// Trains the FEASE model.
    /// This function computes the Gram matrix `G`, inverts it to get `P`,
    /// and then computes the final weight matrix `S`.
    pub fn train(
        &mut self,
        x_mat: &CsMat<f64>, // N x M (Users x Items)
        u_mat: &CsMat<f64>, // N x K (Users x UserFeatures)
        t_mat: &CsMat<f64>, // L x M (ItemFeatures x Items)
    ) -> Result<()> {
        println!("Starting FEASE model training (using nalgebra)...");
        let m = self.num_items;
        let k = self.num_user_features;
        let total_dim = m + k;

        println!("M (Items): {}, K (User Features): {}", m, k);
        println!("Total dimension of Gram matrix: {}x{}", total_dim, total_dim);

        // ---
        // 1. Compute the Gram Matrix G = Z^T * Z in blocks (using sprs)
        // ---
        // Z = [ X  | βU ]   (N x M) | (N x K)
        //     [ αT |  0  ]   (L x M) | (L x K)
        //
        // G = Z^T * Z = [ G_11 | G_12 ]
        //               [ G_21 | G_22 ]
        //
        // G_11 = X^T*X + α^2 * T^T*T   (M x M)
        // G_12 = β * X^T*U             (M x K)
        // G_21 = β * U^T*X             (K x M)
        // G_22 = β^2 * U^T*U           (K x K)
        // ---

        println!("Calculating G_11 = X^T*X + α^2 * T^T*T...");
        let xtx = sparse_transpose_self_multiply(x_mat); // M x M
        let ttt = sparse_transpose_self_multiply(t_mat); // M x M
        let g_11 = &xtx + &(&ttt * (self.alpha * self.alpha));

        println!("Calculating G_12 = β * X^T*U...");
        let x_t = x_mat.transpose_view();
        let g_12 = &(&x_t * u_mat) * self.beta; // (M x N) * (N x K) -> M x K

        println!("Calculating G_21 = β * U^T*X...");
        let u_t = u_mat.transpose_view();
        let g_21 = &(&u_t * x_mat) * self.beta; // (K x N) * (N x M) -> K x M

        println!("Calculating G_22 = β^2 * U^T*U...");
        let utu = sparse_transpose_self_multiply(u_mat); // K x K
        let g_22 = &utu * (self.beta * self.beta);

        // ---
        // 2. Assemble the dense Gram matrix G in nalgebra
        // ---
        println!("Assembling dense Gram matrix G...");
        let mut g = DMatrix::<f64>::zeros(total_dim, total_dim);

        // FIX: Convert from ndarray (returned by .to_dense()) to nalgebra::DMatrix
        // We use `from_row_slice` because `sprs`'s `.to_dense()` creates a
        // row-major `ndarray`, while `nalgebra::DMatrix` is column-major.
        let g_11_dense = g_11.to_dense();
        let g_11_nalgebra =
            DMatrix::from_row_slice(g_11_dense.nrows(), g_11_dense.ncols(), g_11_dense.as_slice().unwrap());

        let g_12_dense = g_12.to_dense();
        let g_12_nalgebra =
            DMatrix::from_row_slice(g_12_dense.nrows(), g_12_dense.ncols(), g_12_dense.as_slice().unwrap());

        let g_21_dense = g_21.to_dense();
        let g_21_nalgebra =
            DMatrix::from_row_slice(g_21_dense.nrows(), g_21_dense.ncols(), g_21_dense.as_slice().unwrap());

        let g_22_dense = g_22.to_dense();
        let g_22_nalgebra =
            DMatrix::from_row_slice(g_22_dense.nrows(), g_22_dense.ncols(), g_22_dense.as_slice().unwrap());

        set_block(&mut g, &g_11_nalgebra, 0, 0);
        set_block(&mut g, &g_12_nalgebra, 0, m);
        set_block(&mut g, &g_21_nalgebra, m, 0);
        set_block(&mut g, &g_22_nalgebra, m, m);

        // ---
        // 3. Compute P = (G + λI)^-1
        // ---
        println!("Calculating P = (G + λI)^-1...");
        let mut g_reg = g.clone(); // G

        // FIX: Use .diagonal_mut().add_scalar_mut() for nalgebra 0.34.1
        // This adds lambda to the diagonal in-place (G + λI)
        g_reg.diagonal_mut().add_scalar_mut(self.lambda_);

        // Compute the inverse
        let p = g_reg
            .try_inverse()
            .ok_or_else(|| anyhow::anyhow!("Failed to invert Gram matrix G. It may be singular."))?;

        // ---
        // 4. Compute S_unconstrained = P * G
        // ---
        println!("Calculating S_unconstrained = P * G...");
        // Note: We use 'g', the original unregularized matrix
        let s_unconstrained = &p * &g;

        // ---
        // 5. Apply the EASE zero-diagonal constraint
        //    S = S_unconstrained - P * D
        //    where D is a diagonal matrix with D_jj = S_unconstrained_jj / P_jj
        // ---
        println!("Applying zero-diagonal constraint...");

        let diag_p = p.diagonal();
        let diag_s_un = s_unconstrained.diagonal();

        // Calculate D_jj = S_unconstrained_jj / P_jj
        // Add a small epsilon to avoid division by zero
        let diag_d_values = diag_s_un.zip_map(&diag_p, |s, p| s / (p + 1e-9));
        let d_mat = DMatrix::from_diagonal(&diag_d_values);

        // S = S_unconstrained - P * D
        self.s_matrix = s_unconstrained - (&p * &d_mat);

        println!("Training complete!");
        Ok(())
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
        // FIX: Correct pattern match for (usize, f64)
        for (item_idx, val) in user_interactions.iter() {
            if *item_idx < self.num_items {
                z_vec[*item_idx] = *val;
            }
        }

        // Fill user features
        // FIX: Correct pattern match for (usize, f64)
        for (feat_idx, val) in user_features.iter() {
            if *feat_idx < self.num_user_features {
                z_vec[self.num_items + *feat_idx] = *val * beta;
            }
        }

        // 2. Convert to nalgebra DVector
        let z = DVector::from_vec(z_vec);

        // 3. Predict scores: p = S^T * z
        // Why S^T * z? The FEASE paper defines the prediction as p = z^T * S.
        // This gives a (1 x total_dim) row vector.
        // In nalgebra, it's easier to work with column vectors.
        // (S^T * z) = (S^T * z)^T = z^T * S
        // (total_dim x total_dim) @ (total_dim x 1) -> (total_dim x 1)
        let p_full = &self.s_matrix * &z; // S * z, not S^T * z. S is symmetric.

        // 4. We only want the item scores, which are the first M entries
        // Slice the resulting DVector
        let p_items = p_full.rows(0, self.num_items);

        // Convert to a standard Vec<f64> to return to Python
        p_items.iter().cloned().collect()
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

/// Helper function to create a sparse matrix from (row, col, data) triplets.
/// Useful for loading data and creating test matrices.
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

        // Dummy mappings (not used by train, but needed by struct)
        let mappings = Mappings {
            user_to_idx: Default::default(),
            idx_to_user: Default::default(),
            item_to_idx: Default::default(),
            idx_to_item: Default::default(),
            user_feature_to_idx: Default::default(),
            idx_to_user_feature: Default::default(),
            item_feature_to_idx: Default::default(),
            idx_to_item_feature: Default::default(),
        };

        let mut model = RustFeaseModel::new(
            n_items,
            n_user_features,
            n_item_features,
            alpha,
            beta,
            lambda,
            mappings,
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
    fn test_fease_model_predict_warm_and_cold() -> Result<()> {
        let n_items = 4;
        let n_user_features = 3;
        let total_dim = n_items + n_user_features;

        // Create a dummy, hand-calculated S matrix for predictable results
        // S = [ S_11 | S_12 ]
        //     [ S_21 | S_22 ]
        // S_11 (4x4, M x M), S_12 (4x3, M x K)
        // S_21 (3x4, K x M), S_22 (3x3, K x K)
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

        let dummy_mappings = Mappings {
            user_to_idx: Default::default(),
            idx_to_user: Default::default(),
            item_to_idx: Default::default(),
            idx_to_item: Default::default(),
            user_feature_to_idx: Default::default(),
            idx_to_user_feature: Default::default(),
            item_feature_to_idx: Default::default(),
            idx_to_item_feature: Default::default(),
        };

        let model = RustFeaseModel {
            s_matrix: s_mat,
            num_items: n_items,
            num_user_features: n_user_features,
            num_item_features: 0, // Not needed for predict
            alpha: 1.0,
            beta: 1.0,
            lambda_: 100.0,
            mappings: dummy_mappings,
        };

        // --- 1. Warm User: interacts with Item 0, has Feature 1 ---
        // z = [1.0, 0.0, 0.0, 0.0 | 0.0, 1.0, 0.0]
        // p = S * z
        // p[0] = S[0,0]*1 + S[0,1]*0 + S[0,2]*0 + S[0,3]*0 + S[0,4]*0 + S[0,5]*1 + S[0,6]*0 = 0
        // p[1] = S[1,0]*1 + ... + S[1,5]*1 + ... = 0.5 (from S[1,0])
        // p[2] = S[2,0]*1 + ... + S[2,5]*1 + ... = 0
        // p[3] = S[3,0]*1 + ... + S[3,5]*1 + ... = 0
        let warm_interactions = vec![(0, 1.0)];
        let warm_features = vec![(1, 1.0)];
        let scores_warm = model.predict(&warm_interactions, &warm_features, 1.0);

        assert_eq!(scores_warm.len(), n_items);
        assert!((scores_warm[0] - 0.0).abs() < 1e-6); // Item 0
        assert!((scores_warm[1] - 0.5).abs() < 1e-6); // Item 1 (recommended)
        assert!((scores_warm[2] - 0.0).abs() < 1e-6); // Item 2
        assert!((scores_warm[3] - 0.0).abs() < 1e-6); // Item 3

        // --- 2. Cold User: NO interactions, has Feature 0 ---
        // z = [0.0, 0.0, 0.0, 0.0 | 1.0, 0.0, 0.0]
        // p = S * z
        // p[0] = S[0,4]*1 = 0
        // p[1] = S[1,4]*1 = 0
        // p[2] = S[2,4]*1 = 1.0 (from S[2,4])
        // p[3] = S[3,4]*1 = 0
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
}