//! Rust-native FEASE model implementation using `nalgebra` and `sprs`.

use nalgebra::{DMatrix, DVector};
use sprs::{CsMat, TriMat};

/// A struct to hold the trained FEASE model, which is essentially the dense
/// (M+K) x (M+K) weight matrix 'S'.
/// This is the internal, pure-Rust struct.
pub struct RustFeaseModel {
    /// The learned weight matrix S.
    /// Dimensions: (items + user_features) x (items + user_features)
    pub s_matrix: DMatrix<f64>,
    pub num_items: usize,
    pub num_user_features: usize,
}

impl RustFeaseModel {
    /// Creates a new, untrained model container.
    pub fn new(num_items: usize, num_user_features: usize) -> Self {
        let total_dim = num_items + num_user_features;
        RustFeaseModel {
            s_matrix: DMatrix::zeros(total_dim, total_dim),
            num_items,
            num_user_features,
        }
    }

    /// Trains the FEASE model based on the "FEASE-U" formulation.
    /// This function is the core of the implementation and is designed
    /// to be memory-efficient by never creating the giant `Z` matrix.
    ///
    /// # Arguments
    /// * `x_mat`: The (N x M) sparse user-item interaction matrix.
    /// * `u_mat`: The (N x K) sparse user-feature matrix.
    /// * `t_mat`: The (L x M) sparse item-feature matrix.
    /// * `alpha`: Weight for item features (α).
    /// * `beta`: Weight for user features (β).
    /// * `lambda`: L2 regularization strength (λ).
    ///
    /// # Panics
    /// Panics if the matrix inversion fails (which should not happen
    /// with proper regularization).
    pub fn train(
        &mut self,
        x_mat: &CsMat<f64>,
        u_mat: &CsMat<f64>,
        t_mat: &CsMat<f64>,
        alpha: f64,
        beta: f64,
        lambda: f64,
    ) {
        println!("Starting FEASE model training (using nalgebra)...");
        let m = x_mat.cols(); // Number of items
        let k = u_mat.cols(); // Number of user features
        let total_dim = m + k;

        println!("M (Items): {}, K (User Features): {}", m, k);
        println!(
            "Total dimension of Gram matrix: {}x{}",
            total_dim, total_dim
        );

        // ---
        // 1. Compute the Gram Matrix G = Z^T * Z in blocks (sparse)
        // ---
        // G = [ G_11 | G_12 ]
        //     [ G_21 | G_22 ]
        //
        // G_11 = X^T*X + α^2 * T^T*T  (M x M)
        // G_12 = β * X^T*U            (M x K)
        // G_21 = β * U^T*X            (K x M)
        // G_22 = β^2 * U^T*U          (K x K)

        println!("Calculating G_11 = X^T*X + α^2 * T^T*T...");
        let xtx = sparse_transpose_self_multiply(x_mat); // M x M
        let ttt = sparse_transpose_self_multiply(t_mat); // M x M
        let g_11 = &xtx + &(&ttt * (alpha * alpha));

        println!("Calculating G_12 = β * X^T*U...");
        let x_t = x_mat.transpose_view();
        let g_12 = &(&x_t * u_mat) * beta; // (M x N) * (N x K) -> M x K

        println!("Calculating G_21 = β * U^T*X...");
        let u_t = u_mat.transpose_view();
        let g_21 = &(&u_t * x_mat) * beta; // (K x N) * (N x M) -> K x M

        println!("Calculating G_22 = β^2 * U^T*U...");
        let utu = sparse_transpose_self_multiply(u_mat); // K x K
        let g_22 = &utu * (beta * beta);

        // ---
        // 2. Assemble the dense Gram matrix G
        //    (Converting from ndarray to nalgebra)
        // ---
        println!("Assembling dense Gram matrix G...");
        let mut g = DMatrix::<f64>::zeros(total_dim, total_dim);

        // Convert from sprs's ndarray (row-major) to nalgebra::DMatrix (col-major)
        // FIX: Use `from_row_slice` for robust ndarray -> nalgebra conversion
        let g_11_nd = g_11.to_dense();
        let g_11_nalgebra = DMatrix::from_row_slice(
            g_11_nd.nrows(),
            g_11_nd.ncols(),
            g_11_nd.as_slice().expect("g_11 ndarray was not contiguous"),
        );
        set_block(&mut g, &g_11_nalgebra, 0, 0);

        // FIX: Use `from_row_slice`
        let g_12_nd = g_12.to_dense();
        let g_12_nalgebra = DMatrix::from_row_slice(
            g_12_nd.nrows(),
            g_12_nd.ncols(),
            g_12_nd.as_slice().expect("g_12 ndarray was not contiguous"),
        );
        set_block(&mut g, &g_12_nalgebra, 0, m);

        // FIX: Use `from_row_slice`
        let g_21_nd = g_21.to_dense();
        let g_21_nalgebra = DMatrix::from_row_slice(
            g_21_nd.nrows(),
            g_21_nd.ncols(),
            g_21_nd.as_slice().expect("g_21 ndarray was not contiguous"),
        );
        set_block(&mut g, &g_21_nalgebra, m, 0);

        // FIX: Use `from_row_slice`
        let g_22_nd = g_22.to_dense();
        let g_22_nalgebra = DMatrix::from_row_slice(
            g_22_nd.nrows(),
            g_22_nd.ncols(),
            g_22_nd.as_slice().expect("g_22 ndarray was not contiguous"),
        );
        set_block(&mut g, &g_22_nalgebra, m, m);

        // ---
        // 3. Compute P = (G + λI)^-1
        // ---
        println!("Calculating P = (G + λI)^-1...");
        let mut g_reg = g.clone();
        for i in 0..g_reg.nrows() {
            g_reg[(i, i)] += lambda;
        }
        let p = g_reg
            .try_inverse()
            .expect("Failed to invert Gram matrix P. Check regularization.");

        // ---
        // 4. Compute S_unconstrained = P * G
        // ---
        println!("Calculating S_unconstrained = P * G...");
        // Note: We use 'g', the original unregularized matrix
        let s_unconstrained = &p * g;

        // ---
        // 5. Apply the EASE zero-diagonal constraint
        //    S = S_unconstrained - P * D
        //    where D is a diagonal matrix with D_jj = S_unconstrained_jj / P_jj
        // ---
        println!("Applying zero-diagonal constraint...");
        let diag_p = p.diagonal();
        let diag_s_un = s_unconstrained.diagonal();

        // Calculate D_jj = S_unconstrained_jj / P_jj
        // Add a small epsilon to avoid division by zero, though P_jj should be stable
        let diag_d_values = diag_s_un.zip_map(&diag_p, |s, p_val| s / (p_val + 1e-9));
        let d_mat = DMatrix::from_diagonal(&diag_d_values);

        // S = S_unconstrained - P * D
        self.s_matrix = s_unconstrained - (&p * d_mat);

        println!("Training complete!");
    }

    /// Predicts scores for a given user.
    ///
    /// # Arguments
    /// * `user_interactions`: A (1 x M) sparse matrix of the user's interactions.
    /// * `user_features`: A (1 x K) sparse matrix of the user's features.
    /// * `beta`: The same user-feature weight (β) used in training.
    ///
    /// # Returns
    /// A dense vector (1 x M) of predicted scores for all items.
    ///
    /// # Panics
    /// Panics if matrix dimensions are incorrect.
    pub fn predict(
        &self,
        user_interactions: &CsMat<f64>,
        user_features: &CsMat<f64>,
        beta: f64,
    ) -> DVector<f64> {
        // 1. Construct the user's input vector z = [x | β*u]
        // We can do this by creating a (1 x M+K) dense vector.
        let mut z_vec = DVector::<f64>::zeros(self.num_items + self.num_user_features);

        // Fill item interactions
        // Corrected iterator: .iter() yields (&value, (row, col))
        // FIX: Changed `&val` to `val` to match the iterator type `(usize, f64)`
        for (item_idx, val) in user_interactions.iter().map(|(&v, (_r, c))| (c, v)) {
            if item_idx < self.num_items {
                z_vec[item_idx] = val;
            }
        }

        // Fill user features
        // Corrected iterator: .iter() yields (&value, (row, col))
        // FIX: Changed `&val` to `val` to match the iterator type `(usize, f64)`
        for (feat_idx, val) in user_features.iter().map(|(&v, (_r, c))| (c, v)) {
            if feat_idx < self.num_user_features {
                z_vec[self.num_items + feat_idx] = val * beta;
            }
        }

        // 2. Predict scores: p = z^T * S
        // (1 x total_dim) @ (total_dim x total_dim) -> (1 x total_dim)
        let p_full = z_vec.transpose() * &self.s_matrix;

        // 3. We only want the item scores, which are the first M entries
        let p_items = p_full.columns(0, self.num_items); // Slices columns 0..M

        // Return as a flat DVector
        p_items.transpose().into()
    }
}

// --- Helper Functions ---

/// Helper function to compute B = A^T * A for a sparse matrix A
fn sparse_transpose_self_multiply(a: &CsMat<f64>) -> CsMat<f64> {
    let a_t = a.transpose_view();
    &a_t * a
}

/// Helper function to set a block in a dense matrix from another dense matrix
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

    /// A simple smoke test to ensure the model trains and predicts without panicking.
    #[test]
    fn test_model_train_and_predict() {
        let num_users = 5;
        let num_items = 4;
        let num_user_features = 3;
        let num_item_features = 2;

        let alpha = 1.0;
        let beta = 1.0;
        let lambda = 100.0;

        // --- Create Dummy Data ---
        let x_mat = create_sparse_matrix(
            num_users,
            num_items,
            vec![(0, 0, 1.0), (1, 1, 1.0), (2, 0, 1.0), (3, 2, 1.0)],
        );
        let u_mat = create_sparse_matrix(
            num_users,
            num_user_features,
            vec![(0, 0, 1.0), (1, 1, 1.0), (4, 2, 1.0)], // User 4 is cold
        );
        let t_mat = create_sparse_matrix(
            num_item_features,
            num_items,
            vec![(0, 0, 1.0), (0, 2, 1.0), (1, 1, 1.0)],
        );

        // --- Train ---
        let mut model = RustFeaseModel::new(num_items, num_user_features);
        model.train(&x_mat, &u_mat, &t_mat, alpha, beta, lambda);

        assert_eq!(model.s_matrix.nrows(), num_items + num_user_features);
        assert_eq!(model.s_matrix.ncols(), num_items + num_user_features);

        // --- Predict (Cold Start) ---
        let u4_interactions = create_sparse_matrix(1, num_items, vec![]); // Empty!
        let u4_features = create_sparse_matrix(1, num_user_features, vec![(0, 2, 1.0)]);

        let u4_scores_vec = model.predict(&u4_interactions, &u4_features, beta);

        assert_eq!(u4_scores_vec.len(), num_items);
        // Scores should be non-zero due to feature contribution
        assert!(u4_scores_vec.sum() != 0.0);

        // --- Predict (Warm Start) ---
        let u0_interactions = create_sparse_matrix(1, num_items, vec![(0, 0, 1.0)]);
        let u0_features = create_sparse_matrix(1, num_user_features, vec![(0, 0, 1.0)]);

        let u0_scores_vec = model.predict(&u0_interactions, &u0_features, beta);

        assert_eq!(u0_scores_vec.len(), num_items);
        assert!(u0_scores_vec.sum() != 0.0);
    }
}
