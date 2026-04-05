//! This file handles the data loading and matrix building.
//! It reads long-format data (e.g., from Parquet) and converts
//! it into the sparse matrices (`X`, `U`, `T`) and the ID/feature
//! mappings required by the model.

use ahash::AHashMap;
use anyhow::Result;
use polars::prelude::*;
use sprs::{CsMat, TriMat};
use std::fs::File;
use std::path::Path;

/// A struct to hold all the string-to-index mappings
/// required for training and prediction.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Mappings {
    pub user_to_idx: AHashMap<String, usize>,
    pub idx_to_user: Vec<String>,
    pub item_to_idx: AHashMap<String, usize>,
    pub idx_to_item: Vec<String>,
    pub user_feature_to_idx: AHashMap<String, usize>,
    pub idx_to_user_feature: Vec<String>,
    pub item_feature_to_idx: AHashMap<String, usize>,
    pub idx_to_item_feature: Vec<String>,
}

/// Main function to build all matrices from long-format files.
///
/// This function is the core of the flexible data pipeline. It reads three
/// separate files (interactions, user features, item features) and
/// constructs the three required CSR matrices and the ID mappings.
pub fn build_matrices(
    interactions_path: &str,
    user_features_path: &str,
    item_features_path: &str,
) -> Result<(CsMat<f64>, CsMat<f64>, CsMat<f64>, Mappings)> {
    println!("Starting matrix build process...");

    // --- 1. Load DataFrames from files ---
    let df_i = read_lazyframe(interactions_path)?.collect()?;
    let df_u = read_lazyframe(user_features_path)?.collect()?;
    let df_t = read_lazyframe(item_features_path)?.collect()?;

    // --- 2. Build Mappings ---
    // We must build user and item mappings from *all* files
    // to include cold-start users/items.
    println!("Building ID and feature mappings...");

    let (user_to_idx, idx_to_user) =
        build_mapping_from_dfs(&[&df_i, &df_u], "user_id")?;
    let (item_to_idx, idx_to_item) =
        build_mapping_from_dfs(&[&df_i, &df_t], "item_id")?;
    let (user_feature_to_idx, idx_to_user_feature) =
        build_mapping_from_dfs(&[&df_u], "feature_name")?;
    let (item_feature_to_idx, idx_to_item_feature) =
        build_mapping_from_dfs(&[&df_t], "feature_name")?;

    let num_users = user_to_idx.len();
    let num_items = item_to_idx.len();
    let num_user_features = user_feature_to_idx.len();
    let num_item_features = item_feature_to_idx.len();

    println!("Mappings complete:");
    println!("  Unique Users: {}", num_users);
    println!("  Unique Items: {}", num_items);
    println!("  Unique User Features: {}", num_user_features);
    println!("  Unique Item Features: {}", num_item_features);

    // --- 3. Build Triplet Lists ---
    // This is more memory-efficient than building TriMat directly
    // as we don't know the exact dimensions until after mapping.
    println!("Building triplet lists...");

    let x_triplets = build_triplets(
        &df_i,
        "user_id",
        "item_id",
        Some("value"),
        &user_to_idx,
        &item_to_idx,
    )?;
    let u_triplets = build_triplets(
        &df_u,
        "user_id",
        "feature_name",
        Some("value"),
        &user_to_idx,
        &user_feature_to_idx,
    )?;
    let t_triplets = build_triplets(
        &df_t,
        "item_id",
        "feature_name",
        Some("value"),
        &item_to_idx,
        &item_feature_to_idx,
    )?;

    // --- 4. Build CSR Matrices ---
    println!("Converting triplets to CSR matrices...");

    // X Matrix: (N x M)
    let mut x_trimat = TriMat::with_capacity((num_users, num_items), x_triplets.len());
    for (r, c, v) in x_triplets {
        x_trimat.add_triplet(r, c, v);
    }
    let x_mat = x_trimat.to_csr();

    // U Matrix: (N x K)
    let mut u_trimat = TriMat::with_capacity((num_users, num_user_features), u_triplets.len());
    for (r, c, v) in u_triplets {
        u_trimat.add_triplet(r, c, v);
    }
    let u_mat = u_trimat.to_csr();

    // T Matrix: (L x M)
    // Note: The FEASE paper defines T as (L x M), where L is item features.
    // Our triplet builder uses (item_id, feature_name), which is (M x L).
    // We must build the (M x L) matrix first, then transpose it to get (L x M).
    let mut t_mat_ml = TriMat::with_capacity((num_items, num_item_features), t_triplets.len());
    for (r, c, v) in t_triplets {
        t_mat_ml.add_triplet(r, c, v);
    }
    let t_mat = t_mat_ml.to_csr().transpose_into(); // Transpose to (L x M)

    println!("Matrix build complete!");

    let mappings = Mappings {
        user_to_idx,
        idx_to_user,
        item_to_idx,
        idx_to_item,
        user_feature_to_idx,
        idx_to_user_feature,
        item_feature_to_idx,
        idx_to_item_feature,
    };

    Ok((x_mat, u_mat, t_mat, mappings))
}

/// Builds a mapping (String -> usize) and its reverse (usize -> String)
/// from one or more DataFrame columns.
fn build_mapping_from_dfs(
    dfs: &[&DataFrame],
    col_name: &str,
) -> Result<(AHashMap<String, usize>, Vec<String>)> {
    let mut unique_strings: AHashMap<String, usize> = AHashMap::new();
    let mut idx_to_string = Vec::new();

    for df in dfs {
        let col = df.column(col_name)?.str()?;
        for opt_val in col.into_iter() {
            if let Some(val) = opt_val {
                if !unique_strings.contains_key(val) {
                    let idx = unique_strings.len();
                    unique_strings.insert(val.to_string(), idx);
                    idx_to_string.push(val.to_string());
                }
            }
        }
    }

    Ok((unique_strings, idx_to_string))
}

/// Builds a list of (row, col, value) triplets from a DataFrame.
fn build_triplets(
    df: &DataFrame,
    row_col_name: &str,
    col_col_name: &str,
    val_col_name: Option<&str>,
    row_map: &AHashMap<String, usize>,
    col_map: &AHashMap<String, usize>,
) -> Result<Vec<(usize, usize, f64)>> {
    let row_series = df.column(row_col_name)?.str()?;
    let col_series = df.column(col_col_name)?.str()?;

    // Get an iterator for values, or use a default of 1.0 if no value column
    let val_iter: Box<dyn Iterator<Item = Option<f64>>> = match val_col_name {
        Some(name) => {
            let val_series = df.column(name)?.f64()?;
            Box::new(val_series.into_iter())
        }
        None => {
            // If no value column, create an iterator that just yields 1.0
            Box::new((0..df.height()).map(|_| Some(1.0)))
        }
    };

    let mut triplets = Vec::with_capacity(df.height());

    for ((opt_row_str, opt_col_str), opt_val) in
        row_series.into_iter().zip(col_series.into_iter()).zip(val_iter)
    {
        if let (Some(row_str), Some(col_str), Some(val)) = (opt_row_str, opt_col_str, opt_val) {
            // Look up the indices. If not found, skip (shouldn't happen if maps
            // are built correctly, but it's safe).
            if let (Some(&row_idx), Some(&col_idx)) = (row_map.get(row_str), col_map.get(col_str)) {
                triplets.push((row_idx, col_idx, val));
            }
        }
    }

    Ok(triplets)
}

/// Reads a Parquet or CSV file from a path into a Polars LazyFrame.
fn read_lazyframe(path_str: &str) -> Result<LazyFrame> {
    let path = Path::new(path_str);
    let extension = path.extension().and_then(|s| s.to_str());

    let lf = match extension {
        Some("parquet") => {
            // Use File::open
            ParquetReader::new(File::open(path)?)
                .finish()?
                .lazy()
        }
        Some("csv") => {
            // FIX: API change .with_infer_schema_length -> .with_infer_schema
            CsvReader::new(File::open(path)?) // Use File::open
                .finish()?
                .lazy()
        }
        _ => {
            return Err(anyhow::anyhow!(
                "Unsupported file type for: {}. Supported types are .parquet and .csv",
                path_str
            ));
        }
    };
    Ok(lf)
}

// --- Unit Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use polars::df;

    // Helper to create a dummy parquet file and return its path
    fn create_dummy_parquet(
        df: &mut DataFrame,
        file_name: &str,
    ) -> Result<String> {
        let path = format!("./{}", file_name);
        let mut file = File::create(&path)?;
        ParquetWriter::new(&mut file).finish(df)?;
        Ok(path)
    }

    #[test]
    fn test_build_matrices_flexible() -> Result<()> {
        // --- 1. Create Dummy DataFrames ---
        let mut df_i = df!(
            "user_id" => ["u1", "u1", "u2", "u3"],
            "item_id" => ["i1", "i2", "i2", "i3"],
            "value" => [1.0, 1.0, 1.0, 1.0],
        )?;

        // u4 is a cold-start user
        let mut df_u = df!(
            "user_id" => ["u1", "u2", "u3", "u4"],
            "feature_name" => ["age_20s", "age_30s", "age_20s", "age_40s"],
            "value" => [1.0, 1.0, 1.0, 1.0],
        )?;

        // i4 is a cold-start item
        let mut df_t = df!(
            "item_id" => ["i1", "i2", "i3", "i4"],
            "feature_name" => ["genre_action", "genre_comedy", "genre_action", "genre_drama"],
            "value" => [1.0, 1.0, 1.0, 1.0],
        )?;

        // --- 2. Write to Parquet files ---
        let i_path = create_dummy_parquet(&mut df_i, "test_i.parquet")?;
        let u_path = create_dummy_parquet(&mut df_u, "test_u.parquet")?;
        let t_path = create_dummy_parquet(&mut df_t, "test_t.parquet")?;

        // --- 3. Run the build_matrices function ---
        let (x_mat, u_mat, t_mat, mappings) =
            build_matrices(&i_path, &u_path, &t_path)?;

        // --- 4. Validate Mappings ---
        assert_eq!(mappings.user_to_idx.len(), 4); // u1, u2, u3, u4
        assert_eq!(mappings.item_to_idx.len(), 4); // i1, i2, i3, i4
        assert_eq!(mappings.user_feature_to_idx.len(), 3); // age_20s, age_30s, age_40s
        assert_eq!(mappings.item_feature_to_idx.len(), 3); // genre_action, genre_comedy, genre_drama

        assert_eq!(mappings.idx_to_user.len(), 4);
        assert_eq!(mappings.idx_to_item.len(), 4);

        // --- 5. Validate Matrix Dimensions ---
        // N = 4, M = 4, K = 3, L = 3
        assert_eq!(x_mat.rows(), 4); // N
        assert_eq!(x_mat.cols(), 4); // M
        assert_eq!(x_mat.nnz(), 4); // 4 interactions

        assert_eq!(u_mat.rows(), 4); // N
        assert_eq!(u_mat.cols(), 3); // K
        assert_eq!(u_mat.nnz(), 4); // 4 user features

        assert_eq!(t_mat.rows(), 3); // L (Item Features)
        assert_eq!(t_mat.cols(), 4); // M (Items)
        assert_eq!(t_mat.nnz(), 4); // 4 item features

        // --- 6. Validate T Matrix (Transpose) ---
        // We check one entry to ensure the (M x L) -> (L x M) transpose worked.
        // `df_t` had ("i1", "genre_action").
        // Let's say i1 -> idx 0, genre_action -> idx 0
        // The triplet from `build_triplets` is (0, 0, 1.0)
        // The `t_mat_ml` (M x L) would have `t_mat_ml[0, 0] = 1.0`
        // The final `t_mat` (L x M) should also have `t_mat[0, 0] = 1.0`
        let i1_idx = *mappings.item_to_idx.get("i1").unwrap();
        let action_idx = *mappings.item_feature_to_idx.get("genre_action").unwrap();

        // Check t_mat (L x M) at (feature, item)
        assert_eq!(t_mat.get(action_idx, i1_idx), Some(&1.0));

        // --- 7. Cleanup ---
        std::fs::remove_file(&i_path)?;
        std::fs::remove_file(&u_path)?;
        std::fs::remove_file(&t_path)?;

        Ok(())
    }
}