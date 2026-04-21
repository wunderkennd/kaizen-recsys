//! Hyperparameter tuning module: grid search and random search over FEASE
//! hyperparameters with k-fold cross-validation.
//!
//! Uses the existing data pipeline and model training directly. Evaluation
//! is based on NDCG@k computed over held-out test users.

use crate::data_pipeline::{self, Mappings};
use crate::evaluation::{build_user_features_map, read_interactions_df, write_parquet};
use crate::model::RustFeaseModel;
use crate::weighting::WeightingConfig;
use ahash::AHashMap;
use anyhow::{Result, anyhow};
use polars::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Parameter types
// ---------------------------------------------------------------------------

/// A single hyperparameter configuration to evaluate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperParams {
    pub alpha: f64,
    pub beta: f64,
    pub lambda_: f64,
    pub meta_weight: f64,
    pub decay_rate: f64,
    pub ips_alpha: f64,
    pub sparsity_threshold: f64,
}

/// Search space for grid search -- each field is a vec of values to try.
#[derive(Debug, Clone)]
pub struct ParamGrid {
    pub alpha: Vec<f64>,
    pub beta: Vec<f64>,
    pub lambda_: Vec<f64>,
    pub meta_weight: Vec<f64>,
    pub decay_rate: Vec<f64>,
    pub ips_alpha: Vec<f64>,
    pub sparsity_threshold: Vec<f64>,
}

/// Result of a single trial in the search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialResult {
    pub params: HyperParams,
    pub mean_score: f64,
    pub fold_scores: Vec<f64>,
}

/// Result of a complete search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub best_params: HyperParams,
    pub best_score: f64,
    pub all_trials: Vec<TrialResult>,
    pub metric_name: String,
}

// ---------------------------------------------------------------------------
// Metrics (NDCG@k)
// ---------------------------------------------------------------------------

/// Computes NDCG@k given recommended item indices and a set of relevant item indices.
///
/// `recommended` is an ordered list of item indices (best first).
/// `relevant` is the set of item indices that are relevant (ground truth).
fn ndcg_at_k(recommended: &[usize], relevant: &ahash::AHashSet<usize>, k: usize) -> f64 {
    if relevant.is_empty() || k == 0 {
        return 0.0;
    }

    let k = k.min(recommended.len());

    // DCG: sum of 1/log2(rank+2) for relevant items in the top-k
    let mut dcg = 0.0;
    for (i, item) in recommended.iter().take(k).enumerate() {
        if relevant.contains(item) {
            dcg += 1.0 / (i as f64 + 2.0).log2();
        }
    }

    // Ideal DCG: best possible DCG with |relevant| items
    let ideal_k = k.min(relevant.len());
    let mut idcg = 0.0;
    for i in 0..ideal_k {
        idcg += 1.0 / (i as f64 + 2.0).log2();
    }

    if idcg == 0.0 { 0.0 } else { dcg / idcg }
}

// ---------------------------------------------------------------------------
// Cartesian product
// ---------------------------------------------------------------------------

/// Generates the cartesian product of all parameter values in the grid.
fn cartesian_product(grid: &ParamGrid) -> Vec<HyperParams> {
    let mut combos = Vec::new();
    for &alpha in &grid.alpha {
        for &beta in &grid.beta {
            for &lambda_ in &grid.lambda_ {
                for &meta_weight in &grid.meta_weight {
                    for &decay_rate in &grid.decay_rate {
                        for &ips_alpha in &grid.ips_alpha {
                            for &sparsity_threshold in &grid.sparsity_threshold {
                                combos.push(HyperParams {
                                    alpha,
                                    beta,
                                    lambda_,
                                    meta_weight,
                                    decay_rate,
                                    ips_alpha,
                                    sparsity_threshold,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    combos
}

// ---------------------------------------------------------------------------
// K-fold split generation
// ---------------------------------------------------------------------------

/// Generate k-fold splits as (train_path, test_path) pairs in a temp directory.
/// Returns the temp dir (caller should keep alive) and the list of fold paths.
fn generate_kfold_splits(
    interactions_path: &str,
    n_folds: usize,
    seed: u64,
) -> Result<(tempfile::TempDir, Vec<(String, String)>)> {
    if n_folds < 2 {
        return Err(anyhow!("n_folds must be >= 2, got {}", n_folds));
    }

    // Read interactions DataFrame
    let df = read_interactions_df(interactions_path)?;

    // Get unique user_ids
    let user_col = df.column("user_id")?.str()?;
    let mut unique_users: Vec<String> = user_col
        .into_iter()
        .flatten()
        .map(|s| s.to_string())
        .collect::<ahash::AHashSet<String>>()
        .into_iter()
        .collect();
    // Sort for deterministic order before shuffling (AHashSet iteration is non-deterministic)
    unique_users.sort();

    if n_folds > unique_users.len() {
        return Err(anyhow!(
            "n_folds ({}) exceeds number of unique users ({})",
            n_folds,
            unique_users.len()
        ));
    }

    // Deterministic shuffle
    let mut rng = StdRng::seed_from_u64(seed);
    unique_users.shuffle(&mut rng);

    // Split into k groups
    let fold_size = unique_users.len() / n_folds;
    let remainder = unique_users.len() % n_folds;

    let mut folds: Vec<Vec<String>> = Vec::with_capacity(n_folds);
    let mut start = 0;
    for i in 0..n_folds {
        let extra = if i < remainder { 1 } else { 0 };
        let end = start + fold_size + extra;
        folds.push(unique_users[start..end].to_vec());
        start = end;
    }

    // Create temp directory for fold files
    let tmp_dir = tempfile::tempdir()?;
    let mut fold_paths = Vec::with_capacity(n_folds);

    for (fold_idx, fold_users) in folds.iter().enumerate() {
        // Test users = fold_idx group; train users = everyone else
        let test_users: ahash::AHashSet<&str> = fold_users.iter().map(|s| s.as_str()).collect();

        // Build boolean mask for train/test
        let user_col = df.column("user_id")?.str()?;
        let mut train_mask = Vec::with_capacity(df.height());
        let mut test_mask = Vec::with_capacity(df.height());
        for val in user_col.into_iter() {
            match val {
                Some(u) => {
                    let is_test = test_users.contains(u);
                    train_mask.push(!is_test);
                    test_mask.push(is_test);
                }
                None => {
                    train_mask.push(false);
                    test_mask.push(false);
                }
            }
        }

        let train_bool = BooleanChunked::from_slice("mask".into(), &train_mask);
        let test_bool = BooleanChunked::from_slice("mask".into(), &test_mask);

        let mut train_df = df.filter(&train_bool)?;
        let mut test_df = df.filter(&test_bool)?;

        let train_path = tmp_dir
            .path()
            .join(format!("fold_{}_train.parquet", fold_idx));
        let test_path = tmp_dir
            .path()
            .join(format!("fold_{}_test.parquet", fold_idx));

        write_parquet(&mut train_df, &train_path.to_string_lossy())?;
        write_parquet(&mut test_df, &test_path.to_string_lossy())?;

        fold_paths.push((
            train_path.to_string_lossy().to_string(),
            test_path.to_string_lossy().to_string(),
        ));
    }

    Ok((tmp_dir, fold_paths))
}

// ---------------------------------------------------------------------------
// Single trial evaluation
// ---------------------------------------------------------------------------

/// Train and evaluate a single parameter configuration on one fold.
/// Returns mean NDCG@k across test users for the specified k.
fn evaluate_trial(
    train_interactions_path: &str,
    test_interactions_path: &str,
    user_features_path: &str,
    item_features_path: &str,
    params: &HyperParams,
    eval_k: usize,
) -> Result<f64> {
    // 1. Build WeightingConfig from params
    let weighting =
        if params.decay_rate > 0.0 || params.ips_alpha > 0.0 || params.sparsity_threshold > 0.0 {
            Some(WeightingConfig {
                event_weights: None,
                decay_rate: params.decay_rate,
                ips_alpha: params.ips_alpha,
                sparsity_threshold: params.sparsity_threshold,
            })
        } else {
            None
        };

    // 2. Build matrices from training data
    let (x_mat, u_mat, t_mat, mappings) = data_pipeline::build_matrices(
        train_interactions_path,
        user_features_path,
        item_features_path,
        weighting.as_ref(),
    )?;

    let num_items = x_mat.cols();
    let num_user_features = u_mat.cols();
    let num_item_features = t_mat.rows();

    // 3. Train model
    let mut model = RustFeaseModel::new(
        num_items,
        num_user_features,
        num_item_features,
        params.alpha,
        params.beta,
        params.lambda_,
        params.meta_weight,
        mappings,
    );
    model.train(&x_mat, &u_mat, &t_mat)?;

    // 4. Apply sparsity pruning if needed
    if params.sparsity_threshold > 0.0 {
        model.prune_sparse(params.sparsity_threshold);
    }

    // 5. Read training interactions to know which items each user has seen
    let train_df = read_interactions_df(train_interactions_path)?;
    let train_user_items = group_user_items(&train_df, &model.mappings)?;

    // 6. Read test interactions and group by user
    let test_df = read_interactions_df(test_interactions_path)?;
    let test_user_items = group_user_items(&test_df, &model.mappings)?;

    // 7. Build user features lookup
    let user_features_map = build_user_features_map(user_features_path, &model.mappings)?;

    // 8. For each test user, predict and compute NDCG@k
    let mut ndcg_sum = 0.0;
    let mut n_users = 0;

    for (user_id, test_items) in &test_user_items {
        if test_items.is_empty() {
            continue;
        }

        // Get the user's training interactions (may be empty for cold-start users)
        let train_items: Vec<(usize, f64)> =
            train_user_items.get(user_id).cloned().unwrap_or_default();

        // Get user features
        let user_feats: Vec<(usize, f64)> =
            user_features_map.get(user_id).cloned().unwrap_or_default();

        // Predict scores for all items
        let scores = model.predict(&train_items, &user_feats, params.beta);

        // Build set of train items to exclude
        let train_item_set: ahash::AHashSet<usize> =
            train_items.iter().map(|(idx, _)| *idx).collect();

        // Rank items by score, excluding train items
        let mut ranked: Vec<(usize, f64)> = scores
            .into_iter()
            .enumerate()
            .filter(|(idx, _)| !train_item_set.contains(idx))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let recommended: Vec<usize> = ranked.iter().map(|(idx, _)| *idx).collect();

        // Relevant items = test items for this user
        let relevant: ahash::AHashSet<usize> = test_items.iter().map(|(idx, _)| *idx).collect();

        ndcg_sum += ndcg_at_k(&recommended, &relevant, eval_k);
        n_users += 1;
    }

    if n_users == 0 {
        return Ok(0.0);
    }

    Ok(ndcg_sum / n_users as f64)
}

// ---------------------------------------------------------------------------
// Grid search
// ---------------------------------------------------------------------------

/// Grid search: evaluates all combinations of parameters in the grid.
pub fn grid_search(
    interactions_path: &str,
    user_features_path: &str,
    item_features_path: &str,
    grid: &ParamGrid,
    n_folds: usize,
    eval_k: usize,
    seed: u64,
) -> Result<SearchResult> {
    let combos = cartesian_product(grid);
    let total = combos.len();
    if total == 0 {
        return Err(anyhow!("Parameter grid produced 0 combinations"));
    }
    log::info!(
        "Grid search: {} parameter combinations, {}-fold CV",
        total,
        n_folds
    );

    // Generate k-fold splits once
    let (_tmp_dir, fold_paths) = generate_kfold_splits(interactions_path, n_folds, seed)?;

    let mut all_trials = Vec::with_capacity(total);
    let mut best_score = f64::NEG_INFINITY;
    let mut best_params = combos[0].clone();

    for (i, params) in combos.iter().enumerate() {
        let mut fold_scores = Vec::with_capacity(n_folds);

        for (train_path, test_path) in &fold_paths {
            let score = evaluate_trial(
                train_path,
                test_path,
                user_features_path,
                item_features_path,
                params,
                eval_k,
            )?;
            fold_scores.push(score);
        }

        let mean_score = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;

        log::info!(
            "Trial {}/{}: alpha={}, beta={}, lambda_={} -> NDCG@{}={:.4}",
            i + 1,
            total,
            params.alpha,
            params.beta,
            params.lambda_,
            eval_k,
            mean_score
        );

        if mean_score > best_score {
            best_score = mean_score;
            best_params = params.clone();
        }

        all_trials.push(TrialResult {
            params: params.clone(),
            mean_score,
            fold_scores,
        });
    }

    Ok(SearchResult {
        best_params,
        best_score,
        all_trials,
        metric_name: format!("ndcg@{}", eval_k),
    })
}

// ---------------------------------------------------------------------------
// Random search
// ---------------------------------------------------------------------------

/// Random search: samples n_trials random parameter configurations from the grid.
#[allow(clippy::too_many_arguments)]
pub fn random_search(
    interactions_path: &str,
    user_features_path: &str,
    item_features_path: &str,
    grid: &ParamGrid,
    n_trials: usize,
    n_folds: usize,
    eval_k: usize,
    seed: u64,
) -> Result<SearchResult> {
    if n_trials == 0 {
        return Err(anyhow!("n_trials must be >= 1"));
    }
    log::info!("Random search: {} trials, {}-fold CV", n_trials, n_folds);

    // Generate k-fold splits once
    let (_tmp_dir, fold_paths) = generate_kfold_splits(interactions_path, n_folds, seed)?;

    // Sample random parameter configs (use seed+1 to decouple from fold generation)
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1));
    let mut sampled_configs = Vec::with_capacity(n_trials);
    for _ in 0..n_trials {
        let params = HyperParams {
            alpha: *grid.alpha.choose(&mut rng).unwrap_or(&1.0),
            beta: *grid.beta.choose(&mut rng).unwrap_or(&1.0),
            lambda_: *grid.lambda_.choose(&mut rng).unwrap_or(&100.0),
            meta_weight: *grid.meta_weight.choose(&mut rng).unwrap_or(&0.0),
            decay_rate: *grid.decay_rate.choose(&mut rng).unwrap_or(&0.0),
            ips_alpha: *grid.ips_alpha.choose(&mut rng).unwrap_or(&0.0),
            sparsity_threshold: *grid.sparsity_threshold.choose(&mut rng).unwrap_or(&0.0),
        };
        sampled_configs.push(params);
    }

    let mut all_trials = Vec::with_capacity(n_trials);
    let mut best_score = f64::NEG_INFINITY;
    let mut best_params = sampled_configs[0].clone();

    for (i, params) in sampled_configs.iter().enumerate() {
        let mut fold_scores = Vec::with_capacity(n_folds);

        for (train_path, test_path) in &fold_paths {
            let score = evaluate_trial(
                train_path,
                test_path,
                user_features_path,
                item_features_path,
                params,
                eval_k,
            )?;
            fold_scores.push(score);
        }

        let mean_score = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;

        log::info!(
            "Trial {}/{}: alpha={}, beta={}, lambda_={} -> NDCG@{}={:.4}",
            i + 1,
            n_trials,
            params.alpha,
            params.beta,
            params.lambda_,
            eval_k,
            mean_score
        );

        if mean_score > best_score {
            best_score = mean_score;
            best_params = params.clone();
        }

        all_trials.push(TrialResult {
            params: params.clone(),
            mean_score,
            fold_scores,
        });
    }

    Ok(SearchResult {
        best_params,
        best_score,
        all_trials,
        metric_name: format!("ndcg@{}", eval_k),
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Groups interactions by user, returning user_id -> Vec<(item_idx, value)>.
fn group_user_items(
    df: &DataFrame,
    mappings: &Mappings,
) -> Result<AHashMap<String, Vec<(usize, f64)>>> {
    let user_col = df.column("user_id")?.str()?;
    let item_col = df.column("item_id")?.str()?;
    let val_col = df.column("value")?.f64()?;

    let mut map: AHashMap<String, Vec<(usize, f64)>> = AHashMap::new();

    for ((user, item), val) in user_col.into_iter().zip(item_col).zip(val_col) {
        if let (Some(u), Some(i), Some(v)) = (user, item, val)
            && let Some(&item_idx) = mappings.item_to_idx.get(i)
        {
            map.entry(u.to_string()).or_default().push((item_idx, v));
        }
    }

    Ok(map)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use polars::df;
    use std::fs::File;
    use std::path::Path;

    /// Helper to create a dummy parquet file in a temp dir and return its path.
    fn create_parquet_in(dir: &Path, name: &str, df: &mut DataFrame) -> Result<String> {
        let path = dir.join(name);
        let mut file = File::create(&path)?;
        ParquetWriter::new(&mut file).finish(df)?;
        Ok(path.to_string_lossy().to_string())
    }

    /// Creates a tiny dataset suitable for tuning tests.
    /// Returns (interactions_path, user_features_path, item_features_path, tmpdir).
    fn create_test_dataset() -> Result<(String, String, String, tempfile::TempDir)> {
        let tmp = tempfile::tempdir()?;

        // 6 users, 4 items — enough for 2- or 3-fold splits
        let mut interactions = df!(
            "user_id" => ["u0","u0","u1","u1","u2","u2","u3","u3","u4","u4","u5","u5"],
            "item_id" => ["i0","i1","i1","i2","i0","i2","i2","i3","i0","i3","i1","i3"],
            "value"   => [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )?;

        let mut user_features = df!(
            "user_id"      => ["u0","u1","u2","u3","u4","u5"],
            "feature_name" => ["f_a","f_b","f_a","f_b","f_a","f_b"],
            "value"        => [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )?;

        let mut item_features = df!(
            "item_id"      => ["i0","i1","i2","i3"],
            "feature_name" => ["g_x","g_y","g_x","g_y"],
            "value"        => [1.0, 1.0, 1.0, 1.0],
        )?;

        let i_path = create_parquet_in(tmp.path(), "interactions.parquet", &mut interactions)?;
        let u_path = create_parquet_in(tmp.path(), "user_features.parquet", &mut user_features)?;
        let t_path = create_parquet_in(tmp.path(), "item_features.parquet", &mut item_features)?;

        Ok((i_path, u_path, t_path, tmp))
    }

    #[test]
    fn test_param_grid_cartesian_product() {
        let grid = ParamGrid {
            alpha: vec![0.5, 1.0],
            beta: vec![0.5, 1.0],
            lambda_: vec![10.0, 100.0],
            meta_weight: vec![0.0],
            decay_rate: vec![0.0],
            ips_alpha: vec![0.0],
            sparsity_threshold: vec![0.0],
        };

        let combos = cartesian_product(&grid);
        // 2 * 2 * 2 * 1 * 1 * 1 * 1 = 8
        assert_eq!(combos.len(), 8);

        // Verify all values appear
        let alphas: ahash::AHashSet<u64> = combos.iter().map(|p| p.alpha.to_bits()).collect();
        assert!(alphas.contains(&0.5_f64.to_bits()));
        assert!(alphas.contains(&1.0_f64.to_bits()));
    }

    #[test]
    fn test_kfold_split_coverage() -> Result<()> {
        let (i_path, _, _, _tmpdir) = create_test_dataset()?;

        let (_tmp_fold_dir, fold_paths) = generate_kfold_splits(&i_path, 3, 42)?;
        assert_eq!(fold_paths.len(), 3);

        // Collect all test user_ids across folds; each user should appear exactly once
        let mut all_test_users: Vec<String> = Vec::new();
        for (_train_path, test_path) in &fold_paths {
            let test_df = read_interactions_df(test_path)?;
            let user_col = test_df.column("user_id")?.str()?;
            let users: ahash::AHashSet<String> = user_col
                .into_iter()
                .flatten()
                .map(|s| s.to_string())
                .collect();
            all_test_users.extend(users);
        }

        // Sort for comparison
        all_test_users.sort();
        all_test_users.dedup();
        assert_eq!(
            all_test_users.len(),
            6,
            "All 6 users should appear in test exactly once across folds"
        );

        // Each fold's train + test should cover all interactions
        for (train_path, test_path) in &fold_paths {
            let train_df = read_interactions_df(train_path)?;
            let test_df = read_interactions_df(test_path)?;
            let total = train_df.height() + test_df.height();
            assert_eq!(total, 12, "train + test should equal total interactions");
        }

        Ok(())
    }

    #[test]
    fn test_ndcg_at_k_basic() {
        // Perfect ranking
        let rec = vec![0, 1, 2];
        let rel: ahash::AHashSet<usize> = [0, 1, 2].into_iter().collect();
        let score = ndcg_at_k(&rec, &rel, 3);
        assert!(
            (score - 1.0).abs() < 1e-10,
            "Perfect ranking should give NDCG=1.0, got {}",
            score
        );

        // No relevant items recommended
        let rec2 = vec![3, 4, 5];
        let score2 = ndcg_at_k(&rec2, &rel, 3);
        assert!(
            score2.abs() < 1e-10,
            "No relevant items should give NDCG=0.0"
        );

        // Empty relevant set
        let empty_rel: ahash::AHashSet<usize> = ahash::AHashSet::new();
        let score3 = ndcg_at_k(&rec, &empty_rel, 3);
        assert!(score3.abs() < 1e-10, "Empty relevant should give NDCG=0.0");
    }

    #[test]
    fn test_grid_search_finds_best() -> Result<()> {
        let (i_path, u_path, t_path, _tmpdir) = create_test_dataset()?;

        let grid = ParamGrid {
            alpha: vec![1.0],
            beta: vec![1.0],
            lambda_: vec![10.0, 500.0],
            meta_weight: vec![0.0],
            decay_rate: vec![0.0],
            ips_alpha: vec![0.0],
            sparsity_threshold: vec![0.0],
        };

        let result = grid_search(&i_path, &u_path, &t_path, &grid, 2, 10, 42)?;

        // Should have exactly 2 trials
        assert_eq!(result.all_trials.len(), 2);
        assert_eq!(result.metric_name, "ndcg@10");

        // The best score should equal one of the trial scores
        let trial_scores: Vec<f64> = result.all_trials.iter().map(|t| t.mean_score).collect();
        assert!(
            trial_scores.contains(&result.best_score),
            "Best score {} should be one of the trial scores {:?}",
            result.best_score,
            trial_scores
        );

        // The best score should be the max
        let max_score = trial_scores
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (result.best_score - max_score).abs() < 1e-10,
            "Best score should be the maximum across trials"
        );

        Ok(())
    }

    #[test]
    fn test_random_search_correct_n_trials() -> Result<()> {
        let (i_path, u_path, t_path, _tmpdir) = create_test_dataset()?;

        let grid = ParamGrid {
            alpha: vec![0.5, 1.0, 2.0],
            beta: vec![0.5, 1.0],
            lambda_: vec![10.0, 50.0, 100.0],
            meta_weight: vec![0.0],
            decay_rate: vec![0.0],
            ips_alpha: vec![0.0],
            sparsity_threshold: vec![0.0],
        };

        let n_trials = 3;
        let result = random_search(&i_path, &u_path, &t_path, &grid, n_trials, 2, 10, 42)?;

        assert_eq!(result.all_trials.len(), n_trials);
        assert_eq!(result.metric_name, "ndcg@10");

        // Each trial should have 2 fold scores
        for trial in &result.all_trials {
            assert_eq!(trial.fold_scores.len(), 2);
        }

        Ok(())
    }
}
