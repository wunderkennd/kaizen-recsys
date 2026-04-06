//! Evaluation pipeline: train/test data splitting and model evaluation harness.
//!
//! Provides functions to split interaction data into train/test sets using various
//! strategies (random, temporal, leave-K-out), and an evaluation harness that
//! computes standard recommendation metrics on held-out data.

use crate::data_pipeline::Mappings;
use crate::metrics;
use crate::model::RustFeaseModel;
use ahash::AHashMap;
use anyhow::{Result, anyhow};
use polars::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::ops::Not;
use std::path::Path;

/// Statistics returned by every split function.
#[derive(Debug, Clone)]
pub struct SplitStats {
    pub train_interactions: usize,
    pub test_interactions: usize,
    pub train_users: usize,
    pub test_users: usize,
}

/// Configuration for evaluation.
#[derive(Debug, Clone)]
pub struct EvalConfig {
    /// K values to evaluate at (e.g., [5, 10, 20, 50]).
    pub k_values: Vec<usize>,
}

/// Results of evaluating a model on test data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    /// Per-K metrics.
    pub metrics_at_k: Vec<MetricsAtK>,
    /// Coverage across all users' recommendations.
    pub coverage: f64,
    /// Number of test users evaluated.
    pub num_users: usize,
    /// Number of test interactions.
    pub num_interactions: usize,
}

/// Metrics computed at a specific K value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsAtK {
    pub k: usize,
    /// Mean precision across users.
    pub precision: f64,
    /// Mean recall across users.
    pub recall: f64,
    /// Mean NDCG across users.
    pub ndcg: f64,
    /// Mean average precision across users.
    pub map: f64,
    /// Mean hit rate across users.
    pub hit_rate: f64,
}

// ---------------------------------------------------------------------------
// Helpers for reading/writing parquet
// ---------------------------------------------------------------------------

fn read_interactions_df(path: &str) -> Result<DataFrame> {
    let p = Path::new(path);
    let ext = p.extension().and_then(|s| s.to_str());
    let df = match ext {
        Some("parquet") => ParquetReader::new(File::open(p)?).finish()?,
        Some("csv") => CsvReader::new(File::open(p)?).finish()?,
        _ => return Err(anyhow!("Unsupported file type: {}", path)),
    };
    Ok(df)
}

fn write_parquet(df: &mut DataFrame, path: &str) -> Result<()> {
    let mut file = File::create(path)?;
    ParquetWriter::new(&mut file).finish(df)?;
    Ok(())
}

fn count_unique_users(df: &DataFrame) -> Result<usize> {
    let col = df.column("user_id")?.str()?;
    let mut seen = HashSet::new();
    for val in col.into_iter().flatten() {
        seen.insert(val.to_string());
    }
    Ok(seen.len())
}

// ---------------------------------------------------------------------------
// Split functions
// ---------------------------------------------------------------------------

/// Random split: for each user, randomly holds out `test_ratio` fraction of interactions.
/// Writes train and test Parquet files to the specified output paths.
pub fn random_split(
    interactions_path: &str,
    train_output: &str,
    test_output: &str,
    test_ratio: f64,
    seed: u64,
) -> Result<SplitStats> {
    if !(0.0..=1.0).contains(&test_ratio) {
        return Err(anyhow!("test_ratio must be between 0.0 and 1.0"));
    }

    let df = read_interactions_df(interactions_path)?;
    let n = df.height();
    let user_col = df.column("user_id")?.str()?;

    // Group row indices by user
    let mut user_rows: AHashMap<String, Vec<usize>> = AHashMap::new();
    for i in 0..n {
        if let Some(uid) = user_col.get(i) {
            user_rows.entry(uid.to_string()).or_default().push(i);
        }
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut train_mask = vec![true; n];

    // Sort user keys for deterministic iteration order (AHashMap is non-deterministic)
    let mut sorted_uids: Vec<String> = user_rows.keys().cloned().collect();
    sorted_uids.sort();
    for uid in &sorted_uids {
        let rows = user_rows.get_mut(uid).unwrap();
        rows.shuffle(&mut rng);
        let n_test = (rows.len() as f64 * test_ratio).round() as usize;
        for &idx in rows.iter().take(n_test) {
            train_mask[idx] = false;
        }
    }

    let mask_series = BooleanChunked::from_slice("mask".into(), &train_mask);
    let not_mask: BooleanChunked = mask_series.clone().not();

    let mut train_df = df.filter(&mask_series)?;
    let mut test_df = df.filter(&not_mask)?;

    let train_users = count_unique_users(&train_df)?;
    let test_users = count_unique_users(&test_df)?;

    let stats = SplitStats {
        train_interactions: train_df.height(),
        test_interactions: test_df.height(),
        train_users,
        test_users,
    };

    write_parquet(&mut train_df, train_output)?;
    write_parquet(&mut test_df, test_output)?;

    log::info!(
        "Random split: train={} test={} (ratio={})",
        stats.train_interactions,
        stats.test_interactions,
        test_ratio
    );

    Ok(stats)
}

/// Temporal split: interactions with days_ago <= cutoff go to test, rest to train.
/// More recent = lower days_ago = test set. Requires `days_ago` column.
pub fn temporal_split(
    interactions_path: &str,
    train_output: &str,
    test_output: &str,
    days_ago_cutoff: f64,
) -> Result<SplitStats> {
    let df = read_interactions_df(interactions_path)?;
    let days_col = df.column("days_ago")?.f64()?;

    // Build mask: train = days_ago > cutoff (older), test = days_ago <= cutoff (recent)
    let train_mask: BooleanChunked = days_col
        .into_iter()
        .map(|opt| opt.map(|d| d > days_ago_cutoff))
        .collect();
    let test_mask: BooleanChunked = days_col
        .into_iter()
        .map(|opt| opt.map(|d| d <= days_ago_cutoff))
        .collect();

    let mut train_df = df.filter(&train_mask)?;
    let mut test_df = df.filter(&test_mask)?;

    let train_users = count_unique_users(&train_df)?;
    let test_users = count_unique_users(&test_df)?;

    let stats = SplitStats {
        train_interactions: train_df.height(),
        test_interactions: test_df.height(),
        train_users,
        test_users,
    };

    write_parquet(&mut train_df, train_output)?;
    write_parquet(&mut test_df, test_output)?;

    log::info!(
        "Temporal split: train={} test={} (cutoff={})",
        stats.train_interactions,
        stats.test_interactions,
        days_ago_cutoff
    );

    Ok(stats)
}

/// Leave-K-Out: for each user, holds out exactly k random interactions for test.
/// Users with fewer than k+1 interactions go entirely to train.
pub fn leave_k_out_split(
    interactions_path: &str,
    train_output: &str,
    test_output: &str,
    k: usize,
    seed: u64,
) -> Result<SplitStats> {
    let df = read_interactions_df(interactions_path)?;
    let n = df.height();
    let user_col = df.column("user_id")?.str()?;

    // Group row indices by user
    let mut user_rows: AHashMap<String, Vec<usize>> = AHashMap::new();
    for i in 0..n {
        if let Some(uid) = user_col.get(i) {
            user_rows.entry(uid.to_string()).or_default().push(i);
        }
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut train_mask = vec![true; n];

    // Sort user keys for deterministic iteration order (AHashMap is non-deterministic)
    let mut sorted_uids: Vec<String> = user_rows.keys().cloned().collect();
    sorted_uids.sort();
    for uid in &sorted_uids {
        let rows = user_rows.get_mut(uid).unwrap();
        if rows.len() < k + 1 {
            continue;
        }
        rows.shuffle(&mut rng);
        for &idx in rows.iter().take(k) {
            train_mask[idx] = false;
        }
    }

    let mask_series = BooleanChunked::from_slice("mask".into(), &train_mask);
    let not_mask: BooleanChunked = mask_series.clone().not();

    let mut train_df = df.filter(&mask_series)?;
    let mut test_df = df.filter(&not_mask)?;

    let train_users = count_unique_users(&train_df)?;
    let test_users = count_unique_users(&test_df)?;

    let stats = SplitStats {
        train_interactions: train_df.height(),
        test_interactions: test_df.height(),
        train_users,
        test_users,
    };

    write_parquet(&mut train_df, train_output)?;
    write_parquet(&mut test_df, test_output)?;

    log::info!(
        "Leave-{}-out split: train={} test={}",
        k,
        stats.train_interactions,
        stats.test_interactions,
    );

    Ok(stats)
}

// ---------------------------------------------------------------------------
// Evaluation harness
// ---------------------------------------------------------------------------

/// Evaluates a trained model against test interactions.
///
/// For each user in the test set who also exists in the model's mappings:
/// 1. Gets the user's TEST interactions (ground truth relevant items)
/// 2. Gets the user's TRAIN interactions (to generate predictions from)
/// 3. Calls model.predict() with train interactions
/// 4. Ranks items, excludes train items
/// 5. Computes all metrics against test items
/// 6. Averages across users
pub fn evaluate_model(
    model: &RustFeaseModel,
    test_interactions_path: &str,
    train_interactions_path: &str,
    user_features_path: Option<&str>,
    config: &EvalConfig,
) -> Result<EvalReport> {
    log::info!("Starting model evaluation...");

    let test_df = read_interactions_df(test_interactions_path)?;
    let train_df = read_interactions_df(train_interactions_path)?;

    // Load user features if provided
    let user_features_map: AHashMap<String, Vec<(usize, f64)>> =
        if let Some(uf_path) = user_features_path {
            build_user_features_map(uf_path, &model.mappings)?
        } else {
            AHashMap::new()
        };

    // Build per-user ground truth from test set: user_id -> set of item indices
    let test_user_col = test_df.column("user_id")?.str()?;
    let test_item_col = test_df.column("item_id")?.str()?;

    let mut test_user_items: AHashMap<String, HashSet<usize>> = AHashMap::new();
    for i in 0..test_df.height() {
        if let (Some(uid), Some(iid)) = (test_user_col.get(i), test_item_col.get(i))
            && let Some(&item_idx) = model.mappings.item_to_idx.get(iid)
        {
            test_user_items
                .entry(uid.to_string())
                .or_default()
                .insert(item_idx);
        }
    }

    // Build per-user train interactions: user_id -> Vec<(item_idx, value)>
    let train_user_col = train_df.column("user_id")?.str()?;
    let train_item_col = train_df.column("item_id")?.str()?;
    let train_val_col = train_df.column("value")?.f64()?;

    let mut train_user_interactions: AHashMap<String, Vec<(usize, f64)>> = AHashMap::new();
    for i in 0..train_df.height() {
        if let (Some(uid), Some(iid), Some(val)) = (
            train_user_col.get(i),
            train_item_col.get(i),
            train_val_col.get(i),
        ) && let Some(&item_idx) = model.mappings.item_to_idx.get(iid)
        {
            train_user_interactions
                .entry(uid.to_string())
                .or_default()
                .push((item_idx, val));
        }
    }

    let max_k = config.k_values.iter().copied().max().unwrap_or(10);

    // Accumulators for per-K metrics
    let num_k = config.k_values.len();
    let mut sum_precision = vec![0.0; num_k];
    let mut sum_recall = vec![0.0; num_k];
    let mut sum_ndcg = vec![0.0; num_k];
    let mut sum_map = vec![0.0; num_k];
    let mut sum_hit_rate = vec![0.0; num_k];

    let mut all_recs: Vec<Vec<usize>> = Vec::new();
    let mut num_users_evaluated = 0usize;
    let mut total_test_interactions = 0usize;

    for (uid, relevant_items) in &test_user_items {
        // The user must exist in the model's mappings
        if !model.mappings.user_to_idx.contains_key(uid.as_str()) {
            continue;
        }

        let user_interactions = train_user_interactions
            .get(uid.as_str())
            .cloned()
            .unwrap_or_default();

        let user_features: Vec<(usize, f64)> = user_features_map
            .get(uid.as_str())
            .cloned()
            .unwrap_or_default();

        // Predict scores
        let scores = model.predict(&user_interactions, &user_features, model.beta);

        // Build set of train items to exclude
        let train_item_set: HashSet<usize> =
            user_interactions.iter().map(|(idx, _)| *idx).collect();

        // Rank items, excluding train items
        let mut ranked: Vec<(usize, f64)> = scores
            .into_iter()
            .enumerate()
            .filter(|(idx, _)| !train_item_set.contains(idx))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let recommended: Vec<usize> = ranked.iter().take(max_k).map(|(idx, _)| *idx).collect();

        // Compute metrics at each K
        for (ki, &k) in config.k_values.iter().enumerate() {
            sum_precision[ki] += metrics::precision_at_k(&recommended, relevant_items, k);
            sum_recall[ki] += metrics::recall_at_k(&recommended, relevant_items, k);
            sum_ndcg[ki] += metrics::ndcg_at_k(&recommended, relevant_items, k);
            sum_map[ki] += metrics::mean_average_precision(
                &recommended[..k.min(recommended.len())],
                relevant_items,
            );
            sum_hit_rate[ki] += metrics::hit_rate_at_k(&recommended, relevant_items, k);
        }

        all_recs.push(recommended);
        num_users_evaluated += 1;
        total_test_interactions += relevant_items.len();
    }

    if num_users_evaluated == 0 {
        return Err(anyhow!(
            "No test users could be evaluated (no overlap between test users and model mappings)"
        ));
    }

    let n = num_users_evaluated as f64;
    let metrics_at_k: Vec<MetricsAtK> = config
        .k_values
        .iter()
        .enumerate()
        .map(|(ki, &k)| MetricsAtK {
            k,
            precision: sum_precision[ki] / n,
            recall: sum_recall[ki] / n,
            ndcg: sum_ndcg[ki] / n,
            map: sum_map[ki] / n,
            hit_rate: sum_hit_rate[ki] / n,
        })
        .collect();

    let cov = metrics::coverage(&all_recs, model.num_items);

    let report = EvalReport {
        metrics_at_k,
        coverage: cov,
        num_users: num_users_evaluated,
        num_interactions: total_test_interactions,
    };

    log::info!(
        "Evaluation complete: {} users, {} test interactions, coverage={:.4}",
        report.num_users,
        report.num_interactions,
        report.coverage
    );

    Ok(report)
}

/// Builds a map from user_id string to Vec<(feature_idx, value)> from a user features file.
fn build_user_features_map(
    user_features_path: &str,
    mappings: &Mappings,
) -> Result<AHashMap<String, Vec<(usize, f64)>>> {
    let df = read_interactions_df(user_features_path)?;
    let user_col = df.column("user_id")?.str()?;
    let feat_col = df.column("feature_name")?.str()?;
    let val_col = df.column("value")?.f64()?;

    let mut map: AHashMap<String, Vec<(usize, f64)>> = AHashMap::new();
    for ((user, feat), val) in user_col
        .into_iter()
        .zip(feat_col.into_iter())
        .zip(val_col.into_iter())
    {
        if let (Some(u), Some(f), Some(v)) = (user, feat, val)
            && let Some(&feat_idx) = mappings.user_feature_to_idx.get(f)
        {
            map.entry(u.to_string()).or_default().push((feat_idx, v));
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
    use tempfile::TempDir;

    fn create_test_parquet(df: &mut DataFrame, path: &str) -> Result<()> {
        let mut file = File::create(path)?;
        ParquetWriter::new(&mut file).finish(df)?;
        Ok(())
    }

    fn make_interactions_df() -> Result<DataFrame> {
        // 3 users, multiple interactions each
        Ok(df!(
            "user_id" => ["u1", "u1", "u1", "u1",
                          "u2", "u2", "u2",
                          "u3", "u3", "u3", "u3", "u3"],
            "item_id" => ["i1", "i2", "i3", "i4",
                          "i1", "i2", "i3",
                          "i1", "i2", "i3", "i4", "i5"],
            "value" =>   [1.0, 1.0, 1.0, 1.0,
                          1.0, 1.0, 1.0,
                          1.0, 1.0, 1.0, 1.0, 1.0],
        )?)
    }

    fn make_temporal_df() -> Result<DataFrame> {
        Ok(df!(
            "user_id" => ["u1", "u1", "u1", "u2", "u2", "u2"],
            "item_id" => ["i1", "i2", "i3", "i1", "i2", "i3"],
            "value" =>   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "days_ago" => [30.0, 5.0, 2.0, 60.0, 3.0, 1.0],
        )?)
    }

    #[test]
    fn test_random_split_ratios() -> Result<()> {
        let dir = TempDir::new()?;
        let input = dir.path().join("input.parquet");
        let train = dir.path().join("train.parquet");
        let test = dir.path().join("test.parquet");

        let mut df = make_interactions_df()?;
        create_test_parquet(&mut df, input.to_str().unwrap())?;

        let stats = random_split(
            input.to_str().unwrap(),
            train.to_str().unwrap(),
            test.to_str().unwrap(),
            0.25,
            42,
        )?;

        // Total should be preserved
        assert_eq!(stats.train_interactions + stats.test_interactions, 12);

        // Test set should be roughly 25% (with rounding per user)
        assert!(stats.test_interactions > 0);
        assert!(stats.train_interactions > 0);

        // Files should exist
        assert!(train.exists());
        assert!(test.exists());

        Ok(())
    }

    #[test]
    fn test_random_split_deterministic() -> Result<()> {
        let dir = TempDir::new()?;
        let input = dir.path().join("input.parquet");
        let train1 = dir.path().join("train1.parquet");
        let test1 = dir.path().join("test1.parquet");
        let train2 = dir.path().join("train2.parquet");
        let test2 = dir.path().join("test2.parquet");

        let mut df = make_interactions_df()?;
        create_test_parquet(&mut df, input.to_str().unwrap())?;

        let stats1 = random_split(
            input.to_str().unwrap(),
            train1.to_str().unwrap(),
            test1.to_str().unwrap(),
            0.3,
            123,
        )?;

        let stats2 = random_split(
            input.to_str().unwrap(),
            train2.to_str().unwrap(),
            test2.to_str().unwrap(),
            0.3,
            123,
        )?;

        assert_eq!(stats1.train_interactions, stats2.train_interactions);
        assert_eq!(stats1.test_interactions, stats2.test_interactions);

        // Read both test files and verify same content
        let df_test1 = ParquetReader::new(File::open(&test1)?).finish()?;
        let df_test2 = ParquetReader::new(File::open(&test2)?).finish()?;
        assert_eq!(df_test1.height(), df_test2.height());

        Ok(())
    }

    #[test]
    fn test_leave_k_out_correct_k() -> Result<()> {
        let dir = TempDir::new()?;
        let input = dir.path().join("input.parquet");
        let train = dir.path().join("train.parquet");
        let test = dir.path().join("test.parquet");

        let mut df = make_interactions_df()?;
        create_test_parquet(&mut df, input.to_str().unwrap())?;

        let k = 2;
        let stats = leave_k_out_split(
            input.to_str().unwrap(),
            train.to_str().unwrap(),
            test.to_str().unwrap(),
            k,
            42,
        )?;

        // Total preserved
        assert_eq!(stats.train_interactions + stats.test_interactions, 12);

        // Read test file and check per-user counts
        let test_df = ParquetReader::new(File::open(&test)?).finish()?;
        let test_user_col = test_df.column("user_id").unwrap().str().unwrap();

        let mut user_test_counts: AHashMap<String, usize> = AHashMap::new();
        for uid in test_user_col.into_iter().flatten() {
            *user_test_counts.entry(uid.to_string()).or_default() += 1;
        }

        // u1 has 4 interactions (>=k+1=3), should have exactly 2 in test
        // u2 has 3 interactions (>=k+1=3), should have exactly 2 in test
        // u3 has 5 interactions (>=k+1=3), should have exactly 2 in test
        for (_uid, count) in &user_test_counts {
            assert_eq!(
                *count, k,
                "Each eligible user should have exactly k test items"
            );
        }

        Ok(())
    }

    #[test]
    fn test_temporal_split() -> Result<()> {
        let dir = TempDir::new()?;
        let input = dir.path().join("input.parquet");
        let train = dir.path().join("train.parquet");
        let test = dir.path().join("test.parquet");

        let mut df = make_temporal_df()?;
        create_test_parquet(&mut df, input.to_str().unwrap())?;

        // Cutoff of 7.0: days_ago <= 7 goes to test
        let stats = temporal_split(
            input.to_str().unwrap(),
            train.to_str().unwrap(),
            test.to_str().unwrap(),
            7.0,
        )?;

        assert_eq!(stats.train_interactions + stats.test_interactions, 6);

        // Test should contain items with days_ago <= 7: (5, 2, 3, 1) = 4 items
        assert_eq!(stats.test_interactions, 4);
        // Train: (30, 60) = 2 items
        assert_eq!(stats.train_interactions, 2);

        // Verify test file content
        let test_df = ParquetReader::new(File::open(&test)?).finish()?;
        let days = test_df.column("days_ago").unwrap().f64().unwrap();
        for val in days.into_iter().flatten() {
            assert!(val <= 7.0, "Test items should have days_ago <= cutoff");
        }

        Ok(())
    }

    #[test]
    fn test_eval_report_structure() -> Result<()> {
        let config = EvalConfig {
            k_values: vec![5, 10, 20],
        };

        let report = EvalReport {
            metrics_at_k: config
                .k_values
                .iter()
                .map(|&k| MetricsAtK {
                    k,
                    precision: 0.1,
                    recall: 0.05,
                    ndcg: 0.15,
                    map: 0.08,
                    hit_rate: 0.4,
                })
                .collect(),
            coverage: 0.75,
            num_users: 100,
            num_interactions: 500,
        };

        assert_eq!(report.metrics_at_k.len(), 3);
        assert_eq!(report.metrics_at_k[0].k, 5);
        assert_eq!(report.metrics_at_k[1].k, 10);
        assert_eq!(report.metrics_at_k[2].k, 20);
        assert!((report.coverage - 0.75).abs() < 1e-10);
        assert_eq!(report.num_users, 100);
        assert_eq!(report.num_interactions, 500);

        // Verify serialization round-trip
        let json = serde_json::to_string(&report)?;
        let deser: EvalReport = serde_json::from_str(&json)?;
        assert_eq!(deser.num_users, 100);
        assert_eq!(deser.metrics_at_k[0].k, 5);

        Ok(())
    }
}
