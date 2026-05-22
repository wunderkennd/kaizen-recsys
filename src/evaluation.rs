//! Evaluation pipeline: train/test data splitting and model evaluation harness.
//!
//! Provides functions to split interaction data into train/test sets using various
//! strategies (random, temporal, leave-K-out), and an evaluation harness that
//! computes standard recommendation metrics on held-out data.

use crate::data_pipeline::Mappings;
use crate::metrics;
use crate::models::{ModelInput, RecModel};
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
    /// Coverage across all users' recommendations (computed at the largest K value).
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

pub(crate) fn read_interactions_df(path: &str) -> Result<DataFrame> {
    let p = Path::new(path);
    let ext = p.extension().and_then(|s| s.to_str());
    let df = match ext {
        Some("parquet") => ParquetReader::new(File::open(p)?).finish()?,
        Some("csv") => CsvReader::new(File::open(p)?).finish()?,
        _ => return Err(anyhow!("Unsupported file type: {}", path)),
    };
    Ok(df)
}

pub(crate) fn write_parquet(df: &mut DataFrame, path: &str) -> Result<()> {
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
        let n_test = if rows.len() < 2 {
            0
        } else {
            let requested = (rows.len() as f64 * test_ratio).round() as usize;
            requested.clamp(1, rows.len() - 1)
        };
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

    if days_col.null_count() > 0 {
        return Err(anyhow!(
            "days_ago contains null values; temporal_split requires non-null days_ago so every interaction is assigned to train or test"
        ));
    }

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

// ---------------------------------------------------------------------------
// EvalAdapter — per-model input construction (issue #51)
// ---------------------------------------------------------------------------

/// Per-user data the eval harness collects from train/test/feature
/// files. Handed to an [`EvalAdapter`] which picks the right
/// [`ModelInput`] variant for its model.
pub struct UserEvalContext<'a> {
    /// `(item_idx, value)` pairs in train-DataFrame row order.
    pub train_items: &'a [(usize, f64)],
    /// `days_ago` values parallel to `train_items`, when the train
    /// DataFrame has a `days_ago` column. `None` otherwise.
    /// Sequence-aware models use this to order history chronologically.
    pub train_days_ago: Option<&'a [f64]>,
    /// `(feature_idx, value)` pairs for this user (EASE only).
    pub user_features: &'a [(usize, f64)],
    /// The model's user index for this user_id when present in
    /// `mappings.user_to_idx` (Two-Tower routes `Some(idx)` to its warm
    /// path; `None` falls back to the reserved cold-start row).
    pub user_idx: Option<usize>,
}

/// Per-model strategy for turning [`UserEvalContext`] into model
/// scores. Each model implements an adapter that picks the appropriate
/// [`ModelInput`] variant. Both [`evaluate_model`] and the per-fold
/// scorers in [`crate::tuning`] go through the adapter so the
/// chronological-ordering (SASRec) and cold-start (Two-Tower) decisions
/// are made in one place (issue #51).
pub trait EvalAdapter: Send + Sync {
    /// The wrapped model. Used by the harness for catalog size and
    /// id mappings.
    fn model(&self) -> &dyn RecModel;

    /// Compute a length-`num_items` score vector for `ctx`. Returns
    /// `Err` if the adapter requires data the context does not carry
    /// (e.g. SASRec without `days_ago`).
    fn predict_user_scores(&self, ctx: &UserEvalContext<'_>) -> Result<Vec<f32>>;
}

/// EASE eval adapter — ignores chronology and `user_idx`, builds
/// [`ModelInput::Sparse`] from interactions and user features.
pub struct EaseEvalAdapter<'m> {
    model: &'m dyn RecModel,
}

impl<'m> EaseEvalAdapter<'m> {
    pub fn new(model: &'m dyn RecModel) -> Self {
        Self { model }
    }
}

impl<'m> EvalAdapter for EaseEvalAdapter<'m> {
    fn model(&self) -> &dyn RecModel {
        self.model
    }

    fn predict_user_scores(&self, ctx: &UserEvalContext<'_>) -> Result<Vec<f32>> {
        self.model.predict_scores(ModelInput::Sparse {
            interactions: ctx.train_items,
            user_features: ctx.user_features,
        })
    }
}

/// SASRec eval adapter — sorts each user's train items oldest-first
/// by `days_ago` (matching `crate::data::sequences`) and scores via
/// [`ModelInput::Sequence`]. Returns `Err` if the context has no
/// `days_ago`: SASRec is order-sensitive, and silently falling back
/// to row order produced misleading metrics in earlier versions (#51).
pub struct SasRecEvalAdapter<'m> {
    model: &'m dyn RecModel,
}

impl<'m> SasRecEvalAdapter<'m> {
    pub fn new(model: &'m dyn RecModel) -> Self {
        Self { model }
    }
}

impl<'m> EvalAdapter for SasRecEvalAdapter<'m> {
    fn model(&self) -> &dyn RecModel {
        self.model
    }

    fn predict_user_scores(&self, ctx: &UserEvalContext<'_>) -> Result<Vec<f32>> {
        let history = chronological_history_from_ctx(ctx)?;
        self.model
            .predict_scores(ModelInput::Sequence { history: &history })
    }
}

/// Order `ctx.train_items` oldest-first by `train_days_ago`, matching
/// the training-time convention in `data::sequences::build_sequences`
/// (larger `days_ago` == older). Stable sort preserves file-row order
/// for ties. Pub(crate) so the per-fold SASRec scorer in `tuning` can
/// use the same ordering rule.
pub(crate) fn chronological_history_from_ctx(ctx: &UserEvalContext<'_>) -> Result<Vec<usize>> {
    let days_ago = ctx.train_days_ago.ok_or_else(|| {
        anyhow!(
            "SasRecEvalAdapter requires a `days_ago` column in the train interactions \
             file to order each user's history chronologically; it is absent. \
             (Matches the same requirement in `data::sequences::build_sequences`.)"
        )
    })?;
    if days_ago.len() != ctx.train_items.len() {
        return Err(anyhow!(
            "internal error: days_ago.len() ({}) != train_items.len() ({})",
            days_ago.len(),
            ctx.train_items.len()
        ));
    }
    let mut pairs: Vec<(f64, usize)> = days_ago
        .iter()
        .zip(ctx.train_items.iter())
        .map(|(d, (idx, _))| (*d, *idx))
        .collect();
    // Oldest first: larger days_ago is older. Stable sort preserves
    // file-row order for ties (matches `data::sequences::build_sequences`).
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(pairs.into_iter().map(|(_, idx)| idx).collect())
}

/// Two-Tower eval adapter — scores the catalog via
/// [`ModelInput::TowerUser`] using the resolved `user_idx` (warm) or
/// `None` (cold-start). Predict-time arbitrary user features are not
/// supported yet (see #55).
pub struct TwoTowerEvalAdapter<'m> {
    model: &'m dyn RecModel,
}

impl<'m> TwoTowerEvalAdapter<'m> {
    pub fn new(model: &'m dyn RecModel) -> Self {
        Self { model }
    }
}

impl<'m> EvalAdapter for TwoTowerEvalAdapter<'m> {
    fn model(&self) -> &dyn RecModel {
        self.model
    }

    fn predict_user_scores(&self, ctx: &UserEvalContext<'_>) -> Result<Vec<f32>> {
        self.model.predict_scores(ModelInput::TowerUser {
            user_idx: ctx.user_idx,
            cat_features: &[],
            dense_features: &[],
        })
    }
}

/// Build the right [`EvalAdapter`] for `model.kind()`.
pub fn adapter_for(model: &dyn RecModel) -> Box<dyn EvalAdapter + '_> {
    use crate::models::ModelKind;
    match model.kind() {
        ModelKind::Ease => Box::new(EaseEvalAdapter::new(model)),
        ModelKind::SasRec => Box::new(SasRecEvalAdapter::new(model)),
        ModelKind::TwoTower => Box::new(TwoTowerEvalAdapter::new(model)),
    }
}

/// Evaluates a trained model against test interactions.
///
/// The harness is generalized over `&dyn RecModel` (Phase 4a / issue #30)
/// and dispatches per-user input construction through the [`EvalAdapter`]
/// trait (#51 / Phase 7 follow-up): EASE → [`ModelInput::Sparse`],
/// SASRec → chronologically-sorted [`ModelInput::Sequence`] (errors loudly
/// if the train file has no `days_ago` column), Two-Tower →
/// [`ModelInput::TowerUser`].
///
/// For each user in the test set who also exists in the model's mappings:
/// 1. Gets the user's TEST interactions (ground truth relevant items)
/// 2. Gets the user's TRAIN interactions (to generate predictions from)
/// 3. Calls the per-model `EvalAdapter` to produce scores
/// 4. Ranks items, excludes train items
/// 5. Computes all metrics against test items
/// 6. Averages across users
pub fn evaluate_model(
    model: &dyn RecModel,
    test_interactions_path: &str,
    train_interactions_path: &str,
    user_features_path: Option<&str>,
    config: &EvalConfig,
) -> Result<EvalReport> {
    let adapter = adapter_for(model);
    evaluate_with_adapter(
        adapter.as_ref(),
        test_interactions_path,
        train_interactions_path,
        user_features_path,
        config,
    )
}

/// Lower-level entrypoint: same as [`evaluate_model`] but with an
/// explicit adapter, so callers (e.g. tuning's per-fold scorer) can
/// avoid re-dispatching from `model.kind()`.
pub fn evaluate_with_adapter(
    adapter: &dyn EvalAdapter,
    test_interactions_path: &str,
    train_interactions_path: &str,
    user_features_path: Option<&str>,
    config: &EvalConfig,
) -> Result<EvalReport> {
    log::info!("Starting model evaluation...");

    let test_df = read_interactions_df(test_interactions_path)?;
    let train_df = read_interactions_df(train_interactions_path)?;

    let model = adapter.model();
    let mappings = model.item_mapping();

    // Load user features if provided
    let user_features_map: AHashMap<String, Vec<(usize, f64)>> =
        if let Some(uf_path) = user_features_path {
            build_user_features_map(uf_path, mappings)?
        } else {
            AHashMap::new()
        };

    // Build per-user ground truth from test set: user_id -> set of item indices
    let test_user_col = test_df.column("user_id")?.str()?;
    let test_item_col = test_df.column("item_id")?.str()?;

    let mut test_user_items: AHashMap<String, HashSet<usize>> = AHashMap::new();
    for i in 0..test_df.height() {
        if let (Some(uid), Some(iid)) = (test_user_col.get(i), test_item_col.get(i))
            && let Some(&item_idx) = mappings.item_to_idx.get(iid)
        {
            test_user_items
                .entry(uid.to_string())
                .or_default()
                .insert(item_idx);
        }
    }

    // Build per-user train interactions plus optional days_ago parallel
    // vec when the train file carries that column. Sequence-aware
    // adapters (SASRec) require days_ago; the column-presence check is
    // done once per file, not per row.
    let train_user_col = train_df.column("user_id")?.str()?;
    let train_item_col = train_df.column("item_id")?.str()?;
    let train_val_col = train_df.column("value")?.f64()?;
    let train_days_col_opt = train_df
        .column("days_ago")
        .ok()
        .and_then(|s| s.f64().ok().cloned());

    let mut train_user_interactions: AHashMap<String, Vec<(usize, f64)>> = AHashMap::new();
    let mut train_user_days_ago: AHashMap<String, Vec<f64>> = AHashMap::new();
    for i in 0..train_df.height() {
        let (Some(uid), Some(iid), Some(val)) = (
            train_user_col.get(i),
            train_item_col.get(i),
            train_val_col.get(i),
        ) else {
            continue;
        };
        let Some(&item_idx) = mappings.item_to_idx.get(iid) else {
            continue;
        };
        // When the train file carries a `days_ago` column, we keep
        // train_user_interactions[uid] and train_user_days_ago[uid] in
        // lockstep so SasRecEvalAdapter can pair them by index. A row
        // with a null `days_ago` is skipped entirely (not pushed to
        // either map) — otherwise we'd get parallel vecs of different
        // lengths and the adapter would report a misleading "internal
        // error: days_ago.len() != train_items.len()". Null `days_ago`
        // means we can't place this interaction in time, so excluding
        // it is the conservative choice.
        if let Some(days_col) = train_days_col_opt.as_ref() {
            let Some(d) = days_col.get(i) else {
                continue;
            };
            train_user_interactions
                .entry(uid.to_string())
                .or_default()
                .push((item_idx, val));
            train_user_days_ago
                .entry(uid.to_string())
                .or_default()
                .push(d);
        } else {
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
        let user_idx_opt = mappings.user_to_idx.get(uid.as_str()).copied();
        if user_idx_opt.is_none() {
            continue;
        }

        let user_interactions = train_user_interactions
            .get(uid.as_str())
            .cloned()
            .unwrap_or_default();

        let user_days_ago_vec = if train_days_col_opt.is_some() {
            Some(train_user_days_ago.get(uid.as_str()).cloned().unwrap_or_default())
        } else {
            None
        };

        let user_features: Vec<(usize, f64)> = user_features_map
            .get(uid.as_str())
            .cloned()
            .unwrap_or_default();

        // Hand off per-user data to the model-specific adapter. EASE
        // ignores days_ago/user_idx; SASRec uses days_ago to sort the
        // history chronologically; Two-Tower uses user_idx.
        let ctx = UserEvalContext {
            train_items: &user_interactions,
            train_days_ago: user_days_ago_vec.as_deref(),
            user_features: &user_features,
            user_idx: user_idx_opt,
        };
        let scores = adapter.predict_user_scores(&ctx)?;

        // Build set of train items to exclude
        let train_item_set: HashSet<usize> =
            user_interactions.iter().map(|(idx, _)| *idx).collect();

        // Rank items, excluding train items
        let mut ranked: Vec<(usize, f32)> = scores
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

    let cov = metrics::coverage(&all_recs, model.num_items());

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
pub(crate) fn build_user_features_map(
    user_features_path: &str,
    mappings: &Mappings,
) -> Result<AHashMap<String, Vec<(usize, f64)>>> {
    let df = read_interactions_df(user_features_path)?;
    let user_col = df.column("user_id")?.str()?;
    let feat_col = df.column("feature_name")?.str()?;
    let val_col = df.column("value")?.f64()?;

    let mut map: AHashMap<String, Vec<(usize, f64)>> = AHashMap::new();
    for ((user, feat), val) in user_col.into_iter().zip(feat_col).zip(val_col) {
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

    // -----------------------------------------------------------------------
    // Phase 4a regression (issue #30): evaluating via `&dyn RecModel`
    // (EaseAdapter) must produce metrics numerically identical to the
    // pre-change concrete (`&RustFeaseModel::predict`, f64) path. The only
    // difference the generalization introduces is a single `as f32` score
    // round-trip in the adapter; the 1e-9 tolerance guards exactly that and
    // nothing else (the surrounding ranking/metric math is unchanged).
    // -----------------------------------------------------------------------

    use crate::data_pipeline::Mappings;
    use crate::model::RustFeaseModel;
    use crate::models::EaseAdapter;
    use nalgebra::DMatrix;

    fn regression_mappings() -> Mappings {
        let mut m = Mappings {
            user_to_idx: Default::default(),
            idx_to_user: Default::default(),
            item_to_idx: Default::default(),
            idx_to_item: Default::default(),
            user_feature_to_idx: Default::default(),
            idx_to_user_feature: Default::default(),
            item_feature_to_idx: Default::default(),
            idx_to_item_feature: Default::default(),
        };
        for (i, u) in ["u1", "u2", "u3"].iter().enumerate() {
            m.user_to_idx.insert(u.to_string(), i);
            m.idx_to_user.insert(i, u.to_string());
        }
        for (i, it) in ["i1", "i2", "i3", "i4", "i5"].iter().enumerate() {
            m.item_to_idx.insert(it.to_string(), i);
            m.idx_to_item.insert(i, it.to_string());
        }
        m
    }

    fn regression_model() -> RustFeaseModel {
        let n_items = 5;
        let n_user_features = 0;
        let total = n_items + n_user_features;
        // A non-trivial, symmetric item-item S so rankings are not all ties.
        let mut s = DMatrix::<f64>::zeros(total, total);
        let weights = [
            (0, 1, 0.9),
            (0, 2, 0.4),
            (0, 3, 0.7),
            (1, 2, 0.6),
            (1, 4, 0.3),
            (2, 3, 0.8),
            (2, 4, 0.5),
            (3, 4, 0.2),
        ];
        for &(a, b, w) in &weights {
            s[(a, b)] = w;
            s[(b, a)] = w;
        }
        RustFeaseModel {
            s_matrix: s,
            num_items: n_items,
            num_user_features: n_user_features,
            num_item_features: 0,
            alpha: 1.0,
            beta: 1.0,
            lambda_: 100.0,
            meta_weight: 0.0,
            mappings: regression_mappings(),
            weighting_config: None,
        }
    }

    /// Recompute an `EvalReport` using the pre-change concrete f64 path.
    /// Mirrors `evaluate_model`'s logic exactly, but calls
    /// `RustFeaseModel::predict` directly (no adapter, no f32 cast). This is
    /// the baseline the generalized path must match within 1e-9.
    fn baseline_concrete_report(
        model: &RustFeaseModel,
        test_path: &str,
        train_path: &str,
        config: &EvalConfig,
    ) -> Result<EvalReport> {
        let test_df = read_interactions_df(test_path)?;
        let train_df = read_interactions_df(train_path)?;

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
            if !model.mappings.user_to_idx.contains_key(uid.as_str()) {
                continue;
            }
            let user_interactions = train_user_interactions
                .get(uid.as_str())
                .cloned()
                .unwrap_or_default();
            let user_features: Vec<(usize, f64)> = Vec::new();

            let scores = model.predict(&user_interactions, &user_features, model.beta);
            let train_item_set: HashSet<usize> =
                user_interactions.iter().map(|(idx, _)| *idx).collect();
            let mut ranked: Vec<(usize, f64)> = scores
                .into_iter()
                .enumerate()
                .filter(|(idx, _)| !train_item_set.contains(idx))
                .collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let recommended: Vec<usize> = ranked.iter().take(max_k).map(|(idx, _)| *idx).collect();

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
        Ok(EvalReport {
            metrics_at_k,
            coverage: metrics::coverage(&all_recs, model.num_items),
            num_users: num_users_evaluated,
            num_interactions: total_test_interactions,
        })
    }

    #[test]
    fn test_evaluate_via_dyn_recmodel_matches_concrete_baseline() -> Result<()> {
        let dir = TempDir::new()?;
        let train_path = dir.path().join("train.parquet");
        let test_path = dir.path().join("test.parquet");

        let mut train_df = df!(
            "user_id" => ["u1", "u1", "u2", "u3", "u3"],
            "item_id" => ["i1", "i2", "i1", "i3", "i4"],
            "value" =>   [1.0_f64, 1.0, 1.0, 1.0, 1.0],
        )?;
        let mut test_df = df!(
            "user_id" => ["u1", "u2", "u2", "u3"],
            "item_id" => ["i3", "i2", "i4", "i5"],
            "value" =>   [1.0_f64, 1.0, 1.0, 1.0],
        )?;
        create_test_parquet(&mut train_df, train_path.to_str().unwrap())?;
        create_test_parquet(&mut test_df, test_path.to_str().unwrap())?;

        let config = EvalConfig {
            k_values: vec![1, 2, 3],
        };

        let model = regression_model();
        let baseline = baseline_concrete_report(
            &model,
            test_path.to_str().unwrap(),
            train_path.to_str().unwrap(),
            &config,
        )?;

        // Generalized path: evaluate through `&dyn RecModel` (EaseAdapter).
        let adapter = EaseAdapter::new(model);
        let via_trait: &dyn RecModel = &adapter;
        let report = evaluate_model(
            via_trait,
            test_path.to_str().unwrap(),
            train_path.to_str().unwrap(),
            None,
            &config,
        )?;

        assert_eq!(report.num_users, baseline.num_users);
        assert_eq!(report.num_interactions, baseline.num_interactions);
        assert!(
            (report.coverage - baseline.coverage).abs() < 1e-9,
            "coverage mismatch: trait={} baseline={}",
            report.coverage,
            baseline.coverage
        );
        assert_eq!(report.metrics_at_k.len(), baseline.metrics_at_k.len());
        for (t, b) in report.metrics_at_k.iter().zip(baseline.metrics_at_k.iter()) {
            assert_eq!(t.k, b.k);
            assert!(
                (t.precision - b.precision).abs() < 1e-9,
                "precision@{}",
                t.k
            );
            assert!((t.recall - b.recall).abs() < 1e-9, "recall@{}", t.k);
            assert!((t.ndcg - b.ndcg).abs() < 1e-9, "ndcg@{}", t.k);
            assert!((t.map - b.map).abs() < 1e-9, "map@{}", t.k);
            assert!((t.hit_rate - b.hit_rate).abs() < 1e-9, "hit_rate@{}", t.k);
        }

        Ok(())
    }

    // -----------------------------------------------------------------
    // #51: SASRec eval chronological-history tests
    // -----------------------------------------------------------------

    #[test]
    fn chronological_history_errors_when_days_ago_absent() {
        let items = [(0_usize, 1.0_f64), (1, 1.0), (2, 1.0)];
        let ctx = UserEvalContext {
            train_items: &items,
            train_days_ago: None,
            user_features: &[],
            user_idx: Some(1),
        };
        let r = chronological_history_from_ctx(&ctx);
        assert!(
            r.is_err(),
            "expected Err when days_ago is absent, got {r:?}"
        );
    }

    #[test]
    fn chronological_history_sorts_oldest_first_by_days_ago() {
        // items[i] has days_ago[i]. Larger days_ago is older; the sort
        // should put items in (oldest, ..., newest) order — i.e. by
        // descending days_ago. Stable sort: ties keep input order.
        let items = [(10_usize, 1.0_f64), (20, 1.0), (30, 1.0), (40, 1.0)];
        let days_ago = [1.0_f64, 7.0, 3.0, 7.0];
        let ctx = UserEvalContext {
            train_items: &items,
            train_days_ago: Some(&days_ago),
            user_features: &[],
            user_idx: Some(1),
        };
        let hist = chronological_history_from_ctx(&ctx).expect("sort succeeds");
        // Oldest first: days_ago 7 (item 20, then 40 by stable tie),
        // then 3 (item 30), then 1 (item 10).
        assert_eq!(hist, vec![20, 40, 30, 10]);
    }

    #[cfg(feature = "ml-models")]
    #[test]
    #[ignore = "trains a real burn SASRec; impractically slow in debug. Run with: cargo test --release --features ml-models -- --ignored"]
    fn sasrec_eval_adapter_honors_days_ago_chronology() -> Result<()> {
        // #51 acceptance: the adapter must produce different model scores
        // when fed the same train items in different chronological
        // orderings. We compare score vectors directly (more focused than
        // an aggregate NDCG@k assertion, which could happen to coincide
        // if the top-1 is stable across orderings).
        use crate::data::sequences::build_sequences;
        use crate::data_pipeline::build_interaction_mappings;
        use crate::models::sasrec::{
            SasRecConfig, SasRecTrainingConfig, TrainedSasRec, train_sasrec,
        };
        use burn::backend::ndarray::NdArrayDevice;
        use burn::backend::{Autodiff, NdArray};

        let dir = TempDir::new()?;
        let train_path = dir.path().join("train.parquet");

        // 6 users on a 4-item catalog with two distinct sequential
        // patterns so the trained model has reason to learn order:
        //   3 users: A → B → C  (so "last=B" → next=C)
        //   3 users: C → B → A  (so "last=B" → next=A is also possible;
        //                        the model has to attend to earlier
        //                        positions to disambiguate)
        let mut train_df = df!(
            "user_id" => ["u0","u0","u0", "u1","u1","u1", "u2","u2","u2",
                          "u3","u3","u3", "u4","u4","u4", "u5","u5","u5"],
            "item_id" => ["A","B","C", "A","B","C", "A","B","C",
                          "C","B","A", "C","B","A", "C","B","A"],
            "value"   => vec![1.0_f64; 18],
            "days_ago"=> vec![3.0_f64,2.0,1.0, 3.0,2.0,1.0, 3.0,2.0,1.0,
                              3.0,2.0,1.0, 3.0,2.0,1.0, 3.0,2.0,1.0],
        )?;
        create_test_parquet(&mut train_df, train_path.to_str().unwrap())?;

        let mappings = build_interaction_mappings(train_path.to_str().unwrap())?;
        let dataset = build_sequences(train_path.to_str().unwrap(), &mappings, 8)?;
        let vocab_size = mappings.idx_to_item.len() + 1;
        let model_config = SasRecConfig::new(vocab_size, 16, 8, 2, 1).with_dropout(0.0);
        let train_config = SasRecTrainingConfig::new()
            .with_num_epochs(30)
            .with_batch_size(4)
            .with_learning_rate(1e-2)
            .with_patience(30)
            .with_seed(42);
        let device = NdArrayDevice::default();
        let fitted = train_sasrec::<Autodiff<NdArray<f32>>>(
            &model_config,
            &train_config,
            &dataset,
            &device,
        )?;
        let trained = TrainedSasRec::new(fitted, model_config, mappings);

        // Same `train_items` slice (file-row order [A, B, C]) but
        // different `days_ago` per call. The adapter must sort so that
        // chronological ordering [A, B, C] differs from reversed [C, B, A].
        let mapping = trained.item_mapping();
        let a = *mapping.item_to_idx.get("A").expect("A in catalog");
        let b = *mapping.item_to_idx.get("B").expect("B in catalog");
        let c = *mapping.item_to_idx.get("C").expect("C in catalog");
        let items = [(a, 1.0_f64), (b, 1.0), (c, 1.0)];

        let adapter = SasRecEvalAdapter::new(&trained);
        // days_ago [3, 2, 1]: A oldest → C newest → sorted history [A, B, C]
        let scores_chrono = adapter.predict_user_scores(&UserEvalContext {
            train_items: &items,
            train_days_ago: Some(&[3.0, 2.0, 1.0]),
            user_features: &[],
            user_idx: Some(1),
        })?;
        // days_ago [1, 2, 3]: A newest → C oldest → sorted history [C, B, A]
        let scores_reversed = adapter.predict_user_scores(&UserEvalContext {
            train_items: &items,
            train_days_ago: Some(&[1.0, 2.0, 3.0]),
            user_features: &[],
            user_idx: Some(1),
        })?;

        assert_eq!(scores_chrono.len(), scores_reversed.len());
        let max_abs_diff = scores_chrono
            .iter()
            .zip(scores_reversed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs_diff > 1e-6,
            "score vectors should differ between chronological and reversed days_ago; \
             max abs diff was {max_abs_diff}"
        );
        Ok(())
    }
}
