//! Availability-aware evaluation (issue #101).
//!
//! Over a multi-year training window an item is not available to every
//! user at every time: it may not have been released yet, it may have
//! left the catalog, or it may be licensed only in some territories. The
//! plain harness ranks the whole catalog and so charges the model for
//! recommending titles the user could not have watched, and counts
//! coverage against items nobody could have been shown.
//!
//! This module loads an **availability table** and turns it into a
//! per-user *eligible* set at that user's *reference time*:
//!
//! - **Table columns** (long format, Parquet or CSV): `item_id` (series
//!   id, same id space as the interactions), optional `territory`
//!   (`"*"` or missing = global), `available_from_days_ago` (f64), and
//!   optional nullable `available_to_days_ago` (f64; null = still
//!   available). Any other column (e.g. `season_id`) is carried for the
//!   producer's benefit and ignored here; several rows per item roll up
//!   to the item, so season-level windows make the *series* eligible
//!   whenever any season is.
//! - **Window semantics** in `days_ago` units (larger = older): a window
//!   contains reference time `r` when `from >= r` and (`to` is null or
//!   `to < r`). A window that ended exactly at `r` does not count.
//! - **Territory**: rows tagged `"*"` apply to everyone. A user's own
//!   territory comes from the categorical user feature named by
//!   [`AvailabilityConfig::user_territory_feature`]; the long-format
//!   feature file names one-hot categoricals `<column>_<value>`, so the
//!   territory is the `<value>` suffix of the first feature whose name
//!   starts with `<column>_` and whose value is positive. Users without
//!   a territory see global rows only.
//! - **Unlisted items are ineligible.** The table defines availability;
//!   an item with no row is never shown to anyone and is reported in
//!   [`AvailabilityReport::num_items_without_availability`] so a
//!   partial table is visible rather than silent.
//! - **Reference time**: [`AvailabilityConfig::reference_days_ago`] when
//!   set (the temporal split's cutoff); otherwise per user, the oldest
//!   held-out interaction (the largest `days_ago` among the user's test
//!   rows), which is what leave-last-K-out needs. The per-user form
//!   requires a non-null `days_ago` column in the test file.

use super::{MetricsAtK, read_interactions_df};
use crate::data_pipeline::Mappings;
use ahash::AHashMap;
use anyhow::{Result, anyhow, bail};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Territory value that applies to every user.
pub const GLOBAL_TERRITORY: &str = "*";

/// Default item-age bucket edges in days: `< 30`, `30–365`, `>= 365`.
pub const DEFAULT_ITEM_AGE_BUCKET_EDGES: [f64; 2] = [30.0, 365.0];

/// Availability inputs for [`super::EvalConfig`].
#[derive(Debug, Clone)]
pub struct AvailabilityConfig {
    /// Path to the availability table (Parquet or CSV).
    pub path: String,
    /// Categorical user-feature *column* carrying territory (see module
    /// docs for the `<column>_<value>` convention). `None` treats every
    /// user as global.
    pub user_territory_feature: Option<String>,
    /// Global reference time in `days_ago`. `None` = per user from the
    /// test file's `days_ago` column.
    pub reference_days_ago: Option<f64>,
    /// Ascending bucket edges (days) for the item-age breakdown. Empty
    /// disables the breakdown.
    pub item_age_bucket_edges: Vec<f64>,
}

impl AvailabilityConfig {
    /// Global rows only, per-user reference time, default age buckets.
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            user_territory_feature: None,
            reference_days_ago: None,
            item_age_bucket_edges: DEFAULT_ITEM_AGE_BUCKET_EDGES.to_vec(),
        }
    }

    pub(crate) fn check(&self) -> Result<()> {
        if let Some(r) = self.reference_days_ago
            && !(r.is_finite() && r >= 0.0)
        {
            bail!("availability.reference_days_ago must be finite and >= 0, got {r}");
        }
        for w in self.item_age_bucket_edges.windows(2) {
            if !matches!(w[0].partial_cmp(&w[1]), Some(std::cmp::Ordering::Less)) {
                bail!(
                    "availability.item_age_bucket_edges must be strictly ascending, got {:?}",
                    self.item_age_bucket_edges
                );
            }
        }
        if let Some(e) = self.item_age_bucket_edges.first()
            && !(e.is_finite() && *e >= 0.0)
        {
            bail!("availability.item_age_bucket_edges must be finite and >= 0");
        }
        Ok(())
    }
}

/// One availability window in `days_ago` units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AvailabilityWindow {
    pub from_days_ago: f64,
    /// `None` = still available.
    pub to_days_ago: Option<f64>,
}

impl AvailabilityWindow {
    /// Whether the item was available at `reference_days_ago`.
    pub fn contains(&self, reference_days_ago: f64) -> bool {
        self.from_days_ago >= reference_days_ago
            && self.to_days_ago.is_none_or(|t| t < reference_days_ago)
    }
}

/// Parsed availability table, keyed by the model's item indices.
#[derive(Debug, Clone, Default)]
pub struct AvailabilityTable {
    /// item_idx -> territory -> windows.
    windows: AHashMap<usize, AHashMap<String, Vec<AvailabilityWindow>>>,
    /// item_idx -> earliest `available_from_days_ago` over all rows.
    first_available: AHashMap<usize, f64>,
    num_items_unlisted: usize,
    num_rows_unknown_item: usize,
}

impl AvailabilityTable {
    /// Load the table at `path`, mapping `item_id` through `mappings`.
    /// Rows for items the model does not know are counted and skipped.
    pub fn load(path: &str, mappings: &Mappings) -> Result<Self> {
        let df =
            read_interactions_df(path).map_err(|e| anyhow!("availability table {path}: {e}"))?;
        let item_col = df
            .column("item_id")
            .map_err(|_| anyhow!("availability table {path}: missing `item_id` column"))?
            .str()?
            .clone();
        let from_col = df
            .column("available_from_days_ago")
            .map_err(|_| {
                anyhow!("availability table {path}: missing `available_from_days_ago` column")
            })?
            .cast(&DataType::Float64)?
            .f64()?
            .clone();
        let to_col = match df.column("available_to_days_ago") {
            Ok(c) => Some(c.cast(&DataType::Float64)?.f64()?.clone()),
            Err(_) => None,
        };
        let territory_col = match df.column("territory") {
            Ok(c) => Some(c.str()?.clone()),
            Err(_) => None,
        };

        let mut table = AvailabilityTable::default();
        for i in 0..df.height() {
            let Some(iid) = item_col.get(i) else {
                continue;
            };
            let Some(&item_idx) = mappings.item_to_idx.get(iid) else {
                table.num_rows_unknown_item += 1;
                continue;
            };
            let Some(from) = from_col.get(i) else {
                bail!("availability table {path}: row {i} has a null `available_from_days_ago`");
            };
            if !from.is_finite() || from < 0.0 {
                bail!("availability table {path}: row {i} has available_from_days_ago={from}");
            }
            let to = to_col.as_ref().and_then(|c| c.get(i));
            if let Some(t) = to
                && (!t.is_finite() || t < 0.0 || t > from)
            {
                bail!(
                    "availability table {path}: row {i} window ends before it starts \
                     (available_from_days_ago={from}, available_to_days_ago={t})"
                );
            }
            let territory = territory_col
                .as_ref()
                .and_then(|c| c.get(i))
                .unwrap_or(GLOBAL_TERRITORY)
                .to_string();
            table
                .windows
                .entry(item_idx)
                .or_default()
                .entry(territory)
                .or_default()
                .push(AvailabilityWindow {
                    from_days_ago: from,
                    to_days_ago: to,
                });
            table
                .first_available
                .entry(item_idx)
                .and_modify(|f| {
                    if from > *f {
                        *f = from;
                    }
                })
                .or_insert(from);
        }
        table.num_items_unlisted = mappings
            .item_to_idx
            .len()
            .saturating_sub(table.windows.len());
        if table.num_items_unlisted > 0 {
            log::warn!(
                "availability table {path}: {} of {} catalog items have no availability row \
                 and are treated as never available",
                table.num_items_unlisted,
                mappings.item_to_idx.len()
            );
        }
        if table.num_rows_unknown_item > 0 {
            log::info!(
                "availability table {path}: skipped {} rows for items unknown to the model",
                table.num_rows_unknown_item
            );
        }
        Ok(table)
    }

    /// Number of catalog items with no availability row.
    pub fn num_items_unlisted(&self) -> usize {
        self.num_items_unlisted
    }

    /// Whether `item_idx` was available to a user in `territory` (or
    /// globally) at `reference_days_ago`.
    pub fn is_eligible(
        &self,
        item_idx: usize,
        territory: Option<&str>,
        reference_days_ago: f64,
    ) -> bool {
        let Some(by_territory) = self.windows.get(&item_idx) else {
            return false;
        };
        let hit = |key: &str| {
            by_territory
                .get(key)
                .is_some_and(|ws| ws.iter().any(|w| w.contains(reference_days_ago)))
        };
        hit(GLOBAL_TERRITORY) || territory.is_some_and(|t| t != GLOBAL_TERRITORY && hit(t))
    }

    /// All items eligible for `territory` at `reference_days_ago`.
    pub fn eligible_set(&self, territory: Option<&str>, reference_days_ago: f64) -> HashSet<usize> {
        self.windows
            .keys()
            .copied()
            .filter(|&idx| self.is_eligible(idx, territory, reference_days_ago))
            .collect()
    }

    /// Days between the item's first availability and `reference_days_ago`
    /// (negative = not yet released at the reference time). `None` for
    /// unlisted items.
    pub fn item_age_days(&self, item_idx: usize, reference_days_ago: f64) -> Option<f64> {
        self.first_available
            .get(&item_idx)
            .map(|first| first - reference_days_ago)
    }
}

/// Read `user_id -> territory` from a long-format user-features file.
///
/// A user's territory is the `<value>` suffix of the first feature named
/// `<feature_col>_<value>` with a positive value (one-hot convention of
/// the ingest's `to_long_format`).
pub fn load_user_territories(
    user_features_path: &str,
    feature_col: &str,
) -> Result<AHashMap<String, String>> {
    let df = read_interactions_df(user_features_path)?;
    let user_col = df.column("user_id")?.str()?;
    let feat_col = df.column("feature_name")?.str()?;
    let val_col = df.column("value")?.cast(&DataType::Float64)?;
    let val_col = val_col.f64()?;
    let prefix = format!("{feature_col}_");
    let mut out: AHashMap<String, String> = AHashMap::new();
    for ((user, feat), val) in user_col.into_iter().zip(feat_col).zip(val_col) {
        let (Some(u), Some(f), Some(v)) = (user, feat, val) else {
            continue;
        };
        if v <= 0.0 {
            continue;
        }
        if let Some(territory) = f.strip_prefix(&prefix)
            && !territory.is_empty()
        {
            out.entry(u.to_string())
                .or_insert_with(|| territory.to_string());
        }
    }
    if out.is_empty() {
        log::warn!(
            "user features {user_features_path}: no `{prefix}<value>` feature found; \
             every user is treated as global"
        );
    }
    Ok(out)
}

/// Metrics restricted to relevant items in one item-age bucket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemAgeBucket {
    /// Human-readable label, e.g. `"<30d"`, `"30-365d"`, `">=365d"`.
    pub label: String,
    pub min_age_days: f64,
    /// `None` = unbounded.
    pub max_age_days: Option<f64>,
    /// Users with at least one eligible relevant item in this bucket.
    pub num_users: usize,
    pub metrics_at_k: Vec<MetricsAtK>,
}

/// Availability section of [`super::EvalReport`]; present only when an
/// availability table was supplied.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailabilityReport {
    /// The global reference time, or `None` when it was per user.
    pub reference_days_ago: Option<f64>,
    /// Size of the union of evaluated users' eligible sets — the
    /// coverage denominator.
    pub num_eligible_items: usize,
    /// Catalog items with no availability row (never eligible).
    pub num_items_without_availability: usize,
    /// Test interactions on items ineligible for their user at the
    /// reference time; dropped from the relevant set, not counted as misses.
    pub num_test_interactions_dropped: usize,
    /// Test users with no eligible relevant item left after filtering.
    pub num_users_skipped: usize,
    /// Breakdown by item age at the reference time.
    pub item_age_buckets: Vec<ItemAgeBucket>,
}

/// Bucket index for `age` given ascending `edges`: `age < edges[0]` → 0,
/// `edges[i-1] <= age < edges[i]` → i, `age >= edges.last()` → `edges.len()`.
pub(crate) fn age_bucket(age: f64, edges: &[f64]) -> usize {
    edges.iter().take_while(|&&e| age >= e).count()
}

/// Labels and bounds for the buckets implied by `edges`.
pub(crate) fn bucket_bounds(edges: &[f64]) -> Vec<(String, f64, Option<f64>)> {
    if edges.is_empty() {
        return Vec::new();
    }
    let fmt = |d: f64| {
        if d.fract() == 0.0 {
            format!("{}", d as i64)
        } else {
            format!("{d}")
        }
    };
    let mut out = Vec::with_capacity(edges.len() + 1);
    out.push((format!("<{}d", fmt(edges[0])), 0.0, Some(edges[0])));
    for w in edges.windows(2) {
        out.push((format!("{}-{}d", fmt(w[0]), fmt(w[1])), w[0], Some(w[1])));
    }
    let last = *edges.last().unwrap();
    out.push((format!(">={}d", fmt(last)), last, None));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::df;
    use std::fs::File;
    use tempfile::TempDir;

    fn mappings(items: &[&str]) -> Mappings {
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
        for (i, it) in items.iter().enumerate() {
            m.item_to_idx.insert(it.to_string(), i);
            m.idx_to_item.insert(i, it.to_string());
        }
        m
    }

    fn write(df: &mut DataFrame, dir: &TempDir, name: &str) -> String {
        let p = dir.path().join(name);
        ParquetWriter::new(File::create(&p).unwrap())
            .finish(df)
            .unwrap();
        p.to_str().unwrap().to_string()
    }

    #[test]
    fn window_semantics_in_days_ago_units() {
        let w = AvailabilityWindow {
            from_days_ago: 100.0,
            to_days_ago: Some(10.0),
        };
        assert!(w.contains(100.0)); // released exactly at the reference time
        assert!(w.contains(50.0));
        assert!(!w.contains(10.0)); // ended exactly at the reference time
        assert!(!w.contains(5.0)); // after it left
        assert!(!w.contains(200.0)); // before release
        let open = AvailabilityWindow {
            from_days_ago: 100.0,
            to_days_ago: None,
        };
        assert!(open.contains(0.0));
        assert!(!open.contains(100.5));
    }

    #[test]
    fn load_rolls_up_seasons_and_territories() {
        let dir = TempDir::new().unwrap();
        let mut df = df!(
            "item_id" => ["s1", "s1", "s2", "s3", "zz"],
            "season_id" => ["s1e1", "s1e2", "s2e1", "s3e1", "zz"],
            "territory" => ["US", "US", "*", "EMEA", "*"],
            "available_from_days_ago" => [400.0_f64, 30.0, 200.0, 50.0, 1.0],
            "available_to_days_ago" => [Some(300.0_f64), None, None, None, None],
        )
        .unwrap();
        let path = write(&mut df, &dir, "av.parquet");
        let m = mappings(&["s1", "s2", "s3", "s4"]);
        let t = AvailabilityTable::load(&path, &m).unwrap();

        // s1 in US: season 1 covered 400..300, season 2 from 30 onward.
        assert!(t.is_eligible(0, Some("US"), 350.0));
        assert!(!t.is_eligible(0, Some("US"), 100.0)); // between the seasons
        assert!(t.is_eligible(0, Some("US"), 10.0));
        assert!(!t.is_eligible(0, Some("EMEA"), 10.0)); // US-only rows
        assert!(!t.is_eligible(0, None, 10.0));
        // s2 is global.
        assert!(t.is_eligible(1, None, 100.0));
        assert!(t.is_eligible(1, Some("EMEA"), 100.0));
        assert!(!t.is_eligible(1, Some("EMEA"), 250.0)); // not yet released
        // s3 EMEA only; s4 unlisted.
        assert!(t.is_eligible(2, Some("EMEA"), 0.0));
        assert!(!t.is_eligible(3, Some("EMEA"), 0.0));
        assert_eq!(t.num_items_unlisted(), 1);
        assert_eq!(t.num_rows_unknown_item, 1);

        assert_eq!(t.eligible_set(Some("US"), 10.0), HashSet::from([0, 1]));
        assert_eq!(t.eligible_set(None, 10.0), HashSet::from([1]));
        // Age at reference time 10: s1 first available 400 days ago -> 390.
        assert_eq!(t.item_age_days(0, 10.0), Some(390.0));
        assert_eq!(t.item_age_days(3, 10.0), None);
    }

    #[test]
    fn load_without_optional_columns_is_global_and_open_ended() {
        let dir = TempDir::new().unwrap();
        let mut df = df!(
            "item_id" => ["a", "b"],
            "available_from_days_ago" => [10_i64, 5],
        )
        .unwrap();
        let path = write(&mut df, &dir, "av.parquet");
        let m = mappings(&["a", "b"]);
        let t = AvailabilityTable::load(&path, &m).unwrap();
        assert!(t.is_eligible(0, Some("JP"), 7.0));
        assert!(!t.is_eligible(1, Some("JP"), 7.0));
        assert_eq!(t.num_items_unlisted(), 0);
    }

    #[test]
    fn load_rejects_inverted_window_and_missing_columns() {
        let dir = TempDir::new().unwrap();
        let mut bad = df!(
            "item_id" => ["a"],
            "available_from_days_ago" => [10.0_f64],
            "available_to_days_ago" => [20.0_f64],
        )
        .unwrap();
        let path = write(&mut bad, &dir, "bad.parquet");
        let err = AvailabilityTable::load(&path, &mappings(&["a"])).unwrap_err();
        assert!(err.to_string().contains("ends before it starts"), "{err}");

        let mut missing = df!("item_id" => ["a"], "territory" => ["*"]).unwrap();
        let path = write(&mut missing, &dir, "missing.parquet");
        let err = AvailabilityTable::load(&path, &mappings(&["a"])).unwrap_err();
        assert!(err.to_string().contains("available_from_days_ago"), "{err}");
    }

    #[test]
    fn user_territories_follow_one_hot_naming() {
        let dir = TempDir::new().unwrap();
        let mut df = df!(
            "user_id" => ["u1", "u1", "u2", "u3"],
            "feature_name" => ["plan_Premium", "region_US", "region_EMEA", "region_JP"],
            "value" => [1.0_f64, 1.0, 1.0, 0.0],
        )
        .unwrap();
        let path = write(&mut df, &dir, "uf.parquet");
        let t = load_user_territories(&path, "region").unwrap();
        assert_eq!(t.get("u1").map(String::as_str), Some("US"));
        assert_eq!(t.get("u2").map(String::as_str), Some("EMEA"));
        assert!(!t.contains_key("u3")); // zero-valued one-hot is not a territory
    }

    #[test]
    fn buckets_and_labels() {
        let edges = [30.0, 365.0];
        assert_eq!(age_bucket(0.0, &edges), 0);
        assert_eq!(age_bucket(29.9, &edges), 0);
        assert_eq!(age_bucket(30.0, &edges), 1);
        assert_eq!(age_bucket(364.0, &edges), 1);
        assert_eq!(age_bucket(365.0, &edges), 2);
        assert_eq!(age_bucket(-5.0, &edges), 0);
        let b = bucket_bounds(&edges);
        assert_eq!(
            b.iter().map(|x| x.0.as_str()).collect::<Vec<_>>(),
            ["<30d", "30-365d", ">=365d"]
        );
        assert_eq!(b[1].1, 30.0);
        assert_eq!(b[2].2, None);
        assert!(bucket_bounds(&[]).is_empty());
    }

    #[test]
    fn config_check_rejects_bad_edges_and_reference() {
        let mut c = AvailabilityConfig::new("x");
        assert!(c.check().is_ok());
        c.item_age_bucket_edges = vec![365.0, 30.0];
        assert!(c.check().is_err());
        c.item_age_bucket_edges = vec![];
        c.reference_days_ago = Some(-1.0);
        assert!(c.check().is_err());
    }
}
