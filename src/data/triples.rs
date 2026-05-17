//! Two-Tower training data: `(user, positive-item)` triples plus a
//! categorical + dense feature loader.
//!
//! ADR-0001 Phase 5 (issue #38). The Two-Tower model consumes, per side
//! (user and item):
//!   * a primary index into an embedding table (the user / item id), and
//!   * a list of *categorical* feature indices (each looked up in a shared
//!     feature embedding table), and
//!   * a vector of *dense* numeric features (fed straight into the tower MLP).
//!
//! The long-format pipeline in `crate::data_pipeline` only produces sparse
//! `value`-weighted features, which collapses categorical/dense into one
//! sparse matrix. Two-Tower needs them kept apart, so this module reads the
//! long-format tables directly via Polars (same reader idiom as
//! `data_pipeline::read_lazyframe`) and builds:
//!   * [`TripleData`] — the `(user_idx, item_idx)` positive pairs, and
//!   * [`FeatureTable`] — per-entity categorical-index lists + dense vectors.
//!
//! Everything here is `ml-models`-gated; EASE-only builds never see it.

// The loaders and `FeatureTable` accessors are exercised by this module's
// tests and by `models::two_tower`; no non-test in-crate caller invokes
// them directly, so the public surface is dead-code allowed (mirrors the
// same allow in `models/mod.rs`).
#![allow(dead_code)]

use ahash::AHashMap;
use anyhow::{Context, Result};
use polars::prelude::*;
use std::fs::File;
use std::path::Path;

/// A single positive training example: user `u` interacted with item `i`.
/// Indices are into the dense user / item embedding tables (0-based,
/// contiguous), assigned in first-seen order while reading interactions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Triple {
    pub user_idx: usize,
    pub item_idx: usize,
}

/// The full set of training triples plus the id↔index mappings needed to
/// translate Python-side string ids and to size the embedding tables.
#[derive(Debug, Clone)]
pub struct TripleData {
    pub triples: Vec<Triple>,
    pub user_to_idx: AHashMap<String, usize>,
    pub idx_to_user: Vec<String>,
    pub item_to_idx: AHashMap<String, usize>,
    pub idx_to_item: Vec<String>,
}

impl TripleData {
    pub fn num_users(&self) -> usize {
        self.idx_to_user.len()
    }

    pub fn num_items(&self) -> usize {
        self.idx_to_item.len()
    }
}

/// Per-entity feature rows for one tower side (users *or* items).
///
/// `cat[e]` is the list of categorical-feature indices active for entity
/// `e` (indices into a shared feature-embedding table of size
/// [`FeatureTable::num_categories`]). `dense[e]` is that entity's dense
/// numeric vector, length [`FeatureTable::dense_dim`] (zero-filled where a
/// feature is absent). Entities with no feature row get an empty `cat`
/// list and an all-zero dense vector — the model still embeds them by id,
/// so cold-start by features degrades gracefully rather than panicking.
#[derive(Debug, Clone)]
pub struct FeatureTable {
    pub cat: Vec<Vec<usize>>,
    pub dense: Vec<Vec<f32>>,
    pub num_categories: usize,
    pub dense_dim: usize,
    /// `feature_name` → categorical slot, in first-seen order.
    pub cat_feature_to_idx: AHashMap<String, usize>,
    /// `feature_name` → dense column, in first-seen order.
    pub dense_feature_to_idx: AHashMap<String, usize>,
}

impl FeatureTable {
    /// An empty table for `n` entities: no categories, no dense columns.
    /// Used when the caller supplies no feature file for a side.
    pub fn empty(n: usize) -> Self {
        Self {
            cat: vec![Vec::new(); n],
            dense: vec![Vec::new(); n],
            num_categories: 0,
            dense_dim: 0,
            cat_feature_to_idx: AHashMap::new(),
            dense_feature_to_idx: AHashMap::new(),
        }
    }
}

/// Reads a Parquet or CSV file into a Polars `DataFrame`.
///
/// Mirrors `data_pipeline::read_lazyframe`'s extension dispatch so the
/// Two-Tower path accepts exactly the same file types as EASE.
fn read_dataframe(path_str: &str) -> Result<DataFrame> {
    let path = Path::new(path_str);
    let extension = path.extension().and_then(|s| s.to_str());
    let df = match extension {
        Some("parquet") => ParquetReader::new(File::open(path)?).finish()?,
        Some("csv") => CsvReader::new(File::open(path)?).finish()?,
        _ => {
            anyhow::bail!("Unsupported file type for {path_str}. Supported: .parquet and .csv")
        }
    };
    Ok(df)
}

/// Pull a string column out of a `DataFrame`, casting non-string columns
/// (e.g. integer ids) to UTF-8 so callers can use numeric ids too.
fn str_column(df: &DataFrame, name: &str) -> Result<Vec<String>> {
    let s = df
        .column(name)
        .with_context(|| format!("missing required column `{name}`"))?
        .cast(&DataType::String)
        .with_context(|| format!("column `{name}` is not castable to string"))?;
    let ca = s.str()?;
    Ok(ca
        .into_iter()
        .map(|o| o.unwrap_or("").to_string())
        .collect())
}

/// Pull an f64 column, casting from any numeric dtype.
fn f64_column(df: &DataFrame, name: &str) -> Result<Vec<f64>> {
    let s = df
        .column(name)
        .with_context(|| format!("missing required column `{name}`"))?
        .cast(&DataType::Float64)
        .with_context(|| format!("column `{name}` is not numeric"))?;
    let ca = s.f64()?;
    Ok(ca.into_iter().map(|o| o.unwrap_or(0.0)).collect())
}

/// Insert `key` into `map`/`vec` if absent, returning its index.
fn intern(map: &mut AHashMap<String, usize>, vec: &mut Vec<String>, key: &str) -> usize {
    if let Some(&i) = map.get(key) {
        return i;
    }
    let i = vec.len();
    map.insert(key.to_string(), i);
    vec.push(key.to_string());
    i
}

/// Load `(user_id, item_id)` positive pairs from a long-format interactions
/// table (`user_id`, `item_id`, optional `value`). A `value <= 0` row is
/// treated as a non-positive and dropped — Two-Tower's in-batch softmax
/// only consumes positives. Indices are assigned in first-seen order to
/// keep runs deterministic given a deterministic file.
pub fn load_triples(interactions_path: &str) -> Result<TripleData> {
    let df = read_dataframe(interactions_path)?;
    let users = str_column(&df, "user_id")?;
    let items = str_column(&df, "item_id")?;
    // `value` is optional; default every row to a positive (1.0).
    let values = match df.column("value") {
        Ok(_) => f64_column(&df, "value")?,
        Err(_) => vec![1.0; users.len()],
    };

    if users.len() != items.len() || users.len() != values.len() {
        anyhow::bail!("interactions columns have mismatched lengths");
    }

    let mut user_to_idx = AHashMap::new();
    let mut idx_to_user = Vec::new();
    let mut item_to_idx = AHashMap::new();
    let mut idx_to_item = Vec::new();
    let mut triples = Vec::with_capacity(users.len());

    for ((u, i), v) in users.iter().zip(items.iter()).zip(values.iter()) {
        if *v <= 0.0 {
            continue;
        }
        let user_idx = intern(&mut user_to_idx, &mut idx_to_user, u);
        let item_idx = intern(&mut item_to_idx, &mut idx_to_item, i);
        triples.push(Triple { user_idx, item_idx });
    }

    if triples.is_empty() {
        anyhow::bail!("no positive interactions found in {interactions_path}");
    }

    Ok(TripleData {
        triples,
        user_to_idx,
        idx_to_user,
        item_to_idx,
        idx_to_item,
    })
}

/// Load a long-format feature table (`<id_col>`, `feature_name`, `value`)
/// into a [`FeatureTable`] sized for `num_entities` (rows are indexed by
/// `entity_to_idx`). A feature is treated as **categorical** when its
/// `value` equals 1.0 *and* it never appears with any other value across
/// the file (classic one-hot indicator); otherwise it is a **dense**
/// numeric column. This split is what lets the Two-Tower towers route
/// one-hot side info through embeddings and continuous side info through
/// the MLP, per ADR-0001 §Context.
///
/// Unknown ids (not in `entity_to_idx`) are skipped — feature rows for
/// entities with no interaction can't be placed in the embedding table.
pub fn load_features(
    features_path: &str,
    id_col: &str,
    entity_to_idx: &AHashMap<String, usize>,
    num_entities: usize,
) -> Result<FeatureTable> {
    let df = read_dataframe(features_path)?;
    let ids = str_column(&df, id_col)?;
    let names = str_column(&df, "feature_name")?;
    let values = f64_column(&df, "value")?;

    if ids.len() != names.len() || ids.len() != values.len() {
        anyhow::bail!("feature columns have mismatched lengths");
    }

    // First pass: decide categorical vs dense per feature_name.
    // Categorical iff every observed value is exactly 1.0.
    let mut all_one: AHashMap<String, bool> = AHashMap::new();
    for (n, v) in names.iter().zip(values.iter()) {
        let e = all_one.entry(n.clone()).or_insert(true);
        if (*v - 1.0).abs() > f64::EPSILON {
            *e = false;
        }
    }

    let mut cat_feature_to_idx: AHashMap<String, usize> = AHashMap::new();
    let mut dense_feature_to_idx: AHashMap<String, usize> = AHashMap::new();
    // Assign slots in first-seen order for determinism.
    for n in &names {
        let categorical = *all_one.get(n).unwrap_or(&true);
        if categorical {
            let next = cat_feature_to_idx.len();
            cat_feature_to_idx.entry(n.clone()).or_insert(next);
        } else {
            let next = dense_feature_to_idx.len();
            dense_feature_to_idx.entry(n.clone()).or_insert(next);
        }
    }

    let num_categories = cat_feature_to_idx.len();
    let dense_dim = dense_feature_to_idx.len();

    let mut cat = vec![Vec::new(); num_entities];
    let mut dense = vec![vec![0.0_f32; dense_dim]; num_entities];

    for ((id, name), value) in ids.iter().zip(names.iter()).zip(values.iter()) {
        let Some(&e) = entity_to_idx.get(id) else {
            continue;
        };
        if e >= num_entities {
            continue;
        }
        if let Some(&c) = cat_feature_to_idx.get(name) {
            // De-duplicate: a one-hot indicator that repeats adds nothing.
            if !cat[e].contains(&c) {
                cat[e].push(c);
            }
        } else if let Some(&d) = dense_feature_to_idx.get(name) {
            dense[e][d] = *value as f32;
        }
    }

    Ok(FeatureTable {
        cat,
        dense,
        num_categories,
        dense_dim,
        cat_feature_to_idx,
        dense_feature_to_idx,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::df;
    use std::io::Write;

    fn write_csv(name: &str, contents: &str) -> String {
        let dir = std::env::temp_dir();
        let path = dir.join(name);
        let mut f = File::create(&path).unwrap();
        f.write_all(contents.as_bytes()).unwrap();
        path.to_str().unwrap().to_string()
    }

    fn write_parquet(name: &str, df: &mut DataFrame) -> String {
        let dir = std::env::temp_dir();
        let path = dir.join(name);
        let mut f = File::create(&path).unwrap();
        ParquetWriter::new(&mut f).finish(df).unwrap();
        path.to_str().unwrap().to_string()
    }

    #[test]
    fn load_triples_assigns_contiguous_indices_and_drops_nonpositive() {
        let path = write_csv(
            "tt_triples.csv",
            "user_id,item_id,value\nu1,i1,1.0\nu1,i2,1.0\nu2,i1,1.0\nu2,i3,0.0\n",
        );
        let data = load_triples(&path).unwrap();

        // 2 users, 2 distinct items SEEN on positive rows (i3 row dropped).
        assert_eq!(data.num_users(), 2);
        assert_eq!(data.num_items(), 2);
        // 3 positive triples (the value=0.0 row is filtered out).
        assert_eq!(data.triples.len(), 3);
        assert_eq!(data.user_to_idx["u1"], 0);
        assert_eq!(data.item_to_idx["i1"], 0);
        assert_eq!(data.item_to_idx["i2"], 1);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_triples_value_column_optional() {
        let path = write_csv("tt_triples_noval.csv", "user_id,item_id\nu1,i1\nu1,i2\n");
        let data = load_triples(&path).unwrap();
        assert_eq!(data.triples.len(), 2);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_features_splits_categorical_and_dense() {
        // age_group is one-hot (all values 1.0) -> categorical.
        // tenure_days has a value != 1.0 -> dense.
        let mut df = df!(
            "user_id" => ["u1", "u1", "u2", "u2"],
            "feature_name" => ["age_20s", "tenure_days", "age_30s", "tenure_days"],
            "value" => [1.0_f64, 365.0, 1.0, 30.0],
        )
        .unwrap();
        let path = write_parquet("tt_feats.parquet", &mut df);

        let mut entity_to_idx = AHashMap::new();
        entity_to_idx.insert("u1".to_string(), 0_usize);
        entity_to_idx.insert("u2".to_string(), 1_usize);

        let ft = load_features(&path, "user_id", &entity_to_idx, 2).unwrap();

        // Two distinct one-hot indicators -> 2 categories.
        assert_eq!(ft.num_categories, 2);
        // One continuous feature -> dense_dim 1.
        assert_eq!(ft.dense_dim, 1);

        let age20s = ft.cat_feature_to_idx["age_20s"];
        let age30s = ft.cat_feature_to_idx["age_30s"];
        assert_eq!(ft.cat[0], vec![age20s]);
        assert_eq!(ft.cat[1], vec![age30s]);

        let tenure = ft.dense_feature_to_idx["tenure_days"];
        assert!((ft.dense[0][tenure] - 365.0).abs() < 1e-6);
        assert!((ft.dense[1][tenure] - 30.0).abs() < 1e-6);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_features_skips_unknown_ids_and_handles_missing_rows() {
        let path = write_csv(
            "tt_feats_unknown.csv",
            "item_id,feature_name,value\ni1,genre_rock,1.0\nUNKNOWN,genre_pop,1.0\n",
        );
        let mut entity_to_idx = AHashMap::new();
        entity_to_idx.insert("i1".to_string(), 0_usize);
        entity_to_idx.insert("i2".to_string(), 1_usize);

        let ft = load_features(&path, "item_id", &entity_to_idx, 2).unwrap();

        // genre_rock for i1; genre_pop for UNKNOWN is dropped, but its
        // feature slot is still allocated (first-seen over the file).
        assert!(!ft.cat[0].is_empty());
        // i2 has no feature row -> empty cat list, all-zero dense.
        assert!(ft.cat[1].is_empty());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn empty_feature_table_is_well_formed() {
        let ft = FeatureTable::empty(3);
        assert_eq!(ft.cat.len(), 3);
        assert_eq!(ft.dense.len(), 3);
        assert_eq!(ft.num_categories, 0);
        assert_eq!(ft.dense_dim, 0);
    }
}
