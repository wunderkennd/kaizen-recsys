//! Sequence data path for SASRec (ADR-0001 Phase 3, issue #36).
//!
//! SASRec consumes per-user, chronologically-ordered item histories. This
//! module turns the long-format interactions table into fixed-length,
//! left-padded causal training sequences.
//!
//! ## Padding / vocab convention
//!
//! Token `0` is reserved as the padding id. A catalog item with index
//! `i` (from [`crate::data_pipeline::Mappings`], `0..num_items`) maps to
//! token `i + 1`. The model vocab size is therefore `num_items + 1`.
//! Left-padding keeps the most recent item at the last position, which is
//! the position SASRec scores at inference time.
//!
//! ## `days_ago` is mandatory
//!
//! SASRec is order-sensitive: shuffling a user's history changes the
//! target. ADR-0001 §Risks records the decision to **fail loudly** rather
//! than silently fall back to file/row order. [`build_sequences`] returns
//! an error if the `days_ago` column is missing or not numeric. Smaller
//! `days_ago` == more recent (consistent with `weighting::apply_temporal_decay`).

// Phase 3 builds the sequence data path; its only non-test consumer is
// the SASRec PyO3 class, which lands in Phase 4 (issue #37). Until then
// `build_sequences` / `item_to_token` / the `vocab_size` field have no
// in-crate caller outside tests. Mirrors the same allow + rationale in
// `models/mod.rs`; it is removed when Phase 4 wires the consumers.
#![allow(dead_code)]

use anyhow::{Context, Result, bail};
use polars::prelude::*;
use std::path::Path;

use crate::data_pipeline::Mappings;

/// The padding token id. Reserved; never a real item.
pub const PAD_TOKEN: usize = 0;

/// A batch of fixed-length, left-padded causal sequences plus their
/// next-item targets.
///
/// Both `inputs` and `targets` are `n_sequences * seq_len` row-major.
/// `targets[k]` is the item that follows `inputs[k]` in the user's
/// history; positions with no successor (and all padding positions) carry
/// [`PAD_TOKEN`] and are masked out of the loss by the training step.
#[derive(Debug, Clone)]
pub struct SequenceDataset {
    /// Left-padded item-token sequences, `n_sequences * seq_len`.
    pub inputs: Vec<i64>,
    /// Next-item targets aligned to `inputs`, `n_sequences * seq_len`.
    pub targets: Vec<i64>,
    /// Fixed sequence length (a.k.a. `max_seq_len`).
    pub seq_len: usize,
    /// Vocab size including the pad token: `num_items + 1`.
    pub vocab_size: usize,
}

impl SequenceDataset {
    /// Number of sequences (one per user with >= 2 interactions).
    pub fn len(&self) -> usize {
        self.inputs.len().checked_div(self.seq_len).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Row-major slice of input tokens for sequence `i`.
    pub fn input_row(&self, i: usize) -> &[i64] {
        &self.inputs[i * self.seq_len..(i + 1) * self.seq_len]
    }

    /// Row-major slice of target tokens for sequence `i`.
    pub fn target_row(&self, i: usize) -> &[i64] {
        &self.targets[i * self.seq_len..(i + 1) * self.seq_len]
    }
}

/// Convert a catalog item index (`0..num_items`) to a sequence token.
#[inline]
pub fn item_to_token(item_idx: usize) -> i64 {
    (item_idx + 1) as i64
}

/// Build left-padded causal training sequences from an interactions file.
///
/// Reads `user_id`, `item_id`, and the **required** `days_ago` column,
/// groups interactions by user, sorts each user's history oldest-first by
/// `days_ago` (descending: larger `days_ago` is older), and emits one
/// fixed-length left-padded sequence per user with the next-item target
/// (input shifted by one).
///
/// `item_to_idx` is the catalog mapping from the data pipeline so tokens
/// agree with the model's embedding table. Items not present in the
/// mapping are skipped (mirrors `data_pipeline::build_triplets`).
///
/// Users with fewer than two in-catalog interactions are dropped — a
/// single item has no next-item target to learn from.
pub fn build_sequences(
    interactions_path: &str,
    mappings: &Mappings,
    seq_len: usize,
) -> Result<SequenceDataset> {
    if seq_len == 0 {
        bail!("build_sequences: seq_len must be >= 1");
    }

    let df = read_interactions(interactions_path)?;

    let user_col = df
        .column("user_id")
        .context("interactions file is missing the required `user_id` column")?
        .str()
        .context("`user_id` column must be Utf8/String")?;
    let item_col = df
        .column("item_id")
        .context("interactions file is missing the required `item_id` column")?
        .str()
        .context("`item_id` column must be Utf8/String")?;

    // SASRec hard-requires `days_ago` for chronological ordering. ADR-0001
    // §Risks: fail loudly rather than silently use row order.
    let days_col = match df.column("days_ago") {
        Ok(c) => c.f64().map_err(|_| {
            anyhow::anyhow!(
                "SASRec requires a numeric `days_ago` column for sequence ordering, \
                 but `days_ago` has dtype {:?}. Provide `days_ago` as Float64.",
                c.dtype()
            )
        })?,
        Err(_) => bail!(
            "SASRec requires a `days_ago` column in the interactions file to order \
             each user's history chronologically; it is absent. (ADR-0001 §Risks: \
             we fail loudly rather than silently fall back to row order.)"
        ),
    };

    // Group (item_token, days_ago) per user, preserving catalog mapping.
    let mut per_user: ahash::AHashMap<&str, Vec<(f64, i64)>> = ahash::AHashMap::new();
    for ((u, it), d) in user_col.into_iter().zip(item_col).zip(days_col) {
        let (Some(u), Some(it), Some(d)) = (u, it, d) else {
            continue;
        };
        let Some(&item_idx) = mappings.item_to_idx.get(it) else {
            continue;
        };
        per_user
            .entry(u)
            .or_default()
            .push((d, item_to_token(item_idx)));
    }

    let vocab_size = mappings.idx_to_item.len() + 1;

    // Deterministic order: sort users by id before building rows so the
    // dataset (and any downstream RNG) is reproducible. AHashMap iteration
    // order is randomized; the rest of the crate sorts keys for the same
    // reason (see CLAUDE.md §Evaluation pipeline).
    let mut users: Vec<&str> = per_user.keys().copied().collect();
    users.sort_unstable();

    let mut inputs: Vec<i64> = Vec::new();
    let mut targets: Vec<i64> = Vec::new();

    for user in users {
        let hist = per_user.get_mut(user).expect("user key just collected");
        if hist.len() < 2 {
            continue;
        }
        // Oldest first: larger `days_ago` == further in the past. Stable
        // sort keeps ties in their original (file) relative order.
        hist.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(std::cmp::Ordering::Equal)
        });

        let items: Vec<i64> = hist.iter().map(|(_, tok)| *tok).collect();

        // Keep the most recent `seq_len + 1` items: `seq_len` inputs plus
        // the final target. Left-pad with PAD when the history is shorter.
        let n = items.len();
        let take = (seq_len + 1).min(n);
        let recent = &items[n - take..];

        let mut row_in = vec![PAD_TOKEN as i64; seq_len];
        let mut row_tgt = vec![PAD_TOKEN as i64; seq_len];

        // recent = [.. , x_{t-1}, x_t]; inputs are recent[..len-1],
        // targets are recent[1..] (shift by one). Right-align both so the
        // newest input sits at the last position.
        let in_len = recent.len() - 1;
        let start = seq_len - in_len;
        row_in[start..start + in_len].copy_from_slice(&recent[..in_len]);
        row_tgt[start..start + in_len].copy_from_slice(&recent[1..1 + in_len]);

        inputs.extend_from_slice(&row_in);
        targets.extend_from_slice(&row_tgt);
    }

    Ok(SequenceDataset {
        inputs,
        targets,
        seq_len,
        vocab_size,
    })
}

/// Read a Parquet or CSV interactions file into a DataFrame.
///
/// Mirrors `data_pipeline::read_lazyframe` so the two paths accept the
/// same files; kept local to keep that module's API private.
fn read_interactions(path_str: &str) -> Result<DataFrame> {
    let path = Path::new(path_str);
    match path.extension().and_then(|s| s.to_str()) {
        Some("parquet") => Ok(ParquetReader::new(std::fs::File::open(path)?).finish()?),
        Some("csv") => Ok(CsvReader::new(std::fs::File::open(path)?).finish()?),
        _ => bail!(
            "Unsupported file type for {}. Supported: .parquet, .csv",
            path_str
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_pipeline::Mappings;

    fn mappings_with_items(items: &[&str]) -> Mappings {
        let mut item_to_idx = ahash::AHashMap::new();
        let mut idx_to_item = Vec::new();
        for (i, it) in items.iter().enumerate() {
            item_to_idx.insert(it.to_string(), i);
            idx_to_item.push(it.to_string());
        }
        Mappings {
            user_to_idx: Default::default(),
            idx_to_user: Default::default(),
            item_to_idx,
            idx_to_item,
            user_feature_to_idx: Default::default(),
            idx_to_user_feature: Default::default(),
            item_feature_to_idx: Default::default(),
            idx_to_item_feature: Default::default(),
        }
    }

    fn write_csv(dir: &Path, name: &str, body: &str) -> String {
        let p = dir.join(name);
        std::fs::write(&p, body).unwrap();
        p.to_str().unwrap().to_string()
    }

    #[test]
    fn missing_days_ago_fails_loudly() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "i.csv",
            "user_id,item_id,value\nu1,a,1.0\nu1,b,1.0\n",
        );
        let m = mappings_with_items(&["a", "b"]);
        let err = build_sequences(&path, &m, 4).unwrap_err();
        assert!(
            err.to_string().contains("days_ago"),
            "error must mention days_ago, got: {err}"
        );
    }

    #[test]
    fn sequences_are_chronological_and_left_padded() {
        let dir = tempfile::tempdir().unwrap();
        // u1: a (3 days ago) -> b (2) -> c (1). Oldest first = a,b,c.
        let path = write_csv(
            dir.path(),
            "i.csv",
            "user_id,item_id,value,days_ago\n\
             u1,c,1.0,1.0\n\
             u1,a,1.0,3.0\n\
             u1,b,1.0,2.0\n",
        );
        let m = mappings_with_items(&["a", "b", "c"]);
        let ds = build_sequences(&path, &m, 4).unwrap();
        assert_eq!(ds.len(), 1);
        assert_eq!(ds.vocab_size, 4); // 3 items + pad
        // tokens: a=1, b=2, c=3. recent = [a,b,c]; inputs=[a,b], targets=[b,c]
        // left-padded into width 4: [0,0,a,b] / [0,0,b,c]
        assert_eq!(ds.input_row(0), &[0, 0, 1, 2]);
        assert_eq!(ds.target_row(0), &[0, 0, 2, 3]);
    }

    #[test]
    fn single_interaction_user_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "i.csv",
            "user_id,item_id,value,days_ago\nu1,a,1.0,1.0\n",
        );
        let m = mappings_with_items(&["a", "b"]);
        let ds = build_sequences(&path, &m, 4).unwrap();
        assert!(ds.is_empty());
    }

    #[test]
    fn long_history_truncates_to_most_recent() {
        let dir = tempfile::tempdir().unwrap();
        // 5 items, seq_len=2 -> keep most recent 3 (2 inputs + 1 target).
        let path = write_csv(
            dir.path(),
            "i.csv",
            "user_id,item_id,value,days_ago\n\
             u1,a,1.0,5.0\n\
             u1,b,1.0,4.0\n\
             u1,c,1.0,3.0\n\
             u1,d,1.0,2.0\n\
             u1,e,1.0,1.0\n",
        );
        let m = mappings_with_items(&["a", "b", "c", "d", "e"]);
        let ds = build_sequences(&path, &m, 2).unwrap();
        // recent = [c,d,e] => inputs=[c,d], targets=[d,e]. c=3,d=4,e=5
        assert_eq!(ds.input_row(0), &[3, 4]);
        assert_eq!(ds.target_row(0), &[4, 5]);
    }
}
