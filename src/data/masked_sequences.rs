//! Masked-sequence data path for BERT4Rec (issue #96).
//!
//! BERT4Rec (Sun et al., CIKM 2019) trains a *bidirectional* transformer
//! with a Cloze objective: random positions in each user's history are
//! replaced by a `[MASK]` token and the model must recover the original
//! item from both its left and right context. This module turns the
//! long-format interactions table into fixed-length, left-padded masked
//! training examples. It is the bidirectional sibling of
//! [`crate::data::sequences`] (SASRec's causal data path) and shares its
//! file reader, `days_ago` requirement, and deterministic user ordering.
//!
//! ## Token vocabulary
//!
//! | token            | meaning                              |
//! |------------------|--------------------------------------|
//! | `0`              | `PAD` (same as SASRec)               |
//! | `1`              | `[MASK]` — the prediction placeholder |
//! | `2..vocab_size`  | catalog item `item_idx + 2`          |
//!
//! so `vocab_size = num_items + 2`.
//!
//! ## Positions
//!
//! Instead of absolute sequence offsets, each token carries a
//! log₂-bucketed `days_ago` ([`days_ago_to_bucket`]): two items watched in
//! the same doubling window share a bucket regardless of absolute date.
//! A 5-year history fits in ~11 buckets while the last few days keep
//! day-level resolution. Padding positions carry bucket `0` and are
//! excluded from attention by the model's pad mask.
//!
//! ## `days_ago` is mandatory
//!
//! Like SASRec, this path **fails loudly** if `days_ago` is missing or not
//! numeric (ADR-0001 §Risks) rather than silently inventing positions.

// `build_masked_sequences` and friends are consumed by `models::bert4rec`
// and the PyO3 layer; several accessors exist for symmetry with
// `SequenceDataset` and are test-only today.
#![allow(dead_code)]

use anyhow::{Result, bail};
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;

use crate::data::sequences::read_interactions;
use crate::data_pipeline::Mappings;

pub use crate::data::days_ago_to_bucket;

/// The padding token id. Reserved; never a real item. Equal to
/// [`crate::data::sequences::PAD_TOKEN`].
pub const PAD_TOKEN: i64 = 0;

/// The `[MASK]` token id: the Cloze prediction placeholder.
pub const MASK_TOKEN: i64 = 1;

/// Number of reserved tokens (`PAD`, `MASK`) in front of the catalog.
pub const NUM_RESERVED_TOKENS: usize = 2;

/// Default share of non-pad positions replaced by `[MASK]`.
pub const DEFAULT_MASK_RATIO: f64 = 0.2;

/// Convert a catalog item index (`0..num_items`) to a sequence token.
#[inline]
pub fn item_to_token(item_idx: usize) -> i64 {
    (item_idx + NUM_RESERVED_TOKENS) as i64
}

/// Inverse of [`item_to_token`]. `None` for `PAD` / `MASK`.
#[inline]
pub fn token_to_item(token: i64) -> Option<usize> {
    usize::try_from(token)
        .ok()
        .and_then(|t| t.checked_sub(NUM_RESERVED_TOKENS))
}

/// A batch of fixed-length, left-padded masked sequences.
///
/// All four token arrays are `n_sequences * seq_len` row-major and
/// aligned position-for-position:
///
/// - `original[k]` — the real token before masking (loss target),
/// - `inputs[k]` — what the model sees (`MASK_TOKEN` where masked),
/// - `mask[k]` — `1` at masked positions, `0` elsewhere (pad included),
/// - `positions[k]` — log₂ `days_ago` bucket (`0` at pad positions).
#[derive(Debug, Clone)]
pub struct MaskedSequenceDataset {
    pub original: Vec<i64>,
    pub inputs: Vec<i64>,
    pub mask: Vec<i64>,
    pub positions: Vec<i64>,
    /// Fixed sequence length (a.k.a. `max_seq_len`).
    pub seq_len: usize,
    /// Vocab size including `PAD` and `MASK`: `num_items + 2`.
    pub vocab_size: usize,
}

impl MaskedSequenceDataset {
    /// Number of sequences (one per user with >= 2 interactions).
    pub fn len(&self) -> usize {
        self.inputs.len().checked_div(self.seq_len).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn row<'a>(&self, v: &'a [i64], i: usize) -> &'a [i64] {
        &v[i * self.seq_len..(i + 1) * self.seq_len]
    }

    /// Row-major slice of unmasked tokens for sequence `i`.
    pub fn original_row(&self, i: usize) -> &[i64] {
        self.row(&self.original, i)
    }

    /// Row-major slice of masked input tokens for sequence `i`.
    pub fn input_row(&self, i: usize) -> &[i64] {
        self.row(&self.inputs, i)
    }

    /// Row-major slice of mask indicators for sequence `i`.
    pub fn mask_row(&self, i: usize) -> &[i64] {
        self.row(&self.mask, i)
    }

    /// Row-major slice of position buckets for sequence `i`.
    pub fn position_row(&self, i: usize) -> &[i64] {
        self.row(&self.positions, i)
    }
}

/// One user's in-catalog history as `(days_ago, item_idx)` pairs,
/// oldest first. Produced by [`read_user_histories`].
pub type UserHistory = Vec<(f64, usize)>;

/// Read the interactions file and group it into per-user chronological
/// histories of catalog item indices.
///
/// Requires `user_id`, `item_id`, and a numeric `days_ago` column (fails
/// loudly otherwise, mirroring `data::sequences`). Items absent from
/// `mappings.item_to_idx` are skipped. Each history is sorted oldest-first
/// (larger `days_ago` == older; stable, so equal-`days_ago` ties keep
/// file order). The returned map is ordered by user id, so iteration is
/// deterministic.
///
/// Users with **at least one** in-catalog interaction are kept — this is
/// the shared reader for both training (which then drops `< 2`) and
/// post-training user-embedding extraction (where a single item is a
/// perfectly good history).
pub fn read_user_histories(
    interactions_path: &str,
    mappings: &Mappings,
) -> Result<BTreeMap<String, UserHistory>> {
    let df = read_interactions(interactions_path)?;

    let user_col = df
        .column("user_id")
        .map_err(|_| anyhow::anyhow!("interactions file is missing the required `user_id` column"))?
        .str()
        .map_err(|_| anyhow::anyhow!("`user_id` column must be Utf8/String"))?;
    let item_col = df
        .column("item_id")
        .map_err(|_| anyhow::anyhow!("interactions file is missing the required `item_id` column"))?
        .str()
        .map_err(|_| anyhow::anyhow!("`item_id` column must be Utf8/String"))?;

    let days_col = match df.column("days_ago") {
        Ok(c) => c.f64().map_err(|_| {
            anyhow::anyhow!(
                "BERT4Rec requires a numeric `days_ago` column for position bucketing, \
                 but `days_ago` has dtype {:?}. Provide `days_ago` as Float64.",
                c.dtype()
            )
        })?,
        Err(_) => bail!(
            "BERT4Rec requires a `days_ago` column in the interactions file to order \
             each user's history and derive relative position buckets; it is absent. \
             (ADR-0001 §Risks: we fail loudly rather than silently fall back to row order.)"
        ),
    };

    let mut per_user: BTreeMap<String, UserHistory> = BTreeMap::new();
    for ((u, it), d) in user_col.into_iter().zip(item_col).zip(days_col) {
        let (Some(u), Some(it), Some(d)) = (u, it, d) else {
            continue;
        };
        let Some(&item_idx) = mappings.item_to_idx.get(it) else {
            continue;
        };
        per_user
            .entry(u.to_string())
            .or_default()
            .push((d, item_idx));
    }

    for hist in per_user.values_mut() {
        // Oldest first: larger `days_ago` == further in the past.
        hist.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    }

    Ok(per_user)
}

/// Build left-padded masked training sequences from an interactions file.
///
/// Pipeline per user (users iterated in sorted-id order for determinism):
///
/// 1. keep the most recent `seq_len` in-catalog items (oldest first),
/// 2. left-pad with `PAD` to `seq_len`,
/// 3. replace each non-pad token with `MASK` independently with
///    probability `mask_ratio` (seeded `StdRng`, so the pattern is
///    reproducible); if a row ends up with no masked position, its most
///    recent item is masked so every row contributes to the Cloze loss,
/// 4. bucket each position's `days_ago` via
///    [`days_ago_to_bucket`]`(d, num_position_buckets)`.
///
/// Users with fewer than two in-catalog interactions are dropped — a
/// masked singleton has no context to recover it from.
///
/// Errors if `seq_len == 0`, `num_position_buckets == 0`, `mask_ratio`
/// is outside `[0, 1]`, or `days_ago` is missing / non-numeric.
pub fn build_masked_sequences(
    interactions_path: &str,
    mappings: &Mappings,
    seq_len: usize,
    mask_ratio: f64,
    num_position_buckets: usize,
    seed: u64,
) -> Result<MaskedSequenceDataset> {
    if seq_len == 0 {
        bail!("build_masked_sequences: seq_len must be >= 1");
    }
    if num_position_buckets == 0 {
        bail!("build_masked_sequences: num_position_buckets must be >= 1");
    }
    if !(0.0..=1.0).contains(&mask_ratio) {
        bail!("build_masked_sequences: mask_ratio must be in [0, 1], got {mask_ratio}");
    }

    let per_user = read_user_histories(interactions_path, mappings)?;
    let vocab_size = mappings.idx_to_item.len() + NUM_RESERVED_TOKENS;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut original: Vec<i64> = Vec::new();
    let mut inputs: Vec<i64> = Vec::new();
    let mut mask: Vec<i64> = Vec::new();
    let mut positions: Vec<i64> = Vec::new();

    for hist in per_user.values() {
        if hist.len() < 2 {
            continue;
        }
        // Most recent `seq_len` items, right-aligned.
        let n = hist.len();
        let take = seq_len.min(n);
        let recent = &hist[n - take..];
        let start = seq_len - take;

        let mut row_orig = vec![PAD_TOKEN; seq_len];
        let mut row_in = vec![PAD_TOKEN; seq_len];
        let mut row_mask = vec![0_i64; seq_len];
        let mut row_pos = vec![0_i64; seq_len];

        let mut any_masked = false;
        for (k, (days_ago, item_idx)) in recent.iter().enumerate() {
            let p = start + k;
            let tok = item_to_token(*item_idx);
            row_orig[p] = tok;
            row_pos[p] = days_ago_to_bucket(*days_ago, num_position_buckets) as i64;
            if rng.gen_bool(mask_ratio) {
                row_in[p] = MASK_TOKEN;
                row_mask[p] = 1;
                any_masked = true;
            } else {
                row_in[p] = tok;
            }
        }
        if !any_masked {
            // Guarantee at least one Cloze target per row; the most recent
            // item is the one next-item-style serving asks about anyway.
            let last = seq_len - 1;
            row_in[last] = MASK_TOKEN;
            row_mask[last] = 1;
        }

        original.extend_from_slice(&row_orig);
        inputs.extend_from_slice(&row_in);
        mask.extend_from_slice(&row_mask);
        positions.extend_from_slice(&row_pos);
    }

    Ok(MaskedSequenceDataset {
        original,
        inputs,
        mask,
        positions,
        seq_len,
        vocab_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

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
    fn test_days_ago_to_bucket() {
        assert_eq!(days_ago_to_bucket(0.0, 32), 0);
        assert_eq!(days_ago_to_bucket(1.0, 32), 0);
        assert_eq!(days_ago_to_bucket(2.0, 32), 1);
        assert_eq!(days_ago_to_bucket(4.0, 32), 2);
        assert_eq!(days_ago_to_bucket(8.0, 32), 3);
        assert_eq!(days_ago_to_bucket(365.0, 32), 8);
        assert_eq!(days_ago_to_bucket(1825.0, 32), 10);
        // Clamped to the table size.
        assert_eq!(days_ago_to_bucket(1825.0, 8), 7);
    }

    #[test]
    fn token_mapping_roundtrips_and_skips_reserved() {
        assert_eq!(item_to_token(0), 2);
        assert_eq!(token_to_item(2), Some(0));
        assert_eq!(token_to_item(PAD_TOKEN), None);
        assert_eq!(token_to_item(MASK_TOKEN), None);
        assert_eq!(token_to_item(-1), None);
    }

    #[test]
    fn test_missing_days_ago_fails_loudly() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "i.csv",
            "user_id,item_id,value\nu1,a,1.0\nu1,b,1.0\n",
        );
        let m = mappings_with_items(&["a", "b"]);
        let err = build_masked_sequences(&path, &m, 4, 0.2, 32, 0).unwrap_err();
        assert!(
            err.to_string().contains("days_ago"),
            "error must mention days_ago, got: {err}"
        );
    }

    #[test]
    fn rejects_bad_mask_ratio_and_zero_lengths() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "i.csv",
            "user_id,item_id,value,days_ago\nu1,a,1.0,1.0\nu1,b,1.0,2.0\n",
        );
        let m = mappings_with_items(&["a", "b"]);
        assert!(build_masked_sequences(&path, &m, 4, 1.5, 32, 0).is_err());
        assert!(build_masked_sequences(&path, &m, 4, -0.1, 32, 0).is_err());
        assert!(build_masked_sequences(&path, &m, 0, 0.2, 32, 0).is_err());
        assert!(build_masked_sequences(&path, &m, 4, 0.2, 0, 0).is_err());
    }

    #[test]
    fn test_masked_sequences_are_left_padded() {
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
        // mask_ratio 0 → nothing randomly masked; the fallback masks the
        // last position only, so the layout is fully predictable.
        let ds = build_masked_sequences(&path, &m, 5, 0.0, 32, 7).unwrap();
        assert_eq!(ds.len(), 1);
        assert_eq!(ds.vocab_size, 5); // 3 items + PAD + MASK
        // tokens: a=2, b=3, c=4; width 5 → [0,0,a,b,c]
        assert_eq!(ds.original_row(0), &[0, 0, 2, 3, 4]);
        assert_eq!(ds.input_row(0), &[0, 0, 2, 3, MASK_TOKEN]);
        assert_eq!(ds.mask_row(0), &[0, 0, 0, 0, 1]);
        // positions: a=3d→1, b=2d→1, c=1d→0; pads 0.
        assert_eq!(ds.position_row(0), &[0, 0, 1, 1, 0]);
        // Pad positions are never masked.
        for (tok, mk) in ds.original_row(0).iter().zip(ds.mask_row(0)) {
            if *tok == PAD_TOKEN {
                assert_eq!(*mk, 0);
            }
        }
    }

    #[test]
    fn long_history_truncates_to_most_recent() {
        let dir = tempfile::tempdir().unwrap();
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
        let ds = build_masked_sequences(&path, &m, 3, 0.0, 32, 0).unwrap();
        // most recent 3 = c,d,e → tokens 4,5,6
        assert_eq!(ds.original_row(0), &[4, 5, 6]);
    }

    #[test]
    fn test_mask_ratio_approximately_correct() {
        let dir = tempfile::tempdir().unwrap();
        // 200 users × 20 items each = 4000 non-pad positions.
        let mut body = String::from("user_id,item_id,value,days_ago\n");
        let items: Vec<String> = (0..20).map(|i| format!("i{i}")).collect();
        for u in 0..200 {
            for (k, it) in items.iter().enumerate() {
                body.push_str(&format!("u{u:03},{it},1.0,{}.0\n", 20 - k));
            }
        }
        let path = write_csv(dir.path(), "i.csv", &body);
        let refs: Vec<&str> = items.iter().map(|s| s.as_str()).collect();
        let m = mappings_with_items(&refs);
        let ds = build_masked_sequences(&path, &m, 20, 0.2, 32, 123).unwrap();
        assert_eq!(ds.len(), 200);
        let non_pad = ds.original.iter().filter(|&&t| t != PAD_TOKEN).count();
        let masked = ds.mask.iter().filter(|&&m| m == 1).count();
        let ratio = masked as f64 / non_pad as f64;
        assert!(
            (0.15..=0.25).contains(&ratio),
            "expected ~20% masked, got {ratio:.3} ({masked}/{non_pad})"
        );
        // Masked positions show MASK in the input and the item in original.
        for k in 0..ds.inputs.len() {
            if ds.mask[k] == 1 {
                assert_eq!(ds.inputs[k], MASK_TOKEN);
                assert!(ds.original[k] >= NUM_RESERVED_TOKENS as i64);
            } else {
                assert_eq!(ds.inputs[k], ds.original[k]);
            }
        }
    }

    #[test]
    fn every_row_has_at_least_one_masked_position() {
        let dir = tempfile::tempdir().unwrap();
        let mut body = String::from("user_id,item_id,value,days_ago\n");
        for u in 0..50 {
            body.push_str(&format!("u{u:02},a,1.0,2.0\nu{u:02},b,1.0,1.0\n"));
        }
        let path = write_csv(dir.path(), "i.csv", &body);
        let m = mappings_with_items(&["a", "b"]);
        // Tiny ratio: most rows would otherwise get no mask at all.
        let ds = build_masked_sequences(&path, &m, 4, 0.01, 32, 5).unwrap();
        for i in 0..ds.len() {
            assert!(
                ds.mask_row(i).iter().any(|&m| m == 1),
                "row {i} has no masked position"
            );
        }
    }

    #[test]
    fn test_single_interaction_user_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "i.csv",
            "user_id,item_id,value,days_ago\nu1,a,1.0,1.0\nu2,a,1.0,1.0\nu2,b,1.0,2.0\n",
        );
        let m = mappings_with_items(&["a", "b"]);
        let ds = build_masked_sequences(&path, &m, 4, 0.2, 32, 0).unwrap();
        assert_eq!(ds.len(), 1, "only u2 (2 items) survives");
        // But the shared reader keeps singletons for embedding extraction.
        let hist = read_user_histories(&path, &m).unwrap();
        assert_eq!(hist.len(), 2);
        assert_eq!(hist["u1"], vec![(1.0, 0)]);
        assert_eq!(hist["u2"], vec![(2.0, 1), (1.0, 0)]); // oldest first
    }

    #[test]
    fn test_deterministic_with_same_seed() {
        let dir = tempfile::tempdir().unwrap();
        let mut body = String::from("user_id,item_id,value,days_ago\n");
        let items: Vec<String> = (0..12).map(|i| format!("i{i}")).collect();
        for u in 0..30 {
            for (k, it) in items.iter().enumerate() {
                body.push_str(&format!("u{u:02},{it},1.0,{}.0\n", 12 - k));
            }
        }
        let path = write_csv(dir.path(), "i.csv", &body);
        let refs: Vec<&str> = items.iter().map(|s| s.as_str()).collect();
        let m = mappings_with_items(&refs);
        let a = build_masked_sequences(&path, &m, 12, 0.3, 32, 99).unwrap();
        let b = build_masked_sequences(&path, &m, 12, 0.3, 32, 99).unwrap();
        assert_eq!(a.inputs, b.inputs);
        assert_eq!(a.mask, b.mask);
        assert_eq!(a.positions, b.positions);
        let c = build_masked_sequences(&path, &m, 12, 0.3, 32, 100).unwrap();
        assert_ne!(
            a.mask, c.mask,
            "a different seed should change the mask pattern"
        );
    }
}
