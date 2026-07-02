//! turbovec (TurboQuant) quantized backend for `AnnBackend` (ADR-0004
//! Phase 2). Feature-gated (`ann`).
//!
//! Verified against the real turbovec 0.9.0 Rust API
//! (`~/.cargo/registry/.../turbovec-0.9.0/src/lib.rs`):
//!
//! - `turbovec::TurboQuantIndex` is the public index type. Construction:
//!   `TurboQuantIndex::new(dim: usize, bit_width: usize) -> Result<Self,
//!   ConstructError>`. `bit_width` must be in `{2, 3, 4}`; `dim` must be a
//!   positive multiple of 8 (`DimNotPositiveMultipleOf8` otherwise). There is
//!   no per-vector id API — ids are *implicit insertion order* (0..n).
//! - `index.add(&mut self, vectors: &[f32])` — appends a flat row-major batch
//!   (`len % dim == 0`). We add all items in one call; item `i` lands at slot
//!   `i`. (Panics on non-finite / `|v| >= 1e16` coords; our embeddings are
//!   finite.)
//! - `index.search(&self, queries: &[f32], k: usize) -> SearchResults` where
//!   `struct SearchResults { scores: Vec<f32>, indices: Vec<i64>, nq, k }`,
//!   with `indices_for_query(qi) -> &[i64]` / `scores_for_query(qi) -> &[f32]`.
//!   (`search_with_mask(queries, k, Option<&[bool]>)` exists for allowlist
//!   filtering, but we post-filter `exclude` like `UsearchBackend` does —
//!   simpler and the mask is a fixed-length bool over *all* slots.)
//! - `index.len() -> usize`, `index.dim() -> usize`, `index.bit_width() ->
//!   usize` for the footprint estimate.
//!
//! Score/ordering convention: turbovec's SIMD heap keeps the *largest* scores
//! (a calibrated inner-product-like similarity) and returns each query's
//! top-k in descending-score order (`search.rs` sorts `b.score.cmp(a.score)`).
//! A *larger* score is a *nearer* neighbor — already matching `exact_top_k`'s
//! descending-dot ordering, so we pass the score through unchanged.
//!
//! Dimension note: TurboQuant requires `dim % 8 == 0`, so the 3-D `clustered`
//! fixtures used by usearch can't be fed directly. The test below pads each
//! 3-D center to 8 dims with zeros (keeping clusters well-separated) and still
//! asserts exact top-1 recovery. The backend itself is dim-agnostic; it reads
//! `dim` from the first item and asserts the multiple-of-8 invariant.
//!
//! Empty-items handling: If `build` is called with an empty items slice, the
//! backend is created in an empty state (no index), and `search` returns an
//! empty result set. This matches `UsearchBackend`'s behavior.

use crate::ann::AnnBackend;
use turbovec::TurboQuantIndex;

/// Quantized (TurboQuant) ANN backend over an item-embedding matrix, queried
/// for approximate maximum-inner-product top-K. Uses 4-bit quantization
/// (better recall than 2-bit).
///
/// When built with an empty items slice, `index` is `None` and `search` returns
/// an empty result set.
#[allow(dead_code)]
pub struct TurbovecBackend {
    index: Option<TurboQuantIndex>,
    dim: usize,
    num_items: usize,
    bit_width: usize,
}

impl AnnBackend for TurbovecBackend {
    fn build(items: &[Vec<f32>]) -> Self {
        if items.is_empty() {
            return Self {
                index: None,
                dim: 0,
                num_items: 0,
                bit_width: 4,
            };
        }
        let dim = items.first().map(|v| v.len()).unwrap_or(0);
        let bit_width = 4;
        let mut index = TurboQuantIndex::new(dim, bit_width)
            .expect("turbovec: index construction failed (dim must be a multiple of 8)");
        // Flatten row-major; item `i` lands at slot `i` (implicit ids).
        let flat: Vec<f32> = items.iter().flat_map(|v| v.iter().copied()).collect();
        index.add(&flat);
        Self {
            index: Some(index),
            dim,
            num_items: items.len(),
            bit_width,
        }
    }

    fn search(&self, query: &[f32], k: usize, exclude: &[usize]) -> Vec<(usize, f32)> {
        if k == 0 || self.index.is_none() {
            return Vec::new();
        }
        // Over-fetch so post-filtering excluded ids still yields up to k.
        let count = (k + exclude.len()).min(self.num_items);
        let index = self.index.as_ref().unwrap();
        let results = index.search(query, count);
        if results.nq == 0 {
            return Vec::new();
        }
        let ex: std::collections::HashSet<usize> = exclude.iter().copied().collect();
        results
            .indices_for_query(0)
            .iter()
            .zip(results.scores_for_query(0).iter())
            // turbovec returns descending-score (larger = nearer), matching
            // `exact_top_k`'s descending-dot convention — pass score through.
            .map(|(&idx, &score)| (idx as usize, score))
            .filter(|(i, _)| !ex.contains(i))
            .take(k)
            .collect()
    }

    fn index_bytes(&self) -> usize {
        match &self.index {
            None => 0,
            Some(idx) => {
                // TurboQuant's headline win is the quantized footprint: each of the
                // `len` vectors is `dim * bit_width` bits of packed codes (plus a
                // per-vector f32 scale).
                let codes_bytes = idx.len() * self.dim * self.bit_width / 8;
                let scale_bytes = idx.len() * std::mem::size_of::<f32>();
                codes_bytes + scale_bytes
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ann::exact::exact_top_k;

    /// Three well-separated clusters, padded from 3-D to 8-D (zeros) so the
    /// vectors satisfy TurboQuant's `dim % 8 == 0` requirement. Clusters stay
    /// well-separated; ANN must recover the exact NN.
    fn clustered(n_per: usize) -> Vec<Vec<f32>> {
        let centers = [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut v = Vec::new();
        for ctr in &centers {
            for j in 0..n_per {
                let jitter = j as f32 * 1e-3;
                // Pad to 8 dims with trailing zeros.
                let mut row = vec![ctr[0] + jitter, ctr[1] + jitter, ctr[2] + jitter];
                row.resize(8, 0.0);
                v.push(row);
            }
        }
        v
    }

    fn query(c: [f32; 3]) -> Vec<f32> {
        let mut q = vec![c[0], c[1], c[2]];
        q.resize(8, 0.0);
        q
    }

    #[test]
    fn recovers_exact_top1_on_clustered_data() {
        let items = clustered(20);
        let be = TurbovecBackend::build(&items);
        let q = query([1.0, 0.0, 0.0]); // cluster 0
        let approx = be.search(&q, 1, &[]);
        let exact = exact_top_k(&q, &items, 1);
        assert_eq!(
            approx[0].0, exact[0].0,
            "turbovec must recover exact NN on separated clusters"
        );
    }

    #[test]
    fn search_respects_exclude() {
        let items = clustered(20);
        let be = TurbovecBackend::build(&items);
        let q = query([1.0, 0.0, 0.0]);
        // exclude the exact top-1; result must not contain it.
        let exact = exact_top_k(&q, &items, 1);
        let excluded = exact[0].0;
        let approx = be.search(&q, 5, &[excluded]);
        assert!(
            approx.iter().all(|(i, _)| *i != excluded),
            "excluded id must not appear"
        );
    }

    #[test]
    fn empty_items_builds_and_searches_empty() {
        let be = TurbovecBackend::build(&[]);
        let result = be.search(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5, &[]);
        assert!(result.is_empty(), "search on empty index must return empty");
        assert_eq!(be.index_bytes(), 0, "empty index must have 0 bytes");
    }
}
