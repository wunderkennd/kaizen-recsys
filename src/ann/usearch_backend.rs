//! usearch HNSW backend for `AnnBackend` (ADR-0004 Phase 2). Feature-gated
//! (`ann`).
//!
//! Verified against the real usearch 2.25.3 Rust API
//! (`~/.cargo/registry/.../usearch-2.25.3/rust/lib.rs`):
//!
//! - `usearch::{Index, IndexOptions, MetricKind, ScalarKind}` are re-exported
//!   (`pub use ffi::{...}`). `pub type Key = u64;`.
//! - `IndexOptions { dimensions, metric, quantization, connectivity,
//!   expansion_add, expansion_search, multi }` (`impl Default`). We set
//!   `metric: MetricKind::IP`, `quantization: ScalarKind::F32`,
//!   `dimensions: dim`, and leave the rest at their defaults
//!   (`connectivity: 0`, `expansion_*: 0` → usearch picks library defaults).
//! - `Index::new(&IndexOptions) -> Result<Index, cxx::Exception>`.
//! - `index.reserve(capacity: usize) -> Result<(), _>`.
//! - `index.add(key: u64, vector: &[f32]) -> Result<(), _>`.
//! - `index.search(query: &[f32], count: usize) -> Result<Matches, _>` where
//!   `struct Matches { keys: Vec<u64>, distances: Vec<f32> }`.
//! - `index.memory_usage() -> usize`.
//!
//! Distance/score convention: usearch documents `MetricKind::IP` as
//! `distance = 1 - dot(a, b)`. So `score = 1.0 - distance` recovers the dot
//! product exactly, and a *smaller* distance is a *larger* dot product —
//! matching `exact_top_k`'s descending-dot ordering once we convert back.
//!
//! Empty-items handling: If `build` is called with an empty items slice, the
//! backend is created in an empty state (no index), and `search` returns an
//! empty result set. This matches `TurbovecBackend`'s behavior.

use crate::ann::AnnBackend;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

/// HNSW (usearch) ANN backend over an item-embedding matrix, queried for
/// approximate maximum-inner-product top-K.
///
/// When built with an empty items slice, `index` is `None` and `search` returns
/// an empty result set.
#[allow(dead_code)]
pub struct UsearchBackend {
    index: Option<Index>,
    dim: usize,
    num_items: usize,
}

impl AnnBackend for UsearchBackend {
    fn build(items: &[Vec<f32>]) -> Self {
        if items.is_empty() {
            return Self {
                index: None,
                dim: 0,
                num_items: 0,
            };
        }
        let dim = items.first().map(|v| v.len()).unwrap_or(0);
        let options = IndexOptions {
            dimensions: dim,
            metric: MetricKind::IP,
            quantization: ScalarKind::F32,
            ..Default::default()
        };
        let index = Index::new(&options).expect("usearch: index construction failed");
        index.reserve(items.len()).expect("usearch: reserve failed");
        for (i, item) in items.iter().enumerate() {
            index.add(i as u64, item).expect("usearch: add failed");
        }
        Self {
            index: Some(index),
            dim,
            num_items: items.len(),
        }
    }

    fn search(&self, query: &[f32], k: usize, exclude: &[usize]) -> Vec<(usize, f32)> {
        if k == 0 || self.index.is_none() {
            return Vec::new();
        }
        // Over-fetch so post-filtering excluded ids still yields up to k.
        let count = (k + exclude.len()).min(self.num_items);
        let index = self.index.as_ref().unwrap();
        let matches = index.search(query, count).expect("usearch: search failed");
        let ex: std::collections::HashSet<usize> = exclude.iter().copied().collect();
        matches
            .keys
            .iter()
            .zip(matches.distances.iter())
            // usearch IP distance is `1 - dot`; recover the dot product so the
            // ordering matches `exact_top_k`'s descending-dot convention.
            .map(|(&key, &dist)| (key as usize, 1.0 - dist))
            .filter(|(i, _)| !ex.contains(i))
            .take(k)
            .collect()
    }

    fn index_bytes(&self) -> usize {
        match &self.index {
            None => 0,
            Some(idx) => {
                let reported = idx.memory_usage();
                if reported > 0 {
                    reported
                } else {
                    self.num_items * self.dim * std::mem::size_of::<f32>()
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ann::exact::exact_top_k;

    /// Three well-separated clusters in 3-D; ANN must recover the exact NN.
    fn clustered(n_per: usize) -> Vec<Vec<f32>> {
        let centers = [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut v = Vec::new();
        for ctr in &centers {
            for j in 0..n_per {
                let jitter = j as f32 * 1e-3;
                v.push(vec![ctr[0] + jitter, ctr[1] + jitter, ctr[2] + jitter]);
            }
        }
        v
    }

    #[test]
    fn recovers_exact_top1_on_clustered_data() {
        let items = clustered(20);
        let be = UsearchBackend::build(&items);
        let query = vec![1.0, 0.0, 0.0]; // cluster 0
        let approx = be.search(&query, 1, &[]);
        let exact = exact_top_k(&query, &items, 1);
        assert_eq!(
            approx[0].0, exact[0].0,
            "usearch must recover exact NN on separated clusters"
        );
    }

    #[test]
    fn search_respects_exclude() {
        let items = clustered(20);
        let be = UsearchBackend::build(&items);
        // exclude the exact top-1; result must not contain it.
        let exact = exact_top_k(&[1.0, 0.0, 0.0], &items, 1);
        let excluded = exact[0].0;
        let approx = be.search(&[1.0, 0.0, 0.0], 5, &[excluded]);
        assert!(
            approx.iter().all(|(i, _)| *i != excluded),
            "excluded id must not appear"
        );
    }

    #[test]
    fn empty_items_builds_and_searches_empty() {
        let be = UsearchBackend::build(&[]);
        let result = be.search(&[1.0, 0.0, 0.0], 5, &[]);
        assert!(result.is_empty(), "search on empty index must return empty");
        assert_eq!(be.index_bytes(), 0, "empty index must have 0 bytes");
    }
}
