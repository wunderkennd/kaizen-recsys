//! Ranking-metric PyO3 wrappers. Moved from `src/lib.rs` per ADR-0003.
//!
//! Each function is a thin shim around `crate::metrics::*`, converting
//! Python-friendly `Vec`/`HashSet` argument types into the borrowed slice
//! references the underlying pure-Rust metrics expect.

use crate::metrics;
use pyo3::prelude::*;
use std::collections::HashSet;

/// Precision@K: fraction of top-K recommendations that are relevant.
#[pyfunction]
#[pyo3(signature = (recommended, relevant, k))]
pub fn precision_at_k(recommended: Vec<usize>, relevant: HashSet<usize>, k: usize) -> f64 {
    metrics::precision_at_k(&recommended, &relevant, k)
}

/// Recall@K: fraction of relevant items captured in the top-K recommendations.
#[pyfunction]
#[pyo3(signature = (recommended, relevant, k))]
pub fn recall_at_k(recommended: Vec<usize>, relevant: HashSet<usize>, k: usize) -> f64 {
    metrics::recall_at_k(&recommended, &relevant, k)
}

/// NDCG@K: Normalized Discounted Cumulative Gain at K (binary relevance).
#[pyfunction]
#[pyo3(signature = (recommended, relevant, k))]
pub fn ndcg_at_k(recommended: Vec<usize>, relevant: HashSet<usize>, k: usize) -> f64 {
    metrics::ndcg_at_k(&recommended, &relevant, k)
}

/// Mean Average Precision over the full recommendation list.
#[pyfunction]
#[pyo3(signature = (recommended, relevant))]
pub fn mean_average_precision(recommended: Vec<usize>, relevant: HashSet<usize>) -> f64 {
    metrics::mean_average_precision(&recommended, &relevant)
}

/// Coverage: fraction of the item catalog recommended across all users.
#[pyfunction]
#[pyo3(signature = (all_recommendations, num_total_items))]
pub fn coverage(all_recommendations: Vec<Vec<usize>>, num_total_items: usize) -> f64 {
    metrics::coverage(&all_recommendations, num_total_items)
}

/// Hit Rate@K: 1.0 if any item in top-K is relevant, else 0.0.
#[pyfunction]
#[pyo3(signature = (recommended, relevant, k))]
pub fn hit_rate_at_k(recommended: Vec<usize>, relevant: HashSet<usize>, k: usize) -> f64 {
    metrics::hit_rate_at_k(&recommended, &relevant, k)
}
