//! Approximate-nearest-neighbor retrieval backends (ADR-0004 Phase 2).
//! Feature-gated (`ann`); not in default or EASE-only builds.
pub mod exact;
pub mod turbovec_backend;
pub mod usearch_backend;

/// A vector-level ANN index built once over an item-embedding matrix, then
/// queried for approximate top-K. The bench builds one per backend and
/// scores its results against `exact::exact_top_k`.
#[allow(dead_code)]
pub trait AnnBackend {
    /// Build the index from `(num_items, dim)` row-major embeddings.
    fn build(items: &[Vec<f32>]) -> Self
    where
        Self: Sized;

    /// Approximate top-K `(item_idx, score)` for `query`, excluding `exclude`.
    fn search(&self, query: &[f32], k: usize, exclude: &[usize]) -> Vec<(usize, f32)>;

    /// Resident index size in bytes (for the memory axis).
    fn index_bytes(&self) -> usize;
}

/// Reference backend: exact search wearing the `AnnBackend` hat. Lets the
/// trait + bench plumbing be tested before any real ANN crate is wired in,
/// and is a recall==1.0 control in the bench.
#[allow(dead_code)]
pub struct ExactBackend {
    items: Vec<Vec<f32>>,
}

impl AnnBackend for ExactBackend {
    fn build(items: &[Vec<f32>]) -> Self {
        Self {
            items: items.to_vec(),
        }
    }
    fn search(&self, query: &[f32], k: usize, exclude: &[usize]) -> Vec<(usize, f32)> {
        let ex: std::collections::HashSet<usize> = exclude.iter().copied().collect();
        let full = exact::exact_top_k(query, &self.items, self.items.len());
        full.into_iter()
            .filter(|(i, _)| !ex.contains(i))
            .take(k)
            .collect()
    }
    fn index_bytes(&self) -> usize {
        self.items
            .iter()
            .map(|v| v.len() * std::mem::size_of::<f32>())
            .sum()
    }
}

#[cfg(test)]
mod backend_tests {
    use super::*;

    #[test]
    fn exact_backend_matches_exact_top_k_and_excludes() {
        let items = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.9, 0.1]];
        let be = ExactBackend::build(&items);
        // Without exclusion, top-1 is item 0.
        assert_eq!(be.search(&[1.0, 0.0], 1, &[])[0].0, 0);
        // Excluding item 0, top-1 becomes item 2 ([0.9,0.1]).
        assert_eq!(be.search(&[1.0, 0.0], 1, &[0])[0].0, 2);
    }
}
