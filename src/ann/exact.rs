//! Exact maximum-inner-product top-K and the recall@K metric — the ground
//! truth the ANN backends are scored against. Mirrors the dense serving
//! path (`crate::serving::filter_sort_top_k`); kept separate so the bench
//! and backends share one tested implementation.

/// Exact MIPS top-K: descending score, ascending-index tie-break (total order,
/// matching `filter_sort_top_k`).
#[allow(dead_code)]
pub fn exact_top_k(query: &[f32], items: &[Vec<f32>], k: usize) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = items
        .iter()
        .enumerate()
        .map(|(i, it)| (i, it.iter().zip(query).map(|(a, b)| a * b).sum::<f32>()))
        .collect();
    let k = k.min(scored.len());
    if k == 0 {
        return Vec::new();
    }
    let cmp = |a: &(usize, f32), b: &(usize, f32)| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    };
    if k < scored.len() {
        scored.select_nth_unstable_by(k - 1, cmp);
        scored.truncate(k);
    }
    scored.sort_unstable_by(cmp);
    scored
}

/// Fraction of the exact top-k item ids that `approx` also returned.
#[allow(dead_code)]
pub fn recall_at_k(approx: &[(usize, f32)], exact: &[(usize, f32)]) -> f32 {
    if exact.is_empty() {
        return 1.0;
    }
    let truth: std::collections::HashSet<usize> = exact.iter().map(|(i, _)| *i).collect();
    let hits = approx.iter().filter(|(i, _)| truth.contains(i)).count();
    hits as f32 / exact.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_top_k_descending_with_index_tiebreak() {
        let items = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.9, 0.1]];
        let top = exact_top_k(&[1.0, 0.0], &items, 2);
        assert_eq!(top.iter().map(|(i, _)| *i).collect::<Vec<_>>(), vec![0, 2]);
    }

    #[test]
    fn exact_top_k_ties_break_by_lower_index() {
        let items = vec![vec![1.0], vec![1.0], vec![0.5]];
        let top = exact_top_k(&[1.0], &items, 2);
        assert_eq!(top[0].0, 0);
        assert_eq!(top[1].0, 1);
    }

    #[test]
    fn recall_full_and_partial() {
        let exact = vec![(1, 0.9), (2, 0.5), (3, 0.4)];
        assert_eq!(recall_at_k(&exact, &exact), 1.0);
        let approx = vec![(1, 0.9), (2, 0.5), (9, 0.3)];
        assert!((recall_at_k(&approx, &exact) - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn recall_of_empty_truth_is_one() {
        assert_eq!(recall_at_k(&[], &[]), 1.0);
    }
}
