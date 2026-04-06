//! Ranking evaluation metrics for recommendation quality assessment.
//!
//! Pure utility functions — no model-specific logic. All functions operate on
//! ordered recommendation lists and ground-truth relevant item sets.

use std::collections::HashSet;

/// Precision@K: fraction of recommended items in the top-K that are relevant.
///
/// Returns |recommended[:k] ∩ relevant| / k.
/// Returns 0.0 if k == 0.
pub fn precision_at_k(recommended: &[usize], relevant: &HashSet<usize>, k: usize) -> f64 {
    if k == 0 {
        return 0.0;
    }
    let hits = recommended
        .iter()
        .take(k)
        .filter(|item| relevant.contains(item))
        .count();
    hits as f64 / k as f64
}

/// Recall@K: fraction of relevant items captured in the top-K recommendations.
///
/// Returns |recommended[:k] ∩ relevant| / |relevant|.
/// Returns 0.0 if relevant is empty.
pub fn recall_at_k(recommended: &[usize], relevant: &HashSet<usize>, k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    let hits = recommended
        .iter()
        .take(k)
        .filter(|item| relevant.contains(item))
        .count();
    hits as f64 / relevant.len() as f64
}

/// NDCG@K: Normalized Discounted Cumulative Gain at K.
///
/// Uses binary relevance: gain = 1.0 if item is relevant, 0.0 otherwise.
/// Discount = 1 / log2(rank + 1) where rank is 1-based.
/// NDCG = DCG@K / IDCG@K.
/// Returns 0.0 if relevant is empty or k == 0.
pub fn ndcg_at_k(recommended: &[usize], relevant: &HashSet<usize>, k: usize) -> f64 {
    if k == 0 || relevant.is_empty() {
        return 0.0;
    }

    // DCG@K
    let dcg: f64 = recommended
        .iter()
        .take(k)
        .enumerate()
        .filter(|(_, item)| relevant.contains(item))
        .map(|(rank, _)| 1.0 / ((rank as f64) + 2.0).log2()) // 0-based rank r → 1-based position r+1 → discount 1/log2(r+2)
        .sum();

    // IDCG@K: best possible DCG with min(|relevant|, k) hits at the top
    let ideal_hits = relevant.len().min(k);
    let idcg: f64 = (0..ideal_hits)
        .map(|rank| 1.0 / ((rank as f64) + 2.0).log2())
        .sum();

    if idcg == 0.0 {
        return 0.0;
    }

    dcg / idcg
}

/// Mean Average Precision (MAP): mean of precision values at each relevant hit position.
///
/// For each position where a relevant item appears, compute precision at that position,
/// then average over all relevant items. Returns 0.0 if relevant is empty.
pub fn mean_average_precision(recommended: &[usize], relevant: &HashSet<usize>) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }

    let mut hits = 0;
    let mut sum_precision = 0.0;

    for (i, item) in recommended.iter().enumerate() {
        if relevant.contains(item) {
            hits += 1;
            sum_precision += hits as f64 / (i + 1) as f64;
        }
    }

    sum_precision / relevant.len() as f64
}

/// Coverage: fraction of the total item catalog recommended across all users.
///
/// Returns |unique items across all recommendation lists| / num_total_items.
/// Returns 0.0 if num_total_items == 0.
pub fn coverage(all_recommendations: &[Vec<usize>], num_total_items: usize) -> f64 {
    if num_total_items == 0 {
        return 0.0;
    }
    let unique_items: HashSet<usize> = all_recommendations
        .iter()
        .flat_map(|recs| recs.iter().copied())
        .collect();
    unique_items.len() as f64 / num_total_items as f64
}

/// Hit Rate@K: 1.0 if any item in recommended[:k] is relevant, else 0.0.
///
/// Returns 0.0 if k == 0.
pub fn hit_rate_at_k(recommended: &[usize], relevant: &HashSet<usize>, k: usize) -> f64 {
    if k == 0 {
        return 0.0;
    }
    let has_hit = recommended
        .iter()
        .take(k)
        .any(|item| relevant.contains(item));
    if has_hit { 1.0 } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_relevant(items: &[usize]) -> HashSet<usize> {
        items.iter().copied().collect()
    }

    // --- precision_at_k ---

    #[test]
    fn test_precision_at_k_basic() {
        let recommended = vec![1, 2, 3, 4, 5];
        let relevant = make_relevant(&[1, 3, 5, 7]);

        // Top-3: items 1, 2, 3 → hits: 1, 3 → 2/3
        assert!((precision_at_k(&recommended, &relevant, 3) - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_precision_at_k_all_relevant() {
        let recommended = vec![1, 2, 3];
        let relevant = make_relevant(&[1, 2, 3]);

        assert!((precision_at_k(&recommended, &relevant, 3) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_precision_at_k_none_relevant() {
        let recommended = vec![1, 2, 3];
        let relevant = make_relevant(&[4, 5, 6]);

        assert!((precision_at_k(&recommended, &relevant, 3) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_precision_at_k_zero_k() {
        let recommended = vec![1, 2, 3];
        let relevant = make_relevant(&[1, 2]);

        assert!((precision_at_k(&recommended, &relevant, 0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_precision_at_k_exceeds_list_length() {
        // k larger than list: only count actual items
        let recommended = vec![1, 2];
        let relevant = make_relevant(&[1, 2, 3]);

        // 2 hits out of k=5 → 2/5
        assert!((precision_at_k(&recommended, &relevant, 5) - 2.0 / 5.0).abs() < 1e-10);
    }

    // --- recall_at_k ---

    #[test]
    fn test_recall_at_k_basic() {
        let recommended = vec![1, 2, 3, 4, 5];
        let relevant = make_relevant(&[1, 3, 5, 7]);

        // Top-3: hits = {1, 3} → 2/4
        assert!((recall_at_k(&recommended, &relevant, 3) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_recall_at_k_full() {
        let recommended = vec![1, 3, 5, 7, 2, 4];
        let relevant = make_relevant(&[1, 3, 5, 7]);

        // Top-4 captures all relevant items → 4/4 = 1.0
        assert!((recall_at_k(&recommended, &relevant, 4) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_recall_at_k_empty_relevant() {
        let recommended = vec![1, 2, 3];
        let relevant = make_relevant(&[]);

        assert!((recall_at_k(&recommended, &relevant, 3) - 0.0).abs() < 1e-10);
    }

    // --- ndcg_at_k ---

    #[test]
    fn test_ndcg_at_k_perfect_ranking() {
        // All relevant items at top → NDCG = 1.0
        let recommended = vec![1, 2, 3, 4, 5];
        let relevant = make_relevant(&[1, 2, 3]);

        assert!((ndcg_at_k(&recommended, &relevant, 3) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ndcg_at_k_imperfect_ranking() {
        // Relevant items: {3, 5}. Recommended: [1, 2, 3, 4, 5]
        // Top-5: hits at positions 3 (rank=2, 0-based) and 5 (rank=4)
        // DCG = 1/log2(3+1) + 1/log2(5+1) = 1/2 + 1/log2(6)
        // IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 1/log2(3)
        let recommended = vec![1, 2, 3, 4, 5];
        let relevant = make_relevant(&[3, 5]);

        let ndcg = ndcg_at_k(&recommended, &relevant, 5);
        let dcg = 1.0 / 4.0_f64.log2() + 1.0 / 6.0_f64.log2();
        let idcg = 1.0 / 2.0_f64.log2() + 1.0 / 3.0_f64.log2();
        let expected = dcg / idcg;

        assert!(
            (ndcg - expected).abs() < 1e-10,
            "ndcg={ndcg}, expected={expected}"
        );
    }

    #[test]
    fn test_ndcg_at_k_single_hit_at_top() {
        let recommended = vec![1, 2, 3];
        let relevant = make_relevant(&[1]);

        // DCG = 1/log2(2) = 1.0, IDCG = 1/log2(2) = 1.0
        assert!((ndcg_at_k(&recommended, &relevant, 3) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ndcg_at_k_zero_k() {
        let recommended = vec![1, 2, 3];
        let relevant = make_relevant(&[1]);

        assert!((ndcg_at_k(&recommended, &relevant, 0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_ndcg_at_k_empty_relevant() {
        let recommended = vec![1, 2, 3];
        let relevant = make_relevant(&[]);

        assert!((ndcg_at_k(&recommended, &relevant, 3) - 0.0).abs() < 1e-10);
    }

    // --- mean_average_precision ---

    #[test]
    fn test_map_basic() {
        // Recommended: [1, 2, 3, 4, 5], Relevant: {1, 3, 5}
        // Hit at pos 0: precision = 1/1
        // Hit at pos 2: precision = 2/3
        // Hit at pos 4: precision = 3/5
        // MAP = (1 + 2/3 + 3/5) / 3
        let recommended = vec![1, 2, 3, 4, 5];
        let relevant = make_relevant(&[1, 3, 5]);

        let map = mean_average_precision(&recommended, &relevant);
        let expected = (1.0 + 2.0 / 3.0 + 3.0 / 5.0) / 3.0;

        assert!((map - expected).abs() < 1e-10);
    }

    #[test]
    fn test_map_perfect() {
        let recommended = vec![1, 2, 3];
        let relevant = make_relevant(&[1, 2, 3]);

        // All hits: (1/1 + 2/2 + 3/3) / 3 = 1.0
        assert!((mean_average_precision(&recommended, &relevant) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_map_no_hits() {
        let recommended = vec![1, 2, 3];
        let relevant = make_relevant(&[4, 5, 6]);

        assert!((mean_average_precision(&recommended, &relevant) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_map_empty_relevant() {
        let recommended = vec![1, 2, 3];
        let relevant = make_relevant(&[]);

        assert!((mean_average_precision(&recommended, &relevant) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_map_relevant_not_in_list() {
        // Some relevant items not in recommendations → they reduce the average
        let recommended = vec![1, 2, 3];
        let relevant = make_relevant(&[1, 99]); // 99 never appears

        // Hit at pos 0: precision = 1/1 = 1.0
        // MAP = 1.0 / 2 = 0.5 (divide by |relevant| = 2)
        assert!((mean_average_precision(&recommended, &relevant) - 0.5).abs() < 1e-10);
    }

    // --- coverage ---

    #[test]
    fn test_coverage_basic() {
        let all_recs = vec![vec![0, 1, 2], vec![2, 3, 4], vec![4, 5]];
        // Unique items: {0, 1, 2, 3, 4, 5} = 6 out of 10
        assert!((coverage(&all_recs, 10) - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_coverage_full() {
        let all_recs = vec![vec![0, 1], vec![2, 3], vec![4]];
        assert!((coverage(&all_recs, 5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_coverage_empty() {
        let all_recs: Vec<Vec<usize>> = vec![vec![], vec![]];
        assert!((coverage(&all_recs, 10) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_coverage_zero_items() {
        let all_recs = vec![vec![1, 2]];
        assert!((coverage(&all_recs, 0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_coverage_duplicates_across_users() {
        // Same items recommended to all users — coverage should not double-count
        let all_recs = vec![vec![0, 1], vec![0, 1], vec![0, 1]];
        assert!((coverage(&all_recs, 10) - 0.2).abs() < 1e-10);
    }

    // --- hit_rate_at_k ---

    #[test]
    fn test_hit_rate_hit() {
        let recommended = vec![10, 20, 30];
        let relevant = make_relevant(&[20, 40]);

        assert!((hit_rate_at_k(&recommended, &relevant, 3) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hit_rate_miss() {
        let recommended = vec![10, 20, 30];
        let relevant = make_relevant(&[40, 50]);

        assert!((hit_rate_at_k(&recommended, &relevant, 3) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_hit_rate_hit_outside_k() {
        let recommended = vec![10, 20, 30, 40];
        let relevant = make_relevant(&[40]);

        // Hit is at position 4, but k=2 → miss
        assert!((hit_rate_at_k(&recommended, &relevant, 2) - 0.0).abs() < 1e-10);
        // k=4 → hit
        assert!((hit_rate_at_k(&recommended, &relevant, 4) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hit_rate_zero_k() {
        let recommended = vec![1, 2, 3];
        let relevant = make_relevant(&[1]);

        assert!((hit_rate_at_k(&recommended, &relevant, 0) - 0.0).abs() < 1e-10);
    }
}
