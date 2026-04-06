//! Advanced interaction weighting: event-type weights, temporal decay, and IPS.
//!
//! These transforms are applied to interaction triplets *before* they enter
//! the sparse X matrix, allowing the FEASE model to incorporate signal quality
//! differences (e.g., purchases vs clicks) and temporal relevance.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for advanced interaction weighting.
///
/// All fields default to no-op values, so passing `WeightingConfig::default()`
/// produces identical results to the unweighted pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightingConfig {
    /// Event-type weights: maps event type string -> weight multiplier.
    /// Applied only when the interactions DataFrame has an `event_type` column.
    /// Example: {"click": 1.0, "cart": 3.0, "purchase": 5.0, "negative": -2.0}
    /// Uses HashMap for serde compatibility; converted to AHashMap at call sites.
    pub event_weights: Option<HashMap<String, f64>>,

    /// Temporal decay rate (exponential: value * exp(-decay_rate * days_ago)).
    /// Applied only when the interactions DataFrame has a `days_ago` column.
    /// 0.0 = no decay.
    pub decay_rate: f64,

    /// Inverse Propensity Scoring alpha.
    /// Propensity = (item_count / max_count) ^ ips_alpha
    /// Reweighted value = original_value / propensity
    /// 0.0 = no IPS correction.
    pub ips_alpha: f64,

    /// Sparsity pruning threshold for the final S matrix.
    /// Entries with |value| < threshold are zeroed after training.
    /// 0.0 = no pruning.
    pub sparsity_threshold: f64,
}

impl Default for WeightingConfig {
    fn default() -> Self {
        Self {
            event_weights: None,
            decay_rate: 0.0,
            ips_alpha: 0.0,
            sparsity_threshold: 0.0,
        }
    }
}

/// Applies event-type weighting to interaction triplets.
///
/// For each triplet, looks up the event type and multiplies the value
/// by the corresponding weight. Unknown event types keep their original value.
pub fn apply_event_weights(
    triplets: &mut [(usize, usize, f64)],
    event_types: &[Option<&str>],
    weights: &HashMap<String, f64>,
) {
    for (triplet, event_type) in triplets.iter_mut().zip(event_types.iter()) {
        if let Some(et) = event_type
            && let Some(&w) = weights.get(*et)
        {
            triplet.2 *= w;
        }
    }
}

/// Applies exponential temporal decay to interaction triplets.
///
/// Each triplet's value is multiplied by exp(-decay_rate * days_ago).
pub fn apply_temporal_decay(
    triplets: &mut [(usize, usize, f64)],
    days_ago: &[Option<f64>],
    decay_rate: f64,
) {
    for (triplet, d) in triplets.iter_mut().zip(days_ago.iter()) {
        if let Some(days) = d {
            triplet.2 *= (-decay_rate * days).exp();
        }
    }
}

/// Applies Inverse Propensity Scoring to interaction triplets.
///
/// Computes per-item popularity propensity and down-weights popular items
/// to debias recommendations toward long-tail content.
pub fn apply_ips(triplets: &mut [(usize, usize, f64)], num_items: usize, ips_alpha: f64) {
    if ips_alpha == 0.0 {
        return;
    }

    // Count interactions per item
    let mut item_counts = vec![0u64; num_items];
    for &(_, col_idx, _) in triplets.iter() {
        if col_idx < num_items {
            item_counts[col_idx] += 1;
        }
    }

    let max_count = *item_counts.iter().max().unwrap_or(&0);
    if max_count == 0 {
        return;
    }

    let max_f = max_count as f64;

    // Reweight: value / propensity where propensity = (count / max_count)^alpha
    for triplet in triplets.iter_mut() {
        let count = item_counts.get(triplet.1).copied().unwrap_or(0);
        if count > 0 {
            let propensity = (count as f64 / max_f).powf(ips_alpha);
            if propensity > 1e-12 {
                triplet.2 /= propensity;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_weights() {
        let mut triplets = vec![(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)];
        let event_types = vec![Some("click"), Some("purchase"), Some("cart")];
        let mut weights = HashMap::new();
        weights.insert("click".to_string(), 1.0);
        weights.insert("purchase".to_string(), 5.0);
        weights.insert("cart".to_string(), 3.0);

        apply_event_weights(&mut triplets, &event_types, &weights);

        assert!((triplets[0].2 - 1.0).abs() < 1e-10);
        assert!((triplets[1].2 - 5.0).abs() < 1e-10);
        assert!((triplets[2].2 - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_event_weights_unknown_type() {
        let mut triplets = vec![(0, 0, 2.0)];
        let event_types = vec![Some("unknown_event")];
        let weights = HashMap::new();

        apply_event_weights(&mut triplets, &event_types, &weights);

        // Unknown event type keeps original value
        assert!((triplets[0].2 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_temporal_decay() {
        let mut triplets = vec![(0, 0, 10.0), (1, 1, 10.0)];
        let days_ago = vec![Some(0.0), Some(100.0)];
        let decay_rate = 0.01;

        apply_temporal_decay(&mut triplets, &days_ago, decay_rate);

        // 0 days ago: no decay
        assert!((triplets[0].2 - 10.0).abs() < 1e-10);
        // 100 days ago: 10 * exp(-0.01 * 100) = 10 * exp(-1) ≈ 3.679
        assert!((triplets[1].2 - 10.0 * (-1.0_f64).exp()).abs() < 1e-6);
    }

    #[test]
    fn test_temporal_decay_zero_rate() {
        let mut triplets = vec![(0, 0, 5.0)];
        let days_ago = vec![Some(365.0)];

        apply_temporal_decay(&mut triplets, &days_ago, 0.0);

        // Zero decay rate = no change
        assert!((triplets[0].2 - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_ips_reweighting() {
        // Item 0: 3 interactions (most popular)
        // Item 1: 1 interaction (least popular)
        let mut triplets = vec![(0, 0, 1.0), (1, 0, 1.0), (2, 0, 1.0), (3, 1, 1.0)];

        apply_ips(&mut triplets, 2, 0.5);

        // Item 0: propensity = (3/3)^0.5 = 1.0, value unchanged
        assert!((triplets[0].2 - 1.0).abs() < 1e-10);
        // Item 1: propensity = (1/3)^0.5 ≈ 0.577, value = 1.0 / 0.577 ≈ 1.732
        assert!((triplets[3].2 - 1.0 / (1.0_f64 / 3.0).powf(0.5)).abs() < 1e-6);
    }

    #[test]
    fn test_ips_zero_alpha() {
        let mut triplets = vec![(0, 0, 5.0)];
        apply_ips(&mut triplets, 1, 0.0);
        assert!((triplets[0].2 - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_default_config_is_no_op() {
        let config = WeightingConfig::default();
        assert!(config.event_weights.is_none());
        assert_eq!(config.decay_rate, 0.0);
        assert_eq!(config.ips_alpha, 0.0);
        assert_eq!(config.sparsity_threshold, 0.0);
    }
}
