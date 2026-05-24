#![allow(dead_code)]
//! Data validation module for pre-training quality checks.
//!
//! Ports the Python `GaussianAnomalyDetector` pattern: compute confidence
//! intervals from historical statistics and verify that current data falls
//! within expected bounds. This catches data pipeline issues (e.g., missing
//! partitions, duplicate loads, schema drift) before they corrupt the model.

use serde::{Deserialize, Serialize};
use std::fmt;
use pyo3::prelude::*;

/// A single confidence-interval check based on Gaussian statistics.
///
/// Given historical `mean` and `std`, the acceptable range is
/// `[mean - std_multiplier * std, mean + std_multiplier * std]`.
/// The `low` / `high` fields store these pre-computed bounds.
#[pyclass(get_all)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianAnomalyDetector {
    /// Lower bound of the confidence interval.
    pub low: f64,
    /// Upper bound of the confidence interval.
    pub high: f64,
    /// Historical mean.
    pub mean: f64,
    /// Historical standard deviation.
    pub std: f64,
    /// Number of standard deviations used for the interval.
    pub std_multiplier: f64,
}

impl GaussianAnomalyDetector {
    /// Creates a new detector from raw historical observations.
    ///
    /// `values` must contain at least 2 elements to compute a meaningful std.
    /// `std_multiplier` controls how wide the acceptance band is (e.g. 3.0 = 3-sigma).
    pub fn from_observations(values: &[f64], std_multiplier: f64) -> Option<Self> {
        if values.len() < 2 {
            return None;
        }
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std = variance.sqrt();

        Some(Self {
            low: mean - std_multiplier * std,
            high: mean + std_multiplier * std,
            mean,
            std,
            std_multiplier,
        })
    }

    /// Creates a detector with explicit bounds (for cases where you have
    /// pre-computed thresholds rather than raw observations).
    pub fn with_bounds(low: f64, high: f64, mean: f64, std: f64, std_multiplier: f64) -> Self {
        Self {
            low,
            high,
            mean,
            std,
            std_multiplier,
        }
    }

    /// Checks whether `value` falls within the confidence interval.
    /// Returns `Ok(z_score)` on pass, `Err(DataCheckFailure)` on fail.
    pub fn check(&self, value: f64, label: &str) -> DataCheckResult {
        let z_score = if self.std > 0.0 {
            (value - self.mean).abs() / self.std
        } else {
            0.0
        };

        if value >= self.low && value <= self.high {
            log::info!(
                "Data check PASSED for '{}': value={:.2}, z_score={:.2}, bounds=[{:.2}, {:.2}]",
                label,
                value,
                z_score,
                self.low,
                self.high
            );
            DataCheckResult::Pass {
                label: label.to_string(),
                value,
                z_score,
            }
        } else {
            log::warn!(
                "Data check FAILED for '{}': value={:.2}, z_score={:.2}, bounds=[{:.2}, {:.2}]",
                label,
                value,
                z_score,
                self.low,
                self.high
            );
            DataCheckResult::Fail {
                label: label.to_string(),
                value,
                z_score,
                low: self.low,
                high: self.high,
            }
        }
    }
}

#[pymethods]
impl GaussianAnomalyDetector {
    #[new]
    #[pyo3(signature = (low, high, mean, std, std_multiplier))]
    pub fn py_new(low: f64, high: f64, mean: f64, std: f64, std_multiplier: f64) -> Self {
        Self {
            low,
            high,
            mean,
            std,
            std_multiplier,
        }
    }

    #[staticmethod]
    #[pyo3(signature = (values, std_multiplier))]
    pub fn fit(values: Vec<f64>, std_multiplier: f64) -> Option<Self> {
        Self::from_observations(&values, std_multiplier)
    }

    #[pyo3(name = "check", signature = (value, label = "unlabeled"))]
    pub fn check_py(&self, value: f64, label: &str) -> bool {
        self.check(value, label).passed()
    }
}

impl fmt::Display for GaussianAnomalyDetector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GaussianAnomalyDetector(mean={:.2}, std={:.2}, bounds=[{:.2}, {:.2}], multiplier={:.1})",
            self.mean, self.std, self.low, self.high, self.std_multiplier
        )
    }
}

/// Result of a single data quality check.
#[derive(Debug, Clone)]
pub enum DataCheckResult {
    Pass {
        label: String,
        value: f64,
        z_score: f64,
    },
    Fail {
        label: String,
        value: f64,
        z_score: f64,
        low: f64,
        high: f64,
    },
}

impl DataCheckResult {
    pub fn passed(&self) -> bool {
        matches!(self, DataCheckResult::Pass { .. })
    }

    pub fn label(&self) -> &str {
        match self {
            DataCheckResult::Pass { label, .. } => label,
            DataCheckResult::Fail { label, .. } => label,
        }
    }
}

impl fmt::Display for DataCheckResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataCheckResult::Pass {
                label,
                value,
                z_score,
            } => write!(
                f,
                "PASS '{}': value={:.2}, z_score={:.2}",
                label, value, z_score
            ),
            DataCheckResult::Fail {
                label,
                value,
                z_score,
                low,
                high,
            } => write!(
                f,
                "FAIL '{}': value={:.2} outside [{:.2}, {:.2}], z_score={:.2}",
                label, value, low, high, z_score
            ),
        }
    }
}

/// Aggregated report from running all data quality checks.
#[derive(Debug, Clone)]
pub struct DataValidationReport {
    pub results: Vec<DataCheckResult>,
}

impl DataValidationReport {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    pub fn add(&mut self, result: DataCheckResult) {
        self.results.push(result);
    }

    /// Returns true if ALL checks passed.
    pub fn all_passed(&self) -> bool {
        self.results.iter().all(|r| r.passed())
    }

    /// Returns only the failed checks.
    pub fn failures(&self) -> Vec<&DataCheckResult> {
        self.results.iter().filter(|r| !r.passed()).collect()
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        let total = self.results.len();
        let passed = self.results.iter().filter(|r| r.passed()).count();
        let failed = total - passed;

        let mut lines = vec![format!(
            "Data Validation: {}/{} checks passed, {} failed",
            passed, total, failed
        )];

        for result in &self.results {
            lines.push(format!("  {}", result));
        }

        lines.join("\n")
    }
}

impl Default for DataValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for data validation thresholds.
///
/// Users can specify the std_multiplier for each metric category.
/// Higher multipliers mean wider acceptance bands (fewer false positives).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataValidationConfig {
    /// Multiplier for distinct-users check (default: 5.0).
    pub distinct_users_multiplier: f64,
    /// Multiplier for distinct-items check (default: 5.0).
    pub distinct_items_multiplier: f64,
    /// Multiplier for total-interactions check (default: 5.0).
    pub interactions_multiplier: f64,
    /// Multiplier for distinct-user-features check (default: 10.0).
    pub user_features_multiplier: f64,
    /// Multiplier for distinct-item-features check (default: 10.0).
    pub item_features_multiplier: f64,
}

impl Default for DataValidationConfig {
    fn default() -> Self {
        Self {
            distinct_users_multiplier: 5.0,
            distinct_items_multiplier: 5.0,
            interactions_multiplier: 5.0,
            user_features_multiplier: 10.0,
            item_features_multiplier: 10.0,
        }
    }
}

/// Validates current data counts against historical baselines.
///
/// `historical` contains one entry per historical observation (e.g., one per day
/// over the last 30 days). `current` is the count from the current run.
///
/// Returns a `DataValidationReport` with pass/fail for each metric.
pub fn validate_data_counts(
    historical_users: &[f64],
    historical_items: &[f64],
    historical_interactions: &[f64],
    current_users: f64,
    current_items: f64,
    current_interactions: f64,
    config: &DataValidationConfig,
) -> DataValidationReport {
    let mut report = DataValidationReport::new();

    if let Some(detector) = GaussianAnomalyDetector::from_observations(
        historical_users,
        config.distinct_users_multiplier,
    ) {
        report.add(detector.check(current_users, "distinct_users"));
    } else {
        log::warn!("Skipping distinct_users check: insufficient historical data");
    }

    if let Some(detector) = GaussianAnomalyDetector::from_observations(
        historical_items,
        config.distinct_items_multiplier,
    ) {
        report.add(detector.check(current_items, "distinct_items"));
    } else {
        log::warn!("Skipping distinct_items check: insufficient historical data");
    }

    if let Some(detector) = GaussianAnomalyDetector::from_observations(
        historical_interactions,
        config.interactions_multiplier,
    ) {
        report.add(detector.check(current_interactions, "total_interactions"));
    } else {
        log::warn!("Skipping total_interactions check: insufficient historical data");
    }

    log::info!("{}", report.summary());
    report
}

// --- Unit Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_detector_from_observations() {
        // 10 days of ~1000 users
        let history = vec![
            1000.0, 1010.0, 990.0, 1005.0, 995.0, 1002.0, 998.0, 1008.0, 992.0, 1000.0,
        ];
        let detector = GaussianAnomalyDetector::from_observations(&history, 3.0).unwrap();

        // Mean should be ~1000
        assert!((detector.mean - 1000.0).abs() < 1.0);
        // Std should be small (~6)
        assert!(detector.std < 10.0);
        // Bounds should be roughly [980, 1020]
        assert!(detector.low > 950.0);
        assert!(detector.high < 1050.0);
    }

    #[test]
    fn test_gaussian_detector_check_pass() {
        let detector = GaussianAnomalyDetector::with_bounds(900.0, 1100.0, 1000.0, 50.0, 2.0);
        let result = detector.check(1000.0, "test_metric");
        assert!(result.passed());
    }

    #[test]
    fn test_gaussian_detector_check_fail_low() {
        let detector = GaussianAnomalyDetector::with_bounds(900.0, 1100.0, 1000.0, 50.0, 2.0);
        let result = detector.check(800.0, "test_metric");
        assert!(!result.passed());
    }

    #[test]
    fn test_gaussian_detector_check_fail_high() {
        let detector = GaussianAnomalyDetector::with_bounds(900.0, 1100.0, 1000.0, 50.0, 2.0);
        let result = detector.check(1200.0, "test_metric");
        assert!(!result.passed());
    }

    #[test]
    fn test_insufficient_history() {
        let history = vec![1000.0]; // Only 1 observation
        assert!(GaussianAnomalyDetector::from_observations(&history, 3.0).is_none());
    }

    #[test]
    fn test_validate_data_counts_all_pass() {
        let hist_users = vec![1000.0, 1010.0, 990.0, 1005.0, 995.0];
        let hist_items = vec![500.0, 510.0, 490.0, 505.0, 495.0];
        let hist_interactions = vec![5000.0, 5100.0, 4900.0, 5050.0, 4950.0];

        let report = validate_data_counts(
            &hist_users,
            &hist_items,
            &hist_interactions,
            1000.0, // current users
            500.0,  // current items
            5000.0, // current interactions
            &DataValidationConfig::default(),
        );

        assert!(report.all_passed());
        assert_eq!(report.results.len(), 3);
        assert!(report.failures().is_empty());
    }

    #[test]
    fn test_validate_data_counts_detects_anomaly() {
        let hist_users = vec![1000.0, 1010.0, 990.0, 1005.0, 995.0];
        let hist_items = vec![500.0, 510.0, 490.0, 505.0, 495.0];
        let hist_interactions = vec![5000.0, 5100.0, 4900.0, 5050.0, 4950.0];

        let report = validate_data_counts(
            &hist_users,
            &hist_items,
            &hist_interactions,
            100.0,  // way below normal — anomaly!
            500.0,  // normal
            5000.0, // normal
            &DataValidationConfig::default(),
        );

        assert!(!report.all_passed());
        assert_eq!(report.failures().len(), 1);
        assert_eq!(report.failures()[0].label(), "distinct_users");
    }

    #[test]
    fn test_data_validation_report_summary() {
        let mut report = DataValidationReport::new();
        report.add(DataCheckResult::Pass {
            label: "metric_a".to_string(),
            value: 100.0,
            z_score: 0.5,
        });
        report.add(DataCheckResult::Fail {
            label: "metric_b".to_string(),
            value: 10.0,
            z_score: 5.0,
            low: 80.0,
            high: 120.0,
        });

        let summary = report.summary();
        assert!(summary.contains("1/2 checks passed"));
        assert!(summary.contains("1 failed"));
        assert!(summary.contains("PASS 'metric_a'"));
        assert!(summary.contains("FAIL 'metric_b'"));
    }
}
