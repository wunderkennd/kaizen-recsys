//! Declarative raw-feature → model-key transformation (issue #71, theme A).
//!
//! At serve time, callers hold raw polymorphic user attributes ("plan":
//! "Premium", "tenure_days": 45) while the trained S-matrix indexes engineered
//! string keys ("plan_Premium", "tenure_30-90d"). Embedding the transformation
//! in the model (zero-drift online inference) requires it to be declarative
//! and serializable — this module is that layer.
//!
//! Ported from PR #62's branch with the review findings fixed:
//! - unknown-value keys no longer double the separator (`plan__unknown`);
//! - prefixes are normalized at construction (stored without a trailing `_`,
//!   formatters always insert one), so categorical and numerical keys follow
//!   one convention;
//! - `NumericalBucketConfig` validates its shape up front instead of silently
//!   falling through to `_unknown` on misconfigured label/boundary lengths.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Strip a single trailing separator so stored prefixes are canonical
/// ("plan_" and "plan" configure the same feature family).
fn normalize_prefix(prefix: &str) -> String {
    prefix.strip_suffix('_').unwrap_or(prefix).to_string()
}

/// Configuration for bucketizing one numeric column into labeled bins.
///
/// `boundaries` are ascending upper bounds (inclusive); `labels` name the
/// `boundaries.len() + 1` buckets they induce: `labels[i]` covers
/// `boundaries[i-1] < v <= boundaries[i]`, with the final label catching
/// everything above the last boundary.
#[pyclass(module = "kzn_recsys._native")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NumericalBucketConfig {
    #[pyo3(get)]
    pub prefix: String,
    #[pyo3(get)]
    pub boundaries: Vec<f64>,
    #[pyo3(get)]
    pub labels: Vec<String>,
}

impl NumericalBucketConfig {
    /// Pure-Rust fallible constructor; the PyO3 `#[new]` wraps it. Keeping
    /// validation free of Python symbols lets `cargo test` exercise it
    /// without linking libpython (extension-module cdylib constraint).
    pub fn try_new(
        prefix: String,
        boundaries: Vec<f64>,
        labels: Vec<String>,
    ) -> Result<Self, String> {
        if labels.len() != boundaries.len() + 1 {
            return Err(format!(
                "labels must have exactly boundaries + 1 entries (one per bucket, \
                 plus the above-last-boundary catch-all): got {} labels for {} boundaries",
                labels.len(),
                boundaries.len()
            ));
        }
        if boundaries.windows(2).any(|w| w[0] >= w[1]) {
            return Err("boundaries must be strictly ascending".to_string());
        }
        Ok(Self {
            prefix: normalize_prefix(&prefix),
            boundaries,
            labels,
        })
    }
}

#[pymethods]
impl NumericalBucketConfig {
    #[new]
    #[pyo3(signature = (prefix, boundaries, labels))]
    pub fn new(prefix: String, boundaries: Vec<f64>, labels: Vec<String>) -> PyResult<Self> {
        Self::try_new(prefix, boundaries, labels).map_err(PyValueError::new_err)
    }
}

/// Declarative feature transformation schema: raw column → engineered keys.
#[pyclass(module = "kzn_recsys._native")]
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct FeatureTransformationSchema {
    /// raw column → key prefix (stored normalized, no trailing `_`).
    pub categorical_features: HashMap<String, String>,
    /// raw column → bucket config.
    pub numerical_features: HashMap<String, NumericalBucketConfig>,
}

#[pymethods]
impl FeatureTransformationSchema {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    #[pyo3(signature = (col, prefix))]
    pub fn add_categorical(&mut self, col: String, prefix: String) {
        self.categorical_features
            .insert(col, normalize_prefix(&prefix));
    }

    #[pyo3(signature = (col, config))]
    pub fn add_numerical(&mut self, col: String, config: NumericalBucketConfig) {
        self.numerical_features.insert(col, config);
    }

    #[getter]
    pub fn get_categorical_features(&self) -> HashMap<String, String> {
        self.categorical_features.clone()
    }

    #[getter]
    pub fn get_numerical_features(&self) -> HashMap<String, NumericalBucketConfig> {
        self.numerical_features.clone()
    }
}

/// One key convention everywhere: `{prefix}_{suffix}`.
fn key(prefix: &str, suffix: &str) -> String {
    format!("{prefix}_{suffix}")
}

/// Transform raw polymorphic features into model-ready sparse string keys.
///
/// Missing columns and unparseable values map to `{prefix}_unknown`, which is
/// a *reachable* key: train-time data can carry the same sentinel so unknowns
/// receive a learned weight instead of silently scoring zero.
pub fn transform_features(
    raw_features: &HashMap<String, serde_json::Value>,
    schema: &FeatureTransformationSchema,
) -> HashMap<String, f64> {
    let mut transformed = HashMap::new();

    for (col, prefix) in &schema.categorical_features {
        let val_str = raw_features.get(col).and_then(|value| match value {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Bool(b) => Some(b.to_string()),
            serde_json::Value::Number(n) => Some(n.to_string()),
            _ => None,
        });
        let suffix = val_str.unwrap_or_else(|| "unknown".to_string());
        transformed.insert(key(prefix, &suffix), 1.0);
    }

    for (col, config) in &schema.numerical_features {
        let val_f64 = raw_features.get(col).and_then(|value| match value {
            serde_json::Value::Number(n) => n.as_f64(),
            serde_json::Value::String(s) => s.parse::<f64>().ok(),
            _ => None,
        });
        let entry = match val_f64 {
            Some(val) => {
                // First boundary >= val names the bucket; above the last
                // boundary falls into the final catch-all label. Validation
                // guarantees labels.len() == boundaries.len() + 1, so the
                // index is always in range.
                let bucket_idx = config
                    .boundaries
                    .iter()
                    .position(|&b| val <= b)
                    .unwrap_or(config.boundaries.len());
                key(&config.prefix, &config.labels[bucket_idx])
            }
            None => key(&config.prefix, "unknown"),
        };
        transformed.insert(entry, 1.0);
    }

    transformed
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tenure_config() -> NumericalBucketConfig {
        NumericalBucketConfig::try_new(
            "tenure".to_string(),
            vec![0.0, 7.0, 30.0, 90.0],
            vec!["0d", "7d", "30d", "90d", "90d+"]
                .into_iter()
                .map(String::from)
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn transforms_categorical_and_bucketed_numerical() {
        let mut schema = FeatureTransformationSchema::new();
        schema.add_categorical("plan".to_string(), "plan_".to_string());
        schema.add_numerical("tenure_days".to_string(), tenure_config());

        let mut raw = HashMap::new();
        raw.insert("plan".to_string(), json!("Premium"));
        raw.insert("tenure_days".to_string(), json!(45));

        let res = transform_features(&raw, &schema);
        assert_eq!(res.get("plan_Premium"), Some(&1.0));
        assert_eq!(res.get("tenure_90d"), Some(&1.0));
    }

    #[test]
    fn unknown_keys_use_single_separator() {
        // PR #62 bug 1: "plan_" + "_unknown" produced the unreachable
        // "plan__unknown". Normalized prefixes + one formatter make the
        // sentinel a real, trainable key.
        let mut schema = FeatureTransformationSchema::new();
        schema.add_categorical("plan".to_string(), "plan_".to_string());
        schema.add_numerical("tenure_days".to_string(), tenure_config());

        let res = transform_features(&HashMap::new(), &schema);
        assert_eq!(res.get("plan_unknown"), Some(&1.0));
        assert_eq!(res.get("tenure_unknown"), Some(&1.0));
        assert!(res.keys().all(|k| !k.contains("__")));
    }

    #[test]
    fn prefix_convention_is_uniform_with_or_without_trailing_underscore() {
        // PR #62 bug 2: categorical and numerical formatters disagreed on
        // whether the prefix carries the separator. Both spellings of the
        // prefix must yield identical keys.
        for prefix in ["plan", "plan_"] {
            let mut schema = FeatureTransformationSchema::new();
            schema.add_categorical("plan".to_string(), prefix.to_string());
            let mut raw = HashMap::new();
            raw.insert("plan".to_string(), json!("Free"));
            let res = transform_features(&raw, &schema);
            assert_eq!(res.get("plan_Free"), Some(&1.0), "prefix {prefix:?}");
        }
    }

    #[test]
    fn bucket_config_validates_shape() {
        // PR #62 bug 3: a labels/boundaries mismatch silently routed values
        // to _unknown. Now it's a construction-time error.
        let err = NumericalBucketConfig::try_new(
            "tenure".to_string(),
            vec![0.0, 7.0],
            vec!["a".to_string(), "b".to_string()], // needs 3
        )
        .unwrap_err();
        assert!(err.contains("boundaries + 1"));

        let err = NumericalBucketConfig::try_new(
            "tenure".to_string(),
            vec![7.0, 0.0],
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
        )
        .unwrap_err();
        assert!(err.contains("ascending"));
    }

    #[test]
    fn bucket_edges_are_inclusive_upper_bounds() {
        let config = tenure_config();
        let mut schema = FeatureTransformationSchema::new();
        schema.add_numerical("tenure_days".to_string(), config);

        let cases = [
            (json!(-3), "tenure_0d"),    // <= 0
            (json!(0), "tenure_0d"),     // boundary inclusive
            (json!(7), "tenure_7d"),     // boundary inclusive
            (json!(45), "tenure_90d"),   // 30 < v <= 90
            (json!(91), "tenure_90d+"),  // above last boundary
            (json!("12"), "tenure_30d"), // numeric string parses
        ];
        for (value, expected) in cases {
            let mut raw = HashMap::new();
            raw.insert("tenure_days".to_string(), value.clone());
            let res = transform_features(&raw, &schema);
            assert_eq!(res.get(expected), Some(&1.0), "value {value}");
        }
    }
}
