use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use pyo3::prelude::*;

/// Configuration for numeric binning/bucketizing.
#[pyclass(module = "kzn_recsys._native")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NumericalBucketConfig {
    #[pyo3(get, set)]
    pub prefix: String,
    #[pyo3(get, set)]
    pub boundaries: Vec<f64>,
    #[pyo3(get, set)]
    pub labels: Vec<String>,
}

#[pymethods]
impl NumericalBucketConfig {
    #[new]
    #[pyo3(signature = (prefix, boundaries, labels))]
    pub fn new(prefix: String, boundaries: Vec<f64>, labels: Vec<String>) -> Self {
        Self {
            prefix,
            boundaries,
            labels,
        }
    }
}

/// Declarative feature transformation schema.
#[pyclass(module = "kzn_recsys._native")]
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct FeatureTransformationSchema {
    pub categorical_features: HashMap<String, String>, // raw_col -> prefix
    pub numerical_features: HashMap<String, NumericalBucketConfig>, // raw_col -> config
}

#[pymethods]
impl FeatureTransformationSchema {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    #[pyo3(signature = (col, prefix))]
    pub fn add_categorical(&mut self, col: String, prefix: String) {
        self.categorical_features.insert(col, prefix);
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

/// Transforms raw features to model-ready sparse string keys.
pub fn transform_features(
    raw_features: &HashMap<String, serde_json::Value>,
    schema: &FeatureTransformationSchema,
) -> HashMap<String, f64> {
    let mut transformed = HashMap::new();

    // 1. Process Categorical features
    for (col, prefix) in &schema.categorical_features {
        match raw_features.get(col) {
            Some(value) => {
                let val_str = match value {
                    serde_json::Value::String(s) => s.clone(),
                    serde_json::Value::Bool(b) => b.to_string(),
                    serde_json::Value::Number(n) => n.to_string(),
                    _ => continue,
                };
                transformed.insert(format!("{}{}", prefix, val_str), 1.0);
            }
            None => {
                transformed.insert(format!("{}_unknown", prefix), 1.0);
            }
        }
    }

    // 2. Process Numerical features
    for (col, config) in &schema.numerical_features {
        match raw_features.get(col) {
            Some(value) => {
                let val_f64 = match value {
                    serde_json::Value::Number(n) => n.as_f64(),
                    serde_json::Value::String(s) => s.parse::<f64>().ok(),
                    _ => None,
                };

                match val_f64 {
                    Some(val) => {
                        // Find the index of the first boundary that is greater than or equal to `val`
                        let mut bucket_idx = config.boundaries.len();
                        for (idx, &boundary) in config.boundaries.iter().enumerate() {
                            if val <= boundary {
                                bucket_idx = idx;
                                break;
                            }
                        }
                        if bucket_idx < config.labels.len() {
                            transformed.insert(format!("{}_{}", config.prefix, config.labels[bucket_idx]), 1.0);
                        } else {
                            transformed.insert(format!("{}_unknown", config.prefix), 1.0);
                        }
                    }
                    None => {
                        transformed.insert(format!("{}_unknown", config.prefix), 1.0);
                    }
                }
            }
            None => {
                transformed.insert(format!("{}_unknown", config.prefix), 1.0);
            }
        }
    }

    transformed
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_transform_features() {
        let mut schema = FeatureTransformationSchema::new();
        schema.add_categorical("plan".to_string(), "plan_".to_string());
        
        let bucket_config = NumericalBucketConfig::new(
            "tenure".to_string(),
            vec![0.0, 7.0, 30.0, 90.0],
            vec!["0d".to_string(), "7d".to_string(), "30d".to_string(), "90d".to_string(), "90d+".to_string()],
        );
        schema.add_numerical("tenure_days".to_string(), bucket_config);

        let mut raw = HashMap::new();
        raw.insert("plan".to_string(), json!("Premium"));
        raw.insert("tenure_days".to_string(), json!(45));

        let res = transform_features(&raw, &schema);
        assert_eq!(res.get("plan_Premium"), Some(&1.0));
        assert_eq!(res.get("tenure_90d"), Some(&1.0));

        // Test missing / null fields
        let empty_raw = HashMap::new();
        let res_empty = transform_features(&empty_raw, &schema);
        assert_eq!(res_empty.get("plan__unknown"), Some(&1.0));
        assert_eq!(res_empty.get("tenure_unknown"), Some(&1.0));
    }
}
