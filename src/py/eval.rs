//! Split + data-validation PyO3 wrappers. Moved from `src/lib.rs` per ADR-0003.
//!
//! These four `#[pyfunction]`s are self-contained shims over
//! `crate::data_validation` (for `validate_data`) and `crate::evaluation`
//! (for the three split strategies). They have no references to
//! `FeaseModel`, `ModelRegistry`, or any other PyO3 type — moving them
//! is purely additive.

use crate::{data_validation, evaluation};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Runs data quality checks against historical baselines.
///
/// Uses the GaussianAnomalyDetector pattern from the Python EASE QA pipeline:
/// computes confidence intervals from historical data and checks if current
/// values fall within the expected range.
///
/// Args:
///     historical_users (list[float]): Historical distinct-user counts (e.g., per day).
///     historical_items (list[float]): Historical distinct-item counts.
///     historical_interactions (list[float]): Historical total-interaction counts.
///     current_users (float): Current run's distinct user count.
///     current_items (float): Current run's distinct item count.
///     current_interactions (float): Current run's total interaction count.
///     config (dict, optional): Override default std multipliers. Keys:
///         - "distinct_users_multiplier" (float, default 5.0)
///         - "distinct_items_multiplier" (float, default 5.0)
///         - "interactions_multiplier" (float, default 5.0)
///
/// Returns:
///     tuple[bool, list[str]]:
///         (all_passed, messages) where messages describe each check result.
#[pyfunction(signature = (
    historical_users,
    historical_items,
    historical_interactions,
    current_users,
    current_items,
    current_interactions,
    config = None
))]
#[allow(clippy::too_many_arguments)]
pub fn validate_data<'py>(
    py: Python<'py>,
    historical_users: Vec<f64>,
    historical_items: Vec<f64>,
    historical_interactions: Vec<f64>,
    current_users: f64,
    current_items: f64,
    current_interactions: f64,
    config: Option<&Bound<'_, PyDict>>,
) -> PyResult<(bool, Bound<'py, PyList>)> {
    let mut validation_config = data_validation::DataValidationConfig::default();

    if let Some(cfg) = config {
        if let Some(val) = cfg.get_item("distinct_users_multiplier")? {
            validation_config.distinct_users_multiplier = val.extract()?;
        }
        if let Some(val) = cfg.get_item("distinct_items_multiplier")? {
            validation_config.distinct_items_multiplier = val.extract()?;
        }
        if let Some(val) = cfg.get_item("interactions_multiplier")? {
            validation_config.interactions_multiplier = val.extract()?;
        }
    }

    let report = data_validation::validate_data_counts(
        &historical_users,
        &historical_items,
        &historical_interactions,
        current_users,
        current_items,
        current_interactions,
        &validation_config,
    );

    let messages: Vec<String> = report.results.iter().map(|r| r.to_string()).collect();
    let py_messages = PyList::new(py, &messages)?;

    Ok((report.all_passed(), py_messages))
}

/// Splits interaction data randomly per user into train and test sets.
///
/// Args:
///     interactions_path (str): Path to interactions Parquet/CSV file.
///     train_output (str): Output path for the train split.
///     test_output (str): Output path for the test split.
///     test_ratio (float): Fraction of each user's interactions to hold out (0.0-1.0).
///     seed (int): Random seed for reproducibility.
///
/// Returns:
///     tuple[int, int, int, int]: (train_interactions, test_interactions, train_users, test_users)
#[pyfunction]
#[pyo3(signature = (interactions_path, train_output, test_output, test_ratio=0.2, seed=42))]
pub fn random_split(
    interactions_path: &str,
    train_output: &str,
    test_output: &str,
    test_ratio: f64,
    seed: u64,
) -> PyResult<(usize, usize, usize, usize)> {
    let stats = evaluation::random_split(
        interactions_path,
        train_output,
        test_output,
        test_ratio,
        seed,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok((
        stats.train_interactions,
        stats.test_interactions,
        stats.train_users,
        stats.test_users,
    ))
}

/// Splits interaction data by time: recent interactions (days_ago <= cutoff) go to test.
///
/// Args:
///     interactions_path (str): Path to interactions Parquet/CSV file (must have `days_ago` column).
///     train_output (str): Output path for the train split.
///     test_output (str): Output path for the test split.
///     days_ago_cutoff (float): Interactions with days_ago <= this go to test.
///
/// Returns:
///     tuple[int, int, int, int]: (train_interactions, test_interactions, train_users, test_users)
#[pyfunction]
#[pyo3(signature = (interactions_path, train_output, test_output, days_ago_cutoff))]
pub fn temporal_split(
    interactions_path: &str,
    train_output: &str,
    test_output: &str,
    days_ago_cutoff: f64,
) -> PyResult<(usize, usize, usize, usize)> {
    let stats = evaluation::temporal_split(
        interactions_path,
        train_output,
        test_output,
        days_ago_cutoff,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok((
        stats.train_interactions,
        stats.test_interactions,
        stats.train_users,
        stats.test_users,
    ))
}

/// Leave-K-Out split: holds out exactly k random interactions per user for test.
///
/// Users with fewer than k+1 interactions go entirely to train.
///
/// Args:
///     interactions_path (str): Path to interactions Parquet/CSV file.
///     train_output (str): Output path for the train split.
///     test_output (str): Output path for the test split.
///     k (int): Number of interactions to hold out per user.
///     seed (int): Random seed for reproducibility.
///
/// Returns:
///     tuple[int, int, int, int]: (train_interactions, test_interactions, train_users, test_users)
#[pyfunction]
#[pyo3(signature = (interactions_path, train_output, test_output, k=1, seed=42))]
pub fn leave_k_out_split(
    interactions_path: &str,
    train_output: &str,
    test_output: &str,
    k: usize,
    seed: u64,
) -> PyResult<(usize, usize, usize, usize)> {
    let stats =
        evaluation::leave_k_out_split(interactions_path, train_output, test_output, k, seed)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok((
        stats.train_interactions,
        stats.test_interactions,
        stats.train_users,
        stats.test_users,
    ))
}
