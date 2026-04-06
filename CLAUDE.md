# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FEASE (Feature-augmented EASE) recommender — a Rust library exposed to Python via PyO3/maturin. Implements the paper "Shallow AutoEncoding Recommender with Cold Start Handling via Side Features." Computes a closed-form weight matrix S from sparse user-item interactions + side features, enabling cold-start recommendations.

## Build & Development Commands

```bash
# Install into current venv (dev mode, compiles Rust + installs Python package)
# IMPORTANT: use .venv/bin/maturin to target the correct Python
.venv/bin/maturin develop

# Build release wheel
maturin build --release --out dist

# Run tests (requires maturin develop first)
.venv/bin/python -m pytest tests/test_model.py -v

# Run a single test
.venv/bin/python -m pytest tests/test_model.py::test_warm_user_prediction -v

# Run Rust unit tests only
cargo test

# Docker wheel build (manylinux2014)
docker build . -t fease-builder
```

**Prerequisites:** Rust toolchain, Python 3.8+, maturin (`pip install maturin`), a virtual environment.

**Important:** This project has `[workspace]` in its Cargo.toml to prevent cargo from inheriting the parent directory's workspace. The `.venv` uses Python 3.14; always use `.venv/bin/python` or `.venv/bin/maturin` for consistent builds.

## Architecture

### Rust-Python Bridge (PyO3)

```
Python caller
    ↓
src/lib.rs           — PyO3 entrypoint: FeaseModel class, build_and_train(), load_model(), validate_data()
    ↓
src/data_pipeline.rs — Reads Parquet/CSV via Polars, builds sparse CSR matrices (X, U, T), applies weighting
src/weighting.rs     — Event-type weights, temporal decay, IPS reweighting configs
    ↓
src/model.rs         — Core FEASE algorithm: block Gram matrix, inversion, S-matrix, sparsity pruning
    ↓
src/serialization.rs — Save/load via serde+bincode with magic bytes + versioning
src/serving.rs       — FeaseModelRegistry (territory routing), batch prediction
src/data_validation.rs — GaussianAnomalyDetector for pre-training data quality checks
```

### Key Rust Modules

- **`lib.rs`**: PyO3 bridge. Exposes `FeaseModel` (predict, predict_batch, predict_similar_items, validate, save), `build_and_train()`, `load_model()`, `validate_data()`.
- **`model.rs`**: Pure Rust `RustFeaseModel`. Training, prediction, MLT similarity, validation, sparsity pruning.
- **`data_pipeline.rs`**: Long-format Parquet/CSV → sparse CSR matrices + string↔index mappings. Hooks for weighting transforms.
- **`weighting.rs`**: `WeightingConfig` struct + functions: `apply_event_weights()`, `apply_temporal_decay()`, `apply_ips()`.
- **`serialization.rs`**: Binary save/load with `FEAS` magic bytes, format versioning, dimension validation on load.
- **`serving.rs`**: `FeaseModelRegistry` for multi-territory model routing, `predict_batch()` / `predict_batch_top_k()`.
- **`data_validation.rs`**: `GaussianAnomalyDetector` — confidence interval checks for data quality.

### Python Layer (`cr_fease/`)

- `__init__.py` — Exports: `FeaseModel`, `build_and_train`, `load_model`, `validate_data`, `EngagementSchema`, `MetadataSchema`
- `schemas.py` — Pydantic models for column validation
- `fease_wrapper.py` — Thin validation wrapper around `build_and_train()`
- `train.py` — CLI training script (`--interactions`, `--user-features`, `--item-features`, `--output`)
- `inference.py` — `FeasePredictor` class for loading saved models and serving predictions
- `fease_train.py` — Databricks end-to-end workflow (PySpark → Parquet → Rust training → predictions)

### `src/schemas/` (Rust)

Legacy schema definitions (account, content, engagement structs). Not wired into the main pipeline.

## Key Concepts

**Memory efficiency:** The combined matrix Z is never materialized. The Gram matrix G is computed in 4 sparse blocks (G_11, G_12, G_21, G_22), keeping memory at O((M+K)^2) independent of user count N.

**Hyperparameters:** `alpha` (item feature weight), `beta` (user feature weight), `lambda_` (L2 regularization, typical: 100-150), `meta_weight` (optional diagonal weighting for metadata rows).

**Advanced weighting:** `decay_rate` (exponential temporal decay on interactions), `ips_alpha` (inverse propensity scoring to debias popular items), `sparsity_threshold` (prune small S-matrix entries), `event_weights` (dict mapping event type to multiplier). All optional with backward-compatible defaults. Weighting requires optional columns in the interactions file: `event_type` (str) and/or `days_ago` (f64).

**Cold-start:** Users with zero interactions still get recommendations through user-feature columns in the S-matrix.

## Key Dependencies

**Rust:** nalgebra (dense LA), sprs (sparse CSR), polars (Parquet/CSV), pyo3 (Python bridge), ahash (fast hashing), bincode+serde (serialization)
**Python:** polars, pydantic, pytest (dev)

## Data Format

The model trains from three long-format tables (Parquet or CSV):
- **Interactions**: `user_id`, `item_id`, `value` (+ optional `event_type`, `days_ago`)
- **User features**: `user_id`, `feature_name`, `value`
- **Item features**: `item_id`, `feature_name`, `value`

## Python Module Name

The compiled Rust library is importable as `rust_fease_recommender`. The Python package is `cr_fease`.
