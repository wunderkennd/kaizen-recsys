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
src/lib.rs           — PyO3 entrypoint: FeaseModel, FeaseRegistry, build_and_train(), split/eval/tuning functions
    ↓
src/data_pipeline.rs — Reads Parquet/CSV via Polars, builds sparse CSR matrices (X, U, T), applies weighting
src/weighting.rs     — Event-type weights, temporal decay, IPS reweighting configs
    ↓
src/model.rs         — Core FEASE algorithm: block Gram matrix, inversion, S-matrix, sparsity pruning
    ↓
src/evaluation.rs    — Train/test splitting (random, temporal, leave-K-out), evaluation harness
src/tuning.rs        — Grid search and random search over hyperparameters with k-fold CV
src/metrics.rs       — Pure ranking metrics: precision, recall, NDCG, MAP, coverage, hit rate
    ↓
src/serialization.rs — Save/load via serde+bincode with magic bytes + format versioning (v1/v2)
src/serving.rs       — FeaseModelRegistry (territory routing), batch prediction (rayon-parallelized)
src/data_validation.rs — GaussianAnomalyDetector for pre-training data quality checks
```

### Key Rust Modules

- **`lib.rs`**: PyO3 bridge. Exposes `FeaseModel` (predict, predict_batch, predict_similar_items, evaluate, validate, save), `FeaseRegistry`, `build_and_train()`, `load_model()`, `validate_data()`, split functions, tuning functions, and standalone metrics.
- **`model.rs`**: Pure Rust `RustFeaseModel`. Training, prediction, MLT similarity, validation, sparsity pruning.
- **`data_pipeline.rs`**: Long-format Parquet/CSV → sparse CSR matrices + string↔index mappings. Hooks for weighting transforms.
- **`weighting.rs`**: `WeightingConfig` struct + functions: `apply_event_weights()`, `apply_temporal_decay()`, `apply_ips()`.
- **`evaluation.rs`**: `random_split()`, `temporal_split()`, `leave_k_out_split()` for data splitting; `evaluate_model()` harness computing per-K metrics + coverage.
- **`tuning.rs`**: `grid_search()` and `random_search()` over `HyperParams` with user-based k-fold CV optimizing NDCG@k.
- **`metrics.rs`**: Pure functions: `precision_at_k`, `recall_at_k`, `ndcg_at_k`, `mean_average_precision`, `coverage`, `hit_rate_at_k`.
- **`serialization.rs`**: Binary save/load with `FEAS` magic bytes, format v2 (persists `WeightingConfig`), v1 backward-compatible migration.
- **`serving.rs`**: `FeaseModelRegistry` for multi-territory model routing, `predict_batch()` / `predict_batch_top_k()` parallelized via rayon.
- **`data_validation.rs`**: `GaussianAnomalyDetector` — confidence interval checks for data quality.

### Python Layer (`kzn_recsys/`)

- `__init__.py` — Exports: `FeaseModel`, `FeaseRegistry`, `build_and_train`, `load_model`, `validate_data`, split functions, tuning functions, metrics, `EngagementSchema`, `MetadataSchema`
- `schemas.py` — Pydantic models for column validation
- `fease_wrapper.py` — Thin validation wrapper around `build_and_train()` with optional advanced weighting params
- `train.py` — CLI training script (`--interactions`, `--user-features`, `--item-features`, `--output`)
- `inference.py` — `FeasePredictor` class for loading saved models and serving predictions
- `fease_train.py` — Databricks end-to-end workflow (PySpark → Parquet → Rust training → predictions)

## Key Concepts

**Memory efficiency:** The combined matrix Z is never materialized. The Gram matrix G is computed in 4 sparse blocks (G_11, G_12, G_21, G_22), keeping memory at O((M+K)^2) independent of user count N.

**Hyperparameters:** `alpha` (item feature weight), `beta` (user feature weight), `lambda_` (L2 regularization, typical: 100-150), `meta_weight` (optional diagonal weighting for metadata rows).

**Advanced weighting:** `decay_rate` (exponential temporal decay on interactions), `ips_alpha` (inverse propensity scoring to debias popular items), `sparsity_threshold` (prune small S-matrix entries), `event_weights` (dict mapping event type to multiplier). All optional with backward-compatible defaults. Weighting requires optional columns in the interactions file: `event_type` (str) and/or `days_ago` (f64).

**Cold-start:** Users with zero interactions still get recommendations through user-feature columns in the S-matrix.

**Evaluation pipeline:** Three split strategies (random, temporal, leave-K-out) produce train/test Parquet files. The evaluation harness computes precision, recall, NDCG, MAP, hit rate at multiple K values, plus catalog coverage. All splits use sorted key iteration before RNG consumption to ensure deterministic results despite AHashMap's randomized hash seeds.

**Hyperparameter tuning:** Grid search and random search over `(alpha, beta, lambda_, meta_weight, decay_rate, ips_alpha, sparsity_threshold)` with user-based k-fold CV. Optimization target is NDCG@k. Results include best params, best score, and all trial details.

## Key Dependencies

**Rust:** nalgebra (dense LA), sprs (sparse CSR), polars (Parquet/CSV), pyo3 (Python bridge), ahash (fast hashing), bincode+serde (serialization), rayon (parallel batch prediction), rand (shuffling for splits/tuning), tempfile (k-fold temp files)
**Python:** polars, pydantic, pytest (dev)

## Data Format

The model trains from three long-format tables (Parquet or CSV):
- **Interactions**: `user_id`, `item_id`, `value` (+ optional `event_type`, `days_ago`)
- **User features**: `user_id`, `feature_name`, `value`
- **Item features**: `item_id`, `feature_name`, `value`

## Work tracking

Track multi-step or in-progress work via GitHub Issues, not markdown
files in the repo. Concretely:

- Do NOT create or maintain `TODO.md`, `STATUS.md`, `ROADMAP.md`, or
  progress banners inside ADRs / design docs.
- Open a GitHub Issue for each unit of work; checklists belong in Issue
  bodies, not in repo files.
- Link Issues from PRs (`Closes #N`, `Tracks #N`); link ADRs from
  Issues, not the other way around.
- This rule targets *progress/roadmap/status* tracking (what is done,
  in progress, blocked, or planned), not factual documentation.
  Describing what the software currently does and how to use it —
  capability/usage docs, feature-flag behavior, present-tense
  architecture — is fine and encouraged in `README.md` / ADRs / design
  docs. The test: state present behavior as fact; do not enumerate
  per-item completion status or phase banners (`(in progress)`,
  "not implemented yet", "lands in Phase N"). Roadmap lives in ADRs +
  Issues.

Rationale: status changes frequently and concurrently across agents
and branches. Tracking it in a single text file produces merge
conflicts every time two branches touch it; Issue state lives in the
GitHub API and is conflict-free.

## PR Review Policy

All PRs require a review from Devin (`devin-ai-integration[bot]`) before merging. When creating PRs, always request review:

```bash
gh pr edit <number> --add-reviewer devin-ai-integration[bot]
```

Do not merge PRs without an approved review from Devin. This is enforced via branch protection on `main`.

## Python Module Name

The compiled Rust extension is a submodule of the Python package: `kzn_recsys._native`. The public Python package is `kzn_recsys`, which re-exports the extension's symbols alongside Python helpers (`SplitResult`, schemas, `fease_wrapper`). End users should import from `kzn_recsys`, not `kzn_recsys._native` directly.
