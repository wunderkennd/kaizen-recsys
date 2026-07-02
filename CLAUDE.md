# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`kzn_recsys` is a multi-model recommender library — a Rust core
exposed to Python via PyO3/maturin. Three models live behind one
`RecModel` trait (see
[ADR-0001](docs/adr/0001-multi-model-architecture.md) and the
[model-comparison guide](docs/guides/model-comparison.md)):

- **EASE** (default) — closed-form, deterministic, feature-augmented
  shallow autoencoder. Implements "Shallow AutoEncoding Recommender with
  Cold Start Handling via Side Features." This is the only model
  compiled by default; published wheels are EASE-only.
- **SASRec** — self-attentive sequential transformer for next-item
  prediction. Requires the `ml-models` Cargo feature.
- **Two-Tower** — dual-tower retrieval with in-batch sampled-softmax and
  learned cold-start prior (id-dropout). Requires the `ml-models` Cargo
  feature.

The evaluation, tuning, and serving layers (`src/evaluation.rs`,
`src/tuning.rs`, `src/serving.rs`) are model-agnostic — they operate on
`&dyn RecModel`. EASE behavior and on-disk outputs are byte-identical to
the pre-multi-model baseline.

## Build & Development Commands

```bash
# Install into current venv (dev mode, compiles Rust + installs Python package)
# IMPORTANT: use .venv/bin/maturin to target the correct Python
.venv/bin/maturin develop

# Same, with SASRec + Two-Tower (adds the `burn` dependency)
.venv/bin/maturin develop --features ml-models

# Build release wheel (EASE-only)
maturin build --release --out dist

# Build release wheel with all models
maturin build --release --features ml-models --out dist

# Run Python tests (requires maturin develop first)
.venv/bin/python -m pytest tests/test_model.py -v
.venv/bin/python -m pytest tests/test_sasrec.py -v       # needs --features ml-models
.venv/bin/python -m pytest tests/test_two_tower.py -v    # needs --features ml-models

# Run a single test
.venv/bin/python -m pytest tests/test_model.py::test_warm_user_prediction -v

# Run Rust unit tests
cargo test                                  # EASE-only
cargo test --features ml-models             # all models
cargo test --features ml-models sasrec      # SASRec only
cargo test --release --features ml-models -- --ignored
                                            # SASRec + Two-Tower end-to-end
                                            # hyperparameter-search tests
                                            # (release-mode burn; debug is
                                            # impractical on hosted CI)

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
src/lib.rs           — PyO3 entrypoint: FeaseModel, ModelRegistry, SASRecModel,
                       TwoTowerModel (last two gated on `ml-models`),
                       build_and_train{,_sasrec,_two_tower}, load_*_model,
                       per-model grid_search_* / random_search_*, split/eval helpers
    ↓
src/models/          — RecModel trait + adapters / impls (see below)
src/data/            — sequences.rs (SASRec), triples.rs (Two-Tower)
src/data_pipeline.rs — Long-format Parquet/CSV → sparse CSR matrices (EASE) and mappings
src/weighting.rs     — Event-type weights, temporal decay, IPS reweighting configs
    ↓
src/model.rs         — Core EASE algorithm: block Gram matrix, inversion, S-matrix, sparsity pruning
    ↓
src/evaluation.rs    — Train/test splitting (random, temporal, leave-K-out),
                       generic evaluate_model harness over &dyn RecModel
src/tuning.rs        — SearchSpace + FoldEvaluator traits; grid/random search generic over
                       any model's HyperParams; per-model schemas (HyperParams /
                       SasRecParams / TwoTowerParams) + FoldEvaluator impls
src/metrics.rs       — Pure ranking metrics: precision, recall, NDCG, MAP, coverage, hit rate
    ↓
src/serialization.rs — EASE save/load (FEAS magic bytes, v1/v2 versioning).
                       SASRec uses FSAS framed format; Two-Tower FTWO. load_model
                       sniffs the magic bytes and dispatches.
src/serving.rs       — ModelRegistry generic over Box<dyn RecModel>, batch
                       prediction (rayon-parallelized)
src/data_validation.rs — GaussianAnomalyDetector for pre-training data quality checks
```

### Key Rust Modules

- **`lib.rs`**: PyO3 bridge. Exposes `FeaseModel` (predict, predict_batch, predict_similar_items, evaluate, validate, save), `ModelRegistry`, EASE `build_and_train()` / `load_model()`, `validate_data()`, split functions, EASE search functions (`grid_search` / `random_search`), per-model search functions (`grid_search_{ease,sasrec,two_tower}` and `random_search_*`), and standalone metrics. Under `ml-models`, also exposes `SASRecModel` / `build_and_train_sasrec` / `load_sasrec_model` and `TwoTowerModel` / `build_and_train_two_tower` / `load_two_tower_model`.
- **`models/mod.rs`**: `RecModel` trait (`kind`, `num_items`, `item_mapping`, `predict_scores(ModelInput<'_>)`, `predict_similar_items`, `validate`, `save`) + `ModelInput` enum (`Sparse`, `Sequence`, `TowerUser`).
- **`models/ease.rs`**: `EaseAdapter` / `EaseAdapterRef` — implement `RecModel` over `RustFeaseModel` with no algorithmic change.
- **`models/sasrec.rs`**: `SasRecConfig`, `SasRecTrainingConfig`, `train_sasrec()`, `TrainedSasRec` (transformer; `burn` backend). Magic-bytes-framed save/load.
- **`models/two_tower.rs`**: `TwoTowerConfig`, `TrainParams`, `train()`, `TrainedTwoTower`. Reserved cold-start row trained via `id_dropout`. Magic-bytes-framed save/load.
- **`model.rs`**: Pure Rust `RustFeaseModel`. EASE training, prediction, MLT similarity, validation, sparsity pruning.
- **`data/sequences.rs`**: Build chronologically-ordered left-padded item sequences for SASRec from a long-format interactions file with a `days_ago` column.
- **`data/triples.rs`**: Load `(user_idx, item_idx)` positive pairs (`TripleData`) and `FeatureTable` (categorical + dense per entity) for Two-Tower. Reserves a cold-start user row at index 0.
- **`data_pipeline.rs`**: Long-format Parquet/CSV → sparse CSR matrices + string↔index mappings (EASE path). Hooks for weighting transforms.
- **`weighting.rs`**: `WeightingConfig` struct + functions: `apply_event_weights()`, `apply_temporal_decay()`, `apply_ips()`.
- **`evaluation.rs`**: `random_split()`, `temporal_split()`, `leave_k_out_split()` for data splitting; `evaluate_model()` harness generic over `&dyn RecModel`. Per-user input construction routes through the `EvalAdapter` trait (`EaseEvalAdapter` → `Sparse`; `SasRecEvalAdapter` → chronologically-sorted `Sequence`, requires `days_ago` in the train file; `TwoTowerEvalAdapter` → `TowerUser`). `evaluate_with_adapter()` is the lower-level entrypoint used by tuning's per-fold scorer (#51).
- **`tuning.rs`**: `SearchSpace` and `FoldEvaluator<P>` traits; `grid_search_with` / `random_search_with` runners generic over `P`. EASE keeps `grid_search()` / `random_search()` (`P = HyperParams`) for callers; per-model entrypoints layer on top. Parallelized via rayon (ADR-0002).
- **`metrics.rs`**: Pure functions: `precision_at_k`, `recall_at_k`, `ndcg_at_k`, `mean_average_precision`, `coverage`, `hit_rate_at_k`.
- **`serialization.rs`**: Binary save/load with `FEAS` magic bytes for EASE (format v2 persists `WeightingConfig`, v1 backward-compatible migration); top-level `load_model()` sniffs the magic bytes and dispatches to EASE / SASRec / Two-Tower loaders.
- **`serving.rs`**: `ModelRegistry` for multi-territory model routing — stores `Box<dyn RecModel>`. `register()` keeps the EASE adapter shortcut; `register_model()` accepts any `RecModel`. String-id-native per-model predict methods (`predict_top_k_ease`, `predict_top_k_sasrec`, `predict_top_k_two_tower`) mirror the standalone model classes' input shapes; the legacy index-based `predict_top_k` is preserved (#56). `predict_batch()` / `predict_batch_top_k()` parallelized via rayon.
- **`data_validation.rs`**: `GaussianAnomalyDetector` — confidence interval checks for data quality.

### Python Layer (`kzn_recsys/`)

- `__init__.py` — Exports when the native extension is present (`_HAS_NATIVE`, i.e. any wheel except the pure-Python `kzn_recsys_spark` one): `FeaseModel`, `ModelRegistry`, `build_and_train`, `load_model`, `validate_data`, split functions, `grid_search` / `grid_search_ease` / `random_search` / `random_search_ease`, metrics. Exports when pydantic + polars are installed (`_HAS_SCHEMAS`): `EngagementSchema`, `MetadataSchema`. The pure-Python install exposes the recommender via the `kzn_recsys.spark` subpackage instead. Conditional on the extension being built with `--features ml-models` (gated by `_HAS_ML_MODELS`): `SASRecModel`, `build_and_train_sasrec`, `load_sasrec_model`, `grid_search_sasrec`, `random_search_sasrec`, `TwoTowerModel`, `build_and_train_two_tower`, `load_two_tower_model`, `grid_search_two_tower`, `random_search_two_tower`.
- `schemas.py` — Pydantic models for column validation
- `fease_wrapper.py` — Thin validation wrapper around `build_and_train()` with optional advanced weighting params
- `train.py` — CLI training script (`--interactions`, `--user-features`, `--item-features`, `--output`)
- `inference.py` — `FeasePredictor` class for loading saved models and serving predictions
- `fease_train.py` — Databricks end-to-end workflow (PySpark → Parquet → Rust training → predictions)
- `onnx_export/` — optional ONNX export (requires the `[onnx]` extra:
  `pip install kzn_recsys[onnx]`): `export_onnx(model, out_dir, *, top_k_default,
  dtype, repeat_penalty_default, mlflow)` writes `model.onnx` (Gemm scoring +
  configurable repeat penalty + eligibility mask + TopK + `raw_scores`), a
  `vocab.json` sidecar, and an optional MLflow pyfunc model. See
  `docs/superpowers/specs/2026-06-01-onnx-export-design.md`. Regenerate the Rust
  ort parity fixtures with `kzn_recsys.onnx_export._write_rust_fixture(model, "tests/fixtures")`.

## Key Concepts

**`ml-models` Cargo feature.** Off by default. Gates the optional `burn`
dependency and the `sasrec` / `two_tower` model modules + their Python
wrappers. With the feature off, the dependency graph, build time, and
wheel size are unchanged; published PyPI wheels are EASE-only by design.

**RecModel trait + ModelInput.** All three models implement `RecModel`,
which exposes `predict_scores(ModelInput<'_>) -> Vec<f32>`. `ModelInput`
is an enum (`Sparse` for EASE, `Sequence` for SASRec, `TowerUser` for
Two-Tower); a model that doesn't support a variant returns `Err` rather
than silently producing nonsense. The trait is **not** GAT-based so it
stays object-safe — registries and the eval harness use `&dyn RecModel`.

**EASE memory efficiency.** The combined matrix Z is never materialized.
The Gram matrix G is computed in 4 sparse blocks (G_11, G_12, G_21,
G_22), keeping memory at `O((M+K)^2)` independent of user count `N`.

**EASE hyperparameters.** `alpha` (item feature weight), `beta` (user
feature weight), `lambda_` (L2 regularization, typical: 100–150),
`meta_weight` (optional diagonal weighting for metadata rows).

**EASE advanced weighting.** `decay_rate` (exponential temporal decay on
interactions), `ips_alpha` (inverse propensity scoring to debias popular
items), `sparsity_threshold` (prune small S-matrix entries),
`event_weights` (dict mapping event type to multiplier). All optional
with backward-compatible defaults. Requires optional columns in the
interactions file: `event_type` (str) and/or `days_ago` (f64).

**SASRec hyperparameters.** `embedding_dim`, `num_heads`, `num_layers`,
`dropout`, `max_seq_len`, `num_epochs`, `batch_size`, `learning_rate`,
`patience`, `seed`. Interactions file **must** include a numeric
`days_ago` column so each user's history is chronologically ordered;
training fails loudly otherwise.

**Two-Tower hyperparameters.** `embedding_dim`, `temperature`,
`learning_rate`, `epochs`, `batch_size`, `id_dropout`, `seed`. The
`id_dropout` fraction is the share of training rows whose user id is
remapped to the reserved cold-start row so that row receives gradient
and learns an average-user prior (PR #46).

**Cold-start.**
- *EASE*: users with zero interactions get recommendations through
  user-feature columns in the S-matrix; works at predict time with
  arbitrary side features.
- *SASRec*: empty history → no recommendation (model is sequence-only).
- *Two-Tower*: unknown user ids fall back to the reserved cold-start row.
  Predict-time arbitrary user features (`predict(user_id, features=...)`)
  are routed through the user-feature-name → category-index / dense-
  column maps persisted on the trained model (file format v5, #55), so
  a cold-start user with informative side info combines its features
  with the learned cold-start prior instead of falling back to the bare
  prior. Unknown feature names are silently skipped.

**Evaluation pipeline.** Three split strategies (random, temporal,
leave-K-out) produce train/test Parquet files. The evaluation harness is
generic over `&dyn RecModel` and computes precision, recall, NDCG, MAP,
hit rate at multiple K values, plus catalog coverage. All splits use
sorted key iteration before RNG consumption to ensure deterministic
results despite AHashMap's randomized hash seeds.

**Hyperparameter tuning.** Grid search and random search over each
model's parameter type with user-based k-fold CV. Optimization target is
NDCG@k. Trials run in parallel via rayon (ADR-0002); `RAYON_NUM_THREADS`
caps concurrency. EASE callers use `grid_search()` / `random_search()`;
new code can use the explicit `grid_search_ease` / `grid_search_sasrec`
/ `grid_search_two_tower` (and `random_search_*`) entrypoints.

## Key Dependencies

**Rust:** nalgebra (dense LA), sprs (sparse CSR), polars (Parquet/CSV), pyo3 (Python bridge), ahash (fast hashing), bincode+serde (serialization), rayon (parallel batch prediction + tuning), rand (shuffling for splits/tuning), tempfile (k-fold temp files), **`burn` + `burn-ndarray` (gated on `ml-models` for SASRec / Two-Tower)**
**Python:** polars, pydantic, pytest (dev)

## Data Format

The models read from long-format tables (Parquet or CSV):
- **Interactions**: `user_id`, `item_id`, `value` (+ optional `event_type` for EASE weighting, **required `days_ago` for SASRec**)
- **User features**: `user_id`, `feature_name`, `value` (used by EASE and Two-Tower)
- **Item features**: `item_id`, `feature_name`, `value` (used by EASE and Two-Tower)

See [`docs/guides/model-comparison.md`](docs/guides/model-comparison.md)
for a side-by-side of data requirements, cold-start behavior, and
hyperparameter starting points.

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

Devin (`devin-ai-integration[bot]`) auto-reviews every PR via the `.github/workflows/request_devin_review.yml` workflow (triggered on `opened` and `ready_for_review`) and via the Devin GitHub App's webhook. **No manual action is required to request the review.**

Do not attempt to add Devin as a reviewer via `gh pr edit --add-reviewer 'devin-ai-integration[bot]'` from a local checkout: the GitHub REST API does not let user PATs request reviews from GitHub Apps, and the call fails with `Could not resolve user`. The CI workflow itself appends `|| true` to acknowledge that the same call can fail under GitHub's own token; the actual review request reaches Devin through its app webhook, not through the reviewer-request endpoint.

This repository is currently solo-developed and `main` is not branch-protected. Devin's review is therefore a strong signal to consult, not a hard merge gate — wait for it on anything non-trivial, but you can self-merge when the change is small, the review is positive, or the situation warrants it. When a collaborator is added, restore branch protection (Settings → Branches, or `gh api -X PUT /repos/wunderkennd/kaizen-recsys/branches/main/protection`) and treat Devin's approval as required again.

## Python Module Name

The compiled Rust extension is a submodule of the Python package: `kzn_recsys._native`. The public Python package is `kzn_recsys`, which re-exports the extension's symbols alongside Python helpers (`SplitResult`, schemas, `fease_wrapper`). End users should import from `kzn_recsys`, not `kzn_recsys._native` directly.
