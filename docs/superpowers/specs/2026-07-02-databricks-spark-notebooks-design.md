# Databricks end-to-end notebooks for `kzn_recsys.spark`

**Date:** 2026-07-02
**Status:** Approved (design)

## Goal

Ship a series of Databricks notebooks that run the pure-Python / PySpark
EASE implementation (`kzn_recsys.spark`, no native extension) end to end —
ingest, feature-engineer, train, tune, evaluate, and predict — packaged as a
single **Databricks Asset Bundle (DAB)** with a multi-task Job, MLflow
logging, and a Delta predictions sink.

This complements, and does not replace, the existing **native**-path
material (`kzn_recsys/fease_train.py`, `notebooks/01-03`,
`kzn_recsys/Databricks_Training_Notebook.ipynb`), which installs the
compiled Rust wheel. The new notebooks exercise the portable Spark path that
runs where the native wheel cannot be installed.

## Design rules

1. **Thin notebooks, tested core.** The genuinely new logic — feature
   engineering a raw engagement/metadata schema into the three long-format
   tables, and synthetic-data generation — lives in a new
   `kzn_recsys/spark/databricks.py` module that ships in the pure-Python
   wheel and is covered by `tests/spark/`. Notebooks are thin orchestration
   over already-tested functions (`build_and_train`, splits, metrics,
   `feature_engineering`, `make_synthetic`). Notebooks themselves are not
   unit-tested; their logic is.
2. **The wheel is a bundle artifact.** DAB builds `packaging/pure-python`
   into `kzn_recsys_spark-*.whl` and attaches it as a task library.
   Databricks Runtime already provides numpy/scipy/pyspark, so install is
   dependency-free.
3. **Parameterized data, synthetic default.** Every notebook runs
   top-to-bottom with no external setup (`data_mode=synthetic`), and reads
   real Unity Catalog Delta tables when `data_mode=delta`.

## Repository layout

New `databricks/` bundle root; existing native `notebooks/` untouched.

```
databricks/
  databricks.yml                     # bundle: targets (dev/prod), artifacts (wheel), variable defaults
  resources/
    kzn_recsys_spark_job.yml         # 4-task Job DAG, cluster spec, widgets -> job params
  src/
    01_ingest_and_features.py        # Databricks source format (# COMMAND ----------)
    02_train_and_tune.py
    03_evaluate.py
    04_predict_and_sink.py
  README.md                          # deploy + run instructions
kzn_recsys/spark/databricks.py       # NEW: feature_engineering(), make_synthetic() — shipped in the wheel, tested
tests/spark/test_databricks.py       # NEW: coverage for the above
```

## New shipped module: `kzn_recsys/spark/databricks.py`

Pure PySpark, imports nothing native. Two public functions:

- `make_synthetic(spark, *, n_users, n_items, seed) -> tuple[DataFrame, DataFrame, DataFrame]`
  — deterministic synthetic `interactions`, `user_features`, `item_features`
  in long format (mirrors the persona structure used by `notebooks/02`).
- `feature_engineering(engagement_df, metadata_df, *, ...) -> tuple[DataFrame, DataFrame, DataFrame]`
  — map a raw engagement table (`user_id`, `item_id`, event columns) and a
  raw metadata table into the three long-format tables
  (`interactions(user_id, item_id, value[, event_type, days_ago])`,
  `user_features(user_id, feature_name, value)`,
  `item_features(item_id, feature_name, value)`).

Both return Spark DataFrames with the exact column contracts the EASE data
path expects, so their output feeds `build_and_train` directly.

## The four notebooks (each a Job task)

All read job parameters via `dbutils.widgets`; `data_mode` (`synthetic` |
`delta`) selects the source. Delta names default to bundle variables.
`taskValues` thread outputs downstream (table names, model path, MLflow run
id).

### 01_ingest_and_features
Resolve `data_mode`. If `synthetic`, call `make_synthetic(...)`; if `delta`,
read the raw engagement + metadata tables (`{catalog}.{schema}.*`). Run
`feature_engineering(...)` -> the three long-format DataFrames. Write each to
Delta (`{catalog}.{schema}.kzn_interactions`, `kzn_user_features`,
`kzn_item_features`). Emit the table names via `dbutils.jobs.taskValues`.

### 02_train_and_tune
Read the three Delta tables. `strategy` widget (`collect` default;
`distributed` shown with a one-cell note on when each wins — `collect` for
portability/moderate data, `distributed` when the Gram build should stay
Spark-side). Optional `do_tune` widget -> `grid_search` / `random_search`
over `lambda_` / `alpha` / `beta` on a user k-fold, selecting best NDCG@k.
Train the final `SparkEaseModel` on full data with the chosen params and an
optional `WeightingConfig`. Save the FEAS artifact to a Unity Catalog Volume
path; log params + the `.feas` artifact to MLflow (run id -> taskValues).

### 03_evaluate
Reload the tables; split with `temporal_split` (default) / `random_split` /
`leave_k_out_split`; retrain on the train fold with the chosen params;
compute precision / recall / NDCG / MAP / hit-rate@K plus catalog coverage.
Log metrics to the same MLflow run. Assert a configurable quality gate
(`NDCG@k >= ndcg_gate`) so the Job task fails loudly on a quality
regression, protecting the downstream predict task.

### 04_predict_and_sink
`load_model()` from the Volume path. Batch-predict top-K for warm users and
a cold-start user (features-only, no interactions). Write recommendations to
a Delta sink (`{catalog}.{schema}.kzn_predictions`, columns `user_id`,
`rank`, `item_id`, `score`, `run_id`).

**Task DAG:** `01 -> 02 -> 03 -> 04`, with `03`'s gate protecting `04`.

## Bundle & Job wiring

### databricks.yml
- `bundle.name: kzn_recsys_spark`.
- `targets`: `dev` (default, `mode: development`) and `prod`.
- `artifacts`: build the wheel from `packaging/pure-python`
  (`python -m build`).
- `variables`: `catalog`, `schema`, `volume_path`, `data_mode`,
  `n_users`, `n_items`, `seed`, `k`, `ndcg_gate`, `strategy`, `do_tune` —
  overridable per target and at the CLI.

### resources/kzn_recsys_spark_job.yml
One Job, four `tasks` (`ingest -> train -> evaluate -> predict`) chained by
`depends_on`. Each task: a `notebook_task` pointing at `../src/NN_*.py`,
`base_parameters` mapping bundle variables to widgets, and
`libraries: [{whl: <built artifact>}]`. A single shared `job_cluster`
(small single-node autoscaling, Spark 3.4+ runtime). `taskValues` thread the
Delta table names, model path, and MLflow run id downstream.

Deploy / run:
```bash
databricks bundle deploy -t dev
databricks bundle run kzn_recsys_spark_job -t dev
```

## Testing & validation

- **`tests/spark/test_databricks.py`** (local `SparkSession` fixture, no
  Databricks): `make_synthetic` produces the expected schema and row counts
  and is seed-deterministic; `feature_engineering` maps a known raw
  engagement/metadata input to the exact three long-format tables (values
  asserted, not just shapes); the output feeds `build_and_train` cleanly
  end to end.
- **Bundle validation**: `databricks bundle validate -t dev` in the manual
  test steps (config lints without a workspace).
- **Notebook logic** is exercised only through the extracted, tested
  functions. Notebooks are not executed in CI (that needs a live cluster);
  this is called out explicitly rather than implied as covered.

## Out of scope

- Model serving endpoints / real-time inference (batch predictions only).
- SASRec / Two-Tower (native-only models).
- Native-path notebooks (already exist; untouched).
- Streaming ingestion (batch Delta reads only).
