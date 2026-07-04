# Databricks Spark Notebooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an end-to-end Databricks Asset Bundle that runs the pure-Python `kzn_recsys.spark` EASE pipeline (ingest → feature-engineer → train → tune → evaluate → predict) as a 4-task Job, with MLflow logging and a Delta predictions sink.

**Architecture:** New tested helpers (`make_synthetic`, `feature_engineering`) ship in the pure-Python wheel under `kzn_recsys/spark/databricks.py`. Four thin Databricks-source notebooks orchestrate the already-tested library (`build_and_train`, splits, tuning, metrics) plus the new helpers. A Databricks Asset Bundle (`databricks/`) builds the wheel as an artifact and wires the notebooks into a multi-task Job.

**Tech Stack:** PySpark (Databricks Runtime 14.3 LTS / Spark 3.5), NumPy/SciPy, Databricks Asset Bundles (`databricks` CLI), MLflow, Delta, Unity Catalog Volumes.

## Global Constraints

- **No native imports.** `kzn_recsys/spark/databricks.py` and all notebooks must import only from `kzn_recsys.spark`, `pyspark`, `numpy`, `scipy`, `mlflow` — never `kzn_recsys._native`, `pydantic`, or `polars`. The Spark wheel runs where those are absent.
- **PySpark ≥ 3.4** API only (Databricks Runtime provides it; do not add it as a wheel dependency).
- **Determinism:** any randomness is seeded; synthetic data is generated in Python under a seeded `random.Random`/`numpy.random.default_rng` and handed to `spark.createDataFrame` (never `rand()`/`monotonically_increasing_id()` — see the existing splits/tuning determinism fixes).
- **Long-format column contracts (exact):**
  - interactions: `user_id: string`, `item_id: string`, `value: double` (+ optional `event_type: string`, `days_ago: double`)
  - user_features: `user_id: string`, `feature_name: string`, `value: double`
  - item_features: `item_id: string`, `feature_name: string`, `value: double`
- **Tests** run under the existing `tests/spark/` session-scoped `spark` fixture (`tests/spark/conftest.py`), invoked as `.venv/bin/python -m pytest`.
- **Commit style:** conventional commits (`feat`/`test`/`docs`/`build`), no attribution footer.

---

### Task 1: `make_synthetic` helper

**Files:**
- Create: `kzn_recsys/spark/databricks.py`
- Test: `tests/spark/test_databricks.py`

**Interfaces:**
- Consumes: a live `SparkSession` (from the test fixture / notebook).
- Produces: `make_synthetic(spark, *, n_users=200, n_items=60, n_personas=4, avg_interactions=15, seed=42) -> tuple[DataFrame, DataFrame, DataFrame]` returning `(interactions_df, user_features_df, item_features_df)` in the exact long-format contracts above. `interactions.value` is `1.0` for every generated (user, item) pair; `user_features` carries `persona={0..n_personas-1}` one-hot rows (`feature_name=f"persona={p}"`, `value=1.0`); `item_features` carries `genre=...` one-hot rows.

- **Step 1: Write the failing test**

```python
# tests/spark/test_databricks.py
import pytest

pytest.importorskip("numpy")
from pyspark.sql import functions as F
from kzn_recsys.spark.databricks import make_synthetic


def test_make_synthetic_schema_and_determinism(spark):
    inter1, users1, items1 = make_synthetic(spark, n_users=50, n_items=20,
                                            n_personas=3, avg_interactions=8, seed=7)

    # exact long-format column contracts
    assert inter1.columns == ["user_id", "item_id", "value"]
    assert users1.columns == ["user_id", "feature_name", "value"]
    assert items1.columns == ["item_id", "feature_name", "value"]

    # value column is all 1.0 for interactions
    assert inter1.filter(F.col("value") != 1.0).count() == 0

    # every user appears with at least one interaction and one persona feature
    assert inter1.select("user_id").distinct().count() == 50
    assert users1.filter(F.col("feature_name").startswith("persona=")).count() == 50
    # every item has a genre feature
    assert items1.select("item_id").distinct().count() == 20

    # seed-determinism: same seed -> identical interaction rows
    inter2, _, _ = make_synthetic(spark, n_users=50, n_items=20,
                                  n_personas=3, avg_interactions=8, seed=7)
    rows1 = sorted((r.user_id, r.item_id) for r in inter1.collect())
    rows2 = sorted((r.user_id, r.item_id) for r in inter2.collect())
    assert rows1 == rows2
```

- **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/spark/test_databricks.py::test_make_synthetic_schema_and_determinism -v`
Expected: FAIL with `ModuleNotFoundError` / `cannot import name 'make_synthetic'`.

- **Step 3: Write minimal implementation**

```python
# kzn_recsys/spark/databricks.py
"""Databricks helpers for the pure-Python Spark EASE pipeline.

Imports nothing native (no kzn_recsys._native, pydantic, or polars): this
module ships in the pure-Python wheel and runs on stock Databricks Runtime.
"""
from __future__ import annotations

import random

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    DoubleType, StringType, StructField, StructType,
)

_INTERACTIONS_SCHEMA = StructType([
    StructField("user_id", StringType(), False),
    StructField("item_id", StringType(), False),
    StructField("value", DoubleType(), False),
])
_FEATURES_SCHEMA_USER = StructType([
    StructField("user_id", StringType(), False),
    StructField("feature_name", StringType(), False),
    StructField("value", DoubleType(), False),
])
_FEATURES_SCHEMA_ITEM = StructType([
    StructField("item_id", StringType(), False),
    StructField("feature_name", StringType(), False),
    StructField("value", DoubleType(), False),
])

_GENRES = ["action", "drama", "comedy", "romance", "thriller", "slice_of_life"]


def make_synthetic(spark: SparkSession, *, n_users: int = 200, n_items: int = 60,
                   n_personas: int = 4, avg_interactions: int = 15,
                   seed: int = 42) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Generate deterministic synthetic long-format tables with persona structure.

    Each persona prefers a contiguous slice of the catalog, so EASE has real
    co-occurrence signal to learn. All randomness is seeded in Python and
    handed to spark.createDataFrame for determinism.
    """
    rng = random.Random(seed)

    # Assign each item a genre and each persona a preferred item slice.
    item_ids = [f"item_{i:04d}" for i in range(n_items)]
    item_genre = {iid: _GENRES[i % len(_GENRES)] for i, iid in enumerate(item_ids)}
    slice_size = max(1, n_items // n_personas)

    inter_rows, user_rows = [], []
    for u in range(n_users):
        uid = f"user_{u:05d}"
        persona = u % n_personas
        user_rows.append((uid, f"persona={persona}", 1.0))
        lo = persona * slice_size
        hi = min(n_items, lo + slice_size)
        pool = item_ids[lo:hi] or item_ids
        k = min(len(pool), max(1, avg_interactions))
        for iid in rng.sample(pool, k):
            inter_rows.append((uid, iid, 1.0))

    item_rows = [(iid, f"genre={item_genre[iid]}", 1.0) for iid in item_ids]

    interactions = spark.createDataFrame(inter_rows, _INTERACTIONS_SCHEMA)
    users = spark.createDataFrame(user_rows, _FEATURES_SCHEMA_USER)
    items = spark.createDataFrame(item_rows, _FEATURES_SCHEMA_ITEM)
    return interactions, users, items
```

- **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/spark/test_databricks.py::test_make_synthetic_schema_and_determinism -v`
Expected: PASS.

- **Step 5: Commit**

```bash
git add kzn_recsys/spark/databricks.py tests/spark/test_databricks.py
git commit -m "feat(spark): synthetic data generator for Databricks notebooks"
```

---

### Task 2: `feature_engineering` helper

**Files:**
- Modify: `kzn_recsys/spark/databricks.py`
- Test: `tests/spark/test_databricks.py`

**Interfaces:**
- Consumes: raw `engagement_df` and `metadata_df` Spark DataFrames.
- Produces: `feature_engineering(engagement_df, metadata_df, *, user_col="user_id", item_col="item_id", value_col="value", event_type_col=None, days_ago_col=None, user_feature_cols=(), item_feature_cols=(), item_key_col="item_id") -> tuple[DataFrame, DataFrame, DataFrame]` returning the three long-format tables. Categorical feature columns are one-hot-encoded as `feature_name=f"{col}={value}"`, `value=1.0`. When neither `event_type_col` nor `days_ago_col` is given, interactions are summed per `(user_id, item_id)`; when either is given, rows pass through with the optional columns preserved (weighting consumes per-event rows).

- **Step 1: Write the failing test**

```python
# add to tests/spark/test_databricks.py
from kzn_recsys.spark.databricks import feature_engineering


def test_feature_engineering_maps_raw_to_long(spark):
    engagement = spark.createDataFrame(
        [
            # user, item, watched, country
            ("u1", "m1", 3.0, "US"),
            ("u1", "m1", 2.0, "US"),   # duplicate (u1,m1) -> summed to 5.0
            ("u1", "m2", 1.0, "US"),
            ("u2", "m2", 4.0, "JP"),
        ],
        ["user_id", "item_id", "watched", "country"],
    )
    metadata = spark.createDataFrame(
        [("m1", "action"), ("m2", "drama")],
        ["item_id", "genre"],
    )

    inter, users, items = feature_engineering(
        engagement, metadata,
        value_col="watched",
        user_feature_cols=["country"],
        item_feature_cols=["genre"],
    )

    assert inter.columns == ["user_id", "item_id", "value"]
    # duplicate (u1,m1) summed
    val = {(r.user_id, r.item_id): r.value for r in inter.collect()}
    assert val[("u1", "m1")] == 5.0
    assert val[("u2", "m2")] == 4.0

    # one-hot user features, deduped per (user, feature)
    ufeat = {(r.user_id, r.feature_name) for r in users.collect()}
    assert ("u1", "country=US") in ufeat
    assert ("u2", "country=JP") in ufeat
    assert users.filter(F.col("value") != 1.0).count() == 0

    # one-hot item features from metadata
    ifeat = {(r.item_id, r.feature_name) for r in items.collect()}
    assert ("m1", "genre=action") in ifeat
    assert ("m2", "genre=drama") in ifeat


def test_feature_engineering_preserves_optional_columns(spark):
    engagement = spark.createDataFrame(
        [("u1", "m1", 1.0, "play", 2.0), ("u1", "m1", 1.0, "like", 1.0)],
        ["user_id", "item_id", "value", "event_type", "days_ago"],
    )
    metadata = spark.createDataFrame([("m1", "action")], ["item_id", "genre"])
    inter, _, _ = feature_engineering(
        engagement, metadata,
        event_type_col="event_type", days_ago_col="days_ago",
        item_feature_cols=["genre"],
    )
    # rows pass through (not summed) with the optional columns present
    assert set(inter.columns) == {"user_id", "item_id", "value", "event_type", "days_ago"}
    assert inter.count() == 2
```

- **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/spark/test_databricks.py -k feature_engineering -v`
Expected: FAIL with `cannot import name 'feature_engineering'`.

- **Step 3: Write minimal implementation**

```python
# add to kzn_recsys/spark/databricks.py
from functools import reduce
from pyspark.sql import functions as F


def _one_hot_long(df: DataFrame, key_col: str, feature_cols, key_out: str) -> DataFrame:
    """Melt categorical columns into (key_out, feature_name, value=1.0) long rows."""
    parts = []
    for col in feature_cols:
        parts.append(
            df.select(
                F.col(key_col).cast("string").alias(key_out),
                F.concat(F.lit(f"{col}="), F.col(col).cast("string")).alias("feature_name"),
            ).where(F.col("feature_name").isNotNull())
        )
    if not parts:
        empty_schema = StructType([
            StructField(key_out, StringType(), False),
            StructField("feature_name", StringType(), False),
            StructField("value", DoubleType(), False),
        ])
        return df.sparkSession.createDataFrame([], empty_schema)
    melted = reduce(DataFrame.unionByName, parts)
    return (melted.withColumn("value", F.lit(1.0))
                  .dropDuplicates([key_out, "feature_name"]))


def feature_engineering(engagement_df: DataFrame, metadata_df: DataFrame, *,
                        user_col: str = "user_id", item_col: str = "item_id",
                        value_col: str = "value",
                        event_type_col: str | None = None,
                        days_ago_col: str | None = None,
                        user_feature_cols=(), item_feature_cols=(),
                        item_key_col: str = "item_id"):
    """Map raw engagement + metadata tables into the three long-format tables."""
    base_cols = [
        F.col(user_col).cast("string").alias("user_id"),
        F.col(item_col).cast("string").alias("item_id"),
        F.col(value_col).cast("double").alias("value"),
    ]
    optional = []
    if event_type_col is not None:
        optional.append(F.col(event_type_col).cast("string").alias("event_type"))
    if days_ago_col is not None:
        optional.append(F.col(days_ago_col).cast("double").alias("days_ago"))

    if optional:
        # preserve per-event rows for weighting
        interactions = engagement_df.select(*base_cols, *optional)
    else:
        interactions = (engagement_df.select(*base_cols)
                        .groupBy("user_id", "item_id")
                        .agg(F.sum("value").alias("value")))

    user_features = _one_hot_long(engagement_df, user_col, list(user_feature_cols), "user_id")
    item_features = _one_hot_long(metadata_df, item_key_col, list(item_feature_cols), "item_id")
    return interactions, user_features, item_features
```

- **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/spark/test_databricks.py -k feature_engineering -v`
Expected: PASS (both tests).

- **Step 5: Commit**

```bash
git add kzn_recsys/spark/databricks.py tests/spark/test_databricks.py
git commit -m "feat(spark): raw->long feature engineering for Databricks notebooks"
```

---

### Task 3: Export helpers + end-to-end integration test

**Files:**
- Modify: `kzn_recsys/spark/__init__.py`
- Test: `tests/spark/test_databricks_e2e.py`

**Interfaces:**
- Consumes: `make_synthetic`, `feature_engineering` (Tasks 1–2); `build_and_train`, `load_model`, `random_split`, `temporal_split`, `grid_search` (existing `kzn_recsys.spark`).
- Produces: `from kzn_recsys.spark import make_synthetic, feature_engineering` works. This test is the proof-of-pipeline the four notebooks orchestrate; if it passes, the notebook logic path is sound.

- **Step 1: Write the failing test**

```python
# tests/spark/test_databricks_e2e.py
import pytest

pytest.importorskip("numpy")
from kzn_recsys.spark import (
    make_synthetic, feature_engineering, build_and_train, load_model,
    random_split, grid_search,
)


def test_end_to_end_pipeline(spark, tmp_path):
    # 01 ingest: synthetic already long-format; feature_engineering is exercised
    # separately, so here we consume make_synthetic output directly.
    interactions, users, items = make_synthetic(spark, n_users=80, n_items=30,
                                                n_personas=4, avg_interactions=10, seed=1)

    # 02 train (collect) + a tiny grid search
    res = grid_search(interactions, users, items,
                      {"lambda_": [50.0, 150.0]}, k_folds=2, eval_k=5, seed=1)
    assert "best_params" in res and "lambda_" in res["best_params"]
    best_lambda = res["best_params"]["lambda_"]

    model = build_and_train(interactions, users, items,
                            lambda_=best_lambda, strategy="collect")
    path = str(tmp_path / "model.feas")
    model.save(path)

    # distributed strategy trains too (parity of interface, not asserting equality here)
    model_dist = build_and_train(interactions, users, items,
                                 lambda_=best_lambda, strategy="distributed")
    assert model_dist.num_items == model.num_items

    # 03 evaluate
    train_df, test_df = random_split(interactions, test_ratio=0.2, seed=1)
    metrics = model.evaluate(test_df, train_df, users, k_values=[5, 10])
    ndcg5 = next(m["ndcg"] for m in metrics["metrics"] if m["k"] == 5)
    assert 0.0 <= ndcg5 <= 1.0

    # 04 predict from the reloaded artifact
    loaded = load_model(path)
    warm_uid = interactions.select("user_id").first()["user_id"]
    warm_inter = {r["item_id"]: r["value"]
                  for r in interactions.filter(interactions.user_id == warm_uid).collect()}
    recs = loaded.predict(warm_inter, {}, top_k=5)
    assert len(recs) <= 5
    # cold-start: no interactions, persona feature only
    cold = loaded.predict({}, {"persona=0": 1.0}, top_k=5)
    assert isinstance(cold, list)
```

- **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/spark/test_databricks_e2e.py -v`
Expected: FAIL with `cannot import name 'make_synthetic' from 'kzn_recsys.spark'`.

- **Step 3: Add the exports**

In `kzn_recsys/spark/__init__.py`, add the import and `__all__` entries:

```python
# after the existing metrics import block
from kzn_recsys.spark.databricks import make_synthetic, feature_engineering
```

And add to `__all__` (alongside the existing names):

```python
    "make_synthetic",
    "feature_engineering",
```

- **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/spark/test_databricks_e2e.py -v`
Expected: PASS.

- **Step 5: Run the full spark suite (no regressions)**

Run: `.venv/bin/python -m pytest tests/spark -q`
Expected: PASS (previous suite + the new tests).

- **Step 6: Commit**

```bash
git add kzn_recsys/spark/__init__.py tests/spark/test_databricks_e2e.py
git commit -m "feat(spark): export Databricks helpers; end-to-end pipeline test"
```

---

### Task 4: Asset Bundle scaffold (databricks.yml + Job + README)

**Files:**
- Create: `databricks/databricks.yml`
- Create: `databricks/resources/kzn_recsys_spark_job.yml`
- Create: `databricks/README.md`
- Create: `databricks/.gitignore`

**Interfaces:**
- Consumes: the pure-Python wheel built from `packaging/pure-python`.
- Produces: a deployable bundle named `kzn_recsys_spark` exposing a Job `kzn_recsys_spark_job` with four notebook tasks (`ingest → train → evaluate → predict`) and the variables the notebooks read as widgets. Task 5–8 notebooks live at `databricks/src/NN_*.py`.

- **Step 1: Write `databricks/databricks.yml`**

```yaml
bundle:
  name: kzn_recsys_spark

variables:
  catalog:      { default: main }
  schema:       { default: kzn_recsys }
  volume:       { default: models }
  data_mode:    { default: synthetic }   # synthetic | delta
  n_users:      { default: "500" }
  n_items:      { default: "120" }
  seed:         { default: "42" }
  k:            { default: "10" }
  ndcg_gate:    { default: "0.0" }
  strategy:     { default: collect }     # collect | distributed
  do_tune:      { default: "false" }
  raw_engagement_table: { default: "" }  # used when data_mode=delta
  raw_metadata_table:   { default: "" }

artifacts:
  spark_wheel:
    type: whl
    build: python -m build --wheel
    path: ../packaging/pure-python

include:
  - resources/*.yml

targets:
  dev:
    mode: development
    default: true
  prod:
    mode: production
```

- **Step 2: Write `databricks/resources/kzn_recsys_spark_job.yml`**

```yaml
resources:
  jobs:
    kzn_recsys_spark_job:
      name: kzn_recsys_spark_job
      job_clusters:
        - job_cluster_key: spark_ease
          new_cluster:
            spark_version: 14.3.x-scala2.12
            node_type_id: Standard_DS3_v2
            num_workers: 1
            data_security_mode: SINGLE_USER
      tasks:
        - task_key: ingest
          job_cluster_key: spark_ease
          notebook_task:
            notebook_path: ../src/01_ingest_and_features.py
            base_parameters:
              catalog: ${var.catalog}
              schema: ${var.schema}
              data_mode: ${var.data_mode}
              n_users: ${var.n_users}
              n_items: ${var.n_items}
              seed: ${var.seed}
              raw_engagement_table: ${var.raw_engagement_table}
              raw_metadata_table: ${var.raw_metadata_table}
          libraries:
            - whl: ${artifacts.spark_wheel.files[0].local_path}
        - task_key: train
          depends_on: [{ task_key: ingest }]
          job_cluster_key: spark_ease
          notebook_task:
            notebook_path: ../src/02_train_and_tune.py
            base_parameters:
              catalog: ${var.catalog}
              schema: ${var.schema}
              volume: ${var.volume}
              strategy: ${var.strategy}
              do_tune: ${var.do_tune}
              k: ${var.k}
              seed: ${var.seed}
          libraries:
            - whl: ${artifacts.spark_wheel.files[0].local_path}
        - task_key: evaluate
          depends_on: [{ task_key: train }]
          job_cluster_key: spark_ease
          notebook_task:
            notebook_path: ../src/03_evaluate.py
            base_parameters:
              catalog: ${var.catalog}
              schema: ${var.schema}
              volume: ${var.volume}
              k: ${var.k}
              ndcg_gate: ${var.ndcg_gate}
              seed: ${var.seed}
          libraries:
            - whl: ${artifacts.spark_wheel.files[0].local_path}
        - task_key: predict
          depends_on: [{ task_key: evaluate }]
          job_cluster_key: spark_ease
          notebook_task:
            notebook_path: ../src/04_predict_and_sink.py
            base_parameters:
              catalog: ${var.catalog}
              schema: ${var.schema}
              volume: ${var.volume}
              k: ${var.k}
          libraries:
            - whl: ${artifacts.spark_wheel.files[0].local_path}
```

- **Step 3: Write `databricks/.gitignore`**

```
.databricks/
*.whl
build/
dist/
```

- **Step 4: Write `databricks/README.md`**

````markdown
# kzn_recsys Spark EASE — Databricks Asset Bundle

Runs the pure-Python `kzn_recsys.spark` EASE pipeline end to end as a 4-task
Databricks Job: **ingest → train → evaluate → predict**. No native extension;
the bundle builds and installs the `kzn_recsys_spark` wheel.

## Prerequisites
- `databricks` CLI ≥ 0.218 authenticated to a workspace (`databricks auth login`).
- Unity Catalog: a catalog + schema you can write to, and a Volume for the model artifact.
- Python `build` installed locally (`pip install build`) so the wheel artifact can be built.

## Deploy & run
```bash
cd databricks
databricks bundle validate -t dev
databricks bundle deploy   -t dev
databricks bundle run kzn_recsys_spark_job -t dev
```

## Configuration
Override any variable at deploy/run time, e.g. run on real Delta tables:
```bash
databricks bundle run kzn_recsys_spark_job -t dev \
  --var="data_mode=delta,raw_engagement_table=main.raw.engagement,raw_metadata_table=main.raw.metadata,do_tune=true"
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `data_mode` | `synthetic` | `synthetic` self-contained demo, or `delta` to read real tables |
| `catalog` / `schema` / `volume` | `main` / `kzn_recsys` / `models` | UC destinations |
| `strategy` | `collect` | `collect` (portable) or `distributed` (Spark-side Gram) |
| `do_tune` | `false` | run grid search before final training |
| `k` | `10` | top-K for metrics + predictions |
| `ndcg_gate` | `0.0` | evaluate task fails if NDCG@k below this |

Outputs: Delta tables `kzn_interactions` / `kzn_user_features` /
`kzn_item_features` / `kzn_predictions`, a FEAS artifact in the Volume, and an
MLflow run with params + metrics.
````

- **Step 5: Validate the bundle config (best-effort)**

Run: `cd databricks && databricks bundle validate -t dev`
Expected: `Validation OK!` (schema + variable references resolve). If the
`databricks` CLI is not installed in this environment, skip with a note —
this step is a config lint, not a code test, and does not block the notebooks.

- **Step 6: Commit**

```bash
git add databricks/databricks.yml databricks/resources/kzn_recsys_spark_job.yml databricks/README.md databricks/.gitignore
git commit -m "build(spark): Databricks Asset Bundle scaffold (job + wheel artifact)"
```

---

### Task 5: Notebook `01_ingest_and_features`

**Files:**
- Create: `databricks/src/01_ingest_and_features.py`

**Interfaces:**
- Consumes: widgets `catalog`, `schema`, `data_mode`, `n_users`, `n_items`, `seed`, `raw_engagement_table`, `raw_metadata_table`; `make_synthetic`, `feature_engineering` from `kzn_recsys.spark`.
- Produces: Delta tables `{catalog}.{schema}.kzn_interactions` / `kzn_user_features` / `kzn_item_features`; sets `taskValues` `interactions_table`, `user_features_table`, `item_features_table`.

- **Step 1: Write the notebook (Databricks source format)**

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Ingest & Feature Engineering
# MAGIC Builds the three long-format tables (`interactions`, `user_features`,
# MAGIC `item_features`) from synthetic data or raw Delta tables.

# COMMAND ----------
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "kzn_recsys")
dbutils.widgets.dropdown("data_mode", "synthetic", ["synthetic", "delta"])
dbutils.widgets.text("n_users", "500")
dbutils.widgets.text("n_items", "120")
dbutils.widgets.text("seed", "42")
dbutils.widgets.text("raw_engagement_table", "")
dbutils.widgets.text("raw_metadata_table", "")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
data_mode = dbutils.widgets.get("data_mode")

# COMMAND ----------
from kzn_recsys.spark import make_synthetic, feature_engineering

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

if data_mode == "synthetic":
    interactions, users, items = make_synthetic(
        spark,
        n_users=int(dbutils.widgets.get("n_users")),
        n_items=int(dbutils.widgets.get("n_items")),
        seed=int(dbutils.widgets.get("seed")),
    )
else:
    engagement = spark.table(dbutils.widgets.get("raw_engagement_table"))
    metadata = spark.table(dbutils.widgets.get("raw_metadata_table"))
    # Adjust these column names to your raw schema. Documented contract:
    #   engagement: user_id, item_id, value (+ optional event_type, days_ago) + user feature cols
    #   metadata:   item_id + item feature cols
    interactions, users, items = feature_engineering(
        engagement, metadata,
        user_col="user_id", item_col="item_id", value_col="value",
        user_feature_cols=[c for c in engagement.columns
                           if c.startswith("user_") and c != "user_id"],
        item_feature_cols=[c for c in metadata.columns if c != "item_id"],
    )

# COMMAND ----------
tables = {
    "kzn_interactions": interactions,
    "kzn_user_features": users,
    "kzn_item_features": items,
}
for name, df in tables.items():
    fq = f"{catalog}.{schema}.{name}"
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(fq)
    print(f"wrote {fq}: {df.count()} rows")

# COMMAND ----------
dbutils.jobs.taskValues.set("interactions_table", f"{catalog}.{schema}.kzn_interactions")
dbutils.jobs.taskValues.set("user_features_table", f"{catalog}.{schema}.kzn_user_features")
dbutils.jobs.taskValues.set("item_features_table", f"{catalog}.{schema}.kzn_item_features")
```

- **Step 2: Verify the ingest logic locally (proxy for the notebook)**

The notebook's data logic is `make_synthetic` / `feature_engineering`, already
covered by `tests/spark/test_databricks.py`. Confirm they still pass:

Run: `.venv/bin/python -m pytest tests/spark/test_databricks.py -q`
Expected: PASS.

- **Step 3: Commit**

```bash
git add databricks/src/01_ingest_and_features.py
git commit -m "feat(spark): Databricks notebook 01 — ingest & feature engineering"
```

---

### Task 6: Notebook `02_train_and_tune`

**Files:**
- Create: `databricks/src/02_train_and_tune.py`

**Interfaces:**
- Consumes: taskValues `interactions_table` / `user_features_table` / `item_features_table` (from Task 5); widgets `catalog`, `schema`, `volume`, `strategy`, `do_tune`, `k`, `seed`; `build_and_train`, `grid_search` from `kzn_recsys.spark`; MLflow.
- Produces: FEAS artifact at `/Volumes/{catalog}/{schema}/{volume}/model.feas`; MLflow run with params + artifact; sets taskValues `model_path`, `mlflow_run_id`, `best_lambda`.

- **Step 1: Write the notebook**

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Train & Tune
# MAGIC Optional grid search, then final EASE training on the full dataset.
# MAGIC Strategy `collect` (portable, default) or `distributed` (Spark-side Gram).

# COMMAND ----------
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "kzn_recsys")
dbutils.widgets.text("volume", "models")
dbutils.widgets.dropdown("strategy", "collect", ["collect", "distributed"])
dbutils.widgets.dropdown("do_tune", "false", ["true", "false"])
dbutils.widgets.text("k", "10")
dbutils.widgets.text("seed", "42")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")
strategy = dbutils.widgets.get("strategy")
do_tune = dbutils.widgets.get("do_tune") == "true"
k = int(dbutils.widgets.get("k"))
seed = int(dbutils.widgets.get("seed"))

def _tv(key, default):
    try:
        return dbutils.jobs.taskValues.get(taskKey="ingest", key=key, debugValue=default)
    except Exception:
        return default

interactions = spark.table(_tv("interactions_table", f"{catalog}.{schema}.kzn_interactions"))
users = spark.table(_tv("user_features_table", f"{catalog}.{schema}.kzn_user_features"))
items = spark.table(_tv("item_features_table", f"{catalog}.{schema}.kzn_item_features"))

# COMMAND ----------
# MAGIC %md
# MAGIC ### Strategy note
# MAGIC `collect` pulls the sparse matrices to the driver (fast, portable, right
# MAGIC for moderate catalogs). `distributed` accumulates the Gram matrix with
# MAGIC Spark joins — use it when the item+feature dimension is too large to
# MAGIC collect but still yields a driver-solvable Gram.

# COMMAND ----------
import mlflow
from kzn_recsys.spark import build_and_train, grid_search

best_lambda = 150.0
mlflow.set_experiment(f"/Shared/kzn_recsys_spark")
with mlflow.start_run() as run:
    if do_tune:
        res = grid_search(interactions, users, items,
                          {"lambda_": [50.0, 100.0, 150.0, 300.0]},
                          k_folds=3, eval_k=k, seed=seed)
        best_lambda = float(res["best_params"]["lambda_"])
        mlflow.log_metric("cv_best_ndcg", float(res["best_score"]))

    mlflow.log_params({"strategy": strategy, "lambda_": best_lambda,
                       "do_tune": do_tune, "eval_k": k})

    model = build_and_train(interactions, users, items,
                            lambda_=best_lambda, strategy=strategy)

    volume_dir = f"/Volumes/{catalog}/{schema}/{volume}"
    dbutils.fs.mkdirs(volume_dir.replace("/Volumes", "dbfs:/Volumes"))
    model_path = f"{volume_dir}/model.feas"
    model.save(model_path)
    mlflow.log_artifact(model_path)

    run_id = run.info.run_id

print(f"trained lambda={best_lambda}, saved {model_path}, run={run_id}")

# COMMAND ----------
dbutils.jobs.taskValues.set("model_path", model_path)
dbutils.jobs.taskValues.set("mlflow_run_id", run_id)
dbutils.jobs.taskValues.set("best_lambda", best_lambda)
```

- **Step 2: Verify the train/tune logic locally (proxy)**

`build_and_train` (both strategies) + `grid_search` + `save` are covered by
`tests/spark/test_databricks_e2e.py`. Confirm:

Run: `.venv/bin/python -m pytest tests/spark/test_databricks_e2e.py -q`
Expected: PASS.

- **Step 3: Commit**

```bash
git add databricks/src/02_train_and_tune.py
git commit -m "feat(spark): Databricks notebook 02 — train & tune with MLflow"
```

---

### Task 7: Notebook `03_evaluate`

**Files:**
- Create: `databricks/src/03_evaluate.py`

**Interfaces:**
- Consumes: taskValues `best_lambda`, `mlflow_run_id`, table names; widgets `catalog`, `schema`, `k`, `ndcg_gate`, `seed`; `build_and_train`, `temporal_split`, `random_split` from `kzn_recsys.spark`; MLflow.
- Produces: metrics logged to the Task-6 MLflow run; raises `AssertionError` (fails the task) when NDCG@k < `ndcg_gate`.

- **Step 1: Write the notebook**

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Evaluate
# MAGIC Holds out a temporal (or random) test split, retrains on the train fold
# MAGIC with the chosen params, and logs ranking metrics. Enforces a quality gate.

# COMMAND ----------
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "kzn_recsys")
dbutils.widgets.text("k", "10")
dbutils.widgets.text("ndcg_gate", "0.0")
dbutils.widgets.text("seed", "42")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
k = int(dbutils.widgets.get("k"))
ndcg_gate = float(dbutils.widgets.get("ndcg_gate"))
seed = int(dbutils.widgets.get("seed"))

def _tv(task, key, default):
    try:
        return dbutils.jobs.taskValues.get(taskKey=task, key=key, debugValue=default)
    except Exception:
        return default

interactions = spark.table(f"{catalog}.{schema}.kzn_interactions")
users = spark.table(f"{catalog}.{schema}.kzn_user_features")
items = spark.table(f"{catalog}.{schema}.kzn_item_features")
best_lambda = float(_tv("train", "best_lambda", 150.0))
run_id = _tv("train", "mlflow_run_id", None)

# COMMAND ----------
import mlflow
from kzn_recsys.spark import build_and_train, random_split, temporal_split

# temporal split needs days_ago; fall back to random if absent.
if "days_ago" in interactions.columns:
    train_df, test_df = temporal_split(interactions, days_ago_cutoff=7.0)
else:
    train_df, test_df = random_split(interactions, test_ratio=0.2, seed=seed)

model = build_and_train(train_df, users, items, lambda_=best_lambda, strategy="collect")
report = model.evaluate(test_df, train_df, users, k_values=[k])
row = next(m for m in report["metrics"] if m["k"] == k)
print(report)

# COMMAND ----------
metrics = {f"test_{name}@{k}": float(row[name])
           for name in ("precision", "recall", "ndcg", "map", "hit_rate")}
metrics["coverage"] = float(report["coverage"])
if run_id:
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
else:
    with mlflow.start_run():
        mlflow.log_metrics(metrics)

assert row["ndcg"] >= ndcg_gate, (
    f"NDCG@{k}={row['ndcg']:.4f} below gate {ndcg_gate}; failing task."
)
print(f"quality gate passed: NDCG@{k}={row['ndcg']:.4f} >= {ndcg_gate}")
```

- **Step 2: Verify the evaluate logic locally (proxy)**

`build_and_train` + split + `evaluate` are covered by
`tests/spark/test_databricks_e2e.py`. Confirm:

Run: `.venv/bin/python -m pytest tests/spark/test_databricks_e2e.py -q`
Expected: PASS.

- **Step 3: Commit**

```bash
git add databricks/src/03_evaluate.py
git commit -m "feat(spark): Databricks notebook 03 — evaluate with MLflow + quality gate"
```

---

### Task 8: Notebook `04_predict_and_sink`

**Files:**
- Create: `databricks/src/04_predict_and_sink.py`

**Interfaces:**
- Consumes: taskValues `model_path` (from Task 6); widgets `catalog`, `schema`, `volume`, `k`; `load_model` from `kzn_recsys.spark`.
- Produces: Delta table `{catalog}.{schema}.kzn_predictions` (`user_id`, `rank`, `item_id`, `score`, `run_id`).

- **Step 1: Write the notebook**

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Predict & Sink
# MAGIC Loads the FEAS artifact, generates top-K recommendations for warm and
# MAGIC cold-start users, and writes them to a Delta table.

# COMMAND ----------
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "kzn_recsys")
dbutils.widgets.text("volume", "models")
dbutils.widgets.text("k", "10")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")
k = int(dbutils.widgets.get("k"))

def _tv(task, key, default):
    try:
        return dbutils.jobs.taskValues.get(taskKey=task, key=key, debugValue=default)
    except Exception:
        return default

model_path = _tv("train", "model_path", f"/Volumes/{catalog}/{schema}/{volume}/model.feas")
run_id = _tv("train", "mlflow_run_id", "")

# COMMAND ----------
from pyspark.sql import Row
from kzn_recsys.spark import load_model

model = load_model(model_path)
interactions = spark.table(f"{catalog}.{schema}.kzn_interactions")
users = spark.table(f"{catalog}.{schema}.kzn_user_features")

# Collect each user's history + features on the driver (EASE predict is per-user).
hist = {}
for r in interactions.select("user_id", "item_id", "value").collect():
    hist.setdefault(r["user_id"], {})[r["item_id"]] = float(r["value"])
feats = {}
for r in users.select("user_id", "feature_name", "value").collect():
    feats.setdefault(r["user_id"], {})[r["feature_name"]] = float(r["value"])

rows = []
for uid, inter in hist.items():
    for rank, (item_id, score) in enumerate(model.predict(inter, feats.get(uid, {}), top_k=k), 1):
        rows.append(Row(user_id=uid, rank=rank, item_id=item_id,
                        score=float(score), run_id=run_id))

# COMMAND ----------
preds = spark.createDataFrame(rows)
sink = f"{catalog}.{schema}.kzn_predictions"
preds.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(sink)
print(f"wrote {sink}: {preds.count()} recommendations")
display(preds.orderBy("user_id", "rank").limit(20))
```

- **Step 2: Verify the predict logic locally (proxy)**

`load_model` + `predict` (warm + cold-start) are covered by
`tests/spark/test_databricks_e2e.py`. Confirm:

Run: `.venv/bin/python -m pytest tests/spark/test_databricks_e2e.py -q`
Expected: PASS.

- **Step 3: Full suite + bundle validate (final)**

Run: `.venv/bin/python -m pytest tests/spark -q`
Expected: PASS.
Run (best-effort): `cd databricks && databricks bundle validate -t dev`
Expected: `Validation OK!` (skip with a note if the CLI is unavailable).

- **Step 4: Commit**

```bash
git add databricks/src/04_predict_and_sink.py
git commit -m "feat(spark): Databricks notebook 04 — predict & Delta sink"
```

---

## Self-Review

**Spec coverage:**
- Parameterized data (synthetic default) → Task 1 (`make_synthetic`), Task 2 (`feature_engineering`), Task 5 (`data_mode` widget). ✓
- DAB packaging → Task 4 (`databricks.yml`, artifact build). ✓
- 4-task Job DAG → Task 4 (`resources/*.yml`), Tasks 5–8 notebooks. ✓
- MLflow logging → Task 6 (params + artifact), Task 7 (metrics). ✓
- collect + distributed demo → Task 6 (`strategy` widget + note), Task 3 (both trained in the e2e test). ✓
- Quality gate → Task 7. ✓
- Delta predictions sink → Task 8. ✓
- Tested core, thin notebooks → Tasks 1–3 tests; Tasks 5–8 verified via the e2e test. ✓
- New module ships in the wheel → Task 3 export (spark subpackage already packaged by `packaging/pure-python`). ✓

**Placeholder scan:** No TBD/TODO; the only intentional adjustment point (raw column names in `delta` mode, Task 5) is documented inline with the exact contract. ✓

**Type consistency:** `make_synthetic` / `feature_engineering` signatures identical across Tasks 1–3 and their notebook call sites (Tasks 5–6); `build_and_train(..., strategy=)`, `evaluate(test, train, users, k_values=[...])`, `grid_search(..., param_grid, k_folds, eval_k, seed)`, `load_model(path)`, and `predict(interactions, features, top_k)` match the verified library signatures. ✓
