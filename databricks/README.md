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

## Series-grain interactions table (issue #102)

`fease_interactions_agg` holds one row per profile × series × day
(`user_id`, `item_id`, `view_date`, `value = sum(ln(seconds + 1))`,
`num_views`), built from the gold viewership table. `days_ago` is derived at
read time, so temporal decay stays tunable and the table never bakes in a
reference date. The SQL in `sql/` is parametrized with `${...}` and run by
`src/00_refresh_interactions_agg.py`:

```bash
# one-time backfill (replaces the table), then validate and record on #102
databricks bundle run fease_interactions_agg_refresh -t dev --params mode=backfill
databricks bundle run fease_interactions_agg_refresh -t dev --params mode=validate
# daily MERGE of the last `agg_lookback_days` complete days (the scheduled default;
# the schedule ships PAUSED — unpause it after the backfill)
databricks bundle run fease_interactions_agg_refresh -t dev
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `agg_catalog` / `agg_schema` / `agg_table` | `dsml_recs` / `dev` / `fease_interactions_agg` | target table |
| `viewership_table` | `cr_prod.gold_db.ds_viewership` | gold source |
| `agg_lookback_days` | `3` | days the MERGE recomputes; must cover the source's late-arrival window |

The MERGE recomputes the window rather than adding deltas, updates changed
rows, inserts new ones and deletes rows that vanished from the source in the
window, so re-running it is a no-op. Consumers: `kzn_recsys/fease_train.py`
(`INTERACTIONS_SOURCE="agg"`) and the helpers in
`kzn_recsys.spark.interactions_agg`.
