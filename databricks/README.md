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

## End-to-end evaluation on production data

`kzn_recsys_eval_job` (`resources/kzn_recsys_eval_job.yml`) evaluates EASE
on the data `kzn_recsys/fease_train.py` trains on, in two tasks that reuse
the notebooks above:

1. **ingest** (`data_mode=agg`) reads `fease_interactions_agg`, derives
   `days_ago` from a pinned `as_of` date, and builds the CR user features
   (latest viewership row: plan, country, region, status, tenure; optional
   interaction-attribute summaries) and item features (content metadata
   keyed by series) with `kzn_recsys.spark.cr_features`, the same code
   `fease_train.py` uses. It writes `kzn_interactions` (pairs, decayed) and
   `kzn_interactions_daily` (the daily grain) into `eval_schema`.
2. **evaluate** holds out the last `days_ago_cutoff` days. The split cuts
   the daily grain: train is every day before the cutoff (decay measured
   from the cutoff), test is the series each user *started* after it. A
   series watched on both sides stays in train and is not a test item:
   evaluation never recommends a user's train items, so it could not be
   hit. The hyperparameters in `param_grid` are grid-searched with user
   k-fold CV on the train fold only, then the model is retrained on the
   whole train fold and scored next to a most-popular baseline on the same
   users. Metrics (`test_*@k`, `baseline_*@k`, `lift_*@k`, `cv_best_ndcg`,
   split sizes) and the eval params go to the `eval_mlflow_experiment` MLflow
   experiment; the notebook's exit value carries the same summary as JSON.

```bash
cd databricks
databricks bundle deploy -t dev
# Smoke test on synthetic data (random split: synthetic data has no days_ago)
databricks bundle run kzn_recsys_eval_job -t dev --var="eval_data_mode=synthetic"
# Real data: a pinned, reproducible backtest with item features
databricks bundle run kzn_recsys_eval_job -t dev \
  --var="metadata_table=<catalog>.<schema>.<content_metadata>,as_of=2026-10-01,activity_window_days=365"
```

Refresh the agg table first (`fease_interactions_agg_refresh`) if it is
stale. The job writes only to `eval_schema` and has no predict task, so it
never touches the demo job's tables or predictions.

| Variable | Default | Purpose |
|----------|---------|---------|
| `eval_schema` | `kzn_recsys_eval` | schema for the job's tables (in `catalog`) |
| `eval_data_mode` | `agg` | `agg`, or `synthetic` for a smoke test |
| `metadata_table` | `""` | media-grain content metadata (`catalog_show_id`, `media_type`, `media_genres`, …); empty = no item features |
| `as_of` | `""` | date `days_ago` is measured from; pin it to make a run reproducible (empty = today) |
| `activity_window_days` | `""` | keep only the last N days of history (empty = all) |
| `decay_rate` | `0.0` | per-day temporal decay applied when collapsing days to pairs |
| `use_attr_features` | `false` | add the interaction-attribute user features (`attrs_table`) |
| `split` / `days_ago_cutoff` / `holdout_k` | `temporal` / `7.0` / `1` | hold-out (shared with the demo job) |
| `eval_do_tune` / `param_grid` / `tune_k_folds` | `true` / `{"lambda_": [50, 100, 150, 300]}` / `3` | grid search on the train fold; `param_grid` also takes `alpha`, `beta`, `meta_weight` |
| `tune_max_users` / `max_eval_users` | `20000` / `50000` | seeded user samples for tuning / scoring (`0` = all); both are collected to the driver |
| `eval_strategy` | `distributed` | training strategy for the evaluated model |
| `availability_table` / `user_territory_feature` | `""` | availability-aware ranking (shared with the demo job) |
| `ndcg_gate` / `k` | `0.0` / `10` | quality gate and cutoff (shared) |

User features describe each user's latest state, including rows after the
cutoff, and new series reach the catalog through their metadata before their
first train interaction; pass an `availability_table` to keep not-yet-released
series out of the ranking.

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

## Interaction and series attribute tables

Two sibling tables of `fease_interactions_agg` carry the per-view and
per-series attributes a factored-input sequence model can embed. The agg
table, its three scripts and its column contract are not touched; the
attributes join back on the same key.

- `fease_interaction_attrs`: one row per profile × series × day, for exactly
  the agg table's keys (same filters, same `min_watch_seconds`, today
  excluded). Columns: `seconds_watched`, `episodes_watched` (distinct
  episode ids), `completed_episodes` (views with seconds / duration ≥
  `completed_ratio`), `max_depth_ratio` (capped at 1), `platform_top`
  (`tv` / `mobile` / `web` / `other`, ties broken in that order),
  `dub_views` / `sub_views` (audio locale ≠ / = `ja-JP`; null audio counts in
  neither), `first_view_ts`. Refreshed with the same recompute-the-window
  MERGE as the agg table.
- `fease_item_attrs`: one row per series. `premiere_date` is the catalog's
  earliest availability (`premiere_source = 'catalog'`), else the earliest
  `view_date` in the agg table (`'first_view'`); plus `episode_count`,
  `avg_episode_seconds`, `first_air_year`. Rebuilt in full on every refresh.

Nothing thresholded or time-relative is stored. Depth, binge and dub-share
buckets and the discovery lag (`view_date − premiere_date`, log₂-bucketed
like the BERT4Rec positions) are derived at read time by
`kzn_recsys.spark.interaction_attrs`. The platform mapping and the
thresholds live in `kzn_recsys/cr_config.py`; the refresh notebook renders
the platform mapping into SQL from there, so the attribute targets need the
`kzn_recsys_spark` wheel on the cluster.

The same notebook drives both tables through a `target` widget:

```bash
# one-time backfills, after the agg table's
databricks bundle run fease_interactions_agg_refresh -t dev --params target=attrs,mode=backfill
databricks bundle run fease_interactions_agg_refresh -t dev --params target=item_attrs,mode=backfill
# validation (key-set parity with the agg table, MERGE idempotency,
# depth / platform distributions, raw → normalized platform values)
databricks bundle run fease_interactions_agg_refresh -t dev --params target=attrs,mode=validate
```

The scheduled job runs `attrs_merge` and `item_attrs_rebuild` after the agg
`merge` task.

| Variable | Default | Purpose |
|----------|---------|---------|
| `attrs_table` / `item_attrs_table` | `fease_interaction_attrs` / `fease_item_attrs` | target tables (same catalog / schema as `agg_table`) |
| `attrs_media_id_col` | `catalog_media_id` | per-episode id on the gold table |
| `attrs_media_duration_col` | `catalog_media_duration` | episode length; the notebook's `duration_to_seconds` widget scales it |
| `attrs_platform_col` | `view_platform_name` | raw platform value |
| `attrs_audio_col` | `view_audio_locale` | audio locale of the view |
| `attrs_completed_ratio` | `0.9` | completed-episode threshold |
| `catalog_table` | `cr_prod.gold_db.ds_catalog` | series / media catalog for `fease_item_attrs` |

The column defaults are not yet confirmed against the live gold schema. A
blank duration, audio or episode-id column turns the dependent attributes
into NULLs (depth `unknown`, no audio counts) rather than failing.

In `kzn_recsys/fease_train.py`, `USE_INTERACTION_ATTR_FEATURES = True` adds
per-user summaries of these attributes to the FEASE user features
(`platform_top_<class>` shares, `dub_pref_<sub|mixed|dub>`,
`depth_completed_share`, `binge_heavy_share`). It defaults to `False`, which
leaves the training inputs byte-identical.
