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
