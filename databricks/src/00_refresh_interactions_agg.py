# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Refresh the series-grain interactions table (issue #102) and its attribute tables
# MAGIC Runs the SQL in `../sql/` with the widget values substituted for the
# MAGIC `${...}` parameters. `target` picks the table:
# MAGIC
# MAGIC - `target=interactions` (default) — `fease_interactions_agg`:
# MAGIC
# MAGIC - `mode=backfill` — `create_fease_interactions_agg.sql`: create the
# MAGIC   table and rebuild it from the full gold history (run once; replaces
# MAGIC   the table).
# MAGIC - `mode=merge` — `merge_fease_interactions_agg.sql`: recompute the last
# MAGIC   `lookback_days` complete days and MERGE them in (idempotent; schedule
# MAGIC   daily after the gold table refreshes).
# MAGIC - `mode=validate` — `validate_fease_interactions_agg.sql`: print the
# MAGIC   validation queries' results to record on the issue.
# MAGIC
# MAGIC - `target=attrs` — `fease_interaction_attrs`, same key set as the agg
# MAGIC   table: `backfill` / `merge` / `validate` run
# MAGIC   `create_` / `merge_` / `validate_fease_interaction_attrs.sql`.
# MAGIC - `target=item_attrs` — `fease_item_attrs`, one row per series:
# MAGIC   `backfill` and `merge` both rebuild it in full
# MAGIC   (`create_fease_item_attrs.sql`); `validate` runs the attrs validation.
# MAGIC
# MAGIC The attribute targets import `kzn_recsys.cr_config` (platform mapping)
# MAGIC and need the `kzn_recsys_spark` wheel on the cluster. A blank
# MAGIC `media_duration_col` / `audio_col` / `media_id_col` (or catalog column)
# MAGIC turns the dependent attributes into NULLs instead of failing.

# COMMAND ----------
import os
import re
from string import Template

dbutils.widgets.dropdown("mode", "merge", ["backfill", "merge", "validate"])
dbutils.widgets.dropdown("target", "interactions", ["interactions", "attrs", "item_attrs"])
dbutils.widgets.text("catalog", "dsml_recs")
dbutils.widgets.text("schema", "dev")
dbutils.widgets.text("table", "fease_interactions_agg")
dbutils.widgets.text("source_table", "cr_prod.gold_db.ds_viewership")
dbutils.widgets.text("user_col", "view_profile_id")
dbutils.widgets.text("item_col", "catalog_show_id")
dbutils.widgets.text("date_col", "view_date")
dbutils.widgets.text("seconds_col", "view_seconds_watched")
dbutils.widgets.text("subsidiary_col", "view_subsidiary")
dbutils.widgets.text("subsidiary", "crunchyroll")
dbutils.widgets.text("min_watch_seconds", "30")
dbutils.widgets.text("lookback_days", "3")
# Attribute tables. Column names below are unverified against the live gold
# schema; see kzn_recsys/cr_config.py.
dbutils.widgets.text("attrs_table", "fease_interaction_attrs")
dbutils.widgets.text("item_attrs_table", "fease_item_attrs")
dbutils.widgets.text("ts_col", "view_ts")
dbutils.widgets.text("media_id_col", "catalog_media_id")
dbutils.widgets.text("media_duration_col", "catalog_media_duration")
dbutils.widgets.text("duration_to_seconds", "1.0")
dbutils.widgets.text("platform_col", "view_platform_name")
dbutils.widgets.text("audio_col", "view_audio_locale")
dbutils.widgets.text("sub_audio_locale", "ja-JP")
dbutils.widgets.text("completed_ratio", "0.9")
dbutils.widgets.text("catalog_table", "cr_prod.gold_db.ds_catalog")
dbutils.widgets.text("catalog_series_col", "catalog_show_id")
dbutils.widgets.text("catalog_media_id_col", "catalog_media_id")
dbutils.widgets.text("catalog_duration_col", "catalog_media_duration")
dbutils.widgets.text("catalog_available_col", "catalog_available_date")
dbutils.widgets.text("catalog_air_date_col", "catalog_original_air_date")

mode = dbutils.widgets.get("mode")
target = dbutils.widgets.get("target")
params = {
    name: dbutils.widgets.get(name)
    for name in (
        "catalog", "schema", "table", "source_table", "user_col", "item_col",
        "date_col", "seconds_col", "subsidiary_col", "subsidiary",
        "min_watch_seconds", "lookback_days",
    )
}
# Numeric parameters are spliced into SQL unquoted: validate them.
float(params["min_watch_seconds"])
if int(params["lookback_days"]) < 1:
    raise ValueError("lookback_days must be >= 1")
# Identifiers are spliced inside backticks: reject anything that could escape them.
for key in ("catalog", "schema", "table", "user_col", "item_col", "date_col",
            "seconds_col", "subsidiary_col"):
    if not re.fullmatch(r"[A-Za-z0-9_]+", params[key]):
        raise ValueError(f"{key}={params[key]!r} is not a plain identifier")
if not re.fullmatch(r"[A-Za-z0-9_.]+", params["source_table"]):
    raise ValueError(f"source_table={params['source_table']!r} is not a table name")
# `subsidiary` is spliced inside a single-quoted SQL literal: allow only
# characters that cannot start an escape or end the literal (no quotes, no
# backslashes, nothing outside [A-Za-z0-9_ -]).
if not re.fullmatch(r"[A-Za-z0-9_ -]+", params["subsidiary"]):
    raise ValueError(f"subsidiary={params['subsidiary']!r} must match [A-Za-z0-9_ -]+")


if target != "interactions":
    attr_names = (
        "attrs_table", "item_attrs_table", "ts_col", "media_id_col", "media_duration_col",
        "duration_to_seconds", "platform_col", "audio_col", "sub_audio_locale",
        "completed_ratio", "catalog_table", "catalog_series_col", "catalog_media_id_col",
        "catalog_duration_col", "catalog_available_col", "catalog_air_date_col",
    )
    params.update({name: dbutils.widgets.get(name).strip() for name in attr_names})
    ratio = float(params["completed_ratio"])
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"completed_ratio={ratio} must be in (0, 1]")
    scale = float(params["duration_to_seconds"])
    if not scale > 0.0:
        raise ValueError(f"duration_to_seconds={scale} must be > 0")
    # Required identifiers.
    for key in ("attrs_table", "item_attrs_table", "ts_col", "platform_col",
                "catalog_series_col", "catalog_media_id_col"):
        if not re.fullmatch(r"[A-Za-z0-9_]+", params[key]):
            raise ValueError(f"{key}={params[key]!r} is not a plain identifier")
    # Optional identifiers: blank means "absent in the source".
    for key in ("media_id_col", "media_duration_col", "audio_col", "catalog_duration_col",
                "catalog_available_col", "catalog_air_date_col"):
        if params[key] and not re.fullmatch(r"[A-Za-z0-9_]+", params[key]):
            raise ValueError(f"{key}={params[key]!r} is not a plain identifier")
    if not re.fullmatch(r"[A-Za-z0-9_.]+", params["catalog_table"]):
        raise ValueError(f"catalog_table={params['catalog_table']!r} is not a table name")
    # Spliced inside a single-quoted SQL literal.
    if not re.fullmatch(r"[A-Za-z0-9_-]+", params["sub_audio_locale"]):
        raise ValueError(f"sub_audio_locale={params['sub_audio_locale']!r} must match [A-Za-z0-9_-]+")

    from kzn_recsys import cr_config as _cfg
    from kzn_recsys.spark.interaction_attrs import platform_case_sql

    def _col_or_null(col, sql_type, wrap="{}"):
        return wrap.format(f"CAST(`{col}` AS {sql_type})") if col else f"CAST(NULL AS {sql_type})"

    params["completed_ratio"] = repr(ratio)
    params["media_id_expr"] = _col_or_null(params["media_id_col"], "STRING")
    params["duration_expr"] = _col_or_null(
        params["media_duration_col"], "DOUBLE", "({} * " + repr(scale) + ")"
    )
    params["audio_expr"] = _col_or_null(params["audio_col"], "STRING")
    params["platform_expr"] = platform_case_sql(f"`{params['platform_col']}`")
    params["platform_rank_expr"] = (
        "CASE `platform` "
        + " ".join(f"WHEN '{c}' THEN {i}" for i, c in enumerate(_cfg.PLATFORM_TIE_ORDER))
        + f" ELSE {len(_cfg.PLATFORM_TIE_ORDER)} END"
    )
    params["catalog_duration_expr"] = _col_or_null(
        params["catalog_duration_col"], "DOUBLE", "({} * " + repr(scale) + ")"
    )
    params["catalog_available_expr"] = _col_or_null(params["catalog_available_col"], "DATE")
    params["catalog_air_year_expr"] = (
        f"YEAR(`{params['catalog_air_date_col']}`)" if params["catalog_air_date_col"]
        else "CAST(NULL AS INT)"
    )

# COMMAND ----------
# The bundle syncs the whole `databricks/` folder, so the SQL lives next to
# this notebook's parent directory in the workspace.
_nb_path = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    .notebookPath().get()
)
_sql_dir = os.path.join("/Workspace" + os.path.dirname(os.path.dirname(_nb_path)), "sql")
_sql_file = {
    ("interactions", "backfill"): "create_fease_interactions_agg.sql",
    ("interactions", "merge"): "merge_fease_interactions_agg.sql",
    ("interactions", "validate"): "validate_fease_interactions_agg.sql",
    ("attrs", "backfill"): "create_fease_interaction_attrs.sql",
    ("attrs", "merge"): "merge_fease_interaction_attrs.sql",
    ("attrs", "validate"): "validate_fease_interaction_attrs.sql",
    # Small table: every refresh is a full rebuild.
    ("item_attrs", "backfill"): "create_fease_item_attrs.sql",
    ("item_attrs", "merge"): "create_fease_item_attrs.sql",
    ("item_attrs", "validate"): "validate_fease_interaction_attrs.sql",
}[(target, mode)]

with open(os.path.join(_sql_dir, _sql_file)) as fh:
    sql_text = Template(fh.read()).substitute(params)


def _statements(text):
    """Split on statement-terminating semicolons, dropping comment-only chunks."""
    for chunk in text.split(";"):
        body = "\n".join(l for l in chunk.splitlines() if not l.strip().startswith("--")).strip()
        if body:
            yield body


# COMMAND ----------
_table_key = {"interactions": "table", "attrs": "attrs_table", "item_attrs": "item_attrs_table"}[target]
print(f"target={target} mode={mode} "
      f"table={params['catalog']}.{params['schema']}.{params[_table_key]} "
      f"source={params['source_table']} lookback_days={params['lookback_days']}")
for i, stmt in enumerate(_statements(sql_text), 1):
    print(f"\n--- statement {i} ---\n{stmt[:400]}{'...' if len(stmt) > 400 else ''}")
    result = spark.sql(stmt)
    if mode == "validate":
        result.show(50, truncate=False)
    else:
        rows = result.collect()
        if rows:
            print(rows)
print("done")
