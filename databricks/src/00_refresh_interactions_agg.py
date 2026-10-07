# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Refresh the series-grain interactions table (issue #102)
# MAGIC Runs the SQL in `../sql/` with the widget values substituted for the
# MAGIC `${...}` parameters:
# MAGIC
# MAGIC - `mode=backfill` — `create_fease_interactions_agg.sql`: create the
# MAGIC   table and rebuild it from the full gold history (run once; replaces
# MAGIC   the table).
# MAGIC - `mode=merge` — `merge_fease_interactions_agg.sql`: recompute the last
# MAGIC   `lookback_days` complete days and MERGE them in (idempotent; schedule
# MAGIC   daily after the gold table refreshes).
# MAGIC - `mode=validate` — `validate_fease_interactions_agg.sql`: print the
# MAGIC   validation queries' results to record on the issue.

# COMMAND ----------
import os
import re
from string import Template

dbutils.widgets.dropdown("mode", "merge", ["backfill", "merge", "validate"])
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

mode = dbutils.widgets.get("mode")
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
if "'" in params["subsidiary"]:
    raise ValueError("subsidiary must not contain quotes")

# COMMAND ----------
# The bundle syncs the whole `databricks/` folder, so the SQL lives next to
# this notebook's parent directory in the workspace.
_nb_path = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    .notebookPath().get()
)
_sql_dir = os.path.join("/Workspace" + os.path.dirname(os.path.dirname(_nb_path)), "sql")
_sql_file = {
    "backfill": "create_fease_interactions_agg.sql",
    "merge": "merge_fease_interactions_agg.sql",
    "validate": "validate_fease_interactions_agg.sql",
}[mode]

with open(os.path.join(_sql_dir, _sql_file)) as fh:
    sql_text = Template(fh.read()).substitute(params)


def _statements(text):
    """Split on statement-terminating semicolons, dropping comment-only chunks."""
    for chunk in text.split(";"):
        body = "\n".join(l for l in chunk.splitlines() if not l.strip().startswith("--")).strip()
        if body:
            yield body


# COMMAND ----------
print(f"mode={mode} target={params['catalog']}.{params['schema']}.{params['table']} "
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
