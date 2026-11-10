# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Ingest & Feature Engineering
# MAGIC Builds the three long-format tables (`interactions`, `user_features`,
# MAGIC `item_features`) from synthetic data, raw Delta tables, or (`agg`) the
# MAGIC series-grain `fease_interactions_agg` table (#102) with the CR user and
# MAGIC item features `kzn_recsys/fease_train.py` trains on.
# MAGIC
# MAGIC `agg` also writes the daily grain to `kzn_interactions_daily`, which the
# MAGIC evaluate task splits on for a temporal hold-out, and the state-only user
# MAGIC features (no interaction-attribute summaries) to `kzn_user_base_features`,
# MAGIC from which the evaluate task rebuilds those summaries per train fold. The
# MAGIC other modes drop both tables so stale ones never shadow their inputs.

# COMMAND ----------
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "kzn_recsys")
dbutils.widgets.dropdown("data_mode", "synthetic", ["synthetic", "delta", "agg"])
dbutils.widgets.text("n_users", "500")
dbutils.widgets.text("n_items", "120")
dbutils.widgets.text("seed", "42")
dbutils.widgets.text("raw_engagement_table", "")
dbutils.widgets.text("raw_metadata_table", "")
# --- data_mode=agg ---
dbutils.widgets.text("agg_table", "dsml_recs.dev.fease_interactions_agg")
dbutils.widgets.text("viewership_table", "cr_prod.gold_db.ds_viewership")  # user features
dbutils.widgets.text("metadata_table", "")  # media-grain content metadata; empty = no item features
dbutils.widgets.text("activity_window_days", "")  # empty = all history
dbutils.widgets.text("as_of", "")  # ISO date days_ago is measured from; empty = today
dbutils.widgets.text("decay_rate", "0.0")  # per-day decay applied when collapsing to pairs
dbutils.widgets.dropdown("use_attr_features", "false", ["true", "false"])
dbutils.widgets.text("attrs_table", "dsml_recs.dev.fease_interaction_attrs")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
data_mode = dbutils.widgets.get("data_mode")

# COMMAND ----------
import re
from kzn_recsys.spark import make_synthetic, feature_engineering

# catalog/schema are interpolated into SQL below: accept plain identifiers only.
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
for name, value in (("catalog", catalog), ("schema", schema)):
    if not _IDENT.match(value):
        raise ValueError(f"{name}={value!r} is not a plain SQL identifier")

spark.sql(f"CREATE CATALOG IF NOT EXISTS `{catalog}`")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")
daily_fq = f"{catalog}.{schema}.kzn_interactions_daily"
base_users_fq = f"{catalog}.{schema}.kzn_user_base_features"

if data_mode == "synthetic":
    n_users = int(dbutils.widgets.get("n_users"))
    interactions, users, items = make_synthetic(
        spark,
        n_users=n_users,
        n_items=int(dbutils.widgets.get("n_items")),
        # a few feature-only users so the predict task exercises cold-start
        n_cold_users=max(1, n_users // 100),
        seed=int(dbutils.widgets.get("seed")),
    )
elif data_mode == "agg":
    from kzn_recsys.spark import (
        cr_item_features, cr_user_features, daily_to_pairs, filter_activity_window,
        with_days_ago,
    )
    from kzn_recsys.spark.interaction_attrs import add_interaction_attr_user_features

    # Pin the reference date once: current_date() is evaluated per query, and
    # a run crossing midnight would otherwise mix two references.
    as_of = (dbutils.widgets.get("as_of").strip()
             or str(spark.sql("SELECT current_date() AS d").first()["d"]))
    window = dbutils.widgets.get("activity_window_days").strip()
    decay_rate = float(dbutils.widgets.get("decay_rate"))
    use_attrs = dbutils.widgets.get("use_attr_features") == "true"
    metadata_table = dbutils.widgets.get("metadata_table").strip()

    daily = filter_activity_window(
        with_days_ago(spark.table(dbutils.widgets.get("agg_table")), as_of=as_of),
        int(window) if window else None,
    ).select("user_id", "item_id", "view_date", "value", "num_views", "days_ago")
    daily.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(daily_fq)
    daily = spark.table(daily_fq)
    print(f"wrote {daily_fq}: {daily.count()} rows (as_of={as_of}, window={window or 'all'})")

    # Decay is applied here, per day; the train / evaluate tasks get the pair
    # frame and must not decay it again.
    interactions = daily_to_pairs(daily, decay_rate=decay_rate)
    # User state as of `as_of`: later viewership rows must not leak into a
    # backtest (and would make reruns drift).
    base_users = cr_user_features(
        spark.table(dbutils.widgets.get("viewership_table")), daily, as_of=as_of)
    base_users.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(base_users_fq)
    base_users = spark.table(base_users_fq)
    attrs_table = dbutils.widgets.get("attrs_table").strip() if use_attrs else ""
    # Summaries over the whole window, for the full-data model; the evaluate
    # task rebuilds them from its train fold only.
    users = add_interaction_attr_user_features(
        base_users, daily, spark.table(attrs_table) if use_attrs else None, enabled=use_attrs,
    )
    if metadata_table:
        items = cr_item_features(spark.table(metadata_table))
    else:
        print("metadata_table is empty: training without item features")
        items = spark.createDataFrame([], "item_id string, feature_name string, value double")
    dbutils.jobs.taskValues.set("as_of", as_of)
    dbutils.jobs.taskValues.set("attrs_table", attrs_table)
else:
    engagement = spark.table(dbutils.widgets.get("raw_engagement_table"))
    metadata = spark.table(dbutils.widgets.get("raw_metadata_table"))
    # Adjust these column names to your raw schema. Documented contract:
    #   engagement: user_id, item_id, value (+ optional event_type, days_ago) + user feature cols
    #   metadata:   item_id + item feature cols
    # Optional columns pass through when present: event_type feeds weighting,
    # days_ago enables the temporal split in the evaluate task.
    interactions, users, items = feature_engineering(
        engagement, metadata,
        user_col="user_id", item_col="item_id", value_col="value",
        event_type_col="event_type" if "event_type" in engagement.columns else None,
        days_ago_col="days_ago" if "days_ago" in engagement.columns else None,
        user_feature_cols=[c for c in engagement.columns
                           if c.startswith("user_") and c != "user_id"],
        item_feature_cols=[c for c in metadata.columns if c != "item_id"],
    )

if data_mode != "agg":
    spark.sql(f"DROP TABLE IF EXISTS {daily_fq}")
    spark.sql(f"DROP TABLE IF EXISTS {base_users_fq}")

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
