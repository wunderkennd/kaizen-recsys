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
