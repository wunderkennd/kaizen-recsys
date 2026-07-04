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
for uid in sorted(set(hist) | set(feats)):
    inter = hist.get(uid, {})
    for rank, (item_id, score) in enumerate(model.predict(inter, feats.get(uid, {}), top_k=k), 1):
        rows.append(Row(user_id=uid, rank=rank, item_id=item_id, score=float(score), run_id=run_id))

# COMMAND ----------
preds = spark.createDataFrame(rows)
sink = f"{catalog}.{schema}.kzn_predictions"
preds.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(sink)
print(f"wrote {sink}: {preds.count()} recommendations")
display(preds.orderBy("user_id", "rank").limit(20))
