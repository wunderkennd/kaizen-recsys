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
dbutils.widgets.text("days_ago_cutoff", "7.0")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
k = int(dbutils.widgets.get("k"))
ndcg_gate = float(dbutils.widgets.get("ndcg_gate"))
seed = int(dbutils.widgets.get("seed"))
days_ago_cutoff = float(dbutils.widgets.get("days_ago_cutoff"))

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
    train_df, test_df = temporal_split(interactions, days_ago_cutoff=days_ago_cutoff)
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

# An empty holdout reports NDCG 0.0 over zero users; that is "not evaluated",
# not "passed", so fail before the threshold comparison.
assert report["num_users"] > 0 and report["num_interactions"] > 0, (
    f"empty holdout (num_users={report['num_users']}, "
    f"num_interactions={report['num_interactions']}); nothing evaluated, failing task."
)
assert row["ndcg"] >= ndcg_gate, (
    f"NDCG@{k}={row['ndcg']:.4f} below gate {ndcg_gate}; failing task."
)
print(f"quality gate passed: NDCG@{k}={row['ndcg']:.4f} >= {ndcg_gate}")
