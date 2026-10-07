# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Evaluate
# MAGIC Holds out a temporal (default), leave-last-K-out or random test split,
# MAGIC retrains on the train fold with the chosen params, and logs ranking
# MAGIC metrics. Enforces a quality gate.
# MAGIC
# MAGIC With `availability_table` set, ranking is restricted to items available
# MAGIC to each user at the reference time (the temporal cutoff, or per user the
# MAGIC oldest held-out interaction under leave-last-K-out) and the report gains
# MAGIC an item-age breakdown. Availability filtering has no reference time under
# MAGIC a random split and is rejected there.

# COMMAND ----------
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "kzn_recsys")
dbutils.widgets.text("k", "10")
dbutils.widgets.text("ndcg_gate", "0.0")
dbutils.widgets.text("seed", "42")
dbutils.widgets.text("days_ago_cutoff", "7.0")
# "temporal" | "leave_last_k" | "random"
dbutils.widgets.text("split", "temporal")
dbutils.widgets.text("holdout_k", "1")
# Optional availability table: item_id, [season_id], [territory],
# available_from_days_ago, [available_to_days_ago]. Empty = full-catalog ranking.
dbutils.widgets.text("availability_table", "")
# Categorical user-feature column carrying territory (one-hot `<col>_<value>`).
dbutils.widgets.text("user_territory_feature", "")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
k = int(dbutils.widgets.get("k"))
ndcg_gate = float(dbutils.widgets.get("ndcg_gate"))
seed = int(dbutils.widgets.get("seed"))
days_ago_cutoff = float(dbutils.widgets.get("days_ago_cutoff"))
split = dbutils.widgets.get("split").strip().lower() or "temporal"
holdout_k = int(dbutils.widgets.get("holdout_k"))
availability_table = dbutils.widgets.get("availability_table").strip() or None
user_territory_feature = dbutils.widgets.get("user_territory_feature").strip() or None

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
from kzn_recsys.spark import (
    build_and_train, leave_last_k_out_split, random_split, temporal_split,
)

# Time-aware splits need days_ago; fall back to random if absent.
if "days_ago" not in interactions.columns and split != "random":
    print(f"no days_ago column; falling back from split={split!r} to random")
    split = "random"

eval_kwargs = {}
if split == "temporal":
    train_df, test_df = temporal_split(interactions, days_ago_cutoff=days_ago_cutoff)
    eval_kwargs["reference_days_ago"] = days_ago_cutoff
elif split == "leave_last_k":
    train_df, test_df = leave_last_k_out_split(interactions, k=holdout_k)
    # reference time is per user: the oldest held-out interaction
elif split == "random":
    if availability_table:
        raise ValueError(
            "availability filtering needs a reference time; a random split has none. "
            "Use split=temporal or split=leave_last_k."
        )
    train_df, test_df = random_split(interactions, test_ratio=0.2, seed=seed)
else:
    raise ValueError(f"unknown split {split!r}; expected temporal | leave_last_k | random")

if availability_table:
    eval_kwargs["availability_df"] = spark.table(availability_table)
    eval_kwargs["user_territory_feature"] = user_territory_feature

model = build_and_train(train_df, users, items, lambda_=best_lambda, strategy="collect")
report = model.evaluate(test_df, train_df, users, k_values=[k], **eval_kwargs)
row = next(m for m in report["metrics"] if m["k"] == k)
print(report)

# COMMAND ----------
metrics = {f"test_{name}@{k}": float(row[name])
           for name in ("precision", "recall", "ndcg", "map", "hit_rate")}
metrics["coverage"] = float(report["coverage"])
if "availability" in report:
    av = report["availability"]
    for name in ("num_eligible_items", "num_items_without_availability",
                 "num_test_interactions_dropped", "num_users_skipped"):
        metrics[f"availability_{name}"] = float(av[name])
    for bucket in av["item_age_buckets"]:
        if bucket["metrics"]:
            b = next(m for m in bucket["metrics"] if m["k"] == k)
            tag = bucket["label"].replace("<", "lt").replace(">=", "ge")
            metrics[f"test_ndcg@{k}_age_{tag}"] = float(b["ndcg"])
            metrics[f"num_users_age_{tag}"] = float(bucket["num_users"])
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
