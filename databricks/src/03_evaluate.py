# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Evaluate
# MAGIC Holds out a temporal (default), leave-last-K-out or random test split,
# MAGIC retrains on the train fold with the chosen params, and logs ranking
# MAGIC metrics next to a most-popular baseline scored on the same users.
# MAGIC Enforces a quality gate.
# MAGIC
# MAGIC When the ingest task wrote `kzn_interactions_daily` (`data_mode=agg`),
# MAGIC the temporal split cuts the daily grain, not the pair frame: train is
# MAGIC every day before the cutoff, test the series each user started after
# MAGIC it (see `kzn_recsys.spark.temporal_split_daily`).
# MAGIC
# MAGIC With `do_tune=true` the hyperparameters are grid-searched here, on the
# MAGIC train fold only, so the test window never informs them. Otherwise the
# MAGIC train task's `best_lambda` is used.
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
# Training strategy for the evaluated model: "collect" | "distributed".
dbutils.widgets.text("strategy", "collect")
# Per-day decay for the daily-grain temporal split; must match the ingest task's.
dbutils.widgets.text("decay_rate", "0.0")
dbutils.widgets.text("do_tune", "false")
dbutils.widgets.text("param_grid", '{"lambda_": [50.0, 100.0, 150.0, 300.0]}')
dbutils.widgets.text("tune_k_folds", "3")
# Users sampled for tuning / for scoring; 0 = all. Both samples are a seeded
# hash of user_id, so they are stable across runs.
dbutils.widgets.text("tune_max_users", "0")
dbutils.widgets.text("max_eval_users", "0")
dbutils.widgets.text("baseline", "true")
# Used only when there is no train-task run to log into.
dbutils.widgets.text("mlflow_experiment", "/Shared/kzn_recsys_spark")

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
strategy = dbutils.widgets.get("strategy").strip() or "collect"
decay_rate = float(dbutils.widgets.get("decay_rate"))
do_tune = dbutils.widgets.get("do_tune").strip().lower() == "true"
tune_k_folds = int(dbutils.widgets.get("tune_k_folds"))
tune_max_users = int(dbutils.widgets.get("tune_max_users"))
max_eval_users = int(dbutils.widgets.get("max_eval_users"))
do_baseline = dbutils.widgets.get("baseline").strip().lower() == "true"
mlflow_experiment = dbutils.widgets.get("mlflow_experiment").strip()

def _tv(task, key, default):
    try:
        return dbutils.jobs.taskValues.get(taskKey=task, key=key, debugValue=default)
    except Exception:
        return default

interactions = spark.table(f"{catalog}.{schema}.kzn_interactions")
users = spark.table(f"{catalog}.{schema}.kzn_user_features")
items = spark.table(f"{catalog}.{schema}.kzn_item_features")
daily_fq = f"{catalog}.{schema}.kzn_interactions_daily"
daily = spark.table(daily_fq) if spark.catalog.tableExists(daily_fq) else None
best_lambda = float(_tv("train", "best_lambda", 150.0))
run_id = _tv("train", "mlflow_run_id", None)
as_of = _tv("ingest", "as_of", "")

# COMMAND ----------
import json

import mlflow
from pyspark.sql import functions as F

from kzn_recsys.spark import (
    PopularityBaseline, build_and_train, grid_search, leave_last_k_out_split,
    random_split, temporal_split, temporal_split_daily,
)

# Time-aware splits need days_ago; fall back to random if absent.
if "days_ago" not in interactions.columns and split != "random":
    print(f"no days_ago column; falling back from split={split!r} to random")
    split = "random"

eval_kwargs = {}
if split == "temporal":
    if daily is not None:
        train_df, test_df = temporal_split_daily(
            daily, days_ago_cutoff=days_ago_cutoff, decay_rate=decay_rate)
    else:
        train_df, test_df = temporal_split(interactions, days_ago_cutoff=days_ago_cutoff)
    if availability_table:  # evaluate() only takes a reference time with a table
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
train_df, test_df = train_df.cache(), test_df.cache()

if availability_table:
    eval_kwargs["availability_df"] = spark.table(availability_table)
    eval_kwargs["user_territory_feature"] = user_territory_feature


def _sample_users(df, n):
    """Up to n distinct users of df, by seeded hash; None (= all) when n <= 0."""
    if n <= 0:
        return None
    return (df.select("user_id").distinct()
            .orderBy(F.xxhash64("user_id", F.lit(seed)), "user_id").limit(n).cache())


def _restrict(df, sample):
    return df if sample is None else df.join(sample, on="user_id", how="left_semi")


sizes = {
    "train_interactions": train_df.count(),
    "test_interactions": test_df.count(),
    "train_users": train_df.select("user_id").distinct().count(),
    "test_users": test_df.select("user_id").distinct().count(),
}
print(f"split={split}: {sizes}")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Tune (train fold only) and train the evaluated model

# COMMAND ----------
params = {"lambda_": best_lambda}
tune_result = None
if do_tune:
    param_grid = json.loads(dbutils.widgets.get("param_grid"))
    tune_users = _sample_users(train_df, tune_max_users)
    tune_result = grid_search(
        _restrict(train_df, tune_users), _restrict(users, tune_users), items,
        param_grid, k_folds=tune_k_folds, eval_k=k, seed=seed,
    )
    params = {name: float(v) for name, v in tune_result["best_params"].items()}
    for t in tune_result["trials"]:
        print(f"  cv ndcg@{k}={t['score']:.4f}  {t['params']}")
    print(f"best params {params} (cv ndcg@{k}={tune_result['best_score']:.4f})")

model = build_and_train(train_df, users, items, strategy=strategy, **params)

eval_users = _sample_users(test_df, max_eval_users)
test_eval, train_eval, users_eval = (
    _restrict(df, eval_users) for df in (test_df, train_df, users)
)
report = model.evaluate(test_eval, train_eval, users_eval, k_values=[k], **eval_kwargs)
row = next(m for m in report["metrics"] if m["k"] == k)
print(report)

baseline_report = None
if do_baseline:
    baseline = PopularityBaseline.fit(train_df, model.mappings)
    baseline_report = baseline.evaluate(
        test_eval, train_eval, users_eval, k_values=[k], **eval_kwargs)
    print(f"popularity baseline: {baseline_report}")

# COMMAND ----------
metric_names = ("precision", "recall", "ndcg", "map", "hit_rate")
metrics = {f"test_{name}@{k}": float(row[name]) for name in metric_names}
metrics["coverage"] = float(report["coverage"])
metrics["num_eval_users"] = float(report["num_users"])
metrics["num_eval_interactions"] = float(report["num_interactions"])
metrics.update({f"num_{name}": float(v) for name, v in sizes.items()})
if tune_result is not None:
    metrics["cv_best_ndcg"] = float(tune_result["best_score"])
if baseline_report is not None:
    base_row = next(m for m in baseline_report["metrics"] if m["k"] == k)
    for name in metric_names:
        metrics[f"baseline_{name}@{k}"] = float(base_row[name])
        if base_row[name] > 0.0:
            metrics[f"lift_{name}@{k}"] = float(row[name]) / float(base_row[name])
    metrics["baseline_coverage"] = float(baseline_report["coverage"])
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

# Prefixed so they never collide with the train task's params in its run
# (MLflow params are immutable once logged).
eval_params = {
    "eval_split": split,
    "eval_days_ago_cutoff": days_ago_cutoff if split == "temporal" else None,
    "eval_holdout_k": holdout_k if split == "leave_last_k" else None,
    "eval_daily_grain": daily is not None and split == "temporal",
    "eval_decay_rate": decay_rate,
    "eval_as_of": as_of or None,
    "eval_strategy": strategy,
    "eval_do_tune": do_tune,
    "eval_max_eval_users": max_eval_users,
    "eval_availability_table": availability_table,
    **{f"eval_{name}": v for name, v in params.items()},
}
eval_params = {key: v for key, v in eval_params.items() if v is not None}
if run_id:
    with mlflow.start_run(run_id=run_id):
        mlflow.log_params(eval_params)
        mlflow.log_metrics(metrics)
else:
    if mlflow_experiment:
        mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run():
        mlflow.log_params(eval_params)
        mlflow.log_metrics(metrics)

# COMMAND ----------
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
dbutils.jobs.taskValues.set("test_ndcg", float(row["ndcg"]))
dbutils.notebook.exit(json.dumps({**eval_params, **metrics}))
