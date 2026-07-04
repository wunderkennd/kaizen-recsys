# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Train & Tune
# MAGIC Optional grid search, then final EASE training on the full dataset.
# MAGIC Strategy `collect` (portable, default) or `distributed` (Spark-side Gram).

# COMMAND ----------
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "kzn_recsys")
dbutils.widgets.text("volume", "models")
dbutils.widgets.dropdown("strategy", "collect", ["collect", "distributed"])
dbutils.widgets.dropdown("do_tune", "false", ["true", "false"])
dbutils.widgets.text("k", "10")
dbutils.widgets.text("seed", "42")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")
strategy = dbutils.widgets.get("strategy")
do_tune = dbutils.widgets.get("do_tune") == "true"
k = int(dbutils.widgets.get("k"))
seed = int(dbutils.widgets.get("seed"))

def _tv(key, default):
    try:
        return dbutils.jobs.taskValues.get(taskKey="ingest", key=key, debugValue=default)
    except Exception:
        return default

interactions = spark.table(_tv("interactions_table", f"{catalog}.{schema}.kzn_interactions"))
users = spark.table(_tv("user_features_table", f"{catalog}.{schema}.kzn_user_features"))
items = spark.table(_tv("item_features_table", f"{catalog}.{schema}.kzn_item_features"))

# COMMAND ----------
# MAGIC %md
# MAGIC ### Strategy note
# MAGIC `collect` pulls the sparse matrices to the driver (fast, portable, right
# MAGIC for moderate catalogs). `distributed` accumulates the Gram matrix with
# MAGIC Spark joins — use it when the item+feature dimension is too large to
# MAGIC collect but still yields a driver-solvable Gram.

# COMMAND ----------
import mlflow
from kzn_recsys.spark import build_and_train, grid_search

best_lambda = 150.0
mlflow.set_experiment("/Shared/kzn_recsys_spark")
with mlflow.start_run() as run:
    if do_tune:
        res = grid_search(interactions, users, items,
                          {"lambda_": [50.0, 100.0, 150.0, 300.0]},
                          k_folds=3, eval_k=k, seed=seed)
        best_lambda = float(res["best_params"]["lambda_"])
        mlflow.log_metric("cv_best_ndcg", float(res["best_score"]))

    mlflow.log_params({"strategy": strategy, "lambda_": best_lambda,
                       "do_tune": do_tune, "eval_k": k})

    model = build_and_train(interactions, users, items,
                            lambda_=best_lambda, strategy=strategy)

    volume_dir = f"/Volumes/{catalog}/{schema}/{volume}"
    dbutils.fs.mkdirs(volume_dir.replace("/Volumes", "dbfs:/Volumes"))
    model_path = f"{volume_dir}/model.feas"
    model.save(model_path)
    mlflow.log_artifact(model_path)

    run_id = run.info.run_id

print(f"trained lambda={best_lambda}, saved {model_path}, run={run_id}")

# COMMAND ----------
dbutils.jobs.taskValues.set("model_path", model_path)
dbutils.jobs.taskValues.set("mlflow_run_id", run_id)
dbutils.jobs.taskValues.set("best_lambda", best_lambda)
