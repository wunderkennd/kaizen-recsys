"""Run the bundle's ingest and evaluate notebooks on local Spark.

The notebooks are executed as plain Python with stand-ins for the Databricks
globals (``dbutils``, a ``spark`` that accepts ``CREATE CATALOG``) and for
``mlflow``, against tables in a throwaway ``spark_catalog`` database.
"""
import datetime as dt
import json
import random
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import pytest

pytestmark = pytest.mark.spark

SRC = Path(__file__).resolve().parents[2] / "databricks" / "src"
DB = "kzn_bundle_nb_test"
AS_OF = dt.date(2026, 10, 7)
SERIES = [f"S{i:02d}" for i in range(24)]


class _Exit(Exception):
    pass


class _Dbutils:
    """The slice of dbutils the notebooks use; one instance per task run."""

    def __init__(self, task, params, task_values):
        widgets = {}

        def _define(name, default, *_):
            widgets.setdefault(name, str(params.get(name, default)))

        def _get_tv(taskKey, key, debugValue=None):
            if (taskKey, key) not in task_values:
                raise ValueError(f"no task value {taskKey}.{key}")
            return task_values[(taskKey, key)]

        def _exit(value):
            raise _Exit(value)

        self.widgets = types.SimpleNamespace(
            text=_define, dropdown=_define, get=lambda name: widgets[name])
        self.jobs = types.SimpleNamespace(taskValues=types.SimpleNamespace(
            set=lambda key, value: task_values.__setitem__((task, key), value), get=_get_tv))
        self.notebook = types.SimpleNamespace(exit=_exit)


class _Spark:
    def __init__(self, session):
        self._s = session

    def sql(self, query):
        if query.strip().upper().startswith("CREATE CATALOG"):
            return None
        return self._s.sql(query)

    def __getattr__(self, name):
        return getattr(self._s, name)


def _run(spark, notebook, task, params, task_values):
    code = compile((SRC / notebook).read_text(), str(SRC / notebook), "exec")
    try:
        exec(code, {"spark": _Spark(spark), "dbutils": _Dbutils(task, params, task_values)})
    except _Exit as e:
        return json.loads(e.args[0])
    return None


@pytest.fixture
def mlflow_log(monkeypatch):
    log = {"params": {}, "metrics": {}, "experiment": None}
    fake = types.SimpleNamespace(
        set_experiment=lambda name: log.__setitem__("experiment", name),
        start_run=lambda run_id=None: nullcontext(),
        log_params=log["params"].update,
        log_metrics=log["metrics"].update,
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    return log


@pytest.fixture
def source_tables(spark, tmp_path):
    """agg / viewership / metadata tables with persona structure: each user
    sticks to one third of the catalog, watched over the last 60 days."""
    spark.sql(f"DROP DATABASE IF EXISTS {DB} CASCADE")
    spark.sql(f"CREATE DATABASE {DB} LOCATION '{tmp_path / 'warehouse'}'")
    rng = random.Random(7)
    agg, views = [], []
    for u in range(60):
        uid, persona = f"p{u:03d}", u % 3
        pool = SERIES[persona * 8:(persona + 1) * 8]
        for item in rng.sample(pool, 6):
            for d in sorted(rng.sample(range(1, 60), 2)):
                agg.append((uid, item, AS_OF - dt.timedelta(days=d), rng.uniform(1, 8), 1))
        views.append((uid, dt.datetime(2026, 10, 1), "crunchyroll", f"plan{persona}",
                      "US", "NA", "Paying", 100))
    spark.createDataFrame(
        agg, "user_id string, item_id string, view_date date, value double, num_views long",
    ).write.saveAsTable(f"{DB}.agg")
    spark.createDataFrame(views, (
        "view_profile_id string, view_ts timestamp, view_subsidiary string, "
        "view_subscription_plan string, account_country_code_account string, "
        "region_major_account string, subscription_status string, account_tenure_days int"
    )).write.saveAsTable(f"{DB}.viewership")
    spark.createDataFrame(
        [(s, "episode", "ja-JP", f"g{i // 8}", "A", f"genre{i // 8}", "tag") for i, s in
         enumerate(SERIES)],
        "catalog_show_id string, media_type string, media_audio_language string, "
        "airtable_primary_genre string, airtable_ca_brand_grade string, "
        "media_genres string, media_tags string",
    ).write.saveAsTable(f"{DB}.metadata")
    yield
    spark.sql(f"DROP DATABASE IF EXISTS {DB} CASCADE")


def test_eval_job_runs_end_to_end_on_the_agg_table(spark, source_tables, mlflow_log):
    common = {"catalog": "spark_catalog", "schema": DB, "seed": "3", "decay_rate": "0.01"}
    task_values = {}
    _run(spark, "01_ingest_and_features.py", "ingest", {
        **common, "data_mode": "agg",
        "agg_table": f"{DB}.agg", "viewership_table": f"{DB}.viewership",
        "metadata_table": f"{DB}.metadata", "as_of": AS_OF.isoformat(),
        "activity_window_days": "90",
    }, task_values)
    assert task_values[("ingest", "as_of")] == AS_OF.isoformat()
    assert spark.catalog.tableExists(f"spark_catalog.{DB}.kzn_interactions_daily")
    assert spark.table(f"{DB}.kzn_item_features").count() > 0

    out = _run(spark, "03_evaluate.py", "evaluate", {
        **common, "split": "temporal", "days_ago_cutoff": "14", "do_tune": "true",
        "param_grid": '{"lambda_": [1.0, 10.0]}', "tune_k_folds": "2",
        "tune_max_users": "40", "max_eval_users": "30", "k": "5",
        "mlflow_experiment": "/Shared/test",
    }, task_values)

    assert out["eval_daily_grain"] is True and out["eval_as_of"] == AS_OF.isoformat()
    assert out["eval_lambda_"] in (1.0, 10.0)
    assert 0 < out["num_eval_users"] <= 30
    assert 0.0 <= out["test_ndcg@5"] <= 1.0 and "baseline_ndcg@5" in out
    assert "cv_best_ndcg" in out
    assert mlflow_log["experiment"] == "/Shared/test"
    assert mlflow_log["metrics"]["test_ndcg@5"] == out["test_ndcg@5"]
    assert task_values[("evaluate", "test_ndcg")] == out["test_ndcg@5"]
    # Persona structure: the model has to beat most-popular here.
    assert out["test_ndcg@5"] > out["baseline_ndcg@5"]

    # Availability-aware: the temporal cutoff becomes the reference time.
    # Persona 0's series became available after the cutoff, so they leave the
    # ranking and the relevant sets.
    spark.createDataFrame(
        [(s, 5.0 if i < 8 else 400.0) for i, s in enumerate(SERIES)],
        "item_id string, available_from_days_ago double",
    ).write.saveAsTable(f"{DB}.availability")
    out = _run(spark, "03_evaluate.py", "evaluate", {
        **common, "split": "temporal", "days_ago_cutoff": "14", "k": "5",
        "availability_table": f"{DB}.availability",
    }, task_values)
    assert out["availability_num_eligible_items"] == 16.0
    assert out["availability_num_test_interactions_dropped"] > 0

    # A later non-agg ingest into the same schema drops the daily table, so
    # evaluate cannot split a stale one; the demo job's defaults still run.
    _run(spark, "01_ingest_and_features.py", "ingest",
         {**common, "data_mode": "synthetic", "n_users": "40", "n_items": "20"}, {})
    assert not spark.catalog.tableExists(f"spark_catalog.{DB}.kzn_interactions_daily")
    out = _run(spark, "03_evaluate.py", "evaluate", {**common, "k": "5"}, {})
    assert out["eval_split"] == "random" and out["eval_lambda_"] == 150.0
    assert out["num_eval_users"] > 0
