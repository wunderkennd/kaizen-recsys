"""Cross-checks the PySpark EASE impl against the native Rust core.

Skipped unless the compiled kzn_recsys._native extension is importable.
Run after: .venv/bin/maturin develop
"""
import numpy as np
import pytest

pytestmark = [pytest.mark.spark, pytest.mark.parity]

_native = pytest.importorskip("kzn_recsys._native")


def _write_long_parquet(rows, cols, tmp_path, name):
    """Write a long-format parquet file. cols is either a list of names (all
    String except the last which is Float64) or a dict mapping name->dtype."""
    import polars as pl
    if isinstance(cols, list):
        # Infer schema: last column is Float64, others are String.
        schema = {c: (pl.Float64 if i == len(cols) - 1 else pl.String)
                  for i, c in enumerate(cols)}
    else:
        schema = cols
    path = str(tmp_path / name)
    if rows:
        pl.DataFrame(rows, schema=list(schema.keys()), orient="row").cast(schema).write_parquet(path)
    else:
        pl.DataFrame({c: pl.Series([], dtype=dtype) for c, dtype in schema.items()}).write_parquet(path)
    return path


def test_pyspark_scores_match_native_within_tol(spark, tmp_path):
    interactions = [("u1", "i1", 1.0), ("u1", "i2", 1.0),
                    ("u2", "i2", 1.0), ("u2", "i3", 1.0),
                    ("u3", "i1", 1.0), ("u3", "i3", 1.0)]
    cols = ["user_id", "item_id", "value"]
    i_path = _write_long_parquet(interactions, cols, tmp_path, "i.parquet")
    u_path = _write_long_parquet([], ["user_id", "feature_name", "value"], tmp_path, "u.parquet")
    t_path = _write_long_parquet([], ["item_id", "feature_name", "value"], tmp_path, "t.parquet")

    native_model = _native.build_and_train(
        interactions_path=i_path, user_features_path=u_path,
        item_features_path=t_path, alpha=1.0, beta=1.0, lambda_=10.0,
    )
    native_recs = dict(native_model.predict({"i1": 1.0}, {}, top_k=3))

    from kzn_recsys.spark import build_and_train as spark_train
    idf = spark.createDataFrame(interactions, cols)
    udf = spark.createDataFrame([], "user_id string, feature_name string, value double")
    tdf = spark.createDataFrame([], "item_id string, feature_name string, value double")
    spark_model = spark_train(idf, udf, tdf, alpha=1.0, beta=1.0, lambda_=10.0)
    spark_recs = dict(spark_model.predict({"i1": 1.0}, {}, top_k=3))

    # Spark predict excludes already-seen items; native may not. Compare on the
    # shared ids (Spark's set must be a subset of native's) and match scores there.
    assert set(spark_recs).issubset(set(native_recs))
    shared = set(native_recs) & set(spark_recs)
    assert shared, "no overlapping recommendations to compare"
    for item_id in shared:
        assert abs(native_recs[item_id] - spark_recs[item_id]) < 1e-5


def test_native_saved_model_loads_in_pyspark(spark, tmp_path):
    interactions = [("u1", "i1", 1.0), ("u1", "i2", 1.0), ("u2", "i2", 1.0)]
    cols = ["user_id", "item_id", "value"]
    i_path = _write_long_parquet(interactions, cols, tmp_path, "i.parquet")
    u_path = _write_long_parquet([], ["user_id", "feature_name", "value"], tmp_path, "u.parquet")
    t_path = _write_long_parquet([], ["item_id", "feature_name", "value"], tmp_path, "t.parquet")
    native_model = _native.build_and_train(
        interactions_path=i_path, user_features_path=u_path,
        item_features_path=t_path, alpha=1.0, beta=1.0, lambda_=10.0,
    )
    model_path = str(tmp_path / "native.fease")
    native_model.save(model_path)

    from kzn_recsys.spark import load_model
    py_model = load_model(model_path)
    native_recs = dict(native_model.predict({"i1": 1.0}, {}, top_k=2))
    py_recs = dict(py_model.predict({"i1": 1.0}, {}, top_k=2))
    # Loaded-from-native model must score the shared ids identically (within tol).
    shared = set(native_recs) & set(py_recs)
    assert shared, "no overlapping recommendations to compare"
    for item_id in shared:
        assert abs(native_recs[item_id] - py_recs[item_id]) < 1e-5


def test_pyspark_saved_model_loads_in_native(spark, tmp_path):
    from kzn_recsys.spark import build_and_train as spark_train
    interactions = [("u1", "i1", 1.0), ("u1", "i2", 1.0), ("u2", "i2", 1.0)]
    cols = ["user_id", "item_id", "value"]
    idf = spark.createDataFrame(interactions, cols)
    udf = spark.createDataFrame([], "user_id string, feature_name string, value double")
    tdf = spark.createDataFrame([], "item_id string, feature_name string, value double")
    spark_model = spark_train(idf, udf, tdf, alpha=1.0, beta=1.0, lambda_=10.0)
    model_path = str(tmp_path / "spark.fease")
    spark_model.save(model_path)

    native_model = _native.load_model(model_path)
    native_recs = dict(native_model.predict({"i1": 1.0}, {}, top_k=2))
    py_recs = dict(spark_model.predict({"i1": 1.0}, {}, top_k=2))
    # A Spark-trained model loaded by the native core must score shared ids identically.
    shared = set(native_recs) & set(py_recs)
    assert shared, "no overlapping recommendations to compare"
    for item_id in py_recs:
        if item_id in shared:
            assert abs(py_recs[item_id] - native_recs[item_id]) < 1e-5
