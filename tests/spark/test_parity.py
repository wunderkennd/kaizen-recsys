"""Cross-checks the PySpark EASE impl against the native Rust core.

Skipped unless the compiled kzn_recsys._native extension is importable.
Run after: .venv/bin/maturin develop
"""
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


def test_availability_report_matches_native(spark, tmp_path):
    """Rust and Spark harnesses agree on an availability-filtered report."""
    import polars as pl
    train_rows = [("u1", "A", 1.0, 50.0), ("u2", "A", 1.0, 50.0),
                  ("u3", "A", 1.0, 50.0), ("u3", "B", 1.0, 40.0), ("u3", "C", 1.0, 30.0),
                  ("u4", "A", 1.0, 50.0), ("u4", "B", 1.0, 40.0)]
    test_rows = [("u1", "C", 1.0, 5.0), ("u2", "B", 1.0, 0.5), ("u2", "C", 1.0, 3.0)]
    av_rows = [("A", "*", 100.0, None), ("B", "*", 1.0, None), ("C", "US", 400.0, 300.0), ("C", "US", 20.0, None)]
    uf_rows = [("u1", "region_US", 1.0), ("u2", "region_EMEA", 1.0)]
    icols = {"user_id": pl.String, "item_id": pl.String, "value": pl.Float64, "days_ago": pl.Float64}
    acols = {"item_id": pl.String, "territory": pl.String,
             "available_from_days_ago": pl.Float64, "available_to_days_ago": pl.Float64}
    train_p = _write_long_parquet(train_rows, icols, tmp_path, "train.parquet")
    test_p = _write_long_parquet(test_rows, icols, tmp_path, "test.parquet")
    av_p = _write_long_parquet(av_rows, acols, tmp_path, "av.parquet")
    uf_p = _write_long_parquet(uf_rows, ["user_id", "feature_name", "value"], tmp_path, "uf.parquet")
    t_p = _write_long_parquet([], ["item_id", "feature_name", "value"], tmp_path, "t.parquet")

    native = _native.build_and_train(interactions_path=train_p, user_features_path=uf_p,
                                     item_features_path=t_p, alpha=1.0, beta=1.0, lambda_=1.0)
    native_report = native.evaluate(test_p, train_p, user_features_path=uf_p, k_values=[1, 2],
                                    availability_path=av_p, user_territory_feature="region")

    from kzn_recsys.spark import build_and_train as spark_train
    ischema = "user_id string, item_id string, value double, days_ago double"
    train_df = spark.createDataFrame(train_rows, ischema)
    test_df = spark.createDataFrame(test_rows, ischema)
    av_df = spark.createDataFrame(
        av_rows, "item_id string, territory string, available_from_days_ago double, available_to_days_ago double")
    uf_df = spark.createDataFrame(uf_rows, ["user_id", "feature_name", "value"])
    t_df = spark.createDataFrame([], "item_id string, feature_name string, value double")
    spark_model = spark_train(train_df, uf_df, t_df, alpha=1.0, beta=1.0, lambda_=1.0)
    spark_report = spark_model.evaluate(test_df, train_df, uf_df, k_values=[1, 2],
                                        availability_df=av_df, user_territory_feature="region")

    for key in ("num_users", "num_interactions"):
        assert native_report[key] == spark_report[key], key
    assert abs(native_report["coverage"] - spark_report["coverage"]) < 1e-9
    for n, s in zip(native_report["metrics"], spark_report["metrics"]):
        assert n["k"] == s["k"]
        for name in ("precision", "recall", "ndcg", "map", "hit_rate"):
            assert abs(n[name] - s[name]) < 1e-6, (n["k"], name)
    na, sa = native_report["availability"], spark_report["availability"]
    for key in ("reference_days_ago", "num_eligible_items", "num_items_without_availability",
                "num_test_interactions_dropped", "num_users_skipped"):
        assert na[key] == sa[key], key
    assert [b["label"] for b in na["item_age_buckets"]] == [b["label"] for b in sa["item_age_buckets"]]
    for nb, sb in zip(na["item_age_buckets"], sa["item_age_buckets"]):
        assert nb["num_users"] == sb["num_users"], nb["label"]
        for n, s in zip(nb["metrics"], sb["metrics"]):
            assert abs(n["ndcg"] - s["ndcg"]) < 1e-6


def test_pair_frame_matches_native_decay_on_daily_rows(spark, tmp_path):
    """Decay applied per day in Spark (daily_to_pairs) then Rust with no
    decay == Rust applying decay per row on the daily grain (#102)."""
    import datetime as dt
    import polars as pl
    from kzn_recsys.spark.interactions_agg import daily_to_pairs, with_days_ago

    as_of = dt.date(2026, 10, 7)
    daily_rows = [("p1", "S1", dt.date(2026, 10, 1), 2.0, 2), ("p1", "S1", dt.date(2026, 10, 5), 1.0, 1),
                  ("p1", "S2", dt.date(2026, 10, 6), 0.5, 1), ("p2", "S1", dt.date(2026, 9, 30), 3.0, 1),
                  ("p2", "S2", dt.date(2026, 10, 3), 1.5, 1)]
    daily = spark.createDataFrame(
        daily_rows, "user_id string, item_id string, view_date date, value double, num_views long")
    daily = with_days_ago(daily, as_of=as_of)
    rate = 0.05
    pairs = daily_to_pairs(daily, decay_rate=rate).collect()

    icols = {"user_id": pl.String, "item_id": pl.String, "value": pl.Float64, "days_ago": pl.Float64}
    pair_p = _write_long_parquet([(r["user_id"], r["item_id"], r["value"], r["days_ago"]) for r in pairs],
                                 icols, tmp_path, "pairs.parquet")
    daily_p = _write_long_parquet(
        [(r["user_id"], r["item_id"], r["value"], r["days_ago"])
         for r in daily.select("user_id", "item_id", "value", "days_ago").collect()],
        icols, tmp_path, "daily.parquet")
    u_p = _write_long_parquet([], ["user_id", "feature_name", "value"], tmp_path, "u.parquet")
    t_p = _write_long_parquet([], ["item_id", "feature_name", "value"], tmp_path, "t.parquet")

    from_pairs = _native.build_and_train(interactions_path=pair_p, user_features_path=u_p,
                                         item_features_path=t_p, lambda_=5.0)
    from_daily = _native.build_and_train(interactions_path=daily_p, user_features_path=u_p,
                                         item_features_path=t_p, lambda_=5.0, decay_rate=rate)
    for user_hist in ({"S1": 1.0}, {"S2": 1.0}):
        a = dict(from_pairs.predict(user_hist, {}, top_k=5))
        b = dict(from_daily.predict(user_hist, {}, top_k=5))
        assert set(a) == set(b)
        for k in a:
            assert abs(a[k] - b[k]) < 1e-6
