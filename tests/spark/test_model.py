import numpy as np
import pytest

pytestmark = pytest.mark.spark

from kzn_recsys.spark import build_and_train, load_model


def _frames(spark):
    interactions = spark.createDataFrame(
        [("u1", "i1", 1.0), ("u1", "i2", 1.0),
         ("u2", "i2", 1.0), ("u2", "i3", 1.0),
         ("u3", "i1", 1.0), ("u3", "i3", 1.0)],
        ["user_id", "item_id", "value"],
    )
    empty_u = spark.createDataFrame([], "user_id string, feature_name string, value double")
    empty_t = spark.createDataFrame([], "item_id string, feature_name string, value double")
    return interactions, empty_u, empty_t


def test_train_and_predict_returns_string_ids(spark):
    i, u, t = _frames(spark)
    model = build_and_train(i, u, t, alpha=1.0, beta=1.0, lambda_=10.0)
    recs = model.predict({"i1": 1.0}, {}, top_k=2)
    assert all(isinstance(item_id, str) for item_id, _ in recs)
    assert len(recs) == 2
    # i1 itself must be excluded by zero-diagonal; recs come from {i2, i3}
    assert {r[0] for r in recs}.issubset({"i2", "i3"})


def test_save_load_roundtrip_predicts_identically(spark, tmp_path):
    i, u, t = _frames(spark)
    model = build_and_train(i, u, t, alpha=1.0, beta=1.0, lambda_=10.0)
    before = model.predict({"i1": 1.0}, {}, top_k=3)
    path = str(tmp_path / "m.fease")
    model.save(path)
    reloaded = load_model(path)
    after = reloaded.predict({"i1": 1.0}, {}, top_k=3)
    assert before == after


def test_build_and_train_rejects_unknown_strategy(spark):
    i, u, t = _frames(spark)
    with pytest.raises(ValueError, match="unknown strategy"):
        build_and_train(i, u, t, alpha=1.0, beta=1.0, lambda_=10.0, strategy="nope")


def test_evaluate_returns_metric_report(spark):
    train = spark.createDataFrame(
        [("u1", "i1", 1.0), ("u1", "i2", 1.0),
         ("u2", "i2", 1.0), ("u2", "i3", 1.0),
         ("u3", "i1", 1.0), ("u3", "i3", 1.0)],
        ["user_id", "item_id", "value"],
    )
    test = spark.createDataFrame(
        [("u1", "i3", 1.0), ("u2", "i1", 1.0)],
        ["user_id", "item_id", "value"],
    )
    empty_u = spark.createDataFrame([], "user_id string, feature_name string, value double")
    empty_t = spark.createDataFrame([], "item_id string, feature_name string, value double")
    model = build_and_train(train, empty_u, empty_t, alpha=1.0, beta=1.0, lambda_=10.0)
    report = model.evaluate(test, train, empty_u, k_values=[1, 2, 3])
    assert "metrics" in report and "coverage" in report
    ks = {m["k"] for m in report["metrics"]}
    assert ks == {1, 2, 3}
    for m in report["metrics"]:
        assert 0.0 <= m["ndcg"] <= 1.0
        assert 0.0 <= m["recall"] <= 1.0
    assert report["num_users"] >= 1
