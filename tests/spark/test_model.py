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


def test_evaluate_map_is_per_k(spark):
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
    by_k = {m["k"]: m["map"] for m in report["metrics"]}
    # MAP@k is non-decreasing in k (truncating to a larger k can only add hits)
    assert by_k[1] <= by_k[2] <= by_k[3]
    for v in by_k.values():
        assert 0.0 <= v <= 1.0


def test_predict_cold_start_user_with_features(spark):
    # Train WITH user features so the feature path is exercised through the facade.
    interactions = spark.createDataFrame(
        [("u1", "i1", 1.0), ("u1", "i2", 1.0),
         ("u2", "i2", 1.0), ("u2", "i3", 1.0),
         ("u3", "i1", 1.0), ("u3", "i3", 1.0)],
        ["user_id", "item_id", "value"],
    )
    user_features = spark.createDataFrame(
        [("u1", "plan_premium", 1.0), ("u2", "plan_premium", 1.0),
         ("u3", "plan_free", 1.0)],
        ["user_id", "feature_name", "value"],
    )
    empty_t = spark.createDataFrame([], "item_id string, feature_name string, value double")
    model = build_and_train(interactions, user_features, empty_t,
                            alpha=1.0, beta=1.0, lambda_=10.0)
    # Cold-start user: no interactions, only a feature -> still gets recs via the
    # user-feature columns of S (predict_scores beta-weights the feature entries).
    recs = model.predict({}, {"plan_premium": 1.0}, top_k=3)
    assert all(isinstance(item_id, str) for item_id, _ in recs)
    assert len(recs) >= 1


def test_ips_weighting_changes_the_model(spark):
    from kzn_recsys.spark import WeightingConfig
    # Skewed popularity: i1 very popular, i3 rare.
    rows = ([("u%d" % u, "i1", 1.0) for u in range(6)] +
            [("u%d" % u, "i2", 1.0) for u in range(3)] +
            [("u0", "i3", 1.0)])
    interactions = spark.createDataFrame(rows, ["user_id", "item_id", "value"])
    empty_u = spark.createDataFrame([], "user_id string, feature_name string, value double")
    empty_t = spark.createDataFrame([], "item_id string, feature_name string, value double")
    plain = build_and_train(interactions, empty_u, empty_t, alpha=1.0, beta=1.0, lambda_=10.0)
    ips = build_and_train(interactions, empty_u, empty_t, alpha=1.0, beta=1.0, lambda_=10.0,
                          weighting=WeightingConfig(event_weights=None, decay_rate=0.0,
                                                    ips_alpha=0.7, sparsity_threshold=0.0))
    # IPS reweights interaction values by item popularity, so the learned S differs.
    assert not np.allclose(plain.s_matrix, ips.s_matrix)
