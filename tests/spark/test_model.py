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


def _availability_frames(spark):
    """u1/u2 trained on A only; B co-watched with A by u3/u4 so it outranks C.
    B released 1 day ago -> ineligible at u1's reference time (5 days ago)."""
    train = spark.createDataFrame(
        [("u1", "A", 1.0, 50.0), ("u2", "A", 1.0, 50.0),
         ("u3", "A", 1.0, 50.0), ("u3", "B", 1.0, 40.0), ("u3", "C", 1.0, 30.0),
         ("u4", "A", 1.0, 50.0), ("u4", "B", 1.0, 40.0)],
        ["user_id", "item_id", "value", "days_ago"],
    )
    test = spark.createDataFrame([("u1", "C", 1.0, 5.0)], ["user_id", "item_id", "value", "days_ago"])
    av = spark.createDataFrame(
        [("A", "A1", "*", 100.0, None), ("B", "B1", "*", 1.0, None), ("C", "C1", "*", 100.0, None)],
        "item_id string, season_id string, territory string, "
        "available_from_days_ago double, available_to_days_ago double",
    )
    users = spark.createDataFrame(
        [("u1", "region_US", 1.0), ("u2", "region_EMEA", 1.0)],
        ["user_id", "feature_name", "value"],
    )
    empty_t = spark.createDataFrame([], "item_id string, feature_name string, value double")
    return train, test, av, users, empty_t


def test_evaluate_with_availability_filters_late_release(spark):
    train, test, av, users, empty_t = _availability_frames(spark)
    model = build_and_train(train, users, empty_t, alpha=1.0, beta=1.0, lambda_=1.0)
    plain = model.evaluate(test, train, users, k_values=[1, 2])
    assert "availability" not in plain
    assert plain["metrics"][0]["precision"] == 0.0  # B outranks C on the full catalog

    report = model.evaluate(test, train, users, k_values=[1, 2], availability_df=av)
    assert report["num_users"] == 1
    assert report["metrics"][0]["precision"] == 1.0
    a = report["availability"]
    assert a["reference_days_ago"] is None
    assert a["num_eligible_items"] == 2
    assert a["num_items_without_availability"] == 0
    assert a["num_test_interactions_dropped"] == 0 and a["num_users_skipped"] == 0
    assert report["coverage"] == 0.5
    assert [b["label"] for b in a["item_age_buckets"]] == ["<30d", "30-365d", ">=365d"]
    assert a["item_age_buckets"][1]["num_users"] == 1
    assert a["item_age_buckets"][1]["metrics"][0]["ndcg"] == 1.0
    assert a["item_age_buckets"][0]["metrics"] == []

    with pytest.raises(ValueError, match="availability_df"):
        model.evaluate(test, train, users, k_values=[1], reference_days_ago=5.0)
    with pytest.raises(ValueError, match="days_ago"):
        model.evaluate(test.drop("days_ago"), train, users, k_values=[1], availability_df=av)


def test_evaluate_with_availability_territory_rollup(spark):
    train, _, _, users, empty_t = _availability_frames(spark)
    test = spark.createDataFrame([("u1", "C", 1.0), ("u2", "C", 1.0)], ["user_id", "item_id", "value"])
    av = spark.createDataFrame(
        [("A", "*", 100.0, None), ("C", "US", 400.0, 300.0), ("C", "US", 20.0, None)],
        "item_id string, territory string, available_from_days_ago double, available_to_days_ago double",
    )
    model = build_and_train(train, users, empty_t, alpha=1.0, beta=1.0, lambda_=1.0)
    report = model.evaluate(test, train, users, k_values=[1], availability_df=av,
                            user_territory_feature="region", reference_days_ago=5.0)
    assert report["num_users"] == 1
    assert report["availability"]["num_users_skipped"] == 1
    assert report["availability"]["num_test_interactions_dropped"] == 1
    assert report["availability"]["num_items_without_availability"] == 1  # B
    assert report["metrics"][0]["hit_rate"] == 1.0
    assert report["availability"]["item_age_buckets"][2]["num_users"] == 1  # first season, 400d
