"""Most-popular baseline scored by the SparkEaseModel harness."""
import pytest

pytestmark = pytest.mark.spark

from kzn_recsys.spark import PopularityBaseline, build_and_train


def _frames(spark):
    inter = spark.createDataFrame([
        ("u1", "A", 1.0), ("u2", "A", 1.0), ("u3", "A", 1.0),
        ("u1", "B", 1.0), ("u2", "B", 1.0),
        ("u3", "C", 1.0),
    ], "user_id string, item_id string, value double")
    users = spark.createDataFrame([], "user_id string, feature_name string, value double")
    items = spark.createDataFrame([("D", "genre=x", 1.0)],
                                  "item_id string, feature_name string, value double")
    return inter, users, items


def test_popularity_ranks_by_distinct_users_and_skips_seen(spark):
    inter, users, items = _frames(spark)
    model = build_and_train(inter, users, items, lambda_=1.0)
    pop = PopularityBaseline.fit(inter, model.mappings)
    assert [i for i, _ in pop.predict({}, {}, top_k=4)] == ["A", "B", "C", "D"]
    assert [i for i, _ in pop.predict({"A": 1.0}, {}, top_k=2)] == ["B", "C"]
    assert [i for i, _ in pop._predict_within({}, {}, 4, {"C", "D"})] == ["C", "D"]


def test_popularity_evaluate_matches_a_hand_count(spark):
    inter, users, items = _frames(spark)
    model = build_and_train(inter, users, items, lambda_=1.0)
    pop = PopularityBaseline.fit(inter, model.mappings)
    test = spark.createDataFrame([("u3", "B", 1.0), ("u1", "D", 1.0)],
                                 "user_id string, item_id string, value double")
    report = pop.evaluate(test, inter, users, k_values=[1])
    # u3 (seen A, C) -> B: hit. u1 (seen A, B) -> C: miss.
    assert report["num_users"] == 2
    assert report["metrics"][0]["hit_rate"] == pytest.approx(0.5)
    assert set(report) == set(model.evaluate(test, inter, users, k_values=[1]))
