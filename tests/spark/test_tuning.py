import pytest

pytestmark = pytest.mark.spark

from kzn_recsys.spark import grid_search, random_search


def _frames(spark):
    rows = []
    for u in range(8):
        for i in range(5):
            rows.append((f"u{u}", f"i{i}", 1.0))
    interactions = spark.createDataFrame(rows, ["user_id", "item_id", "value"])
    empty_u = spark.createDataFrame([], "user_id string, feature_name string, value double")
    empty_t = spark.createDataFrame([], "item_id string, feature_name string, value double")
    return interactions, empty_u, empty_t


def test_grid_search_returns_best_and_all_trials(spark):
    i, u, t = _frames(spark)
    result = grid_search(
        i, u, t,
        param_grid={"lambda_": [1.0, 100.0], "alpha": [1.0]},
        k_folds=2, eval_k=3, seed=1,
    )
    assert "best_params" in result
    assert "best_score" in result
    assert len(result["trials"]) == 2  # 2 lambda values x 1 alpha
    assert "lambda_" in result["best_params"]


def test_random_search_respects_n_iter(spark):
    i, u, t = _frames(spark)
    result = random_search(
        i, u, t,
        param_distributions={"lambda_": [1.0, 10.0, 100.0, 150.0], "beta": [0.5, 1.0]},
        n_iter=3, k_folds=2, eval_k=3, seed=1,
    )
    assert len(result["trials"]) == 3


def test_grid_search_is_deterministic(spark):
    i, u, t = _frames(spark)
    kw = dict(param_grid={"lambda_": [1.0, 100.0]}, k_folds=2, eval_k=3, seed=5)
    r1 = grid_search(i, u, t, **kw)
    r2 = grid_search(i, u, t, **kw)
    s1 = [round(tr["score"], 9) for tr in r1["trials"]]
    s2 = [round(tr["score"], 9) for tr in r2["trials"]]
    assert s1 == s2
    assert r1["best_params"] == r2["best_params"]
