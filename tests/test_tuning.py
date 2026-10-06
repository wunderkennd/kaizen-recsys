"""Strategy-driven hyperparameter search (`tune_*`, issue #97) on EASE.

EASE trains in milliseconds, so these exercise the whole ask/evaluate/observe
loop, the dict-driven space parser, warm starts, and the error paths without
needing the `ml-models` build.
"""

import tempfile
from pathlib import Path

import polars as pl
import pytest

import kzn_recsys as fease


@pytest.fixture(scope="module")
def tuning_data():
    """Six users over four items (enough for 2-fold CV), plus side features."""
    with tempfile.TemporaryDirectory() as tmpdir:
        i_path = Path(tmpdir) / "interactions.parquet"
        u_path = Path(tmpdir) / "user_features.parquet"
        t_path = Path(tmpdir) / "item_features.parquet"
        pl.DataFrame(
            {
                "user_id": ["u0", "u0", "u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4", "u5", "u5"],
                "item_id": ["G0", "G1", "G1", "G2", "G0", "G2", "G2", "G3", "G0", "G3", "G1", "G3"],
                "value": [1.0] * 12,
            }
        ).write_parquet(i_path)
        pl.DataFrame(
            {
                "user_id": ["u0", "u1", "u2", "u3", "u4", "u5"],
                "feature_name": ["plan_A", "plan_B", "plan_A", "plan_B", "plan_A", "plan_B"],
                "value": [1.0] * 6,
            }
        ).write_parquet(u_path)
        pl.DataFrame(
            {
                "item_id": ["G0", "G1", "G2", "G3"],
                "feature_name": ["genre_X", "genre_Y", "genre_X", "genre_Y"],
                "value": [1.0] * 4,
            }
        ).write_parquet(t_path)
        yield str(i_path), str(u_path), str(t_path)


EASE_PARAMS = {
    "alpha", "beta", "lambda_", "meta_weight", "decay_rate", "ips_alpha", "sparsity_threshold",
}


def _check_shape(result, n_trials, strategy):
    assert result["metric"] == "ndcg@10"
    assert result["strategy"] == strategy
    assert set(result["best_params"]) == EASE_PARAMS
    assert len(result["trials"]) == n_trials
    for t in result["trials"]:
        assert set(t["params"]) == EASE_PARAMS
        assert len(t["fold_scores"]) == 2
    assert result["best_score"] == max(t["mean_score"] for t in result["trials"])


def test_tune_ease_grid_matches_grid_search_ease(tuning_data):
    i_path, u_path, t_path = tuning_data
    space = {"alpha": [0.5, 1.0], "lambda_": [10.0, 100.0]}
    tuned = fease.tune_ease(
        i_path, space, user_features_path=u_path, item_features_path=t_path,
        strategy="grid", max_trials=100, n_folds=2, eval_k=10, seed=42,
    )
    legacy = fease.grid_search_ease(
        i_path, u_path, t_path, param_grid=space, n_folds=2, eval_k=10, seed=42
    )
    _check_shape(tuned, 4, "grid")
    assert [t["params"] for t in tuned["trials"]] == [t["params"] for t in legacy["trials"]]
    assert [t["mean_score"] for t in tuned["trials"]] == [t["mean_score"] for t in legacy["trials"]]
    assert tuned["best_params"] == legacy["best_params"]
    # max_trials truncates a grid in grid order.
    two = fease.tune_ease(
        i_path, space, user_features_path=u_path, item_features_path=t_path,
        strategy="grid", max_trials=2, n_folds=2, eval_k=10, seed=42,
    )
    assert [t["params"] for t in two["trials"]] == [t["params"] for t in legacy["trials"][:2]]


def test_tune_ease_random_matches_random_search_ease(tuning_data):
    i_path, u_path, t_path = tuning_data
    space = {"alpha": [0.5, 1.0, 2.0], "lambda_": [10.0, 50.0, 100.0]}
    tuned = fease.tune_ease(
        i_path, space, user_features_path=u_path, item_features_path=t_path,
        strategy="random", max_trials=3, n_folds=2, eval_k=10, seed=7,
    )
    legacy = fease.random_search_ease(
        i_path, u_path, t_path, param_grid=space, n_trials=3, n_folds=2, eval_k=10, seed=7
    )
    _check_shape(tuned, 3, "random")
    assert [t["params"] for t in tuned["trials"]] == [t["params"] for t in legacy["trials"]]


def test_tune_ease_tpe_continuous_space_and_warm_start(tuning_data):
    i_path, u_path, t_path = tuning_data
    space = {
        "lambda_": ("log", 1.0, 1000.0),
        "alpha": {"type": "uniform", "low": 0.5, "high": 2.0},
        "beta": [0.5, 1.0],
    }
    first = fease.tune_ease(
        i_path, space, user_features_path=u_path, item_features_path=t_path,
        strategy="tpe", max_trials=6, n_folds=2, eval_k=10, seed=42, batch_size=2, n_startup=2,
    )
    _check_shape(first, 6, "tpe")
    for t in first["trials"]:
        assert 1.0 <= t["params"]["lambda_"] <= 1000.0
        assert 0.5 <= t["params"]["alpha"] <= 2.0
        assert t["params"]["beta"] in (0.5, 1.0)
        assert t["params"]["meta_weight"] == 0.0  # omitted axis keeps its default

    # Deterministic for a fixed seed.
    again = fease.tune_ease(
        i_path, space, user_features_path=u_path, item_features_path=t_path,
        strategy="tpe", max_trials=6, n_folds=2, eval_k=10, seed=42, batch_size=2, n_startup=2,
    )
    assert [t["params"] for t in again["trials"]] == [t["params"] for t in first["trials"]]

    # Warm start from a previous result's trials: only new trials come back.
    resumed = fease.tune_ease(
        i_path, space, user_features_path=u_path, item_features_path=t_path,
        strategy="tpe", max_trials=2, n_folds=2, eval_k=10, seed=43,
        warm_start=first["trials"],
    )
    _check_shape(resumed, 2, "tpe")


def test_tune_ease_grid_refuses_continuous_axis(tuning_data):
    i_path, u_path, t_path = tuning_data
    with pytest.raises(RuntimeError, match="continuous"):
        fease.tune_ease(
            i_path, {"lambda_": ("log", 1.0, 100.0)},
            user_features_path=u_path, item_features_path=t_path,
            strategy="grid", max_trials=4, n_folds=2,
        )


def test_tune_ease_argument_errors(tuning_data):
    i_path, u_path, t_path = tuning_data
    with pytest.raises(ValueError, match="unknown parameter"):
        fease.tune_ease(
            i_path, {"lambda": [1.0]}, user_features_path=u_path, item_features_path=t_path,
            max_trials=2, n_folds=2,
        )
    with pytest.raises(ValueError, match="strategy"):
        fease.tune_ease(
            i_path, {"lambda_": [1.0, 2.0]}, user_features_path=u_path, item_features_path=t_path,
            strategy="bayes", max_trials=2, n_folds=2,
        )
    with pytest.raises(ValueError, match="range kind"):
        fease.tune_ease(
            i_path, {"lambda_": ("normal", 1.0, 2.0)}, user_features_path=u_path,
            item_features_path=t_path, max_trials=2, n_folds=2,
        )
    with pytest.raises(ValueError, match="user_features_path"):
        fease.tune_ease(i_path, {"lambda_": [1.0, 2.0]}, max_trials=2, n_folds=2)
    with pytest.raises(ValueError, match="warm_start"):
        fease.tune_ease(
            i_path, {"lambda_": [1.0, 2.0]}, user_features_path=u_path, item_features_path=t_path,
            max_trials=2, n_folds=2, warm_start=[{"mean_score": 0.1}],
        )
