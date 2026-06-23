import numpy as np
import pytest

pytestmark = pytest.mark.spark

from kzn_recsys.spark.dataframes import build_mappings, build_csr_inputs


def _frames(spark):
    interactions = spark.createDataFrame(
        [("u1", "i1", 1.0), ("u1", "i2", 1.0), ("u2", "i2", 2.0)],
        ["user_id", "item_id", "value"],
    )
    user_features = spark.createDataFrame(
        [("u1", "plan_premium", 1.0)], ["user_id", "feature_name", "value"]
    )
    item_features = spark.createDataFrame(
        [("i1", "genre_drama", 1.0)], ["item_id", "feature_name", "value"]
    )
    return interactions, user_features, item_features


def test_build_mappings_sorted_and_complete(spark):
    i, u, t = _frames(spark)
    m = build_mappings(i, u, t)
    # users from interactions + user features; items from interactions + item features
    assert m.idx_to_user == ["u1", "u2"]
    assert m.idx_to_item == ["i1", "i2"]
    assert m.idx_to_user_feature == ["plan_premium"]
    assert m.idx_to_item_feature == ["genre_drama"]


def test_build_csr_shapes(spark):
    i, u, t = _frames(spark)
    m = build_mappings(i, u, t)
    X, U, T = build_csr_inputs(i, u, t, m, weighting=None)
    assert X.shape == (2, 2)   # 2 users x 2 items
    assert U.shape == (2, 1)   # 2 users x 1 user-feature
    assert T.shape == (1, 2)   # 1 item-feature x 2 items (transposed)
    # X value for (u2, i2) == 2.0
    assert X[m.user_to_idx["u2"], m.item_to_idx["i2"]] == 2.0
