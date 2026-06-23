import numpy as np
import pytest

pytestmark = pytest.mark.spark

from kzn_recsys.spark.dataframes import build_mappings
from kzn_recsys.spark.gram import gram_collect, gram_distributed
from kzn_recsys.spark.ease_core import EaseParams


def _frames(spark):
    interactions = spark.createDataFrame(
        [("u1", "i1", 1.0), ("u1", "i2", 1.0),
         ("u2", "i2", 2.0), ("u2", "i3", 1.0),
         ("u3", "i1", 1.0), ("u3", "i3", 3.0)],
        ["user_id", "item_id", "value"],
    )
    u = spark.createDataFrame([("u1", "plan_x", 1.0)],
                              ["user_id", "feature_name", "value"])
    t = spark.createDataFrame([("i1", "genre_y", 1.0)],
                              ["item_id", "feature_name", "value"])
    return interactions, u, t


def test_distributed_matches_collect(spark):
    i, u, t = _frames(spark)
    m = build_mappings(i, u, t)
    params = EaseParams(alpha=1.0, beta=1.0, lambda_=10.0)
    S_collect = gram_collect(i, u, t, m, params, weighting=None)
    S_dist = gram_distributed(i, u, t, m, params, weighting=None)
    assert np.allclose(S_collect, S_dist, atol=1e-9)
