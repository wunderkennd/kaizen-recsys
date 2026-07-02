import math

from kzn_recsys.spark.metrics import (
    precision_at_k, recall_at_k, ndcg_at_k, mean_average_precision,
    coverage, hit_rate_at_k,
)


def test_precision_at_k():
    assert abs(precision_at_k([1, 2, 3, 4, 5], {1, 3, 5, 7}, 3) - 2 / 3) < 1e-10
    assert precision_at_k([1, 2, 3], {1, 2}, 0) == 0.0


def test_recall_at_k():
    assert abs(recall_at_k([1, 2, 3, 4, 5], {1, 3, 5, 7}, 3) - 0.5) < 1e-10
    assert recall_at_k([1, 2, 3], set(), 3) == 0.0


def test_ndcg_at_k_imperfect():
    dcg = 1.0 / math.log2(4) + 1.0 / math.log2(6)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert abs(ndcg_at_k([1, 2, 3, 4, 5], {3, 5}, 5) - dcg / idcg) < 1e-10


def test_map():
    expected = (1.0 + 2 / 3 + 3 / 5) / 3
    assert abs(mean_average_precision([1, 2, 3, 4, 5], {1, 3, 5}) - expected) < 1e-10


def test_coverage():
    assert abs(coverage([[0, 1, 2], [2, 3, 4], [4, 5]], 10) - 0.6) < 1e-10
    assert coverage([[1, 2]], 0) == 0.0


def test_hit_rate():
    assert hit_rate_at_k([10, 20, 30], {20, 40}, 3) == 1.0
    assert hit_rate_at_k([10, 20, 30, 40], {40}, 2) == 0.0
