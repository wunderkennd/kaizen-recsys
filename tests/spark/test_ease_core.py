import numpy as np
import pytest

from kzn_recsys.spark.ease_core import EaseParams


def test_ease_params_defaults():
    p = EaseParams()
    assert p.alpha == 1.0
    assert p.beta == 1.0
    assert p.lambda_ == 150.0
    assert p.meta_weight == 0.0


def test_ease_params_explicit():
    p = EaseParams(alpha=2.0, beta=0.5, lambda_=100.0, meta_weight=1.0)
    assert (p.alpha, p.beta, p.lambda_, p.meta_weight) == (2.0, 0.5, 100.0, 1.0)
