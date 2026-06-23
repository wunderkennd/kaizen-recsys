"""Pure NumPy/SciPy EASE math. Mirrors src/model.rs.

No pyspark import here — this module is Spark-free and fast to test.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EaseParams:
    """EASE hyperparameters. Defaults match the Rust core / fease_wrapper."""
    alpha: float = 1.0       # item-feature weight
    beta: float = 1.0        # user-feature weight
    lambda_: float = 150.0   # L2 regularization
    meta_weight: float = 0.0  # diagonal metadata weighting; 0 => treated as 1.0
