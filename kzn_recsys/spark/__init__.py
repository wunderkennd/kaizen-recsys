"""Pure-Python / PySpark EASE implementation.

Imports nothing from kzn_recsys._native, so it works in environments where
the compiled extension cannot be installed.
"""
from kzn_recsys.spark.ease_core import EaseParams
from kzn_recsys.spark.feas_codec import WeightingConfig
from kzn_recsys.spark.model import SparkEaseModel, build_and_train, load_model

__all__ = [
    "EaseParams",
    "WeightingConfig",
    "SparkEaseModel",
    "build_and_train",
    "load_model",
]
