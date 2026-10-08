"""Pure-Python / PySpark EASE implementation.

Imports nothing from kzn_recsys._native, so it works in environments where
the compiled extension cannot be installed.
"""
from kzn_recsys.spark.ease_core import EaseParams
from kzn_recsys.spark.feas_codec import WeightingConfig
from kzn_recsys.spark.model import SparkEaseModel, build_and_train, load_model
from kzn_recsys.spark.tuning import grid_search, random_search
from kzn_recsys.spark.splits import (
    random_split,
    temporal_split,
    leave_k_out_split,
    leave_last_k_out_split,
)
from kzn_recsys.spark.availability import AvailabilityTable
from kzn_recsys.spark.interactions_agg import (
    aggregate_viewership_daily,
    with_days_ago,
    filter_activity_window,
    daily_to_pairs,
    active_users,
)
from kzn_recsys.spark.metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    mean_average_precision,
    coverage,
    hit_rate_at_k,
)
from kzn_recsys.spark.databricks import make_synthetic, feature_engineering

__all__ = [
    "EaseParams",
    "WeightingConfig",
    "SparkEaseModel",
    "build_and_train",
    "load_model",
    "grid_search",
    "random_search",
    "random_split",
    "temporal_split",
    "leave_k_out_split",
    "leave_last_k_out_split",
    "AvailabilityTable",
    "aggregate_viewership_daily",
    "with_days_ago",
    "filter_activity_window",
    "daily_to_pairs",
    "active_users",
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "mean_average_precision",
    "coverage",
    "hit_rate_at_k",
    "make_synthetic",
    "feature_engineering",
]
