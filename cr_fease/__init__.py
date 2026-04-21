"""cr_fease — Python wrapper for the Rust FEASE recommender."""

from rust_fease_recommender import (
    FeaseModel,
    FeaseRegistry,
    build_and_train,
    coverage,
    grid_search_py as grid_search,
    hit_rate_at_k,
    load_model,
    mean_average_precision,
    ndcg_at_k,
    precision_at_k,
    random_search_py as random_search,
    recall_at_k,
    validate_data,
    random_split,
    temporal_split,
    leave_k_out_split,
)
from cr_fease.schemas import EngagementSchema, MetadataSchema

__all__ = [
    "FeaseModel",
    "FeaseRegistry",
    "build_and_train",
    "coverage",
    "grid_search",
    "hit_rate_at_k",
    "load_model",
    "mean_average_precision",
    "ndcg_at_k",
    "precision_at_k",
    "random_search",
    "recall_at_k",
    "validate_data",
    "random_split",
    "temporal_split",
    "leave_k_out_split",
    "EngagementSchema",
    "MetadataSchema",
]
