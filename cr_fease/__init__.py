"""cr_fease — Python wrapper for the Rust FEASE recommender."""

from rust_fease_recommender import (
    FeaseModel,
    FeaseRegistry,
    build_and_train,
    load_model,
    validate_data,
)
from cr_fease.schemas import EngagementSchema, MetadataSchema

__all__ = [
    "FeaseModel",
    "FeaseRegistry",
    "build_and_train",
    "load_model",
    "validate_data",
    "EngagementSchema",
    "MetadataSchema",
]
