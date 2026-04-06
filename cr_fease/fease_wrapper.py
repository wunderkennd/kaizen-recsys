"""High-level Python wrapper with optional schema validation."""

import polars as pl
import rust_fease_recommender as fease
from cr_fease.schemas import EngagementSchema, MetadataSchema


def build_and_train_safe(
    interactions_path: str,
    user_features_path: str,
    item_features_path: str,
    alpha: float = 1.0,
    beta: float = 1.0,
    lambda_: float = 100.0,
    meta_weight: float = 0.0,
) -> fease.FeaseModel:
    """
    Builds and trains a FEASE model from three long-format Parquet/CSV files.

    Args:
        interactions_path: Path to interactions file (user_id, item_id, value).
        user_features_path: Path to user features file (user_id, feature_name, value).
        item_features_path: Path to item features file (item_id, feature_name, value).
        alpha: Weight for item features in the Gram matrix.
        beta: Weight for user features in the Gram matrix.
        lambda_: L2 regularization term.
        meta_weight: Weight for metadata rows (0.0 = equal weighting).

    Returns:
        Trained FeaseModel ready for predictions.
    """
    return fease.build_and_train(
        interactions_path=interactions_path,
        user_features_path=user_features_path,
        item_features_path=item_features_path,
        alpha=alpha,
        beta=beta,
        lambda_=lambda_,
        meta_weight=meta_weight,
    )
