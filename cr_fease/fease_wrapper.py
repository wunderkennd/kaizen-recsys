"""High-level Python wrapper with optional schema validation."""

from __future__ import annotations

from typing import Dict, Optional

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
    decay_rate: float = 0.0,
    ips_alpha: float = 0.0,
    sparsity_threshold: float = 0.0,
    event_weights: Optional[Dict[str, float]] = None,
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
        decay_rate: Exponential temporal decay rate (0.0 = no decay).
            Requires ``days_ago`` column in the interactions file.
        ips_alpha: Inverse propensity scoring strength (0.0 = disabled).
        sparsity_threshold: Prune S-matrix entries below this value (0.0 = no pruning).
        event_weights: Dict mapping event type to weight multiplier (None = disabled).
            Requires ``event_type`` column in the interactions file.

    Returns:
        Trained FeaseModel ready for predictions.
    """
    kwargs: dict = {}
    if decay_rate > 0.0:
        kwargs["decay_rate"] = decay_rate
    if ips_alpha > 0.0:
        kwargs["ips_alpha"] = ips_alpha
    if sparsity_threshold > 0.0:
        kwargs["sparsity_threshold"] = sparsity_threshold
    if event_weights is not None:
        kwargs["event_weights"] = event_weights

    return fease.build_and_train(
        interactions_path=interactions_path,
        user_features_path=user_features_path,
        item_features_path=item_features_path,
        alpha=alpha,
        beta=beta,
        lambda_=lambda_,
        meta_weight=meta_weight,
        **kwargs,
    )
