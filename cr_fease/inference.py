"""Inference utilities for loading and serving a trained FEASE model."""

from typing import Optional
import rust_fease_recommender as fease


def load_and_predict(
    model_path: str,
    interactions: dict[str, float],
    features: dict[str, float],
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """
    Loads a saved model and makes a single prediction.

    Args:
        model_path: Path to the saved .fease model file.
        interactions: Dict mapping item_guid -> interaction value.
        features: Dict mapping feature_name -> value.
        top_k: Number of recommendations to return.

    Returns:
        List of (item_guid, score) tuples sorted descending by score.
    """
    model = fease.load_model(model_path)
    return list(model.predict(interactions, features, top_k=top_k))


class FeasePredictor:
    """Wraps a loaded FEASE model for repeated inference."""

    def __init__(self, model_path: str):
        self.model = fease.load_model(model_path)

    def predict(
        self,
        interactions: dict[str, float],
        features: dict[str, float],
        top_k: int = 100,
    ) -> list[tuple[str, float]]:
        return list(self.model.predict(interactions, features, top_k=top_k))

    def predict_batch(
        self,
        users: list[dict],
        top_k: int = 100,
    ) -> list[list[tuple[str, float]]]:
        return [list(recs) for recs in self.model.predict_batch(users, top_k=top_k)]

    def similar_items(
        self,
        item_guid: str,
        top_k: int = 20,
    ) -> list[tuple[str, float]]:
        return list(self.model.predict_similar_items(item_guid, top_k=top_k))

    @property
    def num_items(self) -> int:
        return self.model.num_items

    @property
    def num_user_features(self) -> int:
        return self.model.num_user_features

    @property
    def num_item_features(self) -> int:
        return self.model.num_item_features
