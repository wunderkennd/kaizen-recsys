"""Reference recommenders to read model metrics against.

A ranking metric on its own says little ("NDCG@10 = 0.08"); the lift over a
most-popular ranking on the same split says whether the model learned
anything beyond popularity.
"""
from __future__ import annotations

import numpy as np

from .model import SparkEaseModel


class PopularityBaseline:
    """Recommend the catalog by descending train popularity (distinct users
    per item), skipping each user's train items; ties go to catalog order.

    The catalog is a trained model's ``mappings``, and ``evaluate`` is
    ``SparkEaseModel.evaluate`` itself, so the report covers the same users,
    relevant sets, availability filter and seen-item exclusion as the
    model's and the two are directly comparable.
    """

    def __init__(self, mappings, item_popularity: dict):
        self.mappings = mappings
        self.num_items = len(mappings.idx_to_item)
        scores = np.zeros(self.num_items, dtype=np.float64)
        for item, count in item_popularity.items():
            j = mappings.item_to_idx.get(item)
            if j is not None:
                scores[j] = float(count)
        self._scores = scores
        self._order = np.argsort(-scores, kind="stable")

    @classmethod
    def fit(cls, train_interactions_df, mappings) -> "PopularityBaseline":
        from pyspark.sql import functions as F
        counts = (train_interactions_df.groupBy("item_id")
                  .agg(F.countDistinct("user_id").alias("n")).collect())
        return cls(mappings, {r["item_id"]: r["n"] for r in counts})

    def _rank(self, interactions: dict, top_k: int, eligible=None):
        m = self.mappings
        seen = {m.item_to_idx[k] for k in interactions if k in m.item_to_idx}
        out = []
        for j in self._order:
            j = int(j)
            if j in seen or (eligible is not None and m.idx_to_item[j] not in eligible):
                continue
            out.append((m.idx_to_item[j], float(self._scores[j])))
            if len(out) == top_k:
                break
        return out

    def predict(self, interactions: dict, features: dict, top_k: int):
        return self._rank(interactions, top_k)

    def _predict_within(self, interactions: dict, features: dict, top_k: int, eligible):
        return self._rank(interactions, top_k, eligible)

    evaluate = SparkEaseModel.evaluate
