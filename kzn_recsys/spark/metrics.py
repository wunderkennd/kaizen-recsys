"""Ranking metrics. Direct port of src/metrics.rs (binary relevance)."""
from __future__ import annotations

import math


def precision_at_k(recommended, relevant, k) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(recommended, relevant, k) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended, relevant, k) -> float:
    if k == 0 or not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 2.0)
              for rank, item in enumerate(recommended[:k]) if item in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 2.0) for rank in range(ideal_hits))
    return 0.0 if idcg == 0.0 else dcg / idcg


def mean_average_precision(recommended, relevant) -> float:
    if not relevant:
        return 0.0
    hits = 0
    sum_precision = 0.0
    for i, item in enumerate(recommended):
        if item in relevant:
            hits += 1
            sum_precision += hits / (i + 1)
    return sum_precision / len(relevant)


def coverage(all_recommendations, num_total_items) -> float:
    if num_total_items == 0:
        return 0.0
    unique = set()
    for recs in all_recommendations:
        unique.update(recs)
    return len(unique) / num_total_items


def hit_rate_at_k(recommended, relevant, k) -> float:
    if k == 0:
        return 0.0
    return 1.0 if any(item in relevant for item in recommended[:k]) else 0.0
