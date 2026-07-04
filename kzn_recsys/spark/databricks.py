"""Databricks helpers for the pure-Python Spark EASE pipeline.

Imports nothing native (no kzn_recsys._native, pydantic, or polars): this
module ships in the pure-Python wheel and runs on stock Databricks Runtime.
"""
from __future__ import annotations

import random

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    DoubleType, StringType, StructField, StructType,
)

_INTERACTIONS_SCHEMA = StructType([
    StructField("user_id", StringType(), False),
    StructField("item_id", StringType(), False),
    StructField("value", DoubleType(), False),
])
_FEATURES_SCHEMA_USER = StructType([
    StructField("user_id", StringType(), False),
    StructField("feature_name", StringType(), False),
    StructField("value", DoubleType(), False),
])
_FEATURES_SCHEMA_ITEM = StructType([
    StructField("item_id", StringType(), False),
    StructField("feature_name", StringType(), False),
    StructField("value", DoubleType(), False),
])

_GENRES = ["action", "drama", "comedy", "romance", "thriller", "slice_of_life"]


def make_synthetic(spark: SparkSession, *, n_users: int = 200, n_items: int = 60,
                   n_personas: int = 4, avg_interactions: int = 15,
                   seed: int = 42) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Generate deterministic synthetic long-format tables with persona structure.

    Each persona prefers a contiguous slice of the catalog, so EASE has real
    co-occurrence signal to learn. All randomness is seeded in Python and
    handed to spark.createDataFrame for determinism.
    """
    rng = random.Random(seed)

    # Assign each item a genre and each persona a preferred item slice.
    item_ids = [f"item_{i:04d}" for i in range(n_items)]
    item_genre = {iid: _GENRES[i % len(_GENRES)] for i, iid in enumerate(item_ids)}
    slice_size = max(1, n_items // n_personas)

    inter_rows, user_rows = [], []
    for u in range(n_users):
        uid = f"user_{u:05d}"
        persona = u % n_personas
        user_rows.append((uid, f"persona={persona}", 1.0))
        lo = persona * slice_size
        hi = min(n_items, lo + slice_size)
        pool = item_ids[lo:hi] or item_ids
        k = min(len(pool), max(1, avg_interactions))
        for iid in rng.sample(pool, k):
            inter_rows.append((uid, iid, 1.0))

    item_rows = [(iid, f"genre={item_genre[iid]}", 1.0) for iid in item_ids]

    interactions = spark.createDataFrame(inter_rows, _INTERACTIONS_SCHEMA)
    users = spark.createDataFrame(user_rows, _FEATURES_SCHEMA_USER)
    items = spark.createDataFrame(item_rows, _FEATURES_SCHEMA_ITEM)
    return interactions, users, items
