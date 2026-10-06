"""Databricks helpers for the pure-Python Spark EASE pipeline.

Imports nothing native (no kzn_recsys._native, pydantic, or polars): this
module ships in the pure-Python wheel and runs on stock Databricks Runtime.
"""
from __future__ import annotations

import random
from functools import reduce

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
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
                   n_cold_users: int = 0,
                   seed: int = 42) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Generate deterministic synthetic long-format tables with persona structure.

    Each persona prefers a contiguous slice of the catalog, so EASE has real
    co-occurrence signal to learn. The persona slices tile the whole catalog,
    so every item can receive interactions even when ``n_items`` is not a
    multiple of ``n_personas``. ``n_cold_users`` appends feature-only users
    (a persona feature, no interactions) to exercise the cold-start path.
    All randomness is seeded in Python and handed to spark.createDataFrame
    for determinism.
    """
    rng = random.Random(seed)

    # Assign each item a genre and each persona a preferred item slice.
    item_ids = [f"item_{i:04d}" for i in range(n_items)]
    item_genre = {iid: _GENRES[i % len(_GENRES)] for i, iid in enumerate(item_ids)}

    def _pool(persona: int) -> list:
        lo = persona * n_items // n_personas
        hi = (persona + 1) * n_items // n_personas
        return item_ids[lo:hi] or item_ids

    inter_rows, user_rows = [], []
    for u in range(n_users):
        uid = f"user_{u:05d}"
        persona = u % n_personas
        user_rows.append((uid, f"persona={persona}", 1.0))
        pool = _pool(persona)
        k = min(len(pool), max(1, avg_interactions))
        for iid in rng.sample(pool, k):
            inter_rows.append((uid, iid, 1.0))
    for c in range(n_cold_users):
        uid = f"user_{n_users + c:05d}"
        user_rows.append((uid, f"persona={c % n_personas}", 1.0))

    item_rows = [(iid, f"genre={item_genre[iid]}", 1.0) for iid in item_ids]

    interactions = spark.createDataFrame(inter_rows, _INTERACTIONS_SCHEMA)
    users = spark.createDataFrame(user_rows, _FEATURES_SCHEMA_USER)
    items = spark.createDataFrame(item_rows, _FEATURES_SCHEMA_ITEM)
    return interactions, users, items


def _one_hot_long(df: DataFrame, key_col: str, feature_cols, key_out: str) -> DataFrame:
    """Melt categorical columns into (key_out, feature_name, value=1.0) long rows."""
    parts = []
    for col in feature_cols:
        parts.append(
            df.select(
                F.col(key_col).cast("string").alias(key_out),
                F.concat(F.lit(f"{col}="), F.col(col).cast("string")).alias("feature_name"),
            ).where(F.col("feature_name").isNotNull())
        )
    if not parts:
        empty_schema = StructType([
            StructField(key_out, StringType(), False),
            StructField("feature_name", StringType(), False),
            StructField("value", DoubleType(), False),
        ])
        return df.sparkSession.createDataFrame([], empty_schema)
    melted = reduce(DataFrame.unionByName, parts)
    return (melted.withColumn("value", F.lit(1.0))
                  .dropDuplicates([key_out, "feature_name"]))


def feature_engineering(engagement_df: DataFrame, metadata_df: DataFrame, *,
                        user_col: str = "user_id", item_col: str = "item_id",
                        value_col: str = "value",
                        event_type_col: str | None = None,
                        days_ago_col: str | None = None,
                        user_feature_cols=(), item_feature_cols=(),
                        item_key_col: str = "item_id"):
    """Map raw engagement + metadata tables into the three long-format tables."""
    base_cols = [
        F.col(user_col).cast("string").alias("user_id"),
        F.col(item_col).cast("string").alias("item_id"),
        F.col(value_col).cast("double").alias("value"),
    ]
    optional = []
    if event_type_col is not None:
        optional.append(F.col(event_type_col).cast("string").alias("event_type"))
    if days_ago_col is not None:
        optional.append(F.col(days_ago_col).cast("double").alias("days_ago"))

    if optional:
        # preserve per-event rows for weighting
        interactions = engagement_df.select(*base_cols, *optional)
    else:
        interactions = (engagement_df.select(*base_cols)
                        .groupBy("user_id", "item_id")
                        .agg(F.sum("value").alias("value")))

    user_features = _one_hot_long(engagement_df, user_col, list(user_feature_cols), "user_id")
    item_features = _one_hot_long(metadata_df, item_key_col, list(item_feature_cols), "item_id")
    return interactions, user_features, item_features
