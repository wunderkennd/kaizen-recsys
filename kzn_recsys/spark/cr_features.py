"""FEASE side features for the content-recommendation (CR) pipelines.

Builds the long-format ``(user_id | item_id, feature_name, value)`` tables
from the gold viewership table and the content metadata. Shared by
``kzn_recsys/fease_train.py`` and the bundle's ingest notebook
(``databricks/src/01_ingest_and_features.py``, ``data_mode=agg``), so the
model the bundle evaluates is fed exactly the features the training
notebook uses.

Feature names follow the ``<column>_<value>`` one-hot convention (not the
``<column>=<value>`` one of ``kzn_recsys.spark.databricks``); the
availability territory loader accepts both.
"""
from __future__ import annotations

from functools import reduce

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

from kzn_recsys import cr_config as cfg
from kzn_recsys.spark.interactions_agg import active_users


def _empty(spark, key_out: str) -> DataFrame:
    return spark.createDataFrame([], StructType([
        StructField(key_out, StringType(), False),
        StructField("feature_name", StringType(), False),
        StructField("value", DoubleType(), False),
    ]))


def to_long_format(df: DataFrame, id_col: str, feature_cols, key_out: str) -> DataFrame:
    """Wide categorical columns -> distinct ``(key_out, "<col>_<value>", 1.0)``
    rows; null values produce no row."""
    parts = [
        df.where(F.col(c).isNotNull()).select(
            F.col(id_col).cast("string").alias(key_out),
            F.concat(F.lit(f"{c}_"), F.col(c)).alias("feature_name"),
            F.lit(1.0).alias("value"),
        )
        for c in feature_cols
    ]
    if not parts:
        return _empty(df.sparkSession, key_out)
    return reduce(DataFrame.unionByName, parts).distinct()


def tenure_bucket(col):
    """Account tenure in days -> ``tenure_{unknown,0d,7d,30d,90d,90d+}``."""
    return (
        F.when(col.isNull(), F.lit("tenure_unknown"))
        .when(col <= 0, F.lit("tenure_0d"))
        .when(col <= 7, F.lit("tenure_7d"))
        .when(col <= 30, F.lit("tenure_30d"))
        .when(col <= 90, F.lit("tenure_90d"))
        .otherwise(F.lit("tenure_90d+"))
    )


def cr_user_features(
    engagement_df: DataFrame,
    interactions_daily_df: DataFrame,
    *,
    user_col: str = cfg.VIEWERSHIP_USER_COL,
    ts_col: str = cfg.VIEWERSHIP_TS_COL,
    categorical_cols=cfg.CR_USER_CATEGORICAL_FEATURES,
    tenure_col: str | None = cfg.CR_USER_TENURE_COL,
    subsidiary_col: str | None = cfg.VIEWERSHIP_SUBSIDIARY_COL,
    subsidiary: str | None = cfg.VIEWERSHIP_SUBSIDIARY,
) -> DataFrame:
    """User features from each user's most recent engagement row.

    Only users present in ``interactions_daily_df`` are kept, so the user
    mapping matches the interactions. The latest row is picked with
    ``row_number()`` over ``ts_col`` descending, ties broken by the feature
    values themselves, so the pick is a pure function of the data.
    """
    feature_cols = [*categorical_cols, *([tenure_col] if tenure_col else [])]
    df = engagement_df
    if subsidiary_col is not None and subsidiary is not None:
        df = df.where(F.col(subsidiary_col) == F.lit(subsidiary))
    base = (
        df.where(F.nullif(F.trim(F.col(user_col)), F.lit("")).isNotNull())
        .join(
            active_users(interactions_daily_df).withColumnRenamed("user_id", "_active_uid"),
            F.col(user_col) == F.col("_active_uid"),
            "inner",
        )
        .drop("_active_uid")
        .select(user_col, ts_col, *feature_cols)
        .withColumn(
            "_rn",
            F.row_number().over(
                Window.partitionBy(user_col).orderBy(
                    F.col(ts_col).desc_nulls_last(),
                    *[F.col(c).desc_nulls_last() for c in feature_cols],
                )
            ),
        )
        .where(F.col("_rn") == 1)
        .drop("_rn")
    )
    out = to_long_format(base, user_col, list(categorical_cols), "user_id")
    if tenure_col:
        out = out.unionByName(base.select(
            F.col(user_col).cast("string").alias("user_id"),
            tenure_bucket(F.col(tenure_col)).alias("feature_name"),
            F.lit(1.0).alias("value"),
        ))
    return out.distinct()


def cr_item_features(
    metadata_df: DataFrame,
    *,
    series_col: str = cfg.METADATA_SERIES_COL,
    categorical_cols=cfg.CR_ITEM_CATEGORICAL_FEATURES,
    multi_value_cols=cfg.CR_ITEM_MULTI_VALUE_FEATURES,
) -> DataFrame:
    """Item features from media-grain metadata, keyed by the series id.

    Every feature lands on ``series_col`` so it matches the series-grain
    interaction items; the trailing ``distinct()`` collapses the episodes of
    a series into one row per feature. ``multi_value_cols`` are
    ``(column, prefix)`` pairs of comma-separated values, each value trimmed.
    """
    meta = metadata_df.where(F.col(series_col).isNotNull())
    parts = [to_long_format(meta, series_col, list(categorical_cols), "item_id")]
    for col, prefix in multi_value_cols:
        parts.append(
            meta.select(
                F.col(series_col).cast("string").alias("item_id"),
                F.explode(F.split(F.col(col), ",")).alias("_val"),
            ).select(
                "item_id",
                F.concat(F.lit(prefix), F.trim(F.col("_val"))).alias("feature_name"),
                F.lit(1.0).alias("value"),
            )
        )
    return (
        reduce(DataFrame.unionByName, parts)
        .where(F.col("feature_name").isNotNull() & (F.col("feature_name") != F.lit("")))
        .distinct()
    )
