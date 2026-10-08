"""Read-side helpers for the series-grain interactions table (issue #102).

The table ``fease_interactions_agg`` holds one row per
``(user_id, item_id, view_date)`` with ``value = sum(ln(seconds + 1))`` over
that day's qualifying views and ``num_views``. It is built and refreshed by
``databricks/sql/*.sql`` (via ``databricks/src/00_refresh_interactions_agg.py``);
``aggregate_viewership_daily`` is the PySpark equivalent of that aggregation
for tests, synthetic data, and ad-hoc windows.

Everything time-related happens at read time:

- ``with_days_ago`` derives ``days_ago`` from ``view_date`` (relative to
  today or to an explicit ``as_of`` date for reproducible backtests), so the
  table never bakes in a reference date.
- ``daily_to_pairs`` collapses the daily grain to one row per
  ``(user_id, item_id)`` for the EASE backends, applying temporal decay per
  *day* before summing (``value = Σ_d value_d · exp(-decay_rate · days_ago_d)``)
  and keeping ``days_ago`` of the most recent day. Both the Rust pipeline
  (``data_pipeline.rs``: decay per triplet, then duplicates summed into the
  CSR matrix) and the Spark distributed Gram (self-join on user: duplicates
  multiply out to the product of sums) would compute the same numbers from
  the daily rows; pre-summing just avoids shipping redundant rows to the
  single-threaded Parquet reader and the quadratic self-join. Because decay
  is already applied, pass ``decay_rate=0`` to whichever backend consumes
  the pair frame.

Sequence models (SASRec, BERT4Rec) need per-event ``days_ago`` and should
consume the daily frame directly.
"""
from __future__ import annotations

from typing import Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

DAILY_COLUMNS = ["user_id", "item_id", "view_date", "value", "num_views"]
PAIR_COLUMNS = ["user_id", "item_id", "value", "days_ago"]


def aggregate_viewership_daily(
    viewership_df: DataFrame,
    *,
    user_col: str,
    item_col: str,
    date_col: str,
    seconds_col: str,
    min_watch_seconds: float = 30.0,
    subsidiary_col: Optional[str] = None,
    subsidiary: Optional[str] = None,
    exclude_today: bool = True,
) -> DataFrame:
    """Event rows -> ``(user_id, item_id, view_date, value, num_views)``.

    Mirrors ``databricks/sql/create_fease_interactions_agg.sql``: drops rows
    with a blank user id, a null item id, or fewer than ``min_watch_seconds``
    seconds watched; ``value`` is ``sum(ln(seconds + 1))``. ``date_col`` may be
    a DATE or a TIMESTAMP (it is cast to a date). ``exclude_today`` drops the
    still-incomplete current day, as the SQL does.
    """
    df = viewership_df
    if subsidiary_col is not None and subsidiary is not None:
        df = df.where(F.col(subsidiary_col) == F.lit(subsidiary))
    df = (
        df.where(F.nullif(F.trim(F.col(user_col)), F.lit("")).isNotNull())
        .where(F.col(item_col).isNotNull())
        .where(F.col(seconds_col) >= F.lit(float(min_watch_seconds)))
        .withColumn("view_date", F.to_date(F.col(date_col)))
    )
    if exclude_today:
        df = df.where(F.col("view_date") < F.current_date())
    return (
        df.groupBy(
            F.col(user_col).cast("string").alias("user_id"),
            F.col(item_col).cast("string").alias("item_id"),
            "view_date",
        )
        .agg(
            F.sum(F.log(F.col(seconds_col) + F.lit(1.0))).alias("value"),
            F.count(F.lit(1)).alias("num_views"),
        )
        .select(*DAILY_COLUMNS)
    )


def with_days_ago(daily_df: DataFrame, *, as_of=None) -> DataFrame:
    """Add ``days_ago`` (double) = days between ``view_date`` and ``as_of``
    (default: today). Pass a fixed ``as_of`` (``datetime.date`` or ISO
    string) to make a backtest reproducible: rows dated *after* ``as_of``
    did not exist at that point in time and are dropped, so a backtest
    never sees the future (or decays it upward)."""
    ref = F.current_date() if as_of is None else F.to_date(F.lit(str(as_of)))
    return (
        daily_df.withColumn("days_ago", F.datediff(ref, F.col("view_date")).cast("double"))
        .where(F.col("days_ago") >= F.lit(0.0))
    )


def filter_activity_window(daily_df: DataFrame, window_days: Optional[int]) -> DataFrame:
    """Keep rows with ``0 <= days_ago <= window_days``; ``None`` keeps
    everything with ``days_ago >= 0`` (the lower bound is a belt-and-braces
    guard for frames whose ``days_ago`` did not come from ``with_days_ago``)."""
    df = daily_df.where(F.col("days_ago") >= F.lit(0.0))
    if window_days is None:
        return df
    return df.where(F.col("days_ago") <= F.lit(float(window_days)))


def daily_to_pairs(daily_df: DataFrame, *, decay_rate: float = 0.0) -> DataFrame:
    """Daily grain -> one row per ``(user_id, item_id)`` for the EASE backends.

    ``value`` is the per-day value summed after temporal decay
    (``exp(-decay_rate * days_ago)``; ``decay_rate=0`` is a plain sum) and
    ``days_ago`` is the most recent day's. Requires ``with_days_ago`` first.
    """
    if decay_rate < 0.0:
        raise ValueError(f"decay_rate must be >= 0, got {decay_rate}")
    if "days_ago" not in daily_df.columns:
        raise ValueError("daily_to_pairs needs a days_ago column; call with_days_ago first")
    weighted = F.col("value")
    if decay_rate > 0.0:
        weighted = weighted * F.exp(F.lit(-float(decay_rate)) * F.col("days_ago"))
    return (
        daily_df.groupBy("user_id", "item_id")
        .agg(F.sum(weighted).alias("value"), F.min("days_ago").alias("days_ago"))
        .select(*PAIR_COLUMNS)
    )


def active_users(daily_df: DataFrame) -> DataFrame:
    """Distinct ``user_id`` column, for restricting user-feature builds to
    users that actually have interactions."""
    return daily_df.select("user_id").distinct()
