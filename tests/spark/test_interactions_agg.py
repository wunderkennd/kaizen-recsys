"""Series-grain interactions helpers (issue #102)."""
import datetime as dt
import math

import pytest

pytestmark = pytest.mark.spark

from kzn_recsys.spark.interactions_agg import (
    active_users,
    aggregate_viewership_daily,
    daily_to_pairs,
    filter_activity_window,
    with_days_ago,
)

AS_OF = dt.date(2026, 10, 7)


def _events(spark):
    """Raw viewership rows: (profile, series, timestamp, seconds, subsidiary)."""
    rows = [
        ("p1", "S1", "2026-10-01 08:00:00", 600.0, "crunchyroll"),   # day 6: two views of S1
        ("p1", "S1", "2026-10-01 21:00:00", 1200.0, "crunchyroll"),
        ("p1", "S1", "2026-10-05 09:00:00", 300.0, "crunchyroll"),   # day 2
        ("p1", "S2", "2026-10-06 09:00:00", 45.0, "crunchyroll"),    # day 1
        ("p1", "S3", "2026-10-06 10:00:00", 10.0, "crunchyroll"),    # below min watch -> dropped
        ("p2", "S1", "2026-09-30 12:00:00", 900.0, "crunchyroll"),   # day 7
        ("",   "S1", "2026-10-02 12:00:00", 900.0, "crunchyroll"),   # blank user -> dropped
        ("p3", None, "2026-10-02 12:00:00", 900.0, "crunchyroll"),   # null item -> dropped
        ("p4", "S1", "2026-10-02 12:00:00", 900.0, "funimation"),    # other subsidiary -> dropped
        ("p5", "S9", "2026-10-07 12:00:00", 900.0, "crunchyroll"),   # today -> excluded
    ]
    return spark.createDataFrame(
        rows, "user string, show string, ts string, secs double, sub string"
    ).withColumn("ts", __import__("pyspark.sql.functions", fromlist=["x"]).to_timestamp("ts"))


def _daily(spark):
    # The current date in Spark is the real today; pin the aggregation's
    # "today" by only asserting on rows that are unambiguously in the past.
    return aggregate_viewership_daily(
        _events(spark), user_col="user", item_col="show", date_col="ts",
        seconds_col="secs", min_watch_seconds=30.0,
        subsidiary_col="sub", subsidiary="crunchyroll", exclude_today=False,
    ).where("view_date < date'2026-10-07'")


def test_aggregate_viewership_daily_matches_the_sql_semantics(spark):
    daily = _daily(spark)
    rows = {(r["user_id"], r["item_id"], r["view_date"]): (r["value"], r["num_views"])
            for r in daily.collect()}
    assert set(rows) == {
        ("p1", "S1", dt.date(2026, 10, 1)),
        ("p1", "S1", dt.date(2026, 10, 5)),
        ("p1", "S2", dt.date(2026, 10, 6)),
        ("p2", "S1", dt.date(2026, 9, 30)),
    }
    v, n = rows[("p1", "S1", dt.date(2026, 10, 1))]
    assert n == 2
    assert v == pytest.approx(math.log(601.0) + math.log(1201.0))
    v, n = rows[("p1", "S2", dt.date(2026, 10, 6))]
    assert n == 1 and v == pytest.approx(math.log(46.0))
    assert daily.columns == ["user_id", "item_id", "view_date", "value", "num_views"]


def test_with_days_ago_and_activity_window(spark):
    daily = with_days_ago(_daily(spark), as_of=AS_OF)
    by_key = {(r["user_id"], r["item_id"], r["view_date"]): r["days_ago"] for r in daily.collect()}
    assert by_key[("p1", "S1", dt.date(2026, 10, 1))] == 6.0
    assert by_key[("p2", "S1", dt.date(2026, 9, 30))] == 7.0
    assert dict(daily.dtypes)["days_ago"] == "double"
    windowed = filter_activity_window(daily, 6)
    assert {r["user_id"] for r in windowed.collect()} == {"p1"}
    assert filter_activity_window(daily, None).count() == daily.count()


def test_as_of_backtest_never_sees_the_future(spark):
    # as_of before the latest views: those rows did not exist yet and must
    # be dropped, not kept with negative days_ago (which decay would amplify).
    daily = with_days_ago(_daily(spark), as_of=dt.date(2026, 10, 4))
    rows = {(r["user_id"], r["item_id"], r["view_date"]): r["days_ago"] for r in daily.collect()}
    assert set(rows) == {("p1", "S1", dt.date(2026, 10, 1)), ("p2", "S1", dt.date(2026, 9, 30))}
    assert min(rows.values()) >= 0.0
    # The window filter guards the lower bound too, for frames built elsewhere.
    from pyspark.sql import functions as F
    tampered = daily.withColumn("days_ago", F.col("days_ago") - F.lit(10.0))
    assert filter_activity_window(tampered, None).count() == 0
    assert filter_activity_window(tampered, 30).count() == 0


def test_daily_to_pairs_sums_per_day_with_decay_and_keeps_latest_day(spark):
    daily = with_days_ago(_daily(spark), as_of=AS_OF)

    plain = {(r["user_id"], r["item_id"]): (r["value"], r["days_ago"])
             for r in daily_to_pairs(daily).collect()}
    assert set(plain) == {("p1", "S1"), ("p1", "S2"), ("p2", "S1")}
    v, d = plain[("p1", "S1")]
    assert v == pytest.approx(math.log(601.0) + math.log(1201.0) + math.log(301.0))
    assert d == 2.0  # most recent day

    rate = 0.1
    decayed = {(r["user_id"], r["item_id"]): r["value"]
               for r in daily_to_pairs(daily, decay_rate=rate).collect()}
    expected = (
        (math.log(601.0) + math.log(1201.0)) * math.exp(-rate * 6)
        + math.log(301.0) * math.exp(-rate * 2)
    )
    assert decayed[("p1", "S1")] == pytest.approx(expected)
    assert daily_to_pairs(daily).columns == ["user_id", "item_id", "value", "days_ago"]

    with pytest.raises(ValueError, match="days_ago"):
        daily_to_pairs(_daily(spark))
    with pytest.raises(ValueError):
        daily_to_pairs(daily, decay_rate=-1.0)

    assert {r["user_id"] for r in active_users(daily).collect()} == {"p1", "p2"}


def test_pair_frame_trains_identically_to_daily_frame(spark):
    """EASE on the pre-summed pair frame == EASE on the daily rows: both the
    collect path (duplicates summed into CSR) and the distributed Gram
    (self-join multiplies out to the product of sums) agree, with decay
    applied per day in Spark vs per row in the backend."""
    import numpy as np
    from kzn_recsys.spark import WeightingConfig, build_and_train

    daily = with_days_ago(_daily(spark), as_of=AS_OF)
    rate = 0.05
    pairs = daily_to_pairs(daily, decay_rate=rate)
    daily_rows = daily.select("user_id", "item_id", "value", "days_ago")
    empty_u = spark.createDataFrame([], "user_id string, feature_name string, value double")
    empty_t = spark.createDataFrame([], "item_id string, feature_name string, value double")

    from_pairs = build_and_train(pairs, empty_u, empty_t, lambda_=5.0, strategy="collect")
    from_daily = build_and_train(
        daily_rows, empty_u, empty_t, lambda_=5.0, strategy="collect",
        weighting=WeightingConfig(decay_rate=rate),
    )
    from_daily_dist = build_and_train(
        daily_rows, empty_u, empty_t, lambda_=5.0, strategy="distributed",
        weighting=WeightingConfig(decay_rate=rate),
    )
    assert np.allclose(from_pairs.s_matrix, from_daily.s_matrix, atol=1e-9)
    assert np.allclose(from_pairs.s_matrix, from_daily_dist.s_matrix, atol=1e-9)
