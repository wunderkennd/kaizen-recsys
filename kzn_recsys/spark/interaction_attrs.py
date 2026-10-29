"""Read-side helpers for the per-interaction and per-series attribute tables.

Two sibling tables of ``fease_interactions_agg`` (issue #102) carry the
attributes a factored-input sequence model will embed, without touching the
agg table, its MERGE, or its column contract:

- ``fease_interaction_attrs``: one row per ``(user_id, item_id, view_date)``,
  the **same key set** as the agg table (same filters, same
  ``min_watch_seconds``, today excluded), with ``seconds_watched``,
  ``episodes_watched``, ``completed_episodes``, ``max_depth_ratio``,
  ``platform_top``, ``dub_views``, ``sub_views`` and ``first_view_ts``.
- ``fease_item_attrs``: one row per series with ``premiere_date`` (+
  ``premiere_source``), ``episode_count``, ``avg_episode_seconds`` and
  ``first_air_year``.

Both are built by ``databricks/sql/*_attrs.sql`` (via
``databricks/src/00_refresh_interactions_agg.py``);
``aggregate_viewership_attrs_daily`` is the PySpark twin of the per-interaction
aggregation for tests and the ``"raw"`` source.

As with ``days_ago``, nothing time-relative or thresholded is stored: the
``with_*`` helpers derive every bucket at read time, and the discovery lag is
computed from ``view_date`` (never from an ``as_of`` date), so it is
backtest-safe by construction.
"""
from __future__ import annotations

from typing import Optional

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from kzn_recsys import cr_config as cfg

KEY_COLUMNS = ["user_id", "item_id", "view_date"]
ATTRS_COLUMNS = [
    *KEY_COLUMNS,
    "seconds_watched",
    "episodes_watched",
    "completed_episodes",
    "max_depth_ratio",
    "platform_top",
    "dub_views",
    "sub_views",
    "first_view_ts",
]
ITEM_ATTRS_COLUMNS = [
    "item_id",
    "premiere_date",
    "premiere_source",
    "episode_count",
    "avg_episode_seconds",
    "first_air_year",
]
DEPTH_BUCKETS = ("sampled", "partial", "completed", "unknown")
BINGE_BUCKETS = ("1", "2-3", "4-6", "7+", "unknown")
DUB_PREF_BUCKETS = ("sub", "mixed", "dub")


# --- Platform normalization -------------------------------------------------

def normalize_platform(col) -> "F.Column":
    """Raw platform value -> ``tv`` / ``mobile`` / ``web`` / ``other``.

    Lower-cases and trims, then matches substrings in
    ``cr_config.PLATFORM_CLASS_PATTERNS`` order; null or unmatched values are
    ``other``. ``platform_case_sql`` renders the same rule as SQL.
    """
    raw = F.lower(F.trim(F.col(col) if isinstance(col, str) else col))
    expr = None
    for cls, patterns in cfg.PLATFORM_CLASS_PATTERNS:
        cond = None
        for p in patterns:
            c = raw.contains(p)
            cond = c if cond is None else (cond | c)
        expr = F.when(cond, F.lit(cls)) if expr is None else expr.when(cond, F.lit(cls))
    return expr.otherwise(F.lit(cfg.PLATFORM_OTHER))


def platform_case_sql(column_sql: str) -> str:
    """SQL ``CASE`` for ``normalize_platform`` over an already-quoted column
    expression (e.g. ``"`view_platform_name`"``). Patterns come from
    ``cr_config`` and are plain ``[a-z0-9_ ]`` strings, so they are safe to
    splice into a quoted literal; anything else is rejected."""
    import re

    raw = f"LOWER(TRIM({column_sql}))"
    whens = []
    for cls, patterns in cfg.PLATFORM_CLASS_PATTERNS:
        for p in (cls, *patterns):
            if not re.fullmatch(r"[a-z0-9_ ]+", p):
                raise ValueError(f"platform pattern {p!r} must match [a-z0-9_ ]+")
        cond = " OR ".join(f"INSTR({raw}, '{p}') > 0" for p in patterns)
        whens.append(f"WHEN {cond} THEN '{cls}'")
    return f"CASE {' '.join(whens)} ELSE '{cfg.PLATFORM_OTHER}' END"


def _tie_rank(platform_col) -> "F.Column":
    order = list(cfg.PLATFORM_TIE_ORDER)
    expr = F.when(platform_col == F.lit(order[0]), F.lit(0))
    for i, cls in enumerate(order[1:], start=1):
        expr = expr.when(platform_col == F.lit(cls), F.lit(i))
    return expr.otherwise(F.lit(len(order)))


# --- Aggregation (PySpark twin of create_fease_interaction_attrs.sql) -------

def aggregate_viewership_attrs_daily(
    viewership_df: DataFrame,
    *,
    user_col: str,
    item_col: str,
    date_col: str,
    ts_col: str,
    seconds_col: str,
    media_id_col: Optional[str],
    media_duration_col: Optional[str],
    platform_col: Optional[str],
    audio_col: Optional[str],
    min_watch_seconds: float = 30.0,
    subsidiary_col: Optional[str] = None,
    subsidiary: Optional[str] = None,
    completed_ratio: float = 0.9,
    duration_to_seconds: float = 1.0,
    sub_audio_locale: str = cfg.SUB_AUDIO_LOCALE,
    exclude_today: bool = True,
) -> DataFrame:
    """Event rows -> one ``fease_interaction_attrs`` row per
    ``(user_id, item_id, view_date)``.

    Filters in the same order as ``aggregate_viewership_daily`` (subsidiary,
    blank user, null item, ``min_watch_seconds``, today) so the two frames
    have identical key sets. ``media_duration_col`` / ``audio_col`` /
    ``media_id_col`` / ``platform_col`` may be ``None`` when the source lacks
    them: the dependent columns become null (``platform_top`` becomes
    ``other``), mirroring a blank widget in the refresh notebook.
    """
    if not 0.0 < completed_ratio <= 1.0:
        raise ValueError(f"completed_ratio must be in (0, 1], got {completed_ratio}")
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

    def _or_null(col, dtype):
        return F.col(col).cast(dtype) if col is not None else F.lit(None).cast(dtype)

    secs = F.col(seconds_col).cast("double")
    duration = _or_null(media_duration_col, "double") * F.lit(float(duration_to_seconds))
    ratio = F.when(duration > 0, secs / duration)
    audio = _or_null(audio_col, "string")
    platform = (
        normalize_platform(platform_col) if platform_col is not None else F.lit(cfg.PLATFORM_OTHER)
    )

    views = df.select(
        F.col(user_col).cast("string").alias("user_id"),
        F.col(item_col).cast("string").alias("item_id"),
        "view_date",
        F.col(ts_col).cast("timestamp").alias("view_ts"),
        secs.alias("seconds"),
        _or_null(media_id_col, "string").alias("media_id"),
        duration.alias("duration"),
        ratio.alias("ratio"),
        platform.alias("platform"),
        audio.alias("audio"),
    )

    per_day = views.groupBy(*KEY_COLUMNS).agg(
        F.sum("seconds").alias("seconds_watched"),
        F.countDistinct("media_id").alias("episodes_watched"),
        F.when(
            F.max("duration") > 0,
            F.sum(F.when(F.col("ratio") >= F.lit(float(completed_ratio)), 1).otherwise(0)),
        ).cast("bigint").alias("completed_episodes"),
        # LEAST skips nulls, so guard it: an unknown ratio must stay null.
        F.max(F.when(F.col("ratio").isNotNull(), F.least(F.col("ratio"), F.lit(1.0))))
        .alias("max_depth_ratio"),
        F.sum(
            F.when(F.col("audio").isNotNull() & (F.col("audio") != F.lit(sub_audio_locale)), 1)
            .otherwise(0)
        ).cast("bigint").alias("dub_views"),
        F.sum(F.when(F.col("audio") == F.lit(sub_audio_locale), 1).otherwise(0))
        .cast("bigint").alias("sub_views"),
        F.min("view_ts").alias("first_view_ts"),
    )

    platform_counts = views.groupBy(*KEY_COLUMNS, "platform").agg(F.count(F.lit(1)).alias("_n"))
    w = Window.partitionBy(*KEY_COLUMNS).orderBy(F.col("_n").desc(), _tie_rank(F.col("platform")))
    top = (
        platform_counts.withColumn("_rn", F.row_number().over(w))
        .where(F.col("_rn") == 1)
        .select(*KEY_COLUMNS, F.col("platform").alias("platform_top"))
    )
    return (
        per_day.join(top, on=KEY_COLUMNS, how="inner")
        .withColumn("episodes_watched", F.col("episodes_watched").cast("bigint"))
        .select(*ATTRS_COLUMNS)
    )


# --- Join + read-time buckets ------------------------------------------------

def join_attrs(daily_df: DataFrame, attrs_df: DataFrame, *, validate: bool = True) -> DataFrame:
    """LEFT JOIN the attribute columns onto the daily frame on
    ``(user_id, item_id, view_date)``.

    Daily rows without attributes keep null attribute columns (every bucket
    helper maps those to ``unknown``). With ``validate``, raises if an attrs
    key inside the daily frame's date range has no daily row: the tables are
    built with identical filters, so such a key means the two drifted. That
    holds for a daily frame restricted to a contiguous date range (as
    ``with_days_ago`` / ``filter_activity_window`` produce); pass
    ``validate=False`` after any other filter (by user, by item, …), or
    filter ``attrs_df`` the same way first.
    """
    attr_cols = [c for c in attrs_df.columns if c not in KEY_COLUMNS]
    clash = [c for c in attr_cols if c in daily_df.columns]
    if clash:
        raise ValueError(f"daily_df already has attribute columns {clash}")
    if validate:
        bounds = daily_df.agg(F.min("view_date").alias("lo"), F.max("view_date").alias("hi")).first()
        if bounds["lo"] is not None:
            orphans = (
                attrs_df.where(F.col("view_date").between(bounds["lo"], bounds["hi"]))
                .select(*KEY_COLUMNS)
                .join(daily_df.select(*KEY_COLUMNS), on=KEY_COLUMNS, how="left_anti")
            )
            n = orphans.limit(1).count()
            if n:
                example = orphans.first()
                raise ValueError(
                    "attrs key missing from the daily frame (the two tables' filters "
                    f"drifted), e.g. {tuple(example)}"
                )
    return daily_df.join(attrs_df.select(*KEY_COLUMNS, *attr_cols), on=KEY_COLUMNS, how="left")


def with_depth_bucket(
    df: DataFrame,
    *,
    partial_threshold: float = cfg.DEPTH_PARTIAL_THRESHOLD,
    completed_threshold: float = cfg.DEPTH_COMPLETED_THRESHOLD,
) -> DataFrame:
    """``depth_bucket`` from ``max_depth_ratio``: ``< partial_threshold`` is
    ``sampled``, ``< completed_threshold`` is ``partial``, else ``completed``;
    null (duration unknown, or no attrs row) is ``unknown``."""
    if not 0.0 < partial_threshold < completed_threshold <= 1.0:
        raise ValueError(
            f"need 0 < partial_threshold < completed_threshold <= 1, got "
            f"{partial_threshold}, {completed_threshold}"
        )
    r = F.col("max_depth_ratio")
    return df.withColumn(
        "depth_bucket",
        F.when(r.isNull(), F.lit("unknown"))
        .when(r < F.lit(float(partial_threshold)), F.lit("sampled"))
        .when(r < F.lit(float(completed_threshold)), F.lit("partial"))
        .otherwise(F.lit("completed")),
    )


def with_binge_bucket(df: DataFrame) -> DataFrame:
    """``binge_bucket`` from ``episodes_watched``: ``1`` / ``2-3`` / ``4-6`` /
    ``7+``; null or 0 (no episode ids that day, or no attrs row) is
    ``unknown``."""
    e = F.col("episodes_watched")
    return df.withColumn(
        "binge_bucket",
        F.when(e.isNull() | (e < 1), F.lit("unknown"))
        .when(e == 1, F.lit("1"))
        .when(e <= 3, F.lit("2-3"))
        .when(e <= 6, F.lit("4-6"))
        .otherwise(F.lit("7+")),
    )


def with_audio_pref(df: DataFrame) -> DataFrame:
    """``dub_share`` = ``dub_views / (dub_views + sub_views)``; null when both
    are 0 or null (audio unknown that day)."""
    dub = F.coalesce(F.col("dub_views"), F.lit(0))
    sub = F.coalesce(F.col("sub_views"), F.lit(0))
    total = dub + sub
    return df.withColumn("dub_share", F.when(total > 0, dub.cast("double") / total))


def log2_bucket(days, num_buckets: int) -> "F.Column":
    """Spark twin of ``src/data/mod.rs::days_ago_to_bucket`` for whole-day
    counts: ``< 1`` -> 0, else ``floor(log2(days))`` clamped to
    ``num_buckets - 1``; null stays null. Uses the bit length of the integer
    rather than a floating ``log2`` so exact powers of two never round down."""
    if num_buckets < 1:
        raise ValueError(f"num_buckets must be >= 1, got {num_buckets}")
    d = F.col(days) if isinstance(days, str) else days
    whole = F.floor(d).cast("long")
    raw = F.length(F.bin(whole)) - F.lit(1)
    return (
        F.when(d.isNull(), F.lit(None))
        .when(d < F.lit(1.0), F.lit(0))
        .otherwise(F.least(raw, F.lit(num_buckets - 1)))
        .cast("int")
    )


def with_discovery_lag(
    df: DataFrame,
    item_attrs_df: DataFrame,
    *,
    num_buckets: int = cfg.DISCOVERY_LAG_NUM_BUCKETS,
) -> DataFrame:
    """Add ``discovery_lag_days`` = ``datediff(view_date, premiere_date)``
    (clamped at 0; null when the series has no premiere date) and
    ``discovery_lag_bucket`` (``log2_bucket``; null when unknown).

    Computed from ``view_date`` only, so a backtest at any ``as_of`` sees the
    same lag the model would have seen on that day."""
    premiere = item_attrs_df.select("item_id", F.col("premiere_date").alias("_premiere"))
    return (
        df.join(premiere, on="item_id", how="left")
        .withColumn(
            "discovery_lag_days",
            # GREATEST skips nulls; an unknown premiere must stay null.
            F.when(
                F.col("_premiere").isNotNull(),
                F.greatest(F.datediff(F.col("view_date"), F.col("_premiere")), F.lit(0)),
            ).cast("double"),
        )
        .withColumn("discovery_lag_bucket", log2_bucket("discovery_lag_days", num_buckets))
        .drop("_premiere")
    )


# --- FEASE user features ------------------------------------------------------

def attrs_to_user_features(
    daily_with_attrs_df: DataFrame,
    *,
    completed_threshold: float = cfg.DEPTH_COMPLETED_THRESHOLD,
    binge_heavy_min: int = cfg.BINGE_HEAVY_MIN_EPISODES,
    dub_share_sub_max: float = cfg.DUB_SHARE_SUB_MAX,
    dub_share_dub_min: float = cfg.DUB_SHARE_DUB_MIN,
) -> DataFrame:
    """Daily rows with attrs (``join_attrs`` output) -> long-format
    ``(user_id, feature_name, value)`` user features FEASE consumes today.

    Shares are over the user's daily rows (one per series per active day)
    that carry the attribute; zero shares are omitted:

    - ``platform_top_<tv|mobile|web|other>``: share of rows whose top
      platform is that class.
    - ``dub_pref_<sub|mixed|dub>`` = 1.0: the user's overall
      ``Σdub / Σ(dub + sub)`` bucketed at ``dub_share_sub_max`` /
      ``dub_share_dub_min``; omitted when no view has a known audio class.
    - ``depth_completed_share``: share of rows with known depth whose
      ``max_depth_ratio >= completed_threshold``.
    - ``binge_heavy_share``: share of rows with known episode counts that
      watched ``>= binge_heavy_min`` episodes.
    """
    if not 0.0 <= dub_share_sub_max < dub_share_dub_min <= 1.0:
        raise ValueError("need 0 <= dub_share_sub_max < dub_share_dub_min <= 1")
    df = daily_with_attrs_df

    platform = (
        df.where(F.col("platform_top").isNotNull())
        .groupBy("user_id")
        .pivot("platform_top", list(cfg.PLATFORM_TIE_ORDER))
        .count()
    )
    rows = df.where(F.col("platform_top").isNotNull()).groupBy("user_id").agg(
        F.count(F.lit(1)).alias("_rows")
    )
    platform_long = None
    for cls in cfg.PLATFORM_TIE_ORDER:
        part = platform.join(rows, "user_id").select(
            "user_id",
            F.lit(f"platform_top_{cls}").alias("feature_name"),
            (F.coalesce(F.col(f"`{cls}`"), F.lit(0)) / F.col("_rows")).cast("double").alias("value"),
        )
        platform_long = part if platform_long is None else platform_long.unionByName(part)

    known_depth = F.col("max_depth_ratio").isNotNull()
    known_eps = F.col("episodes_watched").isNotNull() & (F.col("episodes_watched") >= 1)
    per_user = df.groupBy("user_id").agg(
        F.sum(F.coalesce(F.col("dub_views"), F.lit(0))).alias("_dub"),
        F.sum(F.coalesce(F.col("sub_views"), F.lit(0))).alias("_sub"),
        F.sum(F.when(known_depth, 1).otherwise(0)).alias("_depth_n"),
        F.sum(
            F.when(known_depth & (F.col("max_depth_ratio") >= F.lit(float(completed_threshold))), 1)
            .otherwise(0)
        ).alias("_completed"),
        F.sum(F.when(known_eps, 1).otherwise(0)).alias("_eps_n"),
        F.sum(
            F.when(known_eps & (F.col("episodes_watched") >= F.lit(int(binge_heavy_min))), 1)
            .otherwise(0)
        ).alias("_heavy"),
    )
    total_audio = F.col("_dub") + F.col("_sub")
    share = F.col("_dub") / total_audio
    dub_pref = per_user.where(total_audio > 0).select(
        "user_id",
        F.concat(
            F.lit("dub_pref_"),
            F.when(share <= F.lit(float(dub_share_sub_max)), F.lit("sub"))
            .when(share >= F.lit(float(dub_share_dub_min)), F.lit("dub"))
            .otherwise(F.lit("mixed")),
        ).alias("feature_name"),
        F.lit(1.0).alias("value"),
    )
    depth = per_user.where(F.col("_depth_n") > 0).select(
        "user_id",
        F.lit("depth_completed_share").alias("feature_name"),
        (F.col("_completed") / F.col("_depth_n")).cast("double").alias("value"),
    )
    binge = per_user.where(F.col("_eps_n") > 0).select(
        "user_id",
        F.lit("binge_heavy_share").alias("feature_name"),
        (F.col("_heavy") / F.col("_eps_n")).cast("double").alias("value"),
    )
    return (
        platform_long.unionByName(dub_pref)
        .unionByName(depth)
        .unionByName(binge)
        .where(F.col("value") > 0)
        .select(F.col("user_id").cast("string"), "feature_name", "value")
    )


def add_interaction_attr_user_features(
    user_features_df: DataFrame,
    daily_df: DataFrame,
    attrs_df: Optional[DataFrame],
    *,
    enabled: bool,
) -> DataFrame:
    """``fease_train.py``'s ``USE_INTERACTION_ATTR_FEATURES`` switch.

    Disabled: returns ``user_features_df`` itself, untouched, so the training
    inputs are byte-identical to a run without the attrs tables. Enabled:
    unions ``attrs_to_user_features(join_attrs(daily_df, attrs_df))`` onto it.
    Pass the windowed daily frame the interactions came from, so the
    features describe the same history the model trains on."""
    if not enabled:
        return user_features_df
    if attrs_df is None:
        raise ValueError("enabled=True needs attrs_df")
    extra = attrs_to_user_features(join_attrs(daily_df, attrs_df))
    return user_features_df.unionByName(extra.select(*user_features_df.columns))


def fold_interaction_attr_user_features(
    base_user_features_df: DataFrame,
    daily_df: DataFrame,
    attrs_df: DataFrame,
    *,
    days_ago_cutoff: Optional[float] = None,
    train_pairs_df: Optional[DataFrame] = None,
) -> DataFrame:
    """``add_interaction_attr_user_features`` for an evaluation train fold.

    The summaries are built from the fold's own days only, so held-out views
    never shape the features the model trains and scores with. Give exactly
    one of ``days_ago_cutoff`` (temporal fold: the days older than the
    cutoff) or ``train_pairs_df`` (any other split: every day of the fold's
    ``(user_id, item_id)`` pairs; ``attrs_df`` is restricted to the same
    pairs so ``join_attrs``' drift check still holds).
    """
    if (days_ago_cutoff is None) == (train_pairs_df is None):
        raise ValueError("give exactly one of days_ago_cutoff or train_pairs_df")
    if days_ago_cutoff is not None:
        fold_daily = daily_df.where(F.col("days_ago") > F.lit(float(days_ago_cutoff)))
        fold_attrs = attrs_df
    else:
        keys = train_pairs_df.select("user_id", "item_id").distinct()
        fold_daily = daily_df.join(keys, on=["user_id", "item_id"], how="left_semi")
        fold_attrs = attrs_df.join(keys, on=["user_id", "item_id"], how="left_semi")
    return add_interaction_attr_user_features(
        base_user_features_df, fold_daily, fold_attrs, enabled=True,
    )
