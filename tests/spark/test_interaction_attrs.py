"""Per-interaction / per-series attribute helpers (sibling tables of #102)."""
import datetime as dt
import math
import pathlib
import re
from string import Template

import pytest

pytestmark = pytest.mark.spark

from pyspark.sql import functions as F

from kzn_recsys import cr_config as cfg
from kzn_recsys.spark.interactions_agg import (
    DAILY_COLUMNS,
    aggregate_viewership_daily,
    daily_to_pairs,
    with_days_ago,
)
from kzn_recsys.spark.interaction_attrs import (
    ATTRS_COLUMNS,
    add_interaction_attr_user_features,
    aggregate_viewership_attrs_daily,
    attrs_to_user_features,
    join_attrs,
    log2_bucket,
    normalize_platform,
    platform_case_sql,
    with_audio_pref,
    with_binge_bucket,
    with_depth_bucket,
    with_discovery_lag,
)

SQL_DIR = pathlib.Path(__file__).resolve().parents[2] / "databricks" / "sql"
AS_OF = dt.date(2026, 10, 7)
D1 = dt.date(2026, 10, 1)
D_P2 = dt.date(2026, 9, 30)

_SCHEMA = (
    "user string, show string, ts string, secs double, sub string, "
    "media string, dur double, platform string, audio string"
)


def _events(spark):
    """Raw viewership rows with one row per edge case."""
    today = dt.date.today().isoformat()
    rows = [
        # p1/S1 on 10-01: three views, two distinct episodes, one rewatch.
        # Platforms: 1x iOS (mobile), 2x Roku (tv) -> tv. Audio: 1 sub, 2 dub.
        ("p1", "S1", "2026-10-01 08:00:00", 600.0, "crunchyroll", "E1", 1440.0, "iOS App", "ja-JP"),
        ("p1", "S1", "2026-10-01 21:00:00", 1400.0, "crunchyroll", "E2", 1440.0, "Roku", "en-US"),
        ("p1", "S1", "2026-10-01 22:00:00", 1440.0, "crunchyroll", "E2", 1440.0, " ROKU ", "en-US"),
        # p1/S2: null duration and null audio.
        ("p1", "S2", "2026-10-06 09:00:00", 45.0, "crunchyroll", "E5", None, "Web Browser", None),
        # Below min watch -> dropped from both tables.
        ("p1", "S3", "2026-10-06 10:00:00", 10.0, "crunchyroll", "E9", 1440.0, "Roku", "ja-JP"),
        # p2/S1 on 09-30: platform tie mobile (Android) vs tv (Android TV) -> tv.
        ("p2", "S1", "2026-09-30 12:00:00", 300.0, "crunchyroll", "E1", 1440.0, "Android", "ja-JP"),
        ("p2", "S1", "2026-09-30 13:00:00", 900.0, "crunchyroll", "E3", 1440.0, "Android TV", "ja-JP"),
        # Watched longer than the episode: depth capped at 1.0.
        ("p2", "S2", "2026-10-02 12:00:00", 2000.0, "crunchyroll", "E7", 1440.0, "PlayStation 5", "es-419"),
        # Unknown platform string and null platform -> other.
        ("p3", "S4", "2026-10-03 12:00:00", 100.0, "crunchyroll", None, 0.0, "Fridge", None),
        ("",   "S1", "2026-10-02 12:00:00", 900.0, "crunchyroll", "E1", 1440.0, "Roku", "ja-JP"),   # blank user
        ("p3", None, "2026-10-02 12:00:00", 900.0, "crunchyroll", "E1", 1440.0, "Roku", "ja-JP"),   # null item
        ("p4", "S1", "2026-10-02 12:00:00", 900.0, "funimation", "E1", 1440.0, "Roku", "ja-JP"),    # other subsidiary
        ("p5", "S9", f"{today} 00:00:01", 900.0, "crunchyroll", "E1", 1440.0, "Roku", "ja-JP"),     # today
    ]
    return spark.createDataFrame(rows, _SCHEMA).withColumn("ts", F.to_timestamp("ts"))


def _attrs(spark, events=None, **overrides):
    kw = dict(
        user_col="user", item_col="show", date_col="ts", ts_col="ts", seconds_col="secs",
        media_id_col="media", media_duration_col="dur", platform_col="platform",
        audio_col="audio", min_watch_seconds=30.0, subsidiary_col="sub",
        subsidiary="crunchyroll", completed_ratio=0.9,
    )
    kw.update(overrides)
    return aggregate_viewership_attrs_daily(events if events is not None else _events(spark), **kw)


def _daily(spark):
    return aggregate_viewership_daily(
        _events(spark), user_col="user", item_col="show", date_col="ts",
        seconds_col="secs", min_watch_seconds=30.0,
        subsidiary_col="sub", subsidiary="crunchyroll",
    )


def _by_key(df):
    return {(r["user_id"], r["item_id"], r["view_date"]): r.asDict() for r in df.collect()}


def test_attrs_aggregation_columns_and_edge_cases(spark):
    attrs = _attrs(spark)
    assert attrs.columns == ATTRS_COLUMNS
    rows = _by_key(attrs)
    assert set(rows) == {
        ("p1", "S1", D1),
        ("p1", "S2", dt.date(2026, 10, 6)),
        ("p2", "S1", D_P2),
        ("p2", "S2", dt.date(2026, 10, 2)),
        ("p3", "S4", dt.date(2026, 10, 3)),
    }

    r = rows[("p1", "S1", D1)]
    assert r["seconds_watched"] == pytest.approx(3440.0)
    assert r["episodes_watched"] == 2              # E2 rewatched: distinct episodes
    assert r["completed_episodes"] == 2            # 1400/1440 and 1440/1440 views
    assert r["max_depth_ratio"] == pytest.approx(1.0)
    assert r["platform_top"] == "tv"               # 2 Roku (one padded/upper-case) vs 1 iOS
    assert (r["dub_views"], r["sub_views"]) == (2, 1)
    assert r["first_view_ts"] == dt.datetime(2026, 10, 1, 8, 0, 0)

    r = rows[("p1", "S2", dt.date(2026, 10, 6))]   # null duration and audio
    assert r["completed_episodes"] is None and r["max_depth_ratio"] is None
    assert (r["dub_views"], r["sub_views"]) == (0, 0)
    assert r["platform_top"] == "web"

    r = rows[("p2", "S1", D_P2)]                   # tie: 1 mobile vs 1 tv -> tv by fixed order
    assert r["platform_top"] == "tv"
    assert r["episodes_watched"] == 2 and r["completed_episodes"] == 0
    assert r["max_depth_ratio"] == pytest.approx(900.0 / 1440.0)

    r = rows[("p2", "S2", dt.date(2026, 10, 2))]   # watched past the end: capped
    assert r["max_depth_ratio"] == 1.0
    assert r["platform_top"] == "tv" and (r["dub_views"], r["sub_views"]) == (1, 0)

    r = rows[("p3", "S4", dt.date(2026, 10, 3))]   # zero duration, no episode id, odd platform
    assert r["max_depth_ratio"] is None and r["completed_episodes"] is None
    assert r["episodes_watched"] == 0
    assert r["platform_top"] == "other"

    dtypes = dict(attrs.dtypes)
    assert dtypes["episodes_watched"] == "bigint" and dtypes["completed_episodes"] == "bigint"
    assert dtypes["dub_views"] == "bigint" and dtypes["first_view_ts"] == "timestamp"


def test_attrs_key_set_equals_daily_key_set(spark):
    daily_keys = {tuple(r) for r in _daily(spark).select("user_id", "item_id", "view_date").collect()}
    attrs_keys = {tuple(r) for r in _attrs(spark).select("user_id", "item_id", "view_date").collect()}
    assert attrs_keys == daily_keys
    assert not any(k[0] == "p5" for k in attrs_keys)   # today excluded


def test_absent_source_columns_become_nulls(spark):
    attrs = _attrs(spark, media_id_col=None, media_duration_col=None, audio_col=None,
                   platform_col=None)
    for r in attrs.collect():
        assert r["episodes_watched"] == 0
        assert r["completed_episodes"] is None and r["max_depth_ratio"] is None
        assert (r["dub_views"], r["sub_views"]) == (0, 0)
        assert r["platform_top"] == "other"
    with pytest.raises(ValueError):
        _attrs(spark, completed_ratio=0.0)


def test_normalize_platform_matches_sql_case(spark):
    values = ["iOS App", "Android", "Android TV", " ROKU ", "Fire TV Stick", "Firefox",
              "Web Browser", "PlayStation 5", "LG webOS", "Fridge", None, ""]
    df = spark.createDataFrame([(v,) for v in values], "p string")
    df.createOrReplaceTempView("platform_values")
    py = [r[0] for r in df.select(normalize_platform("p")).collect()]
    sql = [r[0] for r in spark.sql(
        f"SELECT {platform_case_sql('`p`')} FROM platform_values").collect()]
    assert py == sql
    assert py == ["mobile", "mobile", "tv", "tv", "tv", "web",
                  "web", "tv", "tv", "other", "other", "other"]


def _create_attrs_select(source_view):
    """The SELECT of create_fease_interaction_attrs.sql with parameters filled
    the way databricks/src/00_refresh_interactions_agg.py fills them."""
    text = (SQL_DIR / "create_fease_interaction_attrs.sql").read_text()
    params = dict(
        catalog="c", schema="s", attrs_table="t", source_table=source_view,
        user_col="user", item_col="show", date_col="view_date", ts_col="ts",
        seconds_col="secs", subsidiary_col="sub", subsidiary="crunchyroll",
        min_watch_seconds="30.0", completed_ratio="0.9", sub_audio_locale=cfg.SUB_AUDIO_LOCALE,
        media_id_expr="CAST(`media` AS STRING)",
        duration_expr="(CAST(`dur` AS DOUBLE) * 1.0)",
        audio_expr="CAST(`audio` AS STRING)",
        platform_expr=platform_case_sql("`platform`"),
        platform_rank_expr="CASE `platform` "
        + " ".join(f"WHEN '{c}' THEN {i}" for i, c in enumerate(cfg.PLATFORM_TIE_ORDER))
        + f" ELSE {len(cfg.PLATFORM_TIE_ORDER)} END",
    )
    sql = Template(text).substitute(params)
    insert = re.search(r"INSERT OVERWRITE [^\n]+\n(.*?);", sql, re.S)
    return insert.group(1)


def test_create_sql_matches_pyspark_twin(spark):
    # The SQL reads a DATE column; give it one, as the gold table has.
    events = _events(spark).withColumn("view_date", F.to_date("ts"))
    events.createOrReplaceTempView("attrs_events")
    from_sql = _by_key(spark.sql(_create_attrs_select("attrs_events")))
    from_py = _by_key(_attrs(spark, events=events, date_col="view_date"))
    assert set(from_sql) == set(from_py)
    for key, row in from_py.items():
        for col in ATTRS_COLUMNS:
            a, b = from_sql[key][col], row[col]
            if isinstance(b, float):
                assert a == pytest.approx(b), (key, col)
            else:
                assert a == b, (key, col)


def test_join_attrs_left_joins_and_rejects_orphans(spark):
    daily = with_days_ago(_daily(spark), as_of=AS_OF)
    joined = join_attrs(daily, _attrs(spark))
    assert joined.count() == daily.count()
    assert joined.columns[: len(daily.columns)] == daily.columns
    # A daily row without attrs keeps null attributes.
    partial = _attrs(spark).where(F.col("user_id") != "p3")
    j = _by_key(join_attrs(daily, partial))
    assert j[("p3", "S4", dt.date(2026, 10, 3))]["platform_top"] is None
    # An attrs key inside the daily date range with no daily row is drift.
    orphan = spark.createDataFrame(
        [("zz", "S1", dt.date(2026, 10, 2))], "user_id string, item_id string, view_date date"
    )
    extra = _attrs(spark).unionByName(orphan, allowMissingColumns=True)
    with pytest.raises(ValueError, match="drifted"):
        join_attrs(daily, extra)
    join_attrs(daily, extra, validate=False)  # opt-out
    with pytest.raises(ValueError, match="already has"):
        join_attrs(joined, _attrs(spark))


def test_depth_binge_audio_buckets_including_unknown(spark):
    df = spark.createDataFrame(
        [(None, None, None, None), (0.1, 1, 0, 0), (0.25, 2, 3, 1),
         (0.89, 3, 1, 3), (0.9, 4, None, 2), (1.0, 6, 0, 0), (0.5, 7, 2, 0), (0.5, 0, 0, 2)],
        "max_depth_ratio double, episodes_watched bigint, dub_views bigint, sub_views bigint",
    )
    out = with_audio_pref(with_binge_bucket(with_depth_bucket(df))).collect()
    assert [r["depth_bucket"] for r in out] == [
        "unknown", "sampled", "partial", "partial", "completed", "completed", "partial", "partial"]
    assert [r["binge_bucket"] for r in out] == [
        "unknown", "1", "2-3", "2-3", "4-6", "4-6", "7+", "unknown"]
    shares = [r["dub_share"] for r in out]
    assert shares[0] is None and shares[1] is None and shares[5] is None
    assert shares[2] == pytest.approx(0.75) and shares[3] == pytest.approx(0.25)
    assert shares[4] == 0.0 and shares[6] == 1.0 and shares[7] == 0.0
    with pytest.raises(ValueError):
        with_depth_bucket(df, partial_threshold=0.9, completed_threshold=0.5)


def test_log2_bucket_matches_rust_days_ago_to_bucket(spark):
    # Rust: d < 1 -> 0, else floor(log2(d)) clamped to n - 1. For whole days
    # floor(log2(d)) == d.bit_length() - 1, exact at powers of two.
    days = sorted(set(range(0, 2100)) | {2 ** k + o for k in range(1, 40) for o in (-1, 0, 1)})
    df = spark.createDataFrame([(float(d),) for d in days] + [(None,), (0.5,)], "d double")
    for n in (1, 8, 32):
        got = [r[0] for r in df.select(log2_bucket("d", n)).collect()]
        want = [0 if d < 1 else min(int(d).bit_length() - 1, n - 1) for d in days] + [None, 0]
        assert got == want, n
    with pytest.raises(ValueError):
        log2_bucket("d", 0)


def test_discovery_lag_from_view_date_with_unknown_premiere(spark):
    daily = with_days_ago(_daily(spark), as_of=AS_OF)
    items = spark.createDataFrame(
        [("S1", dt.date(2026, 9, 1), "catalog"), ("S2", dt.date(2026, 10, 3), "first_view")],
        "item_id string, premiere_date date, premiere_source string",
    )
    rows = _by_key(with_discovery_lag(daily, items, num_buckets=32))
    r = rows[("p1", "S1", D1)]
    assert r["discovery_lag_days"] == 30.0 and r["discovery_lag_bucket"] == 4
    assert rows[("p2", "S1", D_P2)]["discovery_lag_bucket"] == 4     # 29 days
    r = rows[("p1", "S2", dt.date(2026, 10, 6))]
    assert r["discovery_lag_days"] == 3.0 and r["discovery_lag_bucket"] == 1
    r = rows[("p2", "S2", dt.date(2026, 10, 2))]                      # viewed before premiere
    assert r["discovery_lag_days"] == 0.0 and r["discovery_lag_bucket"] == 0
    r = rows[("p3", "S4", dt.date(2026, 10, 3))]                      # no premiere known
    assert r["discovery_lag_days"] is None and r["discovery_lag_bucket"] is None
    # Independent of as_of: the same view has the same lag in any backtest.
    later = _by_key(with_discovery_lag(with_days_ago(_daily(spark), as_of="2026-10-15"), items))
    assert later[("p1", "S1", D1)]["discovery_lag_days"] == 30.0


def test_attrs_to_user_features(spark):
    daily = with_days_ago(_daily(spark), as_of=AS_OF)
    feats = {(r["user_id"], r["feature_name"]): r["value"]
             for r in attrs_to_user_features(join_attrs(daily, _attrs(spark))).collect()}
    # p1: rows S1 (tv, depth 1.0, 2 eps, dub 2/sub 1) and S2 (web, depth ?, 1 ep, no audio)
    assert feats[("p1", "platform_top_tv")] == pytest.approx(0.5)
    assert feats[("p1", "platform_top_web")] == pytest.approx(0.5)
    assert ("p1", "platform_top_mobile") not in feats            # zero shares omitted
    assert feats[("p1", "dub_pref_mixed")] == 1.0                # 2/3 dub
    assert feats[("p1", "depth_completed_share")] == pytest.approx(1.0)   # 1 known row
    assert ("p1", "binge_heavy_share") not in feats              # no 4+ episode day
    # p2: S1 (tv, 0.625, 2 eps, sub 2) and S2 (tv, 1.0, 1 ep, dub 1) -> 1/3 dub = mixed
    assert feats[("p2", "platform_top_tv")] == pytest.approx(1.0)
    assert feats[("p2", "dub_pref_mixed")] == 1.0
    assert feats[("p2", "depth_completed_share")] == pytest.approx(0.5)
    # p3: no audio, no depth, no episode ids -> only the platform share
    assert {k for k in feats if k[0] == "p3"} == {("p3", "platform_top_other")}
    assert all(v > 0 for v in feats.values())


def test_flag_off_training_inputs_are_byte_identical(spark):
    """USE_INTERACTION_ATTR_FEATURES=False: the user-feature frame is the
    original object and EASE trains to the same S matrix; with the flag on,
    the interactions are untouched and the original features are kept."""
    import numpy as np
    from kzn_recsys.spark import build_and_train

    daily = with_days_ago(_daily(spark), as_of=AS_OF)
    pairs = daily_to_pairs(daily)
    user_feats = spark.createDataFrame(
        [("p1", "plan_premium", 1.0), ("p2", "plan_free", 1.0)],
        "user_id string, feature_name string, value double",
    )
    item_feats = spark.createDataFrame([], "item_id string, feature_name string, value double")

    off = add_interaction_attr_user_features(user_feats, daily, None, enabled=False)
    assert off is user_feats
    base = build_and_train(pairs, user_feats, item_feats, lambda_=5.0, strategy="collect")
    same = build_and_train(pairs, off, item_feats, lambda_=5.0, strategy="collect")
    assert np.array_equal(base.s_matrix, same.s_matrix)

    attrs = _attrs(spark)
    joined = join_attrs(daily, attrs)
    assert sorted(joined.select(*DAILY_COLUMNS).collect()) == sorted(daily.select(*DAILY_COLUMNS).collect())
    assert sorted(daily_to_pairs(joined).collect()) == sorted(pairs.collect())

    on = add_interaction_attr_user_features(user_feats, daily, attrs, enabled=True)
    on_rows = {tuple(r) for r in on.collect()}
    assert {tuple(r) for r in user_feats.collect()} <= on_rows
    assert {r[1] for r in on_rows} - {"plan_premium", "plan_free"} <= {
        "platform_top_tv", "platform_top_mobile", "platform_top_web", "platform_top_other",
        "dub_pref_sub", "dub_pref_mixed", "dub_pref_dub",
        "depth_completed_share", "binge_heavy_share",
    }
    with pytest.raises(ValueError):
        add_interaction_attr_user_features(user_feats, daily, None, enabled=True)


def _notebook_statements():
    """`_statements` from the refresh notebook (which needs dbutils to run)."""
    import ast

    path = SQL_DIR.parent / "src" / "00_refresh_interactions_agg.py"
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_statements")
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    return ns["_statements"]


@pytest.mark.parametrize(
    "name, expected",
    [
        ("create_fease_interactions_agg.sql", ["CREATE", "INSERT", "OPTIMIZE"]),
        ("merge_fease_interactions_agg.sql", ["MERGE"]),
        ("validate_fease_interactions_agg.sql", ["SELECT"] * 6),
        ("create_fease_interaction_attrs.sql", ["CREATE", "INSERT", "OPTIMIZE"]),
        ("merge_fease_interaction_attrs.sql", ["CREATE", "MERGE"]),
        ("validate_fease_interaction_attrs.sql", ["SELECT"] * 7),
        ("create_fease_item_attrs.sql", ["CREATE", "INSERT"]),
    ],
)
def test_refresh_notebook_splits_every_sql_file_into_whole_statements(name, expected):
    # Comments contain semicolons; splitting before stripping them used to
    # hand Spark comment fragments as statements.
    stmts = list(_notebook_statements()((SQL_DIR / name).read_text()))
    assert [s.split()[0].upper() for s in stmts] == expected


def _fold_frames(spark):
    """u1 watches A sub-only before the cutoff and B dubbed only after it."""
    from kzn_recsys.spark.interactions_agg import with_days_ago
    as_of = dt.date(2026, 10, 7)
    days = [("u1", "A", 20), ("u1", "A", 15), ("u1", "B", 3), ("u1", "B", 2)]
    daily = with_days_ago(spark.createDataFrame(
        [(u, i, as_of - dt.timedelta(days=d), 1.0, 1) for u, i, d in days],
        "user_id string, item_id string, view_date date, value double, num_views long",
    ), as_of=as_of)
    attrs = spark.createDataFrame(
        [(u, i, as_of - dt.timedelta(days=d), 600.0, 1, 1, 1.0, "tv",
          1 if i == "B" else 0, 0 if i == "B" else 1, None) for u, i, d in days],
        "user_id string, item_id string, view_date date, seconds_watched double, "
        "episodes_watched long, completed_episodes long, max_depth_ratio double, "
        "platform_top string, dub_views long, sub_views long, first_view_ts timestamp",
    )
    base = spark.createDataFrame([("u1", "plan_Fan", 1.0)],
                                 "user_id string, feature_name string, value double")
    return base, daily, attrs


@pytest.mark.parametrize("fold", ["temporal", "pairs"])
def test_fold_attr_features_never_see_held_out_views(spark, fold):
    from kzn_recsys.spark.interaction_attrs import fold_interaction_attr_user_features
    base, daily, attrs = _fold_frames(spark)
    if fold == "temporal":
        kw = {"days_ago_cutoff": 7.0}
    else:  # e.g. leave-last-1-out: B held out, A is the train pair
        kw = {"train_pairs_df": spark.createDataFrame(
            [("u1", "A", 1.0, 15.0)], "user_id string, item_id string, value double, days_ago double")}
    got = {r["feature_name"]: r["value"]
           for r in fold_interaction_attr_user_features(base, daily, attrs, **kw).collect()}
    assert got["plan_Fan"] == 1.0
    # Only A's (sub) views count: B's dubbed, held-out views do not.
    assert "dub_pref_sub" in got and "dub_pref_dub" not in got and "dub_pref_mixed" not in got
    # The full window would have said "mixed" (2 dub vs 2 sub views).
    full = add_interaction_attr_user_features(base, daily, attrs, enabled=True)
    assert "dub_pref_mixed" in {r["feature_name"] for r in full.collect()}


def test_fold_attr_features_needs_exactly_one_fold(spark):
    from kzn_recsys.spark.interaction_attrs import fold_interaction_attr_user_features
    base, daily, attrs = _fold_frames(spark)
    with pytest.raises(ValueError, match="exactly one"):
        fold_interaction_attr_user_features(base, daily, attrs)
    with pytest.raises(ValueError, match="exactly one"):
        fold_interaction_attr_user_features(base, daily, attrs, days_ago_cutoff=7.0,
                                            train_pairs_df=daily)
