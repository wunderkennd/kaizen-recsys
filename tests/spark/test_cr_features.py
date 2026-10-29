"""CR side features shared by fease_train.py and the bundle's ingest task."""
import datetime as dt

import pytest

pytestmark = pytest.mark.spark

from kzn_recsys.spark.cr_features import cr_item_features, cr_user_features


def _engagement(spark):
    rows = [
        # profile, ts, subsidiary, plan, country, region, status, tenure
        ("p1", "2026-10-01 08:00:00", "crunchyroll", "Fan", "US", "NA", "Paying", 5),
        ("p1", "2026-10-05 08:00:00", "crunchyroll", "Mega", "US", "NA", "Paying", 400),  # latest
        ("p2", "2026-10-03 08:00:00", "crunchyroll", "Free", None, "EU", "Free", None),
        ("p3", "2026-10-03 08:00:00", "crunchyroll", "Fan", "BR", "LA", "Paying", 10),   # inactive
        ("p1", "2026-10-06 08:00:00", "funimation", "Other", "US", "NA", "Paying", 1),   # other sub
        ("  ", "2026-10-06 08:00:00", "crunchyroll", "Fan", "US", "NA", "Paying", 1),    # blank id
    ]
    from pyspark.sql import functions as F
    return spark.createDataFrame(rows, (
        "view_profile_id string, view_ts string, view_subsidiary string, "
        "view_subscription_plan string, account_country_code_account string, "
        "region_major_account string, subscription_status string, account_tenure_days int"
    )).withColumn("view_ts", F.to_timestamp("view_ts"))


def _daily(spark):
    return spark.createDataFrame(
        [("p1", "S1", dt.date(2026, 10, 1), 1.0, 1), ("p2", "S2", dt.date(2026, 10, 2), 1.0, 1)],
        "user_id string, item_id string, view_date date, value double, num_views long",
    )


def test_cr_user_features_latest_row_of_active_users(spark):
    feats = cr_user_features(_engagement(spark), _daily(spark))
    got = {(r["user_id"], r["feature_name"], r["value"]) for r in feats.collect()}
    assert got == {
        ("p1", "view_subscription_plan_Mega", 1.0),
        ("p1", "account_country_code_account_US", 1.0),
        ("p1", "region_major_account_NA", 1.0),
        ("p1", "subscription_status_Paying", 1.0),
        ("p1", "tenure_90d+", 1.0),
        ("p2", "view_subscription_plan_Free", 1.0),
        ("p2", "region_major_account_EU", 1.0),   # null country -> no row
        ("p2", "subscription_status_Free", 1.0),
        ("p2", "tenure_unknown", 1.0),
    }
    assert feats.columns == ["user_id", "feature_name", "value"]


@pytest.mark.parametrize("days,bucket", [
    (0, "tenure_0d"), (7, "tenure_7d"), (8, "tenure_30d"), (90, "tenure_90d"), (91, "tenure_90d+"),
])
def test_cr_user_features_tenure_buckets(spark, days, bucket):
    eng = _engagement(spark).where("view_profile_id = 'p2'").drop("account_tenure_days") \
        .selectExpr("*", f"{days} AS account_tenure_days")
    names = {r["feature_name"] for r in cr_user_features(eng, _daily(spark)).collect()}
    assert bucket in names and len([n for n in names if n.startswith("tenure_")]) == 1


def test_cr_item_features_series_keyed_and_exploded(spark):
    meta = spark.createDataFrame([
        # series, media_type, audio, genre, grade, genres, tags
        ("S1", "episode", "ja-JP", "action", "A", "action, fantasy", "isekai"),
        ("S1", "episode", "en-US", "action", "A", "action", None),       # second episode
        ("S2", "movie", None, "drama", None, "drama,", "tearjerker , school"),
        (None, "episode", "ja-JP", "comedy", "B", "comedy", "x"),        # no series -> dropped
    ], (
        "catalog_show_id string, media_type string, media_audio_language string, "
        "airtable_primary_genre string, airtable_ca_brand_grade string, "
        "media_genres string, media_tags string"
    ))
    got = {(r["item_id"], r["feature_name"]) for r in cr_item_features(meta).collect()}
    assert got == {
        ("S1", "media_type_episode"), ("S1", "media_audio_language_ja-JP"),
        ("S1", "media_audio_language_en-US"), ("S1", "airtable_primary_genre_action"),
        ("S1", "airtable_ca_brand_grade_A"), ("S1", "genre_action"), ("S1", "genre_fantasy"),
        ("S1", "tag_isekai"),
        ("S2", "media_type_movie"), ("S2", "airtable_primary_genre_drama"),
        ("S2", "genre_drama"), ("S2", "genre_"),   # trailing comma, as fease_train.py did
        ("S2", "tag_tearjerker"), ("S2", "tag_school"),
    }


def test_cr_user_features_as_of_ignores_later_state(spark):
    # p1's 2026-10-05 row (plan Mega) postdates as_of: the 2026-10-01 state wins.
    feats = cr_user_features(_engagement(spark), _daily(spark), as_of="2026-10-03")
    got = {(r["user_id"], r["feature_name"]) for r in feats.collect() if r["user_id"] == "p1"}
    assert ("p1", "view_subscription_plan_Fan") in got
    assert ("p1", "tenure_7d") in got
    assert not any("Mega" in f or f == "tenure_90d+" for _, f in got)
