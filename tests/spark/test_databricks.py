import pytest

pytest.importorskip("numpy")
from pyspark.sql import functions as F
from kzn_recsys.spark.databricks import make_synthetic, feature_engineering


def test_make_synthetic_schema_and_determinism(spark):
    inter1, users1, items1 = make_synthetic(spark, n_users=50, n_items=20,
                                            n_personas=3, avg_interactions=8, seed=7)

    # exact long-format column contracts
    assert inter1.columns == ["user_id", "item_id", "value"]
    assert users1.columns == ["user_id", "feature_name", "value"]
    assert items1.columns == ["item_id", "feature_name", "value"]

    # value column is all 1.0 for interactions
    assert inter1.filter(F.col("value") != 1.0).count() == 0

    # every user appears with at least one interaction and one persona feature
    assert inter1.select("user_id").distinct().count() == 50
    assert users1.filter(F.col("feature_name").startswith("persona=")).count() == 50
    # every item has a genre feature
    assert items1.select("item_id").distinct().count() == 20

    # seed-determinism: same seed -> identical interaction rows
    inter2, _, _ = make_synthetic(spark, n_users=50, n_items=20,
                                  n_personas=3, avg_interactions=8, seed=7)
    rows1 = sorted((r.user_id, r.item_id) for r in inter1.collect())
    rows2 = sorted((r.user_id, r.item_id) for r in inter2.collect())
    assert rows1 == rows2


def test_feature_engineering_maps_raw_to_long(spark):
    engagement = spark.createDataFrame(
        [
            # user, item, watched, country
            ("u1", "m1", 3.0, "US"),
            ("u1", "m1", 2.0, "US"),   # duplicate (u1,m1) -> summed to 5.0
            ("u1", "m2", 1.0, "US"),
            ("u2", "m2", 4.0, "JP"),
        ],
        ["user_id", "item_id", "watched", "country"],
    )
    metadata = spark.createDataFrame(
        [("m1", "action"), ("m2", "drama")],
        ["item_id", "genre"],
    )

    inter, users, items = feature_engineering(
        engagement, metadata,
        value_col="watched",
        user_feature_cols=["country"],
        item_feature_cols=["genre"],
    )

    assert inter.columns == ["user_id", "item_id", "value"]
    # duplicate (u1,m1) summed
    val = {(r.user_id, r.item_id): r.value for r in inter.collect()}
    assert val[("u1", "m1")] == 5.0
    assert val[("u2", "m2")] == 4.0

    # one-hot user features, deduped per (user, feature)
    ufeat = {(r.user_id, r.feature_name) for r in users.collect()}
    assert ("u1", "country=US") in ufeat
    assert ("u2", "country=JP") in ufeat
    assert users.filter(F.col("value") != 1.0).count() == 0
    assert users.count() == 2

    # one-hot item features from metadata
    ifeat = {(r.item_id, r.feature_name) for r in items.collect()}
    assert ("m1", "genre=action") in ifeat
    assert ("m2", "genre=drama") in ifeat


def test_feature_engineering_preserves_optional_columns(spark):
    engagement = spark.createDataFrame(
        [("u1", "m1", 1.0, "play", 2.0), ("u1", "m1", 1.0, "like", 1.0)],
        ["user_id", "item_id", "value", "event_type", "days_ago"],
    )
    metadata = spark.createDataFrame([("m1", "action")], ["item_id", "genre"])
    inter, users, _ = feature_engineering(
        engagement, metadata,
        event_type_col="event_type", days_ago_col="days_ago",
        item_feature_cols=["genre"],
    )
    # rows pass through (not summed) with the optional columns present
    assert inter.columns == ["user_id", "item_id", "value", "event_type", "days_ago"]
    assert inter.count() == 2
    # empty feature columns should return the correct schema
    assert users.columns == ["user_id", "feature_name", "value"]
    assert users.count() == 0


def test_make_synthetic_covers_non_divisible_catalog_and_cold_users(spark):
    # 30 items over 4 personas: floor-sized slices would leave items 28-29
    # without any interactions. The slices must tile the whole catalog.
    inter, users, items = make_synthetic(spark, n_users=40, n_items=30,
                                         n_personas=4, avg_interactions=10,
                                         n_cold_users=3, seed=3)
    interacted = {r.item_id for r in inter.select("item_id").distinct().collect()}
    catalog = {r.item_id for r in items.select("item_id").distinct().collect()}
    assert interacted == catalog

    # cold-start users carry a persona feature but no interactions
    warm = {r.user_id for r in inter.select("user_id").distinct().collect()}
    featured = {r.user_id for r in users.select("user_id").distinct().collect()}
    assert len(warm) == 40
    assert len(featured) == 43
    assert featured - warm == {"user_00040", "user_00041", "user_00042"}
