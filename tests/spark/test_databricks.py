import pytest

pytest.importorskip("numpy")
from pyspark.sql import functions as F
from kzn_recsys.spark.databricks import make_synthetic


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
