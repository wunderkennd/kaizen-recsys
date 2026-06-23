import pytest

pytestmark = pytest.mark.spark

from kzn_recsys.spark.splits import random_split, temporal_split, leave_k_out_split


def _interactions(spark):
    rows = [("u1", "i1", 1.0), ("u1", "i2", 1.0), ("u1", "i3", 1.0), ("u1", "i4", 1.0),
            ("u2", "i1", 1.0), ("u2", "i2", 1.0)]
    return spark.createDataFrame(rows, ["user_id", "item_id", "value"])


def test_random_split_holds_out_fraction_and_is_deterministic(spark):
    df = _interactions(spark)
    train1, test1 = random_split(df, test_ratio=0.5, seed=42)
    train2, test2 = random_split(df, test_ratio=0.5, seed=42)
    # determinism: same seed -> same test rows
    assert sorted(test1.collect()) == sorted(test2.collect())
    # every user with >=2 interactions keeps at least one train row
    train_users = {r["user_id"] for r in train1.collect()}
    assert {"u1", "u2"}.issubset(train_users)
    # nothing lost: train + test == original count
    assert train1.count() + test1.count() == df.count()


def test_temporal_split_by_cutoff(spark):
    rows = [("u1", "i1", 1.0, 5.0), ("u1", "i2", 1.0, 50.0)]
    df = spark.createDataFrame(rows, ["user_id", "item_id", "value", "days_ago"])
    train, test = temporal_split(df, days_ago_cutoff=10.0)
    # recent (days_ago <= cutoff) -> test
    assert {r["item_id"] for r in test.collect()} == {"i1"}
    assert {r["item_id"] for r in train.collect()} == {"i2"}


def test_leave_k_out(spark):
    df = _interactions(spark)
    train, test = leave_k_out_split(df, k=1, seed=7)
    # u1 has 4 -> exactly 1 held out; u2 has 2 -> exactly 1 held out
    test_by_user = {}
    for r in test.collect():
        test_by_user.setdefault(r["user_id"], 0)
        test_by_user[r["user_id"]] += 1
    assert test_by_user.get("u1") == 1
    assert test_by_user.get("u2") == 1


def test_random_split_is_exact_complement(spark):
    df = _interactions(spark)
    train, test = random_split(df, test_ratio=0.5, seed=3)
    train_rows = set((r["user_id"], r["item_id"]) for r in train.collect())
    test_rows = set((r["user_id"], r["item_id"]) for r in test.collect())
    all_rows = set((r["user_id"], r["item_id"]) for r in df.collect())
    assert train_rows | test_rows == all_rows      # nothing lost
    assert train_rows & test_rows == set()         # nothing duplicated
