"""Train/test splits on interaction DataFrames. Semantics mirror
src/evaluation.rs; determinism is via a Python-seeded RNG (NOT row-identical
to the Rust StdRng splits — see plan parity fact 5)."""
from __future__ import annotations

import random


def _add_row_index(df):
    # stable per-row id for reproducible masking
    from pyspark.sql import functions as F
    return df.withColumn("_rid", F.monotonically_increasing_id())


def random_split(interactions_df, test_ratio: float, seed: int):
    """For each user, hold out round(test_ratio * n) rows (clamped to [1, n-1]
    for users with >= 2 interactions). Mirrors evaluation.rs:103-173 semantics."""
    if not 0.0 <= test_ratio <= 1.0:
        raise ValueError("test_ratio must be between 0.0 and 1.0")
    df = _add_row_index(interactions_df)
    rows = df.select("user_id", "_rid").collect()
    by_user = {}
    for r in rows:
        by_user.setdefault(r["user_id"], []).append(r["_rid"])

    rng = random.Random(seed)
    test_ids = set()
    for uid in sorted(by_user):
        rids = list(by_user[uid])
        rng.shuffle(rids)
        n = len(rids)
        if n < 2:
            n_test = 0
        else:
            n_test = max(1, min(round(n * test_ratio), n - 1))
        test_ids.update(rids[:n_test])

    from pyspark.sql import functions as F
    test_df = df.where(F.col("_rid").isin(list(test_ids))).drop("_rid")
    train_df = df.where(~F.col("_rid").isin(list(test_ids))).drop("_rid")
    return train_df, test_df


def temporal_split(interactions_df, days_ago_cutoff: float):
    """Recent (days_ago <= cutoff) -> test, older -> train. Mirrors evaluation.rs:177-226."""
    from pyspark.sql import functions as F
    if interactions_df.where(F.col("days_ago").isNull()).count() > 0:
        raise ValueError("days_ago contains nulls; temporal_split requires non-null days_ago")
    test_df = interactions_df.where(F.col("days_ago") <= days_ago_cutoff)
    train_df = interactions_df.where(F.col("days_ago") > days_ago_cutoff)
    return train_df, test_df


def leave_k_out_split(interactions_df, k: int, seed: int):
    """Hold out exactly k rows per user with >= k+1 interactions. Mirrors evaluation.rs:230+."""
    df = _add_row_index(interactions_df)
    rows = df.select("user_id", "_rid").collect()
    by_user = {}
    for r in rows:
        by_user.setdefault(r["user_id"], []).append(r["_rid"])

    rng = random.Random(seed)
    test_ids = set()
    for uid in sorted(by_user):
        rids = list(by_user[uid])
        if len(rids) < k + 1:
            continue
        rng.shuffle(rids)
        test_ids.update(rids[:k])

    from pyspark.sql import functions as F
    test_df = df.where(F.col("_rid").isin(list(test_ids))).drop("_rid")
    train_df = df.where(~F.col("_rid").isin(list(test_ids))).drop("_rid")
    return train_df, test_df
