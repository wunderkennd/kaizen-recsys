"""Train/test splits on interaction DataFrames. Semantics mirror
src/evaluation.rs; determinism is via a Python-seeded RNG (NOT row-identical
to the Rust StdRng splits — see plan parity fact 5).

random_split / leave_k_out_split collect rows to the driver once and use the
stable list index as the row id, then rebuild train/test frames. This avoids
F.monotonically_increasing_id(), which is nondeterministic across the separate
Spark executions that collecting-then-filtering would trigger."""
from __future__ import annotations

import random


def _positions_by_user(rows):
    by_user = {}
    for idx, r in enumerate(rows):
        by_user.setdefault(r["user_id"], []).append(idx)
    return by_user


def _rebuild_from(interactions_df, rows, held_out):
    """Build (train_df, test_df) from the single `rows` materialization."""
    spark = interactions_df.sparkSession
    schema = interactions_df.schema
    test_rows = [rows[i] for i in range(len(rows)) if i in held_out]
    train_rows = [rows[i] for i in range(len(rows)) if i not in held_out]
    return (spark.createDataFrame(train_rows, schema),
            spark.createDataFrame(test_rows, schema))


def random_split(interactions_df, test_ratio: float, seed: int):
    """For each user, hold out round(test_ratio * n) rows (clamped to [1, n-1]
    for users with >= 2 interactions). Mirrors evaluation.rs:103-173 semantics.

    Deterministic given `seed`; not row-identical to the Rust StdRng split.
    """
    if not 0.0 <= test_ratio <= 1.0:
        raise ValueError("test_ratio must be between 0.0 and 1.0")
    rows = interactions_df.collect()
    by_user = _positions_by_user(rows)

    rng = random.Random(seed)
    held_out = set()
    for uid in sorted(by_user):
        positions = list(by_user[uid])
        rng.shuffle(positions)
        n = len(positions)
        if n < 2:
            n_test = 0
        else:
            n_test = max(1, min(round(n * test_ratio), n - 1))
        held_out.update(positions[:n_test])

    return _rebuild_from(interactions_df, rows, held_out)


def temporal_split(interactions_df, days_ago_cutoff: float):
    """Recent (days_ago <= cutoff) -> test, older -> train. Mirrors evaluation.rs:177-226."""
    from pyspark.sql import functions as F
    if interactions_df.where(F.col("days_ago").isNull()).count() > 0:
        raise ValueError("days_ago contains nulls; temporal_split requires non-null days_ago")
    test_df = interactions_df.where(F.col("days_ago") <= days_ago_cutoff)
    train_df = interactions_df.where(F.col("days_ago") > days_ago_cutoff)
    return train_df, test_df


def leave_k_out_split(interactions_df, k: int, seed: int):
    """Hold out exactly k rows per user with >= k+1 interactions. Mirrors evaluation.rs:230+.

    Deterministic given `seed`; not row-identical to the Rust StdRng split.
    """
    rows = interactions_df.collect()
    by_user = _positions_by_user(rows)

    rng = random.Random(seed)
    held_out = set()
    for uid in sorted(by_user):
        positions = list(by_user[uid])
        if len(positions) < k + 1:
            continue
        rng.shuffle(positions)
        held_out.update(positions[:k])

    return _rebuild_from(interactions_df, rows, held_out)


def leave_last_k_out_split(interactions_df, k: int):
    """Hold out each user's k most recently interacted *distinct items* (an
    item's recency is its latest row); every row of a held-out item moves to
    test. Mirrors evaluation.rs::leave_last_k_out_split: deterministic (no
    RNG), ties on days_ago broken by row order, users with < k+1 distinct
    items go entirely to train. Requires a non-null `days_ago` column and
    raises without one.

    Items rather than rows are held out because evaluation excludes every
    train item from a user's candidates: holding out only the latest row of a
    repeated item would make that hold-out an impossible miss.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if "days_ago" not in interactions_df.columns:
        raise ValueError(
            "leave_last_k_out_split requires a `days_ago` column to order each user's "
            "history; use leave_k_out_split for a random hold-out"
        )
    rows = interactions_df.collect()
    by_user = {}  # user -> item -> [min_days_ago, first_idx, [row idx...]]
    for idx, r in enumerate(rows):
        d = r["days_ago"]
        if d is None:
            raise ValueError(f"leave_last_k_out_split: row {idx} has a null days_ago")
        d = float(d)
        items = by_user.setdefault(r["user_id"], {})
        entry = items.get(r["item_id"])
        if entry is None:
            items[r["item_id"]] = [d, idx, [idx]]
        else:
            entry[0] = min(entry[0], d)
            entry[2].append(idx)

    held_out = set()
    for uid in sorted(by_user):
        items = by_user[uid]
        if len(items) < k + 1:
            continue
        ordered = sorted(items.values(), key=lambda e: (e[0], e[1]))
        for _, _, idxs in ordered[:k]:
            held_out.update(idxs)

    return _rebuild_from(interactions_df, rows, held_out)
