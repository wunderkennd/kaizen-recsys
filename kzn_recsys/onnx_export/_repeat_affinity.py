"""Tier C: statistical per-user repeat-affinity table (spec §10, issue #69).

Estimates each user's repeat propensity from interaction history at export
time and maps it to a per-user repeat penalty ``ρ``. The ONNX graph already
takes ``repeat_penalty`` as a per-row input (Tier B), so the learned table
plugs in with no graph change — only a vocab.json entry and a pyfunc lookup.

Method (documented per the issue's acceptance criteria):

- An interaction *event* is one row of the long-format interactions file.
  ``k`` rows for the same ``(user_id, item_id)`` pair mean the user returned
  to that item ``k - 1`` times, so ``repeats_u = rows_u - distinct_pairs_u``.
  If the file is pre-aggregated (one row per pair), no repeat signal exists
  and every user shrinks to the global rate — safe, but a warning is emitted
  when the file shows no repeats at all.
- The raw rate is smoothed toward the global repeat rate with an
  empirical-Bayes prior: ``p_u = (repeats_u + s·p_global) / (events_u + s)``
  where ``s = prior_strength`` (pseudo-event count; default 10).
- ``ρ_u = scale · (1 − p_u)``: a user who never repeats gets the full demotion
  ``scale``; a habitual repeater approaches neutral (``ρ → 0``). The span is
  demote→neutral by design — boosting repeats is a product decision the
  caller can still make per-request via the Tier B input. ``scale`` is in
  raw-score units and defaults to 1.0; tune it to the model's score range.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import polars as pl


def estimate_repeat_affinity(
    interactions: str | Path,
    item_vocab: list[str],
    *,
    scale: float = 1.0,
    prior_strength: float = 10.0,
) -> dict[str, float]:
    """Return a ``user_guid → ρ`` table from a long-format interactions file.

    ``item_vocab`` (the model's ``item_index_to_guid``) is used only for a
    sanity check: if the file's items barely overlap the model's catalog, the
    caller probably passed the wrong dataset.
    """
    if scale < 0:
        raise ValueError("repeat_affinity scale must be non-negative")
    if prior_strength < 0:
        raise ValueError("repeat_affinity prior_strength must be non-negative")

    path = Path(interactions)
    df = pl.read_parquet(path) if path.suffix == ".parquet" else pl.read_csv(path)
    missing = {"user_id", "item_id"} - set(df.columns)
    if missing:
        raise ValueError(f"interactions file lacks required column(s): {sorted(missing)}")
    df = df.select(pl.col("user_id").cast(pl.Utf8), pl.col("item_id").cast(pl.Utf8))

    # Sanity: the interactions should describe (mostly) this model's catalog.
    file_items = set(df.get_column("item_id").unique().to_list())
    known = set(item_vocab)
    overlap = len(file_items & known) / len(file_items) if file_items else 0.0
    if overlap < 0.5:
        warnings.warn(
            f"only {overlap:.0%} of the interactions file's items appear in the "
            "model's item vocab — is this the training dataset for this model?",
            stacklevel=3,
        )

    per_pair = df.group_by(["user_id", "item_id"]).len()
    per_user = per_pair.group_by("user_id").agg(
        pl.col("len").sum().alias("events"),
        (pl.col("len") - 1).sum().alias("repeats"),
    )

    total_events = per_user.get_column("events").sum() or 0
    total_repeats = per_user.get_column("repeats").sum() or 0
    if total_events > 0 and total_repeats == 0:
        warnings.warn(
            "interactions file contains no repeated (user_id, item_id) rows — "
            "if it is pre-aggregated, the per-user table carries no signal and "
            "every user gets the prior rate",
            stacklevel=3,
        )
    p_global = total_repeats / total_events if total_events else 0.0

    table: dict[str, float] = {}
    for user, events, repeats in per_user.iter_rows():
        p_u = (repeats + prior_strength * p_global) / (events + prior_strength)
        table[user] = scale * (1.0 - p_u)
    return table
