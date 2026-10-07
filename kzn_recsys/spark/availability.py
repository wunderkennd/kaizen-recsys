"""Availability-aware evaluation helpers (issue #101), pure Python.

Mirrors ``src/evaluation/availability.rs``: an availability table turns into
a per-user *eligible* item set at that user's *reference time*, and the
harness ranks only eligible items, drops ineligible relevants, and reports
coverage against the eligible catalog.

Table columns (long format): ``item_id`` (series id), optional ``territory``
(``"*"`` or missing = global), ``available_from_days_ago`` (float), optional
nullable ``available_to_days_ago`` (float; null = still available). Any other
column (``season_id`` …) is ignored; several rows per item roll up to the
item.

Window semantics in ``days_ago`` units (larger = older): a window contains
reference time ``r`` when ``from >= r`` and (``to`` is null or ``to < r``).

Territory: rows tagged ``"*"`` apply to everyone. A user's territory is the
``<value>`` suffix of the first user feature named ``<column>_<value>`` with
a positive value (the ingest's one-hot convention). Users without one see
global rows only. Items with no row are never eligible and are counted in
``num_items_without_availability``.
"""
from __future__ import annotations

import math

GLOBAL_TERRITORY = "*"
DEFAULT_ITEM_AGE_BUCKET_EDGES = (30.0, 365.0)


class AvailabilityTable:
    """Eligibility lookups keyed by item *id* (strings), built from rows."""

    def __init__(self):
        self._windows = {}          # item_id -> territory -> [(from, to|None)]
        self._first_available = {}  # item_id -> max(from)
        self.num_rows_unknown_item = 0
        self.num_items_unlisted = 0

    @classmethod
    def from_rows(cls, rows, known_items):
        """``rows``: iterable of mappings with the table's columns.
        ``known_items``: the model's item ids (rows for other ids are skipped)."""
        known = set(known_items)
        t = cls()
        for i, r in enumerate(rows):
            item = r.get("item_id")
            if item is None:
                continue
            if item not in known:
                t.num_rows_unknown_item += 1
                continue
            frm = r.get("available_from_days_ago")
            if frm is None:
                raise ValueError(f"availability row {i}: null available_from_days_ago")
            frm = float(frm)
            if not math.isfinite(frm) or frm < 0.0:
                raise ValueError(f"availability row {i}: available_from_days_ago={frm}")
            to = r.get("available_to_days_ago")
            if to is not None:
                to = float(to)
                if not math.isfinite(to) or to < 0.0 or to > frm:
                    raise ValueError(
                        f"availability row {i}: window ends before it starts "
                        f"(available_from_days_ago={frm}, available_to_days_ago={to})"
                    )
            territory = r.get("territory")
            if territory is None:
                territory = GLOBAL_TERRITORY
            t._windows.setdefault(item, {}).setdefault(territory, []).append((frm, to))
            prev = t._first_available.get(item)
            if prev is None or frm > prev:
                t._first_available[item] = frm
        t.num_items_unlisted = len(known - set(t._windows))
        return t

    @classmethod
    def from_spark(cls, availability_df, known_items):
        cols = [c for c in ("item_id", "territory", "available_from_days_ago",
                            "available_to_days_ago") if c in availability_df.columns]
        if "item_id" not in cols:
            raise ValueError("availability table: missing `item_id` column")
        if "available_from_days_ago" not in cols:
            raise ValueError("availability table: missing `available_from_days_ago` column")
        rows = [r.asDict() for r in availability_df.select(*cols).collect()]
        return cls.from_rows(rows, known_items)

    @staticmethod
    def _contains(window, reference_days_ago):
        frm, to = window
        return frm >= reference_days_ago and (to is None or to < reference_days_ago)

    def is_eligible(self, item_id, territory, reference_days_ago):
        by_t = self._windows.get(item_id)
        if not by_t:
            return False

        def hit(key):
            return any(self._contains(w, reference_days_ago) for w in by_t.get(key, ()))

        return hit(GLOBAL_TERRITORY) or (
            territory is not None and territory != GLOBAL_TERRITORY and hit(territory)
        )

    def eligible_set(self, territory, reference_days_ago):
        return {i for i in self._windows if self.is_eligible(i, territory, reference_days_ago)}

    def item_age_days(self, item_id, reference_days_ago):
        first = self._first_available.get(item_id)
        return None if first is None else first - reference_days_ago


def user_territories(feats_by_user, feature_col):
    """``{user_id: territory}`` from ``{user_id: {feature_name: value}}``
    using the ``<feature_col>_<value>`` one-hot convention."""
    prefix = f"{feature_col}_"
    out = {}
    for uid, feats in feats_by_user.items():
        for name, value in feats.items():
            if value > 0.0 and name.startswith(prefix) and len(name) > len(prefix):
                out[uid] = name[len(prefix):]
                break
    return out


def age_bucket(age, edges):
    """``age < edges[0]`` -> 0, ``edges[i-1] <= age < edges[i]`` -> i,
    ``age >= edges[-1]`` -> ``len(edges)``."""
    n = 0
    for e in edges:
        if age >= e:
            n += 1
        else:
            break
    return n


def bucket_bounds(edges):
    """``[(label, min_age, max_age|None), ...]`` for ``edges``."""
    if not edges:
        return []

    def fmt(d):
        return str(int(d)) if float(d).is_integer() else str(d)

    out = [(f"<{fmt(edges[0])}d", 0.0, float(edges[0]))]
    for lo, hi in zip(edges, edges[1:]):
        out.append((f"{fmt(lo)}-{fmt(hi)}d", float(lo), float(hi)))
    out.append((f">={fmt(edges[-1])}d", float(edges[-1]), None))
    return out


def check_edges(edges):
    edges = [float(e) for e in edges]
    if edges and (not math.isfinite(edges[0]) or edges[0] < 0.0):
        raise ValueError("item_age_bucket_edges must be finite and >= 0")
    for lo, hi in zip(edges, edges[1:]):
        if not lo < hi:
            raise ValueError(f"item_age_bucket_edges must be strictly ascending, got {edges}")
    return edges
