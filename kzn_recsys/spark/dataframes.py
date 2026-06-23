"""Spark DataFrame -> EASE matrix inputs. Mirrors src/data_pipeline.rs.

Index assignment uses sorted-distinct order (deterministic, self-consistent);
it does not reproduce Rust's first-seen order, which is unnecessary because the
mappings are persisted alongside S (see plan parity fact 2).
"""
from __future__ import annotations

from dataclasses import dataclass

import scipy.sparse as sp


@dataclass
class Mappings:
    user_to_idx: dict
    idx_to_user: list
    item_to_idx: dict
    idx_to_item: list
    user_feature_to_idx: dict
    idx_to_user_feature: list
    item_feature_to_idx: dict
    idx_to_item_feature: list


def _sorted_distinct(*dfs_cols):
    """Collect the sorted-distinct union of (df, column) pairs as a list of str."""
    seen = set()
    for df, col in dfs_cols:
        for row in df.select(col).where(f"{col} is not null").distinct().collect():
            seen.add(row[0])
    return sorted(seen)


def _index(values):
    return {v: i for i, v in enumerate(values)}


def build_mappings(interactions_df, user_features_df, item_features_df) -> Mappings:
    users = _sorted_distinct((interactions_df, "user_id"), (user_features_df, "user_id"))
    items = _sorted_distinct((interactions_df, "item_id"), (item_features_df, "item_id"))
    ufeat = _sorted_distinct((user_features_df, "feature_name"))
    ifeat = _sorted_distinct((item_features_df, "feature_name"))
    return Mappings(
        user_to_idx=_index(users), idx_to_user=users,
        item_to_idx=_index(items), idx_to_item=items,
        user_feature_to_idx=_index(ufeat), idx_to_user_feature=ufeat,
        item_feature_to_idx=_index(ifeat), idx_to_item_feature=ifeat,
    )


def _triplets(df, row_col, col_col, row_map, col_map):
    """Collect (row_idx, col_idx, value) for rows whose keys are in both maps."""
    rows, cols, vals = [], [], []
    for r in df.select(row_col, col_col, "value").collect():
        rk, ck, v = r[0], r[1], r[2]
        if rk in row_map and ck in col_map and v is not None:
            rows.append(row_map[rk])
            cols.append(col_map[ck])
            vals.append(float(v))
    return rows, cols, vals


def build_csr_inputs(interactions_df, user_features_df, item_features_df, mappings, weighting):
    """Build (X, U, T) CSR matrices. `weighting` is an optional WeightingConfig.

    X: (N x M), U: (N x K), T: (L x M).  Weighting (event->decay->IPS) is applied
    to interaction values before X is assembled (see apply_weighting in Task 7).
    """
    m = mappings
    N, M = len(m.idx_to_user), len(m.idx_to_item)
    K, L = len(m.idx_to_user_feature), len(m.idx_to_item_feature)

    idf = interactions_df
    if weighting is not None:
        idf = apply_weighting(idf, weighting, m)

    xr, xc, xv = _triplets(idf, "user_id", "item_id", m.user_to_idx, m.item_to_idx)
    ur, uc, uv = _triplets(user_features_df, "user_id", "feature_name",
                           m.user_to_idx, m.user_feature_to_idx)
    tr, tc, tv = _triplets(item_features_df, "item_id", "feature_name",
                           m.item_to_idx, m.item_feature_to_idx)

    X = sp.csr_matrix((xv, (xr, xc)), shape=(N, M))
    U = sp.csr_matrix((uv, (ur, uc)), shape=(N, K))
    # build (M x L) then transpose to (L x M) to match data_pipeline.rs:200-204
    T_ml = sp.csr_matrix((tv, (tr, tc)), shape=(M, L))
    T = T_ml.transpose().tocsr()
    return X, U, T
