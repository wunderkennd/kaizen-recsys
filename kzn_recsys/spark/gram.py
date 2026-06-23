"""Gram-matrix strategies feeding ease_core.train_ease.

Phase 1: collect-to-driver (gram_collect) builds CSR on the driver from Spark
frames and trains via ease_core. Phase 2 adds gram_distributed.
"""
from __future__ import annotations

import numpy as np

from . import dataframes as _df
from . import ease_core as _core
from .ease_core import solve_from_gram


def gram_collect(interactions_df, user_features_df, item_features_df, mappings, params, weighting):
    """Collect-to-driver: build (X,U,T) on the driver and train EASE.

    Returns the trained S matrix (Fortran-order ndarray).
    """
    X, U, T = _df.build_csr_inputs(
        interactions_df, user_features_df, item_features_df, mappings, weighting
    )
    return _core.train_ease(X, U, T, params)


def _block_from_cooccurrence(df, left_key, left_val, right_key, right_val,
                             left_map, right_map, n_left, n_right):
    """Compute a dense (n_left x n_right) block = sum over the join key of
    (left_val * right_val), via a Spark self/cross join on a shared entity key.

    `df` already has columns: join_entity, plus the named value/index columns.
    Returns a dense numpy array.
    """
    from pyspark.sql import functions as F
    agg = (df.groupBy(left_key, right_key)
             .agg(F.sum(F.col(left_val) * F.col(right_val)).alias("v")))
    block = np.zeros((n_left, n_right), dtype=np.float64)
    for r in agg.collect():
        li = left_map.get(r[left_key])
        ri = right_map.get(r[right_key])
        if li is not None and ri is not None:
            block[li, ri] = r["v"]
    return block


def gram_distributed(interactions_df, user_features_df, item_features_df,
                     mappings, params, weighting):
    """Compute the four Gram blocks as Spark aggregations; solve on the driver.

    Mirrors the block algebra of model.rs:130-187 but accumulates ZᵀZ in Spark.
    Only the dense (M+K)² Gram crosses to the driver — memory is independent of N.
    """
    from pyspark.sql import functions as F

    m = mappings
    M = len(m.idx_to_item)
    K = len(m.idx_to_user_feature)
    a, b = params.alpha, params.beta
    w = params.meta_weight if params.meta_weight > 0.0 else 1.0

    idf = interactions_df
    if weighting is not None:
        idf = _df.apply_weighting(idf, weighting, m)

    # XtX[i,j] = sum over users of value(u,i)*value(u,j): self-join interactions on user_id
    left = idf.select(F.col("item_id").alias("li"), F.col("value").alias("lv"),
                      F.col("user_id").alias("uk"))
    right = idf.select(F.col("item_id").alias("ri"), F.col("value").alias("rv"),
                       F.col("user_id").alias("uk2"))
    xtx_df = left.join(right, left.uk == right.uk2)
    XtX = _block_from_cooccurrence(xtx_df, "li", "lv", "ri", "rv",
                                   m.item_to_idx, m.item_to_idx, M, M)

    # TtT[i,j]: for items i,j sharing item-feature f: value(f,i)*value(f,j)
    tf = item_features_df.select(F.col("item_id").alias("it"),
                                 F.col("feature_name").alias("fn"),
                                 F.col("value").alias("fv"))
    tleft = tf.select(F.col("it").alias("li"), F.col("fv").alias("lv"), F.col("fn").alias("fk"))
    tright = tf.select(F.col("it").alias("ri"), F.col("fv").alias("rv"), F.col("fn").alias("fk2"))
    ttt_df = tleft.join(tright, tleft.fk == tright.fk2)
    TtT = _block_from_cooccurrence(ttt_df, "li", "lv", "ri", "rv",
                                   m.item_to_idx, m.item_to_idx, M, M)

    # XtU[i,k] = sum over users of value(u,i)*ufeat(u,k): join interactions to user features
    XtU = np.zeros((M, K), dtype=np.float64)
    UtU = np.zeros((K, K), dtype=np.float64)
    if K > 0:
        uf = user_features_df.select(F.col("user_id").alias("uk2"),
                                     F.col("feature_name").alias("fn"),
                                     F.col("value").alias("fv"))
        xtu_df = (idf.select(F.col("item_id").alias("li"), F.col("value").alias("lv"),
                             F.col("user_id").alias("uk"))
                     .join(uf, F.col("uk") == F.col("uk2")))
        XtU = _block_from_cooccurrence(xtu_df, "li", "lv", "fn", "fv",
                                       m.item_to_idx, m.user_feature_to_idx, M, K)

        # UtU[k,l] = sum over users of ufeat(u,k)*ufeat(u,l): self-join user features on user_id
        uleft = user_features_df.select(F.col("user_id").alias("uk"),
                                        F.col("feature_name").alias("lk"),
                                        F.col("value").alias("lv"))
        uright = user_features_df.select(F.col("user_id").alias("uk2"),
                                         F.col("feature_name").alias("rk"),
                                         F.col("value").alias("rv"))
        utu_df = uleft.join(uright, uleft.uk == uright.uk2)
        UtU = _block_from_cooccurrence(utu_df, "lk", "lv", "rk", "rv",
                                       m.user_feature_to_idx, m.user_feature_to_idx, K, K)

    total = M + K
    G = np.zeros((total, total), dtype=np.float64)
    G[:M, :M] = XtX + w * a * a * TtT
    if K > 0:
        G[:M, M:] = b * XtU
        G[M:, :M] = b * XtU.T
        G[M:, M:] = b * b * UtU
    return solve_from_gram(G, params.lambda_)
