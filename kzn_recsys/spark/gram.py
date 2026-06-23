"""Gram-matrix strategies feeding ease_core.train_ease.

Phase 1: collect-to-driver (gram_collect) builds CSR on the driver from Spark
frames and trains via ease_core. Phase 2 adds gram_distributed.
"""
from __future__ import annotations

from . import dataframes as _df
from . import ease_core as _core


def gram_collect(interactions_df, user_features_df, item_features_df, mappings, params, weighting):
    """Collect-to-driver: build (X,U,T) on the driver and train EASE.

    Returns the trained S matrix (Fortran-order ndarray).
    """
    X, U, T = _df.build_csr_inputs(
        interactions_df, user_features_df, item_features_df, mappings, weighting
    )
    return _core.train_ease(X, U, T, params)
