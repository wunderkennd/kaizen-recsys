"""Pure NumPy/SciPy EASE math. Mirrors src/model.rs.

No pyspark import here — this module is Spark-free and fast to test.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class EaseParams:
    """EASE hyperparameters. Defaults match the Rust core / fease_wrapper."""
    alpha: float = 1.0       # item-feature weight
    beta: float = 1.0        # user-feature weight
    lambda_: float = 150.0   # L2 regularization
    meta_weight: float = 0.0  # diagonal metadata weighting; 0 => treated as 1.0


def train_ease(X, U, T, params: EaseParams) -> np.ndarray:
    """Train EASE, returning the S matrix as a Fortran-order (M+K)x(M+K) array.

    Mirrors RustFeaseModel::train (src/model.rs:87-228).

    Args:
        X: (N x M) users x items, scipy CSR
        U: (N x K) users x user-features, scipy CSR
        T: (L x M) item-features x items, scipy CSR
        params: EaseParams
    """
    M = X.shape[1]
    K = U.shape[1]
    total = M + K

    w = params.meta_weight if params.meta_weight > 0.0 else 1.0
    a, b = params.alpha, params.beta

    # Gram blocks (model.rs:130-145)
    XtX = (X.T @ X)                       # M x M
    TtT = (T.T @ T)                       # M x M
    G11 = (XtX + w * a * a * TtT).toarray()
    G12 = (b * (X.T @ U)).toarray()       # M x K
    G21 = (b * (U.T @ X)).toarray()       # K x M
    G22 = (b * b * (U.T @ U)).toarray()   # K x K

    # Assemble dense G (model.rs:147-187)
    G = np.zeros((total, total), dtype=np.float64)
    G[:M, :M] = G11
    if K > 0:
        G[:M, M:] = G12
        G[M:, :M] = G21
        G[M:, M:] = G22

    # P = inv(G + lambda I)  (model.rs:190-200)
    G.flat[:: total + 1] += params.lambda_  # add lambda to diagonal in place
    P = np.linalg.inv(G)

    # S[i,j] = -P[i,j] / P[j,j], S[j,j] = 0  (model.rs:202-224)
    p_jj = np.diag(P).copy()
    inv = np.where(np.abs(p_jj) > 1e-12, -1.0 / p_jj, 0.0)
    S = P * inv[None, :]
    np.fill_diagonal(S, 0.0)

    # Column-major to match nalgebra / FEAS layout (parity fact 1).
    return np.asfortranarray(S)
