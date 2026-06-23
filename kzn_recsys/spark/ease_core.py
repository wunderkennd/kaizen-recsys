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


def solve_from_gram(G: np.ndarray, lambda_: float) -> np.ndarray:
    """Given the dense Gram G ((M+K)x(M+K)), return S (Fortran-order).

    G is modified in place (lambda added to diagonal). Mirrors model.rs:190-224.
    """
    total = G.shape[0]
    G.flat[:: total + 1] += lambda_
    P = np.linalg.inv(G)
    p_jj = np.diag(P).copy()
    inv = np.where(np.abs(p_jj) > 1e-12, -1.0 / p_jj, 0.0)
    S = P * inv[None, :]
    np.fill_diagonal(S, 0.0)
    return np.asfortranarray(S)


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

    # Column-major to match nalgebra / FEAS layout (parity fact 1).
    return solve_from_gram(G, params.lambda_)


def predict_scores(S, num_items, num_user_features, interactions, features, beta) -> np.ndarray:
    """Score all items for one user. Mirrors RustFeaseModel::predict (model.rs:341-378).

    interactions: list of (item_idx, value); features: list of (feature_idx, value).
    Returns a length-`num_items` float64 array.
    """
    total = num_items + num_user_features
    z = np.zeros(total, dtype=np.float64)
    for item_idx, val in interactions:
        if 0 <= item_idx < num_items:
            z[item_idx] = val
    for feat_idx, val in features:
        if 0 <= feat_idx < num_user_features:
            z[num_items + feat_idx] = val * beta
    return (S @ z)[:num_items]


def predict_similar_items(S, item_idx, num_items, top_k) -> list:
    """Item-item similarity from the S item block. Mirrors model.rs:390+.

    Returns up to top_k (item_index, score) pairs sorted by descending score,
    excluding item_idx itself.
    """
    if item_idx < 0 or item_idx >= num_items:
        return []
    col = np.asarray(S[:num_items, item_idx]).ravel()
    order = np.argsort(-col, kind="stable")
    out = []
    for j in order:
        if int(j) == item_idx:
            continue
        out.append((int(j), float(col[j])))
        if len(out) == top_k:
            break
    return out


def prune_sparse(S, threshold) -> None:
    """Zero entries with |value| < threshold, in place. Mirrors model.rs:233-248.

    S must be a dense numpy array (train_ease returns one); scipy sparse inputs
    are not supported.
    """
    S[np.abs(S) < threshold] = 0.0
