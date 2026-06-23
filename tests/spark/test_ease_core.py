import numpy as np
import pytest
import scipy.sparse as sp

from kzn_recsys.spark.ease_core import EaseParams
from kzn_recsys.spark.ease_core import train_ease


def test_ease_params_defaults():
    p = EaseParams()
    assert p.alpha == 1.0
    assert p.beta == 1.0
    assert p.lambda_ == 150.0
    assert p.meta_weight == 0.0


def test_ease_params_explicit():
    p = EaseParams(alpha=2.0, beta=0.5, lambda_=100.0, meta_weight=1.0)
    assert (p.alpha, p.beta, p.lambda_, p.meta_weight) == (2.0, 0.5, 100.0, 1.0)


def _toy_inputs():
    # 3 users, 3 items, no features (K=0, L=0)
    # User 0: items 0,1 ; User 1: items 1,2 ; User 2: items 0,2
    X = sp.csr_matrix(
        np.array([[1.0, 1.0, 0.0],
                  [0.0, 1.0, 1.0],
                  [1.0, 0.0, 1.0]])
    )
    U = sp.csr_matrix((3, 0))   # N x K, K=0
    T = sp.csr_matrix((0, 3))   # L x M, L=0
    return X, U, T


def test_train_shapes_and_zero_diagonal():
    X, U, T = _toy_inputs()
    S = train_ease(X, U, T, EaseParams(lambda_=10.0))
    # (M+K) x (M+K) == 3 x 3
    assert S.shape == (3, 3)
    # zero diagonal constraint
    assert np.allclose(np.diag(S), 0.0)
    # column-major storage
    assert S.flags["F_CONTIGUOUS"]


def test_train_matches_direct_formula():
    X, U, T = _toy_inputs()
    lam = 10.0
    S = train_ease(X, U, T, EaseParams(alpha=1.0, beta=1.0, lambda_=lam))
    # Reference: G = X^T X (no features), P = inv(G + lam I), B = -P / diag(P), zero diag
    G = (X.T @ X).toarray()
    P = np.linalg.inv(G + lam * np.eye(3))
    B = -P / np.diag(P)[None, :]
    np.fill_diagonal(B, 0.0)
    assert np.allclose(S, B, atol=1e-9)
