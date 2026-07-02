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


from kzn_recsys.spark.ease_core import predict_scores, predict_similar_items, prune_sparse


def test_predict_scores_against_S_at_z():
    X, U, T = _toy_inputs()
    S = train_ease(X, U, T, EaseParams(lambda_=10.0))
    # User with items 0 and 1
    interactions = [(0, 1.0), (1, 1.0)]
    scores = predict_scores(S, num_items=3, num_user_features=0,
                            interactions=interactions, features=[], beta=1.0)
    # Reference: z = [1,1,0], scores = (S @ z)[:3]
    z = np.array([1.0, 1.0, 0.0])
    assert np.allclose(scores, (S @ z)[:3], atol=1e-12)
    assert scores.shape == (3,)


def test_predict_scores_applies_beta_to_features():
    # 1 item, 1 user-feature: total dim 2
    S = np.asfortranarray(np.array([[0.0, 0.5], [0.7, 0.0]]))
    scores = predict_scores(S, num_items=1, num_user_features=1,
                            interactions=[(0, 2.0)], features=[(0, 3.0)], beta=0.5)
    # z = [2.0, 0.5*3.0] = [2.0, 1.5]; score_item0 = 0.0*2.0 + 0.5*1.5 = 0.75
    assert np.allclose(scores, [0.75], atol=1e-12)


def test_predict_similar_items_excludes_self_and_sorts():
    S = np.asfortranarray(np.array([
        [0.0, 0.9, 0.1],
        [0.9, 0.0, 0.5],
        [0.1, 0.5, 0.0],
    ]))
    out = predict_similar_items(S, item_idx=0, num_items=3, top_k=2)
    assert out[0][0] == 1  # highest column-0 score, excluding self
    assert out[1][0] == 2
    assert all(idx != 0 for idx, _ in out)


def test_prune_sparse_zeros_small_entries():
    S = np.asfortranarray(np.array([[0.0, 0.001], [0.5, 0.0]]))
    prune_sparse(S, threshold=0.01)
    assert S[0, 1] == 0.0
    assert S[1, 0] == 0.5


def test_predict_similar_items_out_of_range_returns_empty():
    S = np.asfortranarray(np.zeros((3, 3)))
    assert predict_similar_items(S, item_idx=5, num_items=3, top_k=2) == []
    assert predict_similar_items(S, item_idx=-1, num_items=3, top_k=2) == []
