"""Writes the language-neutral vocab.json sidecar (spec §6)."""
from __future__ import annotations

import json
from pathlib import Path

from . import EXCLUDE_SENTINEL, MASK_PENALTY, OPSET


def write_vocab(
    payload,
    vocab_path: Path,
    *,
    top_k_default: int,
    dtype: str,
    repeat_penalty_default: float,
    per_user_table: dict[str, float] | None = None,
) -> None:
    M, K = payload.num_items, payload.num_user_features
    repeat_policy = {
        "default_penalty": repeat_penalty_default,
        "per_user_table_present": per_user_table is not None,
    }
    if per_user_table is not None:
        # Tier C (spec §10): learned user→ρ, applied by the pyfunc wrapper
        # when the caller doesn't override repeat_penalty per row.
        repeat_policy["per_user_table"] = per_user_table
    vocab = {
        "format_version": 1,
        "model_kind": payload.kind,
        "num_items": M,
        "num_user_features": K,
        "num_item_features": payload.num_item_features,
        "beta": payload.beta,
        "weight_dtype": dtype,
        "opset": OPSET,
        "top_k_default": top_k_default,
        "constants": {"MASK_PENALTY": MASK_PENALTY, "EXCLUDE_SENTINEL": EXCLUDE_SENTINEL},
        "repeat_policy": repeat_policy,
        "io_signature": {
            "inputs": [
                {"name": "interactions", "dtype": "float32", "shape": ["batch", M], "required": True},
                {"name": "features", "dtype": "float32", "shape": ["batch", K], "required": True},
                {"name": "mask", "dtype": "float32", "shape": ["batch", M], "required": False, "default": "all-ones"},
                {"name": "seen", "dtype": "float32", "shape": ["batch", M], "required": False, "default": "all-zeros"},
                {"name": "repeat_penalty", "dtype": "float32", "shape": ["batch", 1], "required": False, "default": "EXCLUDE_SENTINEL"},
                {"name": "k", "dtype": "int64", "shape": [1], "required": False, "default": "top_k_default"},
            ],
            "outputs": [
                {"name": "top_indices", "dtype": "int64", "shape": ["batch", "kc"]},
                {"name": "top_scores", "dtype": "float32", "shape": ["batch", "kc"]},
                {"name": "raw_scores", "dtype": "float32", "shape": ["batch", M]},
            ],
        },
        "item_index_to_guid": payload.item_index_to_guid,
        "feature_name_to_index": payload.feature_name_to_index,
        "provenance": {
            "alpha": payload.alpha,
            "lambda_": payload.lambda_,
            "meta_weight": payload.meta_weight,
            "sparsity_threshold": payload.sparsity_threshold,
            "num_item_features": payload.num_item_features,
        },
    }
    vocab_path.write_text(json.dumps(vocab, indent=2))


def write_vocab_two_tower(
    payload,
    vocab_path: Path,
    *,
    top_k_default: int,
    dtype: str,
    repeat_penalty_default: float,
) -> None:
    """vocab.json for the Two-Tower tower-input contract (#85).

    Differs from EASE: inputs are ``user_idx`` (+ ``cat_ids``/``cat_mask``
    and/or ``dense`` when trained with user features), the id → row map is
    ``user_id_to_index`` (index 0 = cold-start sentinel, absent from the
    map), and the repeat-policy default is neutral (ρ = 0) since Two-Tower
    carries no per-request history.
    """
    M = payload.num_items
    inputs = [
        {
            "name": "user_idx",
            "dtype": "int64",
            "shape": ["batch"],
            "required": True,
            "note": "pass cold_start_user_index (0) for unknown users",
        }
    ]
    if payload.has_cat:
        inputs += [
            {"name": "cat_ids", "dtype": "int64", "shape": ["batch", "C"], "required": False, "default": "no categorical features"},
            {"name": "cat_mask", "dtype": "float32", "shape": ["batch", "C"], "required": False, "default": "no categorical features"},
        ]
    if payload.has_dense:
        inputs.append(
            {"name": "dense", "dtype": "float32", "shape": ["batch", payload.user_dense_dim], "required": False, "default": "all-zeros"}
        )
    inputs += [
        {"name": "mask", "dtype": "float32", "shape": ["batch", M], "required": False, "default": "all-ones"},
        {"name": "seen", "dtype": "float32", "shape": ["batch", M], "required": False, "default": "all-zeros"},
        {"name": "repeat_penalty", "dtype": "float32", "shape": ["batch", 1], "required": False, "default": "repeat_policy.default_penalty"},
        {"name": "k", "dtype": "int64", "shape": [1], "required": False, "default": "top_k_default"},
    ]
    vocab = {
        "format_version": 1,
        "model_kind": payload.kind,
        "num_items": M,
        "num_users": payload.num_users,
        "embedding_dim": payload.embedding_dim,
        "has_cat": payload.has_cat,
        "has_dense": payload.has_dense,
        "num_user_categories": payload.num_user_categories,
        "user_dense_dim": payload.user_dense_dim,
        "weight_dtype": dtype,
        "opset": OPSET,
        "top_k_default": top_k_default,
        "constants": {"MASK_PENALTY": MASK_PENALTY, "EXCLUDE_SENTINEL": EXCLUDE_SENTINEL},
        # Neutral by default (0.0): Two-Tower has no per-request history to
        # derive "seen" from, so nothing is excluded unless the caller feeds
        # an explicit `seen` vector + penalty (issue #70/#85).
        "repeat_policy": {
            "default_penalty": repeat_penalty_default,
            "per_user_table_present": False,
        },
        "io_signature": {
            "inputs": inputs,
            "outputs": [
                {"name": "top_indices", "dtype": "int64", "shape": ["batch", "kc"]},
                {"name": "top_scores", "dtype": "float32", "shape": ["batch", "kc"]},
                {"name": "raw_scores", "dtype": "float32", "shape": ["batch", M]},
            ],
        },
        "item_index_to_guid": payload.item_index_to_guid,
        "user_id_to_index": payload.user_id_to_index,
        "cold_start_user_index": 0,
        "user_cat_feature_to_idx": payload.user_cat_feature_to_idx,
        "user_dense_feature_to_idx": payload.user_dense_feature_to_idx,
    }
    vocab_path.write_text(json.dumps(vocab, indent=2))
