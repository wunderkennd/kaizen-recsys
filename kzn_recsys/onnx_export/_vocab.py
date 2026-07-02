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
