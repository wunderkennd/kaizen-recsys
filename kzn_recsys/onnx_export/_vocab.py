"""Vocab/sidecar JSON writer — stub (Task 6 will implement fully)."""
from __future__ import annotations

import json
from pathlib import Path


def write_vocab(
    payload,
    vocab_path: Path,
    *,
    top_k_default: int,
    dtype: str,
    repeat_penalty_default: float,
) -> None:
    """Write a minimal vocab.json sidecar alongside the ONNX model."""
    vocab = {
        "kind": payload.kind,
        "num_items": payload.num_items,
        "num_user_features": payload.num_user_features,
        "num_item_features": payload.num_item_features,
        "top_k_default": top_k_default,
        "dtype": dtype,
        "repeat_penalty_default": repeat_penalty_default,
        "item_index_to_guid": payload.item_index_to_guid,
        "feature_name_to_index": payload.feature_name_to_index,
    }
    Path(vocab_path).write_text(json.dumps(vocab, indent=2))
