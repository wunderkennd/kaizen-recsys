"""Authoring of the EASE ONNX graph (vanilla ai.onnx, opset 17)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from . import EXCLUDE_SENTINEL, MASK_PENALTY, OPSET

# Items are considered hard-excluded by repeat_penalty when it exceeds this threshold.
# EXCLUDE_SENTINEL == 1e9, so 5e8 is safely above any legitimate boost value.
_SEEN_EXCLUDE_THRESHOLD = EXCLUDE_SENTINEL / 2.0


def build_graph(payload, onnx_path: Path, *, top_k_default: int, repeat_penalty_default: float) -> None:
    M = payload.num_items
    K = payload.num_user_features

    # β-fold: pre-multiply the user-feature columns of S_items by β so the graph
    # consumes raw feature values. W is the Gemm weight, stored [M, M+K].
    W = payload.s_items.astype(np.float64).copy()
    if K > 0:
        W[:, M:] *= payload.beta
    W = W.astype(np.float32)

    initializers = [
        numpy_helper.from_array(W, name="W"),
        numpy_helper.from_array(np.ones((1, M), np.float32), name="mask"),
        numpy_helper.from_array(np.zeros((1, M), np.float32), name="seen"),
        numpy_helper.from_array(np.array([[repeat_penalty_default]], np.float32), name="repeat_penalty"),
        numpy_helper.from_array(np.array([top_k_default], np.int64), name="k"),
        numpy_helper.from_array(np.array(0.0, np.float32), name="zero_const"),
        numpy_helper.from_array(np.array(1.0, np.float32), name="one_const"),
        numpy_helper.from_array(np.array(0.5, np.float32), name="half_const"),
        numpy_helper.from_array(np.array(MASK_PENALTY, np.float32), name="mask_penalty_const"),
        numpy_helper.from_array(np.array([M], np.int64), name="M_const"),
        numpy_helper.from_array(np.array(-np.inf, np.float32), name="neg_inf_const"),
        numpy_helper.from_array(np.array(_SEEN_EXCLUDE_THRESHOLD, np.float32), name="seen_excl_thresh"),
        numpy_helper.from_array(np.array([-1], np.int64), name="reduce_axis"),
    ]

    inputs = [
        helper.make_tensor_value_info("interactions", TensorProto.FLOAT, ["batch", M]),
        helper.make_tensor_value_info("features", TensorProto.FLOAT, ["batch", K]),
        helper.make_tensor_value_info("mask", TensorProto.FLOAT, ["batch", M]),
        helper.make_tensor_value_info("seen", TensorProto.FLOAT, ["batch", M]),
        helper.make_tensor_value_info("repeat_penalty", TensorProto.FLOAT, ["batch", 1]),
        helper.make_tensor_value_info("k", TensorProto.INT64, [1]),
    ]
    outputs = [
        helper.make_tensor_value_info("top_indices", TensorProto.INT64, ["batch", "kc"]),
        helper.make_tensor_value_info("top_scores", TensorProto.FLOAT, ["batch", "kc"]),
        helper.make_tensor_value_info("raw_scores", TensorProto.FLOAT, ["batch", M]),
    ]

    nodes = [
        helper.make_node("Concat", ["interactions", "features"], ["z"], axis=-1),
        # raw_scores = z @ Wᵀ  (also a graph output)
        helper.make_node("Gemm", ["z", "W"], ["raw_scores"], transB=1),
        # seen_eff = max(seen, cast(interactions != 0))
        helper.make_node("Equal", ["interactions", "zero_const"], ["is_zero"]),
        helper.make_node("Not", ["is_zero"], ["is_nonzero"]),
        helper.make_node("Cast", ["is_nonzero"], ["nz_f"], to=TensorProto.FLOAT),
        helper.make_node("Max", ["seen", "nz_f"], ["seen_eff"]),
        # adjusted = raw_scores - repeat_penalty * seen_eff
        helper.make_node("Mul", ["repeat_penalty", "seen_eff"], ["penalty_term"]),
        helper.make_node("Sub", ["raw_scores", "penalty_term"], ["adjusted"]),
        # soft_masked = adjusted + (mask - 1) * MASK_PENALTY
        helper.make_node("Sub", ["mask", "one_const"], ["mask_minus_one"]),
        helper.make_node("Mul", ["mask_minus_one", "mask_penalty_const"], ["mask_term"]),
        helper.make_node("Add", ["adjusted", "mask_term"], ["soft_masked"]),
        # --- Hard exclusion: replace score with -inf for items that must be absent ---
        # hard_mask_excl: mask == 0  →  (mask < 0.5)
        helper.make_node("Less", ["mask", "half_const"], ["hard_mask_excl"]),
        # hard_seen_excl: (seen_eff > 0.5) AND (repeat_penalty > seen_excl_thresh)
        # repeat_penalty shape [batch,1] broadcasts against seen_eff [batch,M]
        helper.make_node("Greater", ["seen_eff", "half_const"], ["seen_eff_bool"]),
        helper.make_node("Greater", ["repeat_penalty", "seen_excl_thresh"], ["rp_above_thresh"]),
        helper.make_node("And", ["seen_eff_bool", "rp_above_thresh"], ["hard_seen_excl"]),
        # hard_excl = hard_mask_excl OR hard_seen_excl
        helper.make_node("Or", ["hard_mask_excl", "hard_seen_excl"], ["hard_excl"]),
        # masked = Where(hard_excl, -inf, soft_masked)
        helper.make_node("Where", ["hard_excl", "neg_inf_const", "soft_masked"], ["masked"]),
        # k_avail = sum(NOT hard_excl, axis=-1), shape [batch]
        helper.make_node("Not", ["hard_excl"], ["avail_bool"]),
        helper.make_node("Cast", ["avail_bool"], ["avail_f"], to=TensorProto.FLOAT),
        # ReduceSum in opset 13+ takes axes as a 1D input tensor, not an attribute.
        helper.make_node("ReduceSum", ["avail_f", "reduce_axis"], ["avail_sum"], keepdims=0),
        helper.make_node("Cast", ["avail_sum"], ["avail_int"], to=TensorProto.INT64),
        # kc = min(k, M, k_avail); TopK
        helper.make_node("Min", ["k", "M_const"], ["k_clamped_M"]),
        helper.make_node("Min", ["k_clamped_M", "avail_int"], ["kc"]),
        helper.make_node(
            "TopK", ["masked", "kc"], ["top_scores", "top_indices"], axis=-1, largest=1, sorted=1
        ),
    ]

    graph = helper.make_graph(nodes, "ease_onnx", inputs, outputs, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = onnx.IR_VERSION  # 13 for onnx 1.21; compatible with onnxruntime 1.26
    onnx.checker.check_model(model)
    onnx.save(model, str(onnx_path))
