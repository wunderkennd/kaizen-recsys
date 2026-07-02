"""Authoring of the Two-Tower ONNX graph (vanilla ai.onnx, opset 17).

Emits the user-tower forward pass (issue #85) — mirroring
``TrainedTwoTower::score_user`` in ``src/models/two_tower.rs`` exactly:

    h  = id_embedding[user_idx]                      (Gather)
    h += mean-pool(cat_embedding[cat_ids], cat_mask)   (if has_cat)
    h += dense @ W_dense + b_dense                     (if has_dense)
    h  = relu(h @ W_hidden + b_hidden)
    z  = h @ W_out + b_out
    z  = z / max(||z||_2, 1e-12)
    raw_scores = z @ item_matrixᵀ

then reuses ``_graph.py``'s repeat-penalty / eligibility-mask / TopK tail
with the same output names (``top_indices``, ``top_scores``, ``raw_scores``).
The only tail difference from EASE: there is no ``interactions`` input to
derive "seen" from, so ``seen_eff`` is the ``seen`` input itself and the
baked ``repeat_penalty`` default is neutral (ρ = 0) — Two-Tower has no
per-request history (issue #70 / #85).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from . import MASK_PENALTY, OPSET

# Same epsilon TrainedTwoTower's forward clamps the L2 norm with.
NORM_EPS = 1e-12


def build_graph_two_tower(
    payload, onnx_path: Path, *, top_k_default: int, repeat_penalty_default: float
) -> None:
    M = payload.num_items
    dim = payload.embedding_dim
    D = payload.user_dense_dim

    initializers = [
        numpy_helper.from_array(payload.id_embedding.astype(np.float32), name="id_emb"),
        numpy_helper.from_array(payload.hidden_w.astype(np.float32), name="hidden_W"),
        numpy_helper.from_array(payload.hidden_b.astype(np.float32), name="hidden_b"),
        numpy_helper.from_array(payload.out_w.astype(np.float32), name="out_W"),
        numpy_helper.from_array(payload.out_b.astype(np.float32), name="out_b"),
        # Gemm(z, item_matrix, transB=1) → scores; rows are L2-normalized.
        numpy_helper.from_array(payload.item_matrix.astype(np.float32), name="item_matrix"),
        # Baked defaults for the optional tail inputs (mirrors _graph.py).
        numpy_helper.from_array(np.ones((1, M), np.float32), name="mask"),
        numpy_helper.from_array(np.zeros((1, M), np.float32), name="seen"),
        numpy_helper.from_array(
            np.array([[repeat_penalty_default]], np.float32), name="repeat_penalty"
        ),
        numpy_helper.from_array(np.array([top_k_default], np.int64), name="k"),
        # Constants.
        numpy_helper.from_array(np.array(1.0, np.float32), name="one_const"),
        numpy_helper.from_array(np.array(MASK_PENALTY, np.float32), name="mask_penalty_const"),
        numpy_helper.from_array(np.array([M], np.int64), name="M_const"),
        numpy_helper.from_array(np.array(NORM_EPS, np.float32), name="norm_eps_const"),
        numpy_helper.from_array(np.array([1], np.int64), name="axis1_const"),
    ]

    inputs = [
        # Callers pass 0 (the reserved row) for cold-start users.
        helper.make_tensor_value_info("user_idx", TensorProto.INT64, ["batch"]),
    ]

    nodes = [
        helper.make_node("Gather", ["id_emb", "user_idx"], ["h_id"], axis=0),
    ]
    h = "h_id"

    if payload.has_cat:
        # cat_ids / cat_mask: dynamic fan-out C, optional — the baked (1, 1)
        # zero default means "no categorical features" (mask all zero pools
        # to 0 exactly like the Rust count-clamped mean-pool).
        # NOTE: optional inputs carry their own batch dim_param ("cat_batch",
        # not "batch"): when omitted, ORT binds the (1, 1) baked default to
        # the declared symbol, and a shared "batch" symbol would then
        # conflict with a real batch > 1 on `user_idx`. The 1-row default
        # broadcasts against [batch, dim] in the Add.
        inputs += [
            helper.make_tensor_value_info("cat_ids", TensorProto.INT64, ["cat_batch", "C"]),
            helper.make_tensor_value_info("cat_mask", TensorProto.FLOAT, ["cat_batch", "C"]),
        ]
        initializers += [
            numpy_helper.from_array(np.zeros((1, 1), np.int64), name="cat_ids"),
            numpy_helper.from_array(np.zeros((1, 1), np.float32), name="cat_mask"),
            numpy_helper.from_array(payload.cat_embedding.astype(np.float32), name="cat_emb"),
            numpy_helper.from_array(np.array([2], np.int64), name="axes2_const"),
        ]
        nodes += [
            helper.make_node("Gather", ["cat_emb", "cat_ids"], ["cat_e"], axis=0),
            # masked mean-pool over the C slots, count clamped >= 1.
            helper.make_node("Unsqueeze", ["cat_mask", "axes2_const"], ["cat_mask3"]),
            helper.make_node("Mul", ["cat_e", "cat_mask3"], ["cat_masked"]),
            helper.make_node("ReduceSum", ["cat_masked", "axis1_const"], ["cat_sum"], keepdims=0),
            helper.make_node("ReduceSum", ["cat_mask", "axis1_const"], ["cat_cnt"], keepdims=1),
            helper.make_node("Clip", ["cat_cnt", "one_const"], ["cat_cnt_c"]),
            helper.make_node("Div", ["cat_sum", "cat_cnt_c"], ["cat_pooled"]),
            helper.make_node("Add", [h, "cat_pooled"], ["h_cat"]),
        ]
        h = "h_cat"

    if payload.has_dense:
        # Optional: the baked all-zeros default still contributes the dense
        # bias, exactly like the Rust forward (dense_proj(0) = b).
        # Own batch symbol for the same omitted-default reason as cat_ids.
        inputs.append(helper.make_tensor_value_info("dense", TensorProto.FLOAT, ["dense_batch", D]))
        initializers += [
            numpy_helper.from_array(np.zeros((1, D), np.float32), name="dense"),
            numpy_helper.from_array(payload.dense_w.astype(np.float32), name="dense_W"),
            numpy_helper.from_array(payload.dense_b.astype(np.float32), name="dense_b"),
        ]
        nodes += [
            helper.make_node("Gemm", ["dense", "dense_W", "dense_b"], ["dense_out"]),
            helper.make_node("Add", [h, "dense_out"], ["h_dense"]),
        ]
        h = "h_dense"

    # Distinct batch symbols again: each optional input's 1-row baked default
    # must not pin the shared "batch" symbol when the input is omitted.
    inputs += [
        helper.make_tensor_value_info("mask", TensorProto.FLOAT, ["mask_batch", M]),
        helper.make_tensor_value_info("seen", TensorProto.FLOAT, ["seen_batch", M]),
        helper.make_tensor_value_info("repeat_penalty", TensorProto.FLOAT, ["rp_batch", 1]),
        helper.make_tensor_value_info("k", TensorProto.INT64, [1]),
    ]
    outputs = [
        helper.make_tensor_value_info("top_indices", TensorProto.INT64, ["batch", "kc"]),
        helper.make_tensor_value_info("top_scores", TensorProto.FLOAT, ["batch", "kc"]),
        helper.make_tensor_value_info("raw_scores", TensorProto.FLOAT, ["batch", M]),
    ]

    nodes += [
        # 2-layer MLP. burn's Linear stores W as [d_in, d_out] (y = x @ W + b),
        # so Gemm needs no transpose.
        helper.make_node("Gemm", [h, "hidden_W", "hidden_b"], ["pre_relu"]),
        helper.make_node("Relu", ["pre_relu"], ["h_relu"]),
        helper.make_node("Gemm", ["h_relu", "out_W", "out_b"], ["z"]),
        # L2 normalize: z / max(||z||_2, eps). ReduceSumSquare keeps axes as
        # an attribute at opset 17 (input-form arrives in 18).
        helper.make_node("ReduceSumSquare", ["z"], ["z_ss"], axes=[1], keepdims=1),
        helper.make_node("Sqrt", ["z_ss"], ["z_norm"]),
        helper.make_node("Clip", ["z_norm", "norm_eps_const"], ["z_norm_c"]),
        helper.make_node("Div", ["z", "z_norm_c"], ["user_vec"]),
        # raw_scores = user_vec @ item_matrixᵀ (also a graph output).
        helper.make_node("Gemm", ["user_vec", "item_matrix"], ["raw_scores"], transB=1),
        # Tail — same structure/names as _graph.py, minus the derived-seen
        # step (no interactions input): adjusted = raw − ρ·seen.
        helper.make_node("Mul", ["repeat_penalty", "seen"], ["penalty_term"]),
        helper.make_node("Sub", ["raw_scores", "penalty_term"], ["adjusted"]),
        # masked = adjusted + (mask − 1) · MASK_PENALTY
        helper.make_node("Sub", ["mask", "one_const"], ["mask_minus_one"]),
        helper.make_node("Mul", ["mask_minus_one", "mask_penalty_const"], ["mask_term"]),
        helper.make_node("Add", ["adjusted", "mask_term"], ["masked"]),
        # kc = min(k, M); TopK
        helper.make_node("Min", ["k", "M_const"], ["kc"]),
        helper.make_node(
            "TopK", ["masked", "kc"], ["top_scores", "top_indices"], axis=-1, largest=1, sorted=1
        ),
    ]

    graph = helper.make_graph(nodes, "two_tower_onnx", inputs, outputs, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = onnx.IR_VERSION
    onnx.checker.check_model(model)
    onnx.save(model, str(onnx_path))
