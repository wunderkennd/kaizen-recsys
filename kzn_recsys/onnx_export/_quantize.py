"""fp16 / int8 post-processing of the fp32 ONNX graph (spec §1 quantization)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _convert_weight_to_fp16(onnx_path: Path) -> None:
    """Rewrite the Gemm weight initialiser W to fp16 in place.

    IO and all sentinel / masking constants remain fp32; only the weight
    matrix and the effective matmul are reduced. A Cast(fp16→fp32) node is
    inserted just before the Gemm so the arithmetic still runs in fp32 on
    most runtimes (per onnxconverter_common keep_io_types semantics).

    Why manual conversion instead of onnxconverter_common.float16?
    The converter library (v1.16.0) has a crash in
    ``remove_unnecessary_cast_node`` when a Cast node has multiple downstream
    consumers (``AttributeError: 'list' object has no attribute 'input'``).
    It also silently clamps 1e9 sentinel constants to fp16-max (65504),
    breaking the repeat-penalty / mask-penalty logic. The targeted manual
    approach avoids both issues.
    """
    m = onnx.load(str(onnx_path))

    # 1. Find the W initialiser and replace with W_fp16 (fp16 tensor).
    w_init = next((init for init in m.graph.initializer if init.name == "W"), None)
    if w_init is None:
        raise ValueError("ONNX graph has no initialiser named 'W'")
    w_fp32 = numpy_helper.to_array(w_init)
    w_fp16 = w_fp32.astype(np.float16)
    m.graph.initializer.remove(w_init)
    m.graph.initializer.append(numpy_helper.from_array(w_fp16, name="W_fp16"))

    # 2. Insert Cast(W_fp16 → fp32) before the Gemm that consumes W.
    cast_node = helper.make_node(
        "Cast",
        inputs=["W_fp16"],
        outputs=["W_fp32_cast"],
        to=TensorProto.FLOAT,
    )
    for node in m.graph.node:
        if node.op_type == "Gemm" and node.input[1] == "W":
            node.input[1] = "W_fp32_cast"
            break

    # 3. Splice cast_node immediately before the Gemm in node order.
    nodes = list(m.graph.node)
    new_nodes: list = []
    for n in nodes:
        if n.op_type == "Gemm":
            new_nodes.append(cast_node)
        new_nodes.append(n)
    del m.graph.node[:]
    m.graph.node.extend(new_nodes)

    onnx.checker.check_model(m)
    onnx.save(m, str(onnx_path))


def quantize(onnx_path: Path, dtype: str) -> None:
    """Rewrite ``onnx_path`` in place at the requested precision.

    IO (and the masking/TopK arithmetic) stay fp32 — only the weight and the
    matmul are reduced — so the output signature and the 1e9 sentinels remain
    safe (spec §4).
    """
    if dtype == "fp16":
        _convert_weight_to_fp16(onnx_path)
    elif dtype == "int8":
        from onnxruntime.quantization import QuantType, quantize_dynamic

        tmp = onnx_path.with_suffix(".int8.onnx")
        quantize_dynamic(str(onnx_path), str(tmp), weight_type=QuantType.QInt8)
        tmp.replace(onnx_path)
    else:
        raise ValueError(f"unsupported quantization dtype: {dtype!r}")
