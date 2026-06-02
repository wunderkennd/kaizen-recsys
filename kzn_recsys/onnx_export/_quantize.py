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
    # "W" is the initializer name defined in _graph.py.
    w_init = next((init for init in m.graph.initializer if init.name == "W"), None)
    if w_init is None:
        raise ValueError("ONNX graph has no initialiser named 'W'")
    w_fp32 = numpy_helper.to_array(w_init)
    w_fp16 = w_fp32.astype(np.float16)
    m.graph.initializer.remove(w_init)
    m.graph.initializer.append(numpy_helper.from_array(w_fp16, name="W_fp16"))

    # 2. Locate the Gemm node that consumes "W" (name coupled to _graph.py) and
    #    rewire it to consume the Cast output.  Fail loud if out of sync with
    #    _graph.py rather than producing a cryptic check_model error.
    gemm = next(
        (
            n
            for n in m.graph.node
            if n.op_type == "Gemm" and len(n.input) > 1 and n.input[1] == "W"
        ),
        None,
    )
    if gemm is None:
        raise ValueError(
            "No Gemm node consuming 'W' found — _quantize.py is out of sync with _graph.py"
        )
    gemm.input[1] = "W_fp32_cast"

    # 3. Insert Cast(W_fp16 → fp32) immediately before the located Gemm in node order.
    cast_node = helper.make_node(
        "Cast",
        inputs=["W_fp16"],
        outputs=["W_fp32_cast"],
        to=TensorProto.FLOAT,
    )
    nodes = list(m.graph.node)
    gemm_idx = nodes.index(gemm)
    new_nodes: list[onnx.NodeProto] = nodes[:gemm_idx] + [cast_node] + nodes[gemm_idx:]
    del m.graph.node[:]
    m.graph.node.extend(new_nodes)

    onnx.checker.check_model(m)
    onnx.save(m, str(onnx_path))


def quantize(onnx_path: Path, dtype: str) -> None:
    """Rewrite ``onnx_path`` in place at the requested precision.

    IO (and the masking/TopK arithmetic) stay fp32 — only the weight and the
    matmul are reduced — so the output signature and the 1e9 sentinels remain
    safe (spec §4).

    Not idempotent / not re-entrant: calling fp16 twice on the same file will
    fail because the ``W`` initializer has been renamed to ``W_fp16`` after the
    first call. Intended for single-pass export-once usage.
    """
    if dtype == "fp16":
        _convert_weight_to_fp16(onnx_path)
    elif dtype == "int8":
        from onnxruntime.quantization import QuantType, quantize_dynamic

        # "int8.onnx" suffix is a temp file; cleaned up on failure so a stale
        # partial file is never left next to the model.
        tmp = onnx_path.with_suffix(".int8.onnx")
        try:
            quantize_dynamic(str(onnx_path), str(tmp), weight_type=QuantType.QInt8)
            tmp.replace(onnx_path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
    else:
        raise ValueError(f"unsupported quantization dtype: {dtype!r}")
