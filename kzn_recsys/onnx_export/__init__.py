"""Optional ONNX export for the EASE model. Requires the ``[onnx]`` extra."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

# Two distinct concepts that currently share the magnitude 1e9 (spec §13 glossary),
# kept as separate names because their roles differ and may diverge:
#   MASK_PENALTY     — graph constant: (mask - 1) * MASK_PENALTY drops eligibility-masked items
#   EXCLUDE_SENTINEL — default repeat_penalty value reproducing "exclude already-watched"
MASK_PENALTY = 1e9
EXCLUDE_SENTINEL = 1e9
OPSET = 17


@dataclasses.dataclass
class ExportPayload:
    kind: str
    s_items: np.ndarray  # (M, M+K) float64, row-major; RAW S[0:M,:] (not β-folded)
    beta: float
    num_items: int
    num_user_features: int
    num_item_features: int
    alpha: float
    lambda_: float
    meta_weight: float
    sparsity_threshold: float | None
    item_index_to_guid: list[str]
    feature_name_to_index: dict[str, int]


@dataclasses.dataclass
class ExportResult:
    onnx_path: Path
    vocab_path: Path
    mlflow_path: Path | None = None


def _payload_from_model(model) -> ExportPayload:
    d = model.export_payload()
    if d["kind"] != "ease":
        raise NotImplementedError(f"ONNX export not yet supported for {d['kind']}")
    rows, cols = d["s_items_shape"]
    s = np.frombuffer(d["s_items_bytes"], dtype="<f8").reshape(rows, cols).copy()
    return ExportPayload(
        kind=d["kind"],
        s_items=s,
        beta=float(d["beta"]),
        num_items=int(d["num_items"]),
        num_user_features=int(d["num_user_features"]),
        num_item_features=int(d["num_item_features"]),
        alpha=float(d["alpha"]),
        lambda_=float(d["lambda_"]),
        meta_weight=float(d["meta_weight"]),
        sparsity_threshold=d["sparsity_threshold"],
        item_index_to_guid=list(d["item_index_to_guid"]),
        feature_name_to_index=dict(d["feature_name_to_index"]),
    )


def _validate_exportable(payload: ExportPayload) -> None:
    """Mirror RustFeaseModel::validate() — refuse to export a bad model."""
    if payload.num_items == 0:
        raise ValueError("Cannot export a model with zero items")
    if np.isnan(payload.s_items).any():
        raise ValueError("S matrix contains NaN values; refusing to export")
    if np.isinf(payload.s_items).any():
        raise ValueError("S matrix contains Inf values; refusing to export")
    if not np.any(payload.s_items):
        raise ValueError("S matrix is all zeros; model may not have learned")


def export_onnx(
    model,
    output_dir: str | Path,
    *,
    top_k_default: int = 100,
    dtype: str = "fp32",
    repeat_penalty_default: str | float = "exclude",
    mlflow: bool = False,
) -> ExportResult:
    """Export a trained EASE ``FeaseModel`` to ONNX + sidecar (+ optional MLflow).

    See ``docs/superpowers/specs/2026-06-01-onnx-export-design.md``.
    """
    if top_k_default <= 0:
        raise ValueError("top_k_default must be positive")
    if dtype not in ("fp32", "fp16", "int8"):
        raise ValueError(f"dtype must be fp32|fp16|int8, got {dtype!r}")

    payload = _payload_from_model(model)  # raises NotImplementedError for non-ease
    _validate_exportable(payload)

    rp_default = (
        EXCLUDE_SENTINEL
        if repeat_penalty_default == "exclude"
        else float(repeat_penalty_default)
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"
    vocab_path = output_dir / "vocab.json"

    from ._graph import build_graph

    build_graph(payload, onnx_path, top_k_default=top_k_default, repeat_penalty_default=rp_default)

    if dtype != "fp32":
        from ._quantize import quantize

        quantize(onnx_path, dtype)

    from ._vocab import write_vocab

    write_vocab(
        payload,
        vocab_path,
        top_k_default=top_k_default,
        dtype=dtype,
        repeat_penalty_default=rp_default,
    )

    mlflow_path = None
    if mlflow:
        from ._mlflow import build_mlflow

        mlflow_path = build_mlflow(onnx_path, vocab_path, output_dir / "mlflow_model")

    return ExportResult(onnx_path=onnx_path, vocab_path=vocab_path, mlflow_path=mlflow_path)


def _write_rust_fixture(model, fixtures_dir) -> None:
    """Emit fixture.onnx + inputs.json + expected.json for the Rust ort test.

    Run once to (re)generate committed fixtures; not part of normal export.
    Exports into a temporary directory and copies only fixture.onnx so that
    tests/fixtures/ contains exactly the three committed files.
    """
    import json as _json
    import shutil as _shutil
    import tempfile as _tempfile

    import numpy as _np

    fixtures_dir = Path(fixtures_dir)
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    # Export into a temp dir so model.onnx / vocab.json don't land in fixtures_dir.
    with _tempfile.TemporaryDirectory() as _tmp:
        _tmp_path = Path(_tmp)
        res = export_onnx(model, _tmp_path)
        _shutil.copy2(res.onnx_path, fixtures_dir / "fixture.onnx")

        payload = _payload_from_model(model)
        M, K = payload.num_items, payload.num_user_features
        inter = _np.zeros(M, _np.float32)
        if M > 0:
            inter[0] = 3.0
        feat = _np.zeros(K, _np.float32)
        if K > 0:
            feat[0] = 1.0
        inputs = {"interactions": inter.tolist(), "features": feat.tolist()}
        (fixtures_dir / "inputs.json").write_text(_json.dumps(inputs))

        import onnxruntime as _ort

        sess = _ort.InferenceSession(str(fixtures_dir / "fixture.onnx"))
        out = sess.run(
            None,
            {
                "interactions": inter.reshape(1, M),
                "features": feat.reshape(1, K),
                "mask": _np.ones((1, M), _np.float32),
                "seen": _np.zeros((1, M), _np.float32),
                "repeat_penalty": _np.array([[0.0]], _np.float32),
                "k": _np.array([M], _np.int64),
            },
        )
        names = [o.name for o in sess.get_outputs()]
        raw = out[names.index("raw_scores")][0]
        (fixtures_dir / "expected.json").write_text(_json.dumps({"raw_scores": raw.tolist()}))
