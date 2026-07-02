"""Optional ONNX export for the EASE and Two-Tower models. Requires the ``[onnx]`` extra."""
from __future__ import annotations

import dataclasses
from pathlib import Path

try:
    import numpy as np
    import onnx as _onnx  # noqa: F401
    import onnxruntime as _onnxruntime  # noqa: F401
except ImportError as _exc:  # pragma: no cover - exercised by the build matrix
    raise ImportError(
        "kzn_recsys ONNX export requires optional dependencies. "
        "Install them with:  pip install 'kzn_recsys[onnx]'"
    ) from _exc

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
class TwoTowerExportPayload:
    """Host copy of the Two-Tower user tower + item catalog matrix (#85).

    All weight arrays are float32. Empty branches (``has_cat`` /
    ``has_dense`` false) carry zero-row arrays. burn's ``Linear`` computes
    ``x @ W + b`` with ``W: (d_in, d_out)``, so no transpose is needed.
    """

    kind: str
    embedding_dim: int
    num_users: int  # includes the reserved cold-start row at index 0
    num_items: int
    has_cat: bool
    has_dense: bool
    num_user_categories: int
    user_dense_dim: int
    id_embedding: np.ndarray  # (num_users, dim); row 0 = cold-start prior
    cat_embedding: np.ndarray  # (num_user_categories, dim) or (0, dim)
    dense_w: np.ndarray  # (user_dense_dim, dim) or (0, dim)
    dense_b: np.ndarray  # (dim,) or (0,)
    hidden_w: np.ndarray  # (dim, dim)
    hidden_b: np.ndarray  # (dim,)
    out_w: np.ndarray  # (dim, dim)
    out_b: np.ndarray  # (dim,)
    item_matrix: np.ndarray  # (num_items, dim), L2-normalized rows
    item_index_to_guid: list[str]
    user_id_to_index: dict[str, int]  # real users only; cold-start = index 0
    user_cat_feature_to_idx: dict[str, int]
    user_dense_feature_to_idx: dict[str, int]


@dataclasses.dataclass
class ExportResult:
    onnx_path: Path
    vocab_path: Path
    mlflow_path: Path | None = None


def _f32(d, key, shape_key=None):
    a = np.frombuffer(d[key], dtype="<f4").copy()
    if shape_key is not None:
        a = a.reshape(d[shape_key])
    return a


def _two_tower_payload(d) -> TwoTowerExportPayload:
    return TwoTowerExportPayload(
        kind=d["kind"],
        embedding_dim=int(d["embedding_dim"]),
        num_users=int(d["num_users"]),
        num_items=int(d["num_items"]),
        has_cat=bool(d["has_cat"]),
        has_dense=bool(d["has_dense"]),
        num_user_categories=int(d["num_user_categories"]),
        user_dense_dim=int(d["user_dense_dim"]),
        id_embedding=_f32(d, "id_embedding_bytes", "id_embedding_shape"),
        cat_embedding=_f32(d, "cat_embedding_bytes", "cat_embedding_shape"),
        dense_w=_f32(d, "dense_w_bytes", "dense_w_shape"),
        dense_b=_f32(d, "dense_b_bytes"),
        hidden_w=_f32(d, "hidden_w_bytes", "hidden_w_shape"),
        hidden_b=_f32(d, "hidden_b_bytes"),
        out_w=_f32(d, "out_w_bytes", "out_w_shape"),
        out_b=_f32(d, "out_b_bytes"),
        item_matrix=_f32(d, "item_matrix_bytes", "item_matrix_shape"),
        item_index_to_guid=list(d["item_index_to_guid"]),
        user_id_to_index=dict(d["user_id_to_index"]),
        user_cat_feature_to_idx=dict(d["user_cat_feature_to_idx"]),
        user_dense_feature_to_idx=dict(d["user_dense_feature_to_idx"]),
    )


def _payload_from_model(model) -> ExportPayload | TwoTowerExportPayload:
    d = model.export_payload()
    if d["kind"] == "two_tower":
        return _two_tower_payload(d)
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
    # Dimension check (Rust validate() check 1) is unnecessary here: s_items shape
    # comes directly from the trained model's export_payload (np.frombuffer(...).reshape(rows, cols)),
    # so dimensions are internally consistent by construction.
    if payload.num_items == 0:
        raise ValueError("Cannot export a model with zero items")
    if np.isnan(payload.s_items).any():
        raise ValueError("S matrix contains NaN values; refusing to export")
    if np.isinf(payload.s_items).any():
        raise ValueError("S matrix contains Inf values; refusing to export")
    if not np.any(payload.s_items):
        raise ValueError("S matrix is all zeros; model may not have learned")


def _validate_exportable_two_tower(payload: TwoTowerExportPayload) -> None:
    """Mirror TrainedTwoTower::validate() plus NaN/Inf weight checks."""
    if payload.num_items == 0:
        raise ValueError("Cannot export a model with zero items")
    weights = {
        "id_embedding": payload.id_embedding,
        "cat_embedding": payload.cat_embedding,
        "dense_w": payload.dense_w,
        "dense_b": payload.dense_b,
        "hidden_w": payload.hidden_w,
        "hidden_b": payload.hidden_b,
        "out_w": payload.out_w,
        "out_b": payload.out_b,
        "item_matrix": payload.item_matrix,
    }
    for name, arr in weights.items():
        if np.isnan(arr).any():
            raise ValueError(f"{name} contains NaN values; refusing to export")
        if np.isinf(arr).any():
            raise ValueError(f"{name} contains Inf values; refusing to export")
    if not np.any(payload.item_matrix):
        raise ValueError("item matrix is all zeros; model may not have learned")


def export_onnx(
    model,
    output_dir: str | Path,
    *,
    top_k_default: int = 100,
    dtype: str = "fp32",
    repeat_penalty_default: str | float = "exclude",
    interactions: str | Path | None = None,
    repeat_affinity_scale: float = 1.0,
    repeat_affinity_prior_strength: float = 10.0,
    mlflow: bool = False,
) -> ExportResult:
    """Export a trained EASE ``FeaseModel`` or ``TwoTowerModel`` to ONNX +
    sidecar (+ optional MLflow, EASE only).

    Pass ``interactions`` (the long-format training interactions file) to
    learn a per-user repeat-affinity table (Tier C, spec §10, EASE only): a
    smoothed ``user_guid → ρ`` estimate persisted in ``vocab.json`` and
    applied by the MLflow wrapper when the caller doesn't override
    ``repeat_penalty``. See ``_repeat_affinity.estimate_repeat_affinity``
    for the estimator.

    Two-Tower (#85): the graph is the user tower (Gather id/cat embeddings →
    dense Gemm → 2-layer MLP → L2 normalize) followed by a MatMul against
    the baked item catalog matrix and the same penalty/mask/TopK tail as
    EASE. The default repeat policy is neutral (ρ = 0) because Two-Tower has
    no per-request history — ``repeat_penalty_default="exclude"`` maps to
    0.0; pass an explicit float to bake a different default. Quantization,
    the MLflow wrapper, and the Tier C repeat-affinity table are not yet
    supported for Two-Tower.

    See ``docs/superpowers/specs/2026-06-01-onnx-export-design.md``.
    """
    if top_k_default <= 0:
        raise ValueError("top_k_default must be positive")
    if dtype not in ("fp32", "fp16", "int8"):
        raise ValueError(f"dtype must be fp32|fp16|int8, got {dtype!r}")

    payload = _payload_from_model(model)  # raises NotImplementedError for e.g. sasrec

    if payload.kind == "two_tower":
        return _export_onnx_two_tower(
            payload,
            output_dir,
            top_k_default=top_k_default,
            dtype=dtype,
            repeat_penalty_default=repeat_penalty_default,
            interactions=interactions,
            mlflow=mlflow,
        )

    _validate_exportable(payload)

    rp_default = (
        EXCLUDE_SENTINEL
        if repeat_penalty_default == "exclude"
        else float(repeat_penalty_default)
    )

    per_user_table = None
    if interactions is not None:
        from ._repeat_affinity import estimate_repeat_affinity

        per_user_table = estimate_repeat_affinity(
            interactions,
            payload.item_index_to_guid,
            scale=repeat_affinity_scale,
            prior_strength=repeat_affinity_prior_strength,
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
        per_user_table=per_user_table,
    )

    mlflow_path = None
    if mlflow:
        from ._mlflow import build_mlflow

        mlflow_path = build_mlflow(onnx_path, vocab_path, output_dir / "mlflow_model")

    return ExportResult(onnx_path=onnx_path, vocab_path=vocab_path, mlflow_path=mlflow_path)


def _export_onnx_two_tower(
    payload: TwoTowerExportPayload,
    output_dir: str | Path,
    *,
    top_k_default: int,
    dtype: str,
    repeat_penalty_default: str | float,
    interactions,
    mlflow: bool,
) -> ExportResult:
    """Two-Tower branch of :func:`export_onnx` (issue #85).

    Scope guards (documented follow-ups, not silent no-ops):
    - ``dtype`` must be ``fp32`` — ``_quantize`` targets the EASE ``W``
      initializer and has not been validated on the tower graph.
    - ``mlflow`` is unsupported — the pyfunc wrapper speaks the EASE
      interactions/features input contract.
    - ``interactions`` (Tier C repeat affinity) is unsupported — Two-Tower
      has no derived-seen path; the neutral ρ = 0 default applies instead.
    """
    if dtype != "fp32":
        raise NotImplementedError(
            "quantized Two-Tower export is not yet supported; use dtype='fp32'"
        )
    if mlflow:
        raise NotImplementedError("MLflow wrapper is not yet supported for Two-Tower export")
    if interactions is not None:
        raise NotImplementedError(
            "per-user repeat-affinity (interactions=...) is EASE-only; "
            "Two-Tower uses a neutral repeat policy (rho = 0)"
        )

    _validate_exportable_two_tower(payload)

    # "exclude" is the signature default aimed at EASE's derived-seen path.
    # Two-Tower has no per-request history, so the baked default is neutral
    # (issue #70/#85). An explicit numeric default is honored as-is.
    rp_default = 0.0 if repeat_penalty_default == "exclude" else float(repeat_penalty_default)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"
    vocab_path = output_dir / "vocab.json"

    from ._graph_two_tower import build_graph_two_tower

    build_graph_two_tower(
        payload, onnx_path, top_k_default=top_k_default, repeat_penalty_default=rp_default
    )

    from ._vocab import write_vocab_two_tower

    write_vocab_two_tower(
        payload,
        vocab_path,
        top_k_default=top_k_default,
        dtype=dtype,
        repeat_penalty_default=rp_default,
    )

    return ExportResult(onnx_path=onnx_path, vocab_path=vocab_path, mlflow_path=None)


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


def _write_rust_fixture_two_tower(model, fixtures_dir) -> None:
    """Emit two_tower_fixture.onnx + two_tower_{inputs,expected}.json for the
    Rust ort parity test (#85).

    ``model`` must be a trained ``TwoTowerModel`` whose user tower has both a
    categorical and a dense branch, so the fixture exercises the full graph.
    Run once to (re)generate committed fixtures; not part of normal export.
    """
    import json as _json
    import shutil as _shutil
    import tempfile as _tempfile

    import numpy as _np

    fixtures_dir = Path(fixtures_dir)
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    payload = _payload_from_model(model)
    if not (payload.kind == "two_tower" and payload.has_cat and payload.has_dense):
        raise ValueError(
            "fixture model must be a Two-Tower with categorical AND dense user features"
        )

    with _tempfile.TemporaryDirectory() as _tmp:
        res = export_onnx(model, Path(_tmp))
        _shutil.copy2(res.onnx_path, fixtures_dir / "two_tower_fixture.onnx")

        M = payload.num_items
        # Warm user 1 with one categorical slot active and a dense value.
        inputs = {
            "user_idx": [1],
            "cat_ids": [[0]],
            "cat_mask": [[1.0]],
            "dense": [[0.5] * payload.user_dense_dim],
        }
        (fixtures_dir / "two_tower_inputs.json").write_text(_json.dumps(inputs))

        import onnxruntime as _ort

        sess = _ort.InferenceSession(str(fixtures_dir / "two_tower_fixture.onnx"))
        out = sess.run(
            None,
            {
                "user_idx": _np.array(inputs["user_idx"], _np.int64),
                "cat_ids": _np.array(inputs["cat_ids"], _np.int64),
                "cat_mask": _np.array(inputs["cat_mask"], _np.float32),
                "dense": _np.array(inputs["dense"], _np.float32),
                "mask": _np.ones((1, M), _np.float32),
                "seen": _np.zeros((1, M), _np.float32),
                "repeat_penalty": _np.array([[0.0]], _np.float32),
                "k": _np.array([M], _np.int64),
            },
        )
        names = [o.name for o in sess.get_outputs()]
        raw = out[names.index("raw_scores")][0]
        (fixtures_dir / "two_tower_expected.json").write_text(
            _json.dumps({"raw_scores": raw.tolist()})
        )
