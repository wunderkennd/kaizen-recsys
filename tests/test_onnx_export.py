import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import polars as pl
import pytest

import kzn_recsys as fease
from kzn_recsys.onnx_export import (
    ExportPayload,
    EXCLUDE_SENTINEL,
    MASK_PENALTY,
    OPSET,
    _payload_from_model,
    _validate_exportable,
)


@pytest.fixture(scope="module")
def trained_model():
    """Tiny trained EASE model (mirrors tests/test_model.py fixture shape)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        i_path = Path(tmpdir) / "interactions.parquet"
        u_path = Path(tmpdir) / "user_features.parquet"
        t_path = Path(tmpdir) / "item_features.parquet"
        pl.DataFrame(
            {"user_id": ["u0", "u0", "u1"], "item_id": ["G0", "G2", "G1"], "value": [5.0, 4.0, 6.0]}
        ).write_parquet(i_path)
        pl.DataFrame(
            {
                "user_id": ["u0", "u0", "u1", "u1", "u2", "u2"],
                "feature_name": ["device_Mobile", "region_US", "device_Mobile", "region_EMEA", "device_Console", "region_APAC"],
                "value": [1.0] * 6,
            }
        ).write_parquet(u_path)
        pl.DataFrame(
            {
                "item_id": ["G0", "G1", "G2", "G3"],
                "feature_name": ["genre_Action", "genre_Comedy", "genre_Action", "genre_Comedy"],
                "value": [1.0] * 4,
            }
        ).write_parquet(t_path)
        yield fease.build_and_train(
            interactions_path=str(i_path),
            user_features_path=str(u_path),
            item_features_path=str(t_path),
            alpha=1.0,
            beta=1.0,
            lambda_=10.0,
            meta_weight=0.0,
        )


def test_export_payload_shapes_and_fields(trained_model):
    d = trained_model.export_payload()
    assert d["kind"] == "ease"
    m, cols = d["s_items_shape"]
    assert cols == d["num_items"] + d["num_user_features"]
    assert m == d["num_items"]
    s = np.frombuffer(d["s_items_bytes"], dtype="<f8").reshape(m, cols)
    assert s.shape == (m, cols)
    # Diagonal of the item-item block is ~zero (EASE zero-diagonal constraint).
    assert np.allclose(np.diag(s[:, :m]), 0.0, atol=1e-6)
    assert isinstance(d["item_index_to_guid"], list) and len(d["item_index_to_guid"]) == m
    assert d["sparsity_threshold"] is None  # no weighting config used
    assert set(d["feature_name_to_index"].values()) == set(range(d["num_user_features"]))


def test_payload_from_model_builds_dataclass(trained_model):
    p = _payload_from_model(trained_model)
    assert isinstance(p, ExportPayload)
    assert p.kind == "ease"
    assert p.s_items.shape == (p.num_items, p.num_items + p.num_user_features)


def test_validate_exportable_rejects_nan(trained_model):
    p = _payload_from_model(trained_model)
    p.s_items[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        _validate_exportable(p)


def test_constants():
    assert EXCLUDE_SENTINEL == 1e9
    assert MASK_PENALTY == 1e9
    assert OPSET == 17


def _build_inputs(payload, interactions_idx_val, feature_idx_val):
    """Dense fp32 input vectors for one user."""
    M, K = payload.num_items, payload.num_user_features
    inter = np.zeros((1, M), np.float32)
    for idx, val in interactions_idx_val:
        inter[0, idx] = val
    feat = np.zeros((1, K), np.float32)
    for idx, val in feature_idx_val:
        feat[0, idx] = val
    return inter, feat


def _ref_scores(payload, inter, feat):
    """Reference raw affinity: β-folded S_items @ [interactions | features]."""
    M = payload.num_items
    W = payload.s_items.copy()
    if payload.num_user_features > 0:
        W[:, M:] *= payload.beta
    z = np.concatenate([inter[0], feat[0]]).astype(np.float64)
    return (W @ z).astype(np.float32)


def test_graph_raw_scores_parity(trained_model, tmp_path):
    from kzn_recsys.onnx_export import _payload_from_model, export_onnx

    payload = _payload_from_model(trained_model)
    res = export_onnx(trained_model, tmp_path)
    sess = ort.InferenceSession(str(res.onnx_path))

    inter, feat = _build_inputs(payload, [(0, 5.0)], [(1, 1.0)])
    out = sess.run(
        None,
        {
            "interactions": inter,
            "features": feat,
            "mask": np.ones((1, payload.num_items), np.float32),
            "seen": np.zeros((1, payload.num_items), np.float32),
            "repeat_penalty": np.array([[0.0]], np.float32),  # neutral → raw == adjusted
            "k": np.array([payload.num_items], np.int64),
        },
    )
    names = [o.name for o in sess.get_outputs()]
    raw = out[names.index("raw_scores")][0]
    np.testing.assert_allclose(raw, _ref_scores(payload, inter, feat), rtol=1e-5, atol=1e-5)
