import json
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


def _run(sess, payload, inter, feat, *, mask=None, seen=None, rp=0.0, k=None):
    M = payload.num_items
    feeds = {
        "interactions": inter,
        "features": feat,
        "mask": np.ones((1, M), np.float32) if mask is None else mask,
        "seen": np.zeros((1, M), np.float32) if seen is None else seen,
        "repeat_penalty": np.array([[rp]], np.float32),
        "k": np.array([M if k is None else k], np.int64),
    }
    out = sess.run(None, feeds)
    names = [o.name for o in sess.get_outputs()]
    return {n: out[i] for i, n in enumerate(names)}


def _rank_of(idx_row, item):
    """Position of `item` in a sorted top_indices row (0 = top)."""
    return idx_row.tolist().index(item)


def test_default_excludes_seen_sinks_to_bottom(trained_model, tmp_path):
    from kzn_recsys.onnx_export import _payload_from_model, export_onnx

    payload = _payload_from_model(trained_model)
    sess = ort.InferenceSession(str(export_onnx(trained_model, tmp_path).onnx_path))
    inter, feat = _build_inputs(payload, [(0, 5.0)], [])
    # Default exclude policy (sentinel) drives the seen item to the LAST rank.
    out = _run(sess, payload, inter, feat, rp=1e9)
    assert out["top_indices"][0][-1] == 0
    assert out["top_indices"][0][0] != 0


def test_repeat_boost_surfaces_seen_item(trained_model, tmp_path):
    from kzn_recsys.onnx_export import _payload_from_model, export_onnx

    payload = _payload_from_model(trained_model)
    sess = ort.InferenceSession(str(export_onnx(trained_model, tmp_path).onnx_path))
    inter, feat = _build_inputs(payload, [(0, 5.0)], [])
    neutral = _run(sess, payload, inter, feat, rp=0.0)["top_scores"][0]
    boosted = _run(sess, payload, inter, feat, rp=-1e6)["top_scores"][0]
    assert boosted.max() > neutral.max()


def test_seen_input_extends_derived_penalty(trained_model, tmp_path):
    from kzn_recsys.onnx_export import _payload_from_model, export_onnx

    payload = _payload_from_model(trained_model)
    M = payload.num_items
    sess = ort.InferenceSession(str(export_onnx(trained_model, tmp_path).onnx_path))
    # Real nonzero interaction on item 1 (so scores are non-degenerate and item 1
    # is "seen" via the derived nonzero path). Item 0 is NOT nonzero-seen.
    inter, feat = _build_inputs(payload, [(1, 5.0)], [])
    derived = _run(sess, payload, inter, feat, rp=1e9)["top_indices"][0]
    # Explicitly mark item 0 as seen too → it must drop in rank vs the derived run.
    seen = np.zeros((1, M), np.float32)
    seen[0, 0] = 1.0
    explicit = _run(sess, payload, inter, feat, seen=seen, rp=1e9)["top_indices"][0]
    assert _rank_of(explicit, 0) > _rank_of(derived, 0)


def test_mask_sinks_item_despite_boost(trained_model, tmp_path):
    from kzn_recsys.onnx_export import _payload_from_model, export_onnx

    payload = _payload_from_model(trained_model)
    M = payload.num_items
    sess = ort.InferenceSession(str(export_onnx(trained_model, tmp_path).onnx_path))
    inter, feat = _build_inputs(payload, [(0, 5.0)], [])
    mask = np.ones((1, M), np.float32)
    mask[0, 0] = 0.0  # exclude item 0 for compliance
    # Even a strong repeat boost cannot lift a masked item: it ranks LAST.
    idx = _run(sess, payload, inter, feat, mask=mask, rp=-1e6)["top_indices"][0]
    assert idx[-1] == 0


def test_top_k_clamped(trained_model, tmp_path):
    from kzn_recsys.onnx_export import _payload_from_model, export_onnx

    payload = _payload_from_model(trained_model)
    M = payload.num_items
    sess = ort.InferenceSession(str(export_onnx(trained_model, tmp_path).onnx_path))
    inter, feat = _build_inputs(payload, [], [(0, 1.0)])
    out = _run(sess, payload, inter, feat, k=M + 50)
    assert out["top_indices"].shape[1] == M  # Min(k, M) clamp


def test_batch_inference(trained_model, tmp_path):
    from kzn_recsys.onnx_export import _payload_from_model, export_onnx

    payload = _payload_from_model(trained_model)
    M, K = payload.num_items, payload.num_user_features
    sess = ort.InferenceSession(str(export_onnx(trained_model, tmp_path).onnx_path))
    # Two users with DIFFERENT exclusions in one batch.
    interactions = np.zeros((2, M), np.float32)
    interactions[0, 0] = 5.0  # user 0 saw item 0
    feats = np.zeros((2, K), np.float32)
    out = sess.run(
        None,
        {
            "interactions": interactions,
            "features": feats,
            "mask": np.ones((2, M), np.float32),
            "seen": np.zeros((2, M), np.float32),
            "repeat_penalty": np.array([[1e9], [1e9]], np.float32),
            "k": np.array([M], np.int64),
        },
    )
    names = [o.name for o in sess.get_outputs()]
    top = out[names.index("top_indices")]
    assert top.shape == (2, M)
    assert top[0][-1] == 0  # user 0's seen item sinks to bottom


def test_vocab_contents(trained_model, tmp_path):
    from kzn_recsys.onnx_export import _payload_from_model, export_onnx

    payload = _payload_from_model(trained_model)
    res = export_onnx(trained_model, tmp_path, top_k_default=7)
    vocab = json.loads(res.vocab_path.read_text())

    assert vocab["format_version"] == 1
    assert vocab["model_kind"] == "ease"
    assert vocab["num_items"] == payload.num_items
    assert vocab["num_user_features"] == payload.num_user_features
    assert vocab["top_k_default"] == 7
    assert vocab["constants"] == {"MASK_PENALTY": 1e9, "EXCLUDE_SENTINEL": 1e9}
    assert vocab["repeat_policy"]["default_penalty"] == 1e9
    assert vocab["repeat_policy"]["per_user_table_present"] is False
    assert len(vocab["item_index_to_guid"]) == payload.num_items
    assert vocab["provenance"]["sparsity_threshold"] is None

    names = [i["name"] for i in vocab["io_signature"]["inputs"]]
    assert names == ["interactions", "features", "mask", "seen", "repeat_penalty", "k"]
    out_names = [o["name"] for o in vocab["io_signature"]["outputs"]]
    assert out_names == ["top_indices", "top_scores", "raw_scores"]


@pytest.fixture(scope="module")
def model_no_user_features():
    """Trained EASE model with K = 0 (no user features)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        i_path = Path(tmpdir) / "interactions.parquet"
        u_path = Path(tmpdir) / "user_features.parquet"
        t_path = Path(tmpdir) / "item_features.parquet"
        pl.DataFrame(
            {"user_id": ["u0", "u0", "u1"], "item_id": ["G0", "G2", "G1"], "value": [5.0, 4.0, 6.0]}
        ).write_parquet(i_path)
        # Empty user features → K = 0 (typed to avoid null-dtype rejection by Polars/Rust).
        pl.DataFrame(
            {"user_id": pl.Series([], dtype=pl.Utf8), "feature_name": pl.Series([], dtype=pl.Utf8), "value": pl.Series([], dtype=pl.Float64)}
        ).write_parquet(u_path)
        pl.DataFrame(
            {"item_id": ["G0", "G1", "G2", "G3"], "feature_name": ["a", "a", "a", "a"], "value": [1.0] * 4}
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


def test_k0_uniform_signature(model_no_user_features, tmp_path):
    from kzn_recsys.onnx_export import _payload_from_model, export_onnx

    payload = _payload_from_model(model_no_user_features)
    assert payload.num_user_features == 0
    res = export_onnx(model_no_user_features, tmp_path)
    sess = ort.InferenceSession(str(res.onnx_path))
    in_names = [i.name for i in sess.get_inputs()]
    assert in_names == ["interactions", "features", "mask", "seen", "repeat_penalty", "k"]
    M = payload.num_items
    out = sess.run(
        None,
        {
            "interactions": np.zeros((1, M), np.float32),
            "features": np.zeros((1, 0), np.float32),  # width-0
            "mask": np.ones((1, M), np.float32),
            "seen": np.zeros((1, M), np.float32),
            "repeat_penalty": np.array([[0.0]], np.float32),
            "k": np.array([M], np.int64),
        },
    )
    names = [o.name for o in sess.get_outputs()]
    assert out[names.index("raw_scores")].shape == (1, M)
