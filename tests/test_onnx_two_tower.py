"""ONNX export parity tests for the Two-Tower model (issue #85).

Requires BOTH the ``[onnx]`` extra (numpy/onnx/onnxruntime) and an extension
built with ``--features ml-models`` — skipped cleanly when either is absent.

The acceptance bar: onnxruntime ``raw_scores`` must match
``TwoTowerModel.predict(...)`` scores within 1e-4 for warm users, the
cold-start row, and feature-carrying inputs.

Scope guards (documented gaps, mirrored by NotImplementedError in the
exporter): quantized dtypes, the MLflow pyfunc wrapper, and the Tier C
repeat-affinity table are EASE-only for now. Unknown-feature-name handling
also lives outside the graph: the ONNX contract takes already-resolved
``cat_ids`` / ``dense`` slots, so callers translate names via the
``user_cat_feature_to_idx`` / ``user_dense_feature_to_idx`` maps in
vocab.json (unknown names are simply not fed — same net effect as the
Rust path's silent skip).
"""

import json
import tempfile
from pathlib import Path

import polars as pl
import pytest

np = pytest.importorskip("numpy")
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

import kzn_recsys as fease

pytestmark = pytest.mark.skipif(
    not getattr(fease, "_HAS_ML_MODELS", False),
    reason="extension built without the `ml-models` feature (no Two-Tower)",
)


def _make_interactions(path: Path) -> None:
    df = pl.DataFrame(
        {
            "user_id": ["u0", "u0", "u0", "u0",
                        "u1", "u1", "u1", "u1",
                        "u2", "u2", "u2", "u2"],
            "item_id": ["A", "B", "C", "D"] * 3,
            "value": [1.0] * 12,
        }
    )
    df.write_parquet(path)


def _make_user_features(path: Path) -> None:
    # Two categorical slots (values exactly 1.0) + one dense column
    # (non-1.0 values), so the exported graph has BOTH tower branches.
    df = pl.DataFrame(
        {
            "user_id": ["u0", "u1", "u2", "u0", "u1", "u2"],
            "feature_name": ["plan_free", "plan_premium", "plan_premium",
                             "tenure_days", "tenure_days", "tenure_days"],
            "value": [1.0, 1.0, 1.0, 10.0, 250.0, 30.0],
        }
    )
    df.write_parquet(path)


def _make_item_features(path: Path) -> None:
    df = pl.DataFrame(
        {
            "item_id": ["A", "B", "C", "D"],
            "feature_name": ["genre_doc"] * 4,
            "value": [1.0] * 4,
        }
    )
    df.write_parquet(path)


@pytest.fixture(scope="module")
def trained_two_tower():
    with tempfile.TemporaryDirectory() as tmp:
        i_path = Path(tmp) / "interactions.parquet"
        u_path = Path(tmp) / "user_features.parquet"
        v_path = Path(tmp) / "item_features.parquet"
        _make_interactions(i_path)
        _make_user_features(u_path)
        _make_item_features(v_path)
        model = fease.build_and_train_two_tower(
            interactions_path=str(i_path),
            user_features_path=str(u_path),
            item_features_path=str(v_path),
            embedding_dim=8,
            temperature=0.1,
            learning_rate=0.05,
            epochs=10,
            batch_size=4,
            id_dropout=0.2,
            seed=42,
        )
        yield model


@pytest.fixture(scope="module")
def exported(trained_two_tower, tmp_path_factory):
    from kzn_recsys.onnx_export import export_onnx

    out = tmp_path_factory.mktemp("two_tower_onnx")
    res = export_onnx(trained_two_tower, out)
    sess = ort.InferenceSession(str(res.onnx_path))
    vocab = json.loads(res.vocab_path.read_text())
    return res, sess, vocab


def _predict_scores_by_index(model, user_id, vocab, features=None):
    """model.predict → per-item-index f32 score vector (full catalog)."""
    n = len(vocab["item_index_to_guid"])
    guid_to_idx = {g: i for i, g in enumerate(vocab["item_index_to_guid"])}
    out = np.zeros(n, np.float32)
    recs = (
        model.predict(user_id, features=features, top_k=n)
        if features is not None
        else model.predict(user_id, top_k=n)
    )
    assert len(recs) == n
    for guid, score in recs:
        out[guid_to_idx[guid]] = score
    return out


def _run(sess, vocab, *, user_idx, cat=None, dense=None, mask=None, seen=None, rp=0.0, k=None):
    M = vocab["num_items"]
    feeds = {
        "user_idx": np.array([user_idx], np.int64),
        "mask": np.ones((1, M), np.float32) if mask is None else mask,
        "seen": np.zeros((1, M), np.float32) if seen is None else seen,
        "repeat_penalty": np.array([[rp]], np.float32),
        "k": np.array([M if k is None else k], np.int64),
    }
    if cat is not None:
        feeds["cat_ids"] = np.array([cat], np.int64)
        feeds["cat_mask"] = np.ones((1, len(cat)), np.float32)
    if dense is not None:
        feeds["dense"] = np.array([dense], np.float32)
    out = sess.run(None, feeds)
    names = [o.name for o in sess.get_outputs()]
    return {n: out[i] for i, n in enumerate(names)}


# ---------------------------------------------------------------------------
# Payload shape
# ---------------------------------------------------------------------------


def test_export_payload_shapes_and_fields(trained_two_tower):
    from kzn_recsys.onnx_export import TwoTowerExportPayload, _payload_from_model

    p = _payload_from_model(trained_two_tower)
    assert isinstance(p, TwoTowerExportPayload)
    assert p.kind == "two_tower"
    d = p.embedding_dim
    assert p.id_embedding.shape == (p.num_users, d)
    assert p.has_cat and p.has_dense
    assert p.cat_embedding.shape == (p.num_user_categories, d)
    assert p.dense_w.shape == (p.user_dense_dim, d)
    assert p.dense_b.shape == (d,)
    assert p.hidden_w.shape == (d, d)
    assert p.out_w.shape == (d, d)
    assert p.item_matrix.shape == (p.num_items, d)
    # Catalog rows are L2-normalized.
    np.testing.assert_allclose(
        np.linalg.norm(p.item_matrix, axis=1), np.ones(p.num_items), rtol=1e-5
    )
    # Cold-start sentinel (index 0) is excluded from the user map.
    assert 0 not in p.user_id_to_index.values()
    assert set(p.user_id_to_index) == {"u0", "u1", "u2"}
    assert set(p.user_cat_feature_to_idx) == {"plan_free", "plan_premium"}
    assert set(p.user_dense_feature_to_idx) == {"tenure_days"}


# ---------------------------------------------------------------------------
# Parity: onnxruntime raw_scores vs TwoTowerModel.predict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("user_id", ["u0", "u1", "u2"])
def test_warm_user_raw_scores_parity(trained_two_tower, exported, user_id):
    _, sess, vocab = exported
    user_idx = vocab["user_id_to_index"][user_id]
    raw = _run(sess, vocab, user_idx=user_idx)["raw_scores"][0]
    expected = _predict_scores_by_index(trained_two_tower, user_id, vocab)
    np.testing.assert_allclose(raw, expected, atol=1e-4)


def test_cold_start_raw_scores_parity(trained_two_tower, exported):
    _, sess, vocab = exported
    assert vocab["cold_start_user_index"] == 0
    raw = _run(sess, vocab, user_idx=0)["raw_scores"][0]
    expected = _predict_scores_by_index(trained_two_tower, "brand_new_user", vocab)
    np.testing.assert_allclose(raw, expected, atol=1e-4)


def test_cold_start_with_categorical_feature_parity(trained_two_tower, exported):
    _, sess, vocab = exported
    cat_idx = vocab["user_cat_feature_to_idx"]["plan_premium"]
    raw = _run(sess, vocab, user_idx=0, cat=[cat_idx])["raw_scores"][0]
    expected = _predict_scores_by_index(
        trained_two_tower, "brand_new_user", vocab, features={"plan_premium": 1.0}
    )
    np.testing.assert_allclose(raw, expected, atol=1e-4)
    # And the feature must actually change the scores vs the bare prior.
    bare = _run(sess, vocab, user_idx=0)["raw_scores"][0]
    assert np.abs(raw - bare).max() > 1e-6


def test_cold_start_with_dense_feature_parity(trained_two_tower, exported):
    _, sess, vocab = exported
    dense_col = vocab["user_dense_feature_to_idx"]["tenure_days"]
    dense = [0.0] * vocab["user_dense_dim"]
    dense[dense_col] = 42.0
    raw = _run(sess, vocab, user_idx=0, dense=dense)["raw_scores"][0]
    expected = _predict_scores_by_index(
        trained_two_tower, "brand_new_user", vocab, features={"tenure_days": 42.0}
    )
    np.testing.assert_allclose(raw, expected, atol=1e-4)


def test_warm_user_with_cat_and_dense_parity(trained_two_tower, exported):
    _, sess, vocab = exported
    user_idx = vocab["user_id_to_index"]["u1"]
    cat_idx = vocab["user_cat_feature_to_idx"]["plan_free"]
    dense_col = vocab["user_dense_feature_to_idx"]["tenure_days"]
    dense = [0.0] * vocab["user_dense_dim"]
    dense[dense_col] = 7.0
    raw = _run(sess, vocab, user_idx=user_idx, cat=[cat_idx], dense=dense)["raw_scores"][0]
    expected = _predict_scores_by_index(
        trained_two_tower, "u1", vocab, features={"plan_free": 1.0, "tenure_days": 7.0}
    )
    np.testing.assert_allclose(raw, expected, atol=1e-4)


def test_top_k_ordering_matches_predict(trained_two_tower, exported):
    _, sess, vocab = exported
    guids = vocab["item_index_to_guid"]
    for user_id in ["u0", "u1", "u2"]:
        out = _run(sess, vocab, user_idx=vocab["user_id_to_index"][user_id])
        onnx_order = [guids[i] for i in out["top_indices"][0]]
        rust_order = [g for g, _ in trained_two_tower.predict(user_id, top_k=len(guids))]
        assert onnx_order == rust_order


# ---------------------------------------------------------------------------
# Tail behavior (mask / seen / repeat_penalty / k) and baked defaults
# ---------------------------------------------------------------------------


def test_default_repeat_policy_is_neutral(exported):
    """Baked defaults: with only user_idx fed, ρ = 0 → top_scores equal the
    raw scores (nothing excluded — Two-Tower has no request history)."""
    _, sess, vocab = exported
    user_idx = vocab["user_id_to_index"]["u0"]
    out = sess.run(None, {"user_idx": np.array([user_idx], np.int64)})
    names = [o.name for o in sess.get_outputs()]
    raw = out[names.index("raw_scores")][0]
    top_scores = out[names.index("top_scores")][0]
    np.testing.assert_allclose(top_scores, np.sort(raw)[::-1][: len(top_scores)], atol=1e-6)
    assert vocab["repeat_policy"]["default_penalty"] == 0.0


def test_seen_with_penalty_sinks_item(exported):
    _, sess, vocab = exported
    M = vocab["num_items"]
    user_idx = vocab["user_id_to_index"]["u0"]
    neutral = _run(sess, vocab, user_idx=user_idx)["top_indices"][0]
    seen = np.zeros((1, M), np.float32)
    seen[0, neutral[0]] = 1.0  # mark the top item as seen
    penalized = _run(sess, vocab, user_idx=user_idx, seen=seen, rp=1e9)["top_indices"][0]
    assert penalized[-1] == neutral[0]  # sinks to the bottom


def test_mask_sinks_item(exported):
    _, sess, vocab = exported
    M = vocab["num_items"]
    user_idx = vocab["user_id_to_index"]["u0"]
    neutral = _run(sess, vocab, user_idx=user_idx)["top_indices"][0]
    mask = np.ones((1, M), np.float32)
    mask[0, neutral[0]] = 0.0
    masked = _run(sess, vocab, user_idx=user_idx, mask=mask)["top_indices"][0]
    assert masked[-1] == neutral[0]


def test_top_k_clamped(exported):
    _, sess, vocab = exported
    M = vocab["num_items"]
    out = _run(sess, vocab, user_idx=0, k=M + 50)
    assert out["top_indices"].shape[1] == M


def test_batch_inference(trained_two_tower, exported):
    _, sess, vocab = exported
    idxs = [vocab["user_id_to_index"]["u0"], 0]  # warm + cold-start in one batch
    M = vocab["num_items"]
    out = sess.run(
        None,
        {
            "user_idx": np.array(idxs, np.int64),
            "mask": np.ones((2, M), np.float32),
            "seen": np.zeros((2, M), np.float32),
            "repeat_penalty": np.zeros((2, 1), np.float32),
            "k": np.array([M], np.int64),
        },
    )
    names = [o.name for o in sess.get_outputs()]
    raw = out[names.index("raw_scores")]
    assert raw.shape == (2, M)
    np.testing.assert_allclose(
        raw[0], _predict_scores_by_index(trained_two_tower, "u0", vocab), atol=1e-4
    )
    np.testing.assert_allclose(
        raw[1], _predict_scores_by_index(trained_two_tower, "brand_new_user", vocab), atol=1e-4
    )


# ---------------------------------------------------------------------------
# vocab.json contract
# ---------------------------------------------------------------------------


def test_vocab_contents(exported):
    _, _, vocab = exported
    assert vocab["model_kind"] == "two_tower"
    assert vocab["num_items"] == 4
    assert vocab["num_users"] == 4  # 3 real + cold-start row
    assert vocab["has_cat"] is True and vocab["has_dense"] is True
    assert vocab["cold_start_user_index"] == 0
    assert set(vocab["user_id_to_index"]) == {"u0", "u1", "u2"}
    assert 0 not in vocab["user_id_to_index"].values()
    assert len(vocab["item_index_to_guid"]) == 4
    names = [i["name"] for i in vocab["io_signature"]["inputs"]]
    assert names == ["user_idx", "cat_ids", "cat_mask", "dense", "mask", "seen", "repeat_penalty", "k"]
    out_names = [o["name"] for o in vocab["io_signature"]["outputs"]]
    assert out_names == ["top_indices", "top_scores", "raw_scores"]
    assert vocab["repeat_policy"] == {"default_penalty": 0.0, "per_user_table_present": False}


# ---------------------------------------------------------------------------
# No-feature model: graph omits the cat/dense branches entirely
# ---------------------------------------------------------------------------


def test_export_without_features(tmp_path):
    from kzn_recsys.onnx_export import export_onnx

    i_path = tmp_path / "interactions.parquet"
    _make_interactions(i_path)
    model = fease.build_and_train_two_tower(
        interactions_path=str(i_path),
        embedding_dim=8,
        temperature=0.1,
        learning_rate=0.05,
        epochs=5,
        batch_size=4,
        id_dropout=0.2,
        seed=42,
    )
    res = export_onnx(model, tmp_path / "out")
    sess = ort.InferenceSession(str(res.onnx_path))
    vocab = json.loads(res.vocab_path.read_text())
    assert vocab["has_cat"] is False and vocab["has_dense"] is False
    names = [i["name"] for i in vocab["io_signature"]["inputs"]]
    assert names == ["user_idx", "mask", "seen", "repeat_penalty", "k"]

    user_idx = vocab["user_id_to_index"]["u0"]
    raw = _run(sess, vocab, user_idx=user_idx)["raw_scores"][0]
    expected = _predict_scores_by_index(model, "u0", vocab)
    np.testing.assert_allclose(raw, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Scope guards
# ---------------------------------------------------------------------------


def test_quantized_dtype_not_implemented(trained_two_tower, tmp_path):
    from kzn_recsys.onnx_export import export_onnx

    with pytest.raises(NotImplementedError, match="quantized"):
        export_onnx(trained_two_tower, tmp_path, dtype="fp16")


def test_mlflow_not_implemented(trained_two_tower, tmp_path):
    from kzn_recsys.onnx_export import export_onnx

    with pytest.raises(NotImplementedError, match="MLflow"):
        export_onnx(trained_two_tower, tmp_path, mlflow=True)


def test_repeat_affinity_not_implemented(trained_two_tower, tmp_path):
    from kzn_recsys.onnx_export import export_onnx

    inter = tmp_path / "i.parquet"
    _make_interactions(inter)
    with pytest.raises(NotImplementedError, match="EASE-only"):
        export_onnx(trained_two_tower, tmp_path, interactions=inter)


def test_explicit_repeat_penalty_default_is_honored(trained_two_tower, tmp_path):
    from kzn_recsys.onnx_export import export_onnx

    res = export_onnx(trained_two_tower, tmp_path, repeat_penalty_default=2.5)
    vocab = json.loads(res.vocab_path.read_text())
    assert vocab["repeat_policy"]["default_penalty"] == 2.5
