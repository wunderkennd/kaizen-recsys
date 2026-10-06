"""BERT4Rec end-to-end smoke test: train -> predict -> embed -> save -> load -> evaluate.

BERT4Rec is compiled only when the Rust extension is built with the
`ml-models` Cargo feature. When the EASE-only wheel is installed these
symbols are absent, so the whole module is skipped.
"""

import tempfile
from pathlib import Path

import polars as pl
import pytest

import kzn_recsys as fease

pytestmark = pytest.mark.skipif(
    not getattr(fease, "_HAS_ML_MODELS", False),
    reason="extension built without the `ml-models` feature (no BERT4Rec)",
)

EMBEDDING_DIM = 16


def _make_interactions(path: Path) -> None:
    """Three users over a 4-item catalog.

    BERT4Rec requires a numeric `days_ago` column: it orders each user's
    history and is log2-bucketed into the model's relative positions.
    """
    df = pl.DataFrame(
        {
            "user_id": [
                "u0", "u0", "u0", "u0",
                "u1", "u1", "u1", "u1",
                "u2", "u2", "u2", "u2",
            ],
            "item_id": [
                "A", "B", "C", "D",
                "A", "B", "C", "D",
                "D", "C", "B", "A",
            ],
            "value": [1.0] * 12,
            # Larger days_ago == older. Spread over log2 buckets 0..3.
            "days_ago": [
                9.0, 5.0, 2.0, 1.0,
                9.0, 5.0, 2.0, 1.0,
                9.0, 5.0, 2.0, 1.0,
            ],
        }
    )
    df.write_parquet(path)


@pytest.fixture(scope="module")
def trained_bert4rec():
    with tempfile.TemporaryDirectory() as tmp:
        i_path = Path(tmp) / "interactions.parquet"
        _make_interactions(i_path)

        model = fease.build_and_train_bert4rec(
            interactions_path=str(i_path),
            embedding_dim=EMBEDDING_DIM,
            max_seq_len=8,
            num_position_buckets=8,
            num_heads=2,
            num_layers=2,
            dropout=0.0,
            mask_ratio=0.3,
            num_epochs=15,
            batch_size=4,
            learning_rate=1e-2,
            patience=15,
            seed=42,
        )
        yield model, tmp, str(i_path)


def test_train_sets_dimensions(trained_bert4rec):
    model, _, _ = trained_bert4rec
    # 4 catalog items (A, B, C, D); PAD + MASK excluded from num_items.
    assert model.num_items == 4
    assert model.embedding_dim == EMBEDDING_DIM
    assert model.max_seq_len == 8
    assert model.num_position_buckets == 8


def test_bert4rec_train_and_predict(trained_bert4rec):
    model, _, _ = trained_bert4rec
    recs = model.predict(["A", "B", "C"], top_k=10)
    assert isinstance(recs, list)
    assert len(recs) >= 1
    item_ids = [r[0] for r in recs]
    # History items are excluded from recommendations.
    assert "A" not in item_ids and "B" not in item_ids and "C" not in item_ids
    scores = [r[1] for r in recs]
    assert all(isinstance(s, float) for s in scores)
    assert scores == sorted(scores, reverse=True)

    # Optional days_ago (parallel to history) is accepted and bucketed.
    with_days = model.predict(["A", "B", "C"], days_ago=[9.0, 5.0, 2.0], top_k=10)
    assert [r[0] for r in with_days] == ["D"]
    # Mismatched lengths are a ValueError, not a silent truncation.
    with pytest.raises(ValueError):
        model.predict(["A", "B"], days_ago=[1.0], top_k=1)


def test_predict_unknown_items_are_skipped(trained_bert4rec):
    model, _, _ = trained_bert4rec
    recs = model.predict(["A", "UNKNOWN_ITEM"], top_k=5)
    assert isinstance(recs, list)
    assert all(r[0] != "A" for r in recs)


def test_bert4rec_embed_users_returns_correct_shape(trained_bert4rec):
    model, _, i_path = trained_bert4rec
    emb = model.embed_users(i_path)
    assert isinstance(emb, dict)
    assert set(emb) == {"u0", "u1", "u2"}
    for vec in emb.values():
        assert len(vec) == EMBEDDING_DIM
        assert all(isinstance(v, float) for v in vec)
    # u0 and u1 have identical histories -> identical embeddings; u2's
    # history is reversed in time -> different positions -> different vector.
    assert emb["u0"] == pytest.approx(emb["u1"])
    assert emb["u2"] != pytest.approx(emb["u0"])
    # The single-user API agrees with the batch extractor.
    single = model.embed_user(["A", "B", "C", "D"], days_ago=[9.0, 5.0, 2.0, 1.0])
    assert single == pytest.approx(emb["u0"], abs=1e-5)
    # Empty / unknown history -> zero vector (hybrid cold-start convention).
    assert model.embed_user([]) == [0.0] * EMBEDDING_DIM
    assert model.embed_user(["NOPE"]) == [0.0] * EMBEDDING_DIM


def test_embed_users_requires_days_ago(trained_bert4rec):
    model, tmp, _ = trained_bert4rec
    bad = Path(tmp) / "no_days.parquet"
    pl.DataFrame({"user_id": ["u0"], "item_id": ["A"], "value": [1.0]}).write_parquet(bad)
    with pytest.raises(ValueError, match="days_ago"):
        model.embed_users(str(bad))


def test_embed_items_covers_catalog(trained_bert4rec):
    model, _, _ = trained_bert4rec
    items = model.embed_items()
    assert set(items) == {"A", "B", "C", "D"}
    assert all(len(v) == EMBEDDING_DIM for v in items.values())


def test_similar_items(trained_bert4rec):
    model, _, _ = trained_bert4rec
    sim = model.predict_similar_items("A", top_k=2)
    assert isinstance(sim, list)
    assert len(sim) <= 2
    assert all(item_id != "A" for item_id, _ in sim)
    assert model.predict_similar_items("NOPE", top_k=2) == []


def test_validate(trained_bert4rec):
    model, _, _ = trained_bert4rec
    passed, messages = model.validate()
    assert passed, f"validation failed: {messages}"


def test_bert4rec_save_load_roundtrip(trained_bert4rec):
    model, tmp, _ = trained_bert4rec
    path = Path(tmp) / "bert4rec.fb4r"
    model.save(str(path))
    assert path.exists()
    assert path.read_bytes()[:4] == b"FB4R"

    loaded = fease.load_bert4rec_model(str(path))
    assert loaded.num_items == model.num_items
    assert loaded.embedding_dim == model.embedding_dim

    before = model.predict(["A", "B"], days_ago=[3.0, 1.0], top_k=4)
    after = loaded.predict(["A", "B"], days_ago=[3.0, 1.0], top_k=4)
    assert [r[0] for r in before] == [r[0] for r in after]
    for (_, sb), (_, sa) in zip(before, after):
        assert abs(sb - sa) < 1e-4
    assert loaded.embed_user(["A", "B"]) == pytest.approx(model.embed_user(["A", "B"]), abs=1e-5)

    # The EASE loader points at the right loader instead of a bare magic error.
    with pytest.raises(Exception, match="load_bert4rec_model"):
        fease.load_model(str(path))


def test_evaluate_runs_via_recmodel_harness(trained_bert4rec):
    model, _, i_path = trained_bert4rec
    report = model.evaluate(
        test_interactions_path=i_path,
        train_interactions_path=i_path,
        k_values=[1, 2],
    )
    assert report["num_users"] >= 1
    assert "coverage" in report
    assert len(report["metrics"]) == 2
    for m in report["metrics"]:
        for key in ("k", "precision", "recall", "ndcg", "map", "hit_rate"):
            assert key in m


def test_registry_routes_bert4rec(trained_bert4rec):
    model, _, _ = trained_bert4rec
    registry = fease.ModelRegistry()
    registry.register_bert4rec("JP", model)
    recs = registry.predict_top_k_bert4rec("JP", ["A", "B", "C"], days_ago=[9.0, 5.0, 2.0], top_k=10)
    assert [r[0] for r in recs] == ["D"]
    direct = model.predict(["A", "B", "C"], days_ago=[9.0, 5.0, 2.0], top_k=10)
    assert abs(recs[0][1] - direct[0][1]) < 1e-5
    # Wrong-model dispatch names the right method.
    with pytest.raises(ValueError, match="predict_top_k_bert4rec"):
        registry.predict_top_k_sasrec("JP", ["A"], top_k=1)


def test_invalid_architecture_is_a_value_error(trained_bert4rec):
    _, _, i_path = trained_bert4rec
    with pytest.raises(ValueError, match="num_heads"):
        fease.build_and_train_bert4rec(
            interactions_path=i_path, embedding_dim=10, num_heads=4, num_epochs=1
        )
