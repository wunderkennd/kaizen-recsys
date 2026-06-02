"""SASRec end-to-end smoke test: train -> predict -> save -> load -> evaluate.

SASRec is compiled only when the Rust extension is built with the
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
    reason="extension built without the `ml-models` feature (no SASRec)",
)


def _make_interactions(path: Path) -> None:
    """Three users with clear, order-bearing item sequences.

    SASRec requires a numeric `days_ago` column to order each user's
    history chronologically (smaller == more recent).
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
            # Oldest first within each user: larger days_ago == older.
            "days_ago": [
                4.0, 3.0, 2.0, 1.0,
                4.0, 3.0, 2.0, 1.0,
                4.0, 3.0, 2.0, 1.0,
            ],
        }
    )
    df.write_parquet(path)


@pytest.fixture(scope="module")
def trained_sasrec():
    with tempfile.TemporaryDirectory() as tmp:
        i_path = Path(tmp) / "interactions.parquet"
        _make_interactions(i_path)

        model = fease.build_and_train_sasrec(
            interactions_path=str(i_path),
            embedding_dim=16,
            max_seq_len=8,
            num_heads=2,
            num_layers=2,
            dropout=0.0,
            num_epochs=15,
            batch_size=4,
            learning_rate=1e-2,
            patience=15,
            seed=42,
        )
        yield model, tmp, str(i_path)


def test_train_sets_dimensions(trained_sasrec):
    model, _, _ = trained_sasrec
    # 4 catalog items (A, B, C, D); pad token excluded from num_items.
    assert model.num_items == 4
    assert model.max_seq_len == 8


def test_predict_returns_ranked_unseen_items(trained_sasrec):
    model, _, _ = trained_sasrec
    recs = model.predict(["A", "B", "C"], top_k=10)
    assert isinstance(recs, list)
    assert len(recs) >= 1
    item_ids = [r[0] for r in recs]
    # History items are excluded from recommendations.
    assert "A" not in item_ids and "B" not in item_ids and "C" not in item_ids
    # Scores are floats, sorted descending.
    scores = [r[1] for r in recs]
    assert all(isinstance(s, float) for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_predict_unknown_items_are_skipped(trained_sasrec):
    model, _, _ = trained_sasrec
    # Unknown ids in history are ignored, not an error.
    recs = model.predict(["A", "UNKNOWN_ITEM"], top_k=5)
    assert isinstance(recs, list)


def test_similar_items(trained_sasrec):
    model, _, _ = trained_sasrec
    sim = model.predict_similar_items("A", top_k=2)
    assert isinstance(sim, list)
    assert len(sim) <= 2
    assert all(item_id != "A" for item_id, _ in sim)
    # Unknown query -> empty list, not an error.
    assert model.predict_similar_items("NOPE", top_k=2) == []


def test_validate(trained_sasrec):
    model, _, _ = trained_sasrec
    passed, messages = model.validate()
    assert passed, f"validation failed: {messages}"


def test_save_load_roundtrip_preserves_predictions(trained_sasrec):
    model, tmp, _ = trained_sasrec
    path = Path(tmp) / "sasrec.fsat"
    model.save(str(path))
    assert path.exists()

    loaded = fease.load_sasrec_model(str(path))
    assert loaded.num_items == model.num_items

    before = model.predict(["A", "B"], top_k=4)
    after = loaded.predict(["A", "B"], top_k=4)
    assert [r[0] for r in before] == [r[0] for r in after]
    for (_, sb), (_, sa) in zip(before, after):
        assert abs(sb - sa) < 1e-4


def test_evaluate_runs_via_recmodel_harness(trained_sasrec):
    model, _, i_path = trained_sasrec
    # The generalized &dyn RecModel eval harness drives SASRec end to end;
    # reuse the training file as both train and test split for the smoke.
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
