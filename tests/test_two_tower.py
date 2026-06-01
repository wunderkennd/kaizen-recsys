"""Two-Tower end-to-end smoke test: train -> predict -> save -> load.

Two-Tower is compiled only when the Rust extension is built with the
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
    reason="extension built without the `ml-models` feature (no Two-Tower)",
)


def _make_interactions(path: Path) -> None:
    """Three users, four items, each user buys every item."""
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
    df = pl.DataFrame(
        {
            "user_id": ["u0", "u1", "u2"],
            "feature_name": ["plan_free", "plan_premium", "plan_premium"],
            "value": [1.0, 1.0, 1.0],
        }
    )
    df.write_parquet(path)


def _make_item_features(path: Path) -> None:
    df = pl.DataFrame(
        {
            "item_id": ["A", "B", "C", "D"],
            "feature_name": ["genre_doc"] * 4,
            "value": [1.0, 1.0, 1.0, 1.0],
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
        yield model, tmp


def test_train_sets_dimensions(trained_two_tower):
    model, _ = trained_two_tower
    assert model.num_items == 4
    # 3 real users + 1 reserved cold-start row at index 0.
    assert model.num_users == 4


def test_predict_warm_user_returns_ranked_items(trained_two_tower):
    model, _ = trained_two_tower
    recs = model.predict("u0", top_k=4)
    assert isinstance(recs, list)
    assert len(recs) == 4
    item_ids = [r[0] for r in recs]
    assert set(item_ids) == {"A", "B", "C", "D"}
    scores = [r[1] for r in recs]
    assert all(isinstance(s, float) for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_predict_unknown_user_falls_back_to_cold_start(trained_two_tower):
    model, _ = trained_two_tower
    # Unknown ids are not an error — the model uses the reserved
    # cold-start row that id-dropout trains as a learned prior.
    recs = model.predict("brand_new_user", top_k=3)
    assert isinstance(recs, list)
    assert len(recs) == 3


def test_similar_items(trained_two_tower):
    model, _ = trained_two_tower
    sim = model.predict_similar_items("A", top_k=2)
    assert isinstance(sim, list)
    assert len(sim) <= 2
    assert all(item_id != "A" for item_id, _ in sim)
    # Unknown query -> empty list, not an error.
    assert model.predict_similar_items("NOPE", top_k=2) == []


def test_validate(trained_two_tower):
    model, _ = trained_two_tower
    passed, messages = model.validate()
    assert passed, f"validation failed: {messages}"


def test_save_load_roundtrip_preserves_predictions(trained_two_tower):
    model, tmp = trained_two_tower
    path = Path(tmp) / "two_tower.ftwo"
    model.save(str(path))
    assert path.exists()

    loaded = fease.load_two_tower_model(str(path))
    assert loaded.num_items == model.num_items
    assert loaded.num_users == model.num_users

    before = model.predict("u0", top_k=4)
    after = loaded.predict("u0", top_k=4)
    assert [r[0] for r in before] == [r[0] for r in after]


def test_train_without_features(tmp_path):
    """Training with no user/item feature files should work (no side info)."""
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
    assert model.num_items == 4
    recs = model.predict("u0", top_k=2)
    assert isinstance(recs, list)
    assert len(recs) == 2


# ---------------------------------------------------------------------------
# #55: predict-time arbitrary user features
# ---------------------------------------------------------------------------


def test_predict_features_arg_is_optional(trained_two_tower):
    """Bare call with no `features` keeps the existing behavior."""
    model, _ = trained_two_tower
    recs_no_features = model.predict("u0", top_k=4)
    recs_empty_features = model.predict("u0", features={}, top_k=4)
    # Same ranking — empty dict is equivalent to no dict.
    assert [r[0] for r in recs_no_features] == [r[0] for r in recs_empty_features]


def test_predict_unknown_feature_names_are_skipped(trained_two_tower):
    """Unknown feature names are silently dropped, not an error."""
    model, _ = trained_two_tower
    recs = model.predict(
        "brand_new_user",
        features={"never_seen": 1.0, "totally_made_up": 99.0},
        top_k=3,
    )
    assert isinstance(recs, list)
    assert len(recs) == 3


def test_predict_cold_start_with_features_differs_from_bare(trained_two_tower):
    """#55 acceptance: a cold-start user with informative features should
    produce a different ranking than the bare cold-start row."""
    model, _ = trained_two_tower
    bare = model.predict("brand_new_user", top_k=4)
    with_feature = model.predict(
        "brand_new_user",
        features={"plan_premium": 1.0},
        top_k=4,
    )
    bare_scores = [s for _, s in bare]
    with_feature_scores = [s for _, s in with_feature]
    # Same items in both (the catalog is the same), but at least one
    # score must differ — the categorical feature embedding has been
    # combined into the user vector.
    score_diff = max(
        abs(a - b) for a, b in zip(bare_scores, with_feature_scores)
    )
    assert score_diff > 1e-6, (
        f"cold-start with feature should differ from bare; max diff = {score_diff}"
    )


def test_save_load_roundtrip_preserves_feature_maps(trained_two_tower):
    """Save/load must preserve the user-feature maps so features work
    after a round-trip (#55 acceptance)."""
    model, tmp = trained_two_tower
    path = Path(tmp) / "two_tower_v5.ftwo"
    model.save(str(path))
    loaded = fease.load_two_tower_model(str(path))

    features = {"plan_premium": 1.0}
    before = model.predict("brand_new_user", features=features, top_k=4)
    after = loaded.predict("brand_new_user", features=features, top_k=4)
    # Item rankings must match byte-for-byte across save/load.
    assert [r[0] for r in before] == [r[0] for r in after]
    # Scores match to within model-output float tolerance.
    for (_, sb), (_, sa) in zip(before, after):
        assert abs(sb - sa) < 1e-5
