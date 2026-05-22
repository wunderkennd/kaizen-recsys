"""Heterogeneous `ModelRegistry` smoke test (#56).

Registers an EASE model, a SASRec model, and a Two-Tower model under
three territories on the same registry, then drives each predict path.
Gated on `ml-models` since SASRec/Two-Tower training only works when the
extension is built with the feature.
"""

import tempfile
from pathlib import Path

import polars as pl
import pytest

import kzn_recsys as fease

pytestmark = pytest.mark.skipif(
    not getattr(fease, "_HAS_ML_MODELS", False),
    reason="extension built without the `ml-models` feature",
)


def _make_ease_inputs(tmp: Path) -> tuple[str, str, str]:
    """Tiny EASE training inputs (interactions + user/item features).

    EASE catalog: items I1, I2, I3, I4. Users U1, U2, U3 with one
    one-hot user feature each.
    """
    inter = pl.DataFrame(
        {
            "user_id": ["U1", "U1", "U2", "U2", "U3", "U3"],
            "item_id": ["I1", "I2", "I2", "I3", "I3", "I4"],
            "value": [1.0] * 6,
        }
    )
    uf = pl.DataFrame(
        {
            "user_id": ["U1", "U2", "U3"],
            "feature_name": ["seg_a", "seg_a", "seg_b"],
            "value": [1.0, 1.0, 1.0],
        }
    )
    it = pl.DataFrame(
        {
            "item_id": ["I1", "I2", "I3", "I4"],
            "feature_name": ["genre"] * 4,
            "value": [1.0, 1.0, 1.0, 1.0],
        }
    )
    i_path = tmp / "ease_interactions.parquet"
    u_path = tmp / "ease_user_features.parquet"
    f_path = tmp / "ease_item_features.parquet"
    inter.write_parquet(i_path)
    uf.write_parquet(u_path)
    it.write_parquet(f_path)
    return str(i_path), str(u_path), str(f_path)


def _make_sasrec_inputs(tmp: Path) -> str:
    """SASRec needs `days_ago` for chronological order."""
    df = pl.DataFrame(
        {
            "user_id": ["U1"] * 4 + ["U2"] * 4 + ["U3"] * 4,
            "item_id": ["A", "B", "C", "D"] * 3,
            "value": [1.0] * 12,
            "days_ago": [4.0, 3.0, 2.0, 1.0] * 3,
        }
    )
    path = tmp / "sasrec_interactions.parquet"
    df.write_parquet(path)
    return str(path)


def _make_two_tower_inputs(tmp: Path) -> tuple[str, str]:
    inter = pl.DataFrame(
        {
            "user_id": ["U1"] * 4 + ["U2"] * 4 + ["U3"] * 4,
            "item_id": ["X", "Y", "Z", "W"] * 3,
            "value": [1.0] * 12,
        }
    )
    uf = pl.DataFrame(
        {
            "user_id": ["U1", "U2", "U3"],
            "feature_name": ["plan_a", "plan_a", "plan_b"],
            "value": [1.0, 1.0, 1.0],
        }
    )
    i_path = tmp / "tt_interactions.parquet"
    u_path = tmp / "tt_user_features.parquet"
    inter.write_parquet(i_path)
    uf.write_parquet(u_path)
    return str(i_path), str(u_path)


@pytest.fixture(scope="module")
def heterogeneous_registry():
    """Build a registry with one EASE, one SASRec, one Two-Tower model."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ease_i, ease_u, ease_f = _make_ease_inputs(tmp)
        sasrec_i = _make_sasrec_inputs(tmp)
        tt_i, tt_u = _make_two_tower_inputs(tmp)

        ease_model = fease.build_and_train(
            interactions_path=ease_i,
            user_features_path=ease_u,
            item_features_path=ease_f,
            alpha=1.0,
            beta=1.0,
            lambda_=100.0,
        )
        sasrec_model = fease.build_and_train_sasrec(
            interactions_path=sasrec_i,
            embedding_dim=8,
            max_seq_len=6,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
            num_epochs=5,
            batch_size=4,
            learning_rate=1e-2,
            patience=5,
            seed=42,
        )
        two_tower_model = fease.build_and_train_two_tower(
            interactions_path=tt_i,
            user_features_path=tt_u,
            embedding_dim=8,
            temperature=0.1,
            learning_rate=0.05,
            epochs=5,
            batch_size=4,
            id_dropout=0.2,
            seed=42,
        )

        registry = fease.ModelRegistry()
        registry.register("US", ease_model)
        registry.register_sasrec("UK", sasrec_model)
        registry.register_two_tower("BR", two_tower_model)

        yield registry


def test_registry_holds_three_model_families(heterogeneous_registry):
    registry = heterogeneous_registry
    assert len(registry) == 3
    assert set(registry.territories()) == {"US", "UK", "BR"}


def test_predict_top_k_ease_routes_through_string_ids(heterogeneous_registry):
    registry = heterogeneous_registry
    recs = registry.predict_top_k_ease(
        "US",
        interactions={"I1": 1.0},
        features={"seg_a": 1.0},
        top_k=3,
    )
    assert isinstance(recs, list)
    item_ids = [r[0] for r in recs]
    # All returned ids are catalog items, and I1 (interacted) is excluded.
    assert all(i in {"I1", "I2", "I3", "I4"} for i in item_ids)
    assert "I1" not in item_ids


def test_predict_top_k_sasrec_routes_through_string_ids(heterogeneous_registry):
    registry = heterogeneous_registry
    recs = registry.predict_top_k_sasrec("UK", history=["A", "B"], top_k=3)
    assert isinstance(recs, list)
    item_ids = [r[0] for r in recs]
    assert all(i in {"A", "B", "C", "D"} for i in item_ids)
    # History items are excluded from recommendations.
    assert "A" not in item_ids and "B" not in item_ids


def test_predict_top_k_two_tower_routes_through_string_ids(heterogeneous_registry):
    registry = heterogeneous_registry
    recs = registry.predict_top_k_two_tower(
        "BR",
        user_id="U1",
        features={"plan_a": 1.0},
        top_k=3,
    )
    assert isinstance(recs, list)
    item_ids = [r[0] for r in recs]
    assert all(i in {"X", "Y", "Z", "W"} for i in item_ids)


def test_predict_top_k_two_tower_cold_start_user(heterogeneous_registry):
    registry = heterogeneous_registry
    # Unknown user falls back to the reserved cold-start row.
    recs = registry.predict_top_k_two_tower("BR", user_id="brand_new", top_k=3)
    assert isinstance(recs, list)
    assert len(recs) == 3


def test_wrong_method_on_territory_errors(heterogeneous_registry):
    """Calling predict_top_k_sasrec on the EASE territory should error
    with a useful message pointing at the correct method."""
    registry = heterogeneous_registry
    with pytest.raises(ValueError) as exc_info:
        registry.predict_top_k_sasrec("US", history=["I1"], top_k=3)
    msg = str(exc_info.value)
    assert "predict_top_k_sasrec" in msg
    assert "predict_top_k_ease" in msg


def test_back_compat_predict_top_k_still_works(heterogeneous_registry):
    """The legacy index-based predict_top_k must still work on EASE
    territories byte-for-byte (#56 preserves this for back-compat)."""
    registry = heterogeneous_registry
    # Index-based: (item_index, value) pairs. For the EASE model trained
    # above, item index 0 corresponds to whichever id was interned first.
    ranked = registry.predict_top_k("US", [(0, 1.0)], top_k=3)
    assert isinstance(ranked, list)
    # Returns (int_idx, float_score) — the old API shape.
    if ranked:
        assert isinstance(ranked[0][0], int)
        assert isinstance(ranked[0][1], float)
