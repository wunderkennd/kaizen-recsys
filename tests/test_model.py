import os
import tempfile
from pathlib import Path

import polars as pl
import pytest
import rust_fease_recommender as fease


@pytest.fixture(scope="session")
def trained_model():
    """
    Creates long-format training data (interactions, user_features, item_features),
    trains a FEASE model, and yields the model for all tests.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        i_path = Path(tmpdir) / "interactions.parquet"
        u_path = Path(tmpdir) / "user_features.parquet"
        t_path = Path(tmpdir) / "item_features.parquet"

        # Interactions: user_id, item_id, value
        # u0 likes G0 and G2 (action shows)
        # u1 likes G1 (comedy movie)
        # u2 is a cold-start user (appears only in user_features)
        interactions_df = pl.DataFrame(
            {
                "user_id": ["u0", "u0", "u1"],
                "item_id": ["G0", "G2", "G1"],
                "value": [5.0, 4.0, 6.0],
            }
        )

        # User features: user_id, feature_name, value
        user_features_df = pl.DataFrame(
            {
                "user_id": [
                    "u0", "u0", "u0",
                    "u1", "u1", "u1",
                    "u2", "u2", "u2",
                ],
                "feature_name": [
                    "device_Mobile", "plan_Premium", "region_US",
                    "device_Mobile", "plan_Free", "region_EMEA",
                    "device_Console", "plan_Premium", "region_APAC",
                ],
                "value": [1.0] * 9,
            }
        )

        # Item features: item_id, feature_name, value
        item_features_df = pl.DataFrame(
            {
                "item_id": [
                    "G0", "G0",
                    "G1", "G1",
                    "G2", "G2",
                    "G3", "G3",
                ],
                "feature_name": [
                    "genre_Action", "type_episode",
                    "genre_Comedy", "type_movie",
                    "genre_Action", "type_episode",
                    "genre_Comedy", "type_movie",
                ],
                "value": [1.0] * 8,
            }
        )

        interactions_df.write_parquet(i_path)
        user_features_df.write_parquet(u_path)
        item_features_df.write_parquet(t_path)

        model = fease.build_and_train(
            interactions_path=str(i_path),
            user_features_path=str(u_path),
            item_features_path=str(t_path),
            alpha=1.0,
            beta=1.0,
            lambda_=10.0,
            meta_weight=0.0,
        )

        yield model, tmpdir


@pytest.fixture(scope="session")
def model(trained_model):
    return trained_model[0]


@pytest.fixture(scope="session")
def tmpdir_path(trained_model):
    return trained_model[1]


# --- Core Tests ---

def test_model_training(model):
    """Tests that the model was trained with the correct dimensions."""
    assert model.num_items == 4  # G0, G1, G2, G3
    assert model.num_user_features > 0
    assert model.num_item_features > 0


def test_warm_user_prediction(model):
    """Tests prediction for a user with known interactions."""
    interactions = {"G0": 5.0, "G2": 4.0}
    features = {"device_Mobile": 1.0, "plan_Premium": 1.0, "region_US": 1.0}

    recs = model.predict(interactions, features, top_k=4)

    assert len(recs) > 0
    # Scores should be non-zero for a warm user
    assert recs[0][1] != 0.0
    print("\nWarm User (u0) Recs:", recs)


def test_cold_user_prediction(model):
    """Tests prediction for a cold-start user (features only)."""
    interactions = {}
    features = {"device_Console": 1.0, "plan_Premium": 1.0, "region_APAC": 1.0}

    recs = model.predict(interactions, features, top_k=4)

    assert len(recs) == 4
    # Cold user with features should get non-zero scores
    assert recs[0][1] != 0.0
    print("\nCold User (u2) Recs:", recs)


def test_unknown_user_prediction(model):
    """Tests prediction for a user with no features and no interactions."""
    recs = model.predict({}, {}, top_k=4)

    assert len(recs) == 4
    # Truly unknown user: all scores should be 0
    for _, score in recs:
        assert score == 0.0


def test_top_k(model):
    """Tests that top_k is respected."""
    features = {"device_Console": 1.0, "plan_Premium": 1.0}

    recs = model.predict({}, features, top_k=2)
    assert len(recs) == 2

    recs_all = model.predict({}, features, top_k=999)
    assert len(recs_all) == 4  # Total number of items


# --- Phase 3+4 Feature Tests ---

def test_predict_similar_items(model):
    """Tests MLT (More-Like-This) item similarity."""
    similar = model.predict_similar_items("G0", top_k=3)

    assert len(similar) <= 3
    # G0 should not appear in its own similar items
    guids = [guid for guid, _ in similar]
    assert "G0" not in guids
    # G2 shares genre_Action with G0, so it should rank higher
    if len(similar) >= 2:
        scores = {guid: score for guid, score in similar}
        if "G2" in scores and "G1" in scores:
            assert scores["G2"] > scores["G1"]
    print("\nSimilar to G0:", similar)


def test_predict_similar_items_unknown(model):
    """Tests MLT for an unknown item returns empty list."""
    similar = model.predict_similar_items("UNKNOWN_ITEM", top_k=5)
    assert len(similar) == 0


def test_validate(model):
    """Tests the model validation report."""
    passed, messages = model.validate()
    assert passed is True
    assert len(messages) > 0


def test_save_load_roundtrip(model, tmpdir_path):
    """Tests saving and loading a model produces identical predictions."""
    save_path = os.path.join(tmpdir_path, "test_model.fease")

    # Save
    model.save(save_path)
    assert os.path.exists(save_path)

    # Load
    loaded = fease.load_model(save_path)
    assert loaded.num_items == model.num_items
    assert loaded.num_user_features == model.num_user_features
    assert loaded.num_item_features == model.num_item_features

    # Predictions should match
    interactions = {"G0": 5.0}
    features = {"device_Mobile": 1.0}
    original_recs = model.predict(interactions, features, top_k=4)
    loaded_recs = loaded.predict(interactions, features, top_k=4)

    for (g1, s1), (g2, s2) in zip(original_recs, loaded_recs):
        assert g1 == g2
        assert abs(s1 - s2) < 1e-10


def test_predict_batch(model):
    """Tests batch prediction matches sequential predictions."""
    users = [
        {"interactions": {"G0": 5.0}, "features": {"device_Mobile": 1.0}},
        {"interactions": {}, "features": {"device_Console": 1.0}},
        {"interactions": {}, "features": {}},
    ]

    batch_results = model.predict_batch(users, top_k=4)
    assert len(batch_results) == 3

    # Compare with sequential predictions
    for user, batch_recs in zip(users, batch_results):
        seq_recs = model.predict(user["interactions"], user["features"], top_k=4)
        for (bg, bs), (sg, ss) in zip(batch_recs, seq_recs):
            assert bg == sg
            assert abs(bs - ss) < 1e-10


def test_validate_data():
    """Tests the data quality validation function."""
    historical_users = [100.0, 105.0, 98.0, 102.0, 101.0]
    historical_items = [50.0, 52.0, 49.0, 51.0, 50.0]
    historical_interactions = [1000.0, 1050.0, 980.0, 1020.0, 1010.0]

    # Normal values should pass
    passed, messages = fease.validate_data(
        historical_users, historical_items, historical_interactions,
        current_users=103.0, current_items=51.0, current_interactions=1030.0,
    )
    assert passed is True

    # Anomalous value should fail
    passed, messages = fease.validate_data(
        historical_users, historical_items, historical_interactions,
        current_users=500.0, current_items=51.0, current_interactions=1030.0,
    )
    assert passed is False


# --- Advanced Weighting Tests ---

@pytest.fixture(scope="session")
def weighting_data():
    """Creates test data with event_type and days_ago columns for weighting tests."""
    tmpdir = tempfile.mkdtemp()
    i_path = Path(tmpdir) / "interactions.parquet"
    u_path = Path(tmpdir) / "user_features.parquet"
    t_path = Path(tmpdir) / "item_features.parquet"

    # Interactions with event_type and days_ago columns
    interactions_df = pl.DataFrame(
        {
            "user_id": ["u0", "u0", "u1", "u1"],
            "item_id": ["G0", "G1", "G1", "G2"],
            "value": [1.0, 1.0, 1.0, 1.0],
            "event_type": ["purchase", "click", "click", "cart"],
            "days_ago": [0.0, 30.0, 10.0, 100.0],
        }
    )

    user_features_df = pl.DataFrame(
        {
            "user_id": ["u0", "u1"],
            "feature_name": ["plan_Premium", "plan_Free"],
            "value": [1.0, 1.0],
        }
    )

    item_features_df = pl.DataFrame(
        {
            "item_id": ["G0", "G1", "G2"],
            "feature_name": ["genre_Action", "genre_Comedy", "genre_Drama"],
            "value": [1.0, 1.0, 1.0],
        }
    )

    interactions_df.write_parquet(i_path)
    user_features_df.write_parquet(u_path)
    item_features_df.write_parquet(t_path)

    return str(i_path), str(u_path), str(t_path)


def test_train_with_ips(weighting_data):
    """Tests that training with IPS succeeds and produces a valid model."""
    i_path, u_path, t_path = weighting_data
    model = fease.build_and_train(
        i_path, u_path, t_path,
        alpha=1.0, beta=1.0, lambda_=10.0,
        ips_alpha=0.5,
    )
    assert model.num_items == 3
    passed, _ = model.validate()
    assert passed


def test_train_with_decay(weighting_data):
    """Tests that training with temporal decay succeeds."""
    i_path, u_path, t_path = weighting_data
    model = fease.build_and_train(
        i_path, u_path, t_path,
        alpha=1.0, beta=1.0, lambda_=10.0,
        decay_rate=0.01,
    )
    assert model.num_items == 3
    passed, _ = model.validate()
    assert passed


def test_train_with_event_weights(weighting_data):
    """Tests that training with event-type weights succeeds."""
    i_path, u_path, t_path = weighting_data
    model = fease.build_and_train(
        i_path, u_path, t_path,
        alpha=1.0, beta=1.0, lambda_=10.0,
        event_weights={"click": 1.0, "cart": 3.0, "purchase": 5.0},
    )
    assert model.num_items == 3
    passed, _ = model.validate()
    assert passed


def test_train_with_pruning(weighting_data):
    """Tests that sparsity pruning produces a valid model."""
    i_path, u_path, t_path = weighting_data
    model = fease.build_and_train(
        i_path, u_path, t_path,
        alpha=1.0, beta=1.0, lambda_=10.0,
        sparsity_threshold=0.001,
    )
    assert model.num_items == 3
    passed, _ = model.validate()
    assert passed


def test_backward_compat(weighting_data):
    """Tests that omitting all weighting params produces same results as before."""
    i_path, u_path, t_path = weighting_data

    # Train without any weighting
    model1 = fease.build_and_train(
        i_path, u_path, t_path,
        alpha=1.0, beta=1.0, lambda_=10.0,
    )

    # Train with explicit defaults (should be identical)
    model2 = fease.build_and_train(
        i_path, u_path, t_path,
        alpha=1.0, beta=1.0, lambda_=10.0,
        decay_rate=0.0, ips_alpha=0.0, sparsity_threshold=0.0,
    )

    interactions = {"G0": 1.0}
    features = {"plan_Premium": 1.0}
    recs1 = model1.predict(interactions, features, top_k=3)
    recs2 = model2.predict(interactions, features, top_k=3)

    for (g1, s1), (g2, s2) in zip(recs1, recs2):
        assert g1 == g2
        assert abs(s1 - s2) < 1e-10


# --- FeaseRegistry Tests ---

@pytest.fixture(scope="session")
def two_territory_models(weighting_data):
    """Trains two separate models with different params to simulate different territories."""
    i_path, u_path, t_path = weighting_data

    model_us = fease.build_and_train(
        i_path, u_path, t_path,
        alpha=1.0, beta=1.0, lambda_=10.0,
    )
    model_br = fease.build_and_train(
        i_path, u_path, t_path,
        alpha=1.0, beta=1.0, lambda_=50.0,  # different lambda
    )
    return model_us, model_br


def test_registry_basic(two_territory_models):
    """Create registry, register 2 models for different territories, verify predict works."""
    model_us, model_br = two_territory_models

    registry = fease.FeaseRegistry()
    assert len(registry) == 0

    registry.register("US", model_us)
    registry.register("BR", model_br)

    assert len(registry) == 2
    territories = registry.territories()
    assert "US" in territories
    assert "BR" in territories

    # Predict in each territory using index-based API
    scores_us = registry.predict("US", [(0, 1.0)])
    scores_br = registry.predict("BR", [(0, 1.0)])

    assert len(scores_us) == model_us.num_items
    assert len(scores_br) == model_br.num_items

    # Different lambda should produce different scores
    assert scores_us != scores_br


def test_registry_fallback(two_territory_models):
    """Create with fallback, verify unknown territory falls back."""
    model_us, model_br = two_territory_models

    registry = fease.FeaseRegistry(fallback_territory="US")
    registry.register("US", model_us)
    registry.register("BR", model_br)

    # Known territory works
    scores_us = registry.predict("US", [(0, 1.0)])
    assert len(scores_us) == model_us.num_items

    # Unknown territory "JP" falls back to "US"
    scores_jp = registry.predict("JP", [(0, 1.0)])
    assert len(scores_jp) == model_us.num_items

    # Fallback scores should exactly match US scores
    for a, b in zip(scores_us, scores_jp):
        assert abs(a - b) < 1e-12


def test_registry_predict_unknown_territory_error(two_territory_models):
    """No fallback, unknown territory raises error."""
    model_us, _ = two_territory_models

    registry = fease.FeaseRegistry()  # No fallback
    registry.register("US", model_us)

    with pytest.raises(ValueError, match="No model registered for territory 'JP'"):
        registry.predict("JP", [(0, 1.0)])


def test_registry_predict_top_k(two_territory_models):
    """Tests predict_top_k on registry, verifying exclusion and ordering."""
    model_us, _ = two_territory_models

    registry = fease.FeaseRegistry()
    registry.register("US", model_us)

    # User interacted with item 0, ask for top 2
    top_recs = registry.predict_top_k("US", [(0, 1.0)], top_k=2)

    assert len(top_recs) <= 2
    # Item 0 should be excluded (user already interacted)
    for idx, _ in top_recs:
        assert idx != 0
    # Results should be sorted descending by score
    if len(top_recs) >= 2:
        assert top_recs[0][1] >= top_recs[1][1]


def test_registry_predict_similar_items(two_territory_models):
    """Tests predict_similar_items on registry."""
    model_us, _ = two_territory_models

    registry = fease.FeaseRegistry()
    registry.register("US", model_us)

    similar = registry.predict_similar_items("US", 0, top_k=2)
    assert len(similar) <= 2
    # Should not contain item 0 itself
    for idx, _ in similar:
        assert idx != 0


def test_registry_bool():
    """Tests __bool__ — empty registry is falsy, non-empty is truthy."""
    registry = fease.FeaseRegistry()
    assert not registry  # empty -> falsy


# --- Ranking Evaluation Metrics Tests ---

import math


def test_precision_at_k():
    """Tests precision@K with known recommendations vs known relevant set."""
    # recommended = [1, 2, 3, 4, 5], relevant = {1, 3, 5}
    # top-3: hits = {1, 3} → 2/3
    assert abs(fease.precision_at_k([1, 2, 3, 4, 5], {1, 3, 5}, 3) - 2.0 / 3.0) < 1e-10

    # All relevant at top → 1.0
    assert abs(fease.precision_at_k([1, 2, 3], {1, 2, 3}, 3) - 1.0) < 1e-10

    # None relevant → 0.0
    assert abs(fease.precision_at_k([1, 2, 3], {4, 5}, 3) - 0.0) < 1e-10

    # k=0 → 0.0
    assert abs(fease.precision_at_k([1, 2], {1}, 0) - 0.0) < 1e-10


def test_recall_at_k():
    """Tests recall@K computation."""
    # top-3 of [1, 2, 3, 4, 5] with relevant {1, 3, 5, 7}: hits = {1, 3} → 2/4
    assert abs(fease.recall_at_k([1, 2, 3, 4, 5], {1, 3, 5, 7}, 3) - 0.5) < 1e-10

    # Full recall
    assert abs(fease.recall_at_k([1, 3, 5, 7], {1, 3, 5, 7}, 4) - 1.0) < 1e-10

    # Empty relevant → 0.0
    assert abs(fease.recall_at_k([1, 2, 3], set(), 3) - 0.0) < 1e-10


def test_ndcg_at_k():
    """Tests NDCG@K with DCG normalization verification."""
    # Perfect ranking: all relevant at top → NDCG = 1.0
    assert abs(fease.ndcg_at_k([1, 2, 3, 4, 5], {1, 2, 3}, 3) - 1.0) < 1e-10

    # Imperfect ranking: relevant = {3, 5}, recommended = [1, 2, 3, 4, 5]
    # Hits at 0-based positions 2 and 4
    # DCG = 1/log2(3+1) + 1/log2(5+1)
    # IDCG = 1/log2(1+1) + 1/log2(2+1)
    dcg = 1.0 / math.log2(4.0) + 1.0 / math.log2(6.0)
    idcg = 1.0 / math.log2(2.0) + 1.0 / math.log2(3.0)
    expected = dcg / idcg
    assert abs(fease.ndcg_at_k([1, 2, 3, 4, 5], {3, 5}, 5) - expected) < 1e-10

    # Single hit at top → 1.0
    assert abs(fease.ndcg_at_k([1, 2, 3], {1}, 3) - 1.0) < 1e-10

    # Empty relevant → 0.0
    assert abs(fease.ndcg_at_k([1, 2, 3], set(), 3) - 0.0) < 1e-10


def test_mean_average_precision():
    """Tests MAP computation."""
    # recommended = [1, 2, 3, 4, 5], relevant = {1, 3, 5}
    # Hit at pos 0: prec = 1/1, pos 2: prec = 2/3, pos 4: prec = 3/5
    # MAP = (1 + 2/3 + 3/5) / 3
    expected = (1.0 + 2.0 / 3.0 + 3.0 / 5.0) / 3.0
    assert abs(fease.mean_average_precision([1, 2, 3, 4, 5], {1, 3, 5}) - expected) < 1e-10

    # Perfect ranking → 1.0
    assert abs(fease.mean_average_precision([1, 2, 3], {1, 2, 3}) - 1.0) < 1e-10

    # No hits → 0.0
    assert abs(fease.mean_average_precision([1, 2, 3], {4, 5, 6}) - 0.0) < 1e-10


def test_coverage():
    """Tests coverage across multiple user recommendation lists."""
    # 3 users, 10 total items, 6 unique recommended
    all_recs = [[0, 1, 2], [2, 3, 4], [4, 5]]
    assert abs(fease.coverage(all_recs, 10) - 0.6) < 1e-10

    # Full coverage
    assert abs(fease.coverage([[0, 1], [2, 3], [4]], 5) - 1.0) < 1e-10

    # Duplicates across users don't inflate coverage
    assert abs(fease.coverage([[0, 1], [0, 1], [0, 1]], 10) - 0.2) < 1e-10

    # Empty recommendations → 0.0
    assert abs(fease.coverage([[], []], 10) - 0.0) < 1e-10


def test_hit_rate_at_k():
    """Tests hit rate@K."""
    # Hit within k
    assert abs(fease.hit_rate_at_k([10, 20, 30], {20, 40}, 3) - 1.0) < 1e-10

    # No hit
    assert abs(fease.hit_rate_at_k([10, 20, 30], {40, 50}, 3) - 0.0) < 1e-10

    # Hit outside k
    assert abs(fease.hit_rate_at_k([10, 20, 30, 40], {40}, 2) - 0.0) < 1e-10
    assert abs(fease.hit_rate_at_k([10, 20, 30, 40], {40}, 4) - 1.0) < 1e-10
