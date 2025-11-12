import os
import tempfile
from pathlib import Path

import polars as pl
import pytest
import rust_fease_recommender as fease


# This fixture creates temporary data, trains a model, and cleans up.
# It runs once for the entire test session.
@pytest.fixture(scope="session")
def trained_model():
    """
    Creates dummy data, saves it to temp parquet files,
    trains a model, and yields the model.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        eng_path = Path(tmpdir) / "eng.parquet"
        meta_path = Path(tmpdir) / "meta.parquet"

        # 1. Create Engagement Data
        # User 0: Warm user, likes item G0
        # User 1: Warm user, likes item G1
        # User 2: Cold user (features only)
        eng_df = pl.DataFrame(
            {
                "anonymous_id": ["u0", "u0", "u1"],
                "view_media_id": ["G0", "G2", "G1"],
                "view_seconds_watched": [300.0, 150.0, 600.0],
                "view_context_device_type": ["Mobile", "Web", "Mobile"],
                "view_subscription_plan": ["Premium", "Premium", "Free"],
                "view_country_code_view": ["US", "US", "DE"],
                "account_country_code_account": ["US", "US", "DE"],
                "account_tenure_days": [400, 400, 10],
                "region_major_account": ["US/CA", "US/CA", "EMEA"],
                "subscription_status": ["Paying", "Paying", "Inactive"],
            },
        )

        # Add cold user features (no interactions)
        # We add a row with no media_id to ensure the user is in the mapping
        cold_user_df = pl.DataFrame(
            {
                "anonymous_id": ["u2"],
                "view_media_id": [None],
                "view_seconds_watched": [None],
                "view_context_device_type": ["Console"],
                "view_subscription_plan": ["Premium"],
                "view_country_code_view": ["JP"],
                "account_country_code_account": ["JP"],
                "account_tenure_days": [1],
                "region_major_account": ["APAC"],
                "subscription_status": ["Free Trial"],
            },
        )

        eng_df = pl.concat([eng_df, cold_user_df])
        eng_df.write_parquet(eng_path)

        # 2. Create Metadata Data
        # G0, G2: Anime
        # G1, G3: Movie
        meta_df = pl.DataFrame(
            {
                "media_guid": ["G0", "G1", "G2", "G3"],
                "media_type": ["episode", "movie", "episode", "movie"],
                "media_audio_language": ["Japanese", "English", "Japanese", "English"],
                "media_hardsub_language": ["English", None, "English", "English"],
                "media_genres": [
                    "Action,Fantasy",
                    "Comedy",
                    "Action,Sci-Fi",
                    "Comedy,Drama",
                ],
                "media_tags": ["tagA,tagB", "tagC", "tagA,tagD", "tagC,tagE"],
                "media_series_title": ["Show A", "Movie B", "Show A", "Movie D"],
                "media_publisher_name": ["Pub 1", "Pub 2", "Pub 1", "Pub 2"],
                "airtable_primary_genre": ["Action", "Comedy", "Action", "Comedy"],
                "airtable_secondary_genres": ["Fantasy", None, "Sci-Fi", "Drama"],
                "airtable_japanese_audience": ["Shounen", None, "Shounen", None],
                "airtable_brand_grade_from_ca_data": ["A", "B", "A", "B"],
                "airtable_original_release_year": ["2020", "2021", "2020", "2022"],
            }
        )
        meta_df.write_parquet(meta_path)

        # 3. Train Model
        model = fease.build_and_train(
            engagement_path=str(eng_path),
            metadata_path=str(meta_path),
            alpha=1.0,
            beta=1.0,
            lambda_=10.0,  # Use a small lambda for distinct test scores
        )

        yield model


def test_model_training(trained_model):
    """Tests that the model was trained with the correct dimensions."""
    assert trained_model.num_items == 4  # G0, G1, G2, G3
    assert trained_model.num_user_features > 5  # Check that features were built
    assert trained_model.num_items > 0
    assert trained_model.num_user_features > 0


def test_warm_user_prediction(trained_model):
    """Tests prediction for a user with known interactions."""
    # User u0 liked G0 and G2 (Action/Fantasy/Sci-Fi)
    # They should prefer G0/G2 over G1/G3 (Comedy)
    interactions = {
        "G0": 5.0,  # (1 + 300).log10()
        "G2": 4.0,  # (1 + 150).log10()
    }
    features = {
        "device_Mobile": 1.0,
        "plan_Premium": 1.0,
        "country_view_US": 1.0,
        "country_acct_US": 1.0,
        "tenure_365d+": 1.0,
        "region_US/CA": 1.0,
        "sub_status_Paying": 1.0,
    }

    recs = trained_model.predict(interactions, features, top_k=4)

    assert len(recs) == 4
    assert recs[0][0] in ["G0", "G2"]  # Top recs should be what they liked
    assert recs[1][0] in ["G0", "G2"]
    assert recs[2][0] in ["G1", "G3"]  # Bottom recs should be other items
    assert recs[3][0] in ["G1", "G3"]
    print("\nWarm User (u0) Recs:", recs)


def test_cold_user_prediction(trained_model):
    """Tests prediction for a cold-start user (features only)."""
    # User u2 has no interactions.
    # Features: Console, Premium, JP, 1 day tenure
    interactions = {}  # Empty
    features = {
        "device_Console": 1.0,
        "plan_Premium": 1.0,
        "country_view_JP": 1.0,
        "country_acct_JP": 1.0,
        "tenure_1-7d": 1.0,
        "region_APAC": 1.0,
        "sub_status_Free Trial": 1.0,
    }

    recs = trained_model.predict(interactions, features, top_k=4)

    assert len(recs) == 4
    # We can't know the *exact* order, but all scores should be non-zero
    assert recs[0][1] != 0.0
    assert recs[1][1] != 0.0

    # Check that scores for similar items are close
    # G0/G2 are "Action", G1/G3 are "Comedy"
    # We expect G0/G2 to have different scores from G1/G3
    scores = dict(recs)
    assert scores["G0"] != scores["G1"]
    assert scores["G2"] != scores["G3"]
    print("\nCold User (u2) Recs:", recs)


def test_unknown_user_prediction(trained_model):
    """Tests prediction for a user with no features and no interactions."""
    interactions = {}
    features = {}

    recs = trained_model.predict(interactions, features, top_k=4)

    assert len(recs) == 4
    # For a *truly* unknown user, all scores should be 0
    # because the input vector `z` is all zeros.
    assert recs[0][1] == 0.0
    assert recs[1][1] == 0.0
    assert recs[2][1] == 0.0
    assert recs[3][1] == 0.0
    print("\nUnknown User Recs:", recs)


def test_top_k(trained_model):
    """Tests that top_k is respected."""
    interactions = {}
    features = {"device_Console": 1.0, "plan_Premium": 1.0}

    recs = trained_model.predict(interactions, features, top_k=2)
    assert len(recs) == 2

    recs_all = trained_model.predict(interactions, features, top_k=999)
    assert len(recs_all) == 4  # Total number of items
