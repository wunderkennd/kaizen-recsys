import tempfile
from pathlib import Path
import pytest
import polars as pl
import kzn_recsys as fease
from kzn_recsys import NumericalBucketConfig, FeatureTransformationSchema, build_and_train

def test_feature_transformation_schema_py():
    schema = FeatureTransformationSchema()
    schema.add_categorical("plan", "plan_")
    
    bucket = NumericalBucketConfig("tenure", [0.0, 7.0, 30.0, 90.0], ["0d", "7d", "30d", "90d", "90d+"])
    schema.add_numerical("tenure_days", bucket)

    cats = schema.categorical_features
    nums = schema.numerical_features
    assert cats["plan"] == "plan_"
    assert nums["tenure_days"].prefix == "tenure"
    assert nums["tenure_days"].boundaries == [0.0, 7.0, 30.0, 90.0]
    assert nums["tenure_days"].labels == ["0d", "7d", "30d", "90d", "90d+"]

def test_model_training_and_predict_raw():
    # Write tiny parquet files for training a dummy model
    with tempfile.TemporaryDirectory() as tmpdir:
        i_path = Path(tmpdir) / "interactions.parquet"
        u_path = Path(tmpdir) / "user_features.parquet"
        t_path = Path(tmpdir) / "item_features.parquet"

        interactions_df = pl.DataFrame({
            "user_id": ["u0", "u0", "u1"],
            "item_id": ["G0", "G2", "G1"],
            "value": [5.0, 4.0, 6.0],
        })
        user_features_df = pl.DataFrame({
            "user_id": ["u0", "u0", "u1"],
            "feature_name": ["plan_Premium", "tenure_30d", "plan_Free"],
            "value": [1.0, 1.0, 1.0],
        })
        item_features_df = pl.DataFrame({
            "item_id": ["G0", "G1", "G2"],
            "feature_name": ["genre_Action", "genre_Comedy", "genre_Action"],
            "value": [1.0, 1.0, 1.0],
        })

        interactions_df.write_parquet(i_path)
        user_features_df.write_parquet(u_path)
        item_features_df.write_parquet(t_path)

        # Set up transformation schema mapping plan and tenure_days
        schema = FeatureTransformationSchema()
        schema.add_categorical("plan", "plan_")
        bucket = NumericalBucketConfig("tenure", [0.0, 7.0, 30.0, 90.0], ["0d", "7d", "30d", "90d", "90d+"])
        schema.add_numerical("tenure_days", bucket)

        # Train model with schema embedded
        model = build_and_train(
            interactions_path=str(i_path),
            user_features_path=str(u_path),
            item_features_path=str(t_path),
            alpha=1.0,
            beta=1.0,
            lambda_=10.0,
            transformation_schema=schema
        )

        # Confirm schema is embedded in the model properties
        assert model.transformation_schema is not None
        assert model.transformation_schema.categorical_features["plan"] == "plan_"

        # 1. Predict using manual preprocessed features
        recs_manual = model.predict(
            interactions={"G0": 5.0},
            features={"plan_Premium": 1.0, "tenure_30d": 1.0},
            top_k=2
        )

        # 2. Predict using predict_raw with raw feature dictionary (strings and integers)
        recs_raw = model.predict_raw(
            interactions={"G0": 5.0},
            raw_features={"plan": "Premium", "tenure_days": 15},
            top_k=2
        )

        # Scores should be identical
        assert recs_manual == recs_raw

        # 3. Save and reload model, confirm schema persists and predicts exactly the same
        save_path = Path(tmpdir) / "model.fease"
        model.save(str(save_path))

        loaded = fease.load_model(str(save_path))
        assert loaded.transformation_schema is not None
        assert loaded.transformation_schema.categorical_features["plan"] == "plan_"

        recs_loaded = loaded.predict_raw(
            interactions={"G0": 5.0},
            raw_features={"plan": "Premium", "tenure_days": 15},
            top_k=2
        )
        assert recs_loaded == recs_raw
