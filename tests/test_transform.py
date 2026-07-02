"""End-to-end tests for the native feature-transformation layer (#71).

Ported from PR #62's branch with the fixed semantics: prefixes are
normalized (no trailing underscore stored), unknown keys use a single
separator, and bucket configs validate their shape at construction.
"""
import tempfile
from pathlib import Path

import polars as pl
import pytest

import kzn_recsys as fease
from kzn_recsys import FeatureTransformationSchema, NumericalBucketConfig, build_and_train


def _tenure_bucket():
    return NumericalBucketConfig(
        "tenure", [0.0, 7.0, 30.0, 90.0], ["0d", "7d", "30d", "90d", "90d+"]
    )


def test_feature_transformation_schema_py():
    schema = FeatureTransformationSchema()
    schema.add_categorical("plan", "plan_")
    schema.add_numerical("tenure_days", _tenure_bucket())

    # Prefixes are stored normalized: "plan_" and "plan" configure the same family.
    assert schema.categorical_features["plan"] == "plan"
    bucket = schema.numerical_features["tenure_days"]
    assert bucket.prefix == "tenure"
    assert bucket.boundaries == [0.0, 7.0, 30.0, 90.0]
    assert bucket.labels == ["0d", "7d", "30d", "90d", "90d+"]


def test_bucket_config_validates_shape():
    with pytest.raises(ValueError, match="boundaries \\+ 1"):
        NumericalBucketConfig("tenure", [0.0, 7.0], ["a", "b"])  # needs 3 labels
    with pytest.raises(ValueError, match="ascending"):
        NumericalBucketConfig("tenure", [7.0, 0.0], ["a", "b", "c"])


def test_model_training_and_predict_raw():
    with tempfile.TemporaryDirectory() as tmpdir:
        i_path = Path(tmpdir) / "interactions.parquet"
        u_path = Path(tmpdir) / "user_features.parquet"
        t_path = Path(tmpdir) / "item_features.parquet"

        pl.DataFrame(
            {
                "user_id": ["u0", "u0", "u1"],
                "item_id": ["G0", "G2", "G1"],
                "value": [5.0, 4.0, 6.0],
            }
        ).write_parquet(i_path)
        pl.DataFrame(
            {
                "user_id": ["u0", "u0", "u1"],
                "feature_name": ["plan_Premium", "tenure_30d", "plan_Free"],
                "value": [1.0, 1.0, 1.0],
            }
        ).write_parquet(u_path)
        pl.DataFrame(
            {
                "item_id": ["G0", "G1", "G2"],
                "feature_name": ["genre_Action", "genre_Comedy", "genre_Action"],
                "value": [1.0, 1.0, 1.0],
            }
        ).write_parquet(t_path)

        schema = FeatureTransformationSchema()
        schema.add_categorical("plan", "plan_")
        schema.add_numerical("tenure_days", _tenure_bucket())

        model = build_and_train(
            interactions_path=str(i_path),
            user_features_path=str(u_path),
            item_features_path=str(t_path),
            alpha=1.0,
            beta=1.0,
            lambda_=10.0,
        )
        assert model.transformation_schema is None
        model.set_transformation_schema(schema)
        assert model.transformation_schema is not None
        assert model.transformation_schema.categorical_features["plan"] == "plan"

        # predict_raw with raw values must equal predict with hand-engineered
        # keys: "Premium" -> plan_Premium, 15 days -> tenure_30d bucket.
        recs_manual = model.predict(
            interactions={"G0": 5.0},
            features={"plan_Premium": 1.0, "tenure_30d": 1.0},
            top_k=2,
        )
        recs_raw = model.predict_raw(
            interactions={"G0": 5.0},
            raw_features={"plan": "Premium", "tenure_days": 15},
            top_k=2,
        )
        assert recs_manual == recs_raw

        # Save (format v3) and reload: schema persists, predictions identical.
        save_path = Path(tmpdir) / "model.fease"
        model.save(str(save_path))
        loaded = fease.load_model(str(save_path))
        assert loaded.transformation_schema is not None
        assert loaded.transformation_schema.categorical_features["plan"] == "plan"
        recs_loaded = loaded.predict_raw(
            interactions={"G0": 5.0},
            raw_features={"plan": "Premium", "tenure_days": 15},
            top_k=2,
        )
        assert recs_loaded == recs_raw

        # Clearing the schema falls back to pass-through semantics.
        loaded.set_transformation_schema(None)
        assert loaded.transformation_schema is None
        recs_passthrough = loaded.predict_raw(
            interactions={"G0": 5.0},
            raw_features={"plan_Premium": 1.0, "tenure_30d": 1.0},
            top_k=2,
        )
        assert recs_passthrough == recs_raw
