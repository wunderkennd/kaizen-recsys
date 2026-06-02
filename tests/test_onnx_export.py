import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import kzn_recsys as fease


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
