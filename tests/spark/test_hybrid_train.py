"""Unit tests for the Spark helpers in `kzn_recsys.hybrid_train` (#96).

These exercise the embedding -> long-format plumbing with a stand-in
BERT4Rec object, so they run on the pure-Python install too (no native
extension, no burn). The end-to-end `run_hybrid_pipeline` needs the
`ml-models` native build and is covered by `tests/test_bert4rec.py` +
a manual Databricks run.
"""
import os

import pytest

pytest.importorskip("numpy")
from pyspark.sql import functions as F

from kzn_recsys.hybrid_train import (
    embeddings_to_long_format,
    extract_user_embeddings,
    pca_reduce_embeddings,
    union_long_features,
    write_single_parquet,
)


class _FakeBert:
    embedding_dim = 3

    def __init__(self, table):
        self._table = table

    def embed_users(self, interactions_path):
        assert interactions_path == "ignored.parquet"
        return dict(self._table)


def test_extract_user_embeddings_is_wide_and_sorted(spark):
    fake = _FakeBert({"zed": [0.5, -1.0, 0.0], "amy": [1.0, 2.0, 3.0]})
    df = extract_user_embeddings(fake, "ignored.parquet", spark)
    assert df.columns == ["user_id", "emb_0", "emb_1", "emb_2"]
    rows = df.collect()
    assert [r.user_id for r in rows] == ["amy", "zed"]
    assert [r.emb_1 for r in rows] == [2.0, -1.0]


def test_embeddings_to_long_format_melts_and_drops_zeros(spark):
    wide = spark.createDataFrame(
        [("u1", 0.5, 0.0), ("u2", 0.0, 0.0)], ["user_id", "emb_0", "emb_1"]
    )
    long = embeddings_to_long_format(wide, "user_id", "bert_emb")
    assert long.columns == ["user_id", "feature_name", "value"]
    rows = {(r.user_id, r.feature_name): r.value for r in long.collect()}
    # Exact zeros dropped: u2 (all-zero = cold-start embedding) has no rows.
    assert rows == {("u1", "bert_emb_0"): 0.5}

    kept = embeddings_to_long_format(wide, "user_id", "bert_emb", drop_zeros=False)
    assert kept.count() == 4
    names = {r.feature_name for r in kept.collect()}
    assert names == {"bert_emb_0", "bert_emb_1"}


def test_embeddings_to_long_format_explicit_columns(spark):
    wide = spark.createDataFrame([("i1", "ignored", 2.0)], ["item_id", "title", "d0"])
    long = embeddings_to_long_format(wide, "item_id", "kg_emb", embedding_cols=["d0"])
    assert [tuple(r) for r in long.collect()] == [("i1", "kg_emb_0", 2.0)]


def test_union_long_features_skips_none(spark):
    a = spark.createDataFrame([("u1", "region=US", 1.0)], ["user_id", "feature_name", "value"])
    b = spark.createDataFrame([("u1", "bert_emb_0", 0.3)], ["user_id", "feature_name", "value"])
    assert union_long_features(None, None) is None
    assert union_long_features(a, None).count() == 1
    assert union_long_features(a, b).count() == 2


def test_pca_reduce_embeddings_shrinks_width(spark):
    rows = [(f"i{k}", float(k), float(2 * k), float(k % 3), 1.0) for k in range(12)]
    wide = spark.createDataFrame(rows, ["item_id", "d0", "d1", "d2", "d3"])
    reduced = pca_reduce_embeddings(wide, "item_id", 2)
    assert reduced.columns == ["item_id", "emb_0", "emb_1"]
    assert reduced.count() == 12
    # Asking for more components than inputs clamps to the input width.
    assert len(pca_reduce_embeddings(wide, "item_id", 10).columns) == 5


def test_write_single_parquet_produces_one_file(spark, tmp_path):
    df = spark.createDataFrame([("u1", "i1", 1.0)], ["user_id", "item_id", "value"])
    out = write_single_parquet(df, str(tmp_path / "interactions.parquet"))
    assert os.path.isfile(out)
    assert not os.path.exists(out + ".spark_tmp")
    back = spark.read.parquet("file:" + out)
    assert back.count() == 1
    assert back.filter(F.col("user_id") == "u1").count() == 1
