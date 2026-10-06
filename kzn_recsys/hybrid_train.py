# Databricks notebook-style training pipeline: BERT4Rec + FEASE hybrid
# (issue #96).
#
#     Full watch history (user x item x days_ago)
#       |
#       +--> BERT4Rec (masked prediction, bidirectional attention)
#       |      '--> user embedding (BERT4REC_EMBEDDING_DIM) --> FEASE user features
#       |
#       +--> FEASE interactions   (user x item x value)
#       +--> FEASE user features  (categorical + bert_emb_* embedding)
#       '--> FEASE item features  (catalog metadata + kg_emb_* MAL KG embedding)
#              '--> FEASE S-matrix --> recommendations
#
# BERT4Rec supplies contextual co-occurrence ("what belongs in this
# collection") as a dense per-user taste vector; pre-computed MyAnimeList
# knowledge-graph embeddings (anime<->user / genre / studio relations,
# PCA-reduced) enrich the item side; FEASE's closed-form solve, side-feature
# cold-start, and existing serving infrastructure produce the final model.
#
# The module is importable (the helpers below are unit-tested under
# `tests/spark/`) and runnable as a Databricks notebook / job: every
# `# COMMAND ----------` block is a cell, and `main()` runs the full flow.
#
# Requirements: the native extension built with `--features ml-models`
# (`fease._HAS_ML_MODELS`), pyspark, and an interactions table that carries
# a numeric `days_ago` column (BERT4Rec fails loudly without it).

from __future__ import annotations

import glob
import os
import shutil
import time
from typing import Iterable, Optional, Sequence

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

import kzn_recsys as fease
from kzn_recsys import cr_config as cfg

# COMMAND ----------

# --
# Step 0: Helpers
# --


def _emb_col(i: int) -> str:
    return f"emb_{i}"


def extract_user_embeddings(
    bert_model,
    interactions_path: str,
    spark: Optional[SparkSession] = None,
) -> DataFrame:
    """Run `bert_model.embed_users(interactions_path)` and return a wide
    Spark DataFrame `[user_id, emb_0, ..., emb_{dim-1}]`.

    Every user with at least one in-catalog interaction in the file gets a
    row. `bert_model` is a `kzn_recsys.Bert4RecModel` (or anything with the
    same `embed_users` / `embedding_dim` surface). Users absent from the
    file (brand-new users) simply have no row, and therefore no `bert_emb_*`
    features in FEASE — their categorical features carry them (#96 §Key
    Design Decisions #5).
    """
    spark = spark or SparkSession.builder.getOrCreate()
    embeddings = bert_model.embed_users(interactions_path)
    dim = int(getattr(bert_model, "embedding_dim", 0)) or (
        len(next(iter(embeddings.values()))) if embeddings else 0
    )
    columns = ["user_id"] + [_emb_col(i) for i in range(dim)]
    rows = [
        (user_id, *[float(v) for v in vec])
        for user_id, vec in sorted(embeddings.items())
    ]
    schema = "user_id string" + "".join(f", {c} double" for c in columns[1:])
    return spark.createDataFrame(rows, schema=schema)


def embeddings_to_long_format(
    df: DataFrame,
    id_col: str,
    prefix: str,
    embedding_cols: Optional[Sequence[str]] = None,
    drop_zeros: bool = True,
) -> DataFrame:
    """Melt a wide embedding table into FEASE's long feature format
    `[id_col, feature_name, value]`.

    `embedding_cols` defaults to every column except `id_col`, in order;
    column `k` becomes feature `f"{prefix}_{k}"`. Exact zeros are dropped by
    default — they add nothing to a sparse CSR feature matrix, and an
    all-zero row (BERT4Rec's "no history" embedding) then yields no
    features at all, which is the intended cold-start behaviour.
    """
    cols = list(embedding_cols) if embedding_cols is not None else [
        c for c in df.columns if c != id_col
    ]
    if not cols:
        return df.sparkSession.createDataFrame(
            [], schema=f"{id_col} string, feature_name string, value double"
        )
    pairs = F.array(
        *[
            F.struct(
                F.lit(f"{prefix}_{k}").alias("feature_name"),
                F.col(c).cast("double").alias("value"),
            )
            for k, c in enumerate(cols)
        ]
    )
    out = (
        df.select(F.col(id_col).cast("string").alias(id_col), F.explode(pairs).alias("p"))
        .select(id_col, F.col("p.feature_name").alias("feature_name"), F.col("p.value").alias("value"))
        .filter(F.col("value").isNotNull())
    )
    if drop_zeros:
        out = out.filter(F.col("value") != 0.0)
    return out


def pca_reduce_embeddings(
    df: DataFrame,
    id_col: str,
    n_components: int,
    embedding_cols: Optional[Sequence[str]] = None,
) -> DataFrame:
    """Project a wide embedding table onto its top `n_components` principal
    components, returning `[id_col, emb_0, ..., emb_{n-1}]`.

    Used to shrink the (typically 128- to 512-dim) MAL knowledge-graph item
    embeddings before they enter FEASE, where every feature column grows
    the Gram matrix quadratically.
    """
    from pyspark.ml.feature import PCA, VectorAssembler
    from pyspark.ml.functions import vector_to_array

    cols = list(embedding_cols) if embedding_cols is not None else [
        c for c in df.columns if c != id_col
    ]
    n_components = min(n_components, len(cols))
    assembled = VectorAssembler(inputCols=cols, outputCol="_features").transform(
        df.select(id_col, *[F.col(c).cast("double").alias(c) for c in cols])
    )
    pca = PCA(k=n_components, inputCol="_features", outputCol="_pca").fit(assembled)
    reduced = pca.transform(assembled).withColumn("_arr", vector_to_array("_pca"))
    return reduced.select(
        id_col, *[F.col("_arr")[i].alias(_emb_col(i)) for i in range(n_components)]
    )


def union_long_features(*frames: Optional[DataFrame]) -> Optional[DataFrame]:
    """Union long-format feature tables (same 3-column shape), skipping
    `None`s. Returns `None` when nothing is given."""
    frames = [f for f in frames if f is not None]
    if not frames:
        return None
    out = frames[0]
    for f in frames[1:]:
        out = out.unionByName(f)
    return out


def write_single_parquet(df: DataFrame, local_path: str) -> str:
    """Write `df` as ONE Parquet *file* at `local_path` (a path the Rust
    reader can open directly, e.g. `/dbfs/tmp/x.parquet`).

    Spark writes directories of part files; the single-threaded Polars
    reader in Rust wants one file, so we coalesce to one partition, write
    to a scratch directory, and move the lone part file into place. The
    scratch dir is a sibling of `local_path`; a `/dbfs/...` local path is
    translated to its `dbfs:/...` Spark URI, any other local path to
    `file:...`.
    """
    scratch = local_path + ".spark_tmp"
    if local_path.startswith("/dbfs/"):
        spark_uri = "dbfs:/" + local_path[len("/dbfs/"):] + ".spark_tmp"
    else:
        spark_uri = "file:" + os.path.abspath(scratch)
    shutil.rmtree(scratch, ignore_errors=True)
    df.coalesce(1).write.mode("overwrite").parquet(spark_uri)
    parts = sorted(glob.glob(os.path.join(scratch, "part-*.parquet")))
    if len(parts) != 1:
        raise RuntimeError(f"expected exactly one part file in {scratch}, found {parts}")
    if os.path.exists(local_path):
        os.remove(local_path)
    shutil.move(parts[0], local_path)
    shutil.rmtree(scratch, ignore_errors=True)
    return local_path


# COMMAND ----------

# --
# Step 1: Train BERT4Rec on the full watch history
# --


def train_bert4rec(interactions_path: str, **overrides):
    """`fease.build_and_train_bert4rec` with `cr_config` defaults.
    `interactions_path` must carry `user_id`, `item_id`, `value`,
    `days_ago`."""
    if not getattr(fease, "_HAS_ML_MODELS", False):
        raise RuntimeError(
            "BERT4Rec requires the native extension built with "
            "`maturin develop --features ml-models`"
        )
    params = dict(
        embedding_dim=cfg.BERT4REC_EMBEDDING_DIM,
        max_seq_len=cfg.BERT4REC_MAX_SEQ_LEN,
        num_position_buckets=cfg.BERT4REC_NUM_POSITION_BUCKETS,
        num_heads=cfg.BERT4REC_NUM_HEADS,
        num_layers=cfg.BERT4REC_NUM_LAYERS,
        dropout=cfg.BERT4REC_DROPOUT,
        mask_ratio=cfg.BERT4REC_MASK_RATIO,
        num_epochs=cfg.BERT4REC_EPOCHS,
        batch_size=cfg.BERT4REC_BATCH_SIZE,
        learning_rate=cfg.BERT4REC_LEARNING_RATE,
        patience=cfg.BERT4REC_PATIENCE,
        seed=cfg.BERT4REC_SEED,
    )
    params.update(overrides)
    t0 = time.time()
    model = fease.build_and_train_bert4rec(interactions_path=interactions_path, **params)
    print(
        f"BERT4Rec trained in {time.time() - t0:.1f}s: "
        f"{model.num_items} items, dim={model.embedding_dim}"
    )
    return model


# COMMAND ----------

# --
# Steps 2-6: the orchestrated flow
# --


def run_hybrid_pipeline(
    spark: SparkSession,
    interactions_path: str,
    work_dir: str,
    model_dir: str,
    user_categorical_features: Optional[DataFrame] = None,
    item_catalog_features: Optional[DataFrame] = None,
    item_kg_embeddings: Optional[DataFrame] = None,
    kg_pca_components: Optional[int] = 32,
    days_ago_cutoff: float = cfg.HYBRID_EVAL_DAYS_AGO_CUTOFF,
    k_values: Iterable[int] = (5, 10, 20),
    bert4rec_overrides: Optional[dict] = None,
    alpha: float = cfg.HYBRID_ALPHA,
    beta: float = cfg.HYBRID_BETA,
    lambda_: float = cfg.HYBRID_LAMBDA,
) -> dict:
    """End-to-end hybrid training.

    1. Train BERT4Rec on `interactions_path` (must carry `days_ago`).
    2. Extract user embeddings -> long format (`bert_emb_0`, ...).
    3. Union with `user_categorical_features` (long format: subscription,
       region, platform, tenure buckets, ...).
    4. Item features = `item_catalog_features` (long) + MAL KG embeddings
       (wide `[item_id, <dims...>]`, PCA-reduced to `kg_pca_components`
       when set) as `kg_emb_*`.
    5. Temporal split at `days_ago_cutoff`; train FEASE on the train split
       with the enriched features and evaluate on the hold-out.
    6. Train the final FEASE on all interactions; save both models under
       `model_dir` (`bert4rec.fb4r`, `fease_hybrid.fease`).

    Returns a dict with the evaluation report, split sizes and model paths.
    `work_dir` holds the intermediate single-file Parquets the Rust side
    reads (use a `/dbfs/...` path on Databricks).
    """
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 1. BERT4Rec
    bert_model = train_bert4rec(interactions_path, **(bert4rec_overrides or {}))

    # 2. + 3. User features
    user_emb_wide = extract_user_embeddings(bert_model, interactions_path, spark)
    user_emb_long = embeddings_to_long_format(
        user_emb_wide, "user_id", cfg.HYBRID_USER_EMBEDDING_PREFIX
    )
    user_features = union_long_features(user_categorical_features, user_emb_long)
    user_features_path = write_single_parquet(
        user_features, os.path.join(work_dir, "user_features.parquet")
    )

    # 4. Item features
    kg_long = None
    if item_kg_embeddings is not None:
        kg_wide = (
            pca_reduce_embeddings(item_kg_embeddings, "item_id", kg_pca_components)
            if kg_pca_components
            else item_kg_embeddings
        )
        kg_long = embeddings_to_long_format(kg_wide, "item_id", cfg.HYBRID_ITEM_EMBEDDING_PREFIX)
    item_features = union_long_features(item_catalog_features, kg_long)
    if item_features is None:
        item_features = spark.createDataFrame(
            [], schema="item_id string, feature_name string, value double"
        )
    item_features_path = write_single_parquet(
        item_features, os.path.join(work_dir, "item_features.parquet")
    )

    # 5. Temporal hold-out evaluation
    split = fease.temporal_split_safe(
        interactions_path, days_ago_cutoff=days_ago_cutoff, output_dir=work_dir
    )
    eval_model = fease.build_and_train(
        interactions_path=split.train_path,
        user_features_path=user_features_path,
        item_features_path=item_features_path,
        alpha=alpha,
        beta=beta,
        lambda_=lambda_,
    )
    report = eval_model.evaluate(
        test_interactions_path=split.test_path,
        train_interactions_path=split.train_path,
        user_features_path=user_features_path,
        k_values=list(k_values),
    )
    for m in report["metrics"]:
        print(
            f"  k={m['k']:>3}  precision={m['precision']:.4f}  recall={m['recall']:.4f}  "
            f"ndcg={m['ndcg']:.4f}  hit_rate={m['hit_rate']:.4f}"
        )

    # 6. Final model on everything + save both
    final_model = fease.build_and_train(
        interactions_path=interactions_path,
        user_features_path=user_features_path,
        item_features_path=item_features_path,
        alpha=alpha,
        beta=beta,
        lambda_=lambda_,
    )
    bert_path = os.path.join(model_dir, "bert4rec.fb4r")
    fease_path = os.path.join(model_dir, "fease_hybrid.fease")
    bert_model.save(bert_path)
    final_model.save(fease_path)
    print(f"Saved BERT4Rec -> {bert_path}\nSaved hybrid FEASE -> {fease_path}")

    return {
        "report": report,
        "split": {
            "train_interactions": split.train_interactions,
            "test_interactions": split.test_interactions,
            "train_users": split.train_users,
            "test_users": split.test_users,
        },
        "bert4rec_path": bert_path,
        "fease_path": fease_path,
        "user_features_path": user_features_path,
        "item_features_path": item_features_path,
        "bert4rec_model": bert_model,
        "fease_model": final_model,
    }


# COMMAND ----------

# --
# Notebook entrypoint
# --
#
# Point these at your tables. The engagement table must expose a numeric
# `days_ago` (compute it from the event timestamp: `datediff(current_date(),
# view_ts)`), and `value` is the log-transformed watch time as in
# `fease_train.py`. User categorical features and item catalog metadata are
# long-format `[id, feature_name, value]` tables; KG embeddings are a wide
# `[item_id, dim_0, ..., dim_N]` table.

INTERACTIONS_TABLE = "your_db.hybrid_interactions"        # user_id, item_id, value, days_ago
USER_FEATURES_TABLE = "your_db.hybrid_user_features"      # user_id, feature_name, value (optional)
ITEM_FEATURES_TABLE = "your_db.hybrid_item_features"      # item_id, feature_name, value (optional)
KG_EMBEDDINGS_TABLE = "your_db.mal_kg_item_embeddings"    # item_id, dim_0..dim_N (optional)

WORK_DIR = "/dbfs/tmp/kzn_hybrid"      # scratch; single-file Parquets for Rust
MODEL_DIR = "/dbfs/models/kzn_hybrid"  # persistent; survives WORK_DIR cleanup


def _optional_table(spark: SparkSession, name: str) -> Optional[DataFrame]:
    try:
        return spark.table(name)
    except Exception as e:  # table absent / not permitted -> feature block skipped
        print(f"Skipping optional table {name}: {e}")
        return None


def main() -> dict:
    spark = SparkSession.builder.getOrCreate()
    os.makedirs(WORK_DIR, exist_ok=True)

    interactions_path = write_single_parquet(
        spark.table(INTERACTIONS_TABLE).select("user_id", "item_id", "value", "days_ago"),
        os.path.join(WORK_DIR, "interactions.parquet"),
    )
    result = run_hybrid_pipeline(
        spark,
        interactions_path=interactions_path,
        work_dir=WORK_DIR,
        model_dir=MODEL_DIR,
        user_categorical_features=_optional_table(spark, USER_FEATURES_TABLE),
        item_catalog_features=_optional_table(spark, ITEM_FEATURES_TABLE),
        item_kg_embeddings=_optional_table(spark, KG_EMBEDDINGS_TABLE),
    )
    print("Hybrid pipeline complete.")
    return result


if __name__ == "__main__":
    main()
