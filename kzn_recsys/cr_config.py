"""Shared configuration constants for the content-recommendation (CR)
training pipelines.

Importable without the native extension, pydantic, polars, or Spark so
Databricks notebooks can read it before any heavy import.
"""

# --- BERT4Rec (issue #96) --------------------------------------------------
# Architecture. `BERT4REC_EMBEDDING_DIM` is also the width of the user
# embedding the hybrid pipeline feeds into FEASE, so it trades BERT4Rec
# expressiveness against FEASE Gram-matrix growth; HPO range [32, 64, 128,
# 256]. Must be divisible by `BERT4REC_NUM_HEADS`.
BERT4REC_EMBEDDING_DIM = 64
BERT4REC_MAX_SEQ_LEN = 200
BERT4REC_NUM_POSITION_BUCKETS = 32  # log2(days_ago) buckets
BERT4REC_NUM_HEADS = 4
BERT4REC_NUM_LAYERS = 2
BERT4REC_DROPOUT = 0.1
BERT4REC_MASK_RATIO = 0.2
# Training.
BERT4REC_EPOCHS = 50
BERT4REC_BATCH_SIZE = 64
BERT4REC_LEARNING_RATE = 1e-3
BERT4REC_PATIENCE = 5
BERT4REC_SEED = 42

# --- Hybrid FEASE (BERT4Rec user embeddings + MAL KG item embeddings) ----
# `beta` weights the user-feature block (where the BERT4Rec embedding
# lives); `alpha` the item-feature block (catalog metadata + KG embedding).
HYBRID_BETA = 0.5
HYBRID_ALPHA = 1.0
HYBRID_LAMBDA = 100.0
# Prefix for the long-format feature names holding the dense embeddings:
# `bert_emb_0 .. bert_emb_{dim-1}` on users, `kg_emb_*` on items.
HYBRID_USER_EMBEDDING_PREFIX = "bert_emb"
HYBRID_ITEM_EMBEDDING_PREFIX = "kg_emb"
# Temporal hold-out used by the hybrid evaluation step.
HYBRID_EVAL_DAYS_AGO_CUTOFF = 30.0

# --- Series-grain interactions table (issue #102) --------------------------
# Daily grain: one row per (profile, series, view_date). Items are series
# (catalog_show_id), not episodes. `days_ago` is derived at read time
# (`kzn_recsys.spark.interactions_agg.with_days_ago`) so decay_rate stays
# tunable. Built / refreshed by databricks/sql/*.sql via
# databricks/src/00_refresh_interactions_agg.py.
INTERACTIONS_AGG_TABLE = "dsml_recs.dev.fease_interactions_agg"
# Gold viewership source and the columns the aggregation reads from it.
VIEWERSHIP_TABLE = "cr_prod.gold_db.ds_viewership"
VIEWERSHIP_USER_COL = "view_profile_id"      # profile grain
VIEWERSHIP_ITEM_COL = "catalog_show_id"      # series grain
VIEWERSHIP_DATE_COL = "view_date"
VIEWERSHIP_TS_COL = "view_ts"                # event timestamp; orders user-feature rows
# Content metadata is media-grain; this column maps each media row to its
# series so item features land on the same ids as the interactions.
METADATA_SERIES_COL = "catalog_show_id"
VIEWERSHIP_SECONDS_COL = "view_seconds_watched"
VIEWERSHIP_SUBSIDIARY_COL = "view_subsidiary"
VIEWERSHIP_SUBSIDIARY = "crunchyroll"
# A view shorter than this does not count as an interaction.
MIN_WATCH_SECONDS = 30.0
# Days the daily MERGE recomputes; must cover the gold table's late-arrival window.
INTERACTIONS_AGG_LOOKBACK_DAYS = 3
