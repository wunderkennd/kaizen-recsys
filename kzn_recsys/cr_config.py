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
