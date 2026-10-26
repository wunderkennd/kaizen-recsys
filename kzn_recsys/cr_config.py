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

# --- FEASE side features (kzn_recsys.spark.cr_features) --------------------
# User features: each profile's most recent viewership row, one-hot encoded
# as `<column>_<value>`, plus a bucketed account tenure.
CR_USER_CATEGORICAL_FEATURES = (
    "view_subscription_plan",
    "account_country_code_account",
    "region_major_account",
    "subscription_status",
)
CR_USER_TENURE_COL = "account_tenure_days"
# Item features from the media-grain content metadata, keyed by
# METADATA_SERIES_COL. `media_series_title` is deliberately absent: at
# series grain it is one feature per item (an identity column).
CR_ITEM_CATEGORICAL_FEATURES = (
    "media_type",
    "media_audio_language",
    "airtable_primary_genre",
    "airtable_ca_brand_grade",
)
# Comma-separated multi-value columns -> (column, feature-name prefix).
CR_ITEM_MULTI_VALUE_FEATURES = (
    ("media_genres", "genre_"),
    ("media_tags", "tag_"),
)
# Days the daily MERGE recomputes; must cover the gold table's late-arrival window.
INTERACTIONS_AGG_LOOKBACK_DAYS = 3

# --- Per-interaction and per-series attribute tables (Phase A) -------------
# Sibling tables of INTERACTIONS_AGG_TABLE, never a widening of it:
#   fease_interaction_attrs: one row per (profile, series, view_date), same
#       key set and filters as the agg table (watch depth, binge, platform,
#       audio counts).
#   fease_item_attrs: one row per series (premiere date, episode count,
#       average episode length, first air year).
# Every bucket (depth, binge, dub share, discovery lag) is derived at read
# time by `kzn_recsys.spark.interaction_attrs`, never stored, so thresholds
# stay tunable and nothing time-relative is baked into a table. Built /
# refreshed by databricks/sql/*_attrs.sql via
# databricks/src/00_refresh_interactions_agg.py (target=attrs|item_attrs).
INTERACTION_ATTRS_TABLE = "dsml_recs.dev.fease_interaction_attrs"
ITEM_ATTRS_TABLE = "dsml_recs.dev.fease_item_attrs"

# Gold-table columns the attribute aggregation reads. UNVERIFIED against the
# live cr_prod.gold_db.ds_viewership schema: the names below come from the
# experience-tokenization proposal or are best guesses. Confirm each before
# the backfill; the refresh notebook also accepts a blank value for the
# duration and audio columns, which turns the dependent attributes into
# NULLs instead of failing.
VIEWERSHIP_MEDIA_ID_COL = "catalog_media_id"           # per-episode id (unverified)
VIEWERSHIP_MEDIA_DURATION_COL = "catalog_media_duration"  # episode length (unverified)
# Multiplier taking VIEWERSHIP_MEDIA_DURATION_COL to seconds (0.001 if the
# column turns out to be milliseconds).
MEDIA_DURATION_TO_SECONDS = 1.0
VIEWERSHIP_PLATFORM_COL = "view_platform_name"         # (unverified)
VIEWERSHIP_AUDIO_COL = "view_audio_locale"             # (unverified; may not exist per view)

# Series-grain catalog used by fease_item_attrs. UNVERIFIED as above.
CATALOG_TABLE = "cr_prod.gold_db.ds_catalog"
CATALOG_SERIES_COL = "catalog_show_id"
CATALOG_MEDIA_ID_COL = "catalog_media_id"
CATALOG_MEDIA_DURATION_COL = "catalog_media_duration"
CATALOG_AVAILABLE_FROM_COL = "catalog_available_date"  # first CR availability
CATALOG_AIR_DATE_COL = "catalog_original_air_date"     # for the era bucket

# A view with seconds / duration at or above this counts as a completed
# episode (stored count `completed_episodes`).
COMPLETED_RATIO = 0.9
# Audio class: views whose audio locale equals this are "sub"; any other
# non-null locale is "dub"; null audio counts as neither.
SUB_AUDIO_LOCALE = "ja-JP"

# Platform normalization, shared by the SQL (generated CASE) and PySpark
# (`interaction_attrs.normalize_platform`). Raw values are lower-cased and
# trimmed, then matched by substring against each class's patterns in the
# order listed (TV before mobile so "android tv" is TV); anything unmatched,
# or null, is "other". Check against the raw value distribution before the
# backfill (validate_fease_interaction_attrs.sql, query 6).
PLATFORM_CLASS_PATTERNS = (
    ("tv", ("tv", "roku", "firetv", "fire_tv", "chromecast", "playstation", "xbox",
            "nintendo", "console", "vizio", "tizen", "webos")),
    ("mobile", ("ios", "iphone", "ipad", "android", "mobile", "tablet")),
    ("web", ("web", "browser", "desktop", "chrome", "safari", "firefox")),
)
PLATFORM_OTHER = "other"
# Deterministic tie-break when two classes have the same view count on a day.
PLATFORM_TIE_ORDER = ("tv", "mobile", "web", "other")

# Read-time bucket thresholds.
DEPTH_PARTIAL_THRESHOLD = 0.25     # max_depth_ratio below this = "sampled"
DEPTH_COMPLETED_THRESHOLD = 0.9    # at or above this = "completed"
BINGE_HEAVY_MIN_EPISODES = 4       # episodes in a day that count as binge-heavy
DUB_SHARE_SUB_MAX = 0.2            # user dub share <= this = "sub" preference
DUB_SHARE_DUB_MIN = 0.8            # user dub share >= this = "dub" preference
DISCOVERY_LAG_NUM_BUCKETS = 32     # log2 buckets, same rule as days_ago positions
