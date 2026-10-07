# Databricks Training & Prediction Notebook for FEASE Recommender
#
# This notebook shows the end-to-end workflow for:
# 1. Installing the custom Rust library (`.whl` file)
# 2. Loading Databricks tables (the series-grain interactions table from
#    issue #102, or raw engagement events; plus content metadata)
# 3. Performing Feature Engineering in PySpark to create the three
#    "long-format" tables (interactions, user_features, item_features).
# 4. Exporting these three tables to temporary Parquet files on DBFS.
# 5. Training the FEASE model by calling the Rust library.
# 6. Running predictions for warm and cold-start users.
# 7. Cleaning up temporary files.

# COMMAND ----------

# --
# Step 1: Library Installation
# --
#
# 1. Build your Rust wheel for Linux (e.g., using `maturin build --release`
#    via Docker, cross-compilation, or a CI/CD pipeline).
#
# 2. Upload the generated wheel (e.g., `kzn_recsys-0.1.0-cp310-cp310-manylinux_x86_64.whl`)
#    to a location on DBFS (e.g., /FileStore/libs/kzn_recsys-0.1.0-....whl)
#
# 3. Install the library on your cluster. You can do this via the Cluster UI
#    (Cluster -> Libraries -> Install New -> DBFS/S3 -> path_to.whl)
#    OR by running a notebook cell *before* this one:
#
# %pip install /dbfs/FileStore/libs/kzn_recsys-0.1.0-cp310-cp310-linux.whl
#
# After installation, you must detach and re-attach the notebook.

import os
import time
from pyspark.sql import SparkSession, DataFrame, Window
import pyspark.sql.functions as F

# Import our library! `kzn_recsys` re-exports the compiled Rust extension
# (`kzn_recsys._native`) plus Python helpers (SplitResult, schemas, wrappers).
import kzn_recsys as fease
from kzn_recsys import cr_config as cfg
from kzn_recsys.spark.interactions_agg import (
    active_users,
    aggregate_viewership_daily,
    daily_to_pairs,
    filter_activity_window,
    with_days_ago,
)

print("Successfully imported 'kzn_recsys'")

# COMMAND ----------

# --
# Step 2: Configuration
# --

# --- Spark Table Configuration ---
# Point these to your actual tables in Databricks.
#
# Interactions come from one of two sources (issue #102):
#   "agg": the series-grain daily table `cfg.INTERACTIONS_AGG_TABLE`
#          (one row per profile x series x day, built by
#          databricks/sql/*.sql). The path for multi-year history.
#   "raw": ENGAGEMENT_TABLE events aggregated to the same daily grain on
#          the fly. For small windows and ad-hoc experiments; identical
#          downstream code.
INTERACTIONS_SOURCE = "agg"
ENGAGEMENT_TABLE = cfg.VIEWERSHIP_TABLE        # used by the "raw" source and user features
METADATA_TABLE = "your_db.content_metadata"
# Columns of ENGAGEMENT_TABLE. The user id must be the same id space the
# agg table was built with (profile id), or user features will not join.
USER_ID_COL = cfg.VIEWERSHIP_USER_COL          # "view_profile_id"
ITEM_ID_COL = cfg.VIEWERSHIP_ITEM_COL          # "catalog_show_id" (series grain)
WATCH_SECONDS_COL = cfg.VIEWERSHIP_SECONDS_COL
SUBSIDIARY_COL = cfg.VIEWERSHIP_SUBSIDIARY_COL
SUBSIDIARY = cfg.VIEWERSHIP_SUBSIDIARY
# Only interactions with days_ago <= this many days are used. None = all
# history in the source. days_ago is derived at read time from view_date.
ACTIVITY_WINDOW_DAYS = None

# --- Training Backend ---
#   "rust":  pair-grain Parquet handoff + the native extension. The whole
#            interactions frame is read on one node.
#   "spark": kzn_recsys.spark.build_and_train(strategy="distributed"): the
#            Gram blocks are Spark aggregations and only the (M+K)^2 matrix
#            reaches the driver, so history length never bottlenecks the
#            driver. Needs numpy/scipy on the cluster. The trained model is
#            saved in the FEAS format the native extension loads, so the
#            prediction / evaluation steps below are unchanged.
TRAINING_BACKEND = "rust"

# --- DBFS Temporary Path Configuration ---
# The Rust library will read from these /dbfs/ paths.
# IMPORTANT: this directory is removed by Step 7 cleanup. Do NOT save the
# trained model here — use MODEL_OUTPUT_DIR below.
TEMP_DIR = "/dbfs/tmp/fease_model_flexible"
TEMP_I_PATH = os.path.join(TEMP_DIR, "interactions.parquet")
TEMP_U_PATH = os.path.join(TEMP_DIR, "user_features.parquet")
TEMP_T_PATH = os.path.join(TEMP_DIR, "item_features.parquet")

# Make sure the /dbfs/ directory exists
os.makedirs(TEMP_DIR, exist_ok=True)

# --- Persistent Model Output Configuration ---
# Trained models are saved here for later inference. This must live OUTSIDE
# TEMP_DIR so it survives the Step 7 cleanup.
MODEL_OUTPUT_DIR = "/dbfs/models/fease"
MODEL_SAVE_PATH = os.path.join(MODEL_OUTPUT_DIR, "model.fease")
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# --- Spark Path Configuration ---
# Spark writes to DBFS paths *without* the /dbfs/ prefix.
SPARK_I_PATH = "file:/tmp/fease_model_flexible/interactions.parquet"
SPARK_U_PATH = "file:/tmp/fease_model_flexible/user_features.parquet"
SPARK_T_PATH = "file:/tmp/fease_model_flexible/item_features.parquet"

# --- Model Hyperparameters ---
ALPHA = 1.0   # Weight for item features
BETA = 1.0    # Weight for user features
LAMBDA = 150.0  # L2 regularization

# --- Advanced Weighting Parameters ---
# Set to 0.0 to disable (backward-compatible defaults). Prefer setting these
# from a `tune_ease` run (strategy="tpe") rather than by hand.
DECAY_RATE = 0.0          # Exponential temporal decay per day (e.g., 0.005)
IPS_ALPHA = 0.0           # Inverse propensity scoring strength (e.g., 0.5)
SPARSITY_THRESHOLD = 0.0  # Prune S-matrix entries below this value (e.g., 0.001)
# DECAY_RATE is applied in Spark when the daily rows are collapsed to
# (user, item) pairs (`daily_to_pairs`), per day, before summing. It is
# therefore never passed to the training backend; passing it as well would
# decay twice. IPS and sparsity are applied by the backend on the pair
# frame, so IPS propensities count users per item, not view events.

# Event-type weight multipliers. The series-grain table carries no
# event_type column (issue #102), so this only applies to a source that
# does; leave None.
EVENT_WEIGHTS = None

# --- Feature Engineering Configuration ---
MIN_WATCH_SECONDS = cfg.MIN_WATCH_SECONDS

# Timestamp / date column of ENGAGEMENT_TABLE; the "raw" source derives
# view_date from it, and user features take each user's latest row by it.
TIMESTAMP_COL = cfg.VIEWERSHIP_DATE_COL  # a DATE or TIMESTAMP column

# COMMAND ----------

# --
# Step 3: Feature Engineering (Python/PySpark)
# --
#
# This is where all your experimentation happens!
# We will create the three required DataFrames (Interactions, User Features, Item Features)
# from the source tables.

spark = SparkSession.builder.getOrCreate()

print(f"Loading Engagement table: {ENGAGEMENT_TABLE}...")
df_eng = spark.table(ENGAGEMENT_TABLE)

print(f"Loading Content Metadata table: {METADATA_TABLE}...")
df_meta = spark.table(METADATA_TABLE)

# ---
# A. Create Interactions DataFrame
# ---
# Daily grain (issue #102): ["user_id", "item_id", "view_date", "value",
# "num_views"] + "days_ago" derived at read time. value is
# sum(ln(seconds + 1)) over a day's views of a series.
#
# `df_interactions_daily` keeps one row per profile x series x day and is
# what the sequence models (SASRec, BERT4Rec) consume. `df_interactions`
# collapses it to one row per (user, item) for EASE, applying DECAY_RATE per
# day before summing and keeping days_ago of the most recent day. Both
# backends would compute the same Gram from the daily rows; the pair frame
# just ships far fewer rows and avoids a quadratic self-join.
print(f"Building Interactions table from source={INTERACTIONS_SOURCE!r}...")

if INTERACTIONS_SOURCE == "agg":
    df_daily = spark.table(cfg.INTERACTIONS_AGG_TABLE)
elif INTERACTIONS_SOURCE == "raw":
    df_daily = aggregate_viewership_daily(
        df_eng,
        user_col=USER_ID_COL,
        item_col=ITEM_ID_COL,
        date_col=TIMESTAMP_COL,
        seconds_col=WATCH_SECONDS_COL,
        min_watch_seconds=MIN_WATCH_SECONDS,
        subsidiary_col=SUBSIDIARY_COL,
        subsidiary=SUBSIDIARY,
    )
else:
    raise ValueError(f"INTERACTIONS_SOURCE={INTERACTIONS_SOURCE!r}; expected 'agg' or 'raw'")

df_interactions_daily = filter_activity_window(with_days_ago(df_daily), ACTIVITY_WINDOW_DAYS)
df_interactions = daily_to_pairs(df_interactions_daily, decay_rate=DECAY_RATE)

# ---
# B. Create User Features DataFrame
# ---
# schema: ["user_id", "feature_name", "value"]
print("Building User Features table...")

# Helper function to create a "long" feature table from a "wide" table
def to_long_format(df: DataFrame, id_col: str, feature_cols: list) -> DataFrame:
    """Melts a DataFrame from wide to long format for features."""
    melted_dfs = []
    for col_name in feature_cols:
        melted_df = (
            df
            .select(
                F.col(id_col),
                F.concat(F.lit(f"{col_name}_"), F.col(col_name)).alias("feature_name")
            )
            .filter(F.col(col_name).isNotNull())
            .withColumn("value", F.lit(1.0))
        )
        melted_dfs.append(melted_df)

    # Union all feature DataFrames
    if not melted_dfs:
        return spark.createDataFrame([], schema="user_id string, feature_name string, value double")

    final_df = melted_dfs[0]
    for i in range(1, len(melted_dfs)):
        final_df = final_df.unionByName(melted_dfs[i])

    return final_df.distinct()

# ---
# Experiment here! Add or remove columns from this list.
# ---
categorical_user_features = [
    "view_subscription_plan",
    "account_country_code_account",
    "region_major_account",
    "subscription_status"
]

# Each user's most recent engagement row gives their "current" state. Only
# users present in the interactions frame are kept, so the full engagement
# table is scanned once for features and the user mapping matches the
# interactions exactly. row_number() over an explicit window is
# deterministic; orderBy + dropDuplicates is not.
df_user_base = (
    df_eng
    .filter(F.col(SUBSIDIARY_COL) == F.lit(SUBSIDIARY))
    .filter(F.nullif(F.trim(F.col(USER_ID_COL)), F.lit("")).isNotNull())
    .join(
        active_users(df_interactions_daily).withColumnRenamed("user_id", "_active_uid"),
        F.col(USER_ID_COL) == F.col("_active_uid"),
        "inner",
    )
    .drop("_active_uid")
    .select(USER_ID_COL, TIMESTAMP_COL, "account_tenure_days", *categorical_user_features)
    .withColumn(
        "_rn",
        F.row_number().over(Window.partitionBy(USER_ID_COL).orderBy(F.col(TIMESTAMP_COL).desc())),
    )
    .filter(F.col("_rn") == 1)
    .drop("_rn")
)

df_user_categorical = to_long_format(df_user_base, USER_ID_COL, categorical_user_features)

# Example of a numerical feature (bucketizing tenure)
df_user_tenure = (
    df_user_base
    .select(USER_ID_COL, "account_tenure_days")
    .withColumn("feature_name",
                F.when(F.col("account_tenure_days").isNull(), F.lit("tenure_unknown"))
                .when(F.col("account_tenure_days") <= 0, F.lit("tenure_0d"))
                .when(F.col("account_tenure_days") <= 7, F.lit("tenure_7d"))
                .when(F.col("account_tenure_days") <= 30, F.lit("tenure_30d"))
                .when(F.col("account_tenure_days") <= 90, F.lit("tenure_90d"))
                .otherwise(F.lit("tenure_90d+"))
                )
    .withColumn("value", F.lit(1.0))
    .select(USER_ID_COL, "feature_name", "value")
)

# Combine all user feature tables
df_user_features = (
    df_user_categorical
    .unionByName(df_user_tenure)
    .withColumnRenamed(USER_ID_COL, "user_id")
    .distinct()
)


# ---
# C. Create Item Features DataFrame
# ---
# schema: ["item_id", "feature_name", "value"]
print("Building Item Features table...")

# Helper for splitting comma-separated features like genres/tags
def split_and_explode(df: DataFrame, id_col: str, feature_col: str, prefix: str) -> DataFrame:
    """Splits a comma-separated string column and explodes it to long format."""
    return (
        df
        .select(
            F.col(id_col),
            F.explode(
                F.split(F.col(feature_col), ",")
            ).alias("feature_val")
        )
        .withColumn("feature_name", F.concat(F.lit(prefix), F.trim(F.col("feature_val"))))
        .withColumn("value", F.lit(1.0))
        .select(id_col, "feature_name", "value")
    )

# ---
# Experiment here! Add or remove features.
# ---
categorical_item_features = [
    "media_type",
    "media_audio_language",
    "media_series_title",
    "airtable_primary_genre",
    "airtable_ca_brand_grade"
]

df_item_categorical = to_long_format(df_meta, "media_guid", categorical_item_features)

# Split/explode features
df_item_genres = split_and_explode(df_meta, "media_guid", "media_genres", "genre_")
df_item_tags = split_and_explode(df_meta, "media_guid", "media_tags", "tag_")

# Combine all item feature tables
df_item_features = (
    df_item_categorical
    .unionByName(df_item_genres)
    .unionByName(df_item_tags)
    .withColumnRenamed("media_guid", "item_id")
    .filter(F.col("feature_name").isNotNull() & (F.col("feature_name") != F.lit("")))
    .distinct()
)


# COMMAND ----------

# --
# Step 4: Write Feature Tables to DBFS
# --

# We coalesce to 1 partition to write a *single* Parquet file.
# This is VASTLY faster for the single-threaded Polars reader in Rust
# than reading a directory of 200+ sharded Parquet files.
#
# The pair-grain frame is written for EASE (both backends; the "rust"
# backend trains from it, and evaluation / tuning read it either way). Set
# WRITE_DAILY_INTERACTIONS to also write the daily grain for SASRec /
# BERT4Rec, which need per-event days_ago.
WRITE_DAILY_INTERACTIONS = False
SPARK_I_DAILY_PATH = SPARK_I_PATH.replace("interactions.parquet", "interactions_daily.parquet")
TEMP_I_DAILY_PATH = TEMP_I_PATH.replace("interactions.parquet", "interactions_daily.parquet")

try:
    if WRITE_DAILY_INTERACTIONS:
        print(f"Writing daily-grain Interactions data to {SPARK_I_DAILY_PATH}...")
        (
            df_interactions_daily
            .select("user_id", "item_id", "value", "days_ago")
            .coalesce(1)
            .write
            .mode("overwrite")
            .parquet(SPARK_I_DAILY_PATH)
        )

    print(f"Writing Interactions data to {SPARK_I_PATH}...")
    start_write = time.time()
    (
        df_interactions
        .coalesce(1)
        .write
        .mode("overwrite")
        .parquet(SPARK_I_PATH)
    )
    print(f"Wrote Interactions data in {time.time() - start_write:.2f}s")

    print(f"Writing User Features data to {SPARK_U_PATH}...")
    start_write = time.time()
    (
        df_user_features
        .coalesce(1)
        .write
        .mode("overwrite")
        .parquet(SPARK_U_PATH)
    )
    print(f"Wrote User Features data in {time.time() - start_write:.2f}s")

    print(f"Writing Item Features data to {SPARK_T_PATH}...")
    start_write = time.time()
    (
        df_item_features
        .coalesce(1)
        .write
        .mode("overwrite")
        .parquet(SPARK_T_PATH)
    )
    print(f"Wrote Item Features data in {time.time() - start_write:.2f}s")

except Exception as e:
    print(f"Error writing Parquet files: {e}")
    # Use dbutils.notebook.exit() to stop the notebook on failure
    dbutils.notebook.exit(f"Failed to write Parquet files: {e}")

# COMMAND ----------

# --
# Step 5: Train the Rust Model
# --

# "rust": load the Parquet files, build all matrices and train in Rust.
# "spark": compute the Gram blocks in Spark and solve on the driver, then
# save the FEAS file and load it with the native extension so Steps 6+ are
# backend-agnostic.
print(f"Starting model training (backend={TRAINING_BACKEND!r})...")
start_train = time.time()

try:
    # Keyword arguments for the optional weighting params. Only non-default
    # values are passed so the Rust API stays backward-compatible. DECAY_RATE
    # is deliberately absent: it was applied in Spark by daily_to_pairs.
    _train_kwargs = {}
    if IPS_ALPHA > 0.0:
        _train_kwargs["ips_alpha"] = IPS_ALPHA
    if SPARSITY_THRESHOLD > 0.0:
        _train_kwargs["sparsity_threshold"] = SPARSITY_THRESHOLD
    if EVENT_WEIGHTS is not None:
        _train_kwargs["event_weights"] = EVENT_WEIGHTS

    if TRAINING_BACKEND == "rust":
        model = fease.build_and_train(
            interactions_path=TEMP_I_PATH,
            user_features_path=TEMP_U_PATH,
            item_features_path=TEMP_T_PATH,
            alpha=ALPHA,
            beta=BETA,
            lambda_=LAMBDA,  # Note the trailing underscore
            **_train_kwargs,
        )
    elif TRAINING_BACKEND == "spark":
        from kzn_recsys.spark import WeightingConfig
        from kzn_recsys.spark import build_and_train as spark_build_and_train

        _weighting = None
        if _train_kwargs:
            _weighting = WeightingConfig(
                event_weights=EVENT_WEIGHTS,
                decay_rate=0.0,  # already applied per day in daily_to_pairs
                ips_alpha=IPS_ALPHA,
                sparsity_threshold=SPARSITY_THRESHOLD,
            )
        spark_model = spark_build_and_train(
            df_interactions,
            df_user_features,
            df_item_features,
            alpha=ALPHA,
            beta=BETA,
            lambda_=LAMBDA,
            weighting=_weighting,
            strategy="distributed",  # only the (M+K)^2 Gram reaches the driver
        )
        spark_model.save(MODEL_SAVE_PATH)
        model = fease.load_model(MODEL_SAVE_PATH)
    else:
        raise ValueError(f"TRAINING_BACKEND={TRAINING_BACKEND!r}; expected 'rust' or 'spark'")

    print(f"Training complete in {time.time() - start_train:.2f}s")
    print(f"Model trained on {model.num_items} items and {model.num_user_features} user features.")

except Exception as e:
    print(f"An error occurred during training: {e}")
    # If training fails, we still want to clean up
    raise e

# COMMAND ----------

# --
# Step 6: Run Predictions
# --

# Now you have a 'model' object in memory, ready for predictions.

# Example 1: Prediction for a WARM user
# (User has interaction history and features)
warm_user_interactions = {
    "GEXU12345": 4.5,  # item_guid: log_watch_time
    "GR9W56789": 3.2
}
warm_user_features = {
    "plan_Premium": 1.0,
    "tenure_90d+": 1.0,
    "country_acct_US": 1.0,
    "region_US/CA": 1.0,
    "sub_status_Paying": 1.0
}

print("\n--- Warm User Predictions ---")
recs_warm = model.predict(warm_user_interactions, warm_user_features, top_k=5)
for guid, score in recs_warm:
    print(f"  {guid}: {score:.4f}")


# Example 2: Prediction for a COLD START user
# (User has NO interaction history, only features)
cold_user_interactions = {}  # Empty dict
cold_user_features = {
    "plan_Free": 1.0,
    "tenure_0d": 1.0,
    "country_acct_DE": 1.0,
    "region_EMEA": 1.0,
    "sub_status_Free Trial": 1.0
}

print("\n--- Cold Start User Predictions ---")
recs_cold = model.predict(cold_user_interactions, cold_user_features, top_k=5)
for guid, score in recs_cold:
    print(f"  {guid}: {score:.4f}")

# COMMAND ----------

# --
# Step 6b: Evaluate Model Quality (Optional)
# --
#
# Split the data and evaluate to get ranking metrics before deploying.

import tempfile

print("\n--- Model Evaluation ---")

# random_split writes the split to disk; pick a workspace it can write to.
SPLIT_DIR = tempfile.mkdtemp(prefix="fease_split_")
train_split = os.path.join(SPLIT_DIR, "train.parquet")
test_split = os.path.join(SPLIT_DIR, "test.parquet")

train_int, test_int, train_users, test_users = fease.random_split(
    interactions_path=TEMP_I_PATH,
    train_output=train_split,
    test_output=test_split,
    test_ratio=0.2,
    seed=42,
)
print(
    f"Split: {train_int} train, {test_int} test interactions "
    f"({train_users} train users, {test_users} test users)"
)

# Train a model on the training split for evaluation. This retrains with the
# native extension on the split regardless of TRAINING_BACKEND; for
# multi-year history use the bundle's 03_evaluate notebook (Spark backend,
# availability-aware) instead.
# Reuse the same _train_kwargs pattern so disabled weighting stays backward-compatible.
eval_model = fease.build_and_train(
    interactions_path=train_split,
    user_features_path=TEMP_U_PATH,
    item_features_path=TEMP_T_PATH,
    alpha=ALPHA,
    beta=BETA,
    lambda_=LAMBDA,
    **_train_kwargs,
)

report = eval_model.evaluate(
    test_interactions_path=test_split,
    train_interactions_path=train_split,
    user_features_path=TEMP_U_PATH,
    k_values=[5, 10, 20, 50],
)
for m in report["metrics"]:
    print(
        f"  @{m['k']}: NDCG={m['ndcg']:.4f}, "
        f"Recall={m['recall']:.4f}, Precision={m['precision']:.4f}"
    )
print(f"  Coverage: {report['coverage']:.4f}")
print(f"  Users evaluated: {report['num_users']}, interactions: {report['num_interactions']}")

# COMMAND ----------

# --
# Step 6c: Save Model (Optional)
# --
#
# Persist the trained model for later inference without re-training.
# MODEL_SAVE_PATH is defined in the configuration section above and points
# to MODEL_OUTPUT_DIR (a persistent location), NOT TEMP_DIR which is wiped
# by Step 7.

model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

# To load later:
# loaded_model = fease.load_model(MODEL_SAVE_PATH)

# COMMAND ----------

# --
# Step 7: Cleanup (Optional but recommended)
# --
#
# Use dbutils.fs.rm to clean up the temporary files/directory
# from DBFS.
dbutils = DBUtils(spark)
try:
    print(f"\nCleaning up temporary directory: {SPARK_I_PATH}...")
    # Use the Spark path for dbutils
    spark_temp_dir = SPARK_I_PATH.replace("file:", "").replace("/interactions.parquet", "")
    dbutils.fs.rm(spark_temp_dir, recurse=True)
    print("Cleanup successful.")
except Exception as e:
    print(f"Warning: Failed to clean up temp files. {e}")