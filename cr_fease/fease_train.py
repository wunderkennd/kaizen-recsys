# Databricks Training & Prediction Notebook for FEASE Recommender
#
# This notebook shows the end-to-end workflow for:
# 1. Installing the custom Rust library (`.whl` file)
# 2. Loading Databricks tables (Engagement & Metadata)
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
# 2. Upload the generated wheel (e.g., `rust_fease_recommender-0.1.0-cp310-cp310-manylinux_x86_64.whl`)
#    to a location on DBFS (e.g., /FileStore/libs/rust_fease_recommender-0.1.0-....whl)
#
# 3. Install the library on your cluster. You can do this via the Cluster UI
#    (Cluster -> Libraries -> Install New -> DBFS/S3 -> path_to.whl)
#    OR by running a notebook cell *before* this one:
#
# %pip install /dbfs/FileStore/libs/rust_fease_recommender-0.1.0-cp310-cp310-linux.whl
#
# After installation, you must detach and re-attach the notebook.

import os
import time
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import StringType

# Import our Rust library!
import rust_fease_recommender as fease

print("Successfully imported 'rust_fease_recommender'")

# COMMAND ----------

# --
# Step 2: Configuration
# --

# --- Spark Table Configuration ---
# Point these to your actual tables in Databricks
ENGAGEMENT_TABLE = "your_db.engagement"
METADATA_TABLE = "your_db.content_metadata"

# --- DBFS Temporary Path Configuration ---
# The Rust library will read from these /dbfs/ paths.
TEMP_DIR = "/dbfs/tmp/fease_model_flexible"
TEMP_I_PATH = os.path.join(TEMP_DIR, "interactions.parquet")
TEMP_U_PATH = os.path.join(TEMP_DIR, "user_features.parquet")
TEMP_T_PATH = os.path.join(TEMP_DIR, "item_features.parquet")

# Make sure the /dbfs/ directory exists
os.makedirs(TEMP_DIR, exist_ok=True)

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
# Set to 0.0 to disable (backward-compatible defaults).
# Values below are from the original EaseConfig in db_pipeline.ipynb.
DECAY_RATE = 0.0          # Exponential temporal decay on interactions (e.g., 0.005)
IPS_ALPHA = 0.0           # Inverse propensity scoring strength (e.g., 0.5)
SPARSITY_THRESHOLD = 0.0  # Prune S-matrix entries below this value (e.g., 0.001)

# Event-type weight multipliers. Set to None to disable event weighting.
# Keys must match the values in the `event_type` column of the interactions table.
# Example: {"click": 1.0, "cart": 3.0, "purchase": 5.0, "negative": -2.0}
EVENT_WEIGHTS = None

# --- Feature Engineering Configuration ---
MIN_WATCH_SECONDS = 30.0

# --- Engagement Column Names ---
# Column in the engagement table that contains the event type (e.g., "click", "purchase").
# Set to None if the column does not exist or event weighting is not needed.
EVENT_TYPE_COL = None      # e.g., "event_type" or "engagement_type"

# Column in the engagement table that contains the event timestamp.
# Used to compute `days_ago` for temporal decay. Set to None to skip.
TIMESTAMP_COL = "view_ts"  # e.g., "view_ts", "event_timestamp"

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
# Base schema: ["user_id", "item_id", "value"]
# Optional columns for advanced weighting: "event_type" (str), "days_ago" (float)
print("Building Interactions table...")

# Start with the base columns
_interaction_cols = [
    F.col("anonymous_id").alias("user_id"),
    F.col("view_media_id").alias("item_id"),
    # We use log-transform for watch time. Add 1 to avoid log(0).
    (F.log(F.col("view_seconds_watched") + 1.0)).alias("value"),
]

# Add event_type column if configured (required for event_weights in Rust)
if EVENT_TYPE_COL is not None:
    _interaction_cols.append(F.col(EVENT_TYPE_COL).cast(StringType()).alias("event_type"))

# Add days_ago column if configured (required for decay_rate in Rust)
if TIMESTAMP_COL is not None and DECAY_RATE > 0.0:
    _interaction_cols.append(
        F.datediff(F.current_date(), F.to_date(F.col(TIMESTAMP_COL)))
        .cast("double")
        .alias("days_ago")
    )

df_interactions = (
    df_eng
    .filter(F.col("view_seconds_watched") >= MIN_WATCH_SECONDS)
    .select(*_interaction_cols)
    .filter(F.col("user_id").isNotNull() & F.col("item_id").isNotNull())
    .distinct() # Or group by user/item and sum/avg value
)

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

# Select only the user features we want from the engagement table
# We take the most recent record for each user to get their "current" state
df_user_base = (
    df_eng
    .select(
        "anonymous_id",
        "view_ts",
        "view_subscription_plan",
        "account_country_code_account",
        "account_tenure_days",
        "region_major_account",
        "subscription_status"
    )
    .filter(F.col("anonymous_id").isNotNull())
    # Get the latest row for each user
    .orderBy(F.col("view_ts").desc())
    .dropDuplicates(["anonymous_id"])
)

# ---
# Experiment here! Add or remove columns from this list.
# ---
categorical_user_features = [
    "view_subscription_plan",
    "account_country_code_account",
    "region_major_account",
    "subscription_status"
]

df_user_categorical = to_long_format(df_user_base, "anonymous_id", categorical_user_features)

# Example of a numerical feature (bucketizing tenure)
df_user_tenure = (
    df_user_base
    .select("anonymous_id", "account_tenure_days")
    .withColumn("feature_name",
                F.when(F.col("account_tenure_days").isNull(), F.lit("tenure_unknown"))
                .when(F.col("account_tenure_days") <= 0, F.lit("tenure_0d"))
                .when(F.col("account_tenure_days") <= 7, F.lit("tenure_7d"))
                .when(F.col("account_tenure_days") <= 30, F.lit("tenure_30d"))
                .when(F.col("account_tenure_days") <= 90, F.lit("tenure_90d"))
                .otherwise(F.lit("tenure_90d+"))
                )
    .withColumn("value", F.lit(1.0))
    .select("anonymous_id", "feature_name", "value")
)

# Combine all user feature tables
df_user_features = (
    df_user_categorical
    .unionByName(df_user_tenure)
    .withColumnRenamed("anonymous_id", "user_id")
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

try:
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

# This one-time step loads data from the Parquet files,
# builds all matrices, and trains the model in Rust.
print("Starting model training (calling Rust library)...")
start_train = time.time()

try:
    # Build keyword arguments for optional weighting params.
    # Only pass non-default values so the Rust API stays backward-compatible.
    _train_kwargs = {}
    if DECAY_RATE > 0.0:
        _train_kwargs["decay_rate"] = DECAY_RATE
    if IPS_ALPHA > 0.0:
        _train_kwargs["ips_alpha"] = IPS_ALPHA
    if SPARSITY_THRESHOLD > 0.0:
        _train_kwargs["sparsity_threshold"] = SPARSITY_THRESHOLD
    if EVENT_WEIGHTS is not None:
        _train_kwargs["event_weights"] = EVENT_WEIGHTS

    model = fease.build_and_train(
        interactions_path=TEMP_I_PATH,
        user_features_path=TEMP_U_PATH,
        item_features_path=TEMP_T_PATH,
        alpha=ALPHA,
        beta=BETA,
        lambda_=LAMBDA,  # Note the trailing underscore
        **_train_kwargs,
    )

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

# Train a model on the training split for evaluation.
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

MODEL_SAVE_PATH = os.path.join(TEMP_DIR, "model.fease")
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