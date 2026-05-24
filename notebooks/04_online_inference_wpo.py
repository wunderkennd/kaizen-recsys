# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Online Inference & Whole Page Optimization (WPO) Quickstart
# 
# This notebook demonstrates how to leverage three key online serving upgrades:
# 1. **Dynamic Threshold QA Validation** (`GaussianAnomalyDetector`): Detecting pre-training data drift dynamically.
# 2. **Rust-Native Feature Preprocessing** (`predict_raw`): Zero-drift prediction directly from raw categorical and numerical values.
# 3. **Whole Page Optimization** (`optimize_layout`): Sub-millisecond layout knapsack solving under consecutive banner and visual height constraints.

# %%
import tempfile
from pathlib import Path
import polars as pl
import kzn_recsys as fease
from kzn_recsys import (
    Format,
    optimize_layout,
    GaussianAnomalyDetector,
    NumericalBucketConfig,
    FeatureTransformationSchema,
    build_and_train
)

# %% [markdown]
# ## 1. Dynamic Threshold QA Validation
# 
# The `GaussianAnomalyDetector` fits statistical bounds over historical data observations ($\mu \pm k \cdot \sigma$) and validates whether current runs are anomalous.

# %%
# Let's say we have 10 days of historical active user counts
historical_active_users = [
    10200.0, 9980.0, 10050.0, 10120.0, 9890.0,
    10010.0, 10080.0, 9950.0, 10250.0, 10000.0
]

# Fit anomaly detector with 3.0 standard deviation multiplier (3-sigma confidence)
detector = GaussianAnomalyDetector.fit(historical_active_users, std_multiplier=3.0)
print("--- Fitted Anomaly Detector ---")
print(f"Mean: {detector.mean:.2f}")
print(f"StdDev: {detector.std:.2f}")
print(f"Acceptance bounds: [{detector.low:.2f}, {detector.high:.2f}]")

# %%
# Check a normal daily count
assert detector.check(10100.0, label="active_users") is True
print("10,100 active users check passed successfully!")

# Check an anomalous daily count (e.g. pipeline partition missing)
is_valid = detector.check(6500.0, label="active_users")
print(f"6,500 active users check returned: {is_valid} (Anomaly detected!)")

# %% [markdown]
# ## 2. Model Training with Embedded Preprocessing Schema
# 
# We build a declarative preprocessing schema in Python, compile it into native Rust, and train a model with the schema permanently embedded and saved.

# %%
# Set up a transformation schema for raw categorical and numerical features
schema = FeatureTransformationSchema()

# 1. Prepend prefix to categorical features: raw "plan" value "Premium" -> "plan_Premium"
schema.add_categorical(col="plan", prefix="plan_")

# 2. Bucketize numeric features: raw "tenure_days" into bin labels based on boundaries
bucket_config = NumericalBucketConfig(
    prefix="tenure",
    boundaries=[0.0, 7.0, 30.0, 90.0],
    labels=["0d", "7d", "30d", "90d", "90d+"]
)
schema.add_numerical(col="tenure_days", config=bucket_config)

# %%
# Create temporary directories for dummy parquet files
with tempfile.TemporaryDirectory() as tmpdir:
    i_path = Path(tmpdir) / "interactions.parquet"
    u_path = Path(tmpdir) / "user_features.parquet"
    t_path = Path(tmpdir) / "item_features.parquet"

    # Save small training datasets
    pl.DataFrame({
        "user_id": ["u0", "u0", "u1"],
        "item_id": ["G0", "G2", "G1"],
        "value": [5.0, 4.0, 6.0],
    }).write_parquet(i_path)

    pl.DataFrame({
        "user_id": ["u0", "u0", "u1"],
        "feature_name": ["plan_Premium", "tenure_30d", "plan_Free"],
        "value": [1.0, 1.0, 1.0],
    }).write_parquet(u_path)

    pl.DataFrame({
        "item_id": ["G0", "G1", "G2"],
        "feature_name": ["genre_Action", "genre_Comedy", "genre_Action"],
        "value": [1.0, 1.0, 1.0],
    }).write_parquet(t_path)

    # Train the model with embedded transformation schema
    print("\n--- Training Model ---")
    model = build_and_train(
        interactions_path=str(i_path),
        user_features_path=str(u_path),
        item_features_path=str(t_path),
        alpha=1.0,
        beta=1.0,
        lambda_=10.0,
        transformation_schema=schema
    )
    print("Model trained successfully!")

    # Save model binary (Format version 3)
    save_path = Path(tmpdir) / "model.fease"
    model.save(str(save_path))
    print(f"Model saved to {save_path}")

    # Reload the model
    print("\n--- Loading Saved Model ---")
    loaded_model = fease.load_model(str(save_path))
    print("Model loaded successfully!")

# %% [markdown]
# ## 3. Online Preprocessing & Microsecond Inference (`predict_raw`)
# 
# We pass raw values directly to the model. The model automatically performs transformations and returns prediction scores with microsecond latency.

# %%
# Raw interactions: item_guid -> log watch time
raw_interactions = {"G0": 5.0}

# Raw online user features (Premium plan, 15 days tenure)
raw_online_features = {
    "plan": "Premium",
    "tenure_days": 15
}

# Run raw prediction
recs = loaded_model.predict_raw(
    interactions=raw_interactions,
    raw_features=raw_online_features,
    top_k=3
)

print("--- Online Inference Results ---")
for item_guid, score in recs:
    print(f"  Item: {item_guid} | Predicted Utility Score: {score:.4f}")

# %% [markdown]
# ## 4. Whole Page Layout Optimization (WPO)
# 
# Finally, we perform Whole Page Optimization. Given multiple visually formatted slots (trays) and a max height pixel budget, we dynamically select the optimal visual layouts.
# 
# Consecutive banner formats are automatically disallowed by the DP solver!

# %%
# Define visual formatting options for Tray Slot 0
tray_0 = {
    "id": 0,
    "options": [
        {"format": "None", "height": 0, "utility": 0.0, "item_count": 0},
        {"format": "Carousel", "height": 2, "utility": 1.5, "item_count": 5},
        {"format": "Banner", "height": 4, "utility": 3.0, "item_count": 1},
    ]
}

# Define visual formatting options for Tray Slot 1
tray_1 = {
    "id": 1,
    "options": [
        {"format": "None", "height": 0, "utility": 0.0, "item_count": 0},
        {"format": "Carousel", "height": 2, "utility": 2.0, "item_count": 5},
        {"format": "Banner", "height": 4, "utility": 4.5, "item_count": 1},
    ]
}

trays = [tray_0, tray_1]

# %%
# Scenario A: Pixel height budget = 8.
# The DP layout optimizer evaluates:
#   - Banner (Utility 3.0) + Banner (Utility 4.5) -> Rejected due to sequential banner constraint!
#   - Carousel (Utility 1.5) + Banner (Utility 4.5) -> Selected (Total utility = 6.0)
utility, layouts = optimize_layout(trays, max_height=8)
print("--- Layout Solution (Height Budget = 8) ---")
print(f"Total Page Utility: {utility:.2f}")
print(f"Selected Layouts: {layouts}")

# %%
# Scenario B: Pixel height budget = 3.
# Solves for the optimal layout fitting in the tight constraint.
utility, layouts = optimize_layout(trays, max_height=3)
print("\n--- Layout Solution (Height Budget = 3) ---")
print(f"Total Page Utility: {utility:.2f}")
print(f"Selected Layouts: {layouts}")
