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
# # Production Serving & Online Inference Service Stub
# 
# This file serves as a production-ready **file stub** demonstrating how to expose the high-performance
# Rust-native `kzn_recsys` model within a high-throughput **FastAPI** web application.
# 
# ### Architectural Flow
# 1. **Model Lifespan Loading**: The `FeaseModel` is loaded exactly once into memory during startup.
# 2. **Online Preprocessing (`predict_raw`)**: Real-time user/context features and interactions are sent directly in raw format (polymorphic inputs). Zero-drift feature binning is executed in Rust at microsecond latency.
# 3. **Whole Page Optimization (`optimize_layout`)**: The top candidates are dynamically solved using the Multiple-Choice Knapsack DP solver to decide which visual layout fits the page layout constraints (e.g., vertical height limits) while avoiding sequential banners.
# 
# To run this microservice locally:
# ```bash
# # Install FastAPI and Uvicorn
# .venv/bin/pip install fastapi uvicorn
# 
# # Run the script directly
# .venv/bin/python notebooks/05_production_serving_stub.py
# ```

# %%
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional
import os
import tempfile
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

import polars as pl
import kzn_recsys as fease
from kzn_recsys import (
    optimize_layout,
    FeatureTransformationSchema,
    NumericalBucketConfig,
    build_and_train
)

# =====================================================================
# 1. Pydantic Models for High-Performance Request/Response Validation
# =====================================================================

class RawUserFeatures(BaseModel):
    """Encapsulates raw, polymorphic online user and context features."""
    plan: str = Field(..., description="Raw subscription/tier level (e.g. 'Premium', 'Free')")
    tenure_days: int = Field(..., description="Days since user registration", ge=0)
    device: Optional[str] = Field("web", description="Device platform")

class RecommendRequest(BaseModel):
    """API request payload structure."""
    user_id: str = Field(..., description="Unique identifier for the active serving user")
    raw_interactions: Dict[str, float] = Field(
        default_factory=dict,
        description="Historical real-time interactions mapping item_guid -> weight"
    )
    raw_features: RawUserFeatures = Field(..., description="Polymorphic raw feature inputs")
    max_height: int = Field(8, description="Discretized visual pixel/height budget for the page", ge=1)
    
    # Visual grid slots (trays) with different options/layout representations
    trays: List[Dict[str, Any]] = Field(
        ...,
        description="List of trays, each containing visual layout options with height and item size specs"
    )

class SelectedLayout(BaseModel):
    """Representation of the chosen layout option for a tray slot."""
    tray_id: int
    format: str
    height: int
    utility: float
    item_count: int

class RecommendResponse(BaseModel):
    """API response payload structure."""
    user_id: str
    total_utility: float
    selected_layouts: List[SelectedLayout]
    latency_ms: float

# =====================================================================
# 2. FastAPI Lifespan Handler for Single-Load Model Management
# =====================================================================

# Global reference for our loaded FEASE inference model
MODEL: Optional[fease.FeaseModel] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages system startup and teardown lifecycles cleanly."""
    global MODEL
    
    # 1. Locate the trained model binary path
    model_path = os.environ.get("FEASE_MODEL_PATH", "model.fease")
    
    # 2. Fallback: If no model binary is found, we automatically bootstrap a small dummy model
    # to guarantee the stub runs out-of-the-box!
    if not Path(model_path).exists():
        print(f"[BOOTSTRAP] Model binary not found at {model_path}. Creating an on-the-fly model...")
        
        # Define embedded transformations
        schema = FeatureTransformationSchema()
        schema.add_categorical(col="plan", prefix="plan_")
        schema.add_numerical(
            col="tenure_days", 
            config=NumericalBucketConfig(
                prefix="tenure",
                boundaries=[0.0, 7.0, 30.0, 90.0],
                labels=["0d", "7d", "30d", "90d", "90d+"]
            )
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            i_path = Path(tmpdir) / "interactions.parquet"
            u_path = Path(tmpdir) / "user_features.parquet"
            t_path = Path(tmpdir) / "item_features.parquet"

            pl.DataFrame({"user_id": ["u0", "u1"], "item_id": ["G0", "G1"], "value": [5.0, 3.0]}).write_parquet(i_path)
            pl.DataFrame({"user_id": ["u0", "u1"], "feature_name": ["plan_Premium", "plan_Free"], "value": [1.0, 1.0]}).write_parquet(u_path)
            pl.DataFrame({"item_id": ["G0", "G1"], "feature_name": ["genre_Action", "genre_Comedy"], "value": [1.0, 1.0]}).write_parquet(t_path)

            trained_model = build_and_train(
                interactions_path=str(i_path),
                user_features_path=str(u_path),
                item_features_path=str(t_path),
                alpha=1.0,
                beta=1.0,
                lambda_=10.0,
                transformation_schema=schema
            )
            trained_model.save(model_path)
            print(f"[BOOTSTRAP] Dummy model trained and written to {model_path}.")
            
    print(f"[STARTUP] Loading FEASE model from {model_path} into memory...")
    try:
        MODEL = fease.load_model(model_path)
        print("[STARTUP] FEASE model successfully loaded.")
    except Exception as e:
        print(f"[STARTUP] Critical Error: Failed to load model! {e}")
        raise e
        
    yield
    
    # Cleanup resources (if any) during teardown
    print("[SHUTDOWN] Unloading model resources...")
    MODEL = None

# Initialize application
app = FastAPI(
    title="Kaizen Recommendation serving microservice",
    version="1.0.0",
    description="High-throughput, sub-millisecond recommender + layout optimization service.",
    lifespan=lifespan
)

# =====================================================================
# 3. High-Performance API Endpoints
# =====================================================================

@app.post("/recommend", response_model=RecommendResponse, status_code=status.HTTP_200_OK)
async def recommend(payload: RecommendRequest):
    """
    Exposes end-to-end recommendation & visual layout optimization.
    
    Flow:
    1. Parse polymorphic user features (categorical + numerical values).
    2. Natively pre-process raw features & fetch scores inside Rust via `predict_raw`.
    3. Feed layout options through `optimize_layout` DP knapsack solver.
    4. Enforce visual constraints (Sequential Banners disallowed!).
    """
    import time
    start_time = time.perf_counter()
    
    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently loading or failed to initialize."
        )
        
    # Step A: Transform raw user features into dictionary mapping
    raw_user_dict = payload.raw_features.model_dump()
    
    # Step B: Native Rust Inference using predict_raw
    try:
        # Predict top candidate scores in microseconds
        raw_recs = MODEL.predict_raw(
            interactions=payload.raw_interactions,
            raw_features=raw_user_dict,
            top_k=20
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rust inference execution failure: {e}"
        )
        
    # Convert recommendations to a fast score lookup table for tray option evaluations
    score_lookup = {item_guid: score for item_guid, score in raw_recs}
    
    # Step C: WPO Input Preparation
    # Format incoming slots with dynamic utility scoring based on individual candidate values
    wpo_trays = []
    for tray in payload.trays:
        tray_id = tray.get("id")
        options = tray.get("options", [])
        
        prepared_options = []
        for opt in options:
            fmt = opt.get("format", "None")
            height = opt.get("height", 0)
            item_count = opt.get("item_count", 0)
            
            # Simple utility calculation: Sum candidate scores matching item_count
            # In a real environment, you might score actual ranked items inside the tray.
            simulated_utility = sum(list(score_lookup.values())[:item_count])
            
            prepared_options.append({
                "format": fmt,
                "height": height,
                "utility": simulated_utility + opt.get("utility_offset", 0.0),
                "item_count": item_count
            })
            
        wpo_trays.append({
            "id": tray_id,
            "options": prepared_options
        })
        
    # Step D: Dynamic Programming WPO Layout Solving
    try:
        total_utility, selections = optimize_layout(wpo_trays, max_height=payload.max_height)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Layout optimizer solver failed: {e}"
        )
        
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    
    # Format layout results to match Pydantic schema
    selected_layouts_out = []
    for i, chosen_format in enumerate(selections):
        # chosen_format is a PyO3 Format object. Map it back to its string name.
        if chosen_format == fease.Format.Carousel:
            chosen_format_name = "Carousel"
        elif chosen_format == fease.Format.Banner:
            chosen_format_name = "Banner"
        elif chosen_format == fease.Format.Grid2x3:
            chosen_format_name = "Grid2x3"
        else:
            chosen_format_name = "None"
            
        orig_tray = wpo_trays[i]
        tray_id = orig_tray["id"]
        
        # Find the option details that match chosen_format_name
        matched_opt = None
        for opt in orig_tray["options"]:
            if opt["format"] == chosen_format_name:
                matched_opt = opt
                break
                
        if matched_opt is None:
            # Safe fallback
            matched_opt = {"format": chosen_format_name, "height": 0, "utility": 0.0, "item_count": 0}
            
        selected_layouts_out.append(SelectedLayout(
            tray_id=tray_id,
            format=matched_opt["format"],
            height=matched_opt["height"],
            utility=matched_opt["utility"],
            item_count=matched_opt["item_count"]
        ))
    
    return RecommendResponse(
        user_id=payload.user_id,
        total_utility=total_utility,
        selected_layouts=selected_layouts_out,
        latency_ms=latency_ms
    )

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Simple heart-beat endpoint confirming system availability."""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "num_items": MODEL.num_items if MODEL else None
    }

# =====================================================================
# 4. Interactive Test Harness
# =====================================================================
# %%
if __name__ == "__main__":
    import threading
    import time
    from fastapi.testclient import TestClient
    
    print("--- SPINNING UP TEST ENVIRONMENT ---")
    
    # Using FastAPI TestClient to test the complete flow end-to-end inside the script!
    with TestClient(app) as client:
        
        # Test 1: Validate Heartbeat
        health_resp = client.get("/health")
        print(f"Health Response: {health_resp.json()}")
        assert health_resp.status_code == 200
        
        # Test 2: Perform end-to-end WPO layout optimization under budget constraints
        payload = {
            "user_id": "user_premium_101",
            "raw_interactions": {
                "G0": 4.5,
                "G1": 1.2
            },
            "raw_features": {
                "plan": "Premium",
                "tenure_days": 45,
                "device": "mobile"
            },
            "max_height": 8,
            "trays": [
                {
                    "id": 0,
                    "options": [
                        {"format": "None", "height": 0, "item_count": 0, "utility_offset": 0.0},
                        {"format": "Carousel", "height": 2, "item_count": 3, "utility_offset": 1.0},
                        {"format": "Banner", "height": 4, "item_count": 1, "utility_offset": 3.0}
                    ]
                },
                {
                    "id": 1,
                    "options": [
                        {"format": "None", "height": 0, "item_count": 0, "utility_offset": 0.0},
                        {"format": "Carousel", "height": 2, "item_count": 2, "utility_offset": 2.5},
                        {"format": "Banner", "height": 4, "item_count": 1, "utility_offset": 5.0}
                    ]
                }
            ]
        }
        
        print("\nSending Recommendation API Request...")
        resp = client.post("/recommend", json=payload)
        
        print(f"Status Code: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print("--- WPO Response Data ---")
            print(f"User ID: {data['user_id']}")
            print(f"Total Page Utility: {data['total_utility']:.4f}")
            print(f"Total Latency: {data['latency_ms']:.3f} ms")
            print("Chosen Layouts:")
            for layout in data["selected_layouts"]:
                print(f"  Tray {layout['tray_id']} -> Selected Format: {layout['format']} (Height: {layout['height']}, Utility: {layout['utility']:.2f})")
        else:
            print(f"API Error Response: {resp.text}")
            
    # Clean up temp file created during bootstrap
    if Path("model.fease").exists():
        os.remove("model.fease")
        print("\nTemporary test binary model.fease deleted.")
