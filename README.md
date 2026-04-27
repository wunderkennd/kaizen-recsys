# Rust FEASE Recommender
(Python Library)

This project implements a recommendation model based on the paper "Shallow AutoEncoding Recommender with Cold Start Handling via Side Features" (FEASE).
It is written in Rust and exposed as a Python library using PyO3. This gives you the performance of Rust for the heavy matrix computations and the ease-of-use of Python for data loading and serving.

This model is similar to the EASE model but is augmented to handle the user cold-start problem by integrating user and item side-features directly into the model's closed-form solution.

## The FEASE-U Model

The core idea is to augment the User-Item matrix (X) with a User-Feature matrix (U) and an Item-Feature matrix (T).

We define a new, combined feature matrix Z:

```
Z = [ X  | βU ]   (N x M) | (N x K)
[ αT |  0  ]   (L x M) | (L x K)
```

Where:
- N = number of users
- M = number of items
- K = number of user features
- L = number of item features
- α, β = scalar weights for the features

The model then learns a weight matrix S (size (M+K) x (M+K)) by solving the standard EASE objective:

```
L(S) = ||Z - ZS||^2 + λ||S||^2
```

This has a closed-form solution that depends on the Gram matrix G = Z^T * Z.

### Key to Memory Efficiency

This implementation's memory efficiency comes from never building the Z matrix, which could be enormous ((N+L) x (M+K)).

Instead, we compute the (M+K) x (M+K) Gram matrix G in blocks, using sparse-sparse matrix multiplication on the inputs:

```
G = Z^T * Z = [ G_11 | G_12 ]
[ G_21 | G_22 ]
```
Where:
- G_11 = X^T*X + α^2 * T^T*T (M x M)
- G_12 = β * X^T*U (M x K)
- G_21 = β * U^T*X (K x M)
- G_22 = β^2 * U^T*U (K x K)

All four blocks are computed efficiently from the sparse X, U, and T matrices. The only large, dense matrices we create are G, P = (G + λI)^-1, and S.

The key takeaway is that the memory bottleneck is O((M+K)^2), which is independent of the number of users (N).

## Building and Using the Python Library

This project is built as a Python library using maturin.

### 1. Prerequisites

- Rust
- Python 3.8+
- A Python virtual environment (recommended)
- maturin

Install maturin:
```
bash
pip install maturin
```
### 2. Building the Library

To build and install the library into your current Python virtual environment in "editable" mode, run:
```
bash
maturin develop
```
To build a wheel for distribution:
```
bash
maturin build --release
```
### 3. Python Usage Example

Once built, you can import and use the library directly in Python.
```python
import rust_fease_recommender as fease
import time

# Define paths to your long-format data files
INTERACTIONS_PATH = "path/to/interactions.parquet"   # user_id, item_id, value
USER_FEATURES_PATH = "path/to/user_features.parquet" # user_id, feature_name, value
ITEM_FEATURES_PATH = "path/to/item_features.parquet" # item_id, feature_name, value

# --- 1. Train the Model ---
print("Starting model training...")
start_time = time.time()

model = fease.build_and_train(
    interactions_path=INTERACTIONS_PATH,
    user_features_path=USER_FEATURES_PATH,
    item_features_path=ITEM_FEATURES_PATH,
    alpha=1.0,    # Weight for item features
    beta=1.0,     # Weight for user features
    lambda_=150.0 # L2 regularization
)
print(f"Training complete in {time.time() - start_time:.2f}s")
print(f"Model: {model.num_items} items, {model.num_user_features} user features.")


# --- 2. Make Predictions ---

# Example 1: Prediction for a WARM user
# (User has interaction history and features)
warm_user_interactions = {
"GEXU12345": 4.5,  # item_guid: log_watch_time
"GR9W56789": 3.2
}
warm_user_features = {
"device_Mobile": 1.0,
"plan_Premium": 1.0,
"tenure_365d+": 1.0
}

print("\n--- Warm User Predictions ---")
recs_warm = model.predict(warm_user_interactions, warm_user_features, top_k=5)
for guid, score in recs_warm:
print(f"  {guid}: {score:.4f}")


# Example 2: Prediction for a COLD START user
# (User has NO interaction history, only features)
cold_user_interactions = {}  # Empty dict
cold_user_features = {
"device_Web": 1.0,
"plan_Free": 1.0,
"tenure_0d": 1.0,
"country_acct_DE": 1.0
}

print("\n--- Cold Start User Predictions ---")
recs_cold = model.predict(cold_user_interactions, cold_user_features, top_k=5)
for guid, score in recs_cold:
print(f"  {guid}: {score:.4f}")
```

## Advanced Weighting

The `build_and_train()` function supports optional advanced weighting parameters:

```python
model = fease.build_and_train(
    interactions_path="interactions.parquet",
    user_features_path="user_features.parquet",
    item_features_path="item_features.parquet",
    alpha=1.0,
    beta=1.0,
    lambda_=150.0,
    # Advanced weighting (all optional, defaults preserve existing behavior):
    decay_rate=0.005,        # Exponential temporal decay (requires `days_ago` column)
    ips_alpha=0.5,           # Inverse propensity scoring (0=off, 1=aggressive)
    sparsity_threshold=0.001, # Prune small S-matrix entries
    event_weights={"click": 1.0, "cart": 3.0, "purchase": 5.0},  # Requires `event_type` column
)
```

The interactions Parquet file can optionally include:
- `event_type` (string): Used with `event_weights` to scale interactions by type
- `days_ago` (float): Used with `decay_rate` for exponential temporal decay

## Model Persistence

```python
# Save a trained model
model.save("model.fease")

# Load it back
loaded_model = fease.load_model("model.fease")
recs = loaded_model.predict(interactions, features, top_k=10)
```

## Batch Prediction

```python
users = [
    {"interactions": {"item1": 5.0}, "features": {"device_Mobile": 1.0}},
    {"interactions": {}, "features": {"plan_Free": 1.0}},
]
batch_results = model.predict_batch(users, top_k=10)
```

## Similar Items (MLT)

Find items similar to a given item using the learned S-matrix ("More Like This"):

```python
similar = model.predict_similar_items("GEXU12345", top_k=10)
for item_guid, score in similar:
    print(f"  {item_guid}: {score:.4f}")
```

## Territory-Aware Model Registry

Maintain multiple trained models keyed by territory/region:

```python
registry = fease.FeaseRegistry()

# Or with a fallback for unknown territories:
registry = fease.FeaseRegistry(fallback_territory="US")

# Register models per territory
registry.register("US", us_model)
registry.register("BR", br_model)

# Route predictions to the correct model
recs = registry.predict_top_k("US", interactions, features, top_k=10)

# List registered territories
print(registry.territories())  # ["US", "BR"]
```

## Evaluation Pipeline

Split data and evaluate model quality with standard ranking metrics:

### Data Splitting

Each splitter writes the train/test partitions to the paths you supply and returns
a `(train_interactions, test_interactions, train_users, test_users)` 4-tuple of counts.

```python
# Random train/test split (80/20)
train_int, test_int, train_users, test_users = fease.random_split(
    interactions_path="interactions.parquet",
    train_output="train.parquet",
    test_output="test.parquet",
    test_ratio=0.2,
    seed=42,
)

# Temporal split (test = interactions within last 7 days)
train_int, test_int, train_users, test_users = fease.temporal_split(
    interactions_path="interactions.parquet",
    train_output="train.parquet",
    test_output="test.parquet",
    days_ago_cutoff=7.0,
)

# Leave-K-out (hold out K items per user)
train_int, test_int, train_users, test_users = fease.leave_k_out_split(
    interactions_path="interactions.parquet",
    train_output="train.parquet",
    test_output="test.parquet",
    k=1,
    seed=42,
)
```

### Model Evaluation

```python
# Evaluate a trained model on held-out data
report = model.evaluate(
    test_interactions_path="test.parquet",
    train_interactions_path="train.parquet",
    user_features_path="user_features.parquet",  # optional
    k_values=[5, 10, 20, 50],
)
for m in report["metrics"]:
    print(f"  @{m['k']}: NDCG={m['ndcg']:.4f}, Recall={m['recall']:.4f}, Precision={m['precision']:.4f}")
print(f"  Coverage: {report['coverage']:.4f}")
print(f"  Users evaluated: {report['num_users']}, interactions: {report['num_interactions']}")
```

### Ranking Metrics (standalone)

```python
recommended = [10, 5, 3, 8, 1]
relevant = {3, 8, 15}

fease.precision_at_k(recommended, relevant, k=5)      # 0.4
fease.recall_at_k(recommended, relevant, k=5)          # 0.667
fease.ndcg_at_k(recommended, relevant, k=5)            # ...
fease.mean_average_precision(recommended, relevant)     # ...
fease.hit_rate_at_k(recommended, relevant, k=5)         # 1.0
fease.coverage(all_recs_list, num_total_items=1000)     # 0.85
```

## Hyperparameter Tuning

Optimize hyperparameters with k-fold cross-validation:

### Grid Search

```python
result = fease.grid_search(
    interactions_path="interactions.parquet",
    user_features_path="user_features.parquet",
    item_features_path="item_features.parquet",
    param_grid={
        "alpha": [0.5, 1.0, 2.0],
        "beta": [0.5, 1.0],
        "lambda_": [50.0, 100.0, 150.0],
    },
    n_folds=5,
    eval_k=20,          # Optimize NDCG@20
    seed=42,
)
print(f"Best NDCG@20: {result['best_score']:.4f}")
print(f"Best params: {result['best_params']}")
```

### Random Search

`random_search` samples uniformly from each list of candidate values (not from a
continuous `[min, max]` range), so the same `param_grid` shape used by
`grid_search` works here too.

```python
result = fease.random_search(
    interactions_path="interactions.parquet",
    user_features_path="user_features.parquet",
    item_features_path="item_features.parquet",
    param_grid={
        "alpha": [0.1, 0.5, 1.0, 2.0, 5.0],
        "beta": [0.1, 0.5, 1.0, 2.0, 5.0],
        "lambda_": [10.0, 50.0, 100.0, 200.0, 500.0],
    },
    n_trials=50,
    n_folds=5,
    eval_k=20,
    seed=42,
)
print(f"Best NDCG@20: {result['best_score']:.4f}")
for trial in result["trials"][:5]:
    print(f"  Score={trial['mean_score']:.4f}, params={trial['params']}")
```

## Data Quality Validation

```python
# Check current data against historical baselines before training
passed, messages = fease.validate_data(
    historical_users=[100.0, 105.0, 98.0],
    historical_items=[50.0, 52.0, 49.0],
    historical_interactions=[1000.0, 1050.0, 980.0],
    current_users=103.0,
    current_items=51.0,
    current_interactions=1030.0,
)
if not passed:
    print("Data quality check failed:", messages)
