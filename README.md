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

# Define paths to your data
ENGAGEMENT_PATH = "path/to/engagement.parquet"
METADATA_PATH = "path/to/metadata.parquet"

# --- 1. Train the Model ---
# This one-time step loads data, builds matrices, and trains the model in Rust.
print("Starting model training...")
start_time = time.time()

try:
model = fease.build_and_train(
engagement_path=ENGAGEMENT_PATH,
metadata_path=METADATA_PATH,
alpha=1.0,   # Weight for item features
beta=1.0,    # Weight for user features
lambda_=150.0 # L2 regularization
)
print(f"Training complete in {time.time() - start_time:.2f}s")
print(f"Model trained on {model.num_items} items and {model.num_user_features} user features.")

except Exception as e:
print(f"An error occurred during training: {e}")
# Handle error (e.g., file not found)
exit()


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
