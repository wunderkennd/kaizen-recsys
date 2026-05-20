# Choosing a model: EASE vs SASRec vs Two-Tower

This crate ships three recommendation models behind one `RecModel` trait
(see [ADR-0001](../adr/0001-multi-model-architecture.md)). They differ
substantially in what they assume about your data, what cold-start
behavior they offer, and what their training cost looks like. This guide
helps you pick.

## Quick decision matrix

| Question | Answer | Pick |
|----------|--------|------|
| Do I have **side features** on users/items, and need a closed-form deterministic model? | Yes | **EASE** |
| Do I have **chronologically-ordered** interaction sequences, and want next-item recommendations? | Yes | **SASRec** |
| Do I need a **dual-tower retrieval** model (separate user/item encoders), and care about learned cold-start? | Yes | **Two-Tower** |
| I just want a strong baseline with minimal data wrangling. | — | **EASE** |
| I have very long item catalogs and need an embedding-based retrieval index. | — | **Two-Tower** |
| I care about *which item the user looked at last*, not just *what they've ever liked*. | — | **SASRec** |

## Comparison

| Dimension | EASE | SASRec | Two-Tower |
|-----------|------|--------|-----------|
| Algorithm class | Closed-form linear autoencoder (Steck 2019) augmented with side features | Self-attentive sequential transformer (Kang & McAuley 2018) | Dual-tower in-batch sampled-softmax retrieval |
| Required cargo feature | (default) | `ml-models` | `ml-models` |
| Required input columns | `user_id`, `item_id`, `value` | `user_id`, `item_id`, `value`, **`days_ago`** | `user_id`, `item_id`, `value` |
| Side features used | Yes — user and item long-format tables fold into the closed-form solution | No — sequence-only | Yes — categorical features route through embeddings; dense features through MLPs |
| Cold-start strategy | Side features substitute for missing user history (the S-matrix has feature rows) | Empty history → no recommendation (model is order-sensitive) | Reserved cold-start user row trained via id-dropout (PR #46) |
| Training | One sparse Gram + dense matrix inversion. Deterministic to floating-point noise. | Stochastic gradient (Adam) over masked next-item prediction. Stochastic but seedable. | Stochastic gradient (Adam) over in-batch sampled-softmax. Stochastic but seedable. |
| Typical training time | Seconds to minutes (dominated by Gram inversion) | Minutes (transformer over CPU `NdArray` backend) | Minutes |
| Memory footprint | `O((M+K)^2)` for the Gram and S-matrices, independent of `N` (users) | `O(embedding_dim × max_seq_len × layers)` | `O(embedding_dim × (M + N + num_categories))` |
| Determinism guarantee | Byte-identical across runs (ADR-0002 §"Decision" #1) | Seedable; not byte-identical (BLAS / Adam moment-order) | Seedable; not byte-identical |
| Persisted format magic bytes | `FEAS` (v1/v2) | `FSAS` | `FTWO` |
| Hyperparameter search | `kzn_recsys.grid_search` / `grid_search_ease` / `random_search` / `random_search_ease` | `grid_search_sasrec` / `random_search_sasrec` | `grid_search_two_tower` / `random_search_two_tower` |
| Python predict-time inputs | `interactions: dict[str, float]`, `features: dict[str, float]` | `history: list[str]` (chronological, oldest first) | `user_id: str` (warm by lookup, unknown → cold-start row) |
| Predict-time side features | Yes (per-call user features) | n/a | **Not yet** — predict only takes a user id |

## Data shape

All three models read the same long-format Parquet/CSV files, plus a
`days_ago` column for SASRec:

- **`interactions.parquet`** — required columns: `user_id` (str),
  `item_id` (str), `value` (float). For SASRec, also: `days_ago` (float).
  Optional everywhere: `event_type` (str) for event weighting in EASE.
- **`user_features.parquet`** — `user_id` (str), `feature_name` (str),
  `value` (float). EASE folds these into the closed-form solution;
  Two-Tower splits one-hot features into categorical embeddings and
  numeric features into a dense MLP; SASRec ignores them.
- **`item_features.parquet`** — same shape with `item_id`. Same usage as
  above per model.

## When to pick which

**EASE.** Default. Strong baseline, no tuning needed to get a usable
model, byte-deterministic across runs, and the only model that natively
handles cold-start users with arbitrary side features at predict time.
Wins when: you have rich categorical user/item metadata and warm users
with sparse histories. Loses when: you care about temporal patterns
("this user just watched X — show them Y").

**SASRec.** Best for catalogs where the *order* of user interactions
carries signal — episodic / serialized content (next episode after this
one), or any context where the most recent item dominates the next pick.
Wins when: histories are long-ish, the catalog has clear sequential
structure, and you have a reliable `days_ago` timestamp. Loses when:
you have lots of side features (ignored), or your test users don't
share history shape with training (cold-start has no good story; the
generalized eval harness scores SASRec via `ModelInput::Sparse` which
discards chronology, so reported metrics are a lower bound — tracked in
[#51](https://github.com/wunderkennd/kaizen-recsys/issues/51)).

**Two-Tower.** Best for large catalogs where you want item embeddings
suitable for ANN retrieval, and where you have rich user/item features
you want the model to encode separately. Wins when: you have many
features that don't fit EASE's "long-format ID" shape cleanly (e.g.,
dense continuous features), or you want decoupled user/item encoders.
Loses when: you need predict-time arbitrary user features (not yet
supported — see Python wrapper docs in [README](../../README.md#two-tower-usage)),
or when warm users have very strong direct co-purchase signals that
EASE captures more cheaply.

## Mixing models

The `FeaseRegistry` is now generic over `&dyn RecModel`, so a single
registry can hold heterogeneous models keyed by territory / segment /
user cohort. See the README's
["Territory-Aware Model Registry"](../../README.md#territory-aware-model-registry)
section. The evaluation harness (`evaluate_model`) and hyperparameter
search runner (`tuning::grid_search_with` / `random_search_with`) work
against any model implementing `RecModel`.

## Hyperparameter starting points

These are not strong recommendations — just where the search functions
in this crate default to or where the included tests succeed.

- **EASE**: `alpha=1.0, beta=1.0, lambda_=100..150`. Tune with
  `kzn_recsys.grid_search`.
- **SASRec**: `embedding_dim=64, num_heads=2, num_layers=2, dropout=0.2,
  max_seq_len=50, learning_rate=1e-3, num_epochs=50, patience=5`. Tune
  with `kzn_recsys.grid_search_sasrec`.
- **Two-Tower**: `embedding_dim=32, temperature=0.05,
  learning_rate=0.01, epochs=50, batch_size=256, id_dropout=0.1`. Tune
  with `kzn_recsys.grid_search_two_tower`.

For all three, k-fold CV is user-based; metric optimization target is
NDCG@k (default `k=10`). `RAYON_NUM_THREADS` caps trial parallelism.
