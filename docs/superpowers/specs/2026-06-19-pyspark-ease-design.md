# PySpark EASE Implementation — Design

**Date:** 2026-06-19
**Status:** Approved (design); pending implementation plan
**Scope:** A pure-Python/PySpark implementation of the EASE recommender,
runnable in environments where the compiled `kzn_recsys._native` extension
cannot be installed.

## Motivation

The published `kzn_recsys` wheels are EASE-only, and the compiled Rust
extension cannot be installed in every target environment (governance-
restricted clusters, serverless runtimes, shared platforms where custom
native wheels are disallowed). The goal is **portability**: a pure-Python
implementation, using only standard PyPI packages, that runs where the
wheel cannot — while staying numerically faithful to the Rust core.

This is explicitly *not* a scale play. The Rust core already parallelizes
batch prediction and tuning via rayon (`src/serving.rs`, `src/tuning.rs`);
distributed training is a secondary benefit of the Spark path, not the
reason for it.

### Decisions locked during brainstorming

- **Models:** EASE only. EASE is closed-form and needs only NumPy/SciPy —
  no deep-learning framework. SASRec and Two-Tower are out of scope.
- **Dependencies:** PySpark + NumPy/SciPy only. No PyTorch, no MLlib
  distributed matrices.
- **Feature surface:** Full parity — train, predict, advanced weighting,
  evaluation, splits, and hyperparameter tuning all runnable in the
  restricted environment.
- **Model interop:** **Both directions.** Models trained by the Rust core
  must load in PySpark, and PySpark-trained models must load in the Rust
  core. This requires a byte-compatible FEAS codec.
- **Parity bar:** Tolerance-based — identical top-K rankings and scores
  within `1e-5` relative on shared fixtures. Bit-exactness is not required
  for scores (BLAS vs nalgebra differ in the last bits) but **is** required
  for the FEAS binary format.
- **Packaging:** Optional-native restructure — one package, one version,
  one name; native symbols conditionally exported.

## Architecture

Layered SciPy core + Spark adapter. The math lives in a pure
NumPy/SciPy module that never imports `pyspark`; Spark is confined to IO,
index-mapping construction, and (optionally) distributed Gram accumulation.

```
kzn_recsys/
  __init__.py          # restructured: _native imports wrapped in
                       # try/except -> _HAS_NATIVE
  spark/
    __init__.py        # public API, mirrors top-level names
    ease_core.py       # pure NumPy/SciPy math — no pyspark — mirrors model.rs
    dataframes.py      # long-format DataFrame -> index mappings + sparse inputs
    gram.py            # Gram strategies: collect-to-driver | distributed 4-block
    model.py           # SparkEaseModel facade: predict / predict_batch /
                       # evaluate / save / load
    feas_codec.py      # pure-Python bincode reader/writer for FEAS v1/v2
    metrics.py         # mirrors metrics.rs (pure functions)
    splits.py          # random / temporal / leave-K-out on DataFrames
    tuning.py          # grid_search / random_search with k-fold CV
```

### Package structure & distribution

1. **`__init__.py` restructure.** The top-level unconditional
   `from kzn_recsys._native import ...` (`kzn_recsys/__init__.py:3`) becomes
   a `try/except ImportError` setting `_HAS_NATIVE`, exactly the pattern
   already used for `_HAS_ML_MODELS` and `_HAS_ONNX` in the same file.
   Without the native module, `import kzn_recsys` still succeeds and
   `kzn_recsys.spark` is fully usable.

2. **Distribution.** One PyPI project, two wheel kinds: the existing
   maturin platform wheels (full native) plus a `py3-none-any` pure-Python
   wheel containing only the Python sources. Pip auto-selects the platform
   wheel where compatible and falls back to the universal wheel where it is
   not — precisely the restricted-environment case. PySpark/NumPy/SciPy
   ship as an extra (`pip install kzn-recsys[spark]`) so the base install
   stays lean.

3. **API parity by naming.** `kzn_recsys.spark` exposes `build_and_train`,
   `load_model`, the metric functions, and split functions with the same
   signatures as the native versions wherever sensible. Data enters as
   Spark DataFrames (or paths); the model object is `SparkEaseModel` with
   the same `predict(interactions, features, top_k)` shape as `FeaseModel`.
   Code written against one surface ports to the other by changing an
   import.

## Components

### 1. EASE core (`ease_core.py`, no Spark)

A direct port of `src/model.rs`, the only component that must hit tolerance
parity, so it mirrors the Rust math operation-for-operation.

- **Inputs:** three `scipy.sparse` CSR matrices — `X` (N×M users×items),
  `U` (N×K users×user-features), `T` (L×M item-features×items) — plus
  `alpha`, `beta`, `lambda_`, `meta_weight`. Built by `dataframes.py`;
  Spark is not imported here.
- **Gram blocks** (`model.rs:124-145`): `w = meta_weight if meta_weight > 0
  else 1.0`; `G11 = XᵀX + w·α²·TᵀT`, `G12 = β·XᵀU`, `G21 = β·UᵀX`,
  `G22 = β²·UᵀU`. Assembled into a dense `(M+K)²` array via
  `scipy.sparse.bmat(...).toarray()`.
- **Solve** (`model.rs:190-224`): `P = inv(G + λI)`, then the closed-form
  `S[i,j] = -P[i,j]/P[j,j]`, `S[j,j] = 0` — vectorized as
  `S = -P / diag(P)[None, :]` with the diagonal zeroed and the same
  `|P_jj| > 1e-12` guard.
- **Predict** (`model.rs:341-378`): `z = [x | β·u]`,
  `scores = (S @ z)[:M]`. `predict_similar_items` queries the item-item
  block the same way.
- **Sparsity pruning** (`model.rs:233-248`): zero S entries with
  `|value| < threshold`.

**Layout caveat.** Rust stores S **column-major**, and the FEAS payload is
column-major `f64` (`serialization.rs:79`). NumPy is row-major by default.
The codec and the core must agree on layout. Keep S as an explicit
Fortran-order array on load and assert shape/order in tests rather than
allowing a silent transpose to corrupt scores.

### 2. FEAS codec (`feas_codec.py`) — interop crux

Both-directions model movement requires load- and save-compatibility with
the Rust binary format. bincode 1.3 with default config is a fixed,
documented encoding: little-endian, fixint (u64 lengths), no field names.
The FEAS payload is `b"FEAS"` (4 bytes) + bincode of the flat
`SerializedModel` struct (`serialization.rs:80-108`), whose field order is
fixed and mirrored field-for-field:

- A struct-driven reader/writer decodes each field in declaration order:
  scalars as LE, `Vec<f64>` as `u64` length + payload, `Vec<String>` and
  `Vec<(String, usize)>` likewise, `Option<WeightingConfig>` as a 1-byte
  tag + body.
- **Version handling:** sniff `version` first; dispatch to the v1 struct
  (no `weighting_config`) or v2, exactly as `into_v2()` does
  (`serialization.rs:48-73`).
- **Round-trip is the test, not a claim:** a Rust-saved `.fease` → Python
  load → Python save → byte-compare against the original is the strongest
  parity check; it runs in CI wherever the wheel can generate fixtures.

This is the one component where tolerance does not apply — bytes must
match — and the highest-risk piece, so the plan front-loads it.

### 3. Spark adapters (`gram.py`, `dataframes.py`, `model.py`)

`gram.py` offers two strategies that feed the same `ease_core` solve:

- **Collect-to-driver** (Phase 1): build index mappings with Spark, collect
  interactions/features to the driver, hand CSR to `ease_core`. Ceiling is
  driver memory — fine for catalogs up to ~10⁴–10⁵ items.
- **Distributed Gram** (Phase 2): compute the four `ZᵀZ` blocks as Spark
  aggregations (self-joins on long-format triples), collect only the
  `(M+K)²` dense Gram to the driver for the solve. The dense inverse is
  inherently driver-side — no distributed solve exists — but the Gram
  **accumulation** scales with interaction count, which is the part that
  blows up. The driver ceiling becomes `(M+K)²`, independent of N,
  mirroring the Rust core's memory argument.

`dataframes.py` converts long-format interaction/feature DataFrames to
string↔index mappings and sparse inputs, and applies the advanced-weighting
transforms (event weights = join-and-multiply, decay = `exp()` column,
IPS = popularity reweight, mirroring `weighting.rs`).

`model.py` is the `SparkEaseModel` facade: `predict`, `predict_batch`,
`evaluate`, `save`, `load`.

### 4. Evaluation & splits (`metrics.py`, `splits.py`)

- **`splits.py`:** `random` / `temporal` / `leave_k_out` as DataFrame ops,
  sorted-key-before-RNG for the same determinism guarantee the Rust splits
  make.
- **`metrics.py`:** pure-function ports of `metrics.rs` (`precision_at_k`,
  `recall_at_k`, `ndcg_at_k`, `mean_average_precision`, `coverage`,
  `hit_rate_at_k`); the evaluation harness scores test users through
  `SparkEaseModel.predict`.

### 5. Tuning (`tuning.py`)

Grid/random search with user k-fold CV, optimizing NDCG@k. Spark provides
trial parallelism — `spark.sparkContext.parallelize(param_grid)`, training
each fold-config combination as a task, mirroring the rayon `(params×fold)`
flat product (ADR-0002) — for the collect-to-driver path where each task is
self-contained.

## Testing & parity strategy

- **Parity fixtures:** a small synthetic dataset trained by the Rust core
  (behind a CI marker that requires the wheel) produces a golden S-matrix +
  predictions; the PySpark core must match top-K rankings exactly and
  scores within `1e-5` relative.
- **Spark-free fast tests:** `ease_core` and `feas_codec` are tested with
  plain pytest, no `SparkSession` — the bulk of coverage, and fast.
- **Spark tests:** a session-scoped `local[*]` fixture; both Gram strategies
  must agree with each other and with `ease_core` on the same data.
- **Codec round-trip:** the byte-exact Rust↔Python test from Component 2.

## Phasing

The phases become the spine of the implementation plan:

1. `ease_core` + collect-to-driver Spark path + FEAS codec — full parity,
   single-node Spark.
2. Distributed Gram strategy.
3. Evaluation + splits.
4. Tuning.

## Out of scope

- SASRec and Two-Tower PySpark implementations.
- Any deep-learning framework dependency.
- Distributed dense matrix inversion (no such primitive exists; the solve
  stays driver-side by design).
