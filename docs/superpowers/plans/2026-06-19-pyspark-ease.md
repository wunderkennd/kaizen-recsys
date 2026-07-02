# PySpark EASE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pure-Python/PySpark implementation of the EASE recommender — train, predict, advanced weighting, evaluation, splits, and tuning — runnable where the compiled `kzn_recsys._native` wheel cannot be installed, with byte-compatible FEAS model interop in both directions.

**Architecture:** A Spark-free NumPy/SciPy math core (`ease_core.py`) mirrors `src/model.rs` operation-for-operation. A pure-Python bincode codec (`feas_codec.py`) reads/writes the Rust FEAS binary format. Spark is confined to IO, index-mapping construction, advanced-weighting transforms, and (Phase 2) distributed Gram accumulation. A `SparkEaseModel` facade ties them together. Everything lives in a new `kzn_recsys.spark` subpackage that imports nothing from `_native`.

**Tech Stack:** Python 3.8+, NumPy, SciPy (`scipy.sparse`), PySpark; pytest for tests. No PyTorch, no Spark MLlib distributed matrices.

**Source of truth (Rust):** `src/model.rs` (EASE math), `src/weighting.rs` (event/decay/IPS), `src/metrics.rs` (ranking metrics), `src/data_pipeline.rs` (mappings + matrix build), `src/serialization.rs` (FEAS format), `src/evaluation.rs` (splits).

---

## Key parity facts (read before starting)

These were established by reading the Rust source. They constrain the implementation:

1. **S matrix is column-major.** `RustFeaseModel.s_matrix` is `nalgebra::DMatrix` (column-major), and the FEAS payload stores `s_data` as a flat column-major `Vec<f64>` reconstructed via `DMatrix::from_vec(nrows, ncols, s_data)` (`serialization.rs:189`). NumPy is row-major by default. **Always treat `s_data` as Fortran-order** when converting to/from a 2-D array.

2. **Index assignment is internal and self-consistent, not cross-impl identical.** Rust builds string→index mappings in *first-seen* order while scanning files (`data_pipeline.rs:224-243`). The FEAS file persists the full mappings *alongside* S. Therefore:
   - PySpark does **not** need to reproduce Rust's exact integer assignment. It only needs each saved `(mappings, S)` pair to be internally consistent.
   - Both-directions interop works because the mapping travels with the matrix.
   - Prediction parity is compared **by item string-id**, so index differences wash out (modulo score ties — see fact 5).

3. **Matrix shapes:** `X` is `(N users × M items)`, `U` is `(N users × K user-features)`, `T` is `(L item-features × M items)` — note T is transposed relative to the natural `(item, feature)` triplet build (`data_pipeline.rs:200-204`).

4. **Weighting order is event → decay → IPS** (`data_pipeline.rs:126-176`), applied to interaction values before they enter X. Formulas (`weighting.rs`): event = multiply by per-type weight (unknown types unchanged); decay = `value * exp(-decay_rate * days_ago)`; IPS propensity = `(item_count / max_count) ^ ips_alpha`, reweighted `value / propensity` (skip if `propensity <= 1e-12`).

5. **Splits are NOT byte-reproducible against Rust.** Rust uses `StdRng::seed_from_u64` + slice `shuffle` (`evaluation.rs:126-144`). Python cannot reproduce that RNG stream. Split parity is therefore *structural* (same hold-out semantics, deterministic given a Python seed), **not** row-identical to Rust. Evaluation parity tests must split **once** and feed the identical train/test frames to both models.

6. **bincode 1.3 default config** = little-endian, fixed-width integers, `u64` lengths, no field names. `Option<T>` = 1 tag byte (`0`/`1`) then the value if present. `Vec<T>`/`String` = `u64` length prefix then elements/UTF-8 bytes. `Vec<(String, usize)>` = `u64` count then each pair. `usize` serializes as `u64`.

---

## File structure

```
kzn_recsys/
  __init__.py                 # MODIFY: wrap _native imports in try/except -> _HAS_NATIVE
  spark/
    __init__.py               # CREATE: public API surface
    ease_core.py              # CREATE: pure NumPy/SciPy EASE math (mirrors model.rs)
    feas_codec.py             # CREATE: bincode FEAS reader/writer (mirrors serialization.rs)
    dataframes.py             # CREATE: DataFrame -> mappings + CSR; weighting transforms
    gram.py                   # CREATE: gram_collect (P1) + gram_distributed (P2)
    model.py                  # CREATE: SparkEaseModel facade, build_and_train, load_model
    metrics.py                # CREATE: ranking metrics (mirrors metrics.rs)
    splits.py                 # CREATE: random/temporal/leave_k_out on DataFrames
    tuning.py                 # CREATE: grid_search / random_search w/ k-fold CV
tests/
  spark/
    conftest.py               # CREATE: session-scoped local[*] SparkSession + markers
    test_ease_core.py         # CREATE
    test_feas_codec.py        # CREATE
    test_dataframes.py        # CREATE
    test_gram_distributed.py  # CREATE (P2)
    test_model.py             # CREATE
    test_metrics.py           # CREATE (P3)
    test_splits.py            # CREATE (P3)
    test_tuning.py            # CREATE (P4)
    test_parity.py            # CREATE: cross-checks vs native wheel (marker: parity)
pyproject.toml                # MODIFY: [spark] extra + pytest markers
```

---

## Task 0: Test infrastructure & dependencies

**Files:**
- Modify: `pyproject.toml:19-26` (optional-dependencies), add `[tool.pytest.ini_options]`
- Create: `tests/spark/conftest.py`
- Create: `tests/spark/__init__.py` (empty)

- **Step 1: Add the `spark` extra and pytest markers to `pyproject.toml`**

Add a `spark` extra under `[project.optional-dependencies]` and a pytest config block. Final state of the two sections:

```toml
[project.optional-dependencies]
onnx = [
    "numpy>=1.24",
    "onnx>=1.16",
    "onnxruntime>=1.18",
    "mlflow>=2.12",
    "sympy>=1.12",  # required by onnxruntime.quantization (int8 path)
]
spark = [
    "numpy>=1.24",
    "scipy>=1.10",
    "pyspark>=3.4",
]

[tool.pytest.ini_options]
markers = [
    "spark: test requires a SparkSession (slow; needs pyspark)",
    "parity: test cross-checks the PySpark impl against the native kzn_recsys._native wheel",
]
```

- **Step 2: Create the empty test package marker**

Create `tests/spark/__init__.py` with no content (makes `tests/spark` importable).

- **Step 3: Create the Spark session fixture**

Create `tests/spark/conftest.py`:

```python
"""Shared fixtures for PySpark EASE tests."""
import pytest


@pytest.fixture(scope="session")
def spark():
    """Session-scoped local SparkSession. Skips the test if pyspark is absent."""
    pyspark = pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("kzn_recsys-spark-tests")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()
```

- **Step 4: Install the extra into the dev venv**

Run: `.venv/bin/python -m pip install -e '.[spark]'`
Expected: installs numpy, scipy, pyspark without error. (The `-e` editable install also makes `kzn_recsys.spark` importable as it is created.)

- **Step 5: Verify pytest discovers the markers**

Run: `.venv/bin/python -m pytest tests/spark/ --markers | grep -E "spark|parity"`
Expected: both `@pytest.mark.spark` and `@pytest.mark.parity` are listed.

- **Step 6: Commit**

```bash
git add pyproject.toml tests/spark/__init__.py tests/spark/conftest.py
git commit -m "test(spark): add spark extra, pytest markers, and SparkSession fixture"
```

---

# Phase 1 — Core math, codec, driver Spark path

The parity spine. After Phase 1 you can train EASE on Spark data (single-node collect), predict, and round-trip FEAS files with the Rust core.

## Task 1: EASE params dataclass

**Files:**
- Create: `kzn_recsys/spark/__init__.py`
- Create: `kzn_recsys/spark/ease_core.py`
- Test: `tests/spark/test_ease_core.py`

- **Step 1: Write the failing test**

Create `tests/spark/test_ease_core.py`:

```python
import numpy as np
import pytest

from kzn_recsys.spark.ease_core import EaseParams


def test_ease_params_defaults():
    p = EaseParams()
    assert p.alpha == 1.0
    assert p.beta == 1.0
    assert p.lambda_ == 150.0
    assert p.meta_weight == 0.0


def test_ease_params_explicit():
    p = EaseParams(alpha=2.0, beta=0.5, lambda_=100.0, meta_weight=1.0)
    assert (p.alpha, p.beta, p.lambda_, p.meta_weight) == (2.0, 0.5, 100.0, 1.0)
```

- **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/spark/test_ease_core.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kzn_recsys.spark'`.

- **Step 3: Create the package init and params**

Create `kzn_recsys/spark/__init__.py`:

```python
"""Pure-Python / PySpark EASE implementation.

Imports nothing from kzn_recsys._native, so it works in environments where
the compiled extension cannot be installed.
"""
```

Create `kzn_recsys/spark/ease_core.py`:

```python
"""Pure NumPy/SciPy EASE math. Mirrors src/model.rs.

No pyspark import here — this module is Spark-free and fast to test.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EaseParams:
    """EASE hyperparameters. Defaults match the Rust core / fease_wrapper."""
    alpha: float = 1.0       # item-feature weight
    beta: float = 1.0        # user-feature weight
    lambda_: float = 150.0   # L2 regularization
    meta_weight: float = 0.0  # diagonal metadata weighting; 0 => treated as 1.0
```

- **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/spark/test_ease_core.py -v`
Expected: PASS (2 passed).

- **Step 5: Commit**

```bash
git add kzn_recsys/spark/__init__.py kzn_recsys/spark/ease_core.py tests/spark/test_ease_core.py
git commit -m "feat(spark): EaseParams dataclass"
```

## Task 2: EASE training (Gram blocks + closed-form solve)

**Files:**
- Modify: `kzn_recsys/spark/ease_core.py`
- Test: `tests/spark/test_ease_core.py`

- **Step 1: Write the failing test**

Append to `tests/spark/test_ease_core.py`:

```python
import scipy.sparse as sp

from kzn_recsys.spark.ease_core import train_ease


def _toy_inputs():
    # 3 users, 3 items, no features (K=0, L=0)
    # User 0: items 0,1 ; User 1: items 1,2 ; User 2: items 0,2
    X = sp.csr_matrix(
        np.array([[1.0, 1.0, 0.0],
                  [0.0, 1.0, 1.0],
                  [1.0, 0.0, 1.0]])
    )
    U = sp.csr_matrix((3, 0))   # N x K, K=0
    T = sp.csr_matrix((0, 3))   # L x M, L=0
    return X, U, T


def test_train_shapes_and_zero_diagonal():
    X, U, T = _toy_inputs()
    S = train_ease(X, U, T, EaseParams(lambda_=10.0))
    # (M+K) x (M+K) == 3 x 3
    assert S.shape == (3, 3)
    # zero diagonal constraint
    assert np.allclose(np.diag(S), 0.0)
    # column-major storage
    assert S.flags["F_CONTIGUOUS"]


def test_train_matches_direct_formula():
    X, U, T = _toy_inputs()
    lam = 10.0
    S = train_ease(X, U, T, EaseParams(alpha=1.0, beta=1.0, lambda_=lam))
    # Reference: G = X^T X (no features), P = inv(G + lam I), B = -P / diag(P), zero diag
    G = (X.T @ X).toarray()
    P = np.linalg.inv(G + lam * np.eye(3))
    B = -P / np.diag(P)[None, :]
    np.fill_diagonal(B, 0.0)
    assert np.allclose(S, B, atol=1e-9)
```

- **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/spark/test_ease_core.py -k train -v`
Expected: FAIL — `ImportError: cannot import name 'train_ease'`.

- **Step 3: Implement `train_ease`**

Append to `kzn_recsys/spark/ease_core.py`:

```python
import numpy as np
import scipy.sparse as sp


def train_ease(X, U, T, params: EaseParams) -> np.ndarray:
    """Train EASE, returning the S matrix as a Fortran-order (M+K)x(M+K) array.

    Mirrors RustFeaseModel::train (src/model.rs:87-228).

    Args:
        X: (N x M) users x items, scipy CSR
        U: (N x K) users x user-features, scipy CSR
        T: (L x M) item-features x items, scipy CSR
        params: EaseParams
    """
    M = X.shape[1]
    K = U.shape[1]
    total = M + K

    w = params.meta_weight if params.meta_weight > 0.0 else 1.0
    a, b = params.alpha, params.beta

    # Gram blocks (model.rs:130-145)
    XtX = (X.T @ X)                       # M x M
    TtT = (T.T @ T)                       # M x M
    G11 = (XtX + w * a * a * TtT).toarray()
    G12 = (b * (X.T @ U)).toarray()       # M x K
    G21 = (b * (U.T @ X)).toarray()       # K x M
    G22 = (b * b * (U.T @ U)).toarray()   # K x K

    # Assemble dense G (model.rs:147-187)
    G = np.zeros((total, total), dtype=np.float64)
    G[:M, :M] = G11
    if K > 0:
        G[:M, M:] = G12
        G[M:, :M] = G21
        G[M:, M:] = G22

    # P = inv(G + lambda I)  (model.rs:190-200)
    G.flat[:: total + 1] += params.lambda_  # add lambda to diagonal in place
    P = np.linalg.inv(G)

    # S[i,j] = -P[i,j] / P[j,j], S[j,j] = 0  (model.rs:202-224)
    p_jj = np.diag(P).copy()
    inv = np.where(np.abs(p_jj) > 1e-12, -1.0 / p_jj, 0.0)
    S = P * inv[None, :]
    np.fill_diagonal(S, 0.0)

    # Column-major to match nalgebra / FEAS layout (parity fact 1).
    return np.asfortranarray(S)
```

- **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/spark/test_ease_core.py -v`
Expected: PASS (4 passed).

- **Step 5: Commit**

```bash
git add kzn_recsys/spark/ease_core.py tests/spark/test_ease_core.py
git commit -m "feat(spark): EASE training (Gram blocks + closed-form S)"
```

## Task 3: Predict, predict-similar, and prune

**Files:**
- Modify: `kzn_recsys/spark/ease_core.py`
- Test: `tests/spark/test_ease_core.py`

- **Step 1: Write the failing test**

Append to `tests/spark/test_ease_core.py`:

```python
from kzn_recsys.spark.ease_core import predict_scores, predict_similar_items, prune_sparse


def test_predict_scores_against_S_at_z():
    X, U, T = _toy_inputs()
    S = train_ease(X, U, T, EaseParams(lambda_=10.0))
    # User with items 0 and 1
    interactions = [(0, 1.0), (1, 1.0)]
    scores = predict_scores(S, num_items=3, num_user_features=0,
                            interactions=interactions, features=[], beta=1.0)
    # Reference: z = [1,1,0], scores = (S @ z)[:3]
    z = np.array([1.0, 1.0, 0.0])
    assert np.allclose(scores, (S @ z)[:3], atol=1e-12)
    assert scores.shape == (3,)


def test_predict_scores_applies_beta_to_features():
    # 1 item, 1 user-feature: total dim 2
    S = np.asfortranarray(np.array([[0.0, 0.5], [0.7, 0.0]]))
    scores = predict_scores(S, num_items=1, num_user_features=1,
                            interactions=[(0, 2.0)], features=[(0, 3.0)], beta=0.5)
    # z = [2.0, 0.5*3.0] = [2.0, 1.5]; score_item0 = 0.0*2.0 + 0.5*1.5 = 0.75
    assert np.allclose(scores, [0.75], atol=1e-12)


def test_predict_similar_items_excludes_self_and_sorts():
    S = np.asfortranarray(np.array([
        [0.0, 0.9, 0.1],
        [0.9, 0.0, 0.5],
        [0.1, 0.5, 0.0],
    ]))
    out = predict_similar_items(S, item_idx=0, num_items=3, top_k=2)
    assert out[0][0] == 1  # highest column-0 score, excluding self
    assert out[1][0] == 2
    assert all(idx != 0 for idx, _ in out)


def test_prune_sparse_zeros_small_entries():
    S = np.asfortranarray(np.array([[0.0, 0.001], [0.5, 0.0]]))
    prune_sparse(S, threshold=0.01)
    assert S[0, 1] == 0.0
    assert S[1, 0] == 0.5
```

- **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/spark/test_ease_core.py -k "predict or prune" -v`
Expected: FAIL — `ImportError: cannot import name 'predict_scores'`.

- **Step 3: Implement the three functions**

Append to `kzn_recsys/spark/ease_core.py`:

```python
def predict_scores(S, num_items, num_user_features, interactions, features, beta):
    """Score all items for one user. Mirrors RustFeaseModel::predict (model.rs:341-378).

    interactions: list of (item_idx, value); features: list of (feature_idx, value).
    Returns a length-`num_items` float64 array.
    """
    total = num_items + num_user_features
    z = np.zeros(total, dtype=np.float64)
    for item_idx, val in interactions:
        if 0 <= item_idx < num_items:
            z[item_idx] = val
    for feat_idx, val in features:
        if 0 <= feat_idx < num_user_features:
            z[num_items + feat_idx] = val * beta
    return (S @ z)[:num_items]


def predict_similar_items(S, item_idx, num_items, top_k):
    """Item-item similarity from the S item block. Mirrors model.rs:390+.

    Returns up to top_k (item_index, score) pairs sorted by descending score,
    excluding item_idx itself.
    """
    col = np.asarray(S[:num_items, item_idx]).ravel()
    order = np.argsort(-col, kind="stable")
    out = []
    for j in order:
        if int(j) == item_idx:
            continue
        out.append((int(j), float(col[j])))
        if len(out) == top_k:
            break
    return out


def prune_sparse(S, threshold) -> None:
    """Zero entries with |value| < threshold, in place. Mirrors model.rs:233-248."""
    S[np.abs(S) < threshold] = 0.0
```

- **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/spark/test_ease_core.py -v`
Expected: PASS (8 passed).

- **Step 5: Commit**

```bash
git add kzn_recsys/spark/ease_core.py tests/spark/test_ease_core.py
git commit -m "feat(spark): EASE predict, predict-similar, prune"
```

## Task 4: bincode primitives for FEAS

**Files:**
- Create: `kzn_recsys/spark/feas_codec.py`
- Test: `tests/spark/test_feas_codec.py`

- **Step 1: Write the failing test**

Create `tests/spark/test_feas_codec.py`:

```python
import io

from kzn_recsys.spark import feas_codec as fc


def test_u64_roundtrip():
    buf = io.BytesIO()
    fc._write_u64(buf, 0)
    fc._write_u64(buf, 1)
    fc._write_u64(buf, 2**40 + 7)
    buf.seek(0)
    assert fc._read_u64(buf) == 0
    assert fc._read_u64(buf) == 1
    assert fc._read_u64(buf) == 2**40 + 7


def test_u64_is_little_endian_8_bytes():
    buf = io.BytesIO()
    fc._write_u64(buf, 1)
    assert buf.getvalue() == b"\x01\x00\x00\x00\x00\x00\x00\x00"


def test_f64_roundtrip():
    buf = io.BytesIO()
    for v in (0.0, -1.5, 3.141592653589793):
        fc._write_f64(buf, v)
    buf.seek(0)
    assert fc._read_f64(buf) == 0.0
    assert fc._read_f64(buf) == -1.5
    assert fc._read_f64(buf) == 3.141592653589793


def test_string_roundtrip_length_prefixed():
    buf = io.BytesIO()
    fc._write_string(buf, "héllo")  # multi-byte UTF-8
    buf.seek(0)
    # u64 length prefix == UTF-8 byte length (6), then bytes
    assert fc._read_string(buf) == "héllo"


def test_vec_f64_roundtrip():
    buf = io.BytesIO()
    fc._write_vec_f64(buf, [1.0, 2.0, 3.0])
    buf.seek(0)
    assert fc._read_vec_f64(buf) == [1.0, 2.0, 3.0]


def test_vec_string_and_pairs_roundtrip():
    buf = io.BytesIO()
    fc._write_vec_string(buf, ["a", "bb"])
    fc._write_vec_pair_string_usize(buf, [("a", 0), ("bb", 1)])
    buf.seek(0)
    assert fc._read_vec_string(buf) == ["a", "bb"]
    assert fc._read_vec_pair_string_usize(buf) == [("a", 0), ("bb", 1)]
```

- **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/spark/test_feas_codec.py -v`
Expected: FAIL — `ModuleNotFoundError` / missing attributes.

- **Step 3: Implement the primitives**

Create `kzn_recsys/spark/feas_codec.py`:

```python
"""Pure-Python reader/writer for the Rust FEAS model format.

Mirrors src/serialization.rs. bincode 1.3 default config: little-endian,
fixed-width ints, u64 length prefixes, no field names. usize serializes as u64.
"""
from __future__ import annotations

import struct

_U64 = struct.Struct("<Q")
_F64 = struct.Struct("<d")


def _read_exact(buf, n: int) -> bytes:
    data = buf.read(n)
    if len(data) != n:
        raise EOFError(f"expected {n} bytes, got {len(data)}")
    return data


def _write_u64(buf, v: int) -> None:
    buf.write(_U64.pack(v))


def _read_u64(buf) -> int:
    return _U64.unpack(_read_exact(buf, 8))[0]


def _write_f64(buf, v: float) -> None:
    buf.write(_F64.pack(v))


def _read_f64(buf) -> float:
    return _F64.unpack(_read_exact(buf, 8))[0]


def _write_string(buf, s: str) -> None:
    raw = s.encode("utf-8")
    _write_u64(buf, len(raw))
    buf.write(raw)


def _read_string(buf) -> str:
    n = _read_u64(buf)
    return _read_exact(buf, n).decode("utf-8")


def _write_vec_f64(buf, xs) -> None:
    _write_u64(buf, len(xs))
    for x in xs:
        _write_f64(buf, float(x))


def _read_vec_f64(buf) -> list:
    n = _read_u64(buf)
    return [_read_f64(buf) for _ in range(n)]


def _write_vec_string(buf, xs) -> None:
    _write_u64(buf, len(xs))
    for s in xs:
        _write_string(buf, s)


def _read_vec_string(buf) -> list:
    n = _read_u64(buf)
    return [_read_string(buf) for _ in range(n)]


def _write_vec_pair_string_usize(buf, pairs) -> None:
    _write_u64(buf, len(pairs))
    for s, i in pairs:
        _write_string(buf, s)
        _write_u64(buf, i)


def _read_vec_pair_string_usize(buf) -> list:
    n = _read_u64(buf)
    return [(_read_string(buf), _read_u64(buf)) for _ in range(n)]
```

- **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/spark/test_feas_codec.py -v`
Expected: PASS (6 passed).

- **Step 5: Commit**

```bash
git add kzn_recsys/spark/feas_codec.py tests/spark/test_feas_codec.py
git commit -m "feat(spark): bincode primitives for FEAS codec"
```

## Task 5: FEAS artifact read/write (v1 + v2)

**Files:**
- Modify: `kzn_recsys/spark/feas_codec.py`
- Test: `tests/spark/test_feas_codec.py`

The struct field order mirrors `SerializedModel` (`serialization.rs:80-108`) exactly:
`version, s_nrows, s_ncols, s_data, num_items, num_user_features, num_item_features, alpha, beta, lambda_, meta_weight, user_to_idx, idx_to_user, item_to_idx, idx_to_item, user_feature_to_idx, idx_to_user_feature, item_feature_to_idx, idx_to_item_feature, [weighting_config (v2 only)]`.
`WeightingConfig` field order (`weighting.rs:14-37`): `event_weights: Option<HashMap<String,f64>>, decay_rate: f64, ips_alpha: f64, sparsity_threshold: f64`.

- **Step 1: Write the failing test**

Append to `tests/spark/test_feas_codec.py`:

```python
import numpy as np

from kzn_recsys.spark.feas_codec import FeaseArtifact, WeightingConfig, write_feas, read_feas


def _toy_artifact(weighting=None, version=2):
    return FeaseArtifact(
        version=version,
        s_nrows=2,
        s_ncols=2,
        # column-major flat: column 0 then column 1
        s_data=np.asfortranarray(np.array([[0.0, 0.5], [0.7, 0.0]])),
        num_items=2,
        num_user_features=0,
        num_item_features=0,
        alpha=1.0, beta=1.0, lambda_=150.0, meta_weight=0.0,
        user_to_idx=[("u0", 0)], idx_to_user=["u0"],
        item_to_idx=[("i0", 0), ("i1", 1)], idx_to_item=["i0", "i1"],
        user_feature_to_idx=[], idx_to_user_feature=[],
        item_feature_to_idx=[], idx_to_item_feature=[],
        weighting_config=weighting,
    )


def test_write_then_read_roundtrip_v2(tmp_path):
    art = _toy_artifact()
    path = tmp_path / "m.fease"
    write_feas(art, str(path))
    back = read_feas(str(path))
    assert back.version == 2
    assert back.s_nrows == 2 and back.s_ncols == 2
    assert np.allclose(back.s_data, art.s_data)
    assert back.s_data.flags["F_CONTIGUOUS"]
    assert back.item_to_idx == [("i0", 0), ("i1", 1)]
    assert back.weighting_config is None


def test_magic_bytes_present(tmp_path):
    path = tmp_path / "m.fease"
    write_feas(_toy_artifact(), str(path))
    with open(path, "rb") as fh:
        assert fh.read(4) == b"FEAS"


def test_weighting_config_roundtrip(tmp_path):
    wc = WeightingConfig(event_weights={"click": 1.0, "purchase": 5.0},
                         decay_rate=0.01, ips_alpha=0.5, sparsity_threshold=0.0)
    path = tmp_path / "m.fease"
    write_feas(_toy_artifact(weighting=wc), str(path))
    back = read_feas(str(path))
    assert back.weighting_config.decay_rate == 0.01
    assert back.weighting_config.ips_alpha == 0.5
    assert back.weighting_config.event_weights == {"click": 1.0, "purchase": 5.0}


def test_byte_exact_reencode(tmp_path):
    """Decode-then-encode must reproduce identical bytes (preserves map order)."""
    wc = WeightingConfig(event_weights={"a": 1.0, "b": 2.0},
                         decay_rate=0.0, ips_alpha=0.0, sparsity_threshold=0.0)
    p1 = tmp_path / "a.fease"
    write_feas(_toy_artifact(weighting=wc), str(p1))
    original = p1.read_bytes()
    back = read_feas(str(p1))
    p2 = tmp_path / "b.fease"
    write_feas(back, str(p2))
    assert p2.read_bytes() == original


def test_v1_has_no_weighting_config(tmp_path):
    art = _toy_artifact(version=1)
    path = tmp_path / "m.fease"
    write_feas(art, str(path))
    back = read_feas(str(path))
    assert back.version == 1
    assert back.weighting_config is None
```

- **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/spark/test_feas_codec.py -k "roundtrip or magic or weighting or byte or v1" -v`
Expected: FAIL — `cannot import name 'FeaseArtifact'`.

- **Step 3: Implement the artifact, codec, and HashMap helpers**

Append to `kzn_recsys/spark/feas_codec.py`:

```python
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_MAGIC = b"FEAS"
_FORMAT_VERSION = 2


@dataclass
class WeightingConfig:
    # event_weights kept as an insertion-ordered dict so decode->encode is byte-stable.
    event_weights: Optional[dict] = None
    decay_rate: float = 0.0
    ips_alpha: float = 0.0
    sparsity_threshold: float = 0.0


@dataclass
class FeaseArtifact:
    version: int
    s_nrows: int
    s_ncols: int
    s_data: np.ndarray  # 2-D Fortran-order (s_nrows x s_ncols), or flat handled in write
    num_items: int
    num_user_features: int
    num_item_features: int
    alpha: float
    beta: float
    lambda_: float
    meta_weight: float
    user_to_idx: list
    idx_to_user: list
    item_to_idx: list
    idx_to_item: list
    user_feature_to_idx: list
    idx_to_user_feature: list
    item_feature_to_idx: list
    idx_to_item_feature: list
    weighting_config: Optional[WeightingConfig] = None


def _write_map_string_f64(buf, m: dict) -> None:
    # bincode HashMap<String,f64> = u64 count then entries in iteration order.
    _write_u64(buf, len(m))
    for k, v in m.items():
        _write_string(buf, k)
        _write_f64(buf, float(v))


def _read_map_string_f64(buf) -> dict:
    n = _read_u64(buf)
    out = {}
    for _ in range(n):
        k = _read_string(buf)
        out[k] = _read_f64(buf)
    return out


def _write_weighting(buf, wc: Optional[WeightingConfig]) -> None:
    # Field is Option<WeightingConfig>: 1 tag byte then body if present.
    if wc is None:
        buf.write(b"\x00")
        return
    buf.write(b"\x01")
    # event_weights: Option<HashMap<String,f64>>
    if wc.event_weights is None:
        buf.write(b"\x00")
    else:
        buf.write(b"\x01")
        _write_map_string_f64(buf, wc.event_weights)
    _write_f64(buf, wc.decay_rate)
    _write_f64(buf, wc.ips_alpha)
    _write_f64(buf, wc.sparsity_threshold)


def _read_weighting(buf) -> Optional[WeightingConfig]:
    tag = _read_exact(buf, 1)
    if tag == b"\x00":
        return None
    ew_tag = _read_exact(buf, 1)
    event_weights = _read_map_string_f64(buf) if ew_tag == b"\x01" else None
    decay_rate = _read_f64(buf)
    ips_alpha = _read_f64(buf)
    sparsity_threshold = _read_f64(buf)
    return WeightingConfig(event_weights, decay_rate, ips_alpha, sparsity_threshold)


def _s_data_flat_colmajor(s_data, nrows, ncols) -> list:
    arr = np.asarray(s_data, dtype=np.float64)
    if arr.ndim == 2:
        arr = np.asfortranarray(arr).reshape(-1, order="F")
    return arr.tolist()


def write_feas(artifact: FeaseArtifact, path: str) -> None:
    import io
    buf = io.BytesIO()
    a = artifact
    _write_u64(buf, a.version)
    _write_u64(buf, a.s_nrows)
    _write_u64(buf, a.s_ncols)
    _write_vec_f64(buf, _s_data_flat_colmajor(a.s_data, a.s_nrows, a.s_ncols))
    _write_u64(buf, a.num_items)
    _write_u64(buf, a.num_user_features)
    _write_u64(buf, a.num_item_features)
    _write_f64(buf, a.alpha)
    _write_f64(buf, a.beta)
    _write_f64(buf, a.lambda_)
    _write_f64(buf, a.meta_weight)
    _write_vec_pair_string_usize(buf, a.user_to_idx)
    _write_vec_string(buf, a.idx_to_user)
    _write_vec_pair_string_usize(buf, a.item_to_idx)
    _write_vec_string(buf, a.idx_to_item)
    _write_vec_pair_string_usize(buf, a.user_feature_to_idx)
    _write_vec_string(buf, a.idx_to_user_feature)
    _write_vec_pair_string_usize(buf, a.item_feature_to_idx)
    _write_vec_string(buf, a.idx_to_item_feature)
    if a.version >= 2:
        _write_weighting(buf, a.weighting_config)
    with open(path, "wb") as fh:
        fh.write(_MAGIC)
        fh.write(buf.getvalue())


def read_feas(path: str) -> FeaseArtifact:
    import io
    with open(path, "rb") as fh:
        blob = fh.read()
    if blob[:4] != _MAGIC:
        raise ValueError(f"not a FEAS file: magic={blob[:4]!r}")
    buf = io.BytesIO(blob[4:])
    version = _read_u64(buf)
    if version not in (1, _FORMAT_VERSION):
        raise ValueError(f"unsupported FEAS version: {version}")
    s_nrows = _read_u64(buf)
    s_ncols = _read_u64(buf)
    s_flat = _read_vec_f64(buf)
    num_items = _read_u64(buf)
    num_user_features = _read_u64(buf)
    num_item_features = _read_u64(buf)
    alpha = _read_f64(buf)
    beta = _read_f64(buf)
    lambda_ = _read_f64(buf)
    meta_weight = _read_f64(buf)
    user_to_idx = _read_vec_pair_string_usize(buf)
    idx_to_user = _read_vec_string(buf)
    item_to_idx = _read_vec_pair_string_usize(buf)
    idx_to_item = _read_vec_string(buf)
    user_feature_to_idx = _read_vec_pair_string_usize(buf)
    idx_to_user_feature = _read_vec_string(buf)
    item_feature_to_idx = _read_vec_pair_string_usize(buf)
    idx_to_item_feature = _read_vec_string(buf)
    weighting_config = _read_weighting(buf) if version >= 2 else None

    s_data = np.reshape(np.asarray(s_flat, dtype=np.float64),
                        (s_nrows, s_ncols), order="F")
    return FeaseArtifact(
        version=version, s_nrows=s_nrows, s_ncols=s_ncols, s_data=np.asfortranarray(s_data),
        num_items=num_items, num_user_features=num_user_features,
        num_item_features=num_item_features, alpha=alpha, beta=beta, lambda_=lambda_,
        meta_weight=meta_weight, user_to_idx=user_to_idx, idx_to_user=idx_to_user,
        item_to_idx=item_to_idx, idx_to_item=idx_to_item,
        user_feature_to_idx=user_feature_to_idx, idx_to_user_feature=idx_to_user_feature,
        item_feature_to_idx=item_feature_to_idx, idx_to_item_feature=idx_to_item_feature,
        weighting_config=weighting_config,
    )
```

- **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/spark/test_feas_codec.py -v`
Expected: PASS (11 passed).

- **Step 5: Commit**

```bash
git add kzn_recsys/spark/feas_codec.py tests/spark/test_feas_codec.py
git commit -m "feat(spark): FEAS artifact read/write (v1 + v2)"
```

## Task 6: DataFrame → mappings + CSR (Spark)

**Files:**
- Create: `kzn_recsys/spark/dataframes.py`
- Test: `tests/spark/test_dataframes.py`

Mappings are built by **sorted distinct** order (deterministic, self-consistent). This differs from Rust's first-seen order, which is fine per parity fact 2.

- **Step 1: Write the failing test**

Create `tests/spark/test_dataframes.py`:

```python
import numpy as np
import pytest

pytestmark = pytest.mark.spark

from kzn_recsys.spark.dataframes import build_mappings, build_csr_inputs


def _frames(spark):
    interactions = spark.createDataFrame(
        [("u1", "i1", 1.0), ("u1", "i2", 1.0), ("u2", "i2", 2.0)],
        ["user_id", "item_id", "value"],
    )
    user_features = spark.createDataFrame(
        [("u1", "plan_premium", 1.0)], ["user_id", "feature_name", "value"]
    )
    item_features = spark.createDataFrame(
        [("i1", "genre_drama", 1.0)], ["item_id", "feature_name", "value"]
    )
    return interactions, user_features, item_features


def test_build_mappings_sorted_and_complete(spark):
    i, u, t = _frames(spark)
    m = build_mappings(i, u, t)
    # users from interactions + user features; items from interactions + item features
    assert m.idx_to_user == ["u1", "u2"]
    assert m.idx_to_item == ["i1", "i2"]
    assert m.idx_to_user_feature == ["plan_premium"]
    assert m.idx_to_item_feature == ["genre_drama"]


def test_build_csr_shapes(spark):
    i, u, t = _frames(spark)
    m = build_mappings(i, u, t)
    X, U, T = build_csr_inputs(i, u, t, m, weighting=None)
    assert X.shape == (2, 2)   # 2 users x 2 items
    assert U.shape == (2, 1)   # 2 users x 1 user-feature
    assert T.shape == (1, 2)   # 1 item-feature x 2 items (transposed)
    # X value for (u2, i2) == 2.0
    assert X[m.user_to_idx["u2"], m.item_to_idx["i2"]] == 2.0
```

- **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/spark/test_dataframes.py -v`
Expected: FAIL — `ModuleNotFoundError` / missing names.

- **Step 3: Implement mappings + CSR build**

Create `kzn_recsys/spark/dataframes.py`:

```python
"""Spark DataFrame -> EASE matrix inputs. Mirrors src/data_pipeline.rs.

Index assignment uses sorted-distinct order (deterministic, self-consistent);
it does not reproduce Rust's first-seen order, which is unnecessary because the
mappings are persisted alongside S (see plan parity fact 2).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class Mappings:
    user_to_idx: dict
    idx_to_user: list
    item_to_idx: dict
    idx_to_item: list
    user_feature_to_idx: dict
    idx_to_user_feature: list
    item_feature_to_idx: dict
    idx_to_item_feature: list


def _sorted_distinct(*dfs_cols):
    """Collect the sorted-distinct union of (df, column) pairs as a list of str."""
    seen = set()
    for df, col in dfs_cols:
        for row in df.select(col).where(f"{col} is not null").distinct().collect():
            seen.add(row[0])
    return sorted(seen)


def _index(values):
    return {v: i for i, v in enumerate(values)}


def build_mappings(interactions_df, user_features_df, item_features_df) -> Mappings:
    users = _sorted_distinct((interactions_df, "user_id"), (user_features_df, "user_id"))
    items = _sorted_distinct((interactions_df, "item_id"), (item_features_df, "item_id"))
    ufeat = _sorted_distinct((user_features_df, "feature_name"))
    ifeat = _sorted_distinct((item_features_df, "feature_name"))
    return Mappings(
        user_to_idx=_index(users), idx_to_user=users,
        item_to_idx=_index(items), idx_to_item=items,
        user_feature_to_idx=_index(ufeat), idx_to_user_feature=ufeat,
        item_feature_to_idx=_index(ifeat), idx_to_item_feature=ifeat,
    )


def _triplets(df, row_col, col_col, row_map, col_map):
    """Collect (row_idx, col_idx, value) for rows whose keys are in both maps."""
    rows, cols, vals = [], [], []
    for r in df.select(row_col, col_col, "value").collect():
        rk, ck, v = r[0], r[1], r[2]
        if rk in row_map and ck in col_map and v is not None:
            rows.append(row_map[rk])
            cols.append(col_map[ck])
            vals.append(float(v))
    return rows, cols, vals


def build_csr_inputs(interactions_df, user_features_df, item_features_df, mappings, weighting):
    """Build (X, U, T) CSR matrices. `weighting` is an optional WeightingConfig.

    X: (N x M), U: (N x K), T: (L x M).  Weighting (event->decay->IPS) is applied
    to interaction values before X is assembled (see apply_weighting).
    """
    m = mappings
    N, M = len(m.idx_to_user), len(m.idx_to_item)
    K, L = len(m.idx_to_user_feature), len(m.idx_to_item_feature)

    idf = interactions_df
    if weighting is not None:
        idf = apply_weighting(idf, weighting, m)

    xr, xc, xv = _triplets(idf, "user_id", "item_id", m.user_to_idx, m.item_to_idx)
    ur, uc, uv = _triplets(user_features_df, "user_id", "feature_name",
                           m.user_to_idx, m.user_feature_to_idx)
    tr, tc, tv = _triplets(item_features_df, "item_id", "feature_name",
                           m.item_to_idx, m.item_feature_to_idx)

    X = sp.csr_matrix((xv, (xr, xc)), shape=(N, M))
    U = sp.csr_matrix((uv, (ur, uc)), shape=(N, K))
    # build (M x L) then transpose to (L x M) to match data_pipeline.rs:200-204
    T_ml = sp.csr_matrix((tv, (tr, tc)), shape=(M, L))
    T = T_ml.transpose().tocsr()
    return X, U, T
```

(`apply_weighting` is added in Task 7; `build_csr_inputs` calls it only when `weighting` is not None, so tests here pass with `weighting=None`.)

- **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/spark/test_dataframes.py -v`
Expected: PASS (2 passed). (Slow: first run boots Spark.)

- **Step 5: Commit**

```bash
git add kzn_recsys/spark/dataframes.py tests/spark/test_dataframes.py
git commit -m "feat(spark): DataFrame -> mappings + CSR inputs"
```

## Task 7: Advanced weighting transforms (Spark)

**Files:**
- Modify: `kzn_recsys/spark/dataframes.py`
- Test: `tests/spark/test_dataframes.py`

- **Step 1: Write the failing test**

Append to `tests/spark/test_dataframes.py`:

```python
from kzn_recsys.spark.feas_codec import WeightingConfig
from kzn_recsys.spark.dataframes import apply_weighting


def test_event_weights_multiply(spark):
    df = spark.createDataFrame(
        [("u1", "i1", 1.0, "click"), ("u1", "i2", 1.0, "purchase")],
        ["user_id", "item_id", "value", "event_type"],
    )
    m = build_mappings(df.select("user_id", "item_id", "value"),
                       spark.createDataFrame([], "user_id string, feature_name string, value double"),
                       spark.createDataFrame([], "item_id string, feature_name string, value double"))
    wc = WeightingConfig(event_weights={"purchase": 5.0}, decay_rate=0.0,
                         ips_alpha=0.0, sparsity_threshold=0.0)
    out = {(r["item_id"]): r["value"] for r in apply_weighting(df, wc, m).collect()}
    assert out["i1"] == 1.0      # unknown -> unchanged
    assert out["i2"] == 5.0      # purchase -> x5


def test_temporal_decay(spark):
    df = spark.createDataFrame(
        [("u1", "i1", 10.0, 0.0), ("u1", "i2", 10.0, 100.0)],
        ["user_id", "item_id", "value", "days_ago"],
    )
    m = build_mappings(df.select("user_id", "item_id", "value"),
                       spark.createDataFrame([], "user_id string, feature_name string, value double"),
                       spark.createDataFrame([], "item_id string, feature_name string, value double"))
    wc = WeightingConfig(event_weights=None, decay_rate=0.01, ips_alpha=0.0, sparsity_threshold=0.0)
    out = {(r["item_id"]): r["value"] for r in apply_weighting(df, wc, m).collect()}
    assert abs(out["i1"] - 10.0) < 1e-9
    assert abs(out["i2"] - 10.0 * np.exp(-1.0)) < 1e-6
```

- **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/spark/test_dataframes.py -k "event_weights or temporal" -v`
Expected: FAIL — `cannot import name 'apply_weighting'`.

- **Step 3: Implement `apply_weighting`**

Append to `kzn_recsys/spark/dataframes.py`:

```python
def apply_weighting(interactions_df, weighting, mappings):
    """Apply event -> decay -> IPS weighting to interaction `value`.

    Mirrors data_pipeline.rs:126-176 + weighting.rs. Returns a new DataFrame.
    Columns event_type / days_ago are used only when present and configured.
    """
    from pyspark.sql import functions as F
    from itertools import chain

    df = interactions_df
    cols = set(df.columns)

    # 1. Event-type weights (requires event_type column)
    if weighting.event_weights and "event_type" in cols:
        pairs = list(chain(*[(F.lit(k), F.lit(float(v)))
                             for k, v in weighting.event_weights.items()]))
        wmap = F.create_map(*pairs)
        mult = F.coalesce(wmap[F.col("event_type")], F.lit(1.0))
        df = df.withColumn("value", F.col("value") * mult)

    # 2. Temporal decay (requires days_ago column)
    if weighting.decay_rate and weighting.decay_rate > 0.0 and "days_ago" in cols:
        df = df.withColumn(
            "value",
            F.when(F.col("days_ago").isNotNull(),
                   F.col("value") * F.exp(F.lit(-weighting.decay_rate) * F.col("days_ago")))
             .otherwise(F.col("value")),
        )

    # 3. IPS: propensity = (item_count / max_count) ^ alpha; value /= propensity
    if weighting.ips_alpha and weighting.ips_alpha > 0.0:
        counts = df.groupBy("item_id").agg(F.count(F.lit(1)).alias("_cnt"))
        max_cnt = counts.agg(F.max("_cnt").alias("_m")).collect()[0]["_m"]
        if max_cnt and max_cnt > 0:
            counts = counts.withColumn(
                "_prop", (F.col("_cnt") / F.lit(float(max_cnt))) ** F.lit(weighting.ips_alpha)
            )
            df = (df.join(counts.select("item_id", "_prop"), on="item_id", how="left")
                    .withColumn("value",
                                F.when(F.col("_prop") > 1e-12, F.col("value") / F.col("_prop"))
                                 .otherwise(F.col("value")))
                    .drop("_prop"))
    return df
```

- **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/spark/test_dataframes.py -v`
Expected: PASS (4 passed).

- **Step 5: Commit**

```bash
git add kzn_recsys/spark/dataframes.py tests/spark/test_dataframes.py
git commit -m "feat(spark): advanced weighting transforms (event/decay/IPS)"
```

## Task 8: Gram collect-to-driver strategy

**Files:**
- Create: `kzn_recsys/spark/gram.py`
- Test: covered via `test_model.py` in Task 9 (gram_collect is a thin wrapper; its correctness is exercised end-to-end).

- **Step 1: Implement `gram_collect`**

Create `kzn_recsys/spark/gram.py`:

```python
"""Gram-matrix strategies feeding ease_core.train_ease.

Phase 1: collect-to-driver (gram_collect) builds CSR on the driver from Spark
frames and trains via ease_core. Phase 2 adds gram_distributed.
"""
from __future__ import annotations

from . import dataframes as _df
from . import ease_core as _core


def gram_collect(interactions_df, user_features_df, item_features_df, mappings, params, weighting):
    """Collect-to-driver: build (X,U,T) on the driver and train EASE.

    Returns the trained S matrix (Fortran-order ndarray).
    """
    X, U, T = _df.build_csr_inputs(
        interactions_df, user_features_df, item_features_df, mappings, weighting
    )
    return _core.train_ease(X, U, T, params)
```

- **Step 2: Verify it imports**

Run: `.venv/bin/python -c "from kzn_recsys.spark.gram import gram_collect; print('ok')"`
Expected: prints `ok`.

- **Step 3: Commit**

```bash
git add kzn_recsys/spark/gram.py
git commit -m "feat(spark): gram collect-to-driver strategy"
```

## Task 9: SparkEaseModel facade + build_and_train + save/load

**Files:**
- Create: `kzn_recsys/spark/model.py`
- Modify: `kzn_recsys/spark/__init__.py` (export public names)
- Test: `tests/spark/test_model.py`

- **Step 1: Write the failing test**

Create `tests/spark/test_model.py`:

```python
import numpy as np
import pytest

pytestmark = pytest.mark.spark

from kzn_recsys.spark import build_and_train, load_model


def _frames(spark):
    interactions = spark.createDataFrame(
        [("u1", "i1", 1.0), ("u1", "i2", 1.0),
         ("u2", "i2", 1.0), ("u2", "i3", 1.0),
         ("u3", "i1", 1.0), ("u3", "i3", 1.0)],
        ["user_id", "item_id", "value"],
    )
    empty_u = spark.createDataFrame([], "user_id string, feature_name string, value double")
    empty_t = spark.createDataFrame([], "item_id string, feature_name string, value double")
    return interactions, empty_u, empty_t


def test_train_and_predict_returns_string_ids(spark):
    i, u, t = _frames(spark)
    model = build_and_train(i, u, t, alpha=1.0, beta=1.0, lambda_=10.0)
    recs = model.predict({"i1": 1.0}, {}, top_k=2)
    assert all(isinstance(item_id, str) for item_id, _ in recs)
    assert len(recs) == 2
    # i1 itself must be excluded by zero-diagonal; recs come from {i2, i3}
    assert {r[0] for r in recs}.issubset({"i2", "i3"})


def test_save_load_roundtrip_predicts_identically(spark, tmp_path):
    i, u, t = _frames(spark)
    model = build_and_train(i, u, t, alpha=1.0, beta=1.0, lambda_=10.0)
    before = model.predict({"i1": 1.0}, {}, top_k=3)
    path = str(tmp_path / "m.fease")
    model.save(path)
    reloaded = load_model(path)
    after = reloaded.predict({"i1": 1.0}, {}, top_k=3)
    assert before == after
```

- **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/spark/test_model.py -v`
Expected: FAIL — `cannot import name 'build_and_train'`.

- **Step 3: Implement the facade**

Create `kzn_recsys/spark/model.py`:

```python
"""SparkEaseModel facade + build_and_train / load_model. Mirrors the public
shape of kzn_recsys.FeaseModel but consumes Spark DataFrames."""
from __future__ import annotations

import numpy as np

from . import dataframes as _df
from . import ease_core as _core
from . import gram as _gram
from . import feas_codec as _codec


class SparkEaseModel:
    def __init__(self, s_matrix, mappings, params, weighting=None, num_item_features=0):
        self.s_matrix = np.asfortranarray(s_matrix)
        self.mappings = mappings
        self.params = params
        self.weighting = weighting
        self.num_items = len(mappings.idx_to_item)
        self.num_user_features = len(mappings.idx_to_user_feature)
        self.num_item_features = num_item_features

    def predict(self, interactions: dict, features: dict, top_k: int):
        """interactions/features are {string_id: value}. Returns [(item_id, score)]."""
        m = self.mappings
        inter_idx = [(m.item_to_idx[k], v) for k, v in interactions.items()
                     if k in m.item_to_idx]
        feat_idx = [(m.user_feature_to_idx[k], v) for k, v in features.items()
                    if k in m.user_feature_to_idx]
        scores = _core.predict_scores(
            self.s_matrix, self.num_items, self.num_user_features,
            inter_idx, feat_idx, self.params.beta,
        )
        # exclude items the user already interacted with
        seen = {m.item_to_idx[k] for k in interactions if k in m.item_to_idx}
        order = np.argsort(-scores, kind="stable")
        out = []
        for j in order:
            j = int(j)
            if j in seen:
                continue
            out.append((m.idx_to_item[j], float(scores[j])))
            if len(out) == top_k:
                break
        return out

    def predict_similar_items(self, item_id: str, top_k: int):
        m = self.mappings
        if item_id not in m.item_to_idx:
            return []
        pairs = _core.predict_similar_items(
            self.s_matrix, m.item_to_idx[item_id], self.num_items, top_k
        )
        return [(m.idx_to_item[j], score) for j, score in pairs]

    def save(self, path: str) -> None:
        m = self.mappings
        wc = self.weighting
        artifact = _codec.FeaseArtifact(
            version=2,
            s_nrows=self.s_matrix.shape[0],
            s_ncols=self.s_matrix.shape[1],
            s_data=self.s_matrix,
            num_items=self.num_items,
            num_user_features=self.num_user_features,
            num_item_features=self.num_item_features,
            alpha=self.params.alpha, beta=self.params.beta,
            lambda_=self.params.lambda_, meta_weight=self.params.meta_weight,
            user_to_idx=list(m.user_to_idx.items()), idx_to_user=m.idx_to_user,
            item_to_idx=list(m.item_to_idx.items()), idx_to_item=m.idx_to_item,
            user_feature_to_idx=list(m.user_feature_to_idx.items()),
            idx_to_user_feature=m.idx_to_user_feature,
            item_feature_to_idx=list(m.item_feature_to_idx.items()),
            idx_to_item_feature=m.idx_to_item_feature,
            weighting_config=wc,
        )
        _codec.write_feas(artifact, path)


def build_and_train(interactions_df, user_features_df, item_features_df,
                    alpha=1.0, beta=1.0, lambda_=150.0, meta_weight=0.0, weighting=None):
    params = _core.EaseParams(alpha=alpha, beta=beta, lambda_=lambda_, meta_weight=meta_weight)
    mappings = _df.build_mappings(interactions_df, user_features_df, item_features_df)
    S = _gram.gram_collect(interactions_df, user_features_df, item_features_df,
                           mappings, params, weighting)
    if weighting is not None and getattr(weighting, "sparsity_threshold", 0.0) > 0.0:
        _core.prune_sparse(S, weighting.sparsity_threshold)
    return SparkEaseModel(S, mappings, params, weighting,
                          num_item_features=len(mappings.idx_to_item_feature))


def load_model(path: str) -> SparkEaseModel:
    art = _codec.read_feas(path)
    mappings = _df.Mappings(
        user_to_idx=dict(art.user_to_idx), idx_to_user=art.idx_to_user,
        item_to_idx=dict(art.item_to_idx), idx_to_item=art.idx_to_item,
        user_feature_to_idx=dict(art.user_feature_to_idx),
        idx_to_user_feature=art.idx_to_user_feature,
        item_feature_to_idx=dict(art.item_feature_to_idx),
        idx_to_item_feature=art.idx_to_item_feature,
    )
    params = _core.EaseParams(alpha=art.alpha, beta=art.beta,
                              lambda_=art.lambda_, meta_weight=art.meta_weight)
    return SparkEaseModel(art.s_data, mappings, params, art.weighting_config,
                          num_item_features=art.num_item_features)
```

- **Step 4: Export the public names**

Append to `kzn_recsys/spark/__init__.py`:

```python
from kzn_recsys.spark.ease_core import EaseParams
from kzn_recsys.spark.feas_codec import WeightingConfig
from kzn_recsys.spark.model import SparkEaseModel, build_and_train, load_model

__all__ = [
    "EaseParams",
    "WeightingConfig",
    "SparkEaseModel",
    "build_and_train",
    "load_model",
]
```

- **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/spark/test_model.py -v`
Expected: PASS (2 passed).

- **Step 6: Commit**

```bash
git add kzn_recsys/spark/model.py kzn_recsys/spark/__init__.py tests/spark/test_model.py
git commit -m "feat(spark): SparkEaseModel facade, build_and_train, save/load"
```

## Task 10: Native parity test (gated on the wheel)

**Files:**
- Test: `tests/spark/test_parity.py`

This is the cross-check that the PySpark scores match the Rust core within `1e-5`, and that FEAS files round-trip both directions. It is skipped automatically wherever `kzn_recsys._native` is absent.

- **Step 1: Write the parity test**

Create `tests/spark/test_parity.py`:

```python
"""Cross-checks the PySpark EASE impl against the native Rust core.

Skipped unless the compiled kzn_recsys._native extension is importable.
Run after: .venv/bin/maturin develop
"""
import numpy as np
import pytest

pytestmark = [pytest.mark.spark, pytest.mark.parity]

_native = pytest.importorskip("kzn_recsys._native")


def _write_long_parquet(rows, cols, tmp_path, name):
    import polars as pl
    path = str(tmp_path / name)
    pl.DataFrame(rows, schema=cols, orient="row").write_parquet(path)
    return path


def test_pyspark_scores_match_native_within_tol(spark, tmp_path):
    interactions = [("u1", "i1", 1.0), ("u1", "i2", 1.0),
                    ("u2", "i2", 1.0), ("u2", "i3", 1.0),
                    ("u3", "i1", 1.0), ("u3", "i3", 1.0)]
    cols = ["user_id", "item_id", "value"]
    i_path = _write_long_parquet(interactions, cols, tmp_path, "i.parquet")
    u_path = _write_long_parquet([], ["user_id", "feature_name", "value"], tmp_path, "u.parquet")
    t_path = _write_long_parquet([], ["item_id", "feature_name", "value"], tmp_path, "t.parquet")

    native_model = _native.build_and_train(
        interactions_path=i_path, user_features_path=u_path,
        item_features_path=t_path, alpha=1.0, beta=1.0, lambda_=10.0,
    )
    native_recs = dict(native_model.predict({"i1": 1.0}, {}, top_k=3))

    from kzn_recsys.spark import build_and_train as spark_train
    idf = spark.createDataFrame(interactions, cols)
    udf = spark.createDataFrame([], "user_id string, feature_name string, value double")
    tdf = spark.createDataFrame([], "item_id string, feature_name string, value double")
    spark_model = spark_train(idf, udf, tdf, alpha=1.0, beta=1.0, lambda_=10.0)
    spark_recs = dict(spark_model.predict({"i1": 1.0}, {}, top_k=3))

    # Spark predict excludes already-seen items; native may not. Compare on the
    # shared ids (Spark's set must be a subset of native's) and match scores there.
    assert set(spark_recs).issubset(set(native_recs))
    shared = set(native_recs) & set(spark_recs)
    assert shared, "no overlapping recommendations to compare"
    for item_id in shared:
        assert abs(native_recs[item_id] - spark_recs[item_id]) < 1e-5


def test_native_saved_model_loads_in_pyspark(spark, tmp_path):
    interactions = [("u1", "i1", 1.0), ("u1", "i2", 1.0), ("u2", "i2", 1.0)]
    cols = ["user_id", "item_id", "value"]
    i_path = _write_long_parquet(interactions, cols, tmp_path, "i.parquet")
    u_path = _write_long_parquet([], ["user_id", "feature_name", "value"], tmp_path, "u.parquet")
    t_path = _write_long_parquet([], ["item_id", "feature_name", "value"], tmp_path, "t.parquet")
    native_model = _native.build_and_train(
        interactions_path=i_path, user_features_path=u_path,
        item_features_path=t_path, alpha=1.0, beta=1.0, lambda_=10.0,
    )
    model_path = str(tmp_path / "native.fease")
    native_model.save(model_path)

    from kzn_recsys.spark import load_model
    py_model = load_model(model_path)
    native_recs = dict(native_model.predict({"i1": 1.0}, {}, top_k=2))
    py_recs = dict(py_model.predict({"i1": 1.0}, {}, top_k=2))
    # Loaded-from-native model must score the shared ids identically (within tol).
    shared = set(native_recs) & set(py_recs)
    assert shared, "no overlapping recommendations to compare"
    for item_id in shared:
        assert abs(native_recs[item_id] - py_recs[item_id]) < 1e-5


def test_pyspark_saved_model_loads_in_native(spark, tmp_path):
    from kzn_recsys.spark import build_and_train as spark_train
    interactions = [("u1", "i1", 1.0), ("u1", "i2", 1.0), ("u2", "i2", 1.0)]
    cols = ["user_id", "item_id", "value"]
    idf = spark.createDataFrame(interactions, cols)
    udf = spark.createDataFrame([], "user_id string, feature_name string, value double")
    tdf = spark.createDataFrame([], "item_id string, feature_name string, value double")
    spark_model = spark_train(idf, udf, tdf, alpha=1.0, beta=1.0, lambda_=10.0)
    model_path = str(tmp_path / "spark.fease")
    spark_model.save(model_path)

    native_model = _native.load_model(model_path)
    native_recs = dict(native_model.predict({"i1": 1.0}, {}, top_k=2))
    py_recs = dict(spark_model.predict({"i1": 1.0}, {}, top_k=2))
    # A Spark-trained model loaded by the native core must score shared ids identically.
    shared = set(native_recs) & set(py_recs)
    assert shared, "no overlapping recommendations to compare"
    for item_id in py_recs:
        if item_id in shared:
            assert abs(py_recs[item_id] - native_recs[item_id]) < 1e-5
```

- **Step 2: Verify the native predict contract before running**

Run: `.venv/bin/maturin develop && .venv/bin/python -c "import kzn_recsys; help(kzn_recsys.FeaseModel.predict)"`
Confirm two things: (a) the argument shape is `predict(interactions_dict, features_dict, top_k=...)` — adjust the parity calls if it differs; (b) whether native `predict` filters already-interacted items. The parity tests are written to tolerate native *not* filtering (Spark's recs must be a subset of native's). If native *does* filter, the subset assertions still hold.

- **Step 3: Run the parity suite**

Run: `.venv/bin/python -m pytest tests/spark/test_parity.py -v`
Expected: PASS (3 passed).

- **Step 4: Commit**

```bash
git add tests/spark/test_parity.py
git commit -m "test(spark): native parity (scores + both-directions FEAS interop)"
```

---

# Phase 2 — Distributed Gram

Removes the driver-memory ceiling on interaction count: the four `ZᵀZ` blocks are computed as Spark aggregations; only the `(M+K)²` dense Gram is collected for the solve.

## Task 11: Distributed Gram accumulation

**Files:**
- Modify: `kzn_recsys/spark/gram.py`
- Modify: `kzn_recsys/spark/ease_core.py` (factor the solve out of `train_ease`)
- Test: `tests/spark/test_gram_distributed.py`

- **Step 1: Factor the solve out of `train_ease` (refactor, no behavior change)**

In `kzn_recsys/spark/ease_core.py`, extract the post-Gram math into `solve_from_gram` and have `train_ease` call it. Replace the body of `train_ease` from the `# P = inv...` line onward with a call:

```python
def solve_from_gram(G: np.ndarray, lambda_: float) -> np.ndarray:
    """Given the dense Gram G ((M+K)x(M+K)), return S (Fortran-order).

    G is modified in place (lambda added to diagonal). Mirrors model.rs:190-224.
    """
    total = G.shape[0]
    G.flat[:: total + 1] += lambda_
    P = np.linalg.inv(G)
    p_jj = np.diag(P).copy()
    inv = np.where(np.abs(p_jj) > 1e-12, -1.0 / p_jj, 0.0)
    S = P * inv[None, :]
    np.fill_diagonal(S, 0.0)
    return np.asfortranarray(S)
```

Then `train_ease` ends with:

```python
    # ... after assembling G ...
    return solve_from_gram(G, params.lambda_)
```

Run: `.venv/bin/python -m pytest tests/spark/test_ease_core.py -v`
Expected: PASS (8 passed) — refactor preserves behavior.

- **Step 2: Write the failing distributed-Gram test**

Create `tests/spark/test_gram_distributed.py`:

```python
import numpy as np
import pytest

pytestmark = pytest.mark.spark

from kzn_recsys.spark.dataframes import build_mappings
from kzn_recsys.spark.gram import gram_collect, gram_distributed
from kzn_recsys.spark.ease_core import EaseParams


def _frames(spark):
    interactions = spark.createDataFrame(
        [("u1", "i1", 1.0), ("u1", "i2", 1.0),
         ("u2", "i2", 2.0), ("u2", "i3", 1.0),
         ("u3", "i1", 1.0), ("u3", "i3", 3.0)],
        ["user_id", "item_id", "value"],
    )
    u = spark.createDataFrame([("u1", "plan_x", 1.0)],
                              ["user_id", "feature_name", "value"])
    t = spark.createDataFrame([("i1", "genre_y", 1.0)],
                              ["item_id", "feature_name", "value"])
    return interactions, u, t


def test_distributed_matches_collect(spark):
    i, u, t = _frames(spark)
    m = build_mappings(i, u, t)
    params = EaseParams(alpha=1.0, beta=1.0, lambda_=10.0)
    S_collect = gram_collect(i, u, t, m, params, weighting=None)
    S_dist = gram_distributed(i, u, t, m, params, weighting=None)
    assert np.allclose(S_collect, S_dist, atol=1e-9)
```

- **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/spark/test_gram_distributed.py -v`
Expected: FAIL — `cannot import name 'gram_distributed'`.

- **Step 4: Implement `gram_distributed`**

Append to `kzn_recsys/spark/gram.py`:

```python
import numpy as np

from .ease_core import solve_from_gram


def _block_from_cooccurrence(df, left_key, left_val, right_key, right_val,
                             left_map, right_map, n_left, n_right):
    """Compute a dense (n_left x n_right) block = sum over the join key of
    (left_val * right_val), via a Spark self/cross join on a shared entity key.

    `df` already has columns: join_entity, plus the named value/index columns.
    Returns a dense numpy array.
    """
    from pyspark.sql import functions as F
    agg = (df.groupBy(left_key, right_key)
             .agg(F.sum(F.col(left_val) * F.col(right_val)).alias("v")))
    block = np.zeros((n_left, n_right), dtype=np.float64)
    for r in agg.collect():
        li = left_map.get(r[left_key])
        ri = right_map.get(r[right_key])
        if li is not None and ri is not None:
            block[li, ri] = r["v"]
    return block


def gram_distributed(interactions_df, user_features_df, item_features_df,
                     mappings, params, weighting):
    """Compute the four Gram blocks as Spark aggregations; solve on the driver.

    Mirrors the block algebra of model.rs:130-187 but accumulates ZᵀZ in Spark.
    Only the dense (M+K)² Gram crosses to the driver — memory is independent of N.
    """
    from pyspark.sql import functions as F
    from . import dataframes as _df

    m = mappings
    M = len(m.idx_to_item)
    K = len(m.idx_to_user_feature)
    L = len(m.idx_to_item_feature)
    a, b = params.alpha, params.beta
    w = params.meta_weight if params.meta_weight > 0.0 else 1.0

    idf = interactions_df
    if weighting is not None:
        idf = _df.apply_weighting(idf, weighting, m)

    # XtX[i,j] = sum over users of value(u,i)*value(u,j): self-join interactions on user_id
    left = idf.select(F.col("item_id").alias("li"), F.col("value").alias("lv"),
                      F.col("user_id").alias("uk"))
    right = idf.select(F.col("item_id").alias("ri"), F.col("value").alias("rv"),
                       F.col("user_id").alias("uk2"))
    xtx_df = left.join(right, left.uk == right.uk2)
    XtX = _block_from_cooccurrence(xtx_df, "li", "lv", "ri", "rv",
                                   m.item_to_idx, m.item_to_idx, M, M)

    # TtT[i,j]: self-join item_features on feature owner item -> over item features
    # T is (L x M); TtT here means (item x item) contribution = sum over item-features.
    # Equivalent: for items i,j sharing item-feature f: value(f,i)*value(f,j).
    tf = item_features_df.select(F.col("item_id").alias("it"),
                                 F.col("feature_name").alias("fn"),
                                 F.col("value").alias("fv"))
    tleft = tf.select(F.col("it").alias("li"), F.col("fv").alias("lv"), F.col("fn").alias("fk"))
    tright = tf.select(F.col("it").alias("ri"), F.col("fv").alias("rv"), F.col("fn").alias("fk2"))
    ttt_df = tleft.join(tright, tleft.fk == tright.fk2)
    TtT = _block_from_cooccurrence(ttt_df, "li", "lv", "ri", "rv",
                                   m.item_to_idx, m.item_to_idx, M, M)

    # XtU[i,k] = sum over users of value(u,i)*ufeat(u,k): join interactions to user features
    uf = user_features_df.select(F.col("user_id").alias("uk2"),
                                 F.col("feature_name").alias("fn"),
                                 F.col("value").alias("fv"))
    xtu_df = (idf.select(F.col("item_id").alias("li"), F.col("value").alias("lv"),
                         F.col("user_id").alias("uk"))
                 .join(uf, F.col("uk") == F.col("uk2")))
    XtU = _block_from_cooccurrence(xtu_df, "li", "lv", "fn", "fv",
                                   m.item_to_idx, m.user_feature_to_idx, M, K)

    # UtU[k,l] = sum over users of ufeat(u,k)*ufeat(u,l): self-join user features on user_id
    uleft = user_features_df.select(F.col("user_id").alias("uk"),
                                    F.col("feature_name").alias("lk"),
                                    F.col("value").alias("lv"))
    uright = user_features_df.select(F.col("user_id").alias("uk2"),
                                     F.col("feature_name").alias("rk"),
                                     F.col("value").alias("rv"))
    utu_df = uleft.join(uright, uleft.uk == uright.uk2)
    UtU = _block_from_cooccurrence(utu_df, "lk", "lv", "rk", "rv",
                                   m.user_feature_to_idx, m.user_feature_to_idx, K, K)

    total = M + K
    G = np.zeros((total, total), dtype=np.float64)
    G[:M, :M] = XtX + w * a * a * TtT
    if K > 0:
        G[:M, M:] = b * XtU
        G[M:, :M] = b * XtU.T
        G[M:, M:] = b * b * UtU
    return solve_from_gram(G, params.lambda_)
```

- **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/spark/test_gram_distributed.py -v`
Expected: PASS (1 passed).

- **Step 6: Wire a strategy switch into `build_and_train`**

In `kzn_recsys/spark/model.py`, change `build_and_train` to accept `strategy="collect"` and dispatch:

```python
def build_and_train(interactions_df, user_features_df, item_features_df,
                    alpha=1.0, beta=1.0, lambda_=150.0, meta_weight=0.0,
                    weighting=None, strategy="collect"):
    params = _core.EaseParams(alpha=alpha, beta=beta, lambda_=lambda_, meta_weight=meta_weight)
    mappings = _df.build_mappings(interactions_df, user_features_df, item_features_df)
    if strategy == "distributed":
        S = _gram.gram_distributed(interactions_df, user_features_df, item_features_df,
                                   mappings, params, weighting)
    else:
        S = _gram.gram_collect(interactions_df, user_features_df, item_features_df,
                               mappings, params, weighting)
    if weighting is not None and getattr(weighting, "sparsity_threshold", 0.0) > 0.0:
        _core.prune_sparse(S, weighting.sparsity_threshold)
    return SparkEaseModel(S, mappings, params, weighting,
                          num_item_features=len(mappings.idx_to_item_feature))
```

Run: `.venv/bin/python -m pytest tests/spark/test_model.py tests/spark/test_gram_distributed.py -v`
Expected: PASS (all). The default `strategy="collect"` keeps Task 9 tests green.

- **Step 7: Commit**

```bash
git add kzn_recsys/spark/gram.py kzn_recsys/spark/ease_core.py kzn_recsys/spark/model.py tests/spark/test_gram_distributed.py
git commit -m "feat(spark): distributed Gram strategy + build_and_train strategy switch"
```

---

# Phase 3 — Evaluation & splits

## Task 12: Ranking metrics (Spark-free port of metrics.rs)

**Files:**
- Create: `kzn_recsys/spark/metrics.py`
- Test: `tests/spark/test_metrics.py`

- **Step 1: Write the failing test**

Create `tests/spark/test_metrics.py`:

```python
import math

from kzn_recsys.spark.metrics import (
    precision_at_k, recall_at_k, ndcg_at_k, mean_average_precision,
    coverage, hit_rate_at_k,
)


def test_precision_at_k():
    assert abs(precision_at_k([1, 2, 3, 4, 5], {1, 3, 5, 7}, 3) - 2 / 3) < 1e-10
    assert precision_at_k([1, 2, 3], {1, 2}, 0) == 0.0


def test_recall_at_k():
    assert abs(recall_at_k([1, 2, 3, 4, 5], {1, 3, 5, 7}, 3) - 0.5) < 1e-10
    assert recall_at_k([1, 2, 3], set(), 3) == 0.0


def test_ndcg_at_k_imperfect():
    dcg = 1.0 / math.log2(4) + 1.0 / math.log2(6)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert abs(ndcg_at_k([1, 2, 3, 4, 5], {3, 5}, 5) - dcg / idcg) < 1e-10


def test_map():
    expected = (1.0 + 2 / 3 + 3 / 5) / 3
    assert abs(mean_average_precision([1, 2, 3, 4, 5], {1, 3, 5}) - expected) < 1e-10


def test_coverage():
    assert abs(coverage([[0, 1, 2], [2, 3, 4], [4, 5]], 10) - 0.6) < 1e-10
    assert coverage([[1, 2]], 0) == 0.0


def test_hit_rate():
    assert hit_rate_at_k([10, 20, 30], {20, 40}, 3) == 1.0
    assert hit_rate_at_k([10, 20, 30, 40], {40}, 2) == 0.0
```

- **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/spark/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- **Step 3: Implement the metrics**

Create `kzn_recsys/spark/metrics.py`:

```python
"""Ranking metrics. Direct port of src/metrics.rs (binary relevance)."""
from __future__ import annotations

import math


def precision_at_k(recommended, relevant, k) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(recommended, relevant, k) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended, relevant, k) -> float:
    if k == 0 or not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 2.0)
              for rank, item in enumerate(recommended[:k]) if item in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 2.0) for rank in range(ideal_hits))
    return 0.0 if idcg == 0.0 else dcg / idcg


def mean_average_precision(recommended, relevant) -> float:
    if not relevant:
        return 0.0
    hits = 0
    sum_precision = 0.0
    for i, item in enumerate(recommended):
        if item in relevant:
            hits += 1
            sum_precision += hits / (i + 1)
    return sum_precision / len(relevant)


def coverage(all_recommendations, num_total_items) -> float:
    if num_total_items == 0:
        return 0.0
    unique = set()
    for recs in all_recommendations:
        unique.update(recs)
    return len(unique) / num_total_items


def hit_rate_at_k(recommended, relevant, k) -> float:
    if k == 0:
        return 0.0
    return 1.0 if any(item in relevant for item in recommended[:k]) else 0.0
```

- **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/spark/test_metrics.py -v`
Expected: PASS (6 passed).

- **Step 5: Commit**

```bash
git add kzn_recsys/spark/metrics.py tests/spark/test_metrics.py
git commit -m "feat(spark): ranking metrics port"
```

## Task 13: Splits (DataFrame, deterministic given seed)

**Files:**
- Create: `kzn_recsys/spark/splits.py`
- Test: `tests/spark/test_splits.py`

Per parity fact 5, these are deterministic given a Python seed but **not** row-identical to the Rust splits. Semantics mirror `evaluation.rs`.

- **Step 1: Write the failing test**

Create `tests/spark/test_splits.py`:

```python
import pytest

pytestmark = pytest.mark.spark

from kzn_recsys.spark.splits import random_split, temporal_split, leave_k_out_split


def _interactions(spark):
    rows = [("u1", "i1", 1.0), ("u1", "i2", 1.0), ("u1", "i3", 1.0), ("u1", "i4", 1.0),
            ("u2", "i1", 1.0), ("u2", "i2", 1.0)]
    return spark.createDataFrame(rows, ["user_id", "item_id", "value"])


def test_random_split_holds_out_fraction_and_is_deterministic(spark):
    df = _interactions(spark)
    train1, test1 = random_split(df, test_ratio=0.5, seed=42)
    train2, test2 = random_split(df, test_ratio=0.5, seed=42)
    # determinism: same seed -> same test rows
    assert sorted(test1.collect()) == sorted(test2.collect())
    # every user with >=2 interactions keeps at least one train row
    train_users = {r["user_id"] for r in train1.collect()}
    assert {"u1", "u2"}.issubset(train_users)
    # nothing lost: train + test == original count
    assert train1.count() + test1.count() == df.count()


def test_temporal_split_by_cutoff(spark):
    rows = [("u1", "i1", 1.0, 5.0), ("u1", "i2", 1.0, 50.0)]
    df = spark.createDataFrame(rows, ["user_id", "item_id", "value", "days_ago"])
    train, test = temporal_split(df, days_ago_cutoff=10.0)
    # recent (days_ago <= cutoff) -> test
    assert {r["item_id"] for r in test.collect()} == {"i1"}
    assert {r["item_id"] for r in train.collect()} == {"i2"}


def test_leave_k_out(spark):
    df = _interactions(spark)
    train, test = leave_k_out_split(df, k=1, seed=7)
    # u1 has 4 -> exactly 1 held out; u2 has 2 -> exactly 1 held out
    test_by_user = {}
    for r in test.collect():
        test_by_user.setdefault(r["user_id"], 0)
        test_by_user[r["user_id"]] += 1
    assert test_by_user.get("u1") == 1
    assert test_by_user.get("u2") == 1
```

- **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/spark/test_splits.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- **Step 3: Implement the splits**

Create `kzn_recsys/spark/splits.py`:

```python
"""Train/test splits on interaction DataFrames. Semantics mirror
src/evaluation.rs; determinism is via a Python-seeded RNG (NOT row-identical
to the Rust StdRng splits — see plan parity fact 5)."""
from __future__ import annotations

import random


def _add_row_index(df):
    # stable per-row id for reproducible masking
    from pyspark.sql import functions as F
    return df.withColumn("_rid", F.monotonically_increasing_id())


def random_split(interactions_df, test_ratio: float, seed: int):
    """For each user, hold out round(test_ratio * n) rows (clamped to [1, n-1]
    for users with >= 2 interactions). Mirrors evaluation.rs:103-173 semantics."""
    if not 0.0 <= test_ratio <= 1.0:
        raise ValueError("test_ratio must be between 0.0 and 1.0")
    df = _add_row_index(interactions_df)
    rows = df.select("user_id", "_rid").collect()
    by_user = {}
    for r in rows:
        by_user.setdefault(r["user_id"], []).append(r["_rid"])

    rng = random.Random(seed)
    test_ids = set()
    for uid in sorted(by_user):
        rids = list(by_user[uid])
        rng.shuffle(rids)
        n = len(rids)
        if n < 2:
            n_test = 0
        else:
            n_test = max(1, min(round(n * test_ratio), n - 1))
        test_ids.update(rids[:n_test])

    from pyspark.sql import functions as F
    test_df = df.where(F.col("_rid").isin(list(test_ids))).drop("_rid")
    train_df = df.where(~F.col("_rid").isin(list(test_ids))).drop("_rid")
    return train_df, test_df


def temporal_split(interactions_df, days_ago_cutoff: float):
    """Recent (days_ago <= cutoff) -> test, older -> train. Mirrors evaluation.rs:177-226."""
    from pyspark.sql import functions as F
    if interactions_df.where(F.col("days_ago").isNull()).count() > 0:
        raise ValueError("days_ago contains nulls; temporal_split requires non-null days_ago")
    test_df = interactions_df.where(F.col("days_ago") <= days_ago_cutoff)
    train_df = interactions_df.where(F.col("days_ago") > days_ago_cutoff)
    return train_df, test_df


def leave_k_out_split(interactions_df, k: int, seed: int):
    """Hold out exactly k rows per user with >= k+1 interactions. Mirrors evaluation.rs:230+."""
    df = _add_row_index(interactions_df)
    rows = df.select("user_id", "_rid").collect()
    by_user = {}
    for r in rows:
        by_user.setdefault(r["user_id"], []).append(r["_rid"])

    rng = random.Random(seed)
    test_ids = set()
    for uid in sorted(by_user):
        rids = list(by_user[uid])
        if len(rids) < k + 1:
            continue
        rng.shuffle(rids)
        test_ids.update(rids[:k])

    from pyspark.sql import functions as F
    test_df = df.where(F.col("_rid").isin(list(test_ids))).drop("_rid")
    train_df = df.where(~F.col("_rid").isin(list(test_ids))).drop("_rid")
    return train_df, test_df
```

- **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/spark/test_splits.py -v`
Expected: PASS (3 passed).

- **Step 5: Commit**

```bash
git add kzn_recsys/spark/splits.py tests/spark/test_splits.py
git commit -m "feat(spark): random/temporal/leave-k-out splits"
```

## Task 14: Evaluation harness (`SparkEaseModel.evaluate`)

**Files:**
- Modify: `kzn_recsys/spark/model.py`
- Test: `tests/spark/test_model.py`

- **Step 1: Write the failing test**

Append to `tests/spark/test_model.py`:

```python
def test_evaluate_returns_metric_report(spark):
    train = spark.createDataFrame(
        [("u1", "i1", 1.0), ("u1", "i2", 1.0),
         ("u2", "i2", 1.0), ("u2", "i3", 1.0),
         ("u3", "i1", 1.0), ("u3", "i3", 1.0)],
        ["user_id", "item_id", "value"],
    )
    test = spark.createDataFrame(
        [("u1", "i3", 1.0), ("u2", "i1", 1.0)],
        ["user_id", "item_id", "value"],
    )
    empty_u = spark.createDataFrame([], "user_id string, feature_name string, value double")
    empty_t = spark.createDataFrame([], "item_id string, feature_name string, value double")
    model = build_and_train(train, empty_u, empty_t, alpha=1.0, beta=1.0, lambda_=10.0)
    report = model.evaluate(test, train, empty_u, k_values=[1, 2, 3])
    assert "metrics" in report and "coverage" in report
    ks = {m["k"] for m in report["metrics"]}
    assert ks == {1, 2, 3}
    for m in report["metrics"]:
        assert 0.0 <= m["ndcg"] <= 1.0
        assert 0.0 <= m["recall"] <= 1.0
    assert report["num_users"] >= 1
```

- **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/spark/test_model.py -k evaluate -v`
Expected: FAIL — `AttributeError: 'SparkEaseModel' object has no attribute 'evaluate'`.

- **Step 3: Implement `evaluate`**

Add to `SparkEaseModel` in `kzn_recsys/spark/model.py` (and add `from . import metrics as _metrics` at the top):

```python
    def evaluate(self, test_interactions_df, train_interactions_df,
                 user_features_df, k_values):
        """Score test users and compute precision/recall/ndcg/map/hit_rate@k + coverage.

        Mirrors src/evaluation.rs::evaluate_model semantics for EASE: each user's
        training interactions form the input; held-out test items are the relevant set.
        """
        m = self.mappings
        max_k = max(k_values)

        def _collect_by_user(df, val=True):
            out = {}
            for r in df.select("user_id", "item_id", "value").collect():
                out.setdefault(r["user_id"], {})[r["item_id"]] = float(r["value"])
            return out

        train_by_user = _collect_by_user(train_interactions_df)
        test_by_user = _collect_by_user(test_interactions_df)

        # user features as {user_id: {feature_name: value}}
        feats_by_user = {}
        for r in user_features_df.select("user_id", "feature_name", "value").collect():
            feats_by_user.setdefault(r["user_id"], {})[r["feature_name"]] = float(r["value"])

        per_k = {k: {"precision": 0.0, "recall": 0.0, "ndcg": 0.0,
                     "map": 0.0, "hit_rate": 0.0} for k in k_values}
        all_recs = []
        n_users = 0
        n_interactions = 0

        for uid, relevant_map in test_by_user.items():
            relevant = set(relevant_map)
            if not relevant:
                continue
            interactions = train_by_user.get(uid, {})
            features = feats_by_user.get(uid, {})
            recs = self.predict(interactions, features, top_k=max_k)
            rec_ids = [item_id for item_id, _ in recs]
            rec_idx = [m.item_to_idx[i] for i in rec_ids if i in m.item_to_idx]
            all_recs.append(rec_idx)
            relevant_idx = {m.item_to_idx[i] for i in relevant if i in m.item_to_idx}
            n_users += 1
            n_interactions += len(relevant)
            for k in k_values:
                per_k[k]["precision"] += _metrics.precision_at_k(rec_idx, relevant_idx, k)
                per_k[k]["recall"] += _metrics.recall_at_k(rec_idx, relevant_idx, k)
                per_k[k]["ndcg"] += _metrics.ndcg_at_k(rec_idx, relevant_idx, k)
                per_k[k]["hit_rate"] += _metrics.hit_rate_at_k(rec_idx, relevant_idx, k)
                per_k[k]["map"] += _metrics.mean_average_precision(rec_idx, relevant_idx)

        denom = max(n_users, 1)
        metrics_out = []
        for k in sorted(k_values):
            metrics_out.append({
                "k": k,
                "precision": per_k[k]["precision"] / denom,
                "recall": per_k[k]["recall"] / denom,
                "ndcg": per_k[k]["ndcg"] / denom,
                "map": per_k[k]["map"] / denom,
                "hit_rate": per_k[k]["hit_rate"] / denom,
            })
        return {
            "metrics": metrics_out,
            "coverage": _metrics.coverage(all_recs, self.num_items),
            "num_users": n_users,
            "num_interactions": n_interactions,
        }
```

- **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/spark/test_model.py -v`
Expected: PASS (all model tests).

- **Step 5: Commit**

```bash
git add kzn_recsys/spark/model.py tests/spark/test_model.py
git commit -m "feat(spark): evaluation harness on SparkEaseModel"
```

---

# Phase 4 — Hyperparameter tuning

## Task 15: Grid & random search with user k-fold CV

**Files:**
- Create: `kzn_recsys/spark/tuning.py`
- Modify: `kzn_recsys/spark/__init__.py` (export search functions)
- Test: `tests/spark/test_tuning.py`

Optimization target is NDCG@k. Folds are over users (each user's interactions partitioned into k folds). For each fold, train on the other folds, evaluate on the held-out fold.

- **Step 1: Write the failing test**

Create `tests/spark/test_tuning.py`:

```python
import pytest

pytestmark = pytest.mark.spark

from kzn_recsys.spark import grid_search, random_search


def _frames(spark):
    rows = []
    for u in range(8):
        for i in range(5):
            rows.append((f"u{u}", f"i{i}", 1.0))
    interactions = spark.createDataFrame(rows, ["user_id", "item_id", "value"])
    empty_u = spark.createDataFrame([], "user_id string, feature_name string, value double")
    empty_t = spark.createDataFrame([], "item_id string, feature_name string, value double")
    return interactions, empty_u, empty_t


def test_grid_search_returns_best_and_all_trials(spark):
    i, u, t = _frames(spark)
    result = grid_search(
        i, u, t,
        param_grid={"lambda_": [1.0, 100.0], "alpha": [1.0]},
        k_folds=2, eval_k=3, seed=1,
    )
    assert "best_params" in result
    assert "best_score" in result
    assert len(result["trials"]) == 2  # 2 lambda values x 1 alpha
    assert "lambda_" in result["best_params"]


def test_random_search_respects_n_iter(spark):
    i, u, t = _frames(spark)
    result = random_search(
        i, u, t,
        param_distributions={"lambda_": [1.0, 10.0, 100.0, 150.0], "beta": [0.5, 1.0]},
        n_iter=3, k_folds=2, eval_k=3, seed=1,
    )
    assert len(result["trials"]) == 3
```

- **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/spark/test_tuning.py -v`
Expected: FAIL — `cannot import name 'grid_search'`.

- **Step 3: Implement tuning**

Create `kzn_recsys/spark/tuning.py`:

```python
"""Grid/random search with user k-fold CV, optimizing NDCG@eval_k.

Folds partition each user's interaction rows into k groups; fold f trains on
the complement and evaluates on fold f. Mirrors the intent of src/tuning.rs.
"""
from __future__ import annotations

import itertools
import random

from pyspark.sql import functions as F

from .model import build_and_train
from . import metrics as _metrics


def _assign_folds(interactions_df, k_folds, seed):
    """Return a DataFrame with an added integer `_fold` in [0, k_folds)."""
    df = interactions_df.withColumn("_rid", F.monotonically_increasing_id())
    rows = df.select("user_id", "_rid").collect()
    by_user = {}
    for r in rows:
        by_user.setdefault(r["user_id"], []).append(r["_rid"])
    rng = random.Random(seed)
    fold_of = {}
    for uid in sorted(by_user):
        rids = list(by_user[uid])
        rng.shuffle(rids)
        for pos, rid in enumerate(rids):
            fold_of[rid] = pos % k_folds
    mapping = df.sparkSession.createDataFrame(
        [(rid, f) for rid, f in fold_of.items()], ["_rid", "_fold"]
    )
    return df.join(mapping, on="_rid", how="left")


def _score_params(interactions_df, user_features_df, item_features_df,
                  params, k_folds, eval_k, seed):
    """Mean NDCG@eval_k across folds for one parameter set."""
    folded = _assign_folds(interactions_df, k_folds, seed).cache()
    ndcgs = []
    for f in range(k_folds):
        train_df = folded.where(F.col("_fold") != f).drop("_rid", "_fold")
        test_df = folded.where(F.col("_fold") == f).drop("_rid", "_fold")
        if test_df.count() == 0:
            continue
        model = build_and_train(
            train_df, user_features_df, item_features_df,
            alpha=params.get("alpha", 1.0), beta=params.get("beta", 1.0),
            lambda_=params.get("lambda_", 150.0),
            meta_weight=params.get("meta_weight", 0.0),
        )
        report = model.evaluate(test_df, train_df, user_features_df, k_values=[eval_k])
        ndcgs.append(report["metrics"][0]["ndcg"])
    folded.unpersist()
    return sum(ndcgs) / len(ndcgs) if ndcgs else 0.0


def _run_trials(interactions_df, user_features_df, item_features_df,
                param_sets, k_folds, eval_k, seed):
    trials = []
    for params in param_sets:
        score = _score_params(interactions_df, user_features_df, item_features_df,
                              params, k_folds, eval_k, seed)
        trials.append({"params": params, "score": score})
    best = max(trials, key=lambda t: t["score"]) if trials else {"params": {}, "score": 0.0}
    return {"best_params": best["params"], "best_score": best["score"], "trials": trials}


def grid_search(interactions_df, user_features_df, item_features_df,
                param_grid, k_folds=3, eval_k=10, seed=42):
    keys = sorted(param_grid)
    param_sets = [dict(zip(keys, combo))
                  for combo in itertools.product(*(param_grid[k] for k in keys))]
    return _run_trials(interactions_df, user_features_df, item_features_df,
                       param_sets, k_folds, eval_k, seed)


def random_search(interactions_df, user_features_df, item_features_df,
                  param_distributions, n_iter=10, k_folds=3, eval_k=10, seed=42):
    rng = random.Random(seed)
    keys = sorted(param_distributions)
    param_sets = []
    for _ in range(n_iter):
        param_sets.append({k: rng.choice(param_distributions[k]) for k in keys})
    return _run_trials(interactions_df, user_features_df, item_features_df,
                       param_sets, k_folds, eval_k, seed)
```

- **Step 4: Export the search functions**

Update `kzn_recsys/spark/__init__.py` to add the imports and `__all__` entries:

```python
from kzn_recsys.spark.tuning import grid_search, random_search
```

And extend `__all__` with `"grid_search"` and `"random_search"`.

- **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/spark/test_tuning.py -v`
Expected: PASS (2 passed).

- **Step 6: Commit**

```bash
git add kzn_recsys/spark/tuning.py kzn_recsys/spark/__init__.py tests/spark/test_tuning.py
git commit -m "feat(spark): grid/random search with user k-fold CV"
```

---

# Phase 5 — Packaging: importable without the native extension

## Task 16: Make `kzn_recsys` import without `_native`

**Files:**
- Modify: `kzn_recsys/__init__.py:3-22`
- Test: extend an existing test or add `tests/spark/test_optional_native.py`

The current top-level `from kzn_recsys._native import (...)` is unconditional (`__init__.py:3`). In a restricted environment without the compiled extension, `import kzn_recsys` would raise. Wrap it in the same `try/except` pattern already used for `_HAS_ML_MODELS` / `_HAS_ONNX`, exposing `_HAS_NATIVE`.

- **Step 1: Write the failing test**

Create `tests/spark/test_optional_native.py`:

```python
def test_spark_subpackage_imports_without_touching_native():
    # Importing the spark subpackage must not require _native.
    import importlib
    mod = importlib.import_module("kzn_recsys.spark")
    assert hasattr(mod, "build_and_train")
    assert hasattr(mod, "load_model")


def test_has_native_flag_exists():
    import kzn_recsys
    assert hasattr(kzn_recsys, "_HAS_NATIVE")
    assert isinstance(kzn_recsys._HAS_NATIVE, bool)
```

- **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/spark/test_optional_native.py -v`
Expected: FAIL — `AttributeError: module 'kzn_recsys' has no attribute '_HAS_NATIVE'`.

- **Step 3: Wrap the native import block**

In `kzn_recsys/__init__.py`, replace the unconditional block at lines 3-28 (the `from kzn_recsys._native import (...)` and the `from kzn_recsys.fease_wrapper import (...)`) so the native portion is guarded. New top of file:

```python
"""kzn_recsys — Python wrapper for the Rust FEASE recommender."""

try:  # The compiled Rust extension is absent in pure-Python (e.g. Spark) installs.
    from kzn_recsys._native import (
        FeaseModel,
        ModelRegistry,
        build_and_train,
        coverage,
        grid_search_ease,
        grid_search_py as grid_search,
        hit_rate_at_k,
        load_model,
        mean_average_precision,
        ndcg_at_k,
        precision_at_k,
        random_search_ease,
        random_search_py as random_search,
        recall_at_k,
        validate_data,
        random_split,
        temporal_split,
        leave_k_out_split,
    )
    _HAS_NATIVE = True
except ImportError:
    _HAS_NATIVE = False

from kzn_recsys.schemas import EngagementSchema, MetadataSchema
```

Move the `from kzn_recsys.fease_wrapper import (...)` import inside the `try` block as well (it depends on native split functions). Then guard the `__all__` additions: build `__all__` starting with always-available names (`EngagementSchema`, `MetadataSchema`), and extend with the native names only `if _HAS_NATIVE:`. Mirror the existing `if _HAS_ML_MODELS:` / `if _HAS_ONNX:` conditional-append structure already in the file (lines 79-101).

- **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/spark/test_optional_native.py -v`
Expected: PASS (2 passed).

- **Step 5: Verify the native path still works**

Run: `.venv/bin/maturin develop && .venv/bin/python -c "import kzn_recsys; print(kzn_recsys._HAS_NATIVE, kzn_recsys.FeaseModel)"`
Expected: prints `True <class 'builtins.FeaseModel'>` and existing `tests/test_model.py` still passes:
`.venv/bin/python -m pytest tests/test_model.py -q`

- **Step 6: Commit**

```bash
git add kzn_recsys/__init__.py tests/spark/test_optional_native.py
git commit -m "feat: import kzn_recsys without the native extension (_HAS_NATIVE)"
```

## Task 17: Pure-Python wheel build (auxiliary)

**Files:**
- Create: `packaging/pure-python/README.md` (documents the build)
- Create: `packaging/pure-python/pyproject.toml` (setuptools-based, pure-Python)

The primary `pyproject.toml` uses the maturin backend, which always compiles a native wheel. To ship a `py3-none-any` wheel for restricted environments, build a separate pure-Python distribution from the same sources. This task documents and configures that auxiliary build; it does not replace the maturin wheel.

- **Step 1: Create the auxiliary pure-Python pyproject**

Create `packaging/pure-python/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "kzn_recsys_spark"
version = "0.1.0"
description = "Pure-Python/PySpark EASE implementation (kzn_recsys.spark), no native extension"
requires-python = ">=3.8"
dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
    "pyspark>=3.4",
]

[tool.setuptools]
# Ship only the pure-Python spark subpackage + the (native-optional) package init.
packages = ["kzn_recsys", "kzn_recsys.spark"]
package-dir = {"" = "../.."}

[tool.setuptools.package-data]
"kzn_recsys" = ["schemas.py", "fease_wrapper.py"]
```

- **Step 2: Document the build**

Create `packaging/pure-python/README.md`:

```markdown
# Pure-Python wheel (`kzn_recsys.spark`)

The main wheel is built by maturin and contains the compiled Rust extension.
For environments where that native wheel cannot be installed, build this
pure-Python distribution, which ships only `kzn_recsys.spark` (NumPy/SciPy/
PySpark EASE) plus the native-optional `kzn_recsys/__init__.py`.

## Build

```bash
cd packaging/pure-python
python -m build --wheel
# -> dist/kzn_recsys_spark-0.1.0-py3-none-any.whl
```

## Install (restricted environment)

```bash
pip install kzn_recsys_spark-0.1.0-py3-none-any.whl
python -c "from kzn_recsys.spark import build_and_train; print('ok')"
```

The two distributions are intentionally separate: `kzn_recsys` (maturin,
native) and `kzn_recsys_spark` (pure-Python). They share the `kzn_recsys`
import namespace; do not install both into the same environment.
```

- **Step 3: Build and verify the wheel**

Run:
```bash
.venv/bin/python -m pip install build
cd packaging/pure-python && ../../.venv/bin/python -m build --wheel
```
Expected: produces `dist/kzn_recsys_spark-0.1.0-py3-none-any.whl`. Confirm the filename ends in `py3-none-any.whl` (platform-independent).

- **Step 4: Commit**

```bash
git add packaging/pure-python/pyproject.toml packaging/pure-python/README.md
git commit -m "build(spark): auxiliary pure-Python wheel for restricted environments"
```

---

## Final verification

- **Run the full Spark suite (native present)**

Run: `.venv/bin/maturin develop && .venv/bin/python -m pytest tests/spark/ -v`
Expected: all tests pass, including the `parity` tests.

- **Run the Spark-free units only (simulating no wheel)**

Run: `.venv/bin/python -m pytest tests/spark/test_ease_core.py tests/spark/test_feas_codec.py tests/spark/test_metrics.py -v`
Expected: all pass without importing `_native` or booting Spark.

- **Confirm existing native tests are unaffected**

Run: `.venv/bin/python -m pytest tests/test_model.py -q`
Expected: pre-existing EASE tests still pass.

---

## Notes for the implementer

- **Parity tolerance:** scores compared at `1e-5` relative; rankings compared by item string-id. Score ties can reorder top-K between impls — if a parity test flips on a tie, assert on the *set* of top-K ids and on per-id scores, not on exact ordinal position.
- **Column-major discipline:** every place S crosses a boundary (train output, codec read/write, model load), it must stay Fortran-order. The tests assert `flags["F_CONTIGUOUS"]` at the boundaries — keep those assertions.
- **Native predict signature:** Task 10 assumes `FeaseModel.predict(interactions_dict, features_dict, top_k=...)`. Verify against the installed extension before running; adjust the parity calls if the signature differs.
- **Spark test speed:** the session fixture boots one JVM for the whole module. Keep Spark tests under the `spark` marker so the Spark-free core can be run quickly in isolation.
```
