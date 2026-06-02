# ONNX Export for FEASE (EASE) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional `kzn_recsys.export_onnx(model, …)` that turns a trained EASE `FeaseModel` into a portable ONNX artifact (scoring + repeat-penalty + eligibility mask + TopK + raw affinity), a `vocab.json` sidecar, and an optional MLflow pyfunc model, verified for parity on both `onnxruntime` (Python) and `ort` (Rust).

**Architecture:** A thin Rust seam (`FeaseModel.export_payload()`) hands the S-matrix bytes + metadata to Python; a Python package `kzn_recsys/onnx_export/` authors the graph with the canonical `onnx` library, writes the sidecar, quantizes, and builds the MLflow wrapper. ONNX deps are an optional `[onnx]` extra; `ort` is a Rust dev-dependency. See spec `docs/superpowers/specs/2026-06-01-onnx-export-design.md`.

**Tech Stack:** Rust (PyO3 0.27, nalgebra), Python (`onnx`, `onnxruntime`, `onnxconverter-common`, `mlflow`, `numpy`), `ort` (Rust ONNX Runtime). Build via maturin; env managed by `uv`.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/onnx_export.rs` | Pure-Rust helper: extract S_items as row-major LE f64 bytes | Create |
| `src/lib.rs` | Register `mod onnx_export`; add `FeaseModel.export_payload()` PyO3 method | Modify |
| `Cargo.toml` | Add `ort` dev-dependency | Modify |
| `pyproject.toml` | Add `[project.optional-dependencies] onnx` | Modify |
| `kzn_recsys/onnx_export/__init__.py` | Public `export_onnx`, dataclasses, dispatch, validation, constants | Create |
| `kzn_recsys/onnx_export/_graph.py` | Build the ONNX graph (Gemm + seen-union + repeat + mask + TopK + raw_scores) | Create |
| `kzn_recsys/onnx_export/_vocab.py` | Write `vocab.json` incl. `io_signature` | Create |
| `kzn_recsys/onnx_export/_quantize.py` | fp16 / int8 post-processing | Create |
| `kzn_recsys/onnx_export/_mlflow.py` | `FeaseOnnxPyfunc` + `build_mlflow` | Create |
| `kzn_recsys/__init__.py` | `_HAS_ONNX` gating; add `export_onnx` to `__all__` | Modify |
| `tests/test_onnx_export.py` | Python parity + behavior tests | Create |
| `tests/fixtures/` | Committed `fixture.onnx`, `inputs.json`, `expected.json` for the Rust parity test | Create |

**Conventions (verified against the repo):**
- Build/install: `.venv/bin/maturin develop` (the `.venv` uses Python 3.14; always target it).
- Python tests: `.venv/bin/python -m pytest tests/test_onnx_export.py -v`.
- Rust tests live **inside** `src/*.rs` as `#[cfg(test)] mod tests` (the crate is `crate-type = ["cdylib"]`, so external `tests/*.rs` integration tests cannot link it). Run with `cargo test`.
- The `.venv` has **no pip**; install Python packages with `uv pip install --python .venv/bin/python …`.
- Public Python package is `kzn_recsys`; the compiled extension is `kzn_recsys._native`.

---

## Task 0: Environment & dependency wiring

**Files:**
- Modify: `pyproject.toml`
- Modify: `Cargo.toml`

- [ ] **Step 1: Add the `onnx` optional extra to `pyproject.toml`**

Insert after the `dependencies = [...]` block in `[project]`:

```toml
[project.optional-dependencies]
onnx = [
    "numpy>=1.24",
    "onnx>=1.16",
    "onnxruntime>=1.18",
    "onnxconverter-common>=1.14",
    "mlflow>=2.12",
]
```

- [ ] **Step 2: Add `ort` as a Rust dev-dependency in `Cargo.toml`**

Replace the empty `[dev-dependencies]` section at the end of `Cargo.toml` with:

```toml
[dev-dependencies]
# ONNX Runtime bindings — used ONLY by the in-crate ONNX parity test
# (src/onnx_export.rs #[cfg(test)]). Not in the default or wheel dependency
# graph. `download-binaries` lets ort fetch a prebuilt ONNX Runtime so CI needs
# no system install.
ort = { version = "2.0.0-rc.10", default-features = false, features = ["download-binaries", "ndarray"] }
ndarray = "0.16"
```

- [ ] **Step 3: Install the onnx extra into the venv**

Run: `uv pip install --python .venv/bin/python "onnx>=1.16" "onnxruntime>=1.18" "onnxconverter-common>=1.14" "mlflow>=2.12" "numpy>=1.24"`
Expected: installs succeed. If a wheel is unavailable for Python 3.14, create a 3.11 venv for the ONNX test runs and note it; the wheel itself is Python-version-agnostic (abi3) so the Rust/maturin build is unaffected.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml Cargo.toml
git commit -m "build: add onnx optional extra and ort dev-dependency"
```

---

## Task 1: Rust helper — S_items as row-major LE f64 bytes

**Files:**
- Create: `src/onnx_export.rs`
- Modify: `src/lib.rs` (register the module)
- Test: `src/onnx_export.rs` (`#[cfg(test)]`)

- [ ] **Step 1: Create `src/onnx_export.rs` with the helper and a failing test**

```rust
//! Pure-Rust helper for the ONNX export seam.
//!
//! Extracts the EASE `S_items` sub-block (the first `M` rows of `S`) in the
//! exact byte layout the Python ONNX authoring layer expects: row-major,
//! little-endian `f64`. nalgebra stores `s_matrix` column-major, so we walk it
//! row-by-row; the Python side then does
//! `np.frombuffer(bytes, dtype="<f8").reshape(M, M + K)` with no transpose.

use crate::model::RustFeaseModel;

/// Returns `(bytes, rows, cols)` where `bytes` is the row-major little-endian
/// `f64` encoding of `S[0..M, 0..M+K]`, `rows == num_items` (M) and
/// `cols == num_items + num_user_features` (M + K).
pub fn s_items_row_major_le_bytes(model: &RustFeaseModel) -> (Vec<u8>, usize, usize) {
    let rows = model.num_items;
    let cols = model.num_items + model.num_user_features;
    let mut bytes = Vec::with_capacity(rows * cols * 8);
    for r in 0..rows {
        for c in 0..cols {
            bytes.extend_from_slice(&model.s_matrix[(r, c)].to_le_bytes());
        }
    }
    (bytes, rows, cols)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_pipeline::Mappings;
    use nalgebra::DMatrix;

    fn dummy_mappings() -> Mappings {
        Mappings {
            user_to_idx: Default::default(),
            idx_to_user: Default::default(),
            item_to_idx: Default::default(),
            idx_to_item: Default::default(),
            user_feature_to_idx: Default::default(),
            idx_to_user_feature: Default::default(),
            item_feature_to_idx: Default::default(),
            idx_to_item_feature: Default::default(),
        }
    }

    #[test]
    fn s_items_bytes_are_row_major_le_and_subset() {
        // M = 2 items, K = 1 user feature → S is 3x3, S_items is the first 2 rows (2x3).
        let m = 2;
        let k = 1;
        let total = m + k;
        let mut s = DMatrix::<f64>::zeros(total, total);
        // Distinct values so row-major order is observable.
        for r in 0..total {
            for c in 0..total {
                s[(r, c)] = (r * 10 + c) as f64;
            }
        }
        let model = RustFeaseModel {
            s_matrix: s,
            num_items: m,
            num_user_features: k,
            num_item_features: 0,
            alpha: 1.0,
            beta: 1.0,
            lambda_: 10.0,
            meta_weight: 0.0,
            mappings: dummy_mappings(),
            weighting_config: None,
        };

        let (bytes, rows, cols) = s_items_row_major_le_bytes(&model);
        assert_eq!((rows, cols), (2, 3));
        assert_eq!(bytes.len(), 2 * 3 * 8);

        // Decode and check row-major order: [00,01,02, 10,11,12]
        let decoded: Vec<f64> = bytes
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(decoded, vec![0.0, 1.0, 2.0, 10.0, 11.0, 12.0]);
    }
}
```

- [ ] **Step 2: Register the module in `src/lib.rs`**

Find the module declarations near the top of `src/lib.rs` (the `mod …;` lines) and add:

```rust
mod onnx_export;
```

- [ ] **Step 3: Run the Rust test to verify it passes**

Run: `cargo test --lib onnx_export`
Expected: `s_items_bytes_are_row_major_le_and_subset ... ok`

- [ ] **Step 4: Commit**

```bash
git add src/onnx_export.rs src/lib.rs
git commit -m "feat(onnx): row-major LE byte extractor for S_items"
```

---

## Task 2: Rust seam — `FeaseModel.export_payload()`

**Files:**
- Modify: `src/lib.rs` (add a method to the `#[pymethods] impl FeaseModel` block)
- Test: `tests/test_onnx_export.py`

- [ ] **Step 1: Write the failing Python test for the payload**

Create `tests/test_onnx_export.py`:

```python
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import kzn_recsys as fease


@pytest.fixture(scope="module")
def trained_model():
    """Tiny trained EASE model (mirrors tests/test_model.py fixture shape)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        i_path = Path(tmpdir) / "interactions.parquet"
        u_path = Path(tmpdir) / "user_features.parquet"
        t_path = Path(tmpdir) / "item_features.parquet"
        pl.DataFrame(
            {"user_id": ["u0", "u0", "u1"], "item_id": ["G0", "G2", "G1"], "value": [5.0, 4.0, 6.0]}
        ).write_parquet(i_path)
        pl.DataFrame(
            {
                "user_id": ["u0", "u0", "u1", "u1", "u2", "u2"],
                "feature_name": ["device_Mobile", "region_US", "device_Mobile", "region_EMEA", "device_Console", "region_APAC"],
                "value": [1.0] * 6,
            }
        ).write_parquet(u_path)
        pl.DataFrame(
            {
                "item_id": ["G0", "G1", "G2", "G3"],
                "feature_name": ["genre_Action", "genre_Comedy", "genre_Action", "genre_Comedy"],
                "value": [1.0] * 4,
            }
        ).write_parquet(t_path)
        yield fease.build_and_train(
            interactions_path=str(i_path),
            user_features_path=str(u_path),
            item_features_path=str(t_path),
            alpha=1.0,
            beta=1.0,
            lambda_=10.0,
            meta_weight=0.0,
        )


def test_export_payload_shapes_and_fields(trained_model):
    d = trained_model.export_payload()
    assert d["kind"] == "ease"
    m, cols = d["s_items_shape"]
    assert cols == d["num_items"] + d["num_user_features"]
    assert m == d["num_items"]
    s = np.frombuffer(d["s_items_bytes"], dtype="<f8").reshape(m, cols)
    assert s.shape == (m, cols)
    # Diagonal of the item-item block is ~zero (EASE zero-diagonal constraint).
    assert np.allclose(np.diag(s[:, :m]), 0.0, atol=1e-6)
    assert isinstance(d["item_index_to_guid"], list) and len(d["item_index_to_guid"]) == m
    assert d["sparsity_threshold"] is None  # no weighting config used
    assert set(d["feature_name_to_index"].values()) == set(range(d["num_user_features"]))
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `.venv/bin/python -m pytest tests/test_onnx_export.py::test_export_payload_shapes_and_fields -v`
Expected: FAIL — `AttributeError: 'FeaseModel' object has no attribute 'export_payload'`

- [ ] **Step 3: Add the `export_payload` method to `FeaseModel` in `src/lib.rs`**

Add these imports to the `use pyo3::types::…` line at the top of `src/lib.rs` (extend the existing import; `PyDict`, `PyList`, `PyString`, `PyFloat` are already imported — add `PyBytes`):

```rust
use pyo3::types::PyBytes;
```

Inside the `#[pymethods] impl FeaseModel { … }` block (the same block that holds `predict`), add:

```rust
/// Returns everything the Python ONNX exporter needs to author the graph:
/// the raw `S_items` weight bytes (row-major little-endian f64) plus its
/// shape, the model hyperparameters, and the id<->index mappings.
///
/// The returned `s_items_bytes` is the RAW `S[0:M, :]` (NOT yet β-folded);
/// the Python layer folds β into the user-feature columns when it builds the
/// graph, keeping that transform testable in one place.
fn export_payload<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    use crate::models::{ModelKind, RecModel};

    let (s_bytes, rows, cols) = crate::onnx_export::s_items_row_major_le_bytes(&self.model);

    let kind = match crate::models::EaseAdapterRef::new(&self.model).kind() {
        ModelKind::Ease => "ease",
        ModelKind::SasRec => "sasrec",
        ModelKind::TwoTower => "two_tower",
    };

    let d = PyDict::new(py);
    d.set_item("kind", kind)?;
    d.set_item("s_items_bytes", PyBytes::new(py, &s_bytes))?;
    d.set_item("s_items_shape", (rows, cols))?;
    d.set_item("beta", self.model.beta)?;
    d.set_item("num_items", self.model.num_items)?;
    d.set_item("num_user_features", self.model.num_user_features)?;
    d.set_item("num_item_features", self.model.num_item_features)?;
    d.set_item("alpha", self.model.alpha)?;
    d.set_item("lambda_", self.model.lambda_)?;
    d.set_item("meta_weight", self.model.meta_weight)?;
    // None when weighting was not used during training (kept distinct from 0.0).
    let sparsity: Option<f64> = self
        .model
        .weighting_config
        .as_ref()
        .map(|w| w.sparsity_threshold);
    d.set_item("sparsity_threshold", sparsity)?;
    d.set_item("item_index_to_guid", self.model.mappings.idx_to_item.clone())?;

    let feat = PyDict::new(py);
    for (name, idx) in self.model.mappings.user_feature_to_idx.iter() {
        feat.set_item(name, *idx)?;
    }
    d.set_item("feature_name_to_index", feat)?;

    Ok(d)
}
```

- [ ] **Step 4: Rebuild the extension and run the test**

Run: `.venv/bin/maturin develop && .venv/bin/python -m pytest tests/test_onnx_export.py::test_export_payload_shapes_and_fields -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs tests/test_onnx_export.py
git commit -m "feat(onnx): FeaseModel.export_payload PyO3 seam"
```

---

## Task 3: Python package scaffold + dispatch + `_HAS_ONNX` gating

**Files:**
- Create: `kzn_recsys/onnx_export/__init__.py`
- Modify: `kzn_recsys/__init__.py`
- Test: `tests/test_onnx_export.py`

- [ ] **Step 1: Write the failing dispatch tests**

Append to `tests/test_onnx_export.py`:

```python
from kzn_recsys.onnx_export import (
    ExportPayload,
    EXCLUDE_SENTINEL,
    MASK_PENALTY,
    OPSET,
    _payload_from_model,
    _validate_exportable,
)


def test_payload_from_model_builds_dataclass(trained_model):
    p = _payload_from_model(trained_model)
    assert isinstance(p, ExportPayload)
    assert p.kind == "ease"
    assert p.s_items.shape == (p.num_items, p.num_items + p.num_user_features)


def test_validate_exportable_rejects_nan(trained_model):
    p = _payload_from_model(trained_model)
    p.s_items[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        _validate_exportable(p)


def test_constants():
    assert EXCLUDE_SENTINEL == 1e9
    assert MASK_PENALTY == 1e9
    assert OPSET == 17
```

- [ ] **Step 2: Run to confirm failure**

Run: `.venv/bin/python -m pytest tests/test_onnx_export.py -k "payload_from_model or validate_exportable or constants" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kzn_recsys.onnx_export'`

- [ ] **Step 3: Create `kzn_recsys/onnx_export/__init__.py`**

```python
"""Optional ONNX export for the EASE model. Requires the ``[onnx]`` extra."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

# Graph/runtime constants (see spec §13 glossary).
MASK_PENALTY = 1e9
EXCLUDE_SENTINEL = 1e9
OPSET = 17


@dataclasses.dataclass
class ExportPayload:
    kind: str
    s_items: np.ndarray  # (M, M+K) float64, row-major; RAW S[0:M,:] (not β-folded)
    beta: float
    num_items: int
    num_user_features: int
    num_item_features: int
    alpha: float
    lambda_: float
    meta_weight: float
    sparsity_threshold: float | None
    item_index_to_guid: list[str]
    feature_name_to_index: dict[str, int]


@dataclasses.dataclass
class ExportResult:
    onnx_path: Path
    vocab_path: Path
    mlflow_path: Path | None = None


def _payload_from_model(model) -> ExportPayload:
    d = model.export_payload()
    if d["kind"] != "ease":
        raise NotImplementedError(f"ONNX export not yet supported for {d['kind']}")
    rows, cols = d["s_items_shape"]
    s = np.frombuffer(d["s_items_bytes"], dtype="<f8").reshape(rows, cols).copy()
    return ExportPayload(
        kind=d["kind"],
        s_items=s,
        beta=float(d["beta"]),
        num_items=int(d["num_items"]),
        num_user_features=int(d["num_user_features"]),
        num_item_features=int(d["num_item_features"]),
        alpha=float(d["alpha"]),
        lambda_=float(d["lambda_"]),
        meta_weight=float(d["meta_weight"]),
        sparsity_threshold=d["sparsity_threshold"],
        item_index_to_guid=list(d["item_index_to_guid"]),
        feature_name_to_index=dict(d["feature_name_to_index"]),
    )


def _validate_exportable(payload: ExportPayload) -> None:
    """Mirror RustFeaseModel::validate() — refuse to export a bad model."""
    if payload.num_items == 0:
        raise ValueError("Cannot export a model with zero items")
    if np.isnan(payload.s_items).any():
        raise ValueError("S matrix contains NaN values; refusing to export")
    if np.isinf(payload.s_items).any():
        raise ValueError("S matrix contains Inf values; refusing to export")
    if not np.any(payload.s_items):
        raise ValueError("S matrix is all zeros; model may not have learned")


def export_onnx(
    model,
    output_dir,
    *,
    top_k_default: int = 100,
    dtype: str = "fp32",
    repeat_penalty_default="exclude",
    mlflow: bool = False,
) -> ExportResult:
    """Export a trained EASE ``FeaseModel`` to ONNX + sidecar (+ optional MLflow).

    See ``docs/superpowers/specs/2026-06-01-onnx-export-design.md``.
    """
    if top_k_default <= 0:
        raise ValueError("top_k_default must be positive")
    if dtype not in ("fp32", "fp16", "int8"):
        raise ValueError(f"dtype must be fp32|fp16|int8, got {dtype!r}")

    payload = _payload_from_model(model)  # raises NotImplementedError for non-ease
    _validate_exportable(payload)

    rp_default = (
        EXCLUDE_SENTINEL
        if repeat_penalty_default == "exclude"
        else float(repeat_penalty_default)
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"
    vocab_path = output_dir / "vocab.json"

    from ._graph import build_graph

    build_graph(payload, onnx_path, top_k_default=top_k_default, repeat_penalty_default=rp_default)

    if dtype != "fp32":
        from ._quantize import quantize

        quantize(onnx_path, dtype)

    from ._vocab import write_vocab

    write_vocab(
        payload,
        vocab_path,
        top_k_default=top_k_default,
        dtype=dtype,
        repeat_penalty_default=rp_default,
    )

    mlflow_path = None
    if mlflow:
        from ._mlflow import build_mlflow

        mlflow_path = build_mlflow(onnx_path, vocab_path, output_dir / "mlflow_model")

    return ExportResult(onnx_path=onnx_path, vocab_path=vocab_path, mlflow_path=mlflow_path)
```

- [ ] **Step 4: Gate `export_onnx` into `kzn_recsys/__init__.py`**

After the existing `_HAS_ML_MODELS` try/except block (around line 49), add:

```python
try:  # pragma: no cover - import guard, exercised by build matrix
    from kzn_recsys.onnx_export import export_onnx  # noqa: F401

    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False
```

Then, after the existing `if _HAS_ML_MODELS:` block that extends `__all__`, add:

```python
if _HAS_ONNX:
    __all__ += ["export_onnx"]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_onnx_export.py -k "payload_from_model or validate_exportable or constants" -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add kzn_recsys/onnx_export/__init__.py kzn_recsys/__init__.py tests/test_onnx_export.py
git commit -m "feat(onnx): export_onnx dispatch, dataclasses, _HAS_ONNX gating"
```

---

## Task 4: Build the ONNX graph (fp32) + raw-score parity

**Files:**
- Create: `kzn_recsys/onnx_export/_graph.py`
- Test: `tests/test_onnx_export.py`

- [ ] **Step 1: Write the failing parity test**

Append to `tests/test_onnx_export.py`:

```python
import onnxruntime as ort


def _build_inputs(payload, interactions_idx_val, feature_idx_val):
    """Dense fp32 input vectors for one user."""
    M, K = payload.num_items, payload.num_user_features
    inter = np.zeros((1, M), np.float32)
    for idx, val in interactions_idx_val:
        inter[0, idx] = val
    feat = np.zeros((1, K), np.float32)
    for idx, val in feature_idx_val:
        feat[0, idx] = val
    return inter, feat


def _ref_scores(payload, inter, feat):
    """Reference raw affinity: β-folded S_items @ [interactions | features]."""
    M = payload.num_items
    W = payload.s_items.copy()
    if payload.num_user_features > 0:
        W[:, M:] *= payload.beta
    z = np.concatenate([inter[0], feat[0]]).astype(np.float64)
    return (W @ z).astype(np.float32)


def test_graph_raw_scores_parity(trained_model, tmp_path):
    from kzn_recsys.onnx_export import _payload_from_model, export_onnx

    payload = _payload_from_model(trained_model)
    res = export_onnx(trained_model, tmp_path)
    sess = ort.InferenceSession(str(res.onnx_path))

    inter, feat = _build_inputs(payload, [(0, 5.0)], [(1, 1.0)])
    out = sess.run(
        None,
        {
            "interactions": inter,
            "features": feat,
            "mask": np.ones((1, payload.num_items), np.float32),
            "seen": np.zeros((1, payload.num_items), np.float32),
            "repeat_penalty": np.array([[0.0]], np.float32),  # neutral → raw == adjusted
            "k": np.array([payload.num_items], np.int64),
        },
    )
    names = [o.name for o in sess.get_outputs()]
    raw = out[names.index("raw_scores")][0]
    np.testing.assert_allclose(raw, _ref_scores(payload, inter, feat), rtol=1e-5, atol=1e-5)
```

- [ ] **Step 2: Run to confirm failure**

Run: `.venv/bin/python -m pytest tests/test_onnx_export.py::test_graph_raw_scores_parity -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kzn_recsys.onnx_export._graph'`

- [ ] **Step 3: Create `kzn_recsys/onnx_export/_graph.py`**

```python
"""Authoring of the EASE ONNX graph (vanilla ai.onnx, opset 17)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from . import MASK_PENALTY, OPSET


def build_graph(payload, onnx_path: Path, *, top_k_default: int, repeat_penalty_default: float) -> None:
    M = payload.num_items
    K = payload.num_user_features

    # β-fold: pre-multiply the user-feature columns of S_items by β so the graph
    # consumes raw feature values. W is the Gemm weight, stored [M, M+K].
    W = payload.s_items.astype(np.float64).copy()
    if K > 0:
        W[:, M:] *= payload.beta
    W = W.astype(np.float32)

    initializers = [
        numpy_helper.from_array(W, name="W"),
        numpy_helper.from_array(np.ones((1, M), np.float32), name="mask"),
        numpy_helper.from_array(np.zeros((1, M), np.float32), name="seen"),
        numpy_helper.from_array(np.array([[repeat_penalty_default]], np.float32), name="repeat_penalty"),
        numpy_helper.from_array(np.array([top_k_default], np.int64), name="k"),
        numpy_helper.from_array(np.array(0.0, np.float32), name="zero_const"),
        numpy_helper.from_array(np.array(1.0, np.float32), name="one_const"),
        numpy_helper.from_array(np.array(MASK_PENALTY, np.float32), name="mask_penalty_const"),
        numpy_helper.from_array(np.array([M], np.int64), name="M_const"),
    ]

    inputs = [
        helper.make_tensor_value_info("interactions", TensorProto.FLOAT, ["batch", M]),
        helper.make_tensor_value_info("features", TensorProto.FLOAT, ["batch", K]),
        helper.make_tensor_value_info("mask", TensorProto.FLOAT, ["batch", M]),
        helper.make_tensor_value_info("seen", TensorProto.FLOAT, ["batch", M]),
        helper.make_tensor_value_info("repeat_penalty", TensorProto.FLOAT, ["batch", 1]),
        helper.make_tensor_value_info("k", TensorProto.INT64, [1]),
    ]
    outputs = [
        helper.make_tensor_value_info("top_indices", TensorProto.INT64, ["batch", "kc"]),
        helper.make_tensor_value_info("top_scores", TensorProto.FLOAT, ["batch", "kc"]),
        helper.make_tensor_value_info("raw_scores", TensorProto.FLOAT, ["batch", M]),
    ]

    nodes = [
        helper.make_node("Concat", ["interactions", "features"], ["z"], axis=-1),
        # raw_scores = z @ Wᵀ  (also a graph output)
        helper.make_node("Gemm", ["z", "W"], ["raw_scores"], transB=1),
        # seen_eff = max(seen, cast(interactions != 0))
        helper.make_node("Equal", ["interactions", "zero_const"], ["is_zero"]),
        helper.make_node("Not", ["is_zero"], ["is_nonzero"]),
        helper.make_node("Cast", ["is_nonzero"], ["nz_f"], to=TensorProto.FLOAT),
        helper.make_node("Max", ["seen", "nz_f"], ["seen_eff"]),
        # adjusted = raw_scores - repeat_penalty * seen_eff
        helper.make_node("Mul", ["repeat_penalty", "seen_eff"], ["penalty_term"]),
        helper.make_node("Sub", ["raw_scores", "penalty_term"], ["adjusted"]),
        # masked = adjusted + (mask - 1) * MASK_PENALTY
        helper.make_node("Sub", ["mask", "one_const"], ["mask_minus_one"]),
        helper.make_node("Mul", ["mask_minus_one", "mask_penalty_const"], ["mask_term"]),
        helper.make_node("Add", ["adjusted", "mask_term"], ["masked"]),
        # kc = min(k, M); TopK
        helper.make_node("Min", ["k", "M_const"], ["kc"]),
        helper.make_node(
            "TopK", ["masked", "kc"], ["top_scores", "top_indices"], axis=-1, largest=1, sorted=1
        ),
    ]

    graph = helper.make_graph(nodes, "ease_onnx", inputs, outputs, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 9  # compatible with onnxruntime >= 1.18
    onnx.checker.check_model(model)
    onnx.save(model, str(onnx_path))
```

- [ ] **Step 4: Run the parity test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_onnx_export.py::test_graph_raw_scores_parity -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add kzn_recsys/onnx_export/_graph.py tests/test_onnx_export.py
git commit -m "feat(onnx): build EASE graph (Gemm+seen+repeat+mask+TopK+raw_scores)"
```

---

## Task 5: Repeat-penalty, seen-union, mask, TopK behavior tests

**Files:**
- Test: `tests/test_onnx_export.py`

These exercise the graph from Task 4. No new implementation expected; if a test fails, fix `_graph.py` minimally.

- [ ] **Step 1: Write the behavior tests**

Append to `tests/test_onnx_export.py`:

```python
def _run(sess, payload, inter, feat, *, mask=None, seen=None, rp=0.0, k=None):
    M = payload.num_items
    feeds = {
        "interactions": inter,
        "features": feat,
        "mask": np.ones((1, M), np.float32) if mask is None else mask,
        "seen": np.zeros((1, M), np.float32) if seen is None else seen,
        "repeat_penalty": np.array([[rp]], np.float32),
        "k": np.array([M if k is None else k], np.int64),
    }
    out = sess.run(None, feeds)
    names = [o.name for o in sess.get_outputs()]
    return {n: out[i] for i, n in enumerate(names)}


def test_default_excludes_seen(trained_model, tmp_path):
    from kzn_recsys.onnx_export import _payload_from_model, export_onnx

    payload = _payload_from_model(trained_model)
    sess = ort.InferenceSession(str(export_onnx(trained_model, tmp_path).onnx_path))
    inter, feat = _build_inputs(payload, [(0, 5.0)], [])
    # Default repeat_penalty (exclude sentinel) baked in → omit it to get default.
    out = sess.run(
        None,
        {
            "interactions": inter,
            "features": feat,
            "mask": np.ones((1, payload.num_items), np.float32),
            "seen": np.zeros((1, payload.num_items), np.float32),
            "repeat_penalty": np.array([[1e9]], np.float32),
            "k": np.array([payload.num_items], np.int64),
        },
    )
    names = [o.name for o in sess.get_outputs()]
    top_idx = out[names.index("top_indices")][0]
    # Item 0 was interacted → must not be the top recommendation.
    assert top_idx[0] != 0


def test_repeat_boost_surfaces_seen_item(trained_model, tmp_path):
    from kzn_recsys.onnx_export import _payload_from_model, export_onnx

    payload = _payload_from_model(trained_model)
    sess = ort.InferenceSession(str(export_onnx(trained_model, tmp_path).onnx_path))
    inter, feat = _build_inputs(payload, [(0, 5.0)], [])
    neutral = _run(sess, payload, inter, feat, rp=0.0)["top_scores"][0]
    boosted = _run(sess, payload, inter, feat, rp=-1e6)["top_scores"][0]
    # Boost lifts the seen item's score versus neutral.
    assert boosted.max() > neutral.max()


def test_seen_input_overrides_zero_value(trained_model, tmp_path):
    from kzn_recsys.onnx_export import _payload_from_model, export_onnx

    payload = _payload_from_model(trained_model)
    M = payload.num_items
    sess = ort.InferenceSession(str(export_onnx(trained_model, tmp_path).onnx_path))
    # Item 0 present as a key with value 0.0 → derived path treats it unseen…
    inter = np.zeros((1, M), np.float32)
    feat = np.zeros((1, payload.num_user_features), np.float32)
    derived = _run(sess, payload, inter, feat, rp=1e9)["top_indices"][0]
    assert 0 in derived.tolist()  # not excluded by derivation alone
    # …but an explicit seen marks it → excluded.
    seen = np.zeros((1, M), np.float32)
    seen[0, 0] = 1.0
    explicit = _run(sess, payload, inter, feat, seen=seen, rp=1e9)["top_indices"][0]
    assert 0 not in explicit.tolist()


def test_mask_overrides_boost(trained_model, tmp_path):
    from kzn_recsys.onnx_export import _payload_from_model, export_onnx

    payload = _payload_from_model(trained_model)
    M = payload.num_items
    sess = ort.InferenceSession(str(export_onnx(trained_model, tmp_path).onnx_path))
    inter, feat = _build_inputs(payload, [(0, 5.0)], [])
    mask = np.ones((1, M), np.float32)
    mask[0, 0] = 0.0  # exclude item 0 for compliance
    idx = _run(sess, payload, inter, feat, mask=mask, rp=-1e6)["top_indices"][0]
    assert 0 not in idx.tolist()  # mask wins even with a strong boost


def test_top_k_clamped(trained_model, tmp_path):
    from kzn_recsys.onnx_export import _payload_from_model, export_onnx

    payload = _payload_from_model(trained_model)
    M = payload.num_items
    sess = ort.InferenceSession(str(export_onnx(trained_model, tmp_path).onnx_path))
    inter, feat = _build_inputs(payload, [], [(0, 1.0)])
    out = _run(sess, payload, inter, feat, k=M + 50)
    assert out["top_indices"].shape[1] == M  # clamped to catalog size
```

- [ ] **Step 2: Run the behavior tests**

Run: `.venv/bin/python -m pytest tests/test_onnx_export.py -k "default_excludes or repeat_boost or seen_input or mask_overrides or top_k_clamped" -v`
Expected: PASS (5 tests). If any fail, fix `_graph.py` minimally and re-run.

- [ ] **Step 3: Commit**

```bash
git add tests/test_onnx_export.py
git commit -m "test(onnx): repeat penalty, seen-union, mask override, top-k clamp"
```

---

## Task 6: vocab.json sidecar (incl. io_signature) + K=0

**Files:**
- Create: `kzn_recsys/onnx_export/_vocab.py`
- Test: `tests/test_onnx_export.py`

- [ ] **Step 1: Write the failing vocab + K=0 tests**

Append to `tests/test_onnx_export.py`:

```python
import json


def test_vocab_contents(trained_model, tmp_path):
    from kzn_recsys.onnx_export import _payload_from_model, export_onnx

    payload = _payload_from_model(trained_model)
    res = export_onnx(trained_model, tmp_path, top_k_default=7)
    vocab = json.loads(res.vocab_path.read_text())

    assert vocab["format_version"] == 1
    assert vocab["model_kind"] == "ease"
    assert vocab["num_items"] == payload.num_items
    assert vocab["num_user_features"] == payload.num_user_features
    assert vocab["top_k_default"] == 7
    assert vocab["constants"] == {"MASK_PENALTY": 1e9, "EXCLUDE_SENTINEL": 1e9}
    assert vocab["repeat_policy"]["default_penalty"] == 1e9
    assert vocab["repeat_policy"]["per_user_table_present"] is False
    assert len(vocab["item_index_to_guid"]) == payload.num_items
    assert vocab["provenance"]["sparsity_threshold"] is None

    names = [i["name"] for i in vocab["io_signature"]["inputs"]]
    assert names == ["interactions", "features", "mask", "seen", "repeat_penalty", "k"]
    out_names = [o["name"] for o in vocab["io_signature"]["outputs"]]
    assert out_names == ["top_indices", "top_scores", "raw_scores"]


@pytest.fixture(scope="module")
def model_no_user_features():
    """Trained EASE model with K = 0 (no user features)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        i_path = Path(tmpdir) / "interactions.parquet"
        u_path = Path(tmpdir) / "user_features.parquet"
        t_path = Path(tmpdir) / "item_features.parquet"
        pl.DataFrame(
            {"user_id": ["u0", "u0", "u1"], "item_id": ["G0", "G2", "G1"], "value": [5.0, 4.0, 6.0]}
        ).write_parquet(i_path)
        # Empty user features → K = 0.
        pl.DataFrame({"user_id": [], "feature_name": [], "value": []}).write_parquet(u_path)
        pl.DataFrame(
            {"item_id": ["G0", "G1", "G2", "G3"], "feature_name": ["a", "a", "a", "a"], "value": [1.0] * 4}
        ).write_parquet(t_path)
        yield fease.build_and_train(
            interactions_path=str(i_path),
            user_features_path=str(u_path),
            item_features_path=str(t_path),
            alpha=1.0,
            beta=1.0,
            lambda_=10.0,
            meta_weight=0.0,
        )


def test_k0_uniform_signature(model_no_user_features, tmp_path):
    from kzn_recsys.onnx_export import _payload_from_model, export_onnx

    payload = _payload_from_model(model_no_user_features)
    assert payload.num_user_features == 0
    res = export_onnx(model_no_user_features, tmp_path)
    sess = ort.InferenceSession(str(res.onnx_path))
    in_names = [i.name for i in sess.get_inputs()]
    assert in_names == ["interactions", "features", "mask", "seen", "repeat_penalty", "k"]
    M = payload.num_items
    out = sess.run(
        None,
        {
            "interactions": np.zeros((1, M), np.float32),
            "features": np.zeros((1, 0), np.float32),  # width-0
            "mask": np.ones((1, M), np.float32),
            "seen": np.zeros((1, M), np.float32),
            "repeat_penalty": np.array([[0.0]], np.float32),
            "k": np.array([M], np.int64),
        },
    )
    names = [o.name for o in sess.get_outputs()]
    assert out[names.index("raw_scores")].shape == (1, M)
```

- [ ] **Step 2: Run to confirm failure**

Run: `.venv/bin/python -m pytest tests/test_onnx_export.py -k "vocab_contents or k0_uniform" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kzn_recsys.onnx_export._vocab'`

- [ ] **Step 3: Create `kzn_recsys/onnx_export/_vocab.py`**

```python
"""Writes the language-neutral vocab.json sidecar (spec §6)."""
from __future__ import annotations

import json
from pathlib import Path

from . import EXCLUDE_SENTINEL, MASK_PENALTY, OPSET


def write_vocab(payload, vocab_path: Path, *, top_k_default: int, dtype: str, repeat_penalty_default: float) -> None:
    M, K = payload.num_items, payload.num_user_features
    vocab = {
        "format_version": 1,
        "model_kind": payload.kind,
        "num_items": M,
        "num_user_features": K,
        "num_item_features": payload.num_item_features,
        "beta": payload.beta,
        "weight_dtype": dtype,
        "opset": OPSET,
        "top_k_default": top_k_default,
        "constants": {"MASK_PENALTY": MASK_PENALTY, "EXCLUDE_SENTINEL": EXCLUDE_SENTINEL},
        "repeat_policy": {
            "default_penalty": repeat_penalty_default,
            "per_user_table_present": False,
        },
        "io_signature": {
            "inputs": [
                {"name": "interactions", "dtype": "float32", "shape": ["batch", M], "required": True},
                {"name": "features", "dtype": "float32", "shape": ["batch", K], "required": True},
                {"name": "mask", "dtype": "float32", "shape": ["batch", M], "required": False, "default": "all-ones"},
                {"name": "seen", "dtype": "float32", "shape": ["batch", M], "required": False, "default": "all-zeros"},
                {"name": "repeat_penalty", "dtype": "float32", "shape": ["batch", 1], "required": False, "default": "EXCLUDE_SENTINEL"},
                {"name": "k", "dtype": "int64", "shape": [1], "required": False, "default": "top_k_default"},
            ],
            "outputs": [
                {"name": "top_indices", "dtype": "int64", "shape": ["batch", "kc"]},
                {"name": "top_scores", "dtype": "float32", "shape": ["batch", "kc"]},
                {"name": "raw_scores", "dtype": "float32", "shape": ["batch", M]},
            ],
        },
        "item_index_to_guid": payload.item_index_to_guid,
        "feature_name_to_index": payload.feature_name_to_index,
        "provenance": {
            "alpha": payload.alpha,
            "lambda_": payload.lambda_,
            "meta_weight": payload.meta_weight,
            "sparsity_threshold": payload.sparsity_threshold,
            "num_item_features": payload.num_item_features,
        },
    }
    vocab_path.write_text(json.dumps(vocab, indent=2))
```

- [ ] **Step 4: Run the vocab + K=0 tests**

Run: `.venv/bin/python -m pytest tests/test_onnx_export.py -k "vocab_contents or k0_uniform" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add kzn_recsys/onnx_export/_vocab.py tests/test_onnx_export.py
git commit -m "feat(onnx): vocab.json sidecar with io_signature; K=0 uniform signature"
```

---

## Task 7: Quantization (fp16 / int8)

**Files:**
- Create: `kzn_recsys/onnx_export/_quantize.py`
- Test: `tests/test_onnx_export.py`

- [ ] **Step 1: Write the failing quantization tests**

Append to `tests/test_onnx_export.py`:

```python
def _topk_set(sess, payload, kk=3):
    inter, feat = _build_inputs(payload, [], [(0, 1.0)])
    out = _run(sess, payload, inter, feat, rp=0.0, k=kk)
    return out["top_indices"][0].tolist()


@pytest.mark.parametrize("dtype", ["fp16", "int8"])
def test_quantized_rank_agreement(trained_model, tmp_path, dtype):
    from kzn_recsys.onnx_export import _payload_from_model, export_onnx

    payload = _payload_from_model(trained_model)
    fp32 = ort.InferenceSession(str(export_onnx(trained_model, tmp_path / "fp32").onnx_path))
    quant = ort.InferenceSession(str(export_onnx(trained_model, tmp_path / dtype, dtype=dtype).onnx_path))
    # Ranking is preserved even though scores shift under quantization.
    assert _topk_set(quant, payload, kk=3) == _topk_set(fp32, payload, kk=3)


def test_quantized_io_is_fp32(trained_model, tmp_path):
    from kzn_recsys.onnx_export import export_onnx

    sess = ort.InferenceSession(str(export_onnx(trained_model, tmp_path, dtype="fp16").onnx_path))
    assert sess.get_outputs()[[o.name for o in sess.get_outputs()].index("raw_scores")].type == "tensor(float)"
```

- [ ] **Step 2: Run to confirm failure**

Run: `.venv/bin/python -m pytest tests/test_onnx_export.py -k "quantized" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kzn_recsys.onnx_export._quantize'`

- [ ] **Step 3: Create `kzn_recsys/onnx_export/_quantize.py`**

```python
"""fp16 / int8 post-processing of the fp32 ONNX graph (spec §1 quantization)."""
from __future__ import annotations

from pathlib import Path

import onnx


def quantize(onnx_path: Path, dtype: str) -> None:
    """Rewrite ``onnx_path`` in place at the requested precision.

    IO (and the masking/TopK arithmetic) stay fp32 — only the weight and the
    matmul are reduced — so the output signature and the 1e9 sentinels remain
    safe (spec §4).
    """
    if dtype == "fp16":
        from onnxconverter_common import float16

        model = onnx.load(str(onnx_path))
        # keep_io_types=True preserves fp32 inputs/outputs; min_positive_val and
        # the default op blocklist keep TopK/Min in fp32.
        converted = float16.convert_float_to_float16(model, keep_io_types=True)
        onnx.save(converted, str(onnx_path))
    elif dtype == "int8":
        from onnxruntime.quantization import QuantType, quantize_dynamic

        tmp = onnx_path.with_suffix(".int8.onnx")
        quantize_dynamic(str(onnx_path), str(tmp), weight_type=QuantType.QInt8)
        tmp.replace(onnx_path)
    else:
        raise ValueError(f"unsupported quantization dtype: {dtype!r}")
```

- [ ] **Step 4: Run the quantization tests**

Run: `.venv/bin/python -m pytest tests/test_onnx_export.py -k "quantized" -v`
Expected: PASS (3 tests: fp16, int8, io-is-fp32)

- [ ] **Step 5: Commit**

```bash
git add kzn_recsys/onnx_export/_quantize.py tests/test_onnx_export.py
git commit -m "feat(onnx): optional fp16/int8 quantization with fp32 IO"
```

---

## Task 8: MLflow pyfunc wrapper

**Files:**
- Create: `kzn_recsys/onnx_export/_mlflow.py`
- Test: `tests/test_onnx_export.py`

- [ ] **Step 1: Write the failing MLflow roundtrip test**

Append to `tests/test_onnx_export.py`:

```python
import pandas as pd


def test_mlflow_roundtrip_guid_in_guid_out(trained_model, tmp_path):
    import mlflow.pyfunc

    from kzn_recsys.onnx_export import export_onnx

    res = export_onnx(trained_model, tmp_path, mlflow=True)
    assert res.mlflow_path is not None

    loaded = mlflow.pyfunc.load_model(str(res.mlflow_path))
    # u0 interacted with G0; default policy must exclude already-watched G0.
    df = pd.DataFrame(
        [{"interactions": {"G0": 5.0}, "features": {}, "exclude": [], "top_k": 3}]
    )
    out = loaded.predict(df)
    guids = list(out["item_guid"])
    assert "G0" not in guids  # excluded by default repeat policy
    assert len(guids) == 3
    assert {"user_row", "rank", "item_guid", "score"} <= set(out.columns)
```

- [ ] **Step 2: Run to confirm failure**

Run: `.venv/bin/python -m pytest tests/test_onnx_export.py::test_mlflow_roundtrip_guid_in_guid_out -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kzn_recsys.onnx_export._mlflow'`

- [ ] **Step 3: Create `kzn_recsys/onnx_export/_mlflow.py`**

This module is **self-contained** — it must NOT import `kzn_recsys._native`, so the served artifact depends only on `onnxruntime`, `numpy`, `pandas`, and `json`.

```python
"""MLflow pyfunc wrapper that serves the ONNX graph by GUID (spec §7)."""
from __future__ import annotations

import json
from pathlib import Path

import mlflow.pyfunc
import numpy as np
import pandas as pd


class FeaseOnnxPyfunc(mlflow.pyfunc.PythonModel):
    """Maps GUIDs↔indices around the numeric ONNX graph. No kzn_recsys import."""

    def load_context(self, context):
        import onnxruntime as ort

        self._vocab = json.loads(Path(context.artifacts["vocab"]).read_text())
        self._sess = ort.InferenceSession(context.artifacts["onnx"])
        self._M = self._vocab["num_items"]
        self._K = self._vocab["num_user_features"]
        self._idx_to_guid = self._vocab["item_index_to_guid"]
        self._guid_to_idx = {g: i for i, g in enumerate(self._idx_to_guid)}
        self._feat_to_idx = self._vocab["feature_name_to_index"]
        self._default_rp = self._vocab["repeat_policy"]["default_penalty"]
        self._default_k = self._vocab["top_k_default"]

    def predict(self, context, model_input, params=None):
        rows = model_input.to_dict(orient="records") if isinstance(model_input, pd.DataFrame) else list(model_input)
        frames = []
        for r, row in enumerate(rows):
            frames.append(self._predict_one(r, row))
        return pd.concat(frames, ignore_index=True)

    def _predict_one(self, row_id, row):
        M, K = self._M, self._K
        inter = np.zeros((1, M), np.float32)
        seen = np.zeros((1, M), np.float32)
        for guid, val in (row.get("interactions") or {}).items():
            idx = self._guid_to_idx.get(guid)
            if idx is None:
                continue  # unknown GUID → skip (catalogs drift)
            inter[0, idx] = float(val)
            seen[0, idx] = 1.0  # key-based "seen", matching Rust semantics
        feat = np.zeros((1, K), np.float32)
        for name, val in (row.get("features") or {}).items():
            idx = self._feat_to_idx.get(name)
            if idx is not None:
                feat[0, idx] = float(val)
        mask = np.ones((1, M), np.float32)
        for guid in (row.get("exclude") or []):
            idx = self._guid_to_idx.get(guid)
            if idx is not None:
                mask[0, idx] = 0.0
        rp = float(row["repeat_penalty"]) if row.get("repeat_penalty") is not None else self._default_rp
        k = int(row["top_k"]) if row.get("top_k") is not None else self._default_k

        out = self._sess.run(
            None,
            {
                "interactions": inter,
                "features": feat,
                "mask": mask,
                "seen": seen,
                "repeat_penalty": np.array([[rp]], np.float32),
                "k": np.array([k], np.int64),
            },
        )
        names = [o.name for o in self._sess.get_outputs()]
        top_idx = out[names.index("top_indices")][0]
        top_scr = out[names.index("top_scores")][0]
        return pd.DataFrame(
            {
                "user_row": row_id,
                "rank": np.arange(len(top_idx)),
                "item_guid": [self._idx_to_guid[i] for i in top_idx],
                "score": top_scr,
            }
        )


def build_mlflow(onnx_path: Path, vocab_path: Path, out_dir: Path) -> Path:
    import mlflow.pyfunc

    if out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    mlflow.pyfunc.save_model(
        path=str(out_dir),
        python_model=FeaseOnnxPyfunc(),
        artifacts={"onnx": str(onnx_path), "vocab": str(vocab_path)},
        code_paths=[str(Path(__file__))],
        pip_requirements=["onnxruntime>=1.18", "numpy>=1.24", "pandas>=1.5"],
    )
    return out_dir
```

- [ ] **Step 4: Run the MLflow roundtrip test**

Run: `.venv/bin/python -m pytest tests/test_onnx_export.py::test_mlflow_roundtrip_guid_in_guid_out -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add kzn_recsys/onnx_export/_mlflow.py tests/test_onnx_export.py
git commit -m "feat(onnx): self-contained MLflow pyfunc wrapper (GUID in/out)"
```

---

## Task 9: Rust `ort` parity test against a committed fixture

**Files:**
- Create: `tests/fixtures/` (committed `fixture.onnx`, `inputs.json`, `expected.json`)
- Modify: `kzn_recsys/onnx_export/__init__.py` (add a fixture-writer helper)
- Modify: `src/onnx_export.rs` (add the `#[cfg(test)]` ort parity test)
- Test: `src/onnx_export.rs`

- [ ] **Step 1: Add a fixture-writer helper to `kzn_recsys/onnx_export/__init__.py`**

Append to `kzn_recsys/onnx_export/__init__.py`:

```python
def _write_rust_fixture(model, fixtures_dir) -> None:
    """Emit fixture.onnx + inputs.json + expected.json for the Rust ort test.

    Run once to (re)generate committed fixtures; not part of normal export.
    """
    import json as _json

    import numpy as _np

    fixtures_dir = Path(fixtures_dir)
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    res = export_onnx(model, fixtures_dir)
    (fixtures_dir / "fixture.onnx").write_bytes(res.onnx_path.read_bytes())

    payload = _payload_from_model(model)
    M, K = payload.num_items, payload.num_user_features
    inter = _np.zeros(M, _np.float32)
    if M > 0:
        inter[0] = 3.0
    feat = _np.zeros(K, _np.float32)
    if K > 0:
        feat[0] = 1.0
    inputs = {"interactions": inter.tolist(), "features": feat.tolist()}
    (fixtures_dir / "inputs.json").write_text(_json.dumps(inputs))

    import onnxruntime as _ort

    sess = _ort.InferenceSession(str(res.onnx_path))
    out = sess.run(
        None,
        {
            "interactions": inter.reshape(1, M),
            "features": feat.reshape(1, K),
            "mask": _np.ones((1, M), _np.float32),
            "seen": _np.zeros((1, M), _np.float32),
            "repeat_penalty": _np.array([[0.0]], _np.float32),
            "k": _np.array([M], _np.int64),
        },
    )
    names = [o.name for o in sess.get_outputs()]
    raw = out[names.index("raw_scores")][0]
    (fixtures_dir / "expected.json").write_text(_json.dumps({"raw_scores": raw.tolist()}))
```

- [ ] **Step 2: Generate and commit the fixtures**

Run:
```bash
.venv/bin/python -c "
import tempfile, pathlib, polars as pl, kzn_recsys as fease
from kzn_recsys.onnx_export import _write_rust_fixture
with tempfile.TemporaryDirectory() as d:
    d = pathlib.Path(d)
    pl.DataFrame({'user_id':['u0','u0','u1'],'item_id':['G0','G2','G1'],'value':[5.0,4.0,6.0]}).write_parquet(d/'i.parquet')
    pl.DataFrame({'user_id':['u0','u1','u2'],'feature_name':['region_US','region_EMEA','region_APAC'],'value':[1.0,1.0,1.0]}).write_parquet(d/'u.parquet')
    pl.DataFrame({'item_id':['G0','G1','G2','G3'],'feature_name':['a','a','a','a'],'value':[1.0]*4}).write_parquet(d/'t.parquet')
    m = fease.build_and_train(interactions_path=str(d/'i.parquet'), user_features_path=str(d/'u.parquet'), item_features_path=str(d/'t.parquet'), alpha=1.0, beta=1.0, lambda_=10.0, meta_weight=0.0)
    _write_rust_fixture(m, 'tests/fixtures')
print('fixtures written')
"
```
Expected: `fixtures written`; `tests/fixtures/{fixture.onnx,inputs.json,expected.json}` exist.

- [ ] **Step 3: Add the failing `ort` parity test to `src/onnx_export.rs`**

Add inside the existing `#[cfg(test)] mod tests { … }` in `src/onnx_export.rs`:

```rust
#[test]
fn ort_fixture_matches_expected_raw_scores() {
    use std::fs;

    // Skip gracefully if fixtures aren't present (e.g., partial checkout).
    let onnx = std::path::Path::new("tests/fixtures/fixture.onnx");
    if !onnx.exists() {
        eprintln!("skipping: tests/fixtures/fixture.onnx missing");
        return;
    }

    let inputs: serde_json::Value =
        serde_json::from_str(&fs::read_to_string("tests/fixtures/inputs.json").unwrap()).unwrap();
    let expected: serde_json::Value =
        serde_json::from_str(&fs::read_to_string("tests/fixtures/expected.json").unwrap()).unwrap();

    let inter: Vec<f32> = inputs["interactions"]
        .as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
    let feat: Vec<f32> = inputs["features"]
        .as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
    let exp: Vec<f32> = expected["raw_scores"]
        .as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();

    let m = inter.len();
    let k = feat.len();

    let mut session = ort::session::Session::builder()
        .unwrap()
        .commit_from_file(onnx)
        .unwrap();

    let interactions = ndarray::Array2::from_shape_vec((1, m), inter).unwrap();
    let features = ndarray::Array2::from_shape_vec((1, k), feat).unwrap();
    let mask = ndarray::Array2::<f32>::ones((1, m));
    let seen = ndarray::Array2::<f32>::zeros((1, m));
    let repeat_penalty = ndarray::Array2::<f32>::zeros((1, 1));
    let kk = ndarray::Array1::<i64>::from_vec(vec![m as i64]);

    let outputs = session
        .run(ort::inputs![
            "interactions" => ort::value::Tensor::from_array(interactions).unwrap(),
            "features" => ort::value::Tensor::from_array(features).unwrap(),
            "mask" => ort::value::Tensor::from_array(mask).unwrap(),
            "seen" => ort::value::Tensor::from_array(seen).unwrap(),
            "repeat_penalty" => ort::value::Tensor::from_array(repeat_penalty).unwrap(),
            "k" => ort::value::Tensor::from_array(kk).unwrap(),
        ])
        .unwrap();

    let (_, raw) = outputs["raw_scores"].try_extract_tensor::<f32>().unwrap();
    assert_eq!(raw.len(), exp.len());
    for (a, b) in raw.iter().zip(exp.iter()) {
        assert!((a - b).abs() < 1e-4, "ort raw_scores mismatch: {a} vs {b}");
    }
}
```

Add to the top of `src/onnx_export.rs` (the test only — keep it out of the non-test build):

```rust
#[cfg(test)]
extern crate ndarray;
```

(`serde_json` is already a normal dependency, so it is available to tests.)

- [ ] **Step 4: Run the Rust parity test**

Run: `cargo test --lib onnx_export::tests::ort_fixture_matches_expected_raw_scores -- --nocapture`
Expected: PASS (first run downloads the ONNX Runtime binary via `ort`'s `download-binaries`). If the exact `ort` 2.x tensor/session API differs from the snippet, adjust to the installed `ort` version's API (the shape is: build session from file, feed named tensors, extract `raw_scores`), keeping the `1e-4` tolerance.

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/ kzn_recsys/onnx_export/__init__.py src/onnx_export.rs
git commit -m "test(onnx): Rust ort parity against committed fixture"
```

---

## Task 10: End-to-end export, unsupported-kind, and docs

**Files:**
- Test: `tests/test_onnx_export.py`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Write the end-to-end + unsupported-kind tests**

Append to `tests/test_onnx_export.py`:

```python
def test_export_writes_all_artifacts(trained_model, tmp_path):
    from kzn_recsys.onnx_export import export_onnx

    res = export_onnx(trained_model, tmp_path, dtype="int8", mlflow=True)
    assert res.onnx_path.exists()
    assert res.vocab_path.exists()
    assert res.mlflow_path.exists()


def test_unsupported_kind_raises(trained_model, monkeypatch):
    from kzn_recsys import onnx_export as ox

    real = trained_model.export_payload

    def fake_payload():
        d = real()
        d["kind"] = "sasrec"
        return d

    monkeypatch.setattr(trained_model, "export_payload", fake_payload, raising=False)
    with pytest.raises(NotImplementedError, match="sasrec"):
        ox._payload_from_model(trained_model)


def test_bad_dtype_raises(trained_model, tmp_path):
    from kzn_recsys.onnx_export import export_onnx

    with pytest.raises(ValueError, match="dtype"):
        export_onnx(trained_model, tmp_path, dtype="bf16")
```

- [ ] **Step 2: Run the full Python suite**

Run: `.venv/bin/python -m pytest tests/test_onnx_export.py -v`
Expected: PASS (all tests)

- [ ] **Step 3: Document the feature in `CLAUDE.md`**

Under the `## Python Layer (kzn_recsys/)` section, add a bullet:

```markdown
- `onnx_export/` — optional ONNX export (requires the `[onnx]` extra):
  `export_onnx(model, out_dir, *, top_k_default, dtype, repeat_penalty_default, mlflow)`
  writes `model.onnx` (Gemm scoring + configurable repeat penalty + eligibility
  mask + TopK + `raw_scores`), a `vocab.json` sidecar, and an optional MLflow
  pyfunc model. See `docs/superpowers/specs/2026-06-01-onnx-export-design.md`.
```

- [ ] **Step 4: Run the Rust suite to confirm nothing regressed**

Run: `cargo test`
Expected: all existing tests + the new `onnx_export` tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_onnx_export.py CLAUDE.md
git commit -m "test(onnx): end-to-end artifacts + unsupported-kind; docs"
```

---

## Self-Review (completed by plan author)

**Spec coverage:** §3.1 export_payload → T2; β-fold + raw-S note → T2/T4; §4 graph (Gemm, seen-union, repeat, mask, TopK, raw_scores, K=0 uniform, optional-input initializers) → T4/T5/T6; §5 API (dtype, repeat_penalty_default, dispatch, `_HAS_ONNX`) → T3/T7; §6 vocab incl. io_signature + sentinels glossary → T6; §7 MLflow pyfunc (GUID I/O, seen-from-keys, unknown-skip) → T8; §8 parity (f32 onnxruntime + Rust ort fixture) → T4/T9; §9 deps (onnx extra, ort dev-dep) → T0; §10 seam/unsupported-kind → T2/T10; §11 edge cases (NaN/Inf/zeros, top_k≤0, K=0, value-0.0 seen) → T3/T5/T6; §12 raw_scores → T4. Deferred items (learned Tier-C table, SASRec/Two-Tower export) intentionally out of scope.

**Placeholder scan:** No TBD/TODO; every code step has complete code and exact run commands. The only adaptivity note is the `ort` 2.x API caveat in T9 Step 4, with the concrete operation shape specified.

**Type consistency:** Rust `export_payload` dict keys (`s_items_bytes`, `s_items_shape`, `kind`, `sparsity_threshold`, …) match `_payload_from_model`'s reads; `ExportPayload`/`ExportResult` field names match across `__init__`, `_graph`, `_vocab`, `_mlflow`; graph I/O names (`interactions, features, mask, seen, repeat_penalty, k → top_indices, top_scores, raw_scores`) are identical across `_graph.py`, `_vocab.py` io_signature, the Python tests, the MLflow wrapper, and the Rust ort test; constants `MASK_PENALTY`/`EXCLUDE_SENTINEL`/`OPSET` defined once in `__init__` and imported elsewhere.
