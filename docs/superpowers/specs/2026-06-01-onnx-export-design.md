# Design: ONNX Export for FEASE (EASE)

- **Status**: Approved (design); pending implementation plan
- **Date**: 2026-06-01
- **Scope**: Optional ONNX export for the linear EASE model, with a seam for
  future models (SASRec, Two-Tower) and a future Rust-native serving path.
- **Related**: ADR-0001 (multi-model architecture); research paper
  `research/silver_torch_research.pdf` (SilverTorch, SIGIR '26).

## 1. Motivation

The primary driver is **ML-platform integration**: produce a portable,
self-contained `.onnx` artifact that drops into Databricks / MLflow Model
Serving (and, by extension, Triton, Vertex, SageMaker, or any ONNX Runtime
host). Today a trained model can only be served through this library's own
Rust/PyO3 `predict` path (`kzn_recsys.FeaseModel.predict`) or the custom
`FEAS`-magic binary format (`src/serialization.rs`). Neither is consumable by
a generic ML serving platform.

ONNX export does **not** replace the existing `.fease` serialization — it is an
additional, optional output format.

### SilverTorch influence

SilverTorch ("A Unified Model-based System to Democratize Large-Scale
Recommendation on GPUs") argues for **model-based serving / "index as
model"**: fold serving-time logic (filtering, nearest-neighbour search,
top-k ranking, score aggregation) *into the served model graph* as tensor
operators, so the runtime executes one forward pass and the client sends one
request — instead of orchestrating separate indexing/filtering/scoring
services.

We adopt that philosophy at EASE's (much smaller) scale:

- The exported graph is not just a scoring matrix multiply. It folds in
  **top-k ranking** and an **optional eligibility mask** ("exclude
  already-seen / ineligible items"), so the platform gets ranked results from
  a single forward pass.
- Item IDs stay **integer inside the graph**; string GUIDs are a boundary
  concern handled by a sidecar vocabulary, mirroring SilverTorch's treatment
  of item ids.
- **Quantization** (SilverTorch's Int8 lever for memory/throughput) maps onto
  EASE's one large weight — the `S` matrix — and is offered as an optional
  export flag.

SilverTorch components that do **not** transfer: ANN/IVF search and the GPU
bloom-filter index exist to *avoid* scoring the full catalog at 10M–80M item
scale. EASE's single dense matmul already scores the entire catalog exactly,
so there is no candidate-generation stage to accelerate. The OverArch /
Value-Model neural re-rank layers are likewise out of scope for the linear
EASE model (they are relevant later to the SASRec / Two-Tower seam).

## 2. Goals and non-goals

### Goals
- Export a trained EASE `FeaseModel` to a portable `.onnx` graph.
- Graph is self-contained: `(interactions, features, mask, k)` →
  `(top_indices, top_scores)`.
- Emit a language-neutral `vocab.json` sidecar (GUID ↔ index, metadata).
- Optionally emit an MLflow `pyfunc` model so Databricks/MLflow callers pass
  item GUIDs directly.
- Optional `dtype` flag for weight precision: `fp32` (default) / `fp16` /
  `int8`.
- Verify numerical parity with the native Rust `predict` path, on **both**
  Python (`onnxruntime`) and Rust (`ort`) runtimes.
- Keep the shipped wheel unchanged: all ONNX dependencies are optional.

### Non-goals (explicit)
- A GPU-native serving engine or hand-written CUDA kernels
  (`NVlabs/cuda-oxide`). GPU acceleration for ONNX comes from ONNX Runtime
  execution providers (CUDA/TensorRT), configured at serving time — there is
  nothing to hand-write.
- LLM-style serving (`EricLBuehler/candle-vllm`). Wrong workload (autoregressive
  text generation), and `candle` was explicitly rejected in ADR-0001 in favour
  of `burn`.
- ONNX export for SASRec / Two-Tower. The seam is designed in; the
  implementation raises `NotImplementedError` until those models' phases land.
- Static int8 calibration. The first cut uses dynamic quantization only.
- A Rust-native serving path. The artifact is *designed* to be Rust-loadable
  (plain-JSON vocab, vanilla-opset graph) so a future `serving.rs` path via
  `ort` needs no rework, but no serving code is written now.

## 3. Architecture

The model abstraction stays in Rust (ADR-0001 intact). Only graph
*serialization* is Python — which it must be, because the MLflow wrapper and
the quantization tooling are Python-native, and parity must be checked against
`onnxruntime`/`onnx` regardless.

```
Rust (minimal seam)                Python  kzn_recsys/onnx_export.py        Tests
─────────────────                  ────────────────────────────────        ─────
FeaseModel.export_payload()  ──►   build_graph()    → model.onnx      Python: onnxruntime parity
  exposes S_items (M×(M+K)),       write_vocab()    → vocab.json       Rust:   ort loads fixture .onnx,
  beta, M, K, mappings,            build_mlflow()   → MLflow model dir         compares vs RustFeaseModel
  kind, validate()                 quantize()       → fp16 / int8
                                   verify_parity()  (onnxruntime)
```

**Construction approach (chosen):** Python authors the graph with the canonical
`onnx` library; `ort` is used for Rust-side parity tests. This uses
battle-tested ONNX authoring + quantization tooling, co-locates every artifact
(graph, vocab, wrapper, quantization, parity check) in one place, and keeps the
Rust change to a single accessor. Rejected alternative: hand-authoring the
`ModelProto` in Rust via `prost` (ADR-0001-literal, but splits the feature
across two languages and rebuilds tooling that the `onnx` lib provides for
free; the MLflow wrapper and quantization would still be Python).

## 4. The ONNX graph contract

EASE prediction (`src/model.rs::predict`) is `p = S · z` with
`z = [x | β·u]`, keeping the first `M` (= `num_items`) entries. Two
simplifications:

1. Only `S[0:M, :]` affects the output, so export `S_items` of shape
   `M × (M + K)` (K = `num_user_features`), not the full `(M+K)²` square.
2. **Fold `β` into the weight at export time**: `S_items[:, M:] *= β`. The
   graph never multiplies by β; β is fixed at export anyway. This removes a
   runtime input the caller could set wrong (SilverTorch's "bake static values
   at publish time" principle).

```
Inputs:
  interactions : float32[batch, M]   dense interaction values (0 = no interaction)
  features     : float32[batch, K]   dense user-feature values
  mask         : float32[batch, M]   eligibility; 1 = keep, 0 = exclude
  k            : int64 scalar         number of items to return

Graph (vanilla ai.onnx, opset 17):
  z          = Concat(interactions, features)   # [batch, M+K]
  scores     = Gemm(z, W, transB=1)             # W = S_items_scaled (M × M+K)
  scores     = Cast(scores, to=float32)         # masking + ranking always in fp32 (see note)
  masked     = scores + (mask - 1) * 1e9        # excluded items sink below TopK
  kc         = Min(k, M)                         # clamp k ≤ catalog size
  topv, topi = TopK(masked, kc, axis=-1, largest=1, sorted=1)

Outputs:
  top_indices : int64[batch, kc]
  top_scores  : float32[batch, kc]
```

Design notes:
- `mask` is a **required** graph input where all-ones means "no filtering".
  This avoids ONNX's awkward optional-input mechanics; the MLflow wrapper
  defaults it to all-ones when the caller omits it. The penalty `(mask-1)*1e9`
  pushes excluded items below any real score so `TopK` never returns them.
  (`1e9` is safely larger than any realistic EASE score magnitude; the value
  is recorded in `vocab.json` for traceability.)
- **Score dtype is always fp32 inside the masking/TopK subgraph and on output**,
  independent of the weight `dtype`. Only the stored weight `W` (and the
  `Gemm`/`MatMulInteger` it feeds) is quantized for `fp16`/`int8`; its result is
  cast to fp32 before masking. This keeps the output signature dtype-stable
  across all `dtype` choices and keeps `mask_penalty` safe — `1e9` would
  overflow fp16 (max ≈ 65504), so masking in fp16 would corrupt scores.
- `k` is a graph **input** (not a baked attribute) so callers can vary top-k
  without re-exporting. `top_k_default` is stored in `vocab.json` and used by
  the MLflow wrapper as the default when a caller omits `top_k`.
- `batch` is a dynamic dimension; the same artifact serves one user or a
  batch.
- Default weight dtype is `float32` (cast from the native `f64`). A documented
  parity tolerance applies (see §8). `float64` exact-parity export is **not**
  in the first cut; if exact parity is later required it can be added as a
  `dtype="fp64"` option.

## 5. Public Python API

```python
kzn_recsys.export_onnx(
    model,                  # a trained FeaseModel
    output_dir,             # directory to write artifacts into
    *,
    top_k_default=100,      # default k, stored in vocab.json + used by wrapper
    dtype="fp32",           # "fp32" | "fp16" | "int8"
    include_mask=True,      # emit the mask input + masking subgraph
    mlflow=False,           # also emit an MLflow pyfunc model directory
) -> ExportResult
```

`ExportResult` carries the written paths: `onnx_path`, `vocab_path`, and
`mlflow_path` (when `mlflow=True`).

Dispatch is on `model.kind`:
- `ease` → build the graph described in §4.
- any other kind → raise
  `NotImplementedError("ONNX export not yet supported for {kind}")`.

Pre-export, the exporter calls the model's existing `validate()` and refuses to
export a model that fails (NaN/Inf in `S`, all-zeros, dimension mismatch).

## 6. vocab.json sidecar

Language-neutral JSON (loadable later by a Rust serving path via serde):

```json
{
  "format_version": 1,
  "model_kind": "ease",
  "num_items": 12000,
  "num_user_features": 340,
  "beta": 0.5,
  "weight_dtype": "fp32",
  "opset": 17,
  "top_k_default": 100,
  "include_mask": true,
  "mask_penalty": 1e9,
  "item_index_to_guid": ["itm_a", "itm_b", "..."],
  "feature_name_to_index": {"age_bucket=2": 0, "country=US": 1},
  "provenance": {
    "alpha": 1.0,
    "lambda_": 100.0,
    "meta_weight": 0.0,
    "weighting_config": null
  }
}
```

- `item_index_to_guid` maps the graph's `top_indices` output back to GUIDs.
- `feature_name_to_index` tells a caller where to place each feature value in
  the dense `features` input vector.
- `provenance` is informational only — `alpha`, `lambda_`, `meta_weight`, and
  any `weighting_config` are already baked into `S` and do not affect
  inference. They are recorded for reproducibility/debugging.

## 7. MLflow pyfunc wrapper

An `mlflow.pyfunc.PythonModel` that packages `model.onnx` + `vocab.json` as
artifacts and serves them via `onnxruntime`. It speaks **GUIDs** so Databricks
Model Serving callers never handle integer indices.

Input — one row per user:

```python
{
  "interactions": {item_guid: value, ...},
  "features":     {feature_name: value, ...},
  "exclude":      [item_guid, ...],   # optional → builds the mask
  "top_k":        int                 # optional → overrides top_k_default
}
```

Output: `DataFrame[user_row, rank, item_guid, score]`.

Behaviour:
- `load_context` builds an `onnxruntime.InferenceSession` from the bundled
  `.onnx` and loads `vocab.json`.
- `predict` vectorizes each row (GUIDs/feature-names → dense tensors via the
  vocab), runs the session, and maps `top_indices` → GUIDs.
- Unknown item GUIDs / feature names in input are **skipped with a warning**,
  not fatal (catalogs drift; a serving call should degrade, not crash).
- Ships with a pinned pip environment (`onnxruntime`, `numpy`).

## 8. Parity verification and tests

### Python — `tests/test_onnx_export.py`
- Train a tiny model, export, run via `onnxruntime`, assert per-item scores
  match `model.predict` within relative tolerance `1e-5` (fp32).
- Warm user and cold-start user (no interactions, features only).
- `mask` exclusion: excluded items never appear in `top_indices`.
- `top_k` clamping: `k > num_items` returns `num_items` results.
- int8 export: assert **top-k rank agreement** (set overlap / order), not score
  equality — quantization perturbs scores but preserves ranking (SilverTorch
  §4.2, §6.2). Testing score equality here would falsely flag int8 as broken.
- MLflow pyfunc roundtrip: GUID-in → GUID-out, ranks match the raw graph.
- Unsupported model kind raises `NotImplementedError`.

### Rust — `ort`-based parity test
- A small fixture `.onnx` plus a JSON of inputs/expected outputs (produced by
  the Python exporter and committed under `tests/fixtures/`) is loaded via
  `ort` and compared against `RustFeaseModel::predict`.
- This validates parity on the runtime a future Rust serving path will actually
  use, closing the cross-language loop.
- `ort` is a Rust **`[dev-dependencies]`** entry (or behind an `onnx-verify`
  feature) so the shipped wheel is unaffected.

## 9. Dependencies and build impact

- **Python:** `onnx`, `onnxruntime`, and `mlflow` are an **optional extra**:
  `pip install kzn-recsys[onnx]`. The base install is unchanged and slim.
- **Rust:** `ort` is dev-only (tests). **No change to the default wheel**,
  consistent with ADR-0001's feature-gate philosophy (EASE-only users see no
  build difference).

## 10. The seam (future models + future Rust serving)

- **Rust:** add `FeaseModel.export_payload()` (PyO3) returning `S_items`,
  `beta`, `M`, `K`, the `Mappings`, and `kind`. Reuses the existing
  `validate()`. No new heavy Rust dependencies on the default build path.
- **Other model kinds:** the Python dispatcher raises `NotImplementedError`
  until SASRec/Two-Tower phases land; each will expose its own export payload
  (burn tensors) behind the same Python protocol — the SilverTorch OverArch /
  Value-Model style post-scoring layers, when they exist, fold into those
  graphs, not EASE's.
- **Future Rust serving:** because `vocab.json` is plain JSON and the `.onnx`
  is vanilla opset, a later `serving.rs` path can load both via `ort` (with
  CUDA/TensorRT execution providers) with zero changes to the artifact format.
  No serving code is written now — only the format guarantee.

## 11. Edge cases and error handling

- Untrained / empty model (no items) → error before writing anything.
- `top_k <= 0` → validation error in the API layer.
- `top_k > num_items` → clamped in-graph by `Min(k, M)`.
- NaN/Inf/all-zeros `S` → refused via `validate()` pre-export.
- Unknown GUID/feature in MLflow wrapper input → skipped with warning.
- `meta_weight` / `weighting_config`: training-time only, already baked into
  `S`; not represented in the graph, recorded in `vocab.json` provenance.
```
