# Design: ONNX Export for FEASE (EASE)

- **Status**: Approved (design); pending implementation plan
- **Date**: 2026-06-01
- **Scope**: Optional ONNX export for the linear EASE model, with a configurable
  repeat-watch preference, a seam for future models (SASRec, Two-Tower), and a
  future Rust-native serving path.
- **Related**: ADR-0001 (multi-model architecture); research paper
  `research/silver_torch_research.pdf` (SilverTorch, SIGIR '26).
- **Revision note**: revised after a code-grounded review — parity baseline
  pinned to the f32 path, `export_payload` return type and matrix layout
  defined, `include_mask` dropped, provenance completed, K=0 handled, and a
  configurable repeat-watch penalty added that generalizes the existing
  hardcoded interacted-item exclusion.

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
  **top-k ranking**, an **eligibility mask**, and a **repeat-watch penalty**,
  so the platform gets ranked, policy-correct results from a single forward
  pass.
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
- Graph is self-contained: `(interactions, features, mask, repeat_penalty, k)`
  → `(top_indices, top_scores)`.
- **Configurable repeat-watch preference**: a per-request (and per-individual)
  penalty on already-seen items, spanning hard-exclude → demote → neutral →
  boost. Default reproduces the native Rust behavior (exclude).
- Emit a language-neutral `vocab.json` sidecar (GUID ↔ index, metadata,
  repeat policy).
- Optionally emit an MLflow `pyfunc` model so Databricks/MLflow callers pass
  item GUIDs directly.
- Optional `dtype` flag for weight precision: `fp32` (default) / `fp16` /
  `int8`.
- Verify numerical parity with the native predict path, on **both** Python
  (`onnxruntime`) and Rust (`ort`) runtimes.
- Keep the shipped wheel unchanged: all ONNX dependencies are optional.

### Non-goals (explicit)
- A **learned per-user repeat-affinity table** (Tier C below). Deferred to a
  follow-up issue. The graph already accepts a per-individual penalty input, so
  a learned source can be plugged in later with no graph change.
- A model-native learned repeat propensity (e.g., logistic on user features, or
  learned inside SASRec/Two-Tower). Further-future.
- A GPU-native serving engine or hand-written CUDA kernels
  (`NVlabs/cuda-oxide`). GPU acceleration for ONNX comes from ONNX Runtime
  execution providers (CUDA/TensorRT), configured at serving time.
- LLM-style serving (`EricLBuehler/candle-vllm`). Wrong workload, and `candle`
  was explicitly rejected in ADR-0001 in favour of `burn`.
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
  → ExportPayload (see §3.1)       write_vocab()    → vocab.json       Rust:   ort loads fixture .onnx,
                                   build_mlflow()   → MLflow model dir         compares vs predict_scores
                                   quantize()       → fp16 / int8
                                   verify_parity()  (onnxruntime)
```

**Construction approach (chosen):** Python authors the graph with the canonical
`onnx` library; `ort` is used for Rust-side parity tests. This uses
battle-tested ONNX authoring + quantization tooling, co-locates every artifact
(graph, vocab, wrapper, quantization, parity check) in one place, and keeps the
Rust change to a single accessor. Rejected alternative: hand-authoring the
`ModelProto` in Rust via `prost` (ADR-0001-literal, but splits the feature
across two languages and rebuilds tooling the `onnx` lib provides for free; the
MLflow wrapper and quantization would still be Python).

### 3.1 The Rust seam: `FeaseModel.export_payload()`

A new PyO3 method on `FeaseModel`. `RustFeaseModel` has no `kind` field —
`kind()` lives on the `RecModel` trait (`src/models/ease.rs`) — so the payload
sources `kind` from `EaseAdapterRef::new(&self.model).kind()`.

Returns a Python dict (assembled into an `@dataclass ExportPayload` on the
Python side) with:

| Field | Type | Notes |
|---|---|---|
| `kind` | `str` | `"ease"` (from `RecModel::kind()`) |
| `s_items` | `numpy.ndarray[float64]`, shape `(M, M+K)` | **row-major / C-contiguous** |
| `beta` | `float` | folded into the weight at export (see §4) |
| `num_items` (M) | `int` | |
| `num_user_features` (K) | `int` | |
| `num_item_features` | `int` | provenance only |
| `item_index_to_guid` | `list[str]` | `mappings.idx_to_item` |
| `feature_name_to_index` | `dict[str,int]` | `mappings.user_feature_to_idx` |
| `alpha`, `lambda_`, `meta_weight` | `float` | provenance |
| `sparsity_threshold` | `float \| None` | from `weighting_config`, if set |
| `weighting_config` | `dict \| None` | provenance |

**Matrix layout.** `RustFeaseModel.s_matrix` is a `nalgebra::DMatrix<f64>`
stored **column-major**, of shape `(M+K) × (M+K)`. The accessor extracts the
`S_items` sub-block `S[0:M, :]` and emits it **row-major** by iterating
`for r in 0..M { for c in 0..(M+K) { push s_matrix[(r, c)] } }`, so the Python
side can `np.asarray(flat).reshape(M, M+K)` (NumPy C-order default) without a
transpose. The Rust accessor — not Python — owns the layout conversion.

The exporter calls the model's existing `validate()` first and refuses to
export a model that fails (NaN/Inf in `S`, all-zeros, dimension mismatch),
matching `RustFeaseModel::validate()` checks 1–4.

## 4. The ONNX graph contract

EASE prediction (`src/model.rs::predict`) is `p = S · z` with
`z = [x | β·u]`, keeping the first `M` (= `num_items`) entries. Two
simplifications:

1. Only `S[0:M, :]` affects the output, so export `S_items` of shape
   `M × (M + K)` (K = `num_user_features`), not the full `(M+K)²` square.
2. **Fold `β` into the weight at export time**: `S_items[:, M:] *= β`. The graph
   never multiplies by β; β is fixed at export anyway. In the Rust path β is
   multiplied into the *feature values* of `z`
   (`src/model.rs`: `z_vec[M + feat_idx] = val * beta`), so baking it into the
   feature columns of the weight and feeding raw feature values is algebraically
   equivalent.

```
Inputs:
  interactions   : float32[batch, M]   dense interaction values (0 = no interaction)
  features       : float32[batch, K]   dense user-feature values  (omitted entirely if K == 0)
  mask           : float32[batch, M]   eligibility; 1 = keep, 0 = exclude  (default all-ones)
  repeat_penalty : float32[batch, 1]   ρ, penalty on already-seen items     (default = exclude sentinel)
  k              : int64 scalar        number of items to return

Graph (vanilla ai.onnx, opset 17):
  z          = Concat(interactions, features)        # [batch, M+K]   (z = interactions if K == 0)
  raw        = Gemm(z, W, transB=1)                  # W stored [M, M+K]; transB=1 → z·Wᵀ → [batch, M]
  scores     = Cast(raw, to=float32)                 # masking + ranking always in fp32 (see notes)
  seen       = Cast(Not(Equal(interactions, 0)), float32)   # [batch, M] indicator of prior interaction
  adjusted   = scores - repeat_penalty * seen        # ρ broadcast [batch,1] over [batch,M]
  masked     = adjusted + (mask - 1) * MASK_PENALTY  # excluded items sink below TopK; MASK_PENALTY = 1e9 (fp32)
  kc         = Min(k, M)                             # clamp k ≤ catalog size
  topv, topi = TopK(masked, kc, axis=-1, largest=1, sorted=1)

Outputs:
  top_indices : int64[batch, kc]
  top_scores  : float32[batch, kc]   the ranked (adjusted+masked) score
```

Design notes:
- **Optional inputs with baked defaults.** `interactions` (and `features` when
  K > 0) are required. `mask`, `repeat_penalty`, and `k` are graph inputs that
  *also* have initializers (ONNX default-valued optional inputs), so a caller
  may omit them and get the baked defaults: `mask` = all-ones (no eligibility
  filtering), `repeat_penalty` = the exclude sentinel, `k` = `top_k_default`.
  This means even a bare `onnxruntime` caller — not just the MLflow wrapper —
  reproduces the native Rust default (exclude already-watched) when passing only
  `interactions`/`features`.
- **Stored vs computed weight shape.** The ONNX initializer `W` holds `S_items`
  in its non-transposed shape `[M, M+K]`. ONNX `Gemm` with `transB=1` computes
  `z[batch, M+K] · Wᵀ[(M+K), M] → [batch, M]`.
- **Repeat-watch penalty `ρ`** generalizes the hardcoded interacted-item
  exclusion (`src/lib.rs` and `src/serving.rs::filter_sort_top_k`):

  | ρ | behavior | use case |
  |---|---|---|
  | `+MASK_PENALTY` (sentinel) | always exclude seen | **default — matches native EASE/SASRec** |
  | `> 0` finite | demote repeats | mostly-fresh, occasional repeat |
  | `0` | neutral; repeats compete on merit | exploratory |
  | `< 0` | boost repeats | consumables, replay, habitual content |

  `seen` is derived in-graph from `interactions ≠ 0` (no extra input). The
  eligibility `mask` (compliance/business hard-filter) and `repeat_penalty`
  (preference) are kept as **separate additive terms** because they answer
  different questions ("is this item *allowed*?" vs "how much do we like
  *re*-showing it?").
- **Score dtype is always fp32** inside the masking/TopK subgraph and on output,
  independent of weight `dtype`. Only the stored weight `W` (and the
  `Gemm`/`MatMulInteger` it feeds) is quantized for `fp16`/`int8`; its result is
  cast to fp32 before any penalty is applied. This keeps the output signature
  dtype-stable and keeps `MASK_PENALTY`/exclude-sentinel safe — `1e9` overflows
  fp16 (max ≈ 65504), so penalties in fp16 would corrupt scores.
- `k` is a graph **input** (not a baked attribute) so callers vary top-k without
  re-exporting. `top_k_default` is stored in `vocab.json` and used by the MLflow
  wrapper as the default.
- `batch` is a dynamic dimension; the same artifact serves one user or a batch.
- **K = 0 (no user features).** When `num_user_features == 0`, the `features`
  input and the `Concat` are omitted; `z = interactions`. This avoids feeding a
  zero-width `float32[batch, 0]` tensor, which not all runtimes handle.
- Default weight dtype is `float32` (cast from native `f64`); parity tolerance
  per §8. `float64` exact-parity export is not in the first cut (could be added
  later as `dtype="fp64"`).

## 5. Public Python API

```python
kzn_recsys.export_onnx(
    model,                          # a trained FeaseModel
    output_dir,                     # directory to write artifacts into
    *,
    top_k_default=100,              # default k; stored in vocab.json + used by wrapper
    dtype="fp32",                   # "fp32" | "fp16" | "int8"
    repeat_penalty_default="exclude",  # "exclude" | float  (per-call override still possible at inference)
    mlflow=False,                   # also emit an MLflow pyfunc model directory
) -> ExportResult
```

- `repeat_penalty_default="exclude"` bakes the exclude sentinel (`+MASK_PENALTY`)
  as the default `repeat_penalty` initializer; a `float` bakes that value.
  Callers can still override per-request via the `repeat_penalty` graph input.
- `ExportResult` carries the written paths: `onnx_path`, `vocab_path`, and
  `mlflow_path` (when `mlflow=True`).
- `include_mask` is **not** a parameter: the mask is always present (all-ones =
  no filtering), so there is only one graph signature to deploy against.
- Dispatch is on the payload's `kind`: `"ease"` builds the graph in §4; any
  other kind raises
  `NotImplementedError("ONNX export not yet supported for {kind}")`.
- `export_onnx` is added to `kzn_recsys.__all__` gated on an optional-import
  flag `_HAS_ONNX` (mirroring the existing `_HAS_ML_MODELS` pattern in
  `kzn_recsys/__init__.py`); importing it without the `[onnx]` extra installed
  raises a clear error naming the extra.

## 6. vocab.json sidecar

Language-neutral JSON (loadable later by a Rust serving path via serde):

```json
{
  "format_version": 1,
  "model_kind": "ease",
  "num_items": 12000,
  "num_user_features": 340,
  "num_item_features": 5,
  "beta": 0.5,
  "weight_dtype": "fp32",
  "opset": 17,
  "top_k_default": 100,
  "mask_penalty": 1e9,
  "repeat_policy": {
    "default_penalty": 1e9,
    "exclude_sentinel": 1e9,
    "per_user_table_present": false
  },
  "item_index_to_guid": ["itm_a", "itm_b", "..."],
  "feature_name_to_index": {"age_bucket=2": 0, "country=US": 1},
  "provenance": {
    "alpha": 1.0,
    "lambda_": 100.0,
    "meta_weight": 0.0,
    "sparsity_threshold": null,
    "num_item_features": 5,
    "weighting_config": null
  }
}
```

- `format_version` is independent of the `FEAS` binary format's v1/v2 (different
  namespace); starts at `1`.
- `item_index_to_guid` maps the graph's `top_indices` → GUIDs.
- `feature_name_to_index` tells a caller where to place each feature value in the
  dense `features` input.
- `repeat_policy.default_penalty` is the baked default `ρ`;
  `per_user_table_present` is `false` in this iteration (reserved for the
  deferred learned Tier C table).
- `provenance` is informational only — `alpha`, `lambda_`, `meta_weight`,
  `sparsity_threshold`, and any `weighting_config` are already baked into `S`
  (sparsity pruning via `RustFeaseModel::prune_sparse` zeroes small entries in
  the exported weights) and do not affect inference; recorded for
  reproducibility.

## 7. MLflow pyfunc wrapper

An `mlflow.pyfunc.PythonModel` that packages `model.onnx` + `vocab.json` as
artifacts and serves them via `onnxruntime`. It speaks **GUIDs** so Databricks
Model Serving callers never handle integer indices.

Input — one row per user:

```python
{
  "interactions":  {item_guid: value, ...},
  "features":      {feature_name: value, ...},
  "exclude":       [item_guid, ...],   # optional → eligibility mask (hard compliance drop)
  "repeat_penalty": float,             # optional → ρ; default = vocab repeat_policy.default_penalty
  "top_k":         int                 # optional → overrides top_k_default
}
```

Output: `DataFrame[user_row, rank, item_guid, score]`.

Behaviour:
- `load_context` builds an `onnxruntime.InferenceSession` and loads
  `vocab.json`.
- `predict` vectorizes each row (GUIDs/feature-names → dense tensors via the
  vocab), runs the session, and maps `top_indices` → GUIDs.
- **Already-watched handling matches the Rust path by default.** `seen` is
  derived in-graph from `interactions`, and the default `repeat_penalty` is the
  exclude sentinel — so a caller who passes `interactions` and nothing else gets
  already-watched items excluded, exactly like `FeaseModel.predict`
  (`src/lib.rs`). Callers override `repeat_penalty` to demote/keep/boost repeats.
- The `exclude` field is the *eligibility* mask (licensing/region), distinct
  from repeat handling.
- Unknown item GUIDs / feature names in input are **skipped with a warning**,
  not fatal (catalogs drift; serving should degrade, not crash).
- Ships with a pinned pip environment (`onnxruntime`, `numpy`).

## 8. Parity verification and tests

### Parity reference
The ONNX graph computes raw scores in fp32, so the parity reference for **raw
scores** is the **f32 path** — `RecModel::predict_scores` via `EaseAdapter`
(`src/models/ease.rs`: `scores.map(|x| x as f32)`), **not** the f64
`RustFeaseModel::predict`. Comparing against f64 would measure the f64→f32 cast,
not ONNX correctness. Tolerance: relative `1e-5` (fp32).

For **top-k membership under the default exclude policy**, the reference is
`FeaseModel.predict` (which excludes interacted items): the ONNX top-k GUIDs
must match modulo float tie-ordering.

### Python — `tests/test_onnx_export.py`
- Raw-score parity vs `predict_scores` (f32), tolerance `1e-5`, with
  `repeat_penalty = 0` and `mask` all-ones (no adjustments).
- Default policy parity: `repeat_penalty = exclude sentinel` → ONNX top-k GUIDs
  match `FeaseModel.predict`.
- Warm user and cold-start user (no interactions, features only).
- Repeat spectrum: `ρ > 0` demotes seen items; `ρ = 0` lets them rank on merit;
  `ρ < 0` boosts seen items above comparable unseen items.
- Eligibility `mask`: excluded items never appear in `top_indices`.
- `top_k` clamping: `k > num_items` returns `num_items` results.
- K = 0 model: graph omits `features`/`Concat` and still scores correctly.
- int8 export: assert **top-k rank agreement** (set overlap / order), not score
  equality — quantization perturbs scores but preserves ranking (SilverTorch
  §4.2, §6.2). Testing score equality here would falsely flag int8 as broken.
- MLflow pyfunc roundtrip: GUID-in → GUID-out; default excludes already-watched;
  `repeat_penalty`/`exclude` overrides behave as specified.
- Unsupported model kind raises `NotImplementedError`.

### Rust — `ort`-based parity test
- The Python exporter writes a small fixture: `fixture.onnx`, plus **separate**
  `inputs.json` and `expected.json` (committed under `tests/fixtures/`), so the
  Rust test verifies the graph against ground-truth scores independently of how
  `ort` loads it.
- The Rust test loads `fixture.onnx` via `ort`, runs the committed inputs, and
  compares to both `expected.json` and a fresh `EaseAdapter::predict_scores`
  computed in-test — closing the cross-language loop on the runtime a future
  Rust serving path will actually use.
- `ort` is a Rust **`[dev-dependencies]`** entry (or behind an `onnx-verify`
  feature) so the shipped wheel is unaffected.

## 9. Dependencies and build impact

- **Python:** `onnx`, `onnxruntime`, and `mlflow` are an **optional extra**:
  `pip install kzn-recsys[onnx]`. The base install is unchanged and slim.
- **Rust:** `ort` is dev-only (tests). **No change to the default wheel**,
  consistent with ADR-0001's feature-gate philosophy.

## 10. The seam (future models + future Rust serving)

- **Rust:** `FeaseModel.export_payload()` (§3.1) is the only Rust addition; no
  new heavy dependencies on the default build path.
- **Other model kinds:** the Python dispatcher raises `NotImplementedError`
  until SASRec/Two-Tower phases land; each will expose its own export payload
  (burn tensors) behind the same Python protocol. **Per-model default repeat
  policy follows native behavior**: EASE and SASRec exclude history
  (`src/serving.rs`), so they default to the exclude sentinel; Two-Tower has no
  per-request history to exclude (`src/serving.rs`: "no per-request history"),
  so it defaults to neutral (`ρ = 0`). The SilverTorch OverArch / Value-Model
  post-scoring layers, when they exist, fold into those models' graphs, not
  EASE's.
- **Learned repeat affinity (deferred):** the graph already takes a
  per-individual `repeat_penalty` input, so the future learned Tier C table (a
  `user→ρ` estimate from interaction history, looked up by the wrapper) plugs in
  with no graph change — only a sidecar table and a wrapper lookup. Tracked as a
  follow-up issue.
- **Future Rust serving:** because `vocab.json` is plain JSON and the `.onnx` is
  vanilla opset, a later `serving.rs` path can load both via `ort` (with
  CUDA/TensorRT EPs) with zero changes to the artifact format. No serving code
  is written now — only the format guarantee.

## 11. Edge cases and error handling

- Untrained / empty model (no items) → error before writing anything.
- `top_k <= 0` → validation error in the API layer.
- `top_k > num_items` → clamped in-graph by `Min(k, M)`.
- NaN/Inf/all-zeros `S` → refused via `validate()` pre-export.
- `dtype="fp16"` with the `1e9` sentinel → safe, because masking/penalties run
  in fp32 (§4); the sentinel never enters fp16 arithmetic.
- K = 0 → features input and Concat omitted (§4).
- Unknown GUID/feature in MLflow wrapper input → skipped with warning.
- `meta_weight` / `weighting_config` / `sparsity_threshold`: training-time only,
  already baked into `S`; not represented in the graph, recorded in `vocab.json`
  provenance.

## 12. Open question — repeat penalty and `top_scores` semantics

The returned `top_scores` are the **ranked (adjusted + masked) scores**, not the
raw affinity. For demoted-but-surfaced repeats this means the reported score
reflects the penalty. If a consumer needs the raw affinity alongside the ranking
score, a second output could be added later; deferred unless a concrete need
appears.
```
