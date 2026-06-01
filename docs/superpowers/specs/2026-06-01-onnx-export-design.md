# Design: ONNX Export for FEASE (EASE)

- **Status**: Approved (design); pending implementation plan
- **Date**: 2026-06-01
- **Scope**: Optional ONNX export for the linear EASE model, with a configurable
  repeat-watch preference, a seam for future models (SASRec, Two-Tower), and a
  future Rust-native serving path.
- **Related**: ADR-0001 (multi-model architecture); research paper
  `research/silver_torch_research.pdf` (SilverTorch, SIGIR '26).
- **Revision note**: revised over two code-grounded review rounds. R1: parity
  baseline pinned to the f32 path, `export_payload` return type and matrix
  layout defined, `include_mask` dropped, provenance completed, configurable
  repeat-watch penalty added. R2: `seen` decoupled from interaction values via
  an optional `seen` input (fidelity with Rust's sparse-key semantics); K=0
  resolved to a uniform signature; a secondary `raw_scores` output added; a
  sentinel glossary and a first-class `io_signature` field added.

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
- Graph is self-contained:
  `(interactions, features, mask, seen, repeat_penalty, k)`
  → `(top_indices, top_scores, raw_scores)`.
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

**The exported weight is not raw `S`.** It is a β-folded, row-subset extract:
`s_items = β-fold(S[0:M, :])`, where the user-feature columns are pre-multiplied
by β (`s_items[:, M:] = β · S[0:M, M:M+K]`; see §4). Callers pass raw feature
values; β lives in the weight. `export_payload` may return either the raw
`S[0:M, :]` plus `beta` (Python folds it) or the already-folded matrix — the
spec assumes the **Rust accessor returns raw `S[0:M, :]` and `beta`
separately**, and the Python `build_graph()` does the fold, so the fold is
testable in one place.

**Matrix layout.** `RustFeaseModel.s_matrix` is a `nalgebra::DMatrix<f64>`
stored **column-major**, of shape `(M+K) × (M+K)`. The accessor extracts the
`S_items` sub-block `S[0:M, :]` and emits it **row-major** by iterating
`for r in 0..M { for c in 0..(M+K) { push s_matrix[(r, c)] } }`, so the Python
side can `np.asarray(flat).reshape(M, M+K)` (NumPy C-order default) without a
transpose. The Rust accessor — not Python — owns the layout conversion.

**`weighting_config = None`.** When weighting was not used during training,
`model.weighting_config` is `None`. `export_payload` returns
`sparsity_threshold = None` (serialized as JSON `null`), not `0.0`, to keep the
"not applied" case distinguishable from "threshold was 0.0":
`self.model.weighting_config.as_ref().map(|wc| wc.sparsity_threshold)`.

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
  interactions   : float32[batch, M]   dense interaction values (0 = no interaction value)
  features       : float32[batch, K]   dense user-feature values (K may be 0 → width-0 tensor; see K=0 note)
  mask           : float32[batch, M]   eligibility; 1 = keep, 0 = exclude          (default all-ones)
  seen           : float32[batch, M]   prior-interaction indicator; 1 = seen        (default all-zeros)
  repeat_penalty : float32[batch, 1]   ρ, penalty on already-seen items             (default EXCLUDE_SENTINEL)
  k              : int64 scalar        number of items to return                    (default top_k_default)

Graph (vanilla ai.onnx, opset 17):
  z          = Concat(interactions, features, axis=-1)     # [batch, M+K]  (width-0 features → [batch, M])
  raw_scores = Cast(Gemm(z, W, transB=1), to=float32)      # W stored [M, M+K]; z·Wᵀ → [batch, M]; OUTPUT
  seen_eff   = Max(seen, Cast(Not(Equal(interactions, 0)), float32))   # union: explicit seen ∪ nonzero-value
  adjusted   = raw_scores - repeat_penalty * seen_eff      # ρ broadcast [batch,1] over [batch,M]
  masked     = adjusted + (mask - 1) * MASK_PENALTY        # excluded items sink below TopK
  kc         = Min(k, M)                                   # clamp k ≤ catalog size
  topv, topi = TopK(masked, kc, axis=-1, largest=1, sorted=1)

Outputs:
  top_indices : int64[batch, kc]     ranked item indices
  top_scores  : float32[batch, kc]   the ranked (adjusted + masked) score TopK ordered on
  raw_scores  : float32[batch, M]    pre-penalty, pre-mask model affinity for ALL items
```

Design notes:
- **Optional inputs with baked defaults.** `interactions` and `features` are
  required (`features` is width-0 when K=0; see below). `mask`, `seen`,
  `repeat_penalty`, and `k` are graph inputs that *also* have initializers (ONNX
  default-valued optional inputs), so a caller may omit them and get the baked
  defaults: `mask` = all-ones (no eligibility filtering), `seen` = all-zeros,
  `repeat_penalty` = `EXCLUDE_SENTINEL`, `k` = `top_k_default`. Even a bare
  `onnxruntime` caller — not just the MLflow wrapper — reproduces the native Rust
  default (exclude already-watched) when passing only `interactions`/`features`,
  because `seen_eff` falls back to the in-graph `interactions ≠ 0` derivation.
- **Stored vs computed weight shape.** The ONNX initializer `W` holds `S_items`
  in its non-transposed shape `[M, M+K]`. ONNX `Gemm` with `transB=1` computes
  `z[batch, M+K] · Wᵀ[(M+K), M] → [batch, M]`.
- **Repeat-watch penalty `ρ`** generalizes the hardcoded interacted-item
  exclusion (`src/lib.rs` and `src/serving.rs::filter_sort_top_k`):

  | ρ | behavior | use case |
  |---|---|---|
  | `EXCLUDE_SENTINEL` (+1e9) | always exclude seen | **default — matches native EASE/SASRec** |
  | `> 0` finite | demote repeats | mostly-fresh, occasional repeat |
  | `0` | neutral; repeats compete on merit | exploratory |
  | `< 0` | boost repeats | consumables, replay, habitual content |

- **`seen` semantics — explicit input ∪ derived (fixes R2 §2.1).** The Rust path
  treats an item as "seen" if it is a **key** in the interaction dict, regardless
  of value (`src/lib.rs`: `HashSet` of interacted indices). Deriving `seen` only
  from `interactions ≠ 0` would miss a key whose value is exactly `0.0` (e.g.,
  an impression with no engagement). So the graph takes an optional `seen` input
  and computes `seen_eff = Max(seen, interactions ≠ 0)`:
  - **bare caller** (no `seen`, default all-zeros) → `seen_eff = interactions ≠ 0`
    — the convenient default, unchanged for the common case;
  - **MLflow wrapper** populates `seen` from the interaction dict **keys** →
    `seen_eff` = key-based, **exact parity with Rust** (every nonzero value is
    also a key, so the union equals the key set);
  - **exposure case** (key present, value `0.0`) → wrapper sets `seen=1` → item
    is penalized, matching Rust.
  One `Max` node, vanilla opset, no branching.
- The eligibility `mask` (compliance/business hard-filter) and `repeat_penalty`
  (preference) are kept as **separate additive terms** because they answer
  different questions ("is this item *allowed*?" vs "how much do we like
  *re*-showing it?"). `mask` always overrides a repeat boost: a masked-out item
  gets `… + (0−1)·MASK_PENALTY`, which dominates any finite negative ρ.
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
- **K = 0 (no user features) — uniform signature (fixes R2 §3.1).** The graph
  **always** has the same input names, including `features`. When
  `num_user_features == 0`, `features` is a width-0 tensor `float32[batch, 0]`
  and the `Concat` is a no-op (`z = interactions`). We deliberately keep a single
  signature rather than a K-dependent variant, so downstream consumers never
  branch on K to build their input dict. ONNX Runtime (our verified target,
  Python `onnxruntime` and Rust `ort`) handles width-0 tensors and zero-input
  `Concat`; a K=0 parity test (§8) locks this in. The authoritative input
  names/shapes are published in `vocab.json.io_signature` (§6).
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

- `repeat_penalty_default="exclude"` bakes `EXCLUDE_SENTINEL` (§13) as the
  default `repeat_penalty` initializer; a `float` bakes that value. Callers can
  still override per-request via the `repeat_penalty` graph input.
- `ExportResult` carries the written paths: `onnx_path`, `vocab_path`, and
  `mlflow_path` (when `mlflow=True`).
- `include_mask` is **not** a parameter: the mask is always present (all-ones =
  no filtering), so there is only one graph signature to deploy against. For the
  same reason there is no `include_raw_scores` flag and no K-dependent variant —
  `raw_scores` is always emitted and the input signature is uniform across K.
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
  "constants": {
    "MASK_PENALTY": 1e9,
    "EXCLUDE_SENTINEL": 1e9
  },
  "repeat_policy": {
    "default_penalty": 1e9,
    "per_user_table_present": false
  },
  "io_signature": {
    "inputs": [
      {"name": "interactions",   "dtype": "float32", "shape": ["batch", 12000], "required": true},
      {"name": "features",       "dtype": "float32", "shape": ["batch", 340],   "required": true},
      {"name": "mask",           "dtype": "float32", "shape": ["batch", 12000], "required": false, "default": "all-ones"},
      {"name": "seen",           "dtype": "float32", "shape": ["batch", 12000], "required": false, "default": "all-zeros"},
      {"name": "repeat_penalty", "dtype": "float32", "shape": ["batch", 1],     "required": false, "default": "EXCLUDE_SENTINEL"},
      {"name": "k",              "dtype": "int64",   "shape": [],               "required": false, "default": "top_k_default"}
    ],
    "outputs": [
      {"name": "top_indices", "dtype": "int64",   "shape": ["batch", "kc"]},
      {"name": "top_scores",  "dtype": "float32", "shape": ["batch", "kc"]},
      {"name": "raw_scores",  "dtype": "float32", "shape": ["batch", 12000]}
    ]
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

- `format_version` is independent of the `FEAS`/`FSAS`/`FTWO` binary formats
  (different namespace; those are at v2/v3/v5); starts at `1`.
- **Two distinct `1e9` constants (R2 §2.3).** `MASK_PENALTY` is the graph
  constant applied via `(mask − 1) · MASK_PENALTY`; `EXCLUDE_SENTINEL` is the
  baked default *value* of the `repeat_penalty` input. They share the magnitude
  `1e9` but live on different code paths and are named separately here to avoid
  runtime-integration confusion.
- `io_signature` is the **authoritative** list of graph input/output names,
  dtypes, shapes, and defaults. Consumers (and the MLflow wrapper) build their
  input dict from this field, never from assumptions. `num_user_features == 0`
  is detectable here (the `features` shape is `["batch", 0]`); the signature
  itself is uniform across K (§4).
- `item_index_to_guid` maps the graph's `top_indices` (and the column index of
  `raw_scores`) → GUIDs.
- `feature_name_to_index` tells a caller where to place each feature value in the
  dense `features` input.
- `repeat_policy.default_penalty` is the baked default `ρ` (= `EXCLUDE_SENTINEL`
  when `repeat_penalty_default="exclude"`); `per_user_table_present` is `false`
  in this iteration (reserved for the deferred learned Tier C table).
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
- **Already-watched handling matches the Rust path exactly.** The wrapper
  populates the `seen` input from the interaction dict **keys** (not from
  nonzero values), and the default `repeat_penalty` is `EXCLUDE_SENTINEL` — so a
  caller passing `interactions` and nothing else gets already-watched items
  excluded with the same key-based semantics as `FeaseModel.predict`
  (`src/lib.rs`), including the value-`0.0` exposure case (R2 §2.1). Callers
  override `repeat_penalty` to demote/keep/boost repeats.
- The `exclude` field is the *eligibility* mask (licensing/region), distinct
  from repeat handling.
- **Filter vs penalize (R2 §3.5).** The Rust serving path *removes* interacted
  items from its returned vector (`filter_sort_top_k`), so it returns
  `M − |interacted|` scores; the ONNX graph *penalizes* them to ≈ `−1e9` and the
  full `raw_scores` output still has `M` columns. For the GUID-keyed top-k
  `DataFrame` the wrapper returns, the results are identical; a consumer reading
  `raw_scores` directly sees all `M` entries (penalized items are not removed,
  only pushed out of the top-k).
- `raw_scores` (pre-penalty affinity) is available on the session output for
  consumers that threshold, blend, or explain; the wrapper exposes it on request
  but does not include it in the default top-k `DataFrame`.
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
- **`seen` semantics**: (a) bare call (no `seen`) excludes items with nonzero
  interaction values; (b) explicit `seen` from keys excludes a key whose value
  is exactly `0.0` (the value-`0.0` fidelity case) while the derived-only path
  does not — proving `seen_eff = Max(seen, interactions ≠ 0)` works.
- `raw_scores` output: equals `predict_scores` (f32) for all M items regardless
  of `mask`/`seen`/`repeat_penalty`, and is unaffected by the penalty/mask
  (tol `1e-5`).
- Eligibility `mask`: excluded items never appear in `top_indices`; mask
  overrides a repeat boost (`ρ < 0` + `mask = 0` → still excluded).
- `top_k` clamping: `k > num_items` returns `num_items` results.
- **K = 0 model (uniform signature)**: pass a width-0 `features` tensor; the
  graph scores correctly and the input names are identical to the K>0 case.
- int8 export: assert **top-k rank agreement** (set overlap / order), not score
  equality — quantization perturbs scores but preserves ranking (SilverTorch
  §4.2, §6.2). Testing score equality here would falsely flag int8 as broken.
- MLflow pyfunc roundtrip: GUID-in → GUID-out; default excludes already-watched
  (key-based, incl. value-`0.0`); `repeat_penalty`/`exclude` overrides behave as
  specified.
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
- `dtype="fp16"` with the `1e9` sentinels → safe, because masking/penalties run
  in fp32 (§4); the sentinels never enter fp16 arithmetic.
- K = 0 → width-0 `features` tensor, uniform signature (§4); no graph variant.
- Item with interaction value exactly `0.0` → "seen" only if the caller marks it
  via the `seen` input (the wrapper does, from dict keys); bare callers relying
  on the derived path treat it as unseen (§4 `seen` note).
- Unknown GUID/feature in MLflow wrapper input → skipped with warning.
- `meta_weight` / `weighting_config` / `sparsity_threshold`: training-time only,
  already baked into `S`; not represented in the graph, recorded in `vocab.json`
  provenance (`None` → `null`, not `0.0`).

## 12. Resolved — `top_scores` vs `raw_scores` semantics

Resolved in favour of exposing **both**. `top_scores` are the ranked
(adjusted + masked) scores `TopK` ordered on — correct for argmax/display
consumers. `raw_scores` (full `[batch, M]`) is the pre-penalty, pre-mask model
affinity, for consumers that threshold, blend (SilverTorch Value-Model style),
A/B-test, or explain. Cost is one extra graph output node (the `Gemm` result is
already materialized — no `Gather` needed since we return all M). `top_scores`
is **never** replaced by raw affinity, which would break the descending-sorted
assumption (a boosted repeat could outrank an item with higher raw affinity). A
consumer wanting raw affinity for only the returned items gathers `raw_scores`
by `top_indices` client-side.

## 13. Glossary

- **`MASK_PENALTY`** — graph constant (`1e9`, fp32) applied as
  `(mask − 1) · MASK_PENALTY`; drives masked-out items below `TopK`.
- **`EXCLUDE_SENTINEL`** — baked default value of the `repeat_penalty` input
  (`1e9`); reproduces native "exclude already-watched". Same magnitude as
  `MASK_PENALTY`, distinct role (a runtime input value vs a graph constant).
- **`seen` / `seen_eff`** — caller-supplied prior-interaction indicator and its
  union with the in-graph `interactions ≠ 0` derivation.
- **`S_items` / `W`** — `β`-folded first-`M`-rows extract of `S`, shape
  `[M, M+K]`, the `Gemm` weight (used with `transB=1`).
- **ρ (`repeat_penalty`)** — signed per-row repeat penalty: `+sentinel` exclude,
  `>0` demote, `0` neutral, `<0` boost.
```
