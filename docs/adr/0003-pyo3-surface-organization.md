# ADR-0003: PyO3 surface organization

- **Status**: Proposed
- **Date**: 2026-06-01
- **Deciders**: project maintainers
- **Supersedes**: —
- **Related**: ADR-0001 (multi-model architecture), ADR-0002 (training and tuning parallelism), Issue #64

## Context

`src/lib.rs` is the single PyO3 surface for the crate. It is currently a
~1360-line file that bundles every Python-visible class, function, and
helper into one module. Concretely it holds:

| Lines (approx.) | Contents |
|---|---|
| 39–427 | `FeaseModel` pyclass with `predict`, `predict_raw`, `predict_batch`, `predict_similar_items`, `validate`, `evaluate`, `save` |
| 437–461 | `load_model` `#[pyfunction]` |
| 484–532 | `validate_data` `#[pyfunction]` |
| 545–634 | `random_split` / `temporal_split` / `leave_k_out_split` `#[pyfunction]`s |
| 654–811 | `build_and_train` `#[pyfunction]` |
| 826–983 | `FeaseRegistry` pyclass (territory routing) |
| 990–1027 | Six metric wrappers (`precision_at_k`, `recall_at_k`, `ndcg_at_k`, `mean_average_precision`, `coverage`, `hit_rate_at_k`) |
| 1056–1143 | `grid_search_py` / `random_search_py` `#[pyfunction]`s |
| 1147–1212 | Private helpers `parse_param_grid` / `search_result_to_py` |
| 1214–1241 | The `_native` `#[pymodule]` registration |
| 1243–1359 | `PyLayoutConstraint` pyclass + `optimize_layout` (added in PR #62) |

A knowledge-graph audit ran via `/graphify` over the codebase clusters
this file as community C1 — "PyO3 Bridge (FeaseModel)" — with
**cohesion 0.077**. Cohesion in this audit is the ratio of intra-community
edges to total edges incident on the community; 0.077 means the nodes
inside the community are barely more connected to each other than they
are to nodes elsewhere in the codebase. The graph's interpretation is
that this file is not one cohesive module — it is a translation layer
for seven distinct subsystems that happen to share an extension
boundary.

Concrete consequences observed today:

- PR #62 (modular WPO constraints + native preprocessing) added 209
  lines to `lib.rs`. Any concurrent PyO3 work in flight has to merge
  against that.
- Reviewers reading the file have to mentally segment seven unrelated
  concerns. The `#[pymodule]` registration block sits between metric
  wrappers and layout wiring, far from the pyclasses it registers.
- ADR-0001's roadmap will land SASRec and Two-Tower model wrappers via
  PyO3 in subsequent phases. With the current structure those become
  more cohesion-eroding additions to `lib.rs`.
- The internal `RustFeaseModel` (pure Rust, in `src/model.rs`) and the
  Python-facing `FeaseModel` (pyclass, in `src/lib.rs`) currently sit
  in completely different organizational layers despite being the same
  conceptual object. New contributors regularly grep `model.rs` for the
  Python surface and miss it.

The PyO3 procedural macro layer is layout-agnostic: `#[pyclass]` and
`#[pyfunction]` definitions live wherever the author puts them, and the
`#[pymodule] fn _native(...)` registration block is the only place that
ties them together. There is no technical reason to bundle them.

## Decision

We will split `src/lib.rs` into a thin entrypoint plus per-subsystem
PyO3 modules under `src/py/`:

| File | Contents |
|---|---|
| `src/lib.rs` | `mod` declarations, `#![recursion_limit]` attribute, the `_native` `#[pymodule]` registration block only. Target: < 50 lines. |
| `src/py/mod.rs` | `pub mod` declarations for the submodules below. |
| `src/py/model.rs` | `FeaseModel` pyclass, `build_and_train`, `load_model`. |
| `src/py/registry.rs` | `FeaseRegistry` pyclass. |
| `src/py/eval.rs` | `random_split`, `temporal_split`, `leave_k_out_split`, `validate_data`. |
| `src/py/metrics.rs` | The six metric wrappers. |
| `src/py/tuning.rs` | `grid_search_py`, `random_search_py`, private helpers `parse_param_grid`, `search_result_to_py`. |
| `src/py/layout.rs` | `PyLayoutConstraint`, `optimize_layout`. |

Naming conventions:

1. The directory is `src/py/` to clearly demarcate the PyO3 boundary
   from the pure-Rust core (`src/model.rs`, `src/evaluation.rs`,
   `src/tuning.rs`, etc.). The boundary is named in the path, not
   buried in a comment.
2. Pyclasses keep their existing Python-facing names (`FeaseModel`,
   `FeaseRegistry`, `LayoutConstraint`). No `Py` prefix on the Rust
   side either, except where ambiguity already exists today (e.g.
   `PyLayoutConstraint` shadows a Rust enum in `src/layout.rs` — the
   prefix stays for that one case).
3. Each `src/py/*.rs` file owns its `use` imports; no shared prelude.

Explicit non-decisions:

4. **Do not rename `_native`** or change `kzn_recsys/__init__.py`'s
   re-export list. The public Python API surface must stay
   byte-identical.
5. **Do not move pure-Rust modules into `src/py/`**. `src/model.rs`,
   `src/evaluation.rs`, `src/tuning.rs`, `src/serving.rs`,
   `src/serialization.rs`, `src/metrics.rs`, `src/data_pipeline.rs`,
   `src/data_validation.rs`, `src/weighting.rs`, `src/layout.rs`,
   `src/transform.rs`, and `src/models/` stay where they are. This
   ADR is only about the PyO3 wrapper layer.
6. **Do not split per concrete model.** A future `src/py/sasrec.rs`
   or `src/py/two_tower.rs` might be added when ADR-0001's later
   phases land, but the initial split groups by *subsystem*
   (model / registry / eval / metrics / tuning / layout), not by
   *model kind*. SASRec's PyO3 surface will most likely land alongside
   `FeaseModel` in `src/py/model.rs` if it shares the wrapper class,
   or as a new file if it does not. That call is out of scope for
   this ADR.

## Alternatives considered

### A. Do nothing — keep one `lib.rs`

- ✅ Zero refactor cost. No risk of breaking the PyO3 registration.
- ❌ Cohesion score continues to drop as ADR-0001 phases add more
  pyclasses. Merge-conflict surface area grows with every PyO3-touching
  PR. New contributors keep grepping the wrong file.

**Rejected** — the cost compounds with every future PR; landing the
split now is cheaper than landing it once SASRec and Two-Tower
wrappers exist.

### B. Split by Python-API distinction (`classes.rs` + `functions.rs`)

Put all `#[pyclass]` definitions in one file, all `#[pyfunction]`
definitions in another.

- ✅ Mechanical rule: PyO3 macro type decides the file.
- ❌ The cohesion finding is about *subsystem*, not *macro type*. A
  `classes.rs` containing `FeaseModel` + `FeaseRegistry` +
  `PyLayoutConstraint` still has the same problem in microcosm: three
  unrelated subsystems sharing a file. Re-introduces the smell at a
  smaller scale.

**Rejected** — solves the wrong axis.

### C. Split per concrete model (`src/py/ease.rs`, future
`src/py/sasrec.rs`, etc.)

Group the PyO3 surface by which recommender it wraps.

- ✅ Aligns visually with `src/models/` (one file per model).
- ❌ The Python surface is broader than per-model wrappers: split
  functions, metric wrappers, layout optimization, and data validation
  are model-agnostic. They would need a `src/py/common.rs` catch-all,
  which is just a renamed `lib.rs`.
- ❌ Premature commitment. SASRec and Two-Tower PyO3 surfaces don't
  exist yet; designing the layout around them now is speculative.

**Rejected** — premature; deferred to a future ADR if/when concrete
model wrappers diverge.

### D. Use a single `src/python.rs` file with `mod` blocks instead of
files

- ✅ One file, no directory.
- ❌ Doesn't reduce the line count. Doesn't help reviewers. The whole
  point of the split is making each subsystem editable independently.

**Rejected** — cosmetic, doesn't address the cohesion finding.

### E. Move everything into `kzn_recsys/_native.pyi` stubs and keep
Rust files as-is

Use Python stub files to document the surface; leave Rust untouched.

- ✅ Improves Python IDE / type-checker experience.
- ❌ Orthogonal to the cohesion problem. Stub files describe the
  surface; this ADR is about reorganizing the surface itself. Both
  could happen; one doesn't replace the other.

**Deferred** — worth doing separately, not as a substitute.

## Consequences

### Positive

- **Cohesion score improves**: re-running `/graphify --update` after
  the split should either split community C1 into multiple smaller
  communities (one per `src/py/*.rs`) or substantially improve C1's
  cohesion score above the current 0.077.
- **Merge-conflict surface shrinks**: PyO3-touching PRs only touch
  the subsystem file they affect, not a single shared file. PR #62's
  WPO additions, the upcoming SASRec wrappers, and any tuning surface
  changes can land in parallel without textual conflicts.
- **Reviewers can read one subsystem at a time**: each `src/py/*.rs`
  file is small enough (50–400 lines) to hold in head.
- **`_native` registration becomes the literal map of the Python
  surface**: reading `src/lib.rs` end-to-end tells you exactly which
  symbols are exposed, in one place, with no implementation noise.
- **Lower discoverability cost for new contributors**: the
  `src/py/` directory name is self-documenting. "Where do I add a new
  Python-callable function?" becomes a one-second question.

### Negative / costs

- **One-time mechanical refactor cost**: ~1300 lines moved, ~7 new
  files created. Affects every PyO3-touching open PR (specifically
  PR #62 if not yet merged).
- **Slightly more PyO3 visibility plumbing**: each pyclass and
  pyfunction needs `pub` so `src/lib.rs` can reference it. Helpers
  like `parse_param_grid` can stay `pub(crate)` or private to their
  module. Modest annotation work, but explicit.
- **Two files instead of one to look at when tracing a Python call
  end-to-end**: the call now flows `kzn_recsys` → `src/lib.rs` (just
  registration) → `src/py/<subsystem>.rs` (the actual wrapper) →
  `src/<core>.rs` (the pure Rust). Three hops instead of two. This
  is offset by the fact that the wrapper layer is now named on the
  path.

### Risks

- **PyO3 registration gaps**: if a `#[pyclass]` or `#[pyfunction]`
  is moved but not re-added to the `_native` block in `src/lib.rs`,
  it disappears from the Python surface silently — Rust compiles
  fine but `import kzn_recsys; kzn_recsys.foo` fails at runtime.
  Mitigation: the acceptance criteria (below) include a `dir()`
  snapshot test asserting the post-refactor Python surface is
  byte-identical to the pre-refactor surface.
- **`use` import drift**: each new file needs its own imports;
  copying them wrong produces compile errors, which are loud and
  caught by `cargo check` per-file during the migration.
- **`#![recursion_limit]` placement**: the existing
  `#![recursion_limit = "256"]` attribute (added for burn-generic
  SASRec instantiation) must stay in `src/lib.rs` — it's a
  crate-level attribute. Documented in the migration steps.

## Phased rollout

This ADR lands as a doc-only PR. The implementation is one PR
(tracked as Issue #64), broken into atomic commits for review
clarity.

| Phase | Scope | Gate |
|-------|-------|------|
| **0** | This ADR. Doc-only. | Merged on `main`. |
| **1** | Create `src/py/` directory and `src/py/mod.rs` with empty `pub mod` declarations. | `cargo build` succeeds with no functional change. |
| **2** | Move one subsystem per commit (`model.rs`, then `registry.rs`, then `eval.rs`, then `metrics.rs`, then `tuning.rs`, then `layout.rs`). Each commit: cut from `lib.rs`, paste into new file with `use` imports, update the `_native` block to import from new location, `cargo check` per commit. | Each commit compiles in isolation. Test suite (88 Rust + 46 Python) passes after the final commit. |
| **3** | Verify Python surface byte-identity via `dir()` snapshot diff. | `python -c "import kzn_recsys; print(sorted(dir(kzn_recsys)))"` output is unchanged from `main`. |
| **4** | Re-run `/graphify --update` on the post-refactor branch. | Community C1's cohesion score either improves materially above 0.077 or C1 splits into multiple smaller communities with individually higher cohesion. |

Phase 1 is a single setup commit. Phases 2 is the bulk of the work,
intentionally split into ~6 atomic commits so that any reviewer can
read one move at a time and so that a partial revert (e.g. if one
subsystem move turns out to need further care) is trivial.

## References

- Issue #64 — implementation tracking issue for this ADR.
- Issue #63 — `evaluate_trial` / `evaluate_model` consolidation, a
  separate cohesion-driven refactor surfaced by the same `/graphify`
  audit. Independent of this ADR.
- ADR-0001 (`docs/adr/0001-multi-model-architecture.md`) — multi-model
  roadmap; future SASRec / Two-Tower wrappers will land into the
  structure this ADR establishes.
- ADR-0002 (`docs/adr/0002-training-parallelism.md`) — established
  `tuning.rs`'s rayon parallelism, the rust-side counterpart to the
  `src/py/tuning.rs` wrapper introduced here.
- PR #62 — most recent PyO3 surface addition (modular WPO + native
  preprocessing); a primary contributor to the current `lib.rs`
  line count.
- PyO3 module documentation: https://pyo3.rs/latest/module
- PyO3 class organization patterns:
  https://pyo3.rs/latest/class.html#class-customizations
