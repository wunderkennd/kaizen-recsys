# ADR-0002: Training and tuning parallelism

- **Status**: Proposed
- **Date**: 2026-05-14
- **Deciders**: project maintainers
- **Supersedes**: —
- **Related**: ADR-0001 (multi-model architecture)

## Context

FEASE training is single-threaded today. The closed-form EASE solve in
`src/model.rs` is dominated by two costs:

1. **Sparse Gram matrix construction** — four blocks (`G_11 = XᵀX`,
   `G_12 = XᵀU`, `G_21 = G_12ᵀ`, `G_22 = UᵀU`) computed sequentially via
   `sprs` sparse matmul. Cost grows with NNZ.
2. **Dense matrix inversion** — `(G + λI)` is inverted via nalgebra's
   built-in dense LU. Cost is `O((M+K)³)`. nalgebra's default backend is
   single-threaded pure Rust.

For typical FEASE workloads (M+K in the few thousands to tens of
thousands), the dense inversion dominates wall-clock time. For
hyperparameter sweeps via `tuning::grid_search` or
`tuning::random_search`, the sequential cost is multiplied by
`n_param_combos × n_folds` — every `(params, fold)` pair trains a fresh
model independently and the whole search runs serially.

Concrete situation:

- `src/tuning.rs::grid_search` loops over `params` then loops over
  `fold_paths`, calling `train_for_kfold` for each `(params, fold)` pair.
- `tempfile::tempdir()` writes fold Parquet files **once** in
  `generate_kfold_splits`; trials only **read** those files. There is
  no shared mutable state across trials — they are pure functions of
  `(params, train_path, test_path)`.
- `rayon` is already a dep, used by `src/serving.rs` for
  parallel batch prediction. Adding parallel iteration to `tuning.rs`
  pulls in zero new dependencies.

The current state leaves two specific wins on the table that don't
require any change to the training math.

## Decision

We will land two independently-shippable parallelism improvements,
in order:

1. **Parallel `(params × fold)` trial execution in `tuning.rs`** via
   rayon's `par_iter`. Default behavior (no feature flag). The outer
   product of parameter combinations and CV folds is embarrassingly
   parallel and uses already-spawned worker threads from rayon's global
   pool. Expected speedup: linear in cores up to `min(n_combos × n_folds,
   num_cpus)`.

2. **Multi-threaded BLAS for the dense Gram inversion** behind a new
   `fast-blas` Cargo feature, **default off**. Enables nalgebra's
   `blas` feature, which delegates LU/Cholesky to a system BLAS
   (OpenBLAS on Linux/Windows, Apple Accelerate on macOS, optionally
   MKL/Netlib). Expected speedup: 2-4× on the inversion step for catalogs
   in the 5K-50K item range; larger for bigger catalogs.

Explicit non-decisions (deferred or rejected):

3. **Parallel Gram-block construction** — defer indefinitely. The four
   block matmuls could be issued via `rayon::scope`, but they are small
   relative to the inversion for any non-toy catalog. Code complexity
   not worth the gain.

4. **Per-territory training parallelism** — out of scope for this ADR.
   When/if we add a batch-train-N-models entrypoint (sibling to
   `FeaseModelRegistry` on the serving side), it's another `par_iter`
   over independent calls to `build_and_train`. Mention it now, design
   it when there's a real consumer.

5. **GPU acceleration** — out of scope. The closed-form solve is
   batched dense LA, not iterative SGD; the win from GPU is marginal
   compared to enabling a tuned BLAS. SASRec and Two-Tower (ADR-0001)
   will have their own GPU story via `burn-tch`/`burn-wgpu` when those
   land.

## Alternatives considered

### A. Do nothing — keep single-threaded training
- ✅ No new code, no new feature flags, no new wheel-distribution
  complexity.
- ❌ Hyperparameter sweeps stay 4-16× slower than necessary on any
  multicore host. A 10-combo × 5-fold sweep that finishes in 5 minutes
  parallel runs ~40 minutes serial today on an 8-core machine.

**Rejected** — easy wins, low risk.

### B. Always-on multi-threaded BLAS (default `fast-blas` feature)
Flip nalgebra's BLAS backend on by default; ship wheels with a bundled
OpenBLAS for every platform.

- ✅ Every user gets the inversion speedup without opting in.
- ❌ Adds 5-50 MB to every wheel (auditwheel-bundled libopenblas on
  manylinux).
- ❌ `build_wheel.yml` cross-platform story (4 platforms) gets harder:
  Apple Accelerate works without bundling, OpenBLAS needs auditwheel on
  Linux and DLL bundling on Windows.
- ❌ Multi-threaded BLAS has historically had subtle numerical
  reproducibility issues (thread-count-dependent rounding); pinning
  exact bit-level reproducibility across hosts becomes harder.

**Rejected** — costs disproportionate to the user population that
needs the speedup. Power users opt in.

### C. Replace nalgebra's inversion with a hand-rolled parallel LU
Skip BLAS entirely; implement Cholesky/LU with rayon-parallelized
blocked decomposition.

- ✅ No system library dependency, no wheel-bundling.
- ❌ Implementing a numerically-stable, well-tested parallel LU is
  weeks of work. nalgebra-lapack and OpenBLAS exist; reimplementing
  them is a textbook example of NIH.

**Rejected** — false economy.

### D. Async/tokio for tuning parallelism instead of rayon
Run trials concurrently via a tokio runtime instead of rayon.

- ✅ Hypothetically composable with async I/O elsewhere.
- ❌ Wrong tool — trials are CPU-bound, not I/O-bound. Tokio's runtime
  is optimized for many small I/O tasks; rayon's work-stealing scheduler
  is optimized for CPU-bound parallel iteration. No existing async I/O
  in the crate to compose with.

**Rejected** — tool mismatch.

### E. Materialize the combined matrix Z and use a parallel sparse matmul
Build `Z = [X | U]` and compute `G = ZᵀZ` in one parallel call.

- ✅ Single matmul, easier to parallelize at the library level.
- ❌ Violates the existing memory-efficiency invariant (CLAUDE.md
  §"Key Concepts"): "Z is never materialized. G is computed in 4 sparse
  blocks, keeping memory at O((M+K)²) independent of user count N."
- ❌ Doesn't address the dominant cost (inversion).

**Rejected** — breaks an explicit invariant for a non-bottleneck.

## Consequences

### Positive
- **Tuning sweeps speed up linearly** in cores for `(n_combos × n_folds)`
  up to host parallelism. On an 8-core CI runner, a 10×5 = 50-trial
  sweep collapses from 50 sequential trains to 8 concurrent batches.
- **Single-model training speeds up 2-4×** on the inversion step for
  users who opt into `fast-blas`. No effect on default wheel users.
- **No change to model output**. Both decisions are parallelism
  rewrites of existing math, not algorithmic changes. Same matrices in,
  same S matrix out, same recommendations.
- **No new always-on dependencies**. Rayon is already in tree. nalgebra
  is already in tree; the BLAS feature is gated.

### Negative / costs
- **Memory footprint of parallel tuning** scales with the number of
  concurrent trials. Each trial holds its own Gram matrix and S matrix
  in memory; for large catalogs this can push host memory. Mitigation:
  rayon's default thread count caps concurrency at `num_cpus`; users
  who need lower concurrency can set `RAYON_NUM_THREADS`.
- **`fast-blas` wheels are not pre-built in v1**. Users opt in by
  building from source with `--features fast-blas`. The standard
  PyPI wheels ship without BLAS. We document this clearly in
  `README.md` when the feature lands.
- **Two ways to "train a FEASE model"** in terms of underlying math
  paths (Rust LU vs. BLAS LU). Same numerical result up to BLAS-vs-pure-
  Rust floating-point ordering; tests should not pin bit-exact outputs.
  Note this in the test files that touch the inversion.
- **Determinism under parallel tuning**: trial *order* in `all_trials`
  becomes non-deterministic; trial *results* stay deterministic per
  trial (RNG is seeded per-trial via `seed + trial_idx`); `best_params`
  selection stays deterministic given the same trial set, because
  ties are broken on the deterministic `trial_idx`.

### Risks
- **Numerical drift between BLAS implementations**: OpenBLAS, MKL,
  Accelerate, and Netlib can produce slightly different floating-point
  results due to ordering of FMA operations. Acceptable for ranking
  (rank order is robust to sub-ulp drift), but tests must use tolerance
  comparisons (already the case).
- **rayon global pool contention**: if a user's runtime already uses
  rayon for other work, our tuning may starve their pool or vice versa.
  Mitigation: document the `RAYON_NUM_THREADS` env var as the standard
  knob, don't construct private thread pools.
- **`fast-blas` build failures** on exotic platforms (musl, manylinux
  variants without a system BLAS). Mitigation: `fast-blas` is opt-in;
  failure is loud and pre-build (linker error), not silent.

## Phased rollout

| Phase | Scope | Gate |
|-------|-------|------|
| **1** | Parallelize `grid_search` and `random_search` trial loops via `par_iter`. Verify `best_params` matches sequential baseline on a regression test. | All existing tuning tests green; new test asserts parallel vs. sequential best_params identical for a fixed seed; trial wall-clock measurably lower on a multi-core CI runner. |
| **2** | Add `fast-blas` Cargo feature: `fast-blas = ["nalgebra/blas"]` plus per-platform BLAS deps. Document opt-in in `README.md`. | `cargo build` (no features) unchanged; `cargo build --features fast-blas` succeeds on Linux/macOS/Windows; default wheel size unchanged; opt-in wheel builds locally and runs an inversion test. |

Phase 1 is a single-PR change with no new deps. Phase 2 adds optional
deps and platform-specific BLAS feature wiring; can be split per
platform if needed. Both phases are independent of ADR-0001's
multi-model work and can land in any order relative to it.

## References

- nalgebra BLAS feature: https://nalgebra.org/docs/user_guide/getting_started/#cargo-features
- rayon parallel iteration: https://docs.rs/rayon/latest/rayon/
- Apple Accelerate framework (macOS, no install needed):
  https://developer.apple.com/documentation/accelerate
- OpenBLAS: https://www.openblas.net/
- auditwheel (Linux wheel bundling): https://github.com/pypa/auditwheel

## Amendment — 2026-05-16 (Phase 2 implementation note)

The original decision text above is retained unchanged as the historical
record. This note reconciles it with what Phase 2 actually shipped (PR #31).

The `fast-blas = ["nalgebra/blas"]` shorthand used in **§Decision #2** and
the **Phase 2** row of the rollout table is *not* literally realizable:
**nalgebra 0.34 (our pinned 0.34.1) exposes no `blas` Cargo feature** — its
LU/Cholesky BLAS delegation lives in the sibling crate `nalgebra-lapack`.

Phase 2 therefore realizes the same decision via an optional
`nalgebra-lapack` 0.27 dependency (the only release tracking `nalgebra
^0.34`), declared `default-features = false` to drop its default
`lapack-netlib` backend, with per-platform backend auto-selection through
`[target.*]` tables: `lapack-accelerate` on macOS, `lapack-openblas`
elsewhere. `invert_gram()` in `src/model.rs` cfg-gates the pure-Rust
`DMatrix::try_inverse` (default) vs. `nalgebra_lapack::LU::inverse()`
(`fast-blas`). Intent, default-off posture, opt-in build-from-source story,
and per-platform backends are all unchanged from the decision above; only
the dependency mechanism differs. See PR #31 for the implementation and
verification matrix.
