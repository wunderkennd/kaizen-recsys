# ADR-0005: Pluggable hyperparameter-search strategies

- **Status**: Accepted
- **Date**: 2026-10-06
- **Deciders**: project maintainers
- **Supersedes**: —
- **Related**: ADR-0002 (rayon-parallel trial runner), issue #97, issue #51
  (`EvalAdapter` routing in the per-fold scorers), issue #96 (BERT4Rec)

## Context

`src/tuning.rs` offered two search methods, grid and random, and they were
not uniformly available: EASE, SASRec and Two-Tower each had
`grid_search_*` / `random_search_*` entrypoints while BERT4Rec had none.
Both methods are non-adaptive — every trial is chosen before the first
result is known — and for the burn-backed models a trial is minutes of
CPU, so exhaustive grids are expensive and random draws waste budget on
regions the first few trials already ruled out.

The space description was also the limiting factor: every `*ParamGrid`
was a struct of candidate *lists*, so there was no way to say "learning
rate anywhere in `[1e-4, 1e-2]`, log-scaled", which is what an adaptive
method needs to model.

Constraints that shaped the design:

- ADR-0002's parallelism must survive: the `(params × fold)` product runs
  on rayon's global pool, and that is where most of the wall-clock win
  for EASE sweeps comes from.
- `grid_search_with` / `random_search_with` are public and used from the
  Python layer; their results for a fixed seed are relied on by the
  determinism tests (`test_parallel_grid_search_matches_sequential`,
  `test_ease_search_matches_legacy_concrete`). Refactoring under them
  must not change a single bit of output.
- No new runtime dependencies for the default build.

## Decision

1. **A model-agnostic space layer** (`src/tuning/space.rs`). A
   `ParamSpace` is an ordered list of named `Axis`es: `Choice(values)`,
   `Uniform`, `LogUniform`, `Int`, `LogInt`. A configuration drawn from it
   is an `Assignment` (one `Value` per axis). The typed grids
   (`ParamGrid`, `SasRecParamGrid`, …) become `Choice`-only spaces.

2. **`SearchSpace` gains codecs, loses hand-written loops.** The trait now
   requires `axes()`, `decode(&Assignment) -> Params` and
   `encode(&Params) -> Assignment`; `combinations()` and `sample_one()`
   are provided in terms of those. Per-model knowledge lives in one
   `ParamSchema` impl (names, defaults, codecs) shared by the typed grid
   and the dict-driven `DynSpace<Schema>` the Python `tune_*` functions
   build. Ordering contract: axes in field order, first axis outermost,
   `Choice` sampled with `SliceRandom::choose` in axis order — which is
   exactly what the old nested loops and per-field `choose` calls did, so
   grid and random output is byte-identical (regression guards:
   `test_grid_combinations_match_legacy_cartesian_product_bitwise`,
   `test_random_search_params_match_legacy_sampling_bitwise`).

3. **An ask-only `SearchStrategy` seam** (`src/tuning/strategy.rs`):
   `propose(space, history, max_n, rng) -> Vec<Assignment>`. The runner
   `tune_with` generates folds once, then loops *ask → evaluate the batch
   in parallel with rayon → append observations* until the strategy
   returns nothing or `max_trials` is reached. Grid and random are
   strategies that return one batch; `grid_search_with` /
   `random_search_with` are thin wrappers over `tune_with`. Warm starts
   are prior `TrialResult`s encoded into the history.

4. **TPE first, GP later if ever.** The Bayesian strategy is a
   Tree-structured Parzen Estimator (Bergstra et al. 2011): split the
   history into the best `gamma` fraction (`l`) and the rest (`g`), fit a
   truncated-Gaussian Parzen estimator per numeric axis (unit interval,
   log axes in log space, Optuna's neighbour-gap bandwidth rule with a
   broad prior component) and Laplace-smoothed frequencies per `Choice`
   axis, then keep the best of `n_candidates` draws from `l` under
   `log l − log g`. TPE is dependency-free, handles mixed
   discrete/continuous/log axes natively, and is Optuna's default, so it
   is directly comparable. A Gaussian-process BO would need a pure-Rust
   GP crate and a continuous-only space; it is not worth that today.

5. **Batches, not fake observations.** To keep ADR-0002's parallelism, TPE
   proposes `batch_size` configurations per round, all drawn from the same
   `l` / `g`. This is independent sampling around the current optimum,
   not the "constant liar" / kriging-believer sequential refinement.
   Smaller batches are more adaptive; larger ones keep more rayon workers
   busy per round; `batch_size = 1` is classic sequential TPE. The
   default is 4.

6. **Scope cut: no multi-fidelity yet.** Successive halving / Hyperband
   over epochs (issue #97 Phase 4) is deferred until a real workload shows
   trial cost, not trial count, is the bottleneck. It would need
   `FoldEvaluator` to accept a fidelity parameter — a small, additive
   extension of the seam defined here.

## Consequences

- Every model gets every strategy with no per-model search code: adding a
  strategy is one file; adding a model is one `ParamSchema` + one
  `FoldEvaluator`.
- Python gains `tune_{ease,sasrec,two_tower,bert4rec}` with a dict space
  (`[..]` lists, `("log", lo, hi)` tuples or `{"type", "low", "high"}`
  dicts), `strategy="grid" | "random" | "tpe"`, `max_trials`, optional
  `batch_size` / `n_startup` / `warm_start`. Results keep the existing
  dict shape plus a `strategy` key. BERT4Rec also gets the classic
  `grid_search_bert4rec` / `random_search_bert4rec`.
- One intentional behaviour change for Rust callers only: an *empty*
  candidate list in a typed grid used to mean "default value, no RNG
  draw"; it is now a one-element axis that does draw. The Python layer
  never passes empty lists, so its outputs are unchanged.
- Determinism: a whole search is a function of `(seed, space, strategy
  settings)`; execution order and thread count do not affect it.
- Acceptance evidence (issue #97): on the Two-Tower fixture, TPE with half
  the budget matches or beats random search's best NDCG@k in a majority
  of seeds (`test_tpe_half_budget_matches_random_best_two_tower`,
  release-mode `--ignored`); on a synthetic 3-axis bowl it wins 7/7 seeds
  at equal budget and 5/7 at half budget
  (`strategy::tests::tpe_beats_random_on_quadratic_bowl`).

## Alternatives considered

- **Optuna bridge** (`kzn_recsys[tune]` extra driving a Python-exposed
  `evaluate_fold`). Rejected as the primary path: Python-side orchestration
  loses rayon parallelism across trials and Optuna is a heavy optional
  dependency. It remains cheap to add on top of the seam later.
- **Gaussian-process BO** (`friedrich` / `egobox`). Rejected for now: new
  dependencies, continuous-only space, and no evidence it beats TPE at
  these budgets.
- **Sequential TPE with constant-liar batching.** Rejected: more code for
  a modest gain at `batch_size ≤ 4`, and it would make results depend on
  the liar value; revisit if large batches prove too exploratory.
