//! PyO3 surface for the crate.
//!
//! Per ADR-0003, each Python-visible subsystem lives in its own file under
//! `src/py/`. The crate root (`src/lib.rs`) holds only module declarations
//! and the `#[pymodule] fn _native` registration block; no `#[pyclass]` or
//! `#[pyfunction]` definitions live there directly.
//!
//! Subsystems land here one at a time as PRs against issue #64. Empty
//! submodules are placeholders until their content is moved from
//! `src/lib.rs`.

pub mod eval;
pub mod metrics;
pub mod registry;
// model, tuning, sasrec, two_tower modules pending follow-up commits.
