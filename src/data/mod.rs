//! Model-specific data paths.
//!
//! The long-format sparse pipeline lives in `crate::data_pipeline`; this
//! module hosts the input shapes the new (ADR-0001 Phase 3+) models need
//! and that do not fit the `(CsMat, CsMat, CsMat)` shape.
//!
//! Phase 3 (issue #36) adds `sequences`: fixed-length left-padded causal
//! item sequences for SASRec. Later phases add their own files behind the
//! same `ml-models` gate, so EASE-only builds never compile any of them.

#[cfg(feature = "ml-models")]
pub mod sequences;
