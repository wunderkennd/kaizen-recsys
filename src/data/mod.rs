//! Model-specific data paths.
//!
//! The long-format sparse pipeline lives in `crate::data_pipeline`; this
//! module hosts model input shapes that do not fit the
//! `(CsMat, CsMat, CsMat)` sparse pipeline. Every submodule is behind the
//! `ml-models` gate, so EASE-only builds compile none of them.
//!
//! - `sequences`: fixed-length, left-padded causal item sequences for
//!   SASRec.
//! - `triples`: `(user, positive-item)` training pairs plus a dense +
//!   categorical feature loader for the Two-Tower model.

#[cfg(feature = "ml-models")]
pub mod sequences;

#[cfg(feature = "ml-models")]
pub mod triples;
