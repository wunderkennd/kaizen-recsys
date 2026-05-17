//! Model-specific data paths.
//!
//! The long-format sparse pipeline lives in `crate::data_pipeline`; this
//! module hosts model input shapes that do not fit the sparse pipeline.
//!
//! The `triples` submodule provides `(user, positive-item)` training
//! pairs plus a dense + categorical feature loader for the Two-Tower
//! model, behind the `ml-models` gate so EASE-only builds skip it.

#[cfg(feature = "ml-models")]
pub mod triples;
