//! SASRec — causal self-attention sequence recommender.
//!
//! Phase 2a (issue #24) only wires the `ml-models` feature gate; this
//! module is intentionally empty. The minimal forward pass lands in
//! Phase 2b (issue #25), training in Phase 3, and the `RecModel` impl
//! plus PyO3 surface in Phase 4. See ADR-0001 for the overall plan.
