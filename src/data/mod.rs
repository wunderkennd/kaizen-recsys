//! Model-specific data paths.
//!
//! The long-format sparse pipeline lives in `crate::data_pipeline`; this
//! module hosts model input shapes that do not fit the
//! `(CsMat, CsMat, CsMat)` sparse pipeline. Every submodule is behind the
//! `ml-models` gate, so EASE-only builds compile none of them.
//!
//! - `sequences`: fixed-length, left-padded causal item sequences for
//!   SASRec.
//! - `masked_sequences`: fixed-length, left-padded *bidirectional*
//!   sequences with random `[MASK]` targets and log-bucketed `days_ago`
//!   positions for BERT4Rec (#96).
//! - `triples`: `(user, positive-item)` training pairs plus a dense +
//!   categorical feature loader for the Two-Tower model.
//!
//! [`days_ago_to_log2_bucket`] / [`days_ago_to_bucket`] live here, outside
//! the feature gate, because the model-agnostic evaluation harness
//! (`crate::evaluation::Bert4RecEvalAdapter`) needs the same bucketing rule
//! the BERT4Rec data path trains with, and `evaluation.rs` compiles in
//! every build.

#[cfg(feature = "ml-models")]
pub mod sequences;

#[cfg(feature = "ml-models")]
pub mod masked_sequences;

#[cfg(feature = "ml-models")]
pub mod triples;

/// Unclamped logarithmic (base-2) recency bucket for a `days_ago` value.
///
/// `0 → 0`, `1 → 0`, `2 → 1`, `4 → 2`, `8 → 3`, `365 → 8`, `1825 → 10`.
/// Anything `< 1` (including negative, `NaN`, and sub-day values) is
/// bucket `0` — "now". Two interactions in the same doubling window share
/// a bucket regardless of absolute date, which is the relative-position
/// behaviour BERT4Rec wants for non-sequential discovery (#96 §Key Design
/// Decisions #1). Callers that own a fixed position-embedding table clamp
/// with [`days_ago_to_bucket`].
#[inline]
pub fn days_ago_to_log2_bucket(days_ago: f64) -> usize {
    if days_ago.is_nan() || days_ago < 1.0 {
        return 0;
    }
    // `as usize` saturates, so an absurdly large `days_ago` is safe.
    days_ago.log2().floor() as usize
}

/// [`days_ago_to_log2_bucket`] clamped to `[0, max_buckets)`.
///
/// `max_buckets == 0` is treated as `1` (everything collapses to bucket
/// `0`) rather than underflowing.
#[inline]
#[cfg_attr(not(feature = "ml-models"), allow(dead_code))]
pub fn days_ago_to_bucket(days_ago: f64, max_buckets: usize) -> usize {
    let cap = max_buckets.max(1) - 1;
    days_ago_to_log2_bucket(days_ago).min(cap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log2_bucket_matches_spec_table() {
        assert_eq!(days_ago_to_log2_bucket(0.0), 0);
        assert_eq!(days_ago_to_log2_bucket(1.0), 0);
        assert_eq!(days_ago_to_log2_bucket(2.0), 1);
        assert_eq!(days_ago_to_log2_bucket(4.0), 2);
        assert_eq!(days_ago_to_log2_bucket(8.0), 3);
        assert_eq!(days_ago_to_log2_bucket(365.0), 8);
        assert_eq!(days_ago_to_log2_bucket(1825.0), 10);
    }

    #[test]
    fn log2_bucket_handles_degenerate_inputs() {
        assert_eq!(days_ago_to_log2_bucket(-3.0), 0);
        assert_eq!(days_ago_to_log2_bucket(0.5), 0);
        assert_eq!(days_ago_to_log2_bucket(f64::NAN), 0);
        assert_eq!(days_ago_to_log2_bucket(f64::INFINITY), usize::MAX);
    }

    #[test]
    fn clamped_bucket_respects_cap() {
        assert_eq!(days_ago_to_bucket(1825.0, 32), 10);
        assert_eq!(days_ago_to_bucket(1825.0, 4), 3);
        assert_eq!(days_ago_to_bucket(1825.0, 1), 0);
        assert_eq!(days_ago_to_bucket(1825.0, 0), 0);
        assert_eq!(days_ago_to_bucket(0.0, 32), 0);
    }
}
