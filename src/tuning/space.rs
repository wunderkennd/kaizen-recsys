//! Model-agnostic parameter spaces for hyperparameter search (issue #97).
//!
//! A [`ParamSpace`] is an ordered list of named [`Axis`]es. Each axis is
//! either a finite [`Axis::Choice`] (the only shape the original grid /
//! random search understood) or a numeric range — [`Axis::Uniform`],
//! [`Axis::LogUniform`], [`Axis::Int`], [`Axis::LogInt`] — so adaptive
//! strategies (TPE, see `strategy.rs`) have a real space to model.
//!
//! A concrete configuration drawn from a space is an [`Assignment`]: one
//! [`Value`] per axis, in axis order. Model-specific parameter structs
//! (`HyperParams`, `SasRecParams`, …) are produced from an `Assignment`
//! by the model's `SearchSpace::decode`, and turned back into one by
//! `encode` for warm starts — so strategies never see model types.
//!
//! Sampling is deliberately simple and deterministic: every axis consumes
//! the shared `StdRng` in axis order, and a `Choice` axis samples with
//! `SliceRandom::choose`, exactly as the original per-field
//! `self.alpha.choose(rng)` did. That is what keeps `random_search_with`
//! byte-identical to its pre-#97 output for a fixed seed.

use anyhow::{Result, anyhow, bail};
use rand::Rng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

/// A single hyperparameter value. Integer-valued knobs (`embedding_dim`,
/// `num_epochs`, …) are `Int`; everything else is `Float`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Float(f64),
    Int(i64),
}

impl Value {
    pub fn as_f64(self) -> f64 {
        match self {
            Value::Float(f) => f,
            Value::Int(i) => i as f64,
        }
    }

    /// Integer view. A `Float` is accepted only when it is integral
    /// (so a user-written `64.0` still means 64) — anything else is an
    /// error, never a silent truncation.
    pub fn as_i64(self) -> Result<i64> {
        match self {
            Value::Int(i) => Ok(i),
            Value::Float(f) if f.is_finite() && f.fract() == 0.0 => Ok(f as i64),
            Value::Float(f) => bail!("expected an integer value, got {f}"),
        }
    }

    pub fn as_usize(self) -> Result<usize> {
        let i = self.as_i64()?;
        usize::try_from(i).map_err(|_| anyhow!("expected a non-negative integer, got {i}"))
    }
}

impl From<f64> for Value {
    fn from(f: f64) -> Self {
        Value::Float(f)
    }
}

impl From<usize> for Value {
    fn from(u: usize) -> Self {
        Value::Int(u as i64)
    }
}

impl From<i64> for Value {
    fn from(i: i64) -> Self {
        Value::Int(i)
    }
}

/// One dimension of a [`ParamSpace`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Axis {
    /// A finite list of candidates (the classic grid axis). Grid search
    /// enumerates it; random search and TPE pick from it.
    Choice(Vec<Value>),
    /// Continuous, uniform on `[low, high]`.
    Uniform { low: f64, high: f64 },
    /// Continuous, uniform in `log(x)` on `[low, high]` (`low > 0`). The
    /// right shape for learning rates and regularization strengths.
    LogUniform { low: f64, high: f64 },
    /// Integer, uniform on `low..=high`.
    Int { low: i64, high: i64 },
    /// Integer, uniform in `log(x)` on `low..=high` (`low >= 1`). The
    /// right shape for embedding dims and epoch counts.
    LogInt { low: i64, high: i64 },
}

impl Axis {
    /// Convenience constructor for a `Choice` axis of floats.
    pub fn choice_f64(values: &[f64]) -> Self {
        Axis::Choice(values.iter().map(|&v| Value::Float(v)).collect())
    }

    /// Convenience constructor for a `Choice` axis of integers.
    pub fn choice_usize(values: &[usize]) -> Self {
        Axis::Choice(values.iter().map(|&v| Value::Int(v as i64)).collect())
    }

    fn validate(&self, name: &str) -> Result<()> {
        match self {
            Axis::Choice(v) if v.is_empty() => bail!("axis `{name}`: Choice list is empty"),
            Axis::Choice(_) => Ok(()),
            Axis::Uniform { low, high } => {
                if !(low.is_finite() && high.is_finite()) || low > high {
                    bail!("axis `{name}`: Uniform needs finite low <= high, got [{low}, {high}]");
                }
                Ok(())
            }
            Axis::LogUniform { low, high } => {
                if !(low.is_finite() && high.is_finite()) || *low <= 0.0 || low > high {
                    bail!(
                        "axis `{name}`: LogUniform needs finite 0 < low <= high, got [{low}, {high}]"
                    );
                }
                Ok(())
            }
            Axis::Int { low, high } => {
                if low > high {
                    bail!("axis `{name}`: Int needs low <= high, got [{low}, {high}]");
                }
                Ok(())
            }
            Axis::LogInt { low, high } => {
                if *low < 1 || low > high {
                    bail!("axis `{name}`: LogInt needs 1 <= low <= high, got [{low}, {high}]");
                }
                Ok(())
            }
        }
    }

    /// `true` for axes with finitely many values (enumerable by grid
    /// search and de-duplicable by adaptive strategies).
    pub fn is_finite(&self) -> bool {
        matches!(
            self,
            Axis::Choice(_) | Axis::Int { .. } | Axis::LogInt { .. }
        )
    }

    /// Number of distinct values for a finite axis, `None` otherwise.
    /// Computed in `i128` and saturated to `usize::MAX`, so a valid but
    /// enormous `Int` range (`i64::MIN..=i64::MAX`) cannot overflow.
    pub fn cardinality(&self) -> Option<usize> {
        match self {
            Axis::Choice(v) => Some(v.len()),
            Axis::Int { low, high } | Axis::LogInt { low, high } => {
                let span = (*high as i128 - *low as i128 + 1).max(0);
                Some(usize::try_from(span).unwrap_or(usize::MAX))
            }
            _ => None,
        }
    }

    /// The `i`-th value of a finite axis in enumeration order (`Choice`
    /// list order; ascending integers for `Int` / `LogInt`). Used by the
    /// lazy grid enumerator so a huge integer range is never materialized.
    ///
    /// # Panics
    ///
    /// On a continuous axis or `i >= cardinality()`.
    pub fn finite_value_at(&self, i: usize) -> Value {
        match self {
            Axis::Choice(v) => v[i],
            Axis::Int { low, .. } | Axis::LogInt { low, .. } => {
                Value::Int((*low as i128 + i as i128) as i64)
            }
            _ => panic!("finite_value_at on a continuous axis"),
        }
    }

    /// `true` if `v` is a value this axis can produce: a listed `Choice`
    /// candidate (numeric equality, so `Int(2)` matches `Float(2.0)`), a
    /// number inside a continuous range, or an integer inside an integer
    /// range.
    pub fn contains(&self, v: Value) -> bool {
        let x = v.as_f64();
        match self {
            Axis::Choice(values) => values.iter().any(|c| c.as_f64() == x),
            Axis::Uniform { low, high } | Axis::LogUniform { low, high } => {
                x.is_finite() && *low <= x && x <= *high
            }
            Axis::Int { low, high } | Axis::LogInt { low, high } => match v.as_i64() {
                Ok(i) => *low <= i && i <= *high,
                Err(_) => false,
            },
        }
    }

    /// Enumerate the values of a finite axis (grid search). Errors for
    /// continuous axes — a grid over a continuum is a caller error, not
    /// something to silently discretize.
    pub fn enumerate(&self, name: &str) -> Result<Vec<Value>> {
        match self {
            Axis::Choice(v) => Ok(v.clone()),
            Axis::Int { low, high } | Axis::LogInt { low, high } => {
                Ok((*low..=*high).map(Value::Int).collect())
            }
            Axis::Uniform { .. } | Axis::LogUniform { .. } => bail!(
                "axis `{name}` is continuous; grid search needs a finite Choice / Int axis \
                 (use random or tpe, or list explicit candidate values)"
            ),
        }
    }

    /// Draw one value. Consumes `rng` exactly once per call for every
    /// axis kind (a single-element `Choice` included).
    pub fn sample(&self, rng: &mut StdRng) -> Value {
        match self {
            Axis::Choice(v) => *v.choose(rng).expect("validated non-empty Choice axis"),
            Axis::Uniform { low, high } => {
                if low == high {
                    let _ = rng.r#gen::<f64>();
                    Value::Float(*low)
                } else {
                    Value::Float(rng.gen_range(*low..=*high).clamp(*low, *high))
                }
            }
            Axis::LogUniform { low, high } => {
                if low == high {
                    let _ = rng.r#gen::<f64>();
                    Value::Float(*low)
                } else {
                    // `exp(ln(high))` can land a rounding error above `high`;
                    // clamp so every sample satisfies `contains`.
                    Value::Float(rng.gen_range(low.ln()..=high.ln()).exp().clamp(*low, *high))
                }
            }
            Axis::Int { low, high } => Value::Int(rng.gen_range(*low..=*high)),
            Axis::LogInt { .. } => {
                let u: f64 = rng.r#gen();
                Value::Int(self.from_unit(u).as_i64().expect("LogInt yields Int"))
            }
        }
    }

    /// Map a value onto the unit interval for density modelling. `None`
    /// for `Choice` axes (categorical) and degenerate ranges.
    pub fn to_unit(&self, v: Value) -> Option<f64> {
        match self {
            Axis::Choice(_) => None,
            Axis::Uniform { low, high } => {
                (high > low).then(|| ((v.as_f64() - low) / (high - low)).clamp(0.0, 1.0))
            }
            Axis::LogUniform { low, high } => (high > low).then(|| {
                ((v.as_f64().max(*low).ln() - low.ln()) / (high.ln() - low.ln())).clamp(0.0, 1.0)
            }),
            Axis::Int { low, high } => (high > low).then(|| {
                ((v.as_f64() - *low as f64) / (*high as f64 - *low as f64)).clamp(0.0, 1.0)
            }),
            Axis::LogInt { low, high } => (high > low).then(|| {
                let (l, h) = (*low as f64, *high as f64);
                ((v.as_f64().max(l).ln() - l.ln()) / (h.ln() - l.ln())).clamp(0.0, 1.0)
            }),
        }
    }

    /// Inverse of [`to_unit`](Self::to_unit) for numeric axes. Integer
    /// axes round to the nearest in-range integer.
    pub fn from_unit(&self, u: f64) -> Value {
        let u = u.clamp(0.0, 1.0);
        match self {
            Axis::Choice(v) => v[((u * v.len() as f64) as usize).min(v.len() - 1)],
            Axis::Uniform { low, high } => {
                Value::Float((low + u * (high - low)).clamp(*low, *high))
            }
            Axis::LogUniform { low, high } => Value::Float(
                (low.ln() + u * (high.ln() - low.ln()))
                    .exp()
                    .clamp(*low, *high),
            ),
            Axis::Int { low, high } => {
                let x = *low as f64 + u * (*high as f64 - *low as f64);
                Value::Int((x.round() as i64).clamp(*low, *high))
            }
            Axis::LogInt { low, high } => {
                let (l, h) = (*low as f64, *high as f64);
                let x = (l.ln() + u * (h.ln() - l.ln())).exp();
                Value::Int((x.round() as i64).clamp(*low, *high))
            }
        }
    }
}

/// A named axis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AxisSpec {
    pub name: String,
    pub axis: Axis,
}

impl AxisSpec {
    pub fn new(name: impl Into<String>, axis: Axis) -> Self {
        Self {
            name: name.into(),
            axis,
        }
    }
}

/// One concrete configuration: a value per axis, in [`ParamSpace`] order.
pub type Assignment = Vec<Value>;

/// An ordered, validated list of named axes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParamSpace {
    axes: Vec<AxisSpec>,
}

impl ParamSpace {
    /// Validate every axis (non-empty choices, ordered finite bounds,
    /// positive log ranges, unique names) and build the space.
    pub fn new(axes: Vec<AxisSpec>) -> Result<Self> {
        if axes.is_empty() {
            bail!("ParamSpace needs at least one axis");
        }
        let mut seen = ahash::AHashSet::new();
        for a in &axes {
            if !seen.insert(a.name.as_str()) {
                bail!("ParamSpace: duplicate axis name `{}`", a.name);
            }
            a.axis.validate(&a.name)?;
        }
        Ok(Self { axes })
    }

    pub fn axes(&self) -> &[AxisSpec] {
        &self.axes
    }

    pub fn len(&self) -> usize {
        self.axes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.axes.is_empty()
    }

    pub fn names(&self) -> Vec<&str> {
        self.axes.iter().map(|a| a.name.as_str()).collect()
    }

    /// Index of the axis called `name`.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.axes.iter().position(|a| a.name == name)
    }

    /// `true` when every axis is finite (the whole space can be
    /// enumerated and exhausted).
    pub fn is_finite(&self) -> bool {
        self.axes.iter().all(|a| a.axis.is_finite())
    }

    /// Total number of distinct configurations for a finite space
    /// (saturating), `None` if any axis is continuous.
    pub fn cardinality(&self) -> Option<usize> {
        self.axes.iter().try_fold(1usize, |acc, a| {
            a.axis.cardinality().map(|c| acc.saturating_mul(c))
        })
    }

    /// Every configuration in nested-loop order: the **first axis is the
    /// outermost loop**, matching the hand-written cartesian products the
    /// typed grids used before #97. Errors on continuous axes.
    pub fn combinations(&self) -> Result<Vec<Assignment>> {
        self.combinations_take(usize::MAX, |_| false)
    }

    /// Lazily enumerate the grid in the same first-axis-outermost order,
    /// skipping assignments for which `skip` is `true`, and stop after
    /// `max_n` have been collected. Only the collected assignments are
    /// ever materialized, so a trial budget bounds memory and work even
    /// on an astronomically large finite space. Errors on continuous axes.
    pub fn combinations_take(
        &self,
        max_n: usize,
        mut skip: impl FnMut(&Assignment) -> bool,
    ) -> Result<Vec<Assignment>> {
        // Validate every axis up front so a continuous axis errors even
        // when `max_n == 0`.
        let radix: Vec<usize> = self
            .axes
            .iter()
            .map(|a| {
                a.axis.cardinality().ok_or_else(|| {
                    anyhow!(
                        "axis `{}` is continuous; grid search needs a finite Choice / Int axis \
                         (use random or tpe, or list explicit candidate values)",
                        a.name
                    )
                })
            })
            .collect::<Result<_>>()?;
        let mut out: Vec<Assignment> = Vec::new();
        if max_n == 0 || radix.contains(&0) {
            return Ok(out);
        }
        // Mixed-radix counter: the last axis is the innermost loop.
        let mut idx = vec![0usize; radix.len()];
        loop {
            let a: Assignment = self
                .axes
                .iter()
                .zip(&idx)
                .map(|(spec, &i)| spec.axis.finite_value_at(i))
                .collect();
            if !skip(&a) {
                out.push(a);
                if out.len() >= max_n {
                    break;
                }
            }
            // Increment with carry; exhausted when the first axis wraps.
            let mut k = idx.len();
            loop {
                if k == 0 {
                    return Ok(out);
                }
                k -= 1;
                idx[k] += 1;
                if idx[k] < radix[k] {
                    break;
                }
                idx[k] = 0;
            }
        }
        Ok(out)
    }

    /// `true` if `a` has one value per axis and every value lies in its
    /// axis (see [`Axis::contains`]). Warm-start trials from a different
    /// space fail this and are ignored by the strategies.
    pub fn contains(&self, a: &Assignment) -> bool {
        a.len() == self.axes.len()
            && self
                .axes
                .iter()
                .zip(a)
                .all(|(spec, v)| spec.axis.contains(*v))
    }

    /// Draw one configuration, sampling each axis independently in axis
    /// order from `rng`.
    pub fn sample_one(&self, rng: &mut StdRng) -> Assignment {
        self.axes.iter().map(|a| a.axis.sample(rng)).collect()
    }

    /// Check an assignment has one value per axis (and integral values
    /// on integer axes).
    pub fn check(&self, a: &Assignment) -> Result<()> {
        if a.len() != self.axes.len() {
            bail!(
                "assignment has {} values but the space has {} axes",
                a.len(),
                self.axes.len()
            );
        }
        for (spec, v) in self.axes.iter().zip(a) {
            if matches!(spec.axis, Axis::Int { .. } | Axis::LogInt { .. }) {
                v.as_i64()
                    .map_err(|e| anyhow!("axis `{}`: {e}", spec.name))?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn space() -> ParamSpace {
        ParamSpace::new(vec![
            AxisSpec::new("a", Axis::choice_f64(&[0.5, 1.0])),
            AxisSpec::new("b", Axis::choice_usize(&[8, 16, 32])),
            AxisSpec::new("c", Axis::Int { low: 1, high: 2 }),
        ])
        .unwrap()
    }

    #[test]
    fn combinations_are_nested_loop_order_first_axis_outermost() {
        let combos = space().combinations().unwrap();
        assert_eq!(combos.len(), 12);
        assert_eq!(
            combos[0],
            vec![Value::Float(0.5), Value::Int(8), Value::Int(1)]
        );
        assert_eq!(
            combos[1],
            vec![Value::Float(0.5), Value::Int(8), Value::Int(2)]
        );
        assert_eq!(
            combos[2],
            vec![Value::Float(0.5), Value::Int(16), Value::Int(1)]
        );
        assert_eq!(
            combos[11],
            vec![Value::Float(1.0), Value::Int(32), Value::Int(2)]
        );
        assert_eq!(space().cardinality(), Some(12));
        assert!(space().is_finite());
    }

    #[test]
    fn continuous_axes_refuse_grid_but_sample_in_range() {
        let s = ParamSpace::new(vec![
            AxisSpec::new(
                "lr",
                Axis::LogUniform {
                    low: 1e-4,
                    high: 1e-1,
                },
            ),
            AxisSpec::new(
                "x",
                Axis::Uniform {
                    low: -1.0,
                    high: 1.0,
                },
            ),
            AxisSpec::new("dim", Axis::LogInt { low: 8, high: 256 }),
        ])
        .unwrap();
        assert!(s.combinations().is_err());
        assert!(!s.is_finite());
        assert_eq!(s.cardinality(), None);
        let mut rng = StdRng::seed_from_u64(3);
        for _ in 0..200 {
            let a = s.sample_one(&mut rng);
            let lr = a[0].as_f64();
            assert!((1e-4..=1e-1).contains(&lr));
            assert!((-1.0..=1.0).contains(&a[1].as_f64()));
            let dim = a[2].as_i64().unwrap();
            assert!((8..=256).contains(&dim));
            s.check(&a).unwrap();
        }
    }

    #[test]
    fn choice_sampling_matches_slice_choose() {
        // The byte-identity guarantee for random search rests on this.
        let values = [0.5_f64, 1.0, 2.0];
        let axis = Axis::choice_f64(&values);
        let mut r1 = StdRng::seed_from_u64(42);
        let mut r2 = StdRng::seed_from_u64(42);
        for _ in 0..50 {
            let via_axis = axis.sample(&mut r1).as_f64();
            let via_choose = *values.choose(&mut r2).unwrap();
            assert_eq!(via_axis.to_bits(), via_choose.to_bits());
        }
    }

    #[test]
    fn unit_roundtrip_on_numeric_axes() {
        let cases = [
            Axis::Uniform {
                low: 2.0,
                high: 6.0,
            },
            Axis::LogUniform {
                low: 1e-3,
                high: 1.0,
            },
            Axis::Int { low: 0, high: 10 },
            Axis::LogInt { low: 4, high: 64 },
        ];
        for axis in cases {
            for u in [0.0, 0.25, 0.5, 1.0] {
                let v = axis.from_unit(u);
                let back = axis.to_unit(v).unwrap();
                let tol = if matches!(axis, Axis::Int { .. } | Axis::LogInt { .. }) {
                    0.2
                } else {
                    1e-9
                };
                assert!((back - u).abs() < tol, "{axis:?}: u={u} -> {v:?} -> {back}");
            }
        }
        assert_eq!(Axis::choice_f64(&[1.0]).to_unit(Value::Float(1.0)), None);
    }

    #[test]
    fn validation_rejects_bad_axes() {
        assert!(ParamSpace::new(vec![]).is_err());
        assert!(ParamSpace::new(vec![AxisSpec::new("a", Axis::Choice(vec![]))]).is_err());
        assert!(
            ParamSpace::new(vec![AxisSpec::new(
                "a",
                Axis::Uniform {
                    low: 2.0,
                    high: 1.0
                }
            )])
            .is_err()
        );
        assert!(
            ParamSpace::new(vec![AxisSpec::new(
                "a",
                Axis::LogUniform {
                    low: 0.0,
                    high: 1.0
                }
            )])
            .is_err()
        );
        assert!(
            ParamSpace::new(vec![AxisSpec::new("a", Axis::LogInt { low: 0, high: 4 })]).is_err()
        );
        assert!(
            ParamSpace::new(vec![
                AxisSpec::new("a", Axis::Int { low: 0, high: 4 }),
                AxisSpec::new("a", Axis::Int { low: 0, high: 4 }),
            ])
            .is_err()
        );
    }

    #[test]
    fn lazy_grid_enumeration_bounds_work_and_skips() {
        // 3 axes × 10^6 values each: 10^18 combinations, untouchable eagerly.
        let huge = ParamSpace::new(vec![
            AxisSpec::new(
                "a",
                Axis::Int {
                    low: 0,
                    high: 999_999,
                },
            ),
            AxisSpec::new(
                "b",
                Axis::Int {
                    low: 0,
                    high: 999_999,
                },
            ),
            AxisSpec::new("c", Axis::choice_f64(&[0.5, 1.0])),
        ])
        .unwrap();
        let first = huge.combinations_take(3, |_| false).unwrap();
        assert_eq!(
            first,
            vec![
                vec![Value::Int(0), Value::Int(0), Value::Float(0.5)],
                vec![Value::Int(0), Value::Int(0), Value::Float(1.0)],
                vec![Value::Int(0), Value::Int(1), Value::Float(0.5)],
            ]
        );
        // Skipping already-seen assignments continues in grid order.
        let s = space();
        let all = s.combinations().unwrap();
        let seen: Vec<Assignment> = all[..5].to_vec();
        let next = s.combinations_take(2, |a| seen.contains(a)).unwrap();
        assert_eq!(next, all[5..7].to_vec());
        // Exhausting the grid while skipping returns what is left.
        let rest = s.combinations_take(100, |a| seen.contains(a)).unwrap();
        assert_eq!(rest, all[5..].to_vec());
        assert!(s.combinations_take(0, |_| false).unwrap().is_empty());
        // Continuous axes still error, even for a zero budget.
        let cont = ParamSpace::new(vec![AxisSpec::new(
            "x",
            Axis::Uniform {
                low: 0.0,
                high: 1.0,
            },
        )])
        .unwrap();
        assert!(cont.combinations_take(0, |_| false).is_err());
    }

    #[test]
    fn cardinality_saturates_on_wide_int_ranges() {
        let wide = Axis::Int {
            low: -1,
            high: i64::MAX,
        };
        // 2^63 + 1 values: fits usize on 64-bit, would overflow i64 math.
        assert_eq!(wide.cardinality(), Some(9_223_372_036_854_775_809));
        let widest = Axis::Int {
            low: i64::MIN,
            high: i64::MAX,
        };
        assert_eq!(widest.cardinality(), Some(usize::MAX));
        assert_eq!(wide.finite_value_at(0), Value::Int(-1));
        assert_eq!(wide.finite_value_at(2), Value::Int(1));
        let s = ParamSpace::new(vec![
            AxisSpec::new("n", wide),
            AxisSpec::new("m", Axis::Int { low: 0, high: 1 }),
        ])
        .unwrap();
        assert_eq!(s.cardinality(), Some(usize::MAX), "product saturates");
        assert_eq!(
            s.combinations_take(3, |_| false).unwrap(),
            vec![
                vec![Value::Int(-1), Value::Int(0)],
                vec![Value::Int(-1), Value::Int(1)],
                vec![Value::Int(0), Value::Int(0)],
            ]
        );
    }

    #[test]
    fn contains_checks_membership_per_axis() {
        let s = space(); // a: {0.5, 1.0}, b: {8, 16, 32}, c: Int 1..=2
        assert!(s.contains(&vec![Value::Float(0.5), Value::Int(8), Value::Int(2)]));
        // Numeric equality across Int / Float.
        assert!(s.contains(&vec![Value::Float(1.0), Value::Float(16.0), Value::Int(1)]));
        assert!(!s.contains(&vec![Value::Float(2.0), Value::Int(8), Value::Int(2)]));
        assert!(!s.contains(&vec![Value::Float(0.5), Value::Int(8), Value::Int(3)]));
        assert!(!s.contains(&vec![Value::Float(0.5), Value::Int(8)]));
        let u = Axis::LogUniform {
            low: 1e-3,
            high: 1.0,
        };
        assert!(u.contains(Value::Float(0.01)));
        assert!(!u.contains(Value::Float(2.0)));
        assert!(!u.contains(Value::Float(f64::NAN)));
    }

    #[test]
    fn samples_and_unit_decodes_are_always_contained() {
        let axes = [
            Axis::Uniform {
                low: -1.0,
                high: 1.0,
            },
            Axis::LogUniform {
                low: 1e-4,
                high: 0.3,
            },
            Axis::LogUniform {
                low: 0.01,
                high: 1.0,
            },
            Axis::Int { low: 3, high: 9 },
            Axis::LogInt { low: 4, high: 64 },
        ];
        let mut rng = StdRng::seed_from_u64(11);
        for axis in &axes {
            for _ in 0..500 {
                let v = axis.sample(&mut rng);
                assert!(axis.contains(v), "{axis:?} sampled {v:?} outside itself");
            }
            for u in [0.0, 1e-9, 0.5, 1.0 - 1e-9, 1.0] {
                let v = axis.from_unit(u);
                assert!(
                    axis.contains(v),
                    "{axis:?} from_unit({u}) = {v:?} outside itself"
                );
            }
        }
    }

    #[test]
    fn value_integer_views() {
        assert_eq!(Value::Float(64.0).as_usize().unwrap(), 64);
        assert!(Value::Float(64.5).as_usize().is_err());
        assert!(Value::Int(-1).as_usize().is_err());
        assert_eq!(Value::Int(3).as_f64(), 3.0);
    }
}
