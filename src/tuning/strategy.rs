//! Search strategies: how the next batch of configurations is chosen
//! (issue #97, ADR-0005).
//!
//! A [`SearchStrategy`] is an *ask* interface over a [`ParamSpace`]: given
//! everything observed so far, propose up to `n` new [`Assignment`]s. The
//! runner (`tuning::tune_with`) evaluates each proposed batch in parallel
//! with rayon, appends the results to the history, and asks again until
//! the strategy returns nothing or the trial budget is spent.
//!
//! Three strategies ship:
//!
//! - [`GridStrategy`] — one batch holding every combination of a finite
//!   space, in nested-loop order. Non-adaptive.
//! - [`RandomStrategy`] — one batch of independent draws. Non-adaptive.
//! - [`TpeStrategy`] — Tree-structured Parzen Estimator (Bergstra et al.,
//!   NeurIPS 2011): after a few random start-up trials, split the history
//!   into the best `gamma` fraction (`l(x)`) and the rest (`g(x)`), fit a
//!   Parzen (kernel-density) estimator per axis to each group, and pick
//!   candidates that maximise `l(x) / g(x)`. Proposes `batch_size`
//!   configurations per round so the rayon pool stays busy; the batch
//!   trade-off is documented on the type.
//!
//! Every strategy draws all randomness from the runner's `StdRng`, so a
//! whole search is reproducible for a fixed seed and thread count.

use super::space::{Assignment, Axis, ParamSpace, Value};
use anyhow::Result;
use rand::Rng;
use rand::rngs::StdRng;

/// One evaluated configuration: the assignment and its mean CV score
/// (higher is better — NDCG@k in this crate).
#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    pub assignment: Assignment,
    pub score: f64,
}

/// Proposes configurations to evaluate next.
pub trait SearchStrategy: Send {
    /// Human-readable name reported in results / logs (`"grid"`,
    /// `"random"`, `"tpe"`).
    fn name(&self) -> &'static str;

    /// Propose up to `max_n` assignments given the `history` of every
    /// observation so far (warm-start entries included). Returning an
    /// empty vector ends the search.
    fn propose(
        &mut self,
        space: &ParamSpace,
        history: &[Observation],
        max_n: usize,
        rng: &mut StdRng,
    ) -> Result<Vec<Assignment>>;
}

// ---------------------------------------------------------------------------
// Grid
// ---------------------------------------------------------------------------

/// Exhaustive enumeration of a finite space in one batch (first axis
/// outermost). A second `propose` call returns nothing. `max_n` truncates
/// the enumeration, so `tune_*` with `strategy="grid"` and a small
/// `max_trials` evaluates the first `max_trials` combinations in grid
/// order.
#[derive(Debug, Default)]
pub struct GridStrategy {
    done: bool,
}

impl GridStrategy {
    pub fn new() -> Self {
        Self::default()
    }
}

impl SearchStrategy for GridStrategy {
    fn name(&self) -> &'static str {
        "grid"
    }

    fn propose(
        &mut self,
        space: &ParamSpace,
        _history: &[Observation],
        max_n: usize,
        _rng: &mut StdRng,
    ) -> Result<Vec<Assignment>> {
        if self.done {
            return Ok(Vec::new());
        }
        self.done = true;
        let mut combos = space.combinations()?;
        combos.truncate(max_n);
        Ok(combos)
    }
}

// ---------------------------------------------------------------------------
// Random
// ---------------------------------------------------------------------------

/// `n_trials` independent draws from the space in one batch. Sampling is
/// sequential on the shared RNG (axis order within each draw), so the
/// set is a deterministic function of the seed — the same contract the
/// original `random_search_with` had.
#[derive(Debug)]
pub struct RandomStrategy {
    n_trials: usize,
    done: bool,
}

impl RandomStrategy {
    pub fn new(n_trials: usize) -> Self {
        Self {
            n_trials,
            done: false,
        }
    }
}

impl SearchStrategy for RandomStrategy {
    fn name(&self) -> &'static str {
        "random"
    }

    fn propose(
        &mut self,
        space: &ParamSpace,
        _history: &[Observation],
        max_n: usize,
        rng: &mut StdRng,
    ) -> Result<Vec<Assignment>> {
        if self.done {
            return Ok(Vec::new());
        }
        self.done = true;
        let n = self.n_trials.min(max_n);
        Ok((0..n).map(|_| space.sample_one(rng)).collect())
    }
}

// ---------------------------------------------------------------------------
// TPE
// ---------------------------------------------------------------------------

/// Tree-structured Parzen Estimator.
///
/// Per round it proposes `batch_size` configurations. Each one is built
/// axis by axis: `n_candidates` draws from the "good" density `l` are
/// scored by `log l(x) − log g(x)` and the best is kept (the independent-
/// axis TPE that Optuna defaults to). Numeric axes are modelled on the
/// unit interval (log axes in log space) with a truncated-Gaussian Parzen
/// estimator plus a broad prior component; `Choice` axes use Laplace-
/// smoothed category frequencies.
///
/// **Batch trade-off.** The `batch_size` members of a round are drawn
/// from the *same* `l` / `g` (no fake observations), so they are
/// independent samples around the current optimum rather than a
/// sequential refinement. Smaller batches are more adaptive; larger ones
/// keep more rayon workers busy per round. `batch_size = 1` is classic
/// sequential TPE. Duplicates of already-evaluated points are resampled
/// (and a fully-enumerated finite space ends the search).
#[derive(Debug, Clone)]
pub struct TpeStrategy {
    /// Random draws before the density model is trusted.
    pub n_startup: usize,
    /// Proposals per round.
    pub batch_size: usize,
    /// Fraction of the history treated as "good" (`ceil(gamma * n)`,
    /// capped at 25, at least one point).
    pub gamma: f64,
    /// Candidates drawn from `l` per axis per proposal.
    pub n_candidates: usize,
}

impl Default for TpeStrategy {
    fn default() -> Self {
        Self {
            n_startup: 10,
            batch_size: 4,
            gamma: 0.1,
            n_candidates: 24,
        }
    }
}

impl TpeStrategy {
    pub fn new() -> Self {
        Self::default()
    }

    /// Defaults scaled to a trial budget: a quarter of the budget as
    /// random start-up (clamped to `3..=10`), batches of four.
    pub fn for_budget(max_trials: usize) -> Self {
        Self {
            n_startup: (max_trials / 4).clamp(3, 10),
            ..Self::default()
        }
    }

    pub fn with_n_startup(mut self, n: usize) -> Self {
        self.n_startup = n;
        self
    }

    pub fn with_batch_size(mut self, b: usize) -> Self {
        self.batch_size = b.max(1);
        self
    }

    pub fn with_gamma(mut self, g: f64) -> Self {
        self.gamma = g.clamp(0.01, 0.99);
        self
    }

    pub fn with_n_candidates(mut self, n: usize) -> Self {
        self.n_candidates = n.max(1);
        self
    }

    /// One TPE proposal: per axis, the best of `n_candidates` draws from
    /// `l` under `log l − log g`.
    fn propose_one(
        &self,
        space: &ParamSpace,
        good: &[&Assignment],
        bad: &[&Assignment],
        rng: &mut StdRng,
    ) -> Assignment {
        space
            .axes()
            .iter()
            .enumerate()
            .map(|(k, spec)| match &spec.axis {
                Axis::Choice(values) => {
                    let l = Categorical::fit(values.len(), good.iter().map(|a| a[k]), values);
                    let g = Categorical::fit(values.len(), bad.iter().map(|a| a[k]), values);
                    let mut best: Option<(usize, f64)> = None;
                    for _ in 0..self.n_candidates {
                        let c = l.sample(rng);
                        let s = l.log_p(c) - g.log_p(c);
                        if best.is_none_or(|(_, bs)| s > bs) {
                            best = Some((c, s));
                        }
                    }
                    values[best.map(|(c, _)| c).unwrap_or(0)]
                }
                axis => {
                    let n_total = good.len() + bad.len();
                    let to_u = |a: &&Assignment| axis.to_unit(a[k]);
                    let l = Parzen::fit(good.iter().filter_map(to_u).collect(), n_total);
                    let g = Parzen::fit(bad.iter().filter_map(to_u).collect(), n_total);
                    let mut best: Option<(f64, f64)> = None;
                    for _ in 0..self.n_candidates {
                        let u = l.sample(rng);
                        let s = l.log_pdf(u) - g.log_pdf(u);
                        if best.is_none_or(|(_, bs)| s > bs) {
                            best = Some((u, s));
                        }
                    }
                    axis.from_unit(best.map(|(u, _)| u).unwrap_or(0.5))
                }
            })
            .collect()
    }
}

impl SearchStrategy for TpeStrategy {
    fn name(&self) -> &'static str {
        "tpe"
    }

    fn propose(
        &mut self,
        space: &ParamSpace,
        history: &[Observation],
        max_n: usize,
        rng: &mut StdRng,
    ) -> Result<Vec<Assignment>> {
        let want = self.batch_size.min(max_n);
        if want == 0 {
            return Ok(Vec::new());
        }

        // A finite space can be exhausted: stop once every configuration
        // has been observed, and never propose a duplicate.
        let finite = space.is_finite();
        let cardinality = space.cardinality().unwrap_or(usize::MAX);
        let mut seen: ahash::AHashSet<Vec<u64>> = if finite {
            history.iter().map(|o| key_of(&o.assignment)).collect()
        } else {
            ahash::AHashSet::new()
        };
        if finite && seen.len() >= cardinality {
            return Ok(Vec::new());
        }

        // Rank history once per round: best first.
        let mut ranked: Vec<&Observation> =
            history.iter().filter(|o| o.score.is_finite()).collect();
        ranked.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let model_ready = ranked.len() >= self.n_startup.max(2);
        // Good set: ceil(gamma * n), capped at 25 (Optuna's rule), at least 1.
        let n_good = ((ranked.len() as f64 * self.gamma).ceil() as usize)
            .min(25)
            .clamp(1, ranked.len().max(1));
        let good: Vec<&Assignment> = ranked.iter().take(n_good).map(|o| &o.assignment).collect();
        let bad: Vec<&Assignment> = ranked.iter().skip(n_good).map(|o| &o.assignment).collect();

        let mut out: Vec<Assignment> = Vec::with_capacity(want);
        let mut attempts = 0usize;
        while out.len() < want {
            if finite && seen.len() >= cardinality {
                break;
            }
            // After the model has had a fair go at an almost-exhausted
            // finite space, fall back to plain sampling so the search
            // always terminates.
            let candidate = if model_ready && attempts < want * 20 {
                self.propose_one(space, &good, &bad, rng)
            } else {
                space.sample_one(rng)
            };
            attempts += 1;
            if finite && !seen.insert(key_of(&candidate)) {
                if attempts > want * 200 {
                    break;
                }
                continue;
            }
            out.push(candidate);
        }
        Ok(out)
    }
}

/// Bit-exact key for de-duplicating assignments on finite spaces.
fn key_of(a: &Assignment) -> Vec<u64> {
    a.iter()
        .map(|v| match v {
            Value::Float(f) => f.to_bits(),
            Value::Int(i) => *i as u64,
        })
        .collect()
}

// --- Parzen estimator on [0, 1] -----------------------------------------

/// Mixture of truncated Gaussians on the unit interval — one component per
/// observation plus a broad prior (`mu = 0.5`, `sigma = 1`) so an empty or
/// one-point group still defines a proper density.
#[derive(Debug)]
struct Parzen {
    mus: Vec<f64>,
    sigmas: Vec<f64>,
    /// Per-component log normaliser of the truncation to [0, 1].
    log_z: Vec<f64>,
}

impl Parzen {
    fn fit(points: Vec<f64>, n_total: usize) -> Self {
        // Optuna-style bandwidths: insert the prior mean among the sorted
        // observations, give each component the *larger* gap to its
        // neighbouring component (endpoints use their single neighbour),
        // then clip to [sigma_min, 1]. Boundaries are deliberately not
        // neighbours — treating them as such inflates the edge kernels
        // until the "good" density is nearly flat and TPE stops
        // exploiting. sigma_min shrinks as the search accumulates points.
        let mut mus: Vec<f64> = points;
        mus.push(0.5);
        mus.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let prior_idx = mus
            .iter()
            .position(|&m| m == 0.5)
            .expect("prior component present");
        let n = mus.len();
        let sigma_max = 1.0;
        let sigma_min = sigma_max / (1.0 + n_total.max(n) as f64).min(100.0);
        let mut sigmas = Vec::with_capacity(n);
        for i in 0..n {
            let left = if i == 0 {
                None
            } else {
                Some(mus[i] - mus[i - 1])
            };
            let right = if i + 1 == n {
                None
            } else {
                Some(mus[i + 1] - mus[i])
            };
            let s = match (left, right) {
                (Some(l), Some(r)) => l.max(r),
                (Some(l), None) => l,
                (None, Some(r)) => r,
                (None, None) => sigma_max,
            };
            sigmas.push(s.clamp(sigma_min, sigma_max));
        }
        // The prior stays broad regardless of its neighbours.
        sigmas[prior_idx] = sigma_max;
        let log_z = mus
            .iter()
            .zip(&sigmas)
            .map(|(&m, &s)| {
                (norm_cdf((1.0 - m) / s) - norm_cdf((0.0 - m) / s))
                    .max(1e-12)
                    .ln()
            })
            .collect();
        Self { mus, sigmas, log_z }
    }

    fn log_pdf(&self, x: f64) -> f64 {
        let k = self.mus.len() as f64;
        let mut acc = f64::NEG_INFINITY;
        for ((&m, &s), &lz) in self.mus.iter().zip(&self.sigmas).zip(&self.log_z) {
            let z = (x - m) / s;
            let lp = -0.5 * z * z - s.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln() - lz;
            acc = log_add(acc, lp);
        }
        acc - k.ln()
    }

    fn sample(&self, rng: &mut StdRng) -> f64 {
        let i = rng.gen_range(0..self.mus.len());
        let (m, s) = (self.mus[i], self.sigmas[i]);
        for _ in 0..16 {
            let x = m + s * standard_normal(rng);
            if (0.0..=1.0).contains(&x) {
                return x;
            }
        }
        (m + s * standard_normal(rng)).clamp(0.0, 1.0)
    }
}

/// Laplace-smoothed categorical over `k` choices.
#[derive(Debug)]
struct Categorical {
    log_p: Vec<f64>,
    cdf: Vec<f64>,
}

impl Categorical {
    fn fit(k: usize, observed: impl Iterator<Item = Value>, values: &[Value]) -> Self {
        let mut counts = vec![1.0_f64; k];
        let mut total = k as f64;
        for v in observed {
            if let Some(i) = values.iter().position(|c| c == &v) {
                counts[i] += 1.0;
                total += 1.0;
            }
        }
        let p: Vec<f64> = counts.iter().map(|c| c / total).collect();
        let mut cdf = Vec::with_capacity(k);
        let mut run = 0.0;
        for &pi in &p {
            run += pi;
            cdf.push(run);
        }
        Self {
            log_p: p.iter().map(|x| x.ln()).collect(),
            cdf,
        }
    }

    fn log_p(&self, c: usize) -> f64 {
        self.log_p[c]
    }

    fn sample(&self, rng: &mut StdRng) -> usize {
        let u: f64 = rng.r#gen();
        self.cdf
            .iter()
            .position(|&c| u < c)
            .unwrap_or(self.cdf.len() - 1)
    }
}

fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// Box–Muller standard normal draw.
fn standard_normal(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.r#gen::<f64>().max(1e-300);
    let u2: f64 = rng.r#gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Standard normal CDF via Abramowitz & Stegun 7.1.26 (|err| < 1.5e-7).
fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tuning::space::AxisSpec;
    use rand::SeedableRng;

    fn finite_space() -> ParamSpace {
        ParamSpace::new(vec![
            AxisSpec::new("a", Axis::choice_f64(&[0.5, 1.0])),
            AxisSpec::new("b", Axis::choice_usize(&[8, 16, 32])),
        ])
        .unwrap()
    }

    #[test]
    fn grid_proposes_everything_once_then_nothing() {
        let s = finite_space();
        let mut g = GridStrategy::new();
        let mut rng = StdRng::seed_from_u64(0);
        let first = g.propose(&s, &[], usize::MAX, &mut rng).unwrap();
        assert_eq!(first, s.combinations().unwrap());
        assert!(g.propose(&s, &[], usize::MAX, &mut rng).unwrap().is_empty());

        let mut g2 = GridStrategy::new();
        assert_eq!(g2.propose(&s, &[], 2, &mut rng).unwrap().len(), 2);
    }

    #[test]
    fn random_matches_sequential_sampling_for_a_seed() {
        let s = finite_space();
        let mut r = RandomStrategy::new(5);
        let mut rng_a = StdRng::seed_from_u64(7);
        let batch = r.propose(&s, &[], usize::MAX, &mut rng_a).unwrap();
        let mut rng_b = StdRng::seed_from_u64(7);
        let expected: Vec<Assignment> = (0..5).map(|_| s.sample_one(&mut rng_b)).collect();
        assert_eq!(batch, expected);
        assert!(
            r.propose(&s, &[], usize::MAX, &mut rng_a)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn erf_and_cdf_are_sane() {
        assert!((erf(0.0)).abs() < 1e-7);
        assert!((erf(1.0) - 0.8427007929).abs() < 1e-6);
        assert!((erf(-1.0) + 0.8427007929).abs() < 1e-6);
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-7);
        assert!((norm_cdf(1.96) - 0.975).abs() < 1e-3);
    }

    #[test]
    fn parzen_density_integrates_to_about_one() {
        let p = Parzen::fit(vec![0.2, 0.25, 0.8], 10);
        let n = 2000;
        let mass: f64 = (0..n)
            .map(|i| p.log_pdf((i as f64 + 0.5) / n as f64).exp() / n as f64)
            .sum();
        assert!((mass - 1.0).abs() < 0.02, "mass = {mass}");
        // Density is higher near the cluster than far from it.
        assert!(p.log_pdf(0.22) > p.log_pdf(0.55));
    }

    #[test]
    fn tpe_random_startup_then_model_and_never_duplicates_on_finite_space() {
        let s = finite_space(); // 6 configs
        let mut t = TpeStrategy::new().with_n_startup(2).with_batch_size(2);
        let mut rng = StdRng::seed_from_u64(1);
        let mut history: Vec<Observation> = Vec::new();
        let mut rounds = 0;
        loop {
            let batch = t.propose(&s, &history, 100, &mut rng).unwrap();
            if batch.is_empty() {
                break;
            }
            rounds += 1;
            assert!(batch.len() <= 2);
            for a in batch {
                assert!(
                    !history.iter().any(|o| o.assignment == a),
                    "duplicate proposal {a:?}"
                );
                let score = a[0].as_f64() + a[1].as_f64();
                history.push(Observation {
                    assignment: a,
                    score,
                });
            }
            assert!(rounds < 20, "did not terminate");
        }
        assert_eq!(history.len(), 6, "a finite space is exhausted exactly");
    }

    /// TPE should beat random on a smooth 2-D objective within the same
    /// budget, robustly across seeds.
    #[test]
    fn tpe_beats_random_on_quadratic_bowl() {
        let s = ParamSpace::new(vec![
            AxisSpec::new(
                "x",
                Axis::Uniform {
                    low: 0.0,
                    high: 1.0,
                },
            ),
            AxisSpec::new(
                "lr",
                Axis::LogUniform {
                    low: 1e-4,
                    high: 1.0,
                },
            ),
            AxisSpec::new("k", Axis::choice_usize(&[1, 2, 3, 4])),
        ])
        .unwrap();
        let objective = |a: &Assignment| -> f64 {
            let x = a[0].as_f64();
            let l = a[1].as_f64().log10(); // optimum at 1e-2
            let k = a[2].as_i64().unwrap() as f64;
            -((x - 0.3).powi(2) + 0.1 * (l + 2.0).powi(2) + 0.05 * (k - 3.0).powi(2))
        };
        let budget = 60;
        let mut tpe_wins = 0;
        let seeds = [1_u64, 2, 3, 4, 5, 6, 7];
        for &seed in &seeds {
            // Random.
            let mut rng = StdRng::seed_from_u64(seed);
            let mut rs = RandomStrategy::new(budget);
            let best_random = rs
                .propose(&s, &[], budget, &mut rng)
                .unwrap()
                .iter()
                .map(objective)
                .fold(f64::NEG_INFINITY, f64::max);
            // TPE with half the budget.
            let mut rng = StdRng::seed_from_u64(seed);
            let mut tpe = TpeStrategy::for_budget(budget / 2).with_batch_size(2);
            let mut history = Vec::new();
            while history.len() < budget / 2 {
                let batch = tpe
                    .propose(&s, &history, budget / 2 - history.len(), &mut rng)
                    .unwrap();
                assert!(!batch.is_empty());
                for a in batch {
                    let score = objective(&a);
                    history.push(Observation {
                        assignment: a,
                        score,
                    });
                }
            }
            let best_tpe = history
                .iter()
                .map(|o| o.score)
                .fold(f64::NEG_INFINITY, f64::max);
            if best_tpe >= best_random {
                tpe_wins += 1;
            }
        }
        assert!(
            tpe_wins * 2 > seeds.len(),
            "TPE (half budget) should match or beat random in a majority of seeds; won {tpe_wins}/{}",
            seeds.len()
        );
    }

    #[test]
    fn tpe_is_deterministic_for_a_seed() {
        let s = finite_space();
        let run = || {
            let mut t = TpeStrategy::new().with_n_startup(2).with_batch_size(2);
            let mut rng = StdRng::seed_from_u64(9);
            let mut history: Vec<Observation> = Vec::new();
            let mut order = Vec::new();
            loop {
                let batch = t.propose(&s, &history, 100, &mut rng).unwrap();
                if batch.is_empty() {
                    break;
                }
                for a in batch {
                    let score = a[1].as_f64();
                    order.push(a.clone());
                    history.push(Observation {
                        assignment: a,
                        score,
                    });
                }
            }
            order
        };
        assert_eq!(run(), run());
    }
}
