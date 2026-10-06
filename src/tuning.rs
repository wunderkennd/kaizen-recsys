//! Hyperparameter tuning module: grid search and random search with
//! k-fold cross-validation.
//!
//! The search machinery (cartesian product, k-fold split generation,
//! parallel trial runner) is model-agnostic: it is generic over the
//! [`FoldEvaluator`] trait, which trains a model on one fold's training
//! split and scores it against that fold's held-out users. The
//! optimization target is NDCG@k.
//!
//! [`EaseFoldEvaluator`] is the EASE implementation. It builds matrices
//! via the data pipeline, trains a [`RustFeaseModel`], and scores users
//! through the [`RecModel`] trait. SASRec and Two-Tower plug in their
//! own [`FoldEvaluator`] implementations behind the same search
//! machinery.
//!
//! The runner is generic over the parameter type `P` (the [`SearchSpace`]
//! trait yields the cartesian product and a seeded random sample), so each
//! model family carries its own architecture-specific schema —
//! [`HyperParams`]/[`ParamGrid`] for EASE,
//! [`SasRecParams`]/[`SasRecParamGrid`] for SASRec, and
//! [`TwoTowerParams`]/[`TwoTowerParamGrid`] for Two-Tower — while sharing
//! the k-fold split generation, the rayon trial runner, and the
//! deterministic result assembly.
//!
//! Issue #97 / ADR-0005 add a strategy seam on top: every search is
//! `tune_with(evaluator, space, strategy, cfg)`, where a [`SearchStrategy`]
//! (grid, random, TPE) proposes batches of model-agnostic [`Assignment`]s
//! over a named [`ParamSpace`], the runner evaluates each batch in
//! parallel, and the model's [`SearchSpace`] impl decodes assignments into
//! its typed params. `grid_search_with` / `random_search_with` are thin
//! wrappers over that loop and produce byte-identical results to their
//! pre-#97 implementations for a fixed seed.

pub mod space;
pub mod strategy;

pub use space::{Assignment, Axis, AxisSpec, ParamSpace, Value};
pub use strategy::{GridStrategy, Observation, RandomStrategy, SearchStrategy, TpeStrategy};

use crate::data_pipeline::{self, Mappings};
use crate::evaluation::{build_user_features_map, read_interactions_df, write_parquet};
use crate::model::RustFeaseModel;
use crate::models::{EaseAdapter, ModelInput, RecModel};
use crate::weighting::WeightingConfig;
use ahash::AHashMap;
use anyhow::{Result, anyhow};
use polars::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Parameter types
// ---------------------------------------------------------------------------

/// A single hyperparameter configuration to evaluate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperParams {
    pub alpha: f64,
    pub beta: f64,
    pub lambda_: f64,
    pub meta_weight: f64,
    pub decay_rate: f64,
    pub ips_alpha: f64,
    pub sparsity_threshold: f64,
}

/// Search space for grid search -- each field is a vec of values to try.
#[derive(Debug, Clone)]
pub struct ParamGrid {
    pub alpha: Vec<f64>,
    pub beta: Vec<f64>,
    pub lambda_: Vec<f64>,
    pub meta_weight: Vec<f64>,
    pub decay_rate: Vec<f64>,
    pub ips_alpha: Vec<f64>,
    pub sparsity_threshold: Vec<f64>,
}

/// Result of a single trial in the search, generic over the model's
/// parameter schema `P` (EASE: [`HyperParams`]; SASRec: [`SasRecParams`];
/// Two-Tower: [`TwoTowerParams`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialResult<P = HyperParams> {
    pub params: P,
    pub mean_score: f64,
    pub fold_scores: Vec<f64>,
}

/// Result of a complete search, generic over the parameter schema `P`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult<P = HyperParams> {
    pub best_params: P,
    pub best_score: f64,
    pub all_trials: Vec<TrialResult<P>>,
    pub metric_name: String,
    /// Which [`SearchStrategy`] produced the trials (`"grid"`, `"random"`,
    /// `"tpe"`).
    #[serde(default)]
    pub strategy: String,
}

// ---------------------------------------------------------------------------
// SASRec parameter schema
// ---------------------------------------------------------------------------

/// A single SASRec hyperparameter configuration. These are the real
/// architecture / optimizer knobs the burn model takes (see
/// [`crate::models::sasrec::SasRecConfig`] /
/// [`crate::models::sasrec::SasRecTrainingConfig`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SasRecParams {
    pub embedding_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub dropout: f64,
    pub learning_rate: f64,
    pub num_epochs: usize,
}

/// Search space for SASRec grid / random search. Each field is a vec of
/// values to try over that architecture knob.
#[derive(Debug, Clone)]
pub struct SasRecParamGrid {
    pub embedding_dim: Vec<usize>,
    pub num_heads: Vec<usize>,
    pub num_layers: Vec<usize>,
    pub dropout: Vec<f64>,
    pub learning_rate: Vec<f64>,
    pub num_epochs: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Two-Tower parameter schema
// ---------------------------------------------------------------------------

/// A single Two-Tower hyperparameter configuration — the real knobs the
/// in-batch sampled-softmax model takes (see
/// [`crate::models::two_tower::TrainParams`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwoTowerParams {
    pub embedding_dim: usize,
    pub temperature: f64,
    pub learning_rate: f64,
    pub id_dropout: f64,
}

/// Search space for Two-Tower grid / random search.
#[derive(Debug, Clone)]
pub struct TwoTowerParamGrid {
    pub embedding_dim: Vec<usize>,
    pub temperature: Vec<f64>,
    pub learning_rate: Vec<f64>,
    pub id_dropout: Vec<f64>,
}

// ---------------------------------------------------------------------------
// BERT4Rec parameter schema (#96 / #97)
// ---------------------------------------------------------------------------

/// A single BERT4Rec hyperparameter configuration (see
/// [`crate::models::bert4rec::Bert4RecConfig`] /
/// [`crate::models::bert4rec::Bert4RecTrainingConfig`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bert4RecParams {
    pub embedding_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub dropout: f64,
    pub mask_ratio: f64,
    pub learning_rate: f64,
    pub num_epochs: usize,
}

/// Search space for BERT4Rec grid / random search.
#[derive(Debug, Clone)]
pub struct Bert4RecParamGrid {
    pub embedding_dim: Vec<usize>,
    pub num_heads: Vec<usize>,
    pub num_layers: Vec<usize>,
    pub dropout: Vec<f64>,
    pub mask_ratio: Vec<f64>,
    pub learning_rate: Vec<f64>,
    pub num_epochs: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Search space abstraction
// ---------------------------------------------------------------------------

/// A model's parameter search space: the only model-specific knowledge the
/// generic runner needs beyond the [`FoldEvaluator`].
///
/// Since #97 the space is described by named [`Axis`]es ([`axes`](Self::axes))
/// and the model supplies the two codecs between a model-agnostic
/// [`Assignment`] and its typed `Params` ([`decode`](Self::decode) /
/// [`encode`](Self::encode)). [`combinations`](Self::combinations) and
/// [`sample_one`](Self::sample_one) are provided in terms of those, so a
/// typed grid and a dict-driven space share one implementation — and so
/// every [`SearchStrategy`] works on every model with no per-model code.
///
/// Ordering / RNG contract (what keeps `grid_search_with` and
/// `random_search_with` byte-identical to their pre-#97 output): the axes
/// are in the typed struct's field order; the grid is enumerated with the
/// first axis outermost; and random sampling draws each axis in order via
/// `SliceRandom::choose` on its candidate list. The one intentional
/// difference: an *empty* candidate list used to mean "default value, no
/// RNG draw" — it now becomes a one-element axis that does draw, so only
/// Rust callers passing empty lists (the Python layer never does) see a
/// different sampled sequence.
pub trait SearchSpace: Send + Sync {
    /// The concrete parameter type this space produces.
    type Params: Clone + Send + Sync;

    /// The named axes, in the model's canonical parameter order.
    fn axes(&self) -> ParamSpace;

    /// Typed params from one value per axis (in [`axes`](Self::axes) order).
    fn decode(&self, a: &Assignment) -> Result<Self::Params>;

    /// Inverse of [`decode`](Self::decode); used to warm-start a strategy
    /// from previous [`TrialResult`]s.
    fn encode(&self, p: &Self::Params) -> Assignment;

    /// Every configuration in the grid, in a deterministic (nested-loop)
    /// order. Errors if an axis is continuous.
    fn combinations(&self) -> Result<Vec<Self::Params>> {
        self.axes()
            .combinations()?
            .iter()
            .map(|a| self.decode(a))
            .collect()
    }

    /// Draw one configuration, sampling each axis independently from `rng`.
    fn sample_one(&self, rng: &mut StdRng) -> Self::Params {
        let a = self.axes().sample_one(rng);
        self.decode(&a)
            .expect("an assignment sampled from the space must decode")
    }
}

// ---------------------------------------------------------------------------
// Per-model parameter schemas: names, defaults, and the Assignment codecs
// ---------------------------------------------------------------------------

/// The per-model knowledge a dict-driven space needs: the canonical axis
/// names, a default axis for any the caller omits, and the codecs.
/// Implemented once per model and shared by its typed `*ParamGrid`
/// (`SearchSpace` impl) and its [`DynSpace`].
pub trait ParamSchema: Send + Sync + 'static {
    type Params: Clone + Send + Sync;
    /// Canonical axis names, in the typed struct's field order.
    const NAMES: &'static [&'static str];
    /// Single-value axis holding the default for `name`.
    fn default_axis(name: &str) -> Axis;
    fn decode(a: &Assignment) -> Result<Self::Params>;
    fn encode(p: &Self::Params) -> Assignment;
}

fn expect_len(a: &Assignment, n: usize, model: &str) -> Result<()> {
    if a.len() != n {
        return Err(anyhow!(
            "{model} assignment has {} values, expected {n}",
            a.len()
        ));
    }
    Ok(())
}

/// EASE schema ([`HyperParams`]).
pub struct EaseSchema;

impl ParamSchema for EaseSchema {
    type Params = HyperParams;
    const NAMES: &'static [&'static str] = &[
        "alpha",
        "beta",
        "lambda_",
        "meta_weight",
        "decay_rate",
        "ips_alpha",
        "sparsity_threshold",
    ];

    fn default_axis(name: &str) -> Axis {
        let v = match name {
            "alpha" | "beta" => 1.0,
            "lambda_" => 100.0,
            _ => 0.0,
        };
        Axis::choice_f64(&[v])
    }

    fn decode(a: &Assignment) -> Result<HyperParams> {
        expect_len(a, 7, "EASE")?;
        Ok(HyperParams {
            alpha: a[0].as_f64(),
            beta: a[1].as_f64(),
            lambda_: a[2].as_f64(),
            meta_weight: a[3].as_f64(),
            decay_rate: a[4].as_f64(),
            ips_alpha: a[5].as_f64(),
            sparsity_threshold: a[6].as_f64(),
        })
    }

    fn encode(p: &HyperParams) -> Assignment {
        vec![
            p.alpha.into(),
            p.beta.into(),
            p.lambda_.into(),
            p.meta_weight.into(),
            p.decay_rate.into(),
            p.ips_alpha.into(),
            p.sparsity_threshold.into(),
        ]
    }
}

/// SASRec schema ([`SasRecParams`]).
pub struct SasRecSchema;

impl ParamSchema for SasRecSchema {
    type Params = SasRecParams;
    const NAMES: &'static [&'static str] = &[
        "embedding_dim",
        "num_heads",
        "num_layers",
        "dropout",
        "learning_rate",
        "num_epochs",
    ];

    fn default_axis(name: &str) -> Axis {
        match name {
            "embedding_dim" => Axis::choice_usize(&[64]),
            "num_heads" => Axis::choice_usize(&[2]),
            "num_layers" => Axis::choice_usize(&[2]),
            "dropout" => Axis::choice_f64(&[0.2]),
            "learning_rate" => Axis::choice_f64(&[1e-3]),
            _ => Axis::choice_usize(&[50]),
        }
    }

    fn decode(a: &Assignment) -> Result<SasRecParams> {
        expect_len(a, 6, "SASRec")?;
        Ok(SasRecParams {
            embedding_dim: a[0].as_usize()?,
            num_heads: a[1].as_usize()?,
            num_layers: a[2].as_usize()?,
            dropout: a[3].as_f64(),
            learning_rate: a[4].as_f64(),
            num_epochs: a[5].as_usize()?,
        })
    }

    fn encode(p: &SasRecParams) -> Assignment {
        vec![
            p.embedding_dim.into(),
            p.num_heads.into(),
            p.num_layers.into(),
            p.dropout.into(),
            p.learning_rate.into(),
            p.num_epochs.into(),
        ]
    }
}

/// Two-Tower schema ([`TwoTowerParams`]).
pub struct TwoTowerSchema;

impl ParamSchema for TwoTowerSchema {
    type Params = TwoTowerParams;
    const NAMES: &'static [&'static str] = &[
        "embedding_dim",
        "temperature",
        "learning_rate",
        "id_dropout",
    ];

    fn default_axis(name: &str) -> Axis {
        match name {
            "embedding_dim" => Axis::choice_usize(&[32]),
            "temperature" => Axis::choice_f64(&[0.05]),
            "learning_rate" => Axis::choice_f64(&[0.01]),
            _ => Axis::choice_f64(&[0.1]),
        }
    }

    fn decode(a: &Assignment) -> Result<TwoTowerParams> {
        expect_len(a, 4, "Two-Tower")?;
        Ok(TwoTowerParams {
            embedding_dim: a[0].as_usize()?,
            temperature: a[1].as_f64(),
            learning_rate: a[2].as_f64(),
            id_dropout: a[3].as_f64(),
        })
    }

    fn encode(p: &TwoTowerParams) -> Assignment {
        vec![
            p.embedding_dim.into(),
            p.temperature.into(),
            p.learning_rate.into(),
            p.id_dropout.into(),
        ]
    }
}

/// BERT4Rec schema ([`Bert4RecParams`]).
pub struct Bert4RecSchema;

impl ParamSchema for Bert4RecSchema {
    type Params = Bert4RecParams;
    const NAMES: &'static [&'static str] = &[
        "embedding_dim",
        "num_heads",
        "num_layers",
        "dropout",
        "mask_ratio",
        "learning_rate",
        "num_epochs",
    ];

    fn default_axis(name: &str) -> Axis {
        match name {
            "embedding_dim" => Axis::choice_usize(&[64]),
            "num_heads" => Axis::choice_usize(&[4]),
            "num_layers" => Axis::choice_usize(&[2]),
            "dropout" => Axis::choice_f64(&[0.1]),
            "mask_ratio" => Axis::choice_f64(&[0.2]),
            "learning_rate" => Axis::choice_f64(&[1e-3]),
            _ => Axis::choice_usize(&[50]),
        }
    }

    fn decode(a: &Assignment) -> Result<Bert4RecParams> {
        expect_len(a, 7, "BERT4Rec")?;
        Ok(Bert4RecParams {
            embedding_dim: a[0].as_usize()?,
            num_heads: a[1].as_usize()?,
            num_layers: a[2].as_usize()?,
            dropout: a[3].as_f64(),
            mask_ratio: a[4].as_f64(),
            learning_rate: a[5].as_f64(),
            num_epochs: a[6].as_usize()?,
        })
    }

    fn encode(p: &Bert4RecParams) -> Assignment {
        vec![
            p.embedding_dim.into(),
            p.num_heads.into(),
            p.num_layers.into(),
            p.dropout.into(),
            p.mask_ratio.into(),
            p.learning_rate.into(),
            p.num_epochs.into(),
        ]
    }
}

/// A dict-driven space for schema `M`: any subset of `M::NAMES` with an
/// arbitrary [`Axis`] each (continuous ranges included); omitted names get
/// `M::default_axis`. This is what the Python `tune_*` entrypoints build.
pub struct DynSpace<M: ParamSchema> {
    space: ParamSpace,
    _schema: std::marker::PhantomData<M>,
}

impl<M: ParamSchema> DynSpace<M> {
    /// Build from caller-supplied axes. Unknown names are an error (a typo
    /// would otherwise silently tune nothing); missing names default.
    pub fn from_axes(axes: Vec<AxisSpec>) -> Result<Self> {
        for a in &axes {
            if !M::NAMES.contains(&a.name.as_str()) {
                return Err(anyhow!(
                    "unknown parameter `{}`; expected one of {:?}",
                    a.name,
                    M::NAMES
                ));
            }
        }
        let ordered: Vec<AxisSpec> = M::NAMES
            .iter()
            .map(|name| {
                axes.iter()
                    .find(|a| a.name == *name)
                    .cloned()
                    .unwrap_or_else(|| AxisSpec::new(*name, M::default_axis(name)))
            })
            .collect();
        Ok(Self {
            space: ParamSpace::new(ordered)?,
            _schema: std::marker::PhantomData,
        })
    }

    pub fn space(&self) -> &ParamSpace {
        &self.space
    }
}

impl<M: ParamSchema> SearchSpace for DynSpace<M> {
    type Params = M::Params;

    fn axes(&self) -> ParamSpace {
        self.space.clone()
    }

    fn decode(&self, a: &Assignment) -> Result<M::Params> {
        M::decode(a)
    }

    fn encode(&self, p: &M::Params) -> Assignment {
        M::encode(p)
    }
}

/// Dict-driven spaces per model.
pub type EaseSpace = DynSpace<EaseSchema>;
pub type SasRecSpace = DynSpace<SasRecSchema>;
pub type TwoTowerSpace = DynSpace<TwoTowerSchema>;
pub type Bert4RecSpace = DynSpace<Bert4RecSchema>;

/// `Choice` axis from a typed grid's candidate list, substituting the
/// schema default for an empty list (see the [`SearchSpace`] contract).
fn choice_or_default<M: ParamSchema>(name: &str, axis: Axis) -> AxisSpec {
    let axis = match axis {
        Axis::Choice(v) if v.is_empty() => M::default_axis(name),
        other => other,
    };
    AxisSpec::new(name, axis)
}

impl SearchSpace for ParamGrid {
    type Params = HyperParams;

    fn axes(&self) -> ParamSpace {
        let lists = [
            &self.alpha,
            &self.beta,
            &self.lambda_,
            &self.meta_weight,
            &self.decay_rate,
            &self.ips_alpha,
            &self.sparsity_threshold,
        ];
        ParamSpace::new(
            EaseSchema::NAMES
                .iter()
                .zip(lists)
                .map(|(n, l)| choice_or_default::<EaseSchema>(n, Axis::choice_f64(l)))
                .collect(),
        )
        .expect("typed EASE grid axes are valid by construction")
    }

    fn decode(&self, a: &Assignment) -> Result<HyperParams> {
        EaseSchema::decode(a)
    }

    fn encode(&self, p: &HyperParams) -> Assignment {
        EaseSchema::encode(p)
    }
}

impl SearchSpace for SasRecParamGrid {
    type Params = SasRecParams;

    fn axes(&self) -> ParamSpace {
        let axes = vec![
            Axis::choice_usize(&self.embedding_dim),
            Axis::choice_usize(&self.num_heads),
            Axis::choice_usize(&self.num_layers),
            Axis::choice_f64(&self.dropout),
            Axis::choice_f64(&self.learning_rate),
            Axis::choice_usize(&self.num_epochs),
        ];
        ParamSpace::new(
            SasRecSchema::NAMES
                .iter()
                .zip(axes)
                .map(|(n, a)| choice_or_default::<SasRecSchema>(n, a))
                .collect(),
        )
        .expect("typed SASRec grid axes are valid by construction")
    }

    fn decode(&self, a: &Assignment) -> Result<SasRecParams> {
        SasRecSchema::decode(a)
    }

    fn encode(&self, p: &SasRecParams) -> Assignment {
        SasRecSchema::encode(p)
    }
}

impl SearchSpace for TwoTowerParamGrid {
    type Params = TwoTowerParams;

    fn axes(&self) -> ParamSpace {
        let axes = vec![
            Axis::choice_usize(&self.embedding_dim),
            Axis::choice_f64(&self.temperature),
            Axis::choice_f64(&self.learning_rate),
            Axis::choice_f64(&self.id_dropout),
        ];
        ParamSpace::new(
            TwoTowerSchema::NAMES
                .iter()
                .zip(axes)
                .map(|(n, a)| choice_or_default::<TwoTowerSchema>(n, a))
                .collect(),
        )
        .expect("typed Two-Tower grid axes are valid by construction")
    }

    fn decode(&self, a: &Assignment) -> Result<TwoTowerParams> {
        TwoTowerSchema::decode(a)
    }

    fn encode(&self, p: &TwoTowerParams) -> Assignment {
        TwoTowerSchema::encode(p)
    }
}

impl SearchSpace for Bert4RecParamGrid {
    type Params = Bert4RecParams;

    fn axes(&self) -> ParamSpace {
        let axes = vec![
            Axis::choice_usize(&self.embedding_dim),
            Axis::choice_usize(&self.num_heads),
            Axis::choice_usize(&self.num_layers),
            Axis::choice_f64(&self.dropout),
            Axis::choice_f64(&self.mask_ratio),
            Axis::choice_f64(&self.learning_rate),
            Axis::choice_usize(&self.num_epochs),
        ];
        ParamSpace::new(
            Bert4RecSchema::NAMES
                .iter()
                .zip(axes)
                .map(|(n, a)| choice_or_default::<Bert4RecSchema>(n, a))
                .collect(),
        )
        .expect("typed BERT4Rec grid axes are valid by construction")
    }

    fn decode(&self, a: &Assignment) -> Result<Bert4RecParams> {
        Bert4RecSchema::decode(a)
    }

    fn encode(&self, p: &Bert4RecParams) -> Assignment {
        Bert4RecSchema::encode(p)
    }
}

// ---------------------------------------------------------------------------
// Metrics (NDCG@k)
// ---------------------------------------------------------------------------

/// Computes NDCG@k given recommended item indices and a set of relevant item indices.
///
/// `recommended` is an ordered list of item indices (best first).
/// `relevant` is the set of item indices that are relevant (ground truth).
fn ndcg_at_k(recommended: &[usize], relevant: &ahash::AHashSet<usize>, k: usize) -> f64 {
    if relevant.is_empty() || k == 0 {
        return 0.0;
    }

    let k = k.min(recommended.len());

    // DCG: sum of 1/log2(rank+2) for relevant items in the top-k
    let mut dcg = 0.0;
    for (i, item) in recommended.iter().take(k).enumerate() {
        if relevant.contains(item) {
            dcg += 1.0 / (i as f64 + 2.0).log2();
        }
    }

    // Ideal DCG: best possible DCG with |relevant| items
    let ideal_k = k.min(relevant.len());
    let mut idcg = 0.0;
    for i in 0..ideal_k {
        idcg += 1.0 / (i as f64 + 2.0).log2();
    }

    if idcg == 0.0 { 0.0 } else { dcg / idcg }
}

// ---------------------------------------------------------------------------
// Cartesian product
// ---------------------------------------------------------------------------

/// Generates the cartesian product of all parameter values in the grid.
///
/// Pre-#97 hand-written enumeration, kept as the test-side reference that
/// `ParamGrid::combinations()` must reproduce exactly (same nested-loop
/// order, first field outermost).
#[cfg(test)]
fn cartesian_product(grid: &ParamGrid) -> Vec<HyperParams> {
    let mut combos = Vec::new();
    for &alpha in &grid.alpha {
        for &beta in &grid.beta {
            for &lambda_ in &grid.lambda_ {
                for &meta_weight in &grid.meta_weight {
                    for &decay_rate in &grid.decay_rate {
                        for &ips_alpha in &grid.ips_alpha {
                            for &sparsity_threshold in &grid.sparsity_threshold {
                                combos.push(HyperParams {
                                    alpha,
                                    beta,
                                    lambda_,
                                    meta_weight,
                                    decay_rate,
                                    ips_alpha,
                                    sparsity_threshold,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    combos
}

// ---------------------------------------------------------------------------
// K-fold split generation
// ---------------------------------------------------------------------------

/// Generate k-fold splits as (train_path, test_path) pairs in a temp directory.
/// Returns the temp dir (caller should keep alive) and the list of fold paths.
fn generate_kfold_splits(
    interactions_path: &str,
    n_folds: usize,
    seed: u64,
) -> Result<(tempfile::TempDir, Vec<(String, String)>)> {
    if n_folds < 2 {
        return Err(anyhow!("n_folds must be >= 2, got {}", n_folds));
    }

    // Read interactions DataFrame
    let df = read_interactions_df(interactions_path)?;

    // Get unique user_ids
    let user_col = df.column("user_id")?.str()?;
    let mut unique_users: Vec<String> = user_col
        .into_iter()
        .flatten()
        .map(|s| s.to_string())
        .collect::<ahash::AHashSet<String>>()
        .into_iter()
        .collect();
    // Sort for deterministic order before shuffling (AHashSet iteration is non-deterministic)
    unique_users.sort();

    if n_folds > unique_users.len() {
        return Err(anyhow!(
            "n_folds ({}) exceeds number of unique users ({})",
            n_folds,
            unique_users.len()
        ));
    }

    // Deterministic shuffle
    let mut rng = StdRng::seed_from_u64(seed);
    unique_users.shuffle(&mut rng);

    // Split into k groups
    let fold_size = unique_users.len() / n_folds;
    let remainder = unique_users.len() % n_folds;

    let mut folds: Vec<Vec<String>> = Vec::with_capacity(n_folds);
    let mut start = 0;
    for i in 0..n_folds {
        let extra = if i < remainder { 1 } else { 0 };
        let end = start + fold_size + extra;
        folds.push(unique_users[start..end].to_vec());
        start = end;
    }

    // Create temp directory for fold files
    let tmp_dir = tempfile::tempdir()?;
    let mut fold_paths = Vec::with_capacity(n_folds);

    for (fold_idx, fold_users) in folds.iter().enumerate() {
        // Test users = fold_idx group; train users = everyone else
        let test_users: ahash::AHashSet<&str> = fold_users.iter().map(|s| s.as_str()).collect();

        // Build boolean mask for train/test
        let user_col = df.column("user_id")?.str()?;
        let mut train_mask = Vec::with_capacity(df.height());
        let mut test_mask = Vec::with_capacity(df.height());
        for val in user_col.into_iter() {
            match val {
                Some(u) => {
                    let is_test = test_users.contains(u);
                    train_mask.push(!is_test);
                    test_mask.push(is_test);
                }
                None => {
                    train_mask.push(false);
                    test_mask.push(false);
                }
            }
        }

        let train_bool = BooleanChunked::from_slice("mask".into(), &train_mask);
        let test_bool = BooleanChunked::from_slice("mask".into(), &test_mask);

        let mut train_df = df.filter(&train_bool)?;
        let mut test_df = df.filter(&test_bool)?;

        let train_path = tmp_dir
            .path()
            .join(format!("fold_{}_train.parquet", fold_idx));
        let test_path = tmp_dir
            .path()
            .join(format!("fold_{}_test.parquet", fold_idx));

        write_parquet(&mut train_df, &train_path.to_string_lossy())?;
        write_parquet(&mut test_df, &test_path.to_string_lossy())?;

        fold_paths.push((
            train_path.to_string_lossy().to_string(),
            test_path.to_string_lossy().to_string(),
        ));
    }

    Ok((tmp_dir, fold_paths))
}

// ---------------------------------------------------------------------------
// Fold evaluator trait
// ---------------------------------------------------------------------------

/// Trains a model on one fold's training split and scores it against that
/// fold's held-out users, returning mean NDCG@k.
///
/// This is the only model-specific seam in the search: the cartesian
/// product, k-fold split generation, parallel runner, and result assembly
/// are all generic over `FoldEvaluator<P>`. EASE is [`EaseFoldEvaluator`]
/// (`P = HyperParams`); SASRec is [`SasRecFoldEvaluator`]
/// (`P = SasRecParams`) and Two-Tower is [`TwoTowerFoldEvaluator`]
/// (`P = TwoTowerParams`).
///
/// Implementors must be `Send + Sync` so the rayon-parallelized
/// `(params × fold)` work product can share one evaluator across threads.
pub trait FoldEvaluator<P>: Send + Sync {
    /// Train on `train_interactions_path` with `params`, then return mean
    /// NDCG@`eval_k` over the users in `test_interactions_path`.
    fn evaluate_fold(
        &self,
        train_interactions_path: &str,
        test_interactions_path: &str,
        params: &P,
        eval_k: usize,
    ) -> Result<f64>;
}

/// Scores a trained model against held-out test users via the
/// [`RecModel`] trait and returns mean NDCG@k.
///
/// Shared by every [`FoldEvaluator`] so the ranking/exclusion/NDCG logic
/// stays in one place. The model is reached as `&dyn RecModel`, so the
/// same scoring path serves EASE, SASRec, and Two-Tower; each evaluator
/// only differs in how it trains and what `ModelInput` it builds.
///
/// `make_input` lets the caller construct the per-user model input
/// (EASE: `ModelInput::Sparse`; sequence/tower models: their own
/// variants) from that user's training interactions and features.
fn score_recmodel_over_test_users<F>(
    model: &dyn RecModel,
    train_user_items: &AHashMap<String, Vec<(usize, f64)>>,
    test_user_items: &AHashMap<String, Vec<(usize, f64)>>,
    user_features_map: &AHashMap<String, Vec<(usize, f64)>>,
    eval_k: usize,
    make_input: F,
) -> Result<f64>
where
    F: for<'a> Fn(&'a [(usize, f64)], &'a [(usize, f64)]) -> ModelInput<'a>,
{
    let mut ndcg_sum = 0.0;
    let mut n_users = 0;

    for (user_id, test_items) in test_user_items {
        if test_items.is_empty() {
            continue;
        }

        let train_items: Vec<(usize, f64)> =
            train_user_items.get(user_id).cloned().unwrap_or_default();
        let user_feats: Vec<(usize, f64)> =
            user_features_map.get(user_id).cloned().unwrap_or_default();

        let scores = model.predict_scores(make_input(&train_items, &user_feats))?;
        ndcg_sum += rank_and_ndcg(&scores, &train_items, test_items, eval_k);
        n_users += 1;
    }

    if n_users == 0 {
        return Ok(0.0);
    }
    Ok(ndcg_sum / n_users as f64)
}

/// Exclude already-seen training items, rank the catalog by score
/// (descending), and return NDCG@k against the held-out test items.
///
/// Shared by every fold scorer so the ranking / exclusion / NDCG logic
/// lives in one place regardless of how the model was fed.
fn rank_and_ndcg(
    scores: &[f32],
    train_items: &[(usize, f64)],
    test_items: &[(usize, f64)],
    eval_k: usize,
) -> f64 {
    let train_item_set: ahash::AHashSet<usize> = train_items.iter().map(|(idx, _)| *idx).collect();
    let mut ranked: Vec<(usize, f32)> = scores
        .iter()
        .copied()
        .enumerate()
        .filter(|(idx, _)| !train_item_set.contains(idx))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let recommended: Vec<usize> = ranked.iter().map(|(idx, _)| *idx).collect();
    let relevant: ahash::AHashSet<usize> = test_items.iter().map(|(idx, _)| *idx).collect();
    ndcg_at_k(&recommended, &relevant, eval_k)
}

/// Two-Tower fold scorer: resolves each test user to its trained user-tower
/// row by id and scores the catalog through `ModelInput::TowerUser`.
///
/// Two-Tower rejects `ModelInput::Sparse`, and its `TowerUser` variant
/// borrows owned per-call slices that the `make_input` closure of
/// [`score_recmodel_over_test_users`] cannot produce, so this is its
/// dedicated scorer. The rank / exclusion / NDCG core is the shared
/// [`rank_and_ndcg`]. A user absent from the trained mapping (unseen in
/// this fold's train split) is scored cold-start (`user_idx = None`),
/// which exercises the model's reserved cold-start row.
#[cfg(feature = "ml-models")]
fn score_two_tower_over_test_users(
    model: &dyn RecModel,
    train_user_items: &AHashMap<String, Vec<(usize, f64)>>,
    test_user_items: &AHashMap<String, Vec<(usize, f64)>>,
    eval_k: usize,
) -> Result<f64> {
    let user_to_idx = &model.item_mapping().user_to_idx;
    let mut ndcg_sum = 0.0;
    let mut n_users = 0;

    for (user_id, test_items) in test_user_items {
        if test_items.is_empty() {
            continue;
        }
        let train_items: Vec<(usize, f64)> =
            train_user_items.get(user_id).cloned().unwrap_or_default();
        let user_idx = user_to_idx.get(user_id).copied();
        let scores = model.predict_scores(ModelInput::TowerUser {
            user_idx,
            cat_features: &[],
            dense_features: &[],
        })?;
        ndcg_sum += rank_and_ndcg(&scores, &train_items, test_items, eval_k);
        n_users += 1;
    }

    if n_users == 0 {
        return Ok(0.0);
    }
    Ok(ndcg_sum / n_users as f64)
}

// ---------------------------------------------------------------------------
// EASE fold evaluator
// ---------------------------------------------------------------------------

/// EASE [`FoldEvaluator`]: builds X/U/T matrices via the data pipeline,
/// trains a [`RustFeaseModel`], and scores users through the [`RecModel`]
/// trait.
///
/// The numeric path is byte-identical to the pre-generalization
/// `evaluate_trial`: EASE's closed-form `predict` is unchanged, and the
/// only difference the trait introduces is the single `as f32` score
/// cast in `EaseAdapter` (the same cast Phase 4a's eval already makes).
/// `test_parallel_grid_search_matches_sequential` and
/// `test_ease_search_matches_legacy_concrete` guard this within 1e-9.
pub struct EaseFoldEvaluator {
    pub user_features_path: String,
    pub item_features_path: String,
}

impl FoldEvaluator<HyperParams> for EaseFoldEvaluator {
    fn evaluate_fold(
        &self,
        train_interactions_path: &str,
        test_interactions_path: &str,
        params: &HyperParams,
        eval_k: usize,
    ) -> Result<f64> {
        evaluate_trial(
            train_interactions_path,
            test_interactions_path,
            &self.user_features_path,
            &self.item_features_path,
            params,
            eval_k,
        )
    }
}

// ---------------------------------------------------------------------------
// Single trial evaluation (EASE)
// ---------------------------------------------------------------------------

/// Train and evaluate a single EASE parameter configuration on one fold.
/// Returns mean NDCG@k across test users for the specified k.
///
/// Trains a [`RustFeaseModel`], wraps it in [`EaseAdapter`], and scores it
/// through the shared `&dyn RecModel` path. EASE's closed-form prediction
/// math is unchanged; the trait only adds the `as f32` output cast.
fn evaluate_trial(
    train_interactions_path: &str,
    test_interactions_path: &str,
    user_features_path: &str,
    item_features_path: &str,
    params: &HyperParams,
    eval_k: usize,
) -> Result<f64> {
    // 1. Build WeightingConfig from params
    let weighting =
        if params.decay_rate > 0.0 || params.ips_alpha > 0.0 || params.sparsity_threshold > 0.0 {
            Some(WeightingConfig {
                event_weights: None,
                decay_rate: params.decay_rate,
                ips_alpha: params.ips_alpha,
                sparsity_threshold: params.sparsity_threshold,
            })
        } else {
            None
        };

    // 2. Build matrices from training data
    let (x_mat, u_mat, t_mat, mappings) = data_pipeline::build_matrices(
        train_interactions_path,
        user_features_path,
        item_features_path,
        weighting.as_ref(),
    )?;

    let num_items = x_mat.cols();
    let num_user_features = u_mat.cols();
    let num_item_features = t_mat.rows();

    // 3. Train model
    let mut model = RustFeaseModel::new(
        num_items,
        num_user_features,
        num_item_features,
        params.alpha,
        params.beta,
        params.lambda_,
        params.meta_weight,
        mappings,
    );
    model.train(&x_mat, &u_mat, &t_mat)?;

    // 4. Apply sparsity pruning if needed
    if params.sparsity_threshold > 0.0 {
        model.prune_sparse(params.sparsity_threshold);
    }

    // 5. Read training interactions to know which items each user has seen
    let train_df = read_interactions_df(train_interactions_path)?;
    let train_user_items = group_user_items(&train_df, &model.mappings)?;

    // 6. Read test interactions and group by user
    let test_df = read_interactions_df(test_interactions_path)?;
    let test_user_items = group_user_items(&test_df, &model.mappings)?;

    // 7. Build user features lookup
    let user_features_map = build_user_features_map(user_features_path, &model.mappings)?;

    // 8. Score every test user through the shared `&dyn RecModel` path.
    //    EASE's `predict` math is unchanged; `EaseAdapter` only adds the
    //    `as f32` output cast (guarded within 1e-9 by the determinism and
    //    legacy-baseline tests).
    let adapter = EaseAdapter::new(model);
    score_recmodel_over_test_users(
        &adapter,
        &train_user_items,
        &test_user_items,
        &user_features_map,
        eval_k,
        |interactions, user_features| ModelInput::Sparse {
            interactions,
            user_features,
        },
    )
}

// ---------------------------------------------------------------------------
// SASRec fold evaluator (ml-models)
// ---------------------------------------------------------------------------

/// SASRec [`FoldEvaluator`]: builds left-padded causal sequences from the
/// fold's training interactions, trains a burn `SasRec` for the trial's
/// architecture/optimizer config, wraps it in [`TrainedSasRec`], and scores
/// held-out users through `SasRecEvalAdapter` so the per-user history is
/// ordered chronologically by `days_ago` before scoring (issue #51).
///
/// SASRec is order-sensitive; the sequence builder requires a numeric
/// `days_ago` column. The per-fold scorer used to pass per-user train items
/// as `ModelInput::Sparse`, which discarded chronology — fixed in #51 by
/// routing through the adapter. `max_seq_len` is fixed per evaluator
/// (not a tuned axis) so every trial sees the same sequence horizon.
#[cfg(feature = "ml-models")]
pub struct SasRecFoldEvaluator {
    /// History length / positional-embedding cap shared by every trial.
    pub max_seq_len: usize,
    /// Mini-batch size for SGD (fixed; not a tuned axis).
    pub batch_size: usize,
    /// Early-stopping patience in epochs.
    pub patience: usize,
    /// Seed for the training loop (kept fixed so trials are comparable).
    pub seed: u64,
}

#[cfg(feature = "ml-models")]
impl Default for SasRecFoldEvaluator {
    fn default() -> Self {
        Self {
            max_seq_len: 50,
            batch_size: 16,
            patience: 5,
            seed: 42,
        }
    }
}

#[cfg(feature = "ml-models")]
impl FoldEvaluator<SasRecParams> for SasRecFoldEvaluator {
    fn evaluate_fold(
        &self,
        train_interactions_path: &str,
        test_interactions_path: &str,
        params: &SasRecParams,
        eval_k: usize,
    ) -> Result<f64> {
        use crate::data::sequences::build_sequences;
        use crate::models::sasrec::{
            SasRecConfig, SasRecTrainingConfig, TrainedSasRec, train_sasrec,
        };
        use burn::backend::ndarray::NdArrayDevice;
        use burn::backend::{Autodiff, NdArray};

        let mappings = data_pipeline::build_interaction_mappings(train_interactions_path)?;
        let dataset = build_sequences(train_interactions_path, &mappings, self.max_seq_len)?;

        let vocab_size = mappings.idx_to_item.len() + 1;
        let model_config = SasRecConfig::new(
            vocab_size,
            params.embedding_dim,
            self.max_seq_len,
            params.num_heads,
            params.num_layers,
        )
        .with_dropout(params.dropout);
        let train_config = SasRecTrainingConfig::new()
            .with_num_epochs(params.num_epochs)
            .with_batch_size(self.batch_size)
            .with_learning_rate(params.learning_rate)
            .with_patience(self.patience)
            .with_seed(self.seed);

        let device = NdArrayDevice::default();
        let fitted = train_sasrec::<Autodiff<NdArray<f32>>>(
            &model_config,
            &train_config,
            &dataset,
            &device,
        )?;
        let trained = TrainedSasRec::new(fitted, model_config, mappings);

        // SASRec per-fold scoring uses `SasRecEvalAdapter` for the
        // chronological-sort + `Sequence` construction, but does *not*
        // go through `evaluate_with_adapter`. That harness skips test
        // users absent from `user_to_idx`, which is fine for the public
        // eval API but wrong here: `generate_kfold_splits` partitions
        // users disjointly, so by construction no test user is in the
        // train fold's `user_to_idx` and the harness would skip them
        // all (NDCG=0 every trial). Mirrors `score_two_tower_over_
        // test_users`, which is its own loop for the same reason.
        let adapter = crate::evaluation::SasRecEvalAdapter::new(&trained);

        let train_df = read_interactions_df(train_interactions_path)?;
        let (train_user_items, train_user_days_ago) =
            group_user_items_with_days_ago(&train_df, trained.item_mapping())?;
        let test_df = read_interactions_df(test_interactions_path)?;
        let test_user_items = group_user_items(&test_df, trained.item_mapping())?;

        let mut ndcg_sum = 0.0;
        let mut n_users = 0;
        for (user_id, test_items) in &test_user_items {
            if test_items.is_empty() {
                continue;
            }
            let train_items: Vec<(usize, f64)> =
                train_user_items.get(user_id).cloned().unwrap_or_default();
            let days_ago_vec: Vec<f64> = train_user_days_ago
                .get(user_id)
                .cloned()
                .unwrap_or_default();

            let ctx = crate::evaluation::UserEvalContext {
                train_items: &train_items,
                // Always Some — group_user_items_with_days_ago already
                // bailed if the train fold had no `days_ago` column.
                // For test users with no train interactions (the common
                // case under user-disjoint k-fold), the slice is empty.
                train_days_ago: Some(&days_ago_vec),
                user_features: &[],
                user_idx: None,
            };
            let scores =
                <_ as crate::evaluation::EvalAdapter>::predict_user_scores(&adapter, &ctx)?;
            ndcg_sum += rank_and_ndcg(&scores, &train_items, test_items, eval_k);
            n_users += 1;
        }

        if n_users == 0 {
            return Ok(0.0);
        }
        Ok(ndcg_sum / n_users as f64)
    }
}

// ---------------------------------------------------------------------------
// Two-Tower fold evaluator (ml-models)
// ---------------------------------------------------------------------------

/// Two-Tower [`FoldEvaluator`]: loads `(user, positive-item)` triples from
/// the fold's training interactions (id-only — no side-feature files in the
/// tuning surface), trains the in-batch sampled-softmax model for the
/// trial's config, and scores held-out users through the dedicated
/// [`score_two_tower_over_test_users`] path (Two-Tower needs
/// `ModelInput::TowerUser`, which the generic `make_input` closure cannot
/// build). Epochs / batch size are fixed per evaluator; the tuned axes are
/// `embedding_dim`, `temperature`, `learning_rate`, and `id_dropout`.
#[cfg(feature = "ml-models")]
pub struct TwoTowerFoldEvaluator {
    /// Training epochs (fixed; tuned axes are dim/temp/lr/id_dropout).
    pub epochs: usize,
    /// Mini-batch size for the sampled-softmax loss.
    pub batch_size: usize,
    /// Seed for the id-dropout RNG (kept fixed so trials are comparable).
    pub seed: u64,
}

#[cfg(feature = "ml-models")]
impl Default for TwoTowerFoldEvaluator {
    fn default() -> Self {
        Self {
            epochs: 50,
            batch_size: 256,
            seed: 0,
        }
    }
}

#[cfg(feature = "ml-models")]
impl FoldEvaluator<TwoTowerParams> for TwoTowerFoldEvaluator {
    fn evaluate_fold(
        &self,
        train_interactions_path: &str,
        test_interactions_path: &str,
        params: &TwoTowerParams,
        eval_k: usize,
    ) -> Result<f64> {
        use crate::data::triples::{FeatureTable, load_triples};
        use crate::models::two_tower::{TrainParams, train};

        let data = load_triples(train_interactions_path)?;
        // Id-only model: tuning has no user/item feature files. Tables are
        // sized to the embedding tables (users include the reserved
        // cold-start row at index 0).
        let user_ft = FeatureTable::empty(data.num_users());
        let item_ft = FeatureTable::empty(data.num_items());

        let trained = train(
            &data,
            &user_ft,
            &item_ft,
            TrainParams {
                embedding_dim: params.embedding_dim,
                temperature: params.temperature,
                learning_rate: params.learning_rate,
                epochs: self.epochs,
                batch_size: self.batch_size,
                id_dropout: params.id_dropout,
                seed: self.seed,
            },
        )?;

        let train_df = read_interactions_df(train_interactions_path)?;
        let train_user_items = group_user_items(&train_df, trained.item_mapping())?;
        let test_df = read_interactions_df(test_interactions_path)?;
        let test_user_items = group_user_items(&test_df, trained.item_mapping())?;

        score_two_tower_over_test_users(&trained, &train_user_items, &test_user_items, eval_k)
    }
}

// ---------------------------------------------------------------------------
// BERT4Rec fold evaluator (ml-models, #96 / #97)
// ---------------------------------------------------------------------------

/// BERT4Rec [`FoldEvaluator`]: builds masked Cloze sequences from the
/// fold's training interactions (`days_ago` required), trains a burn
/// `Bert4Rec` for the trial's config, wraps it in [`TrainedBert4Rec`], and
/// scores held-out users through `Bert4RecEvalAdapter` so each history is
/// ordered by `days_ago` and bucketed exactly as at training time.
/// `max_seq_len`, `num_position_buckets`, `batch_size`, `patience` and
/// `seed` are fixed per evaluator; the tuned axes are the architecture /
/// optimizer knobs in [`Bert4RecParams`].
///
/// Like the SASRec evaluator this loops over test users itself rather
/// than through `evaluate_with_adapter`: k-fold splits are user-disjoint,
/// so no test user is in the train fold's `user_to_idx` and the public
/// harness would skip them all.
#[cfg(feature = "ml-models")]
pub struct Bert4RecFoldEvaluator {
    pub max_seq_len: usize,
    pub num_position_buckets: usize,
    pub batch_size: usize,
    pub patience: usize,
    pub seed: u64,
}

#[cfg(feature = "ml-models")]
impl Default for Bert4RecFoldEvaluator {
    fn default() -> Self {
        Self {
            max_seq_len: 50,
            num_position_buckets: 32,
            batch_size: 64,
            patience: 5,
            seed: 42,
        }
    }
}

#[cfg(feature = "ml-models")]
impl FoldEvaluator<Bert4RecParams> for Bert4RecFoldEvaluator {
    fn evaluate_fold(
        &self,
        train_interactions_path: &str,
        test_interactions_path: &str,
        params: &Bert4RecParams,
        eval_k: usize,
    ) -> Result<f64> {
        use crate::data::masked_sequences::build_masked_sequences;
        use crate::models::bert4rec::{
            Bert4RecConfig, Bert4RecTrainingConfig, TrainedBert4Rec, train_bert4rec,
        };
        use burn::backend::ndarray::NdArrayDevice;
        use burn::backend::{Autodiff, NdArray};

        let mappings = data_pipeline::build_interaction_mappings(train_interactions_path)?;
        let vocab_size = mappings.idx_to_item.len() + 2;
        let model_config = Bert4RecConfig::new(
            vocab_size,
            self.max_seq_len,
            params.num_heads,
            params.num_layers,
        )
        .with_embedding_dim(params.embedding_dim)
        .with_num_position_buckets(self.num_position_buckets)
        .with_dropout(params.dropout)
        .with_mask_ratio(params.mask_ratio);
        model_config.check()?;

        let dataset = build_masked_sequences(
            train_interactions_path,
            &mappings,
            self.max_seq_len,
            params.mask_ratio,
            self.num_position_buckets,
            self.seed,
        )?;
        let train_config = Bert4RecTrainingConfig::new()
            .with_num_epochs(params.num_epochs)
            .with_batch_size(self.batch_size)
            .with_learning_rate(params.learning_rate)
            .with_patience(self.patience)
            .with_seed(self.seed);

        let device = NdArrayDevice::default();
        let fitted = train_bert4rec::<Autodiff<NdArray<f32>>>(
            &model_config,
            &train_config,
            &dataset,
            &device,
        )?;
        let trained = TrainedBert4Rec::new(fitted, model_config, mappings);
        let adapter = crate::evaluation::Bert4RecEvalAdapter::new(&trained);

        let train_df = read_interactions_df(train_interactions_path)?;
        let (train_user_items, train_user_days_ago) =
            group_user_items_with_days_ago(&train_df, trained.item_mapping())?;
        let test_df = read_interactions_df(test_interactions_path)?;
        let test_user_items = group_user_items(&test_df, trained.item_mapping())?;

        let mut ndcg_sum = 0.0;
        let mut n_users = 0;
        for (user_id, test_items) in &test_user_items {
            if test_items.is_empty() {
                continue;
            }
            let train_items: Vec<(usize, f64)> =
                train_user_items.get(user_id).cloned().unwrap_or_default();
            let days_ago_vec: Vec<f64> = train_user_days_ago
                .get(user_id)
                .cloned()
                .unwrap_or_default();
            let ctx = crate::evaluation::UserEvalContext {
                train_items: &train_items,
                train_days_ago: Some(&days_ago_vec),
                user_features: &[],
                user_idx: None,
            };
            let scores =
                <_ as crate::evaluation::EvalAdapter>::predict_user_scores(&adapter, &ctx)?;
            ndcg_sum += rank_and_ndcg(&scores, &train_items, test_items, eval_k);
            n_users += 1;
        }

        if n_users == 0 {
            return Ok(0.0);
        }
        Ok(ndcg_sum / n_users as f64)
    }
}

// ---------------------------------------------------------------------------
// Parallel trial runner
// ---------------------------------------------------------------------------

/// Evaluates every `configs` entry over all CV folds in parallel and returns
/// one [`TrialResult`] per config, in `configs` order.
///
/// The `(params × fold)` work product is embarrassingly parallel: each trial
/// is a pure function of `(params, train_path, test_path)` reading immutable
/// fold Parquet files written once by [`generate_kfold_splits`]. Parallelism
/// uses rayon's global pool (honors `RAYON_NUM_THREADS`); no private pool is
/// constructed, per ADR-0002 §"Risks".
///
/// Determinism is preserved despite non-deterministic completion order:
/// each trial is keyed by its stable index in `configs`, and the fold
/// scores are regrouped by `(trial_idx, fold_idx)` so `fold_scores[i]`
/// always corresponds to `fold_paths[i]`. `trial_offset` / `total` only
/// affect the progress log.
fn evaluate_configs_parallel<P, E>(
    evaluator: &E,
    configs: &[P],
    fold_paths: &[(String, String)],
    eval_k: usize,
    trial_offset: usize,
    total: usize,
) -> Result<Vec<TrialResult<P>>>
where
    P: Clone + Send + Sync,
    E: FoldEvaluator<P>,
{
    let n_configs = configs.len();
    let n_folds = fold_paths.len();

    // Flatten the `(trial, fold)` cartesian product into one flat work list
    // so the rayon pool sees a single even work product. A nested
    // `configs.par_iter()` → `fold_paths.par_iter()` would let the outer
    // parallelism saturate the pool and effectively serialize the inner
    // fold loop for small fold counts; one flat `par_iter` over the
    // `n_configs * n_folds` items distributes work evenly.
    let work: Vec<(usize, usize)> = (0..n_configs)
        .flat_map(|trial_idx| (0..n_folds).map(move |fold_idx| (trial_idx, fold_idx)))
        .collect();

    let mut fold_results: Vec<(usize, usize, f64)> = work
        .par_iter()
        .map(|&(trial_idx, fold_idx)| -> Result<(usize, usize, f64)> {
            let (train_path, test_path) = &fold_paths[fold_idx];
            let score =
                evaluator.evaluate_fold(train_path, test_path, &configs[trial_idx], eval_k)?;
            Ok((trial_idx, fold_idx, score))
        })
        .collect::<Result<Vec<_>>>()?;

    // Regroup deterministically: sorting by `(trial_idx, fold_idx)` makes
    // each trial's fold scores independent of parallel completion order.
    // Each trial has exactly `n_folds` consecutive entries, so
    // `chunks(n_folds)` yields the per-trial groups in ascending
    // `trial_idx` order.
    fold_results.sort_by_key(|&(trial_idx, fold_idx, _)| (trial_idx, fold_idx));

    Ok(fold_results
        .chunks(n_folds)
        .enumerate()
        .map(|(trial_idx, chunk)| {
            debug_assert!(chunk.iter().all(|&(t, _, _)| t == trial_idx));
            let fold_scores: Vec<f64> = chunk.iter().map(|&(_, _, s)| s).collect();
            let mean_score = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
            log::info!(
                "Trial {}/{} -> NDCG@{}={:.4}",
                trial_offset + trial_idx + 1,
                total,
                eval_k,
                mean_score
            );
            TrialResult::<P> {
                params: configs[trial_idx].clone(),
                mean_score,
                fold_scores,
            }
        })
        .collect())
}

/// Assemble a [`SearchResult`] from trials in evaluation order. `best` is
/// the highest `mean_score`, ties broken on the earliest trial, so the
/// result is independent of execution order.
fn assemble_result<P: Clone>(
    all_trials: Vec<TrialResult<P>>,
    eval_k: usize,
    strategy: &str,
) -> Result<SearchResult<P>> {
    let first = all_trials
        .first()
        .ok_or_else(|| anyhow!("search produced no trials"))?;
    let mut best_score = f64::NEG_INFINITY;
    let mut best_params = first.params.clone();
    for trial in &all_trials {
        // Strict `>` keeps the first-seen winner among score ties.
        if trial.mean_score > best_score {
            best_score = trial.mean_score;
            best_params = trial.params.clone();
        }
    }
    Ok(SearchResult {
        best_params,
        best_score,
        all_trials,
        metric_name: format!("ndcg@{}", eval_k),
        strategy: strategy.to_string(),
    })
}

// ---------------------------------------------------------------------------
// Strategy-driven runner (#97 / ADR-0005)
// ---------------------------------------------------------------------------

/// Budget and CV settings for [`tune_with`].
#[derive(Debug, Clone)]
pub struct TuneConfig {
    /// User-based k-fold count (`>= 2`).
    pub n_folds: usize,
    /// NDCG cutoff that is optimised.
    pub eval_k: usize,
    /// Seeds the fold shuffle (`seed`) and the strategy RNG (`seed + 1`,
    /// matching the pre-#97 random search).
    pub seed: u64,
    /// Maximum number of new trials to evaluate.
    pub max_trials: usize,
}

/// Run `strategy` over `space` against `evaluator` with user-based k-fold
/// CV: generate the folds once, then loop *ask → evaluate batch in
/// parallel → observe* until the strategy stops proposing or `max_trials`
/// is reached. `warm_start` trials (e.g. a previous run's `all_trials`)
/// seed the strategy's history but are not re-evaluated and are not part
/// of the returned `all_trials`.
pub fn tune_with<S, E>(
    evaluator: &E,
    interactions_path: &str,
    space: &S,
    strategy: &mut dyn SearchStrategy,
    cfg: &TuneConfig,
    warm_start: &[TrialResult<S::Params>],
) -> Result<SearchResult<S::Params>>
where
    S: SearchSpace,
    E: FoldEvaluator<S::Params>,
{
    if cfg.max_trials == 0 {
        return Err(anyhow!("max_trials must be >= 1"));
    }
    let axes = space.axes();
    log::info!(
        "{} search: up to {} trials over {} axes, {}-fold CV",
        strategy.name(),
        cfg.max_trials,
        axes.len(),
        cfg.n_folds
    );

    // Generate k-fold splits once.
    let (_tmp_dir, fold_paths) = generate_kfold_splits(interactions_path, cfg.n_folds, cfg.seed)?;

    // Strategy RNG decoupled from fold generation (seed+1), as before.
    let mut rng = StdRng::seed_from_u64(cfg.seed.wrapping_add(1));

    let mut history: Vec<Observation> = warm_start
        .iter()
        .map(|t| Observation {
            assignment: space.encode(&t.params),
            score: t.mean_score,
        })
        .collect();
    let mut all_trials: Vec<TrialResult<S::Params>> = Vec::new();

    while all_trials.len() < cfg.max_trials {
        let remaining = cfg.max_trials - all_trials.len();
        let batch = strategy.propose(&axes, &history, remaining, &mut rng)?;
        if batch.is_empty() {
            break;
        }
        let configs: Vec<S::Params> = batch
            .iter()
            .map(|a| {
                axes.check(a)?;
                space.decode(a)
            })
            .collect::<Result<_>>()?;
        let total_hint = if strategy.name() == "tpe" {
            cfg.max_trials
        } else {
            all_trials.len() + configs.len()
        };
        let trials = evaluate_configs_parallel(
            evaluator,
            &configs,
            &fold_paths,
            cfg.eval_k,
            all_trials.len(),
            total_hint,
        )?;
        for (assignment, trial) in batch.into_iter().zip(&trials) {
            history.push(Observation {
                assignment,
                score: trial.mean_score,
            });
        }
        all_trials.extend(trials);
    }

    assemble_result(all_trials, cfg.eval_k, strategy.name())
}

// ---------------------------------------------------------------------------
// Grid search
// ---------------------------------------------------------------------------

/// Generic grid search over any [`FoldEvaluator`].
///
/// Thin wrapper over [`tune_with`] with a [`GridStrategy`]: every
/// combination of the (finite) space runs across all folds in one
/// parallel batch. Errors if the grid is empty or an axis is continuous.
pub fn grid_search_with<S, E>(
    evaluator: &E,
    interactions_path: &str,
    grid: &S,
    n_folds: usize,
    eval_k: usize,
    seed: u64,
) -> Result<SearchResult<S::Params>>
where
    S: SearchSpace,
    E: FoldEvaluator<S::Params>,
{
    let total = grid.axes().cardinality().unwrap_or(0);
    if total == 0 {
        return Err(anyhow!("Parameter grid produced 0 combinations"));
    }
    let cfg = TuneConfig {
        n_folds,
        eval_k,
        seed,
        max_trials: total,
    };
    tune_with(
        evaluator,
        interactions_path,
        grid,
        &mut GridStrategy::new(),
        &cfg,
        &[],
    )
}
/// EASE grid search: evaluates all combinations of parameters in the grid.
///
/// The `(params × fold)` trials run in parallel via rayon's global pool
/// (ADR-0002 Phase 1). The result is deterministic for a fixed `seed` and
/// grid regardless of thread count: see [`evaluate_configs_parallel`]. This is a
/// thin wrapper over [`grid_search_with`] using [`EaseFoldEvaluator`], kept
/// so the existing EASE call sites and determinism guard are unchanged.
pub fn grid_search(
    interactions_path: &str,
    user_features_path: &str,
    item_features_path: &str,
    grid: &ParamGrid,
    n_folds: usize,
    eval_k: usize,
    seed: u64,
) -> Result<SearchResult> {
    let evaluator = EaseFoldEvaluator {
        user_features_path: user_features_path.to_string(),
        item_features_path: item_features_path.to_string(),
    };
    grid_search_with(&evaluator, interactions_path, grid, n_folds, eval_k, seed)
}

// ---------------------------------------------------------------------------
// Random search
// ---------------------------------------------------------------------------

/// Generic random search over any [`FoldEvaluator`].
///
/// Thin wrapper over [`tune_with`] with a [`RandomStrategy`]: `n_trials`
/// configurations are sampled sequentially from an RNG seeded with
/// `seed + 1` (so the set is a deterministic function of `seed`), then
/// evaluated in one parallel batch.
#[allow(clippy::too_many_arguments)]
pub fn random_search_with<S, E>(
    evaluator: &E,
    interactions_path: &str,
    grid: &S,
    n_trials: usize,
    n_folds: usize,
    eval_k: usize,
    seed: u64,
) -> Result<SearchResult<S::Params>>
where
    S: SearchSpace,
    E: FoldEvaluator<S::Params>,
{
    if n_trials == 0 {
        return Err(anyhow!("n_trials must be >= 1"));
    }
    let cfg = TuneConfig {
        n_folds,
        eval_k,
        seed,
        max_trials: n_trials,
    };
    tune_with(
        evaluator,
        interactions_path,
        grid,
        &mut RandomStrategy::new(n_trials),
        &cfg,
        &[],
    )
}
/// EASE random search: samples n_trials random parameter configurations
/// from the grid.
///
/// Thin wrapper over [`random_search_with`] using [`EaseFoldEvaluator`],
/// kept so existing EASE call sites and the determinism guard are
/// unchanged. See [`random_search_with`] for determinism details.
#[allow(clippy::too_many_arguments)]
pub fn random_search(
    interactions_path: &str,
    user_features_path: &str,
    item_features_path: &str,
    grid: &ParamGrid,
    n_trials: usize,
    n_folds: usize,
    eval_k: usize,
    seed: u64,
) -> Result<SearchResult> {
    let evaluator = EaseFoldEvaluator {
        user_features_path: user_features_path.to_string(),
        item_features_path: item_features_path.to_string(),
    };
    random_search_with(
        &evaluator,
        interactions_path,
        grid,
        n_trials,
        n_folds,
        eval_k,
        seed,
    )
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Groups interactions by user, returning user_id -> Vec<(item_idx, value)>.
fn group_user_items(
    df: &DataFrame,
    mappings: &Mappings,
) -> Result<AHashMap<String, Vec<(usize, f64)>>> {
    let user_col = df.column("user_id")?.str()?;
    let item_col = df.column("item_id")?.str()?;
    let val_col = df.column("value")?.f64()?;

    let mut map: AHashMap<String, Vec<(usize, f64)>> = AHashMap::new();

    for ((user, item), val) in user_col.into_iter().zip(item_col).zip(val_col) {
        if let (Some(u), Some(i), Some(v)) = (user, item, val)
            && let Some(&item_idx) = mappings.item_to_idx.get(i)
        {
            map.entry(u.to_string()).or_default().push((item_idx, v));
        }
    }

    Ok(map)
}

/// Per-user (items, days_ago) maps in lockstep — see
/// [`group_user_items_with_days_ago`].
#[cfg(feature = "ml-models")]
type UserItemsWithDaysAgo = (
    AHashMap<String, Vec<(usize, f64)>>,
    AHashMap<String, Vec<f64>>,
);

/// Same as [`group_user_items`] but also collects each row's `days_ago`
/// in parallel. The two per-user vecs are kept in lockstep: a row with a
/// null `days_ago` is skipped entirely so SasRec scorers can pair items
/// and days_ago by index. Used by `SasRecFoldEvaluator`, which requires
/// chronological ordering at scoring time (#51).
#[cfg(feature = "ml-models")]
fn group_user_items_with_days_ago(
    df: &DataFrame,
    mappings: &Mappings,
) -> Result<UserItemsWithDaysAgo> {
    let user_col = df.column("user_id")?.str()?;
    let item_col = df.column("item_id")?.str()?;
    let val_col = df.column("value")?.f64()?;
    let days_col = df.column("days_ago").map_err(|_| {
        anyhow::anyhow!(
            "SasRecFoldEvaluator requires a `days_ago` column in the train fold (matches \
             the same requirement in `data::sequences::build_sequences`)."
        )
    })?;
    let days_col = days_col.f64()?;

    let mut items: AHashMap<String, Vec<(usize, f64)>> = AHashMap::new();
    let mut days: AHashMap<String, Vec<f64>> = AHashMap::new();

    for i in 0..df.height() {
        let (Some(u), Some(it), Some(v)) = (user_col.get(i), item_col.get(i), val_col.get(i))
        else {
            continue;
        };
        let Some(&item_idx) = mappings.item_to_idx.get(it) else {
            continue;
        };
        let Some(d) = days_col.get(i) else {
            // Lockstep skip: a row with null days_ago is excluded from
            // BOTH maps so the parallel vecs stay aligned.
            continue;
        };
        items.entry(u.to_string()).or_default().push((item_idx, v));
        days.entry(u.to_string()).or_default().push(d);
    }

    Ok((items, days))
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use polars::df;
    use std::fs::File;
    use std::path::Path;

    fn make_big_dataset(
        n_users: usize,
        n_items: usize,
    ) -> Result<(String, String, String, tempfile::TempDir)> {
        let tmp = tempfile::tempdir()?;
        let mut u = Vec::new();
        let mut it = Vec::new();
        let mut v = Vec::new();
        for ui in 0..n_users {
            for k in 0..8 {
                u.push(format!("u{}", ui));
                it.push(format!("i{}", (ui * 7 + k * 13) % n_items));
                v.push(1.0_f64);
            }
        }
        let mut interactions = df!("user_id" => u, "item_id" => it, "value" => v)?;
        let uf_id: Vec<String> = (0..n_users).map(|i| format!("u{}", i)).collect();
        let uf_fn: Vec<String> = (0..n_users).map(|i| format!("f{}", i % 5)).collect();
        let uf_v: Vec<f64> = vec![1.0; n_users];
        let mut user_features = df!("user_id" => uf_id, "feature_name" => uf_fn, "value" => uf_v)?;
        let if_id: Vec<String> = (0..n_items).map(|i| format!("i{}", i)).collect();
        let if_fn: Vec<String> = (0..n_items).map(|i| format!("g{}", i % 4)).collect();
        let if_v: Vec<f64> = vec![1.0; n_items];
        let mut item_features = df!("item_id" => if_id, "feature_name" => if_fn, "value" => if_v)?;
        let i_path = create_parquet_in(tmp.path(), "i.parquet", &mut interactions)?;
        let u_path = create_parquet_in(tmp.path(), "u.parquet", &mut user_features)?;
        let t_path = create_parquet_in(tmp.path(), "t.parquet", &mut item_features)?;
        Ok((i_path, u_path, t_path, tmp))
    }

    /// Non-CI timing harness (run with `--ignored --nocapture`). Compares the
    /// genuinely-sequential baseline against the rayon `grid_search` in one
    /// process for a representative grid, and asserts identical best score.
    #[test]
    #[ignore]
    fn bench_parallel_vs_sequential() -> Result<()> {
        let (i, up, tp, _g) = make_big_dataset(120, 60)?;
        let grid = ParamGrid {
            alpha: vec![0.5, 1.0, 2.0],
            beta: vec![0.5, 1.0],
            lambda_: vec![10.0, 100.0, 500.0],
            meta_weight: vec![0.0],
            decay_rate: vec![0.0],
            ips_alpha: vec![0.0],
            sparsity_threshold: vec![0.0],
        };
        let (nf, ek, sd) = (4usize, 10usize, 42u64);
        let t0 = std::time::Instant::now();
        let seq = sequential_grid_baseline(&i, &up, &tp, &grid, nf, ek, sd)?;
        let seq_t = t0.elapsed();
        let t1 = std::time::Instant::now();
        let par = grid_search(&i, &up, &tp, &grid, nf, ek, sd)?;
        let par_t = t1.elapsed();
        eprintln!(
            "BENCH trials={} folds={} threads={} sequential={:?} parallel={:?} speedup={:.2}x",
            par.all_trials.len(),
            nf,
            rayon::current_num_threads(),
            seq_t,
            par_t,
            seq_t.as_secs_f64() / par_t.as_secs_f64()
        );
        // Tolerance, not bit-exact: `evaluate_trial` accumulates NDCG by
        // iterating an `AHashMap` whose iteration order is randomized per
        // process (pre-existing behavior, unrelated to this PR's rayon
        // change). Sub-ULP float drift is expected and rank-irrelevant
        // (ADR-0002 §Negative). The deterministic CI gate is
        // `test_parallel_grid_search_matches_sequential` on a fixed small grid.
        assert!(
            (par.best_score - seq.best_score).abs() < 1e-9,
            "parallel best_score {} vs sequential {}",
            par.best_score,
            seq.best_score
        );
        Ok(())
    }

    /// Helper to create a dummy parquet file in a temp dir and return its path.
    fn create_parquet_in(dir: &Path, name: &str, df: &mut DataFrame) -> Result<String> {
        let path = dir.join(name);
        let mut file = File::create(&path)?;
        ParquetWriter::new(&mut file).finish(df)?;
        Ok(path.to_string_lossy().to_string())
    }

    /// Creates a tiny dataset suitable for tuning tests.
    /// Returns (interactions_path, user_features_path, item_features_path, tmpdir).
    fn create_test_dataset() -> Result<(String, String, String, tempfile::TempDir)> {
        let tmp = tempfile::tempdir()?;

        // 6 users, 4 items — enough for 2- or 3-fold splits
        let mut interactions = df!(
            "user_id" => ["u0","u0","u1","u1","u2","u2","u3","u3","u4","u4","u5","u5"],
            "item_id" => ["i0","i1","i1","i2","i0","i2","i2","i3","i0","i3","i1","i3"],
            "value"   => [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )?;

        let mut user_features = df!(
            "user_id"      => ["u0","u1","u2","u3","u4","u5"],
            "feature_name" => ["f_a","f_b","f_a","f_b","f_a","f_b"],
            "value"        => [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )?;

        let mut item_features = df!(
            "item_id"      => ["i0","i1","i2","i3"],
            "feature_name" => ["g_x","g_y","g_x","g_y"],
            "value"        => [1.0, 1.0, 1.0, 1.0],
        )?;

        let i_path = create_parquet_in(tmp.path(), "interactions.parquet", &mut interactions)?;
        let u_path = create_parquet_in(tmp.path(), "user_features.parquet", &mut user_features)?;
        let t_path = create_parquet_in(tmp.path(), "item_features.parquet", &mut item_features)?;

        Ok((i_path, u_path, t_path, tmp))
    }

    #[test]
    fn test_param_grid_cartesian_product() {
        let grid = ParamGrid {
            alpha: vec![0.5, 1.0],
            beta: vec![0.5, 1.0],
            lambda_: vec![10.0, 100.0],
            meta_weight: vec![0.0],
            decay_rate: vec![0.0],
            ips_alpha: vec![0.0],
            sparsity_threshold: vec![0.0],
        };

        let combos = cartesian_product(&grid);
        // 2 * 2 * 2 * 1 * 1 * 1 * 1 = 8
        assert_eq!(combos.len(), 8);

        // Verify all values appear
        let alphas: ahash::AHashSet<u64> = combos.iter().map(|p| p.alpha.to_bits()).collect();
        assert!(alphas.contains(&0.5_f64.to_bits()));
        assert!(alphas.contains(&1.0_f64.to_bits()));
    }

    #[test]
    fn test_kfold_split_coverage() -> Result<()> {
        let (i_path, _, _, _tmpdir) = create_test_dataset()?;

        let (_tmp_fold_dir, fold_paths) = generate_kfold_splits(&i_path, 3, 42)?;
        assert_eq!(fold_paths.len(), 3);

        // Collect all test user_ids across folds; each user should appear exactly once
        let mut all_test_users: Vec<String> = Vec::new();
        for (_train_path, test_path) in &fold_paths {
            let test_df = read_interactions_df(test_path)?;
            let user_col = test_df.column("user_id")?.str()?;
            let users: ahash::AHashSet<String> = user_col
                .into_iter()
                .flatten()
                .map(|s| s.to_string())
                .collect();
            all_test_users.extend(users);
        }

        // Sort for comparison
        all_test_users.sort();
        all_test_users.dedup();
        assert_eq!(
            all_test_users.len(),
            6,
            "All 6 users should appear in test exactly once across folds"
        );

        // Each fold's train + test should cover all interactions
        for (train_path, test_path) in &fold_paths {
            let train_df = read_interactions_df(train_path)?;
            let test_df = read_interactions_df(test_path)?;
            let total = train_df.height() + test_df.height();
            assert_eq!(total, 12, "train + test should equal total interactions");
        }

        Ok(())
    }

    #[test]
    fn test_ndcg_at_k_basic() {
        // Perfect ranking
        let rec = vec![0, 1, 2];
        let rel: ahash::AHashSet<usize> = [0, 1, 2].into_iter().collect();
        let score = ndcg_at_k(&rec, &rel, 3);
        assert!(
            (score - 1.0).abs() < 1e-10,
            "Perfect ranking should give NDCG=1.0, got {}",
            score
        );

        // No relevant items recommended
        let rec2 = vec![3, 4, 5];
        let score2 = ndcg_at_k(&rec2, &rel, 3);
        assert!(
            score2.abs() < 1e-10,
            "No relevant items should give NDCG=0.0"
        );

        // Empty relevant set
        let empty_rel: ahash::AHashSet<usize> = ahash::AHashSet::new();
        let score3 = ndcg_at_k(&rec, &empty_rel, 3);
        assert!(score3.abs() < 1e-10, "Empty relevant should give NDCG=0.0");
    }

    #[test]
    fn test_grid_search_finds_best() -> Result<()> {
        let (i_path, u_path, t_path, _tmpdir) = create_test_dataset()?;

        let grid = ParamGrid {
            alpha: vec![1.0],
            beta: vec![1.0],
            lambda_: vec![10.0, 500.0],
            meta_weight: vec![0.0],
            decay_rate: vec![0.0],
            ips_alpha: vec![0.0],
            sparsity_threshold: vec![0.0],
        };

        let result = grid_search(&i_path, &u_path, &t_path, &grid, 2, 10, 42)?;

        // Should have exactly 2 trials
        assert_eq!(result.all_trials.len(), 2);
        assert_eq!(result.metric_name, "ndcg@10");

        // The best score should equal one of the trial scores
        let trial_scores: Vec<f64> = result.all_trials.iter().map(|t| t.mean_score).collect();
        assert!(
            trial_scores.contains(&result.best_score),
            "Best score {} should be one of the trial scores {:?}",
            result.best_score,
            trial_scores
        );

        // The best score should be the max
        let max_score = trial_scores
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (result.best_score - max_score).abs() < 1e-10,
            "Best score should be the maximum across trials"
        );

        Ok(())
    }

    #[test]
    fn test_random_search_correct_n_trials() -> Result<()> {
        let (i_path, u_path, t_path, _tmpdir) = create_test_dataset()?;

        let grid = ParamGrid {
            alpha: vec![0.5, 1.0, 2.0],
            beta: vec![0.5, 1.0],
            lambda_: vec![10.0, 50.0, 100.0],
            meta_weight: vec![0.0],
            decay_rate: vec![0.0],
            ips_alpha: vec![0.0],
            sparsity_threshold: vec![0.0],
        };

        let n_trials = 3;
        let result = random_search(&i_path, &u_path, &t_path, &grid, n_trials, 2, 10, 42)?;

        assert_eq!(result.all_trials.len(), n_trials);
        assert_eq!(result.metric_name, "ndcg@10");

        // Each trial should have 2 fold scores
        for trial in &result.all_trials {
            assert_eq!(trial.fold_scores.len(), 2);
        }

        Ok(())
    }

    /// Legacy EASE fold evaluation: trains a `RustFeaseModel` and scores
    /// test users through the *concrete f64* `predict` path, exactly as
    /// `evaluate_trial` did before the `FoldEvaluator`/`RecModel`
    /// generalization (issue #39). No `EaseAdapter`, no `as f32` cast.
    /// This is the byte-for-byte reference the generalized EASE search
    /// must reproduce within float-cast tolerance.
    fn legacy_concrete_evaluate_trial(
        train_interactions_path: &str,
        test_interactions_path: &str,
        user_features_path: &str,
        item_features_path: &str,
        params: &HyperParams,
        eval_k: usize,
    ) -> Result<f64> {
        let weighting =
            if params.decay_rate > 0.0 || params.ips_alpha > 0.0 || params.sparsity_threshold > 0.0
            {
                Some(WeightingConfig {
                    event_weights: None,
                    decay_rate: params.decay_rate,
                    ips_alpha: params.ips_alpha,
                    sparsity_threshold: params.sparsity_threshold,
                })
            } else {
                None
            };

        let (x_mat, u_mat, t_mat, mappings) = data_pipeline::build_matrices(
            train_interactions_path,
            user_features_path,
            item_features_path,
            weighting.as_ref(),
        )?;

        let mut model = RustFeaseModel::new(
            x_mat.cols(),
            u_mat.cols(),
            t_mat.rows(),
            params.alpha,
            params.beta,
            params.lambda_,
            params.meta_weight,
            mappings,
        );
        model.train(&x_mat, &u_mat, &t_mat)?;
        if params.sparsity_threshold > 0.0 {
            model.prune_sparse(params.sparsity_threshold);
        }

        let train_df = read_interactions_df(train_interactions_path)?;
        let train_user_items = group_user_items(&train_df, &model.mappings)?;
        let test_df = read_interactions_df(test_interactions_path)?;
        let test_user_items = group_user_items(&test_df, &model.mappings)?;
        let user_features_map = build_user_features_map(user_features_path, &model.mappings)?;

        let mut ndcg_sum = 0.0;
        let mut n_users = 0;
        for (user_id, test_items) in &test_user_items {
            if test_items.is_empty() {
                continue;
            }
            let train_items: Vec<(usize, f64)> =
                train_user_items.get(user_id).cloned().unwrap_or_default();
            let user_feats: Vec<(usize, f64)> =
                user_features_map.get(user_id).cloned().unwrap_or_default();
            // Concrete f64 path -- no RecModel, no f32 cast.
            let scores = model.predict(&train_items, &user_feats, params.beta);
            let train_item_set: ahash::AHashSet<usize> =
                train_items.iter().map(|(idx, _)| *idx).collect();
            let mut ranked: Vec<(usize, f64)> = scores
                .into_iter()
                .enumerate()
                .filter(|(idx, _)| !train_item_set.contains(idx))
                .collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let recommended: Vec<usize> = ranked.iter().map(|(idx, _)| *idx).collect();
            let relevant: ahash::AHashSet<usize> = test_items.iter().map(|(idx, _)| *idx).collect();
            ndcg_sum += ndcg_at_k(&recommended, &relevant, eval_k);
            n_users += 1;
        }
        if n_users == 0 {
            return Ok(0.0);
        }
        Ok(ndcg_sum / n_users as f64)
    }

    /// Regression guard (issue #39): generalizing EASE search over the
    /// `FoldEvaluator`/`RecModel` traits must not change EASE results.
    /// Compares `grid_search` (now routed through `EaseFoldEvaluator` ->
    /// `EaseAdapter`) against the legacy concrete f64 evaluation for a
    /// fixed seed and grid. The only permitted difference is the single
    /// `as f32` score cast the adapter introduces; tolerance is 1e-9,
    /// far below any value that could flip `best_params` (same bound the
    /// Phase 4a eval baseline guard uses).
    #[test]
    fn test_ease_search_matches_legacy_concrete() -> Result<()> {
        let (i_path, u_path, t_path, _tmpdir) = create_test_dataset()?;

        let grid = ParamGrid {
            alpha: vec![0.5, 1.0],
            beta: vec![1.0],
            lambda_: vec![10.0, 100.0, 500.0],
            meta_weight: vec![0.0],
            decay_rate: vec![0.0],
            ips_alpha: vec![0.0],
            sparsity_threshold: vec![0.0],
        };
        let n_folds = 2;
        let eval_k = 10;
        let seed = 42;

        let generalized = grid_search(&i_path, &u_path, &t_path, &grid, n_folds, eval_k, seed)?;

        // Legacy concrete baseline: same fold splits, same combos, same
        // sequential order, but the pre-#39 direct f64 scoring path.
        let combos = cartesian_product(&grid);
        let (_tmp_dir, fold_paths) = generate_kfold_splits(&i_path, n_folds, seed)?;
        let mut legacy_trials = Vec::with_capacity(combos.len());
        let mut legacy_best_score = f64::NEG_INFINITY;
        let mut legacy_best = combos[0].clone();
        for params in &combos {
            let mut fold_scores = Vec::with_capacity(n_folds);
            for (tr, te) in &fold_paths {
                fold_scores.push(legacy_concrete_evaluate_trial(
                    tr, te, &u_path, &t_path, params, eval_k,
                )?);
            }
            let mean = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
            if mean > legacy_best_score {
                legacy_best_score = mean;
                legacy_best = params.clone();
            }
            legacy_trials.push((params.clone(), mean, fold_scores));
        }

        const TOL: f64 = 1e-9;
        // Decision output must be identical.
        assert_eq!(generalized.best_params.alpha, legacy_best.alpha);
        assert_eq!(generalized.best_params.beta, legacy_best.beta);
        assert_eq!(generalized.best_params.lambda_, legacy_best.lambda_);
        assert!(
            (generalized.best_score - legacy_best_score).abs() < TOL,
            "generalized best_score {} vs legacy concrete {}",
            generalized.best_score,
            legacy_best_score
        );
        // Per-trial scores must match the legacy concrete path within the
        // f32-cast tolerance, in cartesian-product order.
        assert_eq!(generalized.all_trials.len(), legacy_trials.len());
        for (idx, (g, (_, l_mean, l_folds))) in generalized
            .all_trials
            .iter()
            .zip(legacy_trials.iter())
            .enumerate()
        {
            assert!(
                (g.mean_score - l_mean).abs() < TOL,
                "trial {} mean_score: generalized {} vs legacy {}",
                idx,
                g.mean_score,
                l_mean
            );
            for (f, (gf, lf)) in g.fold_scores.iter().zip(l_folds.iter()).enumerate() {
                assert!(
                    (gf - lf).abs() < TOL,
                    "trial {} fold {} score: generalized {} vs legacy {}",
                    idx,
                    f,
                    gf,
                    lf
                );
            }
        }

        Ok(())
    }

    /// Sequential baseline mirroring the pre-parallel `for params { for fold }`
    /// evaluation order. Used to assert the rayon `grid_search` produces a
    /// bit-identical `SearchResult` (ADR-0002 Phase 1 acceptance gate).
    fn sequential_grid_baseline(
        i_path: &str,
        u_path: &str,
        t_path: &str,
        grid: &ParamGrid,
        n_folds: usize,
        eval_k: usize,
        seed: u64,
    ) -> Result<SearchResult> {
        let combos = cartesian_product(grid);
        let (_tmp_dir, fold_paths) = generate_kfold_splits(i_path, n_folds, seed)?;

        let mut all_trials = Vec::with_capacity(combos.len());
        let mut best_score = f64::NEG_INFINITY;
        let mut best_params = combos[0].clone();

        for params in &combos {
            let mut fold_scores = Vec::with_capacity(n_folds);
            for (train_path, test_path) in &fold_paths {
                fold_scores.push(evaluate_trial(
                    train_path, test_path, u_path, t_path, params, eval_k,
                )?);
            }
            let mean_score = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
            if mean_score > best_score {
                best_score = mean_score;
                best_params = params.clone();
            }
            all_trials.push(TrialResult {
                params: params.clone(),
                mean_score,
                fold_scores,
            });
        }

        Ok(SearchResult {
            best_params,
            best_score,
            all_trials,
            metric_name: format!("ndcg@{}", eval_k),
            strategy: "grid".to_string(),
        })
    }

    /// Regression test (ADR-0002 Phase 1 / issue #28):
    /// the parallel `grid_search` must produce identical `best_params`,
    /// `best_score`, and per-trial scores as the sequential baseline for a
    /// fixed seed and small grid, and `all_trials` must be in `trial_idx`
    /// (cartesian-product) order.
    #[test]
    fn test_parallel_grid_search_matches_sequential() -> Result<()> {
        let (i_path, u_path, t_path, _tmpdir) = create_test_dataset()?;

        // Small grid with multiple varying axes -> several distinct trials.
        let grid = ParamGrid {
            alpha: vec![0.5, 1.0],
            beta: vec![1.0],
            lambda_: vec![10.0, 100.0, 500.0],
            meta_weight: vec![0.0],
            decay_rate: vec![0.0],
            ips_alpha: vec![0.0],
            sparsity_threshold: vec![0.0],
        };
        let n_folds = 2;
        let eval_k = 10;
        let seed = 42;

        let parallel = grid_search(&i_path, &u_path, &t_path, &grid, n_folds, eval_k, seed)?;
        let sequential =
            sequential_grid_baseline(&i_path, &u_path, &t_path, &grid, n_folds, eval_k, seed)?;

        // Same number of trials and same metric name.
        assert_eq!(parallel.all_trials.len(), sequential.all_trials.len());
        assert_eq!(parallel.metric_name, sequential.metric_name);

        // The decision output -- best_params -- must be identical. This is
        // the determinism guarantee that matters: which configuration the
        // search picks must not depend on thread count or completion order
        // (ADR-0002 §Negative "Determinism under parallel tuning").
        assert_eq!(parallel.best_params.alpha, sequential.best_params.alpha);
        assert_eq!(parallel.best_params.beta, sequential.best_params.beta);
        assert_eq!(parallel.best_params.lambda_, sequential.best_params.lambda_);

        // best_score within tight tolerance. Scores are NOT pinned bit-exact:
        // the closed-form solve runs through nalgebra's rayon-enabled dense
        // LA, so sub-ULP float drift from FMA ordering is expected and
        // rank-irrelevant (ADR-0002 §Risks). Tolerance is far below any
        // value that could flip best_params.
        const TOL: f64 = 1e-9;
        assert!(
            (parallel.best_score - sequential.best_score).abs() < TOL,
            "parallel best_score {} vs sequential {}",
            parallel.best_score,
            sequential.best_score
        );

        // all_trials must be returned in cartesian-product (trial_idx) order
        // -- this ordering IS pinned exactly, it's the core determinism
        // guarantee of the parallel runner -- and per-trial scores must match
        // the sequential baseline within tolerance.
        let expected_combos = cartesian_product(&grid);
        for (idx, (p_trial, s_trial)) in parallel
            .all_trials
            .iter()
            .zip(sequential.all_trials.iter())
            .enumerate()
        {
            assert_eq!(
                p_trial.params.alpha, expected_combos[idx].alpha,
                "all_trials[{}] not in trial_idx order (alpha)",
                idx
            );
            assert_eq!(
                p_trial.params.lambda_, expected_combos[idx].lambda_,
                "all_trials[{}] not in trial_idx order (lambda_)",
                idx
            );
            assert!(
                (p_trial.mean_score - s_trial.mean_score).abs() < TOL,
                "trial {} mean_score mismatch: parallel {} vs sequential {}",
                idx,
                p_trial.mean_score,
                s_trial.mean_score
            );
            assert_eq!(
                p_trial.fold_scores.len(),
                s_trial.fold_scores.len(),
                "trial {} fold count mismatch",
                idx
            );
            for (f, (pf, sf)) in p_trial
                .fold_scores
                .iter()
                .zip(s_trial.fold_scores.iter())
                .enumerate()
            {
                assert!(
                    (pf - sf).abs() < TOL,
                    "trial {} fold {} score mismatch: parallel {} vs sequential {}",
                    idx,
                    f,
                    pf,
                    sf
                );
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // SASRec / Two-Tower end-to-end search (ml-models)
    // -----------------------------------------------------------------------

    /// A small interactions-only dataset with a `days_ago` column so the
    /// SASRec sequence builder can order each user's history. `n_users`
    /// users each with `per_user` chronologically-spaced interactions over
    /// `n_items` items; every user has >= 2 in-catalog items so no user is
    /// dropped from a fold's train split.
    #[cfg(feature = "ml-models")]
    fn make_seq_dataset(
        n_users: usize,
        n_items: usize,
        per_user: usize,
    ) -> Result<(String, tempfile::TempDir)> {
        let tmp = tempfile::tempdir()?;
        let mut u = Vec::new();
        let mut it = Vec::new();
        let mut v = Vec::new();
        let mut days = Vec::new();
        for ui in 0..n_users {
            for k in 0..per_user {
                u.push(format!("u{}", ui));
                it.push(format!("i{}", (ui + k) % n_items));
                v.push(1.0_f64);
                // Larger days_ago == older; strictly decreasing per user.
                days.push((per_user - k) as f64);
            }
        }
        let mut df = df!(
            "user_id" => u,
            "item_id" => it,
            "value" => v,
            "days_ago" => days,
        )?;
        let path = create_parquet_in(tmp.path(), "seq.parquet", &mut df)?;
        Ok((path, tmp))
    }

    /// SASRec hyperparameter search runs end-to-end: a small real grid +
    /// k-fold CV trains the burn model per (params × fold) through
    /// `SasRecFoldEvaluator` and produces best params/score with the EASE
    /// result shape. Exercises the genuine architecture schema
    /// (embedding_dim / num_heads / num_layers / dropout / learning_rate /
    /// num_epochs), not a placeholder.
    #[cfg(feature = "ml-models")]
    #[test]
    #[ignore = "trains real burn models; impractically slow in a debug CI build. Run with: cargo test --release --features ml-models -- --ignored"]
    fn test_sasrec_grid_search_end_to_end() -> Result<()> {
        let (i_path, _tmp) = make_seq_dataset(12, 6, 5)?;

        let grid = SasRecParamGrid {
            embedding_dim: vec![8, 16],
            num_heads: vec![2],
            num_layers: vec![1],
            dropout: vec![0.0],
            learning_rate: vec![1e-2],
            num_epochs: vec![3],
        };
        let evaluator = SasRecFoldEvaluator {
            max_seq_len: 8,
            batch_size: 8,
            patience: 3,
            seed: 7,
        };

        let result = grid_search_with(&evaluator, &i_path, &grid, 2, 5, 42)?;

        // 2 embedding-dim values × 1 each other axis = 2 trials.
        assert_eq!(result.all_trials.len(), 2);
        assert_eq!(result.metric_name, "ndcg@5");
        for trial in &result.all_trials {
            assert_eq!(trial.fold_scores.len(), 2, "2-fold CV");
            assert!(trial.mean_score.is_finite());
            assert!((0.0..=1.0).contains(&trial.mean_score));
        }
        // best_score is the max over trials and one of the configs ran.
        let max = result
            .all_trials
            .iter()
            .map(|t| t.mean_score)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((result.best_score - max).abs() < 1e-12);
        assert!([8usize, 16].contains(&result.best_params.embedding_dim));
        Ok(())
    }

    /// SASRec random search draws `n_trials` configs from the schema and
    /// runs each through the real evaluator end-to-end.
    #[cfg(feature = "ml-models")]
    #[test]
    #[ignore = "trains real burn models; impractically slow in a debug CI build. Run with: cargo test --release --features ml-models -- --ignored"]
    fn test_sasrec_random_search_end_to_end() -> Result<()> {
        let (i_path, _tmp) = make_seq_dataset(12, 6, 5)?;

        let grid = SasRecParamGrid {
            embedding_dim: vec![8, 16],
            num_heads: vec![2],
            num_layers: vec![1, 2],
            dropout: vec![0.0, 0.1],
            learning_rate: vec![1e-2],
            num_epochs: vec![3],
        };
        let evaluator = SasRecFoldEvaluator {
            max_seq_len: 8,
            batch_size: 8,
            patience: 3,
            seed: 1,
        };

        let n_trials = 3;
        let result = random_search_with(&evaluator, &i_path, &grid, n_trials, 2, 5, 42)?;
        assert_eq!(result.all_trials.len(), n_trials);
        assert_eq!(result.metric_name, "ndcg@5");
        for trial in &result.all_trials {
            assert_eq!(trial.fold_scores.len(), 2);
            assert!(trial.mean_score.is_finite());
        }
        Ok(())
    }

    /// Two-Tower hyperparameter search runs end-to-end: a small real grid
    /// + k-fold CV trains the in-batch sampled-softmax model per
    /// (params × fold) through `TwoTowerFoldEvaluator` and produces best
    /// params/score with the EASE result shape. Exercises the genuine
    /// schema (embedding_dim / temperature / learning_rate / id_dropout).
    #[cfg(feature = "ml-models")]
    #[test]
    #[ignore = "trains real burn models; impractically slow in a debug CI build. Run with: cargo test --release --features ml-models -- --ignored"]
    fn test_two_tower_grid_search_end_to_end() -> Result<()> {
        // Two-Tower doesn't need `days_ago`; reuse the seq dataset (extra
        // column is ignored by the triple loader).
        let (i_path, _tmp) = make_seq_dataset(12, 6, 5)?;

        let grid = TwoTowerParamGrid {
            embedding_dim: vec![8, 16],
            temperature: vec![0.05],
            learning_rate: vec![0.05],
            id_dropout: vec![0.0],
        };
        let evaluator = TwoTowerFoldEvaluator {
            epochs: 10,
            batch_size: 16,
            seed: 0,
        };

        let result = grid_search_with(&evaluator, &i_path, &grid, 2, 5, 42)?;

        assert_eq!(result.all_trials.len(), 2);
        assert_eq!(result.metric_name, "ndcg@5");
        for trial in &result.all_trials {
            assert_eq!(trial.fold_scores.len(), 2);
            assert!(trial.mean_score.is_finite());
            assert!((0.0..=1.0).contains(&trial.mean_score));
        }
        let max = result
            .all_trials
            .iter()
            .map(|t| t.mean_score)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((result.best_score - max).abs() < 1e-12);
        assert!([8usize, 16].contains(&result.best_params.embedding_dim));
        Ok(())
    }

    /// Two-Tower random search draws `n_trials` configs from the schema
    /// and runs each through the real evaluator end-to-end.
    #[cfg(feature = "ml-models")]
    #[test]
    #[ignore = "trains real burn models; impractically slow in a debug CI build. Run with: cargo test --release --features ml-models -- --ignored"]
    fn test_two_tower_random_search_end_to_end() -> Result<()> {
        let (i_path, _tmp) = make_seq_dataset(12, 6, 5)?;

        let grid = TwoTowerParamGrid {
            embedding_dim: vec![8, 16],
            temperature: vec![0.05, 0.1],
            learning_rate: vec![0.05],
            id_dropout: vec![0.0, 0.2],
        };
        let evaluator = TwoTowerFoldEvaluator {
            epochs: 10,
            batch_size: 16,
            seed: 0,
        };

        let n_trials = 3;
        let result = random_search_with(&evaluator, &i_path, &grid, n_trials, 2, 5, 42)?;
        assert_eq!(result.all_trials.len(), n_trials);
        assert_eq!(result.metric_name, "ndcg@5");
        for trial in &result.all_trials {
            assert_eq!(trial.fold_scores.len(), 2);
            assert!(trial.mean_score.is_finite());
        }
        Ok(())
    }

    // -----------------------------------------------------------------
    // #97: strategy seam regression guards + adaptive search
    // -----------------------------------------------------------------

    /// `ParamGrid::combinations()` (now derived from the generic axes)
    /// must reproduce the pre-#97 hand-written cartesian product exactly:
    /// same count, same order, same bits.
    #[test]
    fn test_grid_combinations_match_legacy_cartesian_product_bitwise() -> Result<()> {
        let grid = ParamGrid {
            alpha: vec![0.5, 1.0],
            beta: vec![0.25, 1.0],
            lambda_: vec![10.0, 100.0, 500.0],
            meta_weight: vec![0.0, 0.5],
            decay_rate: vec![0.0],
            ips_alpha: vec![0.0, 0.1],
            sparsity_threshold: vec![0.0],
        };
        let legacy = cartesian_product(&grid);
        let now = grid.combinations()?;
        assert_eq!(legacy.len(), now.len());
        assert_eq!(now.len(), 48);
        for (a, b) in legacy.iter().zip(&now) {
            assert_eq!(EaseSchema::encode(a), EaseSchema::encode(b));
        }
        Ok(())
    }

    /// Pre-#97 random sampling: per-field `choose` in field order from an
    /// RNG seeded with `seed + 1`.
    fn legacy_random_sample(grid: &ParamGrid, n: usize, seed: u64) -> Vec<HyperParams> {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1));
        (0..n)
            .map(|_| HyperParams {
                alpha: *grid.alpha.choose(&mut rng).unwrap(),
                beta: *grid.beta.choose(&mut rng).unwrap(),
                lambda_: *grid.lambda_.choose(&mut rng).unwrap(),
                meta_weight: *grid.meta_weight.choose(&mut rng).unwrap(),
                decay_rate: *grid.decay_rate.choose(&mut rng).unwrap(),
                ips_alpha: *grid.ips_alpha.choose(&mut rng).unwrap(),
                sparsity_threshold: *grid.sparsity_threshold.choose(&mut rng).unwrap(),
            })
            .collect()
    }

    #[test]
    fn test_random_search_params_match_legacy_sampling_bitwise() -> Result<()> {
        let (i_path, u_path, t_path, _tmpdir) = create_test_dataset()?;
        let grid = ParamGrid {
            alpha: vec![0.5, 1.0, 2.0],
            beta: vec![0.5, 1.0],
            lambda_: vec![10.0, 50.0, 100.0],
            meta_weight: vec![0.0],
            decay_rate: vec![0.0],
            ips_alpha: vec![0.0],
            sparsity_threshold: vec![0.0],
        };
        let (n, seed) = (5, 42);
        let result = random_search(&i_path, &u_path, &t_path, &grid, n, 2, 10, seed)?;
        let legacy = legacy_random_sample(&grid, n, seed);
        assert_eq!(result.strategy, "random");
        assert_eq!(result.all_trials.len(), n);
        for (t, l) in result.all_trials.iter().zip(&legacy) {
            assert_eq!(EaseSchema::encode(&t.params), EaseSchema::encode(l));
        }
        Ok(())
    }

    #[test]
    fn test_dyn_space_orders_defaults_and_rejects_unknown() -> Result<()> {
        let s = EaseSpace::from_axes(vec![
            AxisSpec::new(
                "lambda_",
                Axis::LogUniform {
                    low: 1.0,
                    high: 1000.0,
                },
            ),
            AxisSpec::new("alpha", Axis::choice_f64(&[0.5, 2.0])),
        ])?;
        assert_eq!(s.space().names(), EaseSchema::NAMES);
        let mut rng = StdRng::seed_from_u64(0);
        let p = s.sample_one(&mut rng);
        assert!((1.0..=1000.0).contains(&p.lambda_));
        assert!([0.5, 2.0].contains(&p.alpha));
        assert_eq!(p.beta, 1.0, "omitted axes take the schema default");
        assert!(
            s.combinations().is_err(),
            "grid over a continuous axis is refused"
        );
        assert_eq!(s.encode(&p).len(), 7);
        assert_eq!(EaseSchema::encode(&s.decode(&s.encode(&p))?), s.encode(&p));

        assert!(
            EaseSpace::from_axes(vec![AxisSpec::new("lambda", Axis::choice_f64(&[1.0]))]).is_err()
        );
        assert!(
            SasRecSpace::from_axes(vec![AxisSpec::new(
                "embedding_dim",
                Axis::LogInt { low: 8, high: 128 }
            )])
            .is_ok()
        );
        assert!(
            Bert4RecSpace::from_axes(vec![AxisSpec::new("mask_ratio", Axis::choice_f64(&[0.2]))])
                .is_ok()
        );
        // Integer axes reject non-integral values at decode time.
        let bad = Bert4RecSpace::from_axes(vec![])?;
        let mut a = bad.encode(&Bert4RecParams {
            embedding_dim: 64,
            num_heads: 4,
            num_layers: 2,
            dropout: 0.1,
            mask_ratio: 0.2,
            learning_rate: 1e-3,
            num_epochs: 50,
        });
        a[0] = Value::Float(64.5);
        assert!(bad.decode(&a).is_err());
        Ok(())
    }

    #[test]
    fn test_tune_with_grid_truncates_to_max_trials() -> Result<()> {
        let (i_path, u_path, t_path, _tmpdir) = create_test_dataset()?;
        let evaluator = EaseFoldEvaluator {
            user_features_path: u_path,
            item_features_path: t_path,
        };
        let grid = ParamGrid {
            alpha: vec![0.5, 1.0],
            beta: vec![1.0],
            lambda_: vec![10.0, 100.0, 500.0],
            meta_weight: vec![0.0],
            decay_rate: vec![0.0],
            ips_alpha: vec![0.0],
            sparsity_threshold: vec![0.0],
        };
        let cfg = TuneConfig {
            n_folds: 2,
            eval_k: 10,
            seed: 42,
            max_trials: 2,
        };
        let r = tune_with(
            &evaluator,
            &i_path,
            &grid,
            &mut GridStrategy::new(),
            &cfg,
            &[],
        )?;
        assert_eq!(r.strategy, "grid");
        assert_eq!(r.all_trials.len(), 2);
        let combos = grid.combinations()?;
        for (t, c) in r.all_trials.iter().zip(&combos) {
            assert_eq!(EaseSchema::encode(&t.params), EaseSchema::encode(c));
        }
        assert!(
            tune_with(
                &evaluator,
                &i_path,
                &grid,
                &mut GridStrategy::new(),
                &TuneConfig {
                    max_trials: 0,
                    ..cfg
                },
                &[]
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn test_tune_with_tpe_ease_end_to_end_deterministic_and_warm_start() -> Result<()> {
        let (i_path, u_path, t_path, _tmpdir) = create_test_dataset()?;
        let evaluator = EaseFoldEvaluator {
            user_features_path: u_path,
            item_features_path: t_path,
        };
        let space = EaseSpace::from_axes(vec![
            AxisSpec::new(
                "lambda_",
                Axis::LogUniform {
                    low: 1.0,
                    high: 1000.0,
                },
            ),
            AxisSpec::new("alpha", Axis::choice_f64(&[0.5, 1.0, 2.0])),
        ])?;
        let cfg = TuneConfig {
            n_folds: 2,
            eval_k: 10,
            seed: 42,
            max_trials: 6,
        };
        let run = || {
            let mut tpe = TpeStrategy::new().with_n_startup(2).with_batch_size(2);
            tune_with(&evaluator, &i_path, &space, &mut tpe, &cfg, &[])
        };
        let r1 = run()?;
        assert_eq!(r1.strategy, "tpe");
        assert_eq!(r1.metric_name, "ndcg@10");
        assert_eq!(r1.all_trials.len(), 6);
        for t in &r1.all_trials {
            assert!((1.0..=1000.0).contains(&t.params.lambda_));
            assert!([0.5, 1.0, 2.0].contains(&t.params.alpha));
            assert_eq!(t.fold_scores.len(), 2);
            assert!(t.mean_score.is_finite());
        }
        let max = r1
            .all_trials
            .iter()
            .map(|t| t.mean_score)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((r1.best_score - max).abs() < 1e-12);

        // Deterministic for a fixed seed.
        let r2 = run()?;
        for (a, b) in r1.all_trials.iter().zip(&r2.all_trials) {
            assert_eq!(EaseSchema::encode(&a.params), EaseSchema::encode(&b.params));
            assert_eq!(a.mean_score.to_bits(), b.mean_score.to_bits());
        }

        // Warm start: prior trials seed the model; only new trials come back
        // and, with the start-up budget already covered by history, the
        // model path runs from the first proposal.
        let cfg3 = TuneConfig {
            max_trials: 3,
            ..cfg
        };
        let mut tpe3 = TpeStrategy::new().with_n_startup(2).with_batch_size(2);
        let r3 = tune_with(
            &evaluator,
            &i_path,
            &space,
            &mut tpe3,
            &cfg3,
            &r1.all_trials,
        )?;
        assert_eq!(r3.all_trials.len(), 3);
        Ok(())
    }

    /// BERT4Rec grid + random search run end-to-end through the real
    /// evaluator (#97 Phase 1).
    #[cfg(feature = "ml-models")]
    #[test]
    #[ignore = "trains real burn models; impractically slow in a debug CI build. Run with: cargo test --release --features ml-models -- --ignored"]
    fn test_bert4rec_grid_and_random_search_end_to_end() -> Result<()> {
        let (i_path, _tmp) = make_seq_dataset(12, 6, 5)?;
        let grid = Bert4RecParamGrid {
            embedding_dim: vec![8, 16],
            num_heads: vec![2],
            num_layers: vec![1],
            dropout: vec![0.0],
            mask_ratio: vec![0.3],
            learning_rate: vec![1e-2],
            num_epochs: vec![5],
        };
        let evaluator = Bert4RecFoldEvaluator {
            max_seq_len: 8,
            num_position_buckets: 8,
            batch_size: 8,
            patience: 5,
            seed: 42,
        };
        let result = grid_search_with(&evaluator, &i_path, &grid, 2, 5, 42)?;
        assert_eq!(result.all_trials.len(), 2);
        assert_eq!(result.metric_name, "ndcg@5");
        for trial in &result.all_trials {
            assert_eq!(trial.fold_scores.len(), 2);
            assert!(trial.mean_score.is_finite());
            assert!((0.0..=1.0).contains(&trial.mean_score));
        }
        assert!([8usize, 16].contains(&result.best_params.embedding_dim));

        let rr = random_search_with(&evaluator, &i_path, &grid, 3, 2, 5, 42)?;
        assert_eq!(rr.all_trials.len(), 3);
        assert_eq!(rr.strategy, "random");
        Ok(())
    }

    /// #97 acceptance: on the Two-Tower fixture, TPE with half the budget
    /// matches or beats random search's best NDCG@k in a majority of seeds.
    #[cfg(feature = "ml-models")]
    #[test]
    #[ignore = "trains real burn models; impractically slow in a debug CI build. Run with: cargo test --release --features ml-models -- --ignored"]
    fn test_tpe_half_budget_matches_random_best_two_tower() -> Result<()> {
        let (i_path, _tmp) = make_seq_dataset(24, 8, 6)?;
        let evaluator = TwoTowerFoldEvaluator {
            epochs: 10,
            batch_size: 16,
            seed: 0,
        };
        let space = TwoTowerSpace::from_axes(vec![
            AxisSpec::new("embedding_dim", Axis::choice_usize(&[4, 8, 16, 32])),
            AxisSpec::new(
                "temperature",
                Axis::LogUniform {
                    low: 0.01,
                    high: 1.0,
                },
            ),
            AxisSpec::new(
                "learning_rate",
                Axis::LogUniform {
                    low: 1e-3,
                    high: 0.3,
                },
            ),
        ])?;
        let budget = 16;
        let mut wins = 0;
        let seeds = [1_u64, 2, 3];
        for &seed in &seeds {
            let cfg_r = TuneConfig {
                n_folds: 2,
                eval_k: 5,
                seed,
                max_trials: budget,
            };
            let random = tune_with(
                &evaluator,
                &i_path,
                &space,
                &mut RandomStrategy::new(budget),
                &cfg_r,
                &[],
            )?;
            let cfg_t = TuneConfig {
                max_trials: budget / 2,
                ..cfg_r
            };
            let tpe = tune_with(
                &evaluator,
                &i_path,
                &space,
                &mut TpeStrategy::for_budget(budget / 2).with_batch_size(2),
                &cfg_t,
                &[],
            )?;
            assert_eq!(tpe.all_trials.len(), budget / 2);
            if tpe.best_score >= random.best_score - 1e-9 {
                wins += 1;
            }
        }
        assert!(
            wins * 2 > seeds.len(),
            "TPE (half budget) should match/beat random's best in a majority of seeds; won {wins}/{}",
            seeds.len()
        );
        Ok(())
    }
}
