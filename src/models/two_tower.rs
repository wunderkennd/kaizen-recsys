//! Two-Tower — separate user / item embedding networks with categorical
//! and dense numerical features, trained by in-batch sampled softmax.
//!
//! ADR-0001 (issue #38).
//!
//! Architecture (Yi et al., RecSys 2019, "Sampling-Bias-Corrected Neural
//! Modeling for Large Corpus Item Recommendations"):
//!
//!   * A **user tower** and an **item tower**, each = id embedding +
//!     mean-pooled categorical-feature embedding + dense-feature linear,
//!     summed and pushed through a 2-layer MLP. Outputs are
//!     **L2-normalized** and the dot-product score is divided by a
//!     temperature `tau`.
//!   * Training loss is the **in-batch sampled softmax**
//!     `L = -1/B Σ_i log( exp(s_ii) / Σ_j exp(s_ij) )` with two required
//!     corrections (research §5):
//!       - **log-Q popularity correction**: `s_ij -= log p_j`, where
//!         `p_j` is item `j`'s in-batch occurrence frequency, and
//!       - **same-item false-negative masking**: `s_ij = -inf` for
//!         `j != i` whenever `item_id_j == item_id_i` (an in-batch
//!         duplicate of the positive is not a true negative).
//!
//! Cold-start: a user with no id is mapped to a dedicated, *learnable*
//! reserved embedding row (user index 0,
//! [`crate::data::triples::COLD_START_USER_IDX`]). Training applies
//! id-dropout — a fraction of rows have their user id replaced by the
//! reserved index — so that row converges to an average-user prior
//! instead of a hard zero. Such a user still also gets its
//! feature-embedding contribution, so a feature-rich cold-start user
//! blends the learned prior with its side info. The item tower has no
//! reserved row: scoring is always against the known catalog, so a
//! cold-start-item prior would never be exercised.
//!
//! Backend stays generic (`TwoTower<B: Backend>`); training uses
//! `Autodiff<NdArray>`, inference plain `NdArray`. Everything is
//! `ml-models`-gated.

use crate::data::triples::{FeatureTable, TripleData};
use crate::data_pipeline::Mappings;
use crate::model::ValidationReport;
use crate::models::{ModelInput, ModelKind, RecModel};
use ahash::AHashMap;
use anyhow::{Context, Result, anyhow};
use burn::config::Config;
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, Relu};
use burn::optim::AdamConfig;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor, TensorData};
use burn::train::{InferenceStep, TrainOutput, TrainStep};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// CPU inference backend (research §1: `NdArray<f32>` for inference).
type InfB = burn::backend::NdArray<f32>;
/// CPU autodiff training backend (research §1:
/// `Autodiff<NdArray<f32>>` for training).
type TrainB = burn::backend::Autodiff<InfB>;
/// Device handle for the CPU backend.
type Dev = burn::backend::ndarray::NdArrayDevice;

/// Magic bytes for serialized Two-Tower models (ADR-0001 §Decision #4).
const MAGIC: &[u8; 4] = b"FTWO";
/// Framed format version for Two-Tower files.
///
/// v5 persists the `feature_name → categorical-index` and
/// `feature_name → dense-column` maps for the user side (#55), enabling
/// `TwoTowerModel.predict(user_id, features=...)` to translate
/// predict-time string feature names into the integer indices the model
/// expects. v4 files have no such maps; loading a v4 file with v5 code
/// is rejected with a "retrain required" error because (a) we couldn't
/// know the original feature names and (b) silently mapping no features
/// at predict time would be a silent regression for callers that expect
/// the v5 surface to work.
///
/// v4 reserves user embedding row 0 as a learnable cold-start prior and
/// shifts real users to `1..=N`, growing the user id table by one row and
/// renumbering every user index. A v3 (or earlier) param blob is
/// structurally incompatible — its user table is one row short and its
/// indices are off by one, and the cold-start row it lacks can only be
/// obtained by *training* (id-dropout), not by any in-place migration.
/// `load_from` therefore rejects pre-v5 files loudly with a "retrain
/// required" error rather than silently mis-mapping users.
const FORMAT_VERSION: u32 = 5;

/// Construction-time hyperparameters for [`TwoTower`].
#[derive(Config, Debug)]
pub struct TwoTowerConfig {
    pub num_users: usize,
    pub num_items: usize,
    /// Size of the shared user-side categorical-feature embedding table.
    pub num_user_categories: usize,
    /// Size of the shared item-side categorical-feature embedding table.
    pub num_item_categories: usize,
    /// Dense user-feature vector width (0 if none).
    pub user_dense_dim: usize,
    /// Dense item-feature vector width (0 if none).
    pub item_dense_dim: usize,
    /// Embedding / hidden dimension shared by both towers.
    pub embedding_dim: usize,
    /// Softmax temperature; logits are divided by this. Lower = sharper.
    #[config(default = 0.05)]
    pub temperature: f64,
}

/// One tower (user *or* item): id embedding + categorical-feature
/// embedding (mean-pooled) + dense-feature linear, summed, then a 2-layer
/// MLP. A `num_categories == 0` / `dense_dim == 0` side keeps a degenerate
/// table/linear so the module shape is uniform; those paths contribute 0.
#[derive(Module, Debug)]
struct Tower<B: Backend> {
    id_embedding: Embedding<B>,
    cat_embedding: Embedding<B>,
    dense_proj: Linear<B>,
    hidden: Linear<B>,
    out: Linear<B>,
    activation: Relu,
    has_cat: bool,
    has_dense: bool,
}

impl<B: Backend> Tower<B> {
    fn new(
        num_ids: usize,
        num_categories: usize,
        dense_dim: usize,
        dim: usize,
        device: &B::Device,
    ) -> Self {
        let id_embedding = EmbeddingConfig::new(num_ids.max(1), dim).init(device);
        // Embedding tables must be non-empty; sized to max(1, n). The
        // `has_*` flags gate whether they actually contribute.
        let cat_embedding = EmbeddingConfig::new(num_categories.max(1), dim).init(device);
        let dense_proj = LinearConfig::new(dense_dim.max(1), dim).init(device);
        let hidden = LinearConfig::new(dim, dim).init(device);
        let out = LinearConfig::new(dim, dim).init(device);
        Self {
            id_embedding,
            cat_embedding,
            dense_proj,
            hidden,
            out,
            activation: Relu::new(),
            has_cat: num_categories > 0,
            has_dense: dense_dim > 0,
        }
    }

    /// Forward one batch.
    ///
    /// * `ids`           `(B)` — id index per row.
    /// * `cat_ids`       `(B, C)` — padded categorical-feature indices.
    /// * `cat_mask`      `(B, C)` — 1.0 where `cat_ids` is real, 0.0 pad.
    /// * `dense`         `(B, D)` — dense feature vectors.
    ///
    /// Returns the **L2-normalized** `(B, dim)` tower embedding.
    fn forward(
        &self,
        ids: Tensor<B, 1, Int>,
        cat_ids: Tensor<B, 2, Int>,
        cat_mask: Tensor<B, 2>,
        dense: Tensor<B, 2>,
        id_scale: f64,
    ) -> Tensor<B, 2> {
        let [batch, _] = cat_ids.dims();
        let dim = self.id_embedding.weight.val().dims()[1];

        // `id_scale` scales the id-embedding term. It is 1.0 on every
        // current path: warm users use their own row and cold-start users
        // use the reserved learnable row (index 0), so the id term is a
        // meaningful trained vector in both cases — no hard zero. The knob
        // is retained for ablations / callers that want a feature-only
        // vector (pass 0.0).
        let mut h = self
            .id_embedding
            .forward(ids.reshape([batch, 1]))
            .reshape([batch, dim])
            .mul_scalar(id_scale);

        if self.has_cat {
            // Mean-pool categorical embeddings over the non-pad slots.
            let cat_emb = self.cat_embedding.forward(cat_ids); // (B, C, dim)
            let mask3 = cat_mask.clone().reshape([batch, cat_emb.dims()[1], 1]);
            let summed = (cat_emb * mask3).sum_dim(1).reshape([batch, dim]);
            let counts = cat_mask.sum_dim(1).reshape([batch, 1]).clamp_min(1.0); // avoid divide-by-zero for no-feature rows
            h = h + summed / counts;
        }

        if self.has_dense {
            h = h + self.dense_proj.forward(dense);
        }

        let h = self.activation.forward(self.hidden.forward(h));
        let z = self.out.forward(h);

        // L2-normalize: z / ||z||_2 (eps-floored).
        let norm = z
            .clone()
            .powf_scalar(2.0)
            .sum_dim(1)
            .sqrt()
            .clamp_min(1e-12);
        z / norm
    }
}

/// The Two-Tower module.
#[derive(Module, Debug)]
pub struct TwoTower<B: Backend> {
    user_tower: Tower<B>,
    item_tower: Tower<B>,
    temperature: f64,
}

impl TwoTowerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TwoTower<B> {
        TwoTower {
            user_tower: Tower::new(
                self.num_users,
                self.num_user_categories,
                self.user_dense_dim,
                self.embedding_dim,
                device,
            ),
            item_tower: Tower::new(
                self.num_items,
                self.num_item_categories,
                self.item_dense_dim,
                self.embedding_dim,
                device,
            ),
            temperature: self.temperature,
        }
    }
}

/// One mini-batch of `(user, positive-item)` pairs with both sides'
/// features already padded to the batch's max categorical fan-out.
#[derive(Debug, Clone)]
pub struct TwoTowerBatch<B: Backend> {
    user_ids: Tensor<B, 1, Int>,
    user_cat: Tensor<B, 2, Int>,
    user_cat_mask: Tensor<B, 2>,
    user_dense: Tensor<B, 2>,
    item_ids: Tensor<B, 1, Int>,
    item_cat: Tensor<B, 2, Int>,
    item_cat_mask: Tensor<B, 2>,
    item_dense: Tensor<B, 2>,
    /// Raw item indices (length B) — used for same-item masking and the
    /// log-Q frequency estimate. Kept off-device as plain `usize`.
    item_idx: Vec<usize>,
    /// Raw (pre-dropout) user indices (length B). Kept off-device so the
    /// per-epoch id-dropout pass can resample `user_ids` cheaply without
    /// reading the device tensor back.
    user_idx: Vec<usize>,
}

impl<B: Backend> TwoTower<B> {
    /// Score every catalog item for a single user-side embedding.
    /// `user_vec` is `(1, dim)` (already normalized); `item_vecs` is
    /// `(num_items, dim)`. Returns a length-`num_items` score vector.
    fn score_all(&self, user_vec: Tensor<B, 2>, item_vecs: Tensor<B, 2>) -> Vec<f32> {
        let n = item_vecs.dims()[0];
        let dim = item_vecs.dims()[1];
        let scores = (item_vecs * user_vec.reshape([1, dim]))
            .sum_dim(1)
            .reshape([n]);
        let data = scores.into_data();
        data.iter::<f32>().collect()
    }

    /// In-batch sampled-softmax loss with log-Q correction and same-item
    /// false-negative masking (research §5). Returns the scalar loss.
    fn forward_loss(&self, batch: TwoTowerBatch<B>) -> Tensor<B, 1> {
        let device = batch.user_ids.device();
        let b = batch.item_idx.len();

        let u = self.user_tower.forward(
            batch.user_ids,
            batch.user_cat,
            batch.user_cat_mask,
            batch.user_dense,
            1.0,
        ); // (B, dim)
        let v = self.item_tower.forward(
            batch.item_ids,
            batch.item_cat,
            batch.item_cat_mask,
            batch.item_dense,
            1.0,
        ); // (B, dim)

        let dim = u.dims()[1];
        // Score matrix s_ij = <u_i, v_j> / tau   -> (B, B).
        let logits = u.matmul(v.transpose()) / self.temperature;

        // log-Q popularity correction: subtract log p_j, where p_j is the
        // in-batch occurrence frequency of item j's id. Broadcast over rows.
        let mut freq: AHashMap<usize, usize> = AHashMap::new();
        for &it in &batch.item_idx {
            *freq.entry(it).or_insert(0) += 1;
        }
        let log_q: Vec<f32> = batch
            .item_idx
            .iter()
            .map(|it| ((freq[it] as f32) / (b as f32)).ln())
            .collect();
        let log_q = Tensor::<B, 1>::from_data(TensorData::new(log_q, [b]), &device).reshape([1, b]);
        let logits = logits - log_q;

        // Same-item false-negative mask: for off-diagonal (i != j) where
        // item_idx[i] == item_idx[j], set s_ij = -inf so an in-batch
        // duplicate of the positive isn't counted as a negative.
        let mut mask_data = vec![0.0_f32; b * b];
        for i in 0..b {
            for j in 0..b {
                if i != j && batch.item_idx[i] == batch.item_idx[j] {
                    mask_data[i * b + j] = f32::NEG_INFINITY;
                }
            }
        }
        let neg_inf_mask = Tensor::<B, 2>::from_data(TensorData::new(mask_data, [b, b]), &device);
        let logits = logits + neg_inf_mask;

        // Cross-entropy with the diagonal (row i's positive is column i).
        let targets = Tensor::<B, 1, Int>::from_data(
            TensorData::new((0..b as i64).collect::<Vec<_>>(), [b]),
            &device,
        );
        let _ = dim;
        CrossEntropyLossConfig::new()
            .init(&device)
            .forward(logits, targets)
            .reshape([1])
    }
}

// burn 0.21's training abstractions (research §3; `ValidStep` is renamed
// `InferenceStep` in 0.21). The Two-Tower `train` entrypoint drives the
// optimizer through `TrainStep::step` directly — no `LearnerBuilder`
// artifact dir is needed for the tiny datasets this phase targets, and
// the scalar loss is consumed in the loop rather than by `LossMetric`,
// so `Output = ()` (which `burn` implements `ItemLazy` for).
impl<B: AutodiffBackend> TrainStep for TwoTower<B> {
    type Input = TwoTowerBatch<B>;
    type Output = ();

    fn step(&self, batch: TwoTowerBatch<B>) -> TrainOutput<()> {
        let loss = self.forward_loss(batch);
        TrainOutput::new(self, loss.backward(), ())
    }
}

impl<B: Backend> InferenceStep for TwoTower<B> {
    type Input = TwoTowerBatch<B>;
    type Output = ();

    fn step(&self, batch: TwoTowerBatch<B>) {
        let _ = self.forward_loss(batch);
    }
}

/// Metadata persisted alongside the burn param blob (bincode body of the
/// `magic || version || meta || params` frame, per `serialization.rs`).
#[derive(Serialize, Deserialize, Clone, Debug)]
struct TwoTowerMeta {
    num_users: usize,
    num_items: usize,
    num_user_categories: usize,
    num_item_categories: usize,
    user_dense_dim: usize,
    item_dense_dim: usize,
    embedding_dim: usize,
    temperature: f64,
    idx_to_user: Vec<String>,
    idx_to_item: Vec<String>,
    /// Item-side features baked in at train time so prediction can build
    /// the catalog item matrix without re-reading the feature file.
    item_cat: Vec<Vec<usize>>,
    item_dense: Vec<Vec<f32>>,
    /// User-side feature-name → categorical-slot map, in train-time
    /// first-seen order (#55). Lets `predict(user_id, features=...)`
    /// translate predict-time string feature names into the integer
    /// indices the user tower expects. Empty when no user feature file
    /// was provided at training. Uses `std::collections::HashMap` for
    /// the persisted form (ahash lacks serde derives without an opt-in
    /// feature); lookup happens once per `predict` call, so the slight
    /// overhead vs `AHashMap` is irrelevant.
    #[serde(default)]
    user_cat_feature_to_idx: std::collections::HashMap<String, usize>,
    /// User-side feature-name → dense-column map (#55).
    #[serde(default)]
    user_dense_feature_to_idx: std::collections::HashMap<String, usize>,
}

/// A trained, ready-to-serve Two-Tower model on the CPU `NdArray` backend.
/// Holds the catalog item embedding matrix precomputed once at construction
/// so `predict_scores` is a single user-forward + matmul. `Clone` lets the
/// Python `ModelRegistry.register_two_tower(...)` path hand a copy to the
/// registry without taking ownership of the source model (#56).
#[derive(Clone)]
pub struct TrainedTwoTower {
    model: TwoTower<InfB>,
    meta: TwoTowerMeta,
    mappings: Mappings,
    item_matrix: Tensor<InfB, 2>,
    device: Dev,
}

/// Pad a batch's per-row categorical lists to a common width, returning
/// `(ids_flat, mask_flat, width)`.
fn pad_cat(rows: &[Vec<usize>]) -> (Vec<i64>, Vec<f32>, usize) {
    let width = rows.iter().map(|r| r.len()).max().unwrap_or(0).max(1);
    let mut ids = vec![0_i64; rows.len() * width];
    let mut mask = vec![0.0_f32; rows.len() * width];
    for (r, row) in rows.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            ids[r * width + c] = v as i64;
            mask[r * width + c] = 1.0;
        }
    }
    (ids, mask, width)
}

fn flatten_dense(rows: &[Vec<f32>], dim: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows.len() * dim.max(1)];
    if dim == 0 {
        return out;
    }
    for (r, row) in rows.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            out[r * dim + c] = v;
        }
    }
    out
}

/// Build all training batches up front (datasets here are tiny; the
/// overfit test and Python smoke test don't need streaming). Each batch
/// is a contiguous slice of `triples`.
#[allow(clippy::too_many_arguments)]
fn make_batches(
    data: &TripleData,
    user_ft: &FeatureTable,
    item_ft: &FeatureTable,
    batch_size: usize,
    device: &Dev,
) -> Vec<TwoTowerBatch<InfB>> {
    type B = InfB;
    let mut batches = Vec::new();
    for chunk in data.triples.chunks(batch_size.max(1)) {
        let b = chunk.len();
        let user_ids: Vec<i64> = chunk.iter().map(|t| t.user_idx as i64).collect();
        let item_ids: Vec<i64> = chunk.iter().map(|t| t.item_idx as i64).collect();
        let item_idx: Vec<usize> = chunk.iter().map(|t| t.item_idx).collect();
        let user_idx: Vec<usize> = chunk.iter().map(|t| t.user_idx).collect();

        let u_cat: Vec<Vec<usize>> = chunk
            .iter()
            .map(|t| user_ft.cat[t.user_idx].clone())
            .collect();
        let i_cat: Vec<Vec<usize>> = chunk
            .iter()
            .map(|t| item_ft.cat[t.item_idx].clone())
            .collect();
        let (u_cat_ids, u_cat_mask, u_w) = pad_cat(&u_cat);
        let (i_cat_ids, i_cat_mask, i_w) = pad_cat(&i_cat);

        let u_dense_rows: Vec<Vec<f32>> = chunk
            .iter()
            .map(|t| user_ft.dense[t.user_idx].clone())
            .collect();
        let i_dense_rows: Vec<Vec<f32>> = chunk
            .iter()
            .map(|t| item_ft.dense[t.item_idx].clone())
            .collect();
        let u_dense = flatten_dense(&u_dense_rows, user_ft.dense_dim);
        let i_dense = flatten_dense(&i_dense_rows, item_ft.dense_dim);

        batches.push(TwoTowerBatch {
            user_ids: Tensor::<B, 1, Int>::from_data(TensorData::new(user_ids, [b]), device),
            user_cat: Tensor::<B, 2, Int>::from_data(TensorData::new(u_cat_ids, [b, u_w]), device),
            user_cat_mask: Tensor::<B, 2>::from_data(TensorData::new(u_cat_mask, [b, u_w]), device),
            user_dense: Tensor::<B, 2>::from_data(
                TensorData::new(u_dense, [b, user_ft.dense_dim.max(1)]),
                device,
            ),
            item_ids: Tensor::<B, 1, Int>::from_data(TensorData::new(item_ids, [b]), device),
            item_cat: Tensor::<B, 2, Int>::from_data(TensorData::new(i_cat_ids, [b, i_w]), device),
            item_cat_mask: Tensor::<B, 2>::from_data(TensorData::new(i_cat_mask, [b, i_w]), device),
            item_dense: Tensor::<B, 2>::from_data(
                TensorData::new(i_dense, [b, item_ft.dense_dim.max(1)]),
                device,
            ),
            item_idx,
            user_idx,
        });
    }
    batches
}

/// Training hyperparameters for [`train`]. Grouped into one struct so the
/// entrypoint stays under clippy's argument-count lint and later phases
/// can extend it without churning every call site.
#[derive(Debug, Clone, Copy)]
pub struct TrainParams {
    pub embedding_dim: usize,
    pub temperature: f64,
    pub learning_rate: f64,
    pub epochs: usize,
    pub batch_size: usize,
    /// Fraction of training rows whose user id is dropped (remapped to
    /// the reserved cold-start row, [`COLD_START_USER_IDX`]) so that row
    /// receives gradient and learns an average-user prior. With `0.0` the
    /// reserved row is never updated and a cold-start user is no better
    /// than the old hard zero. Resampled per epoch from a seeded RNG so
    /// runs stay deterministic.
    pub id_dropout: f64,
    /// Seed for the id-dropout RNG (kept explicit for reproducibility).
    pub seed: u64,
}

impl Default for TrainParams {
    fn default() -> Self {
        Self {
            embedding_dim: 32,
            temperature: 0.05,
            learning_rate: 0.01,
            epochs: 50,
            batch_size: 256,
            id_dropout: 0.1,
            seed: 0,
        }
    }
}

/// Train a Two-Tower model from triples + feature tables.
///
/// A hand-rolled Adam loop: research §3's `LearnerBuilder` idiom is what
/// the `TrainStep`/`InferenceStep` impls above target, but for the tiny
/// datasets this phase exercises we drive `Optimizer::step` directly (the
/// same call `TrainStep::optimize` makes) so callers need no artifact
/// directory. Returns a [`TrainedTwoTower`] on the CPU backend.
pub fn train(
    data: &TripleData,
    user_ft: &FeatureTable,
    item_ft: &FeatureTable,
    params: TrainParams,
) -> Result<TrainedTwoTower> {
    use burn::optim::{GradientsParams, Optimizer};
    type AB = TrainB;

    let TrainParams {
        embedding_dim,
        temperature,
        learning_rate,
        epochs,
        batch_size,
        id_dropout,
        seed,
    } = params;

    let device = Dev::default();
    let cfg = TwoTowerConfig::new(
        data.num_users(),
        data.num_items(),
        user_ft.num_categories,
        item_ft.num_categories,
        user_ft.dense_dim,
        item_ft.dense_dim,
        embedding_dim,
    )
    .with_temperature(temperature);

    let mut model = cfg.init::<AB>(&device);
    let mut optim = AdamConfig::new().init();

    let batches = make_batches(data, user_ft, item_ft, batch_size, &device);

    // Seeded RNG so id-dropout (and thus the whole run) is reproducible.
    // Dropout is resampled every epoch: a row that kept its id one epoch
    // may be dropped the next, so the reserved row sees a broad slice of
    // the population rather than a fixed subset.
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    for _ in 0..epochs {
        for batch in &batches {
            let ad_batch = to_autodiff_batch(batch, id_dropout, &mut rng);
            let loss = model.forward_loss(ad_batch);
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(learning_rate, model, grads);
        }
    }

    finalize(
        model,
        user_ft,
        data,
        item_ft,
        embedding_dim,
        temperature,
        &device,
    )
}

/// Re-create a batch on the autodiff backend (params train under
/// `Autodiff<NdArray>`; inference uses bare `NdArray`) with id-dropout
/// applied: each row's user id is, with probability `id_dropout`, replaced
/// by the reserved cold-start index ([`COLD_START_USER_IDX`]) so that row
/// receives gradient and learns an average-user prior. Feature embeddings
/// are intentionally *kept* (the id is dropped, not the row's side info),
/// which mirrors the cold-start serving path: reserved id row + real
/// features. `id_dropout == 0.0` reproduces the plain (no-dropout) batch.
fn to_autodiff_batch(
    b: &TwoTowerBatch<InfB>,
    id_dropout: f64,
    rng: &mut impl rand::Rng,
) -> TwoTowerBatch<TrainB> {
    use crate::data::triples::COLD_START_USER_IDX;
    type AB = TrainB;
    let device = Dev::default();
    let user_ids: Vec<i64> = b
        .user_idx
        .iter()
        .map(|&u| {
            // `gen_bool` clamps probability to [0, 1]; 0.0 never drops.
            if id_dropout > 0.0 && rng.gen_bool(id_dropout.clamp(0.0, 1.0)) {
                COLD_START_USER_IDX as i64
            } else {
                u as i64
            }
        })
        .collect();
    let n = user_ids.len();
    TwoTowerBatch {
        user_ids: Tensor::<AB, 1, Int>::from_data(TensorData::new(user_ids, [n]), &device),
        user_cat: Tensor::<AB, 2, Int>::from_data(b.user_cat.to_data(), &device),
        user_cat_mask: Tensor::<AB, 2>::from_data(b.user_cat_mask.to_data(), &device),
        user_dense: Tensor::<AB, 2>::from_data(b.user_dense.to_data(), &device),
        item_ids: Tensor::<AB, 1, Int>::from_data(b.item_ids.to_data(), &device),
        item_cat: Tensor::<AB, 2, Int>::from_data(b.item_cat.to_data(), &device),
        item_cat_mask: Tensor::<AB, 2>::from_data(b.item_cat_mask.to_data(), &device),
        item_dense: Tensor::<AB, 2>::from_data(b.item_dense.to_data(), &device),
        item_idx: b.item_idx.clone(),
        user_idx: b.user_idx.clone(),
    }
}

/// Move trained params onto the inference backend, attach the real
/// metadata + mappings, and precompute the catalog item matrix.
fn finalize(
    trained: TwoTower<TrainB>,
    user_ft: &FeatureTable,
    data: &TripleData,
    item_ft: &FeatureTable,
    embedding_dim: usize,
    temperature: f64,
    device: &Dev,
) -> Result<TrainedTwoTower> {
    type B = InfB;
    // Round-trip params autodiff -> ndarray via the recorder, then
    // reload onto the inference backend with the full config.
    let rec = BinBytesRecorder::<FullPrecisionSettings>::default();
    let blob = rec
        .record(trained.into_record(), ())
        .map_err(|e| anyhow!("recorder.record failed: {e}"))?;
    let cfg = TwoTowerConfig::new(
        data.num_users(),
        data.num_items(),
        user_ft.num_categories,
        item_ft.num_categories,
        user_ft.dense_dim,
        item_ft.dense_dim,
        embedding_dim,
    )
    .with_temperature(temperature);
    let record = rec
        .load(blob, device)
        .map_err(|e| anyhow!("recorder.load failed: {e}"))?;
    let model: TwoTower<B> = cfg.init(device).load_record(record);

    let meta = TwoTowerMeta {
        num_users: data.num_users(),
        num_items: data.num_items(),
        num_user_categories: user_ft.num_categories,
        num_item_categories: item_ft.num_categories,
        user_dense_dim: user_ft.dense_dim,
        item_dense_dim: item_ft.dense_dim,
        embedding_dim,
        temperature,
        idx_to_user: data.idx_to_user.clone(),
        idx_to_item: data.idx_to_item.clone(),
        item_cat: item_ft.cat.clone(),
        item_dense: item_ft.dense.clone(),
        user_cat_feature_to_idx: user_ft
            .cat_feature_to_idx
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect(),
        user_dense_feature_to_idx: user_ft
            .dense_feature_to_idx
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect(),
    };

    let item_matrix = catalog_matrix(&model, &meta, device);
    Ok(TrainedTwoTower {
        model,
        meta,
        mappings: build_mappings(data),
        item_matrix,
        device: *device,
    })
}

fn build_mappings(data: &TripleData) -> Mappings {
    Mappings {
        user_to_idx: data.user_to_idx.clone(),
        idx_to_user: data.idx_to_user.clone(),
        item_to_idx: data.item_to_idx.clone(),
        idx_to_item: data.idx_to_item.clone(),
        user_feature_to_idx: AHashMap::new(),
        idx_to_user_feature: Vec::new(),
        item_feature_to_idx: AHashMap::new(),
        idx_to_item_feature: Vec::new(),
    }
}

/// Build the `(num_items, dim)` catalog matrix by forwarding every item
/// through the item tower once.
fn catalog_matrix(model: &TwoTower<InfB>, meta: &TwoTowerMeta, device: &Dev) -> Tensor<InfB, 2> {
    type B = InfB;
    let n = meta.num_items;
    let ids: Vec<i64> = (0..n as i64).collect();
    let (cat_ids, cat_mask, w) = pad_cat(&meta.item_cat);
    let dense = flatten_dense(&meta.item_dense, meta.item_dense_dim);
    model.item_tower.forward(
        Tensor::<B, 1, Int>::from_data(TensorData::new(ids, [n]), device),
        Tensor::<B, 2, Int>::from_data(TensorData::new(cat_ids, [n, w]), device),
        Tensor::<B, 2>::from_data(TensorData::new(cat_mask, [n, w]), device),
        Tensor::<B, 2>::from_data(
            TensorData::new(dense, [n, meta.item_dense_dim.max(1)]),
            device,
        ),
        1.0,
    )
}

impl TrainedTwoTower {
    /// Serialize to the framed `FTWO || version || bincode(meta) ||
    /// params` format (research §4, mirrors `serialization.rs`).
    pub fn save_to(&self, path: &Path) -> Result<()> {
        let rec = BinBytesRecorder::<FullPrecisionSettings>::default();
        let weights = rec
            .record(self.model.clone().into_record(), ())
            .map_err(|e| anyhow!("recorder.record failed: {e}"))?;
        let meta = bincode::serialize(&self.meta).context("serialize TwoTowerMeta")?;

        let mut out = Vec::with_capacity(4 + 4 + 8 + meta.len() + 8 + weights.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&(meta.len() as u64).to_le_bytes());
        out.extend_from_slice(&meta);
        out.extend_from_slice(&(weights.len() as u64).to_le_bytes());
        out.extend_from_slice(&weights);

        fs::write(path, &out)
            .with_context(|| format!("write Two-Tower model to {}", path.display()))?;
        Ok(())
    }

    /// Load a model written by [`TrainedTwoTower::save_to`].
    pub fn load_from(path: &Path) -> Result<Self> {
        type B = InfB;
        let data = fs::read(path)
            .with_context(|| format!("read Two-Tower model from {}", path.display()))?;
        if data.len() < 8 {
            anyhow::bail!("file too small to be a Two-Tower model: {}", path.display());
        }
        if &data[..4] != MAGIC {
            anyhow::bail!("invalid magic in {} (expected FTWO header)", path.display());
        }
        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if version < FORMAT_VERSION {
            anyhow::bail!(
                "Two-Tower model file {} is format v{version}, but this build \
                 requires v{FORMAT_VERSION}. v5 persists the user-side \
                 feature-name → index maps (#55) that v4 discarded — they \
                 can't be reconstructed without retraining since the train-\
                 time feature names aren't recorded anywhere else. v4 \
                 itself reserves a learnable cold-start user row and \
                 renumbers all user indices vs v3. Retrain the model to \
                 upgrade.",
                path.display()
            );
        }
        if version != FORMAT_VERSION {
            anyhow::bail!(
                "unsupported Two-Tower format version {version} (expected {FORMAT_VERSION})"
            );
        }
        // Bounds-checked framed read: a truncated or corrupt file (any
        // length, including a u64 length prefix that overflows the buffer)
        // returns Err instead of panicking on an out-of-range slice.
        let take = |lo: usize, len: usize| -> Result<&[u8]> {
            let hi = lo
                .checked_add(len)
                .filter(|&h| h <= data.len())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "truncated or corrupt Two-Tower model file: {}",
                        path.display()
                    )
                })?;
            Ok(&data[lo..hi])
        };
        let mut off = 8usize;
        let meta_len = u64::from_le_bytes(take(off, 8)?.try_into().unwrap()) as usize;
        off += 8;
        let meta: TwoTowerMeta =
            bincode::deserialize(take(off, meta_len)?).context("deserialize TwoTowerMeta")?;
        off += meta_len;
        let w_len = u64::from_le_bytes(take(off, 8)?.try_into().unwrap()) as usize;
        off += 8;
        let weights = take(off, w_len)?.to_vec();

        let device = Dev::default();
        let cfg = TwoTowerConfig::new(
            meta.num_users,
            meta.num_items,
            meta.num_user_categories,
            meta.num_item_categories,
            meta.user_dense_dim,
            meta.item_dense_dim,
            meta.embedding_dim,
        )
        .with_temperature(meta.temperature);
        let rec = BinBytesRecorder::<FullPrecisionSettings>::default();
        let record = rec
            .load(weights, &device)
            .map_err(|e| anyhow!("recorder.load failed: {e}"))?;
        let model: TwoTower<B> = cfg.init(&device).load_record(record);

        let mappings = Mappings {
            user_to_idx: meta
                .idx_to_user
                .iter()
                .enumerate()
                // Mirror the train-path invariant (see `load_triples`):
                // index 0 is the reserved cold-start sentinel and is
                // deliberately absent from `user_to_idx`, so a loaded
                // model's mappings match a freshly-trained one and no real
                // id can resolve to the prior row.
                .filter(|(i, _)| *i != crate::data::triples::COLD_START_USER_IDX)
                .map(|(i, s)| (s.clone(), i))
                .collect(),
            idx_to_user: meta.idx_to_user.clone(),
            item_to_idx: meta
                .idx_to_item
                .iter()
                .enumerate()
                .map(|(i, s)| (s.clone(), i))
                .collect(),
            idx_to_item: meta.idx_to_item.clone(),
            user_feature_to_idx: AHashMap::new(),
            idx_to_user_feature: Vec::new(),
            item_feature_to_idx: AHashMap::new(),
            idx_to_item_feature: Vec::new(),
        };
        let item_matrix = catalog_matrix(&model, &meta, &device);
        Ok(Self {
            model,
            meta,
            mappings,
            item_matrix,
            device,
        })
    }

    /// Embed a user-side input and score every catalog item.
    fn score_user(
        &self,
        user_idx: Option<usize>,
        cat_features: &[usize],
        dense_features: &[f32],
    ) -> Result<Vec<f32>> {
        type B = InfB;
        let dev = &self.device;

        // Cold-start: with no id, use the reserved learnable cold-start
        // row (index 0, COLD_START_USER_IDX) at full strength instead of
        // zeroing the id term. id-dropout during training makes that row a
        // learned average-user prior; combined with this user's feature
        // embeddings (if any) it yields a principled cold-start vector.
        // Warm users keep `id_scale = 1.0` on their own row — unchanged.
        let id_scale = 1.0;
        let id = user_idx.unwrap_or(crate::data::triples::COLD_START_USER_IDX) as i64;
        let (cat_ids, cat_mask, w) = pad_cat(std::slice::from_ref(&cat_features.to_vec()));
        let d = self.meta.user_dense_dim;
        let dense = if d == 0 {
            vec![0.0_f32; 1]
        } else {
            let mut v = vec![0.0_f32; d];
            for (i, &x) in dense_features.iter().take(d).enumerate() {
                v[i] = x;
            }
            v
        };

        let user_vec = self.model.user_tower.forward(
            Tensor::<B, 1, Int>::from_data(TensorData::new(vec![id], [1]), dev),
            Tensor::<B, 2, Int>::from_data(TensorData::new(cat_ids, [1, w]), dev),
            Tensor::<B, 2>::from_data(TensorData::new(cat_mask, [1, w]), dev),
            Tensor::<B, 2>::from_data(TensorData::new(dense, [1, d.max(1)]), dev),
            id_scale,
        );

        Ok(self.model.score_all(user_vec, self.item_matrix.clone()))
    }
}

impl RecModel for TrainedTwoTower {
    fn kind(&self) -> ModelKind {
        ModelKind::TwoTower
    }

    fn num_items(&self) -> usize {
        self.meta.num_items
    }

    fn item_mapping(&self) -> &Mappings {
        &self.mappings
    }

    fn predict_scores(&self, input: ModelInput<'_>) -> Result<Vec<f32>> {
        match input {
            ModelInput::TowerUser {
                user_idx,
                cat_features,
                dense_features,
            } => self.score_user(user_idx, cat_features, dense_features),
            ModelInput::Sparse { .. } => Err(anyhow!(
                "Two-Tower does not support ModelInput::Sparse; expected ModelInput::TowerUser"
            )),
            ModelInput::Sequence { .. } => Err(anyhow!(
                "Two-Tower does not support ModelInput::Sequence; expected ModelInput::TowerUser"
            )),
        }
    }

    fn predict_similar_items(&self, item_idx: usize, top_k: usize) -> Result<Vec<(usize, f32)>> {
        if item_idx >= self.meta.num_items {
            return Err(anyhow!(
                "item_idx {item_idx} out of range (num_items = {})",
                self.meta.num_items
            ));
        }
        let dim = self.item_matrix.dims()[1];
        let q = self
            .item_matrix
            .clone()
            .slice([item_idx..item_idx + 1, 0..dim]);
        let scores = self.model.score_all(q, self.item_matrix.clone());
        let mut idx: Vec<(usize, f32)> = scores
            .into_iter()
            .enumerate()
            .filter(|(i, _)| *i != item_idx)
            .collect();
        idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        idx.truncate(top_k);
        Ok(idx)
    }

    fn validate(&self) -> ValidationReport {
        let mut messages = Vec::new();
        let mut passed = true;
        if self.meta.num_items == 0 {
            passed = false;
            messages.push("Two-Tower model has zero items".to_string());
        }
        if self.item_matrix.dims()[0] != self.meta.num_items {
            passed = false;
            messages.push(format!(
                "catalog matrix rows {} != num_items {}",
                self.item_matrix.dims()[0],
                self.meta.num_items
            ));
        }
        ValidationReport { passed, messages }
    }

    fn save(&self, path: &Path) -> Result<()> {
        self.save_to(path)
    }

    /// Translate a `feature_name → value` map into the integer
    /// `(cat_features, dense_features)` pair the user tower expects
    /// (#55). Routing rules:
    /// - Categorical (one-hot) features with non-zero value contribute
    ///   their slot index to `cat_features`. Magnitude unused.
    /// - Dense numeric features fill the matching dense column.
    /// - Unknown feature names are silently skipped.
    fn resolve_user_features(&self, features: &AHashMap<String, f64>) -> (Vec<usize>, Vec<f32>) {
        let dense_dim = self.meta.user_dense_dim;
        let mut cat_features: Vec<usize> = Vec::new();
        let mut dense_features: Vec<f32> = vec![0.0; dense_dim];
        for (name, value) in features {
            if let Some(&cat_idx) = self.meta.user_cat_feature_to_idx.get(name) {
                if *value != 0.0 {
                    cat_features.push(cat_idx);
                }
            } else if let Some(&col) = self.meta.user_dense_feature_to_idx.get(name)
                && col < dense_dim
            {
                dense_features[col] = *value as f32;
            }
        }
        (cat_features, dense_features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::triples::Triple;

    use crate::data::triples::{COLD_START_USER_ID, COLD_START_USER_IDX};

    /// 3 real users at indices 1..=3 (index 0 is the reserved cold-start
    /// row), 3 items at 0..=2; each real user strongly prefers item
    /// `user_idx - 1`.
    fn tiny_data() -> (TripleData, FeatureTable, FeatureTable) {
        let triples = vec![
            Triple {
                user_idx: 1,
                item_idx: 0,
            },
            Triple {
                user_idx: 1,
                item_idx: 0,
            },
            Triple {
                user_idx: 2,
                item_idx: 1,
            },
            Triple {
                user_idx: 2,
                item_idx: 1,
            },
            Triple {
                user_idx: 3,
                item_idx: 2,
            },
            Triple {
                user_idx: 3,
                item_idx: 2,
            },
        ];
        let mut user_to_idx = AHashMap::new();
        let mut item_to_idx = AHashMap::new();
        for i in 0..3 {
            // Real user `u{i}` lives at embedding index i + 1.
            user_to_idx.insert(format!("u{i}"), i + 1);
            item_to_idx.insert(format!("i{i}"), i);
        }
        let data = TripleData {
            triples,
            user_to_idx,
            idx_to_user: vec![
                COLD_START_USER_ID.into(),
                "u0".into(),
                "u1".into(),
                "u2".into(),
            ],
            item_to_idx,
            idx_to_item: vec!["i0".into(), "i1".into(), "i2".into()],
        };
        // User table has 4 rows (reserved + 3); item table has 3.
        (data, FeatureTable::empty(4), FeatureTable::empty(3))
    }

    #[test]
    fn overfits_tiny_data() {
        let (data, uft, ift) = tiny_data();
        let model = train(
            &data,
            &uft,
            &ift,
            TrainParams {
                embedding_dim: 16,
                temperature: 0.05,
                learning_rate: 0.05,
                epochs: 400,
                batch_size: 6,
                // No id-dropout here so the per-user rows fully overfit.
                id_dropout: 0.0,
                seed: 0,
            },
        )
        .expect("train");

        // After overfitting, each real user's top-scored item is its
        // positive. Real users are at embedding indices 1..=3; user index
        // `u` prefers item `u - 1`.
        for u in 1..=3usize {
            let scores = model
                .predict_scores(ModelInput::TowerUser {
                    user_idx: Some(u),
                    cat_features: &[],
                    dense_features: &[],
                })
                .expect("predict");
            let best = scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            assert_eq!(
                best,
                u - 1,
                "user {u} should rank item {} first: {scores:?}",
                u - 1
            );
        }
    }

    #[test]
    fn save_load_roundtrip_preserves_scores() {
        let (data, uft, ift) = tiny_data();
        let model = train(
            &data,
            &uft,
            &ift,
            TrainParams {
                embedding_dim: 8,
                epochs: 50,
                batch_size: 6,
                learning_rate: 0.05,
                ..TrainParams::default()
            },
        )
        .expect("train");

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tt.fease");
        model.save_to(&path).expect("save");
        let loaded = TrainedTwoTower::load_from(&path).expect("load");

        // Cover the reserved cold-start row (0) and the real users (1..=3).
        for u in 0..=3usize {
            let a = model
                .predict_scores(ModelInput::TowerUser {
                    user_idx: Some(u),
                    cat_features: &[],
                    dense_features: &[],
                })
                .unwrap();
            let b = loaded
                .predict_scores(ModelInput::TowerUser {
                    user_idx: Some(u),
                    cat_features: &[],
                    dense_features: &[],
                })
                .unwrap();
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(b.iter()) {
                assert!(
                    (x - y).abs() < 1e-5,
                    "score drift after roundtrip: {x} vs {y}"
                );
            }
        }
    }

    #[test]
    fn rejects_unsupported_input() {
        let (data, uft, ift) = tiny_data();
        let model = train(
            &data,
            &uft,
            &ift,
            TrainParams {
                embedding_dim: 4,
                epochs: 5,
                batch_size: 6,
                ..TrainParams::default()
            },
        )
        .expect("train");
        let r = model.predict_scores(ModelInput::Sequence { history: &[] });
        assert!(r.is_err(), "Two-Tower must reject Sequence input");
    }

    #[test]
    fn similar_items_excludes_query_and_respects_k() {
        let (data, uft, ift) = tiny_data();
        let model = train(
            &data,
            &uft,
            &ift,
            TrainParams {
                embedding_dim: 8,
                epochs: 50,
                batch_size: 6,
                learning_rate: 0.05,
                ..TrainParams::default()
            },
        )
        .expect("train");
        let sim = model.predict_similar_items(0, 2).expect("similar");
        assert!(sim.len() <= 2);
        assert!(sim.iter().all(|(i, _)| *i != 0));
    }

    /// A feature-less cold-start user (`user_idx: None`) must receive the
    /// *trained* reserved-row prior — a non-degenerate score vector that
    /// the model actually learned, not a constant/all-zero fallback. With
    /// id-dropout on, the reserved row gets gradient from a mix of users,
    /// so its scores are non-constant and equal the explicit lookup of the
    /// reserved index (`Some(COLD_START_USER_IDX)`).
    #[test]
    fn coldstart_user_gets_trained_prior_not_constant() {
        let (data, uft, ift) = tiny_data();
        let model = train(
            &data,
            &uft,
            &ift,
            TrainParams {
                embedding_dim: 16,
                epochs: 300,
                batch_size: 6,
                learning_rate: 0.05,
                // Substantial dropout so row 0 is trained meaningfully.
                id_dropout: 0.5,
                seed: 7,
                ..TrainParams::default()
            },
        )
        .expect("train");

        let cold = model
            .predict_scores(ModelInput::TowerUser {
                user_idx: None,
                cat_features: &[],
                dense_features: &[],
            })
            .expect("predict cold");

        // Not a constant/zero vector: a hard-zero id term (the old
        // behavior) with no features would make every item score the same.
        assert_eq!(cold.len(), 3);
        let max = cold.iter().cloned().fold(f32::MIN, f32::max);
        let min = cold.iter().cloned().fold(f32::MAX, f32::min);
        assert!(
            (max - min).abs() > 1e-4,
            "cold-start scores look constant ({cold:?}) — reserved row \
             was not trained"
        );
        assert!(
            cold.iter().any(|s| s.abs() > 1e-6),
            "cold-start scores are all ~zero ({cold:?})"
        );

        // `None` must resolve to the reserved row at full id strength,
        // i.e. be identical to an explicit lookup of COLD_START_USER_IDX.
        let explicit = model
            .predict_scores(ModelInput::TowerUser {
                user_idx: Some(COLD_START_USER_IDX),
                cat_features: &[],
                dense_features: &[],
            })
            .expect("predict explicit reserved");
        for (a, b) in cold.iter().zip(explicit.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "cold-start (None) must equal reserved-row lookup: \
                 {a} vs {b}"
            );
        }
    }

    /// Warm-user behavior is unchanged versus the pre-change (hard-zero)
    /// baseline. The pre-change warm path was: look up the user's own row
    /// at `id_scale = 1.0` and score the catalog — exactly what the
    /// reserved-row change preserves (only the *cold-start* branch moved
    /// from a hard zero to the reserved row). We assert the load-invariant
    /// for warm users (a trained model and its serialized round-trip score
    /// warm users identically, within tight tolerance), and that warm
    /// users are *not* contaminated by the cold-start prior: each warm
    /// user still ranks its own positive first and does not collapse onto
    /// the reserved-row scores — true both with and without id-dropout.
    /// (Cross-*training-run* bit-equality is intentionally not asserted:
    /// burn's parameter initialization is not seeded, only id-dropout is.)
    #[test]
    fn warm_user_scores_match_no_dropout_baseline() {
        let (data, uft, ift) = tiny_data();
        for id_dropout in [0.0_f64, 0.3] {
            let model = train(
                &data,
                &uft,
                &ift,
                TrainParams {
                    embedding_dim: 16,
                    epochs: 300,
                    batch_size: 6,
                    learning_rate: 0.05,
                    id_dropout,
                    seed: 42,
                    ..TrainParams::default()
                },
            )
            .expect("train");

            // Tight-tolerance baseline: serialize + reload must reproduce
            // warm-user scores bit-close (the warm scoring path itself is
            // unchanged by the cold-start work).
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("warm.fease");
            model.save_to(&path).expect("save");
            let reloaded = TrainedTwoTower::load_from(&path).expect("load");

            let cold = model
                .predict_scores(ModelInput::TowerUser {
                    user_idx: None,
                    cat_features: &[],
                    dense_features: &[],
                })
                .unwrap();

            for u in 1..=3usize {
                let a = model
                    .predict_scores(ModelInput::TowerUser {
                        user_idx: Some(u),
                        cat_features: &[],
                        dense_features: &[],
                    })
                    .unwrap();
                let b = reloaded
                    .predict_scores(ModelInput::TowerUser {
                        user_idx: Some(u),
                        cat_features: &[],
                        dense_features: &[],
                    })
                    .unwrap();
                for (x, y) in a.iter().zip(b.iter()) {
                    assert!(
                        (x - y).abs() < 1e-5,
                        "warm user {u} score drift across save/load \
                         (id_dropout={id_dropout}): {x} vs {y}"
                    );
                }
                // Each warm user still ranks its own positive first.
                let best = a
                    .iter()
                    .enumerate()
                    .max_by(|p, q| p.1.partial_cmp(q.1).unwrap())
                    .unwrap()
                    .0;
                assert_eq!(
                    best,
                    u - 1,
                    "warm user {u} mis-ranked (id_dropout={id_dropout}): {a:?}"
                );
                // Warm users use their own row, not the reserved prior:
                // their score vector is not identical to the cold-start
                // vector (which would mean the warm path collapsed onto
                // row 0).
                let same_as_cold = a.iter().zip(cold.iter()).all(|(x, y)| (x - y).abs() < 1e-6);
                assert!(
                    !same_as_cold,
                    "warm user {u} collapsed onto the cold-start prior \
                     (id_dropout={id_dropout})"
                );
            }
        }
    }

    /// Loading a pre-current (older format) file must fail loudly with a
    /// retrain message rather than silently mis-mapping users / missing
    /// the user-feature maps.
    #[test]
    fn rejects_old_format_with_retrain_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("old.fease");
        // Minimal v3 frame: MAGIC + version=3 + empty meta + empty params.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();

        let res = TrainedTwoTower::load_from(&path);
        let err = match res {
            Ok(_) => panic!("old-format file must not load"),
            Err(e) => e,
        };
        let msg = format!("{err}");
        assert!(
            msg.contains("Retrain") || msg.contains("retrain"),
            "expected a retrain-required error, got: {msg}"
        );
        assert!(
            msg.contains("v3") && msg.contains(&format!("v{FORMAT_VERSION}")),
            "got: {msg}"
        );
    }

    /// #55: `resolve_user_features` translates a string `feature_name →
    /// value` dict into the integer `(cat_features, dense_features)`
    /// pair the user tower expects. This unit test exercises the helper
    /// directly so we cover one-hot routing, dense routing, unknown
    /// names, and zero-value categoricals without needing the full
    /// Python predict path.
    #[test]
    fn resolve_user_features_routes_cat_dense_and_skips_unknown() {
        // Build a tiny FeatureTable with one categorical slot ("plan_free")
        // and one dense column ("tenure_days"). The data has 3 users +
        // the reserved cold-start row, with feature row populated only
        // for user index 1.
        let mut uft = FeatureTable::empty(4);
        uft.num_categories = 1;
        uft.dense_dim = 1;
        uft.cat_feature_to_idx.insert("plan_free".into(), 0);
        uft.dense_feature_to_idx.insert("tenure_days".into(), 0);
        uft.cat[1] = vec![0];
        uft.dense[1] = vec![10.0];
        let (data, _, ift) = tiny_data();
        let trained = train(
            &data,
            &uft,
            &ift,
            TrainParams {
                embedding_dim: 4,
                epochs: 1,
                batch_size: 6,
                ..TrainParams::default()
            },
        )
        .expect("train");

        // Categorical with non-zero value → cat_features carries its idx.
        let mut f = ahash::AHashMap::new();
        f.insert("plan_free".to_string(), 1.0);
        let (cat, dense) = trained.resolve_user_features(&f);
        assert_eq!(cat, vec![0]);
        assert_eq!(dense, vec![0.0]);

        // Categorical with zero value → omitted.
        let mut f = ahash::AHashMap::new();
        f.insert("plan_free".to_string(), 0.0);
        let (cat, _dense) = trained.resolve_user_features(&f);
        assert!(cat.is_empty(), "zero-valued cat should be skipped");

        // Dense feature → written into the matching column.
        let mut f = ahash::AHashMap::new();
        f.insert("tenure_days".to_string(), 42.0);
        let (cat, dense) = trained.resolve_user_features(&f);
        assert!(cat.is_empty());
        assert_eq!(dense, vec![42.0]);

        // Unknown name → silently skipped (no panic, no error).
        let mut f = ahash::AHashMap::new();
        f.insert("never_seen".to_string(), 99.0);
        let (cat, dense) = trained.resolve_user_features(&f);
        assert!(cat.is_empty());
        assert_eq!(dense, vec![0.0]);
    }
}
