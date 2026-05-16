//! Two-Tower — separate user / item embedding networks with categorical
//! and dense numerical features, trained by in-batch sampled softmax.
//!
//! ADR-0001 Phase 5 (issue #38). Replaces the Phase 2a stub.
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
//! Cold-start: a user / item with zero interactions still gets a vector
//! from its feature embeddings, so the towers score it without ever
//! having seen its id (ADR-0001 §Risks — the gap pure SASRec can't close).
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
/// Framed format version. EASE is at v2; SASRec/Two-Tower share v3
/// (research §4 "bump to v3").
const FORMAT_VERSION: u32 = 3;

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
    ) -> Tensor<B, 2> {
        let [batch, _] = cat_ids.dims();
        let dim = self.id_embedding.weight.val().dims()[1];

        let mut h = self
            .id_embedding
            .forward(ids.reshape([batch, 1]))
            .reshape([batch, dim]);

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
        ); // (B, dim)
        let v = self.item_tower.forward(
            batch.item_ids,
            batch.item_cat,
            batch.item_cat_mask,
            batch.item_dense,
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
}

/// A trained, ready-to-serve Two-Tower model on the CPU `NdArray` backend.
/// Holds the catalog item embedding matrix precomputed once at construction
/// so `predict_scores` is a single user-forward + matmul.
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
}

impl Default for TrainParams {
    fn default() -> Self {
        Self {
            embedding_dim: 32,
            temperature: 0.05,
            learning_rate: 0.01,
            epochs: 50,
            batch_size: 256,
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
    let ad_batches: Vec<TwoTowerBatch<AB>> = batches.iter().map(to_autodiff_batch).collect();

    for _ in 0..epochs {
        for batch in &ad_batches {
            let loss = model.forward_loss(batch.clone());
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
/// `Autodiff<NdArray>`; inference uses bare `NdArray`).
fn to_autodiff_batch(b: &TwoTowerBatch<InfB>) -> TwoTowerBatch<TrainB> {
    type AB = TrainB;
    let device = Dev::default();
    TwoTowerBatch {
        user_ids: Tensor::<AB, 1, Int>::from_data(b.user_ids.to_data(), &device),
        user_cat: Tensor::<AB, 2, Int>::from_data(b.user_cat.to_data(), &device),
        user_cat_mask: Tensor::<AB, 2>::from_data(b.user_cat_mask.to_data(), &device),
        user_dense: Tensor::<AB, 2>::from_data(b.user_dense.to_data(), &device),
        item_ids: Tensor::<AB, 1, Int>::from_data(b.item_ids.to_data(), &device),
        item_cat: Tensor::<AB, 2, Int>::from_data(b.item_cat.to_data(), &device),
        item_cat_mask: Tensor::<AB, 2>::from_data(b.item_cat_mask.to_data(), &device),
        item_dense: Tensor::<AB, 2>::from_data(b.item_dense.to_data(), &device),
        item_idx: b.item_idx.clone(),
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
        if version != FORMAT_VERSION {
            anyhow::bail!(
                "unsupported Two-Tower format version {version} (expected {FORMAT_VERSION})"
            );
        }
        let mut off = 8;
        let meta_len = u64::from_le_bytes(data[off..off + 8].try_into().unwrap()) as usize;
        off += 8;
        let meta: TwoTowerMeta =
            bincode::deserialize(&data[off..off + meta_len]).context("deserialize TwoTowerMeta")?;
        off += meta_len;
        let w_len = u64::from_le_bytes(data[off..off + 8].try_into().unwrap()) as usize;
        off += 8;
        let weights = data[off..off + w_len].to_vec();

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

        // Cold-start: no id -> use index 0's embedding but rely on
        // features; the tower sums id + feature contributions, so a
        // feature-only user still gets a meaningful vector.
        let id = user_idx.unwrap_or(0) as i64;
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::triples::Triple;

    fn tiny_data() -> (TripleData, FeatureTable, FeatureTable) {
        // 3 users, 3 items, each user strongly prefers one item.
        let triples = vec![
            Triple {
                user_idx: 0,
                item_idx: 0,
            },
            Triple {
                user_idx: 0,
                item_idx: 0,
            },
            Triple {
                user_idx: 1,
                item_idx: 1,
            },
            Triple {
                user_idx: 1,
                item_idx: 1,
            },
            Triple {
                user_idx: 2,
                item_idx: 2,
            },
            Triple {
                user_idx: 2,
                item_idx: 2,
            },
        ];
        let mut user_to_idx = AHashMap::new();
        let mut item_to_idx = AHashMap::new();
        for i in 0..3 {
            user_to_idx.insert(format!("u{i}"), i);
            item_to_idx.insert(format!("i{i}"), i);
        }
        let data = TripleData {
            triples,
            user_to_idx,
            idx_to_user: vec!["u0".into(), "u1".into(), "u2".into()],
            item_to_idx,
            idx_to_item: vec!["i0".into(), "i1".into(), "i2".into()],
        };
        (data, FeatureTable::empty(3), FeatureTable::empty(3))
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
            },
        )
        .expect("train");

        // After overfitting, each user's top-scored item is its positive.
        for u in 0..3 {
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
            assert_eq!(best, u, "user {u} should rank item {u} first: {scores:?}");
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
                temperature: 0.05,
                learning_rate: 0.05,
                epochs: 50,
                batch_size: 6,
            },
        )
        .expect("train");

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tt.fease");
        model.save_to(&path).expect("save");
        let loaded = TrainedTwoTower::load_from(&path).expect("load");

        for u in 0..3 {
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
                temperature: 0.05,
                learning_rate: 0.05,
                epochs: 5,
                batch_size: 6,
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
                temperature: 0.05,
                learning_rate: 0.05,
                epochs: 50,
                batch_size: 6,
            },
        )
        .expect("train");
        let sim = model.predict_similar_items(0, 2).expect("similar");
        assert!(sim.len() <= 2);
        assert!(sim.iter().all(|(i, _)| *i != 0));
    }
}
