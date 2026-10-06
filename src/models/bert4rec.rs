//! BERT4Rec — bidirectional self-attention recommender trained with a
//! Cloze (masked-item) objective. Issue #96.
//!
//! Architecture (Sun et al., CIKM 2019 "BERT4Rec: Sequential
//! Recommendation with Bidirectional Encoder Representations from
//! Transformer"): item embedding + learned *relative-recency* position
//! embedding → N **bidirectional** transformer-encoder blocks → linear
//! projection back to the item vocabulary. `forward` returns raw logits
//! `(batch, seq_len, vocab)`; the training loss is full-softmax
//! cross-entropy **only at masked positions**.
//!
//! Two deliberate departures from SASRec (`crate::models::sasrec`):
//!
//! 1. **No causal mask.** Every non-pad position attends to every other
//!    non-pad position. The model learns "what belongs in this
//!    collection" rather than "what comes next", which fits catalogs
//!    (anime, back-catalog film) where discovery order carries little
//!    signal.
//! 2. **Positions are log₂ `days_ago` buckets**, not sequence offsets
//!    (see `crate::data::masked_sequences`). Two items watched in the
//!    same doubling window share a position regardless of absolute date.
//!
//! Besides scoring, a trained model exposes **user embeddings** (the
//! mean-pooled hidden state over an unmasked history) and the **item
//! embedding table**, which the hybrid pipeline
//! (`kzn_recsys/hybrid_train.py`) feeds into FEASE as dense side
//! features.
//!
//! Persists via burn's `Recorder` inside the framed `FB4R` file format.
//! Gated behind the default-off `ml-models` Cargo feature. The backend
//! stays generic (`Bert4Rec<B: Backend>`); inference uses `NdArray` and
//! training uses `Autodiff<NdArray>`.
// burn 0.21 `#[derive(Config)]` expands to `Self { field: field, .. }`, which
// clippy >= 1.99 lints even inside the derive output (same allow as
// `sasrec.rs` / `two_tower.rs`).
#![allow(clippy::redundant_field_names)]

use burn::config::Config;
use burn::module::{AutodiffModule, Module};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::transformer::{
    TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{ElementConversion, Int, Tensor, TensorData};

use crate::data::masked_sequences::{
    MASK_TOKEN, MaskedSequenceDataset, NUM_RESERVED_TOKENS, PAD_TOKEN, days_ago_to_bucket,
    item_to_token, read_user_histories,
};

/// Hyperparameters for [`Bert4Rec`]. Construction-only knobs (vocab
/// size, dims, depth, bucket count); training-side params live in
/// [`Bert4RecTrainingConfig`].
#[derive(Config, Debug)]
pub struct Bert4RecConfig {
    /// Output logit dimension: `num_items + 2` (reserved `PAD` + `MASK`
    /// tokens, see `data::masked_sequences`).
    pub vocab_size: usize,
    /// Embedding / model dimension `d_model`. Must be divisible by
    /// `num_heads`. Also the length of the user embedding the hybrid
    /// pipeline feeds into FEASE, so it trades expressiveness against
    /// FEASE Gram-matrix growth (HPO range `[32, 64, 128, 256]`).
    #[config(default = 64)]
    pub embedding_dim: usize,
    /// Fixed history length the model is trained and scored on.
    pub max_seq_len: usize,
    /// Size of the relative-recency position table (log₂ `days_ago`
    /// buckets; 32 covers ~11 million years, so it is effectively "all").
    #[config(default = 32)]
    pub num_position_buckets: usize,
    /// Number of attention heads per transformer block.
    pub num_heads: usize,
    /// Number of stacked transformer-encoder blocks.
    pub num_layers: usize,
    /// Dropout probability. Set to 0.0 for deterministic forward passes.
    #[config(default = 0.1)]
    pub dropout: f64,
    /// Share of non-pad positions replaced by `[MASK]` during training.
    /// Recorded here so a saved model documents how it was trained; the
    /// data path (`build_masked_sequences`) is what actually applies it.
    #[config(default = 0.2)]
    pub mask_ratio: f64,
}

impl Bert4RecConfig {
    /// Validate the architecture knobs. burn would panic deep inside the
    /// attention module on `embedding_dim % num_heads != 0`; surfacing it
    /// here gives callers (and the Python layer) a readable error.
    pub fn check(&self) -> anyhow::Result<()> {
        if self.vocab_size <= NUM_RESERVED_TOKENS {
            anyhow::bail!(
                "Bert4RecConfig: vocab_size must exceed the {NUM_RESERVED_TOKENS} reserved \
                 tokens (got {})",
                self.vocab_size
            );
        }
        if self.num_heads == 0 {
            anyhow::bail!("Bert4RecConfig: num_heads must be >= 1");
        }
        if self.embedding_dim == 0 || !self.embedding_dim.is_multiple_of(self.num_heads) {
            anyhow::bail!(
                "Bert4RecConfig: embedding_dim ({}) must be a positive multiple of num_heads ({})",
                self.embedding_dim,
                self.num_heads
            );
        }
        if self.num_layers == 0 {
            anyhow::bail!("Bert4RecConfig: num_layers must be >= 1");
        }
        if self.max_seq_len == 0 {
            anyhow::bail!("Bert4RecConfig: max_seq_len must be >= 1");
        }
        if self.num_position_buckets == 0 {
            anyhow::bail!("Bert4RecConfig: num_position_buckets must be >= 1");
        }
        if !(0.0..=1.0).contains(&self.mask_ratio) {
            anyhow::bail!(
                "Bert4RecConfig: mask_ratio must be in [0, 1] (got {})",
                self.mask_ratio
            );
        }
        if !(0.0..1.0).contains(&self.dropout) {
            anyhow::bail!(
                "Bert4RecConfig: dropout must be in [0, 1) (got {})",
                self.dropout
            );
        }
        Ok(())
    }

    /// Build a [`Bert4Rec`] on `device` using burn's default initializers.
    ///
    /// Panics on an invalid config (see [`Bert4RecConfig::check`]) — call
    /// `check()` first on untrusted input.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Bert4Rec<B> {
        let item_embedding = EmbeddingConfig::new(self.vocab_size, self.embedding_dim).init(device);
        let position_embedding =
            EmbeddingConfig::new(self.num_position_buckets, self.embedding_dim).init(device);
        // Feed-forward inner dim follows the common 4 * d_model rule.
        let transformer = TransformerEncoderConfig::new(
            self.embedding_dim,
            self.embedding_dim * 4,
            self.num_heads,
            self.num_layers,
        )
        .with_dropout(self.dropout)
        .init(device);
        let output_projection = LinearConfig::new(self.embedding_dim, self.vocab_size).init(device);

        Bert4Rec {
            item_embedding,
            position_embedding,
            transformer,
            output_projection,
            max_seq_len: self.max_seq_len,
            num_position_buckets: self.num_position_buckets,
        }
    }
}

/// BERT4Rec model. Generic over the burn backend so the same definition
/// serves CPU inference (`NdArray`) and autodiff training
/// (`Autodiff<NdArray>`).
#[derive(Module, Debug)]
pub struct Bert4Rec<B: Backend> {
    item_embedding: Embedding<B>,
    position_embedding: Embedding<B>,
    transformer: TransformerEncoder<B>,
    output_projection: Linear<B>,
    max_seq_len: usize,
    num_position_buckets: usize,
}

impl<B: Backend> Bert4Rec<B> {
    fn check_shapes(&self, input: &Tensor<B, 2, Int>, positions: &Tensor<B, 2, Int>) {
        let dims = input.dims();
        assert_eq!(
            dims,
            positions.dims(),
            "Bert4Rec: `input` {:?} and `positions` {:?} must have identical shapes",
            dims,
            positions.dims()
        );
        assert!(
            dims[1] <= self.max_seq_len,
            "Bert4Rec: seq_len {} exceeds max_seq_len {}; truncate the sequence before calling",
            dims[1],
            self.max_seq_len
        );
    }

    /// Run the bidirectional encoder and return the hidden states
    /// `(batch, seq_len, embedding_dim)` *before* the output projection.
    ///
    /// `input` is `(batch, seq_len)` of tokens (`0` = pad, `1` = mask,
    /// `>= 2` = catalog item), `positions` the matching `(batch, seq_len)`
    /// recency buckets in `[0, num_position_buckets)`. Pad positions are
    /// excluded from attention as keys via burn's `mask_pad`; there is
    /// **no** autoregressive mask, so every position sees every other
    /// non-pad position.
    ///
    /// # Panics
    ///
    /// If the two shapes differ, `seq_len > max_seq_len`, or any token /
    /// bucket index is out of range for its embedding table (burn's
    /// `Embedding` panics on out-of-range lookups).
    pub fn encode(&self, input: Tensor<B, 2, Int>, positions: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.check_shapes(&input, &positions);
        let pad_mask = input.clone().equal_elem(PAD_TOKEN);

        let embedded =
            self.item_embedding.forward(input) + self.position_embedding.forward(positions);

        self.transformer
            .forward(TransformerEncoderInput::new(embedded).mask_pad(pad_mask))
    }

    /// Full forward pass: [`encode`](Self::encode) followed by the
    /// projection to raw logits `(batch, seq_len, vocab_size)`.
    pub fn forward(&self, input: Tensor<B, 2, Int>, positions: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.output_projection
            .forward(self.encode(input, positions))
    }

    /// Mean-pool [`encode`](Self::encode) hidden states over non-pad
    /// positions → `(batch, embedding_dim)`. Rows with no non-pad
    /// position pool to zero (the divisor is clamped to 1).
    pub fn pooled(&self, input: Tensor<B, 2, Int>, positions: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let [batch, _seq] = input.dims();
        let keep = input.clone().not_equal_elem(PAD_TOKEN).float(); // (b, t)
        let hidden = self.encode(input, positions); // (b, t, d)
        let dim = hidden.dims()[2];
        let summed = (hidden * keep.clone().unsqueeze_dim::<3>(2))
            .sum_dim(1)
            .reshape([batch, dim]);
        let counts = keep.sum_dim(1).clamp_min(1.0); // (b, 1)
        summed / counts
    }

    /// Cloze loss: full-softmax cross-entropy over the vocabulary at the
    /// masked positions only. Targets at non-masked positions are
    /// overwritten with `PAD` and dropped via `with_pad_tokens`, so
    /// context and padding contribute neither loss nor gradient.
    pub fn forward_masked_loss(&self, batch: MaskedBatch<B>) -> Tensor<B, 1> {
        let [b, t] = batch.inputs.dims();
        let logits = self.forward(batch.inputs, batch.positions); // (b, t, vocab)
        let vocab = logits.dims()[2];

        let not_masked = batch.mask.equal_elem(0);
        let targets = batch.original.mask_fill(not_masked, PAD_TOKEN);

        let logits_2d = logits.reshape([b * t, vocab]);
        let targets_1d = targets.reshape([b * t]);

        CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![PAD_TOKEN as usize]))
            .init(&logits_2d.device())
            .forward(logits_2d, targets_1d)
    }
}

// --- Training ------------------------------------------------------------
//
// A hand-rolled Adam loop (the `two_tower.rs` pattern rather than SASRec's
// `Learner`): no artifact directory, no metric store, and the validation
// pass runs on the plain inner backend — which also turns dropout off
// (burn's `Dropout` is a no-op unless `B::ad_enabled()`), so the
// early-stopping signal is deterministic for a given parameter state.

use burn::data::dataloader::batcher::Batcher;

/// One training example, as produced by the data path: four aligned
/// token rows of length `seq_len`.
#[derive(Debug, Clone)]
pub struct MaskedItem {
    pub input: Vec<i64>,
    pub position: Vec<i64>,
    pub original: Vec<i64>,
    pub mask: Vec<i64>,
}

impl MaskedItem {
    /// Copy row `i` out of a [`MaskedSequenceDataset`].
    pub fn from_dataset(ds: &MaskedSequenceDataset, i: usize) -> Self {
        Self {
            input: ds.input_row(i).to_vec(),
            position: ds.position_row(i).to_vec(),
            original: ds.original_row(i).to_vec(),
            mask: ds.mask_row(i).to_vec(),
        }
    }
}

/// A collated batch: every field is `(batch, seq_len)`.
#[derive(Debug, Clone)]
pub struct MaskedBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub positions: Tensor<B, 2, Int>,
    pub original: Tensor<B, 2, Int>,
    pub mask: Tensor<B, 2, Int>,
}

/// Stateless batcher turning `[MaskedItem]` into a tensor batch.
#[derive(Clone, Debug, Default)]
pub struct MaskedBatcher;

fn stack_rows<B: Backend>(
    items: &[MaskedItem],
    pick: impl Fn(&MaskedItem) -> &[i64],
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let batch = items.len();
    let seq_len = items.first().map(|it| it.input.len()).unwrap_or(0);
    let mut flat = Vec::with_capacity(batch * seq_len);
    for it in items {
        flat.extend_from_slice(pick(it));
    }
    Tensor::<B, 1, Int>::from_data(TensorData::new(flat, [batch * seq_len]), device)
        .reshape([batch, seq_len])
}

impl<B: Backend> Batcher<B, MaskedItem, MaskedBatch<B>> for MaskedBatcher {
    fn batch(&self, items: Vec<MaskedItem>, device: &B::Device) -> MaskedBatch<B> {
        MaskedBatch {
            inputs: stack_rows(&items, |it| &it.input, device),
            positions: stack_rows(&items, |it| &it.position, device),
            original: stack_rows(&items, |it| &it.original, device),
            mask: stack_rows(&items, |it| &it.mask, device),
        }
    }
}

/// Knobs for [`train_bert4rec`].
#[derive(Config, Debug)]
pub struct Bert4RecTrainingConfig {
    #[config(default = 50)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 1e-3)]
    pub learning_rate: f64,
    /// Early-stopping patience (epochs without valid-loss improvement).
    #[config(default = 5)]
    pub patience: usize,
    #[config(default = 42)]
    pub seed: u64,
}

/// Train a BERT4Rec model on `dataset`, returning the fitted (inference)
/// model with the autodiff wrapper stripped.
///
/// Mini-batch Adam over the Cloze loss; batch order is reshuffled every
/// epoch from a `StdRng` seeded with `train_config.seed`. After each
/// epoch the mean loss over the same dataset is recomputed on the inner
/// (non-autodiff, dropout-off) backend and used for early stopping: the
/// run stops once that loss has not improved for `patience` consecutive
/// epochs, and the **best-loss** parameters are returned rather than the
/// last ones. (The masking pattern is fixed by the data path, so
/// "validation" here is the training objective evaluated without
/// dropout — the same convention SASRec's learner uses on in-memory
/// datasets.)
pub fn train_bert4rec<B: AutodiffBackend>(
    model_config: &Bert4RecConfig,
    train_config: &Bert4RecTrainingConfig,
    dataset: &MaskedSequenceDataset,
    device: &B::Device,
) -> anyhow::Result<Bert4Rec<B::InnerBackend>> {
    use burn::optim::{AdamConfig, GradientsParams, Optimizer};
    use rand::SeedableRng;
    use rand::seq::SliceRandom;

    model_config.check()?;
    if dataset.is_empty() {
        anyhow::bail!("train_bert4rec: dataset is empty (no users with >= 2 interactions)");
    }
    if dataset.seq_len > model_config.max_seq_len {
        anyhow::bail!(
            "train_bert4rec: dataset seq_len {} exceeds model max_seq_len {}",
            dataset.seq_len,
            model_config.max_seq_len
        );
    }
    if dataset.vocab_size != model_config.vocab_size {
        anyhow::bail!(
            "train_bert4rec: dataset vocab_size {} != model vocab_size {}",
            dataset.vocab_size,
            model_config.vocab_size
        );
    }
    if train_config.batch_size == 0 {
        anyhow::bail!("train_bert4rec: batch_size must be >= 1");
    }

    B::seed(device, train_config.seed);

    let items: Vec<MaskedItem> = (0..dataset.len())
        .map(|i| MaskedItem::from_dataset(dataset, i))
        .collect();
    let batcher = MaskedBatcher;

    let mut model: Bert4Rec<B> = model_config.init(device);
    let mut optim = AdamConfig::new().init();
    let mut rng = rand::rngs::StdRng::seed_from_u64(train_config.seed);
    let mut order: Vec<usize> = (0..items.len()).collect();

    let mut best_loss = f64::INFINITY;
    let mut best_model: Bert4Rec<B::InnerBackend> = model.valid();
    let mut epochs_since_improvement = 0usize;

    for epoch in 0..train_config.num_epochs {
        order.shuffle(&mut rng);
        for chunk in order.chunks(train_config.batch_size) {
            let batch_items: Vec<MaskedItem> = chunk.iter().map(|&i| items[i].clone()).collect();
            let batch: MaskedBatch<B> = batcher.batch(batch_items, device);
            let loss = model.forward_masked_loss(batch);
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(train_config.learning_rate, model, grads);
        }

        // Validation pass on the inner backend (dropout off).
        let inner = model.valid();
        let mut total = 0.0_f64;
        let mut n_batches = 0usize;
        for chunk in items.chunks(train_config.batch_size) {
            let batch: MaskedBatch<B::InnerBackend> = batcher.batch(chunk.to_vec(), device);
            let l: f64 = inner.forward_masked_loss(batch).into_scalar().elem();
            total += l;
            n_batches += 1;
        }
        let valid_loss = total / n_batches.max(1) as f64;
        log::info!(
            "bert4rec epoch {}/{}: valid_loss={valid_loss:.5}",
            epoch + 1,
            train_config.num_epochs
        );

        if valid_loss.is_finite() && valid_loss < best_loss {
            best_loss = valid_loss;
            best_model = inner;
            epochs_since_improvement = 0;
        } else {
            epochs_since_improvement += 1;
            if epochs_since_improvement >= train_config.patience {
                log::info!(
                    "bert4rec early stop at epoch {} (no improvement for {} epochs)",
                    epoch + 1,
                    train_config.patience
                );
                break;
            }
        }
    }

    Ok(best_model)
}

// --- Trained, ready-to-serve wrapper + serialization ---------------------
//
// Framed single file `FB4R || version[u32] || meta_len[u64] ||
// bincode(meta) || w_len[u64] || burn-recorded params`, mirroring the
// `FSAT` / `FTWO` layouts. The meta block embeds the string-id mappings
// so the file is self-describing.

use crate::data_pipeline::Mappings;
use crate::model::ValidationReport;
use crate::models::{ModelInput, ModelKind, RecModel};
use anyhow::{Context, bail};
use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

/// CPU inference backend for a served BERT4Rec model.
type B4InfB = NdArray<f32>;

/// BERT4Rec file magic. Distinct from EASE's `FEAS`, SASRec's `FSAS` /
/// `FSAT`, and Two-Tower's `FTWO` so a loader can auto-detect the model
/// type from the header.
pub const BERT4REC_MAGIC: &[u8; 4] = b"FB4R";

/// Serialization format version for BERT4Rec files.
pub const BERT4REC_FORMAT_VERSION: u32 = 1;

/// Bincode header persisted alongside the burn params blob — everything
/// needed to reconstruct the architecture and translate indices back to
/// catalog ids before loading weights.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Bert4RecMeta {
    pub version: u32,
    pub vocab_size: usize,
    pub embedding_dim: usize,
    pub max_seq_len: usize,
    pub num_position_buckets: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub dropout: f64,
    pub mask_ratio: f64,
    pub idx_to_item: Vec<String>,
    pub idx_to_user: Vec<String>,
}

impl Bert4RecMeta {
    fn to_config(&self) -> Bert4RecConfig {
        Bert4RecConfig::new(
            self.vocab_size,
            self.max_seq_len,
            self.num_heads,
            self.num_layers,
        )
        .with_embedding_dim(self.embedding_dim)
        .with_num_position_buckets(self.num_position_buckets)
        .with_dropout(self.dropout)
        .with_mask_ratio(self.mask_ratio)
    }
}

/// A trained BERT4Rec model on the CPU `NdArray` backend, carrying the
/// catalog mappings so it satisfies [`RecModel`]. `Clone` lets the
/// Python `ModelRegistry.register_bert4rec(...)` path hand a copy to the
/// registry without taking ownership of the source model.
#[derive(Clone)]
pub struct TrainedBert4Rec {
    model: Bert4Rec<B4InfB>,
    config: Bert4RecConfig,
    mappings: Mappings,
    device: NdArrayDevice,
}

impl TrainedBert4Rec {
    /// Wrap a fitted model + config + mappings.
    pub fn new(model: Bert4Rec<B4InfB>, config: Bert4RecConfig, mappings: Mappings) -> Self {
        Self {
            model,
            config,
            mappings,
            device: NdArrayDevice::default(),
        }
    }

    /// Number of catalog items: `vocab_size - 2` (`PAD` + `MASK`).
    pub fn num_items(&self) -> usize {
        self.config.vocab_size.saturating_sub(NUM_RESERVED_TOKENS)
    }

    /// Length of the vectors returned by [`embed_user`](Self::embed_user)
    /// and the rows of [`embed_items`](Self::embed_items).
    pub fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }

    /// The model's fixed history length.
    pub fn config_max_seq_len(&self) -> usize {
        self.config.max_seq_len
    }

    /// Size of the relative-recency position table.
    pub fn num_position_buckets(&self) -> usize {
        self.config.num_position_buckets
    }

    /// Turn a `(catalog idx, bucket)` history into a left-padded
    /// `(tokens, positions)` row of width `max_seq_len`, keeping the most
    /// recent entries. Out-of-catalog indices are skipped; buckets are
    /// clamped to the position table. When `positions` is shorter than
    /// `history`, missing buckets default to `0` ("now"). With
    /// `append_mask`, a `[MASK]` token at bucket `0` is appended as the
    /// final (query) position before truncation.
    fn prepare_row(
        &self,
        history_idx: &[usize],
        positions: &[usize],
        append_mask: bool,
    ) -> (Vec<i64>, Vec<i64>) {
        let n_items = self.num_items();
        let cap = self.config.num_position_buckets - 1;
        let mut pairs: Vec<(i64, i64)> = history_idx
            .iter()
            .enumerate()
            .filter(|(_, i)| **i < n_items)
            .map(|(k, &i)| {
                let bucket = positions.get(k).copied().unwrap_or(0).min(cap);
                (item_to_token(i), bucket as i64)
            })
            .collect();
        if append_mask {
            pairs.push((MASK_TOKEN, 0));
        }

        let seq_len = self.config.max_seq_len;
        let take = pairs.len().min(seq_len);
        let recent = &pairs[pairs.len() - take..];
        let start = seq_len - take;

        let mut toks = vec![PAD_TOKEN; seq_len];
        let mut pos = vec![0_i64; seq_len];
        for (k, (t, p)) in recent.iter().enumerate() {
            toks[start + k] = *t;
            pos[start + k] = *p;
        }
        (toks, pos)
    }

    fn row_tensors(
        &self,
        toks: Vec<i64>,
        pos: Vec<i64>,
    ) -> (Tensor<B4InfB, 2, Int>, Tensor<B4InfB, 2, Int>) {
        let seq_len = toks.len();
        let t = Tensor::<B4InfB, 1, Int>::from_data(TensorData::new(toks, [seq_len]), &self.device)
            .reshape([1, seq_len]);
        let p = Tensor::<B4InfB, 1, Int>::from_data(TensorData::new(pos, [seq_len]), &self.device)
            .reshape([1, seq_len]);
        (t, p)
    }

    /// Dense user embedding: run the **unmasked** history through the
    /// encoder and mean-pool the hidden states over non-pad positions.
    ///
    /// `history` is catalog item indices oldest-first, `positions` the
    /// matching recency buckets (see [`days_ago_to_bucket`]). Returns a
    /// vector of length [`embedding_dim`](Self::embedding_dim); an empty
    /// (or fully out-of-catalog) history yields all zeros, which is what
    /// the hybrid pipeline's cold-start convention expects (#96 §Key
    /// Design Decisions #5).
    pub fn embed_user(&self, history: &[usize], positions: &[usize]) -> Vec<f32> {
        let (toks, pos) = self.prepare_row(history, positions, false);
        if toks.iter().all(|&t| t == PAD_TOKEN) {
            return vec![0.0; self.config.embedding_dim];
        }
        let (t, p) = self.row_tensors(toks, pos);
        self.model
            .pooled(t, p)
            .reshape([self.config.embedding_dim])
            .into_data()
            .convert::<f32>()
            .into_vec()
            .expect("pooled tensor -> Vec<f32>")
    }

    /// Batch user-embedding extraction straight from an interactions
    /// file (`user_id`, `item_id`, **`days_ago`**). Each user's
    /// in-catalog history is ordered oldest-first and bucketed exactly as
    /// at training time, then passed through [`embed_user`](Self::embed_user).
    /// Users with no in-catalog interaction are absent from the result.
    /// The map is ordered by user id, so iteration is deterministic.
    pub fn embed_users_batch(
        &self,
        interactions_path: &str,
    ) -> anyhow::Result<BTreeMap<String, Vec<f32>>> {
        let histories = read_user_histories(interactions_path, &self.mappings)?;
        let buckets = self.config.num_position_buckets;
        let mut out = BTreeMap::new();
        for (user, hist) in histories {
            let items: Vec<usize> = hist.iter().map(|(_, i)| *i).collect();
            let pos: Vec<usize> = hist
                .iter()
                .map(|(d, _)| days_ago_to_bucket(*d, buckets))
                .collect();
            out.insert(user, self.embed_user(&items, &pos));
        }
        Ok(out)
    }

    /// The learned item-embedding table, one row per catalog item in
    /// index order (the `PAD` / `MASK` rows are dropped).
    pub fn embed_items(&self) -> Vec<Vec<f32>> {
        let weight = self.model.item_embedding.weight.val(); // (vocab, dim)
        let [vocab, dim] = weight.dims();
        let flat: Vec<f32> = weight
            .into_data()
            .convert::<f32>()
            .into_vec()
            .expect("embedding weight -> Vec<f32>");
        (NUM_RESERVED_TOKENS..vocab)
            .map(|r| flat[r * dim..(r + 1) * dim].to_vec())
            .collect()
    }

    /// Score every catalog item for a history (oldest first) with the
    /// standard BERT4Rec serving recipe: append `[MASK]` as the final
    /// token (bucket `0`, "now") and read the logits at that position.
    /// Returns `num_items` scores aligned with catalog indices (the
    /// `PAD` / `MASK` logit slots are dropped).
    fn score_items(&self, history: &[usize], positions: &[usize]) -> Vec<f32> {
        let (toks, pos) = self.prepare_row(history, positions, true);
        let seq_len = toks.len();
        let (t, p) = self.row_tensors(toks, pos);
        let logits = self.model.forward(t, p); // (1, seq_len, vocab)
        let vocab = logits.dims()[2];
        let last = logits.slice([0..1, seq_len - 1..seq_len]).reshape([vocab]);
        last.into_data()
            .convert::<f32>()
            .into_vec()
            .expect("logits tensor -> Vec<f32>")
            .into_iter()
            .skip(NUM_RESERVED_TOKENS)
            .collect()
    }

    /// Serialize to the framed `FB4R` format.
    pub fn save_to(&self, path: &Path) -> anyhow::Result<()> {
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let weights: Vec<u8> = recorder
            .record(self.model.clone().into_record(), ())
            .context("failed to record BERT4Rec params")?;

        let meta = Bert4RecMeta {
            version: BERT4REC_FORMAT_VERSION,
            vocab_size: self.config.vocab_size,
            embedding_dim: self.config.embedding_dim,
            max_seq_len: self.config.max_seq_len,
            num_position_buckets: self.config.num_position_buckets,
            num_heads: self.config.num_heads,
            num_layers: self.config.num_layers,
            dropout: self.config.dropout,
            mask_ratio: self.config.mask_ratio,
            idx_to_item: self.mappings.idx_to_item.clone(),
            idx_to_user: self.mappings.idx_to_user.clone(),
        };
        let meta_bytes =
            bincode::serialize(&meta).context("failed to serialize BERT4Rec metadata")?;

        let mut out = Vec::with_capacity(4 + 4 + 8 + meta_bytes.len() + 8 + weights.len());
        out.write_all(BERT4REC_MAGIC)?;
        out.write_all(&BERT4REC_FORMAT_VERSION.to_le_bytes())?;
        out.write_all(&(meta_bytes.len() as u64).to_le_bytes())?;
        out.write_all(&meta_bytes)?;
        out.write_all(&(weights.len() as u64).to_le_bytes())?;
        out.write_all(&weights)?;

        std::fs::write(path, &out)
            .with_context(|| format!("failed to write BERT4Rec model to {}", path.display()))?;
        Ok(())
    }

    /// Load a model written by [`TrainedBert4Rec::save_to`].
    pub fn load_from(path: &Path) -> anyhow::Result<Self> {
        let data = std::fs::read(path)
            .with_context(|| format!("failed to read BERT4Rec model from {}", path.display()))?;

        let take = |lo: usize, len: usize| -> anyhow::Result<&[u8]> {
            let hi = lo
                .checked_add(len)
                .filter(|&h| h <= data.len())
                .ok_or_else(|| {
                    anyhow::anyhow!("truncated or corrupt BERT4Rec file: {}", path.display())
                })?;
            Ok(&data[lo..hi])
        };

        if take(0, 4)? != BERT4REC_MAGIC {
            bail!(
                "invalid magic bytes in {}; expected FB4R header",
                path.display()
            );
        }
        let version = u32::from_le_bytes(take(4, 4)?.try_into().unwrap());
        if version != BERT4REC_FORMAT_VERSION {
            bail!(
                "unsupported BERT4Rec format version {version} (expected {BERT4REC_FORMAT_VERSION})"
            );
        }
        let mut off = 8usize;
        let meta_len = u64::from_le_bytes(take(off, 8)?.try_into().unwrap()) as usize;
        off += 8;
        let meta: Bert4RecMeta =
            bincode::deserialize(take(off, meta_len)?).context("deserialize Bert4RecMeta")?;
        off += meta_len;
        if meta.version != version {
            bail!(
                "BERT4Rec metadata version {} disagrees with header version {version}",
                meta.version
            );
        }
        let w_len = u64::from_le_bytes(take(off, 8)?.try_into().unwrap()) as usize;
        off += 8;
        let weights = take(off, w_len)?.to_vec();

        let config = meta.to_config();
        config.check()?;

        let device = NdArrayDevice::default();
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let record = recorder
            .load(weights, &device)
            .context("failed to load BERT4Rec params blob")?;
        let model: Bert4Rec<B4InfB> = config.init(&device).load_record(record);

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
            user_feature_to_idx: ahash::AHashMap::new(),
            idx_to_user_feature: Vec::new(),
            item_feature_to_idx: ahash::AHashMap::new(),
            idx_to_item_feature: Vec::new(),
        };

        Ok(Self {
            model,
            config,
            mappings,
            device,
        })
    }
}

impl RecModel for TrainedBert4Rec {
    fn kind(&self) -> ModelKind {
        ModelKind::Bert4Rec
    }

    fn num_items(&self) -> usize {
        TrainedBert4Rec::num_items(self)
    }

    fn item_mapping(&self) -> &Mappings {
        &self.mappings
    }

    fn predict_scores(&self, input: ModelInput<'_>) -> anyhow::Result<Vec<f32>> {
        match input {
            ModelInput::MaskedHistory { history, positions } => {
                Ok(self.score_items(history, positions))
            }
            // Order/recency-aware: the eval harness routes through
            // `Bert4RecEvalAdapter`, which sorts by `days_ago` and
            // derives the position buckets. Every other variant is a
            // caller error rather than something to guess at.
            ModelInput::Sparse { .. } => Err(anyhow::anyhow!(
                "BERT4Rec does not support ModelInput::Sparse; expected ModelInput::MaskedHistory \
                 (use crate::evaluation::Bert4RecEvalAdapter for time-aware eval)"
            )),
            ModelInput::Sequence { .. } => Err(anyhow::anyhow!(
                "BERT4Rec does not support ModelInput::Sequence; expected ModelInput::MaskedHistory \
                 (a plain sequence carries no recency buckets)"
            )),
            ModelInput::TowerUser { .. } => Err(anyhow::anyhow!(
                "BERT4Rec does not support ModelInput::TowerUser; expected ModelInput::MaskedHistory"
            )),
        }
    }

    fn predict_similar_items(
        &self,
        item_idx: usize,
        top_k: usize,
    ) -> anyhow::Result<Vec<(usize, f32)>> {
        let n = self.num_items();
        if item_idx >= n {
            return Err(anyhow::anyhow!(
                "item_idx {item_idx} out of range (num_items = {n})"
            ));
        }
        // Cosine similarity over the learned item-embedding rows. Token
        // `idx + 2` is the embedding for catalog item `idx`.
        let weight = self.model.item_embedding.weight.val(); // (vocab, dim)
        let dim = weight.dims()[1];
        let norm = weight
            .clone()
            .powf_scalar(2.0)
            .sum_dim(1)
            .sqrt()
            .clamp_min(1e-12);
        let normed = weight / norm;
        let q_tok = item_idx + NUM_RESERVED_TOKENS;
        let q = normed.clone().slice([q_tok..q_tok + 1, 0..dim]);
        let sims: Vec<f32> = (normed * q.reshape([1, dim]))
            .sum_dim(1)
            .reshape([self.config.vocab_size])
            .into_data()
            .convert::<f32>()
            .into_vec()
            .expect("similarity tensor -> Vec<f32>");
        let mut ranked: Vec<(usize, f32)> = sims
            .into_iter()
            .enumerate()
            // Skip the reserved slots and the query item itself.
            .filter(|(tok, _)| *tok >= NUM_RESERVED_TOKENS && *tok != q_tok)
            .map(|(tok, s)| (tok - NUM_RESERVED_TOKENS, s))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(top_k);
        Ok(ranked)
    }

    fn validate(&self) -> ValidationReport {
        let mut messages = Vec::new();
        let mut passed = true;
        if let Err(e) = self.config.check() {
            passed = false;
            messages.push(e.to_string());
        }
        if self.num_items() == 0 {
            passed = false;
            messages.push("BERT4Rec model has zero catalog items".to_string());
        }
        if self.mappings.idx_to_item.len() != self.num_items() {
            passed = false;
            messages.push(format!(
                "mappings have {} items but vocab implies {}",
                self.mappings.idx_to_item.len(),
                self.num_items()
            ));
        }
        let [rows, cols] = self.model.item_embedding.weight.val().dims();
        if rows != self.config.vocab_size || cols != self.config.embedding_dim {
            passed = false;
            messages.push(format!(
                "item embedding is {rows}x{cols} but config says {}x{}",
                self.config.vocab_size, self.config.embedding_dim
            ));
        }
        ValidationReport { passed, messages }
    }

    fn save(&self, path: &Path) -> anyhow::Result<()> {
        self.save_to(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Autodiff;
    use burn_ndarray::{NdArray, NdArrayDevice};

    type TestBackend = NdArray<f32>;
    type TrainBackend = Autodiff<NdArray<f32>>;

    /// vocab=10 (8 items + PAD + MASK), dim=16, seq_len=8, heads=2,
    /// layers=2, 8 position buckets, no dropout.
    fn test_config() -> Bert4RecConfig {
        Bert4RecConfig::new(10, 8, 2, 2)
            .with_embedding_dim(16)
            .with_num_position_buckets(8)
            .with_dropout(0.0)
    }

    fn zeros_pos(
        batch: usize,
        seq_len: usize,
        device: &NdArrayDevice,
    ) -> Tensor<TestBackend, 2, Int> {
        Tensor::<TestBackend, 2, Int>::zeros([batch, seq_len], device)
    }

    #[test]
    fn test_construction_succeeds() {
        let device = NdArrayDevice::default();
        let _model: Bert4Rec<TestBackend> = test_config().init(&device);
    }

    #[test]
    fn config_check_rejects_bad_head_split() {
        let bad = Bert4RecConfig::new(10, 8, 3, 2).with_embedding_dim(16);
        let err = bad.check().unwrap_err().to_string();
        assert!(err.contains("num_heads"), "unexpected error: {err}");
        assert!(test_config().check().is_ok());
    }

    #[test]
    fn test_forward_shape_is_batch_seq_vocab() {
        let device = NdArrayDevice::default();
        let model: Bert4Rec<TestBackend> = test_config().init(&device);
        let input = Tensor::<TestBackend, 2, Int>::from_data(
            [[0, 0, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 8, 9]],
            &device,
        );
        let logits = model.forward(input, zeros_pos(2, 8, &device));
        assert_eq!(logits.dims(), [2, 8, 10]);
    }

    #[test]
    fn test_forward_is_deterministic() {
        let device = NdArrayDevice::default();
        let model: Bert4Rec<TestBackend> = test_config().init(&device);
        let input = Tensor::<TestBackend, 2, Int>::from_data(
            [[2, 3, 4, 5, 6, 7, 8, 9], [9, 8, 7, 6, 5, 4, 3, 2]],
            &device,
        );
        let pos = Tensor::<TestBackend, 2, Int>::from_data(
            [[7, 6, 5, 4, 3, 2, 1, 0], [7, 6, 5, 4, 3, 2, 1, 0]],
            &device,
        );
        let a = model.forward(input.clone(), pos.clone());
        let b = model.forward(input, pos);
        let diff = (a - b).abs().max().into_scalar();
        assert!(
            diff < 1e-6,
            "forward pass must be deterministic, max diff = {diff}"
        );
    }

    #[test]
    fn test_forward_is_bidirectional() {
        // Changing ONLY the last token must change the logits at the
        // FIRST position. Under a causal mask position 0 can only see
        // itself, so this is impossible for SASRec and is the defining
        // property of BERT4Rec's encoder.
        let device = NdArrayDevice::default();
        let model: Bert4Rec<TestBackend> = test_config().init(&device);
        let a = Tensor::<TestBackend, 2, Int>::from_data([[2, 3, 4, 5, 6, 7, 8, 9]], &device);
        let b = Tensor::<TestBackend, 2, Int>::from_data([[2, 3, 4, 5, 6, 7, 8, 2]], &device);
        let pos = zeros_pos(1, 8, &device);
        let la = model.forward(a, pos.clone());
        let lb = model.forward(b, pos);
        let first_a = la.slice([0..1, 0..1]).reshape([10]);
        let first_b = lb.slice([0..1, 0..1]).reshape([10]);
        let diff = (first_a - first_b).abs().max().into_scalar();
        assert!(
            diff > 1e-6,
            "position-0 logits must depend on the last token (bidirectional), diff = {diff}"
        );
    }

    #[test]
    fn position_buckets_change_the_output() {
        let device = NdArrayDevice::default();
        let model: Bert4Rec<TestBackend> = test_config().init(&device);
        let input = Tensor::<TestBackend, 2, Int>::from_data([[2, 3, 4, 5, 6, 7, 8, 9]], &device);
        let p0 = zeros_pos(1, 8, &device);
        let p1 = Tensor::<TestBackend, 2, Int>::from_data([[7, 6, 5, 4, 3, 2, 1, 0]], &device);
        let diff = (model.forward(input.clone(), p0) - model.forward(input, p1))
            .abs()
            .max()
            .into_scalar();
        assert!(diff > 1e-6, "recency buckets must influence logits");
    }

    #[test]
    fn pad_positions_do_not_leak_into_pooling() {
        // Same real tokens, different amount of left padding → the
        // pooled embedding must be identical (pads are masked out of
        // attention as keys and excluded from the mean).
        let device = NdArrayDevice::default();
        let model: Bert4Rec<TestBackend> = test_config().init(&device);
        let padded = Tensor::<TestBackend, 2, Int>::from_data([[0, 0, 0, 0, 0, 2, 3, 4]], &device);
        let unpadded = Tensor::<TestBackend, 2, Int>::from_data([[2, 3, 4]], &device);
        let a = model.pooled(padded, zeros_pos(1, 8, &device));
        let b = model.pooled(unpadded, zeros_pos(1, 3, &device));
        assert_eq!(a.dims(), [1, 16]);
        assert_eq!(b.dims(), [1, 16]);
        let diff = (a - b).abs().max().into_scalar();
        assert!(
            diff < 1e-5,
            "padding must not change the pooled embedding, diff = {diff}"
        );
    }

    #[test]
    #[should_panic(expected = "exceeds max_seq_len")]
    fn forward_panics_when_seq_len_exceeds_max() {
        let device = NdArrayDevice::default();
        let model: Bert4Rec<TestBackend> = test_config().init(&device);
        let input = Tensor::<TestBackend, 2, Int>::zeros([1, 12], &device);
        let _ = model.forward(input, zeros_pos(1, 12, &device));
    }

    // --- Training + serialization ---

    /// A trivial Cloze dataset over tokens 2..=5 (4 catalog items):
    /// the full collection is always {2,3,4,5}, and the last one is
    /// masked. The model should learn "given {2,3,4}, the missing item
    /// is 5".
    fn tiny_dataset() -> MaskedSequenceDataset {
        let seq_len = 4;
        let row_orig = vec![2_i64, 3, 4, 5];
        let row_in = vec![2_i64, 3, 4, MASK_TOKEN];
        let row_mask = vec![0_i64, 0, 0, 1];
        let row_pos = vec![3_i64, 2, 1, 0];
        let mut original = Vec::new();
        let mut inputs = Vec::new();
        let mut mask = Vec::new();
        let mut positions = Vec::new();
        for _ in 0..8 {
            original.extend_from_slice(&row_orig);
            inputs.extend_from_slice(&row_in);
            mask.extend_from_slice(&row_mask);
            positions.extend_from_slice(&row_pos);
        }
        MaskedSequenceDataset {
            original,
            inputs,
            mask,
            positions,
            seq_len,
            vocab_size: 6, // tokens 0..=5
        }
    }

    fn tiny_model_config() -> Bert4RecConfig {
        Bert4RecConfig::new(6, 4, 2, 2)
            .with_embedding_dim(16)
            .with_num_position_buckets(8)
            .with_dropout(0.0)
    }

    fn train_tiny(seed: u64, epochs: usize) -> Bert4Rec<TestBackend> {
        let device = NdArrayDevice::default();
        let tcfg = Bert4RecTrainingConfig::new()
            .with_num_epochs(epochs)
            .with_batch_size(8)
            .with_learning_rate(1e-2)
            .with_patience(epochs)
            .with_seed(seed);
        train_bert4rec::<TrainBackend>(&tiny_model_config(), &tcfg, &tiny_dataset(), &device)
            .expect("training must succeed")
    }

    #[test]
    fn test_overfits_tiny_masked_dataset() {
        // Same multi-seed robustness pattern as SASRec's overfit test
        // (#48, #90): burn's NdArray init is not reseeded from `B::seed`,
        // so require 2 successes within 5 seeds rather than one lucky run.
        let device = NdArrayDevice::default();
        let mut correct = 0;
        for seed in [1_u64, 2, 3, 4, 5] {
            if correct >= 2 {
                break;
            }
            let model = train_tiny(seed, 200);
            let input = Tensor::<TestBackend, 2, Int>::from_data([[2, 3, 4, MASK_TOKEN]], &device);
            let pos = Tensor::<TestBackend, 2, Int>::from_data([[3, 2, 1, 0]], &device);
            let logits = model.forward(input, pos);
            let last: Vec<f32> = logits
                .slice([0..1, 3..4])
                .reshape([6])
                .into_data()
                .convert::<f32>()
                .into_vec()
                .unwrap();
            let (best, _) = last
                .iter()
                .enumerate()
                .skip(NUM_RESERVED_TOKENS)
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            if best == 5 {
                correct += 1;
            }
        }
        assert!(
            correct >= 2,
            "overfit model should recover token 5 at the masked slot in a majority of \
             runs; only {correct} of up to 5 seeds did"
        );
    }

    #[test]
    fn train_rejects_mismatched_dataset() {
        let device = NdArrayDevice::default();
        let tcfg = Bert4RecTrainingConfig::new().with_num_epochs(1);
        let mut ds = tiny_dataset();
        ds.vocab_size = 7;
        assert!(train_bert4rec::<TrainBackend>(&tiny_model_config(), &tcfg, &ds, &device).is_err());
        let empty = MaskedSequenceDataset {
            original: vec![],
            inputs: vec![],
            mask: vec![],
            positions: vec![],
            seq_len: 4,
            vocab_size: 6,
        };
        assert!(
            train_bert4rec::<TrainBackend>(&tiny_model_config(), &tcfg, &empty, &device).is_err()
        );
    }

    // --- TrainedBert4Rec wrapper + RecModel ---

    /// Mappings for a 4-item catalog (`a,b,c,d` -> idx 0..4). Tokens are
    /// `idx + 2`, so item idx 3 == token 5.
    fn tiny_mappings() -> Mappings {
        let items = ["a", "b", "c", "d"];
        let users = ["u0"];
        Mappings {
            user_to_idx: users
                .iter()
                .enumerate()
                .map(|(i, s)| (s.to_string(), i))
                .collect(),
            idx_to_user: users.iter().map(|s| s.to_string()).collect(),
            item_to_idx: items
                .iter()
                .enumerate()
                .map(|(i, s)| (s.to_string(), i))
                .collect(),
            idx_to_item: items.iter().map(|s| s.to_string()).collect(),
            user_feature_to_idx: Default::default(),
            idx_to_user_feature: Default::default(),
            item_feature_to_idx: Default::default(),
            idx_to_item_feature: Default::default(),
        }
    }

    fn trained_tiny() -> TrainedBert4Rec {
        // Structural / determinism / roundtrip properties only: a short
        // run suffices.
        TrainedBert4Rec::new(train_tiny(42, 10), tiny_model_config(), tiny_mappings())
    }

    #[test]
    fn test_recmodel_kind_and_num_items() {
        let t = trained_tiny();
        assert_eq!(t.kind(), ModelKind::Bert4Rec);
        // vocab 6 (tokens 0..=5) -> 4 catalog items.
        assert_eq!(RecModel::num_items(&t), 4);
        assert_eq!(t.item_mapping().idx_to_item.len(), 4);
        assert_eq!(t.embedding_dim(), 16);
    }

    #[test]
    fn masked_history_scores_catalog_items() {
        let t = trained_tiny();
        let scores = t
            .predict_scores(ModelInput::MaskedHistory {
                history: &[0, 1, 2],
                positions: &[3, 2, 1],
            })
            .expect("MaskedHistory must be supported");
        assert_eq!(scores.len(), 4, "scores must align with num_items");
        assert!(scores.iter().all(|s| s.is_finite()), "scores: {scores:?}");
        let again = t
            .predict_scores(ModelInput::MaskedHistory {
                history: &[0, 1, 2],
                positions: &[3, 2, 1],
            })
            .unwrap();
        assert_eq!(scores, again, "scoring must be deterministic");
        // Out-of-catalog indices and over-long bucket lists are tolerated.
        let loose = t
            .predict_scores(ModelInput::MaskedHistory {
                history: &[0, 99, 2],
                positions: &[500, 1],
            })
            .unwrap();
        assert_eq!(loose.len(), 4);
    }

    #[test]
    fn test_recmodel_rejects_sparse_input() {
        let t = trained_tiny();
        let inter = [(0usize, 1.0f64), (1, 1.0)];
        let r = t.predict_scores(ModelInput::Sparse {
            interactions: &inter,
            user_features: &[],
        });
        assert!(r.is_err(), "Sparse should be rejected, got {r:?}");
    }

    #[test]
    fn test_recmodel_rejects_sequence_input() {
        let t = trained_tiny();
        let r = t.predict_scores(ModelInput::Sequence { history: &[0, 1] });
        assert!(r.is_err(), "Sequence should be rejected, got {r:?}");
        let r = t.predict_scores(ModelInput::TowerUser {
            user_idx: None,
            cat_features: &[],
            dense_features: &[],
        });
        assert!(r.is_err(), "TowerUser should be rejected, got {r:?}");
    }

    #[test]
    fn test_embed_user_shape_matches_config() {
        let t = trained_tiny();
        let e = t.embed_user(&[0, 1, 2], &[3, 2, 1]);
        assert_eq!(e.len(), 16);
        assert!(e.iter().all(|v| v.is_finite()));
        // Empty / out-of-catalog history → zero vector of the same length.
        assert_eq!(t.embed_user(&[], &[]), vec![0.0; 16]);
        assert_eq!(t.embed_user(&[42], &[0]), vec![0.0; 16]);
    }

    #[test]
    fn test_embed_user_deterministic() {
        let t = trained_tiny();
        let a = t.embed_user(&[0, 1, 2], &[3, 2, 1]);
        let b = t.embed_user(&[0, 1, 2], &[3, 2, 1]);
        assert_eq!(a, b);
        // Padding amount must not matter: a 2-item history embeds the
        // same whether or not it is preceded by pads (always is here).
        let c = t.embed_user(&[0, 1], &[3, 2]);
        assert_ne!(a, c, "different histories should embed differently");
    }

    #[test]
    fn embed_items_has_one_row_per_catalog_item() {
        let t = trained_tiny();
        let rows = t.embed_items();
        assert_eq!(rows.len(), 4);
        assert!(rows.iter().all(|r| r.len() == 16));
    }

    #[test]
    fn embed_users_batch_reads_days_ago_and_orders_users() {
        let t = trained_tiny();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("i.csv");
        std::fs::write(
            &path,
            "user_id,item_id,value,days_ago\n\
             zed,a,1.0,10.0\n\
             zed,b,1.0,1.0\n\
             amy,c,1.0,3.0\n\
             nobody,UNKNOWN,1.0,1.0\n",
        )
        .unwrap();
        let out = t.embed_users_batch(path.to_str().unwrap()).unwrap();
        let users: Vec<&String> = out.keys().collect();
        assert_eq!(users, vec!["amy", "zed"]);
        assert!(out.values().all(|v| v.len() == 16));
        // Same user, same history via the direct API: a=idx0 (10d → bucket 3),
        // b=idx1 (1d → bucket 0), oldest first.
        assert_eq!(out["zed"], t.embed_user(&[0, 1], &[3, 0]));

        let bad = dir.path().join("no_days.csv");
        std::fs::write(&bad, "user_id,item_id,value\nzed,a,1.0\n").unwrap();
        let err = t.embed_users_batch(bad.to_str().unwrap()).unwrap_err();
        assert!(err.to_string().contains("days_ago"));
    }

    #[test]
    fn test_recmodel_similar_items_excludes_query() {
        let t = trained_tiny();
        let sim = t.predict_similar_items(0, 2).expect("similar");
        assert!(sim.len() <= 2);
        assert!(sim.iter().all(|(i, _)| *i != 0));
        assert!(sim.iter().all(|(i, _)| *i < 4));
        assert!(t.predict_similar_items(7, 2).is_err());
    }

    #[test]
    fn test_save_load_roundtrip_identical_scores() {
        let t = trained_tiny();
        let before = t
            .predict_scores(ModelInput::MaskedHistory {
                history: &[0, 1, 2],
                positions: &[3, 2, 1],
            })
            .unwrap();
        let emb_before = t.embed_user(&[0, 1, 2], &[3, 2, 1]);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trained.fb4r");
        t.save_to(&path).expect("save");
        assert_eq!(&std::fs::read(&path).unwrap()[..4], BERT4REC_MAGIC);

        let loaded = TrainedBert4Rec::load_from(&path).expect("load");
        assert_eq!(loaded.item_mapping().idx_to_item, vec!["a", "b", "c", "d"]);
        assert_eq!(loaded.item_mapping().idx_to_user, vec!["u0"]);
        assert_eq!(loaded.num_position_buckets(), 8);
        assert_eq!(loaded.config_max_seq_len(), 4);

        let after = loaded
            .predict_scores(ModelInput::MaskedHistory {
                history: &[0, 1, 2],
                positions: &[3, 2, 1],
            })
            .unwrap();
        assert_eq!(before.len(), after.len());
        for (i, (x, y)) in before.iter().zip(after.iter()).enumerate() {
            assert!((x - y).abs() < 1e-5, "score drift at {i}: {x} vs {y}");
        }
        let emb_after = loaded.embed_user(&[0, 1, 2], &[3, 2, 1]);
        for (i, (x, y)) in emb_before.iter().zip(emb_after.iter()).enumerate() {
            assert!((x - y).abs() < 1e-5, "embedding drift at {i}: {x} vs {y}");
        }
    }

    #[test]
    fn test_load_rejects_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.fb4r");
        std::fs::write(&path, b"NOPEnot-a-bert4rec-file").unwrap();
        let msg = match TrainedBert4Rec::load_from(&path) {
            Ok(_) => panic!("expected load to reject bad magic"),
            Err(e) => e.to_string(),
        };
        assert!(msg.contains("magic"), "unexpected error: {msg}");

        let sasrec = dir.path().join("other.fsat");
        std::fs::write(&sasrec, b"FSATxxxxxxxx").unwrap();
        assert!(TrainedBert4Rec::load_from(&sasrec).is_err());
    }

    #[test]
    fn trained_validate_passes_for_consistent_model() {
        let t = trained_tiny();
        let report = t.validate();
        assert!(report.passed, "validation messages: {:?}", report.messages);
    }

    const _: fn() = || {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TrainedBert4Rec>();
    };
}
