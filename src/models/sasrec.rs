//! SASRec — causal self-attention sequence recommender.
//!
//! Architecture (Kang & McAuley, 2018 "Self-Attentive Sequential
//! Recommendation"): item embedding + learned positional embedding →
//! N causal transformer-encoder blocks → linear projection back to the
//! item vocabulary. `forward` returns raw logits `(batch, seq_len,
//! vocab)`; softmax is applied by the training loss and the
//! recommendation scoring path.
//!
//! The model trains by mini-batch SGD (Adam) over fixed-length,
//! left-padded causal sequences (see `crate::data::sequences`) and
//! persists via burn's `Recorder` inside the framed `FSAS` file format.
//! It is gated behind the default-off `ml-models` Cargo feature.
//!
//! The backend stays generic (`SasRec<B: Backend>`); inference uses
//! `NdArray` and training uses `Autodiff<NdArray>`. Causal masking uses
//! burn's `generate_autoregressive_mask` passed through
//! `TransformerEncoderInput` — see burn's text-generation example for
//! the canonical structure.

use burn::config::Config;
use burn::module::Module;
use burn::nn::attention::generate_autoregressive_mask;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::transformer::{
    TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor};
use burn::train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep};

/// Hyperparameters for [`SasRec`]. Construction-only knobs (vocab size,
/// dims, depth); training-side params live in [`SasRecTrainingConfig`].
#[derive(Config, Debug)]
pub struct SasRecConfig {
    /// Number of distinct items (the output logit dimension). Includes
    /// the reserved pad token at index 0 (see `data::sequences`).
    pub vocab_size: usize,
    /// Embedding / model dimension `d_model`.
    pub embedding_dim: usize,
    /// Maximum sequence length the positional embedding supports.
    pub max_seq_len: usize,
    /// Number of attention heads per transformer block.
    pub num_heads: usize,
    /// Number of stacked transformer-encoder blocks.
    pub num_layers: usize,
    /// Dropout probability. Set to 0.0 for deterministic forward passes.
    #[config(default = 0.2)]
    pub dropout: f64,
}

impl SasRecConfig {
    /// Build a [`SasRec`] on `device` using burn's default initializers.
    pub fn init<B: Backend>(&self, device: &B::Device) -> SasRec<B> {
        let item_embedding = EmbeddingConfig::new(self.vocab_size, self.embedding_dim).init(device);
        let positional_embedding =
            EmbeddingConfig::new(self.max_seq_len, self.embedding_dim).init(device);
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

        SasRec {
            item_embedding,
            positional_embedding,
            transformer,
            output_projection,
            max_seq_len: self.max_seq_len,
        }
    }
}

/// SASRec model. Generic over the burn backend so the same definition
/// serves CPU inference (`NdArray`) and autodiff training
/// (`Autodiff<NdArray>`).
#[derive(Module, Debug)]
pub struct SasRec<B: Backend> {
    item_embedding: Embedding<B>,
    positional_embedding: Embedding<B>,
    transformer: TransformerEncoder<B>,
    output_projection: Linear<B>,
    max_seq_len: usize,
}

impl<B: Backend> SasRec<B> {
    /// Run the causal forward pass.
    ///
    /// `sequence` is `(batch, seq_len)` of item indices, oldest first.
    /// Returns logits `(batch, seq_len, vocab_size)` — position `t`'s
    /// logits depend only on positions `≤ t` thanks to the
    /// autoregressive attention mask.
    ///
    /// # Panics
    ///
    /// - If `seq_len > max_seq_len`: the learned positional embedding is
    ///   only defined on `[0, max_seq_len)`. An over-long sequence is a
    ///   caller error, surfaced explicitly here rather than via a silent
    ///   clamp or a cryptic shape-mismatch panic. Truncation
    ///   (keep-most-recent) is the data path's responsibility.
    /// - If any index in `sequence` is `≥ vocab_size`: burn's `Embedding`
    ///   lookup panics on out-of-range indices, so callers must pass
    ///   item indices in `[0, vocab_size)`.
    pub fn forward(&self, sequence: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch_size, seq_len] = sequence.dims();
        let device = sequence.device();

        assert!(
            seq_len <= self.max_seq_len,
            "SASRec.forward: seq_len {seq_len} exceeds max_seq_len {} \
             (positional embedding is only defined on [0, max_seq_len)); \
             truncate the sequence before calling",
            self.max_seq_len
        );

        // Item embeddings: (batch, seq_len, d_model).
        let item_embedded = self.item_embedding.forward(sequence);

        // Learned positional embeddings for [0, seq_len), broadcast over
        // the batch. seq_len <= max_seq_len is guaranteed by the assert above.
        let positions =
            Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device).reshape([1, seq_len]);
        let positions = positions.repeat_dim(0, batch_size);
        let position_embedded = self.positional_embedding.forward(positions);

        let embedded = item_embedded + position_embedded;

        // Causal self-attention: position t attends only to positions ≤ t.
        let mask_attn = generate_autoregressive_mask::<B>(batch_size, seq_len, &device);
        let encoded = self
            .transformer
            .forward(TransformerEncoderInput::new(embedded).mask_attn(mask_attn));

        // Project each position back to the item vocabulary (raw logits).
        self.output_projection.forward(encoded)
    }

    /// Score every item for a single user's history (inference path).
    ///
    /// `history` is item *tokens* (catalog idx + 1; `0` = padding),
    /// oldest first. The sequence is right-aligned / left-padded to
    /// `max_seq_len` and the logits at the final (most recent) position
    /// are returned — `vocab_size` raw scores including the pad slot at
    /// index 0. An empty history returns the no-context logits.
    pub fn score_history(&self, history: &[usize], device: &B::Device) -> Vec<f32> {
        let seq_len = self.max_seq_len;
        let mut row = vec![0_i64; seq_len];
        let take = history.len().min(seq_len);
        if take > 0 {
            for (k, tok) in history[history.len() - take..].iter().enumerate() {
                row[seq_len - take + k] = *tok as i64;
            }
        }
        let input = Tensor::<B, 1, Int>::from_data(row.as_slice(), device).reshape([1, seq_len]);
        let logits = self.forward(input); // (1, seq_len, vocab)
        let vocab = logits.dims()[2];
        let last = logits.slice([0..1, seq_len - 1..seq_len]).reshape([vocab]);
        last.into_data()
            .convert::<f32>()
            .into_vec()
            .expect("logits tensor -> Vec<f32>")
    }
}

// --- Training ------------------------------------------------------------
//
// Training is mini-batch SGD via burn's supervised learner: implement
// `TrainStep`/`InferenceStep`, wire `SupervisedTraining` + `AdamConfig` +
// loss-based early stopping. burn 0.21 uses associated `Input`/`Output`
// types on `TrainStep` and a `SupervisedTraining`/`Learner` split.
// Loss is full-softmax cross-entropy over the item vocabulary with the
// next-item target (input shifted by one); the padding token 0 is masked
// from the loss via `CrossEntropyLossConfig::with_pad_tokens`, so padded
// positions contribute neither attention nor gradient.

use crate::data::sequences::{PAD_TOKEN, SequenceDataset};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::tensor::TensorData;

/// One training example: a left-padded input sequence and its next-item
/// targets (the input shifted by one, produced by the data path).
#[derive(Debug, Clone)]
pub struct SeqItem {
    pub input: Vec<i64>,
    pub target: Vec<i64>,
}

/// In-memory dataset of [`SeqItem`]s built from a [`SequenceDataset`].
#[derive(Debug)]
pub struct SeqDataset {
    items: Vec<SeqItem>,
}

impl SeqDataset {
    pub fn from_sequences(ds: &SequenceDataset) -> Self {
        let items = (0..ds.len())
            .map(|i| SeqItem {
                input: ds.input_row(i).to_vec(),
                target: ds.target_row(i).to_vec(),
            })
            .collect();
        Self { items }
    }
}

impl Dataset<SeqItem> for SeqDataset {
    fn get(&self, index: usize) -> Option<SeqItem> {
        self.items.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.items.len()
    }
}

/// A collated batch: `inputs`/`targets` are `(batch, seq_len)`.
#[derive(Debug, Clone)]
pub struct SeqBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

/// Stateless batcher turning `[SeqItem]` into a padded tensor batch.
#[derive(Clone, Debug, Default)]
pub struct SeqBatcher;

impl<B: Backend> Batcher<B, SeqItem, SeqBatch<B>> for SeqBatcher {
    fn batch(&self, items: Vec<SeqItem>, device: &B::Device) -> SeqBatch<B> {
        let batch = items.len();
        let seq_len = items.first().map(|it| it.input.len()).unwrap_or(0);

        let mut in_flat = Vec::with_capacity(batch * seq_len);
        let mut tgt_flat = Vec::with_capacity(batch * seq_len);
        for it in &items {
            in_flat.extend_from_slice(&it.input);
            tgt_flat.extend_from_slice(&it.target);
        }

        let inputs =
            Tensor::<B, 1, Int>::from_data(TensorData::new(in_flat, [batch * seq_len]), device)
                .reshape([batch, seq_len]);
        let targets =
            Tensor::<B, 1, Int>::from_data(TensorData::new(tgt_flat, [batch * seq_len]), device)
                .reshape([batch, seq_len]);

        SeqBatch { inputs, targets }
    }
}

impl<B: Backend> SasRec<B> {
    /// Full-softmax cross-entropy next-item loss with padding masked.
    ///
    /// Logits `(batch, seq, vocab)` and targets `(batch, seq)` are
    /// flattened to `(batch*seq, vocab)` / `(batch*seq)`;
    /// `CrossEntropyLoss` with `pad_tokens = [PAD_TOKEN]` drops positions
    /// whose target is padding. Returns burn's built-in
    /// [`ClassificationOutput`] so the standard `LossMetric` /
    /// early-stopping machinery works without a hand-rolled `ItemLazy`.
    pub fn forward_loss(&self, batch: SeqBatch<B>) -> ClassificationOutput<B> {
        let [b, t] = batch.inputs.dims();
        let logits = self.forward(batch.inputs); // (b, t, vocab)
        let vocab = logits.dims()[2];

        let logits_2d = logits.reshape([b * t, vocab]);
        let targets_1d = batch.targets.reshape([b * t]);

        let loss = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![PAD_TOKEN]))
            .init(&logits_2d.device())
            .forward(logits_2d.clone(), targets_1d.clone());

        ClassificationOutput::new(loss, logits_2d, targets_1d)
    }
}

// burn 0.21's `TrainStep`/`InferenceStep` use associated `Input`/`Output`
// types. Validation is `InferenceStep` on the *inner* (non-autodiff)
// module, which `#[derive(Module)]` provides automatically; the `Display`
// also required by the `Learner` bound is likewise derived.
impl<B: AutodiffBackend> TrainStep for SasRec<B> {
    type Input = SeqBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: SeqBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let out = self.forward_loss(batch);
        let grads = out.loss.backward();
        TrainOutput::new(self, grads, out)
    }
}

impl<B: Backend> InferenceStep for SasRec<B> {
    type Input = SeqBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: SeqBatch<B>) -> ClassificationOutput<B> {
        self.forward_loss(batch)
    }
}

/// Knobs for [`train_sasrec`].
#[derive(Config, Debug)]
pub struct SasRecTrainingConfig {
    #[config(default = 50)]
    pub num_epochs: usize,
    #[config(default = 16)]
    pub batch_size: usize,
    #[config(default = 1e-3)]
    pub learning_rate: f64,
    /// Early-stopping patience (epochs without valid-loss improvement).
    #[config(default = 5)]
    pub patience: usize,
    #[config(default = 42)]
    pub seed: u64,
}

/// Train a SASRec model on `dataset`, returning the fitted (inference)
/// model with the autodiff wrapper stripped.
///
/// Builds train/valid `DataLoader`s (the same dataset serves both on
/// small in-memory datasets), then wires burn's `SupervisedTraining`
/// with Adam and loss-based early stopping.
pub fn train_sasrec<B: AutodiffBackend>(
    model_config: &SasRecConfig,
    train_config: &SasRecTrainingConfig,
    dataset: &SequenceDataset,
    device: &B::Device,
) -> anyhow::Result<SasRec<B::InnerBackend>> {
    use burn::data::dataloader::DataLoaderBuilder;
    use burn::optim::AdamConfig;
    use burn::train::metric::LossMetric;
    use burn::train::metric::store::{Aggregate, Direction, Split};
    use burn::train::{
        Learner, MetricEarlyStoppingStrategy, StoppingCondition, SupervisedTraining,
    };

    if dataset.is_empty() {
        anyhow::bail!("train_sasrec: dataset is empty (no users with >= 2 interactions)");
    }

    B::seed(device, train_config.seed);

    let batcher = SeqBatcher;
    let loader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(train_config.batch_size)
        .shuffle(train_config.seed)
        .build(SeqDataset::from_sequences(dataset));
    let loader_valid = DataLoaderBuilder::new(batcher)
        .batch_size(train_config.batch_size)
        .build(SeqDataset::from_sequences(dataset));

    let artifact_dir = tempfile::tempdir()
        .map_err(|e| anyhow::anyhow!("failed to create learner artifact dir: {e}"))?;

    let model: SasRec<B> = model_config.init(device);
    let learner = Learner::new(model, AdamConfig::new().init(), train_config.learning_rate);

    let stop_metric = LossMetric::<B>::new();
    let result = SupervisedTraining::new(artifact_dir.path(), loader_train, loader_valid)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .early_stopping(MetricEarlyStoppingStrategy::new(
            &stop_metric,
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince {
                n_epochs: train_config.patience,
            },
        ))
        .num_epochs(train_config.num_epochs)
        .summary()
        .launch(learner);

    // `LearningResult::model` is already the inner (non-autodiff) model.
    Ok(result.model)
}

// --- Serialization -------------------------------------------------------
//
// Framed single file `FSAS || version[u32] || meta_len[u64] ||
// bincode(meta) || w_len[u64] || burn-recorded params`, mirroring
// `serialization.rs`. The shared format-version line is 3; the distinct
// magic means there is no collision with EASE's `FEAS`, so loaders can
// auto-detect the model type from the header.

use anyhow::{Context, bail};
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::Path;

/// SASRec file magic. Distinct from EASE's `FEAS`.
pub const SASREC_MAGIC: &[u8; 4] = b"FSAS";

/// Serialization format version for SASRec files.
pub const SASREC_FORMAT_VERSION: u32 = 3;

/// Bincode header persisted alongside the burn params blob — everything
/// needed to reconstruct the architecture before loading weights.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct SasRecMeta {
    pub version: u32,
    pub vocab_size: usize,
    pub embedding_dim: usize,
    pub max_seq_len: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub dropout: f64,
}

impl SasRecMeta {
    fn from_config(c: &SasRecConfig) -> Self {
        Self {
            version: SASREC_FORMAT_VERSION,
            vocab_size: c.vocab_size,
            embedding_dim: c.embedding_dim,
            max_seq_len: c.max_seq_len,
            num_heads: c.num_heads,
            num_layers: c.num_layers,
            dropout: c.dropout,
        }
    }

    fn to_config(&self) -> SasRecConfig {
        SasRecConfig::new(
            self.vocab_size,
            self.embedding_dim,
            self.max_seq_len,
            self.num_heads,
            self.num_layers,
        )
        .with_dropout(self.dropout)
    }
}

/// Save a trained SASRec model to `path` in the framed `FSAS` format.
pub fn save_sasrec<B: Backend>(
    model: &SasRec<B>,
    config: &SasRecConfig,
    path: &Path,
) -> anyhow::Result<()> {
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    let weights: Vec<u8> = recorder
        .record(model.clone().into_record(), ())
        .context("failed to record SASRec params")?;

    let meta = bincode::serialize(&SasRecMeta::from_config(config))
        .context("failed to serialize SASRec metadata")?;

    let mut out = Vec::with_capacity(4 + 4 + 8 + meta.len() + 8 + weights.len());
    out.write_all(SASREC_MAGIC)?;
    out.write_all(&SASREC_FORMAT_VERSION.to_le_bytes())?;
    out.write_all(&(meta.len() as u64).to_le_bytes())?;
    out.write_all(&meta)?;
    out.write_all(&(weights.len() as u64).to_le_bytes())?;
    out.write_all(&weights)?;

    std::fs::write(path, &out)
        .with_context(|| format!("failed to write SASRec model to {}", path.display()))?;
    Ok(())
}

/// Load a SASRec model previously written by [`save_sasrec`].
pub fn load_sasrec<B: Backend>(
    path: &Path,
    device: &B::Device,
) -> anyhow::Result<(SasRec<B>, SasRecConfig)> {
    let data = std::fs::read(path)
        .with_context(|| format!("failed to read SASRec model from {}", path.display()))?;

    let need = |pos: usize, n: usize, total: usize| -> anyhow::Result<()> {
        if pos + n > total {
            bail!("truncated SASRec file: need {n} bytes at offset {pos}");
        }
        Ok(())
    };

    let mut pos = 0usize;
    need(pos, 4, data.len())?;
    if &data[..4] != SASREC_MAGIC {
        bail!(
            "invalid magic bytes in {}; expected FSAS header",
            path.display()
        );
    }
    pos += 4;

    need(pos, 4, data.len())?;
    let version = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;
    if version != SASREC_FORMAT_VERSION {
        bail!("unsupported SASRec format version {version} (expected {SASREC_FORMAT_VERSION})");
    }

    need(pos, 8, data.len())?;
    let meta_len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
    pos += 8;
    need(pos, meta_len, data.len())?;
    let meta: SasRecMeta = bincode::deserialize(&data[pos..pos + meta_len])
        .context("failed to deserialize SASRec metadata")?;
    pos += meta_len;

    need(pos, 8, data.len())?;
    let w_len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
    pos += 8;
    need(pos, w_len, data.len())?;
    let weights = data[pos..pos + w_len].to_vec();

    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    let record = recorder
        .load(weights, device)
        .context("failed to load SASRec params blob")?;

    let config = meta.to_config();
    let model = config.init::<B>(device).load_record(record);
    Ok((model, config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::{NdArray, NdArrayDevice};

    type TestBackend = NdArray<f32>;

    fn test_config() -> SasRecConfig {
        // vocab=10, dim=16, seq_len=8, heads=2, layers=2.
        SasRecConfig::new(10, 16, 8, 2, 2).with_dropout(0.0)
    }

    #[test]
    fn construction_succeeds() {
        let device = NdArrayDevice::default();
        let _model: SasRec<TestBackend> = test_config().init(&device);
    }

    #[test]
    fn forward_shape_is_batch_seq_vocab() {
        let device = NdArrayDevice::default();
        let model: SasRec<TestBackend> = test_config().init(&device);

        let batch = 2;
        let seq_len = 8;
        let input = Tensor::<TestBackend, 2, Int>::zeros([batch, seq_len], &device);

        let logits = model.forward(input);
        assert_eq!(logits.dims(), [batch, seq_len, 10]);
    }

    #[test]
    fn forward_is_deterministic() {
        let device = NdArrayDevice::default();
        let model: SasRec<TestBackend> = test_config().init(&device);

        let input = Tensor::<TestBackend, 2, Int>::from_data(
            [[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]],
            &device,
        );

        let out_a = model.forward(input.clone());
        let out_b = model.forward(input);

        let diff = (out_a - out_b).abs().max().into_scalar();
        assert!(
            diff < 1e-6,
            "forward pass must be deterministic, max diff = {diff}"
        );
    }

    #[test]
    fn forward_shorter_than_max_seq_len() {
        let device = NdArrayDevice::default();
        let model: SasRec<TestBackend> = test_config().init(&device);

        let batch = 2;
        let seq_len = 4; // < max_seq_len = 8
        let input = Tensor::<TestBackend, 2, Int>::zeros([batch, seq_len], &device);

        let logits = model.forward(input);
        assert_eq!(logits.dims(), [batch, seq_len, 10]);
    }

    #[test]
    #[should_panic(expected = "exceeds max_seq_len")]
    fn forward_panics_when_seq_len_exceeds_max() {
        let device = NdArrayDevice::default();
        let model: SasRec<TestBackend> = test_config().init(&device);

        let input = Tensor::<TestBackend, 2, Int>::zeros([2, 12], &device);
        let _ = model.forward(input);
    }

    // --- Training + serialization ---

    use crate::data::sequences::SequenceDataset;
    use burn::backend::Autodiff;

    type TrainBackend = Autodiff<NdArray<f32>>;

    /// A trivial deterministic next-item dataset: tokens `1,2,3,4`
    /// (vocab includes pad=0). The model should be able to memorize
    /// "given prefix, predict the next token".
    fn tiny_dataset() -> SequenceDataset {
        // seq_len = 4. input = [1,2,3,4], target = [2,3,4,0] (last has no
        // successor -> pad, masked from loss). Repeat so a mini-batch has
        // signal.
        let seq_len = 4;
        let row_in = vec![1_i64, 2, 3, 4];
        let row_tgt = vec![2_i64, 3, 4, 0];
        let mut inputs = Vec::new();
        let mut targets = Vec::new();
        for _ in 0..8 {
            inputs.extend_from_slice(&row_in);
            targets.extend_from_slice(&row_tgt);
        }
        SequenceDataset {
            inputs,
            targets,
            seq_len,
            vocab_size: 5, // tokens 0..=4
        }
    }

    fn tiny_model_config() -> SasRecConfig {
        SasRecConfig::new(5, 16, 4, 2, 2).with_dropout(0.0)
    }

    #[test]
    fn overfits_tiny_next_item_dataset() {
        let device = NdArrayDevice::default();
        let ds = tiny_dataset();
        let mcfg = tiny_model_config();
        let tcfg = SasRecTrainingConfig::new()
            .with_num_epochs(80)
            .with_batch_size(8)
            .with_learning_rate(1e-2)
            .with_patience(80);

        let model = train_sasrec::<TrainBackend>(&mcfg, &tcfg, &ds, &device)
            .expect("training must succeed");

        // Given prefix [1,2,3], the argmax next-item over real items
        // (excluding pad slot 0) should be token 4.
        let scores = model.score_history(&[1, 2, 3], &device);
        assert_eq!(scores.len(), 5);
        let (best, _) = scores
            .iter()
            .enumerate()
            .skip(1) // ignore pad slot
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assert_eq!(
            best, 4,
            "overfit model should predict token 4 after [1,2,3]; scores={scores:?}"
        );
    }

    #[test]
    fn save_load_roundtrip_identical_outputs() {
        let device = NdArrayDevice::default();
        let ds = tiny_dataset();
        let mcfg = tiny_model_config();
        let tcfg = SasRecTrainingConfig::new()
            .with_num_epochs(15)
            .with_batch_size(8)
            .with_patience(15);

        let model = train_sasrec::<TrainBackend>(&mcfg, &tcfg, &ds, &device)
            .expect("training must succeed");

        let before = model.score_history(&[1, 2, 3], &device);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sasrec.fsas");
        save_sasrec(&model, &mcfg, &path).expect("save must succeed");

        let (loaded, loaded_cfg): (SasRec<TestBackend>, _) =
            load_sasrec(&path, &device).expect("load must succeed");
        assert_eq!(loaded_cfg.vocab_size, mcfg.vocab_size);
        assert_eq!(loaded_cfg.max_seq_len, mcfg.max_seq_len);

        let after = loaded.score_history(&[1, 2, 3], &device);
        assert_eq!(before.len(), after.len());
        for (i, (a, b)) in before.iter().zip(after.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "score mismatch at {i} after roundtrip: {a} vs {b}"
            );
        }
    }

    #[test]
    fn load_rejects_bad_magic() {
        let device = NdArrayDevice::default();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.fsas");
        std::fs::write(&path, b"NOPEnot-a-sasrec-file").unwrap();
        let r = load_sasrec::<TestBackend>(&path, &device);
        assert!(r.is_err());
        assert!(r.unwrap_err().to_string().contains("magic"));
    }
}
