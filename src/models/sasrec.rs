//! SASRec — causal self-attention sequence recommender.
//!
//! Phase 2b (issue #25): a minimal model that compiles and runs a
//! forward pass on a tiny input. No training, no PyO3, no `RecModel`
//! impl yet — those land in Phase 3 and Phase 4. See ADR-0001.
//!
//! Architecture (Kang & McAuley, 2018 "Self-Attentive Sequential
//! Recommendation"): item embedding + learned positional embedding →
//! N causal transformer-encoder blocks → linear projection back to the
//! item vocabulary. `forward` returns raw logits `(batch, seq_len,
//! vocab)`; softmax happens later in the training loss (Phase 3) and
//! the recommendation scoring path (Phase 4).
//!
//! The backend stays generic (`SasRec<B: Backend>`); tests instantiate
//! with `NdArray` from `burn-ndarray`. Causal masking uses burn's
//! `generate_autoregressive_mask` passed through `TransformerEncoderInput`
//! — see the research findings on issue #25 and burn's text-generation
//! example for the canonical structure.

use burn::config::Config;
use burn::module::Module;
use burn::nn::attention::generate_autoregressive_mask;
use burn::nn::transformer::{
    TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

/// Hyperparameters for [`SasRec`]. Construction-only knobs (vocab size,
/// dims, depth); training-side params (learning rate, etc.) arrive in
/// Phase 3.
#[derive(Config, Debug)]
pub struct SasRecConfig {
    /// Number of distinct items (the output logit dimension).
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
        // Feed-forward inner dim follows the common 4 * d_model rule;
        // Phase 3 can revisit if training is unstable.
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

/// Minimal SASRec model. Generic over the burn backend so the same
/// definition serves CPU inference (`NdArray`) now and autodiff training
/// later (Phase 3).
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
    ///   clamp or a cryptic shape-mismatch panic. Truncation policy
    ///   (keep-most-recent) belongs to the Phase 3/4 data path.
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::{NdArray, NdArrayDevice};

    type TestBackend = NdArray<f32>;

    fn test_config() -> SasRecConfig {
        // vocab=10, dim=16, seq_len=8, heads=2, layers=2 per issue #25.
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
        // Same config/seed-free init path → identical params; dropout 0.0
        // → no stochasticity in the forward pass.
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

        // seq_len = 12 > max_seq_len = 8: the positional embedding is
        // undefined past max_seq_len, so this is an explicit caller error.
        let input = Tensor::<TestBackend, 2, Int>::zeros([2, 12], &device);
        let _ = model.forward(input);
    }
}
