# FEASE Extension Research: Burn-based SASRec & Two-Tower

> Research-only reference. Compiled 2026-05-16 for the Burn-based SASRec /
> two-tower extension of FEASE. No code in this document has been wired into
> the crate; treat snippets as design guidance to verify against the pinned
> `burn` version at implementation time.

## 1. Burn version & Cargo.toml

Latest stable on crates.io as of 2026-05-16: **`burn` 0.21.0**, released
2026-05-07 (verified via crates.io API).

```toml
[dependencies]
burn = { version = "0.21", default-features = false, features = [
    "ndarray",      # NdArray CPU backend
    "autodiff",     # Autodiff<NdArray> wrapper for training
    "train",        # burn-train: Learner, metrics, checkpointing, early stopping
    "std",
] }
# Optional: native model store (safetensors / burnpack) – new in 0.21
burn-store = { version = "0.21", features = ["std"] }
```

Use `type B = Autodiff<NdArray<f32>>;` for training and `NdArray<f32>` for
inference. Rust edition 2024 is fine (Burn 0.21 builds on stable Rust ≥1.85).

**Breaking changes in last 2–3 releases:**
- **0.21.0 restructure:** `nn` modules split into a dedicated `burn-nn`
  crate; new `burn-store` crate (Safetensors/PyTorch/Burnpack with `ParamId`
  persistence). Facade re-exports preserved — `burn::nn::attention::MultiHeadAttention`
  still works, but doc/source paths moved.
- **0.21.0:** `Gelu` now configurable (must use `Gelu::new()`/`Gelu::default()`);
  explicit conv padding must be symmetric per pair.
- **0.19–0.20:** quantization, distributed collectives, LLVM backend;
  `burn-train` gained the supervised builder split. Optimizer/`Module` trait
  surface stable across these.

Sources: [crates.io burn](https://crates.io/crates/burn),
[0.21.0 notes](https://burn.dev/blog/release-0.21.0/),
[0.19.0 notes](https://burn.dev/blog/release-0.19.0/).

## 2. Attention API for SASRec

Burn **ships both** `MultiHeadAttention` and `TransformerEncoder` (verified
from v0.21.0 source `crates/burn-nn/src/modules/attention/mha.rs`):

- `burn::nn::attention::MultiHeadAttentionConfig { d_model, n_heads, dropout (=0.1), quiet_softmax }`;
  `.init::<B>(device) -> MultiHeadAttention<B>`.
- Input: `MhaInput::<B>::self_attn(Tensor<B,3>)` (`[batch, seq, d_model]`);
  `.mask_attn(Tensor<B,3,Bool>)` for causal, `.mask_pad(Tensor<B,2,Bool>)` for padding.
- `forward(MhaInput) -> MhaOutput { context: Tensor<B,3>, weights: Tensor<B,4> }`.
- Causal mask helper: `burn::nn::attention::generate_autoregressive_mask::<B>(batch, seq_len, device)`
  (lower-triangular).

```rust
use burn::nn::attention::{MultiHeadAttentionConfig, MhaInput, generate_autoregressive_mask};
use burn::nn::{LayerNorm, LayerNormConfig};
use burn::tensor::{Tensor, backend::Backend};

#[derive(burn::module::Module, Debug)]
pub struct SasBlock<B: Backend> {
    attn: burn::nn::attention::MultiHeadAttention<B>,
    norm1: LayerNorm<B>, norm2: LayerNorm<B>,
    ff1: burn::nn::Linear<B>, ff2: burn::nn::Linear<B>,
}

impl<B: Backend> SasBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, t, _] = x.dims();
        let mask = generate_autoregressive_mask::<B>(b, t, &x.device());
        let h = self.norm1.forward(x.clone());
        let a = self.attn.forward(MhaInput::self_attn(h).mask_attn(mask)).context;
        let x = x + a;                                   // residual
        let h = self.norm2.forward(x.clone());
        let f = self.ff2.forward(burn::tensor::activation::relu(self.ff1.forward(h)));
        x + f
    }
}
```

Alternatively use `burn::nn::transformer::TransformerEncoderConfig { d_model,
d_ff, n_heads, n_layers, dropout, norm_first, quiet_softmax }` with an
autoregressive mask via `TransformerEncoderInput`. Sources:
[MultiHeadAttention](https://burn.dev/docs/burn/nn/attention/struct.MultiHeadAttention.html),
[transformer module](https://burn.dev/docs/burn/nn/transformer/index.html).

## 3. Training-loop idiom

Implement `TrainStep`/`ValidStep`, wire through `LearnerBuilder` +
`AdamConfig`. Reference: Burn Book
[Training](https://burn.dev/books/burn/basic-workflow/training.html),
[Learner](https://burn.dev/burn-book/building-blocks/learner.html).

```rust
use burn::train::{TrainStep, ValidStep, TrainOutput, RegressionOutput, LearnerBuilder};
use burn::train::metric::LossMetric;
use burn::train::metric::store::{Aggregate, Direction, Split};
use burn::train::checkpoint::FileCheckpointer;
use burn::train::MetricEarlyStoppingStrategy;
use burn::optim::AdamConfig;

impl<B: AutodiffBackend> TrainStep<SeqBatch<B>, RegressionOutput<B>> for SasRec<B> {
    fn step(&self, batch: SeqBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let out = self.forward_loss(batch);
        TrainOutput::new(self, out.loss.backward(), out)
    }
}
impl<B: Backend> ValidStep<SeqBatch<B>, RegressionOutput<B>> for SasRec<B> {
    fn step(&self, batch: SeqBatch<B>) -> RegressionOutput<B> { self.forward_loss(batch) }
}

let learner = LearnerBuilder::new(artifact_dir)
    .metric_train_numeric(LossMetric::new())
    .metric_valid_numeric(LossMetric::new())
    .with_file_checkpointer(FileCheckpointer::new(/* recorder */, artifact_dir, "model"))
    .early_stopping(MetricEarlyStoppingStrategy::new::<LossMetric<B>>(
        Aggregate::Mean, Direction::Lowest, Split::Valid,
        burn::train::StoppingCondition::NoImprovementSince { n_epochs: 3 }))
    .devices(vec![device.clone()])
    .num_epochs(50)
    .build(model, AdamConfig::new().init(), 1e-3);

let trained = learner.fit(dataloader_train, dataloader_valid);
```

Batching: implement `Batcher<Item, SeqBatch<B>>` and feed
`DataLoaderBuilder::new(batcher).batch_size(N).shuffle(seed).build(dataset)`.

## 4. Save/Load via Recorder / Store

Two co-existing mechanisms in 0.21:
- **Legacy `Recorder`** (`burn::record`): `NamedMpkFileRecorder`/`BinFileRecorder`
  write self-contained files; **`BinBytesRecorder` records to/from `Vec<u8>`**
  (`record(item, ()) -> Vec<u8>`, `load(bytes, device)`). Owns its blob, no
  embedded serde metadata.
- **New `burn-store`**: `BurnpackStore`/`SafetensorsStore` with
  `model.save_into(&mut store)` / `load_from`; Burnpack persists `ParamId`
  (good for resuming training).

**For FEASE's existing magic-byte + bincode-header pattern**, use
`BinBytesRecorder` to serialize params into memory, then write your own framed
single file like `serialization.rs` does today:

```rust
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};

let rec = BinBytesRecorder::<FullPrecisionSettings>::default();
let weights: Vec<u8> = rec.record(model.into_record(), ()).unwrap();

// [b"FEAS"][u8 version][u64 meta_len][bincode meta][u64 w_len][weights]
let meta = bincode::serialize(&SeqModelMeta { /* .. */ })?;
out.write_all(b"FEAS")?; out.write_all(&[3u8])?;
out.write_all(&(meta.len() as u64).to_le_bytes())?; out.write_all(&meta)?;
out.write_all(&(weights.len() as u64).to_le_bytes())?; out.write_all(&weights)?;

// load: parse header → bincode meta → slice weights → rec.load(weights_vec, &device)
let record = rec.load(weights_vec, &device).unwrap();
let model = SasRecConfig::from(&meta).init(&device).load_record(record);
```

Single file, header readable without Burn, consistent with the existing
v1/v2 format (bump to v3). Sources:
[burn::record](https://docs.rs/burn/latest/burn/record/index.html),
[burn-store](https://docs.rs/burn-store).

## 5. Reference implementations

**SASRec** — Kang & McAuley, ICDM 2018.
[arXiv 1808.09781](https://arxiv.org/abs/1808.09781) ·
[PDF](https://cseweb.ucsd.edu/~jmcauley/pdfs/icdm18.pdf) ·
code [github.com/kang205/SASRec](https://github.com/kang205/SASRec).

Input is a fixed-length sequence of the last `n` item IDs (default 50/200),
left-padded with id 0. Item embedding table (**shared** with the
output/prediction layer) + a **learned** positional embedding added
elementwise. 2 self-attention blocks, each = causal MHA (1–2 heads) →
residual+LN → pointwise 2-layer FFN → residual+LN; dropout on embeddings and
within blocks. Causal mask: position `t` attends only to ≤`t`. Loss in the
paper is **binary cross-entropy with one sampled negative per positive**
(BPR-style pairwise also works); modern reimplementations often switch to
**sampled/full softmax cross-entropy** for better accuracy.

Pitfalls: (a) padding positions must be masked from attention *and* excluded
from the loss; (b) share input/output item embeddings; (c) BCE-with-1-negative
trains slowly/noisily — prefer sampled-softmax; (d) labels are the input
sequence shifted by one (predict the *next* item).

**Two-tower in-batch sampled softmax** — Yi et al., RecSys 2019
("Sampling-Bias-Corrected Neural Modeling"), in TF Recommenders. Explainer:
[In-Batch Negatives](http://srome.github.io/In-Batch-Negatives-For-Recommender-Systems/);
recent analysis: [arXiv 2507.09331](https://arxiv.org/abs/2507.09331).

For a batch of `B` (user `u_i`, positive `v_i`) pairs, score
`s_ij = <q(u_i), e(v_j)>`. In-batch softmax:
`L = -1/B Σ_i log( exp(s_ii) / Σ_j exp(s_ij) )`, all in-batch `j≠i` as
negatives. **log-Q correction is standard**: `s_ij ← s_ij − log p_j` where
`p_j` ≈ item `j`'s in-batch occurrence frequency (streaming estimate),
removing popularity bias before softmax.

Same-item-in-batch: if `v_j == v_i` for some `i≠j`, it's a **false negative**
— canonical fix is to mask it: set `s_ij = −inf` for `j≠i` where
`item_id_j == item_id_i`. Typically L2-normalize tower outputs and divide
logits by a temperature `τ`.

## Actionable takeaways

- Target `burn = "0.21"` (nn moved to `burn-nn`, facade re-exports intact).
- Use built-in `MultiHeadAttention` + `generate_autoregressive_mask`.
- Train via `LearnerBuilder` + `AdamConfig` + `MetricEarlyStoppingStrategy`.
- Persist with `BinBytesRecorder` inside FEASE's existing magic-byte+bincode
  framing (bump format to v3).
- SASRec = shared item embeddings + learned positional + 2 causal blocks;
  prefer sampled-softmax loss.
- Two-tower needs both logQ correction and same-item false-negative masking.
