# SilverTorch: A Unified Model-based System to Democratize Large-Scale Recommendation on GPUs

> **Reference notes** extracted from `silver_torch_research.pdf` (the binary is
> gitignored; this markdown is the committed, diffable copy of its content).
> Faithful summary of the paper — section structure, key numbers, and the two
> results tables are transcribed from the source. The "Relevance to
> `kzn_recsys`" box below is our own annotation, not part of the paper.

- **Authors**: Bi Xue\*, Hong Wu\*, Lei Chen\*, Chao Yang\* (equal contribution), et al. — 33 authors, Meta Platforms, Menlo Park, CA, USA
- **Venue**: SIGIR '26 (49th International ACM SIGIR Conference), July 20–24 2026, Melbourne, VIC, Australia
- **Preprint**: arXiv:2511.14881v5 [cs.IR], 8 May 2026
- **DOI**: https://doi.org/10.1145/3805712.3809755
- **CCS / Keywords**: Novelty in information retrieval · Recommendation Serving, GPU Index, Model-based Retrieval

---

> **Relevance to `kzn_recsys` (our annotation).** Directly informs
> [ADR-0004](../docs/adr/0004-retrieval-index-seam.md): SilverTorch is a
> production take on the same problem — embedding-based retrieval where you
> "can't rank all items." It validates several of our design instincts
> (Int8-quantized ANN for memory/latency, exclude/filter co-designed with the
> search) and pushes further (ANN as *in-model* tensor layers, a learned
> OverArch re-ranker over ANN candidates, GPU-native filtering). Its
> Probe-then-Filter equivalence argument and Int8 recall-loss observations are
> useful priors for our `RetrievalIndex` benchmark phase.

---

## Abstract

Serving deep-learning recommendation models (DLRM) at scale is hard. Existing
approaches rely on dedicated ANN indexing and filtering services on CPUs, which
carry non-negligible cost and miss co-design opportunities — making it hard to
support complex architectures like learned similarities and multi-task
retrieval. **SilverTorch** is a model-based serving system that brings all
components into **one unified model**, replacing standalone indexing and
filtering services with **model layers**. Contributions: a **model-based GPU
Bloom index** for feature filtering and a **fused Int8 ANN kernel** for nearest-
neighbor search; co-designing ANN + filtering to cut GPU memory and eliminate
computation; and scaling retrieval via an **OverArch scoring layer** and
**multi-task retrieval with a Value Model**. On industry-scale datasets,
SilverTorch reaches **up to 23.7× higher throughput** than state-of-the-art and
is **13.35× more cost-efficient** than a CPU solution *while improving accuracy*
by serving more complex models.

## 1. Introduction & Motivation

Embedding-based DLRM serving uses a multi-stage design because ranking all items
at inference is impossible:

1. **Retrieval** — narrow candidates to ~thousands by formulating an
   **Approximate Nearest Neighbor (ANN)** search in vector space (commonly
   Faiss, RAFT, or a vector DB like Milvus), plus **feature filtering** to honor
   user constraints (language, eligibility) via inverted indexes.
2. **Ranking** — downstream models score the retrieved set.

Drawbacks of the service-based approach:
- **Authoring divergence** across serving components slows end-to-end dev.
- **Isolated optimization** — each service re-implements versioning, scheduling,
  batching; optimization is fragmented.
- **Client orchestration** of multiple services adds latency from data movement.

Two further challenges: **compute scalability** (bigger candidate pools → bigger
indexes; architectures move beyond dot-product to learned similarities and
transformers over interaction history), and the fact that **no prior work studies
feature filtering on GPUs**, while GPU ANN libraries (Faiss-GPU, Milvus) support
only limited top-k / probes and are hard to customize for recommendation.

**Versioning failure example**: in production serving hundreds of millions of
items, committing user-tower modules takes minutes but building the item kNN
index takes **>4 hours**; the candidate pool versions independently. In 2022, an
incorrect version switch caused a **30% drop** in production metrics. Model-based
retrieval avoids this class of bug.

### Contributions
- **SilverTorch**: a unified model-based serving system in PyTorch; redefines
  standalone ANN + filtering as in-model tensor operators; simplifies client
  orchestration.
- A novel **GPU Bloom index** for filtering + a **fused Int8 ANN kernel**, plus a
  **co-designed index** combining the two — first attempt at feature filtering on
  GPUs and first to apply Bloom-style signatures to recommendation.
- **OverArch** learned-similarity scoring layer + **multi-task retrieval with
  Value Model**; item embeddings pre-computed and cached in the model.
- Evaluation on **10M** and **80M** item pools: up to **23.7×** throughput over
  SOTA, **>5.6%** recall improvement, **13.35×** more cost-efficient.

## 2. Background

- **Two-tower architecture**: User Tower and Item Tower map dense (numerical) and
  sparse (categorical) features into vector space; the interaction layer is
  usually simplified to a **dot product** for serving efficiency.
- Built on PyTorch / TorchRec / TensorFlow / Jax, compiled to GPU kernel graphs.
- GPU ANN libraries (Faiss-GPU, Milvus, CAGRA) cap top-k/probes (Faiss ≤2048
  probes; Milvus top-k ≤1024); modern retrieval feeds **tens of thousands** of
  ANN results into interaction layers, exceeding those limits.
- Bloom index is motivated by **Bitfunnel** (signature-based replacement for
  inverted index in text search) — but Bitfunnel is CPU-only and not designed for
  recommendation, where per-item feature cardinality is much smaller.
- Related GPU retrieval: **LiNR** (LinkedIn) does model-based GPU retrieval but
  lacks a co-designed index and a GPU filtering solution; **Merlin** combines GPU
  libraries across the stack rather than unifying them in one model.

**Table 1 — kNN & filtering systems**

| Name | Usage | GPU | Language | Limit | Service |
|------|-------|-----|----------|-------|---------|
| Elasticsearch | kNN + Filtering | N | Java | — | Y |
| System A (internal) | kNN + Filtering | N | C++ | — | Y |
| Faiss | kNN | Y | C++ | Top-k, nprobe ≤ 2048 | N |
| Milvus | kNN | Y | Go | Top-k ≤ 1024 | Y |

## 3. SilverTorch Overview

All serving components are defined as **model layers**; retrieval is just a
forward pass of tensor operators.

**Publish flow** (after training): load trained models → embedding evaluator
computes item embeddings on GPU → ANN index builder **quantizes to Int8** and
runs **KMeans++** clustering on GPU → Bloom index builder builds signature-based
GPU filtering index → User Tower / OverArch / Value Model **quantized to BFloat16**
as model parameters → model composer combines everything → optimizer compiles
eager-mode model to a graph (lowering + scripting). Output is a **model snapshot**
(weights + index tensors) served in a pure **C++ runtime ("predictor")**. Publish
time drops **from days to ~1 hour**.

**Serving flow**: predictor runs one forward pass — User Tower → user embedding;
Bloom index layer → mask tensor; ANN layer (user emb + mask) → ~**O(10,000)** item
ids; OverArch fetches cached item embeddings, re-ranks, aggregates via Value
Model across tasks → final **O(1,000)** ids to ranking.

## 4. Model Design

**Index as Model.** A query has the shape:

```
ANN_Index(user_emb) AND (feature1=v1 OR feature1=v2 ...) AND (...) ...
# concrete:
ANN_Index(user_emb) AND item_country = "US" AND (item_lang = "EN" OR item_lang = "ES")
```

ANN sub-query served by a **fused Int8 ANN kernel using IVF** (probe nearby
clusters → per-cluster top-k → global top-k). Filtering served by the **Bloom
index** (bitwise GPU computation). Item embeddings + static features are
**pre-computed and cached on GPU** to remove caching-service dependencies.

### 4.1 Bloom Index

Inverted index is a poor GPU fit: recommendation features are broad/dense (no
Zipfian sparsity to exploit) and posting lists are inherently sequential.
**Forward index** enables stateless, parallel per-item evaluation but has
irregular memory access and a large int64 footprint.

**Bloom Index** — an M-bit Bloom filter per item (`VB_i`) and per query (`QB`),
each feature hashed by `k` functions (`hash_i(feature) % N`). Match criterion:

```
R = { V_i ∈ V, V_i = VB_i | QB & VB_i == QB }
```

Optimizations: only examine bits set to 1 in `QB`; **transpose** the bit matrix so
each row is a bit position and each column packs 64 items' bit at that position →
a single **64-bit AND (PTX `and.b64`)** processes 64 items per instruction, storing
64 results in one Int64. For 40M items → 625,000 partitions. False positives
(hash collisions) are tuned low via M and K, and any survivors are removed by
later ranking stages.

### 4.2 Fused Int8 ANN Search

Tensor-native IVF kernel in three steps: (1) dot products between query and
centroid embeddings, (2) top items within selected clusters, (3) global top-k.
The **index-selection** step is the bottleneck (materializes a large temporary
gather tensor); solved with a **fused index-matmul operator** that streams item
embeddings from the table and computes dot products on-the-fly (no intermediate
tensor), one warp per contiguous item tile (coalesced access).

**Int8 quantization** at publish time (global min/max → scale to [−128, 127])
halves the memory footprint and raises throughput with **limited quality loss**;
freed memory lets OverArch score more candidates. Supports large top-k/probes —
**no measurable recall loss at 64 probes, top-2048**. (Caveat in §6.2.1: Int8
cannot reach 0.95 recall, but matching that on Faiss needs prohibitively many
probes.)

### 4.3 ANN + Filtering Co-design

Standard pipeline applies Bloom filtering to the *entire* pool before ANN — but
ANN only scores items in a few probed clusters, so filtering non-probed items is
wasted work. **Co-design reverses the order**: identify probed clusters first,
then filter only items within them. Because ANN probing and Bloom filtering are
**independent conjunctive predicates**, order doesn't change the result —
**Probe-then-Filter and Filter-then-Probe yield identical recall**. The ANN
operator consumes a **1-bit-per-item** mask (vs PyTorch's 8-bit bool), saving
bandwidth.

> **Algorithm 1 — Co-designed ANN Search with Partial Bloom Filtering**
> Inputs: query emb `q`, predicates `F`, item embs `E`, centroids `C`, offsets
> `O`, lengths `L`, Bloom index `B`, probes `n_p`, `k`.
> 1. **ANN probing**: `D ← Distance(q, C)`; `P ← TopK(D, n_p)`
> 2. **Partial filtering** on probed clusters: for `c ∈ P`, `M_c ← BloomSearch(B, F, O[c], L[c])`
> 3. **Fused scoring** with partial masks: for `c ∈ P`, item `d ∈ c` where `M_c[d]=1`: accumulate `⟨q, E[d]⟩`
> 4. **Global top-k**: `R ← ArgTopK(S, k)` → return item IDs + scores

For 81M items / 9,000 clusters at 256 probes: only **2.3M items** scored → **30×
reduction** in both filtering compute and GPU scratch memory.

## 5. Extensibility

### 5.1 OverArch Scoring
Dot-product two-tower similarity has no trainable parameters and oversimplifies
interactions. SilverTorch's cost savings fund a neural **OverArch** re-ranker:
pre-filter `K0` items (`O(10k)–O(100k)`) via dot-product ANN, then an MLP /
stacked self-attention / **Mixture-of-Logits (MoL)** re-ranks the `K0` pairs.
OverArch improves recall more than improving ANN accuracy alone. Item embeddings
and cross-features come straight from the GPU cache.

### 5.2 Multi-Task Retrieval with Value Model
User tower shares a lookup table but applies **task-specific dense layers** (per
task: like / share / comment); **item embeddings are shared** across tasks (they
are semantic representations). One index copy serves batched multi-task queries
without latency regression (vs CPU solutions that replicate the index per task →
linear cost). A **Value Model (VM)** aggregates per-task predictions into one
composite "expected value" score via user-defined JSON-like formulas parsed into
an AST and evaluated in parallel on GPU — improving retrieval/ranking
consistency.

### 5.3 Scale Out
ANN and Bloom indexes **sharded** across GPUs (each handles an item partition);
OverArch + VM **replicated** per GPU. Each GPU computes local pre-filtered
results; embeddings gathered to one GPU for the final result. User-embedding
tables live on **CPU parameter servers**. **QPS-based autoscaling** spins GPU
instances up/down within minutes; extreme bursts are throttled.

## 6. Evaluation

Datasets: **80M** and **10M** item pools, embedding dim **128**, **A100-40G**
GPUs. User tower = **HSTU**; OverArch = **Mixture of Logits**. Replay 5,000
production requests; measure max QPS under a **200 ms P99** budget; 5 repeats
averaged. Baselines: **Baseline-Retrieval** (Faiss-CPU IVF + CPU inverted index),
**Baseline-Retrieval-GPU** (Faiss-GPU IVF + GPU forward index), and SilverTorch
with/without OverArch+VM.

### 6.1 End-to-end
- **80M @ 24 probes** (production setting): SilverTorch **1210 QPS** — **23.7×**
  over CPU, **3.5×–6.7×** over GPU baselines.
- **10M, no sharding**: **3802 QPS** — **165.3×** over CPU, **20.8×** over GPU.
- **Co-design** (ProbeThenFilter) gives **~17–25% QPS** over FilterThenProbe.
- Model footprint on GPU: ~5 GB Bloom index + 5 GB ANN index + 10 GB embedding
  cache (FP16) + 12 GB OverArch/VM weights; user embedding on CPU.

**Table 2 — Cost-efficiency (serving 1000 requests, 80M dataset)**

| Baseline | QPS | TCO/Hour | TCO/1000 Req | Cost Efficiency |
|----------|-----|----------|--------------|-----------------|
| Baseline-CPU | 51 | $28.92 | $0.158 | 1× |
| Baseline-GPU (Faiss2-Forward4) | 186 | $30.29 | $0.045 | 3.48× |
| Baseline-GPU (Faiss2-Forward6) | 340 | $33.27 | $0.027 | 5.8× |
| **SilverTorch** | **1210** | $33.27 | $0.0077 | **20.9×** |
| **SilverTorch-OverArch** | 771 | $33.27 | $0.012 | **13.35×** |

- **Latency**: SilverTorch P99 stays ~**15 ms** regardless of traffic (not
  compute-saturated); baselines grow with QPS. At 32 probes / 10 QPS: 15.3 ms —
  11.4× over CPU, 1.6× over best GPU. ANN search stays ~**2 ms** under high
  traffic; service-based architecture wastes ~**18.9%** of latency on
  network/data transformation.

### 6.2 Component breakdowns
- **ANN** (20M items, dim 128, batch 16): SilverTorch-INT8 lowest latency in all
  cases — at top-k=2048, **2.2×–14.7×** lower than Faiss-GPU (recall 0.35–0.92);
  at top-k=4096 (Faiss-GPU can't support), **31.3×–51×** over HNSW and
  **4.6×–49.2×** over Faiss-CPU. Int8 can't reach 0.95 recall, but that recall
  needs Faiss-CPU 1024 probes / Faiss-GPU 512 probes.
- **Bloom index** (40M items, 6 features × ~10 values, 5 hash fns): **291×–523×**
  over inverted index, **12.6×–42.7×** over forward index; latency constant
  across bit sizes. 512 bits → 6.98% FP rate (1.2 GB); 1024 bits → 0.067% FP.
  Inverted index needs 19.8 GB (1.98× the 2048-bit Bloom). Bit-count heuristic:
  `max_feature_values × hash_functions × collision_buffer` (→ ~1800; they use
  **1024** for all experiments).
- **Co-design** (20M items): at probe=32, scratch memory 35.6 MB → **18.2 MB**
  (Bloom scratch 18.2 → 0.14 MB), latency 1.55 ms → 0.72 ms;
  **1.79×–2.15×** latency improvement overall.

**Table 3 — Recall (probes=32)**

| Task | Method | R@20 | R@100 | R@200 | R@500 | R@1000 | QPS |
|------|--------|------|-------|-------|-------|--------|-----|
| **E-Task** | Baseline | 0.08239 | 0.19179 | 0.29131 | 0.4295 | 0.44127 | 51 |
| | SilverTorch | 0.07163 | 0.20306 | 0.28923 | 0.4237 | 0.44651 | 1210 |
| | SilverTorch-OverArch (Low Bits) | 0.08513 | 0.25391 (+25.04%) | 0.3202 | 0.44301 | 0.4489 | 768 |
| | SilverTorch-OverArch | 0.09181 (+28.2%) | 0.24189 | 0.33148 (+14.6%) | 0.44758 (+5.6%) | 0.45727 (+2.4%) | 771 |
| **C-Task** | Baseline | 0.09642 | 0.25217 | 0.3551 | 0.4971 | 0.5162 | 51 |
| | SilverTorch | 0.09652 | 0.25291 | 0.352 | 0.4969 | 0.51973 | 1210 |
| | SilverTorch-OverArch (Low Bits) | 0.0992 | 0.25103 | 0.355 | 0.512 (+3%) | 0.526 | 768 |
| | SilverTorch-OverArch | 0.0971 (+0.6%) | 0.25733 (+1.7%) | 0.36011 (+2.3%) | 0.50747 | 0.52559 (+1.12%) | 771 |

OverArch+VM improves **E-Task recall 2.4%–35.5%** and **C-Task 1.12%–3%** while
QPS only drops 1210 → 771 (still **15.11×** over baseline). Notably, lowering
Bloom bits 1024 → 768 raises the FP rate (0.00173% → 3.89%) but **does not degrade
end-to-end recall** (occasionally slightly higher) — filtering is a post-hoc
guard, not part of training.

## 7. Discussion & Future Work
- Benefits smaller scales too: one GPU server replaces multiple CPU servers; tiny
  cases can fall back to **CPU PyTorch runtime** via config.
- Rule-based retrieval channels can be folded in via CPU key-value indexing as
  model layers or Bloom-index-only filtering, with OverArch scoring consistently
  across channels.
- **Probe-then-Filter** can under-cover clusters when filter selectivity is very
  high (inherent to IVF); recommendation filters are usually broad, and more
  probes mitigate niche cases.
- **Item freshness**: a self-contained **fresh index** with the same layout as the
  main index, independently/streaming-updated and merged before OverArch (avoids
  the linear-scan cost of append-only watermark approaches). Details left to
  future work.

## 8. Conclusion
SilverTorch is a GPU-native, model-based recommendation serving system that
folds ANN search and Bloom-index filtering into model layers, co-designed and
fused (Int8), extended with OverArch + Value Model. It serves millions of items
across GPUs with **23.7×** throughput and **13.35×** cost-efficiency over SOTA —
a step toward GPU-native recommendation serving.

## References

1. Abadi et al. 2016. *TensorFlow: a system for Large-Scale machine learning.* OSDI 16, 265–283.
2. Ardalani et al. 2022. *Understanding scaling laws for recommendation models.* arXiv:2208.08489.
3. Arthur & Vassilvitskii. 2006. *k-means++: The advantages of careful seeding.* Stanford TR.
4. Baltescu et al. 2022. *ItemSage: Learning product embeddings for shopping recommendations at Pinterest.* KDD, 2703–2711.
5. Borisyuk et al. 2024. *LiNR: Model Based Neural Retrieval on GPUs at LinkedIn.* CIKM, 4366–4373.
6. Brin & Page. 1998. *The anatomy of a large-scale hypertextual web search engine.* Computer Networks 30(1-7), 107–117.
7. Cambazoglu & Baeza-Yates. 2016. *Scalability and efficiency challenges in large-scale web search engines.* SIGIR, 1223–1226.
8. Covington, Adams & Sargin. 2016. *Deep neural networks for YouTube recommendations.* RecSys, 191–198.
9. Ding & Zhai. 2025. *Retrieval with Learned Similarities.* Web Conference 2025, 1626–1637.
10. Douze et al. 2024. *The Faiss library.* arXiv:2401.08281.
11. Frostig, Johnson & Leary. 2019. *Compiling machine learning programs via high-level tracing.* SysML.
12. GitHub 2023. *Faiss on the GPU Limitations.* https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU
13. Goodwin et al. 2017. *Bitfunnel: Revisiting signatures for search.* SIGIR, 605–614.
14. Huang et al. 2020. *Embedding-based retrieval in Facebook search.* KDD, 2553–2561.
15. Ivchenko et al. 2022. *TorchRec: a PyTorch domain library for recommendation systems.* RecSys, 482–483.
16. Jayaram Subramanya et al. 2019. *DiskANN: Fast accurate billion-point nearest neighbor search on a single node.* NeurIPS 32.
17. Johnson, Douze & Jégou. 2019. *Billion-scale similarity search with GPUs.* IEEE TBD 7(3), 535–547.
18. Lewis et al. 2020. *Retrieval-augmented generation for knowledge-intensive NLP tasks.* NeurIPS 33, 9459–9474.
19. Milvus 2023. *Milvus GPU Limitations.* https://milvus.io/docs/gpu_index.md
20. Moritz et al. 2018. *Ray: A distributed framework for emerging AI applications.* OSDI 18, 561–577.
21. Mudigere et al. 2022. *Software-hardware co-design for fast and scalable training of DLRMs.* ISCA, 993–1011.
22. Naumov et al. 2019. *Deep learning recommendation model for personalization and recommendation systems.* arXiv:1906.00091.
23. Oldridge et al. 2020. *Merlin: a GPU accelerated recommendation framework.* IRS.
24. Ootomo et al. 2024. *CAGRA: Highly parallel graph construction and ANN search for GPUs.* ICDE, 4236–4247.
25. Pace et al. 2025. *Lance: Efficient Random Access in Columnar Storage through Adaptive Structural Encodings.* arXiv:2504.15247.
26. Paszke. 2019. *PyTorch: An imperative style, high-performance deep learning library.* arXiv:1912.01703.
27. Pinterest. 2023. *Manas HNSW Realtime: Powering Realtime Embedding-Based Retrieval.* (Pinterest Engineering blog)
28. RapidsAI. 2022. *raft: widely-used algorithms and primitives for data science, graph and ML.* https://github.com/rapidsai/raft
29. Vantage. 2025. *AWS p4d.24xlarge Instance Cost.*
30. Vantage. 2025. *AWS r6i.8xlarge Instance Cost.*
31. Vantage. 2025. *AWS x2idn.24xlarge Instance Cost.*
32. Wang et al. 2018. *Billion-scale commodity embedding for e-commerce recommendation in Alibaba.* KDD, 839–848.
33. Wang et al. 2021. *Milvus: A purpose-built vector data management system.* SIGMOD, 2614–2627.
34. Yi et al. 2019. *Sampling-bias-corrected neural modeling for large corpus item recommendations.* RecSys, 269–277.
35. Zhai et al. 2019. *Learning a unified embedding for visual search at Pinterest.* KDD, 2412–2420.
36. Zhai et al. 2023. *Revisiting neural retrieval on accelerators.* KDD, 5520–5531. (Mixture of Logits)
37. Zhai et al. 2024. *Actions speak louder than words: Trillion-parameter sequential transducers for generative recommendations.* arXiv:2402.17152. (HSTU)
38. Zhang et al. 2024. *Wukong: Towards a scaling law for large-scale recommendation.* arXiv:2403.02545.
39. Zhang et al. 2025. *Optimizing Recall or Relevance? A Multi-Task Multi-Head Approach for Item-to-Item Retrieval.* KDD V.2, 5194–5204.
40. Zhao et al. 2023. *Embedding in recommender systems: A survey.* arXiv:2310.18608.
41. Zhou et al. 2018. *Deep interest network for click-through rate prediction.* KDD, 1059–1068.
