# 视觉自回归建模（VAR）：Next-Scale Prediction

> Diffusion models 在时间上迭代 sample（denoising steps）。VAR 在尺度上迭代 sample——它先预测 1x1 token，再预测 2x2，再预测 4x4，直到最终 resolution，每个 scale 都 condition on 前一个 scale。2024 年论文显示，VAR 在 image generation 上匹配 GPT-style scaling laws，并在相同 compute budget 下超过 DiT。本课构建核心机制。

**类型:** Build
**语言:** Python (with PyTorch)
**先修:** Phase 7 Lesson 03 (Multi-Head Attention), Phase 8 Lesson 06 (DDPM)
**时间:** ~90 minutes

## 要解决的问题

Autoregressive generation 统治了 language modeling，因为它可预测地扩展：更多 compute、更多 parameters、更低 perplexity、更好 outputs。2024 年之前，image generation 有两类主要 AR 尝试：PixelRNN/PixelCNN（pixel-by-pixel）和 DALL-E 1 / Parti / MuseGAN（在 VQ-VAE codes 上 token-by-token）。

两者都受 generation-order problem 困扰。Pixels 和 tokens 排列在 2D grid 中，但 AR model 必须按 1D raster order 访问它们。一个早期 corner pixel 不知道图像最终会变成什么。Generation quality 比 GPT-on-text 扩展得更差，并且在匹配 compute 下从未达到 diffusion-model quality。

VAR 通过改变正在生成的东西来修复 generation-order problem。它不是在空间中一个接一个预测 image tokens，而是以逐渐提高的 resolutions 预测整张 image。Step 1：预测 1x1 token（整体 image “summary”）。Step 2：预测 2x2 token grid（更粗特征）。Step 3：预测 4x4 grid。Step K：预测最终 `(H/8)x(W/8)` grid。

每个 scale 都 attend to 所有 previous scales（按 “scale order” causal），并在自己的 scale 内并行。Order problem 消失了：scale k 的整张 image 在一次 transformer pass 中产生。

## 核心概念

### VQ-VAE Multi-Scale Tokenizer

VAR 需要一个 **multi-scale discrete tokenizer**。对 image x，它产生一串逐渐更高 resolution 的 token grids：

```text
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

每个 z_k 使用同一个 codebook（典型大小 4096-16384）。每个 scale 上的 tokenization 并不独立——训练目标是让每个 scale residual 的和重构 f：

```text
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

这是一个 **residual VQ** 变体。Scale k 捕捉 scales 1..k-1 没捕捉到的内容。Decoder 接收所有 scale embeddings 的和并产生 image。

Multi-scale VQ tokenizer 会先训练一次（像 VQGAN），然后冻结。所有生成工作都由其上的 autoregressive model 完成。

### Next-Scale Prediction

Generative model 是一个 transformer，它看到所有 previous scales 的 tokens，并预测 next scale 的 tokens。

Input sequence structure：

```text
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

Position embeddings 同时编码 scale index 和 scale 内的 spatial position。Attention 在 scale order 上 causal：scale k、position (i, j) 的 token 可以 attend to scales 1..k 的所有 tokens，以及 scale k 本身中按某种 intra-scale order 更早的 tokens（VAR 使用 fixed positional attention，没有 intra-scale causality——同一 scale 内所有 positions 并行预测）。

Training loss：在每个 scale k，给定所有 prior-scale tokens，预测 tokens z_k。对 discrete VQ codes 做 cross-entropy loss。结构与 GPT 相同，只是 “sequence” 现在具有 scale structure。

### Generation

推理时：

```text
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

如果 K = 10 scales，generation 是 10 个 transformer forward passes。每个 pass 并行产生整个 scale——scale 内没有 per-token autoregression。对 256x256 image，这约等于 10 passes，而 DiT 是 28-50。

### Why Next-Scale Wins Over Next-Token

三个结构性优势：

1. **Coarse-to-fine aligns with natural image statistics.** 人类视觉感知和 image datasets 都表现出 scale-dependent regularities：low-frequency structure 稳定且可预测；high-frequency detail 条件依赖于 low-frequency content。Next-scale prediction 利用了这一点。
2. **Parallel generation within scale.** 与 GPT-style token AR 不同，VAR 在一步内产生某个 scale 的所有 tokens。有效 generation length 是 log-scale，而不是 linear。
3. **No generation order bias.** Scale k 的 tokens 能看到 scale k-1 的全部；不存在 “left-of” 或 “above” bias，迫使早期 tokens 在后续 context 可用前提交。

### Scaling Law

Tian et al. 证明 VAR 在 ImageNet 的 FID 上遵循 power-law scaling curve——就像 GPT 的 perplexity 一样。参数或 compute 翻倍会可靠地减半 error。这是第一个像 language models 那样干净展现 scaling behavior 的 image-generative model。结果是，VAR-scale predictions 可以由 compute 预测，而不是每个 architecture 靠经验猜。

### Relationship to Diffusion

VAR 和 diffusion 有相同的数据压缩故事：二者都把 generation problem 拆成一串更容易的 subproblems。

- Diffusion：逐渐加 noise，学习 undo one step。
- VAR：逐渐加 resolution，学习预测 next scale。

它们是穿过问题的不同轴线。二者都得到 tractable conditional distributions。经验上 VAR inference 更快（passes 更少，scale 内完全并行），并在 class-conditional ImageNet 上匹配或超过 DiT。Text-conditional VAR（VARclip、HART）是活跃研究方向。

## 动手实现

在 `code/main.py` 中你将：
1. 在 synthetic “image” data（2D Gaussian rings）上构建 tiny **multi-scale VQ tokenizer**。
2. 训练 **VAR-style transformer** 来 next-scale-predict tokens。
3. 通过调用 transformer 4 次（4 scales）并 decode 来 sample。
4. 验证 scale-ordered training 让 generation 在 scale 内并行。

这是 toy implementation。重点是看见 scale-structured attention mask 和 parallel-within-scale generation 真的在工作。

## 交付成果

本课产出 `outputs/skill-var-tokenizer-designer.md`——一个用于设计 multi-scale tokenizer 的 skill：number of scales、scale ratios、codebook size、residual sharing、decoder architecture。

## 练习

1. **Scale count ablation.** 用 4、6、8、10 scales 训练 VAR。测量 reconstruction quality vs number of autoregressive passes。更多 scales = 更细 residuals = 更好质量，但 passes 更多。

2. **Codebook size.** 用 codebook sizes 512、4096、16384 训练 tokenizers。更大 codebooks 给出更好 reconstruction，但更难预测。找到 knee。

3. **Parallel-within-scale check.** 对训练好的 VAR，显式测量 attention pattern。在 scale k 内，模型是否 attend to cross-scale positions 而不是 intra-scale？验证 mask implementation。

4. **VAR vs DiT scaling.** 对同一个 ImageNet class-conditional task，在匹配 param budgets（例如 33M、130M、458M）下训练 VAR 和 DiT。画 FID vs compute。VAR 应该在每个 size 都领先 DiT——在小尺度复现论文结果。

5. **Text conditioning.** 扩展 VAR，把 text embedding（CLIP pooled）作为额外 conditioning input，通过 adaLN 输入。这是 HART recipe。它能在 text-aligned sampling 上把 FID 改善多少？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| VAR | “Visual AutoRegressive” | 通过在 VQ token grids pyramid 上做 next-scale prediction 来生成图像 |
| Next-scale prediction | “Predict coarser, then finer” | 模型按逐渐升高的 resolution scales 预测 tokens，并 condition on 所有 previous scales |
| Multi-scale VQ tokenizer | “Residual VQ” | 产生 K 个 resolution 递增 token grids 的 VQ-VAE，decoder 汇总所有 scales |
| Scale k | “Pyramid level k” | K 个 resolution levels 之一，从 k=1 的 1x1 到 k=K 的 (H/p)x(W/p) |
| Parallel-within-scale | “One forward per scale” | Scale k 的所有 tokens 在一次 transformer pass 中预测，而不是 autoregressively |
| Causal-across-scales | “Scale-ordered attention” | Scale k 的 token 可以 attend to scales 1..k，但不能 attend to scales k+1..K |
| Residual VQ | “Additive tokenization” | 每个 scale 的 tokens 编码 lower scales 留下的 residual；decoder 汇总所有 scale embeddings |
| VAR scaling law | “Image GPT scaling” | FID 像 language models 的 perplexity 一样，随 compute 遵循可预测 power law |
| HART | “Hybrid VAR + text” | Text-conditional VAR variant，把 MaskGIT-style iterative decoding 与 VAR 的 scale structure 结合 |
| Scale position embedding | “(scale, row, col) triple” | Positional encoding 同时携带 scale index 和 scale 内 spatial coordinates |

## 延伸阅读

- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) — VAR 论文，canonical reference
- [Peebles and Xie, 2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) — DiT，diffusion comparison baseline
- [Esser et al., 2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) — VQGAN，VAR multi-scale tokenizer 扩展的 tokenizer family
- [van den Oord et al., 2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) — VQ-VAE，discrete image tokenization 的基础
- [Tang et al., 2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) — text-conditional VAR
