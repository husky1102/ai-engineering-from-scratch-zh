# Emu3：用于图像和视频生成的 Next-Token Prediction

> BAAI 的 Emu3（Wang et al., 2024 年 9 月）本应终结 2024 年的 diffusion-versus-autoregressive 争论。一个 Llama 风格的 decoder-only transformer，只在 next-token-prediction 目标上训练，跨越文本 + VQ image tokens + 3D VQ video tokens 的统一词表，就能在图像生成上超过 SDXL，并在感知上超过 LLaVA-1.6。没有 CLIP loss。没有 diffusion schedule。推理时使用 classifier-free guidance 提升质量，但核心训练目标就是带 teacher forcing 的 next-token prediction。发表于 Nature。本课阅读 Emu3 的主张：为什么更好的 tokenizer 加上规模就足够，并与 diffusion 方法对比。

**类型:** Learn
**语言:** Python (stdlib, 3D video tokenizer math + autoregressive sampler skeleton)
**先修:** Phase 12 · 11 (Chameleon)
**时间:** ~120 minutes

## 学习目标

- 解释为什么 Emu3 的单一损失 next-token 目标能够工作，尽管长期以来人们认为图像质量必须依赖 diffusion。
- 描述 3D video tokenizer：spatiotemporal VQ codebook 是什么样子，为什么 patch 跨越时间。
- 比较 Emu3 与 Stable Diffusion XL 在训练计算量、推理成本、质量上限上的差异。
- 说出同一个 Emu3 模型扮演的三个角色：Emu3-Gen（image gen）、Emu3-Chat（perception）、Emu3-Stage2（video gen）。

## 要解决的问题

截至 2024 年的传统观点是：图像生成需要 diffusion。论点是：离散图像 token 丢失了太多细节，难以重建，而 autoregressive sampling 会在数千个 token 上累积误差。Stable Diffusion、DALL-E 3、Imagen、Midjourney 都使用某种形式的 diffusion。Chameleon（Lesson 12.11）在小规模上部分推翻了这个观点，但质量没有追上 SDXL。

Emu3 正面攻击这个论点。它的主张是：更好的视觉 tokenizer + 足够规模 + next-token loss = 在同一个也能做感知的模型中，得到超越 diffusion 的图像生成。

它发表时，这个赌注很有争议。两年后，开源统一生成 family（Emu3、Show-o、Janus-Pro、Transfusion）已经成为研究的默认路径；生产级 frontier models 看起来也使用了某种变体。

## 核心概念

### Emu3 tokenizer

关键成分是视觉 tokenizer。Emu3 训练了一个自定义 IBQ-class tokenizer（Inverse Bottleneck Quantizer，SBER-MoVQGAN family），每个 token 做 8x8 resolution-reduction。一张 512x512 图像变成 64x64 = 4096 个 token，codebook size 为 32768。

这比 Chameleon 在 K=8192 下每张 512x512 图像 1024 个 token 更大，但每个 token 更便宜（更小的 codebook lookups、更简单的 codec）。关键指标是：reconstruction PSNR 为 30.5 dB，已经能与 Stable Diffusion 的连续 latent space 的 32 dB 竞争。

对视频来说：3D VQ tokenizer 把一个 spatiotemporal patch（4x4x4 pixels）编码成一个整数。一个 4s clip、8 FPS 共有 32 帧；在 256x256、4x spatial 与 4x temporal reduction 下，token count 是 (256/4) * (256/4) * (32/4) = 64 * 64 * 8 = 32,768 tokens。

tokenizer 质量就是上限。Emu3 的贡献有一部分就是“我们训练了一个非常好的 tokenizer”。

### 单一损失训练

Emu3 使用一个目标：在跨文本 token、2D image token 和 3D video token 的共享词表上做 next-token prediction。训练期间会用 modality-specific factors 乘到权重上，以平衡贡献，但 loss function 是相同的。

训练混合数据包括：
- Image gen: `<text caption> <image> image_tokens </image>`
- Image perception: `<image> image_tokens </image> <question> text_tokens`
- Video gen: `<text caption> <video> video_tokens </video>`
- Video perception: analogous.
- Text only: standard NTP.

模型从数据分布中学习何时发出 image tokens 与 text tokens。生成能力来自模型在 `<image>` tag 之后预测 image tokens。

### Classifier-free guidance 与 temperature

autoregressive image generation 在推理时配合 classifier-free guidance（CFG）会好很多。Emu3 使用了它：生成两次，一次用完整 caption，一次用 empty caption，再用 guidance weight（典型 3.0-7.0）混合 logits。这是 diffusion 中同样的 CFG 技巧，被借用到 autoregressive setting。

Temperature 很重要：过高会产生 artifacts；过低会 mode collapse。Emu3 推荐 perception 使用 temperature 1.0，image generation 使用 0.8。

### 三个角色，一个模型

Emu3 以三个功能上不同的 API 形式发布，但底层是一套权重：

- Emu3-Gen。图像生成。输入文本，输出 image tokens。
- Emu3-Chat。VQA 与 captioning。输入图像（tokens），输出文本。
- Emu3-Stage2。视频生成与 video VQA。输入文本或视频，输出文本或视频。

没有 task-specific heads。只是不同的 prompt templates。同一个 checkpoint。

### Benchmarks

来自 Emu3 论文（2024 年 9 月）：

- 图像生成：在 MJHQ-30K FID 上超过 SDXL（5.4 vs 5.6），GenEval overall（0.54 vs 0.55，统计上打平），Deep-Eval composite 也基本持平。
- 图像感知：在 VQAv2 上超过 LLaVA-1.6（75.1 vs 72.4），在 MMMU 上大致持平。
- 视频生成：4 秒 clip 质量，在 FVD 上与 Sora 时代公开 benchmarked 的模型有竞争力。

这些数字并不总是胜出：Emu3 在这里让一点、在那里赢一点。但“next-token prediction is all you need”这个主张在多模态上是可以辩护的。

### 计算成本

Emu3 用一个 7B 参数模型在约 300 billion multimodal tokens 上训练。GPU-hours 大致相当于 Llama-2-7B pretraining（在 A100-class silicon 上约 2k-4k GPU-years）。Stable Diffusion 3 这样的 diffusion models 训练预算类似，但需要独立 text encoders 和更复杂的 pipelines。

推理时，Emu3 每张图比 SDXL 慢：4096 个 image tokens、30 tok/s，大约每张 512x512 图 2 分钟；SDXL 是 2-5 秒。Speculative decoding 与 KV-cache optimization 会缩小差距，但不能抹平。Autoregressive image gen 计算量很大，这是当前的权衡。

### 为什么重要

Emu3 的深层贡献是概念性的。如果 next-token prediction 能扩展到在图像生成上匹配 diffusion，那么统一模型路径（一个损失、一个 backbone、任意模态）就是可行的。未来模型不再需要独立 text encoders、独立 diffusion schedulers、独立 VAEs。一个 transformer，每种模态一个 tokenizer，然后扩大规模。

Show-o、Janus-Pro 和 InternVL-U 都建立在这个主张之上，或者挑战它。到 2025 年，中国实验室（BAAI、DeepSeek）在这个方向上的发布比美国实验室更激进。

## 实际使用

`code/main.py` 构建两个玩具组件：

- 一个 2D vs 3D VQ tokenizer count calculator：给定 (resolution, patch, clip_length, FPS)，计算 image 与 video 的 token counts。
- 一个带 classifier-free guidance 与 temperature 的 autoregressive image-token sampler。

CFG 实现匹配 Emu3 的 recipe：用 guidance weight 混合 conditional 与 unconditional logits。

## 交付成果

本课产出 `outputs/skill-token-gen-cost-analyzer.md`。给定一个生成产品规格（图像或视频、目标分辨率、质量层级、延迟预算），它会计算 token counts、推理成本，并在 Emu3-family 与 diffusion 之间做选择。

## 练习

1. Emu3 在 8x8 reduction 下每张 512x512 图像产生 4096 个 token。计算 1024x1024 与 2048x2048 的等价 token 数。推理延迟会发生什么？

2. 阅读 Emu3 Section 3.3 中关于 video tokenizer 的内容。描述 3D VQ patch shape，以及为什么它是 4x4x4 而不是 8x8x1。

3. Classifier-free guidance weight 5.0 vs 3.0：视觉效果有什么差别？追踪 `code/main.py` 中的数学。

4. 计算 Emu3-7B 在 300B tokens 上的训练 FLOPs，并与 Stable Diffusion 3 对比。哪个训练更贵？

5. Emu3 在 FID 上超过 SDXL，但在 VQAv2 上没有超过专门 VLM。解释为什么统一损失方法在不同 benchmarks 上相对 specialists 展现出不同优势。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Next-token prediction | “NTP” | 标准自回归损失：给定 token[0..i] 预测 token[i+1]；只要被 token 化，就适用于每种模态 |
| IBQ tokenizer | “Inverse bottleneck quantizer” | 一类 VQ-VAE，codebook 更大（32768+），重建效果优于 Chameleon 的 tokenizer |
| 3D VQ | “Spatiotemporal quantizer” | 由 (time, row, col) 索引的 codebook；一个 token 覆盖 4x4x4 像素立方体 |
| Classifier-free guidance | “CFG” | 用权重 gamma 混合 conditional 与 unconditional logits；推理时提升图像质量 |
| Unified vocabulary | “Shared tokens” | 文本 + 图像 + 视频都来自同一个整数空间；模型预测接下来出现的任意模态 |
| MJHQ-30K | “Image gen benchmark” | 包含 30k prompts 的 Midjourney-quality benchmark；Emu3 在这里报告 FID |

## 延伸阅读

- [Wang et al. — Emu3: Next-Token Prediction is All You Need (arXiv:2409.18869)](https://arxiv.org/abs/2409.18869)
- [Sun et al. — Emu: Generative Pretraining in Multimodality (arXiv:2307.05222)](https://arxiv.org/abs/2307.05222)
- [Liu et al. — LWM (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Yu et al. — MAGVIT-v2 (arXiv:2310.05737)](https://arxiv.org/abs/2310.05737)
- [Tian et al. — VAR (arXiv:2404.02905)](https://arxiv.org/abs/2404.02905)
