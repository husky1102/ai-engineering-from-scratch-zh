# 生成模型：分类与历史

> 每个图像模型、文本模型、视频模型和 3D 模型都能放进五个桶之一。选错桶，你会和数学搏斗好几周。选对桶，过去十二年的进展会在你脑中整齐叠起来。

**类型：** Learn
**语言：** Python
**先修：** Phase 2 (ML Fundamentals), Phase 3 (Deep Learning Core), Phase 7 · 14 (Transformers)
**时间：** ~45 分钟

## 要解决的问题

Generative model 做一件事：给定从某个未知分布 `p_data(x)` 抽取的 training samples，输出看起来像来自同一分布的新 samples。人脸、句子、MIDI files、protein structures：眯起眼睛看，都是同一个问题。

麻烦在于 `p_data` 生活在一个有数百万维的空间里（512x512 RGB image 约 786k dimensions），samples 位于这个空间里一条很薄的 manifold 上，而你可能只有 10M examples。暴力求 density 没希望。每个 generative model 都是在用一个难题换另一个稍微没那么难的问题。

过去十二年里有五个家族活了下来。知道每个家族做了哪种 compromise，就能理解为什么它在某些任务上赢、在另一些任务上崩。

## 核心概念

![Five families of generative models — taxonomy by what they model](../assets/taxonomy.svg)

**1. Explicit density, tractable.** 把 `log p(x)` 写成一个你真的能 evaluate 的和式。Autoregressive models（PixelCNN, WaveNet, GPT）把 `p(x) = ∏ p(x_i | x_<i)` factorize。Normalizing flows（RealNVP, Glow）把 `p(x)` 构造成一个简单 base 的 invertible transform。优点：exact likelihood，干净的 training loss。缺点：autoregressive inference 是 sequential（长序列慢），flows 需要 invertible architectures（架构限制强）。

**2. Explicit density, approximate.** 从下方 bound `log p(x)`（ELBO）并优化这个 bound。VAEs（Kingma 2013）使用带 variational posterior 的 encoder-decoder。Diffusion models（DDPM, Ho 2020）训练一个 denoiser，隐式优化 weighted ELBO。Diffusion 是 2026 年图像、视频和 3D 的主导 backbone。

**3. Implicit density.** 完全跳过 density；学习一个生成 samples 的 generator `G(z)`，以及一个判断真假的 discriminator `D(x)`。GANs（Goodfellow 2014）。Inference 快（一次 forward pass），但 training 出名地不稳定。即使到 2026 年，StyleGAN 1/2/3 在 fixed-domain photorealism（faces, bedrooms）上仍是 state of the art。

**4. Score-based / continuous-time.** 直接学习 log-density 的 gradient `∇_x log p(x)`（score）。Song & Ermon（2019）展示 score matching 会把 diffusion 推广到 SDE。Flow matching（Lipman 2023）是 2024-2026 的 hotness：simulate-free training、更直的 paths、比 DDPM 快 4-10x 的 sampling。Stable Diffusion 3、Flux、AudioCraft 2 都使用 flow matching。

**5. Token-based autoregressive over discrete codes.** 先用 VQ-VAE 或 residual quantizer 把高维数据压缩成一小段 discrete tokens，再用 Transformer 建模 token sequence。Parti、MuseNet、AudioLM、VALL-E、Sora 的 patch tokenizer 都使用这个思路。这是第 1 桶加上 learned tokenizer。

## 简史

| Year | Model | Why it mattered |
|------|-------|-----------------|
| 2013 | VAE (Kingma) | 第一个有可用 training loss 的 deep generative model。 |
| 2014 | GAN (Goodfellow) | Implicit density，没有 likelihood，却有惊人的 sharp samples。 |
| 2015 | DRAW, PixelCNN | Sequential image generation。 |
| 2017 | Glow, RealNVP | Invertible flows；用 depth 得到 exact likelihood。 |
| 2017 | Progressive GAN | 第一批 megapixel faces。 |
| 2019 | StyleGAN / StyleGAN2 | Photorealistic faces 在这个单一 domain 里仍很难被击败。 |
| 2020 | DDPM (Ho) | Diffusion 变得实用。 |
| 2021 | CLIP, DALL-E 1, VQGAN | Text-to-image 进入主流。 |
| 2022 | Imagen, Stable Diffusion 1, DALL-E 2 | Latent diffusion + text conditioning = 商品化。 |
| 2022 | ControlNet, LoRA | 对 pretrained diffusion 做精细控制。 |
| 2023 | SDXL, Midjourney v5, Flow matching | Scale + 更好的 training dynamics。 |
| 2024 | Sora, Stable Diffusion 3, Flux.1 | Video diffusion；flow matching 获胜。 |
| 2025 | Veo 2, Kling 1.5, Runway Gen-3, Nano Banana | Production-grade video。 |
| 2026 | Consistency + Rectified Flow | 从 diffusion backbones 做 one-step sampling。 |

## 五问分诊

读一篇新的 generative model paper 时，先回答这五个问题，再读 method section。

1. **被建模的是什么？** Pixels、latents、discrete tokens、3D Gaussians、meshes、waveforms？
2. **Density 是 explicit 还是 implicit？** 他们有没有写下 `log p(x)`？
3. **Sampling 是 one-shot 还是 iterative？** Iterative 意味着 slower inference；one-shot 通常意味着 adversarial 或 distilled。
4. **Conditioning 是什么：unconditional、class、text、image、pose？** 这会决定 loss 和 architecture scaffolding。
5. **Evaluation 是什么：FID、CLIP score、IS、human preference、task accuracy？** 每个都有已知 failure modes（见 Lesson 14）。

本阶段之后的每节课，你都会重新回答这五个问题。到最后，它们会变成反射。

## 动手实现

本课代码是一个轻量 visualization：从 samples 出发，用三种 toy approaches（kernel density、discrete histogram、nearest-sample “GAN-ish” generator）拟合一个 1-D mixture-of-Gaussians，这样你能在一个屏幕里看见 explicit vs implicit density 的差异。

运行 `code/main.py`。它从一个 two-mode Gaussian mixture 中抽取 2000 samples，然后打印：

```text
explicit density (histogram): p(x in [-0.5, 0.5]) ≈ 0.38
approximate density (KDE):     p(x in [-0.5, 0.5]) ≈ 0.41
implicit (nearest-sample gen): 20 new samples printed, no p(x)
```

注意：前两个能让你问“这个点有多 likely？”第三个不能。这就是 *explicit vs implicit* 区分；它会影响之后每一课。

## 实际使用

2026 年，哪个家族对应哪个任务？

| Task | Best family | Why |
|------|-------------|-----|
| Photoreal faces, narrow domain | StyleGAN 2/3 | 仍然最 sharp，inference 最快。 |
| General text-to-image | Latent diffusion + flow matching | SD3, Flux.1, DALL-E 3。 |
| Fast text-to-image | Rectified flow + distillation | SDXL-Turbo, SD3-Turbo, LCM。 |
| Text-to-video | Diffusion Transformer + flow matching | Sora, Veo 2, Kling。 |
| Speech + music | Token-based AR (AudioLM, VALL-E, MusicGen) or flow matching (AudioCraft 2) | Discrete tokens 便宜地 scale。 |
| 3D scenes | Gaussian Splatting fit, diffusion prior | 3D-GS 用于 reconstruction，diffusion 用于 novel-view。 |
| Density estimation (no sampling) | Flows | 唯一拥有 exact `log p(x)` 的家族。 |
| Simulation / physics | Flow matching, score SDE | Straight-line paths，smooth vector fields。 |

## 交付成果

保存为 `outputs/skill-model-chooser.md`。

这个 skill 接收一个 task description，并输出：（1）该用哪个 family，（2）三个 open 和三个 hosted options 的 ranked list，（3）你该留意的 likely failure mode，以及（4）compute/time budget。

## 练习

1. **Easy.** 对下面五个产品，识别其 family 和 backbone：ChatGPT image、Midjourney v7、Sora、Runway Gen-3、ElevenLabs。证据应来自 public technical reports。
2. **Medium.** 你明天要读的 paper 声称 sampling 比 diffusion 快 100x。写下三个问题，用来检查这个 speedup 在 conditioning 和 high resolution 下是否仍成立。
3. **Hard.** 选择一个你关心的 domain（例如 protein structure、CAD、molecules、trajectories）。对该 domain 当前的 SOTA model 回答五问分诊，并草拟一个更好模型会改变什么。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Generative model | “它会造新东西” | 学习 `p_data(x)` 的 sampler，并可选暴露 `log p(x)`。 |
| Explicit density | “你能 evaluate 它” | 模型提供 closed-form 或 tractable 的 `log p(x)`。 |
| Implicit density | “GAN-style” | 只有 sampler；无法 evaluate 某个给定点的 `p(x)`。 |
| ELBO | “Evidence lower bound” | `log p(x)` 的 tractable lower bound；VAEs 和 diffusion 会优化它。 |
| Score | “Gradient of log-density” | `∇_x log p(x)`；diffusion 和 SDE models 学习这个 field。 |
| Manifold hypothesis | “Data lives on a surface” | 高维数据集中在低维 manifold 上；解释了 dimensionality reduction 为什么有效。 |
| Autoregressive | “预测下一个 piece” | 把 joint factorize 成 conditionals 的乘积。 |
| Latent | “Compressed code” | decoder 能从中 reconstruct input 的低维 representation。 |

## 生产备注：五类模型，五种推理形态

每个 family 映射到不同的 inference-server cost curve。production-inference literature 会把 LLM inference 表述成 prefill + decode；同样的拆分也适用于这里：

- **Autoregressive（bucket 1 and 5）.** Sequential decode 主导 latency；KV-cache、continuous batching 和 speculative decoding 都直接适用。
- **VAE / diffusion / flow-matching（buckets 2 and 4）.** 没有 LLM 意义上的 decode。Cost = `num_steps × step_cost`，而 `step_cost` 是 full latent resolution 上的一次 transformer 或 U-Net forward。Production knobs 是 step count（DDIM / DPM-Solver / distillation）、batch size 和 precision（bf16 / fp8 / int4）。
- **GAN（bucket 3）.** 一次 forward pass。没有 schedule，没有 KV-cache。TTFT ≈ total latency。这就是 StyleGAN 在 narrow-domain UX 上仍然胜出的原因。

当你在 paper abstract 里看到 “faster than diffusion”，把它翻译成“更少 steps × 相同 step cost”或“相同 steps × 更便宜 step cost”。除此之外都是 marketing。

## 延伸阅读

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — GAN paper。
- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — VAE paper。
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM paper。
- [Song et al. (2021). Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) — diffusion as an SDE。
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — flow matching paper。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — Stable Diffusion 3。
