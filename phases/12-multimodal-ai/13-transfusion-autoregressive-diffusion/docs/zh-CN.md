# Transfusion：一个 Transformer 中的自回归文本 + Diffusion 图像

> Chameleon 和 Emu3 把一切都押在离散 token 上。它们能工作，但量化瓶颈很明显：图像质量的平台期低于连续空间 diffusion models。Transfusion（Meta，Zhou et al., 2024 年 8 月）做了相反的下注：保持图像连续，完全去掉 VQ-VAE，并用两个损失训练一个 transformer。文本 token 使用 next-token-prediction。图像 patches 使用 flow-matching / diffusion loss。两个目标优化同一套权重。Stable Diffusion 3 底层架构（MMDiT）是它的近亲。本课阅读 Transfusion 的主张，构建一个玩具双损失 trainer，并追踪让一个 transformer 同时做两件事的 attention mask。

**类型:** Build
**语言:** Python (stdlib, two-loss trainer on MNIST-scale toy)
**先修:** Phase 12 · 11 (Chameleon), Phase 8 (Generative AI)
**时间:** ~180 minutes

## 学习目标

- 连接一个 transformer，让它在一个 backbone 上运行两个损失：文本 token 上的 NTP，以及图像 patches 上的 diffusion MSE。
- 解释为什么图像 patches 内部使用 bidirectional attention、文本 token 上使用 causal attention 是正确的 mask 选择。
- 从计算、质量和代码复杂度比较 Transfusion-style（连续图像、diffusion loss）与 Chameleon-style（离散图像、NTP）。
- 说出 MMDiT 的贡献：每个 block 使用 modality-specific weights，在 residual stream 上做 joint attention。

## 要解决的问题

离散图像 token 与连续图像 token 的争论比 LLM 更早。连续表示（raw pixels、VAE latents）能保留细节。离散 token（VQ indices）符合 transformer 原生词表，但会在量化步骤丢失细节。

Chameleon / Emu3 走离散路线：一个损失、一个架构，但图像保真度受 tokenizer 质量限制。

Diffusion models 走连续路线：图像质量极高，但它是与 LLM 分开的模型，噪声 schedule 工程复杂，也没有和文本生成干净整合的方式。

Transfusion 问：能不能两者兼得？保持图像连续，仍然训练一个模型，把两个损失缝进同一个 gradient step。

## 核心概念

### 双损失架构

单个 decoder-only transformer 处理一个包含以下内容的序列：

- 文本 token（离散，来自 BPE vocab）。
- 图像 patches（连续，16x16 pixel blocks，经 linear embedding 投影到 hidden dim，与 ViT encoder 的输入相同）。
- 标记连续 patches 所在位置的 `<image>` 与 `</image>` tags。

Forward pass 只运行一次。loss 会按 token 选择两个 head 中的一个：

- 对文本 token：在 vocab-logits head 上做标准 cross-entropy。
- 对图像 patches：在连续 patches 上做 diffusion loss，预测添加到每个 patch 的 noise。

梯度流经共享的 transformer body。两个损失同时改进共享权重。

### Attention mask：causal text + bidirectional image

文本 token 必须是 causal 的：不能让文本 token attend 到未来文本，否则 teacher forcing 会被破坏。图像 patches 则代表同一张快照；它们应该在同一个 image block 内彼此 bidirectional attend。

mask：

```text
M[i, j] = 1 if:
  (i is text and j is text and j <= i)   # causal for text
  OR (i is image and j is image and same_image_block(i, j))   # bidirectional within image
  OR (i is text and j is image and j < i_image_end)   # text attends to previous images
  OR (i is image and j is text and j < i_image_start)   # image attends to preceding text
```

训练与推理都将其实现为 block-triangular mask。

### transformer 内部的 Diffusion loss

diffusion loss 是标准做法：给 image patch 加 noise，让模型预测 noise（或者等价地预测 clean patch）。Transfusion 的版本使用 flow matching：预测从 noisy 到 clean 的 velocity field。

训练期间：
1. 对每个 image patch x0，采样一个随机 timestep t。
2. 采样 noise ε，计算 xt = (1-t) * x0 + t * ε（flow matching 的线性插值）。
3. transformer 预测 v_theta(xt, t)；loss = MSE(v_theta(xt, t), ε - x0)。
4. 与来自同一序列的文本 NTP losses 一起 backprop。

推理时，生成是：
- 文本 token：标准 autoregressive sampling。
- 图像 patches：以之前的文本 token 为条件运行 diffusion sampling loop（典型 10-30 steps）。

### MMDiT：Stable Diffusion 3 的变体

Stable Diffusion 3（Esser et al., 2024 年 3 月）在与 Transfusion 接近的时间发布了 MMDiT（Multimodal Diffusion Transformer）。这些架构是兄弟。

MMDiT 的关键差异：

- 每个 block 使用 modality-specific weights。每个 transformer block 对文本 token 与 image patches 有独立的 Q、K、V 和 MLP weights。attention 是 joint（跨模态），其余部分是 modality-specific。
- Rectified flow training。一种特定的 flow-matching 变体，具有已知 sampling，并且数学比 DDPM 更简单。
- Scale。MMDiT 是 SD3（2B 与 8B 参数变体）的 backbone。Transfusion 论文扩展到 7B。

二者收敛到同一个核心想法：一个 transformer 在文本上运行 NTP，在连续图像表示上运行 diffusion。

### 为什么它超过 Chameleon-style

连续 diffusion 与离散 NTP 在图像生成质量上的差距可测。Transfusion 论文报告：

- 在 7B params 下，它在 FID 上比同等大小的 Chameleon-style 模型好 3-5 分。
- 不需要训练 tokenizer：图像 encoder 更简单（Linear projection to hidden，与 ViT 的 input layer 一样）。
- 推理时 image patch denoising 可以并行，而 autoregressive image tokens 不能。

缺点：Transfusion 是双损失模型，训练动态更棘手。Loss weights 需要调节。NTP 与 diffusion 之间的 schedule mismatch 可能导致某个 head 占主导。

### 下游是什么

Janus-Pro（Lesson 12.15）通过解耦理解与生成用的 vision encoder 来细化 Transfusion 的想法：一种用 SigLIP，一种用 VQ，同时共享 transformer body。Show-o（Lesson 12.14）把 diffusion 换成 discrete-diffusion（masked prediction）。统一生成 family 在 Transfusion 之后快速分支。

2026 年能发出图像的生产级 VLM：Gemini 3 Pro、GPT-5、Claude Opus 4.7 的图像生成路径，几乎肯定使用了这个 family 的某种后代。细节是专有的。

## 实际使用

`code/main.py` 在一个微型 MNIST-like 问题上构建玩具 Transfusion：

- 文本 captions 是描述数字（0-9）的短整数序列。
- 图像是 4x4 bytes 网格。
- 一对共享权重的 linear projections 充当 transformer stand-in；文本上做 NTP loss，noisy patches 上做 MSE loss。
- 训练循环交替使用两个损失，attention mask 是显式的。
- 生成在一个 forward pass 中产出文本 caption 与 4x4 image。

transformer 是玩具。双损失 plumbing、attention mask 构造和 inference loop 才是真正的 artifact。

## 交付成果

本课产出 `outputs/skill-two-loss-trainer-designer.md`。给定一个新的多模态训练任务（text + image、text + audio、text + video），它会设计双损失 schedule（loss weights、mask shape、shared vs modality-specific blocks），并标出实现风险。

## 练习

1. 一个 Transfusion-style 模型用 70% text tokens 与 30% image patches 训练。image diffusion loss 的量级约为 text NTP loss 的 10x。什么 loss weights 能平衡它们？

2. 为序列 `[T, T, <image>, P, P, P, P, </image>, T]` 实现 block-triangular mask。把每个 entry 标为 0 或 1。

3. MMDiT 有 modality-specific QKV weights。相比 Transfusion 的完全共享 transformer，这会增加多少参数？在 7B params 下值得吗？

4. 生成：给定 text prompt，模型先为 50 个 token 运行 NTP，然后遇到 `<image>`，接着在 256 patches 上运行 20 个 denoise steps。总共有多少次 forward passes？

5. 阅读 SD3 paper Section 3。描述 rectified flow，以及为什么它比 DDPM 用更少 inference steps 收敛。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| 双损失训练 | “NTP + diffusion” | 单个 transformer 在同一个 gradient step 中同时优化文本 token 上的 cross-entropy 与连续 image patches 上的 MSE |
| Flow matching | “Rectified flow” | 预测从 noise 到 clean data 的 velocity field 的 diffusion 变体；数学比 DDPM 更简单 |
| MMDiT | “Multimodal DiT” | Stable Diffusion 3 的架构：joint attention、modality-specific MLPs 与 norms |
| Block-triangular mask | “Causal text + bidirectional image” | 在文本 token 间保持 causal、在图像区域内保持 bidirectional 的 attention mask |
| 连续图像表示 | “No VQ” | 将 image patches 表示为 real-valued vectors，而不是整数 codebook indices |
| Velocity prediction | “v-parameterization” | 网络输出是 noise 与 data 之间的 velocity field，而不是 noise 本身 |

## 延伸阅读

- [Zhou et al. — Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser et al. — Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao et al. — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
