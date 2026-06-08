# Flamingo 与面向 Few-Shot VLM 的门控交叉注意力

> DeepMind 的 Flamingo（2022）先于其他工作完成了两件事。它证明了单个模型可以处理图像、视频和文本任意交错的序列；也证明了 VLM 可以做上下文学习：给出包含三个（图像，描述）示例对的 few-shot prompt，模型无需任何梯度步骤就能为新图像生成描述。机制是：在冻结 LLM 的既有层之间插入门控交叉注意力层，并使用从零开始的可学习 `tanh` 门，使 LLM 的文本能力在初始化时得以保留。本课会走读 Flamingo 的 Perceiver resampler 和门控交叉注意力架构，也就是 Gemini 交错输入与 Idefics2 视觉 token 的祖先。

**类型:** Learn
**语言:** Python（stdlib，门控交叉注意力 + Perceiver resampler 演示）
**先修:** Phase 12 · 03（BLIP-2 Q-Former）
**时间:** ~120 分钟

## 学习目标

- 解释门控交叉注意力如何通过 `tanh(gate) = 0` 在初始化时保留冻结 LLM 的文本能力。
- 走读 Perceiver resampler：通过交叉注意力把 N 个图像 patch 压成 K 个固定的“latent” query。
- 描述 Flamingo 如何用尊重图像位置的因果 mask 处理交错的图像-文本序列。
- 复现 few-shot 多模态 prompt 结构（3 个图像-描述示例，然后是一个查询图像）。

## 要解决的问题

BLIP-2 会把 32 个视觉 token 喂进冻结 LLM 的输入层。对于每个 prompt 一张图，这很好用。但如果你想输入*多张*图像，并且和文本交错，比如“这是图像 A，描述它；这是图像 B，描述它；现在这是图像 C，描述它”，会怎样？LLM 的 self-attention 需要在同一条流里处理图像 token 和文本 token，而且哪些位置可以 attend 到哪些图像会变得很麻烦。

Flamingo 的答案是：完全不要改变 LLM 的输入流。在既有 LLM block 之间插入额外的 cross-attention 层。文本 token 仍然像往常一样流经 LLM 的 causal self-attention。每隔几个 LLM block，文本 token 还会通过新的门控层 cross-attend 到图像特征。这个 gate（初始化为零）意味着在第 0 步新层就是 no-op，模型表现得和预训练 LLM 完全一样。随着训练推进，gate 打开，视觉信息开始流动。

Flamingo 回答的第二个问题是：一个 prompt 里可能有可变数量的图像（0、1 或很多），该怎么处理？Perceiver resampler 是一个小型 cross-attention 模块，它接受任意数量的 patch，并产生固定数量的视觉 latent token。无论 prompt 里有多少图像，LLM cross-attention 层看到的 shape 都一致。

## 核心概念

### 冻结的 LLM

Flamingo 从冻结的 Chinchilla 70B LLM 开始。全部 70B 权重保持不变。既有文本 self-attention 和 FFN 正常运行。

### Perceiver resampler

对于 prompt 中的每张图像，ViT 会产生 N 个 patch token。Perceiver resampler 有 K 个固定的可学习 latent（Flamingo 使用 K=64）。每个 resampler block 包含两个子步骤：

1. Cross-attention：K 个 latent attend 到 N 个 patch token（Q 来自 latent，K/V 来自 patch）。
2. Latent 内部的 self-attention + FFN。

经过 6 个 resampler block 后，无论 ViT 产生多少 patch，输出都是 K=64 个 dim 1024 的视觉 token。一张 224x224 图像（196 个 patch）和一张 480x480 图像（900 个 patch）都会以 64 个 resampler token 离开。

对于视频，resampler 会沿时间维应用：每一帧的 patch 产生 64 个 latent，时间位置编码让模型区分 `t=0` 和 `t=N`。完整视频变成 `T * 64` 个视觉 token。

### 门控交叉注意力

在冻结 LLM 的每 M 层之间（Flamingo 使用 M=4），插入一个新的门控 cross-attention block：

```text
x_after_llm_block = llm_block(x_before)
cross = cross_attn(x_after, resampler_output)
gated = tanh(alpha) * cross + x_after
x_before_next_block = gated
```

- `alpha` 是一个初始化为零的可学习标量。
- `tanh(0) = 0`，所以初始化时门控分支贡献为零。
- 随着 `alpha` 远离零，cross-attention 的贡献会平滑增长。
- residual connection 意味着即使 gate 完全打开，也不会覆盖 LLM 的文本表示；它只是把视觉信息叠加上去。

这是 Flamingo 中最重要的设计选择：视觉条件是加性的、门控的，并且在初始化时为零。第 0 步的 Flamingo 在纯文本输入上就是完美的 Chinchilla 70B。

### 面向交错输入的 masked cross-attention

在类似“`<image A>` caption A `<image B>` caption B `<image C>` ?”的 prompt 中，每个文本 token 都应该只看到序列中位于它之前的图像。cross-attention mask 强制执行这一点：位置 `t` 的文本 token 只能 attend 到图像索引 `i < i_t` 的图像 resampler token，其中 `i_t` 是位置 `t` 之前最近的图像。“只看最近的前置图像”或“看所有前置图像”都是合理选择；Flamingo 选择了前者。

### 上下文 few-shot 学习

Flamingo prompt 看起来像：

```text
<image1> A photo of a cat. <image2> A photo of a dog. <image3> A photo of a
```

模型看到补全模式，然后输出“bird”（或 image3 展示的任何内容）。没有梯度步骤。冻结 LLM 的上下文学习能力通过门控 cross-attention 传了过来，这是论文的关键结论，也是它重要的原因。

### 训练数据

Flamingo 在三个数据集上训练：

1. MultiModal MassiveWeb（M3W）：4300 万个带有交错图像和文本的网页，重建阅读顺序。
2. Image-Text Pairs（ALIGN + LTIP）：44 亿对。
3. Video-Text Pairs（VTP）：2700 万个短视频片段。

OBELICS（2023）是交错网页语料的开放复现，Idefics、Idefics2 以及大多数开放的“Flamingo-like”模型都在它上面训练。

### OpenFlamingo 与 Otter

OpenFlamingo（2023）是开放复现。架构相同（Perceiver resampler + 冻结 LLaMA 或 MPT 上的门控 cross-attention）。checkpoint 有 3B、4B、9B。由于基础 LLM 更小、数据更少，质量落后于 Flamingo。

Otter（2023）在 OpenFlamingo 基础上使用 MIMIC-IT（多模态指令数据集）做 instruction tuning，证明门控 cross-attention 也适用于指令跟随。

### 后代

- Idefics / Idefics2 / Idefics3：Hugging Face 的门控 cross-attention 谱系，逐步简化（Idefics2 放弃 resampler，改用直接 patch token + adaptive pooling）。
- Flamingo-to-Chameleon 转变：到 2024 年，许多团队转向 early-fusion（Lesson 12.11）；但当必须冻结 backbone 时，Flamingo 风格的门控 cross-attention 仍在生产中使用。
- Gemini 的交错输入：概念上继承了 Flamingo 的交错格式灵活性，尽管具体机制是专有的。

### 与 BLIP-2 对比

| | BLIP-2 | Flamingo |
|---|---|---|
| 视觉桥接 | 输入处一次性 Q-Former | 每 M 层一次门控 cross-attention |
| 视觉 token | 每图 32 个 | 每图每个 cross-attn 层 64 个 |
| 冻结 LLM | 是 | 是 |
| Few-shot 上下文 | 弱 | 强，是论文核心 |
| 交错输入 | 无原生支持 | 有，设计目标 |
| 训练数据 | 1.3 亿对 | 13 亿对 + 4300 万个交错页面 |
| 参数量 | 训练 188M | 训练约 10B（cross-attn 层） |
| 计算 | 8 张 A100 数天 | 数千张 TPUv4 数周 |

预算有限的一图 VQA 选 BLIP-2。需要交错、few-shot 或多图推理时选 Flamingo/Idefics2。

## 实际使用

`code/main.py` 演示：

1. 在 36 个伪 patch token 和 8 个可学习 latent 上运行 Perceiver resampler（纯 Python cross-attention）。
2. 一个门控 cross-attention 步骤：`alpha = 0` 时输出等于输入（LLM 不变），然后 `alpha = 2.0` 时混入视觉贡献。
3. 一个交错 mask 构造器，为“(image 1) (text 1) (image 2) (text 2)”序列产生 2D attention mask。

## 交付成果

本课产出 `outputs/skill-gated-bridge-diagnostic.md`。给定开放 VLM 的 config（是否有 resampler、cross-attn 频率、gate 方案），它会识别 Flamingo 谱系元素并解释冻结策略。它可用于调试为什么某次 fine-tune 降低了文本性能（答案：gate 打开得太快、太宽）。

## 练习

1. 计算 Flamingo-9B 的视觉参数量：9B LLM + 1.4B 门控 cross-attention 层 + 64M resampler。总参数中有多大比例被训练？

2. 在 PyTorch 中实现门控 residual `y = tanh(alpha) * cross + x`。实验展示当 `alpha=0` 时，初始化处 `y==x` 精确成立。

3. 阅读 OpenFlamingo Section 3.2（arXiv:2308.01390），了解当 batch 中每个 prompt 的图像数量不同时它们如何处理多图。描述 padding 策略。

4. 为什么 Flamingo 的 cross-attention mask 让文本 token *只* attend 到最近的前置图像，而不是所有前置图像？阅读 Flamingo paper Section 2.4 并解释取舍。

5. 上下文 few-shot：为一个新的 Flamingo 变体构造一个包含 4 个“image → color of main object”示例的 prompt。描述当示例数量从 0 到 8 变化时预期的准确率模式。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Perceiver resampler | “Fixed-latent cross-attention” | 从可变数量输入 patch 产生 K 个固定 token 的模块 |
| Gated cross-attention | “Tanh-gated bridge” | residual 层 `y = tanh(alpha)*cross + x`，alpha 可学习，初始化为 0 |
| Interleaved input | “Mixed sequence” | 图像和文本按阅读顺序自由混合的 prompt 格式 |
| Frozen LLM | “No LLM gradients” | 文本 LLM 权重不更新；只训练 resampler + cross-attn 层 |
| Few-shot | “In-context examples” | 在 prompt 中给出少量（image, answer）对；模型无需 finetuning 即可泛化 |
| OBELICS | “Interleaved web corpus” | 包含 1.41 亿个网页的开放数据集，图像和文本按阅读顺序排列 |
| Chinchilla | “70B frozen base” | Flamingo 的冻结文本 LLM，来自 DeepMind 的 Chinchilla 论文 |
| Gate schedule | “How alpha moves” | 训练期间 cross-attention gate 打开的速度 |
| Cross-attn frequency | “Every M layers” | 插入门控 cross-attention block 的频率；Flamingo 使用 M=4 |
| OpenFlamingo | “Open reproduction” | MosaicML/LAION 的 3-9B 开放 checkpoint；架构与 Flamingo 相同 |

## 延伸阅读

- [Alayrac et al. — Flamingo (arXiv:2204.14198)](https://arxiv.org/abs/2204.14198) — 原始论文。
- [Awadalla et al. — OpenFlamingo (arXiv:2308.01390)](https://arxiv.org/abs/2308.01390) — 开放复现。
- [Laurençon et al. — OBELICS (arXiv:2306.16527)](https://arxiv.org/abs/2306.16527) — 交错网页语料。
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — 通用 Perceiver 架构。
- [Li et al. — Otter (arXiv:2305.03726)](https://arxiv.org/abs/2305.03726) — instruction-tuned Flamingo 后代。
- [Laurençon et al. — Idefics2 (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246) — Flamingo 方法的现代简化。
