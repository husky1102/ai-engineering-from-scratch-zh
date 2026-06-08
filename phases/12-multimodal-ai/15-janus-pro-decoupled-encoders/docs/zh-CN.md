# Janus-Pro：统一多模态模型的解耦 Encoder

> 统一多模态模型有一个不可避免的张力。理解需要语义特征：SigLIP 或 DINOv2 输出 rich with concept-level information 的向量。生成需要便于重建的 codes：能够组合回清晰像素的 VQ tokens。这两个目标无法在单个 encoder 中兼容。Janus（DeepSeek，2024 年 10 月）和 Janus-Pro（DeepSeek，2025 年 1 月）认为修复办法是停止强求：解耦两个 encoder。任务之间共享 transformer body，但理解经由 SigLIP，生成经由 VQ tokenizer。在 7B 规模下，Janus-Pro 在 GenEval 上超过 DALL-E 3，同时在 MMMU 上匹配 LLaVA。本课阅读为什么两个 encoder 能解决一个 encoder 失败的问题。

**类型:** Build
**语言:** Python (stdlib, dual-encoder routing + shared-body signal)
**先修:** Phase 12 · 13 (Transfusion), Phase 12 · 14 (Show-o)
**时间:** ~120 minutes

## 学习目标

- 解释为什么单个共享 encoder 会在理解质量或生成质量上做出妥协。
- 描述 Janus-Pro 的 routing：理解时输入侧使用 SigLIP features，生成时输入与输出都使用 VQ tokens。
- 追踪让 Janus-Pro 成功而 Janus 没成功的数据混合扩展。
- 比较 decoupled（Janus-Pro）、coupled-continuous（Transfusion）和 coupled-discrete（Show-o）架构。

## 要解决的问题

统一模型在理解和生成之间共享 transformer body。之前的尝试（Chameleon、Show-o、Transfusion）都用一个视觉 tokenizer 同时服务两个方向。tokenizer 是一种妥协：

- 为重建优化（生成）：VQ-VAE 捕获细粒度 pixel detail，但产生的 token 语义一致性较弱。
- 为语义优化（理解）：SigLIP embeddings 会把“cat”图像放到“cat”token 附近，但不能很好重建。

Show-o 与 Transfusion 都为此在某个方向上付出了可见的质量税。Janus-Pro 问：当任务需求不同，为什么还要强迫使用一个 tokenizer？

## 核心概念

### 解耦视觉编码

Janus-Pro 的架构拆开了两个 encoder：

- 理解路径。Input image → SigLIP-SO400m → 2-layer MLP → transformer body。
- 生成路径。Input image（如果基于现有图像 conditioning）→ VQ tokenizer → token IDs → transformer body。
- 输出生成。Image tokens 由 transformer 预测 → VQ decoder → pixels。

transformer body 是共享的。body 上游和下游的所有东西则是 task-specific。

输入由 prompt format 区分：`<understand>` tag 走 SigLIP；`<generate>` 走 VQ。或者 routing 由任务隐式决定。

### 为什么这能工作

理解损失获得 SigLIP features，而 CLIP-style pretraining 已经把它们调成适合 semantic similarity。因为输入特征更适合任务，模型的 perception benchmarks 比 Show-o / Transfusion 更好。

生成损失获得 VQ tokens，而 tokenizer 已经针对重建做过调优。因为 VQ codes 能干净地组合回像素，图像质量优于 Show-o。

共享 transformer body 会看到两种输入分布（SigLIP 与 VQ），并学会同时处理二者。主张是：只要数据足够、参数足够，body 就能吸收这种切换。

### 数据扩展：Janus vs Janus-Pro

Janus（原始版本，arXiv 2410.13848）引入了解耦，但规模较小（1.3B params，数据有限）。Janus-Pro（arXiv 2501.17811）做了扩展：

- 7B params（相对 1.3B）。
- stage 1（alignment）使用 90M image-text pairs，高于 72M。
- stage 2（unified）使用 72M，高于 26M。
- stage 3 增加 200k image-gen instruction samples。

结果：Janus-Pro-7B 在 MMMU 上匹配 LLaVA（60.3 vs ~58），并在 GenEval 上超过 DALL-E 3（0.80 vs 0.67）。一个 open model，在 unified spectrum 的两端都具备竞争力。

### JanusFlow：rectified flow 变体

JanusFlow（arXiv 2411.07975）把 VQ generation path 换成 rectified-flow generation path（连续）。拆分变成 SigLIP-for-understanding + rectified-flow-for-generation。质量上限进一步抬升。架构仍然是 decoupled-encoders-shared-body。

### shared body 的工作

transformer body 处理统一序列，但面对两种输入分布。它的工作是：

- 对理解：消费 SigLIP features + text tokens → autoregressively 发出文本。
- 对生成：消费 text tokens +（可选 image VQ tokens）→ autoregressively 发出 image VQ tokens。

body 在每个 block 内没有 modality-specific weights。它就是你预期会在 Qwen 或 Llama 内部看到的 text-style transformer，再加上两个 input adapters。

有趣的是，这意味着 Janus-Pro 的 body 可以从预训练 LLM 初始化。Janus-Pro 确实从 DeepSeek-MoE-7B 初始化。这一选择很重要：LLM 带来 reasoning ability，而从零开始的 unified models 很难达到这种能力。

### 与 InternVL-U 对比

InternVL-U（Lesson 12.10）是 2026 年的后续。它组合了：

- Native multimodal pretraining（InternVL3 backbone）。
- Decoupled-encoder routing（SigLIP 输入，VQ + diffusion heads 输出）。
- 统一理解 + 生成 + 编辑。

InternVL-U 把 Janus-Pro 的架构选择吸收到一个更大的框架中。decoupled-encoder idea 现在已成为大规模 unified models 的默认选择。

### 局限

解耦 encoders 增加了架构复杂度。两个 tokenizers 要训练，两条输入路径要维护，两组 failure modes。对于不需要生成的产品，Janus-Pro 是过度工程：选择 LLaVA-family 理解模型。

对于不需要理解的产品，Janus-Pro 也过度：选择 Stable Diffusion 3 / Flux 模型。

对于两者都需要的产品，Janus-Pro 现在是参考 open architecture。

## 实际使用

`code/main.py` 模拟 Janus-Pro routing：

- 两个 mock encoders：SigLIP-like（产生 256-dim semantic vectors）与 VQ-like（产生 integer codes）。
- 一个 prompt router，基于 task tag 选择 encoder。
- 一个 shared body（stand-in），不管 token sequence 来自哪个 encoder 都能处理。
- 一个从 stage 1（alignment）切换到 stage 3（instruction tune）的 weighted-sample schedule。

打印 3 个示例的 routed paths：image QA、T2I、image editing。

## 交付成果

本课产出 `outputs/skill-decoupled-encoder-picker.md`。给定一个希望在 frontier-ish quality 上统一 generation + understanding 的产品，它会选择 Janus-Pro、JanusFlow 或 InternVL-U，并给出具体的数据规模建议。

## 练习

1. Janus-Pro-7B 在 GenEval 上超过 DALL-E 3。解释为什么一个 7B open model 能在生成上匹配 frontier proprietary model，但在理解上不能。

2. 实现一个 router function：给定 prompt text，分类为 `understand` 或 `generate`。如何处理像“describe and then sketch”这样的模糊 prompt？

3. JanusFlow 用 rectified flow 替换 VQ path。transformer body 现在输出什么，loss 有何变化？

4. 提出一个 Janus-Pro 架构可以通过再加一个 decoupled encoder 处理的第四种任务。例子：image segmentation（DINO-style）、depth（MiDaS-style）。

5. 阅读 Janus-Pro Section 4.2 中的数据扩展内容。哪个数据 stage 对 T2I 质量相对 Janus 的提升贡献最大？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Decoupled encoding | “Two visual encoders” | 每个方向使用独立 tokenizer 或 encoder：理解用语义 encoder，生成用重建 encoder |
| Shared body | “One transformer” | 单个 transformer 处理任一 encoder 的输出；没有 modality-specific weights |
| SigLIP for understanding | “Semantic features” | CLIP-family vision tower，提供丰富概念特征，但重建能力差 |
| VQ for generation | “Reconstruction codes” | vector-quantized tokens，能干净地解码回像素 |
| JanusFlow | “Rectified-flow variant” | Janus-Pro 的变体，用连续 flow-matching generation head 代替 VQ |
| Routing tag | “Task tag” | 选择输入 encoder 的 prompt marker（`<understand>` / `<generate>`） |

## 延伸阅读

- [Wu et al. — Janus (arXiv:2410.13848)](https://arxiv.org/abs/2410.13848)
- [Chen et al. — Janus-Pro (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)
- [Ma et al. — JanusFlow (arXiv:2411.07975)](https://arxiv.org/abs/2411.07975)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Dong et al. — DreamLLM (arXiv:2309.11499)](https://arxiv.org/abs/2309.11499)
