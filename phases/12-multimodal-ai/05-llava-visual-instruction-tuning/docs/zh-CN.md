# LLaVA 与视觉指令微调

> LLaVA（2023 年 4 月）是地球上被复制最多的多模态架构。它用 2 层 MLP 替代 BLIP-2 的 Q-Former，用朴素 token 拼接替代 Flamingo 的门控 cross-attention，并在 15.8 万个由 GPT-4 从纯文本描述生成的视觉指令轮次上训练。2023 到 2026 年间，任何构建 VLM 的实践者几乎都构建过某个 LLaVA 变体。LLaVA-1.5 加入 AnyRes。LLaVA-NeXT 提升分辨率。LLaVA-OneVision 用一个 recipe 统一单图、多图和视频。本课阅读这个 recipe，实现 projector，并解释为什么“更简单的方案赢了”。

**类型:** Build
**语言:** Python（stdlib，projector + instruction-template builder）
**先修:** Phase 12 · 02（CLIP），Phase 11（LLM Engineering — instruction tuning）
**时间:** ~180 分钟

## 学习目标

- 构建一个 2 层 MLP projector，把 ViT patch embedding（dim 1024）映射到 LLM 的 embedding dim（dim 4096）。
- 走读 LLaVA 的两阶段 recipe：（1）在 55.8 万 caption pair 上做 projector alignment，（2）在 15.8 万 GPT-4 生成轮次上做 visual instruction tuning。
- 构造 LLaVA 格式 prompt，包含 image token placeholder、system prompt 以及 user/assistant 轮次。
- 解释为什么社区从 Q-Former 转向 MLP，尽管 Q-Former 在 token budget 上更省。

## 要解决的问题

BLIP-2 的 Q-Former（Lesson 12.03）把一张图像压缩成 32 个 token。它干净、高效，也适合 benchmark。但它有两个问题。

第一，Q-Former 可训练，但它的 loss 不是最终任务。Stage 1 训练 ITC+ITM+ITG。Stage 2 训练 LM loss。query 学到的是某种中间表示，LLM 随后还要解码它。信息会在瓶颈里丢失。

第二，Q-Former 有 188M 参数，并且在 LLaVA 的 2023 年尺度下，你必须让它和目标 LLM 共同设计。换 LLM，要重训 Q-Former。换 vision encoder，也要重训。每个组合都是一个单独的研发项目。

LLaVA 的答案简单到有点尴尬：取 ViT 的 576 个 patch token，让每个 token 通过一个 2 层 MLP（`1024 → 4096 → 4096`），然后把全部 576 个 token 丢进 LLM 的输入序列。没有瓶颈。没有基于奇怪目标的 stage 1 预训练。只用直接的 LM loss 训练 MLP。

数据从哪里来？LLaVA 的第二个洞见是：用 GPT-4（纯文本）生成指令数据。把某张图像的 COCO caption 和 bounding-box 数据喂给 GPT-4，让它生成对话、描述和复杂推理问题。免费得到 15.8 万个 instruction-response 轮次。无需人工标注。

结果是：一个 VLM 在 8 张 A100 上跑一天，MMMU 上超过 Flamingo，并发布了社区可扩展的开放 checkpoint。到 2023 年末，它已经衍生出 50 多个 fork。

## 核心概念

### 架构

LLaVA-1.5 13B：
- Vision encoder：CLIP ViT-L/14 @ 336（stage 1 冻结，stage 2 可选解冻）。
- Projector：带 GELU activation 的 2 层 MLP，`1024 → 4096 → 4096`。
- LLM：Vicuna-13B（后来是 Llama-3.1-8B）。

图像 + 文本 prompt 的 forward pass：

```text
img -> ViT -> 576 patches of dim 1024
patches -> MLP -> 576 tokens of dim 4096
prompt: system + "<image>" placeholder + user question
replace <image> token with the 576 projected tokens
feed the full sequence to the LLM
decode response
```

图像占用 LLM context 中的 576 个 token。在 2048 context 下，这留下 1472 个文本 token。在 32k context 下，它几乎只是一个舍入误差。

### Stage 1：projector alignment

冻结 ViT。冻结 LLM。只训练 2 层 MLP。数据集：55.8 万 image-caption pair（LAION-CC-SBU）。Loss：在 projected image token 条件下，对 caption 做 language modeling。

batch 128 单 epoch 几小时内即可完成。projector 学会把 ViT-space 映射到 LLM-space。没有任务特定 supervision。

### Stage 2：visual instruction tuning

解冻 projector（仍然可训练）。解冻 LLM（通常全量，有时 LoRA）。在 15.8 万个 visual-instruction 轮次上训练。

指令数据是关键。Liu et al. 这样生成它：
1. 取一张 COCO 图像。
2. 提取文本描述（5 条人工 caption + bounding-box list）。
3. 用三个 prompt template 发给 GPT-4：
   - Conversation：“Generate a back-and-forth dialogue between a user and assistant about this image.”
   - Detailed description：“Give a rich, detailed description of the image.”
   - Complex reasoning：“Ask a question that requires reasoning about the image, then answer it.”
4. 把 GPT-4 输出解析成（instruction, response）对。

这些步骤都不直接接触图像，只使用文本描述。GPT-4 会幻觉出合理的图像内容。存在噪声，但它有效：15.8 万轮足以解锁对话能力。

### 为什么社区复制它

- 不需要调 stage-1-specific loss。始终使用 LM loss。
- Projector 训练耗时是小时，不是天。
- 通过只重训 projector 即可替换 LLM（LLaVA-Llama2、LLaVA-Mistral、LLaVA-Llama3）。
- Visual-instruction 数据流水线使用 GPT-4，针对新领域重新生成也便宜。

### LLaVA-1.5 与 LLaVA-NeXT

LLaVA-1.5（2023 年 10 月）加入：
- 把学术任务数据（VQA、OKVQA、RefCOCO）混入 instruction tuning。
- 更好的 system prompt。
- 2048 → 32k context。

LLaVA-NeXT（2024 年 1 月）加入：
- AnyRes：把高分辨率图像切成 2x2 或 1x3 的 336x336 crop 网格，再加一个全局低分辨率 thumbnail。每个 crop 变成 576 个 token；每张图像总计约 2880 个视觉 token。OCR 和 chart 任务跃升。
- 使用 ShareGPT4V（高质量 GPT-4V caption）的更好 instruction data mixture。
- 更强的基础 LLM（Mistral-7B、Yi-34B）。

### LLaVA-OneVision

Lesson 12.08 会深入讲 OneVision。简短版本：相同 projector，但使用覆盖单图、多图和视频的 curriculum 训练一个模型，共享 visual-token budget。

### 与 Q-Former 对比

| | Q-Former（BLIP-2） | MLP（LLaVA） |
|---|---|---|
| 每图视觉 token | 32 | 576（base）或 2880（AnyRes） |
| 可训练参数 | 188M + LM | 40M + LM |
| Stage 1 loss | ITC+ITM+ITG | 仅 LM |
| LLM drop-in | 需要重训 | 最小重训即可替换 |
| 多图 | 别扭 | 自然（concat） |
| 视频 | 别扭 | 自然（逐帧 concat） |
| Token budget | 小 | 大 |

MLP 赢在简单性和 token 灵活性。Q-Former 赢在 token budget。到 2023 年末，token budget 不再是主要瓶颈（LLM context 增长到 32k-128k+），简单性占了上风。

### Prompt 格式

```text
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...
```

`<image>` 是 placeholder token。在 tokenization 之前，它被替换为 576 个视觉 token（AnyRes 下为 2880）。Tokenizer 看到的序列比训练时略长，但 LLM 可以处理这种新输入，因为 stage 1 已经教过它。

### 参数经济性

LLaVA-1.5-7B 拆分：
- CLIP ViT-L/14 @ 336：303M（stage 1 冻结，stage 2 通常解冻）。
- Projector（2x linear）：约 22M 可训练。
- Llama-7B：7B。
- 总计：7.3B 参数。Stage 2 可训练：完整 7B + 22M projector。

Stage 2 训练成本：8xA100 约 20 小时。这是关键数字：一天、一个节点、可复现。这就是 LLaVA 扩散的原因。

## 实际使用

`code/main.py` 实现：

1. 纯 Python 中的 2 层 MLP projector（toy 尺度 dim 16 → 32 → 32）。
2. Prompt 构建流水线：system prompt + 被 N 个 projected token 替换的 `<image>` + user turn + assistant generation placeholder。
3. 一个可视化器，展示 576-token 视觉块在 LLM context 中的样子（占 2k / 32k / 128k context 的百分比）。

## 交付成果

本课产出 `outputs/skill-llava-vibes-eval.md`。给定一个 LLaVA-family checkpoint，它运行一套 10-prompt vibes-eval（3 个 captioning、3 个 VQA、2 个 reasoning、2 个 refusal）并报告人类可读的 scorecard。它不是 benchmark，而是 smoke test，用来确认 projector 和 LLM 连接良好。

## 练习

1. 计算 `1024 → 4096 → 4096` 的 2 层 MLP projector 的可训练参数量。带 GELU 和 bias 时，它占 LLaVA-13B 的多大比例？

2. 为一个“refusal”场景构造 LLaVA prompt：图像中包含一个私人个体。写出期望的 assistant response。为什么 LLaVA 应该 zero-shot 拒绝，强化这种拒绝需要什么训练数据？

3. 阅读 LLaVA-NeXT blog 的 AnyRes 部分。计算 1344x672 图像在 AnyRes 下的视觉 token 数。与 336x336 的 base 576 token 对比。

4. LLaVA stage-1 projector 使用 caption 的 LM loss 训练。如果跳过 stage 1，直接进入 stage 2（visual instruction tuning）会怎样？引用 Prismatic VLMs ablation（arXiv:2402.07865）作答。

5. LLaVA-Instruct-150k 使用 GPT-4 和 COCO caption 生成 instruction。对于新领域（医学 X-ray、卫星图像），描述生成领域 instruction 的四步数据流水线。每一步可能出什么问题？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Projector | “MLP bridge” | 带 GELU 的 2 层 MLP，把 ViT dim 映射到 LLM dim |
| Image token | “<image> placeholder” | 推理前被 N 个 projected visual token 替换的 prompt marker |
| Visual instruction tuning | “LLaVA stage 2” | 在 GPT-4 生成的（image, instruction, response）三元组上训练 |
| Stage 1 alignment | “Projector pretraining” | 冻结 ViT 和 LLM，用 caption 的 LM loss 训练 projector |
| AnyRes | “Multi-crop tiling” | 把高分辨率图像切成 tile 网格并拼接每个 tile 的视觉 token |
| LLaVA-Instruct | “GPT-4-generated” | 由 COCO caption + GPT-4 合成的 15.8 万个 instruction-response 对 |
| Vision encoder freeze | “Backbone locked” | CLIP 权重在 stage 1 不更新，有时在 stage 2 也不更新 |
| ShareGPT4V | “Better captions” | GPT-4V 生成的 100 万 dense caption，用于更高质量 alignment |
| VQA | “Visual question answering” | 回答关于图像的自由形式问题的任务 |
| Prismatic VLMs | “Design-space paper” | Karamcheti 2024 ablation，系统测试 projector 和数据选择 |

## 延伸阅读

- [Liu et al. — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) — LLaVA 论文。
- [Liu et al. — Improved Baselines with Visual Instruction Tuning (arXiv:2310.03744)](https://arxiv.org/abs/2310.03744) — LLaVA-1.5。
- [Chen et al. — ShareGPT4V (arXiv:2311.12793)](https://arxiv.org/abs/2311.12793) — dense captions 数据集。
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865) — design-space ablations。
- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326) — 统一单图、多图、视频。
