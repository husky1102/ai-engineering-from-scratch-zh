# Show-o 与 Discrete-Diffusion 统一模型

> Transfusion 混合连续与离散表示。Show-o（Xie et al., 2024 年 8 月）走向另一边：文本 token 使用 causal next-token prediction，图像 token 使用 MaskGIT 风格的 masked discrete diffusion。二者位于一个带 hybrid attention mask 的 transformer 中。结果是在一个 backbone、每种模态一个 tokenizer、一个 loss formulation（next-token 扩展到 masked prediction）上统一 VQA、text-to-image、inpainting 和 mixed-modality generation。本课讲解 Show-o 设计：为什么 masked discrete diffusion 是一种并行、少步数的图像生成器，并将它与 Transfusion 和 Emu3 对比。

**类型:** Learn
**语言:** Python (stdlib, masked-discrete-diffusion sampler)
**先修:** Phase 12 · 13 (Transfusion)
**时间:** ~120 minutes

## 学习目标

- 解释 masked discrete diffusion：均匀 mask token，然后让 transformer 恢复它们的 schedule。
- 比较 parallel image decoding（Show-o、MaskGIT）与 autoregressive image decoding（Chameleon、Emu3）在速度和质量上的差异。
- 说出 Show-o 在一个 checkpoint 中处理的三个任务：T2I、VQA、image inpainting。
- 选择一种 masking schedule（cosine、linear、truncated），并推理它对 sample quality 的影响。

## 要解决的问题

Transfusion 的双损失训练有效，但动态更棘手：连续 diffusion loss 与离散 NTP loss 位于不同的数值尺度。平衡 loss weights 是一次超参数搜索。架构有效，但复杂。

Show-o 的答案是：像 Chameleon 一样让两种模态都保持离散，但通过 masked discrete diffusion 并行生成图像，而不是顺序生成。训练目标变成一个单一的 masked-token-prediction，它自然推广了 next-token-prediction。

## 核心概念

### Masked discrete diffusion（MaskGIT）

原始的 Chang et al. (2022) MaskGIT 技巧很优雅。从一个完全 masked 的图像开始（每个 token 都是特殊 `<MASK>` id）。每一步，并行预测所有 masked tokens，然后保留 top-K 置信度最高的预测，重新 mask 其余位置。约 8-16 次迭代后，所有 token 都被填满。每一步 unmask 多少 token 的 schedule 需要调节，cosine schedules 表现很好。

训练很简单：从 [0, 1] 均匀采样 masking ratio，把它应用到图像的 VQ tokens 上，训练 transformer 恢复被 mask 的 token。这正是 BERT 对文本做过的事，只是扩展到图像生成。

### Show-o：一个 transformer，hybrid mask

Show-o 把 MaskGIT 放进 causal-language-model transformer。attention mask 是：

- 文本 token：causal（标准 LLM）。
- 图像 token：在 image block 内 full bidirectional（这样 masked tokens 在预测时可以看到所有其他 image tokens）。
- Text-to-image：文本 attend 到之前的图像，图像 attend 到之前的文本。

训练在以下任务之间切换：
1. 文本序列上的标准 NTP。
2. T2I samples：text → image，其中 image tokens 被 mask，使用 masked-token-prediction loss。
3. VQA samples：image → text，其中 text tokens 被 mask（本质上仍是 NTP）。

统一损失是 `<MASK>` tokens 上的 cross-entropy，覆盖文本 NTP（只有最后一个 token 被“mask”）和图像 masked-diffusion（随机子集被 mask）。

### Parallel sampling

Show-o 生成一张图像大约需要 16 steps，而不是约 1000 次（按 token 自回归）或约 20 次（diffusion）。每一步并行预测所有 masked tokens；提交 top-K 置信度高的 token；重复。

比较：
- Chameleon / Emu3（token 上的自回归）：N_tokens 次 forward passes，通常每张图像 1024-4096 次。
- Transfusion（continuous diffusion）：约 20 steps，每步一次完整 transformer pass。
- Show-o（masked discrete diffusion）：约 16 steps，每步一次完整 transformer pass。

Show-o 比同规模模型上的 Chameleon 更快，大致匹配 Transfusion 的 step count，但每步成本更低（离散 vocab logits vs continuous MSE loss）。

### 一个 checkpoint 中的任务

Show-o 在推理时支持四个任务，由 prompt format 选择：

- 文本生成：标准 autoregressive text output。
- VQA：图像输入，文本输出。
- T2I：文本输入，通过 masked discrete diffusion 输出图像。
- Inpainting：输入部分 token 被 mask 的图像，填补缺失部分。

inpainting 能力来自 masked-prediction training，是免费得到的。把 VQ-token grid 的一个区域 mask 掉，喂入其余部分和 text prompt，预测 masked tokens。

### Masking schedule

每一步 unmask 多少 token 的 schedule 会塑造质量。Show-o 推荐 cosine：

```text
mask_ratio(t) = cos(pi * t / (2 * T))   # t = 0..T
```

step 0 时，所有 token 都 masked（ratio 1.0）。step T 时，没有 token 被 mask。Cosine 把质量集中在 mid-range ratios 上，此时预测最有信息量。Linear schedules 也能工作，但更快进入平台期。

### Show-o2

Show-o2（2025 follow-up，arXiv 2506.15564）扩展了 Show-o：更大的 LLM base、更好的 tokenizer、改进的 mask schedule。同样的架构模式。

### Show-o 的位置

在 2026 年的分类法中：

- Discrete tokens + NTP：Chameleon、Emu3。简单，但推理慢。
- Discrete tokens + masked diffusion：Show-o、MaskGIT、LlamaGen、Muse。并行 sampling，仍受 tokenizer 有损限制。
- Continuous + diffusion：Transfusion、MMDiT、DiT。最高质量，训练更复杂。
- Continuous + flow matching in a VLM：JanusFlow、InternVL-U。最新。

按任务选择：当你希望一个 open model 同时提供 T2I + inpainting + VQA 且速度合理时，用 Show-o；当质量最重要且你承担得起双损失 plumbing 时，用 Transfusion。

## 实际使用

`code/main.py` 模拟 Show-o sampling：

- 一个包含 16 个 VQ tokens 的玩具 grid。
- 一个 mock “transformer”，基于 prompt 和当前 unmasked tokens 预测 logits。
- 使用 cosine schedule，在 8 steps 中做 parallel masked sampling。
- 打印中间状态（mask pattern evolution）和最终 tokens。

运行它，观察 mask 如何一步步消解。

## 交付成果

本课产出 `outputs/skill-unified-gen-model-picker.md`。给定一个既需要理解（VQA、captioning）又需要生成（T2I、inpainting），并且有 open-weights 约束的产品，它会在 Show-o family、Transfusion/MMDiT family、Emu3 / Chameleon family 之间选择，并给出具体权衡。

## 练习

1. Masked discrete diffusion 大约用 16 steps 采样。为什么不是 1 step？如果在 step 0 就 unmask everything，会坏在哪里？

2. Inpainting 对 masked diffusion 是免费的。提出一个真实或假想的产品用例，其中 Show-o 的 inpainting 胜过 specialist model。

3. Cosine schedule vs linear schedule：追踪 T=8 时每一步 unmasked tokens 的数量。哪个更平衡？

4. 一张 512x512 的 Show-o 图像是 1024 个 token。若 vocab K=16384，模型发出 1024 * log2(16384) = 14,336 bits（约 1.75 KiB）数据。Stable Diffusion 输出 512*512*24 bits = 6,291,456 bits（约 768 KiB）的 raw pixels。压缩率是多少？它用质量换来了什么？

5. 阅读 LlamaGen（arXiv:2406.06525）。LlamaGen 的 class-conditional autoregressive image model 与 Show-o 的 masked approach 有何不同？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Masked discrete diffusion | “MaskGIT-style” | 训练模型预测 masked tokens；推理时迭代 unmask 最高置信度预测 |
| Cosine schedule | “Unmask schedule” | 推理步骤中 mask ratio 的衰减；让置信度增长集中在中间区间 |
| Parallel decoding | “All tokens at once” | 每一步在一个 forward pass 中预测完整 masked token 序列，然后提交 top-K |
| Hybrid attention | “Causal + bidirectional” | 对文本 token 使用 causal、对 image blocks 内部使用 bidirectional 的 mask |
| Inpainting | “Fill-in generation” | 以部分 token 被 mask 的图像为条件，预测缺失部分；来自训练目标的免费能力 |
| Commitment rate | “Top-K per step” | 每次迭代有多少 token 被宣布为“done”；控制推理与质量的权衡 |

## 延伸阅读

- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
- [Show-o2 (arXiv:2506.15564)](https://arxiv.org/abs/2506.15564)
- [Chang et al. — MaskGIT (arXiv:2202.04200)](https://arxiv.org/abs/2202.04200)
- [Sun et al. — LlamaGen (arXiv:2406.06525)](https://arxiv.org/abs/2406.06525)
- [Chang et al. — Muse (arXiv:2301.00704)](https://arxiv.org/abs/2301.00704)
