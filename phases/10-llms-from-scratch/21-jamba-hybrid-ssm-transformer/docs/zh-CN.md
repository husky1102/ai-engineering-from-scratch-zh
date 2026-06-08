# Jamba — Hybrid SSM-Transformer

> State space models（SSMs）和 transformers 想要的东西不同。Transformers 通过注意力用二次成本购买质量。SSMs 通过 recurrence 以线性时间推理和常数内存购买速度，但质量落后。AI21 的 Jamba（2024 年 3 月）和 Jamba 1.5（2024 年 8 月）把它们放进同一个模型：每 7 个 Mamba layers 放 1 个 Transformer layer，每隔一个 block 放 MoE，并提供一个能装进单张 80GB GPU 的 256k context window。Mamba-3（ICLR 2026）用 complex-valued state spaces 和 MIMO projections 收紧了 SSM 侧。本课会端到端阅读这两个架构，并解释为什么这个 hybrid recipe 在 pure-SSM 和 pure-Transformer 长上下文尝试都没能活下来的三年扩展中仍然存活。

**类型:** Learn
**语言:** Python (stdlib, layer-mix calculator)
**先修:** Phase 10 · 14 (open-model architectures), Phase 10 · 17 (native sparse attention)
**时间:** ~60 分钟

## 学习目标

- 解释 Jamba block 中的三个 primitives——Transformer layers、Mamba layers、MoE——以及 1:7:even interleaving recipe。
- 从高层说出 SSM 的 recurrence 长什么样，以及为什么它能启用 constant-memory inference。
- 计算 Jamba model 在 256k context 下的 KV cache footprint，并与 pure-Transformer model 需要的内存比较。
- 说出三个 Mamba-3 innovations（exponential-trapezoidal discretization、complex-valued state update、MIMO）以及它们各自针对的问题。

## 要解决的问题

注意力相对序列长度是二次的。State space models 是线性的。这个差异会复利增长：在 256k tokens 时，一个 Transformer attention map 每个 head 有 65B entries；而 SSM 的 recurrent state 不随序列长度改变。

Pure-SSM models（Mamba、Mamba-2）在小规模上能匹配 Transformer perplexity，但在 state-tracking tasks 上落后，并且在一些 in-context retrieval 类别上失败。直觉是：SSMs 把历史压缩进一个 fixed state；当历史很长时，信息会泄漏。Attention 精确记住一切，但付出二次成本。

显然的修复：两者都用。把 Transformer layers 放在需要 exact recall 的地方。其他地方使用 SSM layers。调比例。Jamba 是第一个在规模上交付这个 hybrid recipe 的 production-grade model（52B total，12B active，256k context，single 80GB GPU）。Jamba 1.5 把家族扩展到 398B total / 94B active。Mamba-3（ICLR 2026）是当前最好的 pure-SSM baseline，hybrids 可以围绕它重建。

本课会阅读三篇论文，并产出“选择正确比例”的心智模型。

## 核心概念

### 一页讲完 SSM

State space model 通过一个固定大小 state `h` 处理序列 `x_1, ..., x_N`：

```text
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

每一步，state 通过线性动力学 `A` 演化，接收输入 `B x_t`，并输出 `C h_t`。`A, B, C` 都可以学习。注意关键性质：计算 `y_t` 只需要 `h_{t-1}` 和 `x_t`，不需要任何更早的 `x`。内存是常数。推理是每 token O(1)。

建模质量的技巧在于 `A` 的结构。S4（Gu 2021）使用一个高度结构化矩阵，训练时可以高效地作为长卷积求值。Mamba（Gu, Dao 2023）把固定的 `A, B, C` 换成 data-dependent ones（“selective” 的部分）。Mamba-2（2024）进一步简化结构。Mamba-3（2026）在特定位置重新加入复杂度。

关键性质：对一个 decoder LLM，SSM layer 可以 drop-in 替换 attention layer，用 fixed-size per-layer state 代替增长的 KV cache。

### Jamba block

Jamba block 按两个数字交错 layers：

- `l`：attention-to-Mamba ratio。Jamba 使用 `l = 8`，表示每 7 个 Mamba layers 有 1 个 Transformer layer（7 Mamba + 1 Attention = 每组 8 layers）。
- `e`：MoE frequency。Jamba 使用 `e = 2`，表示每隔一层应用 MoE。

Block 内部的 layer sequence：

```text
M  M  M  M  M  M  M  A    (7 Mamba + 1 Attention)
|  M  |  M  |  M  |  M    (where | marks MoE applied)
```

每个 Jamba block 是 8 层。深度为 4 个 blocks（总共 32 层）时，你会得到 28 个 Mamba 和 4 个 Attention layers。其中 16 个使用 MoE。

### 为什么是 1:7 比例

AI21 做了 ablations：什么 attention-to-Mamba 比例能在他们的 long-context evals 上给出最好的 perplexity-per-parameter 和 in-context recall？

- 太多 attention（1:1）：质量上升，但内存和速度变差。
- 太少 attention（1:15）：内存很好，但 in-context retrieval 失败。
- 甜点：1:7 或 1:8。

直觉是：Transformer layers 处理 exact recall 和 state tracking。Mamba layers 处理便宜的大部分计算。

### 位置编码

Mamba layers 自身具有位置感知能力（来自 recurrence）。原始 Mamba-based hybrids 中的 attention layers 不使用 RoPE——SSM layers 提供位置信息。Jamba 1.5 给 attention layers 添加 RoPE，以改善更长上下文泛化；这是基于经验 long-context evaluation 的事后 refinement。

### 内存预算

对一个 Jamba-1 shape（32 layers：28 Mamba + 4 Attention，hidden 4096，32 attention heads）：

- KV cache（只有 attention layers）：256k BF16 下为 `2 * 4 * 32 * 128 * 256k * 2 = 8.4 GB`。只有 4 个 attention layers 贡献。
- SSM state：`28 * hidden * state_size` per token prefix，但这是每层固定大小，不随 sequence length 缩放。典型 Mamba state 是每 feature 16，hidden 4096：总计 `28 * 4096 * 16 * 2 = 3.7 MB`。

与同 hidden、32 layers、full MHA at 32 heads 的 pure Transformer 比较：256k BF16 下是 `2 * 32 * 32 * 128 * 256k * 2 = 128 GB`。KV cache 降低 8 倍。即使对比大多数 2024 模型使用的 GQA(8) baseline（`2 * 32 * 8 * 128 * 256k * 2 = 32 GB`），Jamba 的 1:7 hybrid 在 16 GB 下仍然小 2 倍。

这就是 AI21 所说 “256k context on a single 80GB GPU” 的含义。Full-MHA pure Transformer 的 KV cache 装不下；即便 GQA baseline 也不给 weights 和 activations 留空间；Jamba 可以。

### Mamba-3：2026 年的 pure-SSM baseline

Mamba-3（ICLR 2026, arXiv:2603.15569）在 pure-SSM 侧引入了三个 innovations：

1. **Exponential-trapezoidal discretization.** 用更具表达力的 recurrence 替换 Mamba-2 中的 Euler-method discretization。在 core recurrence 内部对 state-input 应用 convolution-like operation，而不是对 `x_t` 做 outer convolution。

2. **Complex-valued state update.** 之前的 Mambas 把 state matrix 从 complex（S4）降到 real diagonal（Mamba）再到 scaled identity（Mamba-2）。Mamba-3 重新加入 complex values——等价于 state 上的 data-dependent rotary embedding。这恢复了之前 real-valued simplifications 损失的 state-tracking capabilities。

3. **Multi-input multi-output (MIMO) projections.** 不使用 per-feature scalar projections，而使用 matrix-valued projections。在不增加 decode latency 的情况下改善 modeling power 和 inference-time hardware utilization。

在 1.5B parameters 上，Mamba-3 相对 Gated DeltaNet 将 average downstream accuracy 提高 0.6 点；MIMO variant 再增加 1.2 点，总共提高 1.8 点。在相同 state size 下，Mamba-3 用一半 state 就能匹配 Mamba-2。

Mamba-3 还没有在大规模 production hybrid 中发布——但它显然是下一代 Jamba-class model 中 SSM 侧的候选者。

### 什么时候使用 hybrid

Hybrids 胜出于：

- Context 足够长，pure Transformer KV cache 开始痛苦（64k+）。
- 任务混合短程结构（适合 SSM）和长程 recall（需要 Transformer）。
- 你想部署到单 GPU 内存预算中，而 Transformer KV cache alone 放不下。

Hybrids 失利于：

- Context 很短（低于 16k）。SSM overhead 被浪费；pure Transformer 就很好。
- 任务需要 everywhere-to-everywhere attention（deep reasoning、multi-document cross-reference）。Hybrid 中 attention layers 的稀疏性会伤害表现。
- 你正在扩展到 trillion-parameter frontier models。Pure-Transformer + MLA + MoE（DeepSeek-V3 style）目前正在 capability race 中胜出。

### 竞争格局

| Model | Family | Scale | Unique claim |
|-------|--------|------|-------------|
| Mamba-2 | pure SSM | 3B | linear time, constant memory |
| Jamba | hybrid | 52B/12B | 256k on 80GB |
| Jamba 1.5 Large | hybrid | 398B/94B | enterprise-grade long-context |
| Mamba-3 | pure SSM | 1.5B (paper) | state-tracking restored |
| DeepSeek-V3 | pure Transformer + MoE | 671B/37B | frontier capability |

2026 年的格局：pure-Transformer MoE 主导 frontier，但 hybrids 拥有 256k-plus context niche。Mamba-3 的 state-tracking gains 可能会在下一代中把 hybrid ratios 推得更低（更多 SSM，更少 attention）。

## 实际使用

`code/main.py` 是 hybrid architectures 的 memory calculator。给定 SSM-Transformer ratio 以及 hidden-size / layer-count config，它会计算：

- 目标 context 下的 KV cache。
- SSM state memory。
- 一系列 model shapes 在 context N 下的 total memory。

Calculator 支持：

- Pure-Transformer baseline（KV cache 随 N 增长）。
- Jamba-style 1:7 hybrid。
- Pure-SSM（完全没有 KV cache）。

这些数字对已发布 shapes 直接来自 Jamba-1 和 Jamba-1.5 论文，对假设 variants 则是外推。

真实部署的集成注意事项：

- 大多数 production inference servers（vLLM、SGLang）支持 Jamba 和 Mamba。检查具体版本。
- 在 256k context 下，Jamba 的 memory advantage 会体现在 concurrent-request throughput 上。在同一 VRAM 上，你能装下比 Transformer sequences 更多的 Jamba sequences。
- Mamba-3 作为 standalone model 尚未在 production 中发布——目前是 1.5B 的 research preview。

## 交付成果

本课产出 `outputs/skill-hybrid-picker.md`。给定 workload specification（context length profile、task mix、memory budget），它会在 pure Transformer、Jamba-style hybrid 和 pure SSM 之间做推荐，并显式说明内存与质量取舍。

## 练习

1. 运行 `code/main.py`，计算一个 32-layer pure Transformer（hidden 4096，32 heads）和同 shape 的 Jamba-1 hybrid 在 256k context 下的 KV cache。验证 AI21 论文声称的约 8x memory reduction。

2. 修改 calculator，建模 1:3 hybrid（4 Mamba : 1 Attention）和 1:15 hybrid（14 Mamba : 1 Attention）。绘制 KV cache vs ratio。在什么比例下，KV cache 会等于 SSM state memory？

3. 阅读 Jamba paper（arXiv:2403.19887）Section 3。解释为什么 AI21 使用 Mamba-1 而不是更快的 Mamba-2。提示：hybrid ablation section 记录了原因。

4. 计算 Jamba 1.5 Large（398B total，94B active）中 MoE-every-other-layer 的 parameter overhead。把 active ratio 与 DeepSeek-V3（37B/671B）对比，并解释为什么 Jamba 的架构会把 active ratio 推高。

5. 阅读 Mamba-3 paper（arXiv:2603.15569）Section 3。用三句话解释为什么 complex-valued state update 等价于 data-dependent rotary embedding。把答案与 Phase 7 · Lesson 04 的 RoPE derivation 联系起来。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| State space model (SSM) | “带固定 state 的 recurrence” | 带学习 recurrence `h_t = A h_{t-1} + B x_t` 的层；每 token 常数内存 |
| Selective SSM | “Mamba 的技巧” | Data-dependent A, B, C parameters，让模型以线性时间获得类似 gating 的 selectivity |
| Attention-to-Mamba ratio | “多少 attention layers” | 在 Jamba 中，`l = 8` 表示每 7 个 Mamba layers 有 1 个 attention layer |
| Jamba block | “8-layer group” | 一个 attention + 七个 Mamba + 在交替位置上的 MoE |
| SSM state | “Hidden buffer” | 代替 Mamba layers KV cache 的 fixed-size per-layer state |
| 256k context | “Jamba 的旗舰数字” | Jamba-1 能在单张 80GB GPU 上容纳的序列长度；pure Transformer 在这个大小下不行 |
| Mamba-3 | “2026 pure SSM” | 当前最佳 pure-SSM 架构，带 complex state + MIMO；hybrids 围绕它重建的 baseline |
| MIMO | “Multi-input multi-output” | Mamba-3 innovation，使用 matrix-valued projections 代替 per-feature scalar |
| Exponential-trapezoidal discretization | “Mamba-3 的 recurrence” | 更有表达力的 recurrence，包含 Mamba-2 的 Euler-method discretization |
| Hybrid architecture | “混合 attention 和 SSM” | 任何交错 Transformer 和 SSM layers 的模型；Jamba 是 production archetype |

## 延伸阅读

- [Lieber et al. — Jamba: A Hybrid Transformer-Mamba Language Model (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) — 原始 Jamba paper，ratio ablations，256k context claim
- [AI21 — Jamba 1.5: Hybrid Transformer-Mamba at Scale (arXiv:2408.12570)](https://arxiv.org/abs/2408.12570) — 扩展后的家族，398B/94B 和 12B/52B public releases
- [Gu, Dao — Mamba: Linear-Time Sequence Modeling with Selective State Spaces (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752) — Jamba 所基于的 selective SSM paper
- [Dao, Gu — Mamba-2 (arXiv:2405.21060)](https://arxiv.org/abs/2405.21060) — 简化的 structured-state-space successor
- [Lahoti et al. — Mamba-3 (arXiv:2603.15569, ICLR 2026)](https://arxiv.org/abs/2603.15569) — complex-valued state、MIMO，以及 2026 pure-SSM frontier
- [Gu et al. — Efficiently Modeling Long Sequences with Structured State Spaces (arXiv:2111.00396)](https://arxiv.org/abs/2111.00396) — S4 paper，LLM 中 SSM genealogy 的起点
