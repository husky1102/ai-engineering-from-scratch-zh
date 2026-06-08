# 原生稀疏注意力（DeepSeek NSA）

> 在 64k token 时，注意力会吃掉 70-80% 的解码延迟。每个开放模型实验室都有一个修它的方案。DeepSeek 的 NSA（ACL 2025 best paper）是留下来的那一个：三条并行注意力分支——压缩后的粗粒度 token、选择性保留的细粒度 token，以及用于局部上下文的滑动窗口——再通过一个学习到的门控组合起来。它对硬件友好（kernel-friendly）、原生可训练（用于预训练，而不是推理时外挂），并且在 64k 解码上比 FlashAttention 更快，同时质量持平或超过完整注意力。本课会端到端构建这三条分支，并说明为什么这种稀疏性可以端到端可微。

**类型:** Build
**语言:** Python (stdlib)
**先修:** Phase 7 · 12 (KV cache, flash-attention), Phase 7 · 15 (attention variants), Phase 10 · 16 (differential attention)
**时间:** ~60 分钟

## 学习目标

- 说出 NSA 的三条注意力分支，以及每条分支捕捉什么信息。
- 解释为什么 NSA 是“原生可训练”的，而之前的稀疏注意力方法多是仅限推理。
- 在 64k 上下文下，把压缩块大小和 selection top-k 作为变量，计算 NSA 相对完整注意力节省了多少注意力计算。
- 用 stdlib Python 在一段短合成序列上实现三分支组合，并验证门控权重的行为合理。

## 要解决的问题

完整注意力在序列长度 N 下需要 `O(N^2)` 时间，每层 KV cache 是 `O(N)`。到了 64k token，计算和内存带宽都会变得灾难性。NSA 论文中的理论测算是：在 64k 时，注意力占总解码延迟的 70-80%。下游的一切——TTFT、tokens/sec、每百万 token 成本——都会被注意力成本主导。

稀疏注意力是显然的答案。之前的尝试大致分成两类。固定模式稀疏（sliding-window、strided、block-local）会丢信息，在长程召回任务上失败。推理时稀疏（KV cache pruning、H2O、StreamingLLM）应用在一个用 dense attention 预训练出的模型上，只能拿回一部分潜在加速，因为模型从未被要求通过这种稀疏模式路由信息。

Native Sparse Attention（Yuan et al., DeepSeek + PKU + UW, ACL 2025 best paper, arXiv:2502.11089）两者都做到了：一个模型在预训练期间学习的稀疏模式，加上一个 kernel-aligned 的算法实现，推理时真的能交付计算节省。两年后，NSA 或它的直接后代很可能会成为所有前沿长上下文模型的默认注意力。

## 核心概念

### 三条并行分支

对每个 query，NSA 会对 KV cache 的三种不同视图各跑一次注意力：

1. **压缩分支。** Token 被分组成大小为 `l` 的块（通常是 32 或 64）。每个块通过一个小型学习 MLP 压缩成一个 summary token。Query 对这些压缩 token 做注意力，从而获得整个序列的粗粒度视图。

2. **选择分支。** 使用压缩分支的注意力分数，找出与当前 query 最相关的 top-k 个块。再读取这些块里的细粒度（未压缩）token，并让 query 对它们全部做注意力。可以把压缩分支注意力理解成 selection 的路由信号。

3. **滑动窗口分支。** Query 关注最近的 `W` 个 token（通常是 512）以获得局部上下文。这条分支捕捉结构密集的短程模式（语法、局部共指），这些模式可能被另外两条分支漏掉。

三条分支的输出通过一个学习到的逐位置门控组合：

```text
out = g_cmp * out_cmp + g_sel * out_sel + g_win * out_win
```

`g_cmp, g_sel, g_win` 是一个小 MLP 根据 query 输出的门控权重。它们不一定要和为 1，可以独立地给各分支加权。

### 为什么这是“原生可训练”

选择步骤（top-k blocks）是离散的。离散操作会打断梯度流。之前的稀疏注意力工作要么跳过 selection 的反向传播（限制训练），要么使用连续松弛，但推理时拿不到真正的稀疏性。

NSA 绕开了这个问题：压缩分支注意力本身就是对整个序列的可微粗粒度注意力。Top-k 操作只是复用压缩分支的最高注意力分数，决定要加载哪些细粒度块。梯度会流过压缩分支分数（它们同时影响压缩输出和选择逻辑），被选中块对最终输出的贡献也是可微的。不可微的 `top_k` 操作在前向计算图中只是 no-op——它只控制从内存加载哪些块。

这就是 NSA 可以端到端用于预训练的原因。模型会联合学习如何通过三条分支路由信息，得到一种稀疏模式；推理时，这种模式真的能带来承诺的加速。

### 硬件对齐的 kernel

NSA 的 kernel 是为现代 GPU 内存层次设计的。Kernel 按 GQA 组加载 query（外层循环），为每组取出对应的稀疏 KV 块（内层循环），并在 SRAM 上运行注意力。因为每个 query group 看到相同的 selected blocks（selection 是 per-query-group，而不是 per-query-head），KV 加载可以在组内摊销。算术强度保持较高。

论文报告 Triton kernel 在 64k 解码上比 FlashAttention 快 9 倍，并且速度提升比例随序列长度增长而增长。前向和反向 kernel 都有提供。

### 计算预算

令 `N` 为序列长度，`l` 为压缩块大小，`k` 为 top-k selection 数量，`w` 为滑动窗口，`b` 为选中块大小（通常等于 `l`）。

- 压缩分支：每个 query 有 `O(N/l)` 个 key，因此总量为 `O(N * N / l)`。
- 选择分支：每个 query 有 `O(k * b)` 个 key，因此总量为 `O(N * k * b)`。
- 滑动分支：每个 query 有 `O(w)` 个 key，因此总量为 `O(N * w)`。

总量：`O(N * (N/l + k*b + w))`。

当 `N = 64k, l = 64, k = 16, b = 64, w = 512`：每个 query 的成本是 `1000 + 1024 + 512 = 2536 keys`。完整注意力是 `64000 keys`。计算减少 25 倍。

当 `N = 128k, l = 64, k = 16, b = 64, w = 512`：每个 query 的成本是 `2000 + 1024 + 512 = 3536 keys`。完整注意力是 `128000 keys`。减少 36 倍。收益会随序列长度增长而增长，这正是重点。

### 它和其他方法怎么比

| Method | Differentiable | Real inference speedup | Long-range recall |
|--------|---------------|----------------------|-------------------|
| Sliding window only | yes | yes | fails |
| Strided / block-sparse | yes | yes | partial |
| KV pruning (H2O, StreamingLLM) | N/A (inference-time) | yes | partial |
| MoBA (Moonshot) | partial | yes | good |
| NSA | yes (natively) | yes (9x at 64k) | matches full attention |

MoBA（Moonshot, arXiv:2502.13189）几乎同期发表，也采用了类似“三条路比一条好”的思路，把 MoE 原则应用到注意力块上。NSA 和 MoBA 是理解 2026 长上下文预训练时必须知道的两个架构。

## 动手实现

`code/main.py` 会在一段短合成序列上实现三条分支，并展示：

- 压缩 MLP（为了教学清晰，用一个简单的 mean-pool baseline；真实 NSA 使用学习到的 MLP）。
- 由压缩分支分数驱动的 top-k block selection。
- 对最后 `w` 个 token 的滑动窗口注意力。
- 门控组合。
- 与完整注意力对比的计算量打印。

### Step 1: 把 token 压缩成块

```python
def compress(K, l):
    n = len(K)
    n_blocks = (n + l - 1) // l
    out = []
    for b in range(n_blocks):
        start, end = b * l, min((b + 1) * l, n)
        block = K[start:end]
        summary = [sum(row[d] for row in block) / len(block) for d in range(len(K[0]))]
        out.append(summary)
    return out
```

### Step 2: 压缩分支注意力

让 query 对压缩 key 运行 softmax attention。压缩分支的分数同时作为 top-k selection 的信号。

### Step 3: top-k 块选择

选出压缩块中得分最高的 `k` 个索引。加载这些块中的原始未压缩 token，并对它们运行注意力。

### Step 4: 滑动窗口注意力

取最后 `w` 个 token，并对它们运行标准注意力。

### Step 5: gate + combine

一个小 MLP 根据 query 生成三个门控权重。最终输出是三条分支输出的加权和。

### Step 6: 计算计数

打印每个 query 在各分支关注的 key 数量以及总数。与 `N`（完整注意力）比较。在一个 1024-token 合成样例上，使用 `l = 32, k = 4, w = 128` 时，NSA 每个 query 看到 `32 + 128 + 128 = 288` 个 key，而完整注意力是 1024——少 3.5 倍。

## 实际使用

NSA 正在 DeepSeek 自己的长上下文预训练管线中使用。截至 2026 年 4 月，公共推理栈中的集成状态如下：

- **DeepSeek internal**: 原生支持，已发布权重使用 NSA 或其后继 DSA（Deepseek Sparse Attention）。
- **vLLM**: 面向 DeepSeek-V3.x 权重的实验性 NSA 支持正在开发中。
- **SGLang**: 已发布 NSA benchmark；生产路径跟随 vLLM。
- **llama.cpp / CPU**: 不支持；在 CPU 吞吐下，kernel 拆分的开销不值得。

什么时候该用 NSA：

- 目标是 64k 以上上下文、且有严肃算力预算的预训练或继续训练。
- 推理 DeepSeek 自己的长上下文 checkpoint。权重是 NSA-native 的。

什么时候不该用：

- 服务一个现有的 dense-attention 预训练模型。没有继续训练就无法 retrofit NSA。
- 上下文低于 16k。三分支开销会压过节省。
- Batch-1 交互式聊天。延迟敏感的解码会受益，但只在长上下文下明显。

## 交付成果

本课产出 `outputs/skill-nsa-integrator.md`。给定一个长上下文预训练运行规格，它会生成 NSA 集成计划：压缩块大小、top-k、滑动窗口、gate MLP 宽度、kernel 选择，以及能够证明架构改动合理的具体长上下文 eval。

## 练习

1. 在 1024-token 合成样例上运行 `code/main.py`。在三个 preset 上 sweep `(l, k, w)` 并打印计算量。找出每个 query key-count 最低、同时在 needle-in-haystack 测试上相对完整注意力保持 95% recall 的 preset。

2. 把 mean-pool compressor 替换成一个很小的学习 MLP（2 层，hidden 32）。在一个信号是块均值的合成任务上训练它。测量 held-out 数据上相对 mean-pool baseline 的 perplexity gap。

3. 实现 gate MLP。它接收 query 作为输入并输出三个标量。展示 gate 的行为合理：随机 query 上接近均匀加权；当 query 命中很远的历史块时，对 selected branch 给出高权重。

4. 计算一个 NSA-enabled 70B 模型在 128k 上下文下的 KV cache 内存预算。KV heads 为 8，head dim 为 128，BF16。与完整注意力以及 MLA 比较（Phase 10 · 14 给出了 MLA 的数字）。找出 NSA 的 fine-grained branch KV cache 等于完整注意力的序列长度。

5. 阅读 NSA 论文（arXiv:2502.11089）第 4 节，用三句话解释为什么压缩分支的注意力分数会被复用于 top-k selection，而不是另算一个 routing score。把答案和梯度流联系起来。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Compressed branch | “粗粒度视图” | 对块平均后的 key 做注意力，每个 query 用 O(N/l) 个 key 提供全局上下文 |
| Selected branch | “Top-k 块” | 对压缩分支分数最高的 `k` 个块做细粒度注意力 |
| Sliding window | “局部上下文” | 对最后 `W` 个 token 做注意力，捕捉短程模式 |
| Native trainability | “带着稀疏性预训练” | 稀疏模式是在预训练中学到的，而不是推理时外挂的 |
| Compression block size l | “粗粒度视图的组大小” | 多少 token 被合并成一个 summary；典型值 32-64 |
| Top-k | “保留哪些块” | 读取其未压缩 token 的压缩块数量；典型值 16 |
| Sliding window W | “局部注意力半径” | 通常是 512；更短会伤害局部连贯性，更长会浪费计算 |
| Branch gate | “如何混合三条分支” | 逐位置 MLP 输出，用来加权三条分支的贡献 |
| Hardware alignment | “kernel-friendly sparsity” | 稀疏模式的选择方式能让真实 GPU kernel 达到理论加速 |
| DSA | “NSA 的后继者” | Deepseek Sparse Attention，DeepSeek 谱系中 NSA 之后的架构 |

## 延伸阅读

- [Yuan et al. — Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (arXiv:2502.11089, ACL 2025 Best Paper)](https://arxiv.org/abs/2502.11089) — 原论文
- [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — NSA 目标架构家族
- [Moonshot AI — MoBA: Mixture of Block Attention for Long-Context LLMs (arXiv:2502.13189)](https://arxiv.org/abs/2502.13189) — 同期工作，块上的 MoE-style attention
- [Beltagy et al. — Longformer: The Long-Document Transformer (arXiv:2004.05150)](https://arxiv.org/abs/2004.05150) — sliding-window 的源头
- [Xiao et al. — StreamingLLM: Efficient Streaming Language Models with Attention Sinks (arXiv:2309.17453)](https://arxiv.org/abs/2309.17453) — NSA 改进的推理时稀疏 baseline
- [Dao et al. — FlashAttention-2 (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691) — NSA kernel 在 64k 上击败的完整注意力 baseline
