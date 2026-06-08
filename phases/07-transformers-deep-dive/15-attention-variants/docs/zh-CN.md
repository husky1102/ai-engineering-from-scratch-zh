# Attention Variants — Sliding Window, Sparse, Differential

> Full attention 是一个圆。每个 token 都看见每个 token，memory 为此买单。四种变体会弯曲这个圆的形状，并收回一半成本。

**类型：** Build
**语言：** Python
**先修：** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head), Phase 7 · 12 (KV Cache / Flash Attention)
**时间：** ~60 分钟

## 要解决的问题

Full attention 在 sequence length 上需要 `O(N²)` memory 和 `O(N²)` compute。对一个 128K-context Llama 3 70B 来说，每层有 16 billion attention entries，再乘以 80 layers。Flash Attention（Lesson 12）隐藏了 `O(N²)` activation memory，但不会改变 arithmetic cost：每个 token 仍然 attend 到其他每个 token。

三类变体会改变 attention matrix 本身的 topology：

1. **Sliding window attention (SWA).** 每个 token 只 attend 到固定窗口内的 neighbors，而不是完整 prefix。Memory 和 compute 降到 `O(N · W)`，其中 `W` 是窗口大小。Gemma 2/3、Mistral 7B 的前几层、Phi-3-Long。
2. **Sparse / block attention.** 只有选定的 `(i, j)` pairs 会被打分；其余强制为零权重。Longformer、BigBird、OpenAI sparse transformer。
3. **Differential attention.** 用独立的 Q/K projections 计算两张 attention maps，再相减。它消除会把权重流到前几个 token 的 “attention sink”。Microsoft 的 DIFF Transformer（2024）。

这些会共存。一个 2026 frontier model 常常混合使用：多数层是 SWA-1024，每五层一个 global full attention，还有少数 differential heads 用来清理 retrieval。Gemma 3 的 5:1 SWA-to-global ratio 是当前 textbook default。

## 核心概念

### Sliding Window Attention (SWA)

位置 `i` 的每个 query 只 attend 到 `[i - W, i]`（causal SWA）或 `[i - W/2, i + W/2]`（bidirectional）内的位置。窗口之外的 tokens 在 score matrix 中得到 `-inf`。

```text
full causal:           sliding window (W=4):
positions 0-7          positions 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

对 `N = 8192` 和 `W = 1024`，score matrix 期望上有 1024 × 8192 个 non-zero rows：减少 8×。

**KV cache 会随 SWA 缩小。** 每层只需要保留最近 `W` 个 token 的 K 和 V。对 Gemma-3-ish config（1024 window，128K context），KV cache 下降 128×。

**质量成本。** 只使用 SWA 的 transformers 很难做 long-range retrieval。修复方式：把 SWA layers 和 full-attention layers 交错。Gemma 3 使用 5:1 SWA:global。Mistral 7B 使用 causal-SWA stack，让信息通过重叠窗口“向前流动”：每层把 effective receptive field 扩大 `W`，经过 `L` 层后，模型能回看 `L × W` 个 tokens。

### Sparse / Block Attention

提前选择一个 `N × N` sparsity pattern。三种 canonical shapes：

- **Local + strided（OpenAI sparse transformer）.** Attend 最近 `W` 个 tokens，再 attend 之前每隔 `stride` 的 token。用 `O(N · sqrt(N))` compute 捕获 local 和 long-range。
- **Longformer / BigBird.** Local window + 少量 global tokens（例如 `[CLS]`），这些 tokens attend 到所有人、也被所有人 attend + random-sparse links。在 matched quality 下实现 2× context。
- **Native Sparse Attention（DeepSeek, 2025）.** 学习哪些 `(Q, K)` blocks 重要；在 kernel level 跳过 zero blocks。兼容 FlashAttention。

Sparse attention 是一个 kernel-engineering story。数学很简单（mask score matrix）；收益来自永远不把 zero entries 加载进 SRAM。FlashAttention-3 和 2026 FlexAttention API 让 custom sparse patterns 在 PyTorch 中成为 first-class。

### Differential Attention (DIFF Transformer, 2024)

Regular attention 有一个 “attention sink” 问题：softmax 强制每一行和为 1，所以不想特别 attend 任何内容的 tokens 会把权重倒给第一个 token（或前几个）。这会偷走本该给真实内容的 capacity。

Differential attention 通过计算 **两张** attention maps 并相减来修复：

```text
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

其中 `λ` 是 learned scalar（通常 0.5-0.8）。A1 捕获真实内容权重；A2 捕获 sink。相减会抵消 sink，把权重重新分配给 relevant tokens。

Reported results（Microsoft 2024）：perplexity 降低 5-10%，在相同 trained length 下 effective context 延长 1.5-2×，needle-in-haystack retrieval 更敏锐。

### Variant Comparison

| Variant | Compute | KV cache | Quality vs full | Production use |
|---------|---------|----------|-----------------|----------------|
| Full attention | O(N²) | O(N) per layer | baseline | every model's default layer |
| SWA (window 1024) | O(N·W) | O(W) per layer | -0.1 ppl, good with global layers | Gemma 2/3, Phi-3-Long |
| Local + strided sparse | O(N·√N) | mixed | similar to SWA | OpenAI sparse transformer, Longformer |
| BigBird (local + global + random) | O(N) approx | mixed | matches full at 2× context | early long-context BERT |
| Native Sparse (DeepSeek-V3.2) | O(N · active fraction) | O(N) | within 0.05 ppl | DeepSeek-V3.2, 2025 |
| Differential | O(2·N²) | O(2N) | -5 to -10% ppl | DIFF Transformer, early 2026 models |

## 动手实现

见 `code/main.py`。我们实现一个 causal mask comparator，在 toy sequence 上并排展示 full、SWA、local+strided 和 differential attention。

### Step 1: full causal mask (baseline)

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

来自 Lesson 07 的 baseline。下三角；对角线上方为 zero weight。

### Step 2: sliding window causal mask

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

一个参数：`window`。当 `window >= n` 时，恢复 full causal attention。当 `window = 1` 时，每个 token 只 attend 自己。

### Step 3: local + strided sparse mask

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

Dense local window 加上从 sequence 起点开始每隔 `stride` 个 token。额外层数会让 receptive field 以 log steps 增长。

### Step 4: differential attention

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

两次 attention pass，用 learned mixing coefficient 相减。代码中我们比较 single vs differential 的 attention-sink heatmap，并观察 sink 如何塌陷。

### Step 5: KV cache sizes

在 `N = 131072` 下打印每种 variant 的 per-layer cache size。SWA 和 sparse variants 下降 10-100×。Differential 翻倍。要有意识地支付你的 memory bill。

## 实际使用

2026 production patterns：

```python
from transformers import AutoModelForCausalLM
# Gemma 3 mixes SWA (window=1024) and global layers at 5:1.
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

PyTorch 2.5+ 中的 FlexAttention 接受一个 mask function：

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

这会编译成 custom Triton kernel。对常见 patterns，速度在 FlashAttention-3 的 10% 以内，并且 mask function 是 Python callable。

**什么时候选哪一个：**

- **Pure full attention**：每层都在约 16K context 以内，或 retrieval quality 最重要。
- **SWA + global mix**：long context（>32K）、training 和 inference 都 memory-bound。2026 年 32K 以上的默认选择。
- **Sparse block attention**：custom kernel、custom pattern。留给 specialized workloads（retrieval, audio）。
- **Differential attention**：任何 attention-sink contamination 会造成伤害的 workload（long-context RAG, needle-in-haystack）。

## 交付成果

见 `outputs/skill-attention-variant-picker.md`。这个 skill 会根据 target context length、retrieval demands 和 training/inference compute profile，为新模型选择 attention topology。

## 练习

1. **Easy.** 运行 `code/main.py`。验证 `window=4` 的 SWA 会把每行最近 4 个 tokens 之外的所有位置置零。验证 `window=n` 会 bit-identically 重现 full causal attention。
2. **Medium.** 在 Lesson 07 capstone 之上实现 `window=1024` 的 causal SWA。在 tinyshakespeare 上训练 1,000 steps。相对 full attention，val loss 退化多少？Peak memory 下降多少？
3. **Hard.** 在 capstone model 中实现 Gemma-3-style 5:1 layer mix（5 SWA，1 global）。在 matched parameters 下，把 loss、memory 和 generation quality 与 pure-SWA、pure-global baselines 比较。
4. **Hard.** 用每 head 一个 learned `λ` 实现 differential attention。在 synthetic retrieval task（one needle，2,000 distractors）上训练。与 matched parameters 下的 single-attention baseline 比较 retrieval accuracy。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Sliding window attention (SWA) | “Local attention” | 每个 query attend 自己最近的 `W` 个 tokens；KV cache 缩小到 `O(W)`。 |
| Effective receptive field | “模型能看到多远” | 在一个 `L`-layer SWA stack、window `W` 中，最多能看到 `L × W` 个 tokens。 |
| Longformer / BigBird | “Local + global + random” | 带少量 always-attending global tokens 的 sparse patterns；早期 long-context 方法。 |
| Native Sparse Attention | “DeepSeek 的 kernel trick” | 学习 block-level sparsity；在 kernel level 跳过 zero blocks，同时保持质量。 |
| Differential attention | “两张 map，一张相减” | DIFF Transformer：从第一张 attention map 中减去 learned `λ` 倍的第二张 map，以抵消 attention sinks。 |
| Attention sink | “权重流向 token 0” | Softmax normalization 强制每行和为 1；uninformative queries 会把权重倒给 position 0。 |
| FlexAttention | “Mask-as-Python” | PyTorch 2.5+ API，可把 arbitrary mask functions 编译成 FlashAttention-shape kernels。 |
| Layer type mix | “5:1 SWA-to-global” | 在 stack 中交错 sparse 和 full attention layers，以更低 memory 保持质量。 |

## 延伸阅读

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — canonical sliding-window + global-token paper。
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — local + global + random。
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — OpenAI 的 local+strided pattern。
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) — 1:1 SWA:global mix。
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) — 5:1 mix with window=1024，现在的 textbook default。
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) — DIFF Transformer paper。
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) — DeepSeek-V3.2 的 learned-sparsity attention。
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) — Use It 中 mask-as-callable pattern 的 API reference。
