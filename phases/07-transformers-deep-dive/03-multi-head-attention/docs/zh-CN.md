# Multi-Head Attention

> 一个 attention head 一次学习一种关系。八个 heads 学八种。Heads 近乎免费。多用一些。

**类型：** Build
**语言：** Python
**先修：** Phase 7 · 02 (Self-Attention from Scratch)
**时间：** ~75 分钟

## 要解决的问题

单个 self-attention head 计算一个 attention matrix。这个矩阵捕捉一种关系，通常是最小化当前训练信号 loss 的那一种。如果你的数据里 subject-verb agreement、co-reference、long-range discourse 和 syntactic chunking 全都纠缠在一起，单个 head 会把它们糊进一个 soft-max distribution，丢掉一半信号。

2017 年 Vaswani 论文给出的修复方法：并行运行多个 attention functions，每个都有自己的 Q、K、V projections，然后拼接输出。每个 head 在维度为 `d_model / n_heads` 的更小子空间中运行。总参数量保持相同。表达能力上升。

Multi-head attention 是 2026 年每个 transformer 的默认配置。唯一争论是 *多少个* heads，以及 keys 和 values 是否共享 projections（Grouped-Query Attention、Multi-Query Attention、Multi-head Latent Attention）。

## 核心概念

![Multi-head attention splits, attends, concatenates](../assets/multi-head-attention.svg)

**Split。** 取形状为 `(N, d_model)` 的 `X`。投影到 Q、K、V，每个形状都是 `(N, d_model)`。reshape 为 `(N, n_heads, d_head)`，其中 `d_head = d_model / n_heads`。transpose 为 `(n_heads, N, d_head)`。

**并行 Attend。** 在每个 head 内运行 scaled dot-product attention。每个 head 产生 `(N, d_head)`。这些 heads 在 embedding 的不同子空间中运行，并且在 attention computation 自身期间互不通信。

**Concatenate and project。** 把 heads 堆回 `(N, d_model)`，再乘上形状为 `(d_model, d_model)` 的 learned output matrix `W_o`。`W_o` 是 heads 得以混合的地方。

**为什么有效。** 每个 head 可以专门化，而无需和其他 head 竞争 representational budget。2019-2024 年的 probing studies 展示了不同 head roles：positional heads、attends to the previous token 的 head、copy heads、named-entity heads、induction heads（支撑 in-context learning）。

**2026 年变体谱系：**

| Variant | Q heads | K/V heads | Used by |
|---------|---------|-----------|---------|
| Multi-head (MHA) | N | N | GPT-2, BERT, T5 |
| Multi-query (MQA) | N | 1 | PaLM, Falcon |
| Grouped-query (GQA) | N | G (e.g. N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| Multi-head latent (MLA) | N | compressed to low-rank | DeepSeek-V2, V3 |

GQA 是现代默认选择，因为它能把 KV-cache memory 降低 `N/G` 倍，同时保持几乎完整质量。MLA 进一步把 K/V 压缩进 latent space，然后在 compute time 投影回来，代价是 FLOPs，但节省更多 memory。

## 动手实现

### Step 1：从已有 single-head attention 切分 heads

取 Lesson 02 的 `SelfAttention`，用 split/concat 对包起来。`code/main.py` 中有 numpy implementation；逻辑如下：

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

一次 reshape，一次 transpose。没有循环。这正是 PyTorch 在 `nn.MultiheadAttention` 下面做的事。

### Step 2：按 head 运行 scaled-dot-product attention

每个 head 拿到自己的 Q、K、V slice。Attention 变成 batched matmul：

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

在真实硬件上，`Qh @ Kh.transpose(...)` 是一个 `bmm`。GPU 看到的是形状 `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)` 的单个 batched matmul。增加 heads 近乎免费。

### Step 3：Grouped-Query Attention 变体

只有 key 和 value projections 改变。Q 得到 `n_heads` 个 groups；K 和 V 得到 `n_kv_heads < n_heads` 个 groups，并重复以匹配：

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

推理时这会节省内存，因为 KV cache 中只需要保留 `n_kv_heads` 份，而不是 `n_heads` 份。Llama 3 70B 使用 64 个 query heads 和 8 个 KV heads，KV cache 缩小 8×。

### Step 4：probe 每个 head 学到了什么

在一个短句上用 4 heads 运行 MHA。对每个 head，打印 `(N, N)` attention matrix。即使随机初始化，你也会看到不同 heads 选出不同结构：部分是信号，部分是子空间中的 rotational symmetry。

## 实际使用

在 PyTorch 中，一行版本：

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

PyTorch 2.5+ 中的 GQA：

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K,V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**多少 heads？** 来自 2026 生产模型的经验法则：

| Model size | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| Small (~125M) | 768 | 12 | 64 |
| Base (~350M) | 1024 | 16 | 64 |
| Large (~1B) | 2048 | 16 | 128 |
| Frontier (~70B) | 8192 | 64 | 128 |

`d_head` 几乎总是落在 64 或 128。它是一个 head 能“看到”多少信息的单位。低于 32，heads 开始和 scaling factor `sqrt(d_head)` 打架；高于 256，你会失去“许多小专家”的收益。

## 交付成果

见 `outputs/skill-mha-configurator.md`。这个 skill 会根据 parameter budget、sequence length 和 deployment target，为新 transformer 推荐 head count、kv-head count 和 projection strategy。

## 练习

1. **Easy。** 取 `code/main.py` 中的 MHA，在固定 `d_model=64` 时把 `n_heads` 从 1 改到 16。在 synthetic copy task 上绘制一个 tiny one-layer model 的 loss。更多 heads 是有帮助、平台化，还是有害？
2. **Medium。** 实现 MQA（所有 query heads 共享一个 KV head）。测量相较 full MHA，parameter count 下降多少。计算 N=2048 推理时 KV-cache size 缩小多少。
3. **Hard。** 实现一个 tiny 版 Multi-head Latent Attention：把 K,V 压缩到 rank-`r` latent，把 latent 存在 KV cache 中，在 attention time 解压。在什么 `r` 下，cache memory 低于 full MHA 的 1/8，同时 quality 保持在 validation ppl 的 1 bit 内？

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Head | “A single attention circuit” | 一个维度为 `d_head = d_model / n_heads` 的 Q/K/V projection，带自己的 attention matrix。 |
| d_head | “Head dimension” | 每个 head 的 hidden width；生产中几乎总是 64 或 128。 |
| Split / combine | “Reshape tricks” | attention 前后的 `(N, d_model) ↔ (n_heads, N, d_head)` reshape+transpose。 |
| W_o | “Output projection” | 拼接 heads 后应用的 `(d_model, d_model)` matrix；heads 在这里混合。 |
| MQA | “One KV head” | Multi-Query Attention：单个共享 K/V projection。KV cache 最小，有一些质量损失。 |
| GQA | “The default since Llama 2” | `n_kv_heads < n_heads` 的 Grouped-Query Attention；重复以匹配 Q。 |
| MLA | “DeepSeek's trick” | Multi-head Latent Attention：K,V 被压缩为 low-rank latent，并在 attend time 解压。 |
| Induction head | “The circuit behind in-context learning” | 一对 heads，检测之前出现过的内容并复制其后续内容。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — 原始 multi-head spec。
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — MQA 论文。
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — 训练后如何把 MHA 转为 GQA。
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) — MLA 以及为什么它在 cache memory 上胜过 MHA/GQA。
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — 对 heads 实际作用的 mechanistic look。
