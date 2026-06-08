# Attention Mechanism——突破点

> decoder 不再盯着一个压缩 summary 眯眼看，而是开始查看整个 source。此后的一切都是 attention 加 engineering。

**类型:** Build
**语言:** Python
**先修:** Phase 5 · 09 (Sequence-to-Sequence Models)
**时间:** ~45 minutes

## 要解决的问题

Lesson 09 以一次可测量 failure 结束。一个在 toy copy task 上训练的 GRU encoder-decoder，从 length 5 时 89% accuracy 掉到 length 80 时接近 chance。原因是结构性的，不是 training bug：encoder 捕获到的每一 bit 信息都必须塞进一个 fixed-size hidden state，而 decoder 看不到别的东西。

Bahdanau、Cho 和 Bengio 在 2014 年发表了三行修复。不要只把 final encoder state 给 decoder，而是保留每个 encoder state。在每个 decoder step，计算 encoder states 的 weighted average，权重表示“decoder 此刻需要看 encoder position `i` 多少？”这个 weighted average 就是 context，而且每个 decoder step 都会变化。

这就是完整 idea。Transformers 扩展了它。Self-attention 将它应用到单个 sequence。Multi-head attention 并行运行它。但 2014 版本已经打破 bottleneck；一旦你有了它，转向 transformers 是 engineering，不是概念问题。

## 核心概念

![Bahdanau attention: decoder queries all encoder states](../assets/attention.svg)

在每个 decoder step `t`：

1. 使用 previous decoder hidden state `s_{t-1}` 作为 **query**。
2. 将它与每个 encoder hidden state `h_1, ..., h_T` 打分。每个 encoder position 一个 scalar。
3. 对 scores 做 softmax，得到加和为 1 的 attention weights `α_{t,1}, ..., α_{t,T}`。
4. Context vector `c_t = Σ α_{t,i} * h_i`。encoder states 的 weighted average。
5. Decoder 接收 `c_t` 加 previous output token，产生 next token。

weighted average 是重点。当 decoder 需要将 “Je” 翻译为 “I” 时，它会给 “Je” 对应的 encoder state 高权重，其他位置低权重。当它需要 “not” 时，它会给 “pas” 高权重。context vector 每一步都会重塑。

## Shapes（最容易咬人的地方）

每个人第一次实现 attention 都会在这里出错。慢慢读。

| Thing | Shape | Notes |
|-------|-------|-------|
| Encoder hidden states `H` | `(T_enc, d_h)` | 如果是 BiLSTM，`d_h = 2 * d_hidden` |
| Decoder hidden state `s_{t-1}` | `(d_s,)` | One vector |
| Attention score `e_{t,i}` | scalar | 每个 encoder position 一个 |
| Attention weight `α_{t,i}` | scalar | 对所有 `i` softmax 后 |
| Context vector `c_t` | `(d_h,)` | 与 encoder state shape 相同 |

**Bahdanau (additive) score.** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`。

- `s_{t-1}` 的 shape 是 `(d_s,)`，`h_i` 的 shape 是 `(d_h,)`。
- `W_a` 的 shape 是 `(d_attn, d_s)`。`U_a` 的 shape 是 `(d_attn, d_h)`。
- 它们在 tanh 内部相加后的 shape 是 `(d_attn,)`。
- `v_α` 的 shape 是 `(d_attn,)`。与 `v_α` 的 inner product 会 collapse 为 scalar。**这就是 `v_α` 的作用。**它不是魔法。它是将 attention-dim vector 变成 scalar score 的 projection。

**Luong (multiplicative) score.** 三个 variants：

- `dot`: `e_{t,i} = s_t^T * h_i`。要求 `d_s == d_h`。硬约束。如果 encoder 是 bidirectional，跳过它。
- `general`: `e_{t,i} = s_t^T * W * h_i`，其中 `W` shape 为 `(d_s, d_h)`。移除了 equal-dim constraint。
- `concat`: 本质上是 Bahdanau form。由于前两者更便宜，现在很少用。

**一个值得命名的 Bahdanau / Luong gotcha。** Bahdanau 使用 `s_{t-1}`（生成当前词*之前*的 decoder state）。Luong 使用 `s_t`（生成当前词*之后*的 state）。混用会产生微妙错误 gradients，极难调试。选一篇论文，并坚持它的 convention。

## 动手实现

### Step 1: additive (Bahdanau) attention

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

对照上面的表检查 shapes。`encoder_states` shape 为 `(T_enc, d_h)`。`projected_enc` shape 为 `(T_enc, d_attn)`。`projected_dec` shape 为 `(d_attn,)` 并 broadcast。`combined` shape 为 `(T_enc, d_attn)`。`scores` shape 为 `(T_enc,)`。`weights` shape 为 `(T_enc,)`。`context` shape 为 `(d_h,)`。Ship it。

### Step 2: Luong dot and general

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

每个三行。这就是 Luong paper 能落地的原因。多数 tasks 上 accuracy 相同，代码少很多。

### Step 3: 数值例子

给定三个 encoder states（大致是 “cat”、“sat”、“mat”）和一个最接近第一个的 decoder state，attention distribution 会集中在 position 0。如果 decoder state 转向与最后一个 encoder state 对齐，attention 会移动到 position 2。context vector 会跟着移动。

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```text
weights: [0.464 0.305 0.231]
```

第一行胜出。然后将 decoder state 移近第三个 encoder state，看 weights 如何移动。这就是 attention。它是显式 alignment。

### Step 4: 为什么它是通向 transformers 的桥

将上面的语言翻译成 Q/K/V：

- **Query** = decoder state `s_{t-1}`
- **Key** = encoder states（我们用来打分的对象）
- **Value** = encoder states（我们加权求和的对象）

在 classical attention 中，keys 和 values 是同一个东西。Self-attention 将它们分开：你可以让一个 sequence query 它自身，并为 K 和 V 使用不同 learned projections。Multi-head attention 用不同 learned projections 并行运行它。Transformers 多次 stack 整个 stage，并丢掉 RNNs。

数学是一样的。shapes 是一样的。从 Bahdanau attention 到 scaled dot-product attention 的教学跳跃，主要只是 notation。

## 实际使用

PyTorch 和 TensorFlow 直接提供 attention。

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```text
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

这就是 transformer attention layer。query batch 有 5 positions，key/value batch 有 10 positions，每个 128-dim，8 heads。`output` 是新的 context-augmented queries。`weights` 是可视化的 5x10 alignment matrix。

### 什么时候 classical attention 仍重要

- 教学。single-head、single-layer、RNN-based 版本让每个概念都可见。
- transformers 放不下的 on-device sequence tasks。
- 2014-2017 年的任何论文。不知道 Bahdanau convention 会误读。
- MT 中的 fine-grained alignment analysis。Raw attention weights 即使在 transformer models 上也是 interpretability tool，而读懂它们需要知道它们是什么。

### attention-weight-as-explanation trap

Attention weights 看起来可解释。它们是跨 positions 求和为一的 weights；你能画出来；高值似乎表示“看了这里”。reviewers 喜欢它们。

它们没有看起来那么可解释。Jain and Wallace (2019) 表明，在一些 tasks 上，attention distributions 可以被 permute，甚至被 arbitrary alternatives 替换，而不改变 model predictions。没有 ablation 或 counterfactual check，永远不要把 attention weights 报告为 reasoning evidence。

## 交付成果

保存为 `outputs/prompt-attention-shapes.md`：

```markdown
---
name: attention-shapes
description: Debug shape bugs in attention implementations.
phase: 5
lesson: 10
---

Given a broken attention implementation, you identify the shape mismatch. Output:

1. Which matrix has the wrong shape. Name the tensor.
2. What its shape should be, derived from (d_s, d_h, d_attn, T_enc, T_dec, batch_size).
3. One-line fix. Transpose, reshape, or project.
4. A test to catch regressions. Typically: assert `output.shape == (batch, T_dec, d_h)` and `weights.shape == (batch, T_dec, T_enc)` and `weights.sum(dim=-1) close to 1`.

Refuse to recommend fixes that silently broadcast. Broadcast-hiding bugs surface later as silent accuracy degradation, the worst kind of attention bug.

For Bahdanau confusion, insist the decoder input is `s_{t-1}` (pre-step state). For Luong, `s_t` (post-step state). For dot-product, flag dimension mismatch between query and key as the most common first-time error.
```

## 练习

1. **Easy.** 实现 `softmax` masking，让 encoder 中的 padding tokens 得到 attention weight zero。在 variable-length sequences 的 batch 上测试。
2. **Medium.** 给 Luong `general` form 添加 multi-head attention。将 `d_h` 拆成 `n_heads` 组，每个 head 运行 attention，然后 concatenate。验证 single-head case 与之前实现一致。
3. **Hard.** 在 lesson 09 的 toy copy task 上训练带 Bahdanau attention 的 GRU encoder-decoder。绘制 accuracy vs sequence length。与 no-attention baseline 对比。你应该看到长度增加时差距扩大，从而确认 attention 解除了 bottleneck。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Attention | Looking at things | value sequence 的 weighted average，weights 来自 query-key similarity。 |
| Query, Key, Value | QKV | 三个 projections：Q 发问，K 是匹配对象，V 是返回内容。 |
| Additive attention | Bahdanau | Feed-forward score：`v^T tanh(W q + U k)`。 |
| Multiplicative attention | Luong dot / general | Score 是 `q^T k` 或 `q^T W k`。更便宜，多数 tasks 上 accuracy 相同。 |
| Alignment matrix | The pretty picture | 作为 `(T_dec, T_enc)` grid 的 attention weights。用它看模型关注了什么。 |

## 延伸阅读

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)——原论文。
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)——三种 score variants 及其比较。
- [Jain and Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186)——interpretability caveat。
- [Dive into Deep Learning — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html)——带 PyTorch 的 runnable walkthrough。
