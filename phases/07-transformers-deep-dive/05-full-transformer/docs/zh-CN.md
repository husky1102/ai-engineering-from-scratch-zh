# 完整 Transformer — Encoder + Decoder

> Attention 是主角。其他一切——residual、normalization、feed-forward、cross-attention——都是让你能把它堆深的脚手架。

**类型:** Build
**语言:** Python
**先修:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention), Phase 7 · 04 (Positional Encoding)
**时间:** ~75 minutes

## 要解决的问题

单个 attention layer 是特征提取器，不是完整模型。每层一次 matmul 对语言来说容量不够。你需要深度，而没有合适的管线，深度会崩。

2017 年 Vaswani 论文打包了六个设计决策，把一个 attention layer 变成了可堆叠的 block。之后的每个 transformer——encoder-only (BERT)、decoder-only (GPT)、encoder-decoder (T5)——都继承了同一个骨架。到 2026 年，block 已经被精炼过（RMSNorm、SwiGLU、pre-norm、RoPE），但骨架完全相同。

本课讲这个骨架。后续课程会把它专门化：06 讲 encoder，07 讲 decoder，08 讲 encoder-decoder。

## 核心概念

![Encoder and decoder block internals, wired](../assets/full-transformer.svg)

### 六个组件

1. **Embedding + positional signal.** Tokens → vectors。位置通过 RoPE（现代）或 sinusoidal（经典）注入。
2. **Self-attention.** 每个位置关注所有其他位置。Decoder 中会被 mask。
3. **Feed-forward network (FFN).** 逐位置两层 MLP：`W_2 · activation(W_1 · x)`。默认 expansion ratio 为 4×。
4. **Residual connection.** `x + sublayer(x)`。没有它，超过约 6 层后 gradients 会消失。
5. **Layer normalization.** `LayerNorm` 或 `RMSNorm`（现代）。稳定 residual stream。
6. **Cross-attention (decoder only).** Queries 来自 decoder，keys 和 values 来自 encoder output。

观察一个向量流过一个 block：attention 跨位置混合，residual 把它向前携带，FFN 转换它，norm 让 stream 保持稳定。

```figure
transformer-block
```

### Encoder block（BERT、T5 encoder 使用）

```text
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

Encoder 是 bidirectional。没有 masking。所有位置都能看到所有位置。

### Decoder block（GPT、T5 decoder 使用）

```text
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

Decoder 每个 block 有三个 sublayers。中间的 cross-attention 是信息从 encoder 流向 decoder 的唯一位置。在纯 decoder-only 架构（GPT）中，cross-attention 被省略，只剩 masked self-attention + FFN。

### Pre-norm vs post-norm

原论文：`x + sublayer(LN(x))` vs `LN(x + sublayer(x))`。Post-norm 在 2019 年前后失宠，因为没有仔细 warmup 时它更难深度训练。Pre-norm（在 sublayer *之前* 做 `LN`）是 2026 年默认：Llama、Qwen、GPT-3+、Mistral 都使用它。

### 2026 年现代化 block

Vaswani 2017 使用 LayerNorm + ReLU。现代 stack 替换了两者。生产 block 实际长这样：

| Component | 2017 | 2026 |
|-----------|------|------|
| Normalization | LayerNorm | RMSNorm |
| FFN activation | ReLU | SwiGLU |
| FFN expansion | 4× | 2.6× (SwiGLU uses three matrices, total params match) |
| Position | Sinusoidal absolute | RoPE |
| Attention | Full MHA | GQA (or MLA) |
| Bias terms | Yes | No |

RMSNorm 去掉了 LayerNorm 的 mean-centering（少一次 subtraction），节省计算，并且经验上至少同样稳定。SwiGLU（`Swish(W1 x) ⊙ W3 x`）在 Llama、PaLM、Qwen 论文中都比 ReLU/GELU FFN 稳定高出约 0.5 point ppl。

### Parameter count

对于一个 `d_model = d`、FFN expansion `r` 的 block：

- MHA: `4 · d²`（Q、K、V、O projections）
- FFN (SwiGLU): `3 · d · (r · d)` ≈ `3rd²`
- Norms: 可忽略

在 `d = 4096, r = 2.6, layers = 32`（大致是 Llama 3 8B）时，总量为：`32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B parameters per layer × 32 ≈ 7B`（再加 embeddings 和 head）。与公开参数量相符。

## 动手实现

### Step 1: the building blocks

使用 Lesson 03 的 tiny `Matrix` class（复制到本文件以保持独立）：

- `layer_norm(x, eps=1e-5)` — 减去 mean，除以 std。
- `rms_norm(x, eps=1e-6)` — 除以 RMS。不减 mean。
- `gelu(x)` 和 `silu(x) * W3 x` (SwiGLU)。
- `ffn_swiglu(x, W1, W2, W3)`。
- `encoder_block(x, params)` 和 `decoder_block(x, enc_out, params)`。

完整 wiring 见 `code/main.py`。

### Step 2: wire a 2-layer encoder and a 2-layer decoder

把它们堆起来。把 encoder output 传入每个 decoder cross-attention。在 output projection 之前加一个最终 LN。

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### Step 3: run forward on a toy example

把 6-token source 和 5-token target 送进去。验证 output shape 是 `(5, vocab)`。不训练——本课关注架构，不关注 loss。

### Step 4: swap in RMSNorm + SwiGLU

把 LayerNorm 和 ReLU-FFN 替换为 RMSNorm 和 SwiGLU。确认 shapes 仍然匹配。这就是用一次函数替换完成的 2026 现代化。

## 实际使用

PyTorch/TF 参考实现是：`nn.TransformerEncoderLayer`、`nn.TransformerDecoderLayer`。但大多数 2026 生产代码会自己写 block，因为：

- Flash Attention 在 attention 内部调用，不通过 `nn.MultiheadAttention`。
- GQA / MLA 不在 stdlib reference 里。
- RoPE、RMSNorm、SwiGLU 不是 PyTorch 默认项。

HF `transformers` 有值得阅读的干净参考 block：`modeling_llama.py` 是 2026 canonical decoder-only block。它约 500 行，值得完整走读一次。

**Encoder vs decoder vs encoder-decoder — 什么时候选：**

| Need | Pick | Example |
|------|------|---------|
| Classification, embeddings, QA over text | Encoder-only | BERT, DeBERTa, ModernBERT |
| Text generation, chat, code, reasoning | Decoder-only | GPT, Llama, Claude, Qwen |
| Structured input → structured output (translation, summarization) | Encoder-decoder | T5, BART, Whisper |

Decoder-only 赢下语言任务，是因为它最容易扩展，并且同时处理 comprehension 和 generation。当输入有清晰的“source sequence”身份（translation、speech recognition、structured tasks）时，encoder-decoder 仍然最好。

## 交付成果

见 `outputs/skill-transformer-block-reviewer.md`。这个 skill 会按 2026 默认项审查新的 transformer block 实现，并标出缺失部分（pre-norm、RoPE、RMSNorm、GQA、FFN expansion ratio）。

## 练习

1. **Easy.** 在 `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True` 下统计你的 encoder_block 参数量。实现 block，并用 `sum(p.numel() for p in block.parameters())` 验证。
2. **Medium.** 从 post-norm 切到 pre-norm。初始化二者，并在 random input 上测量堆叠 12 层后的 activation norm。Post-norm 的 activations 应该爆炸；pre-norm 应保持有界。
3. **Hard.** 在 toy copy task（复制反转后的 `x`）上实现 4-layer encoder-decoder。训练 100 steps。报告 loss。换成 RMSNorm + SwiGLU + RoPE 后，loss 会下降吗？

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Block | “一个 transformer layer” | norm + attention + norm + FFN 的堆叠，并包在 residual connections 中。 |
| Residual | “Skip connection” | `x + f(x)` output；让 gradients 能穿过深层 stack。 |
| Pre-norm | “先 normalize，不是后 normalize” | 现代形式：`x + sublayer(LN(x))`。无需 warmup 魔法也能训得更深。 |
| RMSNorm | “没有 mean 的 LayerNorm” | 除以 RMS；少一个操作，经验稳定性相同。 |
| SwiGLU | “大家都换用的 FFN” | `Swish(W1 x) ⊙ W3 x → W2`。在 LM ppl 上优于 ReLU/GELU。 |
| Cross-attention | “decoder 看到 encoder 的方式” | Q 来自 decoder、K/V 来自 encoder outputs 的 MHA。 |
| FFN expansion | “中间 MLP 有多宽” | hidden-size 与 d_model 的比值，通常为 4（LayerNorm）或 2.6（SwiGLU）。 |
| Bias-free | “去掉 +b 项” | 现代 stack 在线性层中省略 biases；ppl 略有改善，模型更小。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原始 block spec。
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — 为什么 pre-norm 在深层训练中胜过 post-norm。
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm。
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — SwiGLU 论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — canonical 2026 decoder-only block。
