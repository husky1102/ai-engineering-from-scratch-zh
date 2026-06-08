# 位置编码 — Sinusoidal、RoPE、ALiBi

> Attention 对排列不敏感。“The cat sat on the mat”和“mat the on sat cat the”如果没有位置信号，会产生同样的输出。三种算法修复了这一点，而且它们各自押注于“位置”到底意味着什么。

**类型:** Build
**语言:** Python
**先修:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention)
**时间:** ~45 minutes

## 要解决的问题

Scaled dot-product attention 不知道顺序。注意力矩阵 `softmax(Q K^T / √d) V` 只由成对相似度计算得到。打乱 `X` 的行，输出的行也会以同样方式被打乱。Attention 内部没有任何东西关心位置。

这对 bag-of-words 模型不是 bug。可对语言、代码、音频、视频，或者任何顺序承载意义的东西来说，这是致命的。

修复方法是以某种方式把位置注入 embedding。三个时代的答案：

1. **Absolute sinusoidal** (Vaswani 2017)。把位置的 `sin/cos` 加到 embedding 上。简单、不需要学习参数，但在训练长度之外外推很差。
2. **RoPE — Rotary Position Embeddings** (Su 2021)。按与位置成比例的角度旋转 Q 和 K 向量。在 dot product 中直接编码*相对*位置。2026 年的主流选择。
3. **ALiBi — Attention with Linear Biases** (Press 2022)。完全跳过 embedding 技巧；根据距离给 attention score 加一个逐 head 的线性惩罚。长度外推非常好。

截至 2026 年，几乎所有 frontier open model 都使用 RoPE：Llama 2/3/4、Qwen 2/3、Mistral、Mixtral、DeepSeek-V3、Kimi。少数 long-context 模型使用 ALiBi 或其现代变体。Absolute sinusoidal 已经主要是历史方案。

## 核心概念

![Sinusoidal absolute vs RoPE rotations vs ALiBi distance bias](../assets/positional-encoding.svg)

### Absolute sinusoidal

预先计算一个形状为 `(max_len, d_model)` 的固定矩阵 `PE`：

```text
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

然后在 attention 之前执行 `X' = X + PE[:N]`。每个维度都是不同频率的 sinusoid。模型学习从相位模式里读出位置。它在 `max_len` 之外会失败：如果模型只见过位置 0–2047，就没有任何信号告诉它位置 2048 会发生什么。

### RoPE

旋转 Q 和 K 向量，而不是 embedding。对于一对维度 `(2i, 2i+1)`：

```text
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 by default
```

用位置 `pos_k` 对 keys 应用同样的旋转。dot product `q'_m · k'_n` 会变成只依赖 `(m - n)` 的函数。也就是说：**attention score 只依赖相对距离**，即使旋转本身是由绝对位置驱动的。漂亮的小技巧。

扩展 RoPE：可以缩放 `base`（NTK-aware、YaRN、LongRoPE），在不重新训练的情况下外推到更长 context。Llama 3 就用这种方式从 8K 扩到 128K context。

### ALiBi

跳过 embedding 技巧，直接偏置 attention scores：

```text
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

其中 `m_h` 是 head-specific slope（例如 `1 / 2^(8·h/H)`）。近的 token 被增强，远的 token 被惩罚。没有训练时成本。论文显示，长度外推优于 sinusoidal，并在原训练长度上匹配 RoPE。

### 2026 年该选什么

| Variant | Extrapolation | Training cost | Used by |
|---------|---------------|---------------|---------|
| Absolute sinusoidal | poor | free | original transformer, early BERT |
| Learned absolute | none | tiny | GPT-2, GPT-3 |
| RoPE | good with scaling | free | Llama 2/3/4, Qwen 2/3, Mistral, DeepSeek-V3, Kimi |
| RoPE + YaRN | excellent | fine-tune stage | Qwen2-1M, Llama 3.1 128K |
| ALiBi | excellent | free | BLOOM, MPT, Baichuan |

RoPE 胜出，是因为它能直接嵌进 attention 而不改变架构，编码相对位置，并且 `base` 这个超参数为 long-context fine-tuning 提供了干净的调节旋钮。

## 动手实现

### Step 1: sinusoidal encoding

见 `code/main.py`。4 行计算：

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

在第一层 attention 之前把它加到 embedding matrix 上。

### Step 2: RoPE applied to Q, K

RoPE 就地作用在 Q 和 K 上。对每一对维度：

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

关键点：对位置 `m` 的 Q 和位置 `n` 的 K 应用同一个函数。它们的 dot product 会在每个坐标对上获得一个 `cos((m-n)·θ_i)` 因子。Attention 免费学到相对位置。

### Step 3: ALiBi slopes and bias

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # add to attention scores before softmax
```

把 `bias[h]` 加到 head `h` 的 `(seq_len, seq_len)` attention score matrix 上，然后 softmax。

### Step 4: verify relative-distance property of RoPE

选两个随机向量 `a, b`。先按 `(pos_a, pos_b)` 旋转，再按 `(pos_a + k, pos_b + k)` 旋转。两个 dot product 必须在浮点误差范围内一致。这个性质就是 RoPE 的核心：它对绝对偏移不变，只关心相对间隔。

## 实际使用

PyTorch 2.5+ 在 `torch.nn.functional` 里提供 RoPE 工具。大多数生产代码使用 `flash_attn` 或 `xformers`，RoPE 会在 attention kernel 内部应用。

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**2026 年的 long-context 技巧：**

- **NTK-aware interpolation.** 从 4K 扩到 16K+ 时，把 `base` 重缩放为 `base * (scale_factor)^(d/(d-2))`。
- **YaRN.** 更聪明的 interpolation，可在长 context 上保留 attention entropy。Llama 3.1 128K 使用它。
- **LongRoPE.** Microsoft 2024 年方法，用 evolutionary search 选择逐维 scale factors。Phi-3-Long 使用它。
- **Position interpolation + fine-tuning.** 直接按扩展因子压缩位置，然后 fine-tune 1–5B tokens。效果意外地好。

## 交付成果

见 `outputs/skill-positional-encoding-picker.md`。这个 skill 会根据目标 context length、extrapolation 需求和训练预算，为新模型选择编码策略。

## 练习

1. **Easy.** 把 `max_len=512, d=128` 的 sinusoidal `PE` matrix 画成 heatmap。确认“维度索引越大，条纹越宽”的模式。
2. **Medium.** 实现 NTK-aware RoPE scaling。在长度 256 的序列上训练一个 tiny LM，然后在长度 1024 上分别测试有无 scaling。测量 perplexity。
3. **Hard.** 在同一个 attention module 中实现 ALiBi 和 RoPE。在长度 512 的 copy task 上训练一个 4-layer transformer。测试时外推到 2048。比较退化幅度。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Positional encoding | “告诉 attention 顺序” | 添加到 embeddings 或 attention 中、用于编码位置的任意信号。 |
| Sinusoidal | “最初那个” | 以几何频率变化的 `sin/cos`，加到 embeddings 上；不能外推。 |
| RoPE | “Rotary embeddings” | 按位置相关角度旋转 Q、K；dot product 编码相对距离。 |
| ALiBi | “Linear bias trick” | 给 attention scores 加 `-m·\|i-j\|`；不需要 embedding，外推很好。 |
| base | “RoPE 的旋钮” | RoPE 中的频率缩放器；增大它可在推理时扩展 context。 |
| NTK-aware | “一种 RoPE scaling trick” | 重缩放 `base`，使 context 扩展时高频维度不被挤压。 |
| YaRN | “更高级的那个” | 保持 attention entropy 的逐维 interpolation+extrapolation。 |
| Extrapolation | “超过训练长度也能工作” | 位置方案能否在训练时见过的 `max_len` 之外继续给出正确输出？ |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — original sinusoidal。
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE 论文。
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — ALiBi。
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — 最先进的 RoPE scaling。
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — Meta 的 Llama 2 long-context 论文。
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — Phi-3-Long 使用、并在 Use It 一节提到的 Microsoft 方法。
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) — 各种 RoPE scaling scheme（default、linear、dynamic、YaRN、LongRoPE、Llama-3）的生产级实现。
