# Build a Transformer from Scratch — The Capstone

> 十三节课。一个模型。不走捷径。

**类型：** Build
**语言：** Python
**先修：** Phase 7 · 01 through 13. 不要跳过。
**时间：** ~120 分钟

## 要解决的问题

你已经读过每篇 paper。你已经实现了 attention、multi-head splits、positional encodings、encoder and decoder blocks、BERT and GPT losses、MoE、KV cache。现在让它们在真实任务上一起工作。

Capstone：在 character-level language modeling task 上端到端训练一个小型 decoder-only transformer。它读 Shakespeare。它生成新的 Shakespeare。它足够小，可以在 laptop 上 10 分钟内训练完。它也足够正确：换成更大的 dataset 和更长训练，就能得到一个真正的 LM。

这是本课程的 “nanoGPT”。它并非原创：Karpathy 2023 年的 nanoGPT tutorial 是每个学生至少会写一次的 reference implementation。我们借用它的形状，并围绕本阶段已经讲过的内容重新整理。

## 核心概念

![Transformer-from-scratch block diagram](../assets/capstone.svg)

标注过的 architecture：

```text
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE option)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07
```

### 我们交付什么

- `GPTConfig`：集中配置所有 hyperparameters。
- `MultiHeadAttention`：causal、batched，带可选 Flash-style pathway（PyTorch 的 `scaled_dot_product_attention`）。
- `SwiGLUFFN`：现代 FFN。
- `Block`：pre-norm、residual-wrapped attention + FFN。
- `GPT`：embeddings、stacked blocks、LM head、generate()。
- 使用 AdamW、cosine LR、gradient clipping 的 training loop。
- Shakespeare text 上的 char-level tokenizer。

### 我们不交付什么

- RoPE：Lesson 04 已经从概念上实现。这里为了简单使用 learned positional embeddings。练习会让你换成 RoPE。
- Generation 期间的 KV cache：每个 generation step 都会在完整 prefix 上重算 attention。更慢但更简单。练习会让你加入 KV cache。
- Flash Attention：如果输入匹配，PyTorch 2.0+ 会自动 dispatch；我们使用 `F.scaled_dot_product_attention`。
- MoE：每个 block 一个 FFN。你在 Lesson 11 已经见过 MoE。

### Target metrics

在 Mac M2 laptop 上，一个 4-layer、4-head、d_model=128 的 GPT 在 `tinyshakespeare.txt` 上训练 2,000 steps：

- Training loss 在大约 6 分钟内从 ~4.2（random）收敛到 ~1.5。
- Sampled output 看起来像 Shakespeare：古风词、换行、像 “ROMEO:” 这样的 proper names 开始出现。
- Val loss（held-out final 10% of text）紧跟 training loss；在这个 size/budget 下没有 overfitting。

## 动手实现

本课使用 PyTorch。安装 `torch`（CPU build 也可以）。见 `code/main.py`。脚本会处理：

- 如果缺失就下载 `tinyshakespeare.txt`（或读取本地副本）。
- Byte-level char tokenizer。
- 90/10 train/val split。
- 在支持的硬件上使用 bf16 autocast 的 training loop。
- 训练完成后的 sampling。

### Step 1: data

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 个 unique characters。很小的 vocabulary。适合 4-byte vocab_size。没有 BPE，没有 tokenizer drama。

### Step 2: model

见 `code/main.py`。这个 block 是 Lesson 05 的 textbook 版本：pre-norm、RMSNorm、SwiGLU、causal MHA。4/4/128 的 parameter count 约为 800K。

### Step 3: training loop

获取一个 length-256 token windows 的 random batch。Forward。Shift-by-one cross-entropy。Backward。AdamW step。Log。重复。

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### Step 4: sample

给定 prompt，反复 forward，从 top-p logits 中 sample、append，并继续。500 tokens 后停止。

### Step 5: read the output

2,000 steps 后：

```text
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

不是 Shakespeare。但已经 Shakespeare-shaped。对约 800K parameters 和 laptop 上 6 分钟来说，这是明确胜利。

## 实际使用

这个 capstone 是一个 reference architecture。要把它推进到真实东西，做三个扩展：

1. **替换 tokenizer。** 使用 BPE（例如 `tiktoken.get_encoding("cl100k_base")`）。Vocab size 会从 65 跳到约 50,000。Model capacity 需要随之扩大来补偿。
2. **在更大的 corpus 上训练。** 使用 `OpenWebText` 或 `fineweb-edu`（HuggingFace）。在单张 A100 上用 10B tokens 训练 125M-param GPT 大约需要 24 小时。
3. **加入 RoPE + KV cache + Flash Attention。** 下面的练习会带你逐个完成。

最终你会得到一个 125M-parameter GPT，能生成流畅英文。它不是 frontier model。但同一条 code path，只是更大，就是 Karpathy、EleutherAI 和 Allen Institute 在 2026 年用来训练 research checkpoints 的东西。

## 交付成果

见 `outputs/skill-transformer-review.md`。这个 skill 会基于前 13 节课，审查一个 transformer-from-scratch implementation 的正确性。

## 练习

1. **Easy.** 运行 `code/main.py`。验证你训练出的模型 final-step validation loss 低于 2.0。把 `max_steps` 从 2,000 改成 5,000：val loss 会继续改善吗？
2. **Medium.** 用 RoPE 替换 learned positional embeddings。在 `MultiHeadAttention` 内对 Q 和 K 应用 rotation。训练并验证 val loss 至少一样低。
3. **Medium.** 在 sampling loop 中实现 KV cache。分别使用 cache 和不使用 cache 生成 500 tokens。Laptop 上 wall-clock 应提升 5-20×。
4. **Hard.** 给模型增加第二个 head，用来预测 next-plus-one token（MTP，DeepSeek-V3 的 Multi-Token Prediction）。联合训练。它有帮助吗？
5. **Hard.** 用 4-expert MoE 替换每个 block 中的单个 FFN。Router + top-2 routing。观察在 matched active parameters 下 val loss 如何变化。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| nanoGPT | “Karpathy 的 tutorial repo” | 最小 decoder-only transformer training code，约 300 LOC；canonical reference。 |
| tinyshakespeare | “标准 toy corpus” | 约 1.1 MB 文本；2015 年以来每个 character-LM tutorial 都会用它。 |
| Tied embeddings | “共享 input/output matrix” | LM head weight = token embedding matrix 的转置；节省参数并改善质量。 |
| bf16 autocast | “Training precision trick” | forward/back 用 bf16，optimizer state 保持 fp32；自 2021 年起成为标准。 |
| Gradient clipping | “Stops spikes” | 把 global grad norm 限制在 1.0；防止 training blowups。 |
| Cosine LR schedule | “2020+ 默认项” | LR 先线性 warmup，再按 cosine shape 衰减到 peak 的 10%。 |
| MFU | “Model FLOP Utilization” | Achieved FLOPs / theoretical peak；2026 年 dense 40%、MoE 30% 就很强。 |
| Val loss | “Held-out loss” | 模型从未见过的数据上的 cross-entropy；overfit detector。 |

## 延伸阅读

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) — 经典 annotated implementation。
