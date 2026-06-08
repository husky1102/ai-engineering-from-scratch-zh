# 为什么是 Transformers：RNN 的问题

> RNN 一次处理一个 token。Transformers 一次处理所有 token。这个单一架构押注改变了 2017 年后深度学习的每一条 scaling curve。

**类型：** Learn
**语言：** Python
**先修：** Phase 3 (Deep Learning Core), Phase 5 · 09 (Sequence-to-Sequence), Phase 5 · 10 (Attention Mechanism)
**时间：** ~45 分钟

## 要解决的问题

2017 年之前，地球上每个 state-of-the-art 序列模型：语言、翻译、语音，都是 recurrent neural network。LSTM 和 GRU 在半个十年里赢下了相当于 ImageNet 的翻译 benchmarks。它们是大家唯一拥有的工具。

它们有三个致命弱点。顺序计算意味着你不能沿时间轴并行化：token `t+1` 需要 token `t` 的 hidden state。一个 1,024-token sequence 意味着在每个 cycle 可以做 1,000,000 次浮点运算的 GPU 上执行 1,024 个串行 step。训练 wall-clock time 在为并行设计的硬件上随 sequence length 线性增长。

梯度消失意味着 50 个 token 之前的信息已经穿过 50 个非线性而被压缩。门控循环单元（LSTM、GRU）缓解了这种挤压，但从未消除它。长程依赖，例如 “the book I read last summer on a plane to Kyoto was…” 经常失败。

固定宽度 hidden states 意味着 encoder 会在 decoder 看见任何东西之前，把整个 source sequence 压进单个 vector。source 是 5 个 token 还是 500 个都无所谓；瓶颈形状相同。

2017 年论文 “Attention Is All You Need” 提出了一个激进想法：完全去掉 recurrence。让每个位置并行 attend 到每个其他位置。用一次大型矩阵乘法训练，而不是 1,024 次顺序计算。

结果到 2026 年主导了每个模态。语言（GPT-5、Claude 4、Llama 4）、视觉（ViT、DINOv2、SAM 3）、音频（Whisper）、生物学（AlphaFold 3）、机器人（RT-2）。同一个 block，不同输入。

## 核心概念

![RNN 顺序计算 vs Transformer 并行 attention](../assets/rnn-vs-transformer.svg)

**Recurrence 是瓶颈。** RNN 计算 `h_t = f(h_{t-1}, x_t)`。每一步依赖前一步。你不能在 `h_4` 之前计算 `h_5`。在有 10,000+ 并行核心的现代 GPU 上，长序列会浪费 99% 的硅。

**Attention 是 broadcast。** Self-attention 对每一对 `(i, j)` 同时计算 `output_i = sum_j(a_ij * v_j)`。整个 N×N attention matrix 在一次 batched matmul 中填满。没有 step 依赖另一个。GPU 喜欢这种形状。

**Speedup 不是常数。** 它是 `O(N)` serial depth 与 `O(1)` serial depth 的差异。实践中，在 N=512、硬件匹配时，transformers 每个 epoch 训练快 5-10×，并且 gap 会随 sequence length 扩大，直到你撞上 attention 的 `O(N²)` memory wall（Flash Attention 之后修复了这个问题，见 Lesson 12）。

**Transformers 的代价。** Attention memory 按 `O(N²)` 缩放。2K context 没问题。128K context 则需要 sliding windows、RoPE extrapolation、Flash Attention tiling 或 linear attention variants。Recurrence 在 time 和 memory 上都是 `O(N)`；transformers 用 memory 换 time，然后通过并行化把 time 赢回来。

**Inductive bias 的转移。** RNN 假设 locality 和 recency。Transformers 不做假设，每一对都是 attention 候选。这就是为什么 transformers 需要更多数据才能训练好，但一旦有数据就能扩展得更远。Chinchilla（2022）形式化了这一点：给定足够 token，同参数量 transformer 总会击败 RNN。

## 动手实现

这里没有神经网络，我们用数值方式模拟核心瓶颈，让你在笔记本上感受到差距。

### Step 1：测量 serial depth

见 `code/main.py`。我们构建两个函数。一个把序列编码成加法链（serial，像 RNN）。另一个把它编码成并行归约（broadcast，像 attention）。数学相同，dependency graph 不同。

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # can't parallelize: h depends on previous h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # every x is independent
```

我们在最长 100,000 个元素的序列上计时二者。RNN 版本是 O(N)，并且只能走单条 CPU pipeline。即使在纯 Python 中，attention-style reduction 在长度 ≥ 1,000 时也会胜出，因为 Python 的 `sum()` 用 C 实现，不会在每步承担解释器开销。

### Step 2：计算理论操作

两个算法都做 N 次加法。区别是 *dependency depth*：在下一个操作能开始前，必须顺序完成多少操作。RNN depth = N。Attention depth = tree reduction 下的 log(N)，或者 parallel scan 下的 1。决定 GPU 时间的是 depth，而不是 op count。

### Step 3：长序列上的经验缩放

我们打印一张 timing table，让 O(N) gap 可见。在 2026 年的 Mac 笔记本上，低于 1,000 个元素的序列太快，难以测量。100,000 的序列会显示出清晰线性扫描。把它扩展到 16,384-token transformer 和 12 层 LSTM 等价模型，你就会看到为什么训练 wall-clock 在 2016 年是 blocker。

## 实际使用

2026 年什么时候仍然选择 RNN：

| Situation | Pick |
|-----------|------|
| Streaming inference，一次一个 token，常量内存 | RNN 或 state-space model（Mamba、RWKV） |
| 超长序列（>1M tokens），attention memory 爆炸 | Linear attention、Mamba 2、Hyena |
| 没有 matmul accelerator 的 edge device | Depthwise-separable RNN 在 FLOPs/watt 上仍胜出 |
| 其他任何情况（training、batched inference、context 到 128K） | Transformer |

Mamba 这样的 state-space models（SSMs）本质上是带结构化参数化的 RNN，给了它们两边的优点：`O(N)` scan memory，通过 selective scan 并行训练。它们以更好的 long-context scaling 恢复了 90% 的 transformer 质量。2026 年多数 frontier labs 训练 hybrid SSM+transformer models（例如 Jamba、Samba）；recurrence 没有死，它只是一个组件。

## 交付成果

见 `outputs/skill-architecture-picker.md`。这个 skill 会根据 length、throughput 和 training-budget constraints，为新的 sequence problem 选择架构。对于超过 1B tokens 的训练运行，它应该始终拒绝推荐 pure RNN，除非明确说明 trade-off。

## 练习

1. **Easy。** 取 `code/main.py` 中的 `rnn_style`，把标量 hidden state 换成长度为 64 的 hidden states vector。重新测量。serial overhead 会随 hidden-state dimension 增长多少？
2. **Medium。** 用纯 Python 实现 parallel prefix-sum（Hillis-Steele scan）。验证它在长度 1024 上产生和 serial scan 相同的数值输出。计算 depth。
3. **Hard。** 把 attention-style reduction 移植到 GPU 上的 PyTorch。随着 sequence length 从 64 sweep 到 65,536，对二者计时。绘图并解释 curve shape。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Recurrence | “RNNs are sequential” | step `t` 依赖 step `t-1` 的计算，迫使沿时间轴串行执行。 |
| Serial depth | “How deep the graph is” | 最长依赖操作链；即使有无限硬件，也会限制 wall-clock。 |
| Attention | “Let tokens look at each other” | 加权和 `sum_j a_ij v_j`，其中 `a_ij` 来自位置 i 和 j 之间的相似度分数。 |
| Context window | “How much the model sees” | attention layer 可接受的输入位置数；二次 memory cost 在这里缩放。 |
| Inductive bias | “Assumptions baked into the architecture” | 关于数据形态的先验；CNN 假设 translation invariance，RNN 假设 recency。 |
| State-space model | “RNN with algebra behind it” | 通过结构化 state-space matrices 参数化 recurrence，以支持并行训练。 |
| Quadratic bottleneck | “Why context costs so much” | Attention memory = sequence length 上的 `O(N²)`；Flash Attention 隐藏常数，不改变 scaling。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 让 recurrence 在主流 NLP 中退场的论文。
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — attention 的诞生地，当时接在 RNN 上。
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — 原始 LSTM 论文，作为记录。
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — 面向 transformers 的现代 recurrent 答案。
