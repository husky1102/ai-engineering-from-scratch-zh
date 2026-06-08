# GPT — Causal Language Modeling

> BERT 能看两边。GPT 只能看过去。三角 mask 是现代 AI 中后果最深远的一行代码。

**类型:** Build
**语言:** Python
**先修:** Phase 7 · 02 (Self-Attention), Phase 7 · 05 (Full Transformer), Phase 7 · 06 (BERT)
**时间:** ~75 minutes

## 要解决的问题

语言模型回答一个问题：给定前 `t-1` 个 tokens，第 `t` 个 token 的概率分布是什么？用这个信号——next-token prediction——训练，你就得到一个能逐 token 生成任意文本的模型。

为了在整段 sequence 上并行端到端训练，你需要让每个位置的预测只依赖更早的位置。否则模型会通过看答案轻易作弊。

Causal mask 做到了这一点。它是一个由 `-inf` 值构成的 upper-triangular matrix，在 softmax 前加到 attention scores 上。softmax 后，这些位置变成 0。每个位置只能关注自己和之前的位置。因为你一次性把它应用到整段 sequence 上，所以一次 forward pass 就能得到 N 个并行 next-token predictions。

GPT-1 (2018)、GPT-2 (2019)、GPT-3 (2020)、GPT-4 (2023)、GPT-5 (2024)、Claude、Llama、Qwen、Mistral、DeepSeek、Kimi——它们全都是 decoder-only causal transformers，核心 loop 相同。只是更大、更好的数据、更好的 RLHF。

## 核心概念

![Causal mask creates a triangular attention matrix](../assets/causal-attention.svg)

### The mask

给定长度为 `N` 的 sequence，构建一个 `N × N` matrix：

```text
M[i, j] = 0       if j <= i
M[i, j] = -inf    if j > i
```

在 softmax 前把 `M` 加到 raw attention scores 上。`exp(-inf) = 0`，所以 masked positions 的权重贡献为零。Attention matrix 的每一行都是只覆盖 previous positions 的 probability distribution。

实现成本：一次 `torch.tril()` 调用。计算时间：纳秒级。对整个领域的影响：一切。

### Parallel training, serial inference

训练：一次 forward pass 处理整个 `(N, d_model)` sequence，计算 N 个 cross-entropy losses（每个位置一个），求和，backprop。沿 sequence 并行。这就是 GPT training 能扩展的原因——你可以在一次 GPU pass 中处理 batch 里的 1M tokens。

推理：逐 token 生成。输入 `[t1, t2, t3]`，得到 `t4`。输入 `[t1, t2, t3, t4]`，得到 `t5`。输入 `[t1, t2, t3, t4, t5]`，得到 `t6`。KV cache（Lesson 12）保存 `t1…tn` 的 hidden states，这样每一步不必重新计算它们。但推理时的串行深度 = 输出长度。这就是 autoregressive tax，也是每个 LLM 的解码延迟瓶颈。

### The loss — shift-by-one

给定 tokens `[t1, t2, t3, t4]`：

- Input: `[t1, t2, t3]`
- Targets: `[t2, t3, t4]`

对每个位置 `i`，计算 `-log P(target_i | inputs[:i+1])`。求和。这就是整段 sequence 的 cross-entropy。

你听过的每个 transformer LM 都用这个 loss 训练。Pre-training、fine-tuning、SFT——loss 相同，数据不同。

### Decoding strategies

训练后，sampling choices 比很多人以为的更重要。

| Method | What it does | When to use |
|--------|--------------|-------------|
| Greedy | Argmax every step | Deterministic tasks, code completion |
| Temperature | Divide logits by T, sample | Creative tasks, higher T = more diversity |
| Top-k | Sample from top-k tokens only | Kills low-probability tails |
| Top-p (nucleus) | Sample from smallest set with cumulative prob ≥ p | 2020+ default; adapts to distribution shape |
| Min-p | Keep tokens with `p > min_p * max_p` | 2024+; better at rejecting long tails than top-p |
| Speculative decoding | Draft model proposes N tokens, big model verifies | 2–3× latency reduction at same quality |

2026 年，对 open-weights models 来说，min-p + temperature 0.7 是一个合理默认值。Speculative decoding 已经是任何生产 inference stack 的标配。

### 让 “GPT recipe” 生效的因素

1. **Decoder-only.** 没有 encoder overhead。每层只做一轮 attention + FFN。
2. **Scaling.** 124M → 1.5B → 175B → trillions。Chinchilla scaling laws（Lesson 13）告诉你如何花 compute。
3. **In-context learning.** 大约在 6B–13B 出现。模型不用 fine-tuning 也能遵循 few-shot examples。
4. **RLHF.** 对 human preferences 做 post-training，把 raw pretrained text 转换成 chat assistants。
5. **Pre-norm + RoPE + SwiGLU.** 稳定大规模训练。

核心架构自 GPT-2 以来变化不大。真正有趣的变化发生在 data、scale 和 post-training。

## 动手实现

### Step 1: the causal mask

见 `code/main.py`。一行：

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

在 softmax 前把它加到 attention scores 上。整个机制就是这样。

### Step 2: a 2-layer GPT-ish model

堆叠两个 decoder blocks（masked self-attention + FFN，没有 cross-attention）。加入 token embedding、positional encoding 和 unembedding（与 token embedding matrix 绑定——这是 GPT-2 以来的标准技巧）。

### Step 3: next-token prediction, end-to-end

在 20-token toy vocab 上，为每个位置产生 logits。针对 shift-by-one target 计算 cross-entropy loss。不做 gradient——这是 forward-pass sanity check。

### Step 4: sampling

实现 greedy、temperature、top-k、top-p、min-p。在固定 prompt 上运行每种策略并比较输出。一个 sampling function 只需 10 行。

## 实际使用

PyTorch，2026 惯用写法：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

底层，`generate()` 会运行 forward pass，取最终位置 logits，采样下一个 token，把它 append，然后重复。每个生产 LLM inference stack（vLLM、TensorRT-LLM、llama.cpp、Ollama、MLX）都实现了同一个 loop，只是做了重度优化——batched prefill、continuous batching、KV cache paging、speculative decoding。

**GPT vs BERT，各一句：** GPT 预测 `P(x_t | x_{<t})`。BERT 预测 `P(x_masked | x_unmasked)`。Loss 决定了模型是否能生成。

## 交付成果

见 `outputs/skill-sampling-tuner.md`。这个 skill 会为新的 generation task 选择 sampling parameters，并标出需要 deterministic decoding 的场景。

## 练习

1. **Easy.** 运行 `code/main.py`，验证 softmax 后的 causal attention matrix 是 lower-triangular。Spot-check：第 3 行应该只在 columns 0–3 有权重。
2. **Medium.** 实现 width 4 的 beam search。在 10 个 short prompts 上比较 beam-4 和 greedy 的 perplexity。Beam 总是赢吗？（Hint: 通常对 translation 是，对 open-ended chat 不是。）
3. **Hard.** 实现 speculative decoding：用 tiny 2-layer model 作为 draft，用 6-layer model 作为 verifier。在 100 个长度 64 的 completions 上测量 wall-clock speedup。确认输出匹配 verifier 的 greedy。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Causal mask | “The triangle” | 加到 attention scores 上的 upper-triangular `-inf` matrix，使位置 `i` 只能看见位置 `≤ i`。 |
| Next-token prediction | “The loss” | 模型分布与每个位置真实 next token 之间的 cross-entropy。 |
| Autoregressive | “一次生成一个” | 把输出反馈为输入；并行只在训练时存在，生成时不存在。 |
| Logits | “Pre-softmax scores” | LM head 在 softmax 前的 raw output；sampling 发生在这些值上。 |
| Temperature | “Creativity knob” | 用 T 除 logits；T→0 = greedy，T→∞ = uniform。 |
| Top-p | “Nucleus sampling” | 把分布截断到 cumulative probability ≥p 的最小集合；从剩余部分采样。 |
| Min-p | “Better than top-p” | 保留 `p ≥ min_p × max_p` 的 tokens；cutoff 会随分布尖锐程度自适应。 |
| Speculative decoding | “Draft + verify” | 便宜模型提出 N 个 tokens；大模型并行验证。 |
| Teacher forcing | “Training trick” | 训练时喂真实 previous token，而不是模型自己的 prediction。每个 seq2seq LM 的标准做法。 |

## 延伸阅读

- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1。
- [Radford et al. (2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2。
- [Brown et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3 和 in-context learning。
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — spec decoding 论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — canonical causal-LM reference code。
