# Transformer 之前的文本生成：N-gram 语言模型

> 如果一个词让模型感到意外，模型就是差的。Perplexity 把“意外”变成数字。Smoothing 让它保持有限。

**类型：** Build
**语言：** Python
**先修：** Phase 5 · 01 (Text Processing), Phase 2 · 14 (Naive Bayes)
**时间：** ~45 minutes

## 要解决的问题

在 transformers 之前，在 RNNs 之前，在 word embeddings 之前，语言模型通过统计一个词跟在前面 `n-1` 个词之后的频率来预测下一个词。统计 "the cat" → "sat" 出现 47 次，"the cat" → "jumped" 出现 12 次，"the cat" → "refrigerator" 出现 0 次。归一化后得到一个概率分布。

这就是 n-gram language model。从 1980 年到 2015 年，它支撑了每一个 speech recognizer、每一个 spell checker，以及每一个 phrase-based machine translation system。当你需要便宜的 on-device language modeling 时，它今天仍然在运行。

有趣的问题是如何处理没见过的 n-grams。一个原始 count-based model 会把没有见过的任何东西赋予零概率，这很灾难，因为句子很长，而几乎每个长句都至少包含一个未见过的序列。五十年的 smoothing 研究修复了这一点。Kneser-Ney smoothing 是结果，而现代 deep learning 继承了它的经验传统。

## 核心概念

![N-gram model: count, smooth, generate](../assets/ngram.svg)

**N-gram probability：** `P(w_i | w_{i-n+1}, ..., w_{i-1})`。固定 `n`（trigrams 通常取 3，4-grams 取 4）。从计数中计算：

```text
P(w | context) = count(context, w) / count(context)
```

**零计数问题。** 训练中未见过的任何 n-gram 都会得到零概率。一项 2007 年针对 Brown corpus 的研究发现，即使是 4-gram model，也有 30% 的 held-out 4-grams 在训练中未出现。没有 smoothing，你无法在任何真实文本上评估。

**Smoothing approaches，按复杂度排序：**

1. **Laplace (add-one)。** 给每个 count 加 1。简单，但在稀有事件上很糟。
2. **Good-Turing。** 基于 frequency-of-frequencies，把 probability mass 从更高频事件重新分配给未见事件。
3. **Interpolation。** 用可调权重组合 n-gram、(n-1)-gram 等估计。
4. **Backoff。** 如果 n-gram 的 count 为零，就退回到 (n-1)-gram。Katz backoff 会对它归一化。
5. **Absolute discounting。** 从所有 counts 中减去一个固定 discount `D`，再重新分配给未见事件。
6. **Kneser-Ney。** Absolute discounting 加上对 lower-order model 的巧妙选择：使用 *continuation probability*（一个词出现在多少种 context 中），而不是 raw frequency。

Kneser-Ney 的洞见很深。"San Francisco" 是一个常见 bigram。Unigram "Francisco" 大多出现在 "San" 之后。朴素 absolute discounting 会给 "Francisco" 很高的 unigram probability（因为 count 很高）。Kneser-Ney 注意到 "Francisco" 只出现在一种 context 中，并相应降低它的 continuation probability。结果是：一个以 "Francisco" 结尾的新 bigram 会得到合适的低概率。

**Evaluation: perplexity。** 在 held-out test set 上，每个词平均 negative log-likelihood 的指数。越低越好。Perplexity 为 100 表示模型的困惑程度相当于在 100 个词中均匀选择。

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

## 动手实现

### Step 1: trigram counts

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

输入是 tokenized sentences 的列表。输出是 n-gram counts 和 context counts。`<s>` 与 `</s>` 是句子边界。

### Step 2: Laplace smoothing

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

给每个 count 加 1。它能 smooth，但会给 unseen events 分配过多 mass，同时也伤害 rare-known events。

### Step 3: Kneser-Ney (bigram, interpolated)

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

有三个移动部件。`continuation_prob` 捕捉“这个词出现在多少种不同 context 中？”（Kneser-Ney 创新）。`lambda_prev` 是 discount 释放出的 mass，用来为 backoff 加权。最终概率是 discounted main term 加上 weighted continuation term。

### Step 4: generating text with sampling

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

按概率成比例 sampling。每个 seed 总是给出不同输出。对于类似 beam search 的输出，在每一步选择 argmax（greedy），并添加一个小的 randomness knob（temperature）。

### Step 5: perplexity

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

越低越好。对于 Brown corpus，一个调优良好的 4-gram KN model perplexity 大约为 140。Transformer LM 在同一个 test set 上能达到 15-30。差距大约是 10 倍。这就是这个领域继续前进的原因。

## 实际使用

- **Classical NLP teaching。** 你能获得的最清晰的 smoothing、MLE 和 perplexity 入门。
- **KenLM。** 生产级 n-gram library。在低延迟重要的 speech 和 MT systems 中用作 rescorer。
- **On-device autocomplete。** 键盘里的 trigram models。至今如此。
- **Baselines。** 在宣布你的 neural LM 很好之前，始终先计算一个 n-gram LM perplexity。如果你的 transformer 没有大幅超过 KN，说明哪里出了问题。

## 交付成果

保存为 `outputs/prompt-lm-baseline.md`：

```markdown
---
name: lm-baseline
description: Build a reproducible n-gram language model baseline before training a neural LM.
phase: 5
lesson: 16
---

Given a corpus and target use (next-word prediction, rescoring, perplexity baseline), output:

1. N-gram order. Trigram for general English, 4-gram if corpus is large, 5-gram for speech rescoring.
2. Smoothing. Modified Kneser-Ney is the default; Laplace only for teaching.
3. Library. `kenlm` for production, `nltk.lm` for teaching, roll your own only to learn.
4. Evaluation. Held-out perplexity with consistent tokenization between train and test sets.

Refuse to report perplexity computed with different tokenization between systems being compared — perplexity numbers are comparable only under identical tokenization. Flag OOV rate in test set; KN handles OOV poorly unless you reserve a special <UNK> token during training.
```

## 练习

1. **Easy.** 在 1,000 句 Shakespeare corpus 上训练一个 trigram LM。生成 20 个句子。它们会局部合理，但全局不连贯。这是 canonical demo。
2. **Medium.** 在 held-out Shakespeare split 上为你的 KN model 实现 perplexity。与 Laplace 对比。你应该看到 KN 将 perplexity 降低 30-50%。
3. **Hard.** 构建一个 trigram spell corrector：给定一个 misspelled word 及其 context，生成候选 corrections，并根据 LM 下的 context probability 排序。在 Birkbeck spelling corpus（公开）上评估。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| N-gram | 词序列 | 连续 `n` 个 tokens 的序列。 |
| Smoothing | 避免零概率 | 重新分配 probability mass，让 unseen events 获得非零概率。 |
| Perplexity | LM 质量指标 | Held-out data 上的 `exp(-average log-prob)`。越低越好。 |
| Backoff | 回退到更短 context | 如果 trigram count 为零，就使用 bigram。Katz backoff 将其形式化。 |
| Kneser-Ney | n-grams 的最佳 smoothing | Absolute discounting + 用于 lower-order model 的 continuation probability。 |
| Continuation probability | KN-specific | `P(w)` 按 `w` 出现的 context 数量加权，而不是按 raw count。 |

## 延伸阅读

- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — n-gram LMs 与 smoothing 的 canonical treatment。
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) — 确立 Kneser-Ney 作为最佳 n-gram smoother 的论文。
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) — 原始 KN 论文。
- [KenLM](https://kheafield.com/code/kenlm/) — 快速的生产级 n-gram LM，2026 年仍用于延迟敏感的应用。
