# GloVe、FastText 与 Subword Embeddings

> Word2Vec 为每个词训练一个 embedding。GloVe 分解共现矩阵。FastText 嵌入词的组成部分。BPE 则通向 transformers。

**类型：** Build
**语言：** Python
**先修：** 第 5 阶段 · 03（从零实现 Word2Vec）
**时间：** 约 45 分钟

## 要解决的问题

Word2Vec 留下了两个开放问题。

第一，还有一条并行研究路线直接分解共现矩阵（LSA、HAL），而不是做在线 skip-gram 更新。Word2Vec 的迭代方法真的更好吗？还是两种方法处理计数的方式造成了差异？**GloVe** 给出了答案：使用精心选择的 loss 做矩阵分解，可以匹配或超过 Word2Vec，而且训练成本更低。

第二，这两种方法都没有处理从未见过的词。`Zoomer-approved`、`dogecoin`、上周刚造出来的任何专有名词、稀有词根的每种屈折形式。**FastText** 通过嵌入字符 n-gram 修复了这一点：一个词是它各个组成部分的和，包括 morphemes，所以即使 out-of-vocabulary 词也能得到合理向量。

第三，一旦 transformers 出现，问题又变了。词级词表大约会在百万量级封顶；真实语言比这开放得多。**Byte-pair encoding（BPE）**及其亲戚通过学习高频 subword 单元词表解决了这个问题，而且能覆盖一切。每个现代 LLM 的每个现代 tokenizer 都是 subword tokenizer。

本课会依次走过三者，然后解释什么时候该选哪个。

## 核心概念

**GloVe（Global Vectors）。** 构建 word-word 共现矩阵 `X`，其中 `X[i][j]` 表示词 `j` 出现在词 `i` 上下文中的次数。训练向量，使 `v_i · v_j + b_i + b_j ≈ log(X[i][j])`。给 loss 加权，避免高频 pair 主导训练。完成。

**FastText。** 一个词是它的字符 n-gram 加上词本身的和。`where` 变成 `<wh, whe, her, ere, re>, <where>`。词向量是这些组件向量之和。训练方式和 Word2Vec 一样。好处是：未见过的词（`whereupon`）可以由已知 n-gram 组合出来。

**BPE（Byte-Pair Encoding）。** 从单个 byte（或字符）的词表开始。统计语料中每个相邻 pair。把最频繁的 pair 合并成新 token。重复 `k` 次。结果是一个包含 `k + 256` 个 token 的词表，高频序列（`ing`、`tion`、`the`）会成为单个 token，稀有词会被拆成熟悉的片段。每个句子都能被 tokenized 成某些东西。

## 动手实现

### GloVe：分解共现矩阵

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

有两个移动部件值得点名。加权函数 `f(x) = (x/x_max)^alpha` 会降低非常高频 pair（如 `(the, and)`）的权重，避免它们主导 loss。最终 embedding 是 `W`（center）和 `W_tilde`（context）两张表的和。把二者相加是论文中的技巧，通常比只用其中一张表现更好。

### FastText：subword-aware embeddings

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

每个词都由它的一组 n-gram 表示（通常是 3 到 6 个字符）。词 embedding 是这些 n-gram embeddings 的和。对 skip-gram 训练来说，把它插到 Word2Vec 原先使用单个向量的位置即可。

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

对于未见过的词，只要它有一些 n-gram 已知，你仍然能得到向量。`whereupon` 和 `where` 共享 `<wh`、`her`、`ere`、`<where`，所以二者会落得比较近。

### BPE：学习得到的 subword 词表

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

第一次迭代会合并最常见的相邻 pair。足够多次迭代后，高频子串（`low`、`est`、`tion`）会成为单个 token，稀有词则会被干净地拆开。

真实 GPT / BERT / T5 tokenizer 会学习 30k-100k 个 merge。结果是：任何文本都会被 tokenized 成有界长度的已知 ID 序列，永远没有 OOV。

## 实际使用

实践中，你很少自己训练这些东西。你会加载预训练 checkpoint。

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

在 transformer 时代使用 BPE 风格的 subword tokenization：

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```text
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

`Ġ` 前缀标记词边界（GPT-2 约定）。每个现代 tokenizer 都是某种 BPE 变体、WordPiece（BERT）或 SentencePiece（T5、LLaMA）。

### 什么时候选哪个

| 场景 | 选择 |
|-----------|------|
| 预训练通用词向量，不需要 OOV 容忍 | GloVe 300d |
| 预训练通用词向量，必须处理拼写错误 / 新词 / 形态丰富语言 | FastText |
| 任何进入 transformer 的内容（训练或推理） | 模型自带的 tokenizer。绝对不要替换。 |
| 从零训练自己的语言模型 | 先在你的语料上训练 BPE 或 SentencePiece tokenizer |
| 使用线性模型的生产文本分类 | 仍然是 TF-IDF。第 02 课。 |

## 交付成果

保存为 `outputs/skill-embeddings-picker.md`：

```markdown
---
name: tokenizer-picker
description: Pick a tokenization approach for a new language model or text pipeline.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

Given a task and dataset description, you output:

1. Tokenization strategy (word-level, BPE, WordPiece, SentencePiece, byte-level). One-sentence reason.
2. Vocabulary size target (e.g., 32k for an English-only LM, 64k-100k for multilingual).
3. Library call with the exact training command. Name the library. Quote the arguments.
4. One reproducibility pitfall. Tokenizer-model mismatch is the single most common silent production bug; call out which pair must be used together.

Refuse to recommend training a custom tokenizer when the user is fine-tuning a pretrained LLM. Refuse to recommend word-level tokenization for any model targeting production inference. Flag non-English / multi-script corpora as needing SentencePiece with byte fallback.
```

## 练习

1. **简单。** 运行 `char_ngrams("playing")` 和 `char_ngrams("played")`。计算两个 n-gram 集合的 Jaccard overlap。你应该会看到大量共享片段（`pla`、`lay`、`play`），这就是为什么 FastText 能很好地跨形态变体迁移。
2. **中等。** 扩展 `learn_bpe`，追踪词表增长。画出 tokens-per-corpus-character 随 merge 次数变化的曲线。你应该会看到一开始压缩很快，然后在约 2-3 chars per token 附近渐近。
3. **困难。** 在莎士比亚全集上训练一个 1k-merge BPE。比较常见词和稀有专有名词的 tokenization。衡量前后每个词的平均 token 数。写下让你意外的发现。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Co-occurrence matrix | 词-词频率表 | `X[i][j]` = 词 `j` 在词 `i` 附近窗口中出现的次数。 |
| Subword | 词的一部分 | 字符 n-gram（FastText）或学习得到的 token（BPE/WordPiece/SentencePiece）。 |
| BPE | Byte-pair encoding | 反复合并最高频相邻 pair，直到词表达到目标大小。 |
| OOV | Out of vocabulary | 模型从未见过的词。Word2Vec/GloVe 会失败。FastText 和 BPE 能处理。 |
| Byte-level BPE | 原始 byte 上的 BPE | GPT-2 的方案。词表从 256 个 byte 开始，因此没有任何东西会 OOV。 |

## 延伸阅读

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — GloVe 论文，七页，仍然是最好的 loss 推导。
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — FastText。
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — 把 BPE 引入现代 NLP 的论文。
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) — BPE、WordPiece 和 SentencePiece 在实践中到底有什么不同。
