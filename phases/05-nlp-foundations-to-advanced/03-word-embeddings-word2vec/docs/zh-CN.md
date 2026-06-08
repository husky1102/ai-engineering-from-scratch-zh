# 词嵌入：从零实现 Word2Vec

> 一个词的意义，来自它身边的词。用这个想法训练一个浅层网络，几何结构就会浮现。

**类型：** Build
**语言：** Python
**先修：** 第 5 阶段 · 02（BoW + TF-IDF），第 3 阶段 · 03（从零实现反向传播）
**时间：** 约 75 分钟

## 要解决的问题

TF-IDF 知道 `dog` 和 `puppy` 是不同的词。它不知道它们的意思几乎相同。一个在 `dog` 上训练过的分类器，不能自然泛化到一条关于 `puppy` 的评论。你可以靠列同义词来临时糊住这个洞，但这会在稀有术语、领域黑话，以及所有你没有预想到的语言上失败。

你想要一种表示，让 `dog` 和 `puppy` 在空间里彼此靠近。让 `king - man + woman` 落在 `queen` 附近。让一个在 `dog` 上训练过的模型，免费把一些信号迁移到 `puppy`。

Word2Vec 给了我们这个空间。两层神经网络，万亿 token 级训练，发表于 2013 年。架构简单到几乎有点尴尬。结果却重塑了 NLP 十年。

## 核心概念

**分布假说**（Firth, 1957）："You shall know a word by the company it keeps." 如果两个词出现在相似上下文中，它们大概率有相似含义。

Word2Vec 有两种形态，二者都利用这个想法。

- **Skip-gram。** 给定中心词，预测周围词。窗口大小为 2 时，`cat -> (the, sat, on)`。
- **CBOW（continuous bag of words）。** 给定周围词，预测中心词。`(the, sat, on) -> cat`。

Skip-gram 训练更慢，但对稀有词更好。因此它成了默认选择。

这个网络只有一个隐藏层，没有非线性。输入是词表上的 one-hot 向量。输出是词表上的 softmax。训练之后，你丢掉输出层。隐藏层权重就是 embeddings。

```text
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

技巧在于：对 10 万个词做 softmax 极其昂贵。Word2Vec 使用 **negative sampling** 把它变成二分类任务。预测“这个上下文词是否出现在这个中心词附近，是或否”。每个训练 pair 只采样少量负例（未共现的词），而不是对整个词表计算 softmax。

## 动手实现

### 第 1 步：从语料生成训练 pair

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

窗口内每个 `(center, context)` pair 都是一个正训练样本。

### 第 2 步：embedding 表

两个矩阵。`W` 是中心词 embedding 表（你最终保留的那个）。`W'` 是上下文词表（通常丢弃，有时会和 `W` 取平均）。

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

小随机初始化。词表大小 10k、维度 100 是现实的；教学时，50 个词 x 16 维已经足够看出几何结构。

### 第 3 步：negative sampling 目标

对每个正 pair `(center, context)`，从词表中随机采样 `k` 个词作为负例。训练模型，让 `W[center] · W'[context]` 对正例高、对负例低。

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

魔法公式是：正 pair 上的 logistic loss（希望 sigmoid 接近 1）加负 pair 上的 logistic loss（希望 sigmoid 接近 0）。梯度会流向两张表。完整推导在原论文里；如果你想真正记住它，拿纸笔走一遍。

### 第 4 步：在玩具语料上训练

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

在大语料上训练足够多 epoch 后，共享上下文的词会拥有相似的中心词 embeddings。在玩具语料上，你只能隐约看到这个效果。在数十亿 token 上，你会非常明显地看到它。

### 第 5 步：类比技巧

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

在预训练的 300d Google News 向量上：

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`。不是因为模型知道什么是王权。原因是向量 `(king - man)` 捕捉到某种类似“royal”的方向，把它加到 `woman` 上，就落在 royal-female 区域附近。

## 实际使用

从零写 Word2Vec 是教学。生产 NLP 使用 `gensim`。

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

真实工作里，你几乎不会自己训练 Word2Vec。你会下载预训练向量。

- **GloVe** — Stanford 的共现矩阵分解方法。50d、100d、200d、300d checkpoint。通用覆盖好。第 04 课会专门讲 GloVe。
- **fastText** — Facebook 的 Word2Vec 扩展，嵌入字符 n-gram。通过组合 subwords 处理 out-of-vocabulary 词。第 04 课。
- **Pretrained Word2Vec on Google News** — 300d，300 万词词表，2013 年发布。今天仍然每天有人下载。

### 2026 年 Word2Vec 仍然会赢的场景

- 轻量级领域检索。在一台笔记本上用医学摘要训练一小时，就能得到通用模型没有捕捉到的专用向量。
- 类比式特征工程。`gender_vector = mean(man - woman pairs)`。从其他词里减掉它，得到一个 gender-neutral 轴。公平性研究里仍在使用。
- 可解释性。100d 小到可以通过 PCA 或 t-SNE 可视化，并真正看到簇形成。
- 任何必须在无 GPU 设备端运行推理的地方。Word2Vec lookup 只是一次取行。

### Word2Vec 失效的地方

一词多义墙。`bank` 只有一个向量。`river bank` 和 `financial bank` 共用它。`table`（电子表格 vs 家具）也共用它。下游分类器无法从这个向量里区分语义。

Contextual embeddings（ELMo、BERT 以及之后每个 transformer）通过根据周围上下文为单词的每次出现生成不同向量，解决了这个问题。这就是从 Word2Vec 到 BERT 的跳跃：从 static 到 contextual。第 7 阶段会覆盖 transformer 的那一半。

Out-of-vocabulary 问题是另一个失败点。如果训练数据里没有 `Zoomer-approved`，Word2Vec 就从未见过它。没有 fallback。fastText 用 subword composition 修复它（第 04 课）。

## 交付成果

保存为 `outputs/skill-embedding-probe.md`：

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.
```

## 练习

1. **简单。** 在一个很小的语料（20 个关于猫狗的句子）上运行训练循环。200 个 epoch 后，验证 `nearest(vocab, W, W[vocab["cat"]])` 的 top 3 里返回了 `dog`。如果没有，增加 epochs 或词表。
2. **中等。** 添加高频词 subsampling。频率高于 `10^-5` 的词会以和频率成比例的概率从训练 pair 中丢弃。衡量它对稀有词相似度的影响。
3. **困难。** 在 20 Newsgroups 语料上训练一个模型。计算两个 bias axis：`he - she` 和 `doctor - nurse`。把职业词投影到这两个轴上。报告哪些职业的 bias gap 最大。这正是公平性研究人员会使用的 probe。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Word embedding | 作为向量的词 | 从上下文中学习到的 dense、低维（通常 100-300）表示。 |
| Skip-gram | Word2Vec 技巧 | 从中心词预测上下文词。比 CBOW 慢，但对稀有词更好。 |
| Negative sampling | 训练捷径 | 用针对 `k` 个随机词的二分类，替代整词表 softmax。 |
| Static embedding | 每个词一个向量 | 无论上下文如何，向量都相同。会在一词多义上失败。 |
| Contextual embedding | 对上下文敏感的向量 | 根据周围词为每次出现生成不同向量。Transformer 产出的就是这个。 |
| OOV | Out of vocabulary | 训练中没见过的词。Word2Vec 无法为它们产生向量。 |

## 延伸阅读

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) — negative-sampling 论文。短，而且可读。
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) — 如果原论文的数学显得太密，这是最清晰的梯度推导。
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) — 实际可用的生产训练设置。
