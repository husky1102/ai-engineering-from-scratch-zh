# 词袋、TF-IDF 与文本表示

> 先计数，再思考。到 2026 年，在定义清楚的任务上，TF-IDF 仍然能赢过 embeddings。

**类型：** Build
**语言：** Python
**先修：** 第 5 阶段 · 01（文本处理），第 2 阶段 · 02（从零实现线性回归）
**时间：** 约 75 分钟

## 要解决的问题

模型需要数字。你手里是字符串。

每条 NLP pipeline 都必须回答同一个问题：怎样把长度可变的 token 流变成分类器可以消费的固定大小向量？这个领域最早落地的答案，是最笨但有效的答案：数词。做成向量。

这个向量承载过的生产 NLP，比任何 embedding 模型都多。垃圾邮件过滤、主题分类、日志异常检测、搜索排序（BM25 之前）、第一波情感分析、学术 NLP benchmark 的第一个十年。到 2026 年，从业者在窄分类任务上仍然会优先拿它试。它快、可解释，而且在词是否出现才是关键信号的任务上，表现常常和 400M 参数的 embedding 模型没什么区别。

本课会从零构建 bag of words，再构建 TF-IDF。然后展示 scikit-learn 如何用三行完成同样的事。最后点名会让你转向 embeddings 的失效模式。

## 核心概念

**Bag of Words（BoW）**丢掉顺序。对每篇文档，统计每个词表词出现了多少次。向量长度就是词表大小。位置 `i` 是第 `i` 个词的计数。

**TF-IDF**对 BoW 重新加权。一个出现在每篇文档里的词没有信息量，所以降低它的权重。一个在语料里罕见、但在单篇文档中频繁出现的词是信号，所以提高它的权重。

```text
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

其中 `TF` 是词在文档中的词频，`df` 是文档频率（包含该词的文档数量），`N` 是文档总数。`log` 会让常见词的权重保持有界。

关键性质：二者都会产生轴可解释的稀疏向量。你可以查看训练后分类器的权重，并读出哪些词把文档推向哪个类别。768 维 BERT embedding 做不到这一点。

## 动手实现

### 第 1 步：构建词表

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

输入：tokenized documents 列表（任何词级 tokenizer 都可以；本课 `code/main.py` 使用了一个简化的小写变体）。输出：`{word: index}` 字典。稳定插入顺序意味着索引 0 是第一篇文档里第一次出现的第一个词。不同工具的约定不同；scikit-learn 会按字母排序。

### 第 2 步：bag of words

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

行是文档。列是词表索引。条目 `[i][j]` 表示“词 `j` 在文档 `i` 里出现了多少次”。文档 1 里 `cat` 出现两次，因为它确实出现了两次。文档 0 里 `ran` 是 0 次，因为它没有出现。

### 第 3 步：词频和文档频率

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

有两个值得点名的平滑技巧。`(n+1)/(d+1)` 避免 `log(x/0)`。末尾的 `+1` 确保一个出现在每篇文档里的词仍然有 IDF 1（而不是 0），这和 scikit-learn 的默认值一致。其他实现会使用原始的 `log(N/df)`。两者都能工作；平滑版本更友好。

### 第 4 步：TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

三篇文档，五个词表词（`the`、`cat`、`sat`、`dog`、`ran`）。`the` 出现在三篇里，所以它的 IDF 低。`dog` 只出现一篇里，所以它的 IDF 高。向量是稀疏的（大多数条目都很小），而有区分力的词会凸显出来。

### 第 5 步：对行做 L2 归一化

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

如果不归一化，较长文档会得到更大的向量，并主导相似度分数。L2 归一化把每篇文档放到单位超球面上。此时行之间的余弦相似度就是点积。

## 实际使用

scikit-learn 提供了生产版本。

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` 在一次调用里完成 tokenization、vocabulary 和 BoW。`TfidfVectorizer` 增加 IDF 加权和 L2 归一化。二者都返回稀疏矩阵。对 10 万篇文档来说，dense 版本放不进内存；在分类器要求 dense 之前，要一直保持 sparse。

会彻底改变结果的旋钮：

| 参数 | 影响 |
|-----|--------|
| `ngram_range=(1, 2)` | 包含 bigram。通常会提升分类。 |
| `min_df=2` | 丢掉少于 2 篇文档中出现的词。能在噪声数据上修剪词表。 |
| `max_df=0.95` | 丢掉超过 95% 文档中出现的词。相当于不用硬编码列表就近似移除 stopword。 |
| `stop_words="english"` | scikit-learn 内置的 stopword 列表。依赖任务；情感分析不应该丢掉否定词。 |
| `sublinear_tf=True` | 用 `1 + log(tf)` 替代原始 `tf`。当某个词在单篇文档里重复很多次时有帮助。 |

### 到 2026 年 TF-IDF 仍然会赢的场景

- 垃圾邮件检测、主题标注、日志异常标记。词是否出现才重要；语义细微差别不重要。
- 低数据量场景（几百个标注样本）。TF-IDF 加 logistic regression 没有预训练成本。
- 任何延迟重要的场景。TF-IDF 加线性模型可以在微秒级回答。用 transformer embedding 一篇文档需要 10-100ms。
- 必须解释预测的系统。查看分类器系数。权重最高的正向词就是原因。

### TF-IDF 失效的地方

语义盲点失效。考虑这两篇文档：

- "The movie was not good at all."
- "The movie was excellent."

一篇是负面影评。一篇是正面影评。它们的 TF-IDF 重叠恰好是 `{the, movie, was}`。bag-of-words 分类器必须死记硬背 `not` 靠近 `good` 会翻转标签。数据足够多时它能学到，但永远不如理解语法的模型自然。

另一个失效：推理时的 out-of-vocabulary 词。一个在 IMDb 影评上训练的 BoW 模型，如果训练里从没出现过 `Zoomer-approved`，就不知道该怎么处理这个 token。Subword embeddings（第 04 课）能处理。TF-IDF 不能。

### 混合方案：TF-IDF 加权 embeddings

2026 年中等数据量分类的务实默认方案：用 TF-IDF 权重作为 word embeddings 上的 attention。

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

你既得到 embeddings 的语义容量，也得到 TF-IDF 对稀有词的强调。分类器在 pooled vector 上训练。在少于约 5 万个标注样本的情感、主题和意图分类中，这通常会超过任一单独方法。

## 交付成果

保存为 `outputs/prompt-vectorization-picker.md`：

```markdown
---
name: vectorization-picker
description: Given a text-classification task, recommend BoW, TF-IDF, embeddings, or a hybrid.
phase: 5
lesson: 02
---

You recommend a text-vectorization strategy. Given a task description, output:

1. Representation (BoW, TF-IDF, transformer embeddings, or a hybrid). Explain why in one sentence.
2. Specific vectorizer configuration. Name the library. Quote the arguments (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`).
3. One failure mode to test before shipping.

Refuse to recommend embeddings when the user has under 500 labeled examples unless they show evidence of semantic failure in a TF-IDF baseline. Refuse to remove stopwords for sentiment analysis (negations carry signal). Flag class imbalance as needing more than a vectorizer change.

Example input: "Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

Example output:

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.
```

## 练习

1. **简单。** 在 L2 归一化后的 TF-IDF 输出上实现 `cosine_similarity(doc_vec_a, doc_vec_b)`。验证相同文档得分为 1.0，词表完全不相交的文档得分为 0.0。
2. **中等。** 给 `bag_of_words` 添加 `n-gram` 支持。参数 `n` 产出 `n`-gram 计数。测试 `n=2` 作用在 `["the", "cat", "sat"]` 上时，会为 `["the cat", "cat sat"]` 产生 bigram 计数。
3. **困难。** 使用 GloVe 100d 向量构建上面的 TF-IDF-weighted-embedding 混合模型（下载一次并缓存）。在 20 Newsgroups 数据集上，对比纯 TF-IDF、纯 mean-pooled embeddings 和混合方案的分类准确率。报告哪个在什么情况下获胜。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| BoW | 词频向量 | 一篇文档中词表词的计数。丢掉顺序。 |
| TF | 词频 | 某个词在文档里的计数，可选择按文档长度归一化。 |
| DF | 文档频率 | 至少包含该词一次的文档数量。 |
| IDF | 逆文档频率 | 平滑后的 `log(N / df)`。降低到处出现的词的权重。 |
| Sparse vector | 大多为零 | 词表通常有 1 万到 10 万个词；任意单篇文档只包含其中很少一部分。 |
| Cosine similarity | 向量夹角 | L2 归一化向量的点积。1 表示相同，0 表示正交。 |

## 延伸阅读

- [scikit-learn — feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — 规范 API 参考，并解释每个旋钮。
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) — 让 TF-IDF 成为十年默认方法的论文。
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — 2026 年视角：旧方法何时获胜，以及为什么。
