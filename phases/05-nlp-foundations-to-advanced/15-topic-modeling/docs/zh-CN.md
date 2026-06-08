# 主题建模：LDA 与 BERTopic

> LDA：文档是主题的混合，主题是词上的分布。BERTopic：文档在嵌入空间中聚类，聚类就是主题。目标相同，分解方式不同。

**类型：** Learn
**语言：** Python
**先修：** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word2Vec)
**时间：** ~45 minutes

## 要解决的问题

你有 10,000 张客户支持工单、50,000 篇新闻文章，或 200,000 条推文。你需要在不逐篇阅读的情况下知道这个集合在谈什么。你没有带标签的类别。你甚至不知道一共有多少类别。

主题建模用无监督方式回答这个问题。给它一个语料库，它会返回一小组连贯主题，并为每篇文档给出这些主题上的分布。

两类算法家族占主导。LDA (2003) 将每篇文档看作潜在主题的混合，并将每个主题看作词上的分布。推断是 Bayesian 的。只要你需要混合成员关系的主题分配，以及可解释的词级概率分布，它今天仍然在生产中使用。

BERTopic (2020) 用 BERT 编码文档，用 UMAP 降维，用 HDBSCAN 聚类，再通过 class-based TF-IDF 抽取主题词。它在短文本、社交媒体，以及任何语义相似性比词重叠更重要的场景中表现更好。一篇文档只得到一个主题，这对长篇内容是一个限制。

本课会为两者建立直觉，并说明面对给定语料库时该选哪一个。

## 核心概念

![LDA mixture model vs BERTopic clustering](../assets/topic-modeling.svg)

**LDA 生成故事。** 每个主题是词上的分布。每篇文档是主题的混合。要在一篇文档中生成一个词，先从该文档的主题混合中采样一个主题，再从该主题的词分布中采样一个词。推断则反过来：给定观测到的词，推断每篇文档的主题分布，以及每个主题的词分布。Collapsed Gibbs sampling 或 variational Bayes 会完成这些数学。

LDA 的关键输出：

- `doc_topic`：矩阵 `(n_docs, n_topics)`，每一行和为 1（文档的主题混合）。
- `topic_word`：矩阵 `(n_topics, vocab_size)`，每一行和为 1（主题的词分布）。

**BERTopic pipeline。**

1. 用 sentence transformer（例如 `all-MiniLM-L6-v2`）编码每篇文档。得到 384 维向量。
2. 用 UMAP 将维度降到约 5 维。BERT 嵌入的维度太高，不适合直接聚类。
3. 用 HDBSCAN 聚类。它是基于密度的，会产生大小可变的聚类和一个 "outlier" 标签。
4. 对每个聚类，在该聚类的文档上计算 class-based TF-IDF，抽取 top words。

输出是每篇文档一个主题（外加 -1 outlier label）。也可以选择通过 HDBSCAN 的 probability vector 得到 soft membership。

## 动手实现

### Step 1: LDA via scikit-learn

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

注意：移除了 stopwords，`min_df` 和 `max_df` 会过滤稀有项与过于普遍的项，使用 CountVectorizer（不是 TfidfVectorizer），因为 LDA 期望 raw counts。

### Step 2: BERTopic (production)

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

对 `Topic != -1` 的过滤会丢弃 BERTopic 的 outlier bucket（HDBSCAN 无法聚类的文档）。`min_topic_size` 控制 HDBSCAN 的最小聚类大小；BERTopic 库默认值是 10。这个示例为了适配本课规模，显式设为 15。对于超过 10,000 篇文档的语料库，把它增大到 50 或 100。

### Step 3: evaluation

两种方法都会输出主题词。问题是这些词是否连贯。

- **Topic coherence (c_v)。** 在滑动窗口上下文中结合 top-word pairs 的 NPMI（normalized pointwise mutual information），将分数聚合成 topic vectors，并用 cosine similarity 比较这些向量。越高越好。使用 `gensim.models.CoherenceModel` 并设置 `coherence="c_v"`。
- **Topic diversity。** 所有主题的 top words 中唯一词的占比。越高越好（主题之间不重叠）。
- **Qualitative inspection。** 阅读每个主题的 top words。它们是否命名了真实事物？人的判断仍然是最后一道防线。

## When to pick which

| Situation | Pick |
|-----------|------|
| 短文本（推文、评论、标题） | BERTopic |
| 有主题混合的长文档 | LDA |
| 没有 GPU / 计算资源有限 | LDA or NMF |
| 需要文档级多主题分布 | LDA |
| 用 LLM integration 做主题标签 | BERTopic (direct support) |
| 资源受限的边缘部署 | LDA |
| 最大化语义连贯性 | BERTopic |

最大的实践考量是文档长度。BERT embeddings 会截断；LDA counts 可以处理任意长度。对于长于 embedding model context 的文档，要么 chunk + aggregate，要么使用 LDA。

## 实际使用

2026 stack：

- **BERTopic。** 短文本以及任何语义重要的场景的默认选择。
- **`gensim.models.LdaModel`。** 生产中的经典 LDA，成熟且经受过实战检验。
- **`sklearn.decomposition.LatentDirichletAllocation`。** 易于实验的 LDA。
- **NMF。** Non-negative matrix factorization。LDA 的快速替代方案，在短文本上质量相近。
- **Top2Vec。** 设计与 BERTopic 类似。社区更小，但在某些 benchmark 上表现不错。
- **FASTopic。** 更新，在超大语料库上比 BERTopic 更快。
- **LLM-based labeling。** 先运行任意 clustering，再 prompt 一个模型为每个 cluster 命名。

## 交付成果

保存为 `outputs/skill-topic-picker.md`：

```markdown
---
name: topic-picker
description: Pick LDA or BERTopic for a corpus. Specify library, knobs, evaluation.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

Given a corpus description (document count, avg length, domain, language, compute budget), output:

1. Algorithm. LDA / NMF / BERTopic / Top2Vec / FASTopic. One-sentence reason.
2. Configuration. Number of topics: `recommended = max(5, round(sqrt(n_docs)))`, clamped to 200 for corpora under 40,000 docs; permit >200 only when the corpus is genuinely large (>40k) and note the increased compute cost. `min_df` / `max_df` filters and embedding model for neural approaches also belong here.
3. Evaluation. Topic coherence (c_v) via `gensim.models.CoherenceModel`, topic diversity, and a 20-sample human read.
4. Failure mode to probe. For LDA, "junk topics" absorbing stopwords and frequent terms. For BERTopic, the -1 outlier cluster swallowing ambiguous documents.

Refuse BERTopic on documents longer than the embedding model's context window without a chunking strategy. Refuse LDA on very short text (tweets, reviews under 10 tokens) as coherence collapses. Flag any n_topics choice below 5 as likely wrong; flag >200 on corpora under 40k docs as likely over-splitting.
```

## 练习

1. **Easy.** 在 20 Newsgroups dataset 上用 5 个主题拟合 LDA。打印每个主题的 top 10 words。手动标注每个主题。算法找到真实类别了吗？
2. **Medium.** 在同一个 20 Newsgroups 子集上拟合 BERTopic。比较它与 LDA 找到的主题数量、top words 和定性连贯性。哪一种更干净地浮现真实类别？
3. **Hard.** 在你的语料库上同时计算 LDA 和 BERTopic 的 c_v coherence。分别用 5、10、20、50 个主题运行。绘制 coherence vs topic count。报告哪种方法在不同 topic counts 上更稳定。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Topic | 语料库谈论的东西 | 词上的概率分布（LDA），或相似文档的一个聚类（BERTopic）。 |
| Mixed membership | 文档属于多个主题 | LDA 为每篇文档分配一个覆盖所有主题的分布。 |
| UMAP | 降维 | 保留局部结构的 manifold learning；用于 BERTopic。 |
| HDBSCAN | 密度聚类 | 找到大小可变的聚类；为 outliers 生成 "noise" label (-1)。 |
| c_v coherence | 主题质量指标 | 在滑动窗口内计算 topic top words 的平均 pointwise mutual information。 |

## 延伸阅读

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — LDA 论文。
- [Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) — BERTopic 论文。
- [Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — 引入 c_v 等指标的论文。
- [BERTopic documentation](https://maartengr.github.io/BERTopic/) — 生产参考。示例非常好。
