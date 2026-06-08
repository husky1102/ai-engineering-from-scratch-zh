# 情感分析

> 经典 NLP 任务。关于经典文本分类你需要知道的大多数东西，都会在这里出现。

**类型：** Build
**语言：** Python
**先修：** 第 5 阶段 · 02（BoW + TF-IDF），第 2 阶段 · 14（朴素贝叶斯）
**时间：** 约 75 分钟

## 要解决的问题

"The food was not great." 是正面还是负面？

情感听起来很简单。评论者说喜欢或不喜欢某个东西。给句子打标签。它之所以成为经典 NLP 任务，是因为每个看起来容易的例子背后都藏着难题。否定会翻转意义。讽刺会把它倒过来。"Not bad at all" 虽然有两个负面编码词，却是正面的。Emoji 携带的信号可能比周围文本更多。领域词汇很重要（音乐评论里的 `tight` 和时尚评论里的 `tight` 不是一回事）。

情感分析是经典 NLP 的工作实验室。如果你理解为什么每个 naive baseline 都有特定失效模式，你就理解了为什么后来每个更丰富的模型会被发明出来。本课会从零构建 Naive Bayes baseline，加入 logistic regression，并点名那些让生产情感分析变成合规级问题的陷阱。

## 核心概念

经典情感分析是两步配方。

1. **表示。** 把文本变成特征向量。BoW、TF-IDF 或 n-grams。
2. **分类。** 在标注样本上拟合线性模型（Naive Bayes、logistic regression、SVM）。

Naive Bayes 是最笨但能工作的模型。假设在给定标签时，每个特征彼此独立。从计数中估计 `P(word | positive)` 和 `P(word | negative)`。推理时，把概率相乘。这个 "naive" 独立假设错得可笑，但结果强得惊人。原因是：面对稀疏文本特征和中等数据量，分类器更关心每个词偏向哪一边，而不是精确的联合概率。

Logistic regression 修复了独立性假设。它为每个特征学习一个权重，包括负权重。`not good` 作为 bigram 特征可以得到负权重。Naive Bayes 对从没见过标签的 bigram 做不到这一点。

## 动手实现

### 第 1 步：一个真实的迷你数据集

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

它有意保持很小。真实工作会使用数万样本（IMDb、SST-2、Yelp polarity）。数学完全相同。

### 第 2 步：从零实现 multinomial Naive Bayes

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

加性平滑（alpha=1.0）就是 Laplace smoothing。没有它，一个从未在某个类别中出现过的词会得到零概率，log 会爆掉。`alpha=0.01` 在实践中很常见。`alpha=1.0` 是教学默认值。

### 第 3 步：从零实现 logistic regression

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

这里 L2 regularization 很重要。文本特征是稀疏的；没有 L2，模型会记住训练样本。先从 `0.01` 开始，再调参。

### 第 4 步：处理否定（失效模式）

考虑 "not good" 和 "not bad"。BoW 分类器看到 `{not, good}` 和 `{not, bad}`，然后根据训练里哪种出现更多来学习。Bigram 分类器看到 `not_good` 和 `not_bad`，把它们当成不同特征来学。通常这就够了。

当你没有 bigrams 时，一个更粗糙但有效的修复是：**negation scoping**。给否定词之后的 token 加 `NOT_` 前缀，直到下一个标点。

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

现在 `good` 和 `NOT_good` 是不同特征。分类器可以给它们相反权重。三行预处理，就能在情感 benchmark 上带来可测量的准确率提升。

### 第 5 步：真正重要的评估指标

如果类别不平衡，单看 accuracy 会误导。真实情感语料通常是 70-80% 正面或 70-80% 负面；一个永远预测多数类的分类器可以拿到 80% accuracy，但毫无价值。以下指标都要报告：

- **Per-class precision and recall。** 每个类别一组。对它们做 macro-average，得到一个尊重类别平衡的单数。
- **Macro-F1（不平衡数据的主指标）。** 每个类别 F1 的平均值，等权重。类别不平衡时，用它替代 accuracy。
- **Weighted-F1（备选）。** 和 macro 类似，但按类别频率加权。当不平衡本身有业务含义时，和 macro-F1 一起报告。
- **Confusion matrix。** 原始计数。信任任何标量指标之前都要检查它；它会揭示模型把哪些类别对混淆了。
- **Per-class error samples。** 每个类别抽 5 个错误预测。读它们。没有任何东西能替代阅读真实错误。

对于严重不平衡的数据（> 95-5 比例），报告 **AUROC** 和 **AUPRC**，而不是 accuracy。AUPRC 对少数类更敏感，而少数类通常正是你关心的东西（垃圾邮件、欺诈、稀有情感）。

**要避免的常见 bug。** 在不平衡数据上报告 micro-F1 而不是 macro-F1，会得到一个看起来很高的数字，因为它被多数类主导。Macro-F1 会强迫你看到少数类表现。

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## 实际使用

scikit-learn 用六行正确完成这件事。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

有三件事要注意。`stop_words=None` 保留否定词。`ngram_range=(1, 2)` 添加 bigrams，让 `not_good` 变成特征。`sublinear_tf=True` 降低重复词的影响。这三个 flag，就是 SST-2 上 75% 准确率 baseline 和 85% 准确率 baseline 的差别。

### 什么时候该用 transformer

- 讽刺检测。经典模型在这里会失败。句号。
- 情感在长评论中途发生转折。
- Aspect-based sentiment。"Camera was great but battery was terrible." 你需要把情感归因到具体 aspect。只能用 transformers 或结构化输出模型。
- 非英语、低资源语言。Multilingual BERT 免费给你一个 zero-shot baseline。

如果你需要上面任何一项，跳到第 7 阶段（transformers deep dive）。否则，Naive Bayes 或 logistic regression 搭配 TF-IDF、bigrams 和否定处理，就是你的 2026 生产 baseline。

### 可复现性陷阱（再次出现）

重新训练情感模型是常规操作。重新评估它们却不是。论文里的 accuracy 数字使用特定 split、特定预处理、特定 tokenizer。如果你没有使用完全相同的 pipeline，就把新模型和某个 baseline 比较，你会得到误导性的 delta。始终在你的 pipeline 上重新生成 baseline，而不是引用论文数字。

## 交付成果

保存为 `outputs/prompt-sentiment-baseline.md`：

```markdown
---
name: sentiment-baseline
description: Design a sentiment analysis baseline for a new dataset.
phase: 5
lesson: 05
---

Given a dataset description (domain, language, size, label granularity, latency budget), you output:

1. Feature extraction recipe. Specify tokenizer, n-gram range, stopword policy (usually keep), negation handling (scoped prefix or bigrams).
2. Classifier. Naive Bayes for baseline, logistic regression for production, transformer only if the domain needs sarcasm / aspects / cross-lingual.
3. Evaluation plan. Report precision, recall, F1, confusion matrix, and per-class error samples (not just scalars).
4. One failure mode to monitor post-deployment. Domain drift and sarcasm are the top two.

Refuse to recommend dropping stopwords for sentiment tasks. Refuse to report accuracy as the sole metric when classes are imbalanced (e.g., 90% positive). Flag subword-rich languages as needing FastText or transformer embeddings over word-level TF-IDF.
```

## 练习

1. **简单。** 把 `apply_negation` 作为 scikit-learn pipeline 中的预处理步骤，并在一个小型情感数据集上衡量 F1 delta。
2. **中等。** 实现 class-weighted logistic regression（给 scikit-learn 传 `class_weight="balanced"`，或自己推导梯度）。在一个合成的 90-10 类别不平衡上衡量效果。
3. **困难。** 通过在情感模型的 residuals 上训练第二个分类器，构建讽刺检测器。记录你的实验设置。当准确率低于 chance 时警告读者（2 类讽刺检测的 chance-level 约为 50%，大多数第一次尝试都会落在那里）。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Polarity | 正面或负面 | 二分类标签；有时扩展到 neutral 或细粒度（5 星）。 |
| Aspect-based sentiment | 每个 aspect 的 polarity | 把情感归因到文本中提到的具体实体或属性。 |
| Negation scoping | 翻转附近 token | 在 "not" 之后给 token 加 `NOT_` 前缀，直到标点。 |
| Laplace smoothing | 给计数加 1 | 防止 Naive Bayes 中出现零概率特征。 |
| L2 regularization | 收缩权重 | 在 loss 中加入 `lambda * sum(w^2)`。对稀疏文本特征必不可少。 |

## 延伸阅读

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — 奠基综述。很长，但前四节覆盖了所有经典内容。
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) — 展示 bigrams + Naive Bayes 在短文本上很难被击败的论文。
- [scikit-learn text feature extraction docs](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — `CountVectorizer`、`TfidfVectorizer` 和你会调的每个旋钮的参考。
