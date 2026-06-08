# 朴素贝叶斯

> “朴素”的假设是错的，但它照样有效。这正是它的美。

**类型:** 构建
**语言:** Python
**先修:** 第 2 阶段，Lessons 01-07（分类、贝叶斯定理）
**时间:** ~75 分钟

## 学习目标

- 从零实现带 Laplace smoothing 的 Multinomial Naive Bayes，用于文本分类
- 解释为什么朴素独立性假设在数学上是错的，却能在实践中产生正确的类别排序
- 比较 Multinomial、Bernoulli 和 Gaussian Naive Bayes 三种变体，并根据给定特征类型选择合适版本
- 在高维稀疏数据上比较 Naive Bayes 与 logistic regression，并解释其中起作用的 bias-variance tradeoff

## 要解决的问题

你需要给文本分类。把邮件分成 spam 或 not-spam。把客户评论分成正面或负面。把支持工单分到不同类别。你有成千上万个特征（每个词一个特征），但训练数据有限。

大多数分类器会在这里卡住。Logistic regression 需要足够多的样本，才能可靠估计成千上万个权重。Decision trees 一次只按一个词划分，很容易严重过拟合。KNN 在 10,000 维空间里没有意义，因为每个点到其他点的距离几乎都一样远。

Naive Bayes 能处理这种场景。它做了一个数学上错误的假设（给定类别后，每个特征都与其他特征独立），但在文本分类上仍然能超过“更聪明”的模型，尤其是在训练集较小时。它只需单次遍历数据就能训练。它可以扩展到数百万个特征。它能给出概率估计（不过由于独立性假设，这些概率通常校准得不好）。

理解为什么一个错误假设能带来好预测，会教会你机器学习中一个根本事实：最好的模型不是最正确的模型，而是对你的数据具有最佳 bias-variance tradeoff 的模型。

## 核心概念

### 贝叶斯定理（快速回顾）

贝叶斯定理会翻转条件概率：

```text
P(class | features) = P(features | class) * P(class) / P(features)
```

我们想要 `P(class | features)`，也就是给定文档中的词后，该文档属于某个类别的概率。它可以由下面几项计算出来：
- `P(features | class)`：在该类别文档中看到这些词的似然
- `P(class)`：类别的先验概率（spam 总体上有多常见？）
- `P(features)`：证据项，对所有类别相同，所以比较类别时可以忽略

`P(class | features)` 最高的类别获胜。

### 朴素独立性假设

精确计算 `P(features | class)` 需要估计所有特征共同出现的联合概率。假设词表有 10,000 个词，你需要估计 2^10,000 种可能组合上的分布。这不可能。

朴素假设是：给定类别后，每个特征都是条件独立的。

```text
P(w1, w2, ..., wn | class) = P(w1 | class) * P(w2 | class) * ... * P(wn | class)
```

你不再估计一个不可能的联合分布，而是估计 n 个简单的逐特征分布。每个分布只需要一个计数。

这个假设显然是错的。在任何文档里，"machine" 和 "learning" 都不是独立的。但分类器不需要正确的概率估计。它需要正确的排序，也就是哪个类别概率最高。独立性假设会引入系统性误差，但这些误差会以相似方式影响所有类别，所以排序仍然能保持正确。

### 为什么它仍然有效

有三个原因：

1. **排序优先于校准。** 分类只需要排名第一的类别正确。即使 P(spam) = 0.99999 而真实概率是 0.7，分类器仍然会正确选择 spam。我们不需要正确概率。我们需要正确赢家。

2. **高偏差、低方差。** 独立性假设是一个很强的先验。它强力约束模型，从而防止过拟合。在训练数据有限时，一个略微错误但稳定的模型，会胜过一个理论上正确但极不稳定的模型。这就是 bias-variance tradeoff 在发挥作用。

3. **特征冗余会相互抵消。** 相关特征提供冗余证据。分类器会重复计算这些证据，但它也会为正确类别重复计算。如果 "machine" 和 "learning" 总是一起出现，它们都会为 "tech" 类别提供证据。NB 会把它们算两次，但它是为正确类别算了两次。

第四个实际原因是：Naive Bayes 极快。训练只是单次遍历数据并统计频率。预测是一次矩阵乘法。你可以在几秒钟内用一百万篇文档训练。这个速度意味着你可以更快迭代、尝试更多特征集，并且比使用较慢模型时运行更多实验。

### 数学逐步展开

我们跟踪一个具体例子。假设有两个类别：spam 和 not-spam。词表有三个词："free"、"money"、"meeting"。

训练数据：
- Spam 邮件提到 "free" 80 次、"money" 60 次、"meeting" 10 次（总共 150 个词）
- Not-spam 邮件提到 "free" 5 次、"money" 10 次、"meeting" 100 次（总共 115 个词）
- 40% 的邮件是 spam，60% 是 not-spam

使用 Laplace smoothing（alpha=1）：

```text
P(free | spam)    = (80 + 1) / (150 + 3) = 81/153 = 0.529
P(money | spam)   = (60 + 1) / (150 + 3) = 61/153 = 0.399
P(meeting | spam) = (10 + 1) / (150 + 3) = 11/153 = 0.072

P(free | not-spam)    = (5 + 1) / (115 + 3) = 6/118 = 0.051
P(money | not-spam)   = (10 + 1) / (115 + 3) = 11/118 = 0.093
P(meeting | not-spam) = (100 + 1) / (115 + 3) = 101/118 = 0.856
```

新邮件包含："free"（2 次）、"money"（1 次）、"meeting"（0 次）。

```text
log P(spam | email) = log(0.4) + 2*log(0.529) + 1*log(0.399) + 0*log(0.072)
                    = -0.916 + 2*(-0.637) + (-0.919) + 0
                    = -3.109

log P(not-spam | email) = log(0.6) + 2*log(0.051) + 1*log(0.093) + 0*log(0.856)
                        = -0.511 + 2*(-2.976) + (-2.375) + 0
                        = -8.838
```

Spam 以很大优势胜出。"free" 出现两次是 spam 的强证据。注意，"meeting" 没有出现会给两个 log 求和都贡献零（0 * log(P)）。在 Multinomial NB 中，未出现的词没有影响。显式建模词缺失的是 Bernoulli NB。

### 三种变体

Naive Bayes 有三种常见形式。每一种都用不同方式建模 `P(feature | class)`。

#### Multinomial Naive Bayes

把每个特征建模为计数。最适合特征是词频或 TF-IDF 值的文本数据。

```text
P(word_i | class) = (count of word_i in class + alpha) / (total words in class + alpha * vocab_size)
```

`alpha` 是 Laplace smoothing（下面会解释）。这个变体是文本分类的主力。

#### Gaussian Naive Bayes

把每个特征建模为正态分布。最适合连续特征。

```text
P(x_i | class) = (1 / sqrt(2 * pi * var)) * exp(-(x_i - mean)^2 / (2 * var))
```

每个类别都会为每个特征拥有自己的均值和方差。当特征在每个类别内部确实遵循钟形曲线时，这种方法效果很好。

#### Bernoulli Naive Bayes

把每个特征建模为二元变量（出现或缺失）。最适合短文本或二元特征向量。

```text
P(word_i | class) = (docs in class containing word_i + alpha) / (total docs in class + 2 * alpha)
```

不同于 Multinomial，Bernoulli 会显式惩罚某个词的缺失。如果 "free" 通常出现在 spam 中，但这封邮件里没有出现，Bernoulli 会把它算作反对 spam 的证据。

### 何时使用每种变体

| 变体 | 特征类型 | 最适合 | 示例 |
|---------|-------------|----------|---------|
| Multinomial | 计数或频率 | 文本分类、bag-of-words | 邮件 spam、主题分类 |
| Gaussian | 连续值 | 具有近似正态特征的表格数据 | Iris 分类、传感器数据 |
| Bernoulli | 二元（0/1） | 短文本、二元特征向量 | SMS spam、出现/缺失特征 |

### Laplace Smoothing

如果某个词出现在测试数据中，但在训练数据的某个特定类别里从未出现过，会发生什么？

不做 smoothing 时：`P(word | class) = 0/N = 0`。一个零被乘进整个乘积，会让 `P(class | features) = 0`，不管其他证据有多强。单个未见过的词会毁掉整个预测，无论有多少其他证据支持它。

Laplace smoothing 会给每个特征计数加一个小计数 `alpha`（通常为 1）：

```text
P(word_i | class) = (count(word_i, class) + alpha) / (total_words_in_class + alpha * vocab_size)
```

当 alpha=1 时，每个词至少都有一个很小的概率。测试邮件中出现 "discombobulate" 不再会杀死 spam 概率。Smoothing 有一个 Bayesian 解释：它等价于给词分布放置一个均匀 Dirichlet prior。

更高的 alpha 意味着更强的 smoothing（分布更均匀）。更低的 alpha 意味着模型更信任数据。Alpha 是一个需要调优的 hyperparameter。

alpha 的影响：

| Alpha | 影响 | 何时使用 |
|-------|--------|-------------|
| 0.001 | 几乎不 smoothing，信任数据 | 训练集很大，预计没有未见特征 |
| 0.1 | 轻度 smoothing | 训练集较大 |
| 1.0 | 标准 Laplace smoothing | 默认起点 |
| 10.0 | 重度 smoothing，压平分布 | 训练集很小，预计有很多未见特征 |

### Log-Space Computation

把数百个概率（每个都小于 1）相乘会导致 floating-point underflow。即使真实值是一个很小的正数，乘积在浮点数中也会变成零。

解决方法：在 log space 中工作。不要相乘概率，而是相加它们的对数：

```text
log P(class | x1, x2, ..., xn) = log P(class) + sum_i log P(xi | class)
```

这会把预测变成一个 dot product：

```text
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

矩阵乘法。这就是 Naive Bayes 预测如此快速的原因：它和单层线性模型是同一种操作。

### Naive Bayes vs Logistic Regression

二者都是用于文本的线性分类器。区别在于它们建模的对象。

| 方面 | Naive Bayes | Logistic Regression |
|--------|------------|-------------------|
| 类型 | Generative（建模 P(X\|Y)） | Discriminative（建模 P(Y\|X)） |
| 训练 | 统计频率 | 优化损失函数 |
| 小数据 | 更好（强先验有帮助） | 更差（样本不足以估计权重） |
| 大数据 | 更差（错误假设会伤害表现） | 更好（边界更灵活） |
| 特征 | 假设独立 | 处理相关性 |
| 速度 | 单次遍历，非常快 | 迭代优化 |
| 校准 | 概率差 | 概率更好 |

经验法则：先从 Naive Bayes 开始。如果你有足够数据且 NB 达到平台期，再切换到 logistic regression。

### 分类流水线

```mermaid
flowchart LR
    A[Raw Text] --> B[Tokenize]
    B --> C[Build Vocabulary]
    C --> D[Count Word Frequencies]
    D --> E[Apply Smoothing]
    E --> F[Compute Log Probabilities]
    F --> G[Predict: argmax P class given words]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

在实践中，我们在 log space 中工作，以避免 floating-point underflow。我们不相乘很多很小的概率，而是相加它们的对数：

```text
log P(class | features) = log P(class) + sum_i log P(feature_i | class)
```

## 动手实现

`code/naive_bayes.py` 中的代码从零实现了 MultinomialNB 和 GaussianNB。

### MultinomialNB

从零实现的流程：

1. **fit(X, y)**：对每个类别，统计每个特征的频率。加入 Laplace smoothing。计算 log probabilities。存储 class priors（类别频率的 log）。

2. **predict_log_proba(X)**：对每个样本，为所有类别计算 log P(class) + sum of log P(feature_i | class)。这是一次矩阵乘法：X @ log_probs.T + log_priors。

3. **predict(X)**：返回 log probability 最高的类别。

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```

关键洞见：完成 fitting 之后，prediction 只是矩阵乘法加上一个 bias。这就是 Naive Bayes 如此快速的原因。

### GaussianNB

对于连续特征，我们为每个类别的每个特征估计均值和方差：

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```

预测会使用每个特征的 Gaussian PDF，并跨特征相乘（在 log space 中相加）。

### Demo：文本分类

代码会生成合成 bag-of-words 数据，模拟两个类别（tech articles 与 sports articles）。每个类别都有不同的词频分布。MultinomialNB 使用词计数对它们分类。

合成数据的工作方式是：我们创建 200 个“词”（特征列）。Words 0-39 在 tech articles 中频率高，在 sports 中频率低。Words 80-119 在 sports 中频率高，在 tech 中频率低。Words 40-79 在两类中都是中等频率。这样就构造了一个现实场景：一些词是强类别指示器，另一些是噪声。

### Demo：连续特征

代码会生成类似 Iris 的数据（3 个类别、4 个特征、Gaussian clusters）。GaussianNB 使用每个类别的均值和方差进行分类。每个类别都有不同中心（mean vector）和不同扩散程度（variance），模拟现实世界中不同类别的测量值会系统性不同的情况。

代码还展示：
- **Smoothing comparison：** 用不同 alpha 值训练 MultinomialNB，以展示 smoothing 强度对 accuracy 的影响。
- **Training size experiment：** NB accuracy 如何随着训练数据从 20 个样本增长到 1600 个样本而提升。即使用很少样本，NB 也能达到不错 accuracy；这是它的主要优势。
- **Confusion matrix：** 每个类别的 precision、recall 和 F1 score，用来展示 NB 在哪里犯错。

### 预测速度

Naive Bayes 预测是一次矩阵乘法。对于 n 个样本、d 个特征和 k 个类别：
- MultinomialNB：一次矩阵乘法 (n x d) @ (d x k) = O(n * d * k)
- GaussianNB：n * k 次 Gaussian PDF 计算，每次跨 d 个特征 = O(n * d * k)

二者在每个维度上都是线性的。对比 KNN（需要计算到所有训练点的距离）或带 RBF kernel 的 SVM（需要对所有 support vectors 做 kernel evaluation）。NB 的预测速度快几个数量级。

## 实际使用

在 sklearn 中，这两个变体都只需一行即可使用：

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test):.3f}")
```

用 sklearn 做文本分类：

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)
```

`naive_bayes.py` 中的代码会在同一份数据上比较 from-scratch implementations 与 sklearn，以验证正确性。

### TF-IDF 与 Naive Bayes

原始词计数会让每次出现的每个词获得相同权重。但像 "the" 和 "is" 这样的常见词会频繁出现在每个类别中，它们不携带信息。TF-IDF（Term Frequency - Inverse Document Frequency）会降低常见词权重，提高罕见且有区分度的词的权重。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

TF-IDF 值是非负的，所以可以和 MultinomialNB 配合使用。TF-IDF + MultinomialNB 的组合是文本分类中最强的 baseline 之一。在少于 10,000 个训练样本的数据集上，它经常击败更复杂的模型。

### 短文本中的 BernoulliNB

对于短文本（tweets、SMS、chat messages），BernoulliNB 可以胜过 MultinomialNB。短文本的词计数很低，因此 MultinomialNB 依赖的频率信息噪声很大。BernoulliNB 只关心出现或缺失，在短文本中更可靠。

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

CountVectorizer 中的 `binary=True` 标志会把所有计数转换成 0/1。如果没有它，BernoulliNB 仍然能运行，但它看到的是计数，而不是它原本设计要处理的数据。

### 校准 NB 概率

NB 概率校准很差。当 NB 说 P(spam) = 0.95 时，真实概率可能是 0.7。如果你需要可靠的概率估计（例如用于设置阈值，或与其他模型组合），使用 sklearn 的 CalibratedClassifierCV：

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

这会使用 cross-validation，在 NB 的 raw scores 之上拟合一个 logistic regression。得到的概率会更接近真实类别频率。

### 常见坑

1. **负特征值。** MultinomialNB 要求非负特征。如果你有负值（比如某些设置下的 TF-IDF 或标准化后的特征），改用 GaussianNB，或者把特征平移到正数。

2. **零方差特征。** GaussianNB 会除以方差。如果某个类别中的某个特征方差为零（所有值都相同），概率计算会崩掉。代码会给所有方差加一个很小的 smoothing term（1e-9）来防止这种情况。

3. **类别不平衡。** 如果 99% 的邮件都是 not-spam，那么先验 P(not-spam) = 0.99 会非常强，以至于压过似然证据。你可以手动设置 class priors，或在 sklearn 中使用 class_prior 参数。

4. **特征缩放。** MultinomialNB 不需要 scaling（它处理计数）。GaussianNB 也不需要 scaling（它估计每个特征的统计量）。相比对特征尺度敏感的 logistic regression 和 SVM，这是一个优势。

## 交付成果

本课产出：
- `outputs/skill-naive-bayes-chooser.md`：用于选择正确 NB 变体的 decision skill
- `code/naive_bayes.py`：从零实现 MultinomialNB 和 GaussianNB，并与 sklearn 比较

### Naive Bayes 何时失败

当独立性假设造成错误排序（而不仅是错误概率）时，NB 会失败。这会发生在：

1. **强特征交互。** 如果类别取决于两个特征的组合，而不是任一单独特征（类似 XOR 的模式），NB 会完全错过。每个单独特征都不提供证据，NB 也无法非线性地组合它们。

2. **高度相关但证据方向相反的特征。** 如果特征 A 指向 "spam"，特征 B 指向 "not-spam"，但 A 和 B 完全相关（现实中它们总是一致），NB 会看到并不存在的冲突证据。

3. **非常大的训练集。** 数据足够多时，像 logistic regression 这样的 discriminative models 会学到真实决策边界并超过 NB。在小数据中有帮助的独立性假设，此时反而会拖累模型。

在实践中，这些失败模式对文本分类并不常见。文本特征数量多、单个特征弱，独立性假设的误差往往会相互抵消。对于少量强相关特征组成的表格数据，优先考虑 logistic regression 或 tree-based models。

## 练习

1. **Smoothing experiment。** 在文本数据上用 alpha 值 0.01、0.1、1.0、10.0 和 100.0 训练 MultinomialNB。绘制 accuracy vs alpha。性能在哪里达到峰值？为什么非常高的 alpha 会伤害表现？

2. **Feature independence test。** 取一个真实文本数据集。挑两个明显相关的词（"machine" 和 "learning"）。计算 P(word1 | class) * P(word2 | class)，并与 P(word1 AND word2 | class) 比较。独立性假设错得有多厉害？它会影响分类准确率吗？

3. **Bernoulli implementation。** 用 BernoulliNB class 扩展代码。把 bag-of-words 转成二元（出现/缺失），并在文本数据上和 MultinomialNB 比较 accuracy。Bernoulli 什么时候会赢？

4. **NB vs Logistic Regression。** 在文本数据上训练二者。从 100 个训练样本开始，增加到 10,000 个。绘制二者的 accuracy vs training set size。Logistic Regression 在什么时候超过 Naive Bayes？

5. **Spam filter。** 构建一个完整 spam 分类器：tokenize 原始邮件文本，build vocabulary，create bag-of-words features，train MultinomialNB，并用 precision 和 recall 评估（不只是 accuracy，为什么？）。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| Naive Bayes | “简单概率分类器” | 使用贝叶斯定理的分类器，并假设给定类别后特征条件独立 |
| Conditional independence | “特征之间不会相互影响” | P(A, B \| C) = P(A \| C) * P(B \| C)，也就是在知道 C 后，知道 B 不会告诉你关于 A 的新信息 |
| Laplace smoothing | “加一平滑” | 给每个特征增加一个小计数，防止零概率主导预测 |
| Prior | “看到数据前的信念” | P(class)，也就是观察任何特征前每个类别的概率 |
| Likelihood | “数据有多吻合” | P(features \| class)，也就是已知类别时观察到这些特征的概率 |
| Posterior | “看到数据后的信念” | P(class \| features)，也就是观察特征后类别的更新概率 |
| Generative model | “建模数据如何生成” | 学习 P(X \| Y) 和 P(Y)，然后用贝叶斯定理得到 P(Y \| X) 的模型 |
| Discriminative model | “建模决策边界” | 不建模 X 如何生成，而是直接学习 P(Y \| X) 的模型 |
| Log probability | “避免 underflow” | 使用 log P 而不是 P，防止许多小数相乘后在浮点数中变成零 |

## 延伸阅读

- [scikit-learn Naive Bayes docs](https://scikit-learn.org/stable/modules/naive_bayes.html)：包含三种变体及其数学细节
- [McCallum and Nigam, A Comparison of Event Models for Naive Bayes Text Classification (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf)：Multinomial 与 Bernoulli 在文本上的经典比较
- [Rennie et al., Tackling the Poor Assumptions of Naive Bayes Text Classifiers (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf)：面向文本的 NB 改进
- [Ng and Jordan, On Discriminative vs. Generative Classifiers (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf)：证明在数据较少时 NB 比 LR 收敛更快
