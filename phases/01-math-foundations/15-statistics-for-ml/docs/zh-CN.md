# 面向机器学习的统计学

> 统计学让你知道模型是真的有效，还是只是运气好。

**类型：** 构建
**语言：** Python
**先修：** 第 1 阶段，第 06 课（概率与分布）、第 07 课（贝叶斯定理）
**时间：** ~120 分钟

## 学习目标

- 从零计算描述性统计量、Pearson/Spearman 相关系数和协方差矩阵
- 执行假设检验（t 检验、卡方检验），并正确解读 p 值和置信区间
- 使用 bootstrap 重采样，在不做分布假设的情况下为任意指标构造置信区间
- 使用效应量度量区分统计显著性和实际显著性

## 要解决的问题

你训练了两个模型。模型 A 在测试集上得分 0.87。模型 B 得分 0.89。你部署了模型 B。三周后，生产指标比以前更差。发生了什么？

模型 B 并没有真正胜过模型 A。0.02 的差异只是噪声。你的测试集太小，或者方差太高，或者两者都有。你把随机性包装成改进并发布了出去。

这种事一直在发生。Kaggle 排行榜洗牌。论文无法复现。A/B 测试只用几百个样本就宣布胜者。根因总是一样：有人跳过了统计学。

统计学给你工具，用来区分信号和噪声。它会告诉你一个差异什么时候是真实的、你应该有多大把握，以及在信任结果之前你需要多少数据。每条 ML 流水线、每次模型比较、每个实验都需要统计学。没有它，你只是在猜。

## 核心概念

### 描述性统计：总结你的数据

在建模任何东西之前，你需要知道数据长什么样。描述性统计把一个数据集压缩成几个数字，用来捕捉它的形状。

**集中趋势的度量** 回答“中间在哪里？”

```text
Mean:   sum of all values / count
        mu = (1/n) * sum(x_i)

Median: middle value when sorted
        Robust to outliers. If you have [1, 2, 3, 4, 1000], the mean is 202
        but the median is 3.

Mode:   most frequent value
        Useful for categorical data. For continuous data, rarely informative.
```

均值是平衡点。中位数是半程标记。当二者分离时，你的分布就是偏斜的。收入分布通常是 mean >> median（亿万富翁造成右偏）。训练期间的损失分布常常是 mean << median（容易样本造成左偏）。

**离散程度的度量** 回答“数据有多分散？”

```text
Variance:   average squared deviation from the mean
            sigma^2 = (1/n) * sum((x_i - mu)^2)

Standard deviation:  square root of variance
                     sigma = sqrt(sigma^2)
                     Same units as the data, so more interpretable.

Range:      max - min
            Sensitive to outliers. Almost never useful alone.

IQR:        Q3 - Q1 (interquartile range)
            The range of the middle 50% of the data.
            Robust to outliers. Used for box plots and outlier detection.
```

**百分位数** 把排序后的数据分成 100 个相等部分。第 25 百分位数（Q1）表示有 25% 的值落在这个点以下。第 50 百分位数就是中位数。第 75 百分位数是 Q3。

```text
For latency monitoring:
  P50 = median latency        (typical user experience)
  P95 = 95th percentile       (bad but not worst case)
  P99 = 99th percentile       (tail latency, often 10x the median)
```

在 ML 中，你会关心推理延迟、预测置信度分布和误差分布的百分位数。一个平均误差很低、但 P99 误差糟糕的模型，对于安全关键应用可能毫无用处。

**样本统计量与总体统计量。** 从样本计算方差时，要除以 (n-1)，而不是 n。这是贝塞尔校正。它补偿了这样一个事实：你的样本均值并不是真实的总体均值。如果分母用 n，你会系统性低估真实方差。用 (n-1) 时，估计量就是无偏的。

```text
Population variance: sigma^2 = (1/N) * sum((x_i - mu)^2)
Sample variance:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
```

实践中：如果 n 很大（数千个样本），差异可以忽略。如果 n 很小（几十个样本），这就很重要。

### 相关性：变量如何一起变化

相关性度量两个变量之间线性关系的强度和方向。

**Pearson 相关系数** 度量线性关联：

```text
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  perfect positive linear relationship
r = -1:  perfect negative linear relationship
r =  0:  no linear relationship (but there might be a nonlinear one!)

Range: [-1, 1]
```

Pearson 假设关系是线性的，并且两个变量大致服从正态分布。它对离群值敏感。一个极端点就可能把 r 从 0.1 拖到 0.9。

**Spearman 秩相关** 度量单调关联：

```text
1. Replace each value with its rank (1, 2, 3, ...)
2. Compute Pearson correlation on the ranks

Spearman catches any monotonic relationship, not just linear.
If y = x^3, Pearson gives r < 1 but Spearman gives rho = 1.
```

**什么时候用哪一个：**

```text
Pearson:    Both variables are continuous and roughly normal.
            You care about the linear relationship specifically.
            No extreme outliers.

Spearman:   Ordinal data (rankings, ratings).
            Data is not normally distributed.
            You suspect a monotonic but not linear relationship.
            Outliers are present.
```

**黄金法则：** 相关性不蕴含因果。冰淇淋销量和溺水死亡人数相关，是因为二者都会在夏天增加。你的模型准确率和参数数量相关，但增加参数并不会自动提高准确率（参见：过拟合）。

### 协方差矩阵

两个变量之间的协方差度量它们如何共同变化：

```text
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X and Y tend to increase together
Cov(X, Y) < 0:  when X increases, Y tends to decrease
Cov(X, Y) = 0:  no linear co-movement
```

对于 d 个特征，协方差矩阵 C 是一个 d x d 矩阵，其中 C[i][j] = Cov(feature_i, feature_j)。对角线条目 C[i][i] 是每个特征的方差。

```text
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Properties:
  - Symmetric: C[i][j] = C[j][i]
  - Positive semi-definite: all eigenvalues >= 0
  - Diagonal = variances
  - Off-diagonal = covariances
```

**与 PCA 的联系。** PCA 会对协方差矩阵做特征分解。特征向量是主成分（最大方差方向）。特征值告诉你每个成分捕获了多少方差。这正是第 10 课讲过的内容，但现在你能看见为什么协方差矩阵是正确的分解对象：它编码了数据中所有成对线性关系。

**与相关性的联系。** 相关矩阵是标准化变量的协方差矩阵（每个变量都除以自己的标准差）。相关性会归一化协方差，让所有值都落在 [-1, 1] 中。

### 假设检验

假设检验是在不确定性下做决策的框架。你从一个主张开始，收集数据，然后判断数据是否与这个主张一致。

**设置：**

```text
Null hypothesis (H0):        the default assumption, usually "no effect"
Alternative hypothesis (H1): what you are trying to show

Example:
  H0: Model A and Model B have the same accuracy
  H1: Model B has higher accuracy than Model A
```

**p 值** 是在 H0 为真时，观察到像你现在这样极端的数据的概率。它不是 H0 为真的概率。这是统计学中最常见的误解。

```text
p-value = P(data this extreme | H0 is true)

If p-value < alpha (typically 0.05):
    Reject H0. The result is "statistically significant."
If p-value >= alpha:
    Fail to reject H0. You do not have enough evidence.
    This does NOT mean H0 is true.
```

**置信区间** 给出一个参数的合理取值范围：

```text
95% confidence interval for the mean:
    x_bar +/- z * (s / sqrt(n))

where z = 1.96 for 95% confidence

Interpretation: if you repeated this experiment many times, 95% of the
computed intervals would contain the true mean. It does NOT mean there
is a 95% probability the true mean is in this specific interval.
```

置信区间的宽度告诉你估计的精度。宽区间意味着高不确定性。窄区间意味着你的估计很精确（但如果数据有偏，它不一定准确）。

### t 检验

t 检验比较均值。它有几种形式。

**单样本 t 检验：** 总体均值是否不同于某个假设值？

```text
t = (x_bar - mu_0) / (s / sqrt(n))

degrees of freedom = n - 1
```

**双样本 t 检验（独立）：** 两组均值是否不同？

```text
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

This is Welch's t-test, which does not assume equal variances.
Always use Welch's unless you have a specific reason for equal variances.
```

**配对 t 检验：** 当测量值成对出现时使用（同一个模型在相同数据划分上评估）：

```text
Compute d_i = x_i - y_i for each pair
Then run a one-sample t-test on the d_i values against mu_0 = 0
```

在 ML 中，配对 t 检验很常见：你在相同的 10 个交叉验证折上运行两个模型，并逐对比较它们的分数。

### 卡方检验

卡方检验检查观察频数是否匹配期望频数。它适合分类数据。

```text
chi^2 = sum((observed - expected)^2 / expected)

Example: does a language model's output distribution match the
training distribution across categories?

Category    Observed   Expected
Positive       120        100
Negative        80        100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

With 1 degree of freedom, chi^2 = 8 gives p < 0.005.
The difference is significant.
```

### 面向 ML 模型的 A/B 测试

ML 中的 A/B 测试不同于网页 A/B 测试。模型比较有一些特定挑战：

```text
1. Same test set:    Both models must be evaluated on identical data.
                     Different test sets make comparison meaningless.

2. Multiple metrics: Accuracy alone is not enough. You need precision,
                     recall, F1, latency, and fairness metrics.

3. Variance:         Use cross-validation or bootstrap to estimate
                     the variance of each metric, not just point estimates.

4. Data leakage:     If the test set was used during model selection,
                     your comparison is biased. Hold out a final test set.
```

**流程：**

```text
1. Define your metric and significance level (alpha = 0.05)
2. Run both models on the same k-fold cross-validation splits
3. Collect paired scores: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Compute differences: d_i = b_i - a_i
5. Run a paired t-test on the differences
6. Check: is the mean difference significantly different from 0?
7. Compute a confidence interval for the mean difference
8. Compute effect size (Cohen's d) to judge practical significance
```

### 统计显著性与实际显著性

一个结果可以在统计上显著，却在实践中没有意义。只要数据足够多，即使微不足道的差异也会变得统计显著。

```text
Example:
  Model A accuracy: 0.9234
  Model B accuracy: 0.9237
  n = 1,000,000 test samples
  p-value = 0.001

Statistically significant? Yes.
Practically significant? A 0.03% improvement is not worth the
engineering cost of deploying a new model.
```

**效应量** 会量化差异有多大，并且不依赖样本量：

```text
Cohen's d = (mean_1 - mean_2) / pooled_std

d = 0.2:  small effect
d = 0.5:  medium effect
d = 0.8:  large effect
```

始终同时报告 p 值和效应量。p 值告诉你差异是否真实。效应量告诉你它是否重要。

### 多重比较问题

当你检验许多假设时，其中一些会因为偶然而“显著”。如果你在 alpha = 0.05 下检验 20 件事，即使什么都不是真的，你也预期会出现 1 个假阳性。

```text
P(at least one false positive) = 1 - (1 - alpha)^m

m = 20 tests, alpha = 0.05:
P(false positive) = 1 - 0.95^20 = 0.64

You have a 64% chance of at least one false positive.
```

**Bonferroni 校正：** 用检验数量来除 alpha。

```text
Adjusted alpha = alpha / m = 0.05 / 20 = 0.0025

Only reject H0 if p-value < 0.0025.
Conservative but simple. Works when tests are independent.
```

在 ML 中，当你跨多个指标比较模型、测试许多超参数配置，或在多个数据集上评估时，这一点很重要。

### Bootstrap 方法

Bootstrapping 通过对数据进行有放回重采样来估计统计量的抽样分布。不需要对底层分布做任何假设。

**算法：**

```text
1. You have n data points
2. Draw n samples WITH replacement (some points appear multiple times,
   some not at all)
3. Compute your statistic on this bootstrap sample
4. Repeat B times (typically B = 1000 to 10000)
5. The distribution of bootstrap statistics approximates the
   sampling distribution
```

**Bootstrap 置信区间（百分位法）：**

```text
Sort the B bootstrap statistics
95% CI = [2.5th percentile, 97.5th percentile]
```

**为什么 bootstrap 对 ML 很重要：**

```text
- Test set accuracy is a point estimate. Bootstrap gives you
  confidence intervals.
- You cannot assume metric distributions are normal (especially
  for AUC, F1, precision at k).
- Bootstrap works for ANY statistic: median, ratio of two means,
  difference in AUC between two models.
- No closed-form formula needed.
```

**用于模型比较的 bootstrap：**

```text
1. You have predictions from Model A and Model B on the same test set
2. For each bootstrap iteration:
   a. Resample test indices with replacement
   b. Compute metric_A and metric_B on the resampled set
   c. Store diff = metric_B - metric_A
3. 95% CI for the difference:
   [2.5th percentile of diffs, 97.5th percentile of diffs]
4. If the CI does not contain 0, the difference is significant
```

这比配对 t 检验更稳健，因为它不做分布假设。

### 参数检验与非参数检验

**参数检验** 假设一个具体分布（通常是正态分布）：

```text
t-test:         assumes normally distributed data (or large n by CLT)
ANOVA:          assumes normality and equal variances
Pearson r:      assumes bivariate normality
```

**非参数检验** 不做分布假设：

```text
Mann-Whitney U:     compares two groups (replaces independent t-test)
Wilcoxon signed-rank: compares paired data (replaces paired t-test)
Spearman rho:       correlation on ranks (replaces Pearson)
Kruskal-Wallis:     compares multiple groups (replaces ANOVA)
```

**什么时候用非参数方法：**

```text
- Small sample size (n < 30) and data is clearly non-normal
- Ordinal data (ratings, rankings)
- Heavy outliers you cannot remove
- Skewed distributions
```

**什么时候用参数方法：**

```text
- Large sample size (CLT makes the test statistic approximately normal)
- Data is roughly symmetric without extreme outliers
- More statistical power (better at detecting real differences)
```

在 ML 实验中，你通常只有很小的 n（5 或 10 个交叉验证折），所以像 Wilcoxon 符号秩检验这样的非参数检验通常比 t 检验更合适。

### 中心极限定理：实践含义

中心极限定理（CLT）说，随着 n 增大，样本均值的分布会接近正态分布，无论底层总体分布是什么。

```text
If X_1, X_2, ..., X_n are iid with mean mu and variance sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    as n -> infinity

Works for n >= 30 in most cases.
For highly skewed distributions, you might need n >= 100.
```

**为什么这对 ML 很重要：**

```text
1. Justifies confidence intervals and t-tests on aggregated metrics
2. Explains why averaging over cross-validation folds gives stable
   estimates even when individual folds vary wildly
3. Mini-batch gradient descent works because the average gradient
   over a batch approximates the true gradient (CLT in action)
4. Ensemble methods: averaging predictions from many models gives
   more stable output than any single model
```

**CLT 不能做什么：**

```text
- Does NOT make your data normal. It makes the MEAN of samples normal.
- Does NOT work for heavy-tailed distributions with infinite variance
  (Cauchy distribution).
- Does NOT apply to dependent data (time series without correction).
```

### ML 论文中常见的统计错误

1. **在训练集上测试。** 这会保证过拟合。始终留出模型在训练期间从未见过的数据。

2. **没有置信区间。** 只报告一个没有不确定性的准确率数字，会让结果不可复现、不可验证。

3. **忽略多重比较。** 测试 50 个配置并在不校正的情况下报告最好的那个，会抬高假阳性率。

4. **混淆统计显著性和实际显著性。** 在 0.01% 的准确率提升上得到 p-value = 0.001 并没有意义。

5. **在类别不平衡数据上使用准确率。** 一个负类占 99% 的数据集上 99% 的准确率，意味着模型什么也没学到。使用 precision、recall、F1 或 AUC。

6. **挑选指标。** 只报告你的模型获胜的那个指标。诚实的评估会报告所有相关指标。

7. **在训练/测试划分之间泄漏信息。** 在划分之前做归一化，或者用未来数据预测过去。

8. **测试集很小且没有方差估计。** 在 100 个样本上评估并声称有 2% 的提升，这是噪声，不是信号。

9. **在数据并不独立时假设独立。** 来自同一位患者的医学影像，来自同一篇文档的多个句子。组内观测值是相关的。

10. **P-hacking。** 尝试不同检验、子集或排除标准，直到得到 p < 0.05。结果只是搜索过程的产物。

## 动手实现

你将实现：

1. **从零实现描述性统计**（mean、median、mode、standard deviation、percentiles、IQR）
2. **相关性函数**（Pearson 和 Spearman，以及协方差矩阵）
3. **假设检验**（单样本 t 检验、双样本 t 检验、卡方检验）
4. **Bootstrap 置信区间**（适用于任意统计量，不需要假设）
5. **A/B 测试模拟器**（生成数据、执行检验、检查 Type I 和 Type II errors）
6. **统计显著性与实际显著性演示**（展示大 n 如何让一切都变得“显著”）

全部从零实现，只使用 `math` 和 `random`。不用 `numpy`，不用 `scipy`。

## 关键术语

| 术语 | 定义 |
|---|---|
| 均值 | 值的总和除以数量。对离群值敏感。 |
| 中位数 | 排序后数据的中间值。对离群值稳健。 |
| 标准差 | 方差的平方根。用原始单位度量离散程度。 |
| 百分位数 | 给定百分比的数据会落在其下方的值。 |
| IQR | 四分位距。Q3 减 Q1。中间 50% 的离散范围。 |
| Pearson 相关 | 度量两个变量之间的线性关联。范围 [-1, 1]。 |
| Spearman 相关 | 使用秩来度量单调关联。 |
| 协方差矩阵 | 所有特征之间成对协方差组成的矩阵。 |
| 零假设 | 默认假设，即没有效应或没有差异。 |
| p 值 | 在零假设为真时，看到如此极端数据的概率。 |
| 置信区间 | 在给定置信水平下，参数的一组合理取值范围。 |
| t 检验 | 检验均值是否存在显著差异。使用 t 分布。 |
| 卡方检验 | 检验观察频数是否不同于期望频数。 |
| 效应量 | 差异的大小，独立于样本量。Cohen's d 很常见。 |
| Bonferroni 校正 | 将显著性阈值除以检验数量，以控制假阳性。 |
| Bootstrap | 通过有放回重采样来估计抽样分布。 |
| Type I error | 假阳性。在 H0 为真时拒绝 H0。 |
| Type II error | 假阴性。在 H0 为假时未能拒绝 H0。 |
| 统计功效 | 正确拒绝错误 H0 的概率。Power = 1 减 Type II error rate。 |
| 中心极限定理 | 随着样本量增长，样本均值会收敛到正态分布。 |
| 参数检验 | 假设数据服从某个具体分布（通常是正态分布）。 |
| 非参数检验 | 不做分布假设。基于秩或符号工作。 |
