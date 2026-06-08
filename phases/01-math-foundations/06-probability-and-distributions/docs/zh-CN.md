# 概率与分布

> 概率是 AI 用来表达不确定性的语言。

**类型：** 学习
**语言：** Python
**先修：** Phase 1，第 01-04 课
**时间：** ~75 分钟

## 学习目标

- 从零开始为 Bernoulli、categorical、Poisson、uniform 和 normal 分布实现 PMF 与 PDF
- 计算期望值和方差，并用中心极限定理解释为什么 Gaussian 如此常见
- 使用数值稳定技巧（减去最大 logit）构建 softmax 和 log-softmax 函数
- 从 logits 计算 cross-entropy loss，并将它连接到 negative log-likelihood

## 要解决的问题

一个分类器输出 `[0.03, 0.91, 0.06]`。一个语言模型要从 50,000 个候选词中选择下一个词。一个 diffusion model 通过从学到的分布中采样来生成图像。这些都是概率在发挥作用。

模型做出的每个预测都是一个概率分布。每个损失函数都在衡量预测分布与真实分布相差多远。每一步训练都在调整参数，让一个分布看起来更像另一个分布。没有概率，你就读不懂任何一篇 ML 论文，调不动任何一个模型，也无法理解为什么训练损失会变成 NaN。

## 核心概念

### 事件、样本空间与概率

样本空间 S 是所有可能结果的集合。事件是样本空间的一个子集。概率把事件映射到 0 到 1 之间的数字。

```text
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

概率的全部内容由三个公理定义：
1. 对任意事件 A，都有 P(A) >= 0
2. P(S) = 1（总会发生某件事）
3. 当 A 和 B 不可能同时发生时，P(A or B) = P(A) + P(B)

其他所有内容（Bayes' theorem、expectations、distributions）都从这三条规则推出。

### 条件概率与独立性

P(A|B) 表示在 B 已经发生的条件下 A 发生的概率。

```text
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

如果知道一个事件发生并不会告诉你另一个事件的任何信息，这两个事件就是独立的：

```text
Independent:   P(A|B) = P(A)
Equivalent to: P(A and B) = P(A) * P(B)
```

抛硬币是独立的。不放回抽牌则不是。

### 概率质量函数与概率密度函数

离散随机变量有概率质量函数（PMF）。每个结果都有一个可以直接读出的具体概率。

```text
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

连续随机变量有概率密度函数（PDF）。单个点上的密度不是概率。概率来自对某个区间上的密度做积分。

```text
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

这个区别在 ML 中很重要。分类输出是 PMF（离散选择）。VAE 的 latent space 使用 PDF（连续变量）。

### 常见分布

**Bernoulli：** 一次试验，两种结果。用于建模二分类。

```text
P(X = 1) = p
P(X = 0) = 1 - p
Mean = p,  Variance = p(1-p)
```

**Categorical：** 一次试验，k 种结果。用于建模多分类（softmax output）。

```text
P(X = i) = p_i,  where sum of p_i = 1
Example: P(cat) = 0.7,  P(dog) = 0.2,  P(bird) = 0.1
```

**Uniform：** 所有结果等可能。用于随机初始化。

```text
Discrete: P(X = k) = 1/n for k in {1, ..., n}
Continuous: f(x) = 1/(b-a) for x in [a, b]
```

**Normal (Gaussian)：** 钟形曲线。由均值（mu）和方差（sigma^2）参数化。

```text
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**Poisson：** 固定区间内稀有事件的计数。用于建模事件发生率。

```text
P(X = k) = (lambda^k * e^(-lambda)) / k!
Mean = lambda,  Variance = lambda
```

### 期望值与方差

期望值是结果的加权平均。

```text
Discrete:   E[X] = sum of x_i * P(X = x_i)
Continuous: E[X] = integral of x * f(x) dx
```

方差衡量围绕均值的离散程度。

```text
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
Standard deviation = sqrt(Var(X))
```

在 ML 中，期望值会以损失函数的形式出现（数据分布上的平均损失）。方差告诉你模型的稳定性。梯度的高方差意味着训练很嘈杂。

### 联合分布与边缘分布

联合分布 P(X, Y) 描述两个随机变量的共同情况。

联合 PMF 示例（X = 天气，Y = 雨伞）：

| | Y=0（无雨伞） | Y=1（有雨伞） | 边缘 P(X) |
|---|---|---|---|
| X=0（晴） | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1（雨） | 0.05 | 0.45 | P(X=1) = 0.50 |
| **边缘 P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

边缘分布会把另一个变量求和消去：

```text
P(X = x) = sum over all y of P(X = x, Y = y)
```

上表中的行总和与列总和就是边缘分布。

### 为什么正态分布无处不在

中心极限定理：许多独立随机变量的和（或平均值）会收敛到正态分布，不管原始分布是什么。

```text
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution.
```

这就是为什么：
- 测量误差近似服从正态分布（来自许多小的独立来源）
- 神经网络的权重初始化会使用正态分布
- SGD 中的梯度噪声近似服从正态分布（许多样本梯度的和）
- 在给定均值和方差时，正态分布是最大熵分布

### 对数概率

原始概率会带来数值问题。把许多很小的概率相乘，很快就会下溢为零。

```text
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0 (underflow after ~30 terms)
```

对数概率解决了这个问题。乘法会变成加法。

```text
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> finite number (no underflow)
```

规则：
- log(a * b) = log(a) + log(b)
- log 概率总是 <= 0（因为 0 < P <= 1）
- 越负 = 越不可能
- Cross-entropy loss 是正确类别的负对数概率

### 作为概率分布的 Softmax

神经网络输出原始分数（logits）。Softmax 会把它们转换成有效的概率分布。

```text
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

softmax trick：在取指数之前减去最大 logit，以防止溢出。

```text
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

Log-softmax 把 softmax 和 log 结合在一起以获得数值稳定性。PyTorch 在 cross-entropy loss 内部会使用它。

### 采样

采样意味着从一个分布中抽取随机值。在 ML 中：
- Dropout 会随机采样哪些神经元要置零
- Data augmentation 会采样随机变换
- 语言模型会从预测分布中采样下一个 token
- Diffusion model 会采样噪声并逐步去噪

从任意分布中采样需要 inverse transform sampling、rejection sampling 或 reparameterization trick（用于 VAE）等技术。

## 动手实现

### 第 1 步：概率基础

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### 第 2 步：从零开始实现 PMF 和 PDF

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### 第 3 步：期望值与方差

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### 第 4 步：从分布中采样

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### 第 5 步：Softmax 与对数概率

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### 第 6 步：中心极限定理演示

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### 第 7 步：可视化

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

包含所有可视化的完整实现在 `code/probability.py` 中。

## 实际使用

使用 NumPy 和 SciPy，上面的所有内容都是一行调用：

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

你已经从零实现过它们。现在你知道这些库调用到底在做什么。

## 练习

1. 为 exponential distribution 实现 inverse transform sampling。通过采样 10,000 个值并将直方图与真实 PDF 比较来验证它。

2. 为两枚加权骰子构建联合分布表。计算边缘分布，并检查这两枚骰子是否独立。

3. 一个 5 类分类器在正确类别为索引 3 时输出 logits `[2.0, 0.5, -1.0, 3.0, 0.1]`。计算它的 cross-entropy loss。然后用 PyTorch 的 `nn.CrossEntropyLoss` 验证你的答案。

4. 编写一个函数，输入一组 log probabilities，返回最可能的序列、总 log probability，以及等价的原始概率。用一个 50 词的句子测试它，其中每个词的概率都是 0.01。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Sample space | “所有可能性” | 实验中每一种可能结果组成的集合 S |
| PMF | “概率函数” | 给出每个离散结果精确概率的函数，所有概率之和为 1 |
| PDF | “概率曲线” | 连续变量的密度函数。对某个区间积分才能得到概率 |
| Conditional probability | “给定某件事后的概率” | P(A\|B) = P(A and B) / P(B)。Bayesian thinking 和 Bayes' theorem 的基础 |
| Independence | “它们互不影响” | P(A and B) = P(A) * P(B)。知道一个事件发生不会告诉你另一个事件的任何信息 |
| Expected value | “平均值” | 所有结果按概率加权后的和。损失函数就是一个期望值 |
| Variance | “有多分散” | 相对均值的平方偏差的期望。高方差 = 嘈杂、不稳定的估计 |
| Normal distribution | “钟形曲线” | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2))。由于 CLT，它到处出现 |
| Central Limit Theorem | “平均值会变成正态” | 许多独立样本的均值会收敛到正态分布，不管来源分布是什么 |
| Joint distribution | “两个变量放在一起” | P(X, Y) 描述 X 和 Y 每一种结果组合的概率 |
| Marginal distribution | “把另一个变量求和消去” | P(X) = sum_y P(X, Y)。从联合分布中恢复单个变量的分布 |
| Log probability | “概率的对数” | log P(x)。把乘积变成求和，防止长序列中的数值下溢 |
| Softmax | “把分数变成概率” | softmax(z_i) = exp(z_i) / sum(exp(z_j))。把实数 logits 映射成有效的概率分布 |
| Cross-entropy | “损失函数” | -sum(p_true * log(p_predicted))。衡量两个分布有多不同。越低越好 |
| Logits | “模型原始输出” | softmax 之前未经归一化的分数。名称来自 logistic function |
| Sampling | “抽取随机值” | 按照概率分布生成值。模型正是这样生成输出的 |

## 延伸阅读

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - 关于平均值为什么会变成正态分布的可视化证明
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf) - 覆盖本文内容以及更多主题的简明参考
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - 为什么数值稳定性重要，以及如何实现它
