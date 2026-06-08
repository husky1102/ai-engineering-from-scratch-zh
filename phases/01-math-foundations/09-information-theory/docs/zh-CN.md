# 信息论

> 信息论度量“惊讶”。损失函数就是建立在它之上。

**类型：** 学习
**语言：** Python
**先修：** Phase 1, Lesson 06 (Probability)
**时间：** ~60 分钟

## 学习目标

- 从零计算熵、交叉熵和 KL 散度，并解释它们之间的关系
- 推导为什么最小化交叉熵损失等价于最大化对数似然
- 计算特征与目标之间的互信息，用它给特征重要性排序
- 将困惑度解释为语言模型在每一步有效选择的词表大小

## 要解决的问题

你训练每一个分类模型时都会调用 `CrossEntropyLoss()`。你在每一篇语言模型论文里都会看到 “perplexity”。你会在 VAE、蒸馏和 RLHF 中读到 KL divergence。这些概念并不是彼此无关。它们都是同一个思想换了不同外衣。

信息论给了你讨论不确定性、压缩和预测的语言。Claude Shannon 在 1948 年发明它，是为了解决通信问题。结果证明，训练神经网络也是一个通信问题：模型试图通过由学习到的权重组成的有噪声信道，传递正确标签。

本课会从零构建每一个公式，让你看清它们从哪里来，以及为什么有效。

## 核心概念

### 信息量（惊讶）

当某件小概率事件发生时，它携带的信息更多。硬币正面朝上？并不意外。中彩票？非常意外。

概率为 p 的事件的信息量是：

```text
I(x) = -log(p(x))
```

使用以 2 为底的 log 会得到 bits。使用自然对数会得到 nats。思想相同，单位不同。

```text
Event              Probability    Surprise (bits)
Fair coin heads    0.5            1.0
Rolling a 6        0.167          2.58
1-in-1000 event    0.001          9.97
Certain event      1.0            0.0
```

必然事件携带零信息。你本来就知道它会发生。

### 熵（平均惊讶）

熵是一种分布在所有可能结果上的期望惊讶。

```text
H(P) = -sum( p(x) * log(p(x)) )  for all x
```

公平硬币在二元变量中拥有最大熵：1 bit。有偏硬币（99% 正面）熵很低：0.08 bits。你几乎已经知道会发生什么，所以每次投掷几乎不会告诉你任何东西。

```text
Fair coin:    H = -(0.5 * log2(0.5) + 0.5 * log2(0.5)) = 1.0 bit
Biased coin:  H = -(0.99 * log2(0.99) + 0.01 * log2(0.01)) = 0.08 bits
```

熵度量一个分布中不可约的不确定性。你无法把它压缩到比熵更低。

### 交叉熵（你每天都在用的损失函数）

当你用分布 Q 去编码实际上来自分布 P 的事件时，交叉熵度量平均惊讶。

```text
H(P, Q) = -sum( p(x) * log(q(x)) )  for all x
```

P 是真实分布（标签）。Q 是模型的预测。如果 Q 与 P 完全匹配，交叉熵就等于熵。任何不匹配都会让它变大。

在分类中，P 是 one-hot 向量（真实类别概率为 1，其他全是 0）。这会把交叉熵简化为：

```text
H(P, Q) = -log(q(true_class))
```

这就是分类任务中完整的交叉熵损失公式。最大化正确类别的预测概率。

### KL 散度（分布之间的距离）

KL 散度度量你用 Q 代替 P 时会多得到多少额外惊讶。

```text
D_KL(P || Q) = sum( p(x) * log(p(x) / q(x)) )  for all x
             = H(P, Q) - H(P)
```

交叉熵等于熵加上 KL 散度。由于真实分布的熵在训练过程中是常数，最小化交叉熵就等价于最小化 KL 散度。你正在把模型的分布推向真实分布。

KL 散度不是对称的：D_KL(P || Q) != D_KL(Q || P)。它不是真正的距离度量。

### 互信息

互信息度量知道一个变量能告诉你关于另一个变量的多少信息。

```text
I(X; Y) = H(X) - H(X|Y)
        = H(X) + H(Y) - H(X, Y)
```

如果 X 和 Y 相互独立，互信息为零。知道其中一个不会告诉你另一个的任何信息。如果它们完全相关，互信息等于任意一个变量的熵。

在特征选择中，某个特征与目标之间的互信息高，意味着这个特征有用。互信息低，意味着它更像噪声。

### 条件熵

H(Y|X) 度量观察到 X 之后，关于 Y 还剩下多少不确定性。

```text
H(Y|X) = H(X,Y) - H(X)
```

两个极端：
- 如果 X 完全决定 Y，那么 H(Y|X) = 0。知道 X 会消除关于 Y 的全部不确定性。例子：X = 摄氏温度，Y = 华氏温度。
- 如果 X 完全不能告诉你关于 Y 的信息，那么 H(Y|X) = H(Y)。知道 X 一点也不会减少你的不确定性。例子：X = 抛硬币结果，Y = 明天的天气。

条件熵总是非负，并且永远不会超过 H(Y)：

```text
0 <= H(Y|X) <= H(Y)
```

在机器学习中，条件熵会出现在决策树里。在每一次划分时，算法会选择能最小化 H(Y|X) 的特征 X，也就是能移除最多标签 Y 不确定性的特征。

### 联合熵

H(X,Y) 是 X 和 Y 共同组成的联合分布的熵。

```text
H(X,Y) = -sum sum p(x,y) * log(p(x,y))   for all x, y
```

关键性质：

```text
H(X,Y) <= H(X) + H(Y)
```

当 X 和 Y 相互独立时取等号。如果它们共享信息，联合熵就小于各自熵的和。那块“少掉的”熵正好就是互信息。

```mermaid
graph TD
    subgraph "Information Venn Diagram"
        direction LR
        HX["H(X)"]
        HY["H(Y)"]
        MI["I(X;Y)<br/>Mutual<br/>Information"]
        HXgY["H(X|Y)<br/>= H(X) - I(X;Y)"]
        HYgX["H(Y|X)<br/>= H(Y) - I(X;Y)"]
        HXY["H(X,Y) = H(X) + H(Y) - I(X;Y)"]
    end

    HXgY --- MI
    MI --- HYgX
    HX -.- HXgY
    HX -.- MI
    HY -.- MI
    HY -.- HYgX
    HXY -.- HXgY
    HXY -.- MI
    HXY -.- HYgX
```

这些关系是：
- H(X,Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)
- I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
- H(X,Y) = H(X) + H(Y) - I(X;Y)

### 互信息（深入）

互信息 I(X;Y) 量化知道一个变量会让你对另一个变量的不确定性减少多少。

```text
I(X;Y) = H(X) - H(X|Y)
       = H(Y) - H(Y|X)
       = H(X) + H(Y) - H(X,Y)
       = sum sum p(x,y) * log(p(x,y) / (p(x) * p(y)))
```

性质：
- I(X;Y) >= 0 恒成立。观察某件事不会让你丢失信息。
- I(X;Y) = 0 当且仅当 X 和 Y 相互独立。
- I(X;Y) = I(Y;X)。它是对称的，不像 KL 散度。
- I(X;X) = H(X)。一个变量与自身共享它的全部信息。

**用于特征选择的互信息。** 在 ML 中，你想要的是能提供目标信息的特征。互信息给了你一个有原则的方法来给特征排序：

1. 对每个特征 X_i，计算 I(X_i; Y)，其中 Y 是目标变量。
2. 按 MI 分数给特征排序。
3. 保留前 k 个特征。

这适用于特征与目标之间的任何关系，包括线性、非线性、单调或非单调关系。相关性只能捕捉线性关系。MI 能捕捉所有关系。

| 方法 | 能检测 | 计算成本 | 处理类别特征？ |
|--------|---------|-------------------|---------------------|
| Pearson correlation | 线性关系 | O(n) | 否 |
| Spearman correlation | 单调关系 | O(n log n) | 否 |
| Mutual information | 任意统计依赖 | O(n log n)，使用分箱 | 是 |

### 标签平滑与交叉熵

标准分类使用硬目标：[0, 0, 1, 0]。真实类别得到概率 1，其他类别得到 0。标签平滑会把它们替换成软目标：

```text
soft_target = (1 - epsilon) * hard_target + epsilon / num_classes
```

当 epsilon = 0.1 且有 4 个类别时：
- 硬目标：  [0, 0, 1, 0]
- 软目标：  [0.025, 0.025, 0.925, 0.025]

从信息论角度看，标签平滑提高了目标分布的熵。硬 one-hot 目标的熵为 0，因为其中没有不确定性。软目标有正熵。

它为什么有帮助：
- 防止模型把 logits 推到极端值（在交叉熵下，要完美匹配 one-hot 目标需要无限 logits）
- 作为正则化：模型不能 100% 自信
- 改善校准：预测概率能更好地反映真实不确定性
- 减少训练行为与推理行为之间的差距

带标签平滑的交叉熵损失变成：

```text
L = (1 - epsilon) * CE(hard_target, prediction) + epsilon * H_uniform(prediction)
```

第二项会惩罚那些远离均匀分布的预测，它是对置信度的直接正则化。

### 为什么交叉熵是分类损失的核心

三种视角，同一个结论。

**信息论视角。** 交叉熵度量使用模型分布而不是真实分布时，你浪费了多少 bits。最小化它，会让模型成为对现实最高效的编码器。

**最大似然视角。** 对 N 个训练样本及其真实类别 y_i：

```text
Likelihood     = product( q(y_i) )
Log-likelihood = sum( log(q(y_i)) )
Negative log-likelihood = -sum( log(q(y_i)) )
```

最后一行就是交叉熵损失。最小化交叉熵 = 最大化训练数据在你的模型下的似然。

**梯度视角。** 交叉熵对 logits 的梯度非常简单，就是 (predicted - true)。干净、稳定、计算快。这也是它与 softmax 完美配合的原因。

### Bits 与 Nats

唯一差异是 log 的底数。

```text
log base 2   -> bits      (information theory tradition)
log base e   -> nats      (machine learning convention)
log base 10  -> hartleys  (rarely used)
```

1 nat = 1/ln(2) bits = 1.4427 bits。PyTorch 和 TensorFlow 默认使用自然对数（nats）。

### 困惑度

困惑度是交叉熵的指数。它告诉你模型在多少个等可能选项之间感到不确定，也就是有效选项数。

```text
Perplexity = 2^H(P,Q)   (if using bits)
Perplexity = e^H(P,Q)   (if using nats)
```

一个困惑度为 50 的语言模型，平均来说就像每一步都必须从 50 个可能的下一个 token 中均匀随机挑选一样困惑。越低越好。

GPT-2 在常见基准上达到约 30 的困惑度。现代模型在表示充分的领域中通常能达到个位数。

## 动手实现

### 第 1 步：信息量与熵

```python
import math

def information_content(p, base=2):
    if p <= 0 or p > 1:
        return float('inf') if p <= 0 else 0.0
    return -math.log(p) / math.log(base)

def entropy(probs, base=2):
    return sum(
        p * information_content(p, base)
        for p in probs if p > 0
    )

fair_coin = [0.5, 0.5]
biased_coin = [0.99, 0.01]
fair_die = [1/6] * 6

print(f"Fair coin entropy:   {entropy(fair_coin):.4f} bits")
print(f"Biased coin entropy: {entropy(biased_coin):.4f} bits")
print(f"Fair die entropy:    {entropy(fair_die):.4f} bits")
```

### 第 2 步：交叉熵与 KL 散度

```python
def cross_entropy(p, q, base=2):
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0:
                return float('inf')
            total += pi * (-math.log(qi) / math.log(base))
    return total

def kl_divergence(p, q, base=2):
    return cross_entropy(p, q, base) - entropy(p, base)

true_dist = [0.7, 0.2, 0.1]
good_model = [0.6, 0.25, 0.15]
bad_model = [0.1, 0.1, 0.8]

print(f"Entropy of true dist:     {entropy(true_dist):.4f} bits")
print(f"CE (good model):          {cross_entropy(true_dist, good_model):.4f} bits")
print(f"CE (bad model):           {cross_entropy(true_dist, bad_model):.4f} bits")
print(f"KL divergence (good):     {kl_divergence(true_dist, good_model):.4f} bits")
print(f"KL divergence (bad):      {kl_divergence(true_dist, bad_model):.4f} bits")
```

### 第 3 步：作为分类损失的交叉熵

```python
def softmax(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def cross_entropy_loss(true_class, logits):
    probs = softmax(logits)
    return -math.log(probs[true_class])

logits = [2.0, 1.0, 0.1]
true_class = 0

probs = softmax(logits)
loss = cross_entropy_loss(true_class, logits)

print(f"Logits:      {logits}")
print(f"Softmax:     {[f'{p:.4f}' for p in probs]}")
print(f"True class:  {true_class}")
print(f"Loss:        {loss:.4f} nats")
print(f"Perplexity:  {math.exp(loss):.2f}")
```

### 第 4 步：交叉熵等于负对数似然

```python
import random

random.seed(42)

n_samples = 1000
n_classes = 3
true_labels = [random.randint(0, n_classes - 1) for _ in range(n_samples)]
model_logits = [[random.gauss(0, 1) for _ in range(n_classes)] for _ in range(n_samples)]

ce_loss = sum(
    cross_entropy_loss(label, logits)
    for label, logits in zip(true_labels, model_logits)
) / n_samples

nll = -sum(
    math.log(softmax(logits)[label])
    for label, logits in zip(true_labels, model_logits)
) / n_samples

print(f"Cross-entropy loss:      {ce_loss:.6f}")
print(f"Negative log-likelihood: {nll:.6f}")
print(f"Difference:              {abs(ce_loss - nll):.2e}")
```

### 第 5 步：互信息

```python
def mutual_information(joint_probs, base=2):
    rows = len(joint_probs)
    cols = len(joint_probs[0])

    margin_x = [sum(joint_probs[i][j] for j in range(cols)) for i in range(rows)]
    margin_y = [sum(joint_probs[i][j] for i in range(rows)) for j in range(cols)]

    mi = 0.0
    for i in range(rows):
        for j in range(cols):
            pxy = joint_probs[i][j]
            if pxy > 0:
                mi += pxy * math.log(pxy / (margin_x[i] * margin_y[j])) / math.log(base)
    return mi

independent = [[0.25, 0.25], [0.25, 0.25]]
dependent = [[0.45, 0.05], [0.05, 0.45]]

print(f"MI (independent): {mutual_information(independent):.4f} bits")
print(f"MI (dependent):   {mutual_information(dependent):.4f} bits")
```

## 实际使用

同样的概念，用 NumPy 来写，这也是你在实践中会使用的方式：

```python
import numpy as np

def np_entropy(p):
    p = np.asarray(p, dtype=float)
    mask = p > 0
    result = np.zeros_like(p)
    result[mask] = p[mask] * np.log(p[mask])
    return -result.sum()

def np_cross_entropy(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    mask = p > 0
    return -(p[mask] * np.log(q[mask])).sum()

def np_kl_divergence(p, q):
    return np_cross_entropy(p, q) - np_entropy(p)

true = np.array([0.7, 0.2, 0.1])
pred = np.array([0.6, 0.25, 0.15])
print(f"Entropy:    {np_entropy(true):.4f} nats")
print(f"Cross-ent:  {np_cross_entropy(true, pred):.4f} nats")
print(f"KL div:     {np_kl_divergence(true, pred):.4f} nats")
```

你从零构建了 `torch.nn.CrossEntropyLoss()` 在内部做的事情。现在你知道为什么训练时损失会下降：模型预测分布正在靠近真实分布，而这个距离用 nats 的浪费信息量来衡量。

## 练习

1. 假设英文字母表服从均匀分布（26 个字母），计算它的熵。然后用真实字母频率估计一次。哪个更高？为什么？

2. 某个样本的真实类别为 1，模型输出 logits [5.0, 2.0, 0.5]。手算交叉熵损失，然后用你的 `cross_entropy_loss` 函数验证。什么样的 logits 会给出零损失？

3. 证明 KL 散度不是对称的。选择两个分布 P 和 Q，计算 D_KL(P || Q) 与 D_KL(Q || P)。解释它们为什么不同。

4. 构建一个函数，为一段 token 预测序列计算困惑度。给定一个由 (true_token_index, predicted_logits) 对组成的列表，返回整个序列的困惑度。

## 关键术语

| 术语 | 人们常说 | 它真正的含义 |
|------|----------------|----------------------|
| Information content | “惊讶” | 编码一个事件所需的 bits（或 nats）数量：-log(p) |
| Entropy | “随机性” | 一个分布所有结果上的平均惊讶。度量不可约的不确定性。 |
| Cross-entropy | “损失函数” | 用模型分布 Q 编码真实分布 P 中事件时的平均惊讶。 |
| KL divergence | “分布之间的距离” | 用 Q 代替 P 时额外浪费的 bits。等于交叉熵减去熵。不是对称的。 |
| Mutual information | “X 和 Y 有多相关” | 知道 Y 后，关于 X 的不确定性减少了多少。为零意味着相互独立。 |
| Softmax | “把 logits 变成概率” | 先指数化再归一化。把任意实值向量映射成合法的概率分布。 |
| Perplexity | “模型有多困惑” | 交叉熵的指数。模型在每一步有效选择的词表大小。 |
| Bits | “Shannon 的单位” | 用以 2 为底的 log 度量信息。一个 bit 解决一次公平硬币投掷的不确定性。 |
| Nats | “ML 的单位” | 用自然对数度量信息。PyTorch 和 TensorFlow 默认使用。 |
| Negative log-likelihood | “NLL loss” | 对 one-hot 标签而言，它与交叉熵损失完全相同。最小化它就是最大化正确预测的概率。 |

## 延伸阅读

- [Shannon 1948: A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)：原始论文，至今仍然可读
- [Visual Information Theory (Chris Olah)](https://colah.github.io/posts/2015-09-Visual-Information/)：关于熵和 KL 散度的最佳可视化解释
- [PyTorch CrossEntropyLoss docs](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)：框架如何实现你刚刚构建的内容
