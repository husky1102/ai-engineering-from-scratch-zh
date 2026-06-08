# 处理不平衡数据

> 当 99% 的数据都是“正常”时，准确率就是谎言。

**类型:** 构建
**语言:** Python
**先修:** Phase 2, Lessons 01-09（尤其是评估指标）
**时间:** ~90 分钟

## 学习目标

- 从零实现 SMOTE，并解释合成过采样与随机复制有何不同
- 使用 F1、AUPRC 和 Matthews Correlation Coefficient 评估不平衡分类器，而不是使用准确率
- 比较类别权重、阈值调优和重采样策略，并为给定的不平衡比例选择合适方法
- 构建完整的不平衡数据流水线，结合 SMOTE、类别权重和阈值优化

## 要解决的问题

你构建了一个欺诈检测模型。它取得了 99.9% 的准确率。你开始庆祝。然后你发现它把每一笔交易都预测为“非欺诈”。

这不是 bug。当只有 0.1% 的交易是欺诈时，这样做是理性的。模型学到：永远猜多数类可以最小化总体错误。它在技术上正确，却完全没用。

只要真实分类任务重要，这种情况就到处存在。疾病诊断：1% 阳性率。网络入侵：0.01% 攻击。制造缺陷：0.5% 缺陷率。垃圾邮件过滤：20% 垃圾邮件。流失预测：5% 流失用户。少数类越关键，它往往越稀有。

准确率会失败，因为它把所有正确预测都同等对待。正确标注一笔合法交易和正确抓住一笔欺诈，在准确率里都只算一分。但抓住欺诈才是模型存在的全部理由。我们需要指标、技术和训练策略，迫使模型关注稀有但重要的类别。

## 核心概念

### 为什么准确率会失败

考虑一个包含 1000 个样本的数据集：990 个负类，10 个正类。一个永远预测负类的模型：

|  | 预测为正类 | 预测为负类 |
|--|---|---|
| 实际为正类 | 0 (TP) | 10 (FN) |
| 实际为负类 | 0 (FP) | 990 (TN) |

Accuracy = (0 + 990) / 1000 = 99.0%

模型抓住了零个欺诈。零个疾病。零个缺陷。但准确率说是 99%。这就是为什么准确率对不平衡问题很危险。

### 更好的指标

**Precision** = TP / (TP + FP)。在所有被标记为正类的样本里，有多少确实是正类？高 precision 表示误报少。

**Recall** = TP / (TP + FN)。在所有实际为正类的样本里，我们抓住了多少？高 recall 表示漏报少。

**F1 Score** = 2 * precision * recall / (precision + recall)。调和平均数。相比算术平均数，它会更严厉地惩罚 precision 与 recall 之间的极端不平衡。

**F-beta Score** = (1 + beta^2) * precision * recall / (beta^2 * precision + recall)。当 beta > 1 时，recall 更重要。当 beta < 1 时，precision 更重要。F2 在欺诈检测中很常见（漏掉欺诈比误报更糟）。

**AUPRC**（Area Under Precision-Recall Curve）。类似 AUC-ROC，但对不平衡数据更有信息量。随机分类器的 AUPRC 等于正类比例（不像 ROC 那样是 0.5）。这让改进更容易被看见。

**Matthews Correlation Coefficient** = (TP * TN - FP * FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))。范围从 -1 到 +1。只有当模型在两个类别上都表现良好时，才会给出高分。即使类别大小差异很大，也能保持平衡。

对上面“永远预测负类”的模型：precision = 0/0（未定义，通常设为 0），recall = 0/10 = 0，F1 = 0，MCC = 0。这些指标会正确地把模型识别为毫无价值。

### 不平衡数据流水线

```mermaid
flowchart TD
    A[Imbalanced Dataset] --> B{Imbalance Ratio?}
    B -->|Mild: 80/20| C[Class Weights]
    B -->|Moderate: 95/5| D[SMOTE + Threshold Tuning]
    B -->|Severe: 99/1| E[SMOTE + Class Weights + Threshold]
    C --> F[Train Model]
    D --> F
    E --> F
    F --> G[Evaluate with F1 / AUPRC / MCC]
    G --> H{Good Enough?}
    H -->|No| I[Try Different Strategy]
    H -->|Yes| J[Deploy with Monitoring]
    I --> B
```

### SMOTE：Synthetic Minority Oversampling Technique

随机过采样会复制已有的少数类样本。这种方法有效，但会带来过拟合风险，因为模型会反复看到完全相同的点。

SMOTE 会创建新的合成少数类样本。这些样本看起来合理，但不是副本。算法如下：

1. 对每个少数类样本 x，在其他少数类样本中找到它的 k 个最近邻
2. 随机选择一个邻居
3. 在 x 与该邻居之间的线段上创建一个新样本

公式：`new_sample = x + random(0, 1) * (neighbor - x)`

这会在真实少数类点之间进行插值，在特征空间的同一区域中创建样本，而不是只复制已有数据。

```mermaid
flowchart LR
    subgraph Original["Original Minority Points"]
        P1["x1 (1.0, 2.0)"]
        P2["x2 (1.5, 2.5)"]
        P3["x3 (2.0, 1.5)"]
    end
    subgraph SMOTE["SMOTE Generation"]
        direction TB
        S1["Pick x1, neighbor x2"]
        S2["random t = 0.4"]
        S3["new = x1 + 0.4*(x2-x1)"]
        S4["new = (1.2, 2.2)"]
        S1 --> S2 --> S3 --> S4
    end
    Original --> SMOTE
    subgraph Result["Augmented Set"]
        R1["x1 (1.0, 2.0)"]
        R2["x2 (1.5, 2.5)"]
        R3["x3 (2.0, 1.5)"]
        R4["synthetic (1.2, 2.2)"]
    end
    SMOTE --> Result
```

### 采样策略对比

**Random Oversampling**：复制少数类样本，使其数量匹配多数类。
- 优点：简单，没有信息损失
- 缺点：完全重复会导致过拟合，增加训练时间

**Random Undersampling**：移除多数类样本，使其数量匹配少数类。
- 优点：训练快，简单
- 缺点：丢弃可能有用的多数类数据，方差更高

**SMOTE**：通过插值创建合成少数类样本。
- 优点：生成新的数据点，相比随机过采样更能减少过拟合
- 缺点：可能在决策边界附近创建有噪声的样本，不考虑多数类分布

| 策略 | 改变的数据 | 风险 | 何时使用 |
|----------|-------------|------|-------------|
| Oversample | 复制少数类 | 过拟合 | 小数据集，中等不平衡 |
| Undersample | 移除多数类 | 信息损失 | 大数据集，希望快速训练 |
| SMOTE | 添加合成少数类 | 边界噪声 | 中等不平衡，有足够少数类样本用于 k-NN |

### 类别权重

不要改变数据，而是改变模型如何对待错误。给少数类误分类分配更高权重。

对于一个有 950 个负类样本和 50 个正类样本的二分类问题：
- 负类权重 = n_samples / (2 * n_negative) = 1000 / (2 * 950) = 0.526
- 正类权重 = n_samples / (2 * n_positive) = 1000 / (2 * 50) = 10.0

正类获得 19 倍权重。误分类一个正类样本的代价，等同于误分类 19 个负类样本。模型被迫关注少数类。

在 logistic regression 中，这会修改损失函数：

```text
weighted_loss = -sum(w_i * [y_i * log(p_i) + (1-y_i) * log(1-p_i)])
```

其中 w_i 取决于样本 i 的类别。

从数学期望上看，类别权重等价于过采样，但不会创建新的数据点。这让它更快，也避免了重复样本带来的过拟合风险。

### 阈值调优

大多数分类器会输出一个概率。默认阈值是 0.5：如果 P(positive) >= 0.5，就预测为正类。但 0.5 是任意选择的。当类别不平衡时，最优阈值通常要低得多。

流程如下：
1. 训练模型
2. 在验证集上获取预测概率
3. 从 0.0 到 1.0 扫描阈值
4. 在每个阈值上计算 F1（或你选择的指标）
5. 选择能最大化指标的阈值

```mermaid
flowchart LR
    A[Model] --> B[Predict Probabilities]
    B --> C[Sweep Thresholds 0.0 to 1.0]
    C --> D[Compute F1 at Each]
    D --> E[Pick Best Threshold]
    E --> F[Use in Production]
```

模型可能会对一笔欺诈交易输出 P(fraud) = 0.15。在阈值 0.5 下，它会被归为非欺诈。在阈值 0.10 下，它会被正确抓住。概率校准不如排序重要 -- 只要欺诈的概率高于非欺诈，就存在一个能把它们分开的阈值。

### 代价敏感学习

类别权重的一般化。不要使用统一代价，而是分配具体的误分类代价：

| | 预测为正类 | 预测为负类 |
|--|---|---|
| 实际为正类 | 0（正确） | C_FN = 100 |
| 实际为负类 | C_FP = 1 | 0（正确） |

漏掉一笔欺诈交易（FN）的代价，是一次误报（FP）的 100 倍。模型优化的是总代价，而不是总错误数。

当你能估计真实世界代价时，这是最有原则的方法。漏诊癌症的代价，与一次误报导致额外活检的代价完全不同。把这些代价显式写出来，会迫使模型做出正确权衡。

### 决策流程图

```mermaid
flowchart TD
    A[Start: Imbalanced Dataset] --> B{How imbalanced?}
    B -->|"< 70/30"| C["Mild: try class weights first"]
    B -->|"70/30 to 95/5"| D["Moderate: SMOTE + class weights"]
    B -->|"> 95/5"| E["Severe: combine multiple strategies"]
    C --> F{Enough data?}
    D --> F
    E --> F
    F -->|"< 1000 samples"| G["Oversample or SMOTE, avoid undersampling"]
    F -->|"1000-10000"| H["SMOTE + threshold tuning"]
    F -->|"> 10000"| I["Undersampling OK, or class weights"]
    G --> J[Train + Evaluate with F1/AUPRC]
    H --> J
    I --> J
    J --> K{Recall high enough?}
    K -->|No| L[Lower threshold]
    K -->|Yes| M{Precision acceptable?}
    M -->|No| N[Raise threshold or add features]
    M -->|Yes| O[Ship it]
```

## 动手实现

### 第 1 步：生成不平衡数据集

```python
import numpy as np


def make_imbalanced_data(n_majority=950, n_minority=50, seed=42):
    rng = np.random.RandomState(seed)

    X_maj = rng.randn(n_majority, 2) * 1.0 + np.array([0.0, 0.0])
    X_min = rng.randn(n_minority, 2) * 0.8 + np.array([2.5, 2.5])

    X = np.vstack([X_maj, X_min])
    y = np.concatenate([np.zeros(n_majority), np.ones(n_minority)])

    shuffle_idx = rng.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]
```

### 第 2 步：从零实现 SMOTE

```python
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def find_k_neighbors(X, idx, k):
    distances = []
    for i in range(len(X)):
        if i == idx:
            continue
        d = euclidean_distance(X[idx], X[i])
        distances.append((i, d))
    distances.sort(key=lambda x: x[1])
    return [d[0] for d in distances[:k]]


def smote(X_minority, k=5, n_synthetic=100, seed=42):
    rng = np.random.RandomState(seed)
    n_samples = len(X_minority)
    k = min(k, n_samples - 1)
    synthetic = []

    for _ in range(n_synthetic):
        idx = rng.randint(0, n_samples)
        neighbors = find_k_neighbors(X_minority, idx, k)
        neighbor_idx = neighbors[rng.randint(0, len(neighbors))]
        t = rng.random()
        new_point = X_minority[idx] + t * (X_minority[neighbor_idx] - X_minority[idx])
        synthetic.append(new_point)

    return np.array(synthetic)
```

### 第 3 步：随机过采样和欠采样

```python
def random_oversample(X, y, seed=42):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    X_resampled = list(X)
    y_resampled = list(y)

    for cls, count in zip(classes, counts):
        if count < max_count:
            cls_indices = np.where(y == cls)[0]
            n_needed = max_count - count
            chosen = rng.choice(cls_indices, size=n_needed, replace=True)
            X_resampled.extend(X[chosen])
            y_resampled.extend(y[chosen])

    X_out = np.array(X_resampled)
    y_out = np.array(y_resampled)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]


def random_undersample(X, y, seed=42):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    X_resampled = []
    y_resampled = []

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        chosen = rng.choice(cls_indices, size=min_count, replace=False)
        X_resampled.extend(X[chosen])
        y_resampled.extend(y[chosen])

    X_out = np.array(X_resampled)
    y_out = np.array(y_resampled)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]
```

### 第 4 步：带类别权重的 logistic regression

```python
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def logistic_regression_weighted(X, y, weights, lr=0.01, epochs=200):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        pred = sigmoid(z)
        error = pred - y
        weighted_error = error * weights

        gradient_w = (X.T @ weighted_error) / n_samples
        gradient_b = np.mean(weighted_error)

        w -= lr * gradient_w
        b -= lr * gradient_b

    return w, b


def compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    weight_map = {}
    for cls, count in zip(classes, counts):
        weight_map[cls] = n_samples / (n_classes * count)
    return np.array([weight_map[yi] for yi in y])
```

### 第 5 步：阈值调优

```python
def find_optimal_threshold(y_true, y_probs, metric="f1"):
    best_threshold = 0.5
    best_score = -1.0

    for threshold in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_probs >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        if metric == "f1":
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        elif metric == "recall":
            score = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        elif metric == "precision":
            score = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score
```

### 第 6 步：评估函数

```python
def confusion_matrix_values(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp, tn, fp, fn


def compute_metrics(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix_values(y_true, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
    }
```

### 第 7 步：比较所有方法

```python
X, y = make_imbalanced_data(950, 50, seed=42)
split = int(0.8 * len(y))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Baseline: no treatment
w_base, b_base = logistic_regression_weighted(
    X_train, y_train, np.ones(len(y_train)), lr=0.1, epochs=300
)
probs_base = sigmoid(X_test @ w_base + b_base)
preds_base = (probs_base >= 0.5).astype(int)

# Oversampled
X_over, y_over = random_oversample(X_train, y_train)
w_over, b_over = logistic_regression_weighted(
    X_over, y_over, np.ones(len(y_over)), lr=0.1, epochs=300
)
preds_over = (sigmoid(X_test @ w_over + b_over) >= 0.5).astype(int)

# SMOTE
minority_mask = y_train == 1
X_minority = X_train[minority_mask]
synthetic = smote(X_minority, k=5, n_synthetic=len(y_train) - 2 * int(minority_mask.sum()))
X_smote = np.vstack([X_train, synthetic])
y_smote = np.concatenate([y_train, np.ones(len(synthetic))])
w_sm, b_sm = logistic_regression_weighted(
    X_smote, y_smote, np.ones(len(y_smote)), lr=0.1, epochs=300
)
preds_smote = (sigmoid(X_test @ w_sm + b_sm) >= 0.5).astype(int)

# Class weights
sample_weights = compute_class_weights(y_train)
w_cw, b_cw = logistic_regression_weighted(
    X_train, y_train, sample_weights, lr=0.1, epochs=300
)
probs_cw = sigmoid(X_test @ w_cw + b_cw)
preds_cw = (probs_cw >= 0.5).astype(int)

# Threshold tuning (tune on held-out validation set, not test set)
probs_val = sigmoid(X_val @ w_cw + b_cw)
best_thresh, best_f1 = find_optimal_threshold(y_val, probs_val, metric="f1")
preds_thresh = (probs_cw >= best_thresh).astype(int)
```

代码文件会在单个脚本中运行上述所有内容，并打印结果。

## 实际使用

使用 scikit-learn 和 imbalanced-learn 时，这些技术都是一行代码：

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

model_weighted = LogisticRegression(class_weight="balanced")
model_weighted.fit(X_train, y_train)
print(classification_report(y_test, model_weighted.predict(X_test)))

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
model_smote = LogisticRegression()
model_smote.fit(X_resampled, y_resampled)
print(classification_report(y_test, model_smote.predict(X_test)))

pipeline = Pipeline([
    ("smote", SMOTE()),
    ("model", LogisticRegression(class_weight="balanced")),
])
pipeline.fit(X_train, y_train)
print(classification_report(y_test, pipeline.predict(X_test)))
```

从零实现会清楚展示每种技术到底在做什么。SMOTE 只是少数类上的 k-NN 插值。类别权重会乘到损失上。阈值调优就是对 cutoff 做一次 for 循环。没有魔法。

## 交付成果

本课产出：
- `outputs/skill-imbalanced-data.md` -- 一个处理不平衡分类问题的决策检查清单

## 练习

1. **Borderline-SMOTE**：修改 SMOTE 实现，只为靠近决策边界的少数类点生成合成样本（这些点的 k-nearest neighbors 包含多数类样本）。在类别重叠的数据集上比较它与标准 SMOTE 的结果。

2. **代价矩阵优化**：实现代价敏感学习，其中代价矩阵是一个参数。创建一个函数，接收代价矩阵并返回能最小化期望代价的最优预测。使用不同代价比例（1:10、1:100、1:1000）测试，并绘制 precision-recall 权衡如何变化。

3. **阈值校准**：实现 Platt scaling（在模型原始输出上拟合一个 logistic regression，以产生校准后的概率）。比较校准前后的 precision-recall curve。展示校准不会改变排序（AUC 保持不变），但会让概率更有意义。

4. **带 balanced bagging 的集成**：训练多个模型，每个模型都使用一个平衡 bootstrap 样本（全部少数类 + 随机多数类子集）。平均它们的预测。将这种方法与单个使用 SMOTE 的模型比较。衡量性能和多次运行之间的方差。

5. **不平衡比例实验**：取一个平衡数据集，逐步提高不平衡比例（50/50、70/30、90/10、95/5、99/1）。对每个比例，分别在使用和不使用 SMOTE 的情况下训练。绘制两种方法的 F1 vs imbalance ratio。到哪个比例时，SMOTE 开始产生有意义的差异？

## 关键术语

| 术语 | 人们常说 | 它真正的意思 |
|------|----------------|----------------------|
| Class imbalance | “一个类别的样本多得多” | 数据集中的类别分布显著偏斜，导致模型偏向多数类 |
| SMOTE | “合成过采样” | 通过在已有少数类样本与其 k 个最近少数类邻居之间插值，创建新的少数类样本 |
| Class weights | “让稀有类别上的错误更昂贵” | 用特定于类别的权重乘以损失函数，让模型更严厉地惩罚少数类误分类 |
| Threshold tuning | “移动决策边界” | 将分类概率 cutoff 从默认 0.5 改为能优化目标指标的值 |
| Precision-recall tradeoff | “你不能两者兼得” | 降低阈值会抓住更多正类（更高 recall），但也会标记更多 false positives（更低 precision）；反之亦然 |
| AUPRC | “PR 曲线下面积” | 将 precision-recall curve 汇总成一个数字；当类别高度不平衡时，比 AUC-ROC 更有信息量 |
| Matthews Correlation Coefficient | “平衡指标” | 预测标签与实际标签之间的相关性；只有当模型在两个类别上都表现良好时才会产生高分 |
| Cost-sensitive learning | “不同错误有不同代价” | 将真实世界的误分类代价纳入训练目标，使模型优化总代价，而不是错误数量 |
| Random oversampling | “复制少数类” | 重复少数类样本以平衡类别数量；简单，但有过拟合到重复点的风险 |

## 延伸阅读

- [SMOTE: Synthetic Minority Over-sampling Technique (Chawla et al., 2002)](https://arxiv.org/abs/1106.1813) -- 原始 SMOTE 论文，仍然是不平衡学习领域引用最多的工作
- [Learning from Imbalanced Data (He & Garcia, 2009)](https://ieeexplore.ieee.org/document/5128907) -- 综合综述，覆盖采样、代价敏感和算法层面的方法
- [imbalanced-learn documentation](https://imbalanced-learn.org/stable/) -- Python 库，包含 SMOTE 变体、欠采样策略和流水线集成
- [The Precision-Recall Plot Is More Informative than the ROC Plot (Saito & Rehmsmeier, 2015)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432) -- 何时以及为什么应该在不平衡问题中优先使用 PR 曲线，而不是 ROC 曲线
