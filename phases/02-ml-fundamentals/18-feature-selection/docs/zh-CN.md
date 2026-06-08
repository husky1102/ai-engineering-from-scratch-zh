# 特征选择

> 更多特征并不代表更好。正确的特征才更好。

**类型:** 构建
**语言:** Python
**先修:** Phase 2, Lessons 01-09, 08（特征工程）
**时间:** ~75 分钟

## 学习目标

- 从零实现过滤方法（方差阈值、互信息、卡方检验）和包装方法（RFE、前向选择）
- 解释为什么互信息能捕捉相关系数会漏掉的非线性特征-目标关系
- 比较 L1 正则化（嵌入式选择）与 RFE（包装式选择），并评估它们的计算权衡
- 构建一个组合多种方法的特征选择流水线，并展示它如何在留出数据上改善泛化能力

## 要解决的问题

你有 500 个特征。模型训练很慢，总是在过拟合，而且没人说得清它学到了什么。你继续添加特征，希望提升性能。结果更糟了。

这就是维度灾难在起作用。随着特征数量增长，特征空间的体积会爆炸式扩大。数据点变得稀疏。点与点之间的距离趋于相似。模型需要指数级更多的数据，才能找到真实模式。噪声特征淹没信号特征。过拟合变成默认结果。

特征选择就是解药。剥离噪声。移除冗余。保留那些真正携带目标信息的特征。结果是：训练更快，泛化更好，模型也更容易解释。

目标不是使用所有可用信息。目标是使用正确的信息。

## 核心概念

### 特征选择的三类方法

每种特征选择方法都属于以下三类之一：

```mermaid
flowchart TD
    A[Feature Selection Methods] --> B[Filter Methods]
    A --> C[Wrapper Methods]
    A --> D[Embedded Methods]

    B --> B1["Variance Threshold"]
    B --> B2["Mutual Information"]
    B --> B3["Chi-squared Test"]
    B --> B4["Correlation Filtering"]

    C --> C1["Recursive Feature Elimination"]
    C --> C2["Forward Selection"]
    C --> C3["Backward Elimination"]

    D --> D1["L1 / Lasso Regularization"]
    D --> D2["Tree-based Importance"]
    D --> D3["Elastic Net"]
```

**过滤方法** 用统计度量独立地给每个特征打分。它们不使用模型。速度很快，但会漏掉特征交互。

**包装方法** 训练模型来评估特征子集。它们用模型性能作为分数。效果通常更好，但代价更高，因为要反复重新训练模型。

**嵌入式方法** 在模型训练过程中选择特征。L1 正则化会把权重推到零。决策树会在最有用的特征上分裂。选择发生在拟合过程中，而不是一个单独步骤。

### 方差阈值

最简单的过滤器。如果某个特征在样本之间几乎不变化，它几乎不携带任何信息。

考虑一个在 1000 个样本里有 999 个都是 0.0 的特征。它的方差接近零。没有模型能用它区分类别。删掉它。

```text
variance(x) = mean((x - mean(x))^2)
```

设置一个阈值（例如 0.01）。丢弃所有方差低于该阈值的特征。这会在完全不查看目标变量的情况下，移除常量或近似常量特征。

何时使用：作为其他方法之前的预处理步骤。它几乎没有成本，却能捕捉显然无用的特征。

局限：一个特征可能有很高方差，但仍然是纯噪声。方差阈值是必要的，但不充分。

### 互信息

互信息衡量：知道特征 X 的取值，能让我们对目标 Y 的不确定性减少多少。

```text
I(X; Y) = sum_x sum_y p(x, y) * log(p(x, y) / (p(x) * p(y)))
```

如果 X 和 Y 独立，那么 p(x, y) = p(x) * p(y)，所以对数项为零，I(X; Y) = 0。X 对 Y 透露的信息越多，互信息越高。

相比相关系数，互信息的关键优势是：它能捕捉非线性关系。某个特征与目标的相关系数可能为零，但由于关系是二次的或周期性的，它仍然可能有很高互信息。

对于连续特征，先离散化成分箱（基于直方图的估计）。分箱数量会影响估计结果，分箱太少会丢失信息，分箱太多会引入噪声。常见选择是 sqrt(n) 个分箱，或 Sturges 规则（1 + log2(n)）。

```mermaid
flowchart LR
    A[Feature X] --> B[Discretize into Bins]
    B --> C["Compute Joint Distribution p(x,y)"]
    C --> D["Compute MI = sum p(x,y) * log(p(x,y) / p(x)p(y))"]
    D --> E["Rank Features by MI Score"]
    E --> F[Select Top K]
```

### 递归特征消除（RFE）

RFE 是一种包装方法。它使用模型自身的特征重要性进行迭代剪枝：

1. 用所有特征训练模型
2. 按重要性对特征排序（线性模型用系数，树模型用不纯度降低）
3. 移除最不重要的特征
4. 重复，直到剩下目标数量的特征

```mermaid
flowchart TD
    A["Start: All N Features"] --> B["Train Model"]
    B --> C["Rank Feature Importances"]
    C --> D["Remove Least Important"]
    D --> E{"Features == Target Count?"}
    E -->|No| B
    E -->|Yes| F["Return Selected Features"]
```

RFE 会考虑特征交互，因为模型会同时看到所有剩余特征。移除一个特征会改变其他特征的重要性。这使它比过滤方法更彻底。

代价是：你要训练模型 N - target 次。如果有 500 个特征，目标是 10 个特征，那就是 490 次训练。对于昂贵模型，这会很慢。你可以通过每轮移除多个特征来加速（例如每轮移除底部 10%）。

### L1（Lasso）正则化

L1 正则化把权重绝对值加入损失函数：

```text
loss = prediction_error + alpha * sum(|w_i|)
```

alpha 参数控制特征被剪掉的激进程度。alpha 越高，越多权重会变成严格的零。

为什么会严格为零？L1 惩罚在权重空间中形成菱形约束区域。最优解倾向于落在菱形的角上，而这些角会让一个或多个权重为零。L2 正则化（ridge）形成圆形约束，权重会缩小，但很少正好变成零。

这就是嵌入式特征选择：模型在训练过程中学习应该忽略哪些特征。权重为零的特征等价于被移除。

优势：只需一次训练；能处理相关特征（选择其中一个，把其他置零）；大多数线性模型实现都内置支持。

局限：只适用于线性模型。无法捕捉非线性的特征重要性。

### 基于树的特征重要性

决策树及其集成模型（随机森林、梯度提升）天然会给特征排序。每次分裂都会降低不纯度（分类任务中是 Gini 或 entropy，回归任务中是方差）。带来更大不纯度降低的特征更重要。

对含有 T 棵树的随机森林：

```text
importance(feature_j) = (1/T) * sum over all trees of
    sum over all nodes splitting on feature_j of
        (n_samples * impurity_decrease)
```

这会给每个特征一个归一化的重要性分数。它能自动处理非线性关系和特征交互。

注意：基于树的重要性会偏向拥有大量唯一值的特征（高基数）。随机 ID 列会显得重要，因为它可以完美分裂每个样本。用 permutation importance 做一次 sanity check。

### 置换重要性

一种模型无关的方法：

1. 训练模型，并在验证数据上记录基线性能
2. 对每个特征：随机打乱它的取值，测量性能下降
3. 下降越大，该特征越重要

如果打乱某个特征不会损害性能，模型就不依赖它。如果性能崩塌，这个特征就是关键特征。

置换重要性避免了基于树的重要性的基数偏差。但它很慢：每个特征都需要一次完整评估，而且通常要重复多次才能稳定。

### 对比表

| 方法 | 类型 | 速度 | 非线性 | 特征交互 |
|--------|------|-------|-----------|---------------------|
| 方差阈值 | 过滤 | 非常快 | 否 | 否 |
| 互信息 | 过滤 | 快 | 是 | 否 |
| 相关性过滤 | 过滤 | 快 | 否 | 否 |
| RFE | 包装 | 慢 | 取决于模型 | 是 |
| L1 / Lasso | 嵌入式 | 快 | 否（线性） | 否 |
| 树重要性 | 嵌入式 | 中等 | 是 | 是 |
| 置换重要性 | 模型无关 | 慢 | 是 | 是 |

### 决策流程图

```mermaid
flowchart TD
    A[Start: Feature Selection] --> B{How many features?}
    B -->|"< 50"| C["Start with variance threshold + mutual information"]
    B -->|"50-500"| D["Variance threshold, then L1 or tree importance"]
    B -->|"> 500"| E["Variance threshold, then mutual info filter, then RFE on survivors"]

    C --> F{Using linear model?}
    D --> F
    E --> F

    F -->|Yes| G["L1 regularization for final selection"]
    F -->|No - trees| H["Tree importance + permutation importance"]
    F -->|No - other| I["RFE with your model"]

    G --> J[Validate: compare selected vs all features]
    H --> J
    I --> J

    J --> K{Performance improved?}
    K -->|Yes| L["Ship with selected features"]
    K -->|No| M["Try different method or keep all features"]
```

## 动手实现

### Step 1: 生成具有已知特征结构的合成数据

```python
import numpy as np


def make_feature_selection_data(n_samples=500, seed=42):
    rng = np.random.RandomState(seed)

    x1 = rng.randn(n_samples)
    x2 = rng.randn(n_samples)
    x3 = rng.randn(n_samples)
    x4 = x1 + 0.1 * rng.randn(n_samples)
    x5 = x2 + 0.1 * rng.randn(n_samples)

    informative = np.column_stack([x1, x2, x3, x4, x5])

    correlated = np.column_stack([
        x1 * 0.9 + 0.1 * rng.randn(n_samples),
        x2 * 0.8 + 0.2 * rng.randn(n_samples),
        x3 * 0.7 + 0.3 * rng.randn(n_samples),
        x1 * 0.5 + x2 * 0.5 + 0.1 * rng.randn(n_samples),
        x2 * 0.6 + x3 * 0.4 + 0.1 * rng.randn(n_samples),
    ])

    noise = rng.randn(n_samples, 10) * 0.5

    X = np.hstack([informative, correlated, noise])
    y = (2 * x1 - 1.5 * x2 + x3 + 0.5 * rng.randn(n_samples) > 0).astype(int)

    feature_names = (
        [f"info_{i}" for i in range(5)]
        + [f"corr_{i}" for i in range(5)]
        + [f"noise_{i}" for i in range(10)]
    )

    return X, y, feature_names
```

我们知道真实结构：特征 0-4 是信息特征（其中 3 和 4 是 0 和 1 的相关副本），特征 5-9 与信息特征相关，特征 10-19 是纯噪声。好的选择方法应该把 0-4 排在最前，把 10-19 排在最后。

### Step 2: 方差阈值

```python
def variance_threshold(X, threshold=0.01):
    variances = np.var(X, axis=0)
    mask = variances > threshold
    return mask, variances
```

### Step 3: 互信息（离散版）

```python
def discretize(x, n_bins=10):
    min_val, max_val = x.min(), x.max()
    if max_val == min_val:
        return np.zeros_like(x, dtype=int)
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    binned = np.digitize(x, bin_edges[1:-1])
    return binned


def mutual_information(X, y, n_bins=10):
    n_samples, n_features = X.shape
    mi_scores = np.zeros(n_features)

    y_vals, y_counts = np.unique(y, return_counts=True)
    p_y = y_counts / n_samples

    for f in range(n_features):
        x_binned = discretize(X[:, f], n_bins)
        x_vals, x_counts = np.unique(x_binned, return_counts=True)
        p_x = dict(zip(x_vals, x_counts / n_samples))

        mi = 0.0
        for xv in x_vals:
            for yi, yv in enumerate(y_vals):
                joint_mask = (x_binned == xv) & (y == yv)
                p_xy = np.sum(joint_mask) / n_samples
                if p_xy > 0:
                    mi += p_xy * np.log(p_xy / (p_x[xv] * p_y[yi]))
        mi_scores[f] = mi

    return mi_scores
```

### Step 4: 递归特征消除

```python
def simple_logistic_importance(X, y, lr=0.1, epochs=100):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        pred = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        error = pred - y
        w -= lr * (X.T @ error) / n_samples
        b -= lr * np.mean(error)

    return w, b


def rfe(X, y, n_features_to_select=5, lr=0.1, epochs=100):
    n_total = X.shape[1]
    remaining = list(range(n_total))
    rankings = np.ones(n_total, dtype=int)
    rank = n_total

    while len(remaining) > n_features_to_select:
        X_subset = X[:, remaining]
        w, _ = simple_logistic_importance(X_subset, y, lr, epochs)
        importances = np.abs(w)

        least_idx = np.argmin(importances)
        original_idx = remaining[least_idx]
        rankings[original_idx] = rank
        rank -= 1
        remaining.pop(least_idx)

    for idx in remaining:
        rankings[idx] = 1

    selected_mask = rankings == 1
    return selected_mask, rankings
```

### Step 5: L1 特征选择

```python
def soft_threshold(w, alpha):
    return np.sign(w) * np.maximum(np.abs(w) - alpha, 0)


def l1_feature_selection(X, y, alpha=0.1, lr=0.01, epochs=500):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        pred = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        error = pred - y

        gradient_w = (X.T @ error) / n_samples
        gradient_b = np.mean(error)

        w -= lr * gradient_w
        w = soft_threshold(w, lr * alpha)
        b -= lr * gradient_b

    selected_mask = np.abs(w) > 1e-6
    return selected_mask, w
```

### Step 6: 基于树的重要性（简单决策树）

```python
def gini_impurity(y):
    if len(y) == 0:
        return 0.0
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)


def best_split(X, y, feature_idx):
    values = np.unique(X[:, feature_idx])
    if len(values) <= 1:
        return None, -1.0

    best_threshold = None
    best_gain = -1.0
    parent_gini = gini_impurity(y)
    n = len(y)

    for i in range(len(values) - 1):
        threshold = (values[i] + values[i + 1]) / 2.0
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)

        if n_left == 0 or n_right == 0:
            continue

        gain = parent_gini - (n_left / n) * gini_impurity(y[left_mask]) - (n_right / n) * gini_impurity(y[right_mask])

        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

    return best_threshold, best_gain


def tree_importance(X, y, n_trees=50, max_depth=5, seed=42):
    rng = np.random.RandomState(seed)
    n_samples, n_features = X.shape
    importances = np.zeros(n_features)

    for _ in range(n_trees):
        sample_idx = rng.choice(n_samples, size=n_samples, replace=True)
        feature_subset = rng.choice(n_features, size=max(1, int(np.sqrt(n_features))), replace=False)

        X_boot = X[sample_idx]
        y_boot = y[sample_idx]

        tree_imp = _build_tree_importance(X_boot, y_boot, feature_subset, max_depth)
        importances += tree_imp

    total = importances.sum()
    if total > 0:
        importances /= total

    return importances


def _build_tree_importance(X, y, feature_subset, max_depth, depth=0):
    n_features = X.shape[1]
    importances = np.zeros(n_features)

    if depth >= max_depth or len(np.unique(y)) <= 1 or len(y) < 4:
        return importances

    best_feature = None
    best_threshold = None
    best_gain = -1.0

    for f in feature_subset:
        threshold, gain = best_split(X, y, f)
        if gain > best_gain:
            best_gain = gain
            best_feature = f
            best_threshold = threshold

    if best_feature is None or best_gain <= 0:
        return importances

    importances[best_feature] += best_gain * len(y)

    left_mask = X[:, best_feature] <= best_threshold
    right_mask = ~left_mask

    importances += _build_tree_importance(X[left_mask], y[left_mask], feature_subset, max_depth, depth + 1)
    importances += _build_tree_importance(X[right_mask], y[right_mask], feature_subset, max_depth, depth + 1)

    return importances
```

### Step 7: 运行所有方法并比较

代码文件会在同一个合成数据集上运行全部五种方法，并打印一张对比表，展示每种方法选择了哪些特征。

## 实际使用

在 scikit-learn 中，特征选择可以内置进流水线：

```python
from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_classif,
    RFE,
    SelectFromModel,
)
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

vt = VarianceThreshold(threshold=0.01)
X_filtered = vt.fit_transform(X)

mi_scores = mutual_info_classif(X, y)
top_k = np.argsort(mi_scores)[-10:]

rfe_selector = RFE(LogisticRegression(), n_features_to_select=10)
rfe_selector.fit(X, y)
X_rfe = rfe_selector.transform(X)

lasso_selector = SelectFromModel(Lasso(alpha=0.01))
lasso_selector.fit(X, y)
X_lasso = lasso_selector.transform(X)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)
importances = rf.feature_importances_
```

从零实现会准确展示每种方法内部发生了什么。方差阈值只是计算 `var(X, axis=0)` 并应用 mask。互信息是在列联表里统计联合频率和边缘频率。RFE 是训练、排序、剪枝的循环。L1 是带 soft-thresholding 步骤的梯度下降。树重要性会累积分裂带来的不纯度降低。没有魔法，只有统计和循环。

sklearn 版本会增加稳健性（例如 `mutual_info_classif` 使用 k-NN 密度估计而不是分箱）、速度（C 实现）以及流水线集成能力。

## 交付成果

本课产出：
- `outputs/skill-feature-selector.md` -- 用于选择合适特征选择方法的快速参考决策树

## 练习

1. **前向选择**：实现 RFE 的反向过程。从零个特征开始。每一步加入能最大幅提升模型性能的特征。当继续加入特征不再有帮助时停止。将所选特征与 RFE 结果对比。哪个更快？哪个效果更好？

2. **稳定性选择**：运行 L1 特征选择 50 次，每次使用数据的随机 80% 子样本，并稍微改变 alpha 值。统计每个特征被选中的频率。被选中次数超过 80% 的特征是“稳定”的。将稳定特征与单次 L1 选择对比。哪种更可靠？

3. **多重共线性检测**：计算所有特征的相关矩阵。实现一个函数：给定相关性阈值（例如 0.9），从每一对高度相关的特征中移除一个（保留与目标互信息更高的那个）。在合成数据集上测试，并验证它会移除冗余的相关特征。

4. **特征选择流水线**：把方差阈值、互信息过滤器和 RFE 串成单条流水线。先移除近零方差特征，再按互信息保留前 50%，最后在幸存特征上运行 RFE。将这条流水线与直接在所有特征上运行 RFE 对比。流水线是否更快？是否同样准确？

5. **从零实现置换重要性**：实现 permutation importance。对每个特征，打乱它的取值 10 次，测量 F1 分数的平均下降。将排序与基于树的重要性对比。找到它们不一致的情况，并解释原因（提示：相关特征）。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| 过滤方法 | “独立给特征打分” | 一种不训练模型、只用统计度量给特征排序的特征选择方法，会孤立评估每个特征 |
| 包装方法 | “用模型挑特征” | 一种通过训练模型评估特征子集，并以模型性能作为选择标准的特征选择方法 |
| 嵌入式方法 | “模型在训练期间选择特征” | 作为模型拟合过程一部分发生的特征选择，例如 L1 正则化把权重推到零 |
| 互信息 | “一个变量告诉你另一个变量多少信息” | 给定 X 后，对 Y 的不确定性减少量；能捕捉线性和非线性依赖 |
| 递归特征消除 | “训练、排序、剪枝、重复” | 一种迭代式包装方法：训练模型，移除最不重要的特征，然后重复直到达到目标数量 |
| L1 / Lasso 正则化 | “杀掉特征的惩罚项” | 把权重绝对值之和加入损失函数，从而把不重要特征的权重推到严格为零 |
| 方差阈值 | “移除常量特征” | 丢弃样本间方差低于指定阈值的特征，过滤掉不携带信息的特征 |
| 特征重要性 | “哪些特征最重要” | 表示每个特征对模型预测贡献程度的分数，可由分裂增益（树）或系数大小（线性模型）计算 |
| 置换重要性 | “打乱并测量损伤” | 通过随机打乱每个特征的取值，并测量模型性能下降来评估特征重要性 |
| 维度灾难 | “特征太多，数据不够” | 添加特征会让特征空间体积指数级增长，使数据稀疏、距离失去意义的现象 |

## 延伸阅读

- [An Introduction to Variable and Feature Selection (Guyon & Elisseeff, 2003)](https://jmlr.org/papers/v3/guyon03a.html) -- 特征选择方法的奠基性综述，至今仍被广泛引用
- [scikit-learn Feature Selection Guide](https://scikit-learn.org/stable/modules/feature_selection.html) -- 过滤、包装和嵌入式方法的实用参考，包含代码示例
- [Stability Selection (Meinshausen & Buhlmann, 2010)](https://arxiv.org/abs/0809.2932) -- 将子采样与特征选择结合，用于获得稳健、可复现的结果
- [Beware Default Random Forest Importances (Strobl et al., 2007)](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-25) -- 展示基于树的重要性中的基数偏差，并提出条件重要性作为替代方案
