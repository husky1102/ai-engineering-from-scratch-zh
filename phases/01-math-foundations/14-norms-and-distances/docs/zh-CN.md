# 范数与距离

> 你的距离函数定义了什么叫“相似”。选错了，后面的所有东西都会坏掉。

**类型：** 构建
**语言：** Python
**先修：** Phase 1, Lessons 01（线性代数直觉）、02（向量、矩阵与运算）
**时间：** ~90 分钟

## 学习目标

- 从零实现 L1、L2、cosine、Mahalanobis、Jaccard 和 edit distance 函数
- 为给定的 ML 任务选择合适的距离度量，并解释其他选择为什么会失败
- 将 L1 与 L2 范数连接到 LASSO 和 Ridge 正则化，以及它们的几何约束区域
- 演示同一个数据集在不同度量下会产生不同的最近邻

## 要解决的问题

你有两个向量。它们可能是词嵌入，可能是用户画像，也可能是像素数组。你需要知道：它们有多接近？

答案完全取决于你选择哪个距离函数。两个数据点在一种度量下可能是最近邻，在另一种度量下却相距很远。你的 KNN 分类器、推荐引擎、向量数据库、聚类算法、损失函数，都会依赖这个选择。选错了，模型就会优化错误的东西。

不存在通用的最佳距离。L2 适合空间数据。Cosine similarity 主导 NLP。Jaccard 处理集合。Edit distance 处理字符串。Mahalanobis 会考虑相关性。Wasserstein 会移动概率质量。每一种距离都编码了关于“相似”含义的不同假设。

本课会从零构建每一种主要距离函数，说明什么时候该使用哪一种工具，并演示同一批数据如何因为度量不同而产生完全不同的最近邻。

## 核心概念

### 范数：度量向量大小

范数度量向量的“大小”。两个向量之间的每个距离函数，都可以写成它们差值的范数：d(a, b) = ||a - b||。所以理解范数，就是在理解距离。

### L1 范数（Manhattan distance）

L1 范数会把所有分量的绝对值相加。

```text
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

它被称为 Manhattan distance，因为它度量的是在城市网格中行走的距离：你只能沿坐标轴移动，不能走对角线。

```text
Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

On a grid, you walk 3 blocks east and 4 blocks north.
```

什么时候使用 L1：
- 高维稀疏数据（文本特征、one-hot 编码）
- 当你希望对异常值更稳健时（单个巨大的差异不会主导结果）
- 特征选择问题（L1 正则化会促进稀疏性）

与 L1 正则化（Lasso）的关系：把 ||w||_1 加到损失函数中，会惩罚权重绝对值之和。这会把小权重推到恰好为零，从而执行自动特征选择。L1 惩罚会在权重空间中形成菱形约束区域，而菱形的角落位于坐标轴上，也就是某些权重为零的位置。

与损失函数的关系：Mean Absolute Error (MAE) 是预测值与目标值之间 L1 距离的平均值。它会线性惩罚所有误差，因此相比 MSE 对异常值更稳健。

### L2 范数（Euclidean distance）

L2 范数是直线距离。它等于各分量平方和的平方根。

```text
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

这就是你在几何课上学过的距离。n 维空间中的 Pythagoras。

```text
Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

The straight line, cutting diagonally through the grid.
```

什么时候使用 L2：
- 低维到中维的连续数据
- 当特征尺度彼此可比时
- 物理距离（空间数据、传感器读数）
- 像素级图像相似度

与 L2 正则化（Ridge）的关系：把 ||w||_2^2 加到损失函数中，会惩罚较大的权重。不同于 L1，它不会把权重推到零。它会按比例把所有权重收缩到接近零。L2 惩罚会形成圆形约束区域，因此坐标轴上没有角落。权重会变小，但很少恰好为零。

与损失函数的关系：Mean Squared Error (MSE) 是 L2 距离平方的平均值。平方会比小误差更重地惩罚大误差。

```text
MAE (L1 loss):  |y - y_hat|         Linear penalty. Robust to outliers.
MSE (L2 loss):  (y - y_hat)^2       Quadratic penalty. Sensitive to outliers.
```

### Lp 范数：通用家族

L1 和 L2 是 Lp 范数的特例：

```text
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

不同的 p 值会产生不同形状的“单位球”（所有距离原点为 1 的点的集合）：

```text
p=1:    Diamond shape      (corners on axes)
p=2:    Circle/sphere      (the usual round ball)
p=3:    Superellipse       (rounded square)
p=inf:  Square/hypercube   (flat sides along axes)
```

### L-infinity 范数（Chebyshev distance）

当 p 趋近于无穷大时，Lp 范数会收敛到最大绝对分量。

```text
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

两点之间的距离由它们差异最大的那个维度决定。其他所有维度都会被忽略。

```text
Point A = (1, 1)
Point B = (4, 5)

L-inf distance = max(|4-1|, |5-1|) = max(3, 4) = 4
```

什么时候使用 L-infinity：
- 当任意单个维度上的最坏偏差很重要时
- 棋盘游戏（国际象棋中的王按 L-infinity 移动：任意方向走一步的代价都是 1）
- 制造公差（每个维度都必须在规格范围内）

### Cosine similarity 与 Cosine distance

Cosine similarity 度量两个向量之间的夹角，忽略它们的大小。

```text
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

它的范围从 -1（方向相反）到 +1（方向相同）。互相垂直的向量 cosine similarity 为 0。

Cosine distance 会把它转换成距离：cosine_distance = 1 - cosine_similarity。范围从 0（方向相同）到 2（方向相反）。

```text
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

为什么 cosine 主导 NLP 和 embeddings：在文本中，文档长度不应该影响相似度。一篇关于猫的文档即使比另一篇关于猫的文档长两倍，也仍然应该是“相似”的。Cosine similarity 忽略大小（长度），只关心方向。两个词分布相同但长度不同的文档会指向同一个方向，并得到 1.0 的 cosine similarity。

什么时候使用 cosine similarity：
- 文本相似度（TF-IDF 向量、word embeddings、sentence embeddings）
- 任何“大小是噪声、方向是信号”的领域
- 推荐系统（用户偏好向量）
- Embedding search（向量数据库几乎总是使用 cosine 或 dot product）

### Dot product similarity 与 Cosine similarity

两个向量的 dot product 是：

```text
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(angle)
```

Cosine similarity 是用两个向量大小归一化后的 dot product。当两个向量已经 unit-normalized（大小 = 1）时，dot product 和 cosine similarity 完全相同。

```text
If ||a|| = 1 and ||b|| = 1:
    a . b = cos(angle between a and b)
```

什么时候它们会不同：dot product 包含大小信息。大小更大的向量会得到更高的 dot product 分数。在某些检索系统中，这很重要，因为你可能希望“更热门”的条目排名更靠前。大小会作为隐式的质量或重要性信号。

```text
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Both agree on direction, but dot product also reflects magnitude.
```

实践中：
- 当你想要纯粹的方向相似度时，使用 cosine similarity
- 当大小承载有意义的信息时，使用 dot product
- 许多向量数据库（Pinecone、Weaviate、Qdrant）允许你在二者之间选择
- 如果你的 embeddings 已经 L2-normalized，那么选择哪一个没有区别

### Mahalanobis distance

Euclidean distance 会平等对待所有维度。但如果你的特征相关，或者尺度不同，L2 会给出误导性的结果。

Mahalanobis distance 会考虑数据的协方差结构。

```text
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

其中 S 是数据的协方差矩阵。

直观地说：Mahalanobis distance 会先对数据去相关并归一化（whitening），然后在变换后的空间中计算 L2 距离。如果 S 是单位矩阵（特征不相关、方差为 1），Mahalanobis distance 就会退化为 Euclidean distance。

```text
Example: height and weight are correlated.
Someone 6'2" and 180 lbs is not unusual.
Someone 5'0" and 180 lbs is unusual.

Euclidean distance might say they are equally far from the mean.
Mahalanobis distance correctly identifies the second as an outlier
because it accounts for the height-weight correlation.
```

什么时候使用 Mahalanobis distance：
- 异常值检测（Mahalanobis distance 离均值很远的点就是异常值）
- 当特征有不同尺度且存在相关性时做分类
- 当你有足够数据来估计可靠的协方差矩阵时
- 制造业质量控制（多变量过程监控）

### Jaccard similarity（用于集合）

Jaccard similarity 度量两个集合之间的重叠。

```text
J(A, B) = |A intersect B| / |A union B|
```

它的范围从 0（没有重叠）到 1（集合完全相同）。Jaccard distance = 1 - Jaccard similarity。

```text
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

Intersection = {cat, fish}         size = 2
Union = {cat, dog, fish, bird, snake}  size = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

什么时候使用 Jaccard：
- 比较标签、类别或特征集合
- 基于词是否出现的文档相似度（而不是词频）
- 近重复检测（用 MinHash 近似 Jaccard）
- 比较二值特征向量（presence/absence 数据）
- 评估分割模型（Intersection over Union = Jaccard）

### Edit distance（Levenshtein distance）

Edit distance 计算把一个字符串转换成另一个字符串所需的最少单字符操作次数。操作包括：插入、删除或替换。

```text
"kitten" -> "sitting"

kitten -> sitten  (substitute k -> s)
sitten -> sittin  (substitute e -> i)
sittin -> sitting (insert g)

Edit distance = 3
```

它使用动态规划计算。填充一个矩阵，其中条目 (i, j) 表示字符串 A 的前 i 个字符与字符串 B 的前 j 个字符之间的 edit distance。

```text
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

什么时候使用 edit distance：
- 拼写检查与纠错
- DNA 序列比对（带加权操作）
- 模糊字符串匹配
- 杂乱文本数据去重

### KL divergence（不是距离，但经常被当作距离使用）

KL divergence 度量一个概率分布与另一个概率分布有多不同。它已在 Lesson 09 覆盖过，但属于本节讨论，因为人们经常把它当作“距离”使用，尽管它并不是距离。

```text
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

关键性质：KL divergence 不是对称的。

```text
D_KL(P || Q) != D_KL(Q || P)
```

这意味着它不满足距离度量的基本要求。它也不满足三角不等式。它是 divergence，不是 distance。

Forward KL (D_KL(P || Q)) 是“mean-seeking”：Q 会试图覆盖 P 的所有模式。
Reverse KL (D_KL(Q || P)) 是“mode-seeking”：Q 会专注于 P 的单个模式。

你会在这些地方看到 KL divergence：
- VAEs（ELBO 中的 KL 项会把 latent distribution 推向 prior）
- Knowledge distillation（student 试图匹配 teacher 的分布）
- RLHF（KL 惩罚让 fine-tuned model 保持接近 base model）
- Policy gradient methods（约束 policy updates）

### Wasserstein distance（Earth Mover's Distance）

Wasserstein distance 度量把一个概率分布转换成另一个概率分布所需的最小“功”。可以把它想成：如果一个分布是一堆土，另一个分布是一个坑，你需要移动多少土、移动多远？

```text
W(P, Q) = inf over all transport plans gamma of E[d(x, y)]
```

对于 1D 分布，它可以简化为累积分布函数之差的绝对值积分：

```text
W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
```

为什么 Wasserstein 重要：
- 它是真正的 metric（对称，满足三角不等式）
- 即使分布不重叠，它也能提供梯度（KL divergence 会变成无穷大）
- 这个性质让它成为 Wasserstein GANs (WGANs) 的核心，解决了原始 GANs 的训练不稳定问题

```text
Distributions with no overlap:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: infinity (log of zero)
Wasserstein: 4 (move all mass 4 bins)

Wasserstein gives a meaningful gradient. KL does not.
```

什么时候使用 Wasserstein：
- GAN 训练（WGAN、WGAN-GP）
- 比较可能不重叠的分布
- Optimal transport 问题
- 图像检索（比较颜色直方图）

### 为什么不同任务需要不同距离

| 任务 | 最佳距离 | 原因 |
|------|--------------|-----|
| 文本相似度 | Cosine | 大小是噪声，方向是含义 |
| 图像像素比较 | L2 | 空间关系重要，特征尺度可比 |
| 稀疏高维特征 | L1 | 稳健，不会放大罕见的大差异 |
| 集合重叠（标签、类别） | Jaccard | 数据天然是集合值，而不是向量值 |
| 字符串匹配 | Edit distance | 操作映射到人类编辑直觉 |
| 异常值检测 | Mahalanobis | 考虑特征相关性和尺度 |
| 比较分布 | KL divergence | 度量用 Q 替代 P 时损失的信息 |
| GAN 训练 | Wasserstein | 即使分布不重叠也能提供梯度 |
| Embeddings（向量数据库） | Cosine 或 dot product | Embeddings 被训练为在方向中编码含义 |
| 推荐 | Dot product | 大小可以编码流行度或置信度 |
| DNA 序列 | Weighted edit distance | 替换代价因核苷酸对而异 |
| 制造业 QC | L-infinity | 任意维度上的最坏偏差都很重要 |

### 与损失函数的关系

损失函数就是应用在预测值与目标值之间的距离函数。

```text
Loss function       Distance it uses       Behavior
MSE                 L2 squared             Penalizes large errors heavily
MAE                 L1                     Penalizes all errors equally
Huber loss          L1 for large errors,   Best of both: robust to outliers,
                    L2 for small errors    smooth gradient near zero
Cross-entropy       KL divergence          Measures distribution mismatch
Hinge loss          max(0, margin - d)     Only penalizes below margin
Triplet loss        L2 (typically)         Pulls positives close, pushes
                                           negatives away
Contrastive loss    L2                     Similar pairs close, dissimilar
                                           pairs beyond margin
```

### 与正则化的关系

正则化会向损失函数中添加一个作用在权重上的范数惩罚。

```text
L1 regularization (Lasso):   loss + lambda * ||w||_1
  -> Sparse weights. Some weights become exactly zero.
  -> Automatic feature selection.
  -> Solution has corners (non-differentiable at zero).

L2 regularization (Ridge):   loss + lambda * ||w||_2^2
  -> Small weights. All weights shrink toward zero.
  -> No feature selection (nothing goes to exactly zero).
  -> Smooth solution everywhere.

Elastic Net:                  loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Combines sparsity of L1 with stability of L2.
  -> Groups of correlated features are kept or dropped together.
```

为什么 L1 会产生稀疏性而 L2 不会：想象二维权重空间中的约束区域。L1 是菱形，L2 是圆。损失函数的等高线（椭圆）最可能先接触菱形的角落，在那里某个权重为零。它们接触圆形时会落在一个光滑点上，在那里两个权重都非零。

### 最近邻搜索

每个距离函数都隐含一个最近邻搜索问题：给定一个查询点，在数据集中找到最接近的点。

在包含 n 个点、每个点 d 维的数据集中，精确最近邻搜索每次查询的复杂度是 O(n * d)。对于大型数据集来说，这太慢了。

Approximate Nearest Neighbor (ANN) 算法会用少量准确率换取巨大的速度提升：

```text
Algorithm         Approach                      Used by
KD-trees          Axis-aligned space partition   scikit-learn (low-dim)
Ball trees        Nested hyperspheres            scikit-learn (medium-dim)
LSH               Random hash projections        Near-duplicate detection
HNSW              Hierarchical navigable         FAISS, Qdrant, Weaviate
                  small-world graph
IVF               Inverted file index with       FAISS (billion-scale)
                  cluster-based search
Product quant.    Compress vectors, search       FAISS (memory-constrained)
                  in compressed space
```

HNSW (Hierarchical Navigable Small World) 是现代向量数据库中的主流算法。它构建一个多层图，每个节点连接到它的近似最近邻。搜索从顶层开始（稀疏、长跳），并下降到底层（稠密、短跳）。

## 动手实现

### Step 1：所有范数与距离函数

完整实现见 `code/distances.py`。每个函数都只用基础 Python 数学从零构建。

### Step 2：同一批数据，不同距离，不同邻居

`distances.py` 中的 demo 会创建一个数据集，选择一个查询点，并展示最近邻如何随距离度量变化而变化。L1 下“最近”的点，在 L2 或 cosine 下可能并不是最近的。

### Step 3：Embedding 相似度搜索

代码包含一个模拟 embedding similarity search，它会使用 cosine similarity 与 L2 distance 为查询寻找最相似的“documents”，展示排序可能如何不同。

## 实际使用

最常见的实际用途：在向量数据库中寻找相似条目。

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

当你调用 `model.encode(text)` 然后搜索向量数据库时，底层发生的就是这些事。Embedding model 会把文本映射到向量。向量数据库会在你的查询向量和每个已存储向量之间计算 cosine similarity（或 dot product），并使用 ANN 算法避免检查所有向量。

## 练习

1. 计算 (1, 2, 3) 与 (4, 0, 6) 之间的 L1、L2 和 L-infinity 距离。验证对任意点对始终有 L-inf <= L2 <= L1。证明为什么这个顺序必然成立。

2. 创建两个向量，使它们的 cosine similarity 很高（> 0.9），但 L2 distance 很大（> 10）。从几何角度解释发生了什么。然后创建两个向量，使它们的 cosine similarity 很低（< 0.3），但 L2 distance 很小（< 0.5）。

3. 实现一个函数，它接收一个数据集和一个查询点，并返回 L1、L2、cosine 和 Mahalanobis distance 下的最近邻。找出一个数据集，让四种度量对哪个点最近全部意见不一。

4. 使用 CDF 方法手算 [0.5, 0.5, 0, 0] 与 [0, 0, 0.5, 0.5] 之间的 Wasserstein distance。再计算 [0.25, 0.25, 0.25, 0.25] 与 [0, 0, 0.5, 0.5] 之间的距离。哪一个更大，为什么？

5. 为近似 Jaccard similarity 实现 MinHash。生成 100 个随机集合，计算所有集合对的精确 Jaccard，并与使用 50、100 和 200 个 hash functions 的 MinHash 近似结果比较。绘制近似误差。

## 关键术语

| 术语 | 人们常说 | 它真正的含义 |
|------|----------------|----------------------|
| Norm | “向量的大小” | 一个把向量映射到非负标量的函数，满足三角不等式、绝对齐次性，并且只有零向量的值为零 |
| L1 norm | “Manhattan distance” | 分量绝对值之和。会在优化中产生稀疏性。对异常值稳健 |
| L2 norm | “Euclidean distance” | 各分量平方和的平方根。Euclidean space 中的直线距离 |
| Lp norm | “Generalized norm” | 绝对分量 p 次幂之和的 p 次根。L1 和 L2 是特例 |
| L-infinity norm | “Max norm” 或 “Chebyshev distance” | 最大绝对分量值。Lp 在 p 趋近无穷大时的极限 |
| Cosine similarity | “向量之间的夹角” | 用两个向量大小归一化后的 dot product。范围从 -1 到 +1。忽略向量长度 |
| Cosine distance | “1 减 cosine similarity” | 把 cosine similarity 转换成距离。范围从 0 到 2 |
| Dot product | “未归一化的 cosine” | 分量逐项乘积之和。等于 cosine similarity 乘以两个向量的大小 |
| Mahalanobis distance | “考虑相关性的距离” | 在一个用数据协方差矩阵 whitening（去相关并归一化）后的空间中的 L2 距离 |
| Jaccard similarity | “集合重叠” | 交集大小除以并集大小。用于集合，不用于向量 |
| Edit distance | “Levenshtein distance” | 把一个字符串转换成另一个字符串所需的最少插入、删除和替换次数 |
| KL divergence | “分布之间的距离” | 不是真正的距离（不对称）。度量用 Q 编码 P 时多出来的信息位 |
| Wasserstein distance | “Earth mover's distance” | 把质量从一个分布运输到另一个分布所需的最小功。真正的 metric |
| Approximate nearest neighbor | “ANN search” | 比精确搜索快得多地寻找近似最近点的算法（HNSW、LSH、IVF） |
| HNSW | “向量数据库算法” | Hierarchical Navigable Small World graph。用于快速 approximate nearest neighbor search 的多层图 |
| L1 regularization | “Lasso” | 把权重的 L1 norm 加到损失中。把权重推到零（稀疏性） |
| L2 regularization | “Ridge” 或 “weight decay” | 把权重的 squared L2 norm 加到损失中。把权重收缩到接近零，但不产生稀疏性 |
| Elastic Net | “L1 + L2” | 结合 L1 和 L2 正则化。比单独使用其中任一种更好地处理相关特征组 |

## 延伸阅读

- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) - Meta 的 billion-scale ANN search 库
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875) - 将 Earth Mover's distance 引入 GANs 的论文
- [Locality-Sensitive Hashing (Indyk & Motwani, 1998)](https://dl.acm.org/doi/10.1145/276698.276876) - 基础 ANN algorithm
- [Efficient Estimation of Word Representations (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) - Word2Vec，cosine similarity 成为 embeddings 默认选择的重要背景
- [sklearn.neighbors documentation](https://scikit-learn.org/stable/modules/neighbors.html) - scikit-learn 中距离度量与 neighbor algorithms 的实用指南
