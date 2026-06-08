# 矩阵变换

> 矩阵是一台重塑空间的机器。理解它对每个点做了什么，你就理解了整个变换。

**类型：** 构建
**语言：** Python, Julia
**先修：** Phase 1，第 01-02 课（线性代数直觉、向量与矩阵运算）
**时间：** ~75 分钟

## 学习目标

- 构造旋转、缩放、剪切和反射矩阵，并将它们应用到 2D 与 3D 点上
- 通过矩阵乘法组合多个变换，并验证顺序确实重要
- 从特征方程计算 2x2 矩阵的 eigenvalue 和 eigenvector
- 解释为什么 eigenvalue 会决定 PCA 方向、RNN 稳定性以及谱聚类行为

## 要解决的问题

你读 PCA 时会看到“求协方差矩阵的 eigenvector”。你读模型稳定性时会看到“检查所有 eigenvalue 的模是否小于 1”。你读数据增强时会看到“应用一次随机旋转”。在你从几何上理解矩阵如何作用于空间之前，这些话都不会真正说得通。

矩阵不只是数字网格。它们是空间机器。旋转矩阵会旋转点。缩放矩阵会拉伸点。剪切矩阵会倾斜点。神经网络对数据施加的每一次变换，都是这些操作之一，或是它们的组合。本课会让这些操作变得具体可见。

## 核心概念

### 把变换看成矩阵

2D 中的每个线性变换都可以写成一个 2x2 矩阵。这个矩阵会准确告诉你基向量 [1, 0] 和 [0, 1] 最终落到哪里。其他一切都随之确定。

```mermaid
graph LR
    subgraph Before["Standard Basis"]
        e1["e1 = [1, 0] (along x)"]
        e2["e2 = [0, 1] (along y)"]
    end
    subgraph Transform["Matrix M"]
        M["M = columns are new basis vectors"]
    end
    subgraph After["After Transformation M"]
        e1p["e1' = new x-basis"]
        e2p["e2' = new y-basis"]
    end
    e1 --> M --> e1p
    e2 --> M --> e2p
```

### 旋转

按角度 theta 做一次 2D 旋转，会保持距离和角度不变。它让每个点沿着圆弧移动。

```mermaid
graph LR
    subgraph Before["Before Rotation"]
        A["A(2, 1)"]
        B["B(0, 2)"]
    end
    subgraph Rot["Rotate 45 degrees"]
        R["R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]"]
    end
    subgraph After["After Rotation"]
        Ap["A'(0.71, 2.12)"]
        Bp["B'(-1.41, 1.41)"]
    end
    A --> R --> Ap
    B --> R --> Bp
```

在 3D 中，你会围绕某个轴旋转。每个轴都有自己的旋转矩阵：

```text
Rz(theta) = | cos  -sin  0 |     Rotate around z-axis
            | sin   cos  0 |     (x-y plane spins, z stays)
            |  0     0   1 |

Rx(theta) = | 1   0     0    |   Rotate around x-axis
            | 0  cos  -sin   |   (y-z plane spins, x stays)
            | 0  sin   cos   |

Ry(theta) = |  cos  0  sin |     Rotate around y-axis
            |   0   1   0  |     (x-z plane spins, y stays)
            | -sin  0  cos |
```

### 缩放

缩放会沿每个轴独立地拉伸或压缩。

```mermaid
graph LR
    subgraph Before["Before Scaling"]
        A["A(2, 1)"]
        B["B(0, 2)"]
    end
    subgraph Scale["Scale sx=2, sy=0.5"]
        S["S = [[2, 0], [0, 0.5]]"]
    end
    subgraph After["After Scaling"]
        Ap["A'(4, 0.5)"]
        Bp["B'(0, 1)"]
    end
    A --> S --> Ap
    B --> S --> Bp
```

### 剪切

剪切会在保持一个轴固定的同时倾斜另一个轴。它会把矩形变成平行四边形。

```mermaid
graph LR
    subgraph Before["Before Shear"]
        A["A(1, 0)"]
        B["B(0, 1)"]
    end
    subgraph Shear["Shear in x, k=1"]
        Sh["Shx = [[1, k], [0, 1]]"]
    end
    subgraph After["After Shear"]
        Ap["A(1, 0) unchanged"]
        Bp["B'(1, 1) shifted"]
    end
    A --> Sh --> Ap
    B --> Sh --> Bp
```

剪切矩阵：
- `Shx = [[1, k], [0, 1]]` 会按 k * y 平移 x
- `Shy = [[1, 0], [k, 1]]` 会按 k * x 平移 y

### 反射

反射会把点沿某个轴或某条线镜像过去。

```mermaid
graph LR
    subgraph Before["Before Reflection"]
        A["A(2, 1)"]
    end
    subgraph Reflect["Reflect across y-axis"]
        R["[[-1, 0], [0, 1]]"]
    end
    subgraph After["After Reflection"]
        Ap["A'(-2, 1)"]
    end
    A --> R --> Ap
```

反射矩阵：
- 关于 y-axis 反射：`[[-1, 0], [0, 1]]`
- 关于 x-axis 反射：`[[1, 0], [0, -1]]`

### 组合：串联多个变换

先应用变换 A，再应用变换 B，等价于把它们的矩阵相乘：`result = B @ A @ point`。顺序很重要。先旋转再缩放，和先缩放再旋转，会得到不同结果。

```mermaid
graph LR
    subgraph Path1["Rotate 90 then Scale (2, 0.5)"]
        P1["(1, 0)"] -->|"Rotate 90"| P2["(0, 1)"] -->|"Scale"| P3["(0, 0.5)"]
    end
```

组合后：`S @ R = [[0, -2], [0.5, 0]]`

```mermaid
graph LR
    subgraph Path2["Scale (2, 0.5) then Rotate 90"]
        Q1["(1, 0)"] -->|"Scale"| Q2["(2, 0)"] -->|"Rotate 90"| Q3["(0, 2)"]
    end
```

组合后：`R @ S = [[0, -0.5], [2, 0]]`

结果不同。矩阵乘法不满足交换律。

### Eigenvalue 与 eigenvector

矩阵作用到大多数向量上时，都会改变它们的方向。Eigenvector 很特殊：矩阵只会缩放它们，从不会旋转它们。这个缩放因子就是 eigenvalue。

```text
A @ v = lambda * v

v is the eigenvector (direction that survives)
lambda is the eigenvalue (how much it stretches)

Example: A = | 2  1 |
             | 1  2 |

Eigenvector [1, 1] with eigenvalue 3:
  A @ [1,1] = [3, 3] = 3 * [1, 1]     (same direction, scaled by 3)

Eigenvector [1, -1] with eigenvalue 1:
  A @ [1,-1] = [1, -1] = 1 * [1, -1]  (same direction, unchanged)
```

这个矩阵会沿 [1, 1] 方向把空间拉伸 3 倍，并让 [1, -1] 保持不变。其他每个方向都是这两个方向的混合。

### 特征分解

如果一个矩阵有 n 个线性无关的 eigenvector，它就可以被分解：

```text
A = V @ D @ V^(-1)

V = matrix whose columns are eigenvectors
D = diagonal matrix of eigenvalues
V^(-1) = inverse of V

This says: rotate into eigenvector coordinates, scale along each axis, rotate back.
```

这句话的意思是：先旋转到 eigenvector 坐标系，沿每个轴缩放，再旋转回来。

### 为什么 eigenvalue 很重要

**PCA。** 协方差矩阵的 eigenvector 就是主成分。eigenvalue 会告诉你每个成分捕获了多少方差。按 eigenvalue 排序，保留前 k 个，你就得到了降维。

**稳定性。** 在循环网络和动力系统中，模大于 1 的 eigenvalue 会让输出爆炸。模小于 1 的 eigenvalue 会让输出消失。这就是用一句话概括的梯度消失/爆炸问题。

**谱方法。** 图神经网络会使用邻接矩阵的 eigenvalue。谱聚类会使用 Laplacian 的 eigenvalue。eigenvector 会揭示图的结构。

### 行列式是体积缩放因子

变换矩阵的 determinant 会告诉你它把面积（2D）或体积（3D）缩放了多少。

```text
det = 1:   area preserved (rotation)
det = 2:   area doubled
det = 0:   space crushed to lower dimension (singular)
det = -1:  area preserved but orientation flipped (reflection)

| det(Rotation) | = 1        (always)
| det(Scale sx, sy) | = sx * sy
| det(Shear) | = 1           (area preserved)
| det(Reflection) | = -1     (orientation flipped)
```

## 动手实现

### 第 1 步：从零实现变换矩阵（Python）

```python
import math

def rotation_2d(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s], [s, c]]

def scaling_2d(sx, sy):
    return [[sx, 0], [0, sy]]

def shearing_2d(kx, ky):
    return [[1, kx], [ky, 1]]

def reflection_x():
    return [[1, 0], [0, -1]]

def reflection_y():
    return [[-1, 0], [0, 1]]

def mat_vec_mul(matrix, vector):
    return [
        sum(matrix[i][j] * vector[j] for j in range(len(vector)))
        for i in range(len(matrix))
    ]

def mat_mul(a, b):
    rows_a, cols_b = len(a), len(b[0])
    cols_a = len(a[0])
    return [
        [sum(a[i][k] * b[k][j] for k in range(cols_a)) for j in range(cols_b)]
        for i in range(rows_a)
    ]

point = [1.0, 0.0]
angle = math.pi / 4

rotated = mat_vec_mul(rotation_2d(angle), point)
print(f"Rotate (1,0) by 45 deg: ({rotated[0]:.4f}, {rotated[1]:.4f})")

scaled = mat_vec_mul(scaling_2d(2, 3), [1.0, 1.0])
print(f"Scale (1,1) by (2,3): ({scaled[0]:.1f}, {scaled[1]:.1f})")

sheared = mat_vec_mul(shearing_2d(1, 0), [1.0, 1.0])
print(f"Shear (1,1) kx=1: ({sheared[0]:.1f}, {sheared[1]:.1f})")

reflected = mat_vec_mul(reflection_y(), [2.0, 1.0])
print(f"Reflect (2,1) across y: ({reflected[0]:.1f}, {reflected[1]:.1f})")
```

### 第 2 步：组合多个变换

```python
R = rotation_2d(math.pi / 2)
S = scaling_2d(2, 0.5)

rotate_then_scale = mat_mul(S, R)
scale_then_rotate = mat_mul(R, S)

point = [1.0, 0.0]
result1 = mat_vec_mul(rotate_then_scale, point)
result2 = mat_vec_mul(scale_then_rotate, point)

print(f"Rotate 90 then scale: ({result1[0]:.2f}, {result1[1]:.2f})")
print(f"Scale then rotate 90: ({result2[0]:.2f}, {result2[1]:.2f})")
print(f"Same? {result1 == result2}")
```

### 第 3 步：从零计算 eigenvalue（2x2）

对于 2x2 矩阵 `[[a, b], [c, d]]`，eigenvalue 满足特征方程：`lambda^2 - (a+d)*lambda + (ad - bc) = 0`。

```python
def eigenvalues_2x2(matrix):
    a, b = matrix[0]
    c, d = matrix[1]
    trace = a + d
    det = a * d - b * c
    discriminant = trace ** 2 - 4 * det
    if discriminant < 0:
        real = trace / 2
        imag = (-discriminant) ** 0.5 / 2
        return (complex(real, imag), complex(real, -imag))
    sqrt_disc = discriminant ** 0.5
    return ((trace + sqrt_disc) / 2, (trace - sqrt_disc) / 2)

def eigenvector_2x2(matrix, eigenvalue):
    a, b = matrix[0]
    c, d = matrix[1]
    if abs(b) > 1e-10:
        v = [b, eigenvalue - a]
    elif abs(c) > 1e-10:
        v = [eigenvalue - d, c]
    else:
        if abs(a - eigenvalue) < 1e-10:
            v = [1, 0]
        else:
            v = [0, 1]
    mag = (v[0] ** 2 + v[1] ** 2) ** 0.5
    return [v[0] / mag, v[1] / mag]

A = [[2, 1], [1, 2]]
vals = eigenvalues_2x2(A)
print(f"Matrix: {A}")
print(f"Eigenvalues: {vals[0]:.4f}, {vals[1]:.4f}")

for val in vals:
    vec = eigenvector_2x2(A, val)
    result = mat_vec_mul(A, vec)
    scaled = [val * vec[0], val * vec[1]]
    print(f"  lambda={val:.1f}, v={[round(x,4) for x in vec]}")
    print(f"    A@v = {[round(x,4) for x in result]}")
    print(f"    l*v = {[round(x,4) for x in scaled]}")
```

### 第 4 步：把 determinant 看成体积缩放因子

```python
def det_2x2(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

print(f"det(rotation 45) = {det_2x2(rotation_2d(math.pi/4)):.4f}")
print(f"det(scale 2,3)   = {det_2x2(scaling_2d(2, 3)):.1f}")
print(f"det(shear kx=1)  = {det_2x2(shearing_2d(1, 0)):.1f}")
print(f"det(reflect y)   = {det_2x2(reflection_y()):.1f}")

singular = [[1, 2], [2, 4]]
print(f"det(singular)     = {det_2x2(singular):.1f}")
print("Singular: columns are proportional, space collapses to a line.")
```

## 实际使用

NumPy 会用优化过的例程处理所有这些操作。

```python
import numpy as np

theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

point = np.array([1.0, 0.0])
print(f"Rotate (1,0) by 45 deg: {R @ point}")

S = np.diag([2.0, 3.0])
composed = S @ R
print(f"Scale(2,3) after Rotate(45): {composed @ point}")

A = np.array([[2, 1], [1, 2]], dtype=float)
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"\nEigenvalues: {eigenvalues}")
print(f"Eigenvectors (columns):\n{eigenvectors}")

for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    print(f"  A @ v{i} = {A @ v}, lambda * v{i} = {lam * v}")

print(f"\ndet(R) = {np.linalg.det(R):.4f}")
print(f"det(S) = {np.linalg.det(S):.1f}")

B = np.array([[3, 1], [0, 2]], dtype=float)
vals, vecs = np.linalg.eig(B)
D = np.diag(vals)
V = vecs
reconstructed = V @ D @ np.linalg.inv(V)
print(f"\nEigendecomposition A = V @ D @ V^-1:")
print(f"Original:\n{B}")
print(f"Reconstructed:\n{reconstructed}")
```

### 使用 NumPy 做 3D 旋转

```python
def rotation_3d_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def rotation_3d_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

point_3d = np.array([1.0, 0.0, 0.0])
rotated_z = rotation_3d_z(np.pi / 2) @ point_3d
rotated_x = rotation_3d_x(np.pi / 2) @ point_3d

print(f"\n3D point: {point_3d}")
print(f"Rotate 90 around z: {np.round(rotated_z, 4)}")
print(f"Rotate 90 around x: {np.round(rotated_x, 4)}")
```

## 交付成果

本课会建立 PCA（Phase 2）和神经网络权重分析所需的几何基础。这里构建的 eigenvalue/eigenvector 代码，和生产级 ML 系统中驱动降维、谱聚类与稳定性分析的算法是同一种算法。

## 练习

1. 对一个单位正方形（四个角为 [0,0]、[1,0]、[1,1]、[0,1]）应用旋转、缩放和剪切。分别打印每次变换后的角点。验证旋转会保持角点之间的距离不变。

2. 使用特征方程手算矩阵 [[4, 2], [1, 3]] 的 eigenvalue。然后用你从零实现的函数和 NumPy 进行验证。

3. 创建三个变换的组合（旋转 30 度，按 [1.5, 0.8] 缩放，使用 kx=0.3 剪切），并将它应用到排列成圆形的 8 个点上。打印变换前后的坐标。计算组合矩阵的 determinant，并验证它等于各个单独 determinant 的乘积。

## 关键术语

| 术语 | 人们常说 | 它真正的意思 |
|------|----------|--------------|
| 旋转矩阵 | “把东西转起来” | 一种正交矩阵，会让点沿圆弧移动，同时保持距离和角度不变。determinant 总是 1。 |
| 缩放矩阵 | “把东西变大” | 一种对角矩阵，会沿每个轴独立地拉伸或压缩。determinant 是各缩放因子的乘积。 |
| 剪切矩阵 | “把东西倾斜” | 一种矩阵，会让一个坐标按另一个坐标的比例平移，把矩形变成平行四边形。determinant 是 1。 |
| 反射 | “把东西镜像过去” | 一种矩阵，会把空间沿某个轴或平面翻转。determinant 是 -1。 |
| 组合 | “做两件事” | 通过相乘变换矩阵来串联操作。顺序很重要：B @ A 表示先应用 A，再应用 B。 |
| Eigenvector | “特殊方向” | 一个只会被矩阵缩放、不会被旋转的方向。它是这个变换的指纹。 |
| Eigenvalue | “拉伸了多少” | 矩阵缩放其 eigenvector 的标量因子。可以是负数（翻转），也可以是复数（旋转）。 |
| 特征分解 | “把矩阵拆开” | 把矩阵写成 V @ D @ V^(-1)，从而分离出它的基本缩放方向和大小。 |
| Determinant | “来自矩阵的一个数字” | 变换缩放面积（2D）或体积（3D）的因子。为零表示该变换不可逆。 |
| 特征方程 | “eigenvalue 从哪里来” | det(A - lambda * I) = 0。它的根就是 eigenvalue。 |

## 延伸阅读

- [3Blue1Brown: Linear Transformations](https://www.3blue1brown.com/lessons/linear-transformations) -- 关于矩阵如何重塑空间的视觉直觉
- [3Blue1Brown: Eigenvectors and Eigenvalues](https://www.3blue1brown.com/lessons/eigenvalues) -- 从几何上解释 eigenvector 含义的优秀可视化讲解
- [MIT 18.06 Lecture 21: Eigenvalues and Eigenvectors](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) -- Gilbert Strang 的经典讲解
