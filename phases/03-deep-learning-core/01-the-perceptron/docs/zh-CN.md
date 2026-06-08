# 感知机

> 感知机是神经网络的原子。把它拆开，你会看到权重、偏置，以及一个决策。

**类型:** Build
**语言:** Python
**先修:** 第 1 阶段（线性代数直觉）
**时间:** ~60 分钟

## 学习目标

- 用 Python 从零实现感知机，包括权重更新规则和阶跃激活函数
- 解释为什么单个感知机只能解决线性可分问题，并演示 XOR 失败案例
- 通过组合 OR、NAND 和 AND 门，构造一个多层感知机来解决 XOR
- 训练一个带 sigmoid 激活和反向传播的两层网络，让它自动学习 XOR

## 要解决的问题

你已经知道向量和点积。你知道矩阵会把输入变换为输出。但机器到底如何*学习*该使用哪种变换？

感知机回答了这个问题。它是最简单的学习机器：取一些输入，乘以权重，加上偏置，然后做出二元决策。接着调整。就是这样。曾经构建过的每个神经网络，都是把这个想法一层层堆叠起来。

理解感知机，就是理解代码里的“学习”究竟是什么意思：不断调整数字，直到输出与现实匹配。

## 核心概念

### 一个神经元，一个决策

感知机接收 n 个输入，把每个输入乘以一个权重，求和，加上偏置，再把结果传入激活函数。

```mermaid
graph LR
    x1["x1"] -- "w1" --> sum["Σ(wi*xi) + b"]
    x2["x2"] -- "w2" --> sum
    x3["x3"] -- "w3" --> sum
    bias["bias"] --> sum
    sum --> step["step(z)"]
    step --> out["output (0 or 1)"]
```

阶跃函数很粗暴：如果加权和加偏置 >= 0，就输出 1。否则输出 0。

```text
step(z) = 1  if z >= 0
           0  if z < 0
```

这就是一个线性分类器。权重和偏置定义了一条线（在更高维空间中是一个超平面），把输入空间分成两个区域。

### 决策边界

对于两个输入，感知机会在 2D 空间中画出一条线：

```text
  x2
  ┤
  │  Class 1        /
  │    (0)          /
  │                /
  │               / w1·x1 + w2·x2 + b = 0
  │              /
  │             /     Class 2
  │            /        (1)
  ┼───────────/──────────── x1
```

线的一侧输出 0。另一侧输出 1。训练会移动这条线，直到它正确分离类别。

### 学习规则

感知机学习规则很简单：

```text
For each training example (x, y_true):
    y_pred = predict(x)
    error = y_true - y_pred

    For each weight:
        w_i = w_i + learning_rate * error * x_i
    bias = bias + learning_rate * error
```

如果预测正确，error = 0，什么都不会改变。如果它预测为 0 但应该是 1，权重会增加。如果它预测为 1 但应该是 0，权重会减少。学习率控制每次调整的大小。

### XOR 问题

问题就在这里出现。看看这些逻辑门：

```text
AND gate:           OR gate:            XOR gate:
x1  x2  out         x1  x2  out         x1  x2  out
0   0   0           0   0   0           0   0   0
0   1   0           0   1   1           0   1   1
1   0   0           1   0   1           1   0   1
1   1   1           1   1   1           1   1   0
```

AND 和 OR 是线性可分的：你可以画一条线，把 0 和 1 分开。XOR 不是。没有任何一条直线可以把 [0,1] 和 [1,0] 同 [0,0] 和 [1,1] 分开。

```text
AND (separable):        XOR (not separable):

  x2                      x2
  1 ┤  0     1            1 ┤  1     0
    │     /                 │
  0 ┤  0 / 0              0 ┤  0     1
    ┼──/──────── x1         ┼──────────── x1
       line works!          no single line works!
```

这是一个根本限制。单个感知机只能解决线性可分问题。Minsky 和 Papert 在 1969 年证明了这一点，并且这几乎让神经网络研究停滞了十年。

修复方法：把感知机堆叠成层。多层感知机可以把两个线性决策组合成一个非线性决策，从而解决 XOR。

## 动手实现

### 第 1 步：Perceptron 类

```python
class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1):
        self.weights = [0.0] * n_inputs
        self.bias = 0.0
        self.lr = learning_rate

    def predict(self, inputs):
        total = sum(w * x for w, x in zip(self.weights, inputs))
        total += self.bias
        return 1 if total >= 0 else 0

    def train(self, training_data, epochs=100):
        for epoch in range(epochs):
            errors = 0
            for inputs, target in training_data:
                prediction = self.predict(inputs)
                error = target - prediction
                if error != 0:
                    errors += 1
                    for i in range(len(self.weights)):
                        self.weights[i] += self.lr * error * inputs[i]
                    self.bias += self.lr * error
            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                return
        print(f"Did not converge after {epochs} epochs")
```

### 第 2 步：在逻辑门上训练

```python
and_data = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1),
]

or_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1),
]

not_data = [
    ([0], 1),
    ([1], 0),
]

print("=== AND Gate ===")
p_and = Perceptron(2)
p_and.train(and_data)
for inputs, _ in and_data:
    print(f"  {inputs} -> {p_and.predict(inputs)}")

print("\n=== OR Gate ===")
p_or = Perceptron(2)
p_or.train(or_data)
for inputs, _ in or_data:
    print(f"  {inputs} -> {p_or.predict(inputs)}")

print("\n=== NOT Gate ===")
p_not = Perceptron(1)
p_not.train(not_data)
for inputs, _ in not_data:
    print(f"  {inputs} -> {p_not.predict(inputs)}")
```

### 第 3 步：观察 XOR 失败

```python
xor_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

print("\n=== XOR Gate (single perceptron) ===")
p_xor = Perceptron(2)
p_xor.train(xor_data, epochs=1000)
for inputs, expected in xor_data:
    result = p_xor.predict(inputs)
    status = "OK" if result == expected else "WRONG"
    print(f"  {inputs} -> {result} (expected {expected}) {status}")
```

它永远不会收敛。这就是单个感知机无法学习 XOR 的硬证据。

### 第 4 步：用两层解决 XOR

诀窍是：XOR = (x1 OR x2) AND NOT (x1 AND x2)。组合三个感知机：

```mermaid
graph LR
    x1["x1"] --> OR["OR neuron"]
    x1 --> NAND["NAND neuron"]
    x2["x2"] --> OR
    x2 --> NAND
    OR --> AND["AND neuron"]
    NAND --> AND
    AND --> out["output"]
```

```python
def xor_network(x1, x2):
    or_neuron = Perceptron(2)
    or_neuron.weights = [1.0, 1.0]
    or_neuron.bias = -0.5

    nand_neuron = Perceptron(2)
    nand_neuron.weights = [-1.0, -1.0]
    nand_neuron.bias = 1.5

    and_neuron = Perceptron(2)
    and_neuron.weights = [1.0, 1.0]
    and_neuron.bias = -1.5

    hidden1 = or_neuron.predict([x1, x2])
    hidden2 = nand_neuron.predict([x1, x2])
    output = and_neuron.predict([hidden1, hidden2])
    return output


print("\n=== XOR Gate (multi-layer network) ===")
for inputs, expected in xor_data:
    result = xor_network(inputs[0], inputs[1])
    print(f"  {inputs} -> {result} (expected {expected})")
```

四种情况全部正确。把感知机堆叠成层，就能创造出单个感知机无法产生的决策边界。

### 第 5 步：训练一个两层网络

第 4 步是手工接线权重。它能解决 XOR，但对真实问题不适用，因为你不会提前知道正确权重。修复方法：用 sigmoid 替换阶跃函数，并通过反向传播自动学习权重。

```python
class TwoLayerNetwork:
    def __init__(self, learning_rate=0.5):
        import random
        random.seed(0)
        self.w_hidden = [[random.uniform(-1, 1), random.uniform(-1, 1)] for _ in range(2)]
        self.b_hidden = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.w_output = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.b_output = random.uniform(-1, 1)
        self.lr = learning_rate

    def sigmoid(self, x):
        import math
        x = max(-500, min(500, x))
        return 1.0 / (1.0 + math.exp(-x))

    def forward(self, inputs):
        self.inputs = inputs
        self.hidden_outputs = []
        for i in range(2):
            z = sum(w * x for w, x in zip(self.w_hidden[i], inputs)) + self.b_hidden[i]
            self.hidden_outputs.append(self.sigmoid(z))
        z_out = sum(w * h for w, h in zip(self.w_output, self.hidden_outputs)) + self.b_output
        self.output = self.sigmoid(z_out)
        return self.output

    def train(self, training_data, epochs=10000):
        for epoch in range(epochs):
            total_error = 0
            for inputs, target in training_data:
                output = self.forward(inputs)
                error = target - output
                total_error += error ** 2

                d_output = error * output * (1 - output)

                saved_w_output = self.w_output[:]
                hidden_deltas = []
                for i in range(2):
                    h = self.hidden_outputs[i]
                    hd = d_output * saved_w_output[i] * h * (1 - h)
                    hidden_deltas.append(hd)

                for i in range(2):
                    self.w_output[i] += self.lr * d_output * self.hidden_outputs[i]
                self.b_output += self.lr * d_output

                for i in range(2):
                    for j in range(len(inputs)):
                        self.w_hidden[i][j] += self.lr * hidden_deltas[i] * inputs[j]
                    self.b_hidden[i] += self.lr * hidden_deltas[i]
```

```python
net = TwoLayerNetwork(learning_rate=2.0)
net.train(xor_data, epochs=10000)
for inputs, expected in xor_data:
    result = net.forward(inputs)
    predicted = 1 if result >= 0.5 else 0
    print(f"  {inputs} -> {result:.4f} (rounded: {predicted}, expected {expected})")
```

这里与第 4 步有两个关键差异。第一，sigmoid 替换了阶跃函数，因为它是平滑的，所以梯度存在。第二，`train` 方法把误差从输出层反向传播到隐藏层，并按每个权重对误差的贡献比例进行调整。这就是 20 行代码里的反向传播。

这是通往第 03 课的桥梁。`d_output` 和 `hidden_deltas` 背后的数学，就是把链式法则应用到网络图上。我们会在那里正式推导它。

## 实际使用

你刚刚从零构建的一切，都存在于一个 import 中：

```python
from sklearn.linear_model import Perceptron as SkPerceptron
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0, 0, 0, 1])

clf = SkPerceptron(max_iter=100, tol=1e-3)
clf.fit(X, y)
print([clf.predict([x])[0] for x in X])
```

五行代码。你的 30 行 `Perceptron` 类做的是同一件事。sklearn 版本增加了收敛检查、多种损失函数和稀疏输入支持，但核心循环完全相同：加权求和、阶跃函数、根据误差更新权重。

真正的差距出现在规模上。生产网络里会改变什么：

- 阶跃函数变成 sigmoid、ReLU 或其他平滑激活
- 权重通过反向传播自动学习（第 03 课）
- 层变得更深：3 层、10 层、100+ 层
- 同一个原则依然成立：每一层都从上一层的输出中创造新特征

单个感知机只能画直线。把它们堆叠起来，你就能画出任何形状。

## 交付成果

本课产出：
- `outputs/skill-perceptron.md` - 一份技能文档，说明什么时候需要单层架构，什么时候需要多层架构

## 练习

1. 在 NAND 门上训练一个感知机（通用门，任何逻辑电路都可以由 NAND 构建）。验证它的权重和偏置构成了有效的决策边界。
2. 修改 Perceptron 类，跟踪每个 epoch 的决策边界（w1*x1 + w2*x2 + b = 0）。打印在 AND 门训练过程中这条线如何移动。
3. 构建一个 3 输入感知机：只有当 3 个输入中至少 2 个为 1 时才输出 1（多数投票函数）。它是线性可分的吗？为什么？

## 关键术语

| 术语 | 人们常说 | 它实际意味着什么 |
|------|----------------|----------------------|
| Perceptron | “一个假神经元” | 一个线性分类器：输入与权重做点积，加上偏置，再经过阶跃函数 |
| Weight | “某个输入有多重要” | 缩放每个输入对决策贡献的乘数 |
| Bias | “阈值” | 一个常数，会平移决策边界，让感知机即使在零输入时也可以触发 |
| Activation function | “把值压扁的东西” | 加权求和之后应用的函数；感知机使用阶跃函数，现代网络使用 sigmoid/ReLU |
| Linearly separable | “你能在它们之间画一条线” | 一个数据集，其中单个超平面可以完美分离类别 |
| XOR problem | “感知机做不了的东西” | 单层网络无法学习非线性可分函数的证明 |
| Decision boundary | “分类器切换的位置” | 将输入空间划分为两个类别的超平面 w*x + b = 0 |
| Multi-layer perceptron | “真正的神经网络” | 堆叠成多层的感知机，每一层的输出都会送入下一层的输入 |

## 延伸阅读

- Frank Rosenblatt, "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain" (1958) -- 开创这一切的原始论文
- Minsky & Papert, "Perceptrons" (1969) -- 证明 XOR 无法由单层网络解决，并让感知机研究停滞十年的书
- Michael Nielsen, "Neural Networks and Deep Learning", Chapter 1 (http://neuralnetworksanddeeplearning.com/) -- 免费在线资源，对感知机如何组合成网络给出了最好的可视化解释
