# 从零实现反向传播

> 反向传播是让学习成为可能的算法。没有它，神经网络只是昂贵的随机数生成器。

**类型:** Build
**语言:** Python
**先修:** Lesson 03.02（Multi-Layer Networks）
**时间:** ~120 分钟

## 学习目标

- 实现一个基于 Value 的 autograd 引擎，构建计算图，并通过拓扑排序计算梯度
- 使用链式法则推导加法、乘法和 sigmoid 的反向传播
- 只使用从零实现的反向传播引擎，在 XOR 和圆形分类上训练多层网络
- 识别深层 sigmoid 网络中的梯度消失问题，并解释为什么梯度会指数级缩小

## 要解决的问题

你的网络有一个隐藏层，包含 768 个输入和 3072 个输出。这就是 2,359,296 个权重。它做错了一次预测。哪些权重导致了这个误差？逐个测试每个权重意味着 230 万次前向传播。反向传播用一次反向传播就能计算全部 230 万个梯度。这不是优化。这是“可训练”和“不可能”之间的差别。

朴素方法是：取一个权重，把它轻轻挪动一个很小的量，再运行一次前向传播，测量 loss 是上升还是下降。这样你得到了这个权重的梯度。现在对网络里的每个权重都这么做。再乘以数千个训练 step 和数百万个数据点。你需要地质年代的时间才能训练出任何有用的东西。

反向传播解决了这个问题。一次前向传播，一次反向传播，所有梯度都算出来。诀窍是把微积分里的链式法则系统地应用到计算图上。这就是让深度学习变得实际可用的算法。没有它，我们还会被困在玩具问题上。

## 核心概念

### 链式法则，应用到网络

你在第 01 阶段第 05 课见过链式法则。快速回顾：如果 y = f(g(x))，那么 dy/dx = f'(g(x)) * g'(x)。你沿着链条把导数相乘。

在神经网络中，“链条”是从输入到 loss 的操作序列。每一层应用权重、加上偏置、经过激活。loss 函数把最终输出与目标进行比较。反向传播沿着这条链反向追踪，计算每个操作对误差的贡献。

### 计算图

每次前向传播都会构建一张图。每个节点是一个操作（multiply、add、sigmoid）。每条边向前携带值，向后携带梯度。

```mermaid
graph LR
    x["x"] --> mul["*"]
    w["w"] --> mul
    mul -- "z1 = w*x" --> add["+"]
    b["b"] --> add
    add -- "z2 = z1 + b" --> sig["sigmoid"]
    sig -- "a = sigmoid(z2)" --> loss["Loss"]
    y["target"] --> loss
```

前向传播：值从左向右流动。x 和 w 产生 z1 = w*x。加上 b 得到 z2。Sigmoid 给出激活 a。使用 loss 函数把 a 与目标 y 比较。

反向传播：梯度从右向左流动。从 dL/da 开始（loss 如何随激活变化）。乘以 da/dz2（sigmoid 导数）。得到 dL/dz2。再分成 dL/db（因为 z2 = z1 + b，所以它等于 dL/dz2）和 dL/dz1。然后 dL/dw = dL/dz1 * x，dL/dx = dL/dz1 * w。

计算图中的每个节点在反向传播期间只有一个任务：接收上游传来的梯度，乘以自己的局部导数，再把它传下去。

### 前向与反向

```mermaid
graph TB
    subgraph Forward["Forward Pass"]
        direction LR
        f1["Input x"] --> f2["z = Wx + b"]
        f2 --> f3["a = sigmoid(z)"]
        f3 --> f4["Loss = (a - y)^2"]
    end
    subgraph Backward["Backward Pass"]
        direction RL
        b4["dL/dL = 1"] --> b3["dL/da = 2(a-y)"]
        b3 --> b2["dL/dz = dL/da * a(1-a)"]
        b2 --> b1["dL/dW = dL/dz * x\ndL/db = dL/dz"]
    end
    Forward --> Backward
```

前向传播会保存每个中间值：z、a，以及每一层的输入。反向传播需要这些保存的值来计算梯度。这就是 backprop 核心的内存-计算权衡。你用内存（存储激活）换速度（一次传播，而不是数百万次）。

### 梯度如何流过网络

对一个 3 层网络，梯度会链式穿过每一层：

```mermaid
graph RL
    L["Loss"] -- "dL/da3" --> L3["Layer 3\na3 = sigmoid(z3)"]
    L3 -- "dL/dz3 = dL/da3 * sigmoid'(z3)" --> L2["Layer 2\na2 = sigmoid(z2)"]
    L2 -- "dL/dz2 = dL/da2 * sigmoid'(z2)" --> L1["Layer 1\na1 = sigmoid(z1)"]
    L1 -- "dL/dz1 = dL/da1 * sigmoid'(z1)" --> I["Input"]
```

在每一层，梯度都会乘以 sigmoid 导数。sigmoid 导数是 a * (1 - a)，最大值是 0.25（当 a = 0.5 时）。三层之后，梯度最多已经被乘以 0.25^3 = 0.0156。十层之后：0.25^10 = 0.000001。

### 梯度消失

这就是梯度消失问题。Sigmoid 把输出压缩在 0 和 1 之间。它的导数总是小于 0.25。堆叠足够多的 sigmoid 层，梯度就会缩到几乎没有。早期层几乎学不到东西，因为它们收到的是接近零的梯度。

```text
sigmoid(z):     Output range [0, 1]
sigmoid'(z):    Max value 0.25 (at z = 0)

After 5 layers:   gradient * 0.25^5 = 0.001x original
After 10 layers:  gradient * 0.25^10 = 0.000001x original
```

这就是深层 sigmoid 网络几乎不可能训练的原因。修复方法——ReLU 及其变体——是第 04 课的主题。现在先理解：backprop 本身工作得很好。问题在于它所穿过的东西。

### 推导 2 层网络的梯度

来看一个具体网络：输入 x，一个带 sigmoid 的隐藏层，一个带 sigmoid 的输出层，以及 MSE loss。

前向传播：
```text
z1 = W1 * x + b1
a1 = sigmoid(z1)
z2 = W2 * a1 + b2
a2 = sigmoid(z2)
L = (a2 - y)^2
```

反向传播（一步步应用链式法则）：
```text
dL/da2 = 2(a2 - y)
da2/dz2 = a2 * (1 - a2)
dL/dz2 = dL/da2 * da2/dz2 = 2(a2 - y) * a2 * (1 - a2)

dL/dW2 = dL/dz2 * a1
dL/db2 = dL/dz2

dL/da1 = dL/dz2 * W2
da1/dz1 = a1 * (1 - a1)
dL/dz1 = dL/da1 * da1/dz1

dL/dW1 = dL/dz1 * x
dL/db1 = dL/dz1
```

每个梯度都是从 loss 往回追踪时，各个局部导数的乘积。这就是反向传播的全部。

## 动手实现

### 第 1 步：Value 节点

我们计算中的每个数字都会成为一个 Value。它存储自己的 data、grad，以及它是如何被创建的（这样它就知道如何向后计算梯度）。

```python
class Value:
    def __init__(self, data, children=(), op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._children = set(children)
        self._op = op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
```

目前没有梯度（0.0）。也还没有 backward function（no-op）。`_children` 会跟踪哪些 Value 生成了这个 Value，这样之后我们就能对图做拓扑排序。

### 第 2 步：带反向函数的操作

每个操作都会创建一个新的 Value，并定义梯度如何反向流过它。

```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
        self.grad += out.grad
        other.grad += out.grad

    out._backward = _backward
    return out

def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad

    out._backward = _backward
    return out
```

对于加法：d(a+b)/da = 1，d(a+b)/db = 1。所以两个输入都会直接得到输出的梯度。

对于乘法：d(a*b)/da = b，d(a*b)/db = a。每个输入都会得到另一个输入的值乘以输出梯度。

`+=` 很关键。一个 Value 可能被多个操作使用。它的梯度是所有路径传回梯度的总和。

### 第 3 步：Sigmoid 和 Loss

```python
import math

def sigmoid(self):
    x = self.data
    x = max(-500, min(500, x))
    s = 1.0 / (1.0 + math.exp(-x))
    out = Value(s, (self,), 'sigmoid')

    def _backward():
        self.grad += (s * (1 - s)) * out.grad

    out._backward = _backward
    return out
```

Sigmoid 导数：sigmoid(x) * (1 - sigmoid(x))。我们已经在前向传播期间计算了 sigmoid(x) = s。复用它。无需额外工作。

```python
def mse_loss(predicted, target):
    diff = predicted + Value(-target)
    return diff * diff
```

单个输出的 MSE：(predicted - target)^2。我们把减法表示为与一个取负 Value 相加。

### 第 4 步：反向传播

拓扑排序保证我们按正确顺序处理节点——在通过某个节点继续传播之前，该节点的梯度已经完全累积。

```python
def backward(self):
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._children:
                build_topo(child)
            topo.append(v)

    build_topo(self)
    self.grad = 1.0
    for v in reversed(topo):
        v._backward()
```

从 loss 开始（gradient = 1.0，因为 dL/dL = 1）。沿排序后的图反向遍历。每个节点的 `_backward` 都会把梯度推给自己的 children。

### 第 5 步：Layer 和 Network

```python
import random

class Neuron:
    def __init__(self, n_inputs):
        scale = (2.0 / n_inputs) ** 0.5
        self.weights = [Value(random.uniform(-scale, scale)) for _ in range(n_inputs)]
        self.bias = Value(0.0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.weights, x)), self.bias)
        return act.sigmoid()

    def parameters(self):
        return self.weights + [self.bias]


class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        params = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params


class Network:
    def __init__(self, sizes):
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Layer(sizes[i], sizes[i + 1]))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            if not isinstance(x, list):
                x = [x]
        return x[0] if len(x) == 1 else x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
```

Neuron 接收输入，计算加权和加偏置，并应用 sigmoid。权重初始化按 sqrt(2/n_inputs) 缩放，以防止更深网络中的 sigmoid 饱和。Layer 是一组 Neuron。Network 是一组 Layer。`parameters()` 方法收集所有可学习的 Value，方便我们更新它们。

### 第 6 步：在 XOR 上训练

```python
random.seed(42)
net = Network([2, 4, 1])

xor_data = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
]

learning_rate = 1.0

for epoch in range(1000):
    total_loss = Value(0.0)
    for inputs, target in xor_data:
        x = [Value(i) for i in inputs]
        pred = net(x)
        loss = mse_loss(pred, target)
        total_loss = total_loss + loss

    net.zero_grad()
    total_loss.backward()

    for p in net.parameters():
        p.data -= learning_rate * p.grad

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {total_loss.data:.6f}")

print("\nXOR Results:")
for inputs, target in xor_data:
    x = [Value(i) for i in inputs]
    pred = net(x)
    print(f"  {inputs} -> {pred.data:.4f} (expected {target})")
```

观察 loss 下降。从随机预测到正确的 XOR 输出，整个过程完全由反向传播计算梯度，并把权重往正确方向轻推。

### 第 7 步：圆形分类

在第 02 课中，你为圆形分类手工调好了权重。现在让网络自己学习它们。

```python
random.seed(7)

def generate_circle_data(n=100):
    data = []
    for _ in range(n):
        x1 = random.uniform(-1.5, 1.5)
        x2 = random.uniform(-1.5, 1.5)
        label = 1.0 if x1 * x1 + x2 * x2 < 1.0 else 0.0
        data.append(([x1, x2], label))
    return data

circle_data = generate_circle_data(80)

circle_net = Network([2, 8, 1])
learning_rate = 0.5

for epoch in range(2000):
    random.shuffle(circle_data)
    total_loss_val = 0.0
    for inputs, target in circle_data:
        x = [Value(i) for i in inputs]
        pred = circle_net(x)
        loss = mse_loss(pred, target)
        circle_net.zero_grad()
        loss.backward()
        for p in circle_net.parameters():
            p.data -= learning_rate * p.grad
        total_loss_val += loss.data

    if epoch % 200 == 0:
        correct = 0
        for inputs, target in circle_data:
            x = [Value(i) for i in inputs]
            pred = circle_net(x)
            predicted_class = 1.0 if pred.data > 0.5 else 0.0
            if predicted_class == target:
                correct += 1
        accuracy = correct / len(circle_data) * 100
        print(f"Epoch {epoch:4d} | Loss: {total_loss_val:.4f} | Accuracy: {accuracy:.1f}%")
```

这里我们使用 online SGD——每个样本之后就更新权重，而不是累积完整 batch。这能更快打破对称性，并避免在完整 loss landscape 上发生 sigmoid 饱和。每个 epoch 都打乱数据，可以防止网络记住顺序。

没有手工调权。网络自己发现了圆形决策边界。这就是反向传播的力量：你定义架构、loss 函数和数据。算法会找出权重。

## 实际使用

PyTorch 用几行就能完成上面的一切。核心思想完全相同——autograd 在前向传播期间构建计算图，再沿图反向追踪以计算梯度。

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.Sigmoid(),
    nn.Linear(4, 1),
    nn.Sigmoid(),
)
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
criterion = nn.MSELoss()

X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

for epoch in range(1000):
    pred = model(X)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("PyTorch XOR Results:")
with torch.no_grad():
    for i in range(4):
        pred = model(X[i])
        print(f"  {X[i].tolist()} -> {pred.item():.4f} (expected {y[i].item()})")
```

`loss.backward()` 就是你的 `total_loss.backward()`。`optimizer.step()` 就是你手写的 `p.data -= lr * p.grad`。`optimizer.zero_grad()` 就是你的 `net.zero_grad()`。同一个算法，工业强度实现。PyTorch 处理 GPU 加速、mixed precision、gradient checkpointing，以及数百种 layer 类型。但反向传播仍然是同样的链式法则，应用到同样的计算图上。

训练会运行前向传播，然后运行反向传播，再更新权重。推理只运行前向传播。没有梯度，没有更新。这个区别很重要，因为推理就是生产环境中发生的事。当你调用 Claude 或 GPT 这样的 API 时，你在运行推理——你的 prompt 向前流过网络，tokens 从另一端出来。没有权重发生变化。理解 backprop 很重要，因为那个网络中的每个权重都是它塑造出来的。

## 交付成果

本课产出：
- `outputs/prompt-gradient-debugger.md` -- 一个可复用 prompt，用于诊断任何神经网络中的梯度问题（vanishing、exploding、NaN）

## 练习

1. 给 Value 类添加一个 `__sub__` 方法（a - b = a + (-1 * b)）。然后实现一个 `__neg__` 方法。用一个简单表达式（例如 (a - b)^2）与手算结果对比，验证梯度正确。

2. 给 Value 添加一个 `relu` 方法（输出 max(0, x)，导数在 x > 0 时为 1，否则为 0）。在隐藏层中用 relu 替换 sigmoid，并重新在 XOR 上训练。比较收敛速度。你应该看到训练更快——这会预告第 04 课。

3. 在 Value 上为整数幂实现一个 `__pow__` 方法。用它把 `mse_loss` 替换成真正的 `(predicted - target) ** 2` 表达式。验证梯度与原实现一致。

4. 给训练循环添加 gradient clipping：调用 `backward()` 之后，把所有梯度裁剪到 [-1, 1]。训练一个更深的网络（4+ 层 sigmoid），并比较有无裁剪时的 loss 曲线。这是你对抗梯度爆炸的第一道防线。

5. 构建一个可视化：在 XOR 训练后，打印网络中每个参数的梯度。找出哪一层的梯度最小。这会演示你在“核心概念”中读到的梯度消失问题。

## 关键术语

| 术语 | 人们常说 | 它实际意味着什么 |
|------|----------------|----------------------|
| Backpropagation | “网络在学习” | 一种算法，通过沿计算图反向应用链式法则，为每个权重计算 dL/dw |
| Computational graph | “网络结构” | 有向无环图，其中节点是操作，边携带值（向前）和梯度（向后） |
| Chain rule | “把导数相乘” | 如果 y = f(g(x))，那么 dy/dx = f'(g(x)) * g'(x)——这是反向传播的数学基础 |
| Gradient | “最陡上升方向” | loss 对某个参数的偏导数——告诉你如何改变该参数以降低 loss |
| Vanishing gradient | “深层网络学不动” | 当梯度穿过 sigmoid 等饱和激活层时，会指数级缩小 |
| Forward pass | “运行网络” | 通过顺序应用每一层操作并保存中间值，从输入计算输出 |
| Backward pass | “计算梯度” | 反向遍历计算图，在每个节点用链式法则累积梯度 |
| Learning rate | “学得多快” | 更新权重时控制步长的标量：w_new = w_old - lr * gradient |
| Topological sort | “正确顺序” | 图节点的一种排序，其中每个节点都出现在它依赖的全部节点之后——保证梯度在传播前已经完全累积 |
| Autograd | “自动微分” | 在前向计算期间构建计算图并自动计算梯度的系统——也就是 PyTorch 的引擎所做的事 |

## 延伸阅读

- Rumelhart, Hinton & Williams, "Learning representations by back-propagating errors" (1986) -- 让反向传播成为主流、并解锁多层网络训练的论文
- 3Blue1Brown, "Neural Networks" series (https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) -- 对反向传播以及梯度如何流过网络的最佳可视化解释
