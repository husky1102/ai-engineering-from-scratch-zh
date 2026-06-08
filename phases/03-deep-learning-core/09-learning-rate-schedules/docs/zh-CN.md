# 学习率调度与 Warmup

> 学习率是最重要的单个超参数。不是架构。不是数据集大小。不是激活函数。是学习率。如果你只调一个东西，就调它。

**类型:** Build
**语言:** Python
**先修:** Lesson 03.06 (Optimizers), Lesson 03.08 (Weight Initialization)
**时间:** ~90 minutes

## 学习目标

- 从零实现 constant、step decay、cosine annealing、warmup + cosine 和 1cycle learning rate schedules
- 演示学习率选择的三种失败模式：divergence（过高）、stalling（过低）和 oscillation（没有衰减）
- 解释为什么基于 Adam 的优化器需要 warmup，以及它如何稳定早期训练
- 在同一任务上比较全部五种 schedule 的收敛速度，并为给定训练预算选择合适方案

## 要解决的问题

把学习率设为 0.1。训练发散——loss 在 3 步内跳到 infinity。设为 0.0001。训练像爬一样慢——100 个 epoch 后，模型几乎没离开随机状态。设为 0.01。训练前 50 个 epoch 有效，但随后 loss 在某个最小值附近振荡，因为步子太大，永远到不了最低点。

最佳学习率不是常数。它会在训练过程中变化。早期你想用大步快速覆盖空间。后期你想用小步安定到一个尖锐最小值里。90% 准确率模型和 95% 准确率模型之间的差距，往往只是 schedule。

过去三年发布的每个重要模型都使用 learning rate schedule。Llama 3 使用 peak lr=3e-4，2000 个 warmup steps，并用 cosine decay 衰减到 3e-5。GPT-3 使用 lr=6e-4，并在 375 million tokens 上 warmup。这些都不是随意选择。它们来自花费数百万美元的大规模超参数扫参。

你需要理解 schedules，因为默认值不会自动适配你的问题。当你 fine-tune 一个 pretrained model 时，正确 schedule 不同于从零训练。当你增加 batch size 时，warmup period 需要改变。当训练在 step 10,000 崩掉时，你需要知道这是 schedule 问题，还是别的地方出了问题。

## 核心概念

### Constant Learning Rate

最简单的办法。选一个数，每一步都用它。

```text
lr(t) = lr_0
```

很少是最优的。它要么对训练后期太高（围绕最小值振荡），要么对训练早期太低（把算力浪费在很小的步子上）。它适合小模型和调试。对任何训练超过一小时的东西来说，都是很糟糕的选择。

### Step Decay

ResNet 时代的老派做法。在固定 epoch 将学习率按某个因子降低（通常是 10 倍）。

```text
lr(t) = lr_0 * gamma^(floor(epoch / step_size))
```

其中 gamma = 0.1 且 step_size = 30 表示：lr 每 30 个 epoch 下降 10 倍。ResNet-50 就使用了这个方式——lr=0.1，并在 epoch 30、60、90 处下降 10 倍。

问题是：最佳衰减点依赖数据集和架构。换到另一个问题，你就得重新调什么时候下降。转换也很突兀——当学习率突然变化时，loss 可能会尖峰。

### Cosine Annealing

从最大学习率平滑衰减到最小学习率，形状遵循余弦曲线：

```text
lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))
```

其中 t 是当前 step，T 是总 step 数。

t=0 时，余弦项是 1，所以 lr = lr_max。t=T 时，余弦项是 -1，所以 lr = lr_min。衰减一开始很温和，中间加速，接近结尾时又变得温和。

这是大多数现代训练运行的默认选择。除了 lr_max 和 lr_min 之外，没有要调的超参数。余弦形状符合一个经验观察：大多数学习发生在训练中段——你希望在这个关键阶段保持合理的步长。

### Warmup：为什么要从小开始

Adam 和其他 adaptive optimizers 会维护梯度均值和方差的运行估计。在 step 0，这些估计初始化为零。前几次梯度更新基于很差的统计量。如果在这段时期学习率很大，模型会迈出巨大且方向很差的步子。

Warmup 修复这一点。从很小的学习率开始（通常是 lr_max / warmup_steps，甚至是零），在前 N 步线性爬升到 lr_max。等达到完整学习率时，Adam 的统计量已经稳定。

```text
lr(t) = lr_max * (t / warmup_steps)     for t < warmup_steps
```

典型 warmup：总训练 step 的 1-5%。Llama 3 训练了约 1.8 trillion tokens，并 warmup 了 2000 步。GPT-3 在 375 million tokens 上进行了 warmup。

### Linear Warmup + Cosine Decay

现代默认选择。先线性升高，再用余弦衰减：

```text
if t < warmup_steps:
    lr(t) = lr_max * (t / warmup_steps)
else:
    progress = (t - warmup_steps) / (total_steps - warmup_steps)
    lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * progress))
```

这正是 Llama、GPT、PaLM 和大多数现代 transformers 使用的方式。warmup 防止早期不稳定。cosine decay 让模型安定到一个好的最小值。

### 1cycle Policy

Leslie Smith 的发现（2018）：在训练前半段把学习率从低值升到高值，然后在后半段降回去。听起来反直觉——为什么要在训练中途*增加*学习率？

理论是：高学习率通过向优化轨迹加入噪声来起正则化作用。模型会在升高阶段探索更多 loss landscape，找到更好的盆地。下降阶段随后在找到的最佳盆地内精修。

```text
Phase 1 (0 to T/2):    lr ramps from lr_max/25 to lr_max
Phase 2 (T/2 to T):    lr ramps from lr_max to lr_max/10000
```

在固定计算预算下，1cycle 通常比 cosine annealing 训练得更快。代价是：你必须提前知道总 step 数。

### Schedule 形状

```mermaid
graph LR
    subgraph "Constant"
        C1["lr"] --- C2["lr"] --- C3["lr"]
    end

    subgraph "Step Decay"
        S1["0.1"] --- S2["0.1"] --- S3["0.01"] --- S4["0.001"]
    end

    subgraph "Cosine Annealing"
        CS1["lr_max"] --> CS2["gradual"] --> CS3["steep"] --> CS4["lr_min"]
    end

    subgraph "Warmup + Cosine"
        WC1["0"] --> WC2["lr_max"] --> WC3["cosine"] --> WC4["lr_min"]
    end
```

### 决策流程图

```mermaid
flowchart TD
    Start["Choosing a LR schedule"] --> Know{"Know total<br/>training steps?"}

    Know -->|"Yes"| Budget{"Compute budget?"}
    Know -->|"No"| Constant["Use constant LR<br/>with manual decay"]

    Budget -->|"Large (days/weeks)"| WarmCos["Warmup + Cosine Decay<br/>(Llama/GPT default)"]
    Budget -->|"Small (hours)"| OneCycle["1cycle Policy<br/>(fastest convergence)"]
    Budget -->|"Moderate"| Cosine["Cosine Annealing<br/>(safe default)"]

    WarmCos --> Warmup["Warmup = 1-5% of steps"]
    OneCycle --> FindLR["Find lr_max with LR range test"]
    Cosine --> MinLR["Set lr_min = lr_max / 10"]
```

### 已发布模型中的真实数值

```mermaid
graph TD
    subgraph "Published LR Configs"
        L3["Llama 3 (405B)<br/>Peak: 3e-4<br/>Warmup: 2000 steps<br/>Schedule: Cosine to 3e-5"]
        G3["GPT-3 (175B)<br/>Peak: 6e-4<br/>Warmup: 375M tokens<br/>Schedule: Cosine to 0"]
        R50["ResNet-50<br/>Peak: 0.1<br/>Warmup: none<br/>Schedule: Step decay x0.1 at 30,60,90"]
        B["BERT (340M)<br/>Peak: 1e-4<br/>Warmup: 10K steps<br/>Schedule: Linear decay"]
    end
```

## 动手实现

### Step 1: Schedule 函数

每个函数接收当前 step，并返回该 step 的学习率。

```python
import math


def constant_schedule(step, lr=0.01, **kwargs):
    return lr


def step_decay_schedule(step, lr=0.1, step_size=100, gamma=0.1, **kwargs):
    return lr * (gamma ** (step // step_size))


def cosine_schedule(step, lr=0.01, total_steps=1000, lr_min=1e-5, **kwargs):
    if step >= total_steps:
        return lr_min
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * step / total_steps))


def warmup_cosine_schedule(step, lr=0.01, total_steps=1000, warmup_steps=100, lr_min=1e-5, **kwargs):
    if total_steps <= warmup_steps:
        return lr * (step / max(warmup_steps, 1))
    if step < warmup_steps:
        return lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * progress))


def one_cycle_schedule(step, lr=0.01, total_steps=1000, **kwargs):
    mid = max(total_steps // 2, 1)
    if step < mid:
        return (lr / 25) + (lr - lr / 25) * step / mid
    else:
        progress = (step - mid) / max(total_steps - mid, 1)
        return lr * (1 - progress) + (lr / 10000) * progress
```

### Step 2: 可视化所有 Schedule

打印一个基于文本的图，展示每种 schedule 如何随训练演化。

```python
def visualize_schedule(name, schedule_fn, total_steps=500, **kwargs):
    steps = list(range(0, total_steps, total_steps // 20))
    if total_steps - 1 not in steps:
        steps.append(total_steps - 1)

    lrs = [schedule_fn(s, total_steps=total_steps, **kwargs) for s in steps]
    max_lr = max(lrs) if max(lrs) > 0 else 1.0

    print(f"\n{name}:")
    for s, lr_val in zip(steps, lrs):
        bar_len = int(lr_val / max_lr * 40)
        bar = "#" * bar_len
        print(f"  Step {s:4d}: lr={lr_val:.6f} {bar}")
```

### Step 3: 训练网络

在 circle dataset 上使用一个简单的双层网络，和前面课程相同，但这次我们改变 schedule。

```python
import random


def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def relu(x):
    return max(0.0, x)


def relu_deriv(x):
    return 1.0 if x > 0 else 0.0


def make_circle_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))
    return data


def train_with_schedule(schedule_fn, schedule_name, data, epochs=300, base_lr=0.05, **kwargs):
    random.seed(0)
    hidden_size = 8
    total_steps = epochs * len(data)

    std = math.sqrt(2.0 / 2)
    w1 = [[random.gauss(0, std) for _ in range(2)] for _ in range(hidden_size)]
    b1 = [0.0] * hidden_size
    w2 = [random.gauss(0, std) for _ in range(hidden_size)]
    b2 = 0.0

    step = 0
    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0
        correct = 0

        for x, target in data:
            lr = schedule_fn(step, lr=base_lr, total_steps=total_steps, **kwargs)

            z1 = []
            h = []
            for i in range(hidden_size):
                z = w1[i][0] * x[0] + w1[i][1] * x[1] + b1[i]
                z1.append(z)
                h.append(relu(z))

            z2 = sum(w2[i] * h[i] for i in range(hidden_size)) + b2
            out = sigmoid(z2)

            error = out - target
            d_out = error * out * (1 - out)

            for i in range(hidden_size):
                d_h = d_out * w2[i] * relu_deriv(z1[i])
                w2[i] -= lr * d_out * h[i]
                for j in range(2):
                    w1[i][j] -= lr * d_h * x[j]
                b1[i] -= lr * d_h
            b2 -= lr * d_out

            total_loss += (out - target) ** 2
            if (out >= 0.5) == (target >= 0.5):
                correct += 1
            step += 1

        avg_loss = total_loss / len(data)
        accuracy = correct / len(data) * 100
        epoch_losses.append(avg_loss)

    return epoch_losses
```

### Step 4: 比较所有 Schedule

用每种 schedule 训练同一个网络，并比较最终 loss 与收敛行为。

```python
def compare_schedules(data):
    configs = [
        ("Constant", constant_schedule, {}),
        ("Step Decay", step_decay_schedule, {"step_size": 15000, "gamma": 0.1}),
        ("Cosine", cosine_schedule, {"lr_min": 1e-5}),
        ("Warmup+Cosine", warmup_cosine_schedule, {"warmup_steps": 3000, "lr_min": 1e-5}),
        ("1cycle", one_cycle_schedule, {}),
    ]

    print(f"\n{'Schedule':<20} {'Start Loss':>12} {'Mid Loss':>12} {'End Loss':>12} {'Best Loss':>12}")
    print("-" * 70)

    for name, schedule_fn, extra_kwargs in configs:
        losses = train_with_schedule(schedule_fn, name, data, epochs=300, base_lr=0.05, **extra_kwargs)
        mid_idx = len(losses) // 2
        best = min(losses)
        print(f"{name:<20} {losses[0]:>12.6f} {losses[mid_idx]:>12.6f} {losses[-1]:>12.6f} {best:>12.6f}")
```

### Step 5: LR 过高与过低

演示三种失败模式：过高（divergence）、过低（crawling）和刚刚好。

```python
def lr_sensitivity(data):
    learning_rates = [1.0, 0.1, 0.01, 0.001, 0.0001]

    print("\nLR Sensitivity (constant schedule, 100 epochs):")
    print(f"  {'LR':>10} {'Start Loss':>12} {'End Loss':>12} {'Status':>15}")
    print("  " + "-" * 52)

    for lr in learning_rates:
        losses = train_with_schedule(constant_schedule, f"lr={lr}", data, epochs=100, base_lr=lr)
        start = losses[0]
        end = losses[-1]

        if end > start or math.isnan(end) or end > 1.0:
            status = "DIVERGED"
        elif end > start * 0.9:
            status = "BARELY MOVED"
        elif end < 0.15:
            status = "CONVERGED"
        else:
            status = "LEARNING"

        end_str = f"{end:.6f}" if not math.isnan(end) else "NaN"
        print(f"  {lr:>10.4f} {start:>12.6f} {end_str:>12} {status:>15}")
```

## 实际使用

PyTorch 在 `torch.optim.lr_scheduler` 中提供 schedulers：

```python
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, StepLR

model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
optimizer = optim.Adam(model.parameters(), lr=3e-4)

scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-5)

for step in range(1000):
    loss = train_step(model, optimizer)
    scheduler.step()
```

对 warmup + cosine，可以使用 lambda scheduler，或 HuggingFace 的 `get_cosine_schedule_with_warmup`：

```python
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=2000,
    num_training_steps=100000,
)
```

HuggingFace 函数正是大多数 Llama 和 GPT fine-tuning scripts 使用的东西。拿不准时，就使用 warmup + cosine，并把 warmup 设为总 step 的 3-5%。它几乎适用于所有情况。

## 交付成果

本课产出：
- `outputs/prompt-lr-schedule-advisor.md`——一个 prompt，用于根据你的训练设置推荐合适的 learning rate schedule 和超参数

## 练习

1. 实现 exponential decay：lr(t) = lr_0 * gamma^t，其中 gamma = 0.999。和 circle dataset 上的 cosine annealing 对比。

2. 实现 learning rate range test（Leslie Smith）：训练几百步，同时把 LR 从 1e-7 指数级提高到 1。绘制 loss vs LR。最佳 max LR 就在 loss 开始上升之前。

3. 使用 warmup + cosine 训练，但改变 warmup length：总 step 的 0%、1%、5%、10%、20%。找到训练最稳定的 sweet spot。

4. 实现带 warm restarts 的 cosine annealing（SGDR）：每 T 步将学习率重置为 lr_max，然后再次衰减。和更长训练运行中的标准 cosine 对比。

5. 构建一个 “schedule surgeon”，它监控 training loss，并在 loss 稳定时自动从 warmup 切换到 cosine；如果 loss plateau 太久，就降低 lr。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Learning rate | “模型学习有多快” | 乘在梯度上的标量，用来决定参数更新大小 |
| Schedule | “随时间改变 LR” | 一个将 training step 映射到 learning rate 的函数，设计目标是优化收敛 |
| Warmup | “从小 LR 开始” | 在前 N 步中将 LR 从接近零线性升高到目标值，以稳定 optimizer statistics |
| Cosine annealing | “平滑 LR 衰减” | 在训练过程中，让 LR 沿余弦曲线从 lr_max 降到 lr_min |
| Step decay | “在里程碑处降低 LR” | 在固定 epoch 间隔将 LR 乘以某个因子（通常是 0.1） |
| 1cycle policy | “先上后下” | Leslie Smith 的方法：在单个 cycle 中先升高 LR 再降低，用于更快收敛 |
| LR range test | “找到最佳学习率” | 短暂训练并逐步提高 LR，以找到 loss 开始发散的位置 |
| Cosine with warm restarts | “重置并重复” | 周期性地把 LR 重置为 lr_max，然后再次衰减（SGDR） |
| Eta min | “LR 的下限” | schedule 最终衰减到的最小学习率 |
| Peak learning rate | “最大 LR” | 训练期间达到的最高 LR，通常在 warmup 之后 |

## 延伸阅读

- Loshchilov & Hutter, “SGDR: Stochastic Gradient Descent with Warm Restarts” (2017)——介绍 cosine annealing 和 warm restarts
- Smith, “Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates” (2018)——1cycle policy 论文
- Touvron et al., “Llama 2: Open Foundation and Fine-Tuned Chat Models” (2023)——记录大规模使用的 warmup + cosine schedule
- Goyal et al., “Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour” (2017)——large batch training 的 linear scaling rule 与 warmup
