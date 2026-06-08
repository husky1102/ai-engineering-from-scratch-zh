# 数值稳定性

> 浮点数是一个会泄漏的抽象。它会在训练过程中咬你一口，而且你往往毫无预感。

**类型：** 构建
**语言：** Python
**先修：** Phase 1, Lessons 01-04
**时间：** ~120 分钟

## 学习目标

- 使用 max-subtraction trick 实现数值稳定的 softmax 和 log-sum-exp
- 在浮点计算中识别 overflow、underflow 和 catastrophic cancellation
- 使用中心有限差分，用数值梯度验证解析梯度
- 解释为什么训练时 bfloat16 通常优于 float16，以及 loss scaling 如何防止梯度下溢

## 要解决的问题

你的模型训练了三个小时，然后 loss 变成 NaN。你加了一行 print。step 9,000 时 logits 还正常。step 9,001 时它们变成了 `inf`。到 step 9,002，每个梯度都是 `nan`，训练已经死掉了。

或者：你的模型顺利训完了，但准确率比论文声称的低 2%。你检查了所有东西。架构一致。超参数一致。数据一致。问题在于论文用的是 float32，而你用了 float16，却没有做正确的 scaling。32 位累积舍入误差悄悄吃掉了你的准确率。

或者：你从零实现 cross-entropy loss。它在小 logits 上运行正常。一旦 logits 超过 100，它就返回 `inf`。softmax overflow 了，因为 `exp(100)` 大于 float32 能表示的范围。每个 ML 框架都用一个两行技巧处理这件事。你之前并不知道这个技巧存在。

数值稳定性不是理论上的担忧。它决定了一次训练是成功，还是悄悄失败。你最终会调试的每一个严重 ML bug，都会在某个层面落到 floating point 上。

## 核心概念

### IEEE 754：计算机如何存储实数

计算机按照 IEEE 754 标准，把实数存成 floating point value。一个 float 有三个部分：符号位、指数和尾数（significand）。

```text
Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

尾数决定精度（有多少有效数字）。指数决定范围（一个数可以有多大或多小）。

```text
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32 给你大约 7 位十进制精度。这意味着它能区分 1.0000001 和 1.0000002，但不能区分 1.00000001 和 1.00000002。超过 7 位之后，一切都只是舍入噪声。

float16 给你大约 3 位精度。它能表示的最大数是 65,504。对于 ML 来说，这小得令人不安，因为 logits、gradients 和 activations 经常超过这个范围。

bfloat16 是 Google 对 float16 范围问题的回答。它和 float32 一样有 8 位 exponent（范围相同，最高到 3.4e38），但只有 7 位 mantissa（精度比 float16 更低）。训练神经网络时，范围比精度更重要，所以 bfloat16 通常胜出。

### 为什么 0.1 + 0.2 != 0.3

数字 0.1 无法在二进制浮点数中被精确表示。在 base 2 中，它是一个循环小数：

```text
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```

Float32 会把它截断到 23 位 mantissa。存储下来的值大约是 0.100000001490116。同样，0.2 存储下来大约是 0.200000002980232。它们的和是 0.300000004470348，而不是 0.3。

```text
In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

这对 ML 很重要，因为：

1. 像 `if loss < threshold` 这样的 loss 比较可能给出错误答案
2. 累积许多小值时（例如数千步里的 gradient updates），结果会偏离真实总和
3. 如果用 `==` 比较 floats，checksums 和 reproducibility tests 会失败

修复方式：永远不要用 `==` 比较 floats。使用 `abs(a - b) < epsilon` 或 `math.isclose()`。

### 灾难性抵消

当你把两个几乎相等的 floating point numbers 相减时，有效数字会互相抵消，剩下的是被推到前导位置的舍入噪声。

```text
a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%
```

这意味着一次减法就带来了 19% 的相对误差。在 ML 中，这会发生在：

- 用大均值数据计算方差：当 E[x] 很大时使用 `E[x^2] - E[x]^2`
- 相减两个几乎相等的 log-probabilities
- 用过小的 epsilon 计算 finite-difference gradients

修复方式：重排公式，避免相减两个很大且几乎相等的数。计算方差时，使用 Welford algorithm，或先把数据中心化。处理 log-probabilities 时，全程在 log-space 中工作。

### Overflow 和 Underflow

Overflow 发生在结果太大、无法表示时。Underflow 发生在结果太小（比最小可表示正数还接近 0）时。

```text
Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0
```

`exp()` 函数是 ML 中 overflow 的主要来源：

```text
exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)
```

`log()` 函数会撞上另一个方向的问题：

```text
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)
```

在 ML 中，`exp()` 出现在 softmax、sigmoid 和 probability computations 里。`log()` 出现在 cross-entropy、log-likelihoods 和 KL divergence 里。没有正确技巧时，`log(exp(x))` 这个组合就是雷区。

### Log-Sum-Exp 技巧

直接计算 `log(sum(exp(x_i)))` 在数值上很危险。如果任何一个 `x_i` 很大，`exp(x_i)` 会 overflow。如果所有 `x_i` 都非常负，每个 `exp(x_i)` 都会 underflow 成 0，而 `log(0)` 是 `-inf`。

技巧是：在取指数之前先减去最大值。

```text
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

为什么它有效：减去 `max(x)` 之后，最大的指数是 `exp(0) = 1`。不可能 overflow。求和里至少有一项是 1，所以总和至少是 1，而 `log(1) = 0`。因此也不可能 underflow 到 `-inf`。

证明：

```text
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (add and subtract c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (factor out exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

令 `c = max(x)`，overflow 就被消除了。

这个技巧在 ML 中随处可见：
- Softmax normalization
- Cross-entropy loss computation
- Sequence models 中的 log-probability summation
- Mixture of Gaussians
- Variational inference

### 为什么 Softmax 需要 Max-Subtraction Trick

Softmax 会把 logits 转换成 probabilities：

```text
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

如果不用这个技巧，logits 为 [100, 101, 102] 时会 overflow：

```text
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

These overflow float32 (max ~3.4e38)? No, 2.69e43 < 3.4e38? Actually:
exp(88.7) is already at the float32 limit.
exp(100) = inf in float32.
```

使用这个技巧时，减去 max(x) = 102：

```text
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

概率完全相同。计算是安全的。这不是优化，而是正确性要求。

### NaN 和 Inf：检测与预防

`nan`（Not a Number）和 `inf`（infinity）会像病毒一样沿着计算传播。梯度更新里有一个 `nan`，weight 就会变成 `nan`，随后每一个输出都会变成 `nan`。训练会在一步之内死亡。

`inf` 如何出现：
- 对很大的正数调用 `exp()`
- 除以零：`1.0 / 0.0`
- accumulations 中的 `float32` overflow

`nan` 如何出现：
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 对负数调用 `sqrt()`
- 对负数调用 `log()`
- 任何涉及已有 `nan` 的 arithmetic

检测：

```python
import math

math.isnan(x)       # True if x is nan
math.isinf(x)       # True if x is +inf or -inf
math.isfinite(x)    # True if x is neither nan nor inf
```

预防策略：

1. Clamp `exp()` 的输入：`exp(clamp(x, -80, 80))`
2. 给分母加 epsilon：`x / (y + 1e-8)`
3. 在 `log()` 内部加 epsilon：`log(x + 1e-8)`
4. 使用稳定实现（log-sum-exp、stable softmax）
5. 使用 gradient clipping 防止 weights 爆炸
6. 调试时，在每次 forward pass 之后检查 `nan`/`inf`

### 数值梯度检查

解析梯度（来自 backpropagation）可能有 bug。Numerical gradient checking 会用有限差分计算梯度，以验证它们。

中心差分公式：

```text
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

它有 O(h^2) 精度，远好于 forward difference `(f(x+h) - f(x)) / h`，后者只有 O(h)。

选择 h：太大，近似会错。太小，灾难性抵消会毁掉结果。`h = 1e-5` 到 `1e-7` 是典型取值。

检查方式：计算解析梯度和数值梯度之间的相对差异。

```text
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

经验规则：
- relative_error < 1e-7：完美，梯度正确
- relative_error < 1e-5：可接受，可能正确
- relative_error > 1e-3：有东西不对
- relative_error > 1：梯度完全错误

实现新的 layer 或 loss function 时，一定要检查梯度。PyTorch 为此提供了 `torch.autograd.gradcheck()`。

### 混合精度训练

现代 GPUs 有专用硬件（Tensor Cores），能让 float16 matrix multiplications 比 float32 快 2-8 倍。Mixed precision training 会利用这一点：

```text
1. Maintain float32 master copy of weights
2. Forward pass in float16 (fast)
3. Compute loss in float32 (prevents overflow)
4. Backward pass in float16 (fast)
5. Scale gradients to float32
6. Update float32 master weights
```

纯 float16 训练的问题：gradients 通常很小（1e-8 或更小）。Float16 会把低于 ~6e-8 的任何东西 underflow 成 0。模型会停止学习，因为所有 gradient updates 都变成了 0。

修复方式是 loss scaling：

```text
1. Multiply loss by a large scale factor (e.g., 1024)
2. Backward pass computes gradients of (loss * 1024)
3. All gradients are 1024x larger (pushed above float16 underflow)
4. Divide gradients by 1024 before updating weights
5. Net effect: same update, but no underflow
```

Dynamic loss scaling 会自动调整 scale factor。从一个较大的值开始（65536）。如果 gradients overflow 到 `inf`，就减半。如果 N 步都没有 overflow，就加倍。

### bfloat16 vs float16：为什么 bfloat16 在训练中胜出

```text
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```

float16 精度更高（10 个 mantissa bits vs 7 个），但范围有限（最大约 ~65,504）。bfloat16 精度更低，但范围和 float32 相同（最大约 ~3.4e38）。

对神经网络训练来说：

- Activations 和 logits 在训练尖峰中经常超过 65,504。float16 会 overflow；bfloat16 能处理。
- float16 需要 loss scaling，但 bfloat16 通常不需要，因为它的范围覆盖了 gradient magnitude spectrum。
- bfloat16 是 float32 的简单截断：丢弃 mantissa 的底部 16 bits。转换很直接，而且 exponent 无损。

float16 更适合 inference，因为此时数值有界，precision 更重要。bfloat16 更适合 training，因为 range 更重要。这也是 TPUs 和现代 NVIDIA GPUs（A100、H100）原生支持 bfloat16 的原因。

### 梯度裁剪

Exploding gradients 发生在 gradients 穿过许多层时指数级增长（常见于 RNNs、deep networks 和 transformers）。一个巨大的 gradient 可以在一步内破坏所有 weights。

两种裁剪方式：

**Clip by value：** 独立 clamp 每个 gradient element。

```text
grad = clamp(grad, -max_val, max_val)
```

简单，但可能改变 gradient vector 的方向。

**Clip by norm：** 缩放整个 gradient vector，使它的 norm 不超过阈值。

```text
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

保留 gradient 的方向。这就是 `torch.nn.utils.clip_grad_norm_()` 做的事情，也是标准选择。

典型取值：transformers 用 `max_norm=1.0`，RL 用 `max_norm=0.5`，更简单的 networks 用 `max_norm=5.0`。

Gradient clipping 不是 hack。它是安全机制。没有它，一个 outlier batch 就可能产生大到足以毁掉数周训练的 gradient。

### 作为数值稳定器的归一化层

Batch normalization、layer normalization 和 RMS normalization 通常被介绍为帮助训练收敛的 regularizers。它们也是 numerical stabilizers。

没有 normalization 时，activations 可能在层间指数级增长或缩小：

```text
Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]
```

Normalization 会在每一层重新居中并重新缩放 activations：

```text
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon`（通常是 1e-5）会在所有 activations 都相同时防止除以零。学得的参数 `gamma` 和 `beta` 让网络恢复它所需的任意 scale。

这会让网络中的 values 始终保持在数值安全范围内，既防止 forward pass 中的 overflow，也防止 backward pass 中的 gradient explosion。

### 常见 ML 数值 Bug

**Bug：Loss 在几个 epochs 后变成 NaN。**
原因：logits 变得太大，softmax overflow。或者 learning rate 太高，weights diverged。
修复：使用 stable softmax（max subtraction）、降低 learning rate、加入 gradient clipping。

**Bug：Loss 卡在 log(num_classes)。**
原因：模型输出接近均匀 probabilities。通常意味着 gradients 正在消失，或者模型完全没有学习。
修复：检查 data labels 是否正确，验证 loss function，检查 dead ReLUs。

**Bug：Validation accuracy 比预期低 1-3%。**
原因：mixed precision 没有正确的 loss scaling。Gradient underflow 会悄悄把小 updates 归零。
修复：启用 dynamic loss scaling，或切换到 bfloat16。

**Bug：某些层的 gradient norms 是 0.0。**
原因：dead ReLU neurons（所有 inputs 都为负），或者 float16 underflow。
修复：使用 LeakyReLU 或 GELU，使用 gradient scaling，检查 weight initialization。

**Bug：Model 在一张 GPU 上正常，但在另一张 GPU 上给出不同结果。**
原因：floating point accumulation order 是非确定性的。GPU parallel reductions 会在不同硬件上以不同顺序求和，而 floating point addition 不满足结合律。
修复：接受小差异（1e-6），或设置 `torch.use_deterministic_algorithms(True)` 并接受速度惩罚。

**Bug：Loss computation 中 `exp()` 返回 `inf`。**
原因：raw logits 没有经过 max-subtraction trick，就被传给了 `exp()`。
修复：使用 `torch.nn.functional.log_softmax()`，它内部实现了 log-sum-exp。

**Bug：从 float32 切换到 float16 后训练发散。**
原因：float16 无法表示低于 6e-8 的 gradient magnitudes，也无法表示高于 65,504 的 activations。
修复：使用带 loss scaling 的 mixed precision（AMP），或者改用 bfloat16。

## 动手实现

### Step 1：展示 floating point precision limits

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### Step 2：实现 naive vs stable softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) would return [nan, nan, nan]
```

### Step 3：实现 stable log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) returns inf
```

### Step 4：实现 stable cross-entropy

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### Step 5：Gradient checking

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## 实际使用

### Mixed precision simulation

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### Gradient clipping

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf detection

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

完整实现见 `code/numerical.py`，其中演示了所有边界情况。

## 交付成果

本课产出：
- `code/numerical.py`，包含 stable softmax、log-sum-exp、cross-entropy、gradient checking 和 mixed precision simulation
- `outputs/prompt-numerical-debugger.md`，用于诊断训练中的 NaN/Inf 和数值问题

这些稳定实现会在 Phase 3 构建 training loop 时再次出现，也会在 Phase 4 实现 attention mechanisms 时再次出现。

## 练习

1. **灾难性抵消。** 使用 naive formula `E[x^2] - E[x]^2`，在 float32 中计算 [1000000.0, 1000001.0, 1000002.0] 的方差。然后用 Welford's online algorithm 再算一次。把误差和真实方差（0.6667）进行比较。

2. **精度搜寻。** 在 Python 中找出最小的正 float32 值 `x`，使得 `1.0 + x == 1.0`。这就是 machine epsilon。验证它是否匹配 `numpy.finfo(numpy.float32).eps`。

3. **Log-sum-exp 边界情况。** 用这些输入测试你的 `logsumexp_stable` 函数：(a) 所有 values 相等，(b) 一个 value 远大于其他值，(c) 所有 values 都非常负（-1000）。验证在 naive version 失败的地方，它能给出正确结果。

4. **对神经网络层做 gradient checking。** 实现一个单独的 linear layer `y = Wx + b` 及其解析 backward pass。使用 `numerical_gradient` 验证一个 3x2 weight matrix 的正确性。

5. **Loss scaling 实验。** 模拟 float16 训练：创建范围在 [1e-9, 1e-3] 内的 random gradients，转换成 float16，并测量有多少比例变成 0。然后应用 loss scaling（乘以 1024），转换成 float16，再 scale back，并再次测量零值比例。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| IEEE 754 | “浮点数标准” | 定义二进制 floating point formats、rounding rules 和特殊值（inf、nan）的国际标准。每个现代 CPU 和 GPU 都实现了它。 |
| Machine epsilon | “精度极限” | 在给定 float format 中，使 1.0 + e != 1.0 成立的最小值 e。对 float32 来说，它大约是 1.19e-7。 |
| Catastrophic cancellation | “减法导致的精度损失” | 相减两个几乎相等的 floating point numbers 时，有效数字会抵消，舍入噪声会主导结果。 |
| Overflow | “数字太大” | 结果超过最大可表示值并变成 inf。exp(89) 会让 float32 overflow。 |
| Underflow | “数字太小” | 结果比最小可表示正数更接近 0，并变成 0.0。exp(-104) 会让 float32 underflow。 |
| Log-sum-exp trick | “先减最大值” | 通过提出 exp(max(x)) 来计算 log(sum(exp(x)))，从而防止 overflow 和 underflow。用于 softmax、cross-entropy 和 log-probability math。 |
| Stable softmax | “不会爆炸的 softmax” | 在 exponentiating 之前减去 max(logits)。数值上结果相同，并且不可能 overflow。 |
| Gradient checking | “验证你的 backprop” | 把 backpropagation 得到的解析梯度与 finite differences 得到的数值梯度比较，以捕捉实现 bug。 |
| Mixed precision | “Float16 forward，float32 backward” | 在速度关键操作中使用低精度 floats，在数值敏感操作中使用更高精度 floats。典型加速是 2-3 倍。 |
| Loss scaling | “防止 gradient underflow” | 在 backprop 之前用一个大常数乘以 loss，让 gradients 留在 float16 可表示范围内，然后在 weight updates 之前除以同一个常数。 |
| bfloat16 | “Brain floating point” | Google 的 16-bit format，包含 8 个 exponent bits（范围和 float32 相同）以及 7 个 mantissa bits（精度低于 float16）。训练时更常用。 |
| Gradient clipping | “限制 gradient norm” | 缩放 gradient vector，使它的 norm 不超过阈值。防止 exploding gradients 破坏 weights。 |
| NaN | “Not a Number” | 由未定义操作（0/0、inf-inf、sqrt(-1)）产生的特殊 float value。会传播到所有后续 arithmetic 中。 |
| Inf | “Infinity” | 由 overflow 或 division by zero 产生的特殊 float value。可以组合产生 NaN（inf - inf、inf * 0）。 |
| Numerical gradient | “暴力导数” | 通过计算 f(x+h) 和 f(x-h)，再除以 2h 来近似 derivative。速度慢，但验证时可靠。 |

## 延伸阅读

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- 权威参考，密集但完整
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) -- NVIDIA 提出 float16 训练中 loss scaling 的论文
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) -- PyTorch 中 mixed precision 的实用指南
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) -- Google 为什么为 TPUs 选择这个格式
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- 一种减少 floating point sums 中舍入误差的算法
