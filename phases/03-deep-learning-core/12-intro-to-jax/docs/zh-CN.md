# JAX 入门

> PyTorch 会 mutate tensors。TensorFlow 会构建 graphs。JAX 会编译 pure functions。最后这一点会改变你思考深度学习的方式。

**类型:** Build
**语言:** Python
**先修:** Phase 03 Lessons 01-10, basic NumPy
**时间:** ~90 minutes

## 学习目标

- 使用 JAX 的 functional API（jax.numpy、jax.grad、jax.jit、jax.vmap）编写 pure-function neural network code
- 解释 PyTorch 的 eager mutation 与 JAX 的 functional compilation model 之间的关键设计差异
- 应用 jit compilation 和 vmap vectorization，相比 naive Python 加速 training loops
- 在 JAX 中训练一个简单网络，并将显式 state management 与 PyTorch 的 object-oriented approach 进行对比

## 要解决的问题

你知道如何在 PyTorch 中构建神经网络。定义一个 `nn.Module`，调用 `.backward()`，让 optimizer step。它能工作。数百万人都在用它。

但 PyTorch 的 DNA 里有一个约束：它在 Python 中 eager 地逐个追踪操作。每个 `tensor + tensor` 都是一次单独的 kernel launch。每个 training step 都会重新解释同一段 Python 代码。在你需要跨 2,048 个 TPUs 训练一个 540-billion-parameter model 之前，这都没问题。到了那时，overhead 会杀死你。

Google DeepMind 用 JAX 训练 Gemini。Anthropic 用 JAX 训练 Claude。这些不是小规模操作——它们是地球上最大的 neural network training runs。它们选择 JAX，是因为 JAX 把你的 training loop 当成一个可编译程序，而不是一串 Python calls。

JAX 是带有三种超能力的 NumPy：automatic differentiation、JIT compilation to XLA，以及 automatic vectorization。你写一个处理单个 example 的函数。JAX 给你一个能处理 batch、计算 gradients、编译成 machine code，并跨多个 devices 运行的函数。所有这些都不需要改变原始函数。

## 核心概念

### JAX Philosophy

JAX 是一个 functional framework。没有 classes，没有 mutable state，没有 `.backward()` method。相反：

| PyTorch | JAX |
|---------|-----|
| 带 state 的 `nn.Module` class | Pure function: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| Eager execution | 通过 XLA 进行 JIT compilation |
| `for x in batch:` manual loop | `jax.vmap(f)` auto-vectorization |
| `DataParallel` / `FSDP` | `jax.pmap(f)` auto-parallelism |
| Mutable `model.parameters()` | Immutable pytree of arrays |

这不是风格偏好。它是 compiler constraint。JIT compilation 需要 pure functions——相同输入总是产生相同输出，没有 side effects。正是这个限制，让 100x speedups 成为可能。

### jax.numpy：熟悉的表层

JAX 在 accelerators 上重新实现了 NumPy API：

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

相同的函数名。相同的 broadcasting rules。相同的 slicing semantics。但 arrays 位于 GPU/TPU 上，而且每个操作都能被 compiler trace。

一个关键差异：JAX arrays 是 immutable。没有 `a[0] = 5`。而是：`a = a.at[0].set(5)`。一开始会别扭一周，然后就会想通——immutability 正是让 `grad`、`jit`、`vmap` 这类 transformations 可组合的原因。

### jax.grad：Functional Autodiff

PyTorch 将 gradients 附到 tensors 上（`.grad`）。JAX 将 gradients 附到 functions 上。

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad` 接收一个函数，并返回一个计算 gradient 的新函数。没有 `.backward()` call。没有存储在 tensors 上的 computation graph。gradient 只是另一个你可以调用、组合或 JIT-compile 的函数。

这可以任意组合：

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

二阶导数。三阶导数。Jacobians。Hessians。都通过组合 `grad` 完成。PyTorch 也能做到（`torch.autograd.functional.hessian`），但它是外挂式的。在 JAX 中，这是基础。

约束是：`grad` 只适用于 pure functions。不能在里面放 print statements（它们会在 tracing 期间运行，而不是 execution 期间）。不能 mutate external state。没有显式 key management，就不能生成 random number。

### jit：编译到 XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

第一次调用时，JAX 会 trace 这个函数——记录发生了哪些操作，但不执行它们。然后它把 trace 交给 XLA（Accelerated Linear Algebra），也就是 Google 面向 TPUs 和 GPUs 的 compiler。XLA 会 fuse operations、消除多余 memory copies，并生成优化后的 machine code。

后续调用会完全跳过 Python。编译后的代码以 C++ 速度在 accelerator 上运行。

JIT 有帮助的场景：
- Training steps（相同计算重复数千次）
- Inference（相同 model，不同 inputs）
- 任何用相似 shape inputs 调用多次的函数

JIT 适得其反的场景：
- Python control flow 依赖 values 的函数（`if x > 0` 且 x 是 traced array）
- 一次性计算（compilation overhead 超过 runtime）
- Debugging（tracing 会隐藏真实 execution）

control flow 限制是真实存在的。`jax.lax.cond` 替代 `if/else`。`jax.lax.scan` 替代 `for` loops。这些不是可选项——它们是 compilation 的代价。

### vmap：Automatic Vectorization

你写一个处理单个 example 的函数：

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap` 将它提升为处理 batch：

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` 表示：不要在 `params` 上做 batch（共享），在 `x` 的 axis 0 上做 batch。没有 manual `for` loop。没有 reshaping。没有手动传递 batch dimension。JAX 会推断 batch dimension，并 vectorize 整个计算。

这不是语法糖。`vmap` 生成 fused vectorized code，比 Python loop 快 10-100 倍。而且它可以和 `jit`、`grad` 组合：

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

Per-example gradients。一行。没有 hack，在 PyTorch 中几乎不可能这样做。

### pmap：跨设备 Data Parallelism

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` 会在所有可用 devices（GPUs/TPUs）上复制函数并拆分 batch。函数内部，`jax.lax.pmean` 和 `jax.lax.psum` 会跨 devices 同步 gradients。

Google 使用 `pmap`（以及它的继任者 `shard_map`）跨数千个 TPU v5e chips 训练 Gemini。编程模型是：写单设备版本，用 `pmap` 包起来，完成。

### Pytrees：通用数据结构

JAX 作用于 “pytrees”——lists、tuples、dicts 和 arrays 的嵌套组合。你的 model parameters 是一个 pytree：

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

每个 JAX transformation——`grad`、`jit`、`vmap`——都知道如何遍历 pytrees。`jax.tree.map(f, tree)` 会把 `f` 应用于每个 leaf。这就是 optimizers 一次性更新所有 parameters 的方式：

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

没有 `.parameters()` method。没有 parameter registration。tree structure 就是 model。

### Functional vs Object-Oriented

PyTorch 将 state 存在 objects 内：

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX 使用带显式 state 的 pure functions：

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

params 被传入。没有东西被存储。没有东西被 mutate。这让每个函数都可测试、可组合、可编译。也意味着你要自己管理 params——或者使用 Flax、Equinox 这样的 library。

### JAX Ecosystem

JAX 给你 primitives。Libraries 给你 ergonomics：

| Library | Role | Style |
|---------|------|-------|
| **Flax** (Google) | Neural network layers | 带显式 state 的 `nn.Module` |
| **Equinox** (Patrick Kidger) | Neural network layers | Pytree-based, Pythonic |
| **Optax** (DeepMind) | Optimizers + LR schedules | Composable gradient transforms |
| **Orbax** (Google) | Checkpointing | Save/restore pytrees |
| **CLU** (Google) | Metrics + logging | Training loop utilities |

Optax 是标准 optimizer library。它将 gradient transformation（Adam、SGD、clipping）和 parameter update 分离，因此组合起来很简单：

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### 什么时候用 JAX 而不是 PyTorch

| Factor | JAX | PyTorch |
|--------|-----|---------|
| TPU support | First-class (Google built both) | Community-maintained (torch_xla) |
| GPU support | Good (CUDA via XLA) | Best-in-class (native CUDA) |
| Debugging | Hard (tracing + compilation) | Easy (eager, line-by-line) |
| Ecosystem | Research-focused (Flax, Equinox) | Massive (HuggingFace, torchvision, etc.) |
| Hiring | Niche (Google/DeepMind/Anthropic) | Mainstream (everywhere) |
| Large-scale training | Superior (XLA, pmap, mesh) | Good (FSDP, DeepSpeed) |
| Prototyping speed | Slower (functional overhead) | Faster (mutate and go) |
| Production inference | TensorFlow Serving, Vertex AI | TorchServe, Triton, ONNX |
| Who uses it | DeepMind (Gemini), Anthropic (Claude) | Meta (Llama), OpenAI (GPT), Stability AI |

诚实答案是：除非你有具体理由使用 JAX，否则用 PyTorch。这些理由包括——TPU access、需要 per-example gradients、在巨大规模上做 multi-device training，或者在 Google/DeepMind/Anthropic 工作。

### JAX 中的 Random Numbers

JAX 没有 global random state。每个 random operation 都需要一个显式 PRNG key：

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

这在一开始很烦。但它保证了跨 devices 和 compilations 的 reproducibility——这是 PyTorch 的 `torch.manual_seed` 在 multi-GPU settings 中无法保证的性质。

## 动手实现

### Step 1: Setup and Data

我们将使用 JAX 和 Optax 在 MNIST 上训练一个 3 层 MLP。784 个 inputs，两个 hidden layers，分别有 256 和 128 个 neurons，10 个 output classes。

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### Step 2: 初始化 Parameters

没有 class。只是一个返回 pytree 的函数：

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

He-initialization，手动完成。三个 PRNG keys 从一个 seed 拆出。每个 weight 都是 nested dict 中的 immutable array。

### Step 3: Forward Pass

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

Pure functions。Params in，prediction out。没有 `self`，没有 stored state。`loss_fn` 从零计算 cross-entropy——softmax、log、negative mean。

### Step 4: JIT-Compiled Training Step

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` 会在一次 pass 中同时返回 loss value 和 gradients。`@jax.jit` decorator 会把这两个函数编译到 XLA。第一次调用之后，每个 training step 都不再触碰 Python。

### Step 5: Training Loop

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 个 epochs。约 97% test accuracy。第一个 epoch 会慢（JIT compilation）。第 2-10 个 epochs 会很快。

注意缺少了什么：没有 `.zero_grad()`，没有 `.backward()`，没有 `.step()`。整个 update 是一次组合函数调用。Gradients 被计算、由 Adam 转换，并应用到 parameters——全部都在 `train_step` 内完成。

## 实际使用

### Flax：Google Standard

Flax 是最常见的 JAX neural network library。它把 `nn.Module` 加了回来，但使用显式 state management：

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

结构与 PyTorch 相同，但 `params` 与 model 分离。`model.init()` 创建 params。`model.apply(params, x)` 运行 forward pass。model object 没有 state。

### Equinox：Pythonic Alternative

Equinox（Patrick Kidger 开发）将 models 表示为 pytrees：

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

model 本身就是一个 pytree。不需要 `.apply()`。Parameters 只是 model 的 leaves。这更接近 JAX 的思维方式。

### Optax：Composable Optimizers

Optax 将 gradient transformation 与 update 解耦：

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

Gradient clipping、learning rate warmup、weight decay——全部组合成一串 transforms。每个 transform 都看到 gradients、修改它们，并把它们传给下一个。没有 monolithic optimizer class。

## 交付成果

**Installation:**

```bash
pip install jax jaxlib optax flax
```

GPU support：

```bash
pip install jax[cuda12]
```

TPU（Google Cloud）：

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**Performance gotchas:**

- 第一次 JIT call 很慢（compilation）。benchmark 前先 warm up。
- 避免在 JIT 内对 JAX arrays 使用 Python loops。使用 `jax.lax.scan` 或 `jax.lax.fori_loop`。
- `jax.debug.print()` 可以在 JIT 内工作。普通 `print()` 不行。
- 用 `jax.profiler` 或 TensorBoard 做 profile。XLA compilation 可能隐藏 bottlenecks。
- JAX 默认预分配 75% 的 GPU memory。设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 可禁用。

**Checkpointing:**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**本课产出：**
- `outputs/prompt-jax-optimizer.md`——一个用于选择正确 JAX optimizer configuration 的 prompt
- `outputs/skill-jax-patterns.md`——一个涵盖 JAX functional patterns 的 skill

## 练习

1. 给 MLP 添加 dropout。在 JAX 中，dropout 需要 PRNG key——将 key 传过 forward pass，并为每个 dropout layer split key。比较有无 dropout 的 test accuracy。

2. 使用 `jax.vmap` 为一批 32 张 MNIST images 计算 per-example gradients。计算每个 example 的 gradient norm。哪些 examples 的 gradients 最大，为什么？

3. 用一个通用的 `mlp_forward(params, x)` 替换手写 forward function，使其适用于任意层数。使用 `jax.tree.leaves` 自动确定 depth。

4. benchmark 带与不带 `@jax.jit` 的 training step。分别计时 100 steps。在你的硬件上 speedup 有多大？第一次调用的 compilation overhead 是多少？

5. 通过组合 `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))` 实现 gradient clipping。分别用和不用 clipping 训练。绘制训练过程中的 gradient norm，观察效果。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| XLA | “让 JAX 变快的东西” | Accelerated Linear Algebra——一个 compiler，能 fuse operations，并从 computation graph 生成优化过的 GPU/TPU kernels |
| JIT | “Just-in-time compilation” | JAX 在第一次调用时 trace 函数、编译到 XLA，之后调用运行编译版本 |
| Pure function | “没有 side effects” | 输出只依赖输入的函数——没有 global state、没有 mutation、没有不带显式 keys 的 randomness |
| vmap | “Auto-batching” | 将处理单个 example 的函数转换为处理 batch 的函数，而无需重写 |
| pmap | “Auto-parallelism” | 在多个 devices 上复制函数，并拆分 input batch |
| Pytree | “Nested dict of arrays” | JAX 可以遍历和转换的 lists、tuples、dicts 和 arrays 的任意嵌套结构 |
| Tracing | “记录计算” | JAX 用 abstract values 执行函数来构建 computation graph，而不计算真实结果 |
| Functional autodiff | “函数的 grad” | 通过转换 functions 来计算 derivatives，而不是将 gradient storage 附到 tensors 上 |
| Optax | “JAX 的 optimizer library” | 一个 composable gradient transformations library——Adam、SGD、clipping、scheduling——可以链式组合 |
| Flax | “JAX 的 nn.Module” | Google 面向 JAX 的 neural network library，在保持 state 显式的同时添加 layer abstractions |

## 延伸阅读

- JAX documentation: https://jax.readthedocs.io/——官方文档，包含关于 grad、jit 和 vmap 的优秀教程
- “JAX: composable transformations of Python+NumPy programs” (Bradbury et al., 2018)——解释设计哲学的原始论文
- Flax documentation: https://flax.readthedocs.io/——Google 面向 JAX 的 neural network library
- Patrick Kidger, “Equinox: neural networks in JAX via callable PyTrees and filtered transformations” (2021)——Flax 的 Pythonic alternative
- DeepMind, “Optax: composable gradient transformation and optimisation”——标准 optimizer library
- “You Don't Know JAX” (Colin Raffel, 2020)——一份实用 JAX gotchas 和 patterns 指南，来自 T5 作者之一
