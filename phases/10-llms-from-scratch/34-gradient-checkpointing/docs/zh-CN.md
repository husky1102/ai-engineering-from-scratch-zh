# 梯度检查点与激活重计算

> Backprop 会保留每个 intermediate activation。在 70B parameters 和 128K context 下，每个 rank 是 3 TB activations。Checkpointing 用 FLOPs 换 memory：不保存，改为 recompute。问题是该丢弃哪些 segments，答案不是“全部丢弃”。

**类型:** Build
**语言:** Python (with numpy, optional torch)
**先修:** Phase 10 Lesson 04 (Pre-Training Mini-GPT), Phase 10 Lesson 05 (Scaling & Distributed)
**时间:** ~70 minutes

## 要解决的问题

训练 transformer 时，每层都会存储 backward 中每个需要求导操作的 inputs：attention inputs、Q/K/V projections、softmax output、FFN inputs、norm outputs 和 residual stream。对于 hidden size `d`、sequence length `L`、batch `B` 的一层，这大约是 `12 * B * L * d` floats per layer。

对 `d=8192, L=8192, B=1`，BF16 下每层是 800 MB。64-layer model 是 51 GB activations——这还没乘 microbatch size，没加 attention-softmax intermediates（每 head `L^2`），也没考虑 tensor-parallel partial copies。

两头收费：BF16 weights 加 optimizer state 也许能放进 80GB，但 activations 会把你推爆。Gradient checkpointing（也叫 activation recomputation）是标准修复。丢弃大多数 activations；backward 时重做 forward 把它们拿回来。代价：额外 FLOPs。收益：memory 按 checkpoint segments 与 total layers 的比例下降。

朴素做法下，checkpointing 每 step 大约多花 33% forward-pass FLOPs。做得好——按 Korthikanti et al. 的 “smart selection” 做 selective checkpointing——可以用低于 5% FLOP overhead 换 5x memory savings。有了 FP8 matmuls、FSDP offload 和 expert-parallel MoE 后，这很重要：你既负担不起 memory，也负担不起浪费的 compute。

## 核心概念

### Backward 实际需要什么

`output = layer(input)`。Backward 想要 `grad_input` 和 `grad_params`。为计算它们，它需要：

- `input`（用于计算 linear layers 的 `grad_params = input.T @ grad_output`）
- 一些 activation derivative intermediates（ReLU/GELU/softmax 的 derivative 依赖 activation value）

Forward pass 会自动把这些存在 autograd graph 里。每个 `tensor.retain_grad()` 和每个需要 input 的 op 都会保留引用。

### Naive Full Checkpointing

把 network 分成 `N` 个 segments。Forward 期间只保存每个 segment 的*输入*。当 backward 需要 intermediates 时，重新运行该 segment 的 forward pass 来 materialize 它们，然后求导。

例子：32-layer transformer 拆成 32 个 segment，每段 1 layer。

- Memory：32 个 layer-inputs（小）vs 32 * (activation volume per layer)（巨大）。
- Extra compute：每个 segment 额外 1 次 forward，即总 forward FLOPs 约多 33%（因为 backward 是 forward 的 2x，完整 step 从 1 + 2 = 3 units 变成 1 + 1 + 2 = 4 units）。

这是 Chen et al. 2016 原始 recipe：每 `sqrt(L)` layers 放一个 checkpoint，以平衡 memory 和 compute。L=64 时，就是 8 checkpoints。

### Selective Checkpointing (Korthikanti 2022)

不是所有 activations 代价相同。Attention softmax output 是 `B*L*L*heads`，随 sequence length *二次*增长。FFN hidden activation 是 `B*L*4d`，线性增长。长序列时 softmax 占主导。

Selective checkpointing 保留便宜的 activations（linear projections、residuals），只 recompute 昂贵的 ones（attention）。你用最少 FLOPs 重新计算，却节省 O(L^2) memory。

Megatron-Core 把它实现为 “selective” activation recomputation。多数 2024+ frontier training runs 都使用它。

### Offload

Recompute 的替代方案：在 forward 和 backward 之间把 activations 送到 CPU RAM。需要 PCIe bandwidth；当 idle bandwidth 超过 rematerialization 成本时有益。Mixed strategies 很常见：一些 layers checkpoint，另一些 offload。

FSDP2 把 offload 作为 first-class option。GPU 在 memory 上 bottleneck、但 CPU-GPU transfer 有余量时，offload 很有价值。

### Recompute Cost Model

每 `k` layers（总共 `L`）做 naive checkpointing 时的 per-step FLOPs：

```text
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # one extra forward per layer in the segment
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

Selective checkpointing 只 recompute attention kernel，而不是整层：

```text
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### Memory Savings Model

每层 activation volume：`A`。`L` 层总 activation memory：`L * A`。

Full checkpoint（segment size 1）：只存 `L * input_volume`（标准 transformer 中约为 `L * 1/10 A`）。节省约 `9 * L * A * 1/10`。

每 `k` layers checkpoint：存 `L/k * A`，加上 active segment 内 `k-1` 层的量。

当 `k = sqrt(L)` 时，memory 和 recompute cost 都按 `sqrt(L)` 缩放——这是 uniform-cost layers 的最优 tradeoff。

### 什么时候不要 Checkpoint

- Pipeline stage 中已经 in-flight 的最内层。它们无论如何都要完成。
- 如果 first 和 last layers 支配 stage compute（transformers 中少见），不要 checkpoint。
- Attention kernels 已经使用 FlashAttention 时——Flash 已经快速 recompute softmax，额外 layer-level checkpointing 带来的收益有限。

### Implementation Patterns

1. **Function wrapper:** 用 `torch.utils.checkpoint.checkpoint(fn, input)` 包住一个 segment。PyTorch 只保存 `input`，backward 时 recompute 其余部分。

2. **Decorator-based:** 把 layers 标记为 checkpointable；trainer 在 config time 决定哪些 segments 被 wrap。

3. **Manual explicit recompute:** 自己写 backward pass，调用自定义 `recompute_forward`，用保存的 input 复制 forward。

三者 functional result 相同。Wrappers 是标准 idiom。

### Interaction with TP / PP / FP8

- **Tensor parallel:** Checkpoint inputs 在 recompute 时必须 gather 或 rescatter；要处理 communication cost。
- **Pipeline parallel:** 典型模式是 checkpoint 每个 pipeline-stage 的 forward，使 reverse-order microbatches 能复用 activation memory。
- **FP8 recompute:** Recompute 期间更新的 amax histories 必须匹配原 forward，否则 FP8 scale 会 drift。多数 frameworks 会 snapshot scale。

## 动手实现

### Step 1: A Toy Model With Segments

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### Step 2: Naive Backward Needing All Activations

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### Step 3: Checkpoint-Every-k Memory

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### Step 4: Cost Model

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### Step 5: Memory Estimator

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### Step 6: Optimal Segment Size

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### Step 7: Selective Checkpoint Decision

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## 实际使用

- **torch.utils.checkpoint**: `from torch.utils.checkpoint import checkpoint` — PyTorch 中的 canonical wrapper。包住一个 function；只存 inputs，backward 时 recompute。
- **Megatron-Core activation recomputation**: 支持 `selective`、`full` 和 `block` modes。2024+ frontier training 的标准配置。
- **FSDP2 offload**: FSDP2 shards 中使用带 `offload_policy` 的 `module.to_empty(device="cpu")`，把 activations shard 到 CPU 而不是 recompute。
- **DeepSpeed ZeRO-Offload**: 面向 optimizer states 和 activations 的 CPU offload，是 checkpointing 的补充。

## 交付成果

本课产出 `outputs/prompt-activation-recompute-policy.md`——一个接收 model config（layers、hidden、seq、batch）和 available GPU memory，并输出 per-layer recompute policy（none / selective / full / offload）的 prompt。

## 练习

1. 验证正确性。运行 `model_forward` + `model_backward`（full activations）vs `model_forward_checkpointed` + `model_backward_checkpointed`（segments）。Parameter gradients 必须在 machine precision 内一致。

2. Sweep segment size `k` from 1 to `L`。画 FLOP overhead 和 memory。找到曲线 knee。

3. 实现 selective checkpointing：存 attention-module input，但不存 intermediates。对 seq=8192 的 32-layer model，测量相对 full-layer checkpointing 的 FLOP overhead。

4. 添加 offload。把 segment inputs 保存到模拟 “CPU buffer”（单独 list）。用 bytes/time 测量 “PCIe bandwidth”，并找出 offload 与 recompute 的 breakeven point。

5. Benchmark 一个真实 PyTorch transformer，分别使用和不使用 `torch.utils.checkpoint`。测量 memory（通过 `torch.cuda.max_memory_allocated`）和 step time。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Gradient checkpointing | “Save memory by redoing forward” | 只存 segment inputs；backward 时 recompute intermediates 以获得支持 gradient 的 tensors |
| Activation recomputation | “Same as checkpointing” | 同一技术在 HPC 语境下的名字 |
| Segment size (k) | “How many layers per checkpoint” | 一起丢弃并 rematerialize intermediates 的层数 |
| Selective checkpointing | “Korthikanti's trick” | 只 recompute expensive-to-store activations（attention softmax）；保留便宜 ones |
| Full checkpointing | “The naive version” | Recompute 每个 segment 中每层的 intermediates |
| Block checkpointing | “Coarse-grained” | Checkpoint 整个 transformer blocks；最大 granularity |
| FLOP overhead | “The compute tax” | 每 step 额外 FLOPs = (recompute FLOPs) / (fwd + bwd FLOPs)；naive 33%，selective 5% |
| Activation offload | “Ship to CPU” | 在 forward->backward 之间把 activations 移到 CPU RAM；recompute 的替代方案 |
| sqrt-L rule | “The classical optimum” | 对 uniform-cost layers，optimal checkpoint spacing 是 sqrt(L) layers |
| Attention-softmax volume | “The O(L^2) problem” | L^2 * heads * batch floats；长 context 下主导 activation memory |

## 延伸阅读

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- formalized gradient checkpointing 的原始论文
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) -- selective activation recomputation 和 formal cost analysis
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) -- 通过 reverse-mode rematerialization 实现 constant-memory 的替代方案
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) -- scale 上的 activation offload
- [PyTorch torch.utils.checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html) -- 标准 API
- [Megatron-Core activation recomputation documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) -- selective、full 和 block modes
