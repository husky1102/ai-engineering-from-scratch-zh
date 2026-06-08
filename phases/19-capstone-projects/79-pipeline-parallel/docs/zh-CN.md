# Pipeline Parallel 与 Bubble 分析

> Tensor parallelism 把矩阵乘法切到多个 rank。Pipeline parallelism 把模型切到多个 rank，每个 rank 一个 stage。Microbatch 在 pipeline 中流动。起点和终点的空闲时间就是 bubble；最小化它就是全部手艺。

**类型:** Build
**语言:** Python
**先修:** Phase 19 Track C lessons 42-49
**时间:** ~90 min

## 学习目标

- 将一个 sequential model 切成 N 个 stage，并模拟跨 N 个 rank 的 forward pipeline。
- 使用 GPipe schedule（只 forward 填充，然后 backward）调度 M 个 microbatch 穿过 pipeline，并计算 bubble fraction。
- 将 bubble 与 Megatron-LM 和 PipeDream 使用的 interleaved 1F1B schedule 进行比较。
- 说明 stage assignment：每个 stage 的 compute 均衡比每个 stage 的 parameter count 均衡更重要。

## 要解决的问题

一个 fp16 的 70B 参数模型仅参数就需要 140 GB。没有消费级 GPU 能容纳它。ZeRO-3 会把参数切分到多个 rank，但仍需要每个 rank 为每个 forward step allgather 完整 layer，为每层支付 log(N) hops。Pipeline parallel 走另一条路：把模型切成 N 个 stage，并把一个 stage 放在一个 rank 上。layer 1 的 forward 在 rank 0 完成后把 activation tensor 交给 rank 1；rank 1 运行 layer 2 后交给 rank 2；如此继续。Backward 反向流动。内存线性下降，因为每个 rank 只持有一个 stage；计算是顺序的，这就是 bubble 问题。

Bubble 是 pipeline 开始时的空闲时间（等待第一个 microbatch 到达最后一个 stage）和结束时的空闲时间（等待最后一个 microbatch 回流排空）。有 M 个 microbatch 和 N 个 stage 时，每个 stage 的 bubble fraction 是 (N-1)/(M+N-1)。M=8、N=4 时是 27%。M=64、N=4 时是 4.5%。当每一步有很多 microbatch 时，bubble 会缩小；这意味着每个 microbatch 的 batch size 要小，而这正是驱动 microbatch 设计的约束。

## 核心概念

```mermaid
flowchart LR
  R0[rank 0: stage 0 / layer 0] --> R1[rank 1: stage 1 / layer 1]
  R1 --> R2[rank 2: stage 2 / layer 2]
  R2 --> R3[rank 3: stage 3 / loss]
  R3 -.backward.-> R2
  R2 -.backward.-> R1
  R1 -.backward.-> R0
```

### GPipe schedule

先用所有 M 个 microbatch 的 forward 填满 pipeline，然后再按反向 drain backward。每个 microbatch 的 activation 必须保留到它自己的 backward，因此内存随 M 线性增长。Forward 需要 M+N-1 个 cycle，backward 又需要 M+N-1 个 cycle。每个 stage 的有效工作是 2M 个 cycle；每个 stage 的 bubble 是 2(N-1) 个 cycle。当每个 forward 和 backward 都耗时一个单位时，bubble fraction 是 (N-1)/(M+N-1)。选择远大于 N 的 M 可以隐藏 bubble。

### 1F1B schedule

交错执行：一旦某个 microbatch 的 forward 到达最后一个 stage，就开始它的 backward，并让它向回流动。这个 schedule 在每个 stage 上交替执行一个 forward 和一个 backward。Bubble 仍是 N-1，但 activation memory 受 pipeline depth 约束，而不是受 microbatch count 约束。生产 pipeline 使用 1F1B（Megatron、PipeDream）。本课先实现 GPipe，因为它更简单；把 1F1B 留作练习。

### 为什么每个 stage 的 compute 均衡很重要

如果 stage 0 需要 50 ms，而 stage 1 需要 100 ms，每个 cycle 都被 stage 1 卡住。其他 stage 每个 cycle 都会空闲 50 ms，等待 stage 1 释放。相同 parameter count 是错误轴线：transformer 的计算主要来自 attention 加 MLP，每层 embedding 有很多参数但计算很少。Stage assignment 应该均衡每个 stage 的 FLOPs，而不是权重数量。

### Microbatch 与 batch

一个 pipeline 运行 M 个 microbatch，每个大小为 B。有效 batch size 是 M*B。Pipeline step 结束时的梯度，是合并的 M*B 个样本上的梯度。Bubble fraction 取决于 M；优化器看到的是 M*B。调 M 意味着在 bubble（高 M 更低）和每个 microbatch 的内存（GPipe 下高 M 带来更高 activation memory）之间权衡。

## 动手实现

`code/main.py` 实现：

- `PipelineStage`：一个小型 `nn.Module`，持有某个 stage 的参数并暴露 `forward(activation)`。
- `Pipeline(stages, num_microbatches)`：在模拟 stage 上使用模拟 wall-clock per stage 编排 GPipe schedule。
- `bubble_fraction(num_stages, num_microbatches)`：闭式形式 (N-1)/(M+N-1)。
- 一个 4-stage demo，打印每个 microbatch 的 trace 和测得的 bubble fraction。

运行：

```bash
python3 code/main.py
```

输出：一个 stage-by-microbatch Gantt chart，以及 bubble percentage 与闭式预测的对比。

## 实际生产中的模式

有三种模式能把 pipeline parallel 加固到可上线。

**Activation checkpointing 与 pipeline 配对。** GPipe 中有 M 个 microbatch 在途时，activation memory 是一个 microbatch 的 M 倍。Activation checkpointing 在 backward 时重新计算 forward，用计算换内存；二者结合，才让长序列的 pipeline 可行。

**Stage balance 是测出来的，不是假设出来的。** 生产团队会跑 profiling pass，在目标硬件上测量实际的逐层 compute（FLOPs 和 wall-clock），然后按测量结果 partition。Megatron-LM 的 `--num-layers-per-stage` flag 接受一个 list，用于在不同 stage 的每层成本不同时允许不均匀的 layer count。

**Send-recv schedule 必须避免 deadlock。** 如果 pipeline 中每个 stage 都先 send 再 receive，会在线路上 deadlock。标准修复是交错：偶数 rank stage 先 send 后 recv，奇数 rank stage 先 recv 后 send。本课显式调度 rank，让这种模式可见。

## 实际使用

生产模式：

- **Megatron-LM.** 大规模 pipeline parallel 的参考。使用 1F1B，并支持 tensor + pipeline + data parallel 组合。
- **DeepSpeed Pipeline.** 与 ZeRO 集成；ZeRO-1 + pipeline 是最大开源模型常见组合。
- **PyTorch Pipe.** PyTorch-native pipeline wrapper，建立在 `torch.distributed.pipeline.sync.Pipe` 上。

## 交付成果

Lesson 80 会把每个 stage 的参数 shard 存入 sharded checkpoint。Lesson 81 会在 end-to-end demo 中组合 DDP + ZeRO + pipeline（精神上如此；demo 为了运行时间保留模拟 pipeline）。

## 练习

1. 实现 1F1B，并验证 bubble fraction 与 GPipe 匹配，但 activation memory 有上界。
2. 在更深的模型上 profile 真实 per-stage time，并按测得的 wall-clock 重新均衡 stage。
3. 增加跨 pipeline microbatch 的梯度累积，并检查该梯度等于等价 full-batch forward 的梯度。
4. 将 pipeline 与 activation checkpointing 配对，测量相对于 compute cost 的内存下降。
5. 将 pipeline 与 DDP 组合（每个 pipeline rank 都在一个 data-parallel group 中复制），并推演 2D schedule。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Pipeline | "Model parallel along depth" | 每个 rank 一个 stage，activation 在 stage 间流动 |
| Bubble | "Pipeline idle time" | 开始 + 结束时某些 stage 没有工作的 (N-1) 步 |
| Microbatch | "Slice of the batch" | 一个 forward/backward 单元；M 增大时 bubble 缩小 |
| GPipe | "Fill then drain" | 所有 M 个 forward 后才有 backward；activation memory 高 |
| 1F1B | "Interleaved schedule" | 每个 stage 一个 forward 一个 backward；activation memory 有界 |

## 延伸阅读

- [Huang et al, GPipe: Efficient Training of Giant Neural Networks](https://arxiv.org/abs/1811.06965)
- [Narayanan et al, PipeDream: Generalized Pipeline Parallelism for DNN Training](https://arxiv.org/abs/1806.03377)
- [Megatron-LM pipeline parallel docs](https://github.com/NVIDIA/Megatron-LM)
- Phase 19 Lesson 76 - schedule 使用的 send/recv primitives
- Phase 19 Lesson 78 - ZeRO 与 pipeline 正交，且常与 pipeline 组合
