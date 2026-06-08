# DualPipe 并行

> DeepSeek-V3 使用 2,048 张 H800 GPU 训练，MoE experts 分散在节点之间。跨节点 expert all-to-all 通信每 1 GPU-hour 计算就要花 1 GPU-hour 通信。GPU 一半时间都在闲着。DualPipe（DeepSeek，2024 年 12 月）是一个双向 pipeline，会把 forward 和 backward 计算与它们触发的 all-to-all comms 重叠起来。Bubbles 下降，吞吐上升；保留两份模型参数（名字里的 “dual”）在 Expert Parallelism 已经把 experts 分散到不同 ranks 后也并不昂贵。本课是一个 Learn-type walkthrough，解释 DualPipe 到底做了什么，以及为什么 Sea AI Lab 的 DualPipeV refinement 能在只付出略微更紧 bubble 的代价下降低 2x 参数成本。

**类型:** Learn
**语言:** Python (stdlib, schedule simulator)
**先修:** Phase 10 · 05 (distributed training, FSDP, DeepSpeed), Phase 10 · 14 (open-model architectures and MoE)
**时间:** ~60 分钟

## 学习目标

- 说出一个 DualPipe forward-backward chunk 的四个组件，以及为什么每个组件都有自己的 overlap window。
- 解释大规模下的 pipeline bubble 问题，以及“bubble-free”在实践中和营销话术中分别是什么意思。
- 手工追踪 8 个 PP ranks、16 个 micro-batches 的 DualPipe schedule，并确认 forward 与 reverse streams 如何填满彼此的 idle slots。
- 说明 DualPipeV（Sea AI Lab, 2025）的取舍：在 Expert Parallelism 不活跃时，用稍大的 bubble 换掉 2x 参数复制。

## 要解决的问题

在 2k H800 GPU 上训练一个 671B MoE 模型会遇到三个相互叠加的瓶颈：

1. **内存压力。** 每张 GPU 持有模型的一片。序列 8k、61 层、128 heads 下的 activation memory 非常巨大。
2. **Pipeline bubbles。** 传统 pipeline parallelism（GPipe, 1F1B）会让 GPU 在等待本 stage 的输入或梯度时闲置。在 8 个 stage 下，即使用 1F1B scheduling，也大约有 12% GPU 时间会变成 bubble。
3. **跨节点 all-to-all。** 使用 expert parallelism 的 MoE 会把 experts 分散到节点之间。每次 forward pass 都会触发一次 all-to-all，把 token dispatch 到它们的 experts，随后还要一次 combine。在 2k GPU 上，这很容易变成 1:1 compute-to-comm ratio。

这些问题各自都有独立解法：memory 用 gradient checkpointing，pipeline bubbles 用 Zero Bubble（Sea AI Lab, 2023），all-to-all 用 expert-parallel comm kernels。DualPipe 做的是让它们协同工作。这个 schedule 会在一个 forward-backward chunk 内重叠 compute 和 comm，从 pipeline 两端同时注入 micro-batches，并用由此产生的 schedule 把 all-to-all 隐藏进 compute windows。

报告结果：pipeline bubbles 几乎消除，DeepSeek-V3 的 14.8T-token 训练运行中 GPU utilization 超过 95%。

## 核心概念

### Pipeline parallelism refresher

把一个 N 层模型切到 P 个设备上。设备 `i` 持有 layers `i * N/P .. (i+1) * N/P - 1`。一个 micro-batch 会从设备 0 到 P-1 做 forward，再从 P-1 到 0 做 backward。每个设备只有在上一个设备发送 output 后才能开始自己的 forward stage，也只有在下游设备发送 upstream gradient 后才能开始 backward。

GPipe（Huang et al., 2019）一次调度一个 micro-batch，会浪费大部分 GPU 时间。1F1B（Narayanan et al., 2021）为多个 micro-batches 交错 forward 和 backward passes。Zero Bubble（Qi et al., 2023）把 backward pass 切成两部分——backward-for-input（B）和 backward-for-weights（W）——并调度它们填充 bubble。Zero Bubble 之后，pipeline 已经几乎收紧。

DualPipe 是下一步。它在此基础上添加两个想法：

### Idea 1: chunk decomposition

每个 forward chunk 被切成四个组件：

- **Attention.** Q/K/V projections、attention、output projection。
- **All-to-all dispatch.** 把 tokens 发送到它们 experts 的跨节点通信。
- **MLP.** MoE expert computation。
- **All-to-all combine.** 把 expert outputs 带回来的跨节点通信。

Backward chunk 会加入这些组件的梯度版本。DualPipe 调度它们，让 all-to-all dispatch 与下一个 chunk 的 attention compute 并行，让 all-to-all combine 与后续 chunk 的 MLP compute 并行。

### Idea 2: bidirectional scheduling

大多数 pipeline schedules 都从 stage 0 注入 micro-batches，并流向 stage P-1。DualPipe 从两端都注入 micro-batches。Stage 0 看到从那里发起的 forward micro-batches；stage P-1 也看到从那里发起的 forward micro-batches。两条 stream 在中间相遇。

要做到这一点，设备 `i` 必须同时持有 early-pipeline layer `i` 和 late-pipeline layer `P - 1 - i`。这就是 DualPipe 里的 “dual”：每个设备保留两份它需要服务的模型层（一份给每个方向）。在 DeepSeek-V3 的规模上，这是 2x parameter replication cost。它是可负担的，因为 Expert Parallelism 已经把 MoE experts 分散得非常薄，复制两份 non-expert layers 相比之下只是小钱。

关键点是：一个方向的 forward stream 和另一个方向的 backward stream 会正好重叠在单向 schedule 产生 bubbles 的地方。Bubbles 消失。

### 手工追踪一个 schedule

考虑 P = 4 ranks、8 micro-batches，分成 4 个 forward / 4 个 reverse。时间从左到右；行是 device ranks。

```text
           Time →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

读 “F4/F5R” 这种记法：rank 1 在同一个 time slot 里同时运行 micro-batch 4 的 forward（沿 pipeline 从左到右）和 micro-batch 5 的 forward（从右到左）。这就是 “bidirectional” 在操作层面的意思。

在 rank 2 上，两条 cross streams 更早重叠；在 rank 0 和 P-1 上，它们最晚重叠。在 schedule 的稳定中间阶段，每个 rank 都运行某个方向的 forward，并与另一个方向的 backward 重叠。Compute 一直忙。Forward pass 的 all-to-all dispatches 隐藏在 backward compute 中。All-to-all combines 隐藏在 forward compute 中。Bubbles 被挤掉。

### Bubble accounting

标准 1F1B pipeline bubble（每个 rank 浪费的时间）：

```text
bubble_1F1B = (P - 1) * forward_chunk_time
```

Zero Bubble refinement 会降低它，但不会降到零。DualPipe 在稳定阶段中，如果 micro-batch count 可以被 2 倍 pipeline depth 整除，就有 zero bubble。在稳定阶段之外（warmup 和 cooldown），仍有一些 bubble，但它不随 micro-batches 数量增长——这是论文强调的关键性质。

营销话术中： “bubble-free”。技术表述中：bubbles 不随 micro-batch count 增长。Sea AI Lab 后续分析（DualPipeV / Cut-in-half）显示，只有 Expert Parallelism 不是瓶颈时才能达到完全 zero-bubble；在 EP-driven all-to-all 下，总会有一些 scheduling compromise。

### DualPipeV：refinement

Sea AI Lab（2025）观察到，当 EP comm overlap 不是重点时，2x parameter replication 很浪费。他们的 DualPipeV schedule 把 bidirectional injection 折叠成一个 “V-shape” schedule，在单份参数拷贝上运行。Bubble 比 DualPipe 略大，但内存节省可观。DeepSeek 在其开源 DualPipe 实现中，把 DualPipeV 作为 EP-off mode 采用。

取舍如下：

| Feature | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|---------|---------|-----------|------|------------|
| Param copies per device | 2 | 1 | 1 | 1 |
| Bubble vs micro-batches | constant | small growth | grows | grows |
| Compute-comm overlap | full | partial | minimal | partial |
| Use when | EP-heavy MoE | dense or EP-light | baseline | any pipeline |

### 对 14.8T-token 运行意味着什么

DeepSeek-V3 的预训练在 2,048 张 H800 GPU 上消耗了 14.8T tokens，总计约 2.8M GPU-hours。如果使用 naive 1F1B，他们会有 12-15% 损失在 pipeline bubbles 上——也就是 340-420K GPU-hours，足够训练一个完整 70B 模型。DualPipe 收回了其中大部分。没有内部日志很难直接量化它的贡献，但论文中的 claim 是训练期间平均 GPU utilization 超过 95%。

对较小运行（低于 1k GPU），DualPipe 往往过度——pipeline bubbles 相对总成本更小，而 dense-model training 很少碰到 all-to-all bottleneck。对数千 GPU 规模的 frontier MoE training，它几乎是必需的。

### 它在 stack 中的位置

- 与 **FSDP**（Phase 10 · 05）互补。FSDP 在 ranks 间 shard model parameters；DualPipe 在 ranks 间调度 compute。两者可以结合。
- 兼容 **ZeRO-3** gradient sharding。两份拷贝复制的 bookkeeping 需要与 ZeRO 的 sharded gradients 协同。
- 需要针对具体 cluster topology 调优的 **custom all-to-all kernels**。DeepSeek 的开源 kernels 是 reference implementation。

## 实际使用

`code/main.py` 是一个 pipeline schedule simulator。它接收 `(P, n_micro_batches, schedule)`，并打印 1F1B、Zero Bubble、DualPipe、DualPipeV 各自的 stable-phase utilization。它是教学工具——数字与论文中的定性 claim 匹配，但不是生产实测 speedup claim。

Simulator 的价值：用不同 P 和 micro-batch counts 运行它，观察 1F1B 的 bubble fraction 如何增长，而 DualPipe 不会。

真实训练运行的集成注意事项：

- 选择一个能干净整除你的 micro-batch count 的 pipeline-parallel depth。
- 确保你的 expert-parallel mesh 支持 bidirectional all-to-all。DeepSeek 的 kernels 是 reference。
- 第一次做时，预计要在 schedule 本身上烧掉一周 debugging 时间。Bookkeeping 很琐碎。
- 监控每个 rank 的 GPU utilization，而不只是 aggregate。DualPipe 的收益来自收紧 stragglers。

## 交付成果

本课产出 `outputs/skill-dualpipe-planner.md`。给定一个训练集群规格（GPU count、topology、interconnect、model shape），它会推荐 pipeline parallelism strategy、要使用的 scheduling algorithm，以及目标规模下的预期 bubble fraction。

## 练习

1. 在 `(P=8, micro_batches=16, schedule=dualpipe)` 和 `(P=8, micro_batches=16, schedule=1f1b)` 上运行 `code/main.py`。计算 GPU utilization 差异，并把它表示成每百万 training tokens 可回收的 GPU-hours。

2. 手工画出 `(P=4, micro_batches=8, schedule=dualpipe)` 的 schedule table。用 micro-batch ID 和方向标记每个 time slot。找出第一个 bubbles 消失的 time slot。

3. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）的 Figure 5。找出 DualPipe forward chunk 中 all-to-all dispatch 的 overlap window。解释 compute schedule 如何隐藏它。

4. 计算一个 P=8 pipeline stages 的 70B dense model 和一个 P=16 pipeline stages 的 671B MoE model 使用 DualPipe 的 2x parameter overhead。说明为什么 MoE case 的 overhead 比例更小（大多数参数是 experts，并被 shard 到一个大的 EP group 上）。

5. 比较 DualPipe 和 Chimera（2021 年的竞争性 bidirectional scheduler）。以论文 Section 3.4 为参考，找出 DualPipe 添加了而 Chimera 没有的两个具体性质。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Pipeline bubble | “每个 rank 的 idle time” | 因 pipeline stage 等待输入或梯度而浪费的 GPU cycles |
| 1F1B | “默认 pipeline schedule” | One forward / one backward interleaved scheduling；DualPipe 击败的 baseline |
| Zero Bubble | “Sea AI Lab 2023” | 把 backward 切成 B（input gradient）和 W（weight gradient）；几乎完全收紧 pipeline |
| DualPipe | “DeepSeek-V3 schedule” | Bidirectional pipeline + compute-comm overlap；bubbles 不随 micro-batch count 增长 |
| DualPipeV | “Cut-in-half” | V-shape refinement，去掉 2x parameter replication，代价是 bubbles 略大 |
| Chunk | “Pipeline work 的单位” | 一个 micro-batch 通过一个 pipeline stage 的 forward 或 backward pass |
| All-to-all dispatch | “把 tokens 发给 experts” | 将 tokens 路由到其分配的 MoE experts 的跨节点通信 |
| All-to-all combine | “把 expert outputs 带回来” | MLP 后收集 expert outputs 的跨节点通信 |
| Expert Parallelism (EP) | “Experts across GPUs” | 在 ranks 间 shard MoE experts，让不同 GPU 持有不同 experts |
| Pipeline Parallelism (PP) | “Layers across GPUs” | 在 ranks 间 shard model layers；DualPipe 所调度的维度 |
| Bubble fraction | “浪费的 GPU 时间” | (bubble_time / total_time)；DualPipe 要推向零的比例 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437), Section 3.3.2 and Figure 5](https://arxiv.org/abs/2412.19437) — 主要 DualPipe reference
- [DeepSeek — DualPipe GitHub repository](https://github.com/deepseek-ai/DualPipe) — 开源 reference implementation，包括 DualPipeV（Cut-in-half）mode
- [Qi et al. — Zero Bubble Pipeline Parallelism (arXiv:2401.10241, Sea AI Lab 2023)](https://arxiv.org/abs/2401.10241) — Zero Bubble predecessor
- [Sea AI Lab — DualPipe could be better without the Dual](https://sail.sea.com/blog/articles/63) — 影响 DeepSeek EP-off mode 的 DualPipeV analysis
- [Narayanan et al. — PipeDream / 1F1B (arXiv:1806.03377, 2018-2021)](https://arxiv.org/abs/1806.03377) — DualPipe 比较的 1F1B schedule
- [Huang et al. — GPipe (arXiv:1811.06965, 2018)](https://arxiv.org/abs/1811.06965) — 原始 pipeline parallelism paper 和 bubble problem
