# 异步与 Hogwild! 推理

> Speculative decoding（Phase 10 · 15）在一个序列内部并行化 token。Multi-agent frameworks 跨整个序列并行化，但需要显式协调（投票、子任务拆分）。Hogwild! Inference（Rodionov et al., arXiv:2504.06261）做的是另一件事：并行运行 N 个相同 LLM 实例，让它们共享同一个 key-value cache。每个 worker 都会立刻看到其他 worker 生成的 tokens。现代 reasoning models——QwQ、DeepSeek-R1——可以通过这个 shared cache 自我协调，不需要任何 fine-tuning。这个方法仍是实验性的，但它打开了一个全新的 inference parallelism 维度，并且与 spec decode 正交。本课会用 stdlib Python 实现一个 two-worker Hogwild! simulator，并解释为什么 shared-cache collaboration 会从现有模型的 reasoning abilities 中涌现出来。

**类型:** Build
**语言:** Python (stdlib)
**先修:** Phase 10 · 12 (inference optimization), Phase 10 · 15 (speculative decoding)
**时间:** ~60 分钟

## 学习目标

- 描述三种常见 parallel-LLM topologies（voting、sub-task、Hogwild!），并说出各自针对什么问题。
- 说明核心 Hogwild! setup：multiple workers、one shared KV cache、通过 self-prompting 产生 emergent coordination。
- 把 worker count `N`、task-level parallelism `p`、coordination overhead `c` 作为变量，计算 Hogwild! 的 wall-time speedup。
- 在 toy problem 上实现 two-worker Hogwild! simulator，并观察 emergent task division。

## 要解决的问题

现代 LLM 通过生成长链 reasoning 来解决难题——5000 tokens 的 step-by-step logic 很常见，深度数学问题上也会出现数万 tokens。在一个 70B 模型上以 35 tokens/sec 解码，50k tokens 需要 24 分钟。这并不交互。

Speculative decoding（Phase 10 · 15）通过在一个序列内部并行化，给你 3-5x 加速。再往后，自回归解码的 sequential dependency 就是硬上限。每个新 token 都依赖之前所有 token。

显然的问题是：能不能跨序列并行化？在同一个问题上运行多个同模型副本，让它们合作，让它们分工？

既有工作：voting ensembles（运行 N 个模型，选 majority answer）、tree-of-thought（分支 reasoning paths 并重组）、multi-agent frameworks（给每个 agent 分配子任务，使用 coordinator）。这些在特定任务领域有帮助。但它们也都引入显式协调机制——voting rules、branch-and-prune logic、agent-to-agent messaging protocols。

Hogwild! Inference 采用不同路径。N 个 workers 共享一个 KV cache。每个 worker 立刻看到所有其他 worker 生成的 tokens，就像它们是自己的 context 一样。这些 workers——无需任何训练或 fine-tuning——会弄清楚如何分工。现代 reasoning models（QwQ、DeepSeek-R1、Claude-family reasoning mode）可以阅读 shared cache，并说出类似“我看到 worker 2 已经处理了 base case，所以我来做 inductive step”的话。

截至 2026 年 4 月，加速与 workload 强相关，且仍处实验阶段。但这个想法值得知道，因为它打开了 inference parallelism 的新轴。

## 核心概念

### Setup

初始化 N 个 worker processes，它们都运行同一个 LLM。不要使用 per-worker KV caches，而是维护一个 shared cache。当 worker `i` 生成 token `t_j` 时，这个 token 被写入 shared cache 的下一个位置。当 worker `k` 走下一步时，它读取 cache 的当前状态（包含所有 N 个 workers 到目前为止生成的一切）。

在 step time，workers 竞争写入 tokens。没有 per-worker position index——cache 是一个单一增长序列。顺序由写入到达时间决定。

### 为什么协调会涌现

Workers 共享同一个 prompt。通常类似于：“You are one of N instances working together on this problem. Each instance reads the shared memory and can see what other instances have written. Avoid redundant work.” Prompt 加 shared cache 就足够了。Reasoning models 会阅读 cache，注意到问题的哪些部分已经被尝试，并且（经常但并不总是）转向未探索部分。

Hogwild! 论文（Rodionov et al., 2025）报告了这些观察：

- Workers 制定计划，并通过 cache 向其他 workers 沟通。
- Workers 注意到其他 workers reasoning 中的错误，并指出它们。
- Workers 在计划失败时适应并提出 alternatives。
- 在被 prompt 要求检查 redundancy 时，workers 会检测重复并 pivot。

这些都不需要 fine-tuning。Emergent behavior 来自模型已经具备的 reasoning capabilities。

### 命名

论文名借用了 Hogwild! SGD（Recht et al., 2011），这是一个 asynchronous-update optimizer。类比是：SGD 的 asynchronous workers 都写入一个 shared parameter vector；Hogwild! Inference 的 workers 都写入一个 shared KV cache。两者都依赖 empirical convergence，而不是 synchronization guarantees。

### RoPE 让这件事可行

Rotary Position Embeddings（RoPE, Su et al. 2021）通过在 Q 和 K vectors 中旋转来编码位置信息。因为 positions 是 rotations，而不是 baked-in offsets，一个 token 的位置可以移动而无需重新计算 KV cache entry。当 worker `i` 写入 shared cache 的位置 `p` 时，其他 workers 读取该位置时可以直接使用 cached entry——不需要 re-rotation。

在 learned-position 或 absolute-position model 中，Hogwild! 会在每次 concurrent write 后都需要 cache invalidation。RoPE 让 cache 保持稳定。

### Wall-time math

令 `T_serial` 为一个 worker 单独解决问题所需时间。令 `p` 为 task-level parallelizable fraction。令 `c` 为每步 coordination overhead（读取扩展后的 cache，决定写什么）。

单 worker 时间：`T_serial`。
如果协调免费，N-worker Hogwild! 时间：`T_serial * ((1 - p) + p / N)`。经典 Amdahl。
有 coordination overhead 时：`T_serial * ((1 - p) + p / N) + c * steps_per_worker`。

要让 worker 有生产力，`c` 必须相对 per-step decode time 很小。对生成 5k+ tokens 的 reasoning models，workers 可以承担数百个 tokens 的 coordination overhead，仍然领先。对短聊天任务，coordination 会主导，Hogwild! 比 serial 更差。

### 具体例子

Reasoning problem：10k tokens chain-of-thought。假设问题有 `p = 0.7` 的 parallelizable content（不同 proof strategies、不同 case analyses），每个 worker 有 `c = 200` tokens coordination overhead。使用 `N = 4` workers：

- Serial time：10000 decode steps。
- Hogwild! time：10000 * (0.3 + 0.7 / 4) + 200 * 4 = 10000 * 0.475 + 800 = 5550 decode steps。
- Speedup：10000 / 5550 = 1.8x。

这很温和。但在更长 reasoning problems（50k tokens）上，coordination overhead 被摊薄，加速会推到 2.5-3x。Hogwild! 是 inference 中的 thread-level parallelism，适用于一个能自然写多线程代码的语言。

### 什么时候使用 Hogwild!

- 长 reasoning problems（数千 tokens），且任务可以跨独立 sub-goals 并行化。
- 已被训练成 step-by-step 思考的 reasoning models。Non-reasoning models 不会很好地自我协调。
- 单节点部署，并且有足够 VRAM 持有 shared cache 加 N 个 worker processes。Cache 是共享的，但每个 worker 有自己的 activation memory。

### 什么时候不该用

- 短交互聊天。Coordination overhead 主导。
- 不能并行化的任务（单条线性 proof、单次 compilation）。N=1 是上限。
- Non-reasoning models。不会出现协调涌现。
- 多节点部署。Shared cache 需要非常快的 cross-worker synchronization。Intra-node 没问题；cross-node 是 latency disaster。

### 实验状态

截至 2026 年 4 月，Hogwild! 是一个有开源 PyTorch 实现的研究方法。还没有生产采用。三个 blocker：

1. 跨 concurrent processes 管理 shared KV cache 是不平凡的工程。
2. Emergent coordination 与任务相关；benchmarks 仍在构建中。
3. 相比 speculative decoding 已经提供的收益，speedups 较温和；两者可以组合，但组合工程又是一层。

值得知道。值得实验。还不值得把产品押在上面。

## 动手实现

`code/main.py` 实现一个 toy Hogwild! simulator：

- 两个 worker processes，每个都是一个 deterministic “LLM”，会以已知概率生成几类 token（work-token、observe-token、coordinate-token）。
- 一个 shared cache（就是 token list），两个 workers 都读写它。
- 一个简单 coordination logic：当 worker 看到另一个 worker 已经在某个 category 中生成足够多 work tokens 时，它会选择不同 category。

Simulator 会在固定 step budget 下运行，并报告：

- 产生的总 work-tokens。
- 总 wall time（worker steps 数）。
- 相对 single worker 的 effective speedup。
- 哪个 worker 写入哪个 token 的 trace。

### Step 1: shared cache

一个所有 workers 都 append 的 list。真实实现中会用简单 locking（Python `threading.Lock`）；这里用 counter 模拟。

### Step 2: worker loop

每个 worker 在每一步：

- 读取当前 shared cache。
- 根据 cache 中已有内容决定要写入的 token category。
- 写入一个 token。

### Step 3: coordination heuristic

如果 category X 已经在 cache 中有 K 个 tokens，而 worker 原本想写 X，则 worker 切到 category Y。这是一个 toy stand-in，对应 reasoning model 的行为：“注意到这部分已经覆盖了，于是去做别的。”

### Step 4: measured speedup

用 N=1 worker 和 N=2 workers、相同总 step budget 运行 simulator。统计产生的 work-tokens。N=2 应该产生约 1.5-1.8x 更多 work-tokens，因为 coordination-driven task division。

### Step 5: stress the coordination

降低 coordination heuristic 的敏感度。再次运行。观察：没有良好协调时，N=2 会重复产生同类 tokens，加速降到 1 以下。这匹配论文观察：只有当 workers 具备自我协调的 reasoning capacity 时，这个技巧才有效。

## 实际使用

截至 2026 年 4 月，Hogwild! 的生产集成仍是 research-grade。来自 Yandex/HSE/IST 的 reference implementation 基于 PyTorch，目标是 DeepSeek-R1 和 QwQ models 上的 single-node multi-process setups。

务实采用路径：

1. Profile 你的 reasoning-task workload。测量 tokens 中 exploratory（multiple strategies、case analyses、search）与 linear 的比例。
2. 如果 exploration 主导，运行一个 two-worker Hogwild! experiment。测量 wall-time improvement。
3. 如果 improvement 低于 1.3x，你处在 coordination-dominated regime。回到 single-worker。
4. 如果 improvement 超过 1.5x，推进到 N=4 并再次测量。Diminishing returns 通常在 N=4-8 左右出现。

与 speculative decoding 组合：每个 Hogwild! worker 都可以独立使用 spec decode。两个 speedups 大致相乘，让一个 3x spec decode 和 1.8x Hogwild! 组合达到相对 naive single-worker decoding 的有效 5.4x。

## 交付成果

本课产出 `outputs/skill-parallel-inference-router.md`。给定 reasoning workload profile（token budget、task parallelism profile、model family、deployment target），它会在 voting、tree-of-thought、multi-agent、Hogwild! 和 speculative decoding strategies 之间路由。

## 练习

1. 使用默认设置运行 `code/main.py`。确认 N=2 Hogwild! 配置在相同 wall time 下产生的 work-tokens 多于 N=1 baseline。

2. 降低 coordination heuristic 的强度（设置 `coordination_weight=0.1`）。重新运行。展示 speedup 崩塌。解释原因：workers 在无法协调时重复劳动。

3. 对一个 50k-token reasoning task 计算预期 Hogwild! speedup，参数为 `p=0.8, c=500`，N=4 workers。再对一个 1k-token chat task 做同样计算，参数为 `p=0.3, c=200`，N=4。为什么一个是胜利，另一个是损失？

4. 阅读 Hogwild! paper 的 Section 4（preliminary evaluation）。找出作者报告的两个 failure modes。描述一个更好的 coordination prompt 如何缓解每个问题。

5. 在 toy 中组合 Hogwild! 和 speculative decoding：每个 worker 内部使用 2-token spec-decode。报告 multiplicative speedup。当两个 workers 都想扩展同一个 shared-cache prefix 时，会出现什么 bookkeeping problem？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Hogwild! | “Parallel workers, shared cache” | N 个相同 LLM 实例并发运行，使用一个 shared KV cache；通过 self-prompting 产生 emergent coordination |
| Shared KV cache | “Coordination medium” | 一个单一增长 KV buffer，所有 workers 都读写；让 token 在 workers 间即时可见 |
| Emergent coordination | “No training needed” | 具备 reasoning 能力的 LLM 可以读取 shared cache 并分工，无需 fine-tuning 或显式协议 |
| Coordination overhead (c) | “用于 orienting 的 tokens” | 每个 worker 读取扩展 cache 并决定下一步的成本；必须相对总 decode time 较小 |
| Parallelizable fraction (p) | “什么可以并行” | Task-level parallelism：总工作中不是内在 sequential 的比例 |
| RoPE enables Hogwild! | “Rotary positions are shift-invariant” | 因 positions 是 rotations，写入 shared cache 不需要重新计算之前 tokens |
| Voting ensemble | “Run N, pick the majority” | 最简单的 parallel inference topology；适合 classification，对 long-form reasoning 较弱 |
| Tree of thought | “Branch and prune” | 探索多个 branches 并 pruning 的 reasoning strategy；显式 coordination logic |
| Multi-agent framework | “Assign sub-tasks” | 每个 agent 获得一个 role；coordinator 编排；protocol overhead 很重 |

## 延伸阅读

- [Rodionov et al. — Hogwild! Inference: Parallel LLM Generation via Concurrent Attention (arXiv:2504.06261)](https://arxiv.org/abs/2504.06261) — Hogwild! paper，对 QwQ 和 DeepSeek-R1 的 preliminary evaluation
- [Recht, Re, Wright, Niu — Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent (arXiv:1106.5730, NeurIPS 2011)](https://arxiv.org/abs/1106.5730) — 原始 Hogwild!，命名来源
- [Su et al. — RoFormer: Enhanced Transformer with Rotary Position Embedding (arXiv:2104.09864)](https://arxiv.org/abs/2104.09864) — RoPE，使 shared-cache inference 可行的性质
- [Yao et al. — Tree of Thoughts: Deliberate Problem Solving with Large Language Models (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) — tree-of-thought reasoning strategy，Hogwild! 与其正交
- [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) — speculative decoding，Hogwild! 可组合的 within-sequence parallelism
- [Hogwild! reference PyTorch implementation](https://github.com/eqimp/hogwild_llm) — 论文实验的 single source of truth
