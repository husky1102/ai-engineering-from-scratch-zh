# 并行、群体与网络化架构

> 与 supervisor 对照：没有 central decider。Agents 读取 shared event bus，异步领取 work，并把 results 写回。LangGraph 明确支持用于 decentralized、dynamic environments 的 “Swarm Architecture”。Matrix（arXiv:2511.21686）把 control flow 和 data flow 都表示为通过 distributed queues 传递的 serialized messages，以消除 orchestrator bottleneck。Tradeoff 是显式的：用 determinism 和 traceability 换 scalability。Swarm 适合有许多 independent sub-problems 的 tasks；它不适合需要 single coherent plan 的 tasks。

**类型：** 学习 + 构建
**语言：** Python (stdlib, `threading`, `queue`)
**先修：** Phase 16 · 05 (Supervisor Pattern), Phase 16 · 04 (Primitive Model)
**时间：** ~75 分钟

## 要解决的问题

Supervisor 可以扩展到少量 workers。那几百个呢？Supervisor 自身会成为 bottleneck：关于谁做什么的每个 decision 都要穿过一个 agent。一个缓慢的 plan step 会 stall 整个 system。

Swarm architectures 反转这个设计。不是 central planner dispatching work，而是 workers 从 shared queue 领取 work。“Coordination” 被烘进 event bus semantics。没有 orchestrator；system 会一直扩展到 queue 的极限。

## 核心概念

### 形状

```text
                ┌──── shared queue ────┐
                │                      │
       ┌────────┼────────┐  ◄──────┬───┘
       ▼        ▼        ▼         │
     Worker  Worker  Worker   Worker
      A       B       C        D
       │        │        │         │
       └────────┴────────┴─────────┘
                 │
                 ▼
            results pool
```

没有 orchestrator。每个 worker 重复：pull 一个 task、process、write result（并可选地 enqueue follow-ups）。

### Swarm 适合什么时候

- **Many independent tasks。** Scraping、transforming、classifying。Tasks 之间互不依赖。
- **Variable-duration work。** 如果一些 tasks 需要 100ms，另一些需要 10s，swarm 会自动 balance load——fast workers 会 pull next jobs。Supervisor 必须预判 duration。
- **Throughput over determinism。** 你关心 total completion time，而不是 strict ordering。

### Swarm 什么时候失败

- **Ordered workflows。** 如果 step 3 需要 step 2 的 output，swarm 可能让 step 3 在 step 2 完成前触发。
- **Global-plan tasks。** Complex research questions 受益于 planner。一个 researchers swarm 会产生 independent facts，而不是 coherent report。
- **Debugging。** 没有 central log，又是 asynchronous work，复现 bug 会很贵。

### Matrix (arXiv:2511.21686)

Matrix 是 2025 年把 swarm 推到自然结论的 paper：control flow 和 data flow 都是在 distributed queues 上的 serialized messages。没有 central coordinator。Fault tolerance 来自 message durability。Scalability 是 message broker 的问题，而不是 system 的问题。

Contribution：一种 programming model，其中 multi-agent coordination 是“这个 agent subscribe 到哪个 message topic？”而不是“supervisor 选择哪个 agent next?”。这让 system 看起来像 pub/sub event mesh。

### LangGraph 的 Swarm Architecture

LangGraph 2025 docs 明确把 “Swarm Architecture” 描述为 multi-agent patterns 之一：agents 是 nodes，但 edges 形成带 cycles 的 directed graph，任何 node 都能从 pool 中被 activated。Worker 通过 condition 从 available work 中选择，而不是由 supervisor assignment。

### Failure mode：starvation 与 hot-spotting

如果所有 workers 都 pull fastest-available task，long-running tasks 会一直不被 pick，直到只剩它们。经典 queue starvation。

Mitigations：
- Priority queues with explicit aging（随着 wait time 提高 priority）。
- Worker specialization：一些 workers 只接 “long” tasks。
- Back-pressure：限制有多少 fast tasks 进入 queue。

### Content-based routing link

Swarm 与 content-based routing（Lesson 22）天然搭配。不是使用 generic queue，而是每种 message type 一个 queue。Specialist workers 只 subscribe 自己的 type。这是可扩展到 thousands of agents 的 message-bus architectures 基础。

## 动手实现

`code/main.py` 实现一个由 4 个 worker threads 组成的 swarm，它们从 shared `queue.Queue` pull。Tasks 有 variable durations（有些 fast，有些 slow）。Demo 对比：

- **Sequential baseline：** 一个 worker 串行处理所有 tasks。
- **Fixed assignment：** 每个 task 预先 assigned 到特定 worker（supervisor-style）。
- **Swarm：** workers 从 shared queue pull。

Swarm 自动 balance load；fixed assignment 会在 assigned task 很慢时让 fast workers idle。

运行：

```text
python3 code/main.py
```

Output 展示 per-worker task counts（swarm distributes unevenly but optimally）和 wall-clock times。

## 实际使用

`outputs/skill-swarm-fit.md` 评估一个 task 应该使用 swarm 还是 supervisor。Inputs：task independence、duration variance、ordering requirements、debuggability needs。

## 交付成果

Checklist：

- **Priority queue with aging。** 防止 long-task starvation。
- **Worker idempotency。** 如果 worker mid-run crash，一个 task 可能会被 pulled more than once。Workers 必须 idempotent。
- **Durable queue。** Production 使用 Kafka、Redis Streams 或 database-backed queue。`queue.Queue` 只在 memory 中。
- **Observability per task。** 每个 task 都有 trace ID；每个 worker 都用它 log start/end。
- **Back-pressure。** 如果 queue 增长快于 workers drain，slow the producer。

## 练习

1. 运行 `code/main.py`。在 variable-duration workload 上，swarm 比 sequential 快多少？比 fixed assignment 快多少？
2. 增加一个 priority queue variant（使用 `queue.PriorityQueue`）。按 task 的 “importance” field 分配 priority。观察 low-priority tasks 在 continuous load 下是否 ever starve。
3. 实现 hot-spot detector：当任意 worker 处理的 tasks 数量达到 slowest worker 的 3× 时 log。这说明 task-duration distribution 有什么特征？
4. 阅读 Matrix paper（arXiv:2511.21686）abstract 和 Section 3。识别 Matrix 接受的一个具体 tradeoff（scalability gain）以及它放弃的一个东西（traceability、determinism）。
5. 把 swarm demo 改成使用由 (task_type, payload) tuples 组成的 `queue.Queue`，workers 只 subscribe 特定 types。当 tasks heterogeneous 时，哪些 routing rules 合理？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Swarm architecture | "Decentralized agents" | Workers 从 shared queue pull；没有 central orchestrator。 |
| Event bus | "Agents subscribe to topics" | 按 type 或 content 把 tasks route 给 workers 的 message broker。 |
| Starvation | "Task never runs" | Low-priority task 因为 higher-priority work 持续到来而一直不被 picked。 |
| Hot-spotting | "One worker drowns" | Load imbalance：一个 worker 拿到大多数 tasks。 |
| Back-pressure | "Slow down the producer" | 当 queue 填满时 signal upstream 停止 producing 的 mechanism。 |
| Idempotent worker | "Safe to re-run" | 一个 task 被 processed twice 时产生相同 result。因为 workers 可能 mid-run crash，所以需要它。 |
| Durable queue | "Survives crashes" | 由 disk 或 replicated storage backed 的 queue；worker crash 时 tasks 不会丢失。 |
| Matrix framework | "Full message-passing swarm" | Data 和 control flow 都是 distributed queues 上的 serialized messages。 |

## 延伸阅读

- [LangGraph workflows and agents — Swarm Architecture](https://docs.langchain.com/oss/python/langgraph/workflows-agents) —— explicit swarm support
- [Matrix — A Decentralized Framework for Multi-Agent Systems](https://arxiv.org/abs/2511.21686) —— full message-passing swarm
- [Anthropic engineering — why supervisor not swarm in Research](https://www.anthropic.com/engineering/multi-agent-research-system) —— 一个特定 production system 为什么明确选择 supervisor 而不是 swarm
- [AutoGen v0.4 actor-model docs](https://microsoft.github.io/autogen/stable/) —— event-driven actor rewrite，比 v0.2 的 GroupChat 更接近 swarm
