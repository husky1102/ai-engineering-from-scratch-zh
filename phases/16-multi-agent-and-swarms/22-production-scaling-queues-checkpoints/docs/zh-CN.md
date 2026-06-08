# Production Scaling：Queues、Checkpoints、Durability

> 将 multi-agent systems 扩展到数千个 concurrent runs，需要 **durable execution**。LangGraph runtime 在每个 super-step 后写一个由 `thread_id` keyed 的 checkpoint（默认 Postgres）；worker crashes 会释放 lease，另一个 worker resume。Agents 可以无限期 sleep，等待 human input。**MegaAgent**（arXiv:2408.09955）运行 per-agent producer-consumer queue，包含三种 states（Idle / Processing / Response）和 two-layer coordination（intra-group chat + inter-group admin chat）。对 LLM streaming 来说，**Fiber/async** 胜过 thread-per-job：threads 99% 时间都在等待 tokens 时 idle，fibers 在 I/O 上 cooperative yield。反方观点：Ashpreet Bedi 的 “Scaling Agentic Software” 主张在 load 证明需要之前，坚持 **FastAPI + Postgres + nothing else**：simple architectures 比预期走得更远。本课构建 durable checkpoint log、带 state transitions 的 per-agent work queue、async-vs-thread demo，并落地务实的 “start simple” rule。

**类型：** 学习 + 构建
**语言：** Python (stdlib, `asyncio`, `sqlite3`)
**先修：** Phase 16 · 09 (Parallel Swarm Networks), Phase 16 · 13 (Shared Memory)
**时间：** ~75 分钟

## 要解决的问题

一个 prototype multi-agent system 在一台 laptop 上用三个 agents 和 in-memory event loop 能工作。你把它搬到 production：

- Agents 有时运行数小时（long research、human-in-the-loop waits）。
- Worker processes 会 crash。重启会丢失 state。
- Peak load 是 average 的 10x；你需要 horizontal scaling。
- 用户按 agent-run 付费；你需要 exactly-once semantics 做 charging。

in-memory event loop 一项都做不到。你需要底层有 durable execution layer。2026 年的 canonical options 是：

1. 带 checkpoints 的 workflow engine（Temporal、LangGraph runtime）。
2. 带 state store 的 message queue（Postgres + SQS/RabbitMQ）。
3. Actor-model frameworks（MegaAgent 的 per-agent producer-consumer）。
4. Hand-rolled FastAPI + Postgres（Bedi 的 argument）。

本课会构建每种方案的 miniature。

## 核心概念

### Durable execution 这个 pattern

durable-execution engine 会在每个 “step”（LangGraph 语言中的 super-step）后持久化完整 program state。crash 时：

```text
worker crashes mid-step
  -> lease timeout
  -> another worker picks up the thread_id
  -> resumes from last checkpoint
  -> no duplicate side effects
```

让它工作需要：

- **Serializable state.** 所有 agent state 都必须可持久化。带 live database connections 的 function closures 无法 surviving。
- **Deterministic resume.** 给定相同 state 和相同 inputs，agent 产生相同 actions（或对 LLM calls defer 到 external deterministic oracle）。
- **Idempotent side effects.** External calls（tool calls、payments）必须 idempotent，或使用 deduplication key。

LangGraph 在每个 super-step 后写 checkpoint；Temporal 在每个 activity 后写；Restate 使用 event-sourced journals。三者实现同一个 pattern。

### LangGraph runtime

每个 agent 有 `thread_id`；state 是 typed dict；每个 super-step 会向 checkpoints table 写一行。resume 时，runtime 从 last checkpoint replay，而不是从头开始。Agents 可以 `interrupt()` 等待 human input；runtime 持久化并释放 worker。当 input 到达时，任意 worker 都能 resume。

这是 2026 年 4 月的 reference production design。

### MegaAgent 的 per-agent queue

arXiv:2408.09955 描述了一个 scale experiment：一个 cluster 中数千个 concurrent agents。architecture：

```text
agent i:
  state ∈ {Idle, Processing, Response}
  in_queue   <- messages addressed to agent i
  out_queue  -> replies + side effects

coordinators:
  intra-group chat  (agents in the same group)
  inter-group admin chat  (high-level routing)
```

two-layer coordination 让 intra-group conversation 可以 dense 发生，而 inter-group 保持 sparse。这是让数千 agents 成本保持 linear 的 pattern。

### Async vs thread-per-job

LLM calls 是 I/O-bound。等待下一个 token 的 thread 99% 时间都是 idle。Threads 每个约 1MB RAM；10,000 个 concurrent calls 时，仅 stacks 就是 10GB。

Fibers（Python `asyncio`、Go goroutines、Rust `tokio`）在 I/O 上 cooperative yield。同样 10,000 calls 可以舒适地放在进程中。在 LLM-agent scale，async 不是 optimization，而是 architecture。

例外：CPU-bound post-processing（embedding、tokenizer tricks）仍然需要 threads 或 processes。把 I/O layer 和 CPU layer 分开。

### Bedi 的 counterpoint

“Scaling Agentic Software”（Ashpreet Bedi, 2026）认为大多数团队在 measured load 之前就 over-engineer。务实 default：

- FastAPI + Postgres。
- 每个 agent run 是一行；state 通过 optimistic concurrency 原地更新。
- 用 `pg_notify` 或简单 Celery worker 做 background jobs。
- retry policy 放在 application code 中。

对于低于约 100 个 concurrent agent-runs 的 manageable tasks，这通常就是你需要的一切。测到它失败时再升级。

规则：当你遇到 simple architectures 无法解决的具体问题时，才采用 durable-execution frameworks。Premature adoption 会把时间烧在不回本的 ceremonies 上。

### Exactly-once semantics

对 paid agent runs，你需要 “exactly-once effective”（at-least-once delivery + idempotent consumer）。工程动作：

- **Dedup key per run.** 在每个 side-effect call 中包含它。
- **Outbox pattern.** side effects 先写入 table，再由 separate process 执行。两个步骤都 idempotent。
- **Compensating transactions.** 当 side effect 成功但 tracking write 失败，schedule compensate。

这些是 database-engineering patterns，不是 LLM-specific。LLM tax 只是 LLM calls 慢；其他都是标准 distributed systems。

### Rainbow deployment

Anthropic 的 multi-agent research system 使用 “rainbow deployments”：多个 agent runtime versions 并发运行，这样 long-running agents 不必在每次 code deploy 时被 kill。对 traffic 的一小片 canary new versions；当旧 versions 的 agents 完成后再 retire。

这是 long-running stateful systems 的标准做法；2026 adaptation 是 agents 可以活数小时，所以 deployment cycles 必须容纳这一点。

### Canonical production checklist

- Durable state（checkpoints、snapshots，或 outbox + replayable log）。
- Idempotent side effects。
- LLM calls 的 async I/O layer。
- 带 dedup 的 at-least-once delivery。
- stateful workloads 的 rainbow/canary deployment。
- Observability：per-agent traces、super-step audit、retry counter。

## 动手实现

`code/main.py` 实现：

- `CheckpointStore`：SQLite-backed checkpoint log，使用 thread-id keys。每个 super-step append 一行。
- `run_with_checkpoint(agent, thread_id)`：模拟 mid-run crash；第二个 worker 从 last checkpoint resume。
- `AgentQueue`：per-agent Idle / Processing / Response state machine，带小型 work queue。
- `demo_async_vs_threads()`：用 asyncio 和 threads 运行 500 个 concurrent simulated “LLM calls”；报告 wall-clock 和 peak memory（近似）。

运行：

```text
python3 code/main.py
```

预期输出：simulated crash 后 checkpoint resume 成功；async version 在 < 1s 内处理 500 concurrent calls；thread version 花费数秒，并且每个 concurrent unit 使用数量级更高的 memory。

## 实际使用

`outputs/skill-scaling-advisor.md` 会根据 load、state-retention needs 和 deploy frequency，建议 durable-execution choice：FastAPI + Postgres、LangGraph runtime、Temporal 或 custom。

## 交付成果

Canonical production hardening：

- **Start simple (Bedi's rule).** FastAPI + Postgres，直到你测到它失败。
- **Instrument everything before optimizing.** per-run latency histogram、per-step time、retry count、failure categorization。
- **Outbox pattern for side effects.** 尤其是 payments 和 external API calls。
- **Rainbow deploys.** deploys 期间永远不要 kill in-flight agent runs。
- **Adopt durable-execution engines (Temporal / LangGraph / Restate) when** 你遇到具体问题：hour-long human-in-the-loop waits、cross-region coordination、complex retry/compensation policies。
- **Async for the I/O layer.** Threads 只用于 CPU-bound post-processing。

## 练习

1. 运行 `code/main.py`。确认 checkpoint resume works；测量 async vs thread concurrency difference。
2. 实现一个 **outbox** table：每个 tool call 先写到 outbox，再由 separate goroutine/task 执行。通过运行同一个 tool call 两次验证 idempotency。
3. 模拟 **rainbow deploy**：两个 concurrent runtime versions；将一半 new thread_ids route 到各自版本；确认旧版本上的 in-flight threads 不会被 interrupt。
4. 阅读 LangGraph runtime doc（见下方链接）。识别哪些 runtime features 在 hand-rolled FastAPI + Postgres 版本中最难 replicate。这是 adopt 的理由，还是可以 defer？
5. 阅读 MegaAgent（arXiv:2408.09955）Section 3。two-layer coordination（intra-group + inter-group admin chat）是 explicit 的。画出如何将它映射到带两个 queue families 的 message queue。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Durable execution | “Persist the program state” | engine 在每个 super-step 后写 state；crash recovery 是 deterministic。 |
| Super-step | “Transactional boundary” | checkpoints 之间的 work unit。LangGraph term。 |
| thread_id | “Agent run identifier” | 绑定 checkpoints 和 resume logic 的 key。 |
| Idempotency | “Safe to retry” | 重复 side effect 与尝试一次产生相同结果。 |
| Outbox pattern | “Decouple side effects” | 将 intent 写到 table；separate executor 执行并标记 done。 |
| At-least-once delivery | “Possible duplicates” | Message queue semantics；dedup key 让 consumer effective-once。 |
| Rainbow deploy | “Overlapping versions” | long-running workloads 期间多个 runtime versions 并发。 |
| Async fiber | “Cooperative yielding” | User-mode concurrency；对 I/O-bound loads 比 threads 便宜。 |
| Checkpoint | “State snapshot” | super-step boundary 上的 serialized state；resume 的 key。 |

## 延伸阅读

- [LangChain — The runtime behind production deep agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — LangGraph runtime design
- [MegaAgent](https://arxiv.org/abs/2408.09955) — per-agent producer-consumer queue；数千 concurrent agents 下的 two-layer coordination
- [Matrix](https://arxiv.org/abs/2511.21686) — 使用 message queues 作为 coordination substrate 的 decentralized framework
- [Temporal docs](https://docs.temporal.io/) — durable execution 的 reference workflow engine
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — 包括 rainbow deployment 的 production lessons
