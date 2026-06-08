# 长时间运行的后台 Agent：持久执行

> 生产级 long-horizon agents 不会跑在 `while True` 里。每一次 LLM call 都会变成一个带 checkpoint、retry 和 replay 的 activity。Temporal 的 OpenAI Agents SDK integration 已于 2026 年 3 月 GA。Claude Code Routines（Anthropic）可以在没有持久本地进程的情况下运行定时 Claude Code 调用。Session 会在人类输入处暂停，跨 deploy 存活，并从按 `thread_id` 索引的最新 checkpoint 恢复。新 ergonomics 背后是一个老模式：workflow orchestration，只是多了一个新输入：LLM calls 是 non-deterministic activities，恢复时必须以 deterministic 方式 replay。

**类型：** 学习
**语言：** Python (stdlib, minimal durable-execution state machine)
**先修：** Phase 15 · 10 (Permission modes), Phase 15 · 01 (Long-horizon agents)
**时间：** ~60 分钟

## 要解决的问题

考虑一个运行四小时的 agent。它调用三个工具，两次提示用户，并发起四十次 LLM calls。运行到一半时，它所在的 host 重启了。会发生什么？

- 在朴素的 `while True` loop 中：一切都会丢失。run 会从头开始。三个 tool calls（带真实 side effects）会再次执行。用户会再次被要求批准已经批准过的事情。四十次 LLM calls 会重新计费。
- 使用 durable execution：run 从最近的 checkpoint 恢复。已经完成的 activities 不会重新执行；它们的结果会从 durable log 中 replay。用户不需要重新批准已经批准过的事情。已经发起过的 LLM calls 不会重新计费。

这是 workflow engines 已经交付十年的同一个模式（Temporal、Cadence、Uber 的 Cherami）。新的地方在于，LLM calls 现在也是一种 activity：non-deterministic、昂贵、带 side effects，并且能干净地套进这个模式。

本课的主线是：long-horizon reliability 会衰减（METR 观察到 “35-minute degradation”，也就是 success rate 随 horizon 大致按二次关系下降）。Durable execution 让 run 可以长过 reliability profile 支持的范围。如果设计正确，这是一种安全失败的新方式；如果设计错误，也会以不安全的方式失败。

## 核心概念

### Activities、workflows 与 replay

- **Workflow**：deterministic orchestration code。定义 activities 的顺序、分支和等待。它必须 deterministic，这样才能从 event log replay，而不会出现令人意外的 divergence。
- **Activity**：non-deterministic、可能失败的 work unit。LLM call、tool call、file write、HTTP request。每个 activity 都会连同 inputs 以及完成后的 outputs 一起记录。
- **Event log**：durable backing store。每个 activity 的 start、complete、fail、retry，以及每个 workflow decision 都会被记录。
- **Replay**：恢复时，workflow code 从开头重新运行；所有已经完成的 activities 返回其 logged result，而不会重新执行。只有尚未完成的 activities 才会真正运行。

这和 React 针对 virtual DOM 重新 render，或者 Git 从 commits 重建 working tree，是同一种形状。Orchestrator 的 determinism 让 durability 变得便宜。

### 为什么 LLM calls 适合这个模式

LLM calls 具有这些特性：
- Non-deterministic（temperature > 0；即使 temperature 0 也会随 model versions 漂移）。
- Expensive（money 和 latency）。
- Potentially failing（rate limits、timeouts）。
- Side-effectful（如果它们调用 tools）。

这正是 activity profile。把每个 LLM call 包装成 activity，就能得到 exponential backoff retry、跨重启 checkpointing，以及用于 debugging 的 replayable trace。

### 按 `thread_id` 索引的 checkpoints

LangGraph、Microsoft Agent Framework、Cloudflare Durable Objects 和 Claude Code Routines 都收敛到同一种 API shape：用 `thread_id`（或等价物）标识 session；每个 state transition 都持久化到 backend（默认 PostgreSQL，dev 用 SQLite，cache 用 Redis）；resume 会读取最新 checkpoint。

Backend 选择很重要：

- **PostgreSQL**：durable、queryable、能跨 deploy 存活。LangGraph 的默认选项。
- **SQLite**：只适合 local-dev；跨 host 会丢数据。
- **Redis**：快，但如果没有配置 AOF/snapshot 就是 ephemeral。
- **Cloudflare Durable Objects**：透明分布式；按 unique key scoped；可存活数小时到数周。

### Human-input 作为一等状态

Propose-then-commit（Lesson 15）需要一个 durable 的 “waiting on human” state。Workflow 暂停，external queue 持有 pending request，而 approval 会从那个精确位置恢复。没有 durability 时，这只是 best-effort；有了它，隔夜 approval 到达后，workflow 会在早晨接着跑。

### 35 分钟退化

METR 观察到，每个被测 agent class 在超过约 35 分钟连续运行后都会出现 reliability decay。任务时长翻倍，failure rate 约翻四倍。Durable execution 不会修复这一点；它只是让你能跑得比 reliability profile 支持的更久。安全模式是把 durability 和 checkpoints 组合起来：re-entry 时需要 fresh HITL，并用 budget kill switches（Lesson 13）在不管 wall-clock time 的情况下限制 total compute。

### 什么时候 durable execution 是错误答案

- 运行短于几分钟且没有 human input。Overhead > benefit。
- 严格 read-only 的 information retrieval。
- 正确性要求在一个 context window 内 end-to-end 完成的任务（某些 reasoning tasks；某些 one-shot generation）。

## 实际使用

`code/main.py` 用 stdlib Python 实现了一个 minimal durable-execution engine。它支持：

- `@activity` decorator：把 inputs 和 outputs 记录到 JSON event log。
- 一个负责串联 activities 的 workflow function。
- `run_or_replay(workflow, event_log)` function：replay 已完成 activities，而不重新执行它们。

Driver 会模拟一个三 activity workflow，在中途 crash，并展示：（a）朴素 retry 会重新执行所有内容；（b）replay 只运行缺失的 activity。

## 交付成果

`outputs/skill-durable-execution-review.md` 会审查一个拟议的 long-running agent deployment 是否具备正确的 durable-execution shape：activities、determinism、checkpoint backend、human-input state，以及 HITL-on-resume policy。

## 练习

1. 运行 `code/main.py`。观察 naive retry 与 replay 在 activity-execution count 上的差异。修改 crash point，并展示 replay count 如何相应变化。

2. 把 toy engine 改成显式使用 `thread_id`。模拟两个 concurrent sessions 共享同一个 engine，并确认它们的 event logs 不会碰撞。

3. 选取 toy engine 中的一个 activity。引入一个 non-determinism（例如 workflow decision 里的 wall-clock timestamp）。演示 replay 时的 divergence。解释真实 engines 如何处理这种问题（side-effect registration、`Workflow.now()` APIs）。

4. 阅读 LangChain 的 “Runtime behind production deep agents” 文章。列出 runtime 持久化的每一种 state，并说明每一种覆盖哪个 failure mode。

5. 为一个 6 小时 autonomous coding task 设计 checkpoint policy。你会在哪里 checkpoint？Resume-on-crash 长什么样？哪些地方需要 fresh HITL？

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| Workflow | “Agent 的脚本” | Deterministic orchestration code；可从 event log replay |
| Activity | “一个步骤” | Non-deterministic unit（LLM call、tool call）；执行前后都记录 |
| Event log | “Backing store” | 每个 state transition 的 durable record |
| Replay | “Resume” | 重新运行 workflow；已完成 activities 返回 logged results 而不重新执行 |
| Checkpoint | “Save point” | 按 thread_id 索引的 persisted state；resume 时 latest-wins |
| thread_id | “Session key” | 用于 scope durable state 的 identifier |
| 35-minute degradation | “Reliability decay” | METR：success rate 随 horizon 约按二次关系下降 |
| Non-determinism | “Replay 时漂移” | Wall clock、random、LLM output；必须注册为 side effect |

## 延伸阅读

- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — budget、turns 和 resume semantics。
- [Microsoft — Agent Framework: human-in-the-loop and checkpointing](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — RequestInfoEvent shape。
- [LangChain — The Runtime Behind Production Deep Agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — 具体 runtime requirements。
- [OpenAI Agents SDK + Temporal integration (Trigger.dev announcement)](https://trigger.dev) — LLM calls 的 activity shape。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 35-minute degradation 参考。
