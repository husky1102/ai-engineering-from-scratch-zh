# LangGraph：状态图与持久执行

> LangGraph 是 2026 年低层状态化编排的参考实现。Agent 是状态机；nodes 是函数；edges 是 transitions；state 是不可变的，并在每一步后 checkpoint。任何失败都能从离开的位置精确恢复。

**类型:** 学习 + 构建
**语言:** Python (stdlib)
**先修:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**时间:** ~75 分钟

## 学习目标

- 描述 LangGraph 的核心模型：包含 immutable state、function nodes、conditional edges 和 post-step checkpoints 的状态机。
- 说出文档强调的四种能力：durable execution、streaming、human-in-the-loop、comprehensive memory。
- 解释 LangGraph 支持的三种编排拓扑：supervisor、peer-to-peer（swarm）、hierarchical（nested subgraphs）。
- 实现一个 stdlib state graph，包含 immutable state、conditional edges 和 checkpoint/resume cycle。

## 要解决的问题

Agents 和 workflows 共享一个问题：当 40 步运行在第 38 步失败时，你想从第 38 步恢复，而不是从头开始。二等公民式的状态模型会迫使 operators 在一个假定每次都是 fresh runs 的库外面硬拼 retries。

LangGraph 的设计答案是：state 是一等 typed object，mutations 是显式的，且每个 node 后都会持久化 checkpoints。Resume 就是一次 `load_state(session_id)` 调用。

## 核心概念

### Graph

一个 graph 由以下部分定义：

- **State type。** 每个 node 读取并变更的 typed dict（或 Pydantic model）。
- **Nodes。** 纯函数 `(state) -> state_update`。返回后，updates 会 merge 到 state。
- **Edges。** Nodes 之间的 conditional 或 direct transitions。
- **Entry and exit。** `START` 和 `END` sentinel nodes 标记边界。

示例：一个 agent 包含 `classify`、`refund`、`bug`、`sales`、`done` nodes -- 也就是 graph 形态的 routing workflow。

### Durable execution

每个 node 返回后，runtime 会序列化 state 并写入 checkpointer（SQLite、Postgres、Redis、自定义）。如果第 N 步失败，runtime 可以 `resume(session_id)`，并以精确 state 从第 N+1 步继续。

LangGraph 文档明确强调了这件事对一些生产用户的重要性：Klarna、Uber、J.P. Morgan。重点不是 graph shape 本身；而是 graph shape 加上 checkpointing 让恢复变便宜。

### Streaming

每个 node 都可以 yield partial output。Graph 会把 per-node-delta events stream 给 caller，让 UI 能随着 graph 运行而更新。

### Human-in-the-loop

在 nodes 之间检查并修改 state。实现方式：在关键 node 前暂停，把 state 展示给人类，接受修改，然后 resume。Checkpointer 让这件事很容易，因为 state 已经被序列化了。

### Memory

短期（一次 run 内 -- state 中的 conversation history）和长期（跨 runs -- 通过 checkpointer 加独立 long-term store 持久化）。LangGraph 通过 tools 与外部 memory systems（Mem0、自定义）集成。

### 三种拓扑

1. **Supervisor。** 中央 router LLM 分发给 specialist subagents。`langgraph-supervisor` 中的 `create_supervisor()`（不过 LangChain 团队在 2026 年建议为了更好控制 context，直接通过 tool calls 来做这件事）。
2. **Swarm / peer-to-peer。** Agents 通过共享 tool surface 直接 hand off。没有中央 router。
3. **Hierarchical。** Supervisors 管理 sub-supervisors，以 nested subgraphs 实现。

### 这个模式容易出错的地方

- **Checkpoints too small。** 只 checkpoint conversation turns 会让 tool state 和 memory writes 无法恢复。Full state 必须可序列化。
- **Non-deterministic nodes。** Resume 假设 node inputs 会产生同样的 state update。Random seeds、wall-clock、external APIs 都必须被捕获。
- **过度使用 conditional edges。** 每条 edge 都是 conditional 的 graph 会变成无法推理的状态机。优先使用带少量分支的 linear chains。

## 动手实现

`code/main.py` 实现了一个 stdlib stateful graph：

- `State` -- 一个 typed dict，包含 `messages`、`step`、`route`、`output`、`human_approval`。
- `Node` -- 接收 state 并返回 update dict 的 callable。
- `StateGraph` -- nodes + edges + conditional edges + run + resume。
- `SQLiteCheckpointer`（in-memory fake）-- 每个 node 后序列化 state；`load(session_id)` 恢复。
- 一个 demo graph：classify -> branch(refund / bug / sales) -> human gate -> send。

运行：

```text
python3 code/main.py
```

Trace 会展示第一次运行在 human gate 失败、状态被持久化，然后 resume 产出最终结果。

## 实际使用

- **LangGraph** -- 参考实现，production-ready。使用 `create_react_agent`、`create_supervisor`，或者构建自己的 graph。
- **AutoGen v0.4**（Lesson 14）-- 面向高并发场景的 actor model 替代方案。
- **Claude Agent SDK**（Lesson 17）-- 带内置 session store 的 managed harness。
- **Custom** -- 当你需要精确控制 state shape 或 checkpointer backend。

## 交付成果

`outputs/skill-state-graph.md` 会在任何目标 runtime 中生成 LangGraph 形态的 state graph，并接好 checkpointing 与 resume。

## 练习

1. 当 classification confidence 低于阈值时，从 `classify` 添加一条通往 `end` 的 conditional edge。在人类手动设置 `route` 后恢复运行。
2. 把 SQLite-like fake 换成真正的 SQLite checkpointer。测量每步 serialization overhead。
3. 实现 parallel edges：两个 nodes 并发运行，由自定义 reducer 合并。Immutable state 在这里带来什么？
4. 阅读 `langgraph-supervisor` reference。把 toy 移植到 `create_supervisor`。比较 trace shapes。
5. 添加 streaming：每个 node 在运行时 yield partial state。打印抵达的 deltas。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| State graph | "Agent as state machine" | Typed state + nodes + edges + reducers |
| Checkpointer | "Persistence backend" | 每个 node 后序列化 state；支持 resume |
| Reducer | "State merger" | 将当前 state 与 node update 组合起来的函数 |
| Conditional edge | "Branch" | 由 state 函数选择的 edge |
| Subgraph | "Nested graph" | 在另一个 graph 内作为 node 使用的 graph |
| Durable execution | "Resume from failure" | 用精确 state 从最后成功的 node 重启 |
| Supervisor | "Router LLM" | Specialist subagents 的中央 dispatcher |
| Swarm | "P2P agents" | Agents 通过 shared tools hand off；没有中央 router |

## 延伸阅读

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) -- 参考文档
- [langgraph-supervisor reference](https://reference.langchain.com/python/langgraph/supervisor/) -- supervisor pattern API
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) -- actor-model 替代方案
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) -- session store 和 subagents
