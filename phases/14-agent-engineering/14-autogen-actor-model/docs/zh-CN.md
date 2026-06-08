# AutoGen v0.4：Actor Model 与 Agent Framework

> AutoGen v0.4（Microsoft Research，2025 年 1 月）围绕 actor model 重新设计了 agent orchestration。Async message exchange、event-driven agents、fault isolation、natural concurrency。如今该框架进入 maintenance mode，同时 Microsoft Agent Framework（2025 年 10 月 public preview）成为后继者。

**类型:** 学习 + 构建
**语言:** Python (stdlib)
**先修:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**时间:** ~75 分钟

## 学习目标

- 描述 actor model：agents 作为 actors，messages 是唯一 IPC，每个 actor 都有 failure isolation。
- 说出 AutoGen v0.4 的三个 API layers -- Core、AgentChat、Extensions -- 以及各自用途。
- 解释为什么将 message delivery 与 handling 解耦会带来 fault isolation 和 natural concurrency。
- 在 Python 中实现一个 stdlib actor runtime，并把双 agent code-review flow 移植到它上面。

## 要解决的问题

大多数 agent frameworks 是同步的：一个 agent 生产，另一个 agent 消费，都在同一个 call stack 里。失败会让整个 stack 崩溃。Concurrency 是后补的。Distribution 需要重写。

AutoGen v0.4 的答案是 actor model。每个 agent 都是一个 actor，拥有 private inbox。Messages 是唯一交互方式。Runtime 将 delivery 与 handling 解耦。Failures 被隔离到单个 actor。Concurrency 是原生的。Distribution 只是不同 transport。

## 核心概念

### Actors

一个 actor 拥有：

- Private state（外部永远不能直接触碰）。
- Inbox（message queue）。
- Handler：`receive(message) -> effects`，其中 effects 可以是 "reply"、"send to other actor"、"spawn new actor"、"update state"、"stop self"。

两个 actors 不能共享 memory。它们只能发送 messages。

### AutoGen v0.4 的三个 API layers

1. **Core。** 低层 actor framework。`AgentRuntime`、`Agent`、`Message`、`Topic`。Async message exchange，event-driven。
2. **AgentChat。** Task-driven high-level API（替代 v0.2 的 ConversableAgent）。`AssistantAgent`、`UserProxyAgent`、`RoundRobinGroupChat`、`SelectorGroupChat`。
3. **Extensions。** Integrations -- OpenAI、Anthropic、Azure、tools、memory。

### 为什么解耦很重要

在 v0.2 模型里，同步调用 `agent_a.chat(agent_b)` 会阻塞 agent_a，直到 agent_b 返回。在 v0.4 中，`send(agent_b, msg)` 会把 message 放进 agent_b 的 inbox 然后返回。Runtime 稍后投递。三个结果：

- **Fault isolation。** Agent B 崩溃不会让 Agent A 崩溃 -- runtime 会捕获 B handler 中的 failure，并决定如何处理（log、retry、dead-letter）。
- **Natural concurrency。** 同时有许多 messages in flight；actors 并发处理自己的 inbox。
- **Distribution-ready。** 无论 actor 在进程内还是另一台主机上，inbox + transport 都是同一个抽象。

### Topologies

- **RoundRobinGroupChat。** Agents 按固定轮换顺序发言。
- **SelectorGroupChat。** Selector agent 根据 conversation context 选择下一个发言者。
- **Magentic-One。** 面向 web browsing、code execution、file handling 的参考 multi-agent team。构建在 AgentChat 之上。

### Observability

内置 OpenTelemetry 支持。每条 message 都会发出一个 span；tool calls 会携带符合 2026 OTel GenAI semantic conventions（Lesson 23）的 `gen_ai.*` attributes。

### 状态：maintenance mode

2026 年初：AutoGen v0.7.x 对研究和 prototyping 来说稳定。Microsoft 已将主动开发转向 Microsoft Agent Framework（2025 年 10 月 1 日 public preview；目标在 2026 年 Q1 末 1.0 GA）。AutoGen patterns 可以顺畅向前移植 -- actor model 才是持久的思想。

## 动手实现

`code/main.py` 实现了一个 stdlib actor runtime：

- `Message` -- 带 `sender`、`recipient`、`topic`、`body` 的 typed payload。
- `Actor` -- 抽象类型，带 `receive(message, runtime)`。
- `Runtime` -- event loop，包含 shared queue、delivery、failure isolation。
- 一个双 actor demo：`ReviewerAgent` review code，`ChecklistAgent` 运行 checklist；它们交换 messages 直到达成 consensus。

运行：

```text
python3 code/main.py
```

Trace 会展示 message delivery、一个 actor 中模拟的 failure 不会让另一个 actor 崩溃，以及最终收敛到 shared verdict。

## 实际使用

- **AutoGen v0.4/v0.7**（maintenance）-- 适合 research、prototyping、multi-agent patterns。
- **Microsoft Agent Framework**（public preview）-- 向前路径；在刷新后的 API 中延续相同 actor-model ideas。
- **LangGraph swarm topology**（Lesson 13）-- 通过 shared-tool handoffs 实现类似模式。
- **Custom actor runtime** -- 当你需要特定 transport（NATS、RabbitMQ、gRPC）。

## 交付成果

`outputs/skill-actor-runtime.md` 会为给定 multi-agent task 生成最小 actor runtime，以及一个 team template（RoundRobin 或 Selector）。

## 练习

1. 添加 dead-letter queue：当 handler 抛错时，把失败 message 停放起来供人类检查。在你的 toy 中 DLQ 命中频率如何？
2. 实现 `SelectorGroupChat`：selector actor 根据 conversation state 选择谁处理下一条 message。
3. 添加 distributed transport：把 in-process queue 换成 JSON-over-HTTP server，让 actors 能在独立进程中运行。
4. 为每条 message 接一个 OTel span（或 no-op 替身）。按 Lesson 23 发出 `gen_ai.agent.name`、`gen_ai.operation.name`。
5. 阅读 AutoGen v0.4 的 architecture post。把你的 toy 移植到真正的 `autogen_core` API。你跳过了哪些生产中重要的东西？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Actor | "Agent" | Private state + inbox + handler；无 shared memory |
| Message | "Event" | Typed payload；actors 交互的唯一方式 |
| Inbox | "Mailbox" | 每个 actor 的 pending messages queue |
| Runtime | "Agent host" | 路由 messages 并隔离 failures 的 event loop |
| Topic | "Channel" | Actors 之间命名的 publish-subscribe route |
| Fault isolation | "Let it crash" | 一个 actor 失败不会让其他 actors 崩溃 |
| RoundRobinGroupChat | "Fixed-rotation team" | Agents 按顺序轮流发言 |
| SelectorGroupChat | "Context-routed team" | Selector 选择下一个发言者 |
| Magentic-One | "Reference team" | 面向 web + code + files 的 multi-agent squad |

## 延伸阅读

- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) -- redesign post
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) -- graph-shaped alternative
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) -- AutoGen 默认发出的 spans
