# Agent Framework 取舍：LangGraph vs CrewAI vs AutoGen vs Agno

> 每个框架都销售同一个 demo（research agent 生成报告），也都隐藏同一个 bug（state schema 与编排层打架）。选择核心抽象匹配你问题形状的框架；其他一切都是你要写两遍的胶水代码。

**类型：** 学习
**语言：** Python
**先修：** Phase 11 · 09（Function Calling），Phase 11 · 16（LangGraph）
**时间：** ~45 分钟

## 要解决的问题

你有一个需要不止一次 LLM 调用的任务。也许它是 research workflow（plan、search、summarize、cite）。也许它是 code-review pipeline（parse diff、critique、patch、validate）。也许它是一个多轮助手，会订机票、写邮件、提交报销。你选择了一个框架。

三天后，你发现框架的抽象会泄漏。CrewAI 给了你 roles，但当“researcher”需要把结构化 plan 交给“writer”时，它会和你作对。AutoGen 给了你 agents 之间的聊天，但没有一等 state，所以你的 checkpoint 是一段 conversation log 的 pickle。LangGraph 给了你 state graph，但要求你在知道 agent 会做什么之前，就命名每个 transition。Agno 给了你 single-agent 抽象，但当你尝试 fan out 到三个并发 worker 时，它会尖叫。

修复办法不是“选择最好的框架”。而是把框架的核心抽象匹配到你问题的形状。本课会画出这张地图。

## 核心概念

![Agent framework matrix：核心抽象 vs 问题形状](../assets/framework-matrix.svg)

四个框架主导了 2026 年的格局。它们的核心抽象并不相同。

| Framework | 核心抽象 | 最适合 | 最不适合 |
|-----------|------------------|----------|-----------|
| **LangGraph** | `StateGraph`：typed state、nodes、conditional edges、checkpointer。 | 拥有显式 state 和 human-in-the-loop interrupts 的 workflow；需要 time-travel debugging 的生产 agent。 | 拓扑未知的松散、角色驱动 brainstorming。 |
| **CrewAI** | `Crew`：roles（goal、backstory）、tasks、process（sequential 或 hierarchical）。 | 带短线性/层级计划的 role-playing 或 persona-driven workflow。 | crew turn history 之外的任何复杂 state；复杂 branching。 |
| **AutoGen** | `ConversableAgent` pair：两个或更多 agent 轮流对话，直到退出条件。 | 多 agent *dialogue*（teacher-student、proposer-critic、actor-reviewer），其中思考从聊天中涌现。 | 已知 DAG 的确定性 workflow；任何需要跨重启持久 state 的任务。 |
| **Agno** | `Agent`：单个 LLM + tools + memory，可组合成 teams。 | 快速构建 single agent 和轻量 team；强 multimodality 和内置 storage drivers。 | 带自定义 reducers 的深度、显式分支 graph。 |

### “抽象”到底是什么意思

一个框架的核心抽象，就是你讲架构时会画在白板上的东西。

- **LangGraph** → 你画一张图。节点是步骤，边是转移，每个点上的 state object 都有类型。心智模型是状态机。
- **CrewAI** → 你画一张组织结构图。每个 role 都有岗位说明，manager 路由 tasks。心智模型是一支小型专家团队。
- **AutoGen** → 你画一个 Slack DM。两个 agent 互发消息；如果需要 moderator，第三个加入。心智模型是聊天。
- **Agno** → 你画一个挂着 tools 的单个盒子。把盒子并排放在一起就是 team。心智模型是“自带电池的 agent”。

### State 问题

state 是大多数框架选择在生产中崩掉的地方。

- **LangGraph。** Typed state（`TypedDict` 或 Pydantic model）、per-field reducers、一等 checkpointer（SQLite/Postgres/Redis）。Resume、interrupt 和 time-travel 都是免费获得的能力。*（参见 Phase 11 · 16。）*
- **CrewAI。** State 通过 `context` 字段以字符串形式在 tasks 之间流动，或通过 `output_pydantic` 以结构化形式流动。开箱没有 durable per-crew store；如果 crew 必须跨重启存活，你要自己接上。
- **AutoGen。** State 是 chat history 和任何用户自定义的 `context`。conversation transcripts 会持久化；任意 workflow state 不会，除非你写 adapters。
- **Agno。** 内置 storage drivers（SQLite、Postgres、Mongo、Redis、DynamoDB），通过 `storage=` 附加到 `Agent`，conversation sessions 和 user memories 会自动持久化。它不是完整 graph checkpointer，而是 session store。

### Branching 问题

每个非平凡 agent 都会分支。谁来决定分支很重要。

- **LangGraph**：你通过 conditional edges 决定。Routing 是带命名分支的 Python 函数。分支是 compiled graph 中的一等对象；checkpointer 会记录走了哪条分支。
- **CrewAI**：hierarchical mode 中由 manager 决定；sequential mode 中由你在构建时决定。Routing 隐含在 task list 里；除了 manager 的 prompt 之外，没有一等的 “if”。
- **AutoGen**：agents 通过聊天决定。分支由谁接下来发言涌现出来。`GroupChatManager` 选择下一位 speaker；你可以手写 `speaker_selection_method`，但默认是 LLM-driven。
- **Agno**：agent 通过下一步调用哪个 tool 决定。Teams 有 coordinator/router/collaborator mode；超出这些的 branching 由开发者负责。

### Observability 问题

- **LangGraph**：通过 LangSmith 或任何 OTel exporter 使用 OpenTelemetry。每次 node transition 都是一个 trace span；checkpoints 同时也是可回放 traces。LangSmith 是第一方选项；Langfuse/Phoenix 也有 adapters。
- **CrewAI**：自 2025 年末起提供一等 OpenTelemetry；集成 Langfuse、Phoenix、Opik、AgentOps。
- **AutoGen**：通过 `autogen-core` 集成 OpenTelemetry；AgentOps 和 Opik 有 connectors。Tracing 粒度是 per-agent-message，而不是 per-node。
- **Agno**：内置 `monitoring=True` flag 和 OpenTelemetry exporters；与 Langfuse 的 session traces 集成紧密。

### 成本与延迟

四个框架都会增加 per-call overhead（框架逻辑、验证、序列化）。按开销从低到高大致排序：Agno ≈ LangGraph < CrewAI ≈ AutoGen。差异主要由框架做了多少额外 LLM routing 决定。CrewAI 的 hierarchical manager 会花 token 决定谁下一步执行；AutoGen 的 `GroupChatManager` 也是如此。LangGraph 只在你写 `llm.invoke` 的地方花 token。Agno 的 single-agent path 很薄。

当每次运行成本重要时，优先使用显式 routing（LangGraph edges、AutoGen `speaker_selection_method`），而不是 LLM-selected routing。

### Interoperability

- **LangGraph** ↔ **LangChain** tools、retrievers、LLMs。一等 MCP adapter（工具作为 MCP servers 导入）。
- **CrewAI** ↔ tools 继承自 `BaseTool`；LangChain tools、LlamaIndex tools 和 MCP tools 都可以适配进来。通过 `allow_delegation=True` 做 crew-to-crew delegation。
- **AutoGen** → `FunctionTool` 包装任意 Python callable；有 MCP adapter。与 AG2 生态中的 agent-to-agent patterns 强耦合。
- **Agno** → `@tool` decorator 或 BaseTool subclass；MCP adapter；tools 可在 agents 和 teams 之间共享。

## 能力要点

> 你能用一句话解释，为什么某个框架适合某个 agent 问题。

构建前 checklist：

1. **画出形状。** 这是 graph（typed state、named transitions）吗？Role play（专家交接工作）吗？Chat（agents 对话直到完成）吗？还是带 tools 的 single agent？
2. **决定谁来分支。** 开发者决定 branching → LangGraph。manager-agent 决定 → CrewAI hierarchical。聊天涌现 → AutoGen。tool-call 决定 → Agno。
3. **检查 state 预算。** 你需要 resume-from-checkpoint 吗？Time-travel 吗？运行中 human interrupts 吗？如果需要，LangGraph 是默认选择；Agno sessions 覆盖 conversation-scoped state。
4. **检查成本预算。** LLM-selected routing 每轮都会额外花 token。如果 agent 每天运行数千次，优先选择显式 routing。
5. **预算框架开销。** 每个框架都是另一个依赖。如果任务只是两次 LLM 调用和一个工具，就写 30 行普通 Python；没有框架比任何框架都便宜。

在你能画出 graph、org chart、chat 或 agent box 之前，拒绝伸手拿框架。拒绝选择一个会迫使你和它的 state model 对抗、无法满足真实需求的框架。

## 决策矩阵

| 问题形状 | 首选框架 | 原因 |
|---------------|---------------------|-----|
| 带 typed state、人工审批、长时间运行的 Workflow DAG | LangGraph | 一等 state、checkpointer、interrupts、time-travel。 |
| 带 distinct roles 的 research / writing pipeline | CrewAI（sequential）或 LangGraph subgraphs | CrewAI 很容易表达 role-per-task；branching 变复杂时用 LangGraph 扩展。 |
| Proposer-critic 或 teacher-student dialogue | AutoGen | 双 agent chat 是它的原生形状。 |
| 带 tools、sessions、memory 的 single agent | Agno | 最薄的设置，内置 storage 和 memory。 |
| 数千个带 reducers 的并行 fanout | LangGraph + `Send` | 唯一拥有一等 parallel-dispatch API 的框架。 |
| 快速 prototype，不承诺框架 | Plain Python + provider SDK | 没有框架就是最快的框架。 |

## 练习

1. **简单。** 取同一个任务：“research Anthropic's headquarters, write a 200-word brief, cite sources”，分别用 LangGraph（四个节点：plan、search、write、cite）和 CrewAI（三个 roles：researcher、writer、editor）实现。报告每次运行的 token 成本和代码行数。
2. **中等。** 用 AutoGen（researcher ↔ writer chat，editor 通过 `GroupChat` 加入）和 Agno（带 `search_tools`、`write_tools` 以及 session store 的 single agent）构建同一任务。按以下维度给四种实现排序：(a) 每次运行成本，(b) crash 后恢复能力，(c) 在 write step 前注入人工审批的能力。
3. **困难。** 构建一个 decision-tree 脚本 `pick_framework.py`，接收一段简短问题描述（JSON：`{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`），并返回一个推荐和一句话理由。用你自己设计的六个 case 验证它。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Orchestration | “agents 如何协调” | 决定下一个运行哪个 node/role/agent 的层。 |
| Durable state | “重启后恢复” | 进程死亡后仍然存活、附加到 checkpoint 或 session store 的 state。 |
| LLM-selected routing | “让模型决定” | planner LLM 每轮选择下一步；灵活，但每次决策都要付 token。 |
| Explicit routing | “开发者决定” | Python 函数或 static edge 选择下一步；便宜且可审计。 |
| Crew | “一个 CrewAI team” | Roles + tasks + process（sequential 或 hierarchical）绑定成一个 runnable。 |
| GroupChat | “AutoGen 的 multi-agent chat” | N 个 agents 之间由 speaker selector 管理的对话。 |
| Team（Agno） | “Multi-agent Agno” | 一组 agents 上的 route / coordinate / collaborate mode。 |
| StateGraph | “LangGraph 的 graph” | Typed-state、node、conditional-edge、checkpointer 抽象。 |

## 延伸阅读

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/)：StateGraph、checkpointers、interrupts、time-travel。
- [CrewAI documentation](https://docs.crewai.com/)：Crews、Flows、Agents、Tasks、Processes。
- [AutoGen documentation](https://microsoft.github.io/autogen/)：ConversableAgent、GroupChat、teams、tools。
- [Agno documentation](https://docs.agno.com/)：Agent、Team、Workflow、storage、memory。
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents)：与框架无关的 pattern library（prompt chaining、routing、parallelization、orchestrator-workers、evaluator-optimizer）。
- [Yao et al., "ReAct: Synergizing Reasoning and Acting" (ICLR 2023)](https://arxiv.org/abs/2210.03629)：每个框架都会包装起来的 loop。
- [Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (2023)](https://arxiv.org/abs/2308.08155)：AutoGen 的设计论文。
- [Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (UIST 2023)](https://arxiv.org/abs/2304.03442)：CrewAI 风格 persona stacks 所依赖的 role-play 基础。
- Phase 11 · 16（LangGraph）：本课拿来对比的框架。
- Phase 11 · 19（Reflexion）：一个能自然映射到 LangGraph、但在 CrewAI 中别扭的模式。
- Phase 11 · 22（Production observability）：如何为你选择的任意框架做 instrumentation。
