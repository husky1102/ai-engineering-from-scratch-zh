# 多智能体原语模型

> 2026 年发布的每个 multi-agent framework——AutoGen、LangGraph、CrewAI、OpenAI Agents SDK、Microsoft Agent Framework——都是四维设计空间中的一个点。四个原语，仅此而已：agent、handoff、shared state、orchestrator。本课从零构建它们，在一个玩具系统里跑通全部四者，然后把每个主流 framework 映射到同一组坐标轴上，让你能用一段话读懂任何新发布的 framework。

**类型：** 学习
**语言：** Python (stdlib)
**先修：** Phase 14 (Agent Engineering), Phase 16 · 01 (Why Multi-Agent)
**时间：** ~60 分钟

## 要解决的问题

每六个月就会有一个新的 multi-agent framework 发布。2023 年的 AutoGen。2024 年的 CrewAI。2024 年的 LangGraph 和 OpenAI Swarm。2025 年 4 月的 Google ADK。2026 年 2 月的 Microsoft Agent Framework RC。每一篇发布稿都宣称自己是“正确的抽象”。

如果你试图一个一个学，很快就会耗尽。APIs 看起来不同。docs 对 “agent” 是什么也说法不一。一个 framework 把 shared memory 叫作 “blackboard”，另一个叫 “message pool”，第三个叫 “StateGraph”。你开始怀疑这个领域只是在原地搅动。

并不是。营销话术下面，四个原语是稳定的。学一次，就能用一段话读懂每个新 framework。

## 核心概念

### 四个原语

1. **Agent** —— 一个 system prompt 加一个 tool list。无状态；每次运行都从它的 system prompt 和当前 message history 开始。
2. **Handoff** —— 从一个 agent 到另一个 agent 的结构化控制权转移。机械上，它可以是一个返回新 agent 的 tool call，也可以是一个按条件跟随的 graph edge。
3. **Shared state** —— 任意可被多个 agent 读取（有时也可写入）的数据结构。Message pool、blackboard、key-value store、vector memory。
4. **Orchestrator** —— 决定谁下一个发言的人或机制。选项包括：显式 graph（确定性）、LLM speaker-selector（软性）、上一个 speaker 的 handoff call（OpenAI Swarm），或 queue 上的 scheduler（swarm architecture）。

这就是整个设计空间。每个 framework 都为每条轴选择默认值；其余只是表层语法。

### 2026 年每个 framework 如何映射

| Framework | Agent | Handoff | Shared state | Orchestrator |
|-----------|-------|---------|--------------|--------------|
| OpenAI Swarm / Agents SDK | `Agent(instructions, tools)` | tool returns Agent | caller's problem | the LLM's next handoff call |
| AutoGen v0.4 / AG2 | `ConversableAgent` | speaker-selector on GroupChat | message pool | selector function (LLM or round-robin) |
| CrewAI | `Agent(role, goal, backstory)` | `Process.Sequential / Hierarchical` | Task outputs chained | manager LLM or static order |
| LangGraph | node function | graph edge + condition | `StateGraph` reducer | the graph, deterministic |
| Microsoft Agent Framework | agent + orchestration patterns | pattern-specific | thread / context | pattern-specific |
| Google ADK | agent + A2A card | A2A task | A2A artifacts | host decides |

表层差异看起来巨大。底层：同样四个旋钮。

### 为什么这很重要

一旦你看见这些原语，framework 比较就会变成一个短 checklist：

- Orchestrator 是信任 LLM 做 routing（Swarm），还是把 routing 固定在 code 里（LangGraph）？
- Shared state 是 full-history（GroupChat），还是 projected（StateGraph reducer）？
- Agents 能修改彼此的 prompts（CrewAI manager），还是只能 hand off（Swarm）？

这三个问题能回答某个 framework 是否适合给定问题的 80%。你不再选购“最好的 multi-agent framework”，而是开始围绕你真正关心的轴做设计。

### 无状态洞察

除了 shared state，每个原语都是无状态的。Agent 是 (prompt, tools) 的函数。Handoff 是一次 function call。Orchestrator 是 scheduler。**系统里唯一有状态的东西是 shared state。** 所有有意思的 bug 都住在那里：memory poisoning（Lesson 15）、message ordering、versioning、write contention。

隐藏 shared state 的 frameworks（Swarm）会把问题推给 caller。集中 shared state 的 frameworks（LangGraph checkpoint、AutoGen pool）让它可检查，但会把协调成本转移到 shared-state implementation 上。

### 单个原语的剖面

#### Agent

```text
Agent = (system_prompt, tools, model, optional_name)
```

没有 memory。没有 state。两个具有相同 system prompt 和 tools 的 agents 可以互换。所有看起来像 per-agent state 的东西，其实都在 shared state 或 handoff protocol 里。

#### Handoff

```text
Handoff = (from_agent, to_agent, reason, payload)
```

三种实现最常见：

- **Function return** —— tool 返回 next agent。这是 OpenAI Swarm pattern。Agents 把 routing 放在自己的 tool schemas 里。
- **Graph edge** —— LangGraph。Edges 是 declarative 的。LLM 产生一个值；condition 选择下一个 node。
- **Speaker selection** —— AutoGen GroupChat。selector function（有时它自己也是一次 LLM call）读取 pool 并选择谁下一个发言。

#### Shared state

```text
SharedState = { messages: [], artifacts: {}, context: {} }
```

最少是一组 messages。经常会更多：structured artifacts（CrewAI Task outputs）、typed context（LangGraph reducers）、external memory（MCP、vector DB）。

两种拓扑：**full pool**（每个 agent 看见每条 message）和 **projected**（agents 看见 role-scoped view）。Full pools 简单但扩展性差。Projected pools 可扩展，但需要预先做 schema design。

#### Orchestrator

```text
Orchestrator = ({state, last_speaker}) -> next_agent
```

四种风格：

- **Static** —— graph 在 build time 固定（LangGraph deterministic、CrewAI Sequential）。
- **LLM-selected** —— 一个 LLM 读取 pool 并选择 next speaker（AutoGen、CrewAI Hierarchical）。
- **Handoff-driven** —— 当前 agent 通过调用 handoff tool 来决定（Swarm）。
- **Queue-driven** —— workers 从 shared queue 拉取任务；没有显式 next-speaker（swarm architectures、Matrix）。

### Frameworks 之间到底变了什么

原语固定后，剩下的设计决策是：

- **Memory strategy** —— ephemeral vs durable checkpointing（LangGraph checkpointer）。
- **Safety boundary** —— 谁可以 approve 一个 handoff（human-in-the-loop）。
- **Cost accounting** —— per-agent token budgets。
- **Observability** —— tracing handoffs、persisting state for replay。

这些都能在原语之上实现。它们都不是新的原语。

## 动手实现

`code/main.py` 用约 150 行 stdlib Python 实现四个原语。没有真实 LLM——每个 agent 都是 scripted policy，让注意力停留在协调结构上。

这个文件导出：

- `Agent` —— name、system prompt、tools、policy function 的 dataclass。
- `Handoff` —— 返回一个新 agent 的函数。
- `SharedState` —— thread-safe message pool。
- `Orchestrator` —— 三个变体：`StaticOrchestrator`、`HandoffOrchestrator`、`LLMSelectorOrchestrator`（simulated）。

demo 让同一个三 agent pipeline（research → write → review）通过全部三种 orchestrator types 运行，并在末尾打印 message pool。你可以看到 outputs 只在 *谁选择下一个* 上不同；跨运行的 agents 和 shared state 完全相同。

运行：

```text
python3 code/main.py
```

预期输出：三次 orchestrator runs，每个 pattern 一次。每次都会打印最终 message pool。如果 researcher 判断已经完成，handoff-driven run 会到达更少 agents——这就是 LLM-routing tradeoff 的微缩版。

## 实际使用

`outputs/skill-primitive-mapper.md` 是一个 skill，它读取任意 multi-agent codebase 或 framework doc，并返回四原语映射。把它跑在新 framework release 上，可以在深入阅读 docs 前先得到一段话的理解。

## 交付成果

采用新 framework 前，先为它写出 primitive mapping。如果写不出来，要么 docs 不完整，要么 framework 正在发明第五个原语（少见——先检查是不是你还没见过的 shared-state flavor）。

把 mapping 固定在你的 architecture doc 里。新队友加入时，先发 mapping，再发 API docs。Framework versions 变化时，diff mapping，而不是 diff changelog。

## 练习

1. 用不同 agent policies 运行 `code/main.py` 三次。观察 orchestrator choice 如何改变哪些 agents 会运行。
2. 实现第四种 orchestrator type：queue-driven，让 agents 轮询 shared state 获取工作。可能发生什么 deadlock，你如何检测它？
3. 取 LangGraph quickstart (https://docs.langchain.com/oss/python/langgraph/workflows-agents)，把它重写成四个原语。LangGraph 的哪些 abstractions 是 1:1 映射，哪些只是 convenience wrappers？
4. 阅读 OpenAI Swarm cookbook (https://developers.openai.com/cookbook/examples/orchestrating_agents)。识别 Swarm 让四个原语中的哪一个最符合人体工学，又把哪一个推给 caller。
5. 在这张表中找一个完全隐藏 shared state 的 framework。解释当 agents 需要在 handoffs 之间协调、但又不能重新读取 history 时，什么会坏掉。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Agent | "An LLM with tools" | 一个 `(system_prompt, tools, model)` triple。无状态。 |
| Handoff | "Transfer of control" | 一个结构化 call，命名 next agent 和可选 payload。三种实现：function return、graph edge、speaker selection。 |
| Shared state | "Memory" / "context" | Multi-agent system 中唯一有状态的部分。Message pool 或 blackboard。 |
| Orchestrator | "Coordinator" | 决定谁下一个运行的人或机制。Static graph、LLM selector、handoff-driven 或 queue-driven。 |
| Primitive | "Abstraction" | 每个 framework 参数化的四条轴之一。不是 framework feature。 |
| Message pool | "Shared chat history" | Full-history shared state。容易推理，但扩展性差。 |
| Projected state | "Scoped view" | Shared state 中 role-specific 的视图。可扩展，但需要 schema design。 |
| Speaker selection | "Who talks next" | 一种 orchestrator pattern，其中 function（通常是 LLM）从 group 中选择 next agent。 |

## 延伸阅读

- [OpenAI cookbook: Orchestrating Agents — Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) —— handoff-driven orchestration 最清晰的表述
- [AutoGen stable docs](https://microsoft.github.io/autogen/stable/) —— GroupChat + speaker selection 是 LLM-selected orchestration 的参考
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) —— graph-edge orchestration 和 reducer-based shared state
- [CrewAI introduction](https://docs.crewai.com/en/introduction) —— role-goal-backstory agents、Sequential / Hierarchical processes
- [AG2 (community AutoGen continuation)](https://github.com/ag2ai/ag2) —— Microsoft 将 v0.4 转入 maintenance 后，仍然活跃的 AutoGen v0.2 line
