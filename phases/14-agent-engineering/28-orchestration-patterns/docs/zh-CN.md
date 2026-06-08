# 编排模式：Supervisor、Swarm、Hierarchical

> 2026 年的 frameworks 中反复出现四种编排模式：supervisor-worker、swarm / peer-to-peer、hierarchical、debate。Anthropic 的指导是：“It's about building the right system for your needs.” 从简单开始；只有当单个 agent 加五种 workflow patterns 不够时，才添加拓扑。

**类型:** 学习 + 构建
**语言:** Python (stdlib)
**先修:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 25 (Multi-Agent Debate)
**时间:** ~60 分钟

## 学习目标

- 说出四种反复出现的编排模式，以及每种适合什么时候使用。
- 描述 2026 年 LangChain 的建议：tool-call-based supervision vs supervisor libraries。
- 解释 Anthropic 的“build the right system”规则，以及它如何约束 topology choice。
- 用 stdlib 和一个 common scripted LLM 实现全部四种模式。

## 要解决的问题

团队经常在真正需要之前就伸手去拿“multi-agent”。四种模式在 frameworks 中反复出现；只要你能命名它们，就能选对一种，或者完全跳过拓扑。

## 核心概念

### Supervisor-worker

- 一个中心 routing LLM 分派给 specialist agents。
- 决策：loop back to self、handoff to specialist、terminate。
- Specialists 彼此不直接对话；所有 routing 都经过 supervisor。

Frameworks：LangGraph `create_supervisor`、Anthropic orchestrator-workers、CrewAI Hierarchical Process。

**2026 LangChain 建议：** 通过直接 tool calls 做 supervision，而不是用 `create_supervisor`。这能提供更细的 context engineering control，你可以精确决定每个 specialist 看到什么。

### Swarm / peer-to-peer

- Agents 通过共享 tool surface 直接 hand off。
- 没有中心 router。
- 比 supervisor 延迟更低（更少 hops）。
- 更难推理（没有单一控制点）。

Frameworks：LangGraph swarm topology、OpenAI Agents SDK handoffs（当所有 agents 都能 hand off 给所有其他 agents 时）。

### Hierarchical

- Supervisors 管理 sub-supervisors，sub-supervisors 再管理 workers。
- 在 LangGraph 中实现为 nested subgraphs；在 CrewAI 中实现为 nested crews。
- 可以扩展到大型 agent populations，代价是运维复杂度上升。

需要它的时机：当单个 supervisor 的 context budget 无法容纳所有 specialists 的描述时。

### Debate

- Parallel proposers + iterative cross-critique（Lesson 25）。
- 严格说不是编排，更像验证；但它会作为 topology choice 出现在 frameworks 中。

### CrewAI Crew vs Flow

CrewAI 形式化了两种部署模式：

- **Flow** 用于确定性的 event-driven automation（生产推荐起点）。
- **Crew** 用于自主的 role-based collaboration。

这与上面的四种模式正交，但会映射到拓扑：Flow 通常是 supervisor 或 hierarchical；Crew 通常是带 LLM router 的 supervisor。

### Anthropic 的指导

“Success in the LLM space isn't about building the most sophisticated system. It's about building the right system for your needs.”

决策顺序：

1. 单个 agent + workflow patterns（Lesson 12）— 从这里开始。
2. Supervisor-worker — 当你有 2-4 个 specialists 时。
3. Swarm — 当 latency 比推理清晰度更重要时。
4. Hierarchical — 只有当 supervisor context budget 失败时。
5. Debate — 当 accuracy 比 cost 更重要时。

### 这种模式容易出错的地方

- **Topology-first thinking。** 在识别 multi-agent 解决什么问题之前就说“我们需要 multi-agent”。
- **Swarm 中的 bouncing handoffs。** A -> B -> A -> B。使用 hop counters。
- **虚假层级。** 因为“enterprise”建三层，但实际只有两个团队。折叠掉。

## 动手实现

`code/main.py` 用 stdlib 和一个 scripted LLM 实现全部四种模式：

- `Supervisor` — 中心 router。
- `Swarm` — peer-to-peer，直接 handoffs。
- `Hierarchical` — supervisors of supervisors。
- `Debate` — parallel proposers + critique。

每种模式都处理同一个三意图任务（refund / bug / sales）。Trace shapes 不同。

运行：

```text
python3 code/main.py
```

输出：每种 pattern 的 trace + op count。Supervisor 最清晰；swarm 最短；hierarchical 最深；debate 最昂贵。

## 实际使用

- **LangGraph** 用于 supervisor 和 hierarchical（nested subgraphs）。
- **OpenAI Agents SDK** 用于 handoffs-as-tools（supervisor-shaped）。
- **CrewAI Flow** 用于生产中的确定性流程。
- **Custom** 用于 debate，或当你需要精确控制时。

## 交付成果

`outputs/skill-orchestration-picker.md` 会选择一种 topology 并实现它。

## 练习

1. 通过移除 router，把 supervisor-worker 转成 swarm。什么坏了？什么变好了？
2. 给 swarm 添加 hop counter：3 次 handoffs 后拒绝。它能捕捉 A->B->A bouncing 吗？
3. 为一个 12-specialist 领域构建两层 hierarchical system。不使用 nesting 时，context budget 在哪里失败？
4. 在 production-shaped workload 上 profile 四种模式。哪一种在哪个指标上胜出（latency、cost、accuracy、debuggability）？
5. 阅读 Anthropic 的 “Building Effective Agents” 文章。把你的每个 production flow 映射到四种之一。有不能干净映射的吗？

## 关键术语

| 术语 | 人们通常怎么说 | 它实际意味着什么 |
|------|----------------|------------------|
| Supervisor-worker | “Router + specialists” | 中心 LLM 分派给 specialists；它们彼此不对话 |
| Swarm | “Peer-to-peer” | 通过共享工具直接 handoffs；没有中心 router |
| Hierarchical | “Supervisors of supervisors” | 面向大型 populations 的 nested subgraphs |
| Debate | “Proposer + critique” | Parallel proposers，cross-critique（Lesson 25） |
| Tool-call-based supervision | “不用库的 supervisor” | 把 supervisor 实现为直接 tool calls，以控制 context |
| Crew | “Autonomous team” | CrewAI 的 role-based collaboration 模式 |
| Flow | “Deterministic workflow” | CrewAI 的 event-driven production 模式 |

## 延伸阅读

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — five patterns + agent vs workflow
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — supervisor、swarm、hierarchical
- [CrewAI docs](https://docs.crewai.com/en/introduction) — Crew vs Flow
- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — debate pattern
