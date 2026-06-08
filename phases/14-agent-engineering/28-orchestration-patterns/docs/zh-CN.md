# 编排模式：监督者、群体与层级式编排

> 2026 年的框架中反复出现四种编排模式：supervisor-worker（监督者-执行者）、swarm / peer-to-peer（群体 / 点对点）、hierarchical（层级式）和 debate（辩论）。Anthropic 的指导是：“构建适合你需求的正确系统。” 从简单开始；只有当单个智能体加五种工作流模式不够时，才添加拓扑。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**先修：** 第 14 阶段 · 12（工作流模式），第 14 阶段 · 25（多智能体辩论）
**时间：** ~60 分钟

## 学习目标

- 说出四种反复出现的编排模式，以及每种适合什么时候使用。
- 描述 2026 年 LangChain 的建议：基于工具调用的监督，而不是依赖监督者库。
- 解释 Anthropic 的“构建正确系统”规则，以及它如何约束拓扑选择。
- 用标准库和一个通用的脚本化 LLM 实现全部四种模式。

## 要解决的问题

团队经常在真正需要之前就伸手去拿“多智能体”。四种模式在框架中反复出现；只要你能命名它们，就能选对一种，或者完全跳过拓扑。

## 核心概念

### Supervisor-worker（监督者-执行者）

- 一个中心路由 LLM 分派给专门智能体。
- 决策：回到自身、移交给专门智能体，或终止。
- 专门智能体彼此不直接对话；所有路由都经过监督者。

框架对应：LangGraph `create_supervisor`、Anthropic orchestrator-workers、CrewAI Hierarchical Process。

**2026 LangChain 建议：** 通过直接工具调用做监督，而不是使用 `create_supervisor`。这能提供更细的上下文工程控制，你可以精确决定每个专门智能体看到什么。

### Swarm / peer-to-peer（群体 / 点对点）

- 智能体通过共享工具面直接移交。
- 没有中心路由器。
- 比监督者模式延迟更低（跳数更少）。
- 更难推理（没有单一控制点）。

框架对应：LangGraph swarm topology、OpenAI Agents SDK handoffs（当所有智能体都能移交给所有其他智能体时）。

### Hierarchical（层级式）

- 监督者管理子监督者，子监督者再管理执行者。
- 在 LangGraph 中实现为嵌套子图；在 CrewAI 中实现为嵌套 crew。
- 可以扩展到大型智能体群体，代价是运维复杂度上升。

需要它的时机：当单个监督者的上下文预算无法容纳所有专门智能体的描述时。

### Debate（辩论）

- 并行提议者 + 迭代交叉批评（第 25 课）。
- 严格说它不是编排，更像验证；但它会作为拓扑选择出现在框架中。

### CrewAI Crew 与 Flow

CrewAI 形式化了两种部署模式：

- **Flow** 用于确定性的事件驱动自动化（生产推荐起点）。
- **Crew** 用于自主的基于角色的协作。

这与上面的四种模式正交，但会映射到拓扑：Flow 通常是监督者模式或层级式模式；Crew 通常是带 LLM 路由器的监督者模式。

### Anthropic 的指导

“LLM 领域的成功不在于构建最复杂的系统，而在于构建适合你需求的正确系统。”

决策顺序：

1. 单个智能体 + 工作流模式（第 12 课）：从这里开始。
2. Supervisor-worker：当你有 2-4 个专门智能体时。
3. Swarm：当延迟比推理清晰度更重要时。
4. Hierarchical：只有当监督者的上下文预算不够时。
5. Debate：当准确率比成本更重要时。

### 这种模式容易出错的地方

- **拓扑优先思维。** 在识别多智能体到底解决什么问题之前，就说“我们需要多智能体”。
- **Swarm 中的反复移交。** A -> B -> A -> B。使用跳数计数器。
- **虚假层级。** 因为“企业级”而建三层，但实际只有两个团队。把它折叠掉。

## 动手实现

`code/main.py` 用标准库和一个脚本化 LLM 实现全部四种模式：

- `Supervisor`：中心路由器。
- `Swarm`：点对点，直接移交。
- `Hierarchical`：监督者之上的监督者。
- `Debate`：并行提议者 + 批评。

每种模式都处理同一个三意图任务（退款 / 缺陷 / 销售）。轨迹形态不同。

运行：

```text
python3 code/main.py
```

输出：每种模式的轨迹和操作次数。Supervisor 最清晰；swarm 最短；hierarchical 最深；debate 最昂贵。

## 实际使用

- **LangGraph** 用于监督者模式和层级式模式（嵌套子图）。
- **OpenAI Agents SDK** 用于把移交当成工具（监督者形态）。
- **CrewAI Flow** 用于生产中的确定性流程。
- **自定义实现** 用于辩论，或用于你需要精确控制的时候。

## 交付成果

`outputs/skill-orchestration-picker.md` 会选择一种拓扑并实现它。

## 练习

1. 通过移除路由器，把 supervisor-worker 转成 swarm。什么坏了？什么变好了？
2. 给 swarm 添加跳数计数器：3 次移交后拒绝。它能捕捉 A->B->A 的反复移交吗？
3. 为一个包含 12 个专门智能体的领域构建两层层级式系统。不使用嵌套时，上下文预算在哪里失败？
4. 在接近生产的负载上剖析四种模式。哪一种在哪个指标上胜出（延迟、成本、准确率、可调试性）？
5. 阅读 Anthropic 的 “Building Effective Agents” 文章。把你的每个生产流程映射到四种之一。有不能干净映射的吗？

## 关键术语

| 术语 | 人们通常怎么说 | 它实际意味着什么 |
|------|----------------|------------------|
| Supervisor-worker | “路由器 + 专门智能体” | 中心 LLM 分派给专门智能体；它们彼此不对话 |
| Swarm | “点对点” | 通过共享工具直接移交；没有中心路由器 |
| Hierarchical | “监督者之上的监督者” | 面向大型智能体群体的嵌套子图 |
| Debate | “提议者 + 批评” | 并行提议者，交叉批评（第 25 课） |
| Tool-call-based supervision | “不用库的监督者” | 把监督者实现为直接工具调用，以控制上下文 |
| Crew | “自主团队” | CrewAI 的基于角色的协作模式 |
| Flow | “确定性工作流” | CrewAI 的事件驱动生产模式 |

## 延伸阅读

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 五种模式 + 智能体与工作流
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — supervisor、swarm、hierarchical
- [CrewAI docs](https://docs.crewai.com/en/introduction) — Crew vs Flow
- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — 辩论模式
