# Anthropic 的工作流模式：简单优先于复杂

> Schluntz 和 Zhang（Anthropic，2024 年 12 月）区分了工作流（预定义路径）和智能体（动态工具使用）。五种工作流模式覆盖了大多数场景。从直接 API 调用开始。只有当步骤无法预测时才加入智能体。

**类型:** 学习 + 构建
**语言:** Python (stdlib)
**先修:** Phase 14 · 01 (Agent Loop)
**时间:** ~60 分钟

## 学习目标

- 说出 Anthropic 的五种工作流模式：prompt chaining、routing、parallelization、orchestrator-workers、evaluator-optimizer。
- 解释 agent-vs-workflow 的区别，以及二者各自的工程成本。
- 识别何时选择 workflow 而不是 agent（以及反过来的情况）。
- 基于 scripted LLM，用 stdlib 实现全部五种模式。

## 要解决的问题

团队常常为本该只需要一个函数调用的问题引入 multi-agent frameworks。代价是真实存在的：框架会增加层次，让 prompts 变得不透明，隐藏控制流，并诱发过早复杂化。Schluntz 和 Zhang 在 2024 年 12 月的文章是行业内被引用最多的反向提醒：从简单开始，只有当复杂性能挣回成本时才增加它。

## 核心概念

### Workflows 与 agents

- **Workflow。** LLM 和工具通过预定义代码路径编排。工程师拥有这张图。
- **Agent。** LLM 动态指挥自己的工具并决定自己的步骤。模型拥有这张图。

二者都有位置。Workflows 更便宜、更快，也更容易调试。Agents 能打开开放式问题，但会让失败模式更难推理。

### Augmented LLM

五种模式的共同基础：一个 LLM 接入三种能力 -- search（retrieval）、tools（actions）、memory（persistence）。任何 API 调用都可以使用这些能力。

### 五种模式

1. **Prompt chaining。** 第 1 次调用的输出作为第 2 次调用的输入。适用于任务有清晰线性拆解的场景。步骤之间可以加入可选的程序化闸门。

2. **Routing。** 一个 classifier LLM 选择要调用的下游 LLM 或工具。适用于类别差异明显、需要不同处理方式的输入（tier-1 support、refund、bug、sales）。

3. **Parallelization。** 并发运行 N 个 LLM 调用，并聚合结果。两种形态：sectioning（不同 chunk）和 voting（同一个 prompt，运行 N 次，做多数投票或综合）。

4. **Orchestrator-workers。** 一个 orchestrator LLM 动态决定要运行哪些 workers（同样是 LLM），并综合它们的输出。类似 agent loops，但 orchestrator 不会无限循环。

5. **Evaluator-optimizer。** 一个 LLM 提出答案，另一个 LLM 评估答案。迭代直到 evaluator 通过。这是 Self-Refine（Lesson 05）的泛化。

### Workflows 胜过 agents 的地方

- **可预测任务。** 如果你能枚举步骤，就应该枚举。
- **成本受限任务。** Workflows 的步骤数有上界；agents 可能螺旋式增长。
- **合规受限任务。** 审计者想读图，而不是从 trajectories 里推断图。

### Agents 胜过 workflows 的地方

- **开放式研究。** 下一步取决于上一步返回了什么。
- **可变长度任务。** 从几分钟到几小时不等、步骤数未知的工作。
- **新领域。** 当你还不知道正确 workflow 时 -- 先探索，再固化。

### Context-engineering 伙伴概念

"Effective context engineering for AI agents"（Anthropic 2025）形式化了相邻学科：200k window 是预算，不是容器。该放入什么、何时 compact、何时让 context 增长。本课程在 Phase 14 的 context compression 课中详细覆盖（本课程重编号前是 Phase 14 earlier lesson 06）。

## 动手实现

`code/main.py` 基于 `ScriptedLLM` 实现全部五种工作流模式：

- `prompt_chain(input, steps)` -- 顺序执行。
- `route(input, classifier, handlers)` -- 分类 + 分发。
- `parallel_vote(prompt, n, aggregator)` -- N 次运行，聚合。
- `orchestrator_workers(task, workers)` -- orchestrator 选择 workers。
- `evaluator_optimizer(task, proposer, evaluator, max_iter)` -- 循环直到通过。

运行：

```text
python3 code/main.py
```

每种模式都会打印自己的 trace。每种模式大约 10-15 行代码；框架成本通常以数千行计。

## 实际使用

- 大多数任务使用直接 API 调用。
- 只有当模式确实需要 durable state（LangGraph）、actor-model concurrency（AutoGen v0.4）或 role templating（CrewAI）时才使用框架。
- 当你想要 Claude Code harness 的形状、但不想重建它时，选择 Claude Agent SDK。

## 交付成果

`outputs/skill-workflow-picker.md` 会为给定任务描述选择正确模式，包括决策理由，以及如果 workflows 不够用时重构为 agent 的路径。

## 练习

1. 用置信度阈值实现 routing。低于阈值 -> 升级给人类。对于 tier-1 support 用例，阈值应该落在哪里？
2. 给 `parallel_vote` 添加 timeout。当某个调用挂起时会发生什么？缺失 votes 时如何聚合？
3. 把 `evaluator_optimizer` 改成 bandit：跨迭代保留 top-2 outputs，这样后期出现的好结果不会被后期坏结果覆盖。
4. 组合 prompt chaining 和 routing：router 选择三条 chains 之一。衡量 token cost 与单个 big-prompt 替代方案的差异。
5. 选择你的一个生产功能。画出 workflow graph。数步骤。这里 agent 真的会更好吗？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Workflow | "Predefined flow" | 工程师拥有的 LLM 与工具调用图 |
| Agent | "Autonomous AI" | 模型拥有的图；动态指挥工具 |
| Augmented LLM | "LLM with tools" | LLM + search + tools + memory；原子单元 |
| Prompt chaining | "Sequential calls" | 调用 N 的输出是调用 N+1 的输入 |
| Routing | "Classifier dispatch" | 选择哪个 chain/model 处理输入 |
| Parallelization | "Fan out" | N 个并发调用；通过 sectioning 或 voting 聚合 |
| Orchestrator-workers | "Dispatcher agent" | Orchestrator LLM 动态选择 specialist LLMs |
| Evaluator-optimizer | "Proposer + judge" | 迭代直到 evaluator 通过；Self-Refine 的泛化 |

## 延伸阅读

- [Anthropic, Building Effective Agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) -- 五种工作流模式
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) -- 伙伴学科
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) -- 何时 stateful graphs 值得其成本
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) -- 产品化后的 orchestrator-workers 模式
