# ReWOO 与 Plan-and-Execute：解耦规划

> ReAct 在同一个 stream 中交错 thought 和 action。ReWOO 把它们分开：先做一个大 plan，再执行。Token 少 5 倍，HotpotQA 准确率 +4%，并且你可以把 planner distill 到 7B 模型。Plan-and-Execute 泛化了它；Plan-and-Act 把它扩展到 web navigation。

**类型:** Build
**语言:** Python（stdlib）
**先修:** Phase 14 · 01（Agent Loop）
**时间:** ~60 分钟

## 学习目标

- 解释为什么 ReWOO 的 Planner / Worker / Solver 拆分比 ReAct 的 interleaved loop 更省 token 且更 robust。
- 实现一个 plan DAG、一个 dependency-ordered executor，以及一个组合 worker outputs 的 solver，全用 stdlib。
- 使用 2026 “five workflow patterns” framing（Anthropic）判断任务应采用 plan-then-execute 还是 interleaved ReAct。
- 识别长程 web 或 mobile tasks 什么时候需要 Plan-and-Act 的 synthetic plan data。

## 要解决的问题

ReAct 的 interleaved thought-action-observation loop 简单灵活，但每个 tool call 都必须携带完整 prior context，包括每个 previous thought。Token usage 随 depth 二次增长。更糟的是，当 tool 在 mid-loop 失败，模型必须从 error observation 中重新推导整个 plan。

ReWOO（Xu et al., arXiv:2305.18323, 2023 年 5 月）注意到这一点并做出押注：先规划完整任务，并行获取 evidence，最后组合答案。一个 LLM call 用于 plan，N 个 tool calls 用于 evidence（可 parallel），一个 LLM call 用于 solve。取舍是更少灵活性（plan 是 static），换取更好的 token efficiency 和更清晰 failure modes。

## 核心概念

### 三个 roles

```text
Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        (tool calls, possibly parallel)
Solver:   user_question, plan_dag, evidence -> final_answer
```

Planner 产生 DAG。每个 node 命名 tool、arguments，以及它依赖哪些 earlier nodes（类似 `#E1`、`#E2` 的 references）。Workers 按 topological order 执行 nodes。Solver 把一切拼起来。

### 为什么 token 少 5 倍

ReAct 的 prompt length 随 step count 线性增长。在第 10 步，prompt 包含 thought 1 加 action 1 加 observation 1，加 thought 2、action 2、observation 2，以此类推。每个 intermediate step 还会冗余包含 original prompt。

ReWOO 支付一次 planner prompt（大）、N 个小 worker prompts（每个只是 tool call，没有 chain）、一次 solver prompt。HotpotQA 上论文测得约 5 倍更少 tokens，同时 absolute accuracy +4。

### 为什么更 robust

如果 ReAct 中 worker 3 失败，loop 必须在 mid-stream 中对 error 推理。在 ReWOO 中，worker 3 返回 error string；solver 在 original plan 的 context 中看到它，并可以 graceful degrade。Failure localization 是 per-node，不是 per-step。

### Planner distillation

论文第二个结果：因为 planner 看不到 observations，你可以用 175B teacher 的 planner outputs fine-tune 一个 7B model。小模型负责 planning；推理时不需要大模型。这现在是标准做法，许多 2026 production agents 使用小 planner 大 executor，或反过来。

### Plan-and-Execute（LangChain，2023）

LangChain 团队 2023 年 8 月的文章把 ReWOO 泛化成一个 pattern name：Plan-and-Execute。Up-front planner 发出 step list，executor 执行每一步，可选 replanner 可以在观察结果后修订。这比 ReWOO 更接近 ReAct（replanner 把 observations 带回 planning），但保留 token savings。

### Plan-and-Act（Erdogan et al., arXiv:2503.09572, ICML 2025）

Plan-and-Act 把该 pattern 扩展到 long-horizon web 和 mobile agents。关键贡献是 synthetic plan data：一个 labeled trajectory generator 产生 plan 显式存在的训练数据。用于 fine-tune planner models，让它们在 WebArena-like tasks 上超过 30-50 步后仍保持工作，而单个 ReAct trajectory 会失去 coherence。

### 何时选择哪种

| Pattern | When |
|---------|------|
| ReAct | 短任务、未知环境、需要 reactive exception handling |
| ReWOO | 工具已知的结构化任务、token-sensitive、parallelizable evidence |
| Plan-and-Execute | 类似 ReWOO，但 partial execution 后需要 replanning |
| Plan-and-Act | Long-horizon（>30 steps）、web/mobile/computer-use |
| Tree of Thoughts | Search 值得付费时（Lesson 04） |

Anthropic 2024 年 12 月 guidance：从最简单的开始。如果任务是一个 tool call 加 summary，不要构建 ReWOO。如果任务是 40 步 research assignment，不要只做 ReAct。

## 动手实现

`code/main.py` 实现一个 toy ReWOO：

- `Planner`：scripted policy，从 prompt 发出 plan DAG。
- `Worker`：通过 registry dispatch 每个 node 的 tool call。
- `Solver`：scripted composition，读取 evidence 并产出 final answer。
- Dependency resolution：把类似 `#E1` 的 references 替换为 earlier worker outputs。

Demo 回答“What is the population of the capital of France, rounded to millions?”，使用两步 plan：（1）查 capital，（2）查 population，然后 solve。

运行它：

```text
python3 code/main.py
```

Trace 先显示完整 plan，再显示 worker results，最后显示 solver composition。把 token count（我们打印 rough character count）与 ReAct-style interleaved run 对比，这类 structured task 上 ReWOO 胜出。

## 实际使用

LangGraph 将 Plan-and-Execute 作为 recipe 提供（ReAct 用 `create_react_agent`，plan-execute 用 custom graphs）。CrewAI 的 Flows 直接编码该 pattern：你预先定义 tasks，Flow DAG 执行它们。Plan-and-Act 的 synthetic data approach 仍主要是 research；runtime pattern（explicit plan DAG）通过 LangGraph 和 CrewAI Flows 进入生产。

## 交付成果

`outputs/skill-rewoo-planner.md` 在给定 tool catalog 的情况下，从 user request 生成 ReWOO plan DAG。它在交给 executor 前验证 plan（acyclic、每个 reference resolved、每个 tool exists）。

## 练习

1. 对 independent plan nodes 并行化 worker execution。一个 6-node DAG 有 2 个 parallel groups 时，它带来什么？
2. 添加 replanner node，在任一 worker 返回 error 时触发。把 ReWOO 变成 Plan-and-Execute 的最小改动是什么？
3. 用小模型（7B class）替换 `Planner`，让 `Solver` 仍用 frontier model。比较端到端 quality，split 在哪里失败？
4. 阅读 ReWOO paper Section 4 关于 planner distillation。概念上复现 175B -> 7B 结果：你需要什么训练数据，如何给 plan quality 打分？
5. 把 toy 移植到 Plan-and-Act 的 trajectory shape：plan 是 sequence，不是 DAG。哪些 tradeoffs 改变了？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| ReWOO | “Reasoning without observations” | 先 plan，再并行 fetch evidence，再 solve，planning prompt 中没有 observations |
| Plan-and-Execute | “LangChain's plan-execute pattern” | ReWOO 加一个 execution 后可选 replanner node |
| Plan-and-Act | “Scaled plan-execute” | 为 long-horizon tasks 使用 synthetic plan training data 的显式 planner/executor split |
| Evidence reference | “#E1, #E2, ...” | Dispatch 时由 prior worker output 替换的 plan-node placeholder |
| Planner distillation | “Small planner, big executor” | 用 large teacher 的 planner traces fine-tune small model |
| Token efficiency | “Fewer round trips” | 论文中 HotpotQA 相比 ReAct token 少 5 倍 |
| DAG executor | “Topological dispatcher” | 按 dependency order 运行 plan nodes；每层可 parallel |

## 延伸阅读

- [Xu et al., ReWOO: Decoupling Reasoning from Observations (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323) — canonical paper
- [Erdogan et al., Plan-and-Act (arXiv:2503.09572)](https://arxiv.org/abs/2503.09572) — 带 synthetic plans 的 scaled planner-executor
- [LangGraph Plan-and-Execute tutorial](https://docs.langchain.com/oss/python/langgraph/overview) — framework recipe
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 选择能工作的最简单 pattern
