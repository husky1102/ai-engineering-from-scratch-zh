# Eval-Driven Agent Development

> Anthropic 的指导是：“start with simple prompts, optimize them with comprehensive evaluation, and add multi-step agentic systems only when needed.” Evaluation 不是最后一步。它是驱动 Phase 14 其他所有选择的外层循环。

**类型:** 学习 + 构建
**语言:** Python (stdlib)
**先修:** All of Phase 14.
**时间:** ~60 分钟

## 学习目标

- 说出三层 evaluation：static benchmarks、custom offline、online production，以及每层的用途。
- 解释 evaluator-optimizer tight loop。
- 描述 2026 年最佳实践：evals 与代码放在一起、在 CI 中运行、gate PRs。
- 把 Phase 14 的每一课连接到它生成的 eval case。

## 要解决的问题

Agents 能通过 demos。它们会在生产中以 demos 无法预测的方式失败。Benchmarks 回答的是“这个模型大体上有能力吗？”，而不是“这个 agent 是否在为我的产品交付正确 patches？”答案是三层 evaluation，持续运行，并把每条 guardrail 和 learned rule 都映射到一个 eval case。

## 核心概念

### 三层 evaluation

1. **Static benchmarks** — 面向代码的 SWE-bench Verified（Lesson 19），面向浏览/桌面的 WebArena/OSWorld（Lesson 20），通用能力的 GAIA（Lesson 19），工具使用的 BFCL V4（Lesson 06）。用于 cross-model comparison 和 regression gating。污染是真实存在的：SWE-bench+ 发现 32.67% 的 solution leakage。始终报告 Verified / +-audited scores。

2. **Custom offline evals** — 你的产品形状：
   - LLM-as-judge（Langfuse、Phoenix、Opik — Lesson 24）。
   - Execution-based（运行 patch，检查 tests）。
   - Trajectory-based（将 action sequences 与 gold 对比；OSWorld-Human 显示 top agents 是 gold 的 1.4-2.7x）。

3. **Online evals** — 生产：
   - Session replays（Langfuse）。
   - Guardrail-triggered alerts（Lesson 16、21）。
   - Per-step cost / latency tracking（Lesson 23 OTel spans）。

### Evaluator-optimizer（Anthropic）

紧密循环：

1. Proposer 生成输出。
2. Evaluator 评判。
3. Refine，直到 evaluator 通过。

这是 Self-Refine（Lesson 05）的泛化。任何你在意的 agent flow 都可以包进 evaluator-optimizer，以提高可靠性。

### 2026 最佳实践

- Evals 与代码放在一起。
- 每个 PR 都在 CI 中运行。
- 用 eval scores gate merge（例如 “no regression > 5% vs main”）。
- 每条 guardrail 都映射到一个 eval case。
- 每条 learned rule（Reflexion、pro-workflow learn-rule）都映射到一个 failure case。

### 把 Phase 14 串起来

Phase 14 的每一课都会生成 eval cases：

| Lesson | 它生成的 eval case |
|--------|--------------------|
| 01 Agent Loop | Budget-exhausted、infinite-loop guard |
| 02 ReWOO | 工具失败时 planner 正确 replan |
| 03 Reflexion | Learned reflections 在 retry 时生效 |
| 05 Self-Refine/CRITIC | Judge 通过 refined output |
| 06 Tool Use | Argument coercion 生效；unknown tools 被拒绝 |
| 07-10 Memory | Retrieval citations 与 sources 匹配；stale facts 失效 |
| 12 Workflow Patterns | 每种 pattern 产出正确 output |
| 13 LangGraph | Resume 精确复现 state |
| 14 AutoGen Actors | DLQ 捕捉 crashed handlers |
| 16 OpenAI Agents SDK | Guardrail 在正确 inputs 上触发 |
| 17 Claude Agent SDK | Subagent results 返回 orchestrator |
| 19-20 Benchmarks | SWE-bench Verified score、WebArena success rate、OSWorld efficiency |
| 21 Computer Use | Per-step safety 捕捉 injected DOM |
| 23 OTel | Spans emit required attributes |
| 26 Failure Modes | Detectors 标注已知 failures |
| 27 Prompt Injection | PVE 拒绝 poisoned retrievals |
| 28 Orchestration | Supervisor route 到正确 specialist |
| 29 Runtime Shapes | DLQ 处理 N% failure |

如果你的 eval suite 覆盖了每一项，就覆盖了 Phase 14。

### Eval-driven development 容易失败的地方

- **没有 baseline。** 没有 last-known-good 的 evals 无法解读。存储 baselines。
- **没有 grounding 的 LLM-judge。** Judges 也会幻觉。CRITIC pattern（Lesson 05）— judge 基于外部工具 grounding。
- **过拟合 evals。** 为 eval 优化会偏离生产有用性。轮换 cases。
- **Flaky evals。** 非确定性 cases 会造成 false alarms。固定 seeds，snapshot state。

## 动手实现

`code/main.py` 是一个 stdlib eval harness：

- 带 categories（benchmark、custom、online）的 case registry。
- 一个 scripted agent under test。
- Evaluator-optimizer loop：propose、judge、refine，直到 pass 或达到 max rounds。
- CI gate：聚合 pass rate + 与 baseline 对比 regression。

运行：

```text
python3 code/main.py
```

输出：每个 case 的 pass/fail、regression flag、CI gate verdict。

## 实际使用

- 在与 agent code 相同的 repo 中编写 eval cases。
- 在每个 PR 上通过 CI 运行它们。
- 在 regression 时让 build 失败。
- 随时间跟踪 pass rate。
- 把每个生产失败都绑定到一个新 case。

## 交付成果

`outputs/skill-eval-suite.md` 会为 agent 产品构建一个三层 eval suite，包含 CI gates 和 regression tracking。

## 练习

1. 取一个你的生产失败。写一个能复现它的 eval case。你的 agent 现在能通过吗？
2. 为你的领域构建一个三维度（factual、tone、scope）的 LLM-judge rubric。给 50 个 sessions 打分。
3. 将 eval suite 接入 CI。在 >=5% regression 时让 build 失败。
4. 添加 trajectory-efficiency metric：agent 走了多少步，与 gold trajectory 相比如何？
5. 将 Phase 14 的每一课映射到你 suite 中的 eval case。有缺失吗？那就是要补的 gap。

## 关键术语

| 术语 | 人们通常怎么说 | 它实际意味着什么 |
|------|----------------|------------------|
| Static benchmark | “Off-the-shelf eval” | SWE-bench、GAIA、AgentBench、WebArena、OSWorld |
| Custom offline eval | “Domain eval” | 面向你产品形状的 LLM-as-judge / exec / trajectory |
| Online eval | “Production eval” | Session replay、guardrail alerts、cost/latency tracking |
| Evaluator-optimizer | “Propose-judge-refine” | 迭代直到 judge 通过 |
| CI gate | “Merge blocker” | 在 eval regression 时让 build 失败 |
| Baseline | “Last-known-good” | 用于检测 regression 的参考分数 |
| Trajectory efficiency | “Steps over gold” | Agent step count 除以 human expert minimum |

## 延伸阅读

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — “start simple, optimize with evals”
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — curated benchmark
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) — tool-use benchmark
- [Langfuse docs](https://langfuse.com/) — 实践中的 evals + session replay
