# 使用 HTN 和 Evolutionary Search 做规划

> Symbolic planning 处理那些计划可以被证明正确的场景。Evolutionary code search 处理那些 fitness function 可机器检查的场景。ChatHTN（2025）和 AlphaEvolve（2025）展示了二者与 LLM 配对后各自解锁了什么。

**类型：** 构建
**语言：** Python (stdlib)
**先修：** Phase 14 · 02 (ReWOO and Plan-and-Execute)
**时间：** ~75 分钟

## 学习目标

- 解释 Hierarchical Task Networks：tasks、methods、operators、preconditions、effects。
- 描述 ChatHTN 的 hybrid loop：symbolic search 加 LLM fallback decomposition。
- 解释 AlphaEvolve 的 evolutionary loop，以及为什么它只在有 programmatic evaluator 时有效。
- 用 stdlib 实现一个 toy HTN planner 和一个 toy evolutionary search。

## 要解决的问题

ReWOO（Lesson 02）、Plan-and-Execute 和 ReAct 覆盖了多数 agent planning。但有两类场景它们覆盖得不好：

1. **Plans with provable correctness.** Scheduling、flight pathing、compliance workflows — 计划必须 by construction sound。一个流畅但偶尔 hallucinate step 的 LLM plan 无法接受。
2. **Optimizations with a machine-checkable fitness function.** Matrix multiplication、scheduling heuristics、compiler passes — 目标不是“一个正确计划”，而是“最佳计划”。

HTN planning 和 AlphaEvolve 解决两个不同问题。二者都把 LLM 当作放大器，而不是替代品。

## 核心概念

### Hierarchical Task Networks

HTN 包含：

- **Tasks** — compound（待分解）和 primitive（可直接执行）。
- **Methods** — 将 compound task 分解成 subtasks 的方式，带 preconditions。
- **Operators** — 带 preconditions 和 effects 的 primitive actions。
- **State** — 一组 facts。

Planning：给定一个 goal task 和 initial state，寻找一串 primitive operators 的 decomposition，并且它们的 preconditions 会按顺序满足。

HTN 比 LLM 更古老，并且仍是 provably-correct plans 的 reference。

### ChatHTN（Gopalakrishnan et al., 2025）

ChatHTN（arXiv:2505.11814）交错执行 symbolic HTN 和 LLM queries：

1. 尝试用已有 methods 分解当前 compound task。
2. 如果没有 method 适用，就问 LLM：“how would you decompose `task` in state `s`?”
3. 将 LLM response 翻译成 candidate subtasks。
4. 根据 operator schema 验证；拒绝 invalid decompositions。
5. Recurse。

论文的中心主张：每个产出的 plan 都是 provably sound，因为 LLM suggestions 只作为 candidate decompositions 进入，绝不直接编辑 plan。Symbolic layer 拥有 correctness；LLM 扩展 method library。

Online method learning（OpenReview `gwYEDY9j2x`, 2025 follow-up）增加了一个 learner，通过 regression 泛化 LLM-produced decompositions，将 LLM query frequency 最多降低 75%。

### AlphaEvolve（Novikov et al., 2025）

AlphaEvolve（arXiv:2506.13131, DeepMind, June 2025）是另一种东西：由 Gemini 2.0 Flash/Pro ensemble 编排的 evolutionary code search。

Loop：

1. 从 seed program + programmatic evaluator（返回 fitness score）开始。
2. LLM ensemble 提出 mutations。
3. 用 evaluator 运行 mutations。
4. 保留最好的；继续 mutate。

已发表成果：

- 56 年来首次改进 4x4 complex matrix multiplication 的 Strassen 结果（48 scalar multiplications）。
- 通过 Borg scheduling heuristic 恢复 Google compute 0.7%。
- 在 frontier workload 上实现 32% FlashAttention speedup。

硬约束：fitness function 必须 machine-checkable。对 prose answers 做 evolutionary search 不会收敛。

### 何时使用哪个

| Problem class | Use | Why |
|---------------|-----|-----|
| Scheduling with hard constraints | HTN + ChatHTN | Provable soundness |
| Compiler optimization | AlphaEvolve | Machine-checkable fitness |
| Multi-step task execution | ReAct / ReWOO | LLM in the loop, no formal guarantees |
| Code improvement with tests | AlphaEvolve | Tests are the evaluator |
| Policy-bound automation | HTN | Preconditions encode policy |

### 这个模式容易出错的地方

- **HTN without operators.** 没有 precondition/effect schemas，soundness claim 就会崩塌。ChatHTN 的“LLM suggests decomposition”要求 schema 能拒绝 invalid moves。
- **AlphaEvolve without a real evaluator.** “Ask the LLM if the code is better”不是 fitness function。Evaluator 必须 deterministic 且快速。
- **Over-engineering.** 多数 agent tasks 并不需要二者。先考虑 ReAct 或 ReWOO。

## 动手实现

`code/main.py` 实现两个 toys：

- 一个 stdlib HTN planner，带 operators、methods、preconditions、effects，以及在没有 method 匹配 compound task 时触发的 `LLMFallback`。“LLM”是 scripted decomposer，所以 planner 可离线运行。
- 一个 stdlib evolutionary search，搜索 arithmetic programs：生成 expressions，让它们在 test set 上最小化 `|f(x) - target|`。Evaluator 是 deterministic 的。

运行：

```text
python3 code/main.py
```

trace 展示 HTN planner 分解 compound task（中途有一次 LLM fallback），以及 evolutionary loop 收敛到 target expression。

## 实际使用

- **HTN planners** — `pyhop`、`SHOP3`，或为 domain-specific policy enforcement 自建。
- **ChatHTN** — research code；pattern（symbolic + LLM fallback）可干净地移植到任意 HTN planner。
- **AlphaEvolve** — DeepMind paper；pattern（ensemble + evaluator）可复现。OpenEvolve 和类似 open-source forks 正在出现。
- **Agent frameworks** — 还没有 first-class HTN 或 AlphaEvolve。把它构建成 subagent 或 background worker。

## 交付成果

`outputs/skill-hybrid-planner.md` 会生成一个 hybrid planner scaffold（HTN 或 evolutionary），并明确限定 LLM role。

## 练习

1. 用 backtracking 扩展 HTN planner：当 operator 的 postcondition 在 runtime 失败时，roll back 并尝试下一个 method。
2. 给 ChatHTN 添加 LLM-method cache：当 LLM 在 state pattern `P` 中分解 task `T` 时，存储结果。下一次调用先重新检查 method library。
3. 把 evolutionary search evaluator 换成真实 test suite。Evolve 一个通过 20 个 test cases 的 sort function；报告收敛所需 generations。
4. 阅读 AlphaEvolve 的 evaluator design notes。为你关心的 domain 设计一个 evaluator（SQL query optimization、test-suite minimization、deployment YAML）。
5. 组合二者：用 HTN 把 compound task 分解成 subtasks，然后在每个 subtask 的 primitive operator 上使用 evolutionary search。它在哪里闪光，在哪里 over-engineer？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| HTN | “Hierarchical planner” | 带 operators、preconditions、effects 的 task decomposition |
| Method | “Decomposition rule” | 将 compound task 拆成 subtasks 的方式 |
| Operator | “Primitive action” | 带 precondition 和 effect 的具体 step |
| ChatHTN | “LLM + HTN” | 没有 method 匹配时，symbolic planner 询问 LLM |
| AlphaEvolve | “Evolutionary code search” | Ensemble LLMs mutate code；deterministic evaluator 选择 |
| Fitness function | “Evaluator” | 对 outputs 的 deterministic、machine-checkable score |
| Online method learning | “Cached LLM decomposition” | 存储并泛化 LLM plans，以降低 query cost |

## 延伸阅读

- [Gopalakrishnan et al., ChatHTN (arXiv:2505.11814)](https://arxiv.org/abs/2505.11814) — symbolic + LLM hybrid planner
- [Novikov et al., AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) — evolutionary code search with LLM mutations
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — when to reach for a planner vs a simple loop
