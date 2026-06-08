# Reflexion：语言形式的强化学习

> Gradient-based RL 需要数千次 trial 和 GPU cluster 才能修复一个 failure mode。Reflexion（Shinn et al., NeurIPS 2023）用自然语言完成这件事：每次 failed trial 后，agent 写一条 reflection，把它存入 episodic memory，并让下一次 trial 以这段 memory 为条件。这是 Letta 的 sleep-time compute、Claude Code 的 CLAUDE.md learnings，以及 pro-workflow 的 learn-rule 背后的 pattern。

**类型:** Build
**语言:** Python（stdlib）
**先修:** Phase 14 · 01（Agent Loop），Phase 14 · 02（ReWOO）
**时间:** ~60 分钟

## 学习目标

- 说出 Reflexion 的三个组件（Actor、Evaluator、Self-Reflector）以及 episodic memory 的角色。
- 实现一个 stdlib Reflexion loop，包含 binary evaluator、reflection buffer 和 fresh re-attempts。
- 针对给定任务，在 scalar、heuristic 和 self-evaluated feedback sources 之间选择。
- 解释为什么 verbal reinforcement 能捕捉 gradient-based RL 需要数千次 trial 才能修复的错误。

## 要解决的问题

Agent 任务失败了。标准 RL 会运行更多数千次 trials、计算 gradients、更新 weights。昂贵、缓慢，而且大多数 production agents 没有为每次失败准备 training budget。

Reflexion（Shinn et al., arXiv:2303.11366）问了另一个问题：如果 agent 只是思考自己为什么失败，然后把那段想法放进 prompt 再试一次呢？没有 weight updates。没有 gradient。只是 trials 之间保存自然语言。

结果是：在 ALFWorld 上，它击败 ReAct 和其他 non-fine-tuned baselines。在 HotpotQA 上，它优于 ReAct。在 code generation（HumanEval/MBPP）上，它达到当时 SOTA。全程没有一个 gradient step。

## 核心概念

### 三个 components

```text
Actor         : generates a trajectory (ReAct-style loop)
Evaluator     : scores the trajectory — binary, heuristic, or self-eval
Self-Reflector: writes a natural-language reflection on the failure
```

再加一个数据结构：

```text
Episodic memory: list of prior reflections, prepended to the next trial's prompt
```

一次 trial 运行 Actor。Evaluator 给它打分。如果分数低，Self-Reflector 产生一条 reflection（“I picked the wrong tool because I misread the question as asking about X when it was asking about Y”）。Reflection 进入 episodic memory。下一次 trial 重新开始，但会看到 reflection。

### 三种 evaluator types

1. **Scalar**：外部 binary signal。ALFWorld 成功或失败。HumanEval tests pass 或 fail。最简单，信号最高。
2. **Heuristic**：预定义 failure signatures。“如果 agent 连续两次产生相同 action，标记为 stuck。”“如果 trajectory 超过 50 步，标记为 inefficient。”
3. **Self-evaluated**：LLM 给自己的 trajectory 打分。无 ground truth 时需要。信号较弱；适合与 tool-grounded verification 搭配（Lesson 05 — CRITIC）。

2026 默认是混合：有 scalar 时用 scalar，没有时用 self-eval，heuristics 作为 safety rails。

### 为什么它能泛化

Reflexion 与其说是新算法，不如说是被命名的 pattern。几乎每个生产“self-healing”agent 都运行某个变体：

- Letta 的 sleep-time compute（Lesson 08）：单独 agent 反思过去 conversations，并写入 memory blocks。
- Claude Code 的 `CLAUDE.md` / “save memory” pattern：把 reflections 捕捉为 learnings，prepend 到 future sessions。
- pro-workflow 的 `/learn-rule` command：把 corrections 捕捉成显式 rules。
- LangGraph 的 reflection nodes：一个给 output 打分并在需要时 route 到 refine 的 node。

它们都来自同一个 insight：自然语言足够丰富，能够在 runs 之间携带“我从失败中学到了什么”。

### 何时有效、何时无效

Reflexion 有效，当：

- 有清晰 failure signal（test failure、tool error、wrong answer）。
- Task class 可复现（同类问题会再次出现）。
- Reflection 有空间改善 trajectory（足够 action budget）。

Reflexion 无效，当：

- Agent 第一次就成功。
- Failure 是外部的（network down、tool broken）。反思“the network was down”对未来 runs 无帮助。
- Reflection 变成迷信，也就是保存了关于一次 flaky run 的叙事。

2026 pitfall：memory rot。Reflections 会累积；有些过时或错误；随着 episodic buffer 增长，re-runs 变慢。缓解：periodic compaction（Lesson 06）、reflection TTL，或单独的 sleep-time cleanup agent（Letta）。

## 动手实现

`code/main.py` 在 toy puzzle 上实现 Reflexion：产生一个 3-element list，使其和为 target。Actor 发出 candidate lists；Evaluator 检查 sum；Self-Reflector 写一行哪里出错。Reflection 进入 episodic memory，供下一次 trial 使用。

组件：

- `Actor`：scripted policy，看到 reflections 后会改进。
- `Evaluator.binary()`：对 target sum 做 pass/fail。
- `SelfReflector`：生成一行 failure diagnosis。
- `EpisodicMemory`：带 TTL semantics 的 bounded list。

运行它：

```text
python3 code/main.py
```

Trace 显示三次 trials。Trial 1 失败，存储 reflection；trial 2 看到 reflection 后改进但仍失败；trial 3 成功。与 baseline run（无 reflection）对比，它会卡在 trial 1 的答案。

## 实际使用

LangGraph 把 reflection 作为 node pattern 提供。Claude Code 的 `/memory` command 和 pro-workflow 的 `/learn-rule` 把 episodic buffer 外部化为 markdown file。Letta 的 sleep-time compute 在 downtime 运行 Self-Reflector，使 primary agent 保持 latency-bound。OpenAI Agents SDK 不直接提供 Reflexion；你可以用按 score 拒绝 trajectories 的 custom Guardrail，以及能跨 runs 存活的 memory `Session` 构建它。

## 交付成果

`outputs/skill-reflexion-buffer.md` 创建并维护一个 episodic buffer，具备 reflection capture、TTL 和 deduplication。给定 task class 和 failure，它会发出真正帮助下一次 trial 的 reflection，而不是泛泛的“be more careful”。

## 练习

1. 从 binary 切换到返回 distance metric（离 target 多远）的 scalar evaluator。它收敛更快吗？
2. 给 reflections 添加 10 trials 的 TTL。超过这个点后，旧 reflections 是伤害还是帮助？
3. 实现 heuristic evaluator：如果同一个 action 重复，就把 trial 标记为 stuck。这如何与 Self-Reflector 交互？
4. 使用忽略 reflections 的 adversarial Actor 运行 Reflexion。迫使 Actor 注意它们的最小 reflection prompt engineering 是什么？
5. 阅读 Reflexion paper Section 4 关于 AlfWorld。概念上复现 130% success-rate improvement：相比 vanilla ReAct，关键 delta 是什么？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Reflexion | “Self-correction” | Shinn et al. 2023，Actor、Evaluator、Self-Reflector 加 episodic memory |
| Verbal reinforcement | “Learning without gradients” | 把 natural-language reflection prepend 到下一次 trial prompt |
| Episodic memory | “Per-task reflections” | 一个 task class 的 prior reflections bounded buffer |
| Scalar evaluator | “Binary success signal” | 来自 ground truth 的 pass/fail 或 numeric score |
| Heuristic evaluator | “Pattern-based detector” | 预定义 failure signatures（例如 stuck-loop、too-many-steps） |
| Self-evaluator | “LLM-as-judge on own trace” | 无 ground truth 时的低信号 fallback，需搭配 tool-grounded verification |
| Memory rot | “Stale reflections” | Episodic buffer 填满 obsolete entries；用 compaction/TTL 修复 |
| Sleep-time reflection | “Async self-reflection” | 在 hot path 外运行 Self-Reflector，让 primary agent 保持快速 |

## 延伸阅读

- [Shinn et al., Reflexion: Language Agents with Verbal Reinforcement Learning (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366) — canonical paper
- [Letta, Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute) — 生产中的 async reflection
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — 把 episodic buffer 作为 context 的一部分管理
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — reflection node pattern
