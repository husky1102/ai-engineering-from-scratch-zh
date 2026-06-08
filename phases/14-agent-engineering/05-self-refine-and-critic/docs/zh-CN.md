# Self-Refine 和 CRITIC：迭代式输出改进

> Self-Refine（Madaan et al., 2023）让一个 LLM 在循环中扮演三个角色：generate、feedback、refine。平均收益：在 7 个任务上绝对提升 +20。CRITIC（Gou et al., 2023）通过把验证路由到外部工具，让 feedback 步骤更可靠。到 2026 年，这个模式已经以“evaluator-optimizer”（Anthropic）或 guardrail loop（OpenAI Agents SDK）的形式进入每个框架。

**类型：** 构建
**语言：** Python (stdlib)
**先修：** Phase 14 · 01 (Agent Loop), Phase 14 · 03 (Reflexion)
**时间：** ~60 分钟

## 学习目标

- 说出 Self-Refine 的三个 prompt（generate、feedback、refine），并解释为什么 history 对 refine prompt 很重要。
- 解释 CRITIC 的关键洞察：没有外部 grounding 时，LLM 不擅长自我验证。
- 用 stdlib 实现带 history 和可选外部 verifier 的 Self-Refine loop。
- 将这个模式映射到 Anthropic 的“evaluator-optimizer”workflow，以及 OpenAI Agents SDK 的 output guardrails。

## 要解决的问题

agent 产出了一个几乎正确的答案。也许一行代码有语法错误。也许摘要太长。也许计划漏了一个边界情况。你真正想要的是：agent 批判自己的输出，然后修正它。

Self-Refine 表明，用单个模型、不用训练数据、不用 RL 也能做到这一点。但有一个问题：LLM 在硬事实上的自我验证很差。CRITIC 给出了修复方式：把 verify 步骤路由到外部工具（search、code interpreter、calculator、test runner）。

这两篇论文合在一起，定义了 2026 年迭代式改进的默认形态：generate，verify（能外部验证时就外部验证），refine，在 verifier 通过时停止。

## 核心概念

### Self-Refine（Madaan et al., NeurIPS 2023）

一个 LLM，三个角色：

```text
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
stop when feedback says "no issues" or budget exhausted.
```

关键细节：`refine` 会看到完整 history，也就是所有先前的 outputs 和 critiques，所以它不会重复犯错。论文做了 ablation：去掉 history，质量会急剧下降。

头条结果：在 7 个任务（math、code、acronym、dialog）上平均绝对提升 +20，包括 GPT-4。无需训练、无需外部工具，单模型完成。

### CRITIC（Gou et al., arXiv:2305.11738, v4 Feb 2024）

Self-Refine 的弱点：feedback 步骤是 LLM 给自己打分。对事实性声明来说这并不可靠（幻觉在生成它的模型眼中往往也显得很有说服力）。CRITIC 用 `verify(task, output, tools)` 替换 `feedback(task, output)`，其中 `tools` 包括：

- 用于事实声明的 search engine。
- 用于代码正确性的 code interpreter。
- 用于算术的 calculator。
- 领域特定 verifier（unit tests、type checkers、linters）。

verifier 会产出一个由工具结果 grounding 的结构化 critique。refiner 随后以这个 critique 为条件进行改写。

头条结果：CRITIC 在事实任务上优于 Self-Refine，因为 critique 有 grounding。在没有外部 verifier 的任务（creative writing、formatting）上，CRITIC 退化为 Self-Refine。

### 停止条件

两种常见形态：

1. **Verifier passes.** 外部测试返回成功。可用时首选（unit tests、type checker、guardrail assertion）。
2. **No feedback issued.** 模型说“the output is fine”。更便宜但不可靠；要和 max-iteration cap 配对。

2026 年默认做法：组合使用它们。“Stop if verifier passes OR model says fine AND iterations >= 2 OR iterations >= max_iterations.”

### Evaluator-Optimizer（Anthropic, 2024）

Anthropic 2024 年 12 月的文章把它命名为五种 workflow pattern 之一。两个角色：

- Evaluator：给输出打分并产出 critique。
- Optimizer：根据 critique 修订输出。

循环直到 evaluator 通过。这就是 Anthropic 语境中的 Self-Refine/CRITIC。Anthropic 补充的关键工程细节：evaluator 和 optimizer 的 prompt 应该有明显不同，避免模型只是 rubber-stamp。

### OpenAI Agents SDK output guardrails

OpenAI Agents SDK 以“output guardrails”的形式提供这个模式。guardrail 是一个 validator，会在 agent 的最终输出上运行。如果 guardrail trip（抛出 `OutputGuardrailTripwireTriggered`），输出会被拒绝，agent 可以重试。guardrails 可以调用工具（CRITIC-style），也可以是纯函数（Self-Refine-style）。

### 2026 年的坑

- **Rubber-stamp loops.** 同一个模型用同一种 prompt 风格同时做 generation 和 critique，会收敛到“looks good to me”。使用结构不同的 prompts，或用一个更小、更便宜的模型做 critique。
- **Over-refinement.** 每次 refine pass 都增加延迟和 tokens。预算 1-3 次；超过后升级到 human review。
- **CRITIC on trivial tasks.** 如果没有外部 verifier，CRITIC 会退化为 Self-Refine；不要为 stub verifier 支付延迟。

## 动手实现

`code/main.py` 在一个 toy task 上实现 Self-Refine 和 CRITIC：给定 topic，产出一个短 bullet list。verifier 检查格式（3 个 bullets，每个少于 60 个字符）。CRITIC 增加一个外部“fact verifier”，惩罚已知幻觉。

组件：

- `generate` — scripted producer。
- `feedback` — LLM-style self-critique。
- `verify_external` — CRITIC-style grounded verifier。
- `refine` — 根据 history 改写 output。
- Stop condition — verifier passes 或最多 4 次 iterations。

运行：

```text
python3 code/main.py
```

比较 Self-Refine 和 CRITIC 的运行。CRITIC 会抓到一个 Self-Refine 漏掉的事实错误，因为外部 verifier 拥有 self-critic 没有的 grounding。

## 实际使用

Anthropic 的 evaluator-optimizer 就是这个模式的 Claude-friendly 表述。OpenAI Agents SDK 的 output guardrails 是 CRITIC-shaped（guardrails 可以调用工具）。LangGraph 提供的 reflection node 读起来像 Self-Refine。Google 的 Gemini 2.5 Computer Use 增加了 per-step safety evaluator，这是 CRITIC 变体：每个 action 在 commit 前都要验证。

## 交付成果

`outputs/skill-refine-loop.md` 会根据 task shape、verifier availability 和 iteration budget 配置一个 evaluator-optimizer loop。它会产出 generator、evaluator/verifier 和 optimizer 的 prompts，以及 stop policy。

## 练习

1. 用 max_iterations=1 运行 toy。CRITIC 仍然有帮助吗？
2. 把外部 verifier 换成一个 noisy verifier（随机 30% false positives）。loop 会怎样？这就是 2026 年大多数 guardrail stacks 的现实。
3. 实现一个“generator-critic on different models”变体：大模型生成，小模型 critique。它能胜过 same-model 吗？
4. 阅读 CRITIC Section 3（arXiv:2305.11738 v4）。说出三类 verification-tool，并分别给一个例子。
5. 将 OpenAI Agents SDK 的 `output_guardrails` 映射到 CRITIC 的 verifier role。SDK 做错了什么，又做对了什么？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Self-Refine | “会修正自己的 LLM” | 单模型中的 Generate -> feedback -> refine loop，带 history |
| CRITIC | “Tool-grounded verification” | 用外部 verifier（search、code、calc、tests）替换 feedback |
| Evaluator-Optimizer | “Anthropic workflow pattern” | 两个角色：evaluator 打分、optimizer 修订；循环到收敛 |
| Output guardrail | “Post-hoc check” | OpenAI Agents SDK validator，在 agent 产出 output 后运行 |
| Verify step | “Critique phase” | 承重决策点：grounded 还是 self-rated |
| Refine history | “模型已经尝试过什么” | 先前 outputs + critiques 被前置到 refine prompt；去掉后质量崩塌 |
| Rubber-stamp loop | “自我认同失败” | Same-prompt critique 返回“looks good”；用结构不同的 prompts 修复 |
| Stop condition | “Convergence test” | Verifier passes OR no feedback AND iteration cap；绝不要只用单一条件 |

## 延伸阅读

- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — canonical paper
- [Gou et al., CRITIC (arXiv:2305.11738)](https://arxiv.org/abs/2305.11738) — tool-grounded verification
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — evaluator-optimizer workflow pattern
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — output guardrails as CRITIC-shaped verifiers
