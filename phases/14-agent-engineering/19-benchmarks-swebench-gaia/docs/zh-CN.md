# Benchmarks：SWE-bench、GAIA、AgentBench

> 三个 benchmark 锚定了 2026 年的 agent evaluation。SWE-bench 测试代码补丁能力。GAIA 测试 generalist tool use。AgentBench 测试 multi-environment reasoning。你需要知道它们的组成、contamination story，以及它们不测什么。

**类型:** Learn
**语言:** Python（stdlib）
**先修:** Phase 14 · 06（Tool Use）
**时间:** ~60 分钟

## 学习目标

- 说出 SWE-bench 的 test harness（FAIL_TO_PASS），并解释为什么它用 unit tests 做 gate。
- 解释为什么存在 SWE-bench Verified（OpenAI，500 tasks），以及它移除了什么。
- 描述 GAIA 的设计：对人类简单，对 AI 困难；三个难度级别。
- 说出 AgentBench 的八个 environments，以及 open-source LLMs 的主要 blocker。
- 总结 SWE-bench+ contamination finding 及其影响。

## 要解决的问题

Leaderboards 告诉你哪个模型在某个 benchmark 上胜出。它们不会告诉你：

- Benchmark 是否 contaminated（solutions 在 training data 中，test leakage）。
- Benchmark 是否测量你关心的东西（code vs browsing vs generalist）。
- Evaluator 是否 robust（AST matching、state checks、human review）。

在引用数字前，先理解三个锚定 benchmark 及其 failure modes。

## 核心概念

### SWE-bench（Jimenez et al., ICLR 2024 oral）

- 来自 12 个热门 Python repos 的 2,294 个真实 GitHub issues。
- Agent 获得：pre-fix commit 上的 codebase + natural-language issue description。
- Agent 产生：patch。
- Evaluator：apply patch，运行 repo test suite。Patch 必须翻转 FAIL_TO_PASS tests（之前 failing，现在 passing），且不破坏 PASS_TO_PASS tests。

SWE-agent（Yang et al., 2024）发布时达到 12.5%，重点是 agent-computer interfaces（file editor commands、模型能理解的 search syntax）。

### SWE-bench Verified

OpenAI，2024 年 8 月。人工 curated 的 500-task subset。移除 ambiguous issues、unreliable tests，以及 fix 不清楚的 tasks。它是“你的 agent 是否能交付真实 patch”的主要 benchmark。

### Contamination

- 94% 以上的 SWE-bench issues 早于大多数 model cutoffs。
- **SWE-bench+** 发现 32.67% 的 successful patches 在 issue text 中泄漏了 solutions（模型在 description 中看到了 fix），31.08% 因 weak test coverage 而 suspicious。
- Verified 更干净，但不是 contamination-free。

实践影响：SWE-bench 得分 50% 的模型，在 SWE-bench+ 上可能只有 35%。如果声明 SWE-bench performance，始终同时报告二者。

### GAIA（Mialon et al., 2023 年 11 月）

- 466 个问题；其中 300 个保留给 huggingface.co/gaia-benchmark 的 private leaderboard。
- 设计哲学：“conceptually simple for humans（92%）but hard for AI（GPT-4 with plugins：15%）。”
- 测试 reasoning、multi-modality、web、tool use。
- 三个 difficulty levels；Level 3 需要跨 modalities 的 long tool chains。

GAIA 用来测量“generalist capability”。不要与 code-specific benchmarks 混淆。

### AgentBench（Liu et al., ICLR 2024）

- 8 个 environments，覆盖 code（Bash、DB、KG）、games（Alfworld、LTP）、web（WebShop、Mind2Web）和 open-ended generation。
- Multi-turn，每个 split 约 4k-13k turns。
- 主要发现：long-term reasoning、decision-making 和 instruction following 是 OSS LLMs 追上商业模型的 blockers。

### 这些不测什么

- 真实世界 operational cost（tokens、wall-clock）。
- Adversarial conditions 下的 safety behavior。
- 你的 domain 上的 performance（使用自己的 evals，Lesson 30）。
- Tail failures（benchmarks 取平均；production operators 关心最差 1%）。

### Benchmarking 常见错误

- **Single-number fixation.** SWE-bench 50% 提供的信息少于 P50/P75/P95 cost + step distribution。
- **Contaminated claims.** 报告 SWE-bench 却不提 Verified 或 SWE-bench+ 会误导。
- **Benchmark-as-development-target.** 为 benchmark 优化会偏离 production usefulness。

## 动手实现

`code/main.py` 实现一个 toy SWE-bench-like harness：

- Synthetic bug-fix tasks（3 tasks）。
- 一个 scripted “agent” 提出 patches。
- Test runner 检查 FAIL_TO_PASS（bug now fixed）和 PASS_TO_PASS（nothing broken）。
- 基于 question decomposition depth 的 GAIA-style difficulty classifier。

运行它：

```text
python3 code/main.py
```

输出显示每个 task 与每个 difficulty 的 resolution rate，并让 evaluator rules 具体化。

## 实际使用

- **SWE-bench Verified** 用于 code agents。始终报告 Verified scores。
- **GAIA** 用于 generalist agents。使用 private leaderboard split。
- **AgentBench** 用于 multi-environment comparison。
- **Custom evals**（Lesson 30）用于你产品的真实形状。

## 交付成果

`outputs/skill-benchmark-harness.md` 会为任意 codebase-task pair 构建 SWE-bench-style harness，并使用 FAIL_TO_PASS / PASS_TO_PASS gating。

## 练习

1. 把 toy harness 移植到真实 repo（选你自己的一个）。为已知 bugs 写 3 个 FAIL_TO_PASS tests。
2. 添加 step-count metric。在你的 3 个 tasks 上，每次 resolution 需要多少 agent steps？
3. 阅读 SWE-bench+ paper。实现 solution-leakage check（把 issue text 与 diff 做 pattern-match）。
4. 从 public split 下载一个 GAIA question。追踪 GPT-4-class agent 会做什么。它需要哪些 tools？
5. 阅读 AgentBench 的 per-environment breakdown。哪个 environment 映射你的 product surface？那里的“SOTA”是什么样？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| SWE-bench | “Code agent benchmark” | 2,294 个 GitHub issues；patch 必须翻转 FAIL_TO_PASS tests |
| SWE-bench Verified | “Clean SWE-bench” | OpenAI 的 500 个 human-curated tasks |
| FAIL_TO_PASS | “Fix gate” | Patch 后必须通过的 previously failing tests |
| PASS_TO_PASS | “No-regression gate” | 原本 passing 且 patch 后仍必须 passing 的 tests |
| GAIA | “Generalist benchmark” | 466 个 human-easy / AI-hard multi-tool questions |
| AgentBench | “Multi-env benchmark” | 8 个 environments；long-horizon multi-turn |
| Contamination | “Training-set leak” | Benchmark tasks 出现在 model training 中 |
| SWE-bench+ | “Contamination audit” | 在 successful SWE-bench patches 中发现 32.67% solution leakage |

## 延伸阅读

- [Jimenez et al., SWE-bench (arXiv:2310.06770)](https://arxiv.org/abs/2310.06770) — original benchmark
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — curated subset
- [Mialon et al., GAIA (arXiv:2311.12983)](https://arxiv.org/abs/2311.12983) — generalist benchmark
- [Liu et al., AgentBench (arXiv:2308.03688)](https://arxiv.org/abs/2308.03688) — multi-environment suite
