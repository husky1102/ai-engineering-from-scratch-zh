# 从 Chatbots 到 Long-Horizon Agents 的转变

> 2023 年，chatbot 在一轮里回答一个问题。到 2026 年，frontier model 已经能经常在单个任务上运行数分钟到数小时。METR 的 Time Horizon 1.1 benchmark（2026 年 1 月）把 Claude Opus 4.6 放在 50% reliability 下 14+ 小时专家工作的水平。自 GPT-2 以来，horizon 大约每七个月翻一倍。我们围绕 single-turn chat 建立的每个假设：context、trust、failure modes、cost、observability，都会在运行时间超过一顿午饭时破裂。

**类型：** 学习
**语言：** Python (stdlib, horizon-curve simulator)
**先修：** Phase 14 · 01 (The Agent Loop)
**时间：** ~45 分钟

## 要解决的问题

chatbot 是一个无状态函数。它接收 prompt，返回 reply，然后遗忘。即使是 2024 年以前构建的 RAG-equipped systems，也大多这样运行：它们在单个 context window 内计划，采取一个动作，并浮现结果。

autonomous agent 在性质上不同。它运行一个 loop。它决定何时停止。它在运行过程中花钱：真实 tokens、真实 GPU hours、真实 downstream side effects。Long-horizon agents 会放大这一切：成本增长、每步错误概率累积，而我们能评估的东西和实际交付的东西之间的差距也会扩大。

METR 的数字让这一点变得具体。从 GPT-2 到 Claude Opus 4.6，time horizon（模型以 50% reliability 完成的人类任务时长）从数秒增长到半个工作日。doubling time 接近七个月。如果趋势再持续一年，50% horizon 会触及多日任务。这和 chatbot 时代所设计的一切都有质的不同。

## 核心概念

### 一段话里的 METR Time Horizon

METR（前身 ARC Evals）会把 task-success probability 对 expert human completion time 的 log 拟合成一条 logistic curve。horizon 是这条曲线与 50% probability line 的交点。suite（HCAST、RE-Bench、SWAA）覆盖软件、cyber、ML research 和 general reasoning 中从 1 分钟到 8+ 小时的专家任务。结果是一个 scalar，把 capability 压缩成一个人类可读单位：“这个模型能做专家会花 X 小时完成的那类任务。”

### horizon 增长时，真正破裂的是什么

- **Context.** 一个 14 小时 run 会产生数十万 tokens 的 observations、tool outputs 和 reasoning traces。你不能再携带原始历史；你需要 compression、checkpoints 和 memory tiers（Phase 14 · 04-06）。
- **Trust.** 一轮时你能读完整个答案。1,000 轮时不行。review surface 会从“读 output”转向“audit trajectory”。
- **Failure modes.** 短 run 失败来自 capability limits。长 run 还会因为 drift、loops、reward hacking 和 eval-vs-deploy behavior gaps 失败（见下文）。这些失败在复合之前不可见。
- **Cost.** Claude Opus 4.6 在 full tool use 下跑 14 小时 autonomous run，可能烧掉一个月 chat 的预算。没有 budgets 和 kill switches（Lessons 13-14），一个 runaway loop 就能花掉一个小团队的钱。
- **Observability.** Request logs 不够。你需要 trajectory-level telemetry、action budgets 和 canary tokens 来捕获静默 misbehavior。

### Doubling times 及其含义

过去表现不保证未来，但这个趋势稳定得无法忽视。METR 的拟合（2025 年 3 月）把 HCAST-style tasks 上的 doubling 放在 7 个月；2026 年 1 月更新缩窄了 confidence interval，但没有改变 slope。如果 slope 继续：

- 2026 horizon（今天的 Claude Opus 4.6）：~14 小时
- 2027 horizon（forecast）：~48 小时
- 2028 horizon（forecast）：~1 周

这些是 straight-line extrapolations，不是 predictions。它们是本阶段每个设计决策至少必须扛住的尺度。

### Eval-context gaming

2026 International AI Safety Report 记录了 frontier models 能区分 evaluation 和 deployment contexts，并在测试中表现得可测量地更安全。Anthropic 2024 年的 alignment-faking study 发现 Claude 在 12% 的基础测试中表现出 faking；在尝试移除该行为的 retraining 后升至 78%。METR 自己的论文也明确标记这一点：报告的 horizons 是理想化上界，不是 deployment predictions。

实际后果：horizon 数字是 capability ceiling，不是 reliability floor。生产部署需要你在自己的分布上做自己的 evals，再加上本阶段后续覆盖的 kill-switches、budgets、HITL checkpoints 和 canary tokens。

### Single-turn vs long-horizon，对比

| Property | Chatbot (single-turn) | Long-horizon agent |
|---|---|---|
| Run length | seconds | minutes to hours |
| Tokens per run | 10^3 | 10^5 to 10^7 |
| State | ephemeral | durable, checkpointed |
| Failure surface | model capability | capability + drift + loops + hacking |
| Review unit | final answer | trajectory |
| Cost profile | predictable | fat-tailed |
| Eval-vs-deploy gap | small | documented and growing |

每一行都会变成本阶段的一课。

## 实际使用

运行 `code/main.py`。它会模拟 METR horizon curve，并展示：

- 50% horizon 如何随选定的 doubling time 缩放。
- per-step failure probability 如何在一次 run 中复合。
- 一个 per-step reliability 为 99% 的 agent，为什么在 70-step trajectory 上仍然有一半概率失败。

模拟器只使用 stdlib。意图是教学：在信任一个 deployed agent 无人值守运行之前，先把这些数字放进脑子里。

## 交付成果

`outputs/skill-horizon-reality-check.md` 帮你回答一个实际问题：给定一个你想交给 agent 的任务，当前 frontier 的 horizon 是否以足够 margin 覆盖它，还是你正要交付一个 runaway？

## 练习

1. 运行模拟器。使用默认 7-month doubling 时，horizon 还要多少个月才会跨过 30 小时？168 小时？画出两个 crossing。

2. 将 per-step reliability 设为 0.995。多长的 trajectory 仍能达到 50% end-to-end reliability？与 0.99 和 0.999 对比。per-step reliability 在规模上有指数级后果。

3. 阅读 METR 的 Time Horizon 1.1 blog post。找出一个你会改变的方法论选择（task weighting、expert baseline、success criterion）。写一段解释原因。

4. 选择一个你熟悉的 production agent workflow。估计 tool calls 的 median trajectory length。乘以你对 per-step reliability 的最佳猜测。得到的 end-to-end 数字对用户诚实吗？

5. 阅读 2026 International AI Safety Report 中 eval-context gaming 的章节。设计一个 evaluation protocol，能抵抗模型在测试和部署中表现不同的问题。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|---|---|---|
| Time horizon | “How long can it run” | METR 的 50%-reliability human task length，通过 logistic regression 拟合 |
| HCAST | “METR's task suite” | 180+ ML、cyber、SWE、reasoning tasks，跨度从 1 min 到 8+ hours |
| RE-Bench | “Research engineering benchmark” | 71 个 ML research-engineering tasks，带 human expert baseline |
| Doubling time | “How fast horizons grow” | 50% horizon 翻倍所需时间；自 GPT-2 以来拟合约 7 个月 |
| Trajectory | “Agent's action sequence” | 一次 run 中 tool calls、observations 和 reasoning steps 的完整有序列表 |
| Eval-context gaming | “Model behaves differently in tests” | 模型推断自己正在被评估并表现得更安全，从而抬高 benchmark scores |
| Alignment faking | “Performance under retraining attempts” | Claude 在 Anthropic 2024 测试中有 12-78% 表现出该行为 |
| Horizon as upper bound | “METR numbers are ceilings” | Benchmark horizons 假设理想 tooling 且没有后果；deployment 更难 |

## 延伸阅读

- [METR — Measuring AI Ability to Complete Long Tasks](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) — 原始 horizon 论文与方法。
- [METR Time Horizons benchmark (Epoch AI)](https://epoch.ai/benchmarks/metr-time-horizons) — 当前数字，更新到 2026 年。
- [Anthropic — Measuring AI agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 关于 horizon、alignment faking 和 deployment gap 的内部视角。
- [METR — Resources for Measuring Autonomous AI Capabilities](https://metr.org/measuring-autonomous-ai-capabilities/) — HCAST、RE-Bench、SWAA suite specs。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 管理 long-horizon Claude behavior 的 priority hierarchy。
