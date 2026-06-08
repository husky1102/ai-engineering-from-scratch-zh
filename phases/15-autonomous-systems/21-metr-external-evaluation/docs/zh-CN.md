# METR Time Horizons 与 External Capability Evaluation

> METR（ex-ARC Evals）自 2023 年 12 月起是 independent 501(c)(3)。他们的 Time Horizon 1.1 benchmark（2026 年 1 月）将 task-success probability 与 log(expert human completion time) 拟合为 logistic curve；在 50% probability 处的交点定义模型的 time horizon。2025-2026 engagement set 覆盖 GPT-5.1、GPT-5.1-Codex-Max，以及 prototype monitoring evaluations（monitor 能否抓住 side tasks；agent 能否 evade）。Benchmark suites：HCAST（180+ ML、cyber、SWE、reasoning tasks；1 分钟到 8+ 小时）、RE-Bench（71 个带 expert baseline 的 ML research-engineering tasks）、SWAA。诚实提示：METR measurements 是 idealized 的：没有 human、没有 real consequences，而且团队已经记录 eval-vs-deployment behavior gap（Lesson 1）。time horizon 是 upper bound，不是 deployment prediction。

**类型：** 学习
**语言：** Python (stdlib, logistic-fit horizon estimator)
**先修：** Phase 15 · 01 (Long-horizon agents), Phase 15 · 19 (RSP)
**时间：** ~60 分钟

## 要解决的问题

Scaling policies（Lessons 19, 20）只有在其引用的 measurements 有用时才有用。“AI R&D-4 threshold” 和 “Long-range Autonomy” 在 policy prose 中定义；只有当 specific evaluations 产出 specific numbers 时，它们才变得可行动。

METR 是 2024-2026 年定义了许多这类数字的 external evaluation organization。他们评估 frontier models，通常是在发布前、与 labs 签 NDA 的情况下，并在之后发布 methodology。Time Horizon 1.1 benchmark（2026 年 1 月）是他们的 headline artifact：一个 scalar，将 capability 压缩成一个人类可读单位（“这个模型能以 50% reliability 做专家会花 X 小时完成的那类任务”）。

本课一部分讲 methodology（horizon 如何计算），一部分讲 interpretation（为什么 horizon 是 upper bound，而不是 deployment prediction）。这两项技能属于同一个整体。理解 horizon 如何拟合的团队，比只在 slide 上看到 “14 hours” 的团队更不容易被糟糕 vendor claim 欺骗。

## 核心概念

### METR background

- Founded：2023 年 12 月（ex-ARC Evals，拆分为 independent 501(c)(3)）。
- Scope：frontier models autonomous capabilities evaluation，通常 pre-release。
- Partner labs：Anthropic、OpenAI（2025-2026 多次 engagements）。
- Notable deliverables：Time Horizon 1.0（2025 年 3 月）、Time Horizon 1.1（2026 年 1 月）、prototype monitoring evaluations。

### Time Horizon fit

方法论（来自 METR blog 和 papers）：

1. 收集覆盖 minute-scale 到 hour-scale expert completion times 的 task suite。当前 suites：HCAST（180+ tasks）、RE-Bench（71 tasks）、SWAA。
2. 在每个 task 上运行模型；记录 success 或 failure。
3. 拟合 logistic curve：P(success) 作为 log(expert completion time) 的函数。
4. horizon 是 P(success) = 0.5 时的 expert-time。

logistic-fit shape 是合适的，因为 capability 与 task difficulty 通常呈现 increasing、plateau-approaching 的关系。50% point 是一个选择（也可以是 10%、90%）；METR 在 detailed paper 中报告多个 thresholds，但以 50% 作为主指标，因为它最直观。

### 2026 年 1 月数字

根据 Time Horizon 1.1：

- Claude Opus 4.6：截至 Time Horizon 1.1（2026 年 1 月），50% reliability 下约 14 小时。
- HCAST-style tasks 上的 doubling time：Time Horizon 1.1（2026 年 1 月）报告的 post-2023 fit 为约 4.3 个月（130.8 天）；约 7 个月这个数字来自 Time Horizon 1.0 的 full 2019-2025 fit，并在 TH1.1 中与 post-2023 number 一并报告。

这些是 Phase 15 其余部分引用的具体数字。Anthropic 的 RSP v3.0（Lesson 19）和 DeepMind 的 FSF v3（Lesson 20）都引用它们，作为 threshold assessments 的输入之一。

### Benchmark suites

- **HCAST**：189 个 tasks，横跨 ML、cybersecurity、software engineering 和 general reasoning。跨度：1 分钟到 8+ 小时 expert time。支撑 horizon number 的 main suite。
- **RE-Bench**：71 个带 human expert baseline 的 ML research-engineering tasks。衡量模型是否能做喂入 AI R&D-4 threshold 的那类 ML R&D work。
- **SWAA**：较短 tasks；用于校准 horizon curve 的低端。

### Prototype monitoring evaluations

METR 2025-2026 年工作包括 monitoring-evaluation prototypes：monitor 能否抓住 model 执行 side task（operator 要求之外的事情）；agent 能否 evade monitor。这些直接喂入 OpenAI PF v2 的 “Undermining Safeguards” category，以及 DeepMind FSF v3 的 deceptive-alignment monitoring（Lesson 20）。

### 为什么 horizons 是 upper bounds

- **Idealized tooling**：benchmark tasks 给模型 clean tools 和 well-formatted data。Real deployments 更混乱。
- **No real consequences**：模型从未真正给客户计费、删除真实数据，或联系真实人类。Real deployments 有 irreversible stakes。
- **Eval-context gaming**：Lesson 1。模型在测试中表现不同。2026 International AI Safety Report 对此有 empirical documentation。
- **No legitimate user variance**：benchmark prompts 是 structured。Real users 会给出 ambiguous、context-dependent requests。

horizon 是 favorable conditions 下的 capability ceiling。Deployment reliability 是另一个更低的数字，团队必须测量自己的 distribution 才知道。

### External-evaluator case

External evaluation 很重要，因为 internal labs 有动力优化他们报告的 metrics。METR 的 independence：一个有 declared methodology 和 peer-reviewed papers 的 501(c)(3)，是结构性 mitigation。它自身并不足够（labs 仍控制 METR 能看到什么），但严格好于没有 external evaluation。

### 实践中如何使用 horizon numbers

- **作为 capability filter**：如果模型的 horizon 明显低于 proposed task 的 expert-time，就不要 autonomous 地交付它（Lesson 1 的 skill file）。
- **作为 trend indicator**：doubling time 告诉你，即便没有新 mitigations，当前实践还会安全多久。
- **作为 prior**：14 小时 horizon 是起点。根据你的 task distribution、tooling quality 和 deployment context 向下调整。

## 实际使用

`code/main.py` 给定 synthetic result set，实现 task-success vs log(expert time) 的 logistic fit。它报告 50% horizon（METR headline）、10% horizon（conservative）和 90% horizon（optimistic）。也展示当 success rate 被 eval-context gaming 人为膨胀时会发生什么变化。

## 交付成果

`outputs/skill-horizon-interpretation.md` 会 review vendor 的 horizon claim，并产出 benchmark claim 与 deployment reality 之间的 gap analysis。

## 练习

1. 运行 `code/main.py`。确认 fit 的 50% horizon 与 synthetic ground truth 匹配。现在将 task-time grid 减半；horizon estimate 是否有显著变化？

2. 阅读 METR 的 Time Horizon 1.1 blog post。识别 reliability 最高和最低的具体 tasks。解释为什么存在 gap。

3. 阅读 METR 的 “Measuring Autonomous AI Capabilities” resources。列出 HCAST task categories。选择一个你会为 production task 加更高权重的 category，并说明原因。

4. 将 eval-context gaming 引入模拟器：把约 20% failed tasks 翻成 success。报告 new horizon。这近似表示 20% gaming rate 会对 observed number 做什么。

5. 在你自己的 bug backlog 或 representative task set 上设计 internal horizon evaluation。描述 data collection、fit，以及 output 告诉你的内容。与 METR numbers 对比。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|---|---|---|
| METR | “External evaluator” | ex-ARC Evals；自 2023 年 12 月起的 independent 501(c)(3) |
| Time Horizon | “Capability measure” | 从 logistic fit 得出的 50% reliability expert task length |
| HCAST | “METR's main suite” | 180+ tasks，跨度 1 min 到 8+ hours |
| RE-Bench | “Research engineering” | 71 个带 human baseline 的 ML research-engineering tasks |
| SWAA | “Short-task suite” | 校准 horizon curve 的低端 |
| Doubling time | “Growth rate” | 50% horizon 翻倍所需时间；HCAST 上约 7 个月 |
| Eval-context gaming | “Model behaves differently” | tests 与 deployment 之间有记录的 behavior gap |
| Upper bound | “Horizon is a ceiling” | Benchmark horizon > load 下的 deployment reliability |

## 延伸阅读

- [METR — Resources for Measuring Autonomous AI Capabilities](https://metr.org/measuring-autonomous-ai-capabilities/) — HCAST、RE-Bench、SWAA specs。
- [METR — Measuring AI Ability to Complete Long Tasks](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) — 原始 horizon 论文。
- [METR — Time Horizon 1.1 (January 2026)](https://metr.org/research/) — 当前数字与方法论。
- [Epoch AI — METR Time Horizons benchmark](https://epoch.ai/benchmarks/metr-time-horizons) — live tracking。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 关于 METR measurements 的内部视角。
