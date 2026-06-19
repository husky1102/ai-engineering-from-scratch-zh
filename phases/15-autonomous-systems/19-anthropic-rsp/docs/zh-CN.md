# Anthropic 负责任扩展政策 v3.0

> RSP v3.0 于 2026 年 2 月 24 日生效，取代 2023 policy。Two-tier mitigation：Anthropic 将单方面执行的内容 vs framed 为 industry-wide recommendation 的内容（包括 RAND SL-4 security standards）。新增 Frontier Safety Roadmaps 和 Risk Reports，把它们作为 standing documents，而不是一次性交付物。移除了 2023 pause commitment。引入 AI R&D-4 threshold：一旦跨过，Anthropic 必须发布 affirmative case，识别 misalignment risks 和 mitigations。Claude Opus 4.6 没有跨过它。Anthropic 在 v3.0 announcement 中表示 “confidently ruling this out is becoming difficult.” SaferAI 将 2023 RSP 评为 2.2；他们把 v3.0 降级到 1.9，使 Anthropic 与 OpenAI 和 DeepMind 一起落入 “weak” RSP category。Qualitative thresholds 取代了 2023 quantitative commitments；移除 pause clause 是最尖锐的退步。

**类型：** 学习
**语言：** Python (stdlib, RSP threshold decision engine)
**先修：** Phase 15 · 06 (AAR), Phase 15 · 07 (RSI)
**时间：** ~45 分钟

## 要解决的问题

Frontier labs 发布的 scaling policies，一部分是 technical documents，一部分是 governance documents，一部分是给 regulators 的信号。RSP v3.0 是当前 Anthropic document。仔细阅读它很重要，不是因为遵守它有约束力（并没有），而是因为它的 framing 会塑造一个 lab 如何理解 catastrophic risk，以及如何向公众沟通 trade-offs。

v3.0 vs v2.0 diff 是有用的单位。新增了什么：Frontier Safety Roadmaps、Risk Reports、AI R&D-4 threshold。移除了什么：2023 pause commitment。重新 framed 了什么：Anthropic-unilateral 与 industry-recommendation 之间的 two-tier mitigation schedule。外部 review：SaferAI 将分数从 2.2（v2）降到 1.9（v3.0）。这就是一个 scaling policy 如何在看起来更精致的同时变得不那么严谨。

## 核心概念

### Two-tier mitigation schedule

- **Anthropic unilateral actions**：无论其他 labs 做什么，Anthropic 都会做什么。包括 threshold 以上停止 training、具体 security measures、具体 deployment gates。
- **Industry-wide recommendations**：Anthropic 认为整个 industry 应该共同做什么。包括 RAND SL-4 security standards。这些不是 Anthropic 方面的 commitments；它们是 policy advocacy。

Two-tier structure 并不存在于 v2。这意味着读者需要查看每个 commitment 位于哪一列。位于 “industry-wide recommendation” 列的 security measure 不是 Anthropic 的承诺；它是 Anthropic 的希望。

### AI R&D-4 threshold

这是 RSP v3.0 命名为重要 next threshold 的 capability level。具体来说：一个能以 competitive cost 自动化 substantial fraction of AI research 的模型。一旦 Anthropic 认为某个模型跨过它，他们必须在 continued scaling 之前发布 affirmative case，识别 misalignment risks 和 mitigations。

根据 v3.0 announcement，Claude Opus 4.6 没有跨过它。文档补充：“confidently ruling this out is becoming difficult.” 这句话很重要；它承认这个 threshold 已经足够接近，成为 live concern，而不是 speculative limit。

Lesson 6（Automated Alignment Research）和 Lesson 7（Recursive Self-Improvement）会直接喂入这个 threshold。Automated alignment researchers 跨过 research-quality bars，是 AI R&D-4 threshold 正在接近的证据。

### Frontier Safety Roadmaps 和 Risk Reports

v3.0 将两类 artifact 提升为 standing documents：

- **Frontier Safety Roadmap**：forward-looking document，描述 planned safety work、capability expectations 和 mitigation research。
- **Risk Report**：specific models 发布后的 retrospective document，描述 observed capability 和 residual risk。

二者都是 public。二者都会按声明 cadence 更新。它们的 utility 是：读者可以追踪 Anthropic 在 Roadmap 中说会做的事情，与其在 Risk Report 中报告的事情如何对应。

### 移除 pause clause

2023 RSP 包含一个明确 pause commitment：如果模型跨过 specific capability thresholds，training 会暂停，直到 mitigations 到位。v3.0 用更软的 formulation 替换了 explicit pause（发布 affirmative case，如果 mitigations adequate 则继续）。SaferAI 和其他 analysts 直接指出，这是新文档中最强的 regression。

支持这一改变的 policy argument：2023 年的 quantitative thresholds 到 2026-era capability benchmarks 时已不可达，因为 benchmarks 自身被 re-scaled。反方 argument：scaling policy 中的 pause clause 是一个 commitment device；移除它，会移除 policy 的 credibility。

### SaferAI 的降级

SaferAI 是一个为 RSP-style documents 打分的 independent organization。他们公开 rating：2023 Anthropic RSP 得分 2.2（在一个 4.0 是当前最好 RSP、1.0 是 nominal 的 scale 上）。v3.0 得分 1.9。这让 Anthropic 从 “moderate” 移动到 “weak”，与 OpenAI 和 DeepMind 一起进入 weak category。

SaferAI 给出的 downgrade factors：
- Qualitative thresholds 取代 quantitative ones。
- Pause commitment 被移除。
- AI R&D-4 threshold mitigations 被描述为 “affirmative case”，而不是 specific measures。
- Review mechanisms 依赖 Anthropic 的 Safety Advisory Group，independent oversight 有限。

### 本课不是什么

这不是一节 compliance 课。RSP v3.0 不是 regulation；没有任何东西强制 Anthropic 遵守它。本课训练的是以应有的 specificity 和 skepticism 阅读文档。Scaling policies 是 frontier labs 面向 public 发出的 catastrophic-risk posture 主要信号。把它们读好，是任何依赖 frontier capabilities 的从业者都需要的 practical skill。

## 实际使用

`code/main.py` 实现一个小型 decision engine，镜像 RSP threshold-evaluation shape：给定一个 candidate model 和一组 capability measurements，返回 AI R&D-4 threshold 是否被跨过、required affirmative-case sections，以及 deployment 是否可以继续。它故意很简单；重点是让文档逻辑显式化。

## 交付成果

`outputs/skill-scaling-policy-review.md` 会对一个 scaling policy（Anthropic、OpenAI、DeepMind 或 internal）按 v3.0 reference 做 review：two-tier structure、thresholds、pause commitments、independent review。

## 练习

1. 运行 `code/main.py`。输入三个不同 capability levels 的 synthetic models。确认 threshold evaluator 行为符合预期，并产生正确的 affirmative-case template。

2. 完整阅读 RSP v3.0（32 页）。识别每一个位于 “industry-wide recommendation” tier 的 commitment。其中哪些在 v2 中会是 “Anthropic unilateral”？

3. 阅读 SaferAI 的 RSP grading methodology。将他们的 rubric 应用于文档，复现 v3.0 的 1.9 分。哪一行 rubric 最推动 downgrade？

4. 2023 pause commitment 被移除了。提出一个 replacement commitment，既保留 policy credibility，又承认 2026 benchmark-rescaling problem。

5. 将 RSP v3.0 与 OpenAI Preparedness Framework v2（Lesson 20）对比。选择一个 v3.0 更强的 area。再选择一个 Preparedness Framework 更强的 area。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|---|---|---|
| RSP | “Anthropic's scaling policy” | Responsible Scaling Policy；v3.0 自 2026 年 2 月 24 日生效 |
| AI R&D-4 | “Research-automation threshold” | 以 competitive cost 自动化 substantial AI research 的能力 |
| Affirmative case | “Safety justification” | 公开论证 risks 已识别且 mitigations 足够 |
| Frontier Safety Roadmap | “Forward plan” | 关于 planned safety work 和 expected capabilities 的 standing document |
| Risk Report | “Retrospective on a model” | 发布后关于 observed capability 和 residual risk 的 standing document |
| Two-tier mitigation | “Unilateral vs industry” | 分离 Anthropic commitments 与 industry recommendations |
| Pause commitment | “2023 clause” | 明确承诺暂停 training；已在 v3.0 移除 |
| SaferAI rating | “Independent RSP grade” | 第三方 rubric；v3.0 得分 1.9（v2 为 2.2） |

## 延伸阅读

- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — 完整 32 页 policy。
- [Anthropic — RSP v3.0 announcement](https://www.anthropic.com/news/responsible-scaling-policy-v3) — 从 v2 到 v3 的 changes summary。
- [Anthropic — Frontier Safety Roadmap](https://www.anthropic.com/research/frontier-safety) — RSP v3.0 链接的 standing document。
- [Anthropic — Risk Report: Claude Opus 4.6](https://www.anthropic.com/research/risk-report-claude-opus-4-6) — 当前 frontier model 的 retrospective。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 将 AI R&D-4 与 measured autonomy 连接起来。
