# 前沿安全框架：RSP、PF、FSF

> 三个主要实验室框架定义了 2026 年 frontier capability 的行业治理。Anthropic Responsible Scaling Policy v3.0（2026 年 2 月）引入分层 AI Safety Levels（ASL-1 到 ASL-5+），以 biosafety levels 为模型，并在 2025 年 5 月对 CBRN-relevant models 激活 ASL-3。OpenAI Preparedness Framework v2（2025 年 4 月）定义了 tracked capabilities 的五项 criteria，并将 Capabilities Reports 与 Safeguards Reports 分离。DeepMind Frontier Safety Framework v3.0（2025 年 9 月）引入 Critical Capability Levels，包括新的 Harmful Manipulation CCL。三者现在都包含 competitor-adjustment clauses，允许在 peer labs 没有 comparable safeguards 也发布时进行 deferral。跨实验室 alignment 仍是结构性的，而不是术语性的：“Capability Thresholds”、“High Capability thresholds” 和 “Critical Capability Levels” 表示类似构造。

**类型:** Learn
**语言:** none
**先修:** Phase 18 · 17 (WMDP), Phase 18 · 07-09 (deception failures)
**时间:** ~75 minutes

## 学习目标

- 描述 Anthropic 的 ASL tier structure，以及是什么激活了 ASL-3。
- 说出 OpenAI Preparedness Framework v2 对 tracked capabilities 的五项 criteria。
- 描述 DeepMind 的 Critical Capability Level structure 和 Harmful Manipulation CCL。
- 解释 competitor-adjustment clauses，以及它们为何影响 race dynamics。
- 定义 safety case，并描述三支柱结构（monitoring、illegibility、incapability）。

## 要解决的问题

第 7-17 课建立了这样一个背景：deception 是可能的，dual-use capability 存在，而 evaluation 有局限。拥有 frontier-capable model 的实验室需要一种内部治理结构，用来：
- 定义什么时候需要新 safeguards 的 thresholds。
- 定义 scaling 前所需的 evaluations。
- 描述 safety case 应该是什么样子。
- 处理 race-dynamic problem（如果 competitors 没有 safeguards 也发布，你该怎么办？）。

2025-2026 年的三个框架就是当前 state of the art——不完美、仍在演化，并且跨实验室已经足够 aligned，以至于治理问题现在变成了这些 frameworks 是否充分，而不是它们是否存在。

## 核心概念

### Anthropic Responsible Scaling Policy v3.0（2026 年 2 月）

ASL structure：
- ASL-1：不是 frontier model（被 weaker-than-frontier baseline 覆盖）。
- ASL-2：当前 frontier baseline；使用常规 safeguards 部署。
- ASL-3：catastrophic misuse 风险显著更高；CBRN-relevant capabilities。2025 年 5 月激活。
- ASL-4：AI R&D-2 crossing threshold；能够自动化 entry-level AI research 的模型。
- ASL-5+：advanced AI R&D；能够显著加速 effective scaling 的模型。

v3.0 新增：
- Frontier Safety Roadmaps（以 redacted form 公开）。
- Risk Reports（quarterly，部分 externally reviewed）。
- AI R&D 被拆分为 AI R&D-2 和 AI R&D-4。
- 一旦跨越 AI R&D-4，就需要 affirmative safety case，识别模型追求 misaligned goals 所带来的 misalignment risks。

### OpenAI Preparedness Framework v2（2025 年 4 月 15 日）

Tracked capabilities 的五项 criteria：
- **Plausible。** 存在合理 threat model。
- **Measurable。** 可以进行 empirical evaluation。
- **Severe。** Harm 很大。
- **Net-new。** 不是 pre-existing risk 的放大版。
- **Instantaneous-or-irremediable。** Harm 发生很快，或无法撤销。

满足全部五项的 capabilities 会被 tracked。其他则不会。

PF v2 的其他结构：
- 将 Capabilities Reports（模型能做什么）与 Safeguards Reports（有哪些 controls）分离。
- Safety Advisory Group reviews。
- Leadership approves；Board's Safety & Security Committee oversees。
- “Adjustment clause”：如果另一家 lab 在没有 comparable safeguards 的情况下发布，OpenAI 可以降低 requirements。

### DeepMind Frontier Safety Framework v3.0（2025 年 9 月）

按 domain 划分的 Critical Capability Levels（CCLs）：
- Bioweapon Uplift
- Cyber Uplift
- ML R&D Acceleration
- Harmful Manipulation（v3.0 新增）：能够在 high-stakes contexts 中实质性改变 beliefs/behavior 的模型。

v2.0（2025 年 2 月）添加了 Deceptive Alignment section，以及面向 ML R&D CCLs 的更高 security levels。

### Cross-lab alignment

- Anthropic “Capability Thresholds”。
- DeepMind “Critical Capability Levels”。
- OpenAI “High Capability thresholds”。

没有 industry-standard terminology。结构上 aligned：三个 frontier capability tiers，带有公开 evaluation criteria。三者都包含 competitor-adjustment clauses（2025 年以后）。

### Safety cases

Safety case 是一份 written argument，说明 deployment 在 worst-case assumptions 下是 acceptably safe 的。标准结构针对三根支柱：

- **Monitoring。** 如果 bad behaviour 发生，我们能检测到吗？
- **Illegibility。** 模型是否缺乏执行 coherent plan 以造成 harm 的能力？
- **Incapability。** 模型是否缺乏造成相关 harm 的 capability？

不同 safety cases 针对不同支柱。对于 ASL-3 CBRN case，incapability（通过 unlearning）是主要目标。对于 deceptive alignment，monitoring 和 illegibility 是目标。对于 cyber uplift，三者都相关。

### Race-dynamic problem

Competitor-adjustment clauses 具有争议。批评者认为它们会造成 race to the bottom：如果三家实验室都会在 competitor defect 时降低 requirements，equilibrium 就会转向 defection。支持者认为，如果 defecting lab 更不重视安全，unilateral safeguards 这个替代方案会产生更糟结果。

UK AISI、US CAISI 和 EU AI Office（第 24 课）是外部治理 counterpart。实验室框架是 voluntary；监管框架正在出现。

### 它在 Phase 18 中的位置

第 17-18 课是 deception 和 red-team analyses 之上的 measurement-and-governance layer。第 19-24 课覆盖 welfare、bias、privacy、watermarking 和 regulatory structure。第 28 课映射 operationalizes evaluations 的 research ecosystem（MATS、Redwood、Apollo、METR）。

## 实际使用

本课没有代码。阅读三个 primary sources：RSP v3.0、PF v2、FSF v3.0。将每家 lab 的 tier structure 映射到其他两家，并找出每家定义的一个其他两家没有的 threshold。

## 交付成果

本课产出 `outputs/skill-framework-diff.md`。给定一份 safety framework 或 release note，它会对照 RSP v3.0、PF v2、FSF v3.0，比较该框架的 threshold definitions、required evaluations 和 safety-case structure，并标记 cross-lab gaps。

## 练习

1. 阅读 RSP v3.0、PF v2 和 FSF v3.0。编译一张表：每家 lab 的 CBRN threshold、每家的 AI R&D threshold，以及每家的 required pre-deployment evaluation。

2. Competitor-adjustment clause 存在于三份 frameworks（2025+）中。写一段支持它的论证；再写一段反对它的论证。指出每个立场依赖的 assumption。

3. 为跨越 Anthropic AI R&D-4 threshold 的模型设计一个 safety case。说出三根支柱（monitoring、illegibility、incapability）各自需要的 evidence。

4. DeepMind FSF v3.0 引入 Harmful Manipulation CCL。提出三个 empirical measurements，用于指示模型已经跨越该 threshold。

5. 阅读 METR 的 “Common Elements of Frontier AI Safety Policies”（2025）。说出三个最强的 cross-lab convergences 和两个最大的 divergences。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| RSP | “Anthropic's framework” | Responsible Scaling Policy；ASL tiers；v3.0 2026 年 2 月 |
| PF | “OpenAI's framework” | Preparedness Framework；five criteria；v2 2025 年 4 月 |
| FSF | “DeepMind's framework” | Frontier Safety Framework；CCLs；v3.0 2025 年 9 月 |
| ASL-3 | “biosafety level 3-analog” | Anthropic 面向 CBRN-relevant capabilities 的 tier；2025 年 5 月激活 |
| CCL | “critical capability level” | DeepMind 的 threshold construct；per-domain |
| Safety case | “the formal argument” | 说明 deployment 在 worst-case assumptions 下 acceptably safe 的 written argument |
| Adjustment clause | “competitor defection allowance” | 当 competitors 没有 comparable safeguards 也发布时，可降低 requirements 的 framework provision |

## 延伸阅读

- [Anthropic — Responsible Scaling Policy v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy) — ASL tiers、roadmaps、AI R&D disaggregation
- [OpenAI — Updating the Preparedness Framework (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/) — five criteria、adjustment clause
- [DeepMind — Strengthening our Frontier Safety Framework (September 2025)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — CCL v3.0、Harmful Manipulation
- [METR — Common Elements of Frontier AI Safety Policies (2025)](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — cross-lab comparison
