# Alignment Research Ecosystem——MATS、Redwood、Apollo、METR

> 五个组织定义了 2026 年非实验室 alignment research layer。MATS（ML Alignment & Theory Scholars）：自 2021 年末以来 527+ researchers，180+ papers，10K+ citations，h-index 47；2024 年夏季 cohort 以 501(c)(3) 注册，约 90 scholars 和 40 mentors；2025 年前 alumni 中 80% 从事 safety/security，200+ 在 Anthropic、DeepMind、OpenAI、UK AISI、RAND、Redwood、METR、Apollo。Redwood Research：由 Buck Shlegeris 创立的 applied alignment lab；提出 AI Control（第 10 课）；与 UK AISI 合作 control safety cases。Apollo Research：为 frontier labs 做 pre-deployment scheming evaluations；撰写 In-Context Scheming（第 8 课）和 Towards Safety Cases for AI Scheming。METR（Model Evaluation and Threat Research）：task-based capability evaluations、autonomous-task time-horizon studies；“Common Elements of Frontier AI Safety Policies” 比较实验室框架。Eleos AI Research：model-welfare pre-deployment evaluations（第 19 课）；进行了 Claude Opus 4 welfare assessment。

**类型:** Learn
**语言:** 无
**先修:** Phase 18 · 01-27 (prior Phase 18 lessons)
**时间:** ~45 minutes

## 学习目标

- 识别非实验室 alignment research ecosystem 的五个组织及其核心产出。
- 描述 MATS 的规模（scholars、papers、h-index）以及它作为 talent pipeline 的作用。
- 描述 Redwood 的 AI Control agenda 以及它与 UK AISI 的合作关系。
- 描述 METR 的 task-based evaluation methodology。

## 要解决的问题

Frontier labs（第 18 课）会在内部产出 safety evaluations，并发布部分结果。实验室之外的 ecosystem 是验证这些 evaluations、率先发现新 failure modes、训练人才的地方。理解这个生态有助于判断哪些研究发现被哪些人信任。

## 核心概念

### MATS（ML Alignment & Theory Scholars）

始于 2021 年末。它是一个 research mentorship program；scholars 会与 senior researcher 一起，在一个具体 alignment problem 上工作 10-12 周。

规模（2026）：
- 自成立以来 527+ researchers。
- 发表 180+ papers。
- 10K+ citations。
- h-index 47。
- 2024 年夏季：90 scholars + 40 mentors；以 501(c)(3) 注册。

职业结果：2025 年前 alumni 中约 80% 正在从事 safety/security。200+ 在 Anthropic、DeepMind、OpenAI、UK AISI、RAND、Redwood、METR、Apollo。

### Redwood Research

Applied alignment lab。由 Buck Shlegeris 创立。提出 AI Control agenda（第 10 课）。与 UK AISI 合作 control safety cases。为 DeepMind 和 Anthropic 的 evaluation design 提供建议。

Canonical papers：Greenblatt、Shlegeris 等人，“AI Control”（arXiv:2312.06942，ICML 2024）；Alignment Faking（Greenblatt、Denison、Wright 等人，arXiv:2412.14093，与 Anthropic 合作）。

风格：具体 threat models、worst-case adversaries、可 stress-tested 的 concrete protocols。

### Apollo Research

为 frontier labs 做 pre-deployment scheming evaluations。撰写 In-Context Scheming（第 8 课，arXiv:2412.04984）。参与 2025 年 OpenAI anti-scheming training collaboration。产出 Towards Safety Cases for AI Scheming（2024）。

风格：在 agentic-setting evaluations 中让 deception 可以涌现；三支柱分解（misalignment、goal-directedness、situational awareness）。

### METR（Model Evaluation and Threat Research）

Task-based capability evaluations。Autonomous-task completion time-horizon studies。“Common Elements of Frontier AI Safety Policies”（metr.org/common-elements，2025）比较实验室框架。

与 Apollo 共同撰写 AI Scheming safety-case sketch。

风格：long-horizon task evaluations、empirical capability measurement、framework synthesis。

### Eleos AI Research

Model-welfare pre-deployment evaluations。进行了系统卡第 5.3 节中记录的 Claude Opus 4 welfare assessment。为第 19 课的 welfare-relevant claims 提供外部 methodology check。

### 流动路径

MATS 训练 researchers。毕业生去 Anthropic、DeepMind、OpenAI（lab safety teams），或去 Redwood、Apollo、METR、Eleos（external evaluation）。External evaluators 与 labs、UK AISI / CAISI 合作。Publications 再把生态反馈给 MATS，供下一届 cohort 使用。

### 为什么这一层重要

Single-source evaluations 不可靠：实验室评估自己的模型存在结构性利益冲突。External evaluators 可以提出并验证实验室可能低报的 failure modes。2024 年 Sleeper Agents paper（第 7 课）是 Anthropic + Redwood；Alignment Faking 是 Anthropic + Redwood；In-Context Scheming 是 Apollo；Anti-Scheming 是 Apollo + OpenAI。Multi-org structure 就是 quality control。

### 它在 Phase 18 中的位置

第 7-11 课引用 Redwood 和 Apollo 的工作；第 18 课引用 METR 的 framework comparison；第 19 课引用 Eleos。第 28 课是后续 Phase 所依赖 ecosystem 的显式组织地图。

## 实际使用

没有代码。阅读 METR 的 “Common Elements of Frontier AI Safety Policies”，把它作为 external synthesis 如何为 lab-internal policy work 增值的例子。

## 交付成果

本课产出 `outputs/skill-ecosystem-map.md`。给定一条 alignment claim 或 evaluation，它会识别组织、publication venue 和 methodological style，并与已知 counterpart organisations 交叉检查。

## 练习

1. 从第 7-15 课中选择一篇论文，识别参与组织。把作者与 MATS alumni 和当前 ecosystem affiliations 交叉检查。

2. 阅读 METR 的 “Common Elements of Frontier AI Safety Policies”。识别他们强调的三个 cross-lab convergences 和两个最大 divergences。

3. MATS 的 career outcomes 约 80% 是 safety/security。论证这种 selection pressure 是 adaptive（训练这个领域）还是 biased（过滤掉 heterodox positions）。

4. Redwood 和 Apollo 都做 control/scheming 工作，但风格不同。选择一个 failure mode，并描述二者会如何调查它。

5. Eleos AI 是唯一一个纯 model-welfare 组织。设计一个假想的第二组织，关注另一个 welfare-adjacent question（cognitive liberty、robotic embodiment 等），并阐明其 methodology。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| MATS | “mentorship program” | ML Alignment & Theory Scholars；自 2021 年以来 527+ researchers |
| Redwood Research | “control lab” | Applied alignment；AI Control 作者；UK AISI partner |
| Apollo Research | “scheming evals” | 为 frontier labs 做 pre-deployment scheming evaluations |
| METR | “task-horizon evals” | Task-based capability evaluations；framework synthesis |
| Eleos AI | “welfare lab” | Model-welfare pre-deployment evaluations |
| Talent pipeline | “MATS -> labs” | MATS graduates 流向 Anthropic、DM、OpenAI、Redwood、Apollo、METR |
| External evaluation | “non-lab check” | 不是由模型生产方完成的 evaluation；增加可信度 |

## 延伸阅读

- [MATS (ML Alignment & Theory Scholars)](https://www.matsprogram.org/) — mentorship program
- [Redwood Research](https://www.redwoodresearch.org/) — AI Control papers
- [Apollo Research](https://www.apolloresearch.ai/) — scheming evaluations
- [METR — Common Elements of Frontier AI Safety Policies](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — framework comparison
- [Eleos AI Research](https://www.eleosai.org/research) — model welfare methodology
