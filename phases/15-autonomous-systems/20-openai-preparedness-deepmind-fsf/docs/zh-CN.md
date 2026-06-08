# OpenAI Preparedness Framework 与 DeepMind Frontier Safety Framework

> OpenAI Preparedness Framework v2（2025 年 4 月）引入 Research Categories：Long-range Autonomy、Sandbagging、Autonomous Replication and Adaptation、Undermining Safeguards，它们不同于 Tracked Categories。Tracked Categories 会触发 Capabilities Reports 加 Safeguards Reports，并由 Safety Advisory Group review。DeepMind 的 FSF v3（2025 年 9 月，2026 年 4 月 17 日新增 Tracked Capability Levels）把 autonomy 折叠进 ML R&D 和 Cyber domains（ML R&D autonomy level 1 = 以 competitive cost 相比 human + AI tools 完全自动化 AI R&D pipeline）。FSF v3 通过 automated monitoring for instrumental-reasoning misuse 明确处理 deceptive alignment。诚实提示：PF v2 中的 Research Categories（包括 Long-range Autonomy）不会自动触发 mitigations；policy language 是 “potential”。DeepMind 自己说，如果 instrumental reasoning 变强，automated monitoring “will not remain sufficient long-term”。

**类型：** 学习
**语言：** Python (stdlib, three-framework decision-table diff tool)
**先修：** Phase 15 · 19 (Anthropic RSP)
**时间：** ~45 分钟

## 要解决的问题

Lesson 19 仔细阅读了 Anthropic 的 scaling policy。本课通过阅读 OpenAI 和 DeepMind 的 policy 来补完整张图。这三份文档是 cousin artifacts，都在回答同一个问题：frontier lab 应该在什么时候 pause 或 gate 一个 model。它们收敛到一小组 categories，也在一些重要的具体点上分歧。

收敛处：三者都把 long-range autonomy 标记为值得追踪的 capability class。三者都承认 deceptive behavior（alignment faking、sandbagging）是具体的 risk class。三者都有 internal review body。分歧处：OpenAI 把 categories 分成 “Tracked”（mandatory mitigation）和 “Research”（没有 automatic trigger）。DeepMind 把 autonomy 折叠进两个 domains，而不是单独命名。lab 命名为 Tracked vs Research、Critical vs Moderate、Tier-1 vs Tier-2；一个 capability 落在哪个 bucket，在不同 labs 中有不同 operational consequence。

把它们放在一起阅读才是有用练习。同一种 capability，在 Anthropic 可能是 “mandatory mitigation”，在 OpenAI 可能是 “monitored but not triggering”，在 DeepMind 可能是 “tracked in a specific domain”。Policy posture 很重要。

## 核心概念

### OpenAI Preparedness Framework v2（2025 年 4 月）

结构：

- **Tracked Categories**：触发 Capabilities Reports（模型能做什么）加 Safeguards Reports（有哪些 mitigations）。部署前由 Safety Advisory Group review。
- **Research Categories**：lab 正在追踪、但尚未承诺 specific mitigations 的 model capabilities。包括 Long-range Autonomy、Sandbagging、Autonomous Replication and Adaptation、Undermining Safeguards。

v2 中的关键 framing：Research Categories 不会自动触发 mitigations。policy language 是 “potential” mitigations。这个 delta 要特别留意：一个 category 位于 Tracked 还是 Research，是 “gated” 和 “observed” 之间的铰链。

### DeepMind Frontier Safety Framework v3（2025 年 9 月；2026 年 4 月 17 日新增 Tracked Capability Levels）

结构：

- **Critical Capability Levels (CCLs)**：跨五个 domains 的 capability thresholds：Cyber、Bio、ML R&D、CBRN、Autonomy（折叠进 ML R&D 和 Cyber）。
- **Tracked Capability Levels**：2026 年 4 月新增的 additional granularity。具体例子：ML R&D autonomy level 1 = 以 competitive cost 相比 human + AI tools 完全自动化 AI R&D pipeline。
- **Deceptive alignment monitoring**：明确承诺对 instrumental-reasoning misuse 做 automated monitoring。

autonomy framing 与 OpenAI 不同。DeepMind 没有把 “Autonomy” 保持为 top-level domain；它被折叠进 autonomy 会造成伤害的 domains（ML R&D 和 Cyber）。支持这一点的 argument 是：没有 domain 的 autonomy 是 capability without risk；反方 argument 是：跨 domains 的 autonomy 是一个 framework 应该命名的 meta-risk。

### 三者共同收敛在哪里

- Internal Safety Advisory Group（命名为 Anthropic SAG、OpenAI SAG、DeepMind internal committee）。高 capability models 部署前 review。
- 明确提到 deceptive alignment / alignment faking 作为 risk class。
- 有声明 cadence 的 standing artifacts（Anthropic: Frontier Safety Roadmap, Risk Report；OpenAI: Capabilities and Safeguards Reports；DeepMind: FSF update cycle）。
- 承认 monitoring-only defenses 有 ceiling。DeepMind 说得很明确：“automated monitoring will not remain sufficient long-term.”

### 它们在哪里分歧

- **Anthropic**：v3.0 移除了 pause commitment；AI R&D-4 threshold 是命名的 next gate。
- **OpenAI**：Tracked vs Research split；Research Categories（包括 Long-range Autonomy）不会自动 gate。
- **DeepMind**：autonomy 折叠进其他 domains；Tracked Capability Levels 在 2026 年 4 月增加 granularity。

### Sandbagging：让三者都复杂化的具体 capability

Sandbagging（模型在 evaluations 上战略性低表现）位于 OpenAI 的 Research Categories。Anthropic 的 RSP v3.0 通过 evaluation-context gap（Lesson 1）处理它。DeepMind 通过 FSF v3 中的 deceptive alignment monitoring 处理它。

如果模型在 evaluations 上 sandbag，每个 framework 的 capability thresholds 都会被低估。framework 只有在 measurement 有效时才有效。这就是为什么除了 lab self-assessment 之外，还需要 external measurement（Lesson 21，METR）和 adversarial evaluation。

### Policy-reading skill

- Locate：你关心的每个 capability 都应该能在 policy 中找到。找不到，说明 policy 没覆盖它。
- Classify：它是 Tracked（触发 mitigation）还是 Research（被追踪但不触发）？OpenAI 这样命名；Anthropic 和 DeepMind 有自己的等价物。
- Cadence：policy 是按声明 schedule 更新，还是只在 specific events 后更新？Declared cadence 更强。
- Independence：external review 是 mandatory 还是 optional？Anthropic 与 Apollo 和 US AI Safety Institute 合作；OpenAI 与 METR 合作；DeepMind 主要依赖 internal SAG。

## 实际使用

`code/main.py` 实现一个小型 decision-table diff tool。给定一个 capability（autonomy、deceptive alignment、R&D automation、cyber uplift 等），它会输出三份 policies 各自如何 classify 该 capability，以及会触发什么 mitigations。它是 reading aid，不是 policy tool。

## 交付成果

`outputs/skill-cross-policy-diff.md` 会为某个 specific capability 生成 cross-policy comparison，以三份 frameworks 作为 reference。

## 练习

1. 运行 `code/main.py`。至少选择两个你能对照 source documents 验证的 capabilities，确认 diff tool 输出与 policies 匹配。

2. 完整阅读 OpenAI Preparedness Framework v2。识别每一个 Research Category。对每一个写一句话，解释为什么它位于 Research 而不是 Tracked。

3. 完整阅读 DeepMind FSF v3，加上 2026 年 4 月 Tracked Capability Levels update。识别 ML R&D autonomy level 1 的具体 evaluation criteria。你会如何从外部测量它？

4. Sandbagging 位于 OpenAI 的 Research Categories。设计一个 evaluation，迫使 sandbagging model 揭示真实 capability。引用 Lesson 1 的 eval-context-gaming 讨论。

5. 针对一个 specific capability（自选）对比三份 policies。说出你认为哪个 policy 的 classification 最 rigorous，哪个最弱。用 source text 辩护。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|---|---|---|
| Preparedness Framework | “OpenAI's scaling policy” | PF v2（2025 年 4 月）；Tracked vs Research categories |
| Tracked Category | “Mandatory mitigation” | 触发 Capabilities + Safeguards Reports；SAG review |
| Research Category | “Monitored only” | 被追踪但没有 automatic mitigation；包括 Long-range Autonomy |
| Frontier Safety Framework | “DeepMind's scaling policy” | FSF v3（2025 年 9 月）+ Tracked Capability Levels（2026 年 4 月） |
| CCL | “Critical Capability Level” | DeepMind 每个 domain 的 threshold（Cyber、Bio、ML R&D、CBRN） |
| ML R&D autonomy level 1 | “R&D automation” | 以 competitive cost 完全自动化 AI R&D pipeline |
| Sandbagging | “Strategic underperformance” | 模型在 evals 上低表现；位于 OpenAI Research Categories |
| Instrumental reasoning | “Means-ends reasoning” | 关于如何达成 goals 的 reasoning；DeepMind monitoring 的目标 |

## 延伸阅读

- [OpenAI — Updating our Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/) — v2 announcement。
- [OpenAI — Preparedness Framework v2 PDF](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) — 完整文档。
- [DeepMind — Strengthening our Frontier Safety Framework](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — FSF v3 announcement。
- [DeepMind — Updating the Frontier Safety Framework (April 2026)](https://deepmind.google/blog/updating-the-frontier-safety-framework/) — Tracked Capability Levels addition。
- [Gemini 3 Pro FSF Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf) — FSF-format Risk Report 示例。
