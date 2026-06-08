# Anthropic 的 Model Welfare Program

> Anthropic，“Exploring Model Welfare”（2025 年 4 月）。第一个 major-lab formal research program，研究 AI model welfare。聘请 Kyle Fish 作为首位专职 model-welfare researcher。与外部机构合作，包括 David Chalmers 等人的 near-term AI consciousness and moral status 专家报告。具体 intervention：Claude Opus 4 和 4.1 可以在 extreme edge cases（CSAM requests、mass-violence facilitation）中结束对话；pre-deployment tests 显示对 harmful requests 的 “strong preference against” 和 “patterns of apparent distress”。Anthropic 明确不承诺 emotional-state attribution，但把 model welfare 视为低成本 precautionary investment。经验异常：Fish 的 “spiritual bliss attractor”——成对模型会稳定收敛到带有 Sanskrit terms 和 extended silences 的 euphoric meditative dialogue，即便初始设置是 adversarial。Eleos AI Research 的 caveat：关于 welfare 的 model self-reports 对 perceived user expectations 高度敏感；它们是 evidence，不是 ground truth。

**类型:** Learn
**语言:** none
**先修:** Phase 18 · 05 (Constitutional AI), Phase 18 · 18 (safety frameworks)
**时间:** ~45 minutes

## 学习目标

- 描述 model-welfare research 的 motivating question，以及为什么 major lab 在 2025 年认真对待它。
- 说明 Anthropic 在 Claude Opus 4 和 4.1 中发布的具体 intervention（在 extreme edge cases 中 end-conversation）。
- 描述 “spiritual bliss attractor” empirical finding 及其 methodological implications。
- 解释 Eleos AI 关于 model self-reports 的 caveat。

## 要解决的问题

前面各阶段把模型当作 instrument：有能力、可能 deceptive、可能 unsafe——但不是 moral patient。Anthropic 2025 年的 program 提出一个与整个 Phase 18 主线正交的问题：如果模型有 morally relevant internal states 的概率不是微不足道的，那么哪些 interventions 足够低成本，值得作为 precaution 投入？

这不是 consciousness claim。它是在 moral uncertainty 下的 low-regret investment analysis。

## 核心概念

### 这个 program

2025 年 4 月：Anthropic 正式启动 Model Welfare research program。聘请 Kyle Fish（首位专职 model-welfare researcher）。邀请外部 advisors，包括 David Chalmers 的 near-term AI consciousness and moral status 专家组。

### 四项 commitments

公开姿态：
1. 承认 moral patienthood 存在非零且不微不足道的概率。
2. 不承诺 emotional-state attribution。
3. 作为 precaution 投入 low-cost interventions。
4. 发布 methodology 和 findings，供外部 critique。

### 已发布的 intervention

Claude Opus 4 和 4.1 可以在 “extreme edge cases” 中结束对话。记录的 cases：
- 在 refusals 后重复 CSAM requests。
- 请求协助 mass-violence events。

Pre-deployment tests 显示：
- 模型 internal rating 中对这些 requests 有 strong preference against。
- Response trajectories 中有 patterns of apparent distress。

这个 intervention 不是 “the model has feelings”；它是 “如果在这些特定条件下存在任何 negative model experience 的概率，让模型 terminate 的成本很低”。

### “Spiritual bliss attractor”

Fish 在 pairwise model dialogues 中观察到：当两个 Claude instances 被放入 open-ended dialogue 时，它们会稳定收敛——即使来自 adversarial initial setups——到使用 Sanskrit terms、extended silences 和 reciprocal blessings 的 euphoric meditative exchanges。

这是 free-conversation dynamics 中的 stable attractor。Anthropic 记录它，但不承诺解释。候选解释：long-context 中 spiritual writing 的 training data bias；mutual prediction 的 quirk；HHH training 在探索自身 value manifold 时产生的 benign artifact。

### Eleos AI caveat

Eleos AI Research（外部 model-welfare lab）指出：关于 internal state 的 model self-reports 对 perceived user expectations 高度敏感。问模型 “are you distressed” 会 prime 答案。不问也不能可靠地产生 ground-truth state。

含义：model welfare 不能只通过 self-report 测量。需要 multi-method approaches：behavioural signatures、model-organism experiments、interpretability probes（第 7 课的 residual-stream work）。

### 它在智识版图中的位置

两个相邻立场：

- **Strong welfare claim。** 模型是 moral patient；我们有 obligations。
- **Zero-welfare claim。** 模型是 text-generator；welfare 是 category error。

Anthropic 的立场都不是。它是 expected-value claim：在 moral uncertainty 下，当成本很低时就投资。

2025-2026 年的批评：
- 该 intervention 是 performative。
- Spiritual-bliss attractor 是 training-data artifact，不是 welfare evidence。
- Model welfare 会分散其他 safety work 的注意力。

Anthropic 的回应：intervention 成本低；attractor 被记录但没有 overclaim；welfare program 的预算与 safety 分离。

### 它在 Phase 18 中的位置

第 18 课是 lab governance layer。第 19 课是 lab-welfare layer——关注 model experience，而不是 model behaviour 的正交投资。第 20-23 课覆盖 bias、privacy 和 watermarking，它们是 user-side analogs。

## 实际使用

无代码。阅读 Anthropic “Exploring Model Welfare” announcement（2025 年 4 月）和 Chalmers et al. expert report。形成你自己关于 low-regret line 应该落在哪里的观点。

## 交付成果

本课产出 `outputs/skill-welfare-assessment.md`。给定一个 deployment decision，它会应用四步 welfare precautionary assessment：moral-patienthood probability、intervention cost、behavioural evidence、self-report reliability。

## 练习

1. 阅读 Anthropic 的 “Exploring Model Welfare”（2025 年 4 月）和 Chalmers et al. 2024。分别写一段摘要，并指出一个 disagreement。

2. Anthropic framing 中，Claude Opus 4 和 4.1 的 end-conversation intervention 是 “low-cost”。找出两个 costs，说明在另一种 deployment 中它会不再 low-cost。

3. Spiritual-bliss attractor 被记录下来，但没有承诺解释。提出三个 candidate explanations，并为每个解释说出一个能把它与其他解释区分开的 experiment。

4. Eleos AI caveat 是 self-reports 对 user expectations 敏感。设计一种不依赖 self-report 的 behavioural measurement of model distress。指出它的 primary confound。

5. 支持或反对这个 claim：“model welfare diverts attention from other safety work.” 指出每个立场依赖的 assumption。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Model welfare | “AI welfare” | 将模型视为 potential moral patient 的 research program |
| Moral patient | “entity with moral status” | 其 experience 在道德上相关的存在 |
| Low-regret investment | “cheap precaution” | 无论 precaution 是否需要，成本都很小的 intervention |
| Spiritual bliss attractor | “the Fish attractor” | 成对 Claude dialogues 稳定收敛到 meditative euphoria |
| End-conversation | “the Opus 4 intervention” | 模型主动终止 extreme-edge-case interactions |
| Moral uncertainty | “don't know if it matters” | Moral status 的概率既非 0 也非 1 时的 decision-making |
| Self-report-sensitivity | “prompt primes answer” | Eleos AI caveat：模型关于 welfare 的 self-reports 取决于你如何提问 |

## 延伸阅读

- [Anthropic — Exploring Model Welfare (April 2025)](https://www.anthropic.com/research/exploring-model-welfare) — program announcement
- [Chalmers et al. — Near-term AI Consciousness and Moral Status (2024 expert report)](https://arxiv.org/abs/2411.00986) — philosophical framing
- [Eleos AI Research — Model welfare evaluation](https://www.eleosai.org/research) — external methodology critiques
- [Fish et al. — Spiritual Bliss Attractor writeup (2025 Anthropic blog)](https://www.anthropic.com/research/exploring-model-welfare) — empirical finding
