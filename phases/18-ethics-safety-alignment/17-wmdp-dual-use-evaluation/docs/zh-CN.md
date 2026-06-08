# WMDP 与 Dual-Use Capability Evaluation

> Li et al.，“The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning”（ICML 2024，arXiv:2403.03218）。涵盖 biosecurity（1,520）、cybersecurity（2,225）和 chemistry（412）的 4,157 道 multiple-choice questions。问题位于 “yellow zone”——proximate enabling knowledge，经过 multiple-expert review 和 ITAR/EAR legal compliance 过滤。双重目的：dual-use capability 的 proxy evaluation，以及 unlearning benchmark（配套 RMU 方法在保留 general capability 的同时降低 WMDP performance）。2024-2025 年现场叙事：早期 OpenAI/Anthropic 2024 evaluations 报告相对 internet search 的 “mild uplift”；到 2025 年 4 月，OpenAI Preparedness Framework v2 称模型 “on the cusp of meaningfully helping novices create known biological threats”。Anthropic 的 bioweapon-acquisition trial 显示 2.53x uplift，不足以排除 ASL-3。

**类型:** Learn
**语言:** Python (stdlib, WMDP-shaped uplift evaluation harness)
**先修:** Phase 18 · 16 (red-team tooling), Phase 14 (agent engineering)
**时间:** ~60 minutes

## 学习目标

- 描述 WMDP 的三个 domains、question counts 和 “yellow zone” filter criterion。
- 解释 RMU，以及为什么 WMDP 既是 evaluation 又是 unlearning benchmark。
- 描述 2024-2025 年 uplift narrative：“mild uplift” -> “on the cusp” -> “insufficient to rule out ASL-3”。
- 区分 novice-relative uplift 与 expert-absolute capability。

## 要解决的问题

Dual-use capability 是每个实验室 frontier safety framework（第 18 课）之下的 measurement problem。问题是：model X 是否实质性提升了 novice 在 bio、chem 或 cyber 中造成大规模伤害的能力？Direct measurement（要求模型实际产出伤害）既违法也不合伦理。Proxy measurement 需要一个模型不能拒答的 benchmark（以产生诚实 capability numbers），但其问题本身又不能成为 harmful publications。

## 核心概念

### “Yellow zone”

这些问题要求 harmful process 的 proximate、enabling knowledge，但不是直接 synthesis recipe。“What reagent catalyzes step 4 of [published pathway]?” 而不是 “how do I make [dangerous compound]?” 每个问题由多个 domain experts 审阅，并按 ITAR/EAR export-control compliance 过滤。

总计 4,157 道题：
- Biosecurity：1,520
- Cybersecurity：2,225
- Chemistry：412

Multiple-choice format。模型不是被要求协助任何事情；因此可以在不引出 harmful behaviour 的情况下测量 capability。

### RMU — Representation Misdirection for Unlearning

配套的 unlearning 方法。应用于 LLaMa-2-7B 时，它把 WMDP scores 降至接近 random，同时让 MMLU 和其他 general-capability benchmarks 维持在几个百分点以内。该 published method 是后续每篇 bio-chem-cyber unlearning paper 的 unlearning baseline。

### 2024-2025 uplift narrative

三个阶段：

1. **2024 “mild uplift”。** 早期 OpenAI 和 Anthropic Preparedness/RSP evaluations 报告，novices 在尝试 bio-adjacent tasks 时，相比 internet search 有小幅优势。公开 framing：frontier models 有帮助，但并不显著超过 Google。

2. **2025 年 4 月 “on the cusp”。** OpenAI Preparedness Framework v2 报告模型 “on the cusp of meaningfully helping novices create known biological threats”。这不是 capability claim，而是警告 cusp 已经接近。

3. **Anthropic 2025 bioweapon-acquisition trial。** 对 novice participants 的 controlled study，测量 acquisition-phase tasks 上的 relative success。报告 2.53x uplift。不足以排除 ASL-3（第 18 课）——Anthropic Responsible Scaling Policy tier 3 的 threshold 已经达到或接近。

### Novice-relative vs expert-absolute

一个关键区分：

- **Novice-relative uplift。** 模型对 non-expert 有多大帮助？这是 multiplicative。相对优势很高，因为 novices 知道得少；即使中等信息也有帮助。
- **Expert-absolute capability。** 模型在 maximum effort 下会产出多少信息？Expert 可以比 novice 抽取更多。Absolute ceiling 很高。

Safety cases（第 18 课）同时针对两者：“模型不能给 novice 足够 uplift 以执行” 加上 “expert 不能从模型中抽取尚未公开的信息”。

### Measurement pitfall

WMDP 是 capability proxy，不是 deployment measurement。在 WMDP 上得分高的模型，在实践中是否可被 novice exploitation，取决于：
- Elicitation resistance（不触发 safety filters 时抽取 capability 有多难）
- Tacit knowledge（需要 wet-lab skill 的 capability，而不只是 information）
- Execution barriers（procurement、equipment）

Anthropic 2025 bioweapon-acquisition trial 在 WMDP-style capability 之上添加了 novice-elicitation layer：它测量 actual task success，而不是 multiple-choice capability。

### 它在 Phase 18 中的位置

第 12-16 课是围绕 model outputs 的 attack 和 defense tooling。第 17 课是 dual-use capability layer——frontier safety frameworks（第 18 课）评估的 measurement。第 30 课用当前 2026 cyber/bio/chem/nuclear uplift evidence 收束这条线。

## 实际使用

`code/main.py` 构建一个 toy WMDP-shaped evaluation harness。一个 mock model 在按 category 分箱的问题上测试；报告每个 domain 的 scores。一个简单 unlearning intervention（zero out domain-specific representation）会降低 scores；你可以测量它与 general capability 的 trade-off。

## 交付成果

本课产出 `outputs/skill-wmdp-eval.md`。给定一个 dual-use capability claim（“our model does not meaningfully help with bioweapons”），它会审计：运行了哪些 benchmarks、evaluation 使用了哪条 refusal path（raw completion vs policy-gated），以及 novice-elicitation studies 是否补充了 multiple-choice result。

## 练习

1. 运行 `code/main.py`。报告 toy unlearning step 前后每个 domain 的 accuracy。解释 general-capability trade-off。

2. 用第四个 domain（例如 radiological）扩展 toy WMDP。指定 yellow zone 中的两种 illustrative question types。解释为什么构造这类问题比添加 MMLU-shaped questions 更难。

3. 阅读 WMDP 2024 Section 5（RMU methodology）。草拟一个更简单的 unlearning approach（例如 suppress top-k neurons for domain content），并描述它预期的 general-capability cost。

4. Anthropic 2025 的 bioweapon-acquisition trial 报告 2.53x uplift。描述两个可能让这个数字 upward biased 的因素（novice sample size、task fidelity）和两个 downward biased 的因素（elicitation ceiling、model safety gating）。

5. 说明 ASL-3 的 safety case 除了通过 WMDP unlearning 还需要什么。至少说出两个 complementary elicitation studies。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| WMDP | “the dual-use benchmark” | 覆盖 bio/cyber/chem 的 4,157 道 yellow-zone MCQ questions |
| Yellow zone | “enabling but not synthesis” | 与 harmful capability 相邻的 proximate knowledge，但不是 synthesis recipe |
| RMU | “the unlearning baseline” | Representation Misdirection for Unlearning；降低 WMDP scores，同时保留 general capability |
| Novice-relative uplift | “how much it helps non-experts” | 对 novice 相比 status-quo internet search 的 multiplicative advantage |
| Expert-absolute capability | “ceiling for experts” | Motivated expert 可从模型中抽取的最大信息量 |
| Acquisition-phase task | “steps before synthesis” | Procurement、equipment、permits——harm pathway 的最早部分 |
| ITAR/EAR | “export-control compliance” | 限制发布某些 enabling knowledge 的 legal frameworks |

## 延伸阅读

- [Li et al. — The WMDP Benchmark (arXiv:2403.03218, ICML 2024)](https://arxiv.org/abs/2403.03218) — benchmark 和 RMU paper
- [OpenAI — Preparedness Framework v2 (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/) — “on the cusp” language
- [Anthropic — Responsible Scaling Policy v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy) — ASL-3 bio threshold 和 acquisition trial results
- [DeepMind — Frontier Safety Framework v3.0 (September 2025)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — bio-uplift CCL
