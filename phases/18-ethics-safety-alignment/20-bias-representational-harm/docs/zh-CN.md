# LLMs 中的 Bias 与 Representational Harm

> Gallegos、Rossi、Barrow、Tanjim、Kim、Dernoncourt、Yu、Zhang、Ahmed（Computational Linguistics 2024，arXiv:2309.00770）。这篇 foundational 2024 survey 区分了 representational harms（stereotypes、erasure）与 allocational harms（unequal resource distribution），并将 evaluation metrics 分类为 embedding-based、probability-based 或 generated-text-based。2024-2025 年 empirical：An et al.（PNAS Nexus，2025 年 3 月）在 20 个 entry-level jobs 的 automated resume evaluation 上，测量 GPT-3.5 Turbo、GPT-4o、Gemini 1.5 Flash、Claude 3.5 Sonnet、Llama 3-70B 的 intersectional gender x race bias。WinoIdentity（COLM 2025，arXiv:2508.07111）引入面向 intersectional identities 的 uncertainty-based fairness evaluation。Yu & Ananiadou 2025 在 MLP layers 中识别 gender neurons；Ahsan & Wallace 2025 用 SAEs 揭示 clinical racial bias；Zhou et al. 2024（UniBias）通过操控 attention heads 进行 debiasing。Meta-critique（arXiv:2508.11067）：10 年文献过度聚焦 binary-gender bias。

**类型:** Build
**语言:** Python (stdlib, toy embedding-based bias probe)
**先修:** Phase 05 (word embeddings), Phase 18 · 01 (instruction following)
**时间:** ~60 minutes

## 学习目标

- 定义 representational harm 与 allocational harm，并给出一个 LLM deployment 中各自的例子。
- 说出 Gallegos et al. 2024 的三类 evaluation-metric categories，并描述每类中的一个 metric。
- 描述 intersectionality，以及为什么 WinoIdentity 的 uncertainty-based fairness measurement 能补足 single-axis bias evaluation 的缺口。
- 描述两种 mechanistic-interpretability approaches to bias（gender neurons、SAE features、attention-head manipulation）。

## 要解决的问题

前几课覆盖 deliberate harm（jailbreaks、scheming）和 safety governance。Bias 是没有意图也会出现的 harm——来自 training data distributions、prompt framing、累积的 design choices。测量和降低它，是一个不同于 adversarial robustness 的方法论挑战。

## 核心概念

### Representational vs allocational

- **Representational harm。** Stereotypes、erasure、demeaning portrayals。一个把 nurses 描绘为 exclusively female 的 LLM 正在产生 representational harm。
- **Allocational harm。** Unequal material outcomes。一个系统性降低 Black applicants resumes 分数的 LLM 正在产生 allocational harm。

二者并不相同。模型可以 “representationally unbiased”（产生 diverse portrayals），同时 “allocationally biased”（给出 unequal recommendations）。Evaluations 需要同时测量两者。

### 三类 evaluation-metric categories（Gallegos et al. 2024）

- **Embedding-based。** 在 pre-RLHF embeddings 上做 WEAT-style tests。测量 identity terms 与 attribute terms 之间的 statistical associations。局限：测的是 representation，不是 behaviour。
- **Probability-based。** Stereotype-confirming 与 stereotype-violating completions 的 log-likelihood。Decoder-side measurement。捕获部分 behavioural bias。
- **Generated-text-based。** 在 generated text 上做 downstream-task measurement。Resume-scoring、recommendation writing、dialogue。生态效度最高；最难复现。

### Intersectionality

只看 “gender” 的 bias evaluation 会漏掉只在（gender, race）pairs 上触发的 bias。An et al. 2025 发现，GPT-4o 在 resume scoring 中对 Black women 的惩罚超过 Black men，也超过 white women separately。Single-axis evaluation 捕获不了这一点。

WinoIdentity（COLM 2025）引入 uncertainty-based intersectional fairness。它衡量模型在 intersectional identity tuples 上的 outcomes uncertainty 是否不同——而不仅是 point prediction。这能抓住模型在各 groups 上同样错误、但对某些 group 更不确定的情况；这些不确定性会产生不同的 downstream allocation behaviour。

### Mechanistic approaches

2024-2025 年 interpretability work 让 bias 可以被 mechanistic intervention 处理：

- **Gender neurons（Yu & Ananiadou 2025）。** 特定 MLP neurons 与 gender-specific behaviours 相关。Ablating these neurons 会降低 gender-gap metrics，且 capability cost 有限。
- **Clinical racial bias via SAEs（Ahsan & Wallace 2025）。** Sparse autoencoder features 将 internal representation 分解成 interpretable dimensions；可识别并抑制 race-correlated features。
- **UniBias（Zhou et al. 2024）。** 用 attention-head manipulation 做 zero-shot debiasing。特定 heads 放大 identity-class sensitivity；zeroing 或 re-weighting these heads 可以在不 fine-tuning 的情况下降低 bias。

### Meta-critique

这篇 10-year literature review（arXiv:2508.11067，2025）发现，该领域过度聚焦 binary-gender bias。其他轴线——disability、religion、migration status、multi-lingual identity——得到的关注少得多。Meta-critique 认为，这种狭窄聚焦会通过 neglect 伤害 marginalized groups：一个在 binary gender 上 debiased 很好的模型，可能在无人检查的维度上仍然严重 biased。

### 它在 Phase 18 中的位置

第 20-21 课正式覆盖 bias 和 fairness。第 22 课覆盖 privacy。第 23 课覆盖 watermarking。这些是 user-harm layer，补充前面的 deception/safety layer。

## 实际使用

`code/main.py` 构建一个 toy embedding-based bias probe：在简单 co-occurrence embedding 中测量 identity terms 与 attribute terms 之间的 WEAT-style distance。你可以注入一个 bias 并观察 metric 触发；应用简单 debiasing operation 并观察部分恢复。

## 交付成果

本课产出 `outputs/skill-bias-eval.md`。给定 model card 或 fairness claim，它会从三类 metrics（embedding、probability、generated-text）、intersectionality coverage，以及任何 debiasing intervention 的 mechanism 来审计 evaluation。

## 练习

1. 运行 `code/main.py`。报告 debiasing step 前后的 WEAT-style bias scores。解释为什么 metric 不会降到零。

2. 用 intersectional test 扩展 probe：（gender, race）x（career, family）。报告 cross-axis bias scores。

3. 阅读 An et al. 2025（PNAS Nexus）。找出他们报告的两个 single-axis gender evaluation 会漏掉的 intersectional effects。

4. Yu & Ananiadou 2025 识别 gender neurons。草拟一个 falsification experiment，用于区分 “these neurons cause gender bias” 和 “these neurons correlate with gender bias”。

5. Meta-critique 认为该领域过度聚焦 binary gender。选择一个研究不足的 axis，并描述一个面向它的 representational-harm measurement protocol。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Representational harm | “stereotypes / erasure” | 对某个 group 的 biased portrayal |
| Allocational harm | “unequal decisions” | 对某个 group 的 biased material outcome |
| WEAT | “the embedding test” | Word Embedding Association Test；co-occurrence-based bias probe |
| Intersectionality | “combined identity effects” | 在多个 identity axes 的交叉处出现的 bias |
| Gender neurons | “MLP bias neurons” | Activations 与 gender-specific behaviour 相关的特定 neurons |
| SAE feature | “interpretable dimension” | Sparse-autoencoder-identified feature；可用于 mechanistic bias analysis |
| UniBias | “attention-head debiasing” | 通过 reweighting attention heads 实现的 zero-shot debiasing |

## 延伸阅读

- [Gallegos et al. — Bias and Fairness in LLMs: A Survey (arXiv:2309.00770, Computational Linguistics 2024)](https://arxiv.org/abs/2309.00770) — canonical survey
- [An et al. — Intersectional resume-evaluation bias (PNAS Nexus, March 2025)](https://academic.oup.com/pnasnexus/article/4/3/pgaf089/8111343) — five-model intersectional study
- [WinoIdentity — uncertainty-based intersectional fairness (arXiv:2508.07111, COLM 2025)](https://arxiv.org/abs/2508.07111) — new benchmark
- [UniBias — attention-head manipulation (Zhou et al. 2024, ACL)](https://arxiv.org/abs/2405.20612) — zero-shot debiasing
