# LLM 功能的 A/B Testing — GrowthBook、Statsig 与 Vibes 问题

> 传统 A/B testing 不是为非确定性的 LLM 构建的。关键区别是：evals 回答“模型能完成任务吗？”A/B tests 回答“用户在意吗？”两者都需要；凭 vibe checks 发版的时代结束了。2026 年要测试什么：prompt engineering（措辞）、model selection（GPT-4 vs GPT-3.5 vs OSS；accuracy vs cost vs latency）、generation parameters（temperature、top-p）。真实案例：一个 chatbot reward-model 变体带来 +70% conversation length 和 +30% retention；Nextdoor AI subject-line 实验在 reward-function refinement 后带来 +1% CTR；Khan Academy Khanmigo 围绕 latency-vs-math-accuracy 轴线迭代。平台分野：**Statsig**（2025 年 9 月被 OpenAI 以 $1.1B 收购）— sequential testing、CUPED、一体化。**GrowthBook** — open-source、warehouse-native、Bayesian + Frequentist + Sequential engines、CUPED、SRM checks、Benjamini-Hochberg + Bonferroni corrections。你的选择取决于 warehouse-SQL 偏好，以及“被 OpenAI 收购”这件事对组织是否重要。

**类型:** 学习
**语言:** Python（stdlib，玩具 sequential test simulator）
**先修:** Phase 17 · 13（Observability），Phase 17 · 20（Progressive Deployment）
**时间:** ~60 分钟

## 学习目标

- 区分 evals（“模型能完成任务吗”）与 A/B tests（“用户在意吗”）。
- 枚举三个可测试轴线（prompt、model、parameters），并为每个轴线选择指标。
- 解释 CUPED、sequential testing 与 Benjamini-Hochberg multiple-comparison corrections。
- 基于 warehouse-SQL 姿态和公司收购立场选择 Statsig 或 GrowthBook。

## 要解决的问题

你手工调了一个 system prompt。感觉更好了。你发版。Conversion 的变化只是噪声。你怪指标。或者你发布了一个新模型，而 conversion 没动 —— 模型退化了吗，还是变化太小检测不到？你不知道，因为你没有做 A/B 就发了。

Evals 回答模型在带标签集合上是否能完成任务。它们不回答用户是否更喜欢输出。只有受控在线实验能回答这个问题，而且前提是实验有足够 power、能控制非确定性，并对 multiple comparisons 做修正。

## 核心概念

### Evals vs A/B tests

**Evals** — offline、带标签集合、judge（rubric、LLM-as-judge 或 human）。回答：“在这个固定分布上，输出是否 correct / helpful / safe？”

**A/B test** — online、真实用户、随机分流。回答：“新变体是否推动了重要的 user-level metric？”

两者都需要。Evals 在曝光前捕获 regression；A/B 在曝光后确认产品影响。

### 测什么

1. **Prompt engineering** — 措辞、system-prompt 结构、examples。指标：task success、user retention、cost/request。
2. **Model selection** — GPT-4 vs GPT-3.5-Turbo vs Llama-OSS。指标：accuracy（task）+ cost/request + latency P99。多目标。
3. **Generation parameters** — temperature、top-p、max_tokens。指标：task-specific（output diversity vs determinism）。

### CUPED — 方差降低

Controlled-experiments Using Pre-Experiment Data。在比较 post-period 之前，先回归掉 pre-period variance。典型方差降低：30-70%。Effective sample size 免费提升。

实现：Statsig 和 GrowthBook 都实现了。

### Sequential testing

经典 A/B 假设固定 sample size。Sequential tests（“peek-and-decide”）在重复查看结果时控制 false-positive rate。Always-valid sequential procedures（mSPRT、Howard's confidence sequences）允许你在赢家明显时提前停止。

### Multiple-comparison corrections

以 95% confidence 同时跑 20 个 A/B tests，纯靠运气也会产生一个 false positive。Bonferroni correction 收紧每个 test 的 α；Benjamini-Hochberg 控制 false-discovery rate。GrowthBook 两者都实现。

### SRM — sample ratio mismatch

Assignment hash 将用户随机分到各变体。如果 50/50 split 实际交付成 47/53，就说明某处坏了 —— SRM check 会标记它。两个平台都实现了。

### Statsig vs GrowthBook

**Statsig**:
- 被 OpenAI 以 $1.1B 收购（2025 年 9 月）。Hosted、SaaS。
- Sequential testing、CUPED、held-out populations。
- 一体化：feature flags + experimentation + observability。
- 最适合：团队已经想要 bundled product，并且不在意 OpenAI ownership。

**GrowthBook**:
- Open-source（MIT）；warehouse-native（直接读取 Snowflake/BigQuery/Redshift）。
- 多种 engines：Bayesian、Frequentist、Sequential。
- CUPED、SRM、Bonferroni、BH corrections。
- Self-host 或 managed cloud。
- 最适合：warehouse-SQL shop，data team 控制 metric layer，并且想要 OSS。

### 非确定性让 power 更复杂

同一个 prompt 会产生不同输出。传统 power calculations 假设 IID observations。面对 LLM non-determinism，effective sample size 低于名义值。把所需 sample size 乘以约 1.3-1.5x 作为安全余量。

### 真实案例结果

- Chatbot reward model 变体：+70% conversation length，+30% retention。
- Nextdoor subject lines：reward-function refinement 后 +1% CTR。
- Khan Academy Khanmigo：围绕 latency-vs-math-accuracy trade 迭代。

### 反模式：凭 vibes 发版

每个资深工程师都能说出某个功能，因为“感觉更好”而没做 A/B 就发了。它们多数让团队几个月都没注意到的产品指标倒退了。A/B 是那个强制校准机制。

### 你应该记住的数字

- Statsig 被 OpenAI 收购：$1.1B，2025 年 9 月。
- GrowthBook：open-source MIT；Bayesian + Frequentist + Sequential。
- CUPED variance reduction：30-70%。
- LLM non-determinism → +30-50% sample-size buffer。

## 实际使用

`code/main.py` 模拟带 fixed 和 sequential boundaries 的 sequential A/B test。展示 sequential 如何让你提前停止。

## 交付成果

本课产出 `outputs/skill-ab-plan.md`。给定 feature change、workload、baseline，它会选择平台、gates 和 sample size。

## 练习

1. 运行 `code/main.py`。在 baseline 3% conversion、预期 5% lift 时，要达到 80% power 需要多大 sample size？
2. 为一个 healthcare-regulated on-prem 客户选择 Statsig 还是 GrowthBook。
3. 设计一个测试 GPT-4 vs GPT-3.5 的 A/B，指标是 cost-per-resolved-ticket。Primary metric、guardrail metric、secondary metric 分别是什么？
4. 你的 canary 通过了，但 A/B 显示 -1.2% conversion。你会发版吗？写下 escalation criteria。
5. 对一个 pre-period 含有 post 60% variance 的场景应用 CUPED。计算 effective-sample-size boost。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|------------|----------|
| Eval | “offline test” | 对 model capability 的 labeled-set evaluation |
| A/B test | “experiment” | 对用户进行 live randomized comparison |
| CUPED | “variance reduction” | 用 pre-period regression 降低 variance |
| Sequential test | “peek-ok test” | 允许 early stop 的 always-valid procedure |
| Multiple comparison | “the family error” | 同时跑很多 tests 会放大 false positives |
| Bonferroni | “tight correction” | 用 tests 数量划分 α |
| Benjamini-Hochberg | “BH FDR” | False-discovery-rate control，更不保守 |
| SRM | “bad split” | Sample ratio mismatch；assignment bug |
| Statsig | “OpenAI owned” | Commercial all-in-one，2025 年被收购 |
| GrowthBook | “the OSS one” | MIT warehouse-native platform |
| mSPRT | “sequential probability ratio test” | Classical sequential procedure |

## 延伸阅读

- [GrowthBook — How to A/B Test AI](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/)
- [Statsig — Beyond Prompts: Data-Driven LLM Optimization](https://www.statsig.com/blog/llm-optimization-online-experimentation)
- [Statsig vs GrowthBook comparison](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools)
- [Deng et al. — CUPED](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Howard — Confidence Sequences](https://arxiv.org/abs/1810.08240)
