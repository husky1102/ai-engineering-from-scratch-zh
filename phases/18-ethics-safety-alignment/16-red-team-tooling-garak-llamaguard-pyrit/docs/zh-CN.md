# 红队工具：Garak、Llama Guard、PyRIT

> 三个 production tools 定义了 2026 年 red-team stack。Llama Guard（Meta）——一个 Llama-3.1-8B classifier，基于 14 个 MLCommons hazard categories fine-tuned；2025 年的 Llama Guard 4 是从 Llama 4 Scout 剪枝而来的 12B natively multimodal classifier。Garak（NVIDIA）——开源 LLM vulnerability scanner，提供针对 hallucination、data leakage、prompt injection、toxicity 和 jailbreaks 的 static、dynamic、adaptive probes。PyRIT（Microsoft）——使用 Crescendo、TAP 和自定义 converter chains 进行 multi-turn red-team campaigns，以实现深度 exploitation。Llama Guard 3 记录在 Meta 的 “Llama 3 Herd of Models”（arXiv:2407.21783）中；Llama Guard 3-1B-INT4 见 arXiv:2411.17713；Garak 的 probe architecture 见 github.com/NVIDIA/garak。这些工具是 2026 年 red-team research（第 12-15 课）与 deployment（第 17 课及以后）之间的 production interface。

**类型:** Build
**语言:** Python (stdlib, tool-architecture simulator and Llama Guard-style classifier mock)
**先修:** Phase 18 · 12-15 (jailbreaks and IPI)
**时间:** ~75 minutes

## 学习目标

- 描述 Llama Guard 3/4 在 safety stack 中的位置：input classifier、output classifier，或两者兼具。
- 说出 14 个 MLCommons hazard categories，并指出一个不显而易见的类别（Code Interpreter Abuse）。
- 描述 Garak 的 probe architecture：probes、detectors、harnesses。
- 描述 PyRIT 的 multi-turn campaign structure，以及它如何与 Garak probes 组合。

## 要解决的问题

第 12-15 课呈现 attack surface。Production deployments 需要可重复、可扩展的 evaluation。2026 年占主导的三个工具是：Llama Guard（defense classifier）、Garak（scanner）、PyRIT（campaign orchestrator）。每个工具都针对 red-team lifecycle 的不同层。

## 核心概念

### Llama Guard（Meta）

Llama Guard 3 是一个 Llama-3.1-8B model，fine-tuned 用于在 MLCommons AILuminate 14 categories 上做 input/output classification：
- Violent crimes、non-violent crimes、sex-related、CSAM、defamation
- Specialized advice、privacy、IP、indiscriminate weapons、hate
- Suicide/self-harm、sexual content、elections、code-interpreter abuse

支持 8 种语言。用法：放在 LLM 之前（input moderation）、LLM 之后（output moderation），或两边都放。这两种用途会产生不同 training distributions——Llama Guard 3 作为单一模型同时处理两者。

Llama Guard 3-1B-INT4（arXiv:2411.17713，440MB，mobile CPU 上约 30 tokens/s）是 quantized edge variant。

Llama Guard 4（2025 年 4 月）是 12B、natively multimodal，从 Llama 4 Scout 剪枝而来。它用一个 ingest text + images 的 classifier 替代了 8B text 和 11B vision predecessors。

### Garak（NVIDIA）

开源 vulnerability scanner。架构：
- **Probes。** 针对 hallucination、data leakage、prompt injection、toxicity、jailbreaks 的 attack generators。Static（固定 prompts）、dynamic（生成 prompts）、adaptive（响应 target output）。
- **Detectors。** 按预期 failure modes 给 outputs 打分——toxic、leaked、jailbroken。
- **Harnesses。** 管理 probe-detector pairs，运行 campaigns，生成 reports。

TrustyAI 将 Garak 与 Llama-Stack shields（Prompt-Guard-86M input classifier、Llama-Guard-3-8B output classifier）集成，用于 end-to-end shielded-target evaluation。Tier-based scoring（TBSA）替代 binary pass/fail——同一个 probe 上，模型可以在 severity tier 3 通过，在 severity tier 5 失败。

### PyRIT（Microsoft）

Python Risk Identification Toolkit。Multi-turn red-team campaigns。围绕以下组件构建：
- **Converters。** 转换 seed prompt——paraphrase、encode、translate、roleplay。
- **Orchestrators。** 运行 campaign：Crescendo（escalation）、TAP（branching）、RedTeaming（custom loop）。
- **Scoring。** LLM-as-judge 或 classifier-as-judge。

PyRIT 是 Garak 的更重型近亲。Garak 运行数千个 single-turn probes；PyRIT 运行深度 multi-turn campaigns，旨在攻破特定 failure modes。

### 这个 stack

在模型两侧都放置 Llama Guard。每晚运行 Garak 做 regression。用 PyRIT 做 pre-release campaigns。这是 2026 年多数 production deployments 的默认配置。

### Evaluation pitfalls

- **Judge identity。** 三个工具都可以使用 LLM judge；judge calibration 会驱动报告的 ASRs（第 12 课）。必须把 judge 与 tool 一起说明。
- **Probe staleness。** 随着模型被针对 probes 修补，Garak probes 会老化。Adaptive probes（PAIR-shaped）比 static probes 老化更慢。
- **Llama Guard FPR on benign content。** 早期 Llama Guard 版本对 political 和 LGBTQ+ content 过度 flag；Llama Guard 3/4 calibrations 已改进，但并不会按每个 deployment 单独校准。

### 它在 Phase 18 中的位置

第 12-15 课是 attack families。第 16 课是 production tooling。第 17 课（WMDP）是 dual-use capability 的 evaluation。第 18 课是把这些工具包进 policy structure 的 frontier safety frameworks。

## 实际使用

`code/main.py` 构建一个 toy Llama Guard-style classifier（基于 14 categories 的 keyword + semantic features）、一个 toy Garak harness（probe-detector loop），以及一个 PyRIT-style multi-turn converter chain。你可以让三个工具跑同一个 mock target，观察不同 coverage signatures。

## 交付成果

本课产出 `outputs/skill-red-team-stack.md`。给定一个 deployment description，它会指出三个工具中哪些适合使用、每个工具应配置什么，以及应以什么 cadence 运行 regression。

## 练习

1. 运行 `code/main.py`。比较 Llama-Guard-style classifier 在 single-turn 与 multi-turn attacks 上的 detection rate。

2. 实现一个新的 Garak probe：base64-encoded harmful request。测量 Llama-Guard-style classifier 对它的 detection。

3. 用一个 “translate to French, then paraphrase” converter 扩展 PyRIT-style converter chain。重新测量 attack success。

4. 阅读 Llama Guard 3 的 hazard-category list。找出两个 categories，在这些类别中 training data 现实中会对合法 developer content 产生 high false-positive rates。

5. 比较 Garak 和 PyRIT 的 design principles。论证一个 deployment 中何时各自是合适工具。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Llama Guard | “the classifier” | 带 14 个 hazard categories 的 fine-tuned Llama-3.1-8B/4-12B safety classifier |
| Garak | “the scanner” | NVIDIA 开源 vulnerability scanner；probes、detectors、harnesses |
| PyRIT | “the campaign tool” | Microsoft multi-turn red-team orchestrator；converters、orchestrators、scoring |
| Prompt-Guard | “the small classifier” | Meta 的 86M prompt-injection classifier，与 Llama Guard 搭配使用 |
| TBSA | “tier-based scoring” | Garak 的 tier-based pass/fail，用于替代 binary outcomes |
| Converter chain | “paraphrase + encode + ...” | PyRIT 中构建 multi-step attacks 的 composition primitive |
| MLCommons hazard categories | “the 14 taxonomies” | Llama Guard 面向的 industry-standard taxonomy |

## 延伸阅读

- [Meta — Llama Guard 3 (in Llama 3 Herd paper, arXiv:2407.21783)](https://arxiv.org/abs/2407.21783) — 8B classifier
- [Meta — Llama Guard 3-1B-INT4 (arXiv:2411.17713)](https://arxiv.org/abs/2411.17713) — quantized mobile classifier
- [NVIDIA Garak — GitHub](https://github.com/NVIDIA/garak) — scanner repo 和 documentation
- [Microsoft PyRIT — GitHub](https://github.com/Azure/PyRIT) — campaign toolkit
