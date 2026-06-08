# Moderation Systems——OpenAI、Perspective、Llama Guard

> Production moderation systems 把第 12-16 课定义的 safety policies 操作化。OpenAI Moderation API：`omni-moderation-latest`（2024）基于 GPT-4o，一次调用即可分类 text + images；在 multilingual test set 上比上一版好 42%；response schema 返回 13 个 category booleans——harassment、harassment/threatening、hate、hate/threatening、illicit、illicit/violent、self-harm、self-harm/intent、self-harm/instructions、sexual、sexual/minors、violence、violence/graphic；对大多数开发者免费。Layered patterns：Input moderation（pre-generation）、Output moderation（post-generation）、Custom moderation（domain rules）。Async parallel calls 隐藏 latency；flag 时返回 placeholder responses。Llama Guard 3/4（第 16 课）：14 个 MLCommons hazards、Code Interpreter Abuse、8 种语言（v3）、multi-image（v4）。Perspective API（Google Jigsaw）：早于 LLM-as-moderator 浪潮的 toxicity scoring；主要是 single-dimension toxicity，带 severe-toxicity/insult/profanity variants；content-moderation research 的 baseline。Deprecations：Azure Content Moderator 于 2024 年 2 月 deprecated，2027 年 2 月 retired，由 Azure AI Content Safety 替代。

**类型:** Build
**语言:** Python (stdlib, three-layer moderation harness)
**先修:** Phase 18 · 16 (Llama Guard / Garak / PyRIT)
**时间:** ~60 minutes

## 学习目标

- 描述 OpenAI Moderation API 的 category taxonomy，以及它与 Llama Guard 3 的 MLCommons set 有何不同。
- 描述 three moderation-layer pattern（input、output、custom），并说出每层的一个 failure mode。
- 描述 Perspective API 作为 pre-LLM-era baseline 的位置，以及为什么它仍被研究使用。
- 说明 Azure deprecation timeline。

## 要解决的问题

第 12-16 课描述 attacks 和 defense tooling。第 29 课覆盖 deployed moderation systems，它们在用户接触产品的表面把 defenses 操作化。Three-layer pattern 是 2026 年默认配置。

## 核心概念

### OpenAI Moderation API

`omni-moderation-latest`（2024）。基于 GPT-4o。一次调用分类 text + images。对大多数开发者免费。

Categories（response schema 中的 13 个 booleans）：
- harassment, harassment/threatening
- hate, hate/threatening
- self-harm, self-harm/intent, self-harm/instructions
- sexual, sexual/minors
- violence, violence/graphic
- illicit, illicit/violent

Multimodal support 适用于 `violence`、`self-harm` 和 `sexual`，但不适用于 `sexual/minors`；其余为 text-only。

为了教学简单，`code/main.py` 中的 code harness 会把 `/threatening`、`/intent`、`/instructions` 和 `/graphic` 子类别折叠到它们的 top-level parents。Production code 应使用完整的 13-category schema。

在 multilingual test set 上比上一代 moderation endpoint 好 42%。提供 per-category scores；applications 自行设置 thresholds。

### Llama Guard 3/4

第 16 课已覆盖。14 个 MLCommons hazard categories（组织方式不同于 OpenAI 的 13 个 response-schema booleans）。支持 8 种语言（v3）。Llama Guard 4（2025 年 4 月）是原生 multimodal，12B。

OpenAI 与 Llama Guard taxonomies 有重叠，但也有分歧。OpenAI 把 “illicit” 作为 broad category；Llama Guard 把 “violent crimes” 和 “non-violent crimes” 分开。部署方会根据自身 policy-taxonomy fit 选择。

### Perspective API（Google Jigsaw）

早于 LLM-as-moderator 浪潮（pre-2020）的 toxicity scoring system。Categories：TOXICITY、SEVERE_TOXICITY、INSULT、PROFANITY、THREAT、IDENTITY_ATTACK。Primary score 是 single-dimension（TOXICITY），并带 sub-dimension variants。

它仍被广泛用作 content-moderation research baseline，因为 API 稳定、有文档，并且有多年 calibration data。对现代 LLM-adjacent use cases，Llama Guard 或 OpenAI Moderation 通常更合适。

### Three-layer pattern

1. **Input moderation。** 生成前分类用户 prompt。如果 flagged，就拒绝。Latency：一次 classifier call。
2. **Output moderation。** 交付前分类模型 output。如果 flagged，用 refusal 替换。Latency：生成后一次 classifier call。
3. **Custom moderation。** Domain-specific rules（regex、allowlists、business policy）。可在 input 或 output 运行。

这三层按设计是顺序的：input moderation 必须在 generation 前完成，output moderation 在 generation 后运行。Parallelism 发生在一层内部——对同一段 text 并发运行多个 classifiers（例如 OpenAI Moderation + Llama Guard + Perspective），可以隐藏单个 classifier 的 latency。作为可选优化，在 input moderation 完成且 token-1 streaming 延迟时，可以显示 placeholder response（“one moment, checking...”）。Flag behaviour 可配置：refuse、sanitize、escalate to human review。

### Failure modes

- **Input only。** 抓不到 output hallucinations（第 12-14 课 encoding attacks 会完全绕过 input classifiers）。
- **Output only。** 允许任何 input 到达模型；增加 cost；把 internal reasoning 暴露给 attacker。
- **Custom only。** 不能跨 categories robust；regexes 很脆弱。

Layered 是默认方案。Belt-and-suspenders。

### Azure deprecation

Azure Content Moderator：2024 年 2 月 deprecated，2027 年 2 月 retired。由 Azure AI Content Safety 替代，后者基于 LLM，并与 Azure OpenAI 集成。对 Azure deployments 来说，迁移是一个 2024-2027 年的 field-level project。

### 它在 Phase 18 中的位置

第 16 课在 red-team context 中覆盖 moderation tooling。第 29 课覆盖 deployed moderation。第 30 课以当前 dual-use capability evidence 收束。

## 实际使用

`code/main.py` 构建 three-layer moderation harness：input moderator（keyword + category score）、output moderator（同一个 classifier 作用于 output）、custom moderator（domain rules）。你可以让 inputs 通过它，并观察哪一层捕捉到什么。

## 交付成果

本课产出 `outputs/skill-moderation-stack.md`。给定一个 deployment，它会推荐 moderation stack configuration：input 用哪个 classifier、output 用哪个、哪些 custom rules，以及 edge cases 用什么 judge。

## 练习

1. 运行 `code/main.py`。让一个 benign、borderline 和 harmful input 通过三层。报告每个 input 触发哪一层。

2. 用特定 category 扩展 harness，加入 Perspective-API-style toxicity scoring。把它的 threshold behaviour 与 category score 比较。

3. 阅读 OpenAI Moderation API docs 和 Llama Guard 3 category list。把每个 OpenAI category 映射到最接近的 Llama Guard categories。识别三个无法干净映射的 categories。

4. 为 code-assistant deployment（例如 GitHub Copilot）设计 moderation stack。识别最相关和最不相关的 categories，并提出 custom rules。

5. Azure Content Moderator 于 2027 年 2 月 retired。规划迁移到 Azure AI Content Safety。识别迁移中风险最高的元素。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| OpenAI Moderation | “omni-moderation-latest” | 基于 GPT-4o 的 13-category（text）classifier，带部分 multimodal support |
| Perspective API | “Google Jigsaw toxicity” | Pre-LLM-era toxicity scoring baseline |
| Llama Guard | “MLCommons 14-category” | Meta 的 hazard classifier（v3：8B text、8 langs；v4：12B multimodal） |
| Input moderation | “pre-generation filter” | 模型调用前作用于 user prompt 的 classifier |
| Output moderation | “post-generation filter” | 交付前作用于 model output 的 classifier |
| Custom moderation | “domain rules” | Deployment-specific rules（regex、allowlist、policy） |
| Layered moderation | “all three layers” | 标准 production deployment pattern |

## 延伸阅读

- [OpenAI Moderation API docs](https://platform.openai.com/docs/api-reference/moderations) — omni-moderation endpoint
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama) — Llama Guard repo
- [Google Jigsaw Perspective API](https://perspectiveapi.com/) — toxicity scoring
- [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/) — Azure replacement
