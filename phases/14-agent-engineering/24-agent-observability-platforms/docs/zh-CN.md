# Agent Observability：Langfuse、Phoenix、Opik

> 三个 open-source agent observability platforms 主导 2026 年。Langfuse（MIT）：每月 600 万+ installs，tracing + prompt management + evals + session replay。Arize Phoenix（Elastic 2.0）：deep agent-specific evals、RAG relevancy、OpenInference auto-instrumentation。Comet Opik（Apache 2.0）：automated prompt optimization、guardrails、LLM-judge hallucination detection。

**类型:** Learn
**语言:** Python（stdlib）
**先修:** Phase 14 · 23（OTel GenAI）
**时间:** ~45 分钟

## 学习目标

- 说出三个顶级 open-source agent observability platforms 及其 licenses。
- 区分每个平台最强处：Langfuse（prompt mgmt + sessions）、Phoenix（RAG + auto-instrumentation）、Opik（optimization + guardrails）。
- 解释为什么到 2026 年，89% organizations 报告已有 agent observability。
- 实现一个带 LLM-judge evaluation 的 stdlib trace-to-dashboard pipeline。

## 要解决的问题

OTel GenAI（Lesson 23）给你 schema。你仍然需要 ingest spans、运行 evaluations、存储 prompt versions、暴露 regressions 的平台。三个竞争者各自强调 lifecycle 的不同部分。

## 核心概念

### Langfuse（MIT）

- 每月 600 万+ SDK installs，19k+ GitHub stars。
- Features：tracing、带 versioning + playground 的 prompt management、evaluations（LLM-as-judge、user feedback、custom）、session replays。
- 2025 年 6 月：曾经商业的 modules（LLM-as-a-judge、annotation queues、prompt experiments、Playground）以 MIT 开源。
- 最强处：带紧密 prompt-management loop 的 end-to-end observability。

### Arize Phoenix（Elastic License 2.0）

- 更深的 agent-specific evaluation：trace clustering、anomaly detection、RAG 的 retrieval relevancy。
- Native OpenInference auto-instrumentation。
- 与 managed Arize AX 搭配用于 production。
- 无 prompt versioning，被定位为配合 broader platforms 的 drift/behavioral-regression tool。
- 最强处：RAG relevancy、behavioral drift、anomaly detection。

### Comet Opik（Apache 2.0）

- 通过 A/B experiments 做 automated prompt optimization。
- Guardrails（PII redaction、topical constraints）。
- LLM-judge hallucination detection。
- Comet 自己测量的 benchmark：Opik logs + evals 23.44s vs Langfuse 327.15s（约 14x gap），vendor benchmarks 只能当 directional。
- 最强处：optimization loop、automated experimentation、guardrail enforcement。

### Industry data

根据 Maxim（2026 field analysis）：89% organizations 已有 agent observability；quality issues 是 top production barrier（32% respondents 提及）。

### 选择哪个

| Need | Pick |
|------|------|
| All-in-one with prompt management | Langfuse |
| Deep RAG evaluation + drift | Phoenix |
| Automated optimization + guardrails | Opik |
| Open licensing, no ELv2 | Langfuse（MIT）或 Opik（Apache 2.0） |
| Datadog / New Relic integration | 任意，它们都 export OTel |

### 这个 pattern 哪里会出错

- **No eval strategy.** 只有 tracing 没有 evaluation，只是昂贵 logging。
- **Self-rolled LLM-judge without grounding.** CRITIC pattern（Lesson 05）适用，judges 需要 external tools 做 factual verification。
- **Prompt versions not tied to traces.** Prod regress 时，你无法 bisect 到导致问题的 prompt。

## 动手实现

`code/main.py` 实现 stdlib trace collector + LLM-judge evaluator：

- Ingest GenAI-shaped spans。
- 按 session 分组，标记 failed runs（guardrail trips、low-confidence evals）。
- 一个 scripted LLM-judge，按 rubric 给 agent responses 打分。
- Dashboard-like summary：failure rate、top failure reasons、eval score distribution。

运行它：

```text
python3 code/main.py
```

输出：per-session eval scores 和 failure categorization，对应 Langfuse/Phoenix/Opik 会展示的内容。

## 实际使用

- **Langfuse** self-hosted 或 cloud；通过 OTel 或其 SDK 接线。
- **Arize Phoenix** self-hosted；auto-instrument OpenInference。
- **Comet Opik** self-hosted 或 cloud；automated optimization loop。
- **Datadog LLM Observability** 适合已经运行 Datadog 的 mixed ops+ML teams。

## 交付成果

`outputs/skill-obs-platform-wiring.md` 选择一个平台，并把 traces + evals + prompt versions 接入现有 agent。

## 练习

1. 导出一周 OTel traces 到 Langfuse cloud（free tier）。哪些 sessions 失败了？为什么？
2. 为你的 domain 编写 LLM-judge rubric（factual correctness、tone、scope adherence）。在 50 条 traces 上测试。
3. 比较 Langfuse prompt versioning 与 Phoenix trace clustering。哪个更快告诉你坏在什么地方？
4. 阅读 Opik guardrail docs。把 PII redaction guardrail 接到你的一个 agent run。
5. 在你的 corpus 上 benchmark 三者。忽略 vendor-published numbers，测你自己的。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Tracing | “Spans collector” | Ingest OTel / SDK spans；按 session index |
| Prompt management | “Prompt CMS” | 绑定 traces 的 versioned prompts |
| LLM-as-judge | “Automated eval” | 单独 LLM 按 rubric 给 agent output 打分 |
| Session replay | “Trace playback” | Step through past runs for debugging |
| RAG relevancy | “Retrieval quality” | Retrieved context 是否匹配 query |
| Trace clustering | “Behavioral grouping” | Cluster similar runs for drift detection |
| Guardrail enforcement | “Policy at log time” | 对 logged content 做 PII/toxicity/scope checks |

## 延伸阅读

- [Langfuse docs](https://langfuse.com/) — tracing、evals、prompt mgmt
- [Arize Phoenix docs](https://docs.arize.com/phoenix) — auto-instrumentation、drift
- [Comet Opik](https://www.comet.com/site/products/opik/) — optimization + guardrails
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 三者消费的 schema
