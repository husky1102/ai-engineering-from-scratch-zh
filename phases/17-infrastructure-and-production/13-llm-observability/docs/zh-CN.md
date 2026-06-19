# LLM 可观测性技术栈选择

> 2026 年 observability 市场分成两类。Development platforms（LangSmith、Langfuse、Comet Opik）把 monitoring 与 evals、prompt management、session replays 捆在一起。Gateway/instrumentation tools（Helicone、SigNoz、OpenLLMetry、Phoenix）专注 telemetry。Langfuse 是 MIT-licensed core，OSS 平衡感很强（free cloud 每月 50K events）。Phoenix 是 OpenTelemetry-native，采用 Elastic License 2.0，非常适合 drift/RAG visualization，但不是持久 production backend。Arize AX 使用 zero-copy Iceberg/Parquet integration，宣称比 monolithic observability 便宜 100x。LangSmith 在 LangChain/LangGraph 场景领先，$39/user/mo，只有 Enterprise 可 self-host。Helicone 是 proxy-based，15-30 分钟 setup，free 100K req/mo，但 agent traces 深度较弱。常见 production pattern：Gateway（Helicone/Portkey）+ eval platform（Phoenix/TruLens），用 OpenTelemetry 粘合。

**类型：** Learn
**语言：** Python (stdlib, toy trace-sampling simulator)
**先修：** Phase 17 · 08 (Inference Metrics), Phase 14 (Agent Engineering)
**时间：** ~60 minutes

## 学习目标

- 区分 development platforms（捆绑 evals + prompts + sessions）与 gateway/telemetry tools（仅 traces + metrics）。
- 将六个主要工具（Langfuse、LangSmith、Phoenix、Arize AX、Helicone、Opik）映射到它们的 licensing、pricing 和 sweet-spot use cases。
- 解释 OpenTelemetry-glue pattern：它让你把 gateway tool 与独立 eval platform 组合起来。
- 说出 2026 年的成本差异化因素（Arize AX 的 zero-copy approach vs monolithic ingest），并给出约 100x 的乘数。

## 要解决的问题

你上线了一个 LLM feature。它能工作。但你看不到 prompt failures、tool loops、latency regressions、cost spikes 或 prompt-cache hit rate。你搜索“LLM observability”，看到八个工具都声称以三种不同价位解决同一个问题。

它们并不解决同一个问题。LangSmith 回答“为什么这个 LangGraph run 失败了？”Phoenix 回答“我的 RAG pipeline 是否 drifting？”Helicone 回答“哪个 app 正在烧 tokens？”Langfuse 回答“我能否 self-host 整个东西？”工具不同，受众也不同。

选择涉及四个轴：stack（LangChain？raw SDK？multi-vendor？）、license tolerance（只接受 MIT？Elastic OK？commercial fine？）、budget（free tier？$100/mo？$1000/mo？）和 self-host（必须？nice-to-have？从不？）。

## 核心概念

### 两个类别

**Development platforms** 把 observability 与 evals、prompt management、dataset versioning、session replay 捆在一起。你运行 experiments，查看哪个 prompt 有效，用 dataset-regression 将新 prompt 与旧 winners 对比。LangSmith、Langfuse、Comet Opik 属于这一类。

**Gateway/telemetry tools** 对 inference calls 做 instrumentation：prompt、response、tokens、latency、model、cost。Helicone、SigNoz、OpenLLMetry、Phoenix 属于这一类。它们更 minimalist，可通过 OpenTelemetry 与单独的 eval tool 组合。

### Langfuse：OSS balance

- Core Apache / MIT licensed；通过 Docker self-host。
- Cloud free tier: 50K events/month。Paid: $29/mo for team。
- Evals、prompt management、traces、datasets。对四类 dev-platform features 覆盖合理。
- Sweet spot：你想要 LangSmith-class features，但必须 self-host 或坚持 OSS license。

### Phoenix（Arize）：telemetry-first、OpenTelemetry-native

- Elastic License 2.0；self-host 很简单。
- RAG 和 drift visualization 很出色。Embedding-space scatter plots 是一等功能。
- 不是为 persistent production backend 设计的，主要是 development-time observability。
- Sweet spot：RAG pipeline development、drift debugging，并与独立 gateway 搭配用于 production。

### Arize AX：scale play

- Commercial。通过 Iceberg/Parquet 做 zero-copy data lake integration。
- 宣称在 scale 下比 monolithic observability（Datadog-class）便宜约 100x。数学逻辑：你把 traces 存在自己 S3 上的 Parquet 中；Arize 直接读取。
- Sweet spot：>10M traces/day、已有 data lake、想要 LLM-specific dashboards 但不想付 Datadog pricing。

### LangSmith：LangChain/LangGraph first

- Commercial，$39/user/month。Self-host 仅限 Enterprise。
- 对 LangChain 和 LangGraph stacks 最强。如果你不在这两者上，它就没那么有吸引力。
- Sweet spot：团队已经 committed to LangChain，且愿意付费。

### Helicone：proxy-based minimum viable

- 通过把 `OPENAI_API_BASE` 换成 Helicone proxy，15-30 分钟 setup。
- MIT licensed；free 100K req/mo，paid $20/mo+。
- 包含 failover、caching、rate limits，也可作为 gateway。
- 对 agent / multi-step traces 的深度较弱。
- Sweet spot：quick start、single-stack app、需要 gateway + observability in one。

### Opik（Comet）：OSS dev platform

- Apache 2.0，fully OSS。
- 与 Langfuse 功能集相近，并继承 Comet 经验。
- Sweet spot：已经在 Comet 上的 ML teams，想在同一个 pane 中获得 LLM observability。

### SigNoz：OpenTelemetry-first full APM

- Apache 2.0。通过 OpenTelemetry 处理 general APM 加 LLM。
- Sweet spot：跨 services 与 LLM calls 的 unified observability。

### 粘合层：OpenTelemetry + GenAI semantic conventions

OpenTelemetry 在 2025 年末发布 GenAI semantic conventions（`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`）。能消费 OTel 的工具可以互操作。正在形成的 production pattern：

1. 每个 LLM call 都发出带 GenAI conventions 的 OTel。
2. 路由到 gateway（Helicone / Portkey）用于 day-to-day。
3. Dual-ship 到 eval platform（Phoenix / Langfuse）用于 regressions。
4. 归档到 data lake（Iceberg），通过 Arize AX 或 DuckDB 做长期分析。

### 陷阱：在错误层做 instrumentation

在 agent framework 内部做 instrumentation（例如添加 LangSmith traces）会把你绑定到该 framework。在 HTTP/OpenAI-SDK layer 做 instrumentation（通过 OpenLLMetry 或你的 gateway）才是 portable 的。

### Sampling：你无法保留一切

超过 1M requests/day 后，full-trace retention 的成本会超过 LLM calls 本身。按规则 sampling：100% errors、100% high-cost、5% success。Aggregates 始终保留；raw 留给 long tail。

### 你应该记住的数字

- Langfuse free cloud: 50K events/month。
- LangSmith: $39/user/month。
- Helicone free: 100K req/month。
- Arize AX claim: scale 下比 monolithic 便宜约 100x。
- OpenTelemetry GenAI conventions: 2025 shipping，2026 widely adopted。

## 实际使用

`code/main.py` 会模拟 1M-trace day 下的不同 retention strategies（100% ingest、sampling、sampling + errors）。它报告 storage cost，以及每种策略会丢失什么。

## 交付成果

本课产出 `outputs/skill-observability-stack.md`。给定 stack、scale、budget、license posture，它会选择 tool(s)。

## 练习

1. 你的团队使用 LangChain，并想要 OSS self-hosted observability。选择 Langfuse 或 Opik，并说明理由。
2. 在 5M traces/day 且 Datadog 报价 $150K/month 的情况下，计算 Arize AX 的 break-even。
3. 设计一组你们组织 guideline 应强制每个 LLM call 携带的 OpenTelemetry GenAI attributes。
4. 论证 Phoenix alone 是否足够用于 production。什么时候不够？
5. Helicone 有 20ms proxy overhead。在 P99 TTFT 300 ms 下，这能接受吗？如果 SLA 是 100 ms 呢？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| OpenLLMetry | “OTel for LLMs” | 面向 LLMs 的 open-source OpenTelemetry instrumentation |
| GenAI conventions | “OTel attributes” | LLM calls 的标准 OTel attribute names |
| LangSmith | “LangChain observability” | 与 LangChain ecosystem 捆绑的 commercial platform |
| Langfuse | “OSS LangSmith” | MIT OSS，功能集相近 |
| Phoenix | “Arize dev tool” | OpenTelemetry-native dev/eval platform |
| Arize AX | “scale observability” | Commercial zero-copy Iceberg/Parquet observability |
| Helicone | “proxy observability” | 收集 LLM telemetry + gateway features 的 HTTP proxy |
| Opik | “Comet LLM” | Comet 出品的 Apache 2.0 OSS dev platform |
| Session replay | “trace rerun” | 重新播放带 tool calls 的完整 agent session |
| Eval | “offline test” | 在 labeled dataset 上运行 candidate model/prompt |

## 延伸阅读

- [SigNoz — Top LLM Observability Tools 2026](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX Alternative analysis](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — Setting Up Langfuse, LangSmith, Helicone, Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix docs](https://docs.arize.com/phoenix)
- [Helicone docs](https://docs.helicone.ai/)
