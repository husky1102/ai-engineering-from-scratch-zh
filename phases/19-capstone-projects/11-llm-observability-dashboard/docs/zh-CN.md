# 综合项目 11：LLM 可观测性与评估仪表盘

> Langfuse 转向 open-core。Arize Phoenix 发布了 2026 GenAI semconv mappings。Helicone 和 Braintrust 都加码 per-user cost attribution。Traceloop 的 OpenLLMetry 成为事实上的 SDK instrumentation。生产形态是：ClickHouse 存 traces，Postgres 存 metadata，Next.js 做 UI，以及一小支 eval jobs 队伍（DeepEval、RAGAS、LLM-judge）在 sampled traces 上运行。构建一个 self-hosted 版本，至少从四个 SDK families 摄取数据，并演示在五分钟内捕捉 injected regression。

**类型：** Capstone
**语言：** TypeScript (UI), Python / TypeScript (ingest + evals), SQL (ClickHouse)
**先修：** Phase 11 (LLM engineering), Phase 13 (tools), Phase 17 (infrastructure), Phase 18 (safety)
**练习阶段：** P11 · P13 · P17 · P18
**时间：** 25 hours

## 要解决的问题

2026 年，每个运行 production traffic 的 AI team 都会在 model 旁边保留一层 observability plane。Cost attribution。Hallucination detection。Drift monitoring。Jailbreak signal。SLO dashboards。PII leak alerts。开源参考实现 Langfuse、Phoenix、OpenLLMetry 都收敛到 OpenTelemetry GenAI semantic conventions，把它作为 ingest schema。现在你可以用一个 SDK instrument OpenAI、Anthropic、Google、LangChain、LlamaIndex 和 vLLM，并发送兼容 spans。

你将构建一个 self-hosted dashboard，从至少四个 SDK families 摄取数据，在 sampled traces 上运行一小组 eval jobs，检测 drift 并 alert。衡量标准：给定一个 deliberate injected regression（一个开始产生 PII 的 prompt），dashboard 能在五分钟内捕捉它并触发 alert。

## 核心概念

Ingest 是 OTLP HTTP。SDK 产生 GenAI-semconv spans：`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`、`gen_ai.response.id`、`llm.prompts`、`llm.completions`。Spans 落入 ClickHouse 做 columnar analytics；metadata（users, sessions, apps）落入 Postgres。

Evals 作为 batch jobs 在 sampled traces 上运行。DeepEval 评分 faithfulness、toxicity 和 answer relevance。当 trace 携带 retrieval context 时，RAGAS 评分 retrieval metrics。Custom LLM-judges 运行 domain-specific checks（PII leak、off-policy response）。Eval runs 会写回同一个 ClickHouse，作为链接到 parent trace 的 eval spans。

Drift detection 观察 embedding-space distributions 随时间的变化（prompt embeddings 上的 PSI 或 KL divergence）以及 eval-score trends。Alerts 进入 Prometheus Alertmanager，然后到 Slack / PagerDuty。UI 使用 Next.js 15 和 Recharts。

## 架构

```text
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## 技术栈

- Ingest: OpenTelemetry SDKs + GenAI semantic conventions; OTLP HTTP transport
- Collector: OpenTelemetry Collector with tail-sampling processor (for cost control)
- Storage: ClickHouse for spans, Postgres for metadata, S3 for raw event archive
- Evals: DeepEval, RAGAS 0.2, Arize Phoenix evaluator pack, custom LLM-judge
- Drift: PSI / KL on pooled prompt embeddings (sentence-transformers) weekly
- Alerting: Prometheus Alertmanager -> Slack / PagerDuty
- UI: Next.js 15 App Router + Recharts + server actions
- SDKs supported out of the box: OpenAI, Anthropic, Google GenAI, LangChain, LlamaIndex, vLLM

## 动手实现

1. **Collector config。** OpenTelemetry Collector，配置 OTLP HTTP receiver、tail-sampler（保留 100% errored traces 和 10% successes），并导出到 ClickHouse 和 S3。

2. **ClickHouse schema。** 表 `spans` 的列镜像 GenAI semconv：`gen_ai_system`、`gen_ai_request_model`、`input_tokens`、`output_tokens`、`latency_ms`、`prompt_hash`、`trace_id`、`parent_span_id`，外加用于 long payloads 的 JSON bag。按 user_id 和 app_id 添加 secondary indexes。

3. **SDK coverage test。** 用每个 SDK（OpenAI、Anthropic、Google、LangChain、LlamaIndex、vLLM）编写一个小 client app，并用 OpenLLMetry auto-instrument。验证每个都会产生 canonical GenAI spans 并落入 ClickHouse。

4. **Eval jobs。** scheduled job 读取 last-15-min sampled traces，并运行 DeepEval faithfulness、toxicity 和 answer relevance。输出是链接到 parent trace 的 eval spans。

5. **Custom LLM-judge。** 一个 PII-leak judge：给定 response，调用 guard LLM 为 PII leak likelihood 打分。高分 responses 进入 triage queue。

6. **Drift detection。** weekly job 计算本周 pooled prompt embeddings 与 trailing 4-week baseline 之间的 PSI。如果 PSI 高于阈值，alert。

7. **Dashboard。** Next.js 15，包含页面：overview（spans/sec、cost/user、p95 latency）、traces（search + waterfall）、evals（faithfulness trend、toxicity）、drift（PSI over time）、alerts。

8. **Alerting chain。** Prometheus exporter 读取 eval score aggregates 和 latency percentiles；Alertmanager 将 warnings 路由到 Slack，将 critical breaches 路由到 PagerDuty。

9. **Regression probe。** 注入 bug：被评估的 chatbot 开始在 1% 的时间泄漏 fake SSNs。测量 MTTR：从 bug deployed 到 Slack alert。

## 实际使用

```text
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## 交付成果

`outputs/skill-llm-observability.md` 是交付物。给定一个 LLM application，dashboard 会摄取它的 traces，运行 evals，对 drift 发出 alerts，并在 Next.js 中呈现 cost/user breakdown。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | Trace-schema coverage | 产生 canonical GenAI spans 的 SDK families 数量（target: 6+） |
| 20 | Eval correctness | DeepEval / RAGAS scores vs hand-labeled set |
| 20 | Dashboard UX | injected regression 上的 MTTR（目标低于 5 minutes） |
| 20 | Cost / scale | 持续摄取 1k spans/sec 且无 backlog |
| 15 | Alerting + drift detection | Prometheus/Alertmanager chain 端到端 exercised |
| **100** | | |

## 练习

1. 为 Haystack framework 添加 custom instrumentation。验证 canonical spans 以忠实的 `gen_ai.*` attributes 落入 ClickHouse。

2. 在同一批 traces 上将 DeepEval 换成 Phoenix evaluators。测量两个 eval engines 之间的 score drift。

3. 锐化 drift detector：按 app-id 而不是全局计算 PSI。展示 per-app drift trails。

4. 添加 "user impact" page：带 sparklines 的 cost-per-user 和 failure-rate-per-user。

5. 构建 tail-sampling policy，保留 100% toxicity > 0.5 的 traces，再对其余 traces 做 10% stratified sample。测量引入的 sampling bias。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| GenAI semconv | "OTel LLM attributes" | 2025 OpenTelemetry spec，用于 LLM span attributes（system, model, tokens） |
| Tail sampling | "Post-trace sample" | Collector 在 trace 完成后决定保留或丢弃（可查看 errors） |
| PSI | "Population stability index" | 比较两个 distributions 的 drift metric；> 0.2 通常表示 meaningful drift |
| LLM-judge | "Eval as model" | 一个 LLM 按 rubric 给另一个 LLM 的 output 打分（faithfulness、toxicity、PII） |
| Tail-sampling policy | "Keep-rule" | 决定哪些 traces persist vs drop 的规则；errored + sample-rate |
| Eval span | "Linked eval trace" | 携带 eval score，并链接到原始 LLM call span 的 child span |
| Cost per user | "Unit economics" | 在一个 window 内归因到某个 user_id 的 dollar cost；关键 product metric |

## 延伸阅读

- [Langfuse](https://github.com/langfuse/langfuse) — reference open-core observability platform
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — alternate reference with strong drift support
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) — auto-instrumentation SDK family
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — ingest schema
- [Helicone](https://www.helicone.ai) — alternate hosted observability
- [Braintrust](https://www.braintrust.dev) — alternate eval-first platform
- [ClickHouse documentation](https://clickhouse.com/docs) — columnar span store
- [DeepEval](https://github.com/confident-ai/deepeval) — evaluator library
