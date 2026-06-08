# OpenTelemetry GenAI Semantic Conventions

> OpenTelemetry 的 GenAI SIG（2024 年 4 月启动）定义了 agent telemetry 的标准 schema。Span names、attributes 和 content-capture rules 跨 vendors 收敛，使 agent traces 在 Datadog、Grafana、Jaeger 和 Honeycomb 中含义一致。

**类型:** Learn + Build
**语言:** Python（stdlib）
**先修:** Phase 14 · 13（LangGraph），Phase 14 · 24（Observability Platforms）
**时间:** ~60 分钟

## 学习目标

- 说出 GenAI span categories：model/client、agent、tool。
- 区分 `invoke_agent` CLIENT vs INTERNAL spans，以及各自何时适用。
- 列出 top-level GenAI attributes：provider name、request model、data-source ID。
- 解释 content-capture contract：opt-in、`OTEL_SEMCONV_STABILITY_OPT_IN`、external-reference recommendation。

## 要解决的问题

每个 vendor 都发明自己的 span names。Ops teams 最终为每个 framework 构建 dashboards。OpenTelemetry 的 GenAI SIG 通过定义全生态都瞄准的一个标准来修复这一点。

## 核心概念

### Span categories

1. **Model / client spans.** 覆盖原始 LLM calls。由 provider SDKs（Anthropic、OpenAI、Bedrock）和 framework model adapters emit。
2. **Agent spans.** `create_agent`（agent constructed 时）和 `invoke_agent`（agent run 时）。
3. **Tool spans.** 每次 tool invocation 一个；通过 parent-child relation 连接到 agent span。

### Agent span naming

- Span name：如果命名，则为 `invoke_agent {gen_ai.agent.name}`；否则 fallback 到 `invoke_agent`。
- Span kind：
  - **CLIENT**：用于 remote agent services（OpenAI Assistants API、Bedrock Agents）。
  - **INTERNAL**：用于 in-process agent frameworks（LangChain、CrewAI、local ReAct）。

### Key attributes

- `gen_ai.provider.name`：`anthropic`、`openai`、`aws.bedrock`、`google.vertex`。
- `gen_ai.request.model`：model ID。
- `gen_ai.response.model`：resolved model（可能因 routing 而不同于 request）。
- `gen_ai.agent.name`：agent identifier。
- `gen_ai.operation.name`：`chat`、`completion`、`invoke_agent`、`tool_call`。
- `gen_ai.data_source.id`：用于 RAG，表示 consult 了哪个 corpus 或 store。

Anthropic、Azure AI Inference、AWS Bedrock、OpenAI 都有 technology-specific conventions。

### Content capture

默认规则：instrumentations SHOULD NOT 默认捕捉 inputs/outputs。通过以下属性 opt-in capture：

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

推荐 production pattern：把 content 存到外部（S3、你的 log store），在 spans 上记录 references（pointer IDs，而不是 prose）。这是 Lesson 27 content-poisoning defense 接入 observability 的方式。

### Stability

截至 2026 年 3 月，大多数 conventions 仍是 experimental。用以下方式 opt in 到 stable preview：

```text
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ 原生把 GenAI attributes 映射进其 LLM Observability schema。其他 backends（Grafana、Honeycomb、Jaeger）支持 raw attributes。

### 这个 pattern 哪里会出错

- **Capturing full prompts in spans.** PII、secrets、customer data 出现在 ops 可读 traces 中。存到外部。
- **No `gen_ai.provider.name`.** 缺少 attribution 时，multi-provider dashboards 会坏。
- **Spans without parent links.** Orphaned tool spans。始终传播 context。
- **Not setting stability opt-in.** Backend upgrade 时 attributes 可能被 rename。

## 动手实现

`code/main.py` 实现一个匹配 GenAI conventions 的 stdlib span emitter：

- 带 GenAI attribute schema 的 `Span`。
- 带 `start_span` 和 nested contexts 的 `Tracer`。
- 一个 scripted agent run，emit：`create_agent`、`invoke_agent`（INTERNAL）、per-tool spans、LLM calls 的 `chat` spans。
- 一种 content-capture mode，把 prompts 存到外部，并在 spans 上记录 IDs。

运行它：

```text
python3 code/main.py
```

输出：带全部 required GenAI attributes 的 span tree，以及显示 opt-in content references 的“external store”。

## 实际使用

- **Datadog LLM Observability**（v1.37+）原生映射 attributes。
- **Langfuse / Phoenix / Opik**（Lesson 24）：auto-instrument ecosystem。
- **Jaeger / Honeycomb / Grafana Tempo**：raw OTel traces；用 GenAI attributes 构建 dashboards。
- **Self-hosted**：运行带 GenAI processor 的 OTel Collector。

## 交付成果

`outputs/skill-otel-genai.md` 把 OTel GenAI spans 接入现有 agent，带 content-capture defaults 和 external-reference storage。

## 练习

1. 用 `invoke_agent`（INTERNAL）+ per-tool spans instrument 你的 Lesson 01 ReAct loop。发送到 Jaeger instance。
2. 以“references only”模式添加 content capture：prompts 写入 SQLite，span attributes 只携带 row IDs。
3. 阅读 `gen_ai.data_source.id` spec。把它接入你的 Lesson 09 Mem0 search。
4. 设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`，并验证 collector 不会 rename 你的 attributes。
5. 构建 dashboard：仅从 GenAI attributes 中分析“哪些 tool errors 与哪些 models 相关”。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| GenAI SIG | “OpenTelemetry GenAI group” | 定义 schema 的 OTel working group |
| invoke_agent | “Agent span” | 表示一次 agent run 的 span name |
| CLIENT span | “Remote call” | 调用 remote agent service 的 span |
| INTERNAL span | “In-process” | In-process agent run 的 span |
| gen_ai.provider.name | “Provider” | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | “RAG source” | Retrieval hit 的 corpus/store |
| Content capture | “Prompt logging” | Opt-in 捕捉 messages；生产中存外部 |
| Stability opt-in | “Preview mode” | 固定 experimental conventions 的 env var |

## 延伸阅读

- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — spec
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 默认 GenAI spans
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 内置 OTel spans
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — W3C trace context propagation
