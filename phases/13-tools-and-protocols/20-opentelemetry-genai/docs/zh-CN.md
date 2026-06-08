# OpenTelemetry GenAI：端到端追踪工具调用

> 一个 agent 调用五个工具、三个 MCP servers 和两个 sub-agents。你需要贯穿所有这些组件的一条 trace。OpenTelemetry GenAI semantic conventions（v1.37 及以上稳定 attributes）是 2026 年标准，并被 Datadog、Langfuse、Arize Phoenix、OpenLLMetry 和 AgentOps 原生支持。本课命名 required attributes，讲解 span hierarchy（agent → LLM → tool），并交付一个可接入任意 OTel exporter 的 stdlib span emitter。

**类型：** 构建
**语言：** Python（stdlib，OTel span emitter）
**先修：** Phase 13 · 07（MCP server）、Phase 13 · 08（MCP client）
**时间：** 约 75 分钟

## 学习目标

- 命名 LLM span 和 tool-execution span 所需的 OTel GenAI attributes。
- 构建覆盖 agent loop、LLM call、tool call 和 MCP client dispatch 的 trace hierarchy。
- 判断哪些内容要 capture（opt-in），哪些要 redact（defaults）。
- 不重写 tool code，就把 spans 发射到本地 collector（Jaeger、Langfuse）。

## 要解决的问题

2026 年 2 月的一个 debug：用户报告“我的 agent 有时需要 30 秒响应，其他时候只要 3 秒”。没有 traces。logs 显示了 LLM call，但没有 tool dispatch，没有 MCP server round-trip，也没有 sub-agent。你只能猜。最后发现：某个 MCP server 偶尔因为 cold-start 挂住。

没有端到端 tracing，你找不到这个问题。OTel GenAI 修复它。

这些 conventions 在 2025-2026 年由 OpenTelemetry semantic-conventions group 定型。它们定义稳定 attribute names，使 Datadog、Langfuse、Phoenix、OpenLLMetry 和 AgentOps 都能解析同一组 spans。一次 instrument，发送到任意 backend。

## 核心概念

### Span hierarchy

```text
agent.invoke_agent  (top, INTERNAL span)
 ├── llm.chat       (CLIENT span)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT span)
 ├── llm.chat       (CLIENT span)
 └── subagent.invoke (INTERNAL)
```

整个结构嵌套在同一个 trace id 下。Span ids 连接 parent-child relationships。

### Required attributes

按照 2025-2026 semconv：

- `gen_ai.operation.name`：`"chat"`、`"text_completion"`、`"embeddings"`、`"execute_tool"`、`"invoke_agent"`。
- `gen_ai.provider.name`：`"openai"`、`"anthropic"`、`"google"`、`"azure_openai"`。
- `gen_ai.request.model`：请求的 model string（例如 `"gpt-4o-2024-08-06"`）。
- `gen_ai.response.model`：实际服务请求的 model。
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`。
- `gen_ai.response.id`：用于 correlation 的 provider response id。

对于 tool spans：

- `gen_ai.tool.name`：tool identifier。
- `gen_ai.tool.call.id`：具体 call id。
- `gen_ai.tool.description`：tool description（optional）。

对于 agent spans：

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`。

### Span kinds

- 对跨越 process boundary 的调用（LLM provider、MCP server）使用 `SpanKind.CLIENT`。
- 对 agent 自身 loop steps 和 tool execution 使用 `SpanKind.INTERNAL`。

### Opt-in content capture

默认情况下，spans 携带 metrics 和 timing，不携带 prompts 或 completions。大 payloads 和 PII 默认关闭。设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 和特定 content-capture env vars 才会包含内容。生产启用前要仔细 review。

### Events on spans

token-level events 可以作为 span events 添加：

- `gen_ai.content.prompt`：input messages。
- `gen_ai.content.completion`：output messages。
- `gen_ai.content.tool_call`：记录下来的 tool call。

Events 在一个 span 内按时间排序，便于详细 replay。

### Exporters

OTel spans 可导出到：

- **Jaeger / Tempo。** OSS，on-prem。
- **Langfuse。** LLM-observability-specific；可视化 token usage。
- **Arize Phoenix。** Evals + tracing combined。
- **Datadog。** 商业；原生解析 `gen_ai.*` attributes。
- **Honeycomb。** Column-oriented；query-friendly。

这些都使用 OTLP 这种 wire format。你的代码不需要关心后端差异。

### Propagation across MCP

当 MCP client 调用 server 时，把 W3C traceparent header 注入请求。Streamable HTTP 支持标准 headers。Stdio 原生不携带 HTTP headers；spec 的 2026 roadmap 讨论在 JSON-RPC calls 上添加 `_meta.traceparent` 字段。

在它发布前：手动把 traceparent 放入每个请求的 `_meta`。server 记录 trace id。

### Metrics

除 spans 外，GenAI semconv 还定义 metrics：

- `gen_ai.client.token.usage`：histogram。
- `gen_ai.client.operation.duration`：histogram。
- `gen_ai.tool.execution.duration`：histogram。

将它们用于不需要 per-call detail 的 dashboards。

### AgentOps layer

AgentOps（2024 年创立）专注 GenAI observability。它包装流行 frameworks（LangGraph、Pydantic AI、CrewAI），自动发射 OTel spans。如果你的 stack 使用受支持 framework，它很有用；否则使用 manual instrumentation。

## 实际使用

`code/main.py` 为一个调用 LLM、dispatch 两个 tools，并进行一次 MCP round-trip 的 agent，将 OTel-shaped spans 发射到 stdout（OTLP-JSON-like format）。没有真实 exporter；本课聚焦 span shape 和 attribute set。把输出粘贴到 OTLP-compatible viewer 中，或直接阅读。

重点查看：

- Trace id 在所有 spans 间共享。
- Parent-child links 通过 `parentSpanId` 编码。
- Required `gen_ai.*` attributes 已填充。
- Content capture 默认关闭；一个场景通过 env var 打开它。

## 交付成果

本课产出 `outputs/skill-otel-genai-instrumentation.md`。给定一个 agent codebase，该 skill 会产出 instrumentation plan：在哪里添加 spans、填充哪些 attributes、面向哪些 exporters。

## 练习

1. 运行 `code/main.py`。数一数 spans，并识别哪个是 CLIENT，哪个是 INTERNAL。

2. 打开 content capture（env var），确认 `gen_ai.content.prompt` 和 `gen_ai.content.completion` events 出现。注意 PII 含义。

3. 添加 tool-execution metric `gen_ai.tool.execution.duration`，并为每次调用以 histogram sample 发射。

4. 将 traceparent 从 parent agent span 传播到 MCP request 的 `_meta.traceparent` 字段。验证 MCP server 会看到同一个 trace id。

5. 阅读 OTel GenAI semconv spec。找出 semconv 中列出但本课代码**没有**发射的一个 attribute。添加它。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| OTel | “OpenTelemetry” | traces、metrics、logs 的开放标准 |
| GenAI semconv | “GenAI semantic conventions” | LLM / tool / agent spans 的稳定 attribute names |
| `gen_ai.*` | “The attribute namespace” | 所有 GenAI attributes 共享此前缀 |
| Span | “Timed operation” | 带 start、end 和 attributes 的工作单元 |
| Trace | “Cross-span ancestry” | 共享同一 trace id 的 span 树 |
| SpanKind | “CLIENT / SERVER / INTERNAL” | 关于 span 方向的提示 |
| OTLP | “OpenTelemetry Line Protocol” | exporters 使用的 wire format |
| Opt-in content | “Prompt / completion capture” | 默认关闭；通过 env var 启用 |
| traceparent | “W3C header” | 跨服务传播 trace context |
| Exporter | “Backend-specific shipper” | 将 spans 发送到 Jaeger / Datadog 等后端的组件 |

## 延伸阅读

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — GenAI spans、metrics 和 events 的 canonical conventions
- [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — LLM 和 tool-execution span attribute list
- [OpenTelemetry — GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — agent-level `invoke_agent` span
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — GitHub-hosted source of truth
- [Datadog — LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — production integration walkthrough
