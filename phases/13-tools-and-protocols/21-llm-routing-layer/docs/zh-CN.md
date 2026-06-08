# LLM Routing Layer：LiteLLM、OpenRouter、Portkey

> Provider lock-in 很昂贵。不同 tool-calling workloads 适合不同模型。Routing gateways 提供一个 API surface、retries、failover、cost tracking 和 guardrails。2026 年三种 archetype 占主导：LiteLLM（open-source self-hosted）、OpenRouter（managed SaaS）、Portkey（production-grade，2026 年 3 月开源）。本课命名 decision criteria，并走读一个 stdlib routing gateway。

**类型:** Learn
**语言:** Python（stdlib，routing + failover + cost tracker）
**先修:** Phase 13 · 02（function calling），Phase 13 · 17（gateways）
**时间:** ~45 分钟

## 学习目标

- 区分 self-hosted、managed 和 production-grade routing options。
- 实现一个 fallback chain，按定义好的 priority order 在 provider failures 上 retry。
- 跨 providers 跟踪 per-request cost 和 token usage。
- 针对给定 production constraint，在 LiteLLM、OpenRouter 和 Portkey 之间做决定。

## 要解决的问题

Provider routing 重要的场景：

1. **Cost.** Claude Sonnet 成本是 Haiku 的 3 倍。Triage task 用 Haiku 足够；synthesis task 值得用 Sonnet。按 request route。

2. **Failover.** OpenAI 出现糟糕的一小时。所有 request 失败。你希望无需 redeploy 就自动 fallback 到 Anthropic。

3. **Latency.** Live chat UI 需要快速 time-to-first-token。Batch summarizer 不需要。按 latency SLA route。

4. **Compliance.** EU 用户必须留在 EU regions。按 region route。

5. **Experimentation.** 在同一 workload 上 A/B 两个模型。按 test bucket route。

为每个 integration 手写这些逻辑很重复。Routing gateway 提供一个 OpenAI-compatible API，并处理其余部分。

## 核心概念

### OpenAI-compatible proxy shape

每个人都说 OpenAI-shape。Routing gateway 暴露 `/v1/chat/completions`，接受 OpenAI schema，并在内部代理到 Anthropic / Gemini / Cohere / Ollama / anything。Client 不关心。

### Model aliases

你的代码不写 `claude-3-5-sonnet-20251022`，而是写 `our_smart_model`。Gateway 把 aliases 映射到真实模型。当 Anthropic 发布 Claude 4，你在 server-side 改 alias；代码不动。

### Fallback chains

```text
primary: openai/gpt-4o
on 5xx: anthropic/claude-3-5-sonnet
on 5xx: google/gemini-1.5-pro
on 5xx: refuse
```

Gateways 在 config 中定义这个。Retries 会计入 budget，避免 fallback cascades 让 cost 爆炸。

### Semantic caching

相同或近似相同 prompts 命中 cache，而不是 provider。重复 agent loops 上可节省 30 到 60%。Keys 是 embedding-based；近似 prompt 共享 cache slot。

### Guardrails

Gateway-level：

- **PII redaction.** 在发送 prompts 前做 regex 或 ML-based pass。
- **Policy violations.** 拒绝包含 prohibited content 的 prompts。
- **Output filters.** 清理 completions 中的泄漏。

Portkey 和 Kong 都提供 opinionated guardrails。LiteLLM 把它们作为 optional。

### Per-key rate limits

一个 API key = 一个 team。Per-key budgets 防止一个 team 消耗共享 quota。大多数 gateways 支持这一点。

### Self-hosted vs managed trade-offs

| Factor | LiteLLM（self-hosted） | OpenRouter（managed） | Portkey（production） |
|--------|----------------------|----------------------|----------------------|
| Code | Open source, Python | Managed SaaS | Open source（Mar 2026）+ managed |
| Setup | Deploy a proxy | Sign up | Either |
| Providers | 100+ | 300+ | 100+ |
| Billing | Your own keys | OpenRouter credits | Your own keys |
| Observability | OpenTelemetry | Dashboard | Full OTel + PII redaction |
| Best for | 想要完全控制的团队 | Rapid prototyping | Production with compliance |

当你有 SRE team 并想要 data sovereignty 时，LiteLLM 胜出。想要单个 subscription 且无 infra 时，OpenRouter 胜出。需要开箱 guardrails 和 compliance 时，Portkey 胜出。

### Cost tracking

每个 request 携带 `provider`、`model`、`input_tokens`、`output_tokens`。乘以 per-model per-token prices（从 gateway 维护的 pricing sheet 拉取）。按 per-user / per-team / per-project 聚合。

### MCP plus routing

Gateway 可以同时 route LLM calls 和 MCP sampling requests。当 sampling request 的 modelPreferences 偏好某个模型时，gateway 翻译到正确 backend。这里 Phase 13 · 17（MCP gateway）与本课 routing gateway 有时会合并成一个 service。

### Routing strategies

- **Static priority.** 列表第一个；出错时 fallback。
- **Load balancing.** Round-robin 或 weighted。
- **Cost-aware.** 选择满足 latency / quality 的最便宜模型。
- **Latency-aware.** 选择过去 N 分钟最快模型。
- **Task-aware.** Prompt classifier 把 coding route 到一个模型，把 summarization route 到另一个。

## 实际使用

`code/main.py` 用约 150 行实现 routing gateway：接受 OpenAI-shaped requests，翻译成 per-provider stubs，运行 priority fallback chain，跟踪 per-request cost，并对 inputs 应用 PII redaction pass。用三个场景运行它：normal request、primary-provider outage triggering fallback、PII leakage caught by redaction。

重点看：

- `ROUTES` dict：alias -> priority-ordered list of concrete providers。
- Fallback loop 在 5xx 上 retry。
- Cost tracker 将 token usage 乘以 per-model rates。
- PII redactor 在 forwarding 前清理 SSN-shaped patterns。

## 交付成果

本课产出 `outputs/skill-routing-config-designer.md`。给定 workload profile（latency、cost、compliance），该 skill 会选择 LiteLLM / OpenRouter / Portkey，并生成 routing config。

## 练习

1. 运行 `code/main.py`。触发 outage scenario；确认 fallback 落到第二 provider，并且 cost attribution 正确。

2. 添加 semantic caching：prompt 的 SHA256 作为 lookup key；cache hit 立即返回。测量 repeated call 上的 cost savings。

3. 添加 prompt classifier，把“code ...” prompts route 到偏 intelligence 的 alias，把“summarize ...” prompts route 到偏 speed 的 alias。

4. 设计 per-team budgets：每个 team 有 monthly spend cap；cap 命中后 gateway 拒绝 requests。选择 enforcement granularity（per-request 或 windowed）。

5. 并排阅读 LiteLLM、OpenRouter 和 Portkey docs。说出每家提供而另外两家不提供的一个 feature。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Routing gateway | “LLM proxy” | 位于多个 providers 前的一层 one-API-surface |
| OpenAI-compatible | “Speaks the OpenAI schema” | 接受 `/v1/chat/completions` shape，并翻译到任意 backend |
| Model alias | “our_smart_model” | 代码中的名字，由 gateway 映射到具体模型 |
| Fallback chain | “Retry list” | 失败时按顺序尝试的 providers 列表 |
| Semantic caching | “Prompt-embedding cache” | Key 是 prompt embedding；near-duplicates 共享 cache hit |
| Guardrails | “Input/output filters” | Redact PII，reject policy violations |
| Per-key rate limit | “Team budget” | 作用域为 API key 的 quota |
| Cost tracking | “Per-request spend” | 聚合 token usage x price per model |
| LiteLLM | “The open proxy” | 可 self-host 的 OSS routing gateway |
| OpenRouter | “The managed SaaS” | 带 credit-based billing 的 hosted gateway |
| Portkey | “The production option” | Open-source + managed，内置 guardrails |

## 延伸阅读

- [LiteLLM — docs](https://docs.litellm.ai/) — self-hosted routing gateway
- [OpenRouter — quickstart](https://openrouter.ai/docs/quickstart) — managed routing SaaS
- [Portkey — docs](https://portkey.ai/docs) — production routing with guardrails
- [TrueFoundry — LiteLLM vs OpenRouter](https://www.truefoundry.com/blog/litellm-vs-openrouter) — decision guide
- [Relayplane — LLM gateway comparison 2026](https://relayplane.com/blog/llm-gateway-comparison-2026) — vendor survey
