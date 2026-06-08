# AI Gateways：LiteLLM、Portkey、Kong AI Gateway、Bifrost

> gateway 位于你的 apps 和 model providers 之间。核心 features 是 provider routing、fallback、retries、rate limiting、secret references、observability、guardrails。2026 market split：**LiteLLM** 是 MIT OSS，100+ providers，OpenAI-compatible，但在约 2000 RPS 附近崩溃（published benchmarks 中 8 GB memory、cascading failures）；最适合 Python、<500 RPS、dev/prototyping。**Portkey** 定位 control-plane（guardrails、PII redaction、jailbreak detection、audit trails），2026 年 3 月转为 Apache 2.0 open-source，20-40 ms latency overhead，$49/mo production tier。**Kong AI Gateway** 基于 Kong Gateway：Kong 自己在同 12 CPUs 上的 benchmark 显示，它比 Portkey 快 228%，比 LiteLLM 快 859%；定价 $100/model/month（Plus tier 最多 5 个）；如果你已经在用 Kong，适合 enterprise。**Bifrost**（Maxim AI）：automatic retries with configurable backoff，OpenAI 429 时 fallback to Anthropic。**Cloudflare / Vercel AI Gateways**：managed、zero-ops、basic retry。Data residency 驱动 self-host decision；Portkey 和 Kong 位于中间，提供 OSS + optional managed。

**类型：** 学习
**语言：** Python (stdlib, toy gateway-routing simulator)
**先修：** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 16 (Model Routing)
**时间：** ~60 分钟

## 学习目标

- 枚举六个 core gateway features（routing、fallback、retries、rate limits、secrets、observability、guardrails）。
- 将四个 2026 gateways（LiteLLM、Portkey、Kong AI、Bifrost）映射到 scale ceilings 和 use cases。
- 引用 Kong benchmark（228% vs Portkey，859% vs LiteLLM），并解释为什么它对 >500 RPS 重要。
- 在 data residency 和 ops budget 下选择 self-hosted vs managed。

## 要解决的问题

你的 product 调用 OpenAI、Anthropic 和 self-hosted Llama。每个 provider 都有不同 SDK、error model、rate limit 和 auth scheme。你想要 failover（如果 OpenAI 429s，尝试 Anthropic）、单一 credential store、统一 observability，以及 per tenant rate limits。

在 app layer 重新发明这些，会把每个 service 和每个 provider 耦合起来。gateway layer 将它 consolidation 到一个 process 和一个 API（通常 OpenAI-compatible）中，再 fan out 到 providers。

## 核心概念

### 六个 core features

1. **Provider routing**：OpenAI、Anthropic、Gemini、self-hosted 等位于一个 API 之后。
2. **Fallback**：429、5xx 或 quality failure 时 retry elsewhere。
3. **Retries**：exponential backoff，bounded attempts。
4. **Rate limits**：per-tenant、per-key、per-model。
5. **Secret references**：runtime 从 vault 拉取 credentials（永远不在 app 中）。
6. **Observability**：OTel + GenAI attributes（Phase 17 · 13）+ cost attribution。
7. **Guardrails**：PII redaction、jailbreak detection、allowed-topics filters。

### LiteLLM：MIT OSS，Python

- 100+ providers、OpenAI-compatible、router config、fallback、basic observability。
- 在 Kong benchmark 中约 2000 RPS 崩溃；8 GB memory footprint，sustained load 下 cascading failures。
- Best fit：Python app、<500 RPS、dev/staging gateways、experimental routing。
- Cost：OSS 为 $0；存在 cloud free tier。

### Portkey：control plane positioning

- 截至 2026 年 3 月为 Apache 2.0 OSS。Guardrails、PII redaction、jailbreak detection、audit trails。
- 每个 request 20-40 ms latency overhead。
- production tier $49/mo，带 retention + SLA。
- Best fit：需要 guardrails + observability bundled 的 regulated industries。

### Kong AI Gateway：scale play

- 基于 Kong Gateway（成熟 API gateway product，lua+OpenResty）。
- Kong 自己在 12-CPU equivalent 上的 benchmark：比 Portkey 快 228%，比 LiteLLM 快 859%。
- Pricing：$100/model/month，Plus tier 最多 5 个。
- Best fit：已经在用 Kong；>1000 RPS；愿意 license。

### Bifrost（Maxim AI）

- 带 configurable backoff 的 automatic retries。
- OpenAI 429 时 fallback to Anthropic 是 canonical recipe。
- 较新的 entrant；commercial。

### Cloudflare AI Gateway / Vercel AI Gateway

- Managed、zero-ops。Basic retry 和 observability。
- Best fit：Cloudflare/Vercel 上 edge-serving JavaScript apps。
- 相比 Kong/Portkey，guardrails 和 rate limits 更有限。

### Self-hosted vs managed

Data residency 是 forcing function。Healthcare 和 finance 默认 self-host（LiteLLM、Portkey OSS 或 Kong）。Consumer products 默认 managed（Cloudflare AI Gateway）或 middle-tier（Portkey managed）。Hybrid：regulated tenant 用 self-hosted，其他用 managed。

### Latency budget

- LiteLLM：typical overhead 5-15 ms。
- Portkey：20-40 ms overhead。
- Kong：3-8 ms overhead。
- Cloudflare/Vercel：1-3 ms overhead（edge advantage）。

Gateway latency 会直接加到 TTFT 上。对 TTFT P99 < 100 ms SLA，选 Kong 或 Cloudflare。对 P99 < 500 ms，任意都可。

### Rate-limit semantics matter

Simple token-bucket 能工作到 moderate scale。Multi-tenant 需要 sliding-window + burst allowance + per-tenant tiering。LiteLLM 提供 token-bucket；Kong 提供 sliding-window；Portkey 提供 tiered。

### Gateway + observability + routing 可以组合

Phase 17 · 13（observability）+ 16（model routing）+ 19（gateways）在 production 中是同一层。选择覆盖三者的一个 tool，或谨慎接线：大多数 2026 deployments 会组合 Helicone（observability）或 Portkey（guardrails）与 Kong（scale）来分担 roles。

### 你应该记住的数字

- LiteLLM：约 2000 RPS 崩溃，8 GB memory。
- Portkey：20-40 ms overhead；自 2026 年 3 月起 Apache 2.0。
- Kong：比 Portkey 快 228%，比 LiteLLM 快 859%。
- Kong pricing：$100/model/month，Plus tier 最多 5 个。
- Cloudflare/Vercel：edge 上 1-3 ms overhead。

## 实际使用

`code/main.py` 在 429/5xx injection 下模拟跨 3 providers 的 gateway routing with fallback。报告 latency、retry rate 和 fallback hit rate。

## 交付成果

本课产出 `outputs/skill-gateway-picker.md`。给定 scale、ops posture、compliance、latency budget，选择 gateway。

## 练习

1. 运行 `code/main.py`。配置 OpenAI→Anthropic→self-hosted fallback。在 5% provider error rate 下，expected hit rate 是多少？
2. 你的 SLA 是 300 ms baseline 上 TTFT P99 < 200 ms。哪些 gateways 仍在 budget 内？
3. healthcare customer 要求 self-hosted + PII redaction + audit。选择 Portkey OSS 还是 Kong。
4. 比较 LiteLLM vs Kong：团队应在什么 RPS ceiling 迁移？
5. 为 multi-tenant SaaS 设计 rate-limit policy：free tier、trial tier、paid tier。Token-bucket 还是 sliding-window？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Gateway | “API broker” | 位于 apps 和 providers 之间的 process |
| LiteLLM | “the MIT one” | Python OSS，100+ providers，2K RPS 崩溃 |
| Portkey | “guardrails gateway” | Control plane + observability，Apache 2.0 |
| Kong AI Gateway | “the scale one” | 基于 Kong Gateway，benchmark leader |
| Bifrost | “Maxim's gateway” | Retries + Anthropic fallback recipe |
| Cloudflare AI Gateway | “edge managed” | Edge-deployed managed gateway，zero-ops |
| PII redaction | “data scrub” | 发送给 model 前用 Regex + NER mask |
| Jailbreak detection | “prompt injection guard” | user input 上的 classifier |
| Audit trail | “regulated log” | 每次 LLM call 的 immutable record |
| Token-bucket | “simple rate limit” | refill-based rate limiter |
| Sliding-window | “precise rate limit” | Time-windowed rate limiter；fairness 更好 |

## 延伸阅读

- [Kong AI Gateway Benchmark](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — AI Gateways 2026 Comparison](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — Top LLM Gateway Tools 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway docs](https://docs.konghq.com/gateway/latest/ai-gateway/)
