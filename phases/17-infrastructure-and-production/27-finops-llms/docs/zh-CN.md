# LLMs 的 FinOps — Unit Economics 与 Multi-Tenant Attribution

> 传统 FinOps 在 LLM spend 上会失效。成本是 token-transactions，不是 resource-uptime。Tags 映射不上 —— API call 是 transaction，不是 asset。工程决策（prompt design、context window、output length）就是财务决策。2026 playbook 要在 day one instrument 三个 attribution dimensions：per-user（`user_id`）用于 seat pricing 和 expansion，per-task（`task_id` + `route`）用于 product surface cost 和 prioritization，per-tenant（`tenant_id`）用于 unit economics 和 renewal。四个 token layers —— prompt、tool、memory、response —— 一个 bucket 会隐藏 spend。Multi-tenant products 的 enforcement ladder：按 tenant rate limits（2-3x expected peak，清晰 429 + retry-after）；daily spend cap（1.5-3x contracted ceiling；触发 rate tightening + alert）；spend z-score > 4 时 kill switches（auto-pause + page on-call）。Attribution patterns：tag-and-aggregate、telemetry-joiner（trace-ID → billing；最高准确率）、sampling-and-extrapolation、model-based allocation、event-sourced、real-time streaming。Unit metric：cost per resolved query、cost per generated artifact —— 不是 $/M tokens。Retroactive tagging 总会漏；在 request creation 时 instrument。

**类型:** 学习
**语言:** Python（stdlib，带 kill switch 的玩具 cost-attribution simulator）
**先修:** Phase 17 · 13（Observability），Phase 17 · 14（Caching）
**时间:** ~60 分钟

## 学习目标

- 解释为什么传统 FinOps（tags + tiers）在 LLM spend 上失效，并说出三个新的 attribution dimensions。
- 枚举四个 token layers（prompt、tool、memory、response），以及为什么 single-bucket billing 会隐藏成本。
- 为 multi-tenant product 设计 enforcement ladder（rate → spend cap → kill switch）。
- 选择 unit metric（cost per resolved query / artifact），而不是 $/M tokens。

## 要解决的问题

你的账单显示 $40,000。你不知道：
- 哪个 tenant 花掉的。
- 哪个 product feature 驱动了这笔花费。
- 是否有 individual user 滥用。
- 问题来自 prompt bloat、tool calls，还是 memory amplification。

Provider-side 的 tag-and-aggregate 适用于 cloud resources（EC2、S3），因为 tags 会传播到 line items。LLM API calls 不会自动带 tags —— 你必须在 call site stamp user/task/tenant 并贯穿下去。Retroactive attribution 总会漏掉 edge cases。

## 核心概念

### 三个 attribution dimensions

**Per-user**（`user_id`）：谁在花费什么。驱动 seat pricing、expansion conversations，并识别 power users。

**Per-task**（`task_id` + `route`）：哪个 product surface 花了多少。驱动 feature prioritization、kill-expensive-features decisions。

**Per-tenant**（`tenant_id`）：哪个 customer 是 profitable。驱动 unit economics、renewal pricing、tier thresholds。

在 day one 就在 call site instrument 三者。Retroactive 总是更差。

### 四个 token layers

| Layer | Example | Typical % of total |
|-------|---------|---------------------|
| Prompt | system + user input | 40-60% |
| Tool | tool-call results fed back | 20-40%（agent workloads） |
| Memory | prior conversation / retrieved docs | 10-30% |
| Response | model output | 10-30% |

把四者全部放到一个 bucket，会让 optimization 失明。把它们拆到你的 attribution schema 里。

### Enforcement ladder

1. **Rate limit** per tenant。2-3x expected peak。返回带 `Retry-After` 的 429。Tenant 感到 friction；不会出现 surprise bill。

2. **Daily spend cap** per tenant。1.5-3x contracted ceiling。触发：tighten rate limit + alert customer-success。

3. **Kill switch**，当 spend z-score > 4（相对 tenant baseline）时触发。Auto-pause tenant；page on-call；escalate to ops + CS。

### Attribution patterns

- **Tag-and-aggregate**：stamp metadata headers；稍后 aggregate。简单；粗略。
- **Telemetry joiner**：通过 trace IDs 将 traces join 到 billing。最高准确率。成熟团队的做法。
- **Sampling + extrapolation**：采样 5-10%，再乘回去。用于粗略 spend 时 cost-effective；会漏 tails。
- **Model-based allocation**：用 regression 推断 cost driver。适合没有 tags 的 legacy data。
- **Event-sourced**：将 cost 作为 stream（Kafka / Kinesis）中的 events。Real-time。
- **Real-time streaming**：dashboard sub-second updates。

### Cost per X 是 unit metric

$/M tokens 是 vendor speak。Product metrics：

- Cost per resolved support ticket。
- Cost per generated article。
- Cost per successful agent task。
- Cost per user-session-minute。

把 cost 绑定到 product outcome。否则 optimization 没有锚点。

### Cost attribution trace shape

```text
trace_id: abc123
  user_id: u_42
  tenant_id: t_7
  task_id: task_classify_doc
  route: model_haiku
  layers:
    prompt_tokens: 1800
    tool_tokens: 600
    memory_tokens: 400
    response_tokens: 150
  cost_usd: 0.0135
  cached_input: true
  batch: false
```

每个 call 都 emit。存入 data lake。按 dimension aggregate。Phase 17 · 13 observability stack 就是它的落点。

### The compounded-savings stack

Stack：cache + batch + route + gateway。四者全有时：
- Cache L2（Phase 17 · 14）：input 约便宜 10x。
- Batch（Phase 17 · 15）：50% off。
- Route to cheap model（Phase 17 · 16）：60% cost reduction。
- Gateway efficiency（Phase 17 · 19）：redundancy + retries。

Best-case stacked：约为 naive baseline 的 ~5-10%。多数团队只启用了 2-3 个 levers；很少有人四个全叠。

### 你应该记住的数字

- Attribution dimensions：per-user、per-task、per-tenant。
- 四个 token layers：prompt、tool、memory、response。
- Kill switch：spend z-score > 4。
- Unit metric：cost per resolved query，而不是 $/M tokens。
- Stacked optimizations：可能达到 baseline 的 ~5-10%。

## 实际使用

`code/main.py` 模拟一个带三层 enforcement ladder 的 multi-tenant LLM service。注入一个 abusive tenant，并演示 kill switch 触发。

## 交付成果

本课产出 `outputs/skill-finops-plan.md`。给定 product 和 scale，它会设计 attribution schema 和 enforcement ladder。

## 练习

1. 运行 `code/main.py`。Kill switch 在什么 z-score 触发？你如何选择 threshold？
2. 设计一个 per-tenant、per-task cost dashboard。最先构建哪 5 个 views？
3. 你的最大 tenant 是 unit-economics-negative。提出三个干预措施，并按 customer impact 排序。
4. 为一个 support product 计算 cost per resolved ticket：3M tokens/ticket，约 800 tickets/day，GPT-5 cached rate。
5. 论证 retroactive tagging 是否可能有效。什么时候可以接受？

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|------------|----------|
| Per-user attribution | “user-level cost” | 每个 call 都 stamp `user_id` |
| Per-task attribution | “feature cost” | `task_id` + `route` 识别 product surface |
| Per-tenant attribution | “customer cost” | `tenant_id`；驱动 unit economics |
| Four token layers | “cost layers” | prompt + tool + memory + response |
| Rate limit | “429 guard” | Gateway 执行的 per-tenant ceiling |
| Daily spend cap | “daily ceiling” | 带 alert 的 tenant-scoped budget |
| Kill switch | “auto-pause” | Spend z-score > 4 触发 auto-suspension |
| Cost per resolved | “product unit metric” | Cost 绑定到 product outcome，而不是 tokens |
| Telemetry joiner | “trace-to-billing” | 最高准确率 attribution pattern |
| Stacked optimization | “cache+batch+route+gateway” | 复合 savings 达到 baseline 的 ~5-10% |

## 延伸阅读

- [FinOps Foundation — FinOps for AI Overview](https://www.finops.org/wg/finops-for-ai-overview/)
- [FinOps School — Cost per Unit 2026 Guide](https://finopsschool.com/blog/cost-per-unit/)
- [Digital Applied — LLM Agent Cost Attribution 2026](https://www.digitalapplied.com/blog/llm-agent-cost-attribution-guide-production-2026)
- [PointFive — Managed LLMs in Azure OpenAI](https://www.pointfive.co/blog/finops-for-ai-economics-of-managed-llms-in-azure-open-ai)
