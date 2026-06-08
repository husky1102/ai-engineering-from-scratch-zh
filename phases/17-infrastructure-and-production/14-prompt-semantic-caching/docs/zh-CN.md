# Prompt Caching 与 Semantic Caching Economics

> **Pricing snapshot dated 2026-04.** 下面的数值 claims 反映本课发布时捕获的 vendor rate cards；在下游引用前请对照链接文档重新验证。

> Caching 发生在两层。L2（provider-level）prompt/prefix caching 会为 repeated prefixes 复用 attention KV：Anthropic 的 prompt-caching docs 宣称长 prompts 上最高可减少 90% 成本和 85% latency；Claude 3.5 Sonnet cache reads 为 $0.30/M，而 fresh 为 $3.00/M，5-minute TTL，1-hour TTL option 有 2x write premium（docs.anthropic.com, 2026-04）。OpenAI prompt caching 对 ≥1024 tokens 的 prompts 自动应用，cached input 相比 fresh 约 90% discount（platform.openai.com, 2026-04）；具体 per-model cached rate 取决于 live rate card。L1（app-level）semantic caching 在 embedding similarity 命中时完全跳过 LLM。Vendor “95% accuracy” 指 match correctness，不是 hit rate：报告的 production hit rates 从 10%（open-ended chat）到 70%（structured FAQ）不等；两个 provider 都没有发布 official baseline，所以把这些当作 community telemetry，而不是 guarantees。production pitfalls：parallelization 会杀死 caching（first cache write 完成前发出的 N 个 parallel requests 会使 spend 膨胀数倍），prefix 内的 dynamic content 会完全阻止 cache hits。ProjectDiscovery 报告称，通过把 dynamic text 移出 cacheable prefix，hit rate 从 7% 提升到 74%（2025-11）。

**类型：** 学习
**语言：** Python (stdlib, toy two-layer cache simulator)
**先修：** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 06 (SGLang RadixAttention)
**时间：** ~60 分钟

## 学习目标

- 区分 L2 prompt/prefix caching（provider 端 KV reuse）与 L1 semantic caching（similar prompts 上绕过 LLM）。
- 解释 Anthropic 的 `cache_control` 显式标记，以及两种 TTL options（5-min vs 1-hour）及其 price multipliers。
- 给定 hit rate、prompt/response mix 和 token prices，计算 expected monthly savings。
- 说出会让账单膨胀 5-10x 的 parallelization anti-pattern，以及让 hit rate 崩塌的 dynamic-content anti-pattern。

## 要解决的问题

你给 RAG service 加了 prompt caching。账单没有变化。你测量 hit rate：7%。你的 prompts 看起来静态，其实不是：system prompt 包含精确到分钟的 current date、request ID，以及为了 diversity 随机重排的 examples。每个 request 都写一个新 cache entry，读零次。

另外，你的 agent 为每个用户问题运行十个 parallel tool calls。十个请求都在第一次 cache write 完成前到达 provider。十次写入，零次读取。你的账单比 “with caching” 本应花费的金额高 5-10x。

Caching 是 protocol，不是 flag。两层，两种不同 failure modes。

## 核心概念

### L2：provider prompt/prefix caching

Provider 存储 cacheable prefix 的 attention KV，并在下一次请求中复用匹配 prefix。你只支付一次 write cost，reads 几乎免费。

**Anthropic（Claude 3.5 / 3.7 / 4 series）**：request 中显式 `cache_control` marker。你标记哪些 blocks 可 cache。TTL：5-minute（write costs 1.25x base）或 1-hour（write costs 2x base）。Cache reads：Claude 3.5 Sonnet 上 $0.30/M vs $3.00/M fresh，即便宜 10x（docs.anthropic.com，截至 2026-04）。不同 model 费率不同（Opus/Haiku separately published）；始终交叉核对 live pricing page。

**OpenAI**：对 prompts ≥1024 tokens 自动 caching（platform.openai.com, 2026-04）。无 explicit flag。当前 gpt-4o/gpt-5 rate cards 上 cached input 约比 fresh 便宜 10x。docs 和 release notes 都没有发布 official hit-rate baseline；community reports 在 careful prompt design 下集中在 30-60%。监控 `usage.cached_tokens` 来测量你自己的。

**Google（Gemini）**：通过 explicit API 做 context caching；1M-token context 意味着 caching 更划算。

**Self-hosted（vLLM, SGLang）**：Phase 17 · 06 覆盖 RadixAttention：同一 pattern 位于你自己的 compute 上。

### L1：app-level semantic caching

在调用 LLM 之前，hash prompt、embed 它，并寻找相似的 cached request（cosine similarity 高于 threshold，通常 0.95+）。命中时返回 cached response。未命中时调用 LLM，并 cache 结果。

Open-source：Redis Vector Similarity、GPTCache、Qdrant。Commercial：Portkey Cache、Helicone Cache。

Vendor accuracy claims 指返回的 cached response 语义上合适的频率，而不是 hit 频率。Production hit rates：

- Open-ended chat：10-15%。
- Structured FAQ / support：40-70%。
- Code questions：20-30%（小 variants 会杀死 hits）。
- Voice agents repeating prompts：50-80%（voice normalization fixed set）。

### Parallelization anti-pattern

你的 agent 并行发起 10 个 tool calls。十个都有相同的 4K-token system prompt。Anthropic cache writes 是 per-request；first cache-write 会在 provider 看到 prompt 后约 300 ms 完成。Requests 2-10 在同一个 millisecond window 到达，每个都看到 cache miss。你支付 10 次 write premiums，0 次 read discounts。

修复：batch with sequential-first：先单独发 request 1，等 1 的 cache populated 后再发 2-10。给第一个 tool call 增加 300 ms；节省 5-10x 账单。

### Dynamic content anti-pattern

你的 system prompt 看起来像：

```text
You are a helpful assistant. The current time is 14:32:17.
User ID: abc123. Today is Tuesday...
```

每个 request 都 unique。每个 request 都写。零 hits。

修复：把真正静态的东西移到 cacheable prefix；把 dynamic content append 到 cache boundary 之后：

```text
[cacheable]
You are a helpful assistant. [rules, examples, instructions]
[/cacheable]
[dynamic, not cached]
Current time: 14:32:17. User: abc123.
```

ProjectDiscovery 用这种方式把 cache hit rate 从 7% 提升到 74%，并发布了 anatomy。

### 为 overnight workloads 叠加 batch + cache

Batch APIs（Phase 17 · 15）提供 24-hour turnaround 下的 50% discount。cached input 再叠加，会在此基础上再带来约 10x。Overnight classification、labeling 和 report generation workloads 可以降到 synchronous-uncached cost 的约 10%。

### 你应该记住的数字

Pricing points 是从 linked vendor docs 抓取的 2026-04 状态，并且每几个月都会漂移：依赖它们前请重新检查。

- Anthropic cached read：Claude 3.5 Sonnet 上 $0.30/M，约比 fresh input 便宜 10x（docs.anthropic.com）。
- Anthropic cache write premium：1.25x（5-min TTL）或 2x（1-hour TTL）。
- OpenAI auto-cache：适用于 ≥1024 tokens 的 prompts；当前 rate cards 上 cached input 价格约为 fresh input 的 10%（platform.openai.com）。
- Semantic cache hit rate（community-reported）：open chat 约 10%；structured FAQ 最高约 70%。不是 vendor-documented baseline。
- ProjectDiscovery：通过把 dynamic 移出 prefix，hit rate 7% → 74%（project blog, 2025-11）。
- Parallelization anti-pattern：当 N 个 parallel requests 错过 first cache write 时，典型报告有 5-10x bill inflation。

## 实际使用

`code/main.py` 模拟 mixed workloads 上的 L1 + L2 caching。报告 hit rates、bill，并展示 parallelization penalty。

## 交付成果

本课产出 `outputs/skill-cache-auditor.md`。给定 prompt template 和 traffic，它会 audit cacheability 并建议 restructure。

## 练习

1. 运行 `code/main.py`。切换 parallelization flag。账单变化多少？
2. 你的 system prompt 有 date。把它移出去。展示 before/after hit rate math。
3. 给定 request arrival rate，计算 1-hour TTL（2x write）与 5-minute TTL（1.25x write）的 break-even。
4. Semantic cache 在 0.95 threshold 命中 20%。在 0.85 命中 50%，但出现 incorrect cached responses。选择正确 threshold 并说明理由。
5. 每个用户问题 batch 10 个 parallel sub-queries。在不增加 end-to-end latency 的情况下，重写为 cache-friendly。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| L2 prompt cache | “prefix cache” | Provider 为 repeated prefix 存储 KV |
| `cache_control` | “Anthropic cache marker” | 标记 cacheable blocks 的 explicit attribute |
| Cache write premium | “write tax” | first miss-to-cache 的额外成本（1.25x 或 2x） |
| L1 semantic cache | “embedding cache” | 调用 LLM 前的 app-level hash-and-embed |
| GPTCache | “LLM caching lib” | 流行 OSS L1 cache library |
| Cache hit rate | “hits / total” | 由 cache 服务的 requests fraction |
| Parallelization anti-pattern | “the N-write trap” | N 个 parallel requests miss cache N 次 |
| Dynamic content trap | “the time-in-prompt trap” | prefix 中的 dynamic bytes 会杀死 hit rate |
| RadixAttention | “intra-replica cache” | SGLang 的 prefix-cache implementation |

## 延伸阅读

- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — official `cache_control` semantics and TTLs。
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching) — automatic caching behavior and eligibility。
- [TianPan — Semantic Caching for LLMs Production](https://tianpan.co/blog/2026-04-10-semantic-caching-llm-production)
- [ProjectDiscovery — Cut LLM Costs 59% With Prompt Caching](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)
- [DigitalOcean / Anthropic — Prompt Caching](https://www.digitalocean.com/blog/prompt-caching-with-digital-ocean)
