# Batch APIs：作为行业标准的 50% Discount

> 每个 major provider 都交付了带 50% discount 和约 24-hour turnaround 的 async batch API。OpenAI、Anthropic、Google，以及多数 inference platforms（Fireworks batch tier、Together batch）都实现了同一 pattern。将 batch 与 prompt caching 叠加，overnight pipelines 可以降到 synchronous-uncached cost 的约 10%。规则极其简单：如果不是 interactive，就应该在 batch 上。Content generation pipelines、document classification、data extraction、report generation、bulk labeling、catalog tagging：任何能容忍 24-hour latency 的工作，在迁移到 batch 之前都是把钱留在桌上。2026 production pattern 是把每个新 LLM workload triage 到三条 lanes：interactive（synchronous with caching）、semi-interactive（async queue with fallback）、batch（overnight, cached input stacked）。假装 interactive 但能容忍数分钟 latency 的 workloads 浪费最多。

**类型：** 学习
**语言：** Python (stdlib, toy batch-vs-sync cost simulator)
**先修：** Phase 17 · 14 (Prompt & Semantic Caching)
**时间：** ~45 分钟

## 学习目标

- 说出三个 provider batch APIs（OpenAI、Anthropic、Google）以及共同的 50% discount + 24h turnaround guarantees。
- 计算 overnight classification workload 上叠加 batch + cached-input 后的成本，并与 synchronous-uncached baseline 对比。
- 将 workload triage 到 interactive / semi-interactive / batch，并为 lane 辩护。
- 说出两个 traps：partial interactivity（用户期望快于 24h）和 output-schema drift（batch file format 因 provider 而异）。

## 要解决的问题

你的团队交付 nightly report generation pipeline。50,000 份 documents，每份 summarize，聚类 summaries，起草 executive brief。同步运行需要 4 小时，花 $2,000/night。你听说了 batch APIs。

batch 给你 50% off。你还在 system prompt（所有 50k calls 共享）上启用 prompt caching。叠加后，账单降到 $180/night：约 baseline 的 9%。同一个 pipeline，三个 config changes。

Batch 是 LLM cost toolkit 中最便宜、却没人拉动的 lever。原因主要是组织性的：团队以为 “real-time”，但 SLA 实际上是 “by morning”。本课讲的是别把 90% 账单留在桌上。

## 核心概念

### 三个 batch APIs

**OpenAI Batch API**：上传包含 request 列表的 JSONL file。承诺 24-hour turnaround（实践中通常约 2-8 小时）。input 和 output tokens 50% discount。`/v1/batches` endpoint。cache-eligible inputs 还能在此基础上得到 cached-input pricing。

**Anthropic Message Batches**：JSONL upload。24-hour turnaround。50% discount。支持 `cache_control`：cache writes 是显式的，reads 在 batch 内自动发生。

**Google Vertex AI Batch Prediction**：BigQuery 或 GCS input。Gemini 类似 50% discount。与 Vertex pipelines 集成。

### Semantic：asynchronous，不是 slow

Batch 是 “我承诺 24 小时内返回”，不是“这会花 24 小时”。典型 P50 是 2-6 小时。Provider 在 GPU inventory underutilized 的 off-peak windows 调度你的 batch。

### 与 caching 叠加

一个 50k-document summarization，拥有相同 4K-token system prompt：

- Synchronous uncached：50000 × ($input × 4000 + $output × 200)，按 full rates。
- Synchronous cached：system prompt 在第一次 write 后被 cached；剩余 49999 次得到 10x cheaper input。
- Batch cached：上述全部，再加上 read 和 write 的 50% discount。

叠加：batch + cache = 约 sync uncached bill 的 10%。任何 overnight 运行并有 shared system prompt 的 workload 都应该使用它。

### Workload triage

**Interactive**：用户等待 response。TTFT 重要。Synchronous call with prompt caching。不能 batch。

**Semi-interactive**：用户提交 task，几分钟后回来查看。Async queue，如果 batch 不可用则 fallback to sync。想象 moderate-volume RAG indexing。

**Batch**：用户预期 “by morning” 或 “next hour” 得到 results。Content pipelines、classification at scale、offline analysis。总是 batch，总是 stack caching。

常见错误：因为 pipeline 是 production，就把一切分类为 interactive。Production 不是 latency spec；SLA 才是。

### Partial-interactivity trap

有些 features 看起来 interactive，但能容忍 5-10 分钟。例如：nightly customer health report 上有 “refresh” button。用户点击 refresh；等待 10 分钟可以接受。团队却把它做成 synchronous。50 个 concurrent refreshes 的成本，是 batched-and-delivered-via-email 的 10x。

要问的问题：“24-hour 对这个用户意味着什么？” 如果答案是 “they wouldn't notice”，batch it。

### Output-schema trap

Batch file formats 因 provider 而异：

- OpenAI：JSONL，一行一个 request。
- Anthropic：JSONL，一行一条 message；response format embedded。
- Vertex：BigQuery table 或 GCS prefix with TFRecord。

跨 providers 写 “one batch client” 意味着每个 provider 都需要 adapter code。宣称 multi-provider batch 的 gateways（Portkey、LiteLLM 某些 tiers）仍然只是 thin-wrap raw format。

### 你应该记住的数字

- providers 的 batch discount：input + output 统一 50% flat。
- Turnaround SLA：保证 24 小时，典型 P50 是 2-6 小时。
- 叠加 batch + cached input：约为 sync uncached cost 的 10%。
- Workload triage rule：如果 24h latency 可接受，always batch。

## 实际使用

`code/main.py` 为 50k-document workload 计算 sync、sync+cache、batch、batch+cache 的成本。报告 $ 和 percent savings。

## 交付成果

本课产出 `outputs/skill-batch-triager.md`。给定 workload characteristics，它会 triage 到 interactive/semi/batch 并估算 savings。

## 练习

1. 运行 `code/main.py`。对 100k-doc pipeline、3K-token system prompt、500-token output，计算 full stack（batch + cache）相对 sync baseline 的 savings。
2. 选择你熟悉真实产品中的三个 features。将每个 triage 到 interactive/semi/batch。
3. 用户投诉 report 花了 3 小时。这是 batch mis-triage 还是 legitimate interactive？写出 decision criterion。
4. 你的 batch API return SLA 是 24h，但 P99 是 20 小时。你如何向用户沟通？edge case 下 downstream system behavior 是什么？
5. 计算 break-even：shared-prefix length 到多少时，batch + cache 会比 overnight 在自有 reserved GPU 上运行更便宜？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Batch API | “async discount” | 24h turnaround 下 50% off |
| JSONL | “batch format” | 一行一个 JSON request；OpenAI/Anthropic standard |
| Message Batches | “Anthropic batch” | Anthropic batch API 的 product name |
| Batch prediction | “Vertex batch” | Vertex AI 的 batch API product |
| Turnaround SLA | “24h promise” | guarantee，不是 typical；typical 是 2-6h |
| Workload triage | “interactivity decision” | Interactive / semi / batch routing decision |
| Output schema | “response format” | per-provider JSONL layout；not portable |
| Stacked discount | “batch + cache” | 两者都适用时，约为 uncached sync bill 的 10% |

## 延伸阅读

- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) — JSONL format and `/v1/batches` semantics。
- [Anthropic Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) — batch format and `cache_control` interaction。
- [Vertex AI Batch Prediction](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/batch-prediction) — Gemini batch semantics。
- [Finout — OpenAI vs Anthropic API Pricing 2026](https://www.finout.io/blog/openai-vs-anthropic-api-pricing-comparison)
- [Zen Van Riel — LLM API Cost Comparison 2026](https://zenvanriel.com/ai-engineer-blog/llm-api-cost-comparison-2026/)
