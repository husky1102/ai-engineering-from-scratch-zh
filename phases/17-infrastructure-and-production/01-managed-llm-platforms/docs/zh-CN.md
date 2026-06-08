# 托管 LLM 平台：Bedrock、Vertex AI、Azure OpenAI

> 三家 hyperscalers，三种不同策略。AWS Bedrock 是 model marketplace，把 Claude、Llama、Titan、Stability、Cohere 放在一个 API 后面。Azure OpenAI 是独家 OpenAI partnership 加上用于 dedicated capacity 的 Provisioned Throughput Units (PTUs)。Vertex AI 以 Gemini 为先，拥有最强的 long-context 和 multimodal 叙事。2026 年，Artificial Analysis 在 Llama 3.1 405B 等价部署上测得 Azure OpenAI median 约 50 ms，Bedrock 约 75 ms，PTUs 解释了这个差距，因为 dedicated capacity 胜过 shared on-demand。决策规则不是“哪个最快”，而是“哪个 model catalog 和 FinOps surface 匹配我的产品”。本课教你把 tradeoffs 写下来再选择，而不是凭感觉。

**类型：** Learn
**语言：** Python (stdlib, toy cost-and-latency comparator)
**先修：** Phase 11 (LLM Engineering), Phase 13 (Tools & Protocols)
**时间：** ~60 minutes

## 学习目标

- 说出三种平台策略（marketplace vs exclusive vs Gemini-first），并把每种策略匹配到一个 product use case。
- 解释 Azure OpenAI 中的 Provisioned Throughput Units (PTUs) 购买的是什么，以及为什么 on-demand Bedrock 在 405B 规模上通常会慢约 25 ms。
- 画出每个平台的 FinOps attribution surface（Bedrock Application Inference Profiles vs Vertex project-per-team vs Azure scopes + PTU reservations）。
- 写下一条 “two-provider minimum” policy，并解释为什么 single-vendor lock-in 是 2026 年代价高昂的错误。

## 要解决的问题

你为产品选择了 Claude 3.7 Sonnet。现在需要把它服务出来。你可以直接调用 Anthropic API，也可以通过 AWS Bedrock 调用，或者走一个 gateway。Direct API 最简单；Bedrock 增加 BAAs、VPC endpoints、IAM 和 CloudWatch attribution。Gateway 增加跨 providers 的 failover、unified billing 和 rate limits。

更深的问题是 catalog。如果同一个产品中需要 Claude、Llama 和 Gemini，除非同时使用 Bedrock 加 Vertex 加 Azure OpenAI，否则无法从一个地方买齐。Hyperscalers 并不可互换，它们各自对谁拥有 model layer 下了不同赌注。

本课映射三种赌注、latency gap、FinOps gap 和 lock-in risk。

## 核心概念

### 三种策略

**AWS Bedrock**：marketplace。Claude (Anthropic)、Llama (Meta)、Titan (AWS first-party)、Stability (image)、Cohere (embeddings)、Mistral，以及 image 和 embedding sub-catalogs。一个 API，一个 IAM surface，一个 CloudWatch export。Bedrock 的赌注是：客户想要 optionality 多于想要单一模型。

**Azure OpenAI**：exclusive partnership。你在 Azure datacenters 中获得 GPT-4 / 4o / 5 / o-series、DALL·E、Whisper，以及 OpenAI models 的 fine-tuning。“Azure OpenAI Service” catalog 中没有非 OpenAI models，这些会去 Azure AI Foundry（单独产品）。Azure 的赌注是 OpenAI 仍然处在 frontier，客户想要围绕这段特定关系的 enterprise controls。

**Vertex AI**：Gemini first，其他次之。Gemini 1.5 / 2.0 / 2.5 Flash and Pro，加上 Model Garden（third-party）。Vertex 的赌注是 multimodal long-context，1M-token Gemini context 是差异化点。

### 规模下的 latency gap

Artificial Analysis 持续运行 benchmarks。在等价的 Llama 3.1 405B deployments（shared on-demand）上，Azure OpenAI median first-token latency 约为 50 ms；Bedrock 约为 75 ms。这个差距不是 AWS 失败，而是 capacity model 差异。Azure 销售 PTUs (Provisioned Throughput Units)，为你的 tenant 预留 GPU capacity。Bedrock 的等价物（Provisioned Throughput）也存在，但起价约为每 unit $21/hour，大多数客户仍留在 shared on-demand。

On-demand shared capacity 会和其他所有客户的流量竞争。Dedicated capacity 不会。如果你的 product SLA 是 TTFT < 100 ms at P99，要么购买 Azure PTUs，要么购买 Bedrock Provisioned Throughput，要么接受默认 variance。

### Provisioned Throughput 经济性

Azure PTUs：一块预留的 inference compute。对 predictable workloads，相比 on-demand 最高可节省约 70%。费用按小时固定，不管有没有 traffic，idle 时也要为 reservation 付费。Break-even 通常在 40-60% sustained utilization 左右。

Bedrock Provisioned Throughput：根据 model 和 region 不同，每小时 $21-$50。数学类似，break-even 约在半峰值 utilization。需要 monthly commitment。

Vertex provisioned capacity 按 Gemini SKU 销售；价格因 model 和 region 而异，公开程度更低。

### FinOps surface：真正的差异化点

**Bedrock Application Inference Profiles** 是 marketplace 中最清晰的 attribution。用 `team`、`product`、`feature` 给 profile 打标签；所有 model invocations 都通过它路由；CloudWatch 无需后处理就能按 profile 拆分 cost。该功能于 2025 年加入，仍然是最细粒度的 hyperscaler native 方案。

**Vertex** attribution 是 project-per-team 加 labels-everywhere。把每个 team 建模为一个 GCP project，在每个 resource 上加 labels，并使用 BigQuery Billing Export + DataStudio 做 rollups。工作更多，但 BigQuery 让你可以对 cost data 做任意 SQL。

**Azure** 依赖 subscription/resource-group scopes 加 tags，PTU reservations 是 first-class cost object。Tags 从 resource groups 继承，而不是从 requests 继承，因此 per-request attribution 需要 Application Insights custom metrics，或一个负责写入 headers 的 gateway。

模式是：Bedrock 原生最清晰，Vertex 通过 BigQuery 最灵活，Azure 若不额外 instrumentation 则最不透明。

### Lock-in 是 2026 年的风险

当一个模型占据主导时，single-hyperscaler commitment 没问题。2026 年，frontier 每月都在移动：某个季度是 Claude 3.7，下个季度是 Gemini 2.5，再下个季度是 GPT-5。锁定到一个平台，会把你挡在三分之二的 frontier 之外。

有效团队采用的模式是：任何 product-critical LLM call 都至少 two-provider minimum。Bedrock 加 Azure OpenAI 是常见组合，一个提供 Claude，一个提供 GPT，在同一个 gateway 后面做 failover。Cost uplift 可以忽略，因为 gateway 会做 optimal routing；outages（例如 Azure OpenAI 2025 年 1 月 incident、AWS us-east-1 outage）期间的 availability uplift 才是决定性的。

### Data residency、BAAs 与受监管行业

Bedrock：大多数 regions 提供 BAAs；VPC endpoints；guardrails。常见 fintech default。
Azure OpenAI：HIPAA、SOC 2、ISO 27001；EU data residency；enterprise-regulated default。
Vertex：HIPAA、GDPR、按 region 的 data residency；Google Cloud 的 compliance stack。

三者都满足基本 checkbox。差异在于 data retention policies、logs 如何处理，以及 abuse-monitoring 是否读取你的 traffic（多数默认 opt-in；enterprise 可 opt-out）。

### 你应该记住的数字

- Azure OpenAI 在 Llama 3.1 405B 等价部署上的 median TTFT：约 50 ms（使用 PTUs）。
- Bedrock on-demand median TTFT：约 75 ms。
- Bedrock Provisioned Throughput：每 unit $21-$50/hr。
- Azure PTU break-even：约 40-60% sustained utilization。
- 高 utilization 下，PTU 相比 on-demand 最高节省 70%。

## 实际使用

`code/main.py` 在一个 synthetic workload 上比较三个平台：它建模 on-demand vs PTU economics、TTFT variance 和 cost attribution fidelity。运行它，看看 PTUs 在哪里划算，以及 marketplace 的 model breadth 在哪里超过 TTFT gap。

## 交付成果

本课产出 `outputs/skill-managed-platform-picker.md`。给定 workload profile（models needed、TTFT SLA、daily volume、compliance requirements），它会推荐 primary platform、fallback 和 FinOps instrumentation plan。

## 练习

1. 运行 `code/main.py`。对于 70B class model，Azure PTU 在什么 sustained utilization 下胜过 on-demand？计算 break-even，并与广告中的 40-60% 区间比较。
2. 你的产品需要 Claude 3.7 Sonnet 和 GPT-4o。设计一个 two-provider deployment：哪个模型走哪个 hyperscaler，前面放什么 gateway，failover policy 是什么？
3. 一个受监管的医疗客户要求 BAAs、US-East data residency 和 sub-100ms P99 TTFT。选择一个平台，并用三个具体功能说明理由。
4. 你发现 Bedrock bill 本月涨了 4x，但 traffic 没变。没有 Application Inference Profiles 时，如何找到责任方？有 profiles 时，需要多久？
5. 阅读 Azure OpenAI 和 Bedrock pricing pages。对于 100M-token/month 的 Claude workload，哪个更便宜：direct Anthropic API、Bedrock on-demand，还是 Bedrock Provisioned Throughput？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Bedrock | "AWS LLM service" | 跨 Claude、Llama、Titan、Mistral、Cohere 的 model marketplace |
| Azure OpenAI | "Azure's ChatGPT" | Azure datacenters 中带 enterprise controls 的独家 OpenAI models |
| Vertex AI | "Google's LLM" | Gemini-first platform，Model Garden 提供 third-party models |
| PTU | "dedicated capacity" | Provisioned Throughput Unit，即预留 inference GPUs，按小时计价 |
| Application Inference Profile | "Bedrock tagging" | 带 tags 的 per-product cost/usage profile，CloudWatch-native |
| Model Garden | "Vertex catalog" | Vertex AI 的 third-party model section，与 Gemini 分离 |
| Two-provider minimum | "LLM redundancy" | 每条 critical LLM path 都跨 ≥2 hyperscalers 运行的 policy |
| BAA | "HIPAA paperwork" | Business Associate Agreement；PHI 必需；三家都提供 |
| Abuse monitoring | "the log watcher" | Provider-side safety scan on prompts/outputs；enterprise 可 opt-out |

## 延伸阅读

- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/) — 权威 rate card 和 Provisioned Throughput pricing。
- [Azure OpenAI Service Pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) — PTU economics 和 rate cards。
- [Vertex AI Generative AI Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing) — Gemini tiers 和 Model Garden surcharges。
- [Artificial Analysis LLM Leaderboard](https://artificialanalysis.ai/) — 跨 providers 的 continuous latency and throughput benchmarks。
- [The AI Journal — AWS Bedrock vs Azure OpenAI CTO Guide 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) — enterprise decision framework。
- [Finout — Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) — attribution mechanics side-by-side。
