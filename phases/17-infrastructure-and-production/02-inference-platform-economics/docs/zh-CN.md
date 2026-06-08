# 推理平台经济学：Fireworks、Together、Baseten、Modal、Replicate、Anyscale

> 2026 年的 inference market 已经不再是 GPU time rental。它分化为 custom silicon（Groq、Cerebras、SambaNova）、GPU platforms（Baseten、Together、Fireworks、Modal）和 API-first marketplaces（Replicate、DeepInfra）。Fireworks 在 2026 年 5 月 1 日把 GPU 每小时价格上调 $1，$4B valuation 和 10T+ tokens/day 告诉你 volume-driven model 行得通。Baseten 于 2026 年 1 月以 $5B valuation 完成 $300M Series E。竞争定位规则很简单：Fireworks 优化 latency，Together 优化 catalog breadth，Baseten 优化 enterprise polish，Modal 优化 Python-native DX，Replicate 优化 multimodal reach，Anyscale 优化 distributed Python。本课给你一个可以交给 founder 的矩阵。

**类型：** Learn
**语言：** Python (stdlib, toy per-call economics comparator)
**先修：** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 04 (vLLM Serving Internals)
**时间：** ~60 minutes

## 学习目标

- 说出三类 market segments（custom silicon、GPU platforms、API-first），并把每个 vendor 映射到对应 segment。
- 解释为什么 “per-token” API pricing model 会向 serving engine 的 cost curve 压缩，而不是向 hardware 的 cost curve 压缩。
- 跨至少三个 vendors 计算 effective cost per request，并解释什么时候 per-minute（Baseten、Modal）胜过 per-token。
- 识别给定 workload 的正确默认平台（serverless bursty、steady high-throughput、fine-tuned variants、multimodal）。

## 要解决的问题

你已经评估过托管 hyperscaler platforms。你决定需要一个更窄、更快的 provider：Fireworks 用于 latency，Together 用于 breadth，Baseten 用于 fine-tuned custom model。现在你有六个真实选择，而 pricing pages 并不对齐。Fireworks 显示 $/M tokens；Baseten 显示 $/minute；Modal 显示 $/second；Replicate 显示 $/prediction。没有 workload model，就无法逐项比较。

更糟的是，每个 pricing page 背后的 business model 都不同。Fireworks 在 shared GPUs 上运行自己的 custom engine（FireAttention）；per-token rate 反映的是它们的 utilization curve。Baseten 给你 Truss + dedicated GPUs；per-minute 反映 exclusivity。Modal 是真正的 Python serverless，按秒计费，cold starts 低于秒级。相同输出（一个 LLM response），三种不同 cost functions。

本课会建模这六个平台，并告诉你何时各自胜出。

## 核心概念

### 三个 segments

**Custom silicon**：Groq (LPU)、Cerebras (WSE)、SambaNova (RDU)。在同一模型上，decode 通常比 GPU-based cluster 快 5-10x。Per-token price 更高（2025 年末 Groq 在 Llama-70B 上约 $0.99/M），但在 latency-sensitive use cases 中难以匹敌。Groq 是 voice agents 和 real-time translation 的 production pick。

**GPU platforms**：Baseten、Together、Fireworks、Modal、Anyscale。运行在 NVIDIA（2026 年 H100、H200、B200）或有时 AMD 上。处在 “raw GPU rental”（RunPod、Lambda）和 “hyperscaler managed service”（Bedrock）之间的经济层。

**API-first marketplaces**：Replicate、DeepInfra、OpenRouter、Fal。Catalog 广，按 prediction 或按秒付费，强调 time-to-first-call。

### Fireworks：latency-optimized GPU platform

- FireAttention engine（custom）；宣传在等价 configs 下 latency 比 vLLM 低 4x。
- Batch tier 对 non-interactive workloads 约为 serverless rate 的 50%。
- Fine-tuned model 按 base model 相同 rate 服务，这是相比会对你的 LoRA 收 premium 的 providers 的真正差异化点。
- 2026 年中：on-demand GPU rental 自 2026 年 5 月 1 日起上调 $1/hour。大规模时可协商 volume pricing。
- Financial signal：$4B valuation，处理 10T+ tokens/day。

### Together：breadth-optimized

- 200+ models，包括上游发布数日内的 open-source releases。
- 等价 LLM models 上比 Replicate 便宜 50-70%；“AI Native Cloud” 定位是 volume 和 catalog。
- Inference + fine-tuning + training 统一在一个 API 中。

### Baseten：enterprise-polish-optimized

- Truss framework：用一个 manifest 打包 model dependencies、secrets、serving config。
- GPU 覆盖 T4 到 B200。Per-minute billing，并有合理的 cold-start mitigation。
- SOC 2 Type II，HIPAA-ready。常见 fintech 和 healthcare pick。
- $5B valuation，2026 年 1 月 Series E（CapitalG、IVP、NVIDIA 投资 $300M）。

### Modal：Python-native-optimized

- 纯 Python 的 infrastructure-as-code。用 `@modal.function(gpu="A100")` 装饰一个 function，然后一条命令 deploy。
- Per-second billing。Cold starts 在 pre-warming 后为 2-4s；小模型 <1s。
- $87M Series B，valuation $1.1B（2025）。独立调查中的 developer experience score 最强。

### Replicate：multimodal breadth

- Pay-per-prediction。Image、video 和 audio models 的默认平台。
- Integration ecosystem（Zapier、Vercel、CMS plugins）。
- 在 LLM per-token rates 上竞争力较弱，但赢在 multimodal variety。

### Anyscale：Ray-native

- 构建在 Ray 之上；RayTurbo 是 Anyscale 的 proprietary inference engine（与 vLLM 竞争）。
- 最适合 distributed Python workloads，其中 inference step 是更大 graph 中的一个 node。
- Managed Ray clusters；与 Ray AIR 和 Ray Serve 紧密集成。

### Per-token versus per-minute：何时谁赢

Per-token 适合 latency-insensitive 且 bursty 的 workload，因为只按实际使用付费。Per-minute 适合 utilization 高且可预测的 workload，一旦你能让 GPU 饱和，它就会胜过 per-token。

粗略规则：对于超过 dedicated GPU 约 30% sustained utilization 的 workloads，per-minute（Baseten、Modal）开始胜过 per-token（Fireworks、Together）。低于这个水平，per-token 胜出，因为你避免为空闲付费。

### Custom engine 才是真正的 moat

每个高于 vLLM 和 SGLang 的平台都声称有 custom engine。FireAttention、RayTurbo、Baseten 的 inference stack。Custom-engine claims 带有 marketing 色彩，诚实的表述是：vLLM + SGLang 代表约 80% 的 production open-source inference，而 platform layer 的差异化在于 DX、attribution 和 SLAs。

### 你应该记住的数字

- Fireworks GPU rental：自 2026 年 5 月 1 日起上调 $1/hr。
- Fireworks claim：在等价 configs 下 latency 比 vLLM 低 4x。
- Together：LLMs 上比 Replicate 便宜 50-70%。
- Baseten valuation：$5B（Series E，2026 年 1 月，$300M round）。
- Modal valuation：$1.1B（Series B，2025）。
- Sustained utilization 超过约 30% 后，per-minute 胜过 per-token。

## 实际使用

`code/main.py` 在 synthetic workload 上跨 pricing models 比较六个 vendors。报告 $/day 和 effective $/M tokens。运行它，找到 per-token 和 per-minute 之间的 break-even。

## 交付成果

本课产出 `outputs/skill-inference-platform-picker.md`。给定 workload profile、SLA 和 budget，它会选择 primary inference platform，并点名 runner-up。

## 练习

1. 运行 `code/main.py`。对于一张 H100 上的 70B model，Baseten（per-minute）在什么 sustained utilization 下胜过 Fireworks（per-token）？自己推导 crossover，并与 rule of thumb 比较。
2. 你的产品服务 image generation、chat 和 speech-to-text。为每种 modality 选择平台，并说明统一它们的 gateway pattern。
3. Fireworks 把你的 primary model 价格上调 $1/hr。如果 40% traffic 迁移到 batch tier（50% off），建模 blended cost impact。
4. 受监管客户要求 SOC 2 Type II + HIPAA + dedicated GPUs。哪三个平台可行，哪一个在 FinOps 上胜出？
5. 比较 Llama 3.1 70B 在 Fireworks serverless、Together on-demand、Baseten dedicated 和 Replicate API 上每 1,000 predictions 的成本。10 predictions/day 时哪个最便宜？10,000 时呢？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Custom silicon | "non-GPU chips" | Groq LPU、Cerebras WSE、SambaNova RDU，针对 decode 优化 |
| FireAttention | "Fireworks engine" | Custom attention kernel；宣传 latency 比 vLLM 低 4x |
| Truss | "Baseten's format" | Model packaging manifest；dependencies + secrets + serving config |
| Per-token | "API pricing" | 按消耗的 tokens 收费；不为空闲付费 |
| Per-minute | "dedicated pricing" | 按 wall-clock GPU time 收费；高 utilization 时胜出 |
| Per-prediction | "Replicate pricing" | 按 model invocation 收费；image/video 中常见 |
| RayTurbo | "Anyscale engine" | Ray 上的 proprietary inference；在 Ray clusters 上与 vLLM 竞争 |
| Batch tier | "50% off" | 降低费率的 non-interactive queue；Fireworks、OpenAI 中常见 |
| Fine-tuned at base rate | "Fireworks LoRA" | LoRA-served requests 按 base model rate 收费（差异化点） |

## 延伸阅读

- [Fireworks Pricing](https://fireworks.ai/pricing) — per-token rates、batch tier、GPU rental。
- [Baseten Pricing](https://www.baseten.co/pricing/) — per-minute rates、committed capacity、enterprise tiers。
- [Modal Pricing](https://modal.com/pricing) — per-second GPU rates 和 free tier。
- [Together AI Pricing](https://www.together.ai/pricing) — model catalog 和 per-token rates。
- [Anyscale Pricing](https://www.anyscale.com/pricing) — RayTurbo 和 managed Ray pricing。
- [Northflank — Fireworks AI Alternatives](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) — comparative assessment。
- [Infrabase — AI Inference API Providers 2026](https://infrabase.ai/blog/ai-inference-api-providers-compared) — vendor landscape。
