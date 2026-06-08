# Model Routing 作为 Cost-Reduction Primitive

> dynamic broker 会评估每个 request（task type、token length、embedding similarity、confidence），把 simple queries 发给 cheap model，将 complex ones escalate 到 frontier model。也称为 model cascading。Production case studies 显示，在 US/UK/EU deployments 中，iso-quality 下 cost reduction 为 20-60%；high-volume SaaS 上 30% routing efficiency improvement 会转化为六位数 annual savings。2026 语境是 LLM inference prices 每年下降约 10x：GPT-4-class token 从 2022 年末的 $20/M 到 2026 年约 $0.40/M。大部分下降来自更好的 serving stacks（Phase 17 · 04-09），不是 hardware。Routing 是你在不造成 product regression 的情况下，把价格下降转化为 margin 的方式。failure mode 是 cheap-model drift：route 把 40% 推给 weaker model，reasoning tasks 上 quality 下降 3-5%，一个季度没人注意。用 online quality metrics gate routes，而不仅是 offline eval sets。

**类型：** 学习
**语言：** Python (stdlib, toy cascading router simulator)
**先修：** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 19 (AI Gateways)
**时间：** ~60 分钟

## 学习目标

- 解释 model cascading：cheap-first + confidence check，low confidence 时 escalate。
- 枚举四个 routing signals（task classification、prompt length、embedding similarity to known-hard set、first-pass self-confidence）。
- 在目标 routing split 和 quality loss tolerance 下计算 expected blended cost。
- 说出能捕获 cheap-model creep 的 drift-monitoring metric（online quality gate）。

## 要解决的问题

你的服务在 GPT-5 上花 $80k/month。analytics 显示 70% queries 很简单：“what time is it in Paris?” “rephrase this sentence.” Haiku-class model 能以 3% 成本完美处理这些。30% 需要 GPT-5 的 reasoning：coding、math、multi-step planning。

如果你把 70% route 到 cheap，30% route 到 expensive，你的账单会在相同 product quality 下下降约 65%。这就是 routing。诀窍是构建 broker，而不让 quality regression。

## 核心概念

### 四个 routing signals

1. **Task classification**：simple/complex/codegen/math/chat。可以是 rules-based classifier、small LLM（Haiku-class at $0.25/M），或到 labeled buckets 的 embedding similarity。输出：route = cheap / balanced / frontier。

2. **Prompt length**：prompts >4K tokens 往往需要 frontier 来保持 coherence。prompts <500 tokens 通常不需要。

3. **Embedding similarity to known-hard set**：如果 query 靠近（cosine > 0.88）known-hard bucket，直接 escalate 到 frontier。

4. **Self-confidence from first-pass**：先发给 cheap；如果 model 的 log-probs 显示 low confidence，或者它 refuses，或者输出 hedging language，就 retry on frontier。给约 10% traffic 增加 P95 latency，但在其他 90% 上节省 50%+。

### 三种 patterns

**Pre-route**（classifier up front）：增加约 5-10ms latency；overall 最快。

**Cascade**（cheap-first, low confidence 时 escalate）：约 1.2x median latency（cheap run plus verify），escalated 时约 2x。quality floor 最好。

**Ensemble route**（sample 中并行运行 cheap 和 frontier，由 reward-model 选择）：最高 quality，最高 cost；只用于 critical A/B。

### Implementation

AI gateways（Phase 17 · 19）暴露 routing。LiteLLM 有带 fallback 和 cost-routing 的 `router` config。Portkey 有 guards + routing。Kong AI Gateway 有 plugin-based routing。OpenRouter 的 model marketplace 暴露 recommendation API。

Open-source：RouteLLM（LMSYS）、Not Diamond（commercial）、Prompt Mule。

### 2026 price curve

| Model class | Late 2022 | 2026 | Change |
|-------------|-----------|------|--------|
| GPT-4-level quality | ~$20/M | ~$0.40/M | 50x cheaper |
| Frontier (GPT-5, Claude 4) | — | ~$3-10/M | new tier |

大部分改进来自 serving efficiency：Phase 17 · 04-09 的核心课程变成 provider-side cost drops。Routing 让你在 app layer 捕获这些 gains，而不是等所有用户迁移到 cheap tier。

### Drift 是真正风险

你的 route 把 40% 发给 cheap model。六个月后，task distribution shift（用户更 sophisticated，问更长问题）。router 没注意到，因为 classifier 是基于 Q1 data 训练的。Quality 静默下降。没人足够大声投诉。你是在 competitor benchmark 中发现自己输了。

用 online quality metrics gate routes：

- 每个 route 上的 user thumbs-up / thumbs-down。
- 每个 route 上 held-out sample（5%）的 automated LLM-judge。
- Escalation rate：如果 cascade up-route >30%，cheap model 正被 over-routed。
- 每个 route 的 refusal rate。

### 你应该记住的数字

- 2026 routing savings at iso-quality：case studies 20-60%。
- LLM price drop 2022-2026：aggregate 每年约 10x。
- GPT-4-level 2022 vs 2026：~$20/M → ~$0.40/M。
- Cascade latency impact：约 1.2x median，escalated 约 2x（~10% traffic）。

## 实际使用

`code/main.py` 在 mixed workload 上模拟 pre-route、cascade 和 ensemble。报告 blended cost、quality loss 和 escalation rate。

## 交付成果

本课产出 `outputs/skill-router-plan.md`。给定 workload 和 quality budget，它会选择 routing pattern 和 signals。

## 练习

1. 运行 `code/main.py`。在什么 accuracy floor 下，cascade 胜过 pre-route？
2. 你的 user base 是 30% enterprise（complex queries）、70% free tier（simple）。设计 routing split。用什么 online metric gate 它？
3. 一个 route 让 quality 下降 2%，但节省 40%。是否 ship？取决于 product：两边都论证。
4. 用 OpenAI / Anthropic APIs 的 logprobs 实现 confidence check。你会从什么 threshold 开始？
5. 六个月里 escalation rate 从 8% 升到 22%。诊断三个原因，并给出每个的 fix。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Model routing | “cost broker” | 每个 request 动态选择 model |
| Model cascade | “cheap-first escalate” | 先跑 cheap，low confidence 时 fall through 到 frontier |
| Pre-route | “classify first” | upfront classifier；不 re-run |
| Ensemble route | “parallel pick” | 运行多个，由 reward-model 选择 best |
| Escalation rate | “uprouted %” | cascade requests 中被 escalated 的 fraction |
| RouteLLM | “LMSYS router” | OSS router library |
| Not Diamond | “commercial router” | SaaS model-routing product |
| Drift | “cheap creep” | distribution shift，但 router 没注意到 |
| Online quality gate | “live check” | 对 live traffic 做 automated LLM-judge sampling |

## 延伸阅读

- [AbhyashSuchi — Model Routing LLM 2026 Best Practices](https://abhyashsuchi.in/model-routing-llm-2026-best-practices/)
- [Lukas Brunner — Rise of Inference Optimization 2026](https://dev.to/lukas_brunner/the-rise-of-inference-optimization-the-real-llm-infra-trend-shaping-2026-4e4o)
- [RouteLLM paper / code](https://github.com/lm-sys/RouteLLM)
- [Not Diamond — model routing](https://www.notdiamond.ai/)
- [OpenRouter](https://openrouter.ai/) — multi-model gateway with routing primitives。
