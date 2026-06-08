# LLM Production 的 Chaos Engineering

> 到 2026 年，LLMs 的 chaos engineering 已经是一门独立纪律。在 production 中运行 experiments 前的 prerequisites：已定义 SLI/SLO、trace+metric+log observability、automated rollback、runbooks、on-call。架构有四个 planes：control（experiment scheduler）、target（services、infra、data stores）、safety（guards + abort + traffic filters）、observability（metrics + traces + logs）、feedback（进入 SLO adjustments）。Guardrails 是强制要求：如果 daily error-budget burn > 2x expected，burn-rate alerts 会暂停 experiments；suppression windows + trace-ID correlation 去重 alert noise。节奏：weekly small canary + SLO review；monthly game day + postmortem；quarterly cross-team resilience audit + dependency mapping。LLM-specific experiments：memory overload、network failures、provider outages、malformed prompts、KV cache eviction storms。Tooling：Harness Chaos Engineering（LLM-derived recommendations、blast-radius downscaling、MCP tool integration）；LitmusChaos（CNCF）；Chaos Mesh（CNCF Kubernetes-native）。

**类型:** 学习
**语言:** Python（stdlib，玩具 chaos experiment runner）
**先修:** Phase 17 · 23（SRE for AI），Phase 17 · 13（Observability）
**时间:** ~60 分钟

## 学习目标

- 说出五个 chaos engineering prerequisites（SLI/SLO、observability、rollback、runbooks、on-call），并解释跳过任何一个为什么会破坏实践。
- 画出四个 planes（control、target、safety、observability）以及进入 SLO 的 feedback loop。
- 枚举五个 LLM-specific experiments（memory overload、network fail、provider outage、malformed prompt、KV eviction storm）。
- 在给定 stack 时选择工具 —— Harness、LitmusChaos、Chaos Mesh。

## 要解决的问题

传统 stacks 中的 chaos testing 已经成熟。LLM stacks 添加了新的 failure modes。一个带 poison character 的 4K-token prompt 会让 tokenizer 卡住 12 秒。上游 provider 返回 429；你的 gateway 重试；你的 service 因 retry-amplified concurrency 而 OOM。Burst load 下的 KV cache eviction storm 会造成 re-prefill cascades，进而打满 compute。

这些都不会出现在 unit tests 里。Chaos engineering 是在用户发现之前发现它们的方法。

## 核心概念

### Prerequisites

没有以下条件，不要在 production 中运行 chaos：

1. **SLI/SLO** — 已定义 service-level indicators 和 objectives。
2. **Observability** — traces、metrics、logs，已接入 dashboards。
3. **Automated rollback** — Phase 17 · 20 policy-flag rollback。
4. **Runbooks** — 结构化，Phase 17 · 23。
5. **On-call** — 有人响应。

缺任何一个，chaos 都会变成真实 incident。

### Four planes + feedback

**Control plane** — experiment scheduler（Litmus workflow、Chaos Mesh schedule、Harness UI）。

**Target plane** — services、pods、nodes、load balancers、data stores。

**Safety plane** — kill switch、suppression windows、blast-radius limits、error-budget gates。

**Observability plane** — 常规 metrics + trace-ID correlation，用于区分 chaos-induced failures 与 natural failures。

**Feedback loop** — findings 回流到 SLO adjustment、runbook updates、code fixes。

### Guardrails 是强制要求

- **Burn-rate alert**：如果 daily error-budget burn 超过 2x expected，则暂停 experiment。
- **Suppression windows**：在 experiment 期间静默 blast radius 内的非实验 alerts。
- **Trace-ID correlation**：所有 experiment-induced errors 都携带 tag，让 on-call 能 dedupe。

### 五个 LLM-specific experiments

1. **Memory overload** — 通过发送 high concurrency 的 long-context requests 强制 KV cache preemption storm。观察：service 是 graceful shed 还是 crash？

2. **Network failure** — 切断 inference gateway 与 provider 之间的 connectivity。观察：fallback 是否在 SLA 内启动？（Phase 17 · 19）

3. **Provider outage simulation** — OpenAI 100% 返回 429。观察：routing 是否 failover 到 Anthropic？（Phase 17 · 16、19）

4. **Malformed prompt** — 注入 tokenizer-stalling payload（例如 deeply nested unicode、huge UTF-8 codepoint）。观察：单个 request 是否会锁死一个 worker？

5. **KV eviction storm** — 通过 saturating vLLM block budget 强制 eviction。观察：LMCache 是否恢复，还是 service degradation？

### Cadence

- **Weekly** — staging 中的小 canary experiments，也许 5% prod。
- **Monthly** — 围绕特定 scenario 的 scheduled game day；跨团队参与；postmortem。
- **Quarterly** — cross-team resilience audit；dependency map update。

### Tooling

- **Harness Chaos Engineering** — commercial；AI-derived experiment recommendations；blast-radius downscaling；MCP tool integration。
- **LitmusChaos** — CNCF graduated；基于 Kubernetes workflow。
- **Chaos Mesh** — CNCF sandbox；Kubernetes-native CRD style。
- **Gremlin** — commercial；广泛支持。
- **AWS FIS** / **Azure Chaos Studio** — managed cloud offerings。

### 从小处开始

第一个 experiment：在 steady traffic 下 pod-kill 一个 decode replica。观察 rerouting 和 recovery。如果它可用且看起来安全，再升级到 network chaos。

第一个 LLM-specific experiment：注入一个 provider 429，持续 5 分钟。观察 fallback。大多数团队会发现他们的 fallback 没有被完整测试。

### 你应该记住的数字

- 四个 planes：control、target、safety、observability。
- Burn-rate pause：2x expected daily budget burn。
- Cadence：weekly canary、monthly game day、quarterly audit。
- 五个 LLM experiments：memory、network、provider、malformed prompt、KV storm。

## 实际使用

`code/main.py` 模拟三个带 safety plane gates 的 chaos experiments。报告哪些 experiments 会触发 burn-rate abort。

## 交付成果

本课产出 `outputs/skill-chaos-plan.md`。给定 stack 和 maturity，它会选择前三个 experiments 和 tooling。

## 练习

1. 运行 `code/main.py`。哪个 experiment 触发 burn-rate gate，为什么？
2. 为基于 vLLM 的 RAG service 设计前五个 chaos experiments。包含 success criteria。
3. 你的 burn-rate alert 暂停了一个 experiment。你如何确定 root cause —— chaos 还是 natural？
4. 论证 chaos 应该在 production 中运行，还是只在 staging 中运行。什么时候 production 才是正确答案？
5. 说出三个 generic network-chaos 无法复现的 LLM-specific failure modes。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|------------|----------|
| SLI / SLO | “service targets” | Indicator + objective；必需 prerequisite |
| Blast radius | “scope” | 受 experiment 影响的 services / users 集合 |
| Burn-rate alert | “budget gate” | 当 error-budget burn rate > 2x expected 时触发 |
| Game day | “monthly drill” | Scheduled cross-team chaos exercise |
| LitmusChaos | “CNCF workflow” | Graduated CNCF Kubernetes chaos tool |
| Chaos Mesh | “CNCF CRD” | CNCF sandbox Kubernetes-native chaos |
| Harness CE | “commercial AI-assisted” | 带 AI recommendations 的 Harness chaos |
| Malformed prompt | “tokenizer bomb” | 会让 tokenization 卡住的 input |
| KV eviction storm | “preemption cascade” | 触发 re-prefills 的 mass eviction |

## 延伸阅读

- [DevSecOps School — Chaos Engineering 2026 Guide](https://devsecopsschool.com/blog/chaos-engineering/)
- [Ankush Sharma — Observability for LLMs (book)](https://www.amazon.com/Observability-Large-Language-Models-Engineering-ebook/dp/B0DJSR65TR)
- [LitmusChaos (CNCF)](https://litmuschaos.io/)
- [Chaos Mesh (CNCF)](https://chaos-mesh.org/)
- [Harness Chaos Engineering](https://www.harness.io/products/chaos-engineering)
- [AWS FIS](https://aws.amazon.com/fis/)
