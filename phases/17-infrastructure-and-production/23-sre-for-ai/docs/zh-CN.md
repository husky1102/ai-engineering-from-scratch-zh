# AI 的 SRE — 多 Agent 事故响应、Runbooks、预测性检测

> AI SRE 使用通过 RAG 接入基础设施数据（logs、runbooks、service topology）的 LLMs，自动化调查、记录和协调阶段。2026 年的架构模式是 multi-agent orchestration —— 专门化 agents（logs、metrics、runbooks）由 supervisor 协调；AI 提出 hypotheses 和 queries，人类批准 judgment calls。Datadog Bits AI 和 Azure SRE Agent 已将其作为 managed products 交付。Runbooks 正在演进：NeuBird Hawkeye 使用 adversarial evaluation（两个模型分析同一 incident；agreement = confidence，disagreement = uncertainty）；operational memory 会在团队变更后保留下来。Auto-remediation 保持谨慎：AI 建议，人类批准。Fully autonomous action 仅限窄范围（restart pod、rollback specific deploy）并带 tight guardrails —— 任何兜售“set it and forget it”的人都在过度承诺。新前沿：pre-incident prediction。MIT research 报告，一个基于 historical logs + GPU temps + API error patterns 训练的 LLM，能提前 10-15 分钟预测 89% outages。预测：到 2026 年底，95% enterprise LLMs 具备 automated failover。

**类型:** 学习
**语言:** Python（stdlib，玩具 multi-agent incident triage simulator）
**先修:** Phase 17 · 13（Observability），Phase 17 · 24（Chaos Engineering）
**时间:** ~60 分钟

## 学习目标

- 画出 multi-agent AI SRE architecture：supervisor + specialized agents（logs、metrics、runbooks）+ human approval gate。
- 解释为什么 auto-remediation 是窄范围（restart pod、revert deploy），而不是宽范围（re-architect service）。
- 说出 adversarial evaluation pattern（NeuBird Hawkeye）：two models agree = confidence；disagree = escalate。
- 引用 MIT 89% early-detection result 以及 operational constraint：没有 actuation 的 predictions 只是 dashboards。

## 要解决的问题

值班工程师凌晨 3 点收到 page：“checkout error rate 高。”他们检查 Datadog、Loki、三个 runbooks、deploy log。30 分钟后，他们意识到 root cause 是 KV cache spike 触发的 vLLM OOM。他们 restart pod；error 清除。

到 2026 年，这段调查的前 20 分钟可以自动化。按 service 对 logs 分组、关联 recent deploys、匹配 runbooks —— 这些都是 RAG + tool-use。一个受监督的 agent 可以做 first-pass triage，并在 human 打开 Datadog 前给出 hypothesis。

Fully autonomous remediation 是另一个问题。Restart pod：安全。Scale GPU pool：如果 policy 允许则安全。Re-architect the service：绝对不行。纪律在于画出这条窄边界。

## 核心概念

### Multi-agent architecture

```text
          Incident
             │
             ▼
        Supervisor
        /    |    \
       ▼     ▼     ▼
  Log agent  Metric agent  Runbook agent
       │     │     │
       └─────┴─────┘
             │
             ▼
        Hypothesis + evidence
             │
             ▼
        Human approval
             │
             ▼
        Action (narrow set)
```

Supervisor 将 incident 拆成 sub-queries。Specialized agents 拥有 tool access（log search、PromQL、doc retrieval）。Supervisor 综合信息，向 human 展示 hypothesis + evidence。Human 批准或重定向。

### Auto-remediation scope

**安全（窄范围）**：restart pod、revert specific deploy、在预先批准的边界内 scale pool、enable pre-approved feature flag。

**不安全（宽范围）**：change service topology、modify resource limits、deploy new code、change IAM、alter databases。

任何兜售“set it and forget it”的人都在过度承诺。随着 AI SRE 成熟，安全集合会扩大，但边界是真实存在的。

### Adversarial evaluation（NeuBird Hawkeye）

两个模型独立分析同一个 incident。如果它们在 root cause 上一致，confidence 高。如果不一致，就升级给 human，并展示两个 hypotheses。这个模式简单，是过滤 hallucinated root causes 的有效手段。

### Operational memory

Team turnover 是传统 SRE 的无声杀手 —— tribal knowledge 会离开。AI SRE 将 runbooks + post-mortems 存入 vector DB；agents 在每次新 incident 中检索。当新工程师加入时，AI 拥有完整历史。

### Pre-incident prediction

MIT 2025 research：基于 historical logs、GPU temperatures、API error patterns 训练的 LLM，在 test set 上能提前 10-15 分钟预测 89% outages。

现实检查：没有 actuation 的 predictions 只是 dashboards。运营问题是：“当我们预测到时，要做什么？”Pre-emptive drain？Pager？Auto-scale？答案取决于 policy。

### 2026 年产品

- **Datadog Bits AI** — Datadog 内的 managed SRE copilot。
- **Azure SRE Agent** — Azure-native。
- **NeuBird Hawkeye** — adversarial eval + operational memory。
- **PagerDuty AIOps** — triage + deduplication。
- **Incident.io Autopilot** — incident commander + coordination。

### Runbooks as code

Runbooks 从 Confluence pages 演进为带结构化章节的 versioned markdown（symptom、hypothesis、verify、act）。结构化 runbooks 能喂出更好的 RAG retrieval。任何 AI-SRE rollout 都应该先把非结构化 runbooks 转成结构化。

### 你应该记住的数字

- MIT early-detection：89% outages，10-15 min lead time。
- Multi-agent triage：supervisor +（logs、metrics、runbooks）+ human。
- 安全 auto-remediation set：restart pod、revert deploy、scale within bounds。
- Adversarial eval：两个模型 independent；agreement = confidence。

## 实际使用

`code/main.py` 模拟 multi-agent triage：log agent 找到 error，metric agent 找到 CPU spike，runbook agent 匹配 known issue。Supervisor 对 hypotheses 排序。

## 交付成果

本课产出 `outputs/skill-ai-sre-plan.md`。给定当前 on-call、incident volume、team maturity，它会设计 AI SRE rollout。

## 练习

1. 运行 `code/main.py`。如果 log 和 metric agents 不一致会怎样？Supervisor 如何解决？
2. 为你的 service 定义三个“safe” auto-remediation actions。逐一说明理由。
3. 写一个 structured runbook template：sections、required fields、verification commands。
4. Predictive detection 在 12 min lead 时触发。你的 policy 是什么 —— pager、pre-drain，还是二者都有？
5. 论证一个 3 人团队在 2026 年应该采用 AI SRE 还是等待。考虑 maturity、volume、risk。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|------------|----------|
| AI SRE | “agent for on-call” | LLM-backed incident investigation + coordination |
| Supervisor agent | “the orchestrator” | 将 incidents 拆成 sub-queries 的 top-level agent |
| Specialized agent | “domain agent” | 拥有 tool access 的 sub-agent（logs、metrics、runbooks） |
| Auto-remediation | “AI fixes it” | 窄范围预先批准的 action；不是 broad re-architecture |
| Operational memory | “vector runbooks” | 用于 RAG 的 post-mortems + runbooks in vector DB |
| Adversarial eval | “two-model check” | 独立 analyses；agreement = confidence |
| NeuBird Hawkeye | “the adversarial one” | 具备 adversarial-eval + memory pattern 的产品 |
| Bits AI | “Datadog's SRE agent” | Datadog-managed AI SRE |
| Pre-incident prediction | “early detection” | outage prediction 的 10-15 min lead time |

## 延伸阅读

- [incident.io — AI SRE Complete Guide 2026](https://incident.io/blog/what-is-ai-sre-complete-guide-2026)
- [InfoQ — Human-Centred AI for SRE](https://www.infoq.com/news/2026/01/opsworker-ai-sre/)
- [DZone — AI in SRE 2026](https://dzone.com/articles/ai-in-sre-whats-actually-coming-in-2026)
- [Datadog Bits AI](https://www.datadoghq.com/product/bits-ai/)
- [NeuBird Hawkeye](https://www.neubird.ai/)
- [awesome-ai-sre](https://github.com/agamm/awesome-ai-sre)
