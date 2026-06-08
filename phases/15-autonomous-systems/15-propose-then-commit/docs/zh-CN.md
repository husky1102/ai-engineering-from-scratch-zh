# Human-in-the-Loop：Propose-Then-Commit

> 2026 年围绕 HITL 的共识很具体。它不是 “agent 提问，user 点击 Approve”。它是 propose-then-commit：proposed action 会用 idempotency key 持久化到 durable store；向 reviewer 展示 intent、data lineage、permissions touched、blast radius 和 rollback plan；只有 positive acknowledgement 后才 commit；execution 后还要 verify，确认 side effect 确实发生。LangGraph 的 `interrupt()` 加 PostgreSQL checkpointing、Microsoft Agent Framework 的 `RequestInfoEvent`、Cloudflare 的 `waitForApproval()` 都实现了同一种形状。Canonical failure mode 是 rubber-stamp approval：“Approve?” 没有 review 就被点击。文档化的 mitigation 是带显式 checklist 的 challenge-and-response。

**类型：** 学习
**语言：** Python (stdlib, propose-then-commit state machine with idempotency)
**先修：** Phase 15 · 12 (Durable execution), Phase 15 · 14 (Tripwires)
**时间：** ~60 分钟

## 要解决的问题

Agent 要采取一个 action。User 必须决定：approve 还是不 approve。如果这个 decision 是瞬间完成的，那它很可能不是 review。如果 decision 是 structured，它会慢一些，但可信。工程问题是：如何让 structured review 成为 least resistance 的路径。

2023-era HITL pattern 是 synchronous prompt：“Agent wants to send email to X with body Y — approve?” User 点击 Approve。所有人都觉得系统安全。实践中，这个 surface 极易被 rubber-stamped：users approve 很快，approvals 几乎不预测什么；当 agent 出错时，audit trail 显示的是一长串 user 已经想不起来的 approvals。

2026 pattern，也就是 propose-then-commit，把 HITL 移到 durable substrate 上，附加 structured metadata，并要求 positive commit。每个 managed agent SDK 都交付了一个版本：LangGraph `interrupt()`、Microsoft Agent Framework `RequestInfoEvent`、Cloudflare `waitForApproval()`。API 名字不同，形状相同。

## 核心概念

### Propose-then-commit state machine

1. **Propose。** Agent 生成一个 proposed action。它被持久化到 durable store（PostgreSQL、Redis、Durable Object）。包含：
   - intent（agent 为什么这样做）
   - data lineage（哪个 source 导致这个 proposal）
   - permissions touched（哪些 scopes / files / endpoints）
   - blast radius（最坏情况是什么）
   - rollback plan（如果 committed，如何撤销）
   - idempotency key（每个 proposal 唯一；重新提交返回同一条 record）
2. **Surface。** Reviewer 看到 proposal 和所有 metadata。Reviewer 是人（不是 agent review 自己）。
3. **Commit。** Positive acknowledgement。Action 执行。
4. **Verify。** Execution 后，回读 side effect 并确认。如果 verify step 失败，system 处于已知 bad state，alerting 会介入。

### Idempotency key

没有 idempotency key 时，transient failure 后的 retry 会把已批准 action 执行两次。具体例子：user 批准 “transfer $100 from A to B”。Network blips。Workflow retries。User 只批准了一次，但 transfer 执行了两次。Idempotency key 把 approval 绑定到单个 unique side effect；第二次 execution 是 no-op。

这和 Stripe、AWS APIs 使用的 idempotency pattern 相同。Microsoft Agent Framework docs 明确把它复用于 agent approvals。

### Durability：为什么 approvals 能活得比 processes 更久

Approval waiting room 是 agent 不拥有的一块 state。Workflow 暂停（Lesson 12）。当 approval 到达时，workflow 从那个精确位置恢复。这就是 LangGraph 为什么把 `interrupt()` 和 PostgreSQL checkpointing 搭配，而不是只用 in-memory state：两天后的 approval 仍然能找到完整 workflow。

### Rubber-stamp approvals 与 challenge-and-response mitigation

HITL 的默认 UI（“Approve” / “Reject” buttons）会产生快速 approvals，却没有真正 review。文档化 mitigation：challenge-and-response checklist，它要求 reviewer 对具体问题作出 positive answers 后，Approve button 才会启用。具体形状：

- “Do you understand what resource this touches? [ ]”
- “Have you verified the blast radius is acceptable? [ ]”
- “Do you have a rollback plan if this fails? [ ]”

这不是为了 bureaucracy 而 bureaucracy，而是 forcing function。无法勾选这些 boxes 的 reviewer，要么请求澄清（escalation），要么拒绝（safe default）。Anthropic agent-safety research 明确把 checklist-driven HITL 作为 rubber-stamp approval patterns 的 mitigation。

### 什么算 consequential

不是每个 action 都需要 propose-then-commit。2026 guidance：

- **Consequential actions**（总是 HITL）：irreversible writes、financial transactions、outbound communication、production database changes、destructive file-system operations。
- **Reversible actions**（有时 HITL）：local files edits、staging-env changes、带清晰 rollback 的 reversible writes。
- **Reads and inspections**（从不 HITL）：reading a file、listing resources、calling a read-only API。

### Post-action verification

“The commit ran” 不等于 “the side effect happened”。Network-partition 和 race conditions 可能让 workflow 以为成功，而 backend 没有 persist。Verify step 会在 commit 后重新读取 target resource 来确认。这和 database transactions 中的 `RETURNING` clauses，或 AWS `PutObject` 后 `GetObject` 是同一个模式。

### EU AI Act Article 14

Article 14 要求 EU 高风险 AI systems 具备 effective human oversight。“Effective” 不是装饰。监管语言明确排除 rubber-stamp patterns。带 challenge-and-response 的 propose-then-commit，是 Microsoft Agent Governance Toolkit compliance docs 中能经受 Article 14 scrutiny 的形状。

## 实际使用

`code/main.py` 用 stdlib Python 实现一个 propose-then-commit state machine。Durable store 是 JSON file。Idempotency key 是 (thread_id, action_signature) 的 hash。Driver 模拟三种情况：clean approval flow、transient failure 后的 retry（不能 double-execute），以及 rubber-stamp default 与 challenge-and-response flow 的对比。

## 交付成果

`outputs/skill-hitl-design.md` 会 review 一个拟议 HITL workflow 的 propose-then-commit shape，并标记 missing metadata、idempotency、verification 或 challenge-and-response layers。

## 练习

1. 运行 `code/main.py`。确认 approved proposal 的 retry 会使用 durable record，而不会重新执行。然后把 idempotency key 改成包含 timestamp，展示 retry 会 double-execute。

2. 用 `rollback` field 扩展 proposal record。模拟一个 verify step 失败的 execution。展示 rollback 自动触发。

3. 阅读 Microsoft Agent Framework 的 `RequestInfoEvent` docs。识别 API 包含但 toy engine 缺少的一个 metadata field。加入它，并解释它防护什么。

4. 为一个具体 action（例如 “post to a public Twitter account”）设计 challenge-and-response checklist。Reviewer 必须回答哪三个问题？为什么是这三个？

5. 选择一个 synchronous “Approve?” prompt 已足够的场景（不需要 durable store）。解释为什么，并说出你接受的 risk class。

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| Propose-then-commit | “Two-phase approval” | Persisted proposal + positive commit + verify |
| Idempotency key | “Retry-safe token” | 每个 proposal 唯一；第二次 execution no-ops |
| Data lineage | “Where it came from” | 产生 proposal 的具体 source content |
| Blast radius | “Worst case” | Action 出错时的 effect scope |
| Rubber-stamp | “Fast approval” | 没有 genuine review 就点击 “Approve” |
| Challenge-and-response | “Forcing checklist” | Reviewer 必须明确承认 specific questions |
| RequestInfoEvent | “MS Agent Framework primitive” | 带 structured metadata 的 durable HITL request |
| `interrupt()` / `waitForApproval()` | “Framework primitives” | LangGraph / Cloudflare 中同一形状的 equivalents |

## 延伸阅读

- [Microsoft Agent Framework — Human in the loop](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — `RequestInfoEvent`、durable approvals。
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — `waitForApproval()` 和 Durable Objects。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — HITL 作为 long-horizon risk 的 mitigation。
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — high-risk systems 的 regulatory baseline。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — oversight 周围的 constitutional framing。
