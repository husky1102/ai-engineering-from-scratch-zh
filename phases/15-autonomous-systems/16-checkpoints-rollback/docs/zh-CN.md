# Checkpoints 与 Rollback

> 每个 graph-state transition 都会 persist。当 worker crash 时，它的 lease 过期，另一个 worker 会从 latest checkpoint 接手。Cloudflare Durable Objects 可以跨数小时或数周持有 state。Propose-then-commit（Lesson 15）为每个 action 定义 rollback plan。Post-action verification 关闭这个 loop。EU AI Act Article 14 要求高风险 systems 必须具备 effective human oversight；实践中，这意味着 checkpoints 必须 queryable，rollbacks 必须 rehearsed，audit trail 必须跨 deploy 存活。尖锐 failure mode 是：没有 idempotency keys 和 precondition checks 时，transient failure 后的 retry 可能 double-execute 一个已经 approved action。Post-action verification 正是捕获它的机制。

**类型：** 学习
**语言：** Python (stdlib, checkpoint and rollback state machine)
**先修：** Phase 15 · 12 (Durable execution), Phase 15 · 15 (Propose-then-commit)
**时间：** ~60 分钟

## 要解决的问题

Durable execution（Lesson 12）让 crashed agent 可恢复。Propose-then-commit（Lesson 15）让 approved action 可 audit。本课把它们接起来：当一个 approved action 部分执行、crash、再 resume 时，会发生什么？Rollback 什么时候运行，又针对什么 state 运行？

真实 systems 会用不同方式接线：

- **LangGraph** 把每个 graph-state transition checkpoint 到 PostgreSQL。Worker crash 时，lease 释放，另一个 worker 从 latest checkpoint resume。Workflows 会在 `interrupt()` 处暂停，而 `interrupt()` 本身也会 persist。
- **Cloudflare Durable Objects** 跨数小时或数周持有 per-key state。把 computation 与 approved action 的 storage co-locate。
- **Microsoft Agent Framework** 在 workflow API 中暴露 `Checkpoint` primitives；replay 加 idempotency 覆盖 retries。

在所有情况下，真正有效的组合都是：idempotency key（防 double-execute）+ precondition check（state 仍然是 approval 时所依据的 state）+ post-action verify（side effect 确实发生）+ verify-fail 时 rollback。

## 核心概念

### 每个 transition 都 persist

Graph-state transition 是任何把 workflow 从一个 named state 移到另一个 named state 的步骤。朴素实现只在特定 commit points persist；production implementations 会 persist 每个 transition。成本（多几次 writes）相对于 reliability gain 很小：replay 可以落在任何位置，lease recovery 更精确。

### Lease recovery

Worker crash 时，workflow 不会丢失；lease（一个短生命周期 claim，表示此 worker 正在执行这个 run）只是过期。另一个 worker 拿起 latest checkpoint 并 resume。Lease mechanism 让 production systems 能在 rolling deploys 中不丢失 in-flight work。

### Idempotency 加 preconditions

Idempotency 本身不够。考虑：workflow 被批准执行 “transfer $100 from A to B when balance > $1000”。Workflow committed，mid-execution crash，然后 resume。如果只检查 idempotency key，并让 execution resume，transfer 只运行一次（正确）。但如果在 crash 与 resume 之间，A 的 balance 通过另一个 workflow 降到 $500 呢？Idempotency check 仍然通过，precondition 不通过。没有 precondition check，我们就会制造 overdraft。

每个 consequential action 都需要两者：

- **Idempotency key**：防止 double-execute。
- **Precondition check**：确认 state 仍然与 approved action 一致。

### Post-action verification

“Tool returned 200” 不是 verification。真正的 verification 会重新读取 target state，并确认 side effect 确实发生。Patterns：

- Database update：`UPDATE ... RETURNING *`，然后 assert returned row 匹配 intended state。
- Email send：submission 后检查 sent-folder 中的 message ID。
- File write：读回 file 并 hash。
- API call：对 target resource 做后续 `GET`。

如果 verify 失败，workflow 处于 known-bad state。Rollback 会介入。

### Rollback plans

Propose-then-commit（Lesson 15）中的每个 consequential action 都携带 rollback plan。类型：

- **In-band rollback**：直接反转 side effect（`INSERT` 后 `DELETE`，send 后 `Send-correction-email`）。
- **Compensating transaction**：一个 neutralize 原 action 的新 action（标准 SAGA pattern）。
- **Out-of-band rollback**：alert human、暂停 workflow、留下 bad state 供 investigation。

No-op rollback（“we cannot undo this”）必须在 proposal 中命名。没有 rollback 的 actions 在 commit time 需要更强 HITL（Lesson 15 challenge-and-response）。

### EU AI Act Article 14 的 operational reading

Article 14 要求高风险 systems 具备 “effective human oversight”。在 operational terms 中，implementers 会把它理解为：

- Checkpoints 可被 auditor query。
- Rollbacks 已 rehearsed（至少 end-to-end 测试过一次）。
- Audit trail 跨 deploy 存活（checkpoint backend 不是 ephemeral）。
- Failed verifications 会 alert，而不是 silent log。

一个 workflow 如果 mid-commit crash、resume，并在没有 verify + rollback pathway 的情况下完成 side effect，就经不起 Article 14 test。

### 尖锐 failure mode：double-execute

这个领域最常见的 production incident：

1. Action approved，idempotency key k。
2. Commit starts，executes，returns 200。
3. Workflow 在 persist “committed” status 前 crash。
4. Workflow resumes；看到 “approved but not committed”；re-executes。
5. Side effect 触发两次。

Mitigation：execution 前 persist 一个 “in-flight” intent，用 idempotency key 执行，然后只在 post-action verification 成功后标记 “committed”。如果 action fire 了但 status write 失败，你知道要 verify，并在必要时 re-fire。如果 status write 成功但 action 失败，你会 verify，并通过 recovery path 恰好 fire 一次。

## 实际使用

`code/main.py` 实现一个带 checkpoint 的 workflow，包含 idempotency、preconditions、verify 和 rollback。Driver 模拟四个 scenarios：clean run、crash 后 retry（idempotency 捕获）、precondition fail（workflow aborts without firing）、verify fail（rollback fires）。

## 交付成果

`outputs/skill-rollback-rehearsal.md` 会为一个拟议 workflow 设计 rollback-rehearsal test，并 audit checkpoint backend 的 audit-trail persistence。

## 练习

1. 运行 `code/main.py`。验证四个 scenarios。对于 crash-during-commit case，确认 action 在 retries 中恰好 fire 一次。

2. 修改 “mark as done first, then do it” pattern，让 status write 在 action 之后 fire。重新运行 crash scenario。测量触发了多少 duplicate actions。

3. 为一个具体 production action（例如 “post to a Slack channel”）设计 rollback plan。分类为 in-band、compensating 或 out-of-band。说明选择理由。

4. 选择一个你熟悉的 workflow。识别每个 state transition。为每个标记 durability requirement（persist / do not persist）。数一数当前没有 persist 的有多少。

5. Rehearsed-rollback test：设计一个 end-to-end test，运行真实 workflow，让它 crash，并确认 rollback path fire。这个 test assert 什么？

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| Checkpoint | “Save point” | 每个 graph-state transition 都 persist 到 durable store |
| Lease | “Worker claim” | Worker 正在执行某个 run 的短生命周期 claim；crash 时 expire |
| Precondition | “State gate” | 断言 state 仍与 approved action 一致 |
| Post-action verify | “Re-read check” | 确认 side effect 确实发生在 target system 中 |
| In-band rollback | “Direct undo” | 用 inverse operation 反转 side effect |
| Compensating transaction | “SAGA undo” | 一个 neutralize 原 action 的新 action |
| Mark-as-done-first | “Status write order” | 从 commit 返回前先 persist committed status |
| Article 14 | “EU AI Act human oversight” | Operational：queryable checkpoints、rehearsed rollbacks、auditable trail |

## 延伸阅读

- [Microsoft Agent Framework — Checkpointing and HITL](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — checkpoint primitives 与 lease recovery。
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — Durable Objects 作为 state substrate。
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — regulatory baseline。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — long-horizon workflows 的 reliability framing。
- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — Claude Code Routines 的 workflow shape。
