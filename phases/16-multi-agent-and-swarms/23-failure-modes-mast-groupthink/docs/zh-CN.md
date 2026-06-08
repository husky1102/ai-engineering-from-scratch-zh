# Failure Modes：MAST、Groupthink、Monoculture、Cascading Errors

> 2026 年的 reference taxonomy 是 **MAST**（Cemri et al., NeurIPS 2025, arXiv:2503.13657），来源于 7 个 state-of-the-art open-source MAS 的 1642 条 execution traces，显示 **41-86.7% failure rate**。三个 root categories：**Specification Problems**（41.77%）：role ambiguity、unclear task definitions；**Coordination Failures**（36.94%）：communication breakdowns、state desync；**Verification Gaps**（21.30%）：missing validation、absent quality checks。**Groupthink** family（arXiv:2508.05687）补充：monoculture collapse（same base model → correlated failures）、conformity bias（agents reinforce each other's errors）、deficient theory of mind、mixed-motive dynamics、cascading reliability failures。Cascading example：retry storms，其中 payment failure 触发 order retries，再触发 inventory retries，然后压垮 inventory service（数秒内 10x load，需要 circuit breakers）。Memory poisoning：一个 agent 的 hallucination 进入 shared memory，下游 agents 把它当 fact；accuracy 逐渐 decay，让 root-cause diagnosis 很痛苦。**STRATUS**（NeurIPS 2025）报告，通过 specialized detection / diagnosis / validation agents，mitigation-success 提升 1.5x。本课把 failure modes 当成 first-class engineering targets。

**类型：** 学习
**语言：** Python (stdlib)
**先修：** Phase 16 · 13 (Shared Memory), Phase 16 · 14 (Consensus and BFT), Phase 16 · 15 (Voting and Debate Topology)
**时间：** ~75 分钟

## 要解决的问题

Multi-agent systems 在真实任务上有 41-86.7% 的失败率（Cemri et al. 2025 在 7 个 open-source MAS 上测得）。这不是靠 “just add more agents” 能 debug 的。失败有结构性原因。MAST taxonomy 给了你 categories。本课将每个 category 映射到具体 detection、diagnosis 和 mitigation pattern，让这些数字不再显得任意。

2026 年 production practice 是将 failure modes 当成 design inputs。直到你能指向每个 MAST category 并说出已经部署的 mitigation，你的 architecture 才算 “good enough”。

## 核心概念

### MAST categories

**Specification Problems（41.77% failures）。** agent 的 task 没定义得足够紧。例子：

- Role ambiguity：两个 agents 都认为自己是 reviewer。
- Task underspecified：“summarize this”，但用户想要 specific angle。
- Success criteria implicit：agent 无法判断自己是否成功。

Mitigations：
- 写 explicit role contracts。每个 agent 的 prompt 说明它做什么，*以及不做什么*。
- 每个 task 有 acceptance tests。agent 开始前定义 “done looks like X.”
- Pre-flight spec check：dispatch 前由 separate agent review task definition。

**Coordination Failures（36.94%）。** Communication 或 state breakdowns。

例子：
- 两个 agents 在没有 synchronization 的情况下更新 shared state。
- agents 之间 message lost（queue failure、timeout）。
- State drift：agent A 认为 task done；agent B 仍在 executing。

Mitigations：
- 带 optimistic concurrency 的 versioned shared state。
- critical messages 使用 explicit acknowledgment（retry until acked）。
- periodic state-sync checkpoints；早期检测 drift。

**Verification Gaps（21.30%）。** outputs 没有 independent check。

例子：
- 一个 agent 声称成功；没人验证。
- 一串 agents 每个都信任前一个 output。
- emergent composed behavior 缺少 test coverage。

Mitigations：
- Independent verifier agent（Lesson 13）。Read-only，independent source access。
- Explicit handoff contract：“A 的 output 必须通过 checker C，B 才能开始。”
- 用于 post-hoc analysis 的 outcome logging。

### Groupthink family（arXiv:2508.05687）

当 agents homogenize 或 mimic each other 时，会出现五类相关 failures：

**Monoculture collapse.** Same base model or training data → correlated errors。三个 agents 共享同一个 LLM 时，它们共享 hallucinations。

**Conformity bias.** Agents 向最响亮或最 confident 的 peer 靠拢，即便 peer 是错的。

**Deficient ToM.** Agents 无法建模彼此 beliefs；coordination 瓦解（Lesson 18）。

**Mixed-motive dynamics.** incentives 部分 aligned 的 agents 会漂向 compromise-middle，结果谁都不满足。

**Cascading reliability failures.** 一个 component 的 error pattern 会触发 dependent components 中的 error patterns。

### Cascading example：retry storm

一个经典 2026 incident pattern：

```text
payment service fails 10% of requests
   ↓
order agent retries payment (exponential backoff but naive)
   ↓
each retry is a new order-inventory check
   ↓
inventory service sees 2x normal load
   ↓
inventory service starts timing out
   ↓
every order retries inventory check
   ↓
inventory service sees 10x normal load
   ↓
cluster goes down
```

修复是经典做法：**circuit breakers**。当 downstream error rate 超过 threshold，short-circuit 为 cached 或 default results。再加上每个 request 的 capped retry budgets。

Circuit breakers 是少数可以从 distributed systems 直接借来、无需修改的 multi-agent failure mitigation。

### Memory poisoning（重访）

来自 Lesson 13：一个 agent 的 hallucination 成为 shared-memory fact；下游 agents 在 poisoned fact 上 reasoning。用 MAST 术语说，这是 shared-memory layer 的 verification gap。

症状是 gradual accuracy decay。你不会得到 crash；你会得到难以 root-cause 的 slow drift。

Mitigation：append-only log、provenance、unwritable verifier。Lesson 13 已覆盖。

### STRATUS：用于 failure detection 的 specialized agents

STRATUS（NeurIPS 2025）报告，如果部署以下 agents，mitigation-success 可提升 1.5x：

- **Detection agent.** 观察 symptom patterns（high disagreement、retry spikes、accuracy drift）。
- **Diagnosis agent.** 给定 symptoms，从 MAST taxonomy 推断 likely root cause。
- **Validation agent.** mitigation 应用后，检查 symptoms 是否清除。

这是应用于 agent systems 的 SRE-style incident response。三个 roles 都可以是带 specialized prompts 的 LLM agents。

### Failure-mode audit

2026 best practice 是 annual（或每个 major release）failure-mode audit：

1. **Trace sample.** 收集约 1000 条真实 execution traces。
2. **Categorize.** 对每条 trace 的 failures，映射到 MAST + Groupthink categories。
3. **Compute failure-by-category rate.** 哪些 categories 主导你的系统？
4. **Rank mitigations.** 哪个 fix 会消除最多 failures？
5. **Pick 2-3 mitigations.** 实现；下个季度重新 audit。

discipline 比具体选择更重要。没有 audits，failures 会混成噪声，永远无法系统性处理。

### 当 systems 静默失败

最危险的 failure category 是 silent correctness failure。loudly fail 的系统（crash、exception、alert）可以被监控。产生 plausible-but-wrong outputs 的系统无法被 exception logs 检测。这就是为什么 verification gaps 虽然只占 21.30% by count，却是每次失败成本最高的 category。

投入：
- Sample-based human review。
- Golden-dataset regression tests。
- 重要 outputs 上的 cross-agent cross-checking。

### Failure vs slow failure

有些 failures 是 immediate，有些是 slow。Immediate failures（timeout、schema mismatch、auth error）检测便宜。Slow failures（memory poisoning、monoculture drift、role ambiguity）检测和预防昂贵。

2026 engineering move：instrument slow-failure proxies，让你在 drift 变成 visible error 前捕获它。Agreement rate、retry rate、output-length distribution，以及 consecutive agent versions 之间的 edit-distance，都是有用 proxies。

## 动手实现

`code/main.py` 实现：

- `FailureTaxonomy`：将 simulated incidents 分类到 MAST + Groupthink categories。
- `CircuitBreaker`：classic pattern；error rate 超过 threshold 时 open。
- `RetryStormSimulator`：展示 cascading failure；切换 circuit breaker on / off。
- `DetectionAgent`：scripted STRATUS-style symptom matcher。

运行：

```text
python3 code/main.py
```

预期输出：
- 没有 circuit breaker 的 retry storm：inventory errors 爆炸（simulated）。
- 有 circuit breaker：cap at threshold；提供 degraded-mode responses。
- detection agent 标记 pattern 并命名 MAST category。

## 实际使用

`outputs/skill-mast-auditor.md` 会对 multi-agent system 运行 MAST-style failure-mode audit。Traces → categorization → mitigation ranking。

## 交付成果

Production 中的 failure-mode discipline：

- **MAST audit per quarter.** 不要 annual。categories 会随系统增长而变化。
- **Circuit breakers everywhere.** 每个 outbound call 到任何 dependent service。默认 open threshold 为 5-10% error rate。
- **Golden datasets.** 小而高质量，hand-audited。每周进行 regression-test。
- **STRATUS trio.** Detection + Diagnosis + Validation agents 监控 production。先只从 detection agent 开始；当 symptoms noisy 时再加 diagnosis。
- **Failure budget.** 按 category 显式设置 failure rate SLO。超出 budget 会触发 stop-shipping conversation。

## 练习

1. 运行 `code/main.py`。确认 circuit breaker caps retry storm。改变 failure threshold 并观察 tradeoff。
2. 实现一个 **slow-failure proxy**：3 个 parallel agents 的 agreement rate。当它 sharply drops 时触发 alert。通过逐渐相关化 agent outputs 来模拟 monoculture drift。
3. 阅读 Cemri et al.（arXiv:2503.13657）。选择他们的 7 个 MAS systems 之一，并映射其 top 3 failure categories。这些与 MAST 预测相比如何？
4. 阅读 Groupthink paper（arXiv:2508.05687）。识别五种 patterns 中哪一种在 production 中最难 detect。提出一个 proxy metric。
5. 为你熟悉的 specific multi-agent system 设计一个 STRATUS-style detection-diagnosis-validation trio。detection 观察哪些 symptoms？diagnosis 推荐什么 mitigations？validation 如何确认它们有效？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| MAST | “The 2026 taxonomy” | Cemri 2025；3 个 root categories + 14 个 failure sub-types。 |
| Specification Problem | “Role ambiguity” | task 或 role under-defined；agents 不知道该做什么。 |
| Coordination Failure | “State drift” | agents 之间的 communication 或 sync breakdown。 |
| Verification Gap | “No one checked” | outputs 未经 independent validation 就被接受。 |
| Groupthink family | “Homogeneity failures” | Monoculture、conformity、deficient ToM、mixed-motive、cascading。 |
| Monoculture collapse | “Same model, same hallucinations” | 由 shared base model 或 training data 导致的 correlated errors。 |
| Retry storm | “Cascading error amplification” | 一个 failure 触发 retries，放大 downstream load。 |
| Circuit breaker | “Fail fast on error rate” | error rate 超过 threshold 时 open；用 default short-circuit。 |
| STRATUS | “Incident response trio” | Detection + diagnosis + validation agents。mitigation success 提升 1.5x。 |
| Memory poisoning | “Hallucinations propagate” | shared-memory fact 被污染；downstream agents 在 poison 上 reasoning。 |

## 延伸阅读

- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST taxonomy，NeurIPS 2025
- [Groupthink failures in multi-agent LLMs](https://arxiv.org/abs/2508.05687) — monoculture、conformity 与 five-family taxonomy
- [STRATUS — specialized agents for MAS incident response](https://neurips.cc/) — NeurIPS 2025 proceedings entry（detection + diagnosis + validation）
- [Release It! — stability patterns (Nygard)](https://pragprog.com/titles/mnee2/release-it-second-edition/) — canonical circuit-breaker reference
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — production failure-mode notes
