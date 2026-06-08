# Shared Memory 与 Blackboard Patterns

> 2026 年的多智能体系统中并存两种 approach：**message pool**（每个人都看到每个人的 messages，如 AutoGen GroupChat 或 MetaGPT）和**带 subscription 的 blackboard**（agents 订阅相关 events，如 Context-Aware MCP 或 Matrix framework）。两者都是多智能体系统里唯一有状态的部分，也意味着有趣的 bug 都住在这里。参考 failure mode 是 **memory poisoning**：一个 agent 幻觉出一个“fact”，其他 agents 把它当成已验证事实，accuracy 以渐进方式衰减，比立即 crash 更难 debug。本课从 stdlib 构建两种结构，注入一次 poisoning attack，并展示生产中真正有效的三种 mitigations。

**类型：** Learn + Build
**语言：** Python (stdlib, `threading`)
**先修：** Phase 16 · 04 (Primitive Model), Phase 16 · 09 (Parallel Swarm Networks)
**时间：** ~75 分钟

## 要解决的问题

多智能体系统需要一个地方让 agents 共享 facts。一个字面方案是“把所有东西都通过 messages 传递”，但这只是用额外复制重新发明 shared state。另一个方案是“给每个人一个 global log”，但 global logs 会无限增长，而且很容易被 poisoning。第三个方案是“为每个 agent 投影一个 view”，可扩展但 schema-heavy。

当某个 agent 幻觉并把幻觉写入 shared state 时，每个读取该 state 的下游 agent 都会把这个幻觉当成事实采纳。等人类发现时，reasoning chain 已经深入五步，根因却是第三条写入的 message。debug 多智能体 accuracy decay 比 debug crash 更难。

这就是 memory poisoning。它是 MAST taxonomy（Cemri et al., arXiv:2503.13657）中记录第二多的 failure family，而且是结构性的：任何没有 provenance 和不可写 verifier 的 shared-memory design 最终都会表现出它。

## 核心概念

### 两种主要 topology

**Full message pool。** 每个 agent 读取每条 message。AutoGen GroupChat 和 MetaGPT 使用这种方式。简单、透明、可检查，但很难扩展到 ~10 个 agents 以上，因为每个 agent 的 context 会被其他 agents 的工作填满。

```text
agent-A ──write──▶ ┌────────────────┐ ◀──read── agent-D
                   │ message pool   │
agent-B ──write──▶ │                │ ◀──read── agent-E
                   │ (global log)   │
agent-C ──write──▶ └────────────────┘ ◀──read── agent-F
```

**Blackboard with subscription。** Agents 声明它们感兴趣的 topics；substrate 只路由相关 messages。CA-MCP（arXiv:2601.11595）和 Matrix decentralized framework（arXiv:2511.21686）使用这种方式。它扩展得更远，但需要 upfront schema design，才能让 subscriptions 有意义。

```text
                   ┌─ topic: prices ──┐
agent-A ──pub────▶ │                  │ ──▶ agent-D (subscribed)
                   ├─ topic: orders ──┤
agent-B ──pub────▶ │                  │ ──▶ agent-E (subscribed)
                   ├─ topic: alerts ──┤
agent-C ──pub────▶ │                  │ ──▶ agent-F (subscribed)
                   └──────────────────┘
```

### 各自什么时候胜出

- **Full pool** 在 agents 很少（< 10）、角色异质、conversation 是 short-horizon 时胜出。每个人都看到一切时，推理“谁说了什么”非常直接。
- **Blackboard** 在 agents 很多、角色同质但实例众多（swarms）、conversation 长时间运行时胜出。Routing 节省 token cost，也减少 context pollution。

生产系统常混合两者：顶层（planning layer）用小型 full pool，下层（worker layer）用 blackboards。

### Memory poisoning，一个场景

三个 agents 做一个 research task。Agent A 是 retrieval agent。Agent B 是 summarizer。Agent C 是 analyst。

1. A 抓取一个页面，并向 shared state 写入 message：“The study reports a 42% accuracy improvement.”
2. 实际抓取页面写的是“4.2% improvement”。A 幻觉了一个小数点。
3. B 读取 shared state 后写入：“Large 42% accuracy gain reported (source: A).”
4. C 读取 shared state 后写入：“Recommend adoption — 42% lift is transformative.”
5. 最终报告引用了一个从未存在过的 42% 数字。

没有 agent crash。没有 test fail。系统“工作了”。幻觉通过 shared state 从一个 agent 的 context 跨进了每个下游 agent 的 reasoning。

### 为什么这是结构性问题

没有 shared state 时，agent A 的幻觉留在 A 的 context 中。下游 agents 会重新 fetch 或重新 derive，可能捕捉到错误。有了 naive shared state，A 的 context 变成每个人的 context，幻觉被洗成事实。

问题不在 shared state 本身，而在于 shared state **没有 provenance，也没有 independent verifier**。三种 mitigation 处理这个问题：

1. **为每次 write 标注 provenance。** shared state 中每个 entry 都记录谁写的、何时写的、在什么 prompt 下写的，以及（如果适用）agent 引用了什么 source。下游 agents 根据 provenance 带着怀疑去读。
2. **对 writes 做 versioning；把它们视作 append-only。** correction 是一个 supersedes 旧 entry 的新 entry，不是 in-place update。audit trail 被保留。
3. **保留至少一个不能写 shared state 的 agent。** read-only verifier agent 抽样 entries、重新 fetch sources，并标记 inconsistencies。因为它不能写入 pool，所以它不能被 pool poisoning。

### Blackboard 先例（Hayes-Roth, 1985）

Blackboard pattern 比 LLM agents 早四十年。Hayes-Roth（1985, "A Blackboard Architecture for Control"）描述了 specialist Knowledge Sources：它们观察 global blackboard、贡献 partial solutions，并触发其他 sources。2026 年的 blackboard（CA-MCP、Matrix）是同一种 pattern，只是 Knowledge Sources 变成 LLM agents，partial solutions 变成 JSON blobs。旧文献已经记录了 write contention、opportunistic control 和 consistency 的解决方案，现代系统正在重新发现它们。

### Projection vs full view

纯 blackboard 给每个 subscriber 同样的 projection（topic-scoped）。更激进的设计是 **per-agent projection**：每个 agent 得到一个针对自身 role 定制的 view。LangGraph 的 state reducers 是 2026 年的 canonical implementation：reducer function 把 global state 折叠成 role-specific slice。

Per-agent projection 扩展得更远，但需要 schema。没有 schema 时，你会在每个 agent 的 prompt 里重建 ad-hoc projection。

### Write-contention patterns

多个 agents 同时写入是并发问题，不只是 LLM 问题。三种 pattern 有效：

- **Sequential writer (single producer)。** 所有 writes 通过一个 coordinator agent 串行化。简单，但形成 bottleneck。
- **Optimistic concurrency with versioning。** 每个 entry 有 version；writers 在 version mismatch 时 fail 并 retry。经典 database technique。
- **Topic partitioning。** 不同 agents 拥有不同 topics。没有 cross-topic contention。需要设计 partition boundaries。

多数 2026 frameworks 默认 sequential writer，因为 LLM calls 足够慢，contention 罕见，而且 bottleneck 不伤性能。

### 不可写 verifier

最关键的 mitigation 是 read-only verifier。实现规则：

- Verifier 与团队共享 state（读取 blackboard 或 pool）。
- Verifier 没有 shared state 的 write handle，只能写入单独的 verification channel。
- Verifier 独立 fetch writes 中引用的 sources。标记 disagreement。
- Verifier 自己的 outputs 会被路由给 human 或单独的 decision agent，绝不会 fed back into the pool。

没有这种隔离时，verifier 的 outputs 会变成 pool 中的新 entries，这意味着 poisoned pool 会 poison verifier，verifier 又会 poison 它的 verifications。

## 动手实现

`code/main.py` 用 stdlib Python 实现两种 topologies，加上一个 toy poisoning attack 和三种 mitigations。

- `MessagePool` — thread-safe append-only log with full read-out。
- `Blackboard` — topic-keyed pub/sub with per-agent subscriptions。
- `ProvenanceEntry` — 每次 write 记录 (writer, timestamp, prompt_hash, source_uri)。
- `PoisoningScenario` — 运行一个三 agent research task，其中 agent A 幻觉了一个小数点。打印 final report。
- `Verifier` — read-only agent，重新 fetch sources 并标记 inconsistencies。在 verifier present 的情况下运行同一场景。

运行：

```text
python3 code/main.py
```

预期输出：
- Run 1（no verifier）：幻觉出的 42% 传播到 final report。
- Run 2（with verifier）：verifier 标记 inconsistency，pool 被标记为 "flagged"，final report 包含 retraction。

## 实际使用

`outputs/skill-memory-auditor.md` 是一个 skill，用于审计任何 multi-agent system 的 shared-memory design 是否具备 provenance、versioning 和 verifier separation。在新的 multi-agent architectures 投产前运行它。

## 交付成果

对任何 shared-memory design：

- 在每次 write 上记录 provenance：`(writer, timestamp, prompt_hash, tool_calls_cited, source_uri)`。
- 让 log append-only。Corrections 是引用被 supersede 条目的新 entries。
- 部署至少一个拥有独立 source access 的 read-only verifier agent。
- 把 verifier output 路由到单独 channel，不要放回 shared pool。
- 记录 supersessions 在 writes 中的比例；比例上升是 hallucination patterns 的早期证据。

## 练习

1. 运行 `code/main.py`。确认 run 1 传播 hallucination，而 run 2 捕捉到它。
2. 添加第二个 hallucination：agent B 编造 dataset size。verifier 应该同时捕捉两者，而且不需要针对任一 hallucination hand-tuned。
3. 把 full pool 切换成带 topic partitions（`prices`、`summaries`、`analyses`）的 blackboard。Topic partitioning 会让哪些 poisoning scenarios 更难成功？又对哪些没有帮助？
4. 阅读 Hayes-Roth（1985, "A Blackboard Architecture for Control"）。找出论文中两个本课没有讨论、但 2026 系统会受益的 control patterns。
5. 阅读 CA-MCP（arXiv:2601.11595）。把它的 Shared Context Store 映射到 `code/main.py` 中的 MessagePool 或 Blackboard class。CA-MCP 在此基础上增加了哪些 primitives？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Message pool | “Shared chat history” | 每个 agent 都读取的 append-only log。完全透明，可扩展性差。 |
| Blackboard | “Shared workspace” | Topic-keyed pub/sub。Agents 订阅相关 topics。扩展得更远。 |
| Provenance | “谁写了什么” | 每次 write 的 metadata：writer、timestamp、prompt、sources。 |
| Memory poisoning | “幻觉传播” | 一个 agent 的错误进入 shared state，下游 agents 把它当成事实。 |
| Append-only | “没有 in-place updates” | Corrections 是 supersede 旧条目的新 entries。保留 audit trail。 |
| Unwritable verifier | “Independent auditor” | read-only agent，会重新 fetch sources 并标记 inconsistencies。 |
| Projection | “Scoped view” | 从 global state 计算出的 per-agent view。LangGraph reducers 是 canonical case。 |
| Knowledge Source | “Specialist agent” | Hayes-Roth 1985 对 blackboard participant 的术语。 |

## 延伸阅读

- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST taxonomy；memory poisoning 是 coordination-failure sub-family
- [CA-MCP — Context-Aware Multi-Server MCP](https://arxiv.org/abs/2601.11595) — 用于 coordinated MCP servers 的 Shared Context Store
- [Matrix — decentralized multi-agent framework](https://arxiv.org/abs/2511.21686) — 没有 central orchestrator 的 message-queue-based blackboard
- [LangGraph state and reducers](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — 生产中的 per-agent projection pattern
- [Anthropic — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — 来自 production deployment 的 provenance 和 verification notes
