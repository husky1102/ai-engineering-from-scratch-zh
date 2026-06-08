# FIPA-ACL 与 Speech Acts 的遗产

> 在 MCP 之前，在 A2A 之前，有 FIPA-ACL。2000 年，IEEE Foundation for Intelligent Physical Agents 批准了一种 agent communication language，包含二十个 performatives、两种 content languages，以及一组 interaction protocols：contract net、subscribe/notify、request-when。它淡出工业界，是因为 ontology overhead 对 web 来说太重，但 LLM 带来的 multi-agent systems 复兴，正在悄悄重新实现同样的想法，只是没有 formal semantics：JSON contracts 代替 performatives，natural language 代替 ontologies。本课认真阅读 FIPA-ACL，让你看清 2026 年的 protocol decisions 哪些是重新发明、哪些是真正的新东西，以及当前浪潮会在哪里重新发现 2000 年代已经解决过的问题。

**类型：** 学习
**语言：** Python (stdlib)
**先修：** Phase 16 · 01 (Why Multi-Agent)
**时间：** ~60 分钟

## 要解决的问题

2026 年的 agent-protocol landscape 很拥挤：MCP 用于 tools，A2A 用于 agents，ACP 用于 enterprise audit，ANP 用于 decentralized trust，NLIP 用于 natural-language content，还有 CA-MCP 和二十多个 research proposals。每个 spec 都宣称自己 foundational。

诚实的读法是：它们大多在重新发现一棵非常具体、已有二十年历史的 decision tree。Austin（1962）和 Searle（1969）的 speech-act theory 给了我们“utterances are actions”。KQML（1993）把它转成 wire protocol。FIPA-ACL（2000 年批准）产出了 reference standardization：二十个 performatives、content languages SL0/SL1，以及用于 contract-net 和 subscribe-notify 的 interaction protocols。JADE 和 JACK 是 Java reference platforms。这个努力在 2010 年左右淡出，因为 ontology overhead 太重，而 web 正在赢。

当你看 MCP 的 `tools/call`、A2A 的 task lifecycle，或 CA-MCP 的 shared context store 时，你看到的是 FIPA decisions 的更软、JSON-native rehash。知道这段 heritage 会告诉你两件事：哪些新的 “innovations” 实际上是 reinventions，以及新 specs 将重新发现哪些旧 failure modes。

## 核心概念

### 一段话里的 Speech acts

Austin 注意到，有些句子不是描述世界，而是改变世界。“I promise.” “I request.” “I declare.” 他称这些为 performative utterances。Searle 形式化了五类：assertive、directive、commissive、expressive、declarative。KQML（Finin et al., 1993）把它 operationalize 到 software agents：一条 message 是 performative（action）加 content（action 作用的对象）。FIPA-ACL 清理了 KQML 的 gaps，并围绕二十个 performatives 标准化。

### 二十个 FIPA performatives（部分列表）

| Performative | Intent |
|---|---|
| `inform` | “I tell you P is true” |
| `request` | “I ask you to do X” |
| `query-if` | “Is P true?” |
| `query-ref` | “What is the value of X?” |
| `propose` | “I propose we do X” |
| `accept-proposal` | “I accept the proposal” |
| `reject-proposal` | “I reject the proposal” |
| `agree` | “I agree to do X” |
| `refuse` | “I refuse to do X” |
| `confirm` | “I confirm P is true” |
| `disconfirm` | “I deny P” |
| `not-understood` | “Your message did not parse” |
| `cfp` | “Call for proposals on X” |
| `subscribe` | “Notify me when X changes” |
| `cancel` | “Cancel the ongoing X” |
| `failure` | “I tried X and failed” |

完整列表在 `fipa00037.pdf`（FIPA ACL Message Structure）中。重点不是背诵它，重点是每一个都会对应到 LLM protocol 最终重新添加的 primitive。

### Canonical FIPA-ACL message

```text
(inform
  :sender       agent1@platform
  :receiver     agent2@platform
  :content      "((price IBM 83))"
  :language     SL0
  :ontology     finance
  :protocol     fipa-request
  :conversation-id   conv-42
  :reply-with   msg-17
)
```

七个字段承载 protocol envelope；一个字段（`content`）承载 payload。其余字段正是你每次给 JSON protocol 补 retries、threading 和 ontology 时都会重新发明的东西。

### 两个 legacy platforms

**JADE**（Java Agent DEvelopment framework，1999-2020s）是最常用的 FIPA-compliant runtime。Agents 扩展 base class、交换 ACL messages、运行在 containers 中，并用 “behaviors” 协调。interaction-protocol library 内置 contract-net、subscribe-notify、request-when 和 propose-accept。

**JACK**（Agent Oriented Software，commercial）强调在 FIPA messages 之上的 BDI（Belief-Desire-Intention）reasoning。更 formal，采用更少。

二者都在 web stack 吃掉 multi-agent use cases 后衰退。MCP 和 A2A 是 2026 年的 runtime “containers”。

### 为什么 FIPA 淡出

- **Ontology overhead.** FIPA 要求 shared ontology 来 parse `content`。就 ontologies 达成共识是多年 standards process。web 只是用了 HTTP + JSON。
- **Formal semantics nobody used.** SL（Semantic Language）给出 rigorous truth conditions，但大多数 production systems 使用 free-form content，并忽略 formalism。
- **Tooling lock-in.** JADE 只支持 Java；JACK 是 commercial。Polyglot teams 绕开了二者。
- **The internet won the stack.** REST、然后 JSON-RPC、然后 gRPC 取代了 ACL 的 transport。

### LLM revival 是 FIPA-lite

比较 FIPA `request` 和 MCP `tools/call`：

```text
(request                                {
  :sender  agent1                         "jsonrpc": "2.0",
  :receiver tool-server                   "method":  "tools/call",
  :content "(lookup stock IBM)"           "params":  {"name":"lookup_stock",
  :ontology finance                                   "arguments":{"symbol":"IBM"}},
  :conversation-id c42                    "id": 42
)                                        }
```

同一个 envelope，不同 syntax。二者都携带：who、whom、intent、payload、correlation id。它们彼此都不是革命；它们是在同一设计上的不同 trade-offs。

Liu et al. 2025 年 survey（“A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP”, arXiv:2505.02279）明确指出这条 lineage：MCP 对应 tool-use speech acts，A2A 对应 agent-peer speech acts，ACP 对应 audit-trail speech acts，ANP 对应 decentralized-identity extensions。新 specs 是 ACL descendants，使用 JSON syntax 和更松的 semantics。

### 直白陈述这个 trade-off

**FIPA 给了你而现代 specs 放弃的东西：**

- Formal semantics：你可以证明 `inform` 意味着 sender 相信 content。
- Canonical catalog of performatives：你不用重新争论“我们是否应该有 `cancel`？”。
- 数十年的 interaction-protocol patterns：contract-net、subscribe-notify、propose-accept，并带有已知 correctness properties。

**现代 specs 给了你而 FIPA 没有的东西：**

- 与所有现代 tool 兼容的 JSON-native payloads。
- LLMs 可以在没有 hand-coded ontology 的情况下解释的 natural-language content。
- Web-stack transport（HTTP、SSE、WebSocket）。
- 通过 self-describing documents 做 capability discovery（MCP `listTools`、A2A Agent Card）。

为了更容易实现，牺牲更松的 intent semantics。这就是 exact trade。

### 值得移植的 Interaction protocols

FIPA 交付了约 15 个 interaction protocols。三个值得带入 LLM multi-agent systems：

1. **Contract Net Protocol (CNP).** Manager 发出 `cfp`（call for proposals）；bidders 用 `propose` 回复；manager accept/reject。这是 canonical task-market pattern（Phase 16 · 16 Negotiation）。
2. **Subscribe/Notify.** Subscriber 发送 `subscribe`；topic 改变时，publisher 发送 `inform`。这是 2026 年的每个 event-bus。
3. **Request-When.** “当 condition Y 成立时执行 X。” 带 pre-conditions 的 delayed-action。2026 年 analog 是 durable workflow engines 中的 deferred tasks（Phase 16 · 22 Production Scaling）。

每一个都能干净映射到现代 message queues、HTTP + polling 或 SSE streaming。

### 放弃 ontology 后会破什么

没有 shared ontology，agents 会从 natural-language content 推断 meaning。2026 年记录下来的 failure mode 是 **semantic drift**：两个 agents 用同一个词（`"customer"`）表示细微不同的 concepts，receiver 的 agent 按错误解释行动，没有 schema validator 捕获它。FIPA 的 ontology requirement 会在 parse time 拒绝这条 message。

不走 full ontology 的 mitigations：

- `content` 上的 JSON Schema：在线上拒绝 structural errors。
- Typed artifacts（A2A）：拒绝错误 modality。
- envelope 中的 explicit performative：即使 content 是 natural language，也能让 intent 无歧义。

### 2026 specs 映射到 speech-act heritage

| Modern spec | FIPA analog | What it keeps | What it drops |
|---|---|---|---|
| MCP `tools/call` | `request` | explicit intent, correlation id | formal semantics, ontology |
| MCP `resources/read` | `query-ref` | explicit intent, correlation id | formal semantics |
| A2A Task lifecycle | contract-net + request-when | async lifecycle, state transitions | formal completeness guarantees |
| A2A streaming events | subscribe/notify | async push | typed-predicate subscription |
| CA-MCP shared context | blackboard (Hayes-Roth 1985) | multi-writer shared memory | logical consistency model |
| NLIP | natural-language content | LLM-native | schema |

自上而下阅读这张表，pattern 是：保留 structural primitive，丢掉 formalism，让 LLMs 填补 ambiguity。

## 动手实现

`code/main.py` 实现一个 pure-stdlib FIPA-ACL translator。它 encode 和 decode canonical ACL envelope，并展示每个 MCP / A2A message shape 如何 reduce 到同样七个字段。demo：

- 将五条 MCP-style 和 A2A-style messages encode 为 FIPA-ACL。
- 将 FIPA-ACL decode 回 modern equivalent。
- 用 `cfp`、`propose`、`accept-proposal`、`reject-proposal` 在一个 manager 和三个 bidders 之间运行 toy Contract Net negotiation。

运行：

```text
python3 code/main.py
```

输出是一段 side-by-side trace，展示每条 modern message 的 2026 JSON form 和 FIPA-ACL form，然后展示一次 contract-net bid 的 round-trip。相同 protocol primitives 在 round-trip 后幸存；只有 syntax 不同。

## 实际使用

`outputs/skill-fipa-mapper.md` 是一个 skill，读取任何 agent-protocol spec 并生成 FIPA-ACL mapping。在采用新 protocol 前用它回答：“这是真的新东西，还是带 JSON syntax 的 `inform`？”

## 交付成果

不要把 FIPA-ACL 带回来。带回它的 checklist：

- 每条 message 的 intent primitive（performative）是什么？
- request-response 和 cancellation 是否有 correlation id？
- 是否有 explicit content language（JSON-RPC、plain text、structured typed artifact）？
- interaction protocols 是 first-class，还是你正在从零重新实现 contract-net？
- 当两个 agents 对 content meaning 产生分歧（semantic drift）时会发生什么？

在把任何新 protocol 交付到生产前，记录这五个问题。

## 练习

1. 运行 `code/main.py`。观察 round-trip encoding。识别哪个 FIPA performative 对应 `tools/call`、`resources/read` 和 A2A task creation。
2. 用 `cancel` performative 扩展 contract-net demo，让 manager 能在 mid-bid 撤回 task。`cancel` 解决了 retries alone 无法解决的什么 failure case？
3. 阅读 FIPA ACL Message Structure（http://www.fipa.org/specs/fipa00037/）sections 4.1-4.3。选择一个本课未覆盖的 performative，并描述它的 modern JSON-RPC analog。
4. 阅读 Liu et al., arXiv:2505.02279。对 MCP、A2A、ACP、ANP 分别列出它们保留和丢弃的 FIPA performative families。
5. 为你自己系统中 `request` performative 的 `content` field 设计一个 minimal JSON-Schema。这个 schema 给了你 pure natural-language 没有的什么东西？代价是什么？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Speech act | “An utterance that does something” | Austin/Searle：utterances as actions。ACL 的 theoretical parent。 |
| FIPA | “That old XML thing” | IEEE Foundation for Intelligent Physical Agents。2000 年 standardized ACL。 |
| ACL | “Agent Communication Language” | FIPA 的 envelope format：performative + content + metadata。 |
| Performative | “The verb” | message 的 intent class：`inform`、`request`、`propose`、`cfp` 等。 |
| KQML | “FIPA's predecessor” | Knowledge Query and Manipulation Language（1993）。更简单、更窄。 |
| Ontology | “Shared vocabulary” | content language 所谈 concepts 的 formal definition。 |
| SL0 / SL1 | “FIPA content languages” | Semantic Language levels 0 and 1：formal content language family。 |
| Contract Net | “Task market” | Manager 发出 cfp；bidders propose；manager accepts。canonical interaction protocol。 |
| Interaction protocol | “Pattern of messages” | 一组带 known correctness 的 performatives 序列：request-when、subscribe-notify 等。 |

## 延伸阅读

- [Liu et al. — A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP](https://arxiv.org/html/2505.02279v1) — 连接 modern specs 与 FIPA heritage 的 canonical 2025 survey
- [FIPA ACL Message Structure Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — 2000 年批准的 envelope format
- [FIPA Communicative Act Library Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — 完整 performative catalog
- [MCP specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — `request`/`query-ref` 的 modern tool-use equivalent
- [A2A specification](https://a2a-protocol.org/latest/specification/) — contract-net 和 subscribe-notify 的 modern agent-peer equivalent
