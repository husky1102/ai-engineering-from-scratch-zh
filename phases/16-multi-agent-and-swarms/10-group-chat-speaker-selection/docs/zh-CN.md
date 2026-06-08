# Group Chat 与 Speaker Selection

> AutoGen GroupChat 和 AG2 GroupChat 在 N 个 agents 之间共享一个 conversation；selector function（LLM、round-robin 或 custom）选择谁下一个发言。这是 emergent multi-agent conversation 的原型——agents 不知道自己在 static graph 中的角色，它们只是响应 shared pool。AutoGen v0.2 的 GroupChat semantics 保留在 AG2 fork 中；AutoGen v0.4 把它重写成 event-driven actor model。Microsoft 在 2026 年 2 月把 AutoGen 放入 maintenance mode，并将其与 Semantic Kernel 合并为 Microsoft Agent Framework（RC February 2026）。GroupChat primitive 在 AG2 和 Microsoft Agent Framework 中都存活下来——学一次，到处用。

**类型：** 学习 + 构建
**语言：** Python (stdlib)
**先修：** Phase 16 · 04 (Primitive Model)
**时间：** ~60 分钟

## 要解决的问题

Static graphs（LangGraph）在 workflow 已知时很好。真实 conversations 并不是 static：有时 coder 问 reviewer，有时问 researcher，有时问 writer。Hardcoding every possible handoff 会产生 edge explosion。你想要的是 *agents reacting to a shared pool*，再用某个 function 决定谁说话 next。

这正是 AutoGen GroupChat 做的事。

## 核心概念

### 形状

```text
              ┌─── shared pool ────┐
              │   m1  m2  m3  ...  │
              └─────────┬──────────┘
                        │ (everyone reads all)
      ┌───────┬─────────┼─────────┬───────┐
      ▼       ▼         ▼         ▼       ▼
    Agent A  Agent B  Agent C  Agent D  Selector
                                           │
                                           ▼
                                  "next speaker = C"
```

每个 agent 都看见每条 message。每一 turn 都会调用一个 selector function 来选择谁下一个发言。

### 三种 selector flavors

**Round-robin。** Fixed cycle。Deterministic。随 N 线性扩展，但忽略 context——即使 topic 是 legal review，coder 也会拿到 turn。

**LLM-selected。** 一次 LLM call，读取 recent pool 并返回 best next speaker。Context-aware 但慢：每一 turn 都增加一次 LLM call。AutoGen 默认。

**Custom。** 一个带有你想要的任何逻辑的 Python function。Typical：LLM-selected with fallback rules（例如 “always give the verifier the turn after the coder”）。

### ConversableAgent API

```text
agent = ConversableAgent(
    name="coder",
    system_message="You write Python.",
    llm_config={...},
)
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

`GroupChatManager` 持有 selector。当一个 agent 完成 turn，manager 调用 selector，后者返回 next agent。Loop 持续到 termination condition。

### Termination

三种常见 patterns：

- **Max rounds。** Total turns 的 hard cap。
- **"TERMINATE" token。** Agents 可以 emit 一个 sentinel message；manager 看到它就停止。
- **Goal-reached check。** 一个 lightweight verifier 每 turn 运行，并在 chat 完成时停止。

### AutoGen → AG2 split 与 Microsoft Agent Framework merge

2025 年初，Microsoft 开始围绕 event-driven actor model 对 AutoGen（v0.4）进行 major rewrite。Community 将 AutoGen v0.2 的 GroupChat semantics fork 为 AG2，保留 early adopters 已经集成的 API。

2026 年 2 月，Microsoft 宣布 AutoGen 会进入 maintenance mode，其 event-driven actor model 合并进 **Microsoft Agent Framework**（RC February 2026，现在已与 Semantic Kernel 合并）。GroupChat concept 在两条路线中都存活；implementation details 不同。对于 v0.2-compatible code，AG2 是 preferred upstream。

### GroupChat 适合什么时候

- **Emergent conversations。** 你不想 pre-wire every possible next-speaker。
- **Role-mixing tasks。** Coder 问 researcher，researcher 问 archivist，archivist 又问 coder。Flow 不是 DAG。
- **Exploratory problem-solving。** 想象 “brainstorm meeting”，而不是 “assembly line”。

### 它什么时候失败

- **Strict determinism。** LLM selector 可能 inconsistent。同样 prompt，不同 runs，不同 next speakers。
- **Sycophancy cascades。** Agents 服从说得最 confident 的人。需要显式 counter-prompt。
- **Context bloat。** 每个 agent 读取每条 message；10 turns 后 context 很大。使用 projections（Lesson 15）来 scope views。
- **Hot speakers。** 一个 agent 因为 selector 偏好它的 specialties 而 dominate conversation。把 speaker balance 引入 selector feature。

### Group chat vs supervisor

同样的 primitives，不同 defaults：

- Supervisor：一个 agent plans，其他 agents execute。Selector 是 “ask the planner what to do.”
- Group chat：所有 agents 是 peers；selector 是 shared pool 上的 function。

两者都使用 Lesson 04 的四个 primitives。Group chat 默认 LLM-selected orchestration 和 full-pool shared state。

## 动手实现

`code/main.py` 用 stdlib 从零实现一个 GroupChat。三个 agents（coder、reviewer、manager）、round-robin 和 LLM-selected variants，以及基于 `TERMINATE` token 的 termination。

Demo 会打印 conversation transcript，以及两个 variants 的 selector decision trace。

运行：

```text
python3 code/main.py
```

## 实际使用

`outputs/skill-groupchat-selector.md` 为给定 task 配置 GroupChat selector——round-robin vs LLM-selected vs custom，以及要使用哪些 selector inputs（recent messages、agent specialties、turn counts）。

## 交付成果

Checklist：

- **Max rounds cap。** 永远设置。Typical tasks 为 10-20。
- **Speaker-balance metric。** Track turns per agent；当 imbalance 超过 threshold 时 alert。
- **Termination token。** `TERMINATE` 或 dedicated verifier agent。
- **Projection or scoped memory。** 约 10 messages 后，考虑只给每个 agent scoped view，以防止 context bloat。
- **Selector logging。** 对 LLM-selected variants，log selector input 和 choice。否则 debugging 不可能。

## 练习

1. 运行 `code/main.py`。比较 round-robin 与 LLM-selected 下的 conversation。每种情况下哪个 agent dominate？
2. 在 selector 中增加 “max-speaks-per-agent” rule。它如何影响 transcript？
3. 实现 goal-reached termination：当 reviewer 返回 “approved” 时停止。它有多频繁能在 round cap 之前触发？
4. 阅读 AutoGen stable docs 上的 GroupChat（https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html）。识别 `GroupChatManager` 使用的 default selector。
5. 阅读 AG2 repo（https://github.com/ag2ai/ag2），并比较它的 v0.2 GroupChat 与 v0.4 event-driven version。v0.4 增加了哪个具体 property（throughput、fault-tolerance、composability）？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| GroupChat | "Agents in one chat room" | Shared message pool + selector function。AutoGen / AG2 primitive。 |
| Speaker selection | "Who talks next" | 选择 next agent 的 function。Round-robin、LLM-selected 或 custom。 |
| GroupChatManager | "The meeting host" | AutoGen component，拥有 selector 并 loop over turns。 |
| ConversableAgent | "The base agent" | AutoGen base class；可以 send 和 receive messages 的 agent。 |
| Termination token | "The 'stop' word" | 结束 chat 的 sentinel string（通常是 `TERMINATE`）。 |
| Hot speaker | "One agent dominates" | Selector 持续选择同一个 agent 的 failure mode。 |
| Context bloat | "Pool grows unbounded" | 每个 agent 读取每条 prior message；context 随 turns 增长。 |
| Projection | "Scoped view" | Shared pool 中的 role-specific view，用来防止 context bloat。 |

## 延伸阅读

- [AutoGen group chat docs](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html) —— reference implementation
- [AG2 repo](https://github.com/ag2ai/ag2) —— community AutoGen v0.2 continuation
- [Microsoft Agent Framework docs](https://microsoft.github.io/agent-framework/) —— merged successor，RC February 2026
- [AutoGen v0.4 release notes](https://microsoft.github.io/autogen/stable/) —— event-driven actor model rewrite details
