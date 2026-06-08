# Handoffs 与 Routines：无状态编排

> OpenAI 的 Swarm（2024 年 10 月）把多智能体编排提炼成两个原语：**routines**（把 instructions + tools 放进 system prompt）和 **handoffs**（返回另一个 Agent 的工具）。没有状态机，没有分支 DSL：LLM 通过调用正确的 handoff tool 来路由。OpenAI Agents SDK（2025 年 3 月）是它的生产级继任者。Swarm 本身仍然是最干净的概念参考：全部源码只有几百行。这个模式传播很快，因为 API 表面大致就是“agent = prompt + tools；handoff = function returning agent”。局限：它是无状态的，所以 memory 是调用方的问题。

**类型：** Learn + Build
**语言：** Python (stdlib)
**先修：** Phase 16 · 04 (Primitive Model)
**时间：** ~60 分钟

## 要解决的问题

每个多智能体框架都想让你学习它的 DSL：LangGraph 的节点和边，CrewAI 的 crews 和 tasks，AutoGen 的 GroupChat 和 managers。这些 DSL 确实是真实的抽象，但它们会让事情感觉比实际需要更重。

Swarm 走向相反方向：使用模型已经具备的 tool-calling 能力。Handoffs 变成 tool calls。编排器就是当前持有对话的那个 agent。状态机隐含在各个 agent 的 system prompts 里。

## 核心概念

### 两个原语

**Routine。** 定义一个 agent 角色和可用工具的 system prompt。可以把它看成有作用域的一组指令：“你是 triage agent；如果用户询问退款，就 hand off 给 refund agent。”

**Handoff。** agent 可以调用的一个工具，它返回新的 Agent 对象。Swarm runtime 检测到 Agent 返回值后，会在下一轮切换 active agent。

这就是全部抽象。

```text
def transfer_to_refunds():
    return refund_agent  # Swarm sees Agent return → switch active agent

triage_agent = Agent(
    name="triage",
    instructions="Route the user to the right specialist.",
    functions=[transfer_to_refunds, transfer_to_sales, transfer_to_support],
)
```

triage agent 的 system prompt 会让它根据用户消息选择正确的 handoff。LLM 的 tool-calling 负责路由。

### 为什么它会传播

- **API 很小。** 只需要学习两个概念。
- **使用模型已经会做的事。** Tool calling 已经在多个提供商那里达到生产级。
- **没有状态机负担。** 你不描述 graph；agent 的 prompts 描述它们会 hand off 给谁。

### 无状态权衡

Swarm 明确在两次运行之间保持无状态。框架会在一次 run 期间保留 message history，但不会持久化任何东西。Memory、连续性、长任务：全部都是调用方的问题。

在生产中（OpenAI Agents SDK，2025 年 3 月），这是主要变化之一：SDK 在保留 handoff 原语的同时，增加了内置 session management、guardrails 和 tracing。

### Swarm/handoffs 适合什么

- **Triage patterns。** 前线 agent 把用户路由给 specialist。
- **Skill-based handoffs。** “如果任务需要代码，调用 coder；如果需要研究，调用 researcher。”
- **短而有边界的对话。** Customer support、FAQ-to-ticket、简单 workflow。

### Swarm 的困难场景

- **带共享 memory 的长 session。** Handoff 会把 conversation state 重置成新 agent 的 prompt 加 history。如果没有调用方管理 memory，就没有跨 agent 的持久状态。
- **并行执行。** Handoff 是一次一个：active agent 被切换。并行性需要调用方编排多个 Swarm runs。
- **审计和回放。** 无状态 runs 很难精确回放；LLM 的 handoff 选择不是确定性的。

### OpenAI Agents SDK（2025 年 3 月）

生产级继任者增加了：

- **Session state。** 跨 runs 的持久 thread。
- **Guardrails。** 输入/输出校验 hooks。
- **Tracing。** 每个 tool call 和 handoff 都会被记录。
- **Handoff filters。** 控制 handoff 时转移哪些上下文。

handoff 原语保留下来；生产工程体验被加在它周围。

### Swarm vs GroupChat

两者都使用 LLM-driven routing，但区别在于**谁选择下一个**：

- GroupChat：selector（function 或 LLM）从外部选择下一个 speaker。
- Swarm：current agent 通过调用 handoff tool 选择它的继任者。

Swarm 是“agent decides what's next”；GroupChat 是“manager decides what's next”。Swarm 的决策存在 active agent 的 tool call 里；GroupChat 的决策存在 `GroupChatManager` 里。

## 动手实现

`code/main.py` 从零实现 Swarm：一个 Agent dataclass，一个 handoff 机制（tool 返回 Agent），以及一个检测 agent switches 的 run loop。

Demo：triage agent 路由到 refund、sales 或 support specialists。每个 specialist 都有自己的 tools。run loop 会打印每次 handoff。

运行：

```text
python3 code/main.py
```

## 实际使用

`outputs/skill-handoff-designer.md` 会为给定任务设计 handoff topology：有哪些 agents、它们可以调用哪些 handoffs、转移哪些 context。

## 交付成果

Checklist：

- **Handoff logging。** 每次 handoff 都写入 trace event，包含 from-agent、to-agent、context snapshot。
- **Context transfer rules。** 决定 handoff 时移动什么：完整 history（昂贵）、最后 N 条 messages，或 summary。
- **Guardrail on handoff。** handoff 到拥有不同 tool permissions 的 specialist 时必须经过认证；否则 prompt injection 可以强迫不想要的 handoffs。
- **Loop detection。** 两个 agents 互相 hand back and forth 是常见故障；用简单的 last-K ring check 检测。
- **Fallback agent。** 如果 handoff target 不存在，回退到安全默认值。

## 练习

1. 运行 `code/main.py`，triage 到 refund agent。确认第二轮的 active agent 是 refund。
2. 添加 loop-detection 规则：如果同两个 agents 连续 hand off 了 3 次，强制退出。设计 fallback。
3. 阅读 OpenAI Agents SDK 关于 handoff filters 的文档。实现一个“summarize-on-handoff”版本：outgoing agent 在 incoming agent 接管前，把 context 压缩成 bullet summary。
4. 比较 Swarm handoff 和 GroupChatManager selector。哪种模式会让 prompt injection 更糟，为什么？
5. 阅读 Swarm cookbook（https://developers.openai.com/cookbook/examples/orchestrating_agents）。找出 Swarm 作出的一个明确设计决策，并判断 OpenAI Agents SDK 是改变了它还是保留了它。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Routine | “agent prompt” | System prompt + tool list。定义 role 和可用 handoffs。 |
| Handoff | “转给另一个 agent” | active agent 可以调用的一个 tool，它返回新的 Agent。runtime 会切换 active agent。 |
| Stateless | “runs 之间没有 memory” | Swarm 不持久化任何东西；memory 是调用方的责任。 |
| Active agent | “现在谁在说话” | 当前持有对话的 agent。Handoff 会改变它。 |
| Context transfer | “handoff 时移动什么” | incoming agent 能看到哪些 history 的策略：full、last N，或 summarized。 |
| Handoff loop | “agents ping-pong” | 两个 agents 不断把控制权交回给对方的 failure mode。 |
| OpenAI Agents SDK | “Production Swarm” | 2025 年 3 月的继任者；在 handoff 原语上增加 sessions、guardrails、tracing。 |
| Handoff filter | “transfer gate” | SDK feature，用来在 handoff 边界检查和修改 context。 |

## 延伸阅读

- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — 参考表述
- [OpenAI Swarm repo](https://github.com/openai/swarm) — 原始实现，保留为概念参考
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — 带 sessions 和 tracing 的生产级继任者
- [Anthropic handoff-in-Claude notes](https://docs.anthropic.com/en/docs/claude-code) — Claude Code subagents 如何通过 `Task` 使用类似 handoff 的模式
