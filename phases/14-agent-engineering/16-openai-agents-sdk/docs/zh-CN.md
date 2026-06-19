# OpenAI Agents SDK：交接、护栏与追踪

> OpenAI Agents SDK 是构建在 Responses API 之上的轻量 multi-agent framework。五个 primitives：Agent、Handoff、Guardrail、Session、Tracing。Handoffs 是名为 `transfer_to_<agent>` 的 tools。Guardrails 会在 input 或 output 上触发。Tracing 默认开启。

**类型:** 学习 + 构建
**语言:** Python (stdlib)
**先修:** Phase 14 · 01 (Agent Loop), Phase 14 · 06 (Tool Use)
**时间:** ~75 分钟

## 学习目标

- 说出 OpenAI Agents SDK 的五个 primitives。
- 解释 handoffs：为什么它们被建模为 tools，模型看到的 name shape 是什么，以及 context 如何 transfer。
- 区分 input guardrails、output guardrails 和 tool guardrails；解释 `run_in_parallel` 与 blocking mode。
- 实现一个包含 handoffs + guardrails + span-style tracing 的 stdlib runtime。

## 要解决的问题

无法干净 delegation 的 agents 最终会把所有内容塞进一个 prompt。没有 guardrails 的 agents 会发出 PII、policy-violating output，或无限循环。OpenAI 的 SDK 将让 multi-agent work 可控的三个 primitives 固化下来。

## 核心概念

### 五个 primitives

1. **Agent。** LLM + instructions + tools + handoffs。
2. **Handoff。** Delegation 到另一个 agent。向模型呈现为名为 `transfer_to_<agent_name>` 的 tool。
3. **Guardrail。** 在 input（仅第一个 agent）、output（仅最后一个 agent）或 tool invocation（每个 function tool）上进行 validation。
4. **Session。** 跨 turns 自动保存 conversation history。
5. **Tracing。** LLM generations、tool calls、handoffs、guardrails 的内置 spans。

### Handoffs as tools

模型会在 tool list 中看到 `transfer_to_billing_agent`。调用它表示 runtime 要：

1. 复制 conversation context（或通过 beta 版 `nest_handoff_history` 折叠它）。
2. 用目标 agent 的 instructions 初始化目标 agent。
3. 继续让目标 agent 运行。

这是产品化后的 supervisor pattern（Lesson 13 / Lesson 28）。

### Guardrails

三种类型：

- **Input guardrails。** 在第一个 agent 的 input 上运行。任何 LLM call 之前就拒绝 unsafe 或 out-of-scope requests。
- **Output guardrails。** 在最后一个 agent 的 output 上运行。捕获 PII leaks、policy violations、malformed responses。
- **Tool guardrails。** 按每个 function-tool 运行。验证 arguments、检查 permissions、审计 execution。

Mode：

- **Parallel**（默认）。Guardrail LLM 与 main LLM 并行运行。Tail latency 更低。如果触发，main LLM 的工作会被丢弃（浪费 tokens）。
- **Blocking**（`run_in_parallel=False`）。Guardrail LLM 先运行。如果触发，不会浪费 main call tokens。

Tripwires 会抛出 `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`。

### Tracing

默认开启。每次 LLM generation、tool call、handoff 和 guardrail 都会发出一个 span。`OPENAI_AGENTS_DISABLE_TRACING=1` 可退出。`add_trace_processor(processor)` 会把 spans 同时扇出到你自己的 backend 和 OpenAI。

### Sessions

`Session` 将 conversation history 存到 backend（SQLite、Redis、自定义）。`Runner.run(agent, input, session=session)` 会自动加载并追加。

### 这个模式容易出错的地方

- **Handoff drift。** Agent A hand off 给 Agent B，B 又 hand back 给 A。添加 hop counter。
- **Guardrail bypass。** Tool guardrails 只在 function tools 上触发；built-in tools（file reader、web fetch）需要单独 policy。
- **Over-tracing。** Spans 中有敏感内容。与 OTel GenAI content-capture rules（Lesson 23）配套 -- 外部存储，用 ID 引用。

## 动手实现

`code/main.py` 用 stdlib 实现 SDK 形态：

- `Agent`、`FunctionTool`、`Handoff`（作为带 transfer semantics 的 function tool）。
- `Runner`，包含 input/output/tool guardrails、handoff dispatch 和 hop counter。
- 一个 simple span emitter，用于展示 trace shape。
- 一个 triage agent，会基于用户 query hand off 到 billing 或 support；某个 input 会触发 guardrail。

运行：

```text
python3 code/main.py
```

Trace 会展示两次成功 handoffs、一次 input guardrail trip，以及一棵与真实 SDK 发出的结构相似的 span tree。

## 实际使用

- **OpenAI Agents SDK** 用于 OpenAI-first products。
- **Claude Agent SDK**（Lesson 17）用于 Claude-first products。
- **LangGraph**（Lesson 13）用于需要 explicit state 和 durable resume 的场景。
- **Custom** 用于需要精确控制（voice、multi-provider、federated deployments）的场景。

## 交付成果

`outputs/skill-agents-sdk-scaffold.md` 会 scaffold 一个 Agents SDK app，包含 triage agent、handoffs、input/output/tool guardrails、session store 和 trace processor。

## 练习

1. 添加 handoff hop counter：N 次 transfers 后拒绝。跟踪 behavior。
2. 把 `nest_handoff_history` 实现成一个 option -- transfer 前把 prior messages 折叠成一段 summary。
3. 编写一个 blocking output guardrail。比较会触发它的 prompts 和会通过的 prompts 的 latency。
4. 把 `add_trace_processor` 接到 JSON logger。每个 span 发出的 shape 是什么？
5. 阅读 SDK docs。把你的 stdlib toy 移植到 `openai-agents-python`。你有哪些地方建模错了？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Agent | "LLM + instructions" | SDK 中的 Agent type；拥有 tools 和 handoffs |
| Handoff | "Transfer" | 模型调用以 delegate 给另一个 agent 的 tool |
| Guardrail | "Policy check" | Input / output / tool invocation 上的 validation |
| Tripwire | "Guardrail trip" | Guardrail 拒绝时抛出的 exception |
| Session | "History store" | Runs 之间持久化的 conversation memory |
| Tracing | "Spans" | 对 LLM + tool + handoff + guardrail 的内置 observability |
| Blocking guardrail | "Sequential check" | Guardrail 先运行；触发时不浪费 tokens |
| Parallel guardrail | "Concurrent check" | Guardrail 并行运行；latency 更低，触发时浪费 tokens |

## 延伸阅读

- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) -- primitives、handoffs、guardrails、tracing
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) -- Claude-flavored counterpart
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) -- 何时该使用 handoffs
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) -- Agents SDK spans 映射到的标准
