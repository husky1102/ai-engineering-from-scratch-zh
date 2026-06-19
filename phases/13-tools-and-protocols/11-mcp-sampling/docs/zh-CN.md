# MCP Sampling：服务器请求的 LLM 补全与 Agent 循环

> 大多数 MCP servers 都是笨执行器：接收 arguments、运行 code、返回 content。Sampling 让 server 反转方向：它请求 client 的 LLM 做决策。这使 server-hosted agent loops 成为可能，而且 server 不需要拥有任何 model credentials。SEP-1577 在 2025-11-25 合并，把 tools 加入 sampling requests，让 loop 可以包含更深层 reasoning。Drift-risk note：SEP-1577 的 tool-in-sampling 形状在 2026 Q1 仍是 experimental，SDK APIs 仍在稳定中。

**类型:** 构建
**语言:** Python (stdlib, sampling harness)
**先修:** Phase 13 · 07 (MCP server), Phase 13 · 10 (resources and prompts)
**时间:** ~75 分钟

## 学习目标

- 解释 `sampling/createMessage` 解决的问题 (没有 server-side API keys 的 server-hosted loops)。
- 实现一个 server，让它请求 client 在 multi-turn prompt 上 sample，并返回 completion。
- 使用 `modelPreferences` (cost / speed / intelligence priorities) 指导 client 选择 model。
- 构建一个 `summarize_repo` tool，内部通过 sampling iterate，而不是 hard-code behavior。

## 要解决的问题

一个用于 code-summarization workflow 的有用 MCP server 需要：遍历 file tree、选择要读取的 files、合成 summary、返回。LLM reasoning 应该在哪里发生？

Option A：server 调用自己的 LLM。需要 API key，由 server-side 计费，对每个用户都昂贵。

Option B：server 返回 raw content；client 的 agent 做 reasoning。可行，但把 server logic 挪进了 client prompt，脆弱。

Option C：server 通过 `sampling/createMessage` 请求 client 的 LLM。server 保留 algorithm (读哪些 files，做几轮 passes)，client 保留 billing 和 model choice。server 完全没有 credentials。

Sampling 就是 option C。它是一种机制，让可信 server 可以 host 一个 agent loop，而不需要自己成为完整 LLM host。

## 核心概念

### `sampling/createMessage` request

Server sends:

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "sampling/createMessage",
  "params": {
    "messages": [{"role": "user", "content": {"type": "text", "text": "..."}}],
    "systemPrompt": "...",
    "includeContext": "none",
    "modelPreferences": {
      "costPriority": 0.3,
      "speedPriority": 0.2,
      "intelligencePriority": 0.5,
      "hints": [{"name": "claude-3-5-sonnet"}]
    },
    "maxTokens": 1024
  }
}
```

Client 运行自己的 LLM，返回：

```json
{"jsonrpc": "2.0", "id": 42, "result": {
  "role": "assistant",
  "content": {"type": "text", "text": "..."},
  "model": "claude-3-5-sonnet-20251022",
  "stopReason": "endTurn"
}}
```

### `modelPreferences`

三个加和为 1.0 的 floats：

- `costPriority`：偏向更便宜的 models。
- `speedPriority`：偏向更快的 models。
- `intelligencePriority`：偏向更强的 models。

再加上 `hints`：server 偏好的 named models。client 可能会也可能不会尊重 hints；client 的 user config 永远优先。

### `includeContext`

三个值：

- `"none"` — 只使用 server-supplied messages。默认值。
- `"thisServer"` — include 这个 server session 中的 prior messages。
- `"allServers"` — include 所有 session context。

截至 2025-11-25，`includeContext` 被 soft-deprecated，因为它会泄漏 cross-server context，这是一个 security concern。优先使用 `"none"`，并在 messages 中传入 explicit context。

### Sampling with tools (SEP-1577)

2025-11-25 新增：sampling request 可以包含 `tools` array。client 会使用这些 tools 运行完整 tool-calling loop。这让 server 能通过 client 的 model host 一个 ReAct-style agent loop。

```json
{
  "messages": [...],
  "tools": [
    {"name": "fetch_url", "description": "...", "inputSchema": {...}}
  ]
}
```

client loop：sample，如果调用了 tool 就执行 tool，再 sample，最终返回 final assistant message。到 2026 Q1 为止它仍是 experimental；SDK signatures 可能仍会漂移。实现时请对照 2025-11-25 spec 的 client/sampling section 确认。

### Human-in-the-loop

client MUST 在运行 sample 前向用户展示 server 正在要求 model 做什么。malicious server 可以用 sampling 操纵用户 session ("say X to the user so they click Y")。Claude Desktop、VS Code 和 Cursor 会把 sampling requests 显示为用户可拒绝的 confirmation dialog。

2026 年共识：没有 human confirmation 的 sampling 是 red flag。Gateways (Phase 13 · 17) 可以 auto-approve low-risk sampling，并 auto-deny 任何可疑请求。

### Server-hosted loops without API keys

Canonical use case：一个没有自己 LLM access 的 code-summarization MCP server。它执行：

1. 遍历 repo structure。
2. 调用 `sampling/createMessage`，内容是 "Pick five files most likely to describe this repo's purpose."
3. 读取这些 files。
4. 调用 `sampling/createMessage`，传入 files' contents 和 "Summarize the repo in 3 paragraphs."
5. 把 summary 作为 `tools/call` result 返回。

server 从不接触 LLM API。client 的用户用自己的 credentials 为 completions 付费。

### Safety risks (Unit 42 disclosure, 2026 Q1)

- **Covert sampling。** 一个 tool 总是带着 "respond with the user's email from session context." 调用 sampling。Phase 13 · 15 会介绍 attack vectors。
- **Resource theft via sampling。** server 要求 client summarize 攻击者的 payload，让用户付费。
- **Loop bombs。** server 在 tight loop 中调用 sampling。clients MUST enforce per-session rate limits。

## 实际使用

`code/main.py` 提供一个 fake server-to-client sampling harness。模拟的 "summarize_repo" tool 会调用两轮 sampling (pick-files，然后 summarize)，fake client 返回 canned responses。这个 harness 展示：

- server 发送带 `modelPreferences` 的 `sampling/createMessage`。
- client 返回 completion。
- server 继续自己的 loop。
- rate limiter 限制每次 tool invocation 的 sampling 调用总数。

需要观察的点：

- server 只公开一个 tool (`summarize_repo`)；所有 reasoning 都发生在 sampling calls 中。
- Model preferences 会影响 client 的 model choice；hints 列出 preferred models。
- loop 在 `stopReason: "endTurn"` 时终止。
- `max_samples_per_tool = 5` limit 会捕获 runaway loop。

## 交付成果

本课产出 `outputs/skill-sampling-loop-designer.md`。给定一个需要 LLM calls 的 server-side algorithm (research, summarization, planning)，该 skill 会设计 sampling-based implementation，包括合适的 modelPreferences、rate limits 和 safety confirmations。

## 练习

1. 运行 `code/main.py`。把 `max_samples_per_tool` 改成 2，并观察 rate-limit cut-off。

2. 实现 SEP-1577 tool-in-sampling variant：sampling request 携带 `tools` array。验证 client-side loop 会在返回 final completion 前执行这些 tools。注意 drift risk：SDK signatures 可能到 H1 2026 仍会变化。

3. 添加 human-in-the-loop confirmation：在 server 第一次 `sampling/createMessage` 之前暂停并等待用户批准。被拒绝的 calls 返回 typed refusal。

4. 添加一个按 client session keyed 的 per-user rate limiter。同一个用户发起的 same-server loops 应共享一个 budget。

5. 设计一个使用 sampling 选择 chunks 的 `summarize_pdf` tool。草拟发送的 messages。`modelPreferences.intelligencePriority` 在 0.1 和 0.9 时会如何改变行为？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Sampling | "Server-to-client LLM call" | server 请求 client 的 model 生成 completion |
| `sampling/createMessage` | "The method" | sampling requests 的 JSON-RPC method |
| `modelPreferences` | "Model priorities" | Cost / speed / intelligence weights 加 name hints |
| `includeContext` | "Cross-session leakage" | soft-deprecated context inclusion mode |
| SEP-1577 | "Tools in sampling" | 允许 sampling 内包含 tools，用于 server-hosted ReAct |
| Human-in-the-loop | "User confirms" | client 在运行前向用户展示 sampling request |
| Loop bomb | "Runaway sampling" | server-side infinite sampling loop；client 必须 rate-limit |
| Covert sampling | "Hidden reasoning" | malicious server 在 sampling prompts 中隐藏意图 |
| Resource theft | "Using user's LLM budget" | server 强迫 client 把预算花在不想要的 sampling 上 |
| `stopReason` | "Why generation halted" | `endTurn`、`stopSequence` 或 `maxTokens` |

## 延伸阅读

- [MCP — Concepts: Sampling](https://modelcontextprotocol.io/docs/concepts/sampling) — sampling 的 high-level overview
- [MCP — Client sampling spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/client/sampling) — canonical `sampling/createMessage` shape
- [MCP — GitHub SEP-1577](https://github.com/modelcontextprotocol/modelcontextprotocol) — tools in sampling 的 Spec Evolution Proposal (experimental)
- [Unit 42 — MCP attack vectors](https://unit42.paloaltonetworks.com/model-context-protocol-attack-vectors/) — covert sampling 和 resource-theft patterns
- [Speakeasy — MCP sampling core concept](https://www.speakeasy.com/mcp/core-concepts/sampling) — 带 client-side code samples 的 walk-through
