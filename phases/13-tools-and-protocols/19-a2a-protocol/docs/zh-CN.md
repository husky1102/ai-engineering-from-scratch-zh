# A2A：Agent-to-Agent Protocol

> MCP 是 agent-to-tool。A2A（Agent2Agent）是 agent-to-agent：一个开放协议，让基于不同 frameworks 构建、内部不透明的 agents 能够协作。它由 Google 于 2025 年 4 月发布，2025 年 6 月捐赠给 Linux Foundation，2026 年 4 月达到 v1.0，并获得包括 AWS、Cisco、Microsoft、Salesforce、SAP 和 ServiceNow 在内的 150+ 支持者。它吸收了 IBM 的 ACP，并加入 AP2 payments extension。本课讲解 Agent Card、Task lifecycle 和两种 transport bindings。

**类型：** 构建
**语言：** Python（stdlib，Agent Card + Task harness）
**先修：** Phase 13 · 06（MCP fundamentals）、Phase 13 · 08（MCP client）
**时间：** 约 75 分钟

## 学习目标

- 区分 agent-to-tool（MCP）与 agent-to-agent（A2A）use cases。
- 在 `/.well-known/agent.json` 发布包含 skills 和 endpoint metadata 的 Agent Card。
- 走通 Task lifecycle（submitted → working → input-required → completed / failed / canceled / rejected）。
- 使用带 Parts（text、file、data）的 Messages，并用 Artifacts 作为 outputs。

## 要解决的问题

一个 customer-service agent 需要把 report-writing 委托给专门的 writer agent。A2A 之前的选项：

- Custom REST API。能用，但每对集成都是 one-off。
- Shared codebase。要求两个 agents 运行在同一 framework。
- MCP。不合适：MCP 是调用工具，不是让两个 agents 在保留各自不透明内部推理的同时协作。

A2A 填补这个空白。它把交互建模为一个 agent 向另一个 agent 发送 Task，带 lifecycle、messages 和 artifacts。被调用 agent 的内部状态保持不透明，调用方只看到 task state transitions 和最终 outputs。

A2A 是“让跨 frameworks 的 agents 彼此对话”的协议。它不取代 MCP；两者互补。

## 核心概念

### Agent Card

每个 A2A-compliant agent 都在 `/.well-known/agent.json` 发布一张 card：

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

Discovery 基于 URL：fetch card，了解 A2A endpoint 的 URL，并枚举 skills。

### Signed Agent Cards (AP2)

AP2 extension（2025 年 9 月）为 Agent Cards 增加 cryptographic signatures。publisher 用 JWT 签署自己的 card；consumers 验证。它能防止 impersonation。

### Task lifecycle

```text
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (loop via message)
```

Clients 以 `tasks/send` 发起。被调用 agent 经过多个 states；clients 通过 SSE 订阅 state updates，或进行 poll。

### Messages and Parts

一条 message 携带一个或多个 Parts：

- `text`：纯内容。
- `file`：带 mimeType 的 base64 blob。
- `data`：typed JSON payload（给被调用 agent 的 structured input）。

示例：

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### Artifacts

Outputs 是 Artifacts，而不是 raw strings。Artifact 是一个有名称、有类型的 output：

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

Artifacts 可以以 chunks 形式 stream。caller 负责累积。

### 两种 transport bindings

1. **JSON-RPC over HTTP。** `/a2a` endpoint，POST requests，可选 SSE streaming。默认 binding。
2. **gRPC。** 面向原生使用 gRPC 的 enterprise environments。

两种 bindings 承载相同的逻辑 message shape。

### Opacity preservation

一个关键设计原则：被调用 agent 的内部状态是不透明的。caller 看到 task state 和 artifacts。被调用 agent 的 chain-of-thought、工具调用、sub-agent delegation 全部不可见。这不同于 MCP，MCP 中 tool calls 是透明的。

理由：A2A 让竞争对手能够协作而不暴露内部实现。A2A 可以是“调用这个 customer-service agent”，而 caller 不需要知道该 agent 如何实现服务。

### Timeline

- **2025-04-09。** Google 宣布 A2A。
- **2025-06-23。** 捐赠给 Linux Foundation。
- **2025-08。** 吸收 IBM 的 ACP。
- **2025-09。** AP2 extension（Agent Payments）发布。
- **2026-04。** v1.0 发布，150+ organizations 支持。

### Relationship to MCP

| Dimension | MCP | A2A |
|-----------|-----|-----|
| Use case | Agent-to-tool | Agent-to-agent |
| Opacity | Transparent tool calls | Opaque inner reasoning |
| Typical caller | Agent runtime | Another agent |
| State | Tool-call result | Task with lifecycle |
| Authorization | OAuth 2.1 (Phase 13 · 16) | JWT-signed Agent Cards (AP2) |
| Transport | Stdio / Streamable HTTP | JSON-RPC over HTTP / gRPC |

当你想调用一个具体工具时使用 MCP。当你想把整个任务委托给另一个 agent 时使用 A2A。许多生产系统会同时使用两者：agent 用 MCP 构建工具层，用 A2A 构建协作层。

## 实际使用

`code/main.py` 实现一个 minimal A2A harness：research agent 发布自己的 card，writer agent 接收包含 PDF 和文本指令 parts 的 `tasks/send`，经历 working → input_required → working → completed，并返回 text artifact。全部使用 stdlib；用 in-memory transport 聚焦 message shapes。

重点查看：

- Agent Card JSON shape。
- Task id assignment 和 state transitions。
- Mixed-type parts 的 Messages。
- 任务中途的 input-required branch。
- completion 时返回 Artifact。

## 交付成果

本课产出 `outputs/skill-a2a-agent-spec.md`。给定一个应当可被其他 agents 调用的新 agent，该 skill 会生成 Agent Card JSON、skills schema 和 endpoint blueprint。

## 练习

1. 运行 `code/main.py`。追踪完整 Task lifecycle，包括被调用 agent 请求澄清时的 input-required pause。

2. 添加 signed Agent Card。对 card 的 canonical JSON 用 HMAC 签名。编写 verifier，并确认 mutated card 验证失败。

3. 实现 task streaming：writer agent 通过 SSE 发射三个 incremental artifact chunks，caller 累积它们。

4. 设计一个包装 MCP server 的 A2A agent。把每个 MCP tool 映射为 A2A skill。记下 trade-offs：会失去哪些 opacity？

5. 阅读 A2A v1.0 announcement，找出截至 2026 年 4 月还没有任何 framework 实现的一项 feature。（提示：它与 multi-hop task delegation 有关。）

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| A2A | “Agent-to-Agent protocol” | 用于不透明 agent 协作的开放协议 |
| Agent Card | “`.well-known/agent.json`” | 描述 agent skills 和 endpoint 的已发布 metadata |
| Skill | “A callable unit” | agent 支持的命名 operation（类似 MCP tool） |
| Task | “Unit of delegation” | 带 lifecycle 和最终 artifact 的工作项 |
| Message | “Task input” | 携带 Parts（text、file、data） |
| Part | “Typed chunk” | message 中的 `text` / `file` / `data` 元素 |
| Artifact | “Task output” | completion 时返回的命名、typed output |
| AP2 | “Agent Payments Protocol” | 用于信任和支付的 Signed Agent Cards extension |
| Opacity | “Black-box collaboration” | 被调用 agent 的内部对 caller 隐藏 |
| Input-required | “Task pause” | agent 需要更多信息时的 lifecycle state |

## 延伸阅读

- [a2a-protocol.org](https://a2a-protocol.org/latest/) — canonical A2A specification
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) — reference implementations 和 SDKs
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) — 2025 年 6 月 governance transfer
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) — roadmap 和 partner momentum
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) — v1.0 release notes 和 backward-compat guidance
