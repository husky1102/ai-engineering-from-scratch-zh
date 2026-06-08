# MCP Fundamentals：Primitives、Lifecycle、JSON-RPC Base

> MCP 之前的每个 integration 都是一次性的。Model Context Protocol 最初由 Anthropic 于 2024 年 11 月发布，现在由 Linux Foundation 的 Agentic AI Foundation 维护，它标准化 discovery 和 invocation，使任意 client 都能与任意 server 通信。2025-11-25 spec 命名了六个 primitives（三个 server，三个 client）、一个三阶段 lifecycle，以及 JSON-RPC 2.0 wire format。学会这些，本 phase 的 MCP 章节其余部分就只是阅读。

**类型:** Learn
**语言:** Python（stdlib，JSON-RPC parser）
**先修:** Phase 13 · 01 through 05（the tool interface and function calling）
**时间:** ~45 分钟

## 学习目标

- 说出全部六个 MCP primitives（server 上的 tools、resources、prompts；client 上的 roots、sampling、elicitation），并分别给出一个 use case。
- 走读三阶段 lifecycle（initialize、operation、shutdown），并说明每个阶段谁发送哪条 message。
- Parse 并 emit JSON-RPC 2.0 request、response 和 notification envelopes。
- 解释 `initialize` 时的 capability negotiation 是什么，以及没有它会坏掉什么。

## 要解决的问题

MCP 之前，每个 tool-using agent 都有自己的 protocol。Cursor 有一个 MCP-shaped 但不兼容的 tool system。Claude Desktop 发布了另一个。VS Code 的 Copilot extension 有第三个。一个团队构建“Postgres query”tool，需要把同一个 tool 写三遍，分别适配不同 host API。复用它需要复制代码。

结果是一次性 integrations 的寒武纪爆发，以及 ecosystem velocity 的天花板。

MCP 通过标准化 wire format 修复这一点。单个 MCP server 可以在每个 MCP client 中工作：Claude Desktop、ChatGPT、Cursor、VS Code、Gemini、Goose、Zed、Windsurf，截至 2026 年 4 月已有 300+ clients。每月 SDK downloads 1.1 亿。公开 servers 10,000+。Linux Foundation 于 2025 年 12 月在新的 Agentic AI Foundation 下接手治理。

本 phase 使用的 spec revision 是 **2025-11-25**。它加入 async Tasks（SEP-1686）、URL-mode elicitation（SEP-1036）、sampling with tools（SEP-1577）、incremental scope consent（SEP-835）以及 OAuth 2.1 resource-indicator semantics。Phase 13 · 09 through 16 覆盖这些 extensions。本课停在 base。

## 核心概念

### 三个 server primitives

1. **Tools.** Callable actions。与 Phase 13 · 01 相同的四步 loop。
2. **Resources.** 暴露的数据。按 URI addressable 的 read-only content：`file:///path`、`db://query/...`、custom schemes。
3. **Prompts.** 可复用 templates。Host UI 中的 slash-commands；server 提供 template，client 填 arguments。

### 三个 client primitives

4. **Roots.** Server 被允许触碰的 URI 集合。Client 声明它们；server 遵守。
5. **Sampling.** Server 请求 client 的模型执行 completion。使 server-hosted agent loops 无需 server-side API keys。
6. **Elicitation.** Server 在 mid-flight 向 client 的用户请求 structured input。Forms 或 URLs（SEP-1036）。

MCP 中每个 capability 都恰好属于这六者之一。Phase 13 · 10 through 14 会深入每个 primitive。

### Wire format：JSON-RPC 2.0

每条 message 都是带这些 fields 的 JSON object：

- Requests：`{jsonrpc: "2.0", id, method, params}`。
- Responses：`{jsonrpc: "2.0", id, result | error}`。
- Notifications：`{jsonrpc: "2.0", method, params}`，没有 `id`，也不期待 response。

Base spec 有约 15 个 methods，按 primitive 分组。重要 ones：

- `initialize` / `initialized`（handshake）
- `tools/list`、`tools/call`
- `resources/list`、`resources/read`、`resources/subscribe`
- `prompts/list`、`prompts/get`
- `sampling/createMessage`（server-to-client）
- `notifications/tools/list_changed`、`notifications/resources/updated`、`notifications/progress`

### 三阶段 lifecycle

**Phase 1：initialize。**

Client 发送带 `capabilities` 和 `clientInfo` 的 `initialize`。Server 返回自己的 `capabilities`、`serverInfo` 和它使用的 spec version。Client 在消化 response 后发送 `notifications/initialized`。从这之后，任一方都可以按协商出的 capabilities 发送 requests。

**Phase 2：operation。**

双向。Client 调用 `tools/list` 进行 discovery，然后用 `tools/call` invocation。Server 如果声明了 capability，可以发送 `sampling/createMessage`。Server 在 tool set 变化时可以发送 `notifications/tools/list_changed`。当用户改变 root scope 时，client 可以发送 `notifications/roots/list_changed`。

**Phase 3：shutdown。**

任一方关闭 transport。MCP 没有结构化 shutdown method；transport（stdio 或 Streamable HTTP，Phase 13 · 09）承载 end-of-connection signal。

### Capability negotiation

`initialize` handshake 中的 `capabilities` 是 contract。Server 示例：

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

Server 声明它可以发出 `tools/list_changed` notifications，并支持 `resources/subscribe`。Client 通过声明自己的 capabilities 同意：

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

如果 client 没有声明 `sampling`，server 就不得调用 `sampling/createMessage`。对称地，如果 server 没有声明 `resources.subscribe`，client 就不得尝试 subscribe。

这就是防止 ecosystem drift 的机制。不支持 sampling 的 client 仍是 valid MCP client；不调用 `sampling` 的 server 仍是 valid MCP server。它们只是不会一起使用该 feature。

### Structured content 与 error shapes

`tools/call` 返回一个 `content` array，其中包含 typed blocks：`text`、`image`、`resource`。Phase 13 · 14 会把 MCP Apps（`ui://` interactive UI）加入这个列表。

Errors 使用 JSON-RPC error codes。Spec-defined additions：`-32002` “Resource not found”、`-32603` “Internal error”，以及作为 `error.data` 的 MCP-specific error data。

### Client capabilities vs tool call details

常见困惑：`capabilities.tools` 表示 client 是否支持 tool-list-changed notifications。Client 是否会调用某个具体 tool 是由其模型驱动的 runtime choice，不是 capability flag。Capability flag 是 spec-level contract。模型选择是正交的。

### 为什么是 JSON-RPC 而不是 REST？

JSON-RPC 2.0（2010）是轻量级双向协议。REST 是 client-initiated。MCP 需要 server-initiated messages（sampling、notifications），所以带有对称 request/response shape 的 JSON-RPC 很自然。JSON-RPC 也能干净地组合在 stdio 和 WebSocket/Streamable HTTP 上，无需重新发明 HTTP request shape。

## 实际使用

`code/main.py` 交付一个最小 JSON-RPC 2.0 parser 和 emitter，然后手动走过 `initialize` → `tools/list` → `tools/call` → `shutdown` sequence，打印每条 message。没有真实 transport，只有 message shapes。与延伸阅读中的 spec 对比，验证每个 envelope。

重点看：

- `initialize` 双向声明 capabilities；response 有 `serverInfo` 和 `protocolVersion: "2025-11-25"`。
- `tools/list` 返回 `tools` array；每个 entry 有 `name`、`description`、`inputSchema`。
- `tools/call` 使用 `params.name` 和 `params.arguments`。
- Response `content` 是 `{type, text}` blocks 的 array。

## 交付成果

本课产出 `outputs/skill-mcp-handshake-tracer.md`。给定一个 MCP client-server interaction 的 pcap-style transcript，该 skill 会标注每条 message 属于哪个 primitive、哪个 lifecycle phase，以及依赖哪个 capability。

## 练习

1. 运行 `code/main.py`。找出 capability negotiation 发生的那一行，并描述如果 server 没有声明 `tools.listChanged` 会改变什么。

2. 扩展 parser 以处理 `notifications/progress`。Message shape：`{method: "notifications/progress", params: {progressToken, progress, total}}`。在 long-running `tools/call` 进行中 emit 它，并确认 client handler 会显示 progress bar。

3. 从头到尾阅读 MCP 2025-11-25 spec，整个文档大约 80 页。找出大多数 server 不需要的一个 capability flag。提示：它与 resource subscription 有关。

4. 在纸上草拟一个假想“cron job”feature 应属于哪个 primitive。（提示：server 希望 client 在计划时间 invoke 它。今天六个 primitives 中没有一个合适。）MCP 2026 roadmap 有该功能的 draft SEP。

5. 解析 GitHub 上一个开放 MCP server 的 session log。统计 request vs response vs notification messages。计算 lifecycle vs operation traffic 的比例。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| MCP | “Model Context Protocol” | 用于 model-to-tool discovery 和 invocation 的开放协议 |
| Server primitive | “What a server exposes” | tools（actions）、resources（data）、prompts（templates） |
| Client primitive | “What a client lets servers use” | roots（scope）、sampling（LLM callbacks）、elicitation（user input） |
| JSON-RPC 2.0 | “The wire format” | 对称 request/response/notification envelopes |
| `initialize` handshake | “Capability negotiation” | 第一组 message pair；servers 和 clients 声明支持的 features |
| `tools/list` | “Discovery” | Client 向 server 请求当前 tool set |
| `tools/call` | “Invocation” | Client 请求 server 带 arguments 执行 tool |
| `notifications/*_changed` | “Mutation events” | Server 告诉 client 其 primitive list 已变化 |
| Content block | “Typed result” | Tool result 中的 `{type: "text" \| "image" \| "resource" \| "ui_resource"}` |
| SEP | “Spec Evolution Proposal” | 命名的 draft proposal（例如 async Tasks 的 SEP-1686） |

## 延伸阅读

- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — canonical spec document
- [Model Context Protocol — Architecture concepts](https://modelcontextprotocol.io/docs/concepts/architecture) — six-primitive mental model
- [Anthropic — Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) — 2024 年 11 月 launch post
- [MCP blog — First MCP anniversary](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 一周年 retrospective 和 2025-11-25 spec changes
- [WorkOS — MCP 2025-11-25 spec update](https://workos.com/blog/mcp-2025-11-25-spec-update) — SEP-1686、1036、1577、835、1724 概览
