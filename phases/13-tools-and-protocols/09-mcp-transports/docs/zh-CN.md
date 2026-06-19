# MCP 传输：stdio、Streamable HTTP 与 SSE 迁移

> stdio 只适合本地，离开本机就不适用。Streamable HTTP (2025-03-26) 是 remote standard。旧的 HTTP+SSE transport 已被弃用，并将在 2026 年中移除。选错 transport 会带来一次 migration；选对 transport 则能得到一个可 remote-hostable、具备 session continuity 和 DNS-rebinding protection 的 MCP server。

**类型:** 学习
**语言:** Python (stdlib, Streamable HTTP endpoint skeleton)
**先修:** Phase 13 · 07, 08 (MCP server and client)
**时间:** ~45 分钟

## 学习目标

- 根据 deployment shape (local vs remote, single-process vs fleet) 在 stdio 和 Streamable HTTP 之间选择。
- 实现 Streamable HTTP single-endpoint pattern：POST 处理 requests，GET 处理 session stream。
- 强制 `Origin` validation 和 session-id semantics，以抵御 DNS-rebinding。
- 在 2026 年中 removal deadlines 之前，把 legacy HTTP+SSE server 迁移到 Streamable HTTP。

## 要解决的问题

第一个 MCP remote transport (2024-11) 是 HTTP+SSE：两个 endpoints，一个用于 client 的 POSTs，另一个 Server-Sent-Events channel 用于 server-to-client stream。它能工作。但也很笨重：每个 session 两个 endpoints，在某些 CDNs 前方会遇到 broken caches，并且强依赖 long-lived SSE connections，而一些 WAFs 会激进地终止这些连接。

2025-03-26 spec 用 Streamable HTTP 替代了它：一个 endpoint，POST 用于 client requests，GET 用于建立 session stream，两者共享 `Mcp-Session-Id` header。从那之后新建或迁移的每个 server 都使用 Streamable HTTP。旧 SSE mode 正在被弃用 —— Atlassian Rovo 于 2026 年 6 月 30 日移除，Keboola 于 2026 年 4 月 1 日移除，大多数剩余 enterprise servers 会在 2026 年底前移除。

而 stdio 对 local servers 仍然重要。Claude Desktop、VS Code 和每个 IDE-shaped client 都通过 stdio 启动 servers。正确的 mental model：stdio 用于 "this machine"，Streamable HTTP 用于 "over the network"。不要混用。

## 核心概念

### stdio

- Child-process transport。client 启动 server，通过 stdin/stdout 通信。
- 一行一个 JSON object。Newline-delimited。
- 没有 session id；process identity 就是 session。
- 不需要 auth (child 继承 parent 的 trust boundary)。
- 永远不要用于 remote servers —— 你需要 SSH 或 socat 来 tunnel，既然如此就应该使用 Streamable HTTP。

### Streamable HTTP

单 endpoint `/mcp` (或任意 path)。支持三种 HTTP methods：

- **POST /mcp。** client 发送 JSON-RPC message。server 回复单个 JSON response，或一个包含 one-or-more responses 的 SSE stream (适用于 batched responses 和与该 request 相关的 notifications)。
- **GET /mcp。** client 打开 long-lived SSE channel。server 用它发送 server-to-client requests (sampling, notifications, elicitation)。
- **DELETE /mcp。** client 显式终止 session。

Sessions 由 server 在第一次 response 上设置、client 在后续每个 request 中 echo 的 `Mcp-Session-Id` header 标识。Session ids MUST 是 cryptographically random (128+ bits)；为了安全，拒绝 client-chosen ids。

### Single endpoint vs two

旧 spec 的 two-endpoint mode 在 2026 年仍然可调用 —— spec 声明它为 "legacy compatible"。但所有新 servers 都应使用 single-endpoint。官方 SDKs 发出 single-endpoint；只有在与尚未迁移的 remote 对话时才使用 legacy mode。

### `Origin` validation and DNS-rebinding

Browsers 目前不是 MCP clients，但攻击者可以制作一个网页，诱导 browser 向 `localhost:1234/mcp` POST —— 那里可能运行着用户的 local MCP server。如果 server 不检查 `Origin`，browser 的 same-origin policy 也救不了它，因为 `Origin: http://evil.com` 是有效的 cross-origin。

2025-11-25 spec 要求 servers 拒绝 `Origin` 不在 allowlist 中的 requests。allowlist 通常包含 MCP client host (`https://claude.ai`、`vscode-webview://*`) 以及 local UIs 使用的 localhost variants。

### Session id lifecycle

1. client 发送第一个 request，不带 `Mcp-Session-Id`。
2. server 分配 random id，并在 response header 上设置 `Mcp-Session-Id`。
3. client 在所有后续 requests 以及用于 stream 的 `GET /mcp` 上 echo 该 header。
4. session 可以被 server revoked；client 在后续 requests 上看到 404，必须重新 initialize。
5. client 可以显式 DELETE session，以便干净 shutdown。

### Keepalive and reconnect

SSE connections 会断开。client 用相同 `Mcp-Session-Id` 重新 GET 来重建连接。server MUST queue outage 期间错过的 events (在合理窗口内)，并通过 client echo 的 `last-event-id` header replay。

Phase 13 · 13 会介绍 Tasks，它能让 long-running work 即使经历 full-session reconnect 也能存活。

### Backwards compatibility probe

想同时支持旧新 servers 的 client：

1. POST 到 `/mcp`。
2. 如果 response 是带 JSON 或 SSE 的 `200 OK`，这是 Streamable HTTP。
3. 如果 response 是带 `Content-Type: text/event-stream` 且有指向 secondary endpoint 的 `Location` header 的 `200 OK`，这是 legacy HTTP+SSE；跟随 `Location`。

### Cloudflare, ngrok, and hosting

2026 年的 production remote MCP servers 运行在 Cloudflare Workers (配合其 MCP Agents SDK)、Vercel Functions 或 containerized Node/Python 上。关键点：你的 hosting 必须支持 SSE GET 所需的 long-lived HTTP connections。Vercel 的 free tier 限制为 10 秒，不适合。Cloudflare Workers 支持 indefinite streams。

### Gateway composition

当你用 gateway 置于多个 MCP servers 之前 (Phase 13 · 17) 时，gateway 是一个单独的 Streamable HTTP endpoint，会 rewrite session ids 并 multiplex upstream。Tools 在 gateway layer 合并；client 看到的是一个 logical server。

### Transport failure modes

- **stdio SIGPIPE。** Child process 在 mid-write 时死亡会触发 SIGPIPE；servers 应该干净退出。clients 应检测 EOF 并把 session 标记为 dead。
- **HTTP 502 / 504。** Cloudflare、nginx 和其他 proxies 在 upstream failure 时发出这些状态。Streamable HTTP clients 应在短暂 backoff 后 retry 一次。
- **SSE connection drop。** TCP RST、proxy timeout 或 client network change 会关闭 stream。client 带 `Mcp-Session-Id` 和可选 `last-event-id` reconnect 以 resume。
- **Session revocation。** server 使 session id 失效；client 在下一个 request 看到 404。client 必须重新 handshake。
- **Clock skew。** client 上的 Resource-TTL calculations 与 server 分歧。client 应把 server timestamps 视为 authoritative。

### When to bypass Streamable HTTP

一些 enterprises 会在自己的 networks 内把 MCP servers 部署在 gRPC 或 message-queue transports 后面。这不是标准方式 —— MCP spec 没有正式定义这些。Gateways 可以对 MCP clients 暴露 Streamable HTTP surface，同时内部使用 gRPC。保持 external surface spec-compliant；translation 由 gateway 负责。

## 实际使用

`code/main.py` 使用 `http.server` (stdlib) 实现一个 minimal Streamable HTTP endpoint。它处理 `/mcp` 上的 POST、GET 和 DELETE，在第一次 response 上设置 `Mcp-Session-Id`，验证 `Origin`，并拒绝来自非 allowlisted origins 的 requests。handler 复用了 Lesson 07 notes server 的 dispatch logic。

需要观察的点：

- POST handler 读取 JSON-RPC body、dispatch，并写出 JSON response (single-response variant；SSE variant 在结构上相似)。
- `Origin` check 会拒绝默认的 `http://evil.example` probe，但接受 `http://localhost`。
- Session ids 是 random 128-bit hex strings；server 在内存中保存 per-session state。

## 交付成果

本课产出 `outputs/skill-mcp-transport-migrator.md`。给定一个 HTTP+SSE (legacy) MCP server，该 skill 会产出迁移到 Streamable HTTP 的计划，包括 session-id continuity、Origin checks 和 backwards-compatible probe support。

## 练习

1. 运行 `code/main.py`。从 `curl` POST 一个 `initialize`，观察 `Mcp-Session-Id` response header。再 POST 第二个 request 并 echo 该 header，验证 session continuity。

2. 添加一个会打开 SSE stream 的 GET handler。每五秒发送一个 `notifications/progress` event。用相同 session id 重新 GET 来 reconnect，并确认 server 接受它。

3. 实现 `last-event-id` replay logic。在 reconnect 时 replay 该 id 之后生成的所有 events。

4. 扩展 `Origin` validation，以支持 wildcard pattern (`https://*.example.com`)，并确认它接受 `https://app.example.com` 但拒绝 `https://evil.example.com.attacker.net`。

5. 从 official registry 取一个 legacy HTTP+SSE server (有好几个)，并草拟迁移方案：endpoint handling、session id generation 和 header semantics 会发生哪些变化。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| stdio transport | "Local child process" | 基于 stdin/stdout 的 JSON-RPC，newline-delimited |
| Streamable HTTP | "The remote transport" | Single-endpoint POST + GET + optional SSE，2025-03-26 spec |
| HTTP+SSE | "Legacy" | 正在 2026 年中移除的 two-endpoint model |
| `Mcp-Session-Id` | "Session header" | server-assigned random id，会在每个后续 request 上 echo |
| `Origin` allowlist | "DNS-rebinding defense" | 拒绝 Origin 未被批准的 requests |
| Single endpoint | "One URL" | `/mcp` 处理所有 session operations 的 POST / GET / DELETE |
| `last-event-id` | "SSE replay" | 用于恢复 dropped stream 且不丢失 events 的 header |
| Backwards-compat probe | "Old vs new detection" | client response-shape check，自动选择 transport |
| Long-lived HTTP | "SSE streaming" | server 在一个 TCP connection 上推送数分钟或数小时的 events |
| Session revocation | "Force re-init" | server 使 session id 失效；client 必须重新 handshake |

## 延伸阅读

- [MCP — Basic transports spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports) — stdio 和 Streamable HTTP 的 canonical reference
- [MCP — Basic transports spec 2025-03-26](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports) — 引入 Streamable HTTP 的 revision
- [Cloudflare — MCP transport](https://developers.cloudflare.com/agents/model-context-protocol/transport/) — Workers-hosted Streamable HTTP patterns
- [AWS — MCP transport mechanisms](https://builder.aws.com/content/35A0IphCeLvYzly9Sw40G1dVNzc/mcp-transport-mechanisms-stdio-vs-streamable-http) — across deployment shapes 的比较
- [Atlassian — HTTP+SSE deprecation notice](https://community.atlassian.com/forums/Atlassian-Remote-MCP-Server/HTTP-SSE-Deprecation-Notice/ba-p/3205484) — 具体 migration deadline 示例
