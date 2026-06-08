# 构建 MCP Client — Discovery、Invocation、Session Management

> 大多数 MCP 内容都会交付 server tutorials，然后对 client 一笔带过。真正困难的编排在 client code 里：process spawning、capability negotiation、跨多个 servers 合并 tool lists、sampling callbacks、reconnection，以及 namespace collision resolution。本课构建一个 multi-server client，把三个不同 MCP servers 提升为一个给 model 使用的 flat tool namespace。

**类型:** 构建
**语言:** Python (stdlib, multi-server MCP client)
**先修:** Phase 13 · 07 (building an MCP server)
**时间:** ~75 分钟

## 学习目标

- 把 MCP server 作为 child process 启动，完成 `initialize`，并发送 `notifications/initialized`。
- 维护 per-server session state (capabilities, tool list, last-seen notification ids)。
- 把多个 servers 的 tool lists 合并成一个 namespace，并处理 collisions。
- 把 tool call 路由到拥有它的 server，并重新组装 response。

## 要解决的问题

真实 agent host (Claude Desktop, Cursor, Goose, Gemini CLI) 会一次加载多个 MCP servers。用户可能同时运行 filesystem server、Postgres server 和 GitHub server。client 的工作：

1. 启动每个 server。
2. 分别完成 handshake。
3. 对每个 server 调用 `tools/list`，并把结果 flatten。
4. 当 model 发出 `notes_search` 时，在 merged namespace 中查找，并路由到正确 server。
5. 处理来自任意 server 的 notifications (`tools/list_changed`)，同时不阻塞。
6. 在 transport failure 时 reconnect。

手写这些逻辑，是把 "toy" 和 "serviceable" 区分开的关键。官方 SDK 会封装它，但 mental model 必须属于你自己。

## 核心概念

### Child-process spawning

使用 `subprocess.Popen`，并设置 `stdin=PIPE, stdout=PIPE, stderr=PIPE`。设置 `bufsize=1`，用 text mode 做 line-by-line reads。每个 server 是一个 process；client 为每个 server 持有一个 `Popen` handle。

### Per-server session state

每个 server 对应一个 `Session` object，保存：

- `process` — Popen handle。
- `capabilities` — server 在 `initialize` 中声明的能力。
- `tools` — 上一次 `tools/list` result。
- `pending` — request id 到等待 response 的 promise/future 的 map。

Requests 天然是 async；发给 server A 的 `tools/call` 不应因为 server B 正在 mid-call 而阻塞。可以使用 threads with queues，也可以使用 asyncio。

### Merged namespace

当 client 看到聚合后的 tool list 时，names 可能冲突。两个 servers 可能都公开 `search`。client 有三种选择：

1. **按 server name 加前缀。** `notes/search`、`files/search`。清楚但不美观。
2. **静默 first-come。** 后来 server 的 `search` 覆盖先前的。风险高；会隐藏 collisions。
3. **拒绝 collision。** 拒绝加载第二个 server；通知用户。对 security-sensitive hosts 最安全。

Claude Desktop 使用 prefix-by-server。Cursor 使用 collision rejection 并给出清楚错误。VS Code MCP 也采用 prefix-by-server。

### Routing

合并之后，一个 dispatch table 把 `tool_name -> session` 映射起来。model 按 name 发出 call；client 找到 session，把 `tools/call` message 写入该 server 的 stdin，然后等待 response。

### Sampling callback

如果 server 在 `initialize` 中声明了 `sampling` capability，它可以发送 `sampling/createMessage`，请求 client 运行自己的 LLM。client 必须：

1. 阻塞对该 server 的后续 requests，直到 sample resolves；如果 implementation 支持 concurrency，也可以 pipeline。
2. 调用自己的 LLM provider。
3. 把 response 发回 server。

Lesson 11 会端到端讲 sampling。本课为了完整性只 stub 它。

### Notification handling

`notifications/tools/list_changed` 意味着重新调用 `tools/list`。`notifications/resources/updated` 意味着如果该 resource 正在使用，就重新读取。Notifications 不能产生 responses —— 不要尝试 ack 它们。

一个常见 client bug：在 `tools/call` 上阻塞 read loop，而 notification 滞留在 stream 里。使用一个 background reader thread，把每条 message 推入 queue；main thread 再 dequeue 并 dispatch。

### Reconnection

Transport 可能失败：server crashed、OS killed the process、stdio pipe broke。client 检测 stdout 上的 EOF，并把 session 视为 dead。选项：

- 静默重启 server 并重新 handshake。对纯 read-only servers 可以接受。
- 把失败暴露给用户。对有 user-visible sessions 的 stateful servers 可以接受。

Phase 13 · 09 会介绍 Streamable HTTP reconnection semantics；stdio 更简单。

### Keepalive and session id

Streamable HTTP 使用 `Mcp-Session-Id` header。Stdio 没有 session id —— process identity 就是 session。Keepalive pings 是可选的；stdio pipes 不会因为 inactivity 而断开。

## 实际使用

`code/main.py` 会把三个模拟 MCP servers 作为 subprocesses 启动，分别 handshake，合并它们的 tool lists，并把 tool calls 路由到正确的 server。这里的 "servers" 实际上是运行 toy responders 的其他 Python processes (没有真实 LLM)。运行它可以看到：

- 三次 initialization，每个都有自己的 capability set。
- 三个 `tools/list` results 合并成一个 7-tool namespace。
- 基于 tool name 的 routing decision。
- 通过 namespace prefixing 阻止 collision。

需要观察的点：

- `Session` dataclass 干净地保存 per-server state。
- background reader thread 会 drain stdout 上的每一行，而不阻塞 main thread。
- dispatch table 是简单的 `dict[str, Session]`。
- collision handling 是显式的：当两个 servers 声明相同 name 时，后来的那个会带 prefix 重新命名。

## 交付成果

本课产出 `outputs/skill-mcp-client-harness.md`。给定一组声明式 MCP servers (name, command, args)，该 skill 会生成一个 harness，启动它们、合并 tool lists，并交付一个带 collision resolution 的 routing function。

## 练习

1. 运行 `code/main.py`，观察 server spawn log。用 SIGTERM 杀掉其中一个 simulated server process，观察 client 如何检测 EOF 并把该 session 标记为 dead。

2. 实现 namespace prefixing。当两个 servers 都公开 `search` 时，把第二个重命名为 `<server>/search`。更新 dispatch table 并验证 tool calls 能正确路由。

3. 为 server restart 添加 connection-pool-style backoff：连续失败时 exponential backoff，上限 30 秒，三次失败后向用户发出 notification。

4. 草拟一个支持 100 个 concurrent MCP servers 的 client。什么 data structure 会替代简单 dispatch dict？(提示：用于 prefix namespacing 的 trie，再加一个 tool-count-per-server metric。)

5. 把 client 移植到官方 MCP Python SDK。SDK 封装了 `stdio_client` 和 `ClientSession`。代码应从约 200 行缩短到约 40 行，同时保留 multi-server routing。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| MCP client | "The agent host" | 启动 servers 并编排 tool calls 的 process |
| Session | "Per-server state" | Capabilities、tool list 和 pending-request bookkeeping |
| Merged namespace | "One tool list" | 所有 active servers 上的 flat tool names 集合 |
| Namespace collision | "Two servers same tool" | client 必须 prefix、reject 或 first-come duplicate |
| Routing | "Who gets this call?" | 从 tool name dispatch 到 owning server |
| Background reader | "Non-blocking stdout" | 把 server stdout drain 到 queue 的 thread 或 task |
| Sampling callback | "LLM-as-a-service" | client 对 server 发出的 `sampling/createMessage` 的 handler |
| `notifications/*_changed` | "Primitive mutated" | 表示 client 必须 re-discover 或 re-read |
| Reconnection policy | "When server dies" | transport 失败时的 restart semantics |
| Stdio session | "Process = session" | 没有 session id；child process lifetime 就是 session |

## 延伸阅读

- [Model Context Protocol — Client spec](https://modelcontextprotocol.io/specification/2025-11-25/client) — canonical client behavior
- [MCP — Quickstart client guide](https://modelcontextprotocol.io/quickstart/client) — 使用 Python SDK 的 hello-world client tutorial
- [MCP Python SDK — client module](https://github.com/modelcontextprotocol/python-sdk) — reference `ClientSession` 和 `stdio_client`
- [MCP TypeScript SDK — Client](https://github.com/modelcontextprotocol/typescript-sdk) — TS parallel
- [VS Code — MCP in extensions](https://code.visualstudio.com/api/extension-guides/ai/mcp) — VS Code 如何在单个 editor host 中 multiplex 多个 MCP servers
