# 构建 MCP Server — Python + TypeScript SDK

> 大多数 MCP 教程只展示 stdio hello-world。真实服务器会公开 tools、resources 和 prompts，处理 capability negotiation，发出结构化错误，并且在不同 SDK 中保持一致。本课端到端构建一个 notes server：stdlib stdio transport、JSON-RPC dispatch、三种 server primitives，以及一种纯函数风格，等你进阶时可以直接迁移到 Python SDK 的 FastMCP 或 TypeScript SDK。

**类型:** 构建
**语言:** Python (stdlib, stdio MCP server)
**先修:** Phase 13 · 06 (MCP fundamentals)
**时间:** ~75 分钟

## 学习目标

- 实现 `initialize`、`tools/list`、`tools/call`、`resources/list`、`resources/read`、`prompts/list` 和 `prompts/get` 方法。
- 编写一个 dispatch loop，从 stdin 读取 JSON-RPC messages，并把 responses 写到 stdout。
- 按 JSON-RPC 2.0 spec 和 MCP 的附加 codes 发出结构化 error responses。
- 在不重写 tool logic 的前提下，把 stdlib implementation 迁移到 FastMCP (Python SDK) 或 TypeScript SDK。

## 要解决的问题

在你使用 remote transport (Phase 13 · 09) 或 auth layer (Phase 13 · 16) 之前，需要先有一个干净的 local server。Local 指的是 stdio：server 由 client 作为 child process 启动，messages 通过 stdin/stdout 按 newline-delimited 方式流动。

2025-11-25 spec 规定 stdio messages 编码为 JSON objects，并显式使用 `\n` 分隔。这里没有 SSE；SSE 是旧的 remote mode，正在 2026 年中被移除 (Atlassian 的 Rovo MCP server 于 2026 年 6 月 30 日弃用，Keboola 于 2026 年 4 月 1 日弃用)。对于 stdio，一行一个 JSON object 就是完整 wire format。

notes server 是一个很好的形状，因为它会练习全部三种 server primitives。Tools 执行 mutation (`notes_create`)。Resources 暴露数据 (`notes://{id}`)。Prompts 交付 templates (`review_note`)。本课的结构可以推广到任何 domain。

## 核心概念

### Dispatch loop

```text
loop:
  line = stdin.readline()
  msg = json.loads(line)
  if has id:
    handle request -> write response
  else:
    handle notification -> no response
```

三条规则：

- 不要向 stdout 打印任何不是 JSON-RPC envelope 的内容。Debug logs 写到 stderr。
- 每个 request MUST 匹配一个带相同 `id` 的 response。
- Notifications MUST NOT 被响应。

### 实现 `initialize`

```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }
```

只声明你真正支持的内容。client 依靠 capability set 来控制功能是否可用。

### 实现 `tools/list` 和 `tools/call`

`tools/list` 返回 `{tools: [...]}`，其中每个 entry 都有 `name`、`description`、`inputSchema`。`tools/call` 接收 `{name, arguments}`，并返回 `{content: [blocks], isError: bool}`。

Content blocks 有类型。最常见的是：

```json
{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

Tool errors 有两种形状。Protocol-level errors (unknown method, bad params) 是 JSON-RPC errors。Tool-level errors (valid call but the tool failed) 作为 `{content: [...], isError: true}` 返回。这样 model 能在自己的 context 里看到失败信息。

### 实现 resources

Resources 按设计是 read-only。`resources/list` 返回 manifest；`resources/read` 返回 content。URIs 可以是 `file://...`、`http://...`，也可以是像 `notes://` 这样的 custom scheme。

当你把数据暴露为 resource 而不是 tool 时：

- model 不会 "call" 它；client 可以在用户请求时把它注入 context。
- Subscriptions 让 server 能在 resource 变化时推送 updates (Phase 13 · 10)。
- Phase 13 · 14 会用 `ui://` 把它扩展为 interactive resources。

### 实现 prompts

Prompts 是带 named arguments 的 templates。host 会把它们显示成 slash-commands。一个 `review_note` prompt 可以接收 `note_id` argument，并生成一个 multi-message prompt template，client 再把它喂给自己的 model。

### Stdio transport 的细节

- Newline-delimited JSON。没有 length-prefixed framing。
- 不要 buffer。每次 write 后调用 `sys.stdout.flush()`。
- client 控制 lifetime。当 stdin 关闭 (EOF) 时，干净退出。
- 不要静默处理 SIGPIPE；记录日志并退出。

### Annotations

每个 tool 都可以携带 `annotations` 来描述安全属性：

- `readOnlyHint: true` — 纯读取，可安全重试。
- `destructiveHint: true` — 不可逆 side effects；client 应要求确认。
- `idempotentHint: true` — 相同 inputs 产生相同 outputs。
- `openWorldHint: true` — 与 external systems 交互。

client 用这些信息决定 UX (confirmation dialogs, status indicators) 和 routing (Phase 13 · 17)。

### 进阶迁移路径

`code/main.py` 中的 stdlib server 大约 180 行。FastMCP (Python) 会把同样的逻辑压缩成 decorator-style：

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

TypeScript SDK 也有等价形状。当你准备好时，这条进阶路径可以直接替换；概念 (capabilities, dispatch, content blocks) 是相同的。

## 实际使用

`code/main.py` 是一个完整的 notes MCP server，通过 stdio 运行，只使用 stdlib。它处理 `initialize`，为三个 tools (`notes_list`、`notes_search`、`notes_create`) 处理 `tools/list` 和 `tools/call`，为每条 note 处理 `resources/list` 和 `resources/read`，并提供一个 `review_note` prompt。你可以通过管道发送 JSON-RPC messages 来驱动它：

```text
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

需要观察的点：

- dispatcher 是一个以 method name 为 key 的 `dict[str, Callable]`。
- 每个 tool executor 返回 content blocks 列表，而不是裸字符串。
- 当 executor 抛错时会设置 `isError: true`。

## 交付成果

本课产出 `outputs/skill-mcp-server-scaffolder.md`。给定一个 domain (notes, tickets, files, database)，该 skill 会用正确的 tools / resources / prompts 划分和 SDK 进阶迁移路径 scaffold 一个 MCP server。

## 练习

1. 运行 `code/main.py`，并用手写 JSON-RPC messages 驱动它。练习 `notes_create`，然后用 `resources/read` 取回新 note。

2. 添加一个带 `annotations: {destructiveHint: true}` 的 `notes_delete` tool。确认 client 会显示 confirmation dialog (这需要真实 host；Claude Desktop 可以)。

3. 实现 `resources/subscribe`，让 server 在 note 被修改时推送 `notifications/resources/updated`。添加一个 keepalive task。

4. 把 server 移植到 FastMCP。Python 文件应缩短到 80 行以内。wire behavior 必须一致；用同一个 JSON-RPC test harness 验证。

5. 阅读 spec 的 `server/tools` section，找出一个本课 server 尚未实现的 tool definition 字段。(提示：有好几个；任选一个并添加它。)

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| MCP server | "The thing that exposes tools" | 通过 stdio 或 HTTP 说 MCP JSON-RPC 的 process |
| stdio transport | "Child process model" | Server 由 client 启动；通过 stdin/stdout 通信 |
| Dispatcher | "Method router" | JSON-RPC method name 到 handler function 的 map |
| Content block | "Tool result chunk" | tool response 的 `content` array 中的 typed element |
| `isError` | "Tool-level failure" | 表示 tool 失败；与 JSON-RPC error 区分 |
| Annotations | "Safety hints" | readOnly / destructive / idempotent / openWorld flags |
| FastMCP | "Python SDK" | MCP protocol 之上的 decorator-based higher-level framework |
| Resource URI | "Addressable data" | 标识 resource 的 `file://`、`db://` 或 custom scheme |
| Prompt template | "Slash-command brief" | server-supplied template，带 host UIs 使用的 argument slots |
| Capability declaration | "Feature toggle" | 在 `initialize` 中声明的 per-primitive flags |

## 延伸阅读

- [Model Context Protocol — Python SDK](https://github.com/modelcontextprotocol/python-sdk) — reference Python implementation
- [Model Context Protocol — TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) — parallel TS implementation
- [FastMCP — server framework](https://gofastmcp.com/) — MCP servers 的 decorator-style Python API
- [MCP — Quickstart server guide](https://modelcontextprotocol.io/quickstart/server) — 使用任一 SDK 的端到端 tutorial
- [MCP — Server tools spec](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) — tools/* messages 的完整 reference
