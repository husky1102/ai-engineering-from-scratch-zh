# MCP Resources and Prompts — Tools 之外的 Context Exposure

> Tools 吸引了 MCP 90% 的注意力。另两个 server primitives 解决的是不同问题。Resources 暴露可读取的数据；prompts 暴露可复用 templates，作为 slash-commands。许多 servers 应该用 resources，而不是把 reads 包装成 tools；也应该用 prompts，而不是在 client prompts 中 hard-code workflows。本课给出 decision rule，并走读 `resources/*` 和 `prompts/*` messages。

**类型:** 构建
**语言:** Python (stdlib, resource + prompt handler)
**先修:** Phase 13 · 07 (MCP server)
**时间:** ~45 分钟

## 学习目标

- 针对给定 domain，决定把 capability 暴露为 tool、resource 还是 prompt。
- 实现 `resources/list`、`resources/read`、`resources/subscribe`，并处理 `notifications/resources/updated`。
- 使用 argument templates 实现 `prompts/list` 和 `prompts/get`。
- 识别 host 何时把 prompts 显示为 slash-commands，何时作为 auto-injected context。

## 要解决的问题

notes app 的 naive MCP server 会把一切都公开成 tools：`notes_read`、`notes_list`、`notes_search`。这把每次 data access 都包装成 model-driven tool call。后果：

- 对每个可能受益于 context 的 query，model 都必须决定是否调用 `notes_read`。
- Read-only content 无法被 subscribed，也无法 stream 到 host 的 side panel。
- Client UIs (Claude Desktop 的 resource attachment panel、Cursor 的 "Include file" picker) 无法展示这些数据。

正确划分是：把数据暴露为 resource，把 mutating 或 computed actions 暴露为 tools，把 reusable multi-step workflows 暴露为 prompts。每种 primitive 都有自己的 UX affordance 和 access pattern。

## 核心概念

### Tools vs resources vs prompts — decision rule

| Capability | Primitive |
|------------|-----------|
| 用户想 search、filter 或 transform data | tool |
| 用户想让 host 把这份 data 作为 context include | resource |
| 用户想复用一个 templated workflow | prompt |

Guideline：如果 model 会从每个相关 query 都调用它中受益，它就是 tool。如果用户会从把它 attach 到 conversation 中受益，它就是 resource。如果用户想复用的单位是完整 multi-step workflow，它就是 prompt。

### Resources

`resources/list` 返回 `{resources: [{uri, name, mimeType, description?}]}`。`resources/read` 接收 `{uri}`，并返回 `{contents: [{uri, mimeType, text | blob}]}`。

URIs 可以是任何可寻址内容：

- `file:///Users/alice/notes/mcp.md`
- `postgres://my-db/query/SELECT ...`
- `notes://note-14` (custom scheme)
- `memory://session-2026-04-22/recent` (server-specific)

`contents[]` 同时支持 text 和 binary。Binary 使用 `blob` 作为 base64-encoded string，并带上 `mimeType`。

### Resource subscriptions

在 capabilities 中声明 `{resources: {subscribe: true}}`。client 调用 `resources/subscribe {uri}`。resource 变化时，server 发送 `notifications/resources/updated {uri}`。client 重新读取。

Use case：一个 resources 是磁盘文件的 notes server；file watcher 触发 update notifications；Claude Desktop 在 host 外部编辑该文件时重新把它拉入 context。

### Resource templates (2025-11-25 addition)

`resourceTemplates` 让你公开一个 parameterized URI pattern：`notes://{id}`，其中 `id` 是 completion target。client 可以在 resource picker 中 autocomplete ids。

### Prompts

`prompts/list` 返回 `{prompts: [{name, description, arguments?}]}`。`prompts/get` 接收 `{name, arguments}`，并返回 `{description, messages: [{role, content}]}`。

prompt 是一个 template，会填充成 host 喂给其 model 的 messages 列表。例如，`code_review` prompt 接收 `file_path` argument，并返回一个 three-message sequence：system message、带 file body 的 user message，以及带 reasoning template 的 assistant kickoff。

### Hosts and prompts

Claude Desktop、VS Code 和 Cursor 会在 chat UI 中把 prompts 暴露为 slash-commands。用户输入 `/code_review`，并从表单中选择 arguments。server 的 prompt 是 "user shortcut" 与 "full prompt sent to model" 之间的 contract。

并非每个 client 都支持 prompts —— 检查 capability negotiation。声明了 prompt capability 但 client 不支持 prompt 的 server，不会看到 slash commands。

### The "list changed" notification

当集合发生 mutation 时，resources 和 prompts 都会发出 `notifications/list_changed`。一个刚导入 20 条新 notes 的 notes server 会发出 `notifications/resources/list_changed`；client 重新调用 `resources/list` 以获取新增项。

### Content type conventions

For text：`mimeType: "text/plain"`、`text/markdown`、`application/json`。
For binary：`image/png`、`application/pdf`，再加 `blob` field。
For MCP Apps (Lesson 14)：`ui://` URI 中使用 `text/html;profile=mcp-app`。

### Dynamic resources

resource URI 不一定对应静态文件。`notes://recent` 可以每次 read 都返回最新五条 notes。`db://query/users/active` 可以执行 parameterized query。server 可以自由动态计算 content。

规则：如果 client 能按 URI cache，那么 URI 必须稳定。如果 computation 是 one-shot，URI 应包含 timestamp 或 nonce，避免 client cache stale out。

### Subscriptions vs polling

支持 subscription 的 clients 可以通过 `notifications/resources/updated` 获得 server push。还不支持 subscription 的 clients 或 hosts 通过重新读取来 poll。两者都符合 spec。server 的 capability declaration 会告诉 client 它支持哪种方式。

Subscriptions 的代价：server 上的 per-session state (谁订阅了什么)。保持 subscribed set bounded；disconnected clients 应该 timeout。

### Prompts vs system prompts

MCP 中的 prompts 不是 system prompts。host 的 system prompt (它自己的 operating instructions) 与 MCP prompts (由用户调用的 server-supplied templates) 并排存在。行为良好的 client 从不让 server prompt 覆盖自己的 system prompt；它会把它们 layer 在一起。

## 实际使用

`code/main.py` 在 Lesson 07 的 notes server 基础上扩展了：

- 每条 note 的 resources (`notes://note-1` 等)，支持 `resources/subscribe`。
- 一个会渲染为 three-message template 的 `review_note` prompt。
- 一个 file-watcher simulation，会在 note 被修改时发出 `notifications/resources/updated`。
- 一个 `notes://recent` dynamic resource，总是返回最新五条 notes。

运行 demo 来查看完整 flow。

## 交付成果

本课产出 `outputs/skill-primitive-splitter.md`。给定一个拟议的 MCP server，该 skill 会把每个 capability 分类为 tool / resource / prompt，并给出 rationale。

## 练习

1. 运行 `code/main.py`。观察初始 resource list，然后触发一次 note edit，并验证 `notifications/resources/updated` event 被发出。

2. 添加一个 `resources/list_changed` emitter：创建新 note 时发送 notification，让 clients re-discover。

3. 为 GitHub MCP server 设计三个 prompts：`summarize_pr`、`triage_issue`、`release_notes`。每个都带 argument schemas。prompt body 应无需进一步 edits 就能运行。

4. 拿 Lesson 07 server 中的一个现有 tool，分类它应该保留为 tool，还是拆成 resource plus tool pair。用一句话说明理由。

5. 阅读 spec 的 `server/resources` 和 `server/prompts` sections。找出 `resources/read` 中一个很少被填充但 spec 支持的字段。提示：看 resource content 上的 `_meta`。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Resource | "Exposed data" | host 可以读取的 URI-addressable content |
| Resource URI | "Pointer to data" | Scheme-prefixed identifier (`file://`, `notes://`, etc.) |
| `resources/subscribe` | "Watch for changes" | client opt-in 的 server-push updates，针对特定 URI |
| `notifications/resources/updated` | "Resource changed" | 告诉 client subscribed resource 有新 content 的 signal |
| Resource template | "Parameterized URI" | 带 host picker completion hints 的 URI pattern |
| Prompt | "Slash-command template" | 带 argument slots 的 named multi-message template |
| Prompt arguments | "Template inputs" | host 在渲染前收集的 typed parameters |
| `prompts/get` | "Render template" | server 返回填充后的 message list |
| Content block | "Typed chunk" | `{type: text \| image \| resource \| ui_resource}` |
| Slash-command UX | "User shortcut" | host 把 prompts 显示为以 `/` 开头的 commands |

## 延伸阅读

- [MCP — Concepts: Resources](https://modelcontextprotocol.io/docs/concepts/resources) — resource URIs、subscriptions 和 templates
- [MCP — Concepts: Prompts](https://modelcontextprotocol.io/docs/concepts/prompts) — prompt templates 和 slash-command integration
- [MCP — Server resources spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/resources) — 完整的 `resources/*` message reference
- [MCP — Server prompts spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts) — 完整的 `prompts/*` message reference
- [MCP — Protocol info site: resources](https://modelcontextprotocol.info/docs/concepts/resources/) — 扩展官方文档的 community guide
