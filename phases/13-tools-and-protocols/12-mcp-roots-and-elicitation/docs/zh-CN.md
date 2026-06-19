# Roots 与 Elicitation：作用域和中途用户输入

> 用户一打开不同项目，hard-coded paths 就会坏掉。用户给得不够具体时，预填 tool arguments 也会坏。Roots 把 server 限定到用户控制的一组 URIs；elicitation 在 tool-call 中途暂停，通过 form 或 URL 向用户请求 structured input。两个 client primitives，分别修复两类常见 MCP failure modes。SEP-1036 (URL-mode elicitation, 2025-11-25) 在 H1 2026 仍是 experimental —— 依赖它之前请检查 SDK versions。

**类型:** 构建
**语言:** Python (stdlib, roots + elicitation demo)
**先修:** Phase 13 · 07 (MCP server)
**时间:** ~45 分钟

## 学习目标

- 声明 `roots` 并响应 `notifications/roots/list_changed`。
- 把 server file operations 限制在 declared root set 内的 URIs。
- 使用 `elicitation/create` 在 tool-call 中途请求用户确认或 structured input。
- 在 form-mode 和 URL-mode elicitation 之间选择 (后者是 experimental；注意 drift-risk)。

## 要解决的问题

notes MCP server 在 production 中会遇到两个具体失败。

**Broken path assumption。** server 按 `~/notes` 编写。换一台机器后，用户的 notes 在 `~/Documents/Notes`，tool call 要么静默失败 (no file found)，要么更糟，写到了错误位置。

**Missing argument the user would know。** 用户说 "delete the old TPS report note"。model 调用 `notes_delete(title: "TPS report")`，但 2023、2024 和 2025 各有一条匹配 note。tool 不能猜。返回 "ambiguous" 很烦；对三条全部执行则是灾难。

Roots 修复第一个问题：client 在 `initialize` 时声明 server 可以触碰的 URIs 集合。Elicitation 修复第二个问题：server 暂停 tool call，发送 `elicitation/create`，要求用户选择其中一个。

## 核心概念

### Roots

client 在 `initialize` 中声明 root list：

```json
{
  "capabilities": {"roots": {"listChanged": true}}
}
```

server 随后可以调用 `roots/list`：

```json
{"roots": [{"uri": "file:///Users/alice/Documents/Notes", "name": "Notes"}]}
```

Servers MUST 把 roots 当作边界：任何 root set 外的 file read 或 write 都会被拒绝。这不是由 client 强制执行的 (server 仍然是用户信任的 code)，但 spec-compliant servers 会遵守。

当用户添加或删除 root 时，client 发送 `notifications/roots/list_changed`。server 重新调用 `roots/list` 并更新边界。

### Why roots are a client primitive

Roots 由 client 声明，因为它们代表用户的 consent model。用户告诉 Claude Desktop "give this notes server access to these two directories"。server 不能扩大这个 scope。

### Elicitation: the form-mode default

`elicitation/create` 接收一个 form schema 加一个 natural-language prompt：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Delete 'TPS report'? Multiple notes match; pick one.",
    "requestedSchema": {
      "type": "object",
      "properties": {
        "note_id": {
          "type": "string",
          "enum": ["note-3", "note-7", "note-14"]
        },
        "confirm": {"type": "boolean"}
      },
      "required": ["note_id", "confirm"]
    }
  }
}
```

client 渲染一个 form，收集用户答案，并返回：

```json
{
  "action": "accept",
  "content": {"note_id": "note-14", "confirm": true}
}
```

三种可能 actions：`accept` (用户填写了它)、`decline` (用户关闭了它)、`cancel` (用户终止整个 tool call)。

Form schemas 是 flat 的 —— v1 不支持 nested objects。SDKs 通常会拒绝比 single layer 更复杂的内容。

### Elicitation: URL mode (SEP-1036, experimental)

2025-11-25 新增。server 发送 URL，而不是 schema：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Sign in to GitHub",
    "url": "https://github.com/login/oauth/authorize?client_id=..."
  }
}
```

client 在 browser 中打开 URL，等待 completion，并在用户回来时返回。适用于 OAuth flows、payment authorization 和 document signing 这类 form 不足以表达的场景。

Drift-risk note：SEP-1036 response shape 仍在稳定中；一些 SDKs 返回 callback URL，另一些返回 completion token。production 中使用 URL mode 前，请阅读你所用 SDK 的 release notes。

### When elicitation is the right tool

- destructive actions 之前的用户确认 (destructive hint + elicitation)。
- Disambiguation (从 N 个 matches 中选择一个)。
- First-run setup (API keys, directories, preferences)。
- OAuth-style flows (URL mode)。

### When elicitation is wrong

- 填充 model 本可以用自然语言追问到的 required tool arguments。使用普通 re-prompt，而不是 elicitation dialog。
- High-frequency calls。Elicitation 会打断 conversation；不要在 loop 内触发。
- server 可以事后 validate 的任何内容。validate、返回 error，让 model 用 text 向用户询问。

### Human-in-the-loop bridge

Elicitation 加 sampling 共同构成 MCP 的 "human-in-the-loop" model。server 的 agent loop 可以暂停以获取用户输入 (elicitation) 或 model reasoning (sampling)。Phase 13 · 11 介绍了 sampling；本课介绍 elicitation。把二者组合起来，就能获得完整 mid-loop control。

## 实际使用

`code/main.py` 扩展 notes server，加入：

- `roots/list` response，server 会在 root-list-changed notifications 后重新查询它。
- 一个 `notes_delete` tool，当多个 notes 匹配时使用 `elicitation/create` 来 disambiguate。
- 一个 `notes_setup` tool，使用 URL-mode elicitation 打开 first-run config page (模拟)。
- 一个 boundary check，拒绝对 declared roots 之外 URIs 的操作。

demo 运行三个 scenarios：happy path (一个 match)、disambiguation (三个 matches，elicitation 触发)、out-of-root-write (被拒绝)。

## 交付成果

本课产出 `outputs/skill-elicitation-form-designer.md`。给定一个可能需要用户确认或 disambiguation 的 tool，该 skill 会设计 elicitation form schema 和 message template。

## 练习

1. 运行 `code/main.py`。触发 disambiguation path；确认 simulated user answer 会被路由回 tool。

2. 添加一个新 tool `notes_archive`，每次都要求 elicitation confirmation (destructive hint)。检查 UX：这和 model 用 text 重新询问相比如何？

3. 为 first-run OAuth flow 实现 URL-mode elicitation。注意 drift risk，并添加 SDK-version guard。

4. 扩展 `roots/list` handling：notification 到达时，server 应原子地重新读取并重新扫描可能已经 out of scope 的 open file handles。

5. 阅读 GitHub 上的 SEP-1036 issue discussion thread。找出一个会影响 servers 应如何处理 URL-mode callbacks 的 open question。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Root | "Consent boundary" | client 允许 server 触碰的 URI |
| `roots/list` | "Server asks for scope" | client 返回当前 root set |
| `notifications/roots/list_changed` | "User changed scope" | client 表示 root set 已发生 mutation |
| Elicitation | "Ask the user mid-call" | server-initiated request，用于 structured user input |
| `elicitation/create` | "The method" | elicitation requests 的 JSON-RPC method |
| Form mode | "Schema-driven form" | 由 flat JSON Schema 渲染成 client UI 中的 form |
| URL mode | "Browser redirect" | SEP-1036 experimental；打开 URL 并等待 |
| `accept` / `decline` / `cancel` | "User response outcomes" | server 需要处理的三条分支 |
| Disambiguation | "Pick one" | tool 有 N 个 candidates 时的常见 elicitation use case |
| Flat form | "Top-level properties only" | Elicitation schemas 不能嵌套 |

## 延伸阅读

- [MCP — Client roots spec](https://modelcontextprotocol.io/specification/draft/client/roots) — canonical roots reference
- [MCP — Client elicitation spec](https://modelcontextprotocol.io/specification/draft/client/elicitation) — canonical elicitation reference
- [Cisco — What's new in MCP elicitation, structured content, OAuth enhancements](https://blogs.cisco.com/developer/whats-new-in-mcp-elicitation-structured-content-and-oauth-enhancements) — 2025-11-25 additions walk-through
- [MCP — GitHub SEP-1036](https://github.com/modelcontextprotocol/modelcontextprotocol) — URL-mode elicitation proposal (experimental, drift-risk)
- [The New Stack — How elicitation brings human-in-the-loop to AI tools](https://thenewstack.io/how-elicitation-in-mcp-brings-human-in-the-loop-to-ai-tools/) — UX walkthrough
