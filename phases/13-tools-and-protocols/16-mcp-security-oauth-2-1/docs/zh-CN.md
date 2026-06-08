# MCP Security II：OAuth 2.1、Resource Indicators、Incremental Scopes

> 远程 MCP 服务器需要授权，而不只是认证。2025-11-25 spec 与 OAuth 2.1 + PKCE + resource indicators（RFC 8707）+ protected-resource metadata（RFC 9728）对齐。SEP-835 增加了 incremental scope consent，并通过 403 WWW-Authenticate 执行 step-up authorization。本课把 step-up flow 实现为状态机，让你看清每一次跳转。

**类型：** 构建
**语言：** Python（stdlib，OAuth state machine simulator）
**先修：** Phase 13 · 09（transports）、Phase 13 · 15（security I）
**时间：** 约 75 分钟

## 学习目标

- 区分 resource server 与 authorization server 的职责。
- 走通受 PKCE 保护的 OAuth 2.1 authorization code flow。
- 使用 `resource`（RFC 8707）和 protected-resource metadata（RFC 9728）防止 confused-deputy attacks。
- 实现 step-up authorization：server 返回带 WWW-Authenticate 的 403，请求更高 scope；client 重新提示用户 consent 并重试。

## 要解决的问题

早期 MCP（2025 年之前）让远程服务器使用临时 API keys，甚至没有 auth。2025-11-25 spec 用完整 OAuth 2.1 profile 补上了这个缺口。

三个真实需求：

- **普通远程服务器。** 用户安装访问其 Notion / GitHub / Gmail 的远程 MCP server。OAuth 2.1 with PKCE 是正确形态。
- **Scope escalation。** 已获 `notes:read` 的 notes server 后续可能因为某个具体动作需要 `notes:write`。不必重跑完整 flow，step-up（SEP-835）会请求额外 scope。
- **Confused deputy prevention。** client 持有 audience-scoped 给 Server A 的 token。恶意 Server A 试图把该 token 提交给 Server B。Resource indicators（RFC 8707）把 token 固定到目标 audience。

OAuth 2.1 并不新。新的是 MCP 的 profile：明确要求的 flows（仅 authorization code + PKCE；默认没有 implicit、没有 client credentials）、每次 token request 必须带 resource indicators，以及发布 protected-resource metadata 让 clients 知道该去哪里。

## 核心概念

### 角色

- **Client。** MCP client（Claude Desktop、Cursor 等）。
- **Resource server。** MCP server（notes、GitHub、Postgres 等）。
- **Authorization server。** 签发 tokens。可以与 resource server 是同一服务，也可以是单独 IdP（Auth0、Keycloak、Cognito）。

在 MCP profile 中，resource 和 authorization servers 可以是同一 host，但应当通过 URL 区分。

### Authorization code + PKCE

流程：

1. Client 生成 `code_verifier`（随机值）和 `code_challenge`（SHA256）。
2. Client 将用户重定向到 `/authorize?response_type=code&client_id=...&redirect_uri=...&scope=notes:read&code_challenge=...&resource=https://notes.example.com`。
3. 用户 consent。Authorization server 重定向到 `redirect_uri?code=...`。
4. Client POST 到 `/token?grant_type=authorization_code&code=...&code_verifier=...&resource=...`。
5. Authorization server 校验 verifier 的 hash 是否匹配存储的 challenge，并签发 access token。
6. Client 使用该 token：对 resource server 的每个请求都带 `Authorization: Bearer ...`。

PKCE 防止 authorization-code interception attacks。Resource indicators 防止 token 在别处有效。

### Protected-resource metadata (RFC 9728)

resource server 发布 `.well-known/oauth-protected-resource` 文档：

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:read", "notes:write", "notes:delete"]
}
```

Client 从 resource server 发现 authorization server。这样减少配置，client 只需要 resource URL。

### Resource indicators (RFC 8707)

token request 中的 `resource` 参数会固定 token 的目标 audience。签发出的 token 包含 `aud: "https://notes.example.com"`。另一个 MCP server 收到此 token 后检查 `aud` 并拒绝。

### Scope model

Scopes 是空格分隔的字符串。常见 MCP 约定：

- `notes:read`、`notes:write`、`notes:delete`
- `admin:*` 表示 admin capabilities（少用）
- `profile:read` 表示 identity

Scope selection 应遵循 least-privilege：只请求当前需要的，在需要更多时 step up。

### Step-up authorization (SEP-835)

用户授予 `notes:read`。之后他们要求 agent 删除一条笔记。服务器响应：

```text
HTTP/1.1 403 Forbidden
WWW-Authenticate: Bearer error="insufficient_scope",
    scope="notes:delete", resource="https://notes.example.com"
```

Client 看到 insufficient_scope error，用 consent dialog 提示用户授予额外 scope，为它执行一个 mini OAuth flow，然后用新 token 重试请求。

### Token audience validation

每次请求：server 检查 `token.aud == self.resource_url`。不匹配 = 401。这会阻止 cross-server token reuse。

### Short-lived tokens and rotation

Access tokens 应当短生命周期（默认 1 小时）。Refresh tokens 每次 refresh 都要 rotate。client 在后台处理 silent refresh。

### No token passthrough

Sampling servers（Phase 13 · 11）绝不能把 client 的 token 传递给其他服务。sampling request 就是边界。

### Confused deputy prevention

Token 绑定到 `aud`。Client 绑定到 `client_id`。每个请求都要同时验证两者。spec 明确禁止旧的 “pass-the-token” 模式，这种模式在 MCP 之前的远程工具生态中很常见。

### Client ID discovery

每个 MCP client 都在固定 URL 发布自己的 metadata。Authorization servers 可以获取 client 的 metadata document，以发现 redirect URIs 和联系信息。这移除了手工 client registration。

### Gateways and OAuth

Phase 13 · 17 展示企业 gateway 如何处理 OAuth：gateway 持有 upstream servers 的 credentials，发给 client 的 tokens 由 gateway 签发，upstream tokens 永远不离开 gateway。这会翻转 trust model：用户只需向 gateway 认证一次；gateway 处理 N 个服务器授权。

## 实际使用

`code/main.py` 将完整 OAuth 2.1 step-up flow 模拟为状态机。它实现：

- PKCE code-verifier / challenge generation。
- 带 resource indicator 的 authorization code flow。
- Protected-resource metadata endpoint。
- 带 audience check 的 token validation。
- 在 `insufficient_scope` 上 step-up。

本课没有 HTTP server；状态机在内存中运行，便于你追踪每一次跳转。Phase 13 · 17 的 gateway 课程会把它接到真实 transport。

## 交付成果

本课产出 `outputs/skill-oauth-scope-planner.md`。给定一个带 tools 的远程 MCP server，该 skill 会设计 scope set、pinning rules 和 step-up policy。

## 练习

1. 运行 `code/main.py`。追踪双 scope step-up flow。注意 step-up 时哪些跳转会重复。

2. 添加 refresh-token rotation：每次 refresh 签发新的 refresh token，并让旧 token 失效。模拟被盗 refresh token 在 rotation 后被使用，确认它失败。

3. 使用 stdlib http.server 将 protected-resource metadata endpoint 实现为真实 HTTP response。参考 Lesson 09 的 /mcp endpoint。

4. 为 GitHub MCP server 设计 scope hierarchy：read repo、write PR、approve PR、merge PR、admin。在每个层级之间使用 step-up。

5. 阅读 RFC 8707 和 RFC 9728。找出 9728 中 MCP 用法不同于 RFC 示例的一个字段。（提示：它与 `scopes_supported` 有关。）

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| OAuth 2.1 | “Modern OAuth” | 要求 PKCE 并禁止 implicit flow 的整合版 RFC |
| PKCE | “Proof-of-possession” | code verifier + challenge，用于击败 authorization-code interception |
| Resource indicator | “Token audience” | RFC 8707 `resource` 参数，将 token 固定到一个服务器 |
| Protected-resource metadata | “Discovery doc” | RFC 9728 `.well-known/oauth-protected-resource` |
| Step-up authorization | “Incremental consent” | SEP-835 按需添加 scopes 的 flow |
| `insufficient_scope` | “403 with WWW-Authenticate” | server 发出的重新 consent 更大 scope 的信号 |
| Confused deputy | “Token reuse across services” | 可信持有方不当转发 token 的攻击 |
| Short-lived token | “Access token TTL” | 快速过期的 Bearer；refresh token 用于续期 |
| Scope hierarchy | “Least privilege stack” | 分级 scope set，层级之间 step-up |
| Client ID metadata | “Client discovery doc” | client 发布自身 OAuth metadata 的 URL |

## 延伸阅读

- [MCP — Authorization spec](https://modelcontextprotocol.io/specification/draft/basic/authorization) — canonical MCP OAuth profile
- [den.dev — MCP November authorization spec](https://den.dev/blog/mcp-november-authorization-spec/) — 2025-11-25 变更 walkthrough
- [RFC 8707 — Resource indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — audience-pinning RFC
- [RFC 9728 — OAuth 2.0 protected resource metadata](https://datatracker.ietf.org/doc/html/rfc9728) — discovery-document RFC
- [Aembit — MCP OAuth 2.1, PKCE and the future of AI authorization](https://aembit.io/blog/mcp-oauth-2-1-pkce-and-the-future-of-ai-authorization/) — 实用 step-up-flow walkthrough
