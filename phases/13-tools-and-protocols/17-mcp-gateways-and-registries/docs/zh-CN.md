# MCP Gateways and Registries：企业控制平面

> 企业不能让每个开发者随意安装随机 MCP servers。gateway 集中处理 auth、RBAC、audit、rate limiting、caching 和 tool-poisoning detection，然后把合并后的 tool surface 作为单个 MCP endpoint 暴露。Official MCP Registry（Anthropic + GitHub + PulseMCP + Microsoft，namespace-verified）是 canonical upstream。本课说明 gateway 放在哪里，走读一个最小实现，并概览 2026 年供应商格局。

**类型：** 学习
**语言：** Python（stdlib，minimal gateway）
**先修：** Phase 13 · 15（tool poisoning）、Phase 13 · 16（OAuth 2.1）
**时间：** 约 45 分钟

## 学习目标

- 解释 MCP gateway 的位置（位于 MCP clients 与多个 backend MCP servers 之间）。
- 实现 gateway 的五项职责：auth、RBAC、audit、rate limit、policy。
- 在 gateway 层执行 pinned-tool-hash manifest。
- 区分 Official MCP Registry 与 metaregistries（Glama、MCPMarket、MCP.so、Smithery、LobeHub）。

## 要解决的问题

一家 Fortune 500 公司有 30 个已批准 MCP servers、5000 名开发者、合规与审计要求，以及一个希望集中 policy 的安全团队。让每个开发者在 IDE 中安装任意服务器完全不可接受。

gateway pattern：

1. Gateway 作为单个 Streamable HTTP endpoint 运行，开发者连接到它。
2. Gateway 持有每个 backend MCP server 的 credentials。
3. 每个开发者请求都通过 gateway 自己的 OAuth 认证并设定 scope。
4. Gateway 将调用路由到 backend server，同时应用 policy。
5. 所有调用都写入 audit log。

Cloudflare MCP Portals、Kong AI Gateway、IBM ContextForge、MintMCP、TrueFoundry、Envoy AI Gateway 都在 2025-2026 年发布了 gateways 或 gateway features。

与此同时，Official MCP Registry 作为 canonical upstream 发布：它收录经过 curated、namespace-verified、reverse-DNS-named 的服务器，供 gateway 拉取。Metaregistries（Glama、MCPMarket、MCP.so、Smithery、LobeHub）则聚合多个来源的服务器。

## 核心概念

### 五项 gateway 职责

1. **Auth。** 用 OAuth 2.1 识别开发者；映射到用户 roles。
2. **RBAC。** Per-user policy：哪些 servers、哪些 tools、哪些 scopes。
3. **Audit。** 每次调用都记录 who、what、when、result。
4. **Rate limit。** Per-user / per-tool / per-server caps，用来防止滥用。
5. **Policy。** 拒绝 poisoned descriptions、执行 Rule of Two、redact PII。

### Gateway as a single endpoint

对开发者来说，gateway 看起来像一个 MCP server。内部它路由到 N 个 backends。Session ids（Phase 13 · 09）会在边界处重写。

### Credential vaulting

开发者永远看不到 backend tokens。gateway 持有它们（或代理到实际持有它们的 identity provider）。在 gateway 上拥有 `notes:read` 的开发者，可以传递性地用 gateway 自己的 backend credentials 访问 notes MCP server，但只能在绑定该传递访问的 policy 下进行。

### Tool-hash pinning at the gateway

gateway 持有 approved tool descriptions 的 manifest（SHA256 hashes）。在 discovery 时，它获取每个 backend 的 `tools/list`，把 hashes 与 manifest 对比，并移除任何 description 已发生变更的工具。这是 Phase 13 · 15 的 rug-pull 防御集中应用。

### Policy-as-code

高级 gateways 用 OPA/Rego、Kyverno 或 Styra 表达 policy。像“用户 `alice` 只能在 org `acme` 的 repos 上调用 `github.open_pr`”这样的规则会以声明式方式编码。简单 gateways 使用手写 Python。两种形态都有效。

### Session-aware routing

当用户 session 包含多个服务器时，gateway 会 multiplex：开发者的单个 MCP session 持有 N 个 backend sessions，每个服务器一个。来自任意 backend 的 notifications 都经 gateway 路由到开发者 session。

### Namespace merging

Gateways 会合并所有 backends 的 tool namespaces，通常在冲突时加前缀。`github.open_pr`、`notes.search`。这样路由没有歧义。

### Registries

- **Official MCP Registry (`registry.modelcontextprotocol.io`)。** 在 Anthropic、GitHub、PulseMCP、Microsoft 监管下发布。Namespace-verified（reverse-DNS：`io.github.user/server`）。预先过滤基本质量。
- **Glama。** 以搜索为中心的 metaregistry，聚合许多来源。
- **MCPMarket。** 偏商业目录，包含 vendor listings。
- **MCP.so。** 社区目录；开放提交。
- **Smithery。** 类似 package manager 的安装流程。
- **LobeHub。** 集成在 LobeChat app UI 中的 registry。

Enterprise gateways 默认从 Official Registry 拉取，允许管理员从 metaregistries curated additions，并拒绝任何未 pin 的内容。

### Reverse-DNS naming

Official Registry 要求公共服务器使用 reverse-DNS names：`io.github.alice/notes`。Namespaces 防止抢注，并让信任委托更清晰。

### Vendor survey, April 2026

| Vendor | Strength |
|--------|----------|
| Cloudflare MCP Portals | Edge-hosted; OAuth integrated; free tier |
| Kong AI Gateway | K8s-native; fine-grained policy; logs to OpenTelemetry |
| IBM ContextForge | Enterprise IAM; compliance; audit export |
| TrueFoundry | DevOps-leaning; metrics-first |
| MintMCP | Developer-platform oriented |
| Envoy AI Gateway | Open-source; customizable filters |

Phase 17（production infrastructure）会更深入讨论 gateway operations。

## 实际使用

`code/main.py` 用约 150 行实现一个 minimal gateway：通过 fake Bearer token 认证用户，持有 per-user RBAC policy，把请求路由到两个 backend MCP servers，把每次调用写入 audit log，执行 rate limit，并拒绝 description hash 不匹配 pinned manifest 的任何 backend tool。

重点查看：

- `RBAC` dict 以 `user_id` 为 key，包含允许的 `server_tool` entries。
- `AUDIT_LOG` 是 append-only event list。
- Rate limit 使用 per user 的 token bucket。
- Pinned manifest 是 `server::tool -> hash` 的 dict。

## 交付成果

本课产出 `outputs/skill-gateway-bootstrap.md`。给定一个 enterprise MCP plan（users、backends、compliance），该 skill 会生成 gateway configuration spec。

## 练习

1. 运行 `code/main.py`。以允许用户发起一次调用；再以不允许用户发起；然后发起一个超出 rate limit 的 burst。验证三种 flow。

2. 添加一条 policy，在结果返回 client 之前 redact PII。对 SSN 形状的字符串使用简单 regex pass；记下缺口（emails、phone numbers）。

3. 扩展 audit log，使其发射 OpenTelemetry GenAI spans。Phase 13 · 20 会覆盖准确 attributes。

4. 为一个 50 人开发团队设计 RBAC policy，含五个 backends（notes、github、postgres、jira、slack）。谁对每个 backend 只有 read-only？谁拥有 write？

5. 从头到尾阅读 Cloudflare enterprise MCP 文章。找出 Cloudflare 提供但这个 stdlib gateway 没有的一项 feature。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Gateway | “MCP proxy” | clients 与 backends 之间的集中式服务器 |
| Credential vaulting | “Backend tokens stay server-side” | 开发者永远看不到 upstream tokens |
| Session-aware routing | “Multi-backend session” | Gateway 为每个 developer session multiplex N 个 backend sessions |
| Tool-hash pinning | “Approved manifest” | 每个已批准工具描述的 SHA256；集中阻断 rug-pulls |
| RBAC | “Per-user policy” | 面向 tools 与 servers 的 role-based access control |
| Policy-as-code | “Declarative rules” | 在 gateway 执行的 OPA/Rego、Kyverno、Styra policies |
| Audit log | “Who, what, when” | 用于合规的 append-only event log |
| Rate limit | “Per-user token bucket” | 防滥用的 per-minute caps |
| Official MCP Registry | “Canonical upstream” | `registry.modelcontextprotocol.io`，namespace-verified |
| Reverse-DNS naming | “Registry namespace” | `io.github.user/server` 约定 |

## 延伸阅读

- [Official MCP Registry](https://registry.modelcontextprotocol.io/) — canonical upstream，namespace-verified
- [Cloudflare — Enterprise MCP](https://blog.cloudflare.com/enterprise-mcp/) — gateway pattern with OAuth and policy
- [agentic-community — MCP gateway registry](https://github.com/agentic-community/mcp-gateway-registry) — open-source reference gateway
- [TrueFoundry — What is an MCP gateway?](https://www.truefoundry.com/blog/what-is-mcp-gateway) — feature comparison article
- [IBM — MCP context forge](https://github.com/IBM/mcp-context-forge) — IBM 的 enterprise gateway
