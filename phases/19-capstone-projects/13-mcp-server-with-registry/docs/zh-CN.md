# 综合项目 13 — 带注册中心和治理的 MCP Server

> 到 2026 年，Model Context Protocol 不再是未来，而是默认的 tool-use 规范。Anthropic、OpenAI、Google 和所有主流 IDE 都提供 MCP client。Pinterest 公开了内部 MCP server 生态。AAIF Registry 在 `.well-known` 下正式化了能力元数据。AWS ECS 发布了 reference stateless deployment。Block 的 goose-agent 把同一协议放进 hosted assistant。2026 年的生产形态是：StreamableHTTP transport、OAuth 2.1 scopes、OPA policy gating，以及让平台团队发现、验证并启用 server 的 registry。把它端到端构建出来。

**类型:** Capstone
**语言:** Python（server，通过 FastMCP）或 TypeScript（@modelcontextprotocol/sdk），Go（registry service）
**先修:** Phase 11（LLM engineering），Phase 13（tools and MCP），Phase 14（agents），Phase 17（infrastructure），Phase 18（safety）
**覆盖阶段:** P11 · P13 · P14 · P17 · P18
**时间:** 25 小时

## 要解决的问题

MCP 已经成为 tool-use 的通用语。Claude Code、Cursor 3、Amp、OpenCode、Gemini CLI 以及每个 managed agent 现在都会消费 MCP servers。生产挑战不在于编写 server（FastMCP 让这件事很容易），而在于带着企业要求规模化部署它们：per-tenant OAuth scopes、对 destructive tools 的 OPA policy、StreamableHTTP stateless scaling、用于发现的 registry、每次 tool call 的 audit logs。Pinterest 的内部 MCP 生态和 AAIF Registry 规范设定了 2026 年的标准。

你将构建一个暴露 10 个内部工具的 MCP server（Postgres read-only、S3 listing、Jira、Linear、Datadog 等）、一个用于平台发现的 registry UI，以及 destructive tools 的 human-approval gate。Load test 要证明 StreamableHTTP 的水平扩展。Audit trail 要满足企业安全审查。

## 核心概念

MCP 2026 revision 要求 StreamableHTTP 作为默认 transport。不同于早期的 stdio-and-SSE 形态，StreamableHTTP 默认无状态：单个 HTTP endpoint 接收 JSON-RPC requests、流式传输 responses，并支持用于 notifications 的 long-lived connections。无状态意味着可以在 load balancer 后面水平扩展。

授权使用 OAuth 2.1 和 per-tool scopes。Token 携带 `jira:read`、`s3:list`、`postgres:query:readonly` 等 scopes。MCP server 在 tool-call time 检查 scopes，而不只是在 session start 检查。对于高风险工具，如果某次调用的 scope 在最近 N 分钟内没有提升到 `approved:by:human`，server 就会拒绝；这个提升来自 Slack review card。

Registry 是一个独立服务。每个 MCP server 都暴露 `.well-known/mcp-capabilities` 文档，其中包含 tool manifest、transport URL、auth requirements。Registry 负责轮询、验证和索引。平台团队使用 registry UI 查看有哪些工具、需要哪些 scopes、由哪个团队拥有。

## 架构

```text
MCP client (Claude Code, Cursor 3, ...)
          |
          v
StreamableHTTP over HTTPS (JSON-RPC + streaming)
          |
          v
MCP server (FastMCP) behind load balancer
          |
   +------+------+---------+----------+------------+
   v             v         v          v            v
Postgres    S3 listing  Jira       Linear     Datadog
(read-only) (paged)     (read)     (read)     (query)
          |
   +------+-------------+
   v                    v
 OPA policy gate   destructive tool MCP (separate server)
                        |
                        v
                   human approval via Slack
                        |
                        v
                   audit log (append-only, per-tenant)

  registry service
     |
     v  GET /.well-known/mcp-capabilities from each server
     v
     UI: search / validate / enable-disable / ownership
```

## 技术栈

- Server framework：FastMCP（Python）或 `@modelcontextprotocol/sdk`（TypeScript）
- Transport：基于 HTTPS 的 StreamableHTTP（stateless）
- Auth：OAuth 2.1，workload identity 通过 SPIFFE / SPIRE
- Policy：每个工具的 OPA / Rego rules；每次请求调用 policy decision service
- Registry：自托管，消费 `.well-known/mcp-capabilities` manifests
- Human approval：Slack interactive message，用于 destructive tools
- Deployment：AWS ECS Fargate 或 Fly.io；每租户一个 server，或共享 server 加 tenant scoping
- Audit：每租户 bucket 中的 structured JSONL，带 per-call lineage

## 动手实现

1. **工具表面。** 暴露 10 个内部工具：Postgres read-only query、S3 list objects、Jira search/fetch、Linear search/fetch、Datadog metric query、PagerDuty on-call lookup、GitHub read-only、Notion search、Slack search、Salesforce read。每个工具都有 typed schema 和 scope label。

2. **FastMCP server。** 挂载工具。配置 StreamableHTTP transport。添加 middleware 做 OAuth token introspection 和 scope enforcement。

3. **OPA policy。** 每个工具一份 Rego policy：哪些 scopes 允许调用、应用什么 PII redaction、payload-size caps 是多少。每次 tool call 都调用 decision service。

4. **Registry service。** 单独的 Go 或 TS 服务，轮询已注册 server 的 `.well-known/mcp-capabilities`，用 JSON Schema 验证，并暴露 list / search / validate / enable-disable UI。

5. **Capability manifest。** 每个 server 暴露 `.well-known/mcp-capabilities`，包含：tool list、auth requirements、transport URL、owner team、SLO。

6. **Destructive tool separation。** 会修改状态的工具（Jira create、Linear create、Postgres write）放在第二个 MCP server 上，并使用更严格的 auth flow：tokens 必须在 15 分钟内通过 Slack card 提升得到 `approved:by:human` scope。

7. **Audit log。** 每租户 append-only JSONL：`{timestamp, user, tool, args_redacted, response_redacted, outcome}`。写入前通过 Presidio 做 PII redaction。

8. **Load test。** StreamableHTTP 上 100 个并发 clients。通过增加第二个 replica 演示水平扩展；展示 load balancer 在没有 session stickiness 的情况下重新分配流量。

9. **Conformance tests。** 对两个 servers 运行官方 MCP conformance suite。通过所有 mandatory sections。

## 实际使用

```text
$ curl -H "Authorization: Bearer eyJhbGc..." \
       -X POST https://mcp.internal.example.com/ \
       -d '{"jsonrpc":"2.0","method":"tools/call",
            "params":{"name":"postgres.readonly","arguments":{"sql":"SELECT 1"}}}'
[registry]   capability validated: postgres.readonly v1.2
[policy]    scope postgres:query:readonly present; allowed
[audit]     logged: user=u42 tool=postgres.readonly outcome=ok
response:    { "result": { "rows": [[1]] } }
```

## 交付成果

`outputs/skill-mcp-server.md` 描述交付物。一个 production-grade MCP server + registry + audit layer，用于带 OAuth 2.1 scopes 和 OPA gating 的内部工具。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | Spec conformance | StreamableHTTP + capability manifest 通过 MCP conformance tests |
| 20 | Security | Scope enforcement、所有工具的 OPA coverage、secret hygiene |
| 20 | Observability | 带 PII redaction 的 per-tool-call audit log |
| 20 | Scale | 100-client load test 水平扩展示范 |
| 15 | Registry UX | Discover / validate / enable-disable workflow |
| **100** | | |

## 练习

1. 添加一个新工具（Confluence search）。不触碰 core server，通过 registry validation flow 发布它。

2. 编写一个 OPA policy，redact Postgres query results 中列名为 `email`、`ssn` 或 `phone` 的列。用 probe query 练习。

3. Benchmark StreamableHTTP vs stdio 的本地延迟。报告 per-call p50/p95。

4. 实现 per-tenant quota：每个租户每个工具每分钟最多 N 次调用。通过第二条 OPA rule 强制执行。

5. 从 [mcp-conformance-tests](https://github.com/modelcontextprotocol/conformance) 运行 MCP conformance suite，并修复每个 failure。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| StreamableHTTP | “2026 MCP transport” | 无状态 HTTP + streaming；取代网络化 server 中的 SSE + stdio |
| Capability manifest | “Well-known doc” | `.well-known/mcp-capabilities`，包含 tool list、auth、transport URL |
| OPA / Rego | “Policy engine” | Open Policy Agent，用外部规则授权 tool calls |
| Scope elevation | “Approved-by-human” | 通过 Slack approval 授予的短期 scope；destructive tools 必需 |
| Registry | “Tool discovery” | 从 capability manifests 索引 MCP servers 的服务 |
| Workload identity | “SPIFFE / SPIRE” | 用于 OAuth token issuance 的加密服务身份 |
| Conformance suite | “Spec tests” | StreamableHTTP + tool manifest correctness 的官方 MCP test battery |

## 延伸阅读

- [Model Context Protocol 2026 Roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP、capability metadata、registry
- [AAIF MCP Registry spec](https://github.com/modelcontextprotocol/registry) — 2026 registry spec
- [AWS ECS reference deployment](https://aws.amazon.com/blogs/containers/deploying-model-context-protocol-mcp-servers-on-amazon-ecs/) — reference production deployment
- [Pinterest internal MCP ecosystem](https://www.infoq.com/news/2026/04/pinterest-mcp-ecosystem/) — reference internal deployment
- [Block `goose` MCP usage](https://block.github.io/goose/) — reference agent consumption pattern
- [FastMCP](https://github.com/jlowin/fastmcp) — Python server framework
- [Open Policy Agent](https://www.openpolicyagent.org/) — policy engine reference
- [SPIFFE / SPIRE](https://spiffe.io) — workload identity reference
