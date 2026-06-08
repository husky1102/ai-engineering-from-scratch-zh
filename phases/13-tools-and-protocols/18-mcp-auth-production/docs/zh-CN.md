# 生产环境中的 MCP 认证：接入注册、JWKS 刷新与受众固定令牌

> 第 16 课在内存中搭建了 OAuth 2.1 状态机。到 2026 年，每个交付给真实组织的 MCP server 都必须接入生产认证：客户端接入要能支撑不断增长的 client 群体（优先使用 Client ID Metadata Documents，dynamic client registration 作为向后兼容 fallback）；客户端要能发现授权服务器 metadata（RFC 8414 *或* OpenID Connect Discovery）；JWKS 缓存刷新不能让凌晨 3 点的 token validation 中断；token 必须固定 audience，拒绝跨资源 replay。本课用 authorization server、resource server（MCP server）和 client 三个角色建模完整认证表面，让你追踪从 discovery 到已验证 tool call 的每一次跳转。
>
> **规范说明（2025-11-25）：** 2025 年 11 月 MCP authorization spec 将 Dynamic Client Registration 从 `SHOULD` 降级为 `MAY`，并把 **Client ID Metadata Documents (CIMD)** 设为推荐的默认接入机制。本课按规范优先级同时教授两者；代码为了让走查完全自包含在一个进程里，所以保留 DCR。

**类型：** 构建
**语言：** Python（stdlib）
**先修：** 第 13 阶段第 16 课（OAuth 2.1 状态机）、第 13 阶段第 17 课（网关）
**时间：** 约 90 分钟

## 学习目标

- 通过 RFC 8414 metadata 发现 authorization server，并验证它满足契约。
- 实现 RFC 7591 dynamic client registration，让 MCP clients 无需管理员介入即可完成接入。
- 按计划缓存并刷新 JWKS keys，使签名验证能承受 key roll-over。
- 使用 RFC 8707 resource indicators 将 tokens 固定到单个 MCP resource，并拒绝 confused-deputy reuse。
- 清晰分离 authorization server、resource server、client 三个角色，让每个角色只执行属于自己的检查。
- 阅读 IdP 能力矩阵，并在 IdP 不能满足 MCP auth profile 时拒绝部署。

## 要解决的问题

第 16 课的模拟器在内存中运行 OAuth 2.1。生产环境有三个只靠内存模拟器看不到的运维缺口。

第一个缺口是客户端接入。真实组织会运行数百个 MCP servers 和数千个 MCP clients，运营人员不可能把每个 Cursor 用户都手工注册成 OAuth client。2025-11-25 spec 给 clients 定义了优先顺序：已有预注册 `client_id` 时直接使用；否则使用 **Client ID Metadata Document**，也就是让 client 用自己控制的 HTTPS URL 标识自己，并由 authorization server *拉取* metadata；再不行才回退到 **RFC 7591 dynamic client registration**，也就是让 client *推送* `POST /register` 并当场收到 `client_id`；最后才提示用户。CIMD 是推荐默认值，因为它保留基于 DNS 的信任模型，同时移除逐服务器 registration；DCR 则保留用于向后兼容。二者的入口点都来自 authorization server metadata：CIMD 看 `client_id_metadata_document_supported`，DCR 看 `registration_endpoint`。

第二个缺口是 key rotation。JWT validation 依赖 authorization server 的 signing keys，这些 keys 以 JSON Web Key Set (JWKS) 发布。authorization server 会按计划 rotate keys（通常每小时一次，事故响应时可能更快）。MCP server 如果只在启动时获取一次 JWKS，在 rotation window 之前验证正常，之后所有请求都会失败，直到重启。生产环境会把 JWKS 做成带 refresh job 的缓存：在旧 keys 过期前覆盖缓存；cache miss 时再执行一次 fallback fetch，用来处理“token 由比缓存更新的 key 签名，但 token 先于下一次刷新到达”的情况。

第三个缺口是 audience binding。第 16 课引入了 RFC 8707 resource indicators。在生产中，该 indicator 会变成每次请求上的硬性 claim check。MCP server 将 `token.aud` 与自身 canonical resource URL 比较，不匹配就以 HTTP 401 拒绝。这是抵御同一 trust mesh 中的 upstream MCP server（或持有某一服务器 token 的恶意 client）把 token 重放到另一台服务器的唯一防线。

本课把每个缺口映射到认证表面上的具体部件。metadata document 是一个 HTTP endpoint。JWKS cache refresh 由 scheduled job 和 key-value cache 组成。JWT validation 是 resource server 在 dispatch 任何 tool 前运行的 routine。保持三个角色分离，每个角色只执行自己拥有的检查：authorization server 签发并 rotate keys，resource server 缓存并验证，client 负责发现并完成接入。

## 核心概念

### RFC 8414：OAuth Authorization Server Metadata

位于 `/.well-known/oauth-authorization-server` 的文档描述了 client 所需的一切：

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

给定 MCP resource URL 后，client 会执行链式 discovery：RFC 9728 的 `oauth-protected-resource`（resource server 文档）先命名 issuer，然后 `oauth-authorization-server`（本 RFC）再命名每个 endpoint。client 永远不会 hard-code authorization URL。

在信任某个 IdP 用于 MCP 前，你要验证的契约：

- `code_challenge_methods_supported` 包含 `S256`（RFC 7636 的 PKCE）。spec 很明确：如果这个字段**缺失**，authorization server 不支持 PKCE，client **MUST** 拒绝继续。
- `grant_types_supported` 包含 `authorization_code`，并拒绝 `password` 和 `implicit`。
- 至少公布一种接入路径：`client_id_metadata_document_supported: true`（CIMD，优先）**或** `registration_endpoint`（RFC 7591 DCR，fallback）。任一种都满足契约；你不再硬性要求 DCR。
- `response_types_supported` 对 OAuth 2.1 来说恰好是 `["code"]`。

如果缺少 `S256`，MCP server 会拒绝部署到这个 IdP；PKCE 没有降级模式。如果*两种*接入路径都没有公布，且你没有预注册的 `client_id`，也无法完成接入；这是 deployment manifest 错误，不是代码错误。

### RFC 9728（回顾）：Protected Resource Metadata

第 16 课已覆盖 RFC 9728。生产环境中的增量是：这个文档是 client 寻找被*此* MCP server 信任的 authorization servers 的唯一位置。单个 MCP server 可以接受来自多个 IdP 的 tokens（一个给员工，一个给合作伙伴）。RFC 9728 声明这个集合；RFC 8414 说明每个 IdP 支持什么。

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### Client ID Metadata Documents（推荐默认值）

CIMD 将 registration 从 *push* 反转为 *pull*。client 不再请求 authorization server mint 一个 `client_id`，而是使用自己控制的 HTTPS URL **作为** `client_id`。该 URL 解析为 JSON metadata document；authorization server 在 OAuth flow 中按需获取它。信任根植于 DNS：如果服务器运营方信任 `app.example.com`，它就信任从 `https://app.example.com/client.json` 提供的 client。没有 registration round-trip，没有会耗尽的 `client_id` namespace，也没有要在每个服务器间同步的 per-server state。

client 托管的 metadata document：

```json
{
  "client_id": "https://app.example.com/oauth/client.json",
  "client_name": "Example MCP Client",
  "client_uri": "https://app.example.com",
  "redirect_uris": ["http://127.0.0.1:7333/callback", "http://localhost:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none"
}
```

文档中的 `client_id` 值**必须**等于它被提供的 URL（authorization server 会验证；不匹配会被拒绝）。authorization server 通过在 RFC 8414 metadata 中设置 `client_id_metadata_document_supported: true` 宣布支持。

spec 对两个安全事实说得很直接：

- **SSRF。** authorization server 会获取攻击者提供的 URL。它必须防御 server-side request forgery（不能 fetch 内部/admin endpoints）。
- **localhost impersonation。** 仅 CIMD 无法阻止本地攻击者声称一个合法 client 的 metadata URL 并绑定任意 `localhost` redirect。authorization server **MUST** 在 consent 时清晰展示 redirect URI hostname，并且 **SHOULD** 对只包含 `localhost` 的 redirects 发出警告。

因为 CIMD 不需要 server-side state，所以不需要像 DCR 那样搭建 registrar。client 侧是只读的：从一个静态 HTTPS endpoint 提供 metadata document，让 authorization server 拉取即可。

### RFC 7591：Dynamic Client Registration（fallback / 向后兼容）

DCR 现在是 `MAY`，保留用于向后兼容 2025-11-25 之前的部署，以及尚不支持 CIMD 的 IdPs。没有它（也没有 CIMD 或 pre-registration）时，每个 MCP client（Cursor、Claude Desktop、自定义 agent）都需要和 IdP admin 进行带外交换。有了 DCR，client 会发送：

```json
POST /register
Content-Type: application/json

{
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

server 返回 `client_id` 和用于后续更新的 `registration_access_token`：

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`token_endpoint_auth_method: none` 是在用户设备上运行的 MCP clients 的正确默认值。它们只获得 `client_id`，没有可被外传的 `client_secret`。PKCE 提供 public clients 需要的 proof-of-possession。

三个生产陷阱：

- registration endpoint 必须按 source IP 做 rate-limit。否则恶意行为者可以脚本化数百万个 fake registrations，耗尽 `client_id` namespace。在 registrar 处理请求前先运行 rate-limit check。
- 某些 enterprise IdPs 要求 `software_statement`（为 client 背书的 signed JWT）。本课 mock 跳过它；生产环境要接入 verification step，拒绝除 localhost redirect URIs 以外的任何 unsigned registrations。
- `registration_access_token` 必须以 hash 形式存储，而不是 plaintext。该 token 被盗意味着攻击者可以重写 client 的 redirect URIs。

### RFC 8707（回顾）：Resource Indicators

第 16 课已建立其形态。生产规则是：每个 token request 包含 `resource=<canonical-mcp-url>`，MCP server 在每次调用上验证 `token.aud` 是否匹配自身 resource URL。canonical URI 是服务器*最具体*的标识符：使用小写 scheme 和 host，不含 fragment，通常不带 trailing slash。path component 并不会按规则剥离；当它需要用于标识单个 MCP server 时，spec 会保留它。`https://mcp.example.com`、`https://mcp.example.com/mcp`、`https://mcp.example.com:8443` 和 `https://mcp.example.com/server/mcp` 都是有效 canonical URIs。为每个服务器选一个，并把 `aud` 精确固定到它。（本课 mock 为简洁使用类似 `https://notes.example.com` 的 bare-host audiences；在同一 origin 下托管多个 MCP servers 的部署会用 path 区分它们。）

### RFC 7636（回顾）：PKCE

PKCE 在 OAuth 2.1 中是强制项。本课的 authorization-code flow 总是携带 `code_challenge` 和 `code_verifier`。server 会拒绝任何没有 verifier，或 verifier hash 不匹配已存 challenge 的 token request。

### MCP Spec 2025-11-25 认证画像

MCP spec（2025-11-25）精确定义了 MCP server 的 authorization layer 必须做什么：

- 实现 RFC 9728 protected-resource metadata，并通过 401 上的 `WWW-Authenticate: Bearer resource_metadata="..."` header **或** well-known URI `/.well-known/oauth-protected-resource`（SEP-985 让 header 变为可选，并提供 well-known fallback）提供其位置。metadata 的 `authorization_servers` 字段**必须**命名至少一个 server。
- 在**每个**请求上只接受经 `Authorization: Bearer ...` 传递的 tokens，绝不通过 query string，也绝不能只在 session start 验证。
- 每次请求都验证 `aud`、`iss`、`exp` 和 required scopes。server **MUST** 验证 token 是专门签发给它的（audience）；缺失或不匹配的 `aud` 会被拒绝，绝不能视为 wildcard。
- 在 401/403 上返回携带 `error=...`、`resource_metadata="<PRM-URL>"` 参数（metadata document 的 URL，*不是* bare resource）以及 `insufficient_scope`（403）上的 `scope="..."` 的 `WWW-Authenticate: Bearer`。注意：参数是 `resource_metadata`，也就是 discovery pointer；challenge 中没有 `resource` 参数。
- Authorization-server discovery 接受 RFC 8414 OAuth metadata **或** OpenID Connect Discovery 1.0；clients 必须按优先顺序尝试两个 well-known suffixes。
- client（不是 server）防御 **mix-up attacks**：它在 redirect 前记录 expected `issuer`，并在 redeem code 前验证 `iss` authorization-response 参数（RFC 9207）。仅靠 PKCE 不能阻止 mix-up，因为 client 会把自己的 `code_verifier` 交给被引导到的任何 token endpoint。

OAuth 2.1 draft 是底层；RFC 8414/7591/8707/9728/9207 + RFC 7636 + CIMD 是认证表面；MCP spec 是 profile。

### IdP 能力矩阵

不是每个 IdP 都支持完整 MCP profile。下表记录的是截至 2025-11-25 spec 的事实能力声明。它是*部署门槛*，不是推荐清单。

CIMD 随 2025-11-25 spec 发布，而底层 OAuth draft 直到 2025 年 10 月才被采用，所以 vendor support 仍在到来。请把下面的 “CIMD” 理解为“截至今天的状态，请在你的 tenant 中验证”，而不是永久结论。

| IdP 类别 | AS metadata (8414/OIDC) | CIMD | RFC 7591 DCR | RFC 8707 resource | RFC 7636 S256 PKCE | 说明 |
|---|---|---|---|---|---|---|
| 自托管（Keycloak） | 是 | 发展中 | 是 | 是（24.x 起） | 是 | 本课 MCP profile 的 reference IdP；DCR path 端到端完整，CIMD 跟进新 spec。 |
| 企业 SSO（Microsoft Entra ID） | 是 | 发展中 | 是（高级 tier） | 是 | 是 | DCR availability 随 tenant tier 不同；部署前在目标 tenant 验证。 |
| 企业 SSO（Okta） | 是 | 发展中 | 是（Okta CIC / Auth0） | 是 | 是 | Auth0（现在是 Okta CIC）支持 DCR；classic Okta orgs 需要 admin pre-registration。 |
| 社交登录 IdPs（通用） | 不一 | 否 | 很少 | 很少 | 是 | 大多数 social IdPs 把 clients 当作 static partners；没有 self-service enrollment。只把它用作 identity source，在上层叠加自己的 MCP-aware authorization server。 |
| 自定义 / 自研 | 取决于实现 | 取决于实现 | 取决于实现 | 取决于实现 | 取决于实现 | 如果自己实现，就实现完整 profile 并优先 CIMD。跳过 PKCE 或 audience binding 会破坏 MCP auth contract。 |

deployment manifest 的拒绝规则：如果所选 IdP 没有在 `code_challenge_methods_supported` 中列出 `S256`，MCP server 拒绝启动，因为 PKCE 没有降级模式。客户端接入是较软的门槛：你需要*一种*可用路径（预注册的 `client_id`、`client_id_metadata_document_supported: true` 或 `registration_endpoint`）。单独缺少 DCR 不再触发拒绝，因为 CIMD 或 pre-registration 可以覆盖。

### JWKS 刷新模式（AS rotate，resource server refresh）

把两个动词分开，因为混淆它们是真实生产 bug：

- **Rotate** 是 *authorization server* 做的事：mint 新 signing key，把它发布进 JWKS，并稍后 retire 旧 key。resource server 与此无关，也不能做这件事，因为它没有 IdP 的 private keys。
- **Refresh** 是 *resource server* 做的事：重新 `GET` 已发布 JWKS 到缓存中。这是 resource server 唯一会执行的 JWKS 动作。

生产故障模式是 stale cache。用 scheduled refresh job 加 key-value cache 解决。resource server 运行一个 job（cron、timer，或运行时提供的任何机制），按固定间隔获取 `<issuer>/.well-known/jwks.json` 并覆盖 `cache[issuer] = {keys, fetched_at}`。validator 从该 cache 读取。若 token 的 `kid` 不在 cache 中，则触发**一次**同步 refresh 作为 fallback，然后重新检查。这同时处理两种情况：scheduled refresh，以及在 key-overlap windows 中，下一次 scheduled refresh 之前，一个由全新 key 签名的 token 先到达。

fallback **必须是 re-fetch，绝不是 rotate**。如果把 cache-miss path 接到 rotate-and-mint，会破坏两件事：（1）mint 出的新 key 的 `kid` *仍然* 不匹配 token，所以 lookup 还是失败；（2）攻击者用随机 `kid` values 喷 token，会强迫系统无限创建 keys，形成自我 DoS。re-fetch 是幂等的，所以 bogus `kid` 最多浪费一次 fetch。

cache 形态：

```json
{
  "https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

同时有两个 keys 是稳定状态。authorization servers 先引入下一个 key（`k_2026_04`），再 retire 前一个（`k_2026_03`），因此旧 key 签发的 tokens 在过期前仍有效。cache 持有并集；validator 通过 `kid` 选择。

### 验证例程

MCP server 在 dispatch 任何 tool 前运行 validation。`code/main.py` 使用的形态：

```python
result = server.validate(bearer_token, required_scope="mcp:tools.invoke")
if not result["valid"]:
    return {"status": result["status"], "WWW-Authenticate": result["www_authenticate"]}
```

`validate` 解码 JWT，从 JWKS cache 解析 signing key（miss 时 refresh 一次），验证 signature，然后检查 `iss` 是否在 allow-list 中、`aud` 是否匹配此 server 的 canonical resource、`exp` 和 required scope，并在第一个失败处返回 `WWW-Authenticate` challenge。把它作为 resource server 上的单一 routine，意味着每个入口（每次 tool call、每种 transport）都会经过相同检查；不存在不验证就能到达 tool 的路径。

### Audience replay 走查（access-token privilege restriction）

Server A（`notes.example.com`）和 Server B（`tasks.example.com`）都注册到同一个 authorization server。Server A 被攻陷。攻击者拿到用户的 notes token，并 replay 到 Server B。

Server B 的 validator：

1. Decode JWT，通过 `kid` 获取 JWKS，验证 signature。
2. 检查 `iss` 是否在其 protected-resource metadata 的 `authorization_servers` 中。（通过：同一个 IdP。）
3. 检查 `aud == "https://tasks.example.com"`。（失败：token 的 `aud` 是 `https://notes.example.com`。）
4. 返回 401，带 `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch", resource_metadata="https://tasks.example.com/.well-known/oauth-protected-resource"`。

audience claim 是协议层抵御该攻击的唯一防御。为了性能跳过它是最常见的生产错误；validator 必须在每个请求上运行，而不只是 session start。spec 称之为 **access-token privilege restriction**：MCP server `MUST` 拒绝任何 audience 中没有命名自己的 token。

> **命名说明。** spec 将 *confused deputy* 这个术语保留给一个相关但不同的问题：MCP server 作为第三方 API 的 OAuth **proxy**，使用 static client ID，并在没有获得 per-client user consent 的情况下转发 token。Audience binding 修复上面的 replay；confused-deputy 的修复是 per-client consent **加上**永远不要把 inbound token pass through 到 upstream APIs（MCP server `MUST` 获取自己单独的 upstream token）。

### Mix-up attacks（server 无法提供的 client-side defense）

client 在生命周期中会与许多 authorization servers 对话。恶意 AS 可以试图让 client 把诚实 AS 的 authorization code redeem 到攻击者的 token endpoint。Audience binding 在这里无济于事，因为攻击发生在任何 token 出现之前。防御位于 client（RFC 9207）：

1. redirect 前，client 从已验证 AS metadata 记录 expected `issuer`。
2. authorization response 上，client 在把 code 发送到任何地方之前，将返回的 `iss` 参数与已记录 issuer 比较（简单字符串比较，不做 normalization）。
3. 不匹配（或 AS 已声明 `authorization_response_iss_parameter_supported` 但 `iss` 缺失）→ 拒绝，并且甚至不要展示 `error` fields。

仅靠 PKCE 不能阻止 mix-up，因为 client 会把自己的 `code_verifier` 交给被引导到的任意 token endpoint。这就是 spec 为什么把 issuer 与 PKCE verifier 和 `state` 一起按请求记录。

### 失败模式

- **Stale JWKS。** AS rotate key 后，validator 拒绝有效 tokens。修复是上面的 cron-refresh + cache-miss-refetch pattern。绝不要缓存 JWKS 却没有 refresh job。
- **Rotate-as-fall-back。** 把 cache-miss path 接到 rotate-and-mint 而不是 re-fetch 是真实 bug：它永远生成不出缺失的 `kid`，还会把 attacker-controlled `kid` values 变成 key-creation DoS。fallback 必须是幂等的 `refresh-jwks`。
- **Missing `aud` claim。** 某些 IdPs 默认会省略 `aud`，除非 token request 中存在 `resource`。validator 必须拒绝缺失 `aud` 的 tokens，不能把缺失视为 wildcard。
- **Mix-up via missing `iss` check。** client 如果没有把 RFC 9207 `iss` authorization-response 参数与 redirect 前记录的 issuer 校验，就可能被引导到攻击者 token endpoint 去 redeem 诚实 AS 的 code。这是 client-side failure；resource server 无法补偿。
- **Scope upgrade race。** 同一用户的两个并发 step-up flows 都可能成功，并产生两个 scope 不同的 access tokens。validator 必须使用请求中提交的 token，而不是查找“用户当前 scope”，后者会创建 TOCTOU window。
- **Registration token theft。** 泄露的 `registration_access_token` 允许攻击者重写 redirect URIs。静态存储时 hash；每次更新要求 client 出示明文；怀疑泄露时 rotate。
- **`iss` not pinned。** 接受任意 `iss` 的 validator 允许攻击者搭建自己的 authorization server，为目标 audience 注册 client 并签发 tokens。protected-resource metadata 的 `authorization_servers` 列表就是 allow-list；必须执行。

## 实际使用

`code/main.py` 用 stdlib Python 和三个角色走完整生产 flow：`AuthorizationServer`、`ResourceServer` 和 `Client`。流程：

1. Authorization server 在 `/.well-known/oauth-authorization-server` 发布 RFC 8414 metadata。
2. MCP client 调用 metadata endpoint，并检查其接入选项（CIMD 的 `client_id_metadata_document_supported`、DCR 的 `registration_endpoint`）和 `S256` PKCE support。
3. walkthrough 走 DCR fallback path：client post 到 `/register`（RFC 7591）并收到 `client_id`。（CIMD client 会改为出示自己的 HTTPS `client_id` URL，并跳过此步骤。）
4. MCP client 运行受 PKCE 保护的 authorization code flow（RFC 7636），并带上 `resource` indicator（RFC 8707）。
5. MCP client 用 `Authorization: Bearer ...` 调用 MCP server 上的一个 tool。
6. MCP server 运行 `validate`，从 JWKS cache 解析 signing key。
7. IdP rotate 一个 key；scheduled refresh 将 JWKS 重新拉入 cache。
8. 下一次调用不重启也能用 refreshed keys 验证，之前的 token 在 overlap window 中仍可验证。
9. 针对另一个 MCP resource 的 audience-replay attempt 收到带 `audience mismatch` 和 `resource_metadata` pointer 的 401。

这里的 JWT 使用 HS256 和 shared secret（这样本课只靠 stdlib 就能运行）。生产使用 RS256 或 EdDSA，并配合上面的 JWKS pattern；validation logic 其余部分相同。因为 IdP 和 resource server 在同一进程中，`refresh_jwks` 直接读取 authorization server 的 key list；在线路上它是对 `jwks_uri` 的 HTTP `GET`。

## 交付成果

本课产出 `outputs/skill-mcp-auth.md`。给定 MCP server config 和 IdP capability set，该 skill 会给出需要搭建的认证表面：protected-resource metadata、要使用的接入路径（CIMD、pre-registration 或 DCR fallback）、JWKS refresh schedule、scope mapping，以及 IdP 不支持完整 RFC profile 时要应用的 refusal rules。

## 练习

1. 运行 `code/main.py`。追踪 flow。注意 IdP 如何在第 6 步 rotate key，scheduled `refresh_jwks` 如何重新拉取已发布集合，以及 old token（overlap window）和 fresh token 都如何无需重启即可验证。

2. 向 protected-resource metadata 的 `authorization_servers` 列表添加一个新 IdP。签发一个由新 IdP 签名的 token，确认 validator 接受它。签发一个未列出 IdP 签名的 token，确认 validator 以 `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"` 拒绝。

3. 向 `register_client` 添加一个 rate-limit check，并确保它在 registrar 接受请求之前运行。使用一个以 IP 为 key 的小 dict 保存每个 source IP 的 token-bucket。

4. 阅读 RFC 7591，找出本课 `/register` handler 没有验证的两个字段。添加验证。（提示：`software_statement` 和 `redirect_uris` URI scheme。）

5. 添加 Client ID Metadata Document path。提供一个 `client_id` 等于自身 URL 的 `client.json`，并让 authorization server fetch 并验证它（若 `client_id` ≠ URL 则拒绝）。确认 CIMD client 无需 `register_client` call 即可完成接入。

6. 证明 DoS 修复。向 validator 发送一个带随机 `kid` 的 token，确认 `refresh_jwks` 最多运行一次，且 authorization server 的 key count 不会增长。然后故意把 fallback 重新接到 rotate-and-mint，观察每个 bogus token 都让 key count 增加；之后恢复 re-fetch。

7. 实现 mix-up section 中 client-side RFC 9207 `iss` check：authorization request 前记录 expected issuer，然后拒绝 `iss` 不匹配的 authorization response。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| ASM | “OAuth metadata document” | RFC 8414 `/.well-known/oauth-authorization-server` JSON |
| CIMD | “Client metadata URL” | Client ID Metadata Document：用作 `client_id` 的 HTTPS URL；AS 拉取 JSON。自 2025-11-25 起为推荐默认值 |
| DCR | “自助客户端注册” | RFC 7591 `POST /register` flow；在 2025-11-25 被降级为 `MAY` fallback |
| JWKS | “JWT 验证公钥” | JSON Web Key Set，从 `jwks_uri` 获取，并以 `kid` 索引 |
| Rotate vs refresh | “更新密钥” | *Rotate* = AS mint/retire signing keys；*refresh* = resource server 重新获取已发布集合。resource servers 只会 refresh |
| Resource indicator | “受众参数” | RFC 8707 `resource` 参数，将 token 固定到一个 server |
| `aud` claim | “受众” | validator 与 canonical resource URL 比较的 JWT claim |
| Audience replay | “令牌重放” | 签发给 Server A 的 token 被提交给 Server B；通过 audience validation 防御（spec：access-token privilege restriction） |
| Confused deputy | “代理令牌误用” | 使用 static client ID 的 MCP proxy 在没有 per-client consent 时转发 token；不同于 audience replay |
| Mix-up attack | “错误的 token endpoint” | client 被引导到攻击者 endpoint 去 redeem 诚实 AS 的 code；通过 client-side RFC 9207 `iss` 防御 |
| `iss` allow-list | “可信授权服务器” | protected-resource metadata 的 `authorization_servers` 命名的集合 |
| `resource_metadata` | “PRM 文档位置” | 401/403 上命名 RFC 9728 metadata URL 的 `WWW-Authenticate` 参数 |
| Public client | “原生或浏览器客户端” | 没有 `client_secret` 的 OAuth client；PKCE 作补偿 |
| `WWW-Authenticate` | “401/403 响应头” | 携带驱动 client recovery 的 `Bearer error=...` 指令 |

## 延伸阅读

- [MCP — Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization) — 本课实现的 MCP auth profile
- [MCP blog — One Year of MCP: November 2025 Spec Release](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 2025-11-25 中的变化（CIMD、XAA、DCR 降级）
- [Aaron Parecki — Client Registration in the November 2025 MCP Authorization Spec](https://aaronparecki.com/2025/11/25/1/mcp-authorization-spec-update) — 优先 CIMD 而不是 DCR 的理由
- [OAuth Client ID Metadata Document (draft-ietf-oauth-client-id-metadata-document-00)](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-client-id-metadata-document-00) — CIMD
- [RFC 8414 — OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414) — discovery contract
- [RFC 7591 — OAuth 2.0 Dynamic Client Registration Protocol](https://datatracker.ietf.org/doc/html/rfc7591) — DCR（fallback path）
- [RFC 7636 — Proof Key for Code Exchange (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) — public-client proof-of-possession
- [RFC 8707 — Resource Indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — audience pinning
- [RFC 9728 — OAuth 2.0 Protected Resource Metadata](https://datatracker.ietf.org/doc/html/rfc9728) — resource server discovery
- [RFC 9207 — OAuth 2.0 Authorization Server Issuer Identification](https://datatracker.ietf.org/doc/html/rfc9207) — 防御 mix-up attacks 的 `iss` 参数
- [OAuth 2.1 draft](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1) — 合并后的 OAuth 底层规范
