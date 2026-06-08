# MCP Apps：通过 `ui://` 提供交互式 UI 资源

> 纯文本工具输出限制了 agent 能展示的内容。MCP Apps（SEP-1724，2026 年 1 月 26 日正式发布）让工具可以返回沙箱化的交互式 HTML，并内嵌渲染在 Claude Desktop、ChatGPT、Cursor、Goose 和 VS Code 中。仪表盘、表单、地图、3D 场景，都能通过同一个扩展实现。本课讲解 `ui://` 资源方案、`text/html;profile=mcp-app` MIME、iframe 沙箱 postMessage 协议，以及让服务器渲染 HTML 所带来的安全面。

**类型：** 构建
**语言：** Python（stdlib，UI 资源发射器）、HTML（示例 app）
**先修：** Phase 13 · 07（MCP server）、Phase 13 · 10（resources）
**时间：** 约 75 分钟

## 学习目标

- 从一次工具调用中返回 `ui://` 资源，并设置正确的 MIME 与元数据。
- 用 `_meta.ui.resourceUri`、`_meta.ui.csp` 和 `_meta.ui.permissions` 声明工具关联的 UI。
- 实现 iframe 沙箱 postMessage JSON-RPC，用于 UI 到 host 的通信。
- 应用能防御 UI 发起攻击的 CSP 与 permissions-policy 默认值。

## 要解决的问题

一个 2025 年时代的 `visualize_timeline` 工具可以返回“这里是按时间排序的 14 条笔记：……”。那只是一个段落。用户真正想要的是可交互的时间线。在 MCP Apps 之前，可选方案是客户端专属的小组件 API（Claude artifacts、OpenAI Custom GPT HTML），或者完全没有 UI。

MCP Apps（SEP-1724，2026 年 1 月 26 日发布）标准化了这份契约。工具结果包含一个 `resource`，其 URI 是 `ui://...`，MIME 是 `text/html;profile=mcp-app`。host 会在沙箱 iframe 中渲染它，并施加受限 CSP；除非显式授予，否则没有网络访问。iframe 内的 UI 通过一个很小的 postMessage JSON-RPC 方言向 host 发消息。

每个兼容客户端（Claude Desktop、ChatGPT、Goose、VS Code）都会以同样方式渲染同一个 `ui://` 资源。一个服务器、一个 HTML bundle、通用 UI。

## 核心概念

### `ui://` 资源方案

工具返回：

```json
{
  "content": [
    {"type": "text", "text": "Here is your notes timeline:"},
    {"type": "ui_resource", "uri": "ui://notes/timeline"}
  ],
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline",
      "csp": {
        "defaultSrc": "'self'",
        "scriptSrc": "'self' 'unsafe-inline'",
        "connectSrc": "'self'"
      },
      "permissions": []
    }
  }
}
```

随后 host 对 `ui://notes/timeline` URI 调用 `resources/read`，拿到：

```json
{
  "contents": [{
    "uri": "ui://notes/timeline",
    "mimeType": "text/html;profile=mcp-app",
    "text": "<!doctype html>..."
  }]
}
```

### Iframe 沙箱

host 在一个沙箱化 `<iframe>` 中渲染 HTML，配置包括：

- `sandbox="allow-scripts allow-same-origin"`（或按服务器声明使用更严格配置）
- 通过响应头应用服务器声明的 CSP。
- 没有来自 host origin 的 cookies，也没有 localStorage。
- 网络访问受 CSP 中 `connectSrc` 限制。

### postMessage 协议

iframe 通过 `window.postMessage` 与 host 通信。它使用一个很小的 JSON-RPC 2.0 方言：

始终把 `targetOrigin` 固定到对端的精确 origin；接收侧在处理任何 payload 之前，都要根据 allowlist 校验 `event.origin`。这个通道两侧都不要使用 `"*"`，因为消息体会携带工具调用和资源读取。

```js
// iframe to host  (pin to host origin)
window.parent.postMessage({
  jsonrpc: "2.0",
  id: 1,
  method: "host.callTool",
  params: { name: "notes_update", arguments: { id: "note-14", title: "..." } }
}, "https://host.example.com");

// host to iframe  (pin to iframe origin)
iframe.contentWindow.postMessage({
  jsonrpc: "2.0",
  id: 1,
  result: { content: [...] }
}, "https://iframe.example.com");

// receiver on both sides
window.addEventListener("message", (event) => {
  if (event.origin !== "https://expected-peer.example.com") return;
  // safe to process event.data
});
```

UI 可以调用的 host 侧方法包括：

- `host.callTool(name, arguments)`：调用服务器工具。
- `host.readResource(uri)`：读取 MCP 资源。
- `host.getPrompt(name, arguments)`：获取 prompt 模板。
- `host.close()`：关闭 UI。

每次调用仍然要走 MCP 协议，并继承服务器权限。

### 权限

`_meta.ui.permissions` 列表请求额外能力：

- `camera`：访问用户摄像头（用于扫描文档类 UI）。
- `microphone`：语音输入。
- `geolocation`：位置。
- `network:*`：比单独 `connectSrc` 更宽的网络访问。

每项权限都会在 UI 渲染前向用户展示一次提示。

### 安全风险

iframe 中的 HTML 仍然是 HTML。新增攻击面包括：

- **通过 UI 做 prompt injection。** 恶意服务器 UI 可以展示看起来像系统消息的文本来欺骗用户。host 渲染时应当明显区分服务器 UI 与 host UI。
- **通过 `connectSrc` 外传。** 如果 CSP 允许 `connect-src: *`，UI 可以把数据发到任何地方。默认值应当严格。
- **Clickjacking。** UI 覆盖 host chrome。host 必须阻止 z-index 操作并强制 opacity 规则。
- **窃取焦点。** UI 抢占键盘焦点并捕获下一条消息。host 必须拦截。

Phase 13 · 15 会在 MCP security 中深入讨论这些问题；本课只是先引入。

### `ui/initialize` 握手

iframe 加载后，会通过 postMessage 发送 `ui/initialize`：

```json
{"jsonrpc": "2.0", "id": 0, "method": "ui/initialize",
 "params": {"theme": "dark", "locale": "en-US", "sessionId": "..."}}
```

host 回应 capabilities 和 session token。UI 在之后每次 host 调用中都使用该 session token。

### AppRenderer / AppFrame SDK 原语

ext-apps SDK 暴露两个便捷原语：

- `AppRenderer`（服务器侧）：包装 React / Vue / Solid component，并以正确 MIME 和元数据发射 `ui://` 资源。
- `AppFrame`（客户端侧）：接收资源、挂载 iframe，并协调 postMessage。

你可以使用这些原语，也可以手写 HTML 和 JSON-RPC。

### 生态状态

MCP Apps 于 2026 年 1 月 26 日发布。截至 2026 年 4 月的客户端支持：

- **Claude Desktop。** 自 2026 年 1 月起完整支持。
- **ChatGPT。** 通过 Apps SDK 完整支持（底层是同一套 MCP Apps 协议）。
- **Cursor。** Beta；通过设置启用。
- **VS Code。** 仅 Insider builds。
- **Goose。** 完整支持。
- **Zed、Windsurf。** 已列入路线图。

生产中的服务器场景包括：仪表盘、地图可视化、数据表、图表构建器、沙箱 IDE 预览。

## 实际使用

`code/main.py` 扩展了 notes server，加入一个 `visualize_timeline` 工具。它返回 `ui://notes/timeline` 资源，并提供一个面向该 URI 的 `resources/read` handler，返回一个很小但完整的 HTML bundle，里面有 SVG 时间线。HTML 用 stdlib 模板生成，不需要构建系统。由于 stdlib 不能驱动浏览器，postMessage 以 JS 注释形式勾勒。

重点查看：

- 工具响应上的 `_meta.ui` 携带 resourceUri、CSP、permissions。
- HTML 在无网络访问下渲染；所有数据都内联。
- JS 通过 `window.parent.postMessage` 调用 `host.callTool`（在这个 stdlib demo 中只是文档化且不实际执行）。

## 交付成果

本课产出 `outputs/skill-mcp-apps-spec.md`。给定一个适合交互式 UI 的工具，该 skill 会生成完整 MCP Apps 契约：`ui://` URI、CSP、permissions、postMessage 入口点，以及安全检查清单。

## 练习

1. 运行 `code/main.py` 并检查发射出的 HTML。直接在浏览器中打开 HTML，确认 SVG 能渲染。然后草拟 UI 用来调用 `host.callTool("notes_update", ...)` 的 postMessage 契约。

2. 收紧 CSP：移除 `'unsafe-inline'`，改用基于 nonce 的 script policy。HTML 生成代码需要做哪些变化？

3. 添加第二个 UI 资源 `ui://notes/editor`，包含一个用于就地编辑笔记的表单。用户提交时，iframe 调用 `host.callTool("notes_update", ...)`。

4. 审计 UI 的攻击面。恶意服务器可以在哪里注入内容？iframe 沙箱能防住什么，不能防住什么？

5. 阅读 SEP-1724 spec，找出 MCP Apps SDK 中此玩具实现没有使用的一项能力。（提示：component-level state sync。）

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| MCP Apps | “交互式 UI 资源” | 2026-01-26 发布的 SEP-1724 扩展 |
| `ui://` | “App URI scheme” | UI bundle 的资源方案 |
| `text/html;profile=mcp-app` | “MIME” | MCP App HTML 的 content-type |
| Iframe sandbox | “渲染容器” | 结合 CSP 和权限对 UI 进行浏览器沙箱化 |
| postMessage JSON-RPC | “UI-to-host wire” | 用于 host 调用的轻量 JSON-RPC-over-postMessage 方言 |
| `_meta.ui` | “Tool-UI binding” | 把工具结果连接到 UI 资源的元数据 |
| CSP | “Content-Security-Policy” | 声明脚本、网络、样式等允许来源 |
| AppRenderer | “Server SDK primitive” | 将 framework component 转成 `ui://` 资源 |
| AppFrame | “Client SDK primitive” | 挂载 iframe 并协调 postMessage 的 helper |
| `ui/initialize` | “Handshake” | UI 发给 host 的第一条 postMessage |

## 延伸阅读

- [MCP ext-apps — GitHub](https://github.com/modelcontextprotocol/ext-apps) — reference implementation 和 SDK
- [MCP Apps specification 2026-01-26](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx) — 正式 spec 文档
- [MCP — Apps extension overview](https://modelcontextprotocol.io/extensions/apps/overview) — 高层文档
- [MCP blog — MCP Apps launch](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/) — 2026 年 1 月发布文章
- [MCP Apps API reference](https://apps.extensions.modelcontextprotocol.io/api/) — JSDoc 风格 SDK reference
