# Async Tasks (SEP-1686) — Call-Now, Fetch-Later for Long-Running Work

> 真实 agent work 需要数分钟到数小时：CI runs、deep-research synthesis、batch exports。Synchronous tool calls 会断开连接、超时，或阻塞 UI。SEP-1686 在 2025-11-25 合并，加入 Tasks primitive：任何 request 都可以被增强为 task，result 可以稍后 fetch，或通过 state notifications stream。Drift-risk note：Tasks 在 H1 2026 仍是 experimental；SDK surface 仍在围绕 spec 设计。

**类型:** 构建
**语言:** Python (stdlib, async task state machine)
**先修:** Phase 13 · 07 (MCP server), Phase 13 · 09 (transports)
**时间:** ~75 分钟

## 学习目标

- 识别何时应把 tool 从 synchronous 提升为 task-augmented (server-side work >30 秒)。
- 走读 task lifecycle：`working` → `input_required` → `completed` / `failed` / `cancelled`。
- 持久化 task state，避免 crashes 丢失 in-flight work。
- 正确 poll `tasks/status` 并 fetch `tasks/result`。

## 要解决的问题

一个 `generate_report` tool 会运行多分钟 extraction pipeline。在 synchronous model 下有几种选择：

1. 让 connection 保持打开三分钟。Remote transports 会断开；clients 会超时；UIs 会冻结。
2. 立即返回 placeholder；要求 client poll 一个 custom endpoint。破坏 MCP uniformity。
3. Fire-and-forget；没有 result。

都不好。SEP-1686 添加了第四种：task augmentation。任何 request (通常是 `tools/call`) 都可以被标记为 task。server 立即返回 task id。client poll `tasks/status`，完成后 fetch `tasks/result`。Server-side state 可以跨 restarts 存活。

## 核心概念

### Task augmentation

request 通过设置 `params._meta.task.required: true` (或 `optional: true`，由 server 决定) 变成 task。server 立即响应：

```json
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "_meta": {
      "task": {
        "id": "tsk_9f7b...",
        "state": "working",
        "ttl": 900000
      }
    }
  }
}
```

`ttl` 是 server 对保留 state 的承诺；ttl 之后 task result 会被丢弃。

### Per-tool opt-in

Tool annotations 可以声明 task support：

- `taskSupport: "forbidden"` — 这个 tool 总是同步运行。适合 fast tools。
- `taskSupport: "optional"` — client 可以请求 task-augmentation。
- `taskSupport: "required"` — client MUST 使用 task augmentation。

`generate_report` tool 应为 `required`。`notes_search` tool 应为 `forbidden`。

### States

```text
working  -> input_required -> working  (loop via elicitation)
working  -> completed
working  -> failed
working  -> cancelled
```

State machine 是 append-only：一旦进入 `completed`、`failed` 或 `cancelled`，task 就是 terminal。

### Methods

- `tasks/status {taskId}` — 返回 current state 和 progress hint。
- `tasks/result {taskId}` — 如果尚未完成则 block 或返回 404。
- `tasks/cancel {taskId}` — idempotent；terminal states 会忽略。
- `tasks/list` — 可选；枚举 active 和 recently-completed tasks。

### Streaming state changes

server 支持时，client 可以订阅 state notifications：

```text
server -> notifications/tasks/updated {taskId, state, progress?}
```

stream 而不是 poll 的 clients 会得到更好的 UX。Polling 始终作为 minimal surface 被支持。

### Durable state

spec 要求声明 task support 的 servers 持久化 state。crash 不应丢失 ttl 内的 completed results。stores 可以从 SQLite 到 Redis 到 filesystem。本课 Lesson 13 harness 使用 filesystem。

### Cancellation semantics

`tasks/cancel` 是 idempotent。如果 task 正在 mid-execution，server 会尝试停止它 (检查 executor-cooperative cancellation)。如果已经 terminal，则 request 是 no-op。

### Crash recovery

server process 重启时：

1. 加载所有 persisted task states。
2. 把 process 死亡时仍处于 `working` 的 tasks 标记为 `failed`，error 为 `CRASH_RECOVERY`。
3. 在 ttl 内保留 `completed` / `failed` / `cancelled`。

### Async tasks plus sampling

task 本身可以调用 `sampling/createMessage`。这就是 long-running research tasks 的工作方式：server 的 task thread 按需 sample client 的 model，同时 client UI 显示 task 为 `working`，并展示周期性 progress updates。

### Why this is experimental

SEP-1686 在 2025-11-25 发布，但更广泛 roadmap 指出了三个 open issues：durable subscription primitives、subtasks (parent-child task relationships) 和 result-TTL standardization。预计 spec 会在 2026 年继续演进。Production code 应只把 Tasks 的 common case 当作稳定，并对未来 SDK changes for subtasks 做 guard。

## 实际使用

`code/main.py` 实现了 durable task store (filesystem-backed) 和一个在 background thread 中运行的 `generate_report` tool。clients 调用该 tool 后立即得到 task id，在 worker 更新 progress 时 poll `tasks/status`，完成后 fetch `tasks/result`。Cancellation 可用；通过杀掉 worker thread 并 reload state 来模拟 crash recovery。

需要观察的点：

- Task state JSON 持久化到 `/tmp/lesson-13-tasks/<id>.json`。
- Worker thread 更新 `progress` field；poll 时可以看到它推进。
- client-side cancellation 设置一个 event；worker 检查它并提前退出。
- "crash" 后 state reload 会把 in-flight task 标记为 `failed`，并带上 `CRASH_RECOVERY`。

## 交付成果

本课产出 `outputs/skill-task-store-designer.md`。给定一个 long-running tool (research, build, export)，该 skill 会设计 task store (state shape, ttl, durability)，选择正确的 taskSupport flag，并草拟 progress notifications。

## 练习

1. 运行 `code/main.py`。启动一个 `generate_report` task，poll status，然后 fetch result。

2. 在 mid-run 添加一次 `tasks/cancel` call。验证 worker 会遵守它，state 变为 `cancelled`。

3. 模拟 crash recovery：杀掉 worker thread，重启 loader，并观察 `CRASH_RECOVERY` failure mode。

4. 把 store 扩展到 SQLite。durability 收益相同；query options 变多 (列出来自 session X 的所有 tasks)。

5. 阅读 2026 年 MCP roadmap post。找出一个最可能在下一年影响 SDK API design 的 Tasks-related open issue。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Task | "Long-running tool call" | 用 `_meta.task` 增强以 async execution 的 request |
| SEP-1686 | "Tasks spec" | 在 2025-11-25 添加 Tasks 的 Spec Evolution Proposal |
| `_meta.task` | "Task envelope" | 包含 id、state、ttl 的 per-request metadata |
| taskSupport | "Tool flag" | 每个 tool 的 `forbidden` / `optional` / `required` |
| `tasks/status` | "Poll method" | 获取 current state 和 optional progress hint |
| `tasks/result` | "Fetch result" | 返回 completed payload；未完成时返回 404 |
| `tasks/cancel` | "Stop it" | Idempotent cancellation request |
| ttl | "Retention budget" | server 承诺保留 task state 的毫秒数 |
| `notifications/tasks/updated` | "State push" | server-initiated state-change event |
| Durable store | "Crash-safe state" | Filesystem / SQLite / Redis persistence layer |

## 延伸阅读

- [MCP — GitHub SEP-1686 issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1686) — originating proposal 和完整 discussion
- [WorkOS — MCP async tasks for AI agent workflows](https://workos.com/blog/mcp-async-tasks-ai-agent-workflows) — 带 rationale 的 design walkthrough
- [DeepWiki — MCP task system and async operations](https://deepwiki.com/modelcontextprotocol/modelcontextprotocol/2.7-task-system-and-async-operations) — mechanics 和 state machine
- [FastMCP — Tasks](https://gofastmcp.com/servers/tasks) — SDK-level task implementation patterns
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 包含 subtasks 的 open issues 和 2026 priorities
