# Claude Agent SDK：Subagents 与 Session Store

> Claude Agent SDK 是 Claude Code harness 的 library 形态。内置 tools、用于 context isolation 的 subagents、hooks、W3C trace propagation、session store parity。Claude Managed Agents 是面向 long-running async work 的 hosted alternative。

**类型:** 学习 + 构建
**语言:** Python (stdlib)
**先修:** Phase 14 · 01 (Agent Loop), Phase 14 · 10 (Skill Libraries)
**时间:** ~75 分钟

## 学习目标

- 解释 Anthropic Client SDK（raw API）与 Claude Agent SDK（harness shape）的区别。
- 描述 subagents -- parallelization 和 context isolation -- 以及何时使用它们。
- 说出 Python SDK 的 session store surface（`append`、`load`、`list_sessions`、`delete`、`list_subkeys`）以及 `--session-mirror` 的作用。
- 实现一个 stdlib harness，包含 built-in tools、带 isolated context 的 subagent spawning、lifecycle hooks 和 session store。

## 要解决的问题

Raw LLM API 给你一次 round-trip。生产 agent 需要 tool execution、MCP servers、lifecycle hooks、subagent spawning、session persistence、trace propagation。Claude Agent SDK 将这种形态作为 library 发出来 -- 也就是 Claude Code 使用的同一个 harness，但暴露给 custom agents。

## 核心概念

### Client SDK vs Agent SDK

- **Client SDK (`anthropic`)。** Raw Messages API。Loop、tools、state 都由你拥有。
- **Agent SDK (`claude-agent-sdk`)。** Built-in tool execution、MCP connections、hooks、subagent spawning、session store。把 Claude Code loop 作为 library。

### Built-in tools

SDK 开箱提供 10+ tools：file read/write、shell、grep、glob、web fetch 等。Custom tools 通过标准 tool-schema interface 注册。

### Subagents

Anthropic 文档记录的两个用途：

1. **Parallelization。** 并发运行独立工作。"Find the test file for each of these 20 modules" 就是 20 个 parallel subagent tasks。
2. **Context isolation。** Subagents 使用自己的 context window；只有 results 回到 orchestrator。Orchestrator 的 budget 得以保留。

Python SDK 的近期新增项：`list_subagents()`、`get_subagent_messages()`，用于读取 subagent transcripts。

### Session store

与 TypeScript 的协议 parity：

- `append(session_id, message)` -- 添加一个 turn。
- `load(session_id)` -- 恢复 conversation。
- `list_sessions()` -- 枚举。
- `delete(session_id)` -- 删除，并 cascade 到 subagent sessions。
- `list_subkeys(session_id)` -- 列出 subagent keys。

`--session-mirror`（CLI flag）会在 transcript stream 时把它 mirror 到外部文件，便于 debugging。

### Hooks

你可以注册的 lifecycle hooks：

- `PreToolUse`、`PostToolUse` -- gate 或 audit tool calls。
- `SessionStart`、`SessionEnd` -- set up 和 tear down。
- `UserPromptSubmit` -- 在模型看到 user input 前对它采取动作。
- `PreCompact` -- 在 context compaction 前运行。
- `Stop` -- agent exit 时 cleanup。
- `Notification` -- side-channel alerts。

Hooks 是 pro-workflow（Phase 14 curriculum reference）以及类似系统添加 cross-cutting behavior 的方式。

### W3C trace context

Caller 上 active 的 OTel spans 会通过 W3C trace context headers 传播到 CLI subprocess。整个 multi-process trace 会在你的 backend 中显示为一条 trace。

### Claude Managed Agents

Hosted alternative（beta header `managed-agents-2026-04-01`）。Long-running async work、built-in prompt caching、built-in compaction。用 managed infrastructure 换取更少控制。

### 这个模式容易出错的地方

- **Subagent over-spawn。** 为 100 个 tiny tasks 启动 100 个 subagents。Overhead 占主导。改为 batch。
- **Hook creep。** 每个团队都添加 hooks；startup time 膨胀。每季度 review hooks。
- **Session bloat。** Sessions 累积；size 增长。使用 `list_sessions` + expiry policy。

## 动手实现

`code/main.py` 用 stdlib 实现 SDK 形态：

- `Tool`、`ToolRegistry`，包含 built-in `read_file`、`write_file`、`list_dir`。
- `Subagent` -- private context、isolated run、results returned。
- `SessionStore` -- append、load、list、delete、list_subkeys。
- `Hooks` -- `pre_tool_use`、`post_tool_use`、`session_start`、`session_end`。
- 一个 demo：main agent 并发启动 3 个 subagents（各自 isolated），聚合 results，持久化 session。

运行：

```text
python3 code/main.py
```

Trace 会展示 subagent context isolation（orchestrator context size 保持有界）、hook execution 和 session persistence。

## 实际使用

- **Claude Agent SDK** 用于想要 Claude Code harness shape 的 Claude-first products。
- **Claude Managed Agents** 用于 hosted long-running async work。
- **OpenAI Agents SDK**（Lesson 16）用于 OpenAI-first counterparts。
- **LangGraph + custom tools** 如果你想要 graph-shaped state machine。

## 交付成果

`outputs/skill-claude-agent-scaffold.md` 会 scaffold 一个 Claude Agent SDK app，包含 subagents、hooks、session store、MCP server attachment 和 W3C trace propagation。

## 练习

1. 添加一个 subagent spawner，将 20 个 tasks 批成 5 个 parallel subagents 一组。测量 orchestrator context size 与 one-per-task 的差异。
2. 实现一个 `PreToolUse` hook，对 `write_file` calls 做 rate limit（每 session 每分钟 5 次）。跟踪 behavior。
3. 接好 `list_subkeys` 来渲染 subagent tree。Deep nesting 看起来是什么样？
4. 把 toy 移植到真实的 `claude-agent-sdk` Python package。Tool registration 有什么变化？
5. 阅读 Claude Managed Agents docs。什么时候你会从 self-hosted 切到 managed？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Agent SDK | "Claude Code as a library" | Harness shape：tools、MCP、hooks、subagents、session store |
| Subagent | "Child agent" | 独立 context，自己的 budget；results 向上冒泡 |
| Session store | "Conversation DB" | 持久化、加载、列出、删除 turns，并 cascade subagents |
| Hook | "Lifecycle callback" | Pre/post tool、session、prompt submit、compact、stop |
| W3C trace context | "Cross-process trace" | Parent span 传播到 CLI subprocess |
| Managed Agents | "Hosted harness" | Anthropic-hosted long-running async work |
| `--session-mirror` | "Transcript mirror" | Session turns stream 时写到外部文件 |
| MCP server | "Tool surface" | 附加到 agent 的外部 tool/resource source |

## 延伸阅读

- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) -- Claude Code 的 library 形态
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) -- production patterns
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) -- hosted alternative
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) -- counterpart
