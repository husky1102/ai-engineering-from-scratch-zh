# Capstone：构建完整 Tool Ecosystem

> Phase 13 教过每个部件。这个 capstone 把它们接成一个 production-shaped system：带 tools + resources + prompts + tasks + UI 的 MCP server，边缘 OAuth 2.1，一个 RBAC gateway，一个 multi-server client，一个 A2A sub-agent call，进入 collector 的 OTel tracing，CI 中的 tool-poisoning detection，以及 AGENTS.md + SKILL.md bundle。结束时你可以为每个架构选择辩护。

**类型:** Build
**语言:** Python（stdlib，end-to-end ecosystem harness）
**先修:** Phase 13 · 01 through 21
**时间:** ~120 分钟

## 学习目标

- 组合一个暴露 tools、resources、prompts 以及带 `ui://` app 的 task 的 MCP server。
- 用 OAuth 2.1 gateway 前置 server，并强制 RBAC 和 pinned hashes。
- 编写一个 multi-server client，用 OTel GenAI attributes 做端到端 tracing。
- 把 workload 的一部分委托给 A2A sub-agent；验证 opacity 得到保留。
- 用 AGENTS.md + SKILL.md 打包整个 stack，让其他 agents 可以驱动它。

## 要解决的问题

交付“research and report”系统：

- User asks：“summarize the three most-cited 2026 arXiv papers on agent protocols.”
- System：通过 MCP 搜索 arXiv；经由 A2A 把 paper summarization 委托给 specialized writer agent；聚合结果；把 interactive report 渲染成 MCP Apps `ui://` resource；把每一步记录到 OTel。

Phase 13 的所有 primitives 都会出现。这不是玩具，Anthropic（Claude Research product）、OpenAI（GPTs with Apps SDK）和第三方在 2026 年发布的 production research-assistant systems 就是这个形状。

## 核心概念

### 架构

```text
[user] -> [client] -> [gateway (OAuth 2.1 + RBAC)] -> [research MCP server]
                                                      |
                                                      +- MCP tool: arxiv_search (pure)
                                                      +- MCP resource: notes://recent
                                                      +- MCP prompt: /research_topic
                                                      +- MCP task: generate_report (long)
                                                      +- MCP Apps UI: ui://report/current
                                                      +- A2A call: writer-agent (tasks/send)
                                                      |
                                                      +- OTel GenAI spans
```

### Trace hierarchy

```text
agent.invoke_agent
 ├── llm.chat (kick off)
 ├── mcp.call -> tools/call arxiv_search
 ├── mcp.call -> resources/read notes://recent
 ├── mcp.call -> prompts/get research_topic
 ├── a2a.tasks/send -> writer-agent
 │    └── task transitions (opaque internals)
 ├── mcp.call -> tools/call generate_report (task-augmented)
 │    └── tasks/status polling
 │    └── tasks/result (completed, returns ui:// resource)
 └── llm.chat (final synthesis)
```

一个 trace id。每个 span 都有正确的 `gen_ai.*` attributes。

### Security posture

- OAuth 2.1 + PKCE，resource indicator 把 audience pin 到 gateway。
- Gateway 持有 upstream credentials；user 永远看不到它们。
- RBAC：`alice` 有 `research:read`、`research:write`，可以调用所有 tools。`bob` 有 `research:read`，不能调用 `generate_report`。
- Pinned description manifest：丢弃任何 tool hashes 变化的 server。
- Rule of Two audit：没有 tool 同时组合 untrusted input、sensitive data 和 consequential action。

### Rendering

最终 `generate_report` task 返回 content blocks 加一个 `ui://report/current` resource。Client 的 host（Claude Desktop 等）在 sandbox iframe 中渲染 interactive dashboard。Dashboard 包含排序后的 paper list、citation counts，以及一个按钮，用户点击任意 paper 时调用 `host.callTool('summarize_paper', {arxiv_id})`。

### Packaging

整套系统发布为：

```text
research-system/
  AGENTS.md                     # project conventions
  skills/
    run-research/
      SKILL.md                  # the top-level workflow
  servers/
    research-mcp/               # the MCP server
      pyproject.toml
      src/
  agents/
    writer/                     # the A2A agent
  gateway/
    config.yaml                 # RBAC + pinned manifest
```

用户用 `docker compose up` 部署。Claude Code、Cursor、Codex 和 opencode 用户可以通过 invoke `run-research` skill 来驱动系统。

### Phase 13 每课贡献了什么

| Lesson | Capstone 使用了什么 |
|--------|------------------------|
| 01-05 | Tool interface、provider-portability、parallel calls、schemas、linting |
| 06-10 | MCP primitives、server、client、transports、resources + prompts |
| 11-14 | Sampling、roots + elicitation、async tasks、`ui://` apps |
| 15-17 | Tool poisoning、OAuth 2.1、gateway + registry |
| 18 | A2A sub-agent delegation |
| 19 | OTel GenAI tracing |
| 20 | Routing gateway for the LLM layer |
| 21 | SKILL.md + AGENTS.md packaging |

## 实际使用

`code/main.py` 把之前 lessons 的 patterns 缝成一个 runnable demo。全 stdlib、全 in-process，所以你可以端到端阅读。它为 research-and-report scenario 运行完整 flow：与 gateway handshake、模拟 OAuth 2.1、merged tools/list、把 generate_report 作为 task、A2A call 到 writer、返回 ui:// resource、emit OTel spans。

重点看：

- 每个 hop 共享一个 trace id。
- Gateway policy 阻止第二个 user 写入。
- Task lifecycle 从 working → completed，并返回 text 与 ui:// content。
- A2A call 的 inner state 对 orchestrator opaque。
- AGENTS.md 和 SKILL.md 是另一个 agent 复现 workflow 所需的唯一 files。

## 交付成果

本课产出 `outputs/skill-ecosystem-blueprint.md`。给定一个 product need（research、summarization、automation），该 skill 会产出完整 architecture：哪些 MCP primitives、哪些 gateway controls、哪些 A2A calls、哪些 telemetry、哪些 packaging。

## 练习

1. 运行 `code/main.py`。注意 single trace id 以及 spans 如何嵌套。统计 demo 触及 Phase 13 中多少 primitives。

2. 扩展 demo：添加第二个 backend MCP server（例如 `bibliography`），并确认 gateway 把它的 tools merge 到同一 namespace。

3. 把 fake A2A writer agent 换成在 subprocess 上运行的真实 agent。使用 Lesson 19 harness。

4. 在 orchestrator 和 LLM 之间的 routing gateway 中加入 PII redaction step。确认 user query 中的 emails 被 scrub。

5. 为维护此系统的队友写一个 AGENTS.md。阅读应少于五分钟，并给出他们在 Cursor 或 Codex 中驱动 capstone 所需的一切。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Capstone | “Phase-13 integration demo” | 使用每个 primitive 的端到端系统 |
| Research and report | “The scenario” | Search、summarize、render pattern |
| Ecosystem | “All the pieces together” | Server + client + gateway + sub-agent + telemetry + package |
| Trace hierarchy | “Single trace id” | 每个 hop 的 span 共享 trace；通过 span ids 形成 parent-child |
| Gateway-issued token | “Transitive auth” | Client 只看到 gateway token；gateway 持有 upstream creds |
| Merged namespace | “All tools in one flat list” | Gateway 处 multi-server merge，collision 时加 prefix |
| Opacity boundary | “A2A call hides internals” | Sub-agent reasoning 对 orchestrator 不可见 |
| Three-layer stack | “AGENTS.md + SKILL.md + MCP” | Project context + workflow + tools |
| Defense-in-depth | “Multiple security layers” | Pinned hashes、OAuth、RBAC、Rule of Two、audit log |
| Spec compliance matrix | “What we ship that the spec requires” | 把 deliverables 映射到 2025-11-25 requirements 的 checklist |

## 延伸阅读

- [MCP — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — consolidated reference
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — protocol 的未来方向
- [a2a-protocol.org](https://a2a-protocol.org/latest/) — A2A v1.0 reference
- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — canonical tracing conventions
- [Anthropic — Claude Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview) — production agent runtime patterns
