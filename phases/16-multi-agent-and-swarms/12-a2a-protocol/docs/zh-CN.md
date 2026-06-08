# A2A：Agent-to-Agent Protocol

> Google 在 2025 年 4 月发布 A2A；到 2026 年 4 月，规范位于 https://a2a-protocol.org/latest/specification/，并已有 150+ 组织支持它。A2A 是 MCP（Lesson 13）的横向补充：MCP 是纵向的（agent ↔ tools），A2A 是 peer-to-peer 的（agent ↔ agent）。它定义 Agent Cards（discovery）、带 artifacts 的 tasks（text、structured data、video）、不透明 task lifecycles 和 auth。生产系统越来越常把 MCP 与 A2A 搭配使用。Google Cloud 在 2025-2026 年期间把 A2A 支持集成进 Vertex AI Agent Builder。

**类型：** Learn + Build
**语言：** Python (stdlib, `http.server`, `json`)
**先修：** Phase 16 · 04 (Primitive Model)
**时间：** ~75 分钟

## 要解决的问题

你的 agent 需要调用另一个系统上的另一个 agent。怎么调用？你可以暴露一个 HTTP endpoint，定义一套 bespoke JSON schema，然后希望对方也说这套协议。每一对 agents 都会变成一次 custom integration。

A2A 是这类调用的通用 wire protocol。标准 discovery、标准 task model、标准 transport、标准 artifacts。就像 HTTP+REST，但把 agents 当成一等公民。

## 核心概念

### 四个元素

**Agent Card。** 位于 `/.well-known/agent.json` 的 JSON document，描述该 agent：name、skills、endpoints、supported modalities、auth requirements。Discovery 通过读取 card 完成。

```text
GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": {
      "tasks": "https://agent.example.com/tasks"
    },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }
```

**Task。** 工作单元。一个异步、有状态的对象，拥有生命周期：`submitted → working → completed / failed / canceled`。client 发送 task，然后轮询或订阅更新。

**Artifact。** task 产生的结果类型。Text、structured JSON、image、video、audio。Artifacts 有类型，所以不同 modalities 都是一等公民。

**Opaque lifecycle。** A2A 不规定 remote agent *如何*解决 task。client 看到 state transitions 和 artifacts；implementation 可以自由使用任意 framework。

### MCP/A2A 分工

- **MCP**（Lesson 13）：agent ↔ tool。agent 通过 JSON-RPC 读写 tool server。默认无状态。
- **A2A**：agent ↔ agent。Peer protocol；两边都是拥有自己 reasoning 的 agents。

生产级 multi-agent systems 会同时使用两者。一个 A2A peer 会调用它那一侧的 MCP tools。这个分工让两类 concern 保持清晰。

### Discovery flow

```text
Client                     Agent server
  ├──GET /.well-known/agent.json──>
  <──Agent Card JSON─────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working, 42% done──────
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts──
```

或者使用 streaming：订阅 `/tasks/{id}/events` 的 SSE 来接收 push updates。

### Auth

A2A 支持三种常见模式：

- **Bearer token** — OAuth2 或 opaque。
- **mTLS** — mutual TLS；组织之间互相证明身份。
- **Signed requests** — 对 payload 做 HMAC。

Auth 会在 Agent Card 里声明；clients 发现后遵守。

### 到 2026 年 4 月已有 150+ 组织

企业采用推动了 A2A 的规模。核心意思是：A2A 成了企业 agent systems 跨越 trust boundaries 的方式。Google Cloud 发布了 Vertex AI Agent Builder 的 A2A 支持；Microsoft Agent Framework 支持它；多数主要 frameworks（LangGraph、CrewAI、AutoGen）都提供 A2A adapters。

### A2A 的优势场景

- **跨组织调用。** 公司 A 的 agent 调用公司 B 的 agent。没有 A2A 时，每一对都是 bespoke contract。
- **异构 frameworks。** LangGraph agent 调用 CrewAI agent，再调用 custom Python agent。A2A 负责标准化。
- **Typed artifacts。** Video result、structured JSON、audio 都是一等公民。
- **长时间运行的 tasks。** Opaque lifecycle + polling 让持续数小时的 tasks 变得直接。

### A2A 的困难场景

- **延迟敏感的 micro-calls。** A2A 的 lifecycle 是 async。亚毫秒级 agent-to-agent 不适合；用 direct RPC。
- **紧耦合的 in-process agents。** 如果两个 agents 都运行在同一个 Python process，A2A 的 HTTP round-trip 就过重。
- **小团队。** 规范开销是真实存在的；internal-only agents 可能不需要这种正式性。

### A2A vs ACP, ANP, NLIP

2024-2026 年出现了几个相关规范：

- **ACP**（IBM/Linux Foundation）— A2A 的前身，范围更窄。
- **ANP**（Agent Network Protocol）— 更强调 peer discovery，decentralized-first。
- **NLIP**（Ecma Natural Language Interaction Protocol，2025 年 12 月标准化）— natural-language content type。

截至 2026 年 4 月，A2A 是采用最广的 peer protocol。比较可参考 arXiv:2505.02279（Liu et al., "A Survey of Agent Interoperability Protocols"）。

## 动手实现

`code/main.py` 使用 `http.server` 和 JSON 实现一个 A2A-minimal server 与 client。server：

- 暴露 `/.well-known/agent.json`，
- 接受 `POST /tasks`，
- 管理 task state，
- 在 `GET /tasks/{id}` 时返回 artifacts。

client：

- 获取 Agent Card，
- 提交 task，
- 轮询直到 completion，
- 读取 artifact。

运行：

```text
python3 code/main.py
```

脚本会在 background thread 中启动 server，然后运行 client 访问它。你会看到完整流程：discovery、submit、poll、artifact。

## 实际使用

`outputs/skill-a2a-integrator.md` 会设计一个 A2A integration：Agent Card contents、task schemas、auth choice、streaming vs polling。

## 交付成果

Checklist：

- **Pin the spec version。** A2A 仍在演进；Agent Card 应声明 protocol version。
- **Idempotent task creation。** 重复提交（network retries）应该产生同一个 task。
- **Artifact schemas。** 声明 agent 返回什么 shape；consumers 应该 validate。
- **Rate limits + auth。** A2A 面向公网；应用标准 web security。
- **Dead-letter for failed tasks。** 随时间检查 recurring failure types。

## 练习

1. 运行 `code/main.py`。确认 client 发现 server 并收到正确 artifact。
2. 给 server 添加第二个 skill（例如 "summarize"）。更新 Agent Card。写一个 client，根据 task type 选择 skill。
3. 实现一个 SSE streaming endpoint：`/tasks/{id}/events`，用于发出 state changes。client 需要有什么不同？
4. 阅读 A2A spec（https://a2a-protocol.org/latest/specification/）。找出三件 spec 强制要求但本 demo 没有实现的事。
5. 比较 A2A（Agent Card discovery）和 MCP（通过 `listTools` 做 server-side capability listing）。self-describing agents 与 capability-probing 之间的权衡是什么？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| A2A | “Agent-to-agent” | agent 跨系统调用其他 agents 的 peer protocol。Google 2025。 |
| Agent Card | “agent 的 business card” | 位于 `/.well-known/agent.json` 的 JSON，描述 skills、endpoints、auth。 |
| Task | “工作单元” | 异步有状态对象，带 lifecycle；completion 时产生 artifacts。 |
| Artifact | “结果” | Typed output：text、structured JSON、image、video、audio。一等媒体。 |
| Opaque lifecycle | “怎么解决是 agent 自己的事” | client 看到 state transitions；server 可自由选择 framework/tools。 |
| Discovery | “找到 agent” | `GET /.well-known/agent.json` 返回 card。 |
| MCP vs A2A | “tools vs peers” | MCP：纵向 agent ↔ tool。A2A：横向 agent ↔ agent。 |
| ACP / ANP / NLIP | “Sibling protocols” | 相邻规范；A2A 是 2026 年采用最广的规范。 |

## 延伸阅读

- [A2A specification](https://a2a-protocol.org/latest/specification/) — canonical spec
- [Google Developers Blog — A2A announcement](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) — 2025 年 4 月 launch post
- [A2A GitHub repo](https://github.com/a2aproject/A2A) — reference implementations and SDKs
- [Liu et al. — A Survey of Agent Interoperability Protocols](https://arxiv.org/html/2505.02279v1) — MCP、ACP、A2A、ANP 对比
