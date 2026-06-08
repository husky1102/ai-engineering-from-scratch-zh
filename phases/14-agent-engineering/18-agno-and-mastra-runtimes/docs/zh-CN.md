# Agno 与 Mastra：生产运行时

> Agno（Python）和 Mastra（TypeScript）是 2026 年的一组 production-runtime pairing。Agno 目标是 microsecond agent instantiation 和 stateless FastAPI backends。Mastra 在 Vercel AI SDK substrate 上提供 agents、tools、workflows、unified model routing 和 composite storage。

**类型:** 学习
**语言:** Python, TypeScript
**先修:** Phase 14 · 01 (Agent Loop), Phase 14 · 13 (LangGraph)
**时间:** ~45 分钟

## 学习目标

- 识别 Agno 的 performance targets，以及它们什么时候重要。
- 说出 Mastra 的三个 primitives -- Agents、Tools、Workflows -- 以及支持的 server adapters。
- 解释为什么 stateless session-scoped FastAPI backend 是推荐的 Agno production path。
- 为给定 stack 选择 Agno 或 Mastra（Python-first vs TypeScript-first）。

## 要解决的问题

LangGraph、AutoGen、CrewAI 都偏 framework-heavy。想要 "just the agent loop, fast, in my runtime" 的团队会转向 Agno（Python）或 Mastra（TypeScript）。两者都用一部分 framework-owned primitives 换取 raw speed，以及与周边 stack 更紧密的适配。

## 核心概念

### Agno

- Python runtime，前身是 Phi-data。
- "No graphs, chains, or convoluted patterns -- just pure python."
- 文档中的 performance targets：~2μs agent instantiation、每个 agent ~3.75 KiB memory、~23 model providers。
- Production path：stateless session-scoped FastAPI backend。每个 request 启动一个 fresh agent；session state 存在 DB 中。
- 原生 multimodal（text、image、audio、video、file）和 agentic RAG。

当你每秒有数千个 short-lived agents（chat fan-in、evaluation pipelines）时，speed targets 很重要。当一个 agent 要运行 10 分钟时，它们就没那么重要。

### Mastra

- TypeScript，构建在 Vercel AI SDK 之上。
- 三个 primitives：**Agents**、**Tools**（Zod-typed）、**Workflows**。
- Unified Model Router -- 跨 94 个 providers 的 3,300+ models（2026 年 3 月）。
- Composite storage：memory、workflows、observability 可接到不同 backends；大规模 observability 推荐 ClickHouse。
- Apache 2.0，源码中的 `ee/` directories 使用 source-available enterprise license。
- 支持 Express、Hono、Fastify、Koa 的 server adapters；first-class Next.js 和 Astro integration。
- 提供 Mastra Studio（localhost:4111）用于 debugging。
- 1.0（2026 年 1 月）时有 22k+ GitHub stars、300k+ weekly npm downloads。

### Positioning

二者都不是想成为 LangGraph。它们竞争的是：

- **Language fit。** Agno 面向 Python-first teams；Mastra 面向 TypeScript-first。
- **Runtime ergonomics。** Agno = near-zero overhead；Mastra = 与 Vercel ecosystem 集成。
- **Observability。** 两者都集成 Langfuse/Phoenix/Opik（Lesson 24），但 Mastra Studio 是 first-party。

### 何时选择各自

- **Agno** -- Python backend、许多 short-lived agents、强 performance requirements、FastAPI shop。
- **Mastra** -- TypeScript backend、Next.js / Vercel deploy、unified multi-provider model routing、Zod-typed tools。
- **LangGraph**（Lesson 13）-- 当 durable state 和 explicit graph reasoning 比 raw speed 更重要时。
- **OpenAI / Claude Agent SDK** -- 当你想要 provider productized shape（Lessons 16-17）时。

### 这个模式容易出错的地方

- **Perf-for-perf's-sake。** 因为 "2μs" 听起来很棒就选择 Agno，但 workload 是每个 request 只有一次 slow agent call。Overhead 不是 bottleneck。
- **Ecosystem lock-in。** Mastra 的 Vercel-flavored integration 在 Vercel 上是加分项，在别处则是减分项。
- **Enterprise license confusion。** Mastra 的 `ee/` directories 是 source-available，不是 Apache 2.0。如果计划 fork，请阅读 licenses。

## 动手实现

本课主要是比较型 -- 没有一个单独 code artifact 能公允地覆盖两个框架。见 `code/main.py` 中的 side-by-side toy：一个最小的 "run an agent, stream the output, persist session" flow 实现了两次（一次 Agno-shaped，一次 Mastra-shaped）。

运行：

```text
python3 code/main.py
```

两个 traces 在结构上不同，但功能等价。

## 实际使用

- **Agno** -- 需要速度和 FastAPI shape 的 Python backend。
- **Mastra** -- 拥有许多 providers 和 workflow primitives 的 TypeScript backend。
- 二者都提供 first-party observability hooks。二者都集成 Langfuse。

## 交付成果

`outputs/skill-runtime-picker.md` 会基于 stack、latency budget 和 operational shape，在 Agno、Mastra、LangGraph 或 provider SDK 之间做选择。

## 练习

1. 阅读 Agno docs。把 stdlib ReAct loop（Lesson 01）移植到 Agno。什么消失了？什么留下了？
2. 阅读 Mastra docs。把同一个 loop 移植到 Mastra。Tool typing（Zod vs nothing）发生了什么变化？
3. Benchmark：测量你 stack 上的 agent instantiation latency。Agno 的 2μs 对你的 workload 重要吗？
4. 设计一次 migration：如果你一直在 Python 中运行 CrewAI，迁移到 Agno 会破坏什么？
5. 阅读 Mastra 的 `ee/` license terms。哪些限制会影响 open-source fork？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Agno | "Fast Python agents" | Stateless session-scoped agent runtime |
| Mastra | "TypeScript agents on Vercel AI SDK" | Agents + Tools + Workflows + Model Router |
| Unified Model Router | "Multi-provider access" | 跨 94 个 providers 的 3,300+ models 的单一 client |
| Composite storage | "Multiple backends" | Memory/workflows/observability 各接到不同 store |
| Mastra Studio | "Local debugger" | 用于 introspecting agents 的 localhost:4111 UI |
| Source-available | "Not OSS" | 允许阅读源码但限制 commercial use 的 license |

## 延伸阅读

- [Agno Agent Framework docs](https://www.agno.com/agent-framework) -- performance targets、FastAPI integration
- [Mastra docs](https://mastra.ai/docs) -- primitives、server adapters、Model Router
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) -- stateful-graph alternative
- [Comet Opik](https://www.comet.com/site/products/opik/) -- Mastra integrations 引用的 observability comparisons
