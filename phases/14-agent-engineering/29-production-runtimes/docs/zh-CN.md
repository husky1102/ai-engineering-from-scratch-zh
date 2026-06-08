# 生产运行时：Queue、Event、Cron

> 生产 agents 运行在六种 runtime shapes 上：request-response、streaming、durable execution、queue-based background、event-driven 和 scheduled。先选 shape，再选 framework。Observability 在每一种 shape 中都是 load-bearing。

**类型:** 学习
**语言:** Python (stdlib)
**先修:** Phase 14 · 13 (LangGraph), Phase 14 · 22 (Voice)
**时间:** ~60 分钟

## 学习目标

- 说出六种生产 runtime shapes，并把每一种匹配到 framework / product pattern。
- 解释为什么 durable execution（LangGraph）对 long-horizon tasks 很重要。
- 描述 event-driven runtime，以及 Claude Managed Agents 什么时候适合。
- 解释 multi-step agents 中 observability-as-load-bearing 的主张。

## 要解决的问题

生产 agents 的失败方式是 Jupyter notebook 暴露不出来的：第 37 步网络超时、用户在语音通话中途挂断、cron job 在机器重启时死亡、background worker 内存耗尽。Runtime shape 决定哪些失败可以幸存。

## 核心概念

### Request-response

- 同步 HTTP。用户等待完成。
- 只适合短任务（<30s）。
- Stacks：Agno（Python + FastAPI）、Mastra（TypeScript + Express/Hono/Fastify/Koa）。
- Observability：标准 HTTP access logs + OTel spans。

### Streaming

- SSE 或 WebSocket，用于渐进输出。
- LiveKit 将它扩展到语音/视频的 WebRTC（Lesson 22）。
- Stacks：任何支持 streaming 的 framework + 能处理 SSE/WS 的 frontend。
- Observability：per-chunk timing、first-token latency、tail latency。

### Durable execution

- 每一步之后 checkpoint state；失败时自动 resume。
- AutoGen v0.4 actor model 将失败隔离到一个 agent（Lesson 14）。
- LangGraph 的核心差异化能力（Lesson 13）。
- 当 step count 未知且恢复成本高时必不可少。

### Queue-based / background

- Job 进入 queue，workers 取走处理，结果通过 webhooks 或 pub/sub 回流。
- 对 long-horizon agents 必不可少（Anthropic 的 computer use 公告中说，每个任务可能有 dozens-to-hundreds of steps）。
- Stacks：Celery（Python）、BullMQ（Node）、SQS + Lambda（AWS）、custom。
- Observability：queue depth、per-job latency distribution、DLQ size。

### Event-driven

- Agents 订阅 triggers：new email、PR opened、cron fire。
- Claude Managed Agents 开箱覆盖这一点（Lesson 17）。
- CrewAI Flows（Lesson 15）组织 event-driven deterministic workflows。
- Observability：trigger source、event-to-start latency、agent latency。

### Scheduled

- 周期性运行的 cron-shaped agents。
- 与 durable execution 结合，让失败的 nightly run 在下一个 tick resume。
- Stacks：Kubernetes CronJob + durable framework；hosted（Render cron、Vercel cron）。

### 2026 部署模式

- **CrewAI Flows** 用于 event-driven production。
- **Agno** stateless FastAPI 用于 Python microservices。
- **Mastra** server adapters（Express、Hono、Fastify、Koa）用于 embedding。
- **Pipecat Cloud / LiveKit Cloud** 用于 managed voice（Lesson 22）。
- **Claude Managed Agents** 用于 hosted long-running async。

### Observability 是 load-bearing

没有 OpenTelemetry GenAI spans（Lesson 23）加 Langfuse/Phoenix/Opik backend（Lesson 24），你无法调试一个在第 40 步失败的 multi-step agent。这在生产中不是可选项。它是“我们快速 debug”和“我们从头 replay 并加更多 logging”之间的差别。

### 生产运行时容易失败的地方

- **错误的 shape 选择。** 给 5 分钟任务选择 request-response。用户挂断；workers 堆积；retries 叠加。
- **没有 DLQ。** Queue workers 没有 dead-letter。失败 jobs 消失。
- **不透明的 background work。** Background agent 没有 trace export 就运行。直到用户报告前，失败都不可见。
- **跳过 durable state。** 任何 > 30 秒且你不能承受重启的 run，都需要 durable execution。

## 动手实现

`code/main.py` 是一个 stdlib multi-shape demo：

- Request-response endpoint（普通函数）。
- Streaming handler（generator）。
- 带 DLQ 的 queue-based worker。
- Event trigger registry。
- Cron-shaped scheduler。

运行：

```bash
python3 code/main.py
```

输出：五条 traces，展示同一任务上每种 shape 的行为。同一套 agent logic，不同 outer shells。Durable execution（第六种 shape）已经在 Lesson 13 用 LangGraph checkpointing 专门覆盖，因此这里有意不展开。

## 实际使用

- **Request-response** 用于 chat-style UX。
- **Streaming** 用于 progressive responses。
- **Durable** 用于 long-horizon tasks。
- **Queue** 用于 batch / async / long-running。
- **Event** 用于 agent reactivity。
- **Cron** 用于 housekeeping（memory consolidation、evals、cost reports）。

## 交付成果

`outputs/skill-runtime-shape.md` 会为一个任务选择 runtime shape，并接好 observability requirements。

## 练习

1. 把你的 Lesson 01 ReAct loop 移植到你 stack 中的全部六种 shapes。哪种 shape 适合哪种 product surface？
2. 给 queue-based demo 添加 DLQ。模拟 10% job failure；显示 DLQ size。
3. 编写一个 cron-triggered eval agent，每晚针对当天 top 20 traces 运行。
4. 实现带 backpressure 的 streaming：如果 client 很慢，就暂停 agent。这和 turn budget 如何交互？
5. 阅读 Claude Managed Agents docs。什么时候你会把 self-hosted long-horizon agent 迁移到 managed？

## 关键术语

| 术语 | 人们通常怎么说 | 它实际意味着什么 |
|------|----------------|------------------|
| Request-response | “Synchronous” | 用户等待；只适合短任务 |
| Streaming | “SSE / WS” | 渐进输出；更好的 UX；可按 chunk 观测 latency |
| Durable execution | “Resume from failure” | Checkpointed state；从上一步重启 |
| Queue-based | “Background jobs” | Producer / worker pool / DLQ |
| Event-driven | “Trigger-based” | Agent 响应外部事件 |
| DLQ | “Dead-letter queue” | 失败 jobs 的停车场 |
| Claude Managed Agents | “Hosted harness” | Anthropic-hosted long-running async，带 caching + compaction |

## 延伸阅读

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — durable execution details
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — hosted long-running async
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — “dozens-to-hundreds of steps per task”
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — actor-model fault isolation
