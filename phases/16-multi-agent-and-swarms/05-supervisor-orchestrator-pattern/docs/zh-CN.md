# Supervisor / Orchestrator-Worker Pattern

> 一个 lead agent 负责规划和委派；specialized workers 在并行 contexts 中执行并回报。这就是 Anthropic Research system 背后的 pattern（Claude Opus 4 作为 lead，Sonnet 4 作为 subagents），在 internal research evals 上相对 single-agent Opus 4 提升 +90.2%。Anthropic 的 engineering post 报告称，BrowseComp 上 80% 的 variance 仅由 token usage 解释——multi-agent 获胜很大程度上是因为每个 subagent 都获得一个新鲜 context window。本课从 primitives 构建 supervisor pattern，并覆盖 production deployments 中到 2026 年仍然重要的 engineering lessons。

**类型：** 学习 + 构建
**语言：** Python (stdlib, `threading`)
**先修：** Phase 16 · 04 (Primitive Model)
**时间：** ~75 分钟

## 要解决的问题

Research 是 single-agent systems 最容易失败的原型任务。你问“2023 到 2026 年之间 multi-agent systems 发生了什么变化？”一个 single agent 顺序读取五篇 papers，把半个 context 填满它们的文本，然后还要一起推理所有内容。等它读到第五篇时，它已经忘了第一篇。它无法并行化。

Supervisor pattern 修复这个问题：一个 lead agent 规划搜索，把每个 sub-question 委派给一个 worker，然后 synthesizes。每个 worker 都为一个 narrow question 获得自己的 200k-token window。Lead 永远不看 raw papers——只看 worker summaries。

Anthropic 的 production Research system 报告称，相比 single Opus 4，它在 internal research evals 上提升 +90.2%。同一篇 post 还指出，BrowseComp variance 的 80% 由 *token usage alone* 解释。每个 subagent 的 fresh context 是主要机制。

## 核心概念

### 这个 pattern

```text
                 ┌──────────────┐
                 │   Lead       │  plans, decomposes,
                 │  (Opus 4)    │  synthesizes
                 └──┬────┬───┬──┘
                    │    │   │
            ┌───────┘    │   └───────┐
            ▼            ▼           ▼
      ┌─────────┐  ┌─────────┐  ┌─────────┐
      │ Worker1 │  │ Worker2 │  │ Worker3 │
      │(Sonnet) │  │(Sonnet) │  │(Sonnet) │
      └─────────┘  └─────────┘  └─────────┘
         fresh       fresh        fresh
         context     context      context
```

Lead 永远不读取 raw materials。Workers 在 lead synthesizes 之前永远看不到彼此的 work。每条 arrow 都是一个带有 narrow artifact 的 handoff。

### 为什么它会赢

三种机制：

1. **每个 subagent 都有 fresh context。** 探索 “FIPA-ACL heritage” 的 worker 不会携带 lead 规划时花掉的 40k tokens。它为一个问题获得一个 200k window。
2. **通过 prompt 形成 specialization。** Lead 的 prompt 是 “decompose and synthesize”，不是 “research”。每个 worker 的 prompt 很窄：“find what changed in X”。聚焦的 prompts 产生聚焦的 outputs。
3. **Parallelism。** Workers 并发运行。Wall-clock time 大约是 `max(worker_times) + plan + synthesis`，不是 `sum(worker_times)`。

### Engineering lessons (Anthropic 2025)

Anthropic post 列出了几个到 2026 年仍然相关的 production lessons：

- **Scale effort to query complexity。** Simple queries：一个 agent，3-10 次 tool calls。Complex queries：10+ agents。必须由 lead 估计，而不是 caller。
- **Broad then narrow。** 先分解成 broad sub-questions，如果答案值得深入，再为每个 sub-question 生成更多 workers。
- **Rainbow deployments。** Agents 是 long-running 且 stateful 的。传统 blue-green 不适用。Anthropic 使用 rainbow：逐步推出新版本，同时让旧版本自然 drain。
- **Token usage dominates。** Multi-agent 约为 single-agent 的 15× tokens。只有 task value 足以证明成本合理时才运行它。

### LangGraph 的转向

LangGraph 最初发布了一个 `langgraph-supervisor` library，其中有 high-level `create_supervisor` helper。2025 年，LangChain 把推荐方式转为通过 tool-calling 直接实现 supervisor pattern，因为 tool calls 能更好地控制 *supervisor sees what*（context engineering）。这个 library 仍然可用；docs 现在推荐 tool-calling form。

### Failure modes

- **Lead hallucinates the plan。** 如果 lead 生成的 sub-questions 没有分解真实问题，workers 会在错误目标上做精确 research。
- **Workers over-explore。** 没有显式 scope boundaries，workers 会漂移到 assigned sub-question 之外，污染 synthesis step。
- **Synthesis conflicts。** 两个 workers 返回互相矛盾的 facts。Lead 必须 either re-ask（增加一轮）或显式标出 disagreement。Silent picking of one side 是最糟的失败：user 永远不知道 disagreement 发生过。

### 什么时候 supervisor 是错的

- **Sequential tasks。** 如果 step 2 字面上需要 step 1 的 output，parallelism 没有收益。使用 pipeline（CrewAI Sequential、LangGraph linear graph）。
- **Simple queries。** Single-agent 更快也更便宜。在生成 workers 前使用 lead 的 “scale effort” check。
- **Strict determinism。** Supervisor 使用 LLM-selected delegation。当 audit/replay 比 adaptability 更重要时，static graphs 更好。

## 动手实现

`code/main.py` 使用 `threading` 实现一个由三个 parallel workers 组成的 supervisor。Lead 把 query 分解成 sub-questions，workers 在每个 sub-question 上并发运行，然后 lead synthesizes。没有真实 LLMs——workers 是 scripted，用来模拟 fetch-and-summarize。

关键结构：

- `Lead.plan(query)` 把 query 拆成 3 个 sub-questions。
- `Worker.run(sub_q)` 返回一个 fake summary（production 中可以是任何 tool-using agent）。
- `Lead.run(query)` 在 threads 中启动 workers、join，并 synthesizes。

运行：

```text
python3 code/main.py
```

Output 展示 plan、带 start/end timestamps 的 parallel worker traces，以及 final synthesis。你可以看见 wall-clock wins：三个 0.3 秒 workers 在约 0.35 秒内完成，而不是 0.9 秒。

## 实际使用

`outputs/skill-supervisor-designer.md` 接收一个 user query，并生成 supervisor-pattern design：lead system prompt、worker roles、sub-question decomposition rules，以及 synthesis template。在构建新的 research-style agent system 前使用它。

## 交付成果

部署 supervisor pattern 前的 checklist：

- **Model pairing。** Lead 使用 reasoning-tier model（Opus class、`o3` class）。Workers 使用更快、更便宜的 model（Sonnet、`o4-mini`）。
- **Worker timeout。** 任何超过 2× median runtime 的 worker 都被 kill；lead 要么用更窄 scope 重新 spawn，要么在没有它的情况下继续。
- **Token cap per worker。** Hard limit（比如 expected synthesis input 的 10×）防止 runaway worker 炸掉 budget。
- **Observability。** Trace lead's plan、每个 worker's tool calls，以及 synthesis。这是任何 post-hoc debugging 的基础。
- **Rainbow rollout。** Stateful long-running agents 需要 gradual version transition，而不是 hot swap。

## 练习

1. 运行 `code/main.py`，然后修改 lead，让它 spawn 5 个 workers 而不是 3 个。观察 wall-clock effect。在这个 demo 中，worker count 到多少时 spawn overhead 会超过 parallel savings？
2. 实现 worker timeout：kill 任何运行超过 0.5 秒的 worker，并让 lead synthesize remaining results。你需要什么 observability 才知道某个 worker 被 cut？
3. 给 lead 的 synthesis 增加 conflict-detection step：如果两个 workers 返回 contradictory answers，lead 标出 disagreement，而不是选择其中一个。不调用 LLM 时，你如何 detect contradiction？
4. 阅读 Anthropic 的 Research-system engineering post。列出这个 toy demo 如果要进 production，必须采用的三种 practices。
5. 比较 LangGraph 的 `create_supervisor`（legacy）与新的 tool-calling recommendation。哪一个能更好控制 supervisor sees what？为什么 Anthropic 明确只把 sub-answers，而不是 raw worker context，传入 synthesis？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Supervisor | "Lead agent" | 一个 orchestrator agent，负责 planning、delegating 和 synthesizing。它自己不做具体工作。 |
| Worker | "Subagent" | 由 supervisor 调用的 focused agent，scope 很窄，并有自己的 context window。 |
| Orchestrator-worker | "Supervisor pattern" | 同一件事，不同名字。2026 年 literature 两者都用。 |
| Fresh context | "Clean window" | Worker 的 context 从它的 system prompt 和 assigned question 开始，而不是从 lead 的 history 开始。 |
| Rainbow deployment | "Gradual rollout" | Long-running stateful agents 需要 versioned drain-and-replace，而不是 blue-green。 |
| Token dominance | "Context is the variable" | 根据 Anthropic，research-eval variance 的 80% 来自 total tokens used，而不是 model choice。 |
| Scale effort | "Match agent count to complexity" | Lead 估计 query difficulty，并相应 spawn 1 个 vs 10+ workers。 |
| Synthesis conflict | "Workers disagree" | 两个 workers 返回 contradictory facts；lead 必须暴露 disagreement，而不是 silent pick one。 |

## 延伸阅读

- [Anthropic engineering — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) —— supervisor pattern 的 production reference
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) —— tool-calling supervisor 现在是 recommended form
- [LangGraph supervisor reference](https://reference.langchain.com/python/langgraph-supervisor) —— legacy helper，2026 production 中仍有人使用
- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) —— handoff-based supervisor variant
