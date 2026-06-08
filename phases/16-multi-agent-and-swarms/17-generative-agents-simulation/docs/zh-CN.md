# Generative Agents 与 Emergent Simulation

> Park et al. 2023（UIST '23, arXiv:2304.03442）在 **Smallville** 这个 25-agent sandbox 中使用三部分架构：**memory stream**（natural-language log）、**reflection**（agent 对自身 stream 生成的 higher-level syntheses）和 **plan**（day-level behavior，然后是 sub-plans）。标志性结果是 Valentine's Day party emergence：一个 agent 被种下“wants to throw a Valentine's Day party”，没有进一步 scripting，邀请在群体中传播，日期被协调，party 最终发生——而其他 24 个 agents 起初完全不知道这件事。Ablations 显示三个组件都对 believability 必不可少。已记录的 failures 是 spatial-norm errors（进入已关闭的商店、共用 single-person bathrooms）。这是 2026 年 agent simulations 和 multi-agent social evaluation 的参考架构。

**类型：** Learn + Build
**语言：** Python (stdlib)
**先修：** Phase 16 · 04 (Primitive Model), Phase 16 · 13 (Shared Memory)
**时间：** ~75 分钟

## 要解决的问题

大多数 multi-agent systems 是紧密 scripted teams：planner 计划，coder 写代码，reviewer 审查。这适合 well-defined tasks。它无法捕捉当 agents 拥有 memory、priorities 和 open world 时产生的 emergent、unscripted behavior。Research、society simulation，以及越来越多的 game AI 都需要第二种系统。

Smallville architecture 是它的 benchmark。在 Park 2023 之前，最好的 agent simulations 是浅层 script-followers；之后，这个 pattern 成为 open worlds 中 generative agents 的默认架构。如果你在 2026 年构建 agent simulation，你要么使用 Smallville 的三个组件，要么明确说明为什么不使用。

## 核心概念

### 三个组件

**Memory stream。** observations、actions、reflections 和 plans 的 append-only log。每个 entry 有 timestamp、type、description（natural language）以及 derived metadata：**recency**、**importance**（agent 自评 1-10）和 **relevance**（与 current query 的 cosine similarity）。

```text
[2026-02-14 09:12:03] observation: Isabella Rodriguez asked me if I like jazz
[2026-02-14 09:14:22] reflection:   I enjoy long conversations about music
[2026-02-14 10:05:00] plan:         Attend Isabella's Valentine's Day party tonight
```

Memory retrieval 结合三个 scores：`score = w_recency * e^(-decay * age) + w_importance * importance + w_relevance * cos_sim`。Top-k entries 进入 current prompt。

**Reflection。** 周期性地（每 N 条 memories 或发生 important events 时），agent 从 recent memories 生成 higher-order syntheses。Reflection entries 会被写回 stream，并像任何其他 memory 一样可检索。这是 agents 建立“understandings”的方式，也是该架构中的 long-term beliefs 等价物。

**Plan。** 自顶向下分解。先是 broad strokes 的 day-level plan（“go to work, have dinner with Klaus”）。然后是 hour-level plans。再到 action-level plans。Plans 是可修订的：当 observation 与 plan 矛盾时，agent 会 replan 受影响的 segment。

### 为什么三个都重要（ablation）

Park et al. 做了分别移除 observation、reflection 和 plan 的 ablations。每个 ablation 都伤害 believability：

- 没有 **observation**，agent 会错过 context，并基于过时 beliefs 行动。
- 没有 **reflection**，agent 无法形成 higher-order beliefs；interactions 会保持浅层。
- 没有 **plan**，behavior 变成 reactive noise；goals 会消散。

Human raters 给出的 believability scores 在三个组件都存在时最高；去掉任一个都会产生可测量 regression。

### Valentine's Day emergence

一个 agent，Isabella Rodriguez，被种下目标“wants to throw a Valentine's Day party at Hobbs Cafe on Feb 14 at 5pm”。其他 24 个 agents 没有这样的 seed。在模拟的数天内：

1. Isabella 的 plan 包含邀请别人。
2. 每次邀请都成为 neighbor memory stream 中的 observation。
3. 该 neighbor 的 reflection 生成 beliefs：“Isabella is throwing a party.”
4. neighbor 的 plan 纳入“attend party on Feb 14”。
5. Neighbors 告诉其他 neighbors。邀请在没有 central coordination 的情况下传播。
6. 2 月 14 日下午 5 点，若干 agents 汇聚到 Hobbs Cafe。

这是 technical sense 上的 emergence：system-level behavior（一个 party）从 local interactions（bilateral invitations + individual planning）中出现，没有 central orchestrator。

### 已记录的 failure modes

Park et al. 明确记录了：

- **Spatial norm errors。** Agents 走进已关闭的 stores。Agents 试图使用同一个 single-person bathroom。Agents 在不该吃东西的 rooms 里吃饭。模型不会仅从 environment 推断 social-physical norms。
- **Memory overflow。** 深度 simulation runs 会让 memory-retrieval cost 增长。实用补救：periodic memory compaction（summarize-and-prune）以及对 low-importance entries 做 decay。
- **Reflection hallucination。** Reflections 可能发明 memory stream 中不存在的 relationships。Mitigation：在 reflection prompts 中包含 source memory ids，并在 retrieval time verify。

这些是 production-relevant failure modes：任何 2026 年的 agent simulation 都会继承它们。

### 三组件实现规则

1. **Memory is append-only。** 永远不要 mutate memory entry。Corrections 是新 entries。
2. **Importance scores are cheap。** 在 write time 调用 LLM 给 importance 评 1-10。缓存 score。
3. **Retrieval is ranked, not filtered。** 按 combined score 取 Top-k；不要使用 hard filters（会丢失 context）。
4. **Reflection runs periodically。** 当未处理 memories 的 importance 总和超过 threshold（例如 150）时触发。
5. **Plans are revisable。** 当新的 observation 与 plan 矛盾时，只 regenerate 受影响的 segment，不是整个 plan。

### Smallville 之外的 generative agents

2024-2026 年的后续文献扩展了该架构：

- **用于 policy / market research 的 multi-agent social simulation。** Smallville-like populations 模拟用户对 features 的 behavior response。比 A/B tests 更快；accuracy 仍有争议。
- **Games 的 NPC AI。** 带 Smallville agents 的 RPGs 会产生 emergent storylines，而不是 scripted quests。
- **Generative-agent evaluation benchmarks。** metric 不再是 task accuracy，而是 believability + 长时间运行中的 behavior coherence。

该架构是参考架构。Extensions 会替换组件（memory 用 vector store、retrieval-augmented reflection、neurosymbolic plan），但保留三部分结构。

### 这为什么对 multi-agent engineering 重要

Smallville 是一个 proof of concept：只要组件正确，multi-agent emergence 就很便宜。这个架构已经在 open-source models 上复现（更小的 LLMs 会让 believability 平滑下降，而不是骤降）。任何需要 **emergent social behavior** 的生产系统都会使用这种形状。任何需要 **tight task execution** 的系统，则使用本 phase 前面介绍的 supervisor / roles / primitives patterns。

## 动手实现

`code/main.py` 用 stdlib Python 和 scripted agent policies（没有真实 LLM）实现三个组件。demo 以微缩方式复现 Valentine's-party emergence：

- `MemoryStream` — append-only log with recency/importance/relevance retrieval。
- `reflect(stream)` — 对 recent high-importance memories 做 scripted reflection。
- `plan(agent_state)` — 基于 current beliefs 的 day-level 和 hour-level plans。
- Scenario：5 agents。Agent 1 起始目标是“throw party at 5pm”。经过 simulated ticks，邀请传播，agents 汇聚。

运行：

```text
python3 code/main.py
```

预期输出：tick-by-tick trace。到最后一个 tick，5 个 agents 中至少 3 个会在 plan 中出现 party，并且汇聚到 party location。这个单一 seed 在没有 orchestrator 的情况下产生了 coordinated arrival。

## 实际使用

`outputs/skill-simulation-designer.md` 设计 generative-agent simulation：agents 数量、memory schema、reflection cadence、plan horizon 和 evaluation metric。

## 交付成果

Production simulations 的规则：

- **Memory is the database。** 规模化时选择真实 store（vector DB、Postgres）。In-memory stdlib 只适合 prototypes。
- **Log the retrieval trace。** 对每个 action，记录驱动它的 top-k memories。这就是你的 debug ability。
- **Budget per-agent tokens。** 每个 agent 每个 tick 的 retrieve + reflect + plan 是 O(k) LLM calls。N agents × T ticks × calls-per-tick 可能压垮预算。
- **Compact memory periodically。** summarize-and-prune low-importance entries。Retention policy 是设计决策，不是细节。
- **Detect spatial / social norm violations** explicitly。该架构不会自己学会它们。

## 练习

1. 运行 `code/main.py`。确认 3+ agents 汇聚到 party。把 agents 增加到 10：emergence 还会发生吗？
2. 移除 reflection step。behavior 看起来如何？映射到 Park 2023 的 ablation finding。
3. 引入一个竞争性的 seeded goal（“Klaus wants to give a research talk at 5pm”）。Agents 会分裂，还是某个 goal 会占主导？决定因素是什么？
4. 添加 spatial constraints：Hobbs Cafe 最多容纳 4 个 agents。simulation 能优雅处理 overflow，还是会击中“single-person bathroom” failure pattern？
5. 阅读 Park et al.（arXiv:2304.03442）Section 6（emergent behavior experiments）。找出一个你的 miniature 无法复现的 behavior。你需要增强该架构的哪个组件？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Memory stream | “agent 的 diary” | observations、actions、reflections、plans 的 append-only log。 |
| Recency | “memory 有多新” | 按 age 做 exponential-decay score。 |
| Importance | “agent 有多在意” | write time 自评 1-10。缓存。 |
| Relevance | “与 current query 有多相关” | Cosine similarity（embedding-based）。 |
| Reflection | “Higher-order belief” | 从 recent memories 生成的 synthesis，会作为新 memory 重新 ingested。 |
| Plan | “Day/hour/action decomposition” | Top-down plan tree。observations 矛盾时可修订。 |
| Smallville | “Park 2023 的 sandbox” | 产生 Valentine's Day emergence 的 25-agent simulation。 |
| Believability | “质量指标” | human-rater score，用于判断 behavior 是否像 plausible agent。 |

## 延伸阅读

- [Park et al. — Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) — reference architecture
- [UIST '23 paper page](https://dl.acm.org/doi/10.1145/3586183.3606763) — publication venue
- [Smallville code release](https://github.com/joonspk-research/generative_agents) — reference Python implementation
- [Hayes-Roth 1985 — A Blackboard Architecture for Control](https://www.sciencedirect.com/science/article/abs/pii/0004370285900639) — structured-memory agents 的 prior art
