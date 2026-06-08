# Memory Blocks 和 Sleep-Time Compute（Letta）

> MemGPT 在 2024 年变成了 Letta。2026 年的演进增加了两个想法：模型可以直接编辑的离散 functional memory blocks，以及一个在 primary agent 空闲时异步 consolidation memory 的 sleep-time agent。这就是把 memory 扩展到单次 conversation 之外的方式。

**类型：** 构建
**语言：** Python (stdlib)
**先修：** Phase 14 · 07 (MemGPT)
**时间：** ~75 分钟

## 学习目标

- 说出 Letta 使用的三层 memory tiers（core、recall、archival）以及每一层的角色。
- 解释 memory-block pattern：Human block、Persona block 和 user-defined blocks 都是一等 typed objects。
- 描述什么是 sleep-time compute、为什么它位于 critical path 之外，以及为什么它可以运行比 primary agent 更强的模型。
- 实现一个 scripted two-agent loop：primary agent 提供 responses，sleep-time agent 在 turns 之间 consolidation blocks。

## 要解决的问题

MemGPT（Lesson 07）解决了 virtual-memory control flow。随后出现了三个 production problems：

1. **Latency.** 每个 memory operation 都在 critical path 上。如果 agent 必须在用户等待时 prune、summarize 或 reconcile，tail latency 会爆炸。
2. **Memory rot.** Writes 持续累积。被矛盾事实覆盖的旧 facts 仍然存在。Retrieval 被 stale content 淹没。
3. **Structure loss.** Flat archival store 无法表达“The Human block is always in the prompt; the Persona block is always in the prompt; the Task block swaps per session.”

Letta（letta.com）是 2026 年的重写。Memory blocks 让结构显式化；sleep-time compute 把 consolidation 移到 critical path 之外。

## 核心概念

### 三层

| Tier | Scope | Where it lives | Written by |
|------|-------|----------------|------------|
| Core | Always visible | Inside the main prompt | Agent tool call + sleep-time rewrites |
| Recall | Conversation history | Retrievable | Automatic turn logging |
| Archival | Arbitrary facts | Vector + KV + graph | Agent tool call + sleep-time ingest |

Core 是 MemGPT core。Recall 是 conversation buffer 及其被 evict 的尾部。Archival 是 external store。这个拆分清理了 MemGPT two-tier 设计中的重载。

### Memory blocks

block 是 core tier 中 typed、persistent、editable 的 section。原始 MemGPT paper 定义了两个：

- **Human block** — 关于用户的 facts（name、role、preferences、goals）。
- **Persona block** — agent 的 self-concept（identity、tone、constraints）。

Letta 将其泛化为任意 user-defined blocks：当前 goal 的 `Task` block、codebase facts 的 `Project` block、hard constraints 的 `Safety` block。每个 block 都有 `id`、`label`、`value`、`limit`（character cap）、`description`（让模型知道何时编辑它）。

Blocks 可通过 tool surface 编辑：

- `block_append(label, text)`
- `block_replace(label, old, new)`
- `block_read(label)`
- `block_summarize(label)` — 压缩接近 limit 的 block。

### Sleep-time compute

Letta 在 2025 年加入的想法：在后台、critical path 之外运行第二个 agent。Sleep-time agents 处理 conversation transcripts 和 codebase context，把 `learned_context` 写入 shared blocks，并 consolidation 或 invalidate archival records。

随之产生的性质：

- **No latency cost.** Primary responses 不等待 memory ops。
- **Stronger model allowed.** Sleep-time agent 可以是更昂贵、更慢的模型，因为它不受 latency 约束。
- **Natural consolidation window.** 用户不等待时，进行 dedup、summarize、invalidate contradicted facts。

这个形态和人类工作的方式一致：你先做任务，睡一觉，long-term memory 在夜里沉淀下来。

### Letta V1 和 native reasoning

Letta V1（`letta_v1_agent`, 2026）废弃 `send_message`/heartbeat 和 inline `Thought:` tokens，改用 native reasoning。Responses API（OpenAI）和带 extended thinking 的 Messages API（Anthropic）会在单独 channel 上发出 reasoning，并跨 turns 传递（生产中跨 providers encrypted）。control loop 仍然是 ReAct。thought trace 是结构性的，不是 prompt-shaped。

### 这个模式容易出错的地方

- **Block bloat.** 无限 `block_append` 很快撞到 limit。在会越过 cap 的 write 之前接入 block summarizer。
- **Silent drift.** Sleep-time agent 改写了 block，而 primary agent 从未注意到。给 blocks 做 versioning，并在 trace 中暴露 diffs。
- **Poisoned consolidation.** Sleep-time agent 把 attacker-reachable content 处理进 core。Lesson 27 同样适用于 sleep-time surface。

## 动手实现

`code/main.py` 实现：

- `Block` — id、label、value、limit、description。
- `BlockStore` — CRUD + `near_limit(label)` helper。
- 两个 scripted agents — `PrimaryAgent` 服务一个 turn，`SleepTimeAgent` 在 turns 之间 consolidation。
- 一段 trace，展示三轮 conversation、block writes，以及一个 sleep-time pass：summarize 一个 block 并 invalidate 一个 stale fact。

运行：

```text
python3 code/main.py
```

transcript 展示了这个拆分：primary turns 很快，并产出 raw writes；sleep pass 负责 compact 和清理。

## 实际使用

- **Letta**（letta.com）作为 reference implementation。可以 self-host，也可以用 managed cloud。
- **Claude Agent SDK skills** 作为 block-shaped knowledge — skill 是一个命名、versioned、retrievable 的 instructions block，agent 会按需加载。
- **Custom builds** 适合想控制 storage backend 的团队。使用 Letta API contract，便于日后迁移。

## 交付成果

`outputs/skill-memory-blocks.md` 会为任意 runtime 生成 Letta-shaped block system，带 sleep-time hooks，并包含 safety rules 和 citation wiring。

## 练习

1. 添加一个 `block_summarize` tool，当 `near_limit` 返回 true 时，把 block value 替换为 model-generated summary。哪个 trigger threshold 能同时最小化 summarization calls 和 block overflow？
2. 实现 archival 的 sleep-time dedup：text 有 >90% token overlap 的两条 records 合并为一条。只在 sleep pass 中做，绝不在 critical path 上做。
3. 给 blocks 做 versioning。每次 write 都记录 old value 和 diff。暴露 `block_history(label)`，让 operators 可以 debug“为什么 agent 忘了 X”。
4. 把 sleep-time agents 当作 untrusted writers。它们触碰 Persona 或 Safety block 时，提交前要求 second-agent review。
5. 将示例移植为使用 Letta API（`letta_v1_agent`）。block schema 有什么变化，native reasoning 如何改变 trace shape？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Memory block | “Editable prompt section” | core memory 中 typed、persistent、LLM-editable 的 segment |
| Human block | “User memory” | 关于用户的 facts，固定在 core 中 |
| Persona block | “Agent identity” | self-concept、tone、constraints，固定在 core 中 |
| Sleep-time compute | “Async memory work” | 第二个 agent 在 critical path 之外做 consolidation |
| Core / Recall / Archival | “Tiers” | 三层 memory split：always-visible / conversation / external |
| Block limit | “Cap” | 每个 block 的 character limit，迫使 summarization |
| Native reasoning | “Thinking channel” | Provider-level reasoning output，而不是 prompt-level `Thought:` |
| Learned context | “Sleep output” | sleep-time agent 写入 shared blocks 的 facts |

## 延伸阅读

- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) — the block pattern
- [Letta, Sleep-time Compute blog](https://www.letta.com/blog/sleep-time-compute) — async consolidation
- [Letta, Rearchitecting the Agent Loop](https://www.letta.com/blog/letta-v1-agent) — native reasoning rewrite
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — the origin
