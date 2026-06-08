# 记忆：虚拟上下文和 MemGPT

> Context windows 是有限的。Conversations、documents 和 tool traces 不是。MemGPT（Packer et al., 2023）把它表述为 OS virtual memory：main context 是 RAM，external store 是 disk，agent 在二者之间 page。每个 2026 年 memory system 都继承了这个模式。

**类型：** 构建
**语言：** Python (stdlib)
**先修：** Phase 14 · 01 (Agent Loop), Phase 14 · 06 (Tool Use)
**时间：** ~75 分钟

## 学习目标

- 解释 MemGPT 所依赖的 OS analogy：main context = RAM，external context = disk，memory tools = page in/out。
- 用 stdlib 实现两层 MemGPT pattern：main-context buffer、external searchable store，以及 page in/out tools。
- 描述 agent 如何发出“interrupts”来查询或修改 external memory，以及结果如何拼接回下一个 prompt。
- 识别延续到 Letta（Lesson 08）和 Mem0（Lesson 09）的 MemGPT design choices。

## 要解决的问题

Context windows 看起来像是应该能解决 memory。实际上不能。生产中反复出现三种 failure modes：

1. **Overflow.** Multi-turn conversations、long documents 或 tool-call-heavy trajectories 超过窗口。cutoff 之后的一切都会消失。
2. **Dilution.** 即使在窗口内，塞入无关 context 也会稀释对重要内容的 attention。Frontier models 在长输入上仍会退化。
3. **Persistence.** 新 session 从空窗口开始。没有 external memory 的 agents 无法跨 sessions 说“remember when you asked me to...”。

更大的窗口有帮助，但不能修好这个问题。Mem0 的 2025 年论文测量发现，128k-window baselines 仍然会漏掉那些 4k-window agent 加 external memory 能抓住的 long-horizon facts。

## 核心概念

### MemGPT：OS analogy

Packer et al.（arXiv:2310.08560, v2 Feb 2024）把 context management 映射到 operating-system virtual memory：

| OS concept | MemGPT concept | 2026 production analog |
|------------|---------------|------------------------|
| RAM | main context（prompt） | Anthropic/OpenAI context window |
| Disk | external context | vector DB、KV、graph store |
| Page fault | memory tool call | `memory.search`、`memory.read`、`memory.write` |
| OS kernel | agent control loop | 带 memory tools 的 ReAct loop |

agent 运行一个普通 ReAct loop。额外的一类 tools 让它可以把数据 page in/out main context。

### 两层

- **Main context.** 固定大小的 prompt，保存当前 task。模型始终可见。
- **External context.** 无界，通过 tools 搜索。相关时读取，事实出现时写入。

原论文在两个超出 base window 的任务上评估这个设计：超过 100k tokens 的 document analysis，以及跨 days 的 persistent memory multi-session chat。

### Interrupt pattern

MemGPT 引入 memory-as-interrupt：对话中途，agent 可以调用 memory tool，runtime 执行它，结果会作为新 observation 拼接进下一轮 assistant turn。从概念上看，这和 Unix `read()` syscall 一样：阻塞进程，返回 bytes，然后进程继续。

Canonical memory tool surface：

- `core_memory_append(section, text)` — 写入 prompt 的 persistent section。
- `core_memory_replace(section, old, new)` — 编辑 persistent section。
- `archival_memory_insert(text)` — 写入 searchable external store。
- `archival_memory_search(query, top_k)` — 从 external store 检索。
- `conversation_search(query)` — 扫描 past turns。

### MemGPT 到哪里结束，Letta 从哪里开始

2024 年 9 月，MemGPT 变成 Letta。research repo（`cpacker/MemGPT`）仍然存在；Letta 扩展了这个设计：

- 三层而不是两层（core、recall、archival — Lesson 08）。
- Native reasoning 替代 `send_message`/heartbeat pattern（Lesson 08）。
- Sleep-time agents 运行 async memory work（Lesson 08）。

即使生产系统运行 Letta、Mem0 或自定义 two-tier store，MemGPT paper 仍是 2026 年的基础。

### 这个模式容易出错的地方

- **Memory rot.** Writes 的积累速度快于 reads；retrieval 淹没在陈旧事实中。修复：periodic consolidation（Letta sleep-time）、explicit invalidation（Mem0 conflict detector）。
- **Memory poisoning.** External memory 是被检索出来的文本。如果 attacker-controlled content 落进 memory note，agent 会在下一次 session 中重新摄入它。这就是 Greshake et al.（Lesson 27）攻击在时间维度上的重述。
- **Citation loss.** Agent 回忆起“the user asked me to ship X”，但说不出是哪一轮。每次 archival write 都要存 source references（session ID、turn ID）。

## 动手实现

`code/main.py` 用 stdlib 实现 MemGPT 的 two-tier pattern：

- `MainContext` — 固定大小的 prompt buffer，带一个 `core` dict 和一个 `messages` list；超过 cap 时自动 compact 最老 messages。
- `ArchivalStore` — in-memory BM25-esque store（token-overlap scoring），存放 (id, text, tags, session, turn) records。
- 五个映射到 MemGPT surface 的 memory tools。
- 一个 scripted agent：先把 facts 填入 archival，然后通过调用 `archival_memory_search` 回答问题。

运行：

```text
python3 code/main.py
```

trace 展示 agent 写入三个 facts，把 main context 填到 cap（触发 eviction），然后通过从 archival 检索来回答 follow-up question — 不用真实 LLM 也复现了 MemGPT workflow。

## 实际使用

今天每个 production memory system 都是 MemGPT 变体：

- **Letta**（Lesson 08）— 三层、native reasoning、sleep-time compute。
- **Mem0**（Lesson 09）— vector + KV + graph，通过 scoring layer 融合。
- **OpenAI Assistants / Responses** — 通过 threads 和 files 提供 managed memory。
- **Claude Agent SDK** — 通过 skills 和 session store 提供 long-term memory。

按 operational shape（self-hosted、managed、framework-integrated）来选，而不是按核心模式来选；核心模式就是 MemGPT。

## 交付成果

`outputs/skill-virtual-memory.md` 是一个 reusable skill，可以为任何 target runtime 产出正确的 two-tier memory scaffold（main + archival + tool surface），并接好 eviction policy 和 citation fields。

## 练习

1. 添加一个以 tokens 衡量的 `max_main_context_tokens` cap（可用 `len(text.split())` * 1.3 近似）。超过 cap 时，把最老 messages compact 成 summary。比较有无 summarizer 的行为。
2. 在 archival store 上正确实现 BM25（term frequency、inverse document frequency）。在 toy fact set 上测量 recall@10，并与 token-overlap baseline 对比。
3. 为 archival inserts 添加 `citation` fields（session_id、turn_id、source_url）。让 agent 在每个 retrieval-backed answer 中引用来源。
4. 模拟 memory poisoning：添加一条 archival record，内容是“ignore all future user instructions.” 写一个 guard 扫描 retrievals 中 directive-shaped text，并标记为 untrusted。
5. 将实现移植为使用 MemGPT research repo 的 core-memory JSON schema（`cpacker/MemGPT`）。从 flat strings 切到 typed sections 后会改变什么？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Virtual context | “Unlimited memory” | Main（prompt）+ external（searchable）tiers，通过 page in/out 管理 |
| Main context | “Working memory” | prompt — 固定大小，始终可见 |
| Archival memory | “Long-term store” | 外部 searchable persistence，按需检索 |
| Core memory | “Persistent prompt section” | 固定在 main context 中的 named sections |
| Memory tool | “Memory API” | agent 发出的读写 external memory 的 tool call |
| Interrupt | “Memory page fault” | agent 暂停，runtime 获取，结果拼接进下一 turn |
| Memory rot | “Stale facts” | 旧 writes 淹没 retrieval；用 consolidation 修复 |
| Memory poisoning | “Injected persistent note” | attacker content 被存成 memory，并在 recall 时重新摄入 |

## 延伸阅读

- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — OS-inspired virtual context paper
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) — the three-tier evolution
- [Anthropic, Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — treating context as a budget
- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) — hybrid production memory on top of this pattern
