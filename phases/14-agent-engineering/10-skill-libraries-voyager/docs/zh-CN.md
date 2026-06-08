# 技能库和终身学习（Voyager）

> Voyager（Wang et al., TMLR 2024）把可执行代码视为 skill。Skills 是命名的、可检索的、可组合的，并由 environment feedback 持续 refinement。这是 Claude Agent SDK skills、skillkit 和 2026 年 skill-library pattern 的 reference architecture。

**类型：** 构建
**语言：** Python (stdlib)
**先修：** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Letta Blocks)
**时间：** ~75 分钟

## 学习目标

- 说出 Voyager 的三个 components：automatic curriculum、skill library、iterative prompting，以及各自角色。
- 解释为什么 Voyager 让 action space 成为 code，而不是 primitive commands。
- 用 stdlib 实现带 registration、retrieval、composition 和 failure-driven refinement 的 skill library。
- 将 Voyager pattern 映射到 2026 年的 Claude Agent SDK skills 和 skillkit ecosystem。

## 要解决的问题

每个 session 都从零重建能力的 agents 会做错三件事：

1. **Waste tokens.** 每个 task 都重新诱发同样的 reasoning。
2. **Lose progress.** session A 学到的 correction 不会迁移到 session B。
3. **Fail on long-horizon composition.** 复杂 tasks 需要 capability hierarchies；one-shot prompts 无法表达它们。

Voyager 的答案：把每个 reusable capability 当作一个命名 code chunk 存入 library，可通过 similarity 检索，可和其他 skills 组合，并由 execution feedback refinement。

## 核心概念

### 三个 components

Voyager（arXiv:2305.16291）围绕三部分组织 agent：

1. **Automatic curriculum.** curiosity-driven proposer 会根据 agent 当前 skill set 和 environment state 选择下一个 task。Exploration 是 bottom-up 的。
2. **Skill library.** 每个 skill 都是 executable code。task 成功时添加新 skills。Skills 通过 query-to-description similarity 检索。
3. **Iterative prompting mechanism.** 失败时，agent 接收 execution errors、environment feedback 和 self-verification output，然后 refine skill。

Minecraft evaluation（Wang et al., 2024）：相比 baselines，unique items 多 3.3x，stone tools 快 8.5x，iron tools 快 6.4x，map traversal 长 2.3x。这些数字是 Minecraft-specific 的，但 pattern 可以迁移。

### Action space = code

多数 agents 发出 primitive commands。Voyager 发出 JavaScript functions。一个 skill 是：

```text
async function craftIronPickaxe(bot) {
  await mineIron(bot, 3);
  await mineStick(bot, 2);
  await placeCraftingTable(bot);
  await craft(bot, 'iron_pickaxe');
}
```

由 sub-skills 组合而成。按 description 和 embedding keyed 存储。检索出来的是 program，不是 prompt。

这就是 2026 年 Claude Agent SDK skill：一个命名、可检索的 code chunk 加 instructions，agent 按需加载。

### Skill retrieval

新 task：“make a diamond pickaxe”。Agent：

1. 嵌入 task description。
2. 查询 skill library，取 top-k similar skills。
3. 检索 `craftIronPickaxe`、`mineDiamond`、`placeCraftingTable` 等。
4. 用检索到的 primitives + 新逻辑组合出新 skill。

这就是 MCP resources（Phase 13）和 Agent SDK skills 实现的模式：在 knowledge/code surface 上做 retrieval，并 scoped 到当前 task。

### Iterative refinement

Voyager 的 feedback loop：

1. Agent 写一个 skill。
2. Skill 在 environment 中运行。
3. 返回三类信号之一：`success`、`error`（with stack trace）、`self-verification failure`。
4. Agent 用该信号作为 context 改写 skill。
5. 循环直到 success 或达到 max rounds。

这是 Self-Refine（Lesson 05）应用到 code generation，并用 environment-grounded verification 作为验证。CRITIC（Lesson 05）是同一个模式，只是 verifier 是 external tools。

### Curriculum 和 exploration

Voyager 的 curriculum module 会根据 agent 已有什么、还没做过什么，提出“build a shelter near the lake”这类 tasks。proposer 使用 environment state + skill inventory 选择略高于当前能力的 task，也就是 exploration sweet spot。

对 production agents 来说，这转化为“what's missing”operator：给定当前 skill library 和一个 domain，我们还没覆盖哪些 skills？团队通常通过 curriculum review 手动实现。

### 这个模式容易出错的地方

- **Skill library rot.** 同一个 skill 用略有差异的 descriptions 添加了 10 次。在 write 时添加 deduplication；retrieval 只返回一个。
- **Composed-skill drift.** Parent skill 依赖一个被 refined 的 child。固定到 v1 的 parent 不会神奇地自动兼容 v3。
- **Retrieval quality.** 当 library 增长到几百个 skills 以上时，基于 skill descriptions 的 vector retrieval 会退化。用 tag filters 和 hard constraints（“only skills with `category=tooling`”）补充。

## 动手实现

`code/main.py` 实现一个 stdlib skill library：

- `Skill` — name、description、code（as string）、version、tags、dependencies。
- `SkillLibrary` — register、search（token overlap）、compose（deps 的 topological sort）和 refine（update 时 version bump）。
- 一个 scripted agent：注册三个 primitive skills，组合第四个，遇到一次 failure，然后 refinement。

运行：

```text
python3 code/main.py
```

trace 展示 library writes、retrieval、composition、failed execution 和 v2 refinement — 从端到端复现 Voyager loop。

## 实际使用

- **Claude Agent SDK skills**（Anthropic）— 2026 年 reference：每个 skill 都有 description、code 和 instructions；在 agent session 中按需加载。
- **skillkit**（npm: skillkit）— 面向 32+ AI coding agents 的 cross-agent skill management。
- **Custom skill libraries** — domain-specific（数据 agents 的 SQL skills、infra agents 的 Terraform skills）。Voyager pattern 可以向下缩放。
- **OpenAI Agents SDK `tools`** — 低端形态；每个 tool 都是 lightweight skill。

## 交付成果

`outputs/skill-skill-library.md` 会为任意 target runtime 生成 Voyager-shaped skill library，包含 registration、retrieval、versioning 和 refinement。

## 练习

1. 给 `compose()` 添加 dependency-cycle detector。当 skill A 依赖 B、B 又依赖 A 时会发生什么？Error 还是 warning？
2. 实现 per-skill version pinning。当 parent skill 组合 child `crafting@1` 时，对 `crafting@2` 的 refinement 不得 silently upgrade parent。
3. 用 sentence-transformers embeddings（或 BM25 stdlib impl）替换 token-overlap retrieval。在 50-skill toy library 上测量 retrieval@5。
4. 添加一个“curriculum”agent：给定当前 library 和 domain description，提出 5 个 missing skills。每周调用它。
5. 阅读 Anthropic 的 Claude Agent SDK skill docs。把 toy library 移植到 SDK 的 skill schema。discoverability 有什么变化？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Skill | “Reusable capability” | 命名 code chunk + description，可通过 similarity 检索 |
| Skill library | “Agent memory of how-to” | skills 的 persistent store，可搜索且可组合 |
| Curriculum | “Task proposer” | 由当前 capability gap 驱动的 bottom-up goal generator |
| Composition | “Skill DAG” | Skills 调用 skills；执行时按拓扑顺序排序 |
| Iterative refinement | “Self-correcting loop” | Env feedback + errors + self-verification 折回下一版本 |
| Action-space-as-code | “Programmatic actions” | 发出 functions，而不是 primitive commands，用于 temporally extended behavior |
| Dedup on write | “Skill collapse” | 近重复 descriptions 合并到一个 canonical skill |

## 延伸阅读

- [Wang et al., Voyager (arXiv:2305.16291)](https://arxiv.org/abs/2305.16291) — the original skill-library paper
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — skills as the 2026 productization
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — skills and subagents in practice
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — the refinement loop underneath Voyager
