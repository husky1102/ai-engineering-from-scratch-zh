# Action Budgets、Iteration Caps 与 Cost Governors

> 一个中型 e-commerce agent 的月度 LLM 成本在团队启用 “order-tracking” skill 后，从 $1,200 跳到 $4,800。这不是 pricing bug，而是一个 agent 找到了新的 loop，并在其中持续花钱。Microsoft 的 Agent Governance Toolkit（2026 年 4 月 2 日）把这类问题的防御 codify 为：per-request `max_tokens`、per-task token 和 dollar budgets、per-day/month caps、iteration caps、tiered model routing、prompt caching、context windowing、昂贵 actions 上的 HITL checkpoints、budget breach 时的 kill switches。Anthropic 的 Claude Code Agent SDK 用不同名字交付了同样的 primitives。Financial velocity limits，例如 10 分钟内超过 $50 就切断 access，比 monthly caps 更快捕获 loop。

**类型：** 学习
**语言：** Python (stdlib, layered cost-governor simulator)
**先修：** Phase 15 · 10 (Permission modes), Phase 15 · 12 (Durable execution)
**时间：** ~60 分钟

## 要解决的问题

Autonomous agents 每一轮都会花真钱。Chatbot 的坏输出是一条坏回复；agent 的坏 loop 是账单。行业文档中给这个 failure mode 的术语是 “Denial of Wallet”：agent 持续 reasoning、持续 tool-calling、持续 billing，而且没有任何东西阻止它，因为系统没有被设计成会阻止它。

修复方式不是一个数字，而是在不同 time scales 和 granularities 上叠一组 limits：per-request、per-task、per-hour、per-day、per-month。设计良好的 stack 会在几分钟内抓住 runaway loop，在几小时内抓住 slow leak，在一天内抓住 bad release。当 agent 是 long-horizon 且 autonomous 时，同一套 stack 才能让 budget 仍然成立。

这是一节工程课：数学很简单，团队失败在 discipline。下面的 limits 要么来自 Microsoft Agent Governance Toolkit，要么来自 Anthropic Claude Code Agent SDK docs。

## 核心概念

### Cost-governor stack

1. **每个 request 的 `max_tokens`。** 简单。防止单个 call 产生无界 completion。
2. **Per-task token budget。** 在整个 run 内，不要超过 N tokens。到 cap 时 hard stop。
3. **Per-task dollar budget。** 和 tokens 一样，但单位是 currency。Claude Code 中是 `max_budget_usd`。
4. **Per-tool call cap。** 不超过 N 次 `WebFetch` calls、N 次 `shell_exec` calls 等。
5. **Iteration cap (`max_turns`)。** Agent loop iterations 总数；防止 infinite reasoning loops。
6. **Per-minute / per-hour / per-day / per-month cap。** Rolling windows。用不同 time scales 捕获 leaks。
7. **Financial velocity limit。** 例如 “如果 10 分钟内 spend 超过 $50，就 cut access”。在 monthly caps 触发前捕获 loop-based burn。
8. **Tiered model routing。** 默认使用较小 model；只有 classifier 判断 task 值得时才升级到更大 model。
9. **Prompt caching。** System prompt 和 stable context 存在 provider cache 中；重新发送的 token cost 接近零。
10. **Context windowing。** 通过 compaction / summarization 让 active context 低于阈值；直接降低 token-cost。
11. **昂贵 actions 上的 HITL checkpoints。** 在已知昂贵的 action 之前（长 tool call、大 download、昂贵 model upgrade），要求 human tap。
12. **Budget breach 上的 kill switch。** 任何 cap 触发时 session abort。记录 cap；需要单独的 re-enable path。

### 为什么是 stack，而不是一个 cap

单个 monthly cap 只会在钱包已经空了之后才抓住 runaway agent。单个 per-request cap 无法抓住 session 级问题。不同 failure modes 需要不同 time scales：

- **Runaway loop**（agent 卡在 5 秒 retry 中）：由 velocity limit 捕获。
- **Slow leak**（agent 每个 task 做了约 2x expected work）：由 daily cap 捕获。
- **Bad release**（新版本使用 5x tokens）：由 weekly / monthly cap 捕获。
- **Legitimate surge**（真实需求，不是 bug）：由 hour / day cap 捕获，并留下清晰 log。

### Claude Code 的 budget surface

Claude Code Agent SDK 暴露（public docs）：

- `max_turns`：iteration cap。
- `max_budget_usd`：dollar cap；breach 时 session abort。
- `allowed_tools` / `disallowed_tools`：tool allowlist 和 denylist。
- Tool use 前的 hook points，用于 custom cost-accounting。

把这些与 permission-mode ladder（Lesson 10）组合起来。没有 `max_budget_usd` 的 `autoMode` session 是 ungoverned autonomy。Anthropic 明确把 Auto Mode framed 为需要 budget controls；classifier 与 cost 是正交的。

### EU AI Act、OWASP Agentic Top 10

Microsoft 的 Agent Governance Toolkit 覆盖 OWASP Agentic Top 10 和 EU AI Act Article 14（human oversight）要求。对于 EU 生产环境，logging 和 cap enforcement 不是可选项。

### 观察到的 $1,200 → $4,800 案例

Microsoft docs 中的真实案例：一个 e-commerce agent 在加入新 tool 后，monthly cost 翻了三倍。这个 tool 允许 agent 在每个 session 中 poll order status。没有 loop detection。没有 per-tool cap。没有 week-over-week growth alert。修复方式是 per-tool cap 加 daily-growth alert。这是一个模板：每个新 tool surface 都是新的潜在 loop；每个新 tool 都需要自己的 cap 和 alert。

## 实际使用

`code/main.py` 模拟一个有 layered cost-governor stack 和没有该 stack 的 agent run。模拟 agent 会在若干 turns 后 drift 进入 polling loop；layered stack 会在 velocity window 内捕获它，而 single monthly cap 要到几天后才会触发。

## 交付成果

`outputs/skill-agent-budget-audit.md` 会 audit 一个拟议 agent deployment 的 cost-governor stack，并标记缺失的 layers。

## 练习

1. 运行 `code/main.py`。确认在 polling-loop trajectory 上，velocity limit 会先于 iteration cap 触发。然后禁用 velocity limit，测量 agent 在 iteration cap 捕获之前 “spends” 了多少。

2. 为 browser agent（Lesson 11）设计一组 per-tool caps。哪个 tool 需要最紧的 cap？哪个 tool 可以在无风险情况下 unbounded 运行？

3. 阅读 Microsoft Agent Governance Toolkit docs。列出 toolkit 命名的每种 cap type。把每一种映射到一个 failure mode（runaway loop、slow leak、bad release、surge）。

4. 为一个真实任务的 unattended overnight run 定价（例如 “triage 50 issues in a repo”）。把 `max_budget_usd` 设为 point estimate 的 2x。说明为什么是 2x。

5. Claude Code 的 `max_budget_usd` 在 session aggregate cost 上触发。设计一个你会从外部 enforce 的 complementary velocity limit。什么会触发 cut-off，re-enable 是什么样？

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| Denial of Wallet | “Runaway bill” | Agent loop 在没有 cap 阻止的情况下生成 spend |
| max_tokens | “Per-request cap” | 单个 completion 大小的上限 |
| max_turns | “Iteration cap” | 一个 session 中 agent loop iterations 的上限 |
| max_budget_usd | “Dollar kill switch” | Session cost cap；breach 时 abort |
| Velocity limit | “Rate cap” | 短窗口内 spend 的限制（例如 $50 / 10 min） |
| Tiered routing | “Small model first” | 默认便宜 model；只有 classifier 判断值得时才 escalate |
| Prompt caching | “Cached system prompt” | Provider-side cache 将重新发送 token cost 降到接近零 |
| HITL checkpoint | “Human approval gate” | 昂贵 action 前需要 human tap |

## 延伸阅读

- [Anthropic Claude Code Agent SDK — agent loop and budgets](https://code.claude.com/docs/en/agent-sdk/agent-loop) — `max_turns`、`max_budget_usd`、tool allowlists。
- [Microsoft Agent Framework — human-in-the-loop and governance](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — cost-governor checkpoints。
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — provider-side cost controls。
- [Anthropic — Prompt caching (Claude API docs)](https://platform.claude.com/docs/en/prompt-caching) — caching mechanics。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — long-horizon agents 的 cost profile。
