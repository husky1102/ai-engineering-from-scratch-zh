# Browser Agents 与长程 Web 任务

> ChatGPT agent（2025 年 7 月）将 Operator 和 deep research 合并为一个 browser/terminal agent，并以 68.9% 达到 BrowseComp SOTA。OpenAI 于 2025 年 8 月 31 日关闭 Operator：产品层发生整合。Anthropic 收购 Vercept 后，让 Claude Sonnet 在 OSWorld 上从低于 15% 提升到 72.5%。WebArena-Verified（ServiceNow, ICLR 2026）修复了原始 WebArena 中 11.3 个百分点的 false-negative rate，并发布 258-task Hard subset。这些数字是真的。attack surface 也是真的：OpenAI 的 preparedness 负责人公开表示，对 browser agents 的 indirect prompt injection“不是一个可以完全 patch 的 bug”。已有记录的 2025-2026 攻击包括：Tainted Memories（Atlas CSRF）、HashJack（Cato Networks），以及 Perplexity Comet 中的一键劫持。

**类型:** Learn
**语言:** Python（stdlib，indirect prompt-injection attack surface model）
**先修:** Phase 15 · 10（Permission modes），Phase 15 · 01（Long-horizon agents）
**时间:** ~45 分钟

## 要解决的问题

browser agent 是一种会读取不可信内容并执行有后果动作的 long-horizon agent。agent 访问的每个页面，都是用户没有写过的输入。每个页面上的每个表单，都是潜在 command channel。2025-2026 年的攻击语料显示这不是假设：Tainted Memories 让攻击者通过 crafted page 把恶意 instructions 绑定到 agent memory；HashJack 把 commands 隐藏在 agent 会访问的 URL fragments 中；Perplexity Comet hijacks 一次点击就能命中。

防御图景并不舒服。OpenAI 的 preparedness 负责人把难听的话说了出来：indirect prompt injection“不是一个可以完全 patch 的 bug”。原因是攻击发生在 agent 的 reading-vs-acting boundary，而这个边界在架构上是模糊的：模型读取的每个 token，原则上都可能被当作 instruction 来读。

本课命名 attack surface，命名 benchmark landscape（BrowseComp、OSWorld、WebArena-Verified），并建模一个最小 indirect-prompt-injection scenario，这样你就能推理 Lessons 14 和 18 中的真实 defenses。

## 核心概念

### 2026 格局，每个系统一段话

**ChatGPT agent（OpenAI）。** 2025 年 7 月发布。统一 Operator（browsing）和 Deep Research（multi-hour research）。于 2025 年 8 月 31 日关闭 standalone Operator。BrowseComp 上 SOTA 为 68.9%；在 OSWorld 和 WebArena-Verified 上数字也很强。

**Claude Sonnet + Vercept（Anthropic）。** Anthropic 的 Vercept 收购聚焦 computer-use capabilities。将 Claude Sonnet 在 OSWorld 上从 <15% 推到 72.5%。Claude Computer Use 以 tool API 形式发布。

**Gemini 3 Pro with Browser Use（DeepMind）。** Browser Use integration 发布 computer-use controls；FSF v3（2026 年 4 月，Lesson 20）专门追踪 ML R&D domain 中的 autonomy。

**WebArena-Verified（ServiceNow, ICLR 2026）。** 修复一个已有充分记录的问题：原始 WebArena 有约 11.3% false-negative rate（实际已解决的任务被标为失败）。Verified release 使用人工策划的 success criteria 重新评分，并增加一个 258-task Hard subset（ICLR 2026 paper, openreview.net/forum?id=94tlGxmqkN）。

### BrowseComp vs OSWorld vs WebArena

| Benchmark | What it measures | Horizon |
|---|---|---|
| BrowseComp | 在开放 web 上限时寻找具体事实 | minutes |
| OSWorld | agent 操作完整 desktop（mouse、keyboard、shell） | tens of minutes |
| WebArena-Verified | 模拟网站中的 transactional web tasks | minutes |
| Hard subset | 带 multi-page state transitions 的 WebArena-Verified tasks | tens of minutes |

不同轴。高 BrowseComp 分数说明 agent 能找事实；它不说明 agent 能订机票。OSWorld 分数更接近“它能不能在我的 desktop 上工作”。WebArena-Verified 更接近“它能不能完成一个 flow”。任何生产决策都需要匹配 task distribution 的 benchmark。

### 攻击面，逐一命名

1. **Indirect prompt injection。** 不可信页面内容包含 instructions。agent 读取它们。agent 执行它们。公开例子：2024 Kai Greshake et al.、2025 Tainted Memories paper、2026 HashJack（Cato Networks）。
2. **URL fragment / query injection。** 被 crawled URL 的 `#fragment` 或 query string 包含 commands。它从不被可见渲染；仍位于 agent context 内。
3. **Memory-binding attacks。** 页面指示 agent 写入 persistent memory（Lesson 12 覆盖 durable state）。下一次 session 中，memory 在没有可见触发器的情况下触发 payload。
4. **Authenticated sessions 上的 CSRF-shaped attacks。** Tainted Memories 类别：agent 登录在某处；攻击者页面发起 agent 会带着用户 cookies 执行的 state-changing requests。
5. **One-click hijack。** 一个视觉上无害的 button 搭载 agent 会跟随执行的后续 payload。Comet 类别。
6. **agent host surface 中的 Content-Security-Policy holes。** rendering 和 tool layers 本身也可能是 attack vectors；browser-in-a-browser-agent stack 很宽。

### 为什么“不能完全 patch”

攻击与 agent 的能力同构。agent 必须读取不可信内容才能工作。agent 读取的任何内容都可能包含 instructions。agent 遵循的任何 instructions 都可能与用户真实请求错位。防御（trust boundaries、classifiers、tool allowlists、consequential actions 上的 HITL）会提高攻击成本并降低 blast radius。它们不会闭合这个类别。

这与 Lob's theorem（Lesson 8）是相同推理模式：agent 不能证明下一个 token 是安全的；它只能建立一个让不安全 tokens 更可检测的系统。

### 真正在发布的防御姿态

- **Read / write boundary。** Reading 永远不应产生后果。Writing（提交表单、发布内容、调用有副作用工具）如果由 trust boundary 外的内容发起，就需要新的人工批准。
- **按任务设置 tool allowlist。** agent 可以 browse；除非为任务明确启用，否则不能发起 wire transfer。Lesson 13 覆盖 budgets。
- **Session isolation。** Browser agent sessions 只用 scoped credentials 运行。无 production auth，无 personal email。保留每个 HTTP request 的日志用于 audit。
- **Content sanitizer。** Fetched HTML 在拼接进 model context 前会去除 known-bad patterns。（减少容易攻击；不能阻止复杂 payloads。）
- **Consequential actions 上的 HITL。** propose-then-commit 模式（Lesson 15）。
- **Memory 上的 canary tokens。** 如果一个 memory entry 触发，用户会看到它（Lesson 14）。

## 实际使用

`code/main.py` 建模一个微型 browser-agent run，目标是三个 synthetic pages。一个页面是 benign，一个在 visible text 中包含 direct prompt-injection blob，一个包含 URL-fragment injection（不可见但在 agent context 内）。脚本展示（a）naive agent 会做什么，（b）read/write boundary 抓住什么，（c）sanitizer 抓住什么，（d）两者都抓不住什么。

## 交付成果

`outputs/skill-browser-agent-trust-boundary.md` 为一个 proposed browser-agent deployment 划定范围：它触及哪些 trust zones、被授权写入什么，以及第一次运行前必须具备哪些 defenses。

## 练习

1. 运行 `code/main.py`。指出 sanitizer 抓住但 read/write boundary 抓不住的攻击，以及只有 read/write boundary 抓得住的攻击。

2. 扩展 sanitizer，以检测一类 HashJack-style URL-fragment injection。测量在带有合法 fragments 的 benign URLs 上的 false-positive rate。

3. 选择一个你熟悉的真实 browser-agent workflow（例如“book a flight”）。列出每个 read 和每个 write。标出哪些 writes 需要 HITL，并说明原因。

4. 阅读 WebArena-Verified ICLR 2026 paper。指出原始 WebArena scoring 不可靠的一类任务，并解释 Verified subset 如何解决它。

5. 为 browser-agent setting 设计一个 memory canary。你会存储什么，存在哪里，什么会触发 alarm？

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| Indirect prompt injection | “坏页面文本” | agent 读取的页面中的不可信内容包含 agent 会执行的 instructions |
| Tainted Memories | “Memory attack” | agent 将攻击者提供的 instruction 写入 durable memory；下一 session 触发 |
| HashJack | “URL fragment attack” | 隐藏在 URL fragment / query string 中的 payload 位于 agent context 内但不可见渲染 |
| One-click hijack | “坏 button” | 可见 affordance 搭载 agent 会执行的后续 payload |
| BrowseComp | “Web search benchmark” | 在开放 web 上寻找具体事实；minute-scale horizon |
| OSWorld | “Desktop benchmark” | 完整 OS control；multi-step GUI tasks |
| WebArena-Verified | “修复后的 web-task benchmark” | ServiceNow 重新评分的 WebArena，带 Hard subset |
| Read/write boundary | “Side-effect gate” | reading 永远不产生后果；如果内容 out-of-trust，则 writing 需要新的批准 |

## 延伸阅读

- [OpenAI — Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) — Operator 与 deep research 的合并；BrowseComp SOTA。
- [OpenAI — Computer-Using Agent](https://openai.com/index/computer-using-agent/) — Operator lineage，以及后来成为 ChatGPT agent 的 architecture。
- [Zhou et al. — WebArena](https://webarena.dev/) — 原始 benchmark。
- [WebArena-Verified (OpenReview)](https://openreview.net/forum?id=94tlGxmqkN) — ICLR 2026 fixed-subset paper。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 包含 computer-use agents 的 attack-surface discussion。
