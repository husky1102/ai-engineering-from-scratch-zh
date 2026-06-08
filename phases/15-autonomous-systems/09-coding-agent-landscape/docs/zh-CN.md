# 自主编码 Agent 格局（2026）

> SWE-bench Verified 在不到三年里从 4% 走到 80.9%。同一个 Claude Sonnet 4.5 在 SWE-agent v1 上得分 43.2%，在 Cline autonomous 上得分 59.8%——围绕模型的 scaffolding 如今和模型本身一样重要。OpenHands（原 OpenDevin）是最活跃的 MIT-licensed platform，其 CodeAct loop 会直接在沙箱中执行 Python actions，而不是 JSON tool calls。头条数字隐藏了一个方法论问题：SWE-bench Verified 的 500 个任务中有 161 个只需要 1-2 行修改，而同一批 frontier models 在 SWE-bench Pro（10+ 行任务）上只有 23-59%。

**类型:** Learn
**语言:** Python（stdlib，CodeAct 与 JSON tool-call 对比）
**先修:** Phase 14 · 07（Tool use），Phase 15 · 01（Long-horizon agents）
**时间:** ~45 分钟

## 要解决的问题

“哪个 coding agent 最好”是错误问题。正确问题是：在匹配我工作的 task distribution 上，使用我会在生产中运行的 scaffolding，我能得到怎样的端到端可靠性？

2022 到 2026 年之间，领域学到的是 scaffolding 承重：retrieval layer、planner、sandbox、edit-verify loop、feedback format。Claude Sonnet 4.5 在 SWE-agent v1 上的 SWE-bench Verified 得分是 43.2%；同一模型放进 Cline 的 autonomous scaffold 得分是 59.8%。同一组 weights，相差 16.6 个绝对百分点。base model 是组件；loop 才是产品。

伴随问题是 benchmark saturation 会隐藏 regressions。SWE-bench Verified 已接近饱和，而 easy-task tail（500 个任务中有 161 个只需要 ≤2 行）会拉高顶部分数。真实世界质量更适合在 SWE-bench Pro 这类分布上测量（10+ 行修改），同样的领先系统在那里仍然只有 23-59%。

## 核心概念

### SWE-bench，一段话

SWE-bench（Jimenez et al.）取真实 GitHub issues 和 ground-truth patches，要求 agent 产出一个能让 test suite 通过的 patch。SWE-bench Verified（OpenAI, 2024）是一个经人工策划的 500 任务子集，移除了模糊和损坏任务。SWE-bench Pro 是更难的后继版本：任务要求 10+ 行修改，当前 frontier agents 得分在 23-59%。

### 2022 → 2026 曲线真正展示了什么

- **2022**：研究模型在 raw SWE-bench 上约 4%。
- **2024**：GPT-4 + Devin-style scaffolding 约 14%；SWE-agent 约 12%。
- **2025**：Claude 3.5/3.7 Sonnet 在 Aider 和 SWE-agent 内推动到 40-55% 区间。
- **2026**：Claude Sonnet 4.5 和 frontier competitors 在 SWE-bench Verified 上达到 70-80%+。Epoch AI 的 leaderboard 实时跟踪这些分数。

斜率来自三个复合来源：更好的 base models、更好的 scaffolding（CodeAct、reflection、verifier loops），以及更好的 benchmarks（Verified 移除噪声）。

### CodeAct vs JSON tool calls

OpenHands（All-Hands-AI, arXiv:2407.16741，原 OpenDevin）押注了一种特定架构：模型不是发出由 host 解码和执行的 JSON tool calls，而是发出 Python code，由 Jupyter-style kernel 在沙箱中运行。agent 可以在一个 action 内遍历文件、串联工具，并捕捉自己的异常。

取舍：

- **JSON tool calls**：每个 action 是一个 turn；易审计；compositionality 有限；默认更安全，因为每次 call 都通过显式 validator。
- **CodeAct**：一个 action 可以是一整个 program；compositional；需要 hardened sandbox（OpenHands 使用 Docker isolation）；失败模式包括沙箱 runtime 允许的任何事情。

两种架构都在生产中。CodeAct 在开放平台中占主导（OpenHands、smolagents）。JSON tool calls 在 managed services 中仍占主导（Anthropic Managed Agents、OpenAI Assistants），在那里 provider 控制 executor。

### 2026 格局中的 scaffolds

| Scaffold | License | Execution model | Notable property |
|---|---|---|---|
| OpenHands（OpenDevin） | MIT | Docker 中的 CodeAct | 最活跃的开放平台；event-stream replayable |
| SWE-agent | MIT | Agent-Computer Interface（ACI） | 第一个端到端 SWE-bench scaffold |
| Aider | Apache-2 | 本地 repo 中 edit-via-diff | 最小 scaffold，强 regression stability |
| Cline | Apache-2 | 带 tool policy 的 VS Code agent | Sonnet 4.5 上最高分开放 scaffold |
| Devin（Cognition） | Proprietary | Managed VM + planner | 第一个“AI software engineer”产品类别 |
| Claude Code | Proprietary | Permission modes + routines | Lesson 10 详细覆盖 agent loop |

### 为什么 scaffolding 占主导

一次 coding run 是一条 long-horizon trajectory（Lesson 1）。可靠性会跨步骤复合。scaffolding 在三个地方买到分数：

1. **Retrieval**：找到应该读取的文件是沉默瓶颈。SWE-agent 的 ACI、OpenHands 的 file-index、Aider 的 repo-map 都在攻克它。
2. **Verifier loop**：运行测试、读取 stack traces 并重新尝试，会在 SWE-bench 上带来 10+ 分的差异。
3. **Failure containment**：出错时能回滚的 sandbox 能防止损害复合。有无 verifier loop 的同一个模型，看起来像两个不同产品。

### Benchmark saturation 与真实分布

OpenHands 作者和 Epoch AI 都指出 SWE-bench Verified 有一条 easy tail：500 个任务中有 161 个只需要 1-2 行修改。高分部分由这条 tail 驱动。SWE-bench Pro 限制为 10+ 行修改，即使 frontier systems 也只返回 23-59% 的分数。你的生产分布几乎肯定更接近 Pro，而不是 Verified。

选择 agent 的含义：在你自己的 bug backlog 上运行一个 Pro-like subset。真正重要的是它在代表你所交付工作的任务上的分数。

## 实际使用

`code/main.py` 在固定 mini-task distribution 上比较两个玩具 agent scaffolds：

1. 一个 **JSON tool-call** scaffold，每个 turn 只执行一个 action。
2. 一个 **CodeAct** scaffold，每个 action 可以发出一个小 Python snippet。

两者都使用 stub “model”（确定性规则），因此对比会隔离 scaffold 与 model quality。输出展示 CodeAct scaffold 以更少 turns 解出更多任务，代价是更大的 per-action blast radius。

## 交付成果

`outputs/skill-scaffold-audit.md` 帮助你在采用前审计一个 proposed coding-agent scaffold：retrieval quality、verifier presence、sandbox isolation，以及 benchmark-to-distribution fit。

## 练习

1. 运行 `code/main.py`。每种 scaffold 在同一个 task set 上分别需要多少 turns？各自的 per-action blast radius 是什么？

2. 阅读 OpenHands paper（arXiv:2407.16741）。论文认为 CodeAct 在复杂任务上胜过 JSON tool calls。指出论文承认的一个 failure mode，并用一句话说明该模式在生产中何时会占主导。

3. 从你的 bug backlog 中选择一个需要跨两个文件修改 10+ 行的任务。估计一个 frontier model 在（a）JSON tool calls 和（b）CodeAct 下的端到端成功概率。说明差距理由。

4. SWE-bench Verified 有 161 个 single-file、1-2 行任务。构造一个排除它们的分数。leaderboard 会如何洗牌？

5. 阅读 “Introducing SWE-bench Verified”（OpenAI）。解释用于移除 ambiguous tasks 的具体 methodology，并说出一种该 curation 会漏掉的类别。

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| SWE-bench | “Coding benchmark” | 带 ground-truth patches 和 test suites 的真实 GitHub issues |
| SWE-bench Verified | “清洗后的子集” | 500 个人工策划任务，存在 easier-tail |
| SWE-bench Pro | “更难子集” | 10+ 行修改；frontier 位于 23-59% |
| CodeAct | “Code-as-action” | agent 发出 Python；Jupyter-style kernel 在沙箱中执行 |
| JSON tool call | “Function calling” | 每个 action 都是执行前会验证的 structured JSON payload |
| Scaffold | “Agent framework” | 围绕 base model 的 retrieval + planner + executor + verifier loop |
| ACI（Agent-Computer Interface） | “SWE-agent 的格式” | 为 LLM ergonomics 设计的 command set，而不是人类 shells |
| Verifier loop | “Test-and-retry” | 运行测试、读取输出、修订 patch；最大的非模型可靠性增益 |

## 延伸阅读

- [Jimenez et al. — SWE-bench](https://www.swebench.com/) — 原始 benchmark 和 methodology。
- [OpenAI — Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — curated subset 的构建方式。
- [Wang et al. — OpenHands: An Open Platform for AI Software Developers](https://arxiv.org/abs/2407.16741) — CodeAct architecture 和 event-stream design。
- [Epoch AI — SWE-bench leaderboard](https://epoch.ai/benchmarks) — live-tracked scores。
- [Anthropic — Measuring agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — long-horizon coding-agent reliability framing。
