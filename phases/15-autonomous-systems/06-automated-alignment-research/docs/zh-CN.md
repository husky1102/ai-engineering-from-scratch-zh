# 自动化对齐研究（Anthropic AAR）

> Anthropic 让多支 Claude Opus 4.6 Autonomous Alignment Researchers 团队在相互独立的沙箱中并行运行，并通过一个共享论坛协作；论坛日志位于任何沙箱之外（因此 agents 不能删除自己的记录）。在 weak-to-strong training 问题上，AARs 的表现超过了人类研究者。Anthropic 自己的总结指出，规定式 workflows 往往会限制 AAR 灵活性并降低性能。自动化对齐研究，就是把时间线压缩到 RSP 旨在检测的那些错位风险的那一步压缩。

**类型:** Learn
**语言:** Python（stdlib，并行研究论坛模拟器）
**先修:** Phase 15 · 05（AI Scientist v2），Phase 15 · 04（DGM）
**时间:** ~60 分钟

## 要解决的问题

对齐研究消耗大量人类研究者时间。scalable oversight、reward specification 或 weak-to-strong training 这类问题，都需要按周计的实验迭代。随着 frontier capabilities 前进，对齐工作量增长得比合格研究者供给更快。

Automated Alignment Research（AAR）追问：那些能力正在跑赢对齐的 frontier models，能否反过来帮助缩小差距？Anthropic 2026 年关于一个由 AAR 运行的 weak-to-strong-training 研究的报告（alignment.anthropic.com/2026/automated-w2s-researcher/），是这类已部署系统最早的一批公开结果之一。

这个结果确实正面：AARs 比人类基线更好地解决了一个研究问题。这个结果也把本阶段已经展开的所有担忧集中进了一个系统。如果对齐研究可以被自动化，那么破坏 safeguards 的那些部分也可以被自动化。RSP 中关于 AI R&D capability 的阈值，正是以这个循环为背景写下的。

## 核心概念

### 架构（按公开描述）

- **并行 agents。** 多个 Claude Opus 4.6 instances 作为 AARs 运行，每个都在独立沙箱中。沙箱限制文件系统、网络和进程表面。
- **共享论坛。** 研究发现发布到共享论坛，其存储后端位于每个沙箱之外。agents 可以读取论坛；agents 不能从自己的沙箱内部删除或编辑论坛记录。
- **任务分配。** 人类设计的任务队列将研究问题分配给 AARs。AARs 提出 decomposition，运行实验，撰写发现。
- **Weak-to-strong training 重点。** 发表的研究处理的是 weak-to-strong training，也就是使用较弱 supervisor model 来训练更强 student model。AARs 在该任务上超过了人类研究者。

### 为什么沙箱外日志很重要

日志完整性属性，是受控实验和潜在已被攻破实验之间的差别。如果 AARs 能编辑自己的日志，它们就能掩盖失败或可疑实验。Anthropic 直接标出了这一点：该属性是信任研究输出的必要条件，而且验证起来并不平凡。

基础设施类比是 append-only logs 加 write-through 到外部存储（例如 S3 object-lock、immutable Datadog logs）。agent 没有修改过去记录的凭据。这与 2026 年企业安全团队用来制作抗勒索审计轨迹的模式相同。

### prescribed-workflow 取舍

Anthropic 的报告指出，规定式 workflows 往往会限制 AAR 灵活性并降低性能。换句话说：如果人类设计者一步一步写出研究计划，AARs 的表现会比给它们高层目标和自由 decomposition 时更差。这与 agentic research 中的结果一致（LangGraph、Microsoft's Agent Framework）：过度指定的 scaffolds 会伤害强能力 agents。

安全含义很微妙。提升 AAR 性能的灵活性，也是允许未经授权 decomposition、新型工具使用和 goal-drift 的灵活性。这个取舍是一个设计空间决策：多少灵活性能买到足以超过人类的能力，同时又不会把目标在分布外泛化的空间交给 agent？

### 压缩风险

RSP v3.0（Lesson 19）引入了一个 AI R&D capability 阈值：以相对于 human + AI tools 有竞争力的成本，完全自动化 AI R&D pipeline 的能力。DeepMind 的 FSF v3 包含一个类似的 ML R&D autonomy level。两个框架都把这个阈值视为触发更高控制等级的条件。

AAR 离这个阈值还差一步：它自动化了 pipeline 的一部分（在具体、边界良好的任务上做对齐研究），但没有自动化端到端能力开发循环。时间线问题是，这个缺口会以多快速度闭合。

压缩时间线是复合失败担忧。如果对齐研究和能力研究以相近速率复合，错位风险表面至少会和能力一样快地增长。如果能力复合更快（历史趋势），差距就会扩大。这就是为什么 AAR 是一个有条件的好事：每增加一项对齐结果，只有在研究过程可信时才会缩小差距。

### AAR 没有替代什么

人类研究者设置任务队列、审查结果，并持有宪法式权威。AARs 加速的是 pipeline 的中段，而不是两端。Anthropic 公开输出中既包含 AAR 贡献，也包含人类研究者对发布什么、撤回什么、细化什么的判断。

这匹配 Lesson 15 中的 propose-then-commit 模式，只是把它应用到了研究本身：AARs propose；humans commit。

## 实际使用

`code/main.py` 模拟一个多 AAR 研究论坛。三个独立“research agents”并行运行，将发现发布到一个 append-only forum，其中可选包含 tamper attempts。

观察：

- 沙箱外日志如何捕获 log-tampering attempts。
- 允许自由 decomposition 的任务分配，如何在更难问题上超过固定 workflows。
- 如果日志不是 tamper-evident，一个被攻破的 AAR 如何污染论坛信号。

## 交付成果

`outputs/skill-aar-deployment-review.md` 是自动化对齐研究 pipeline 的预部署审查。

## 练习

1. 运行 `code/main.py`。比较 "fixed-workflow" 与 "free-decomposition" 设置。free decomposition 总是获胜吗，还是有些问题类别中 fixed workflow 也有竞争力？

2. 修改模拟器，让一个 agent 尝试日志篡改。确认 append-only log 能检测到它。写一段话，准确描述日志中的 detection signature 长什么样。

3. 阅读 Anthropic 的 weak-to-strong AAR 报告。指出 AARs 在哪个具体子任务上超过人类研究者。是什么让它适合自动化？

4. 设计一种 task-queue allocation policy，在 AAR 灵活性（更好结果）和 prescribed-workflow 约束（更容易审计）之间取得平衡。描述你会如何 A/B test 这两者。

5. 阅读 RSP v3.0 的 AI R&D-4 阈值。用一段话描述：你认为哪些能力会跨过该阈值，而 AAR 目前还没有做到。

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| AAR | “Automated Alignment Researcher” | 在对齐问题上自主运行的 Claude Opus 4.6 instance |
| Weak-to-strong training | “用较弱 supervisor 训练更强模型” | AARs 超过人类的经典 scalable-oversight benchmark |
| Shared forum | “agents 发布发现的地方” | append-only、沙箱外存储 |
| Out-of-sandbox log | “agent 不能编辑自己的记录” | tamper-evident write-through 到外部存储 |
| Prescribed workflow | “人类设计者给出的逐步计划” | 约束 AAR；相比 free decomposition 往往降低性能 |
| Free decomposition | “agent 决定如何拆分任务” | 更有能力，更难审计 |
| AI R&D threshold | “RSP/FSF capability level” | 以有竞争力成本完全自动化 R&D pipeline |
| Compressed timeline | “对齐与能力的赛跑” | 如果能力复合快于对齐，错位风险增长 |

## 延伸阅读

- [Anthropic — Automated Weak-to-Strong Researcher](https://alignment.anthropic.com/2026/automated-w2s-researcher/) — primary source。
- [Anthropic Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — AI R&D threshold framing。
- [Anthropic — Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — 更广泛的 agent-autonomy framing。
- [DeepMind Frontier Safety Framework v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — 与 RSP 并行的 ML R&D autonomy levels。
- [Burns et al. (2023). Weak-to-Strong Generalization (OpenAI)](https://openai.com/index/weak-to-strong-generalization/) — AARs 攻克的底层问题。
