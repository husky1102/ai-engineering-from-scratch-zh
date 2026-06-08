# 失败模式：为什么 Agent 会坏掉

> MASFT（Berkeley，2025）把多 Agent 失败模式归纳为 3 类 14 种。Microsoft 的 Taxonomy 说明了既有 AI 失败如何在 agentic 场景中被放大。行业现场数据收敛到五种反复出现的模式：幻觉动作、范围蔓延、级联错误、上下文丢失、工具误用。

**类型:** 学习 + 构建
**语言:** Python (stdlib)
**先修:** Phase 14 · 05 (Self-Refine and CRITIC), Phase 14 · 24 (Observability)
**时间:** ~60 分钟

## 学习目标

- 说出 MASFT 的三类失败，以及每类至少四个具体模式。
- 解释为什么 agentic 失败会放大既有 AI 失败模式（偏见、幻觉）。
- 描述行业中反复出现的五种模式及其缓解方式。
- 实现一个 stdlib 检测器，用 failure-mode 标签标注 agent traces。

## 要解决的问题

团队发布的 agents 在 90% 的 traces 上能工作。那 10% 的失败不是随机噪声，而是落入少数反复出现的类别。只要你能命名它们，就能监控它们并修复它们。

## 核心概念

### MASFT（Berkeley，arXiv:2503.13657）

Multi-Agent System Failure Taxonomy。14 种失败模式聚成 3 类。标注者间 Cohen's Kappa 为 0.88，说明这些类别可以被可靠地区分。

核心主张：失败是多 Agent 系统中的根本设计缺陷，不是靠更好的基础模型就能修掉的 LLM 局限。

### Microsoft Taxonomy of Failure Mode in Agentic AI Systems

- 既有 AI 失败（偏见、幻觉、数据泄漏）会在 agentic 场景中被放大。
- 自主性会带来新的失败：规模化的非预期动作、工具误用、任务漂移。
- 这份 whitepaper 是 agentic 产品的风险登记册。

### Characterizing Faults in Agentic AI（arXiv:2603.06847）

- 失败来自编排、内部状态演化和环境交互。
- 不只是“坏代码”或“坏模型输出”。

### LLM Agent Hallucinations Survey（arXiv:2509.18970）

两种主要表现：

1. **Instruction-following Deviation** — agent 没有遵循 system prompt。
2. **Long-range Contextual Misuse** — agent 忘记或误用早期轮次的上下文。

子意图错误：Omission（漏掉步骤）、Redundancy（重复步骤）、Disorder（步骤顺序错误）。

### 行业中反复出现的五种模式

Arize、Galileo、NimbleBrain 2024-2026 的现场分析收敛到：

1. **幻觉动作。** Agent 调用不存在的工具，或编造参数。
2. **范围蔓延。** Agent 把任务扩展到用户请求之外（创建额外 PR、发送额外邮件）。
3. **级联错误。** 一个错误调用触发下游影响。一个幻觉 SKU 会触发四个 API 调用，变成多系统事故。
4. **上下文丢失。** 长周期任务忘记早期轮次的约束。
5. **工具误用。** 用错误参数调用正确工具，或者完全调用了错误工具。

级联是杀手级问题。Agents 无法区分“我失败了”和“任务不可能完成”，而且经常在 400 错误上幻觉出成功消息来闭合循环。

### 缓解：每一步都设门禁

在推理链的每一步设置自动验证门禁，检查事实依据是否与环境状态一致。具体来说：

- 每步 safety classifier（Lesson 21）。
- Tool-call 参数校验（Lesson 06）。
- 将检索内容与已知事实交叉检查（Lesson 05，CRITIC）。
- 通过重新探测状态来检测成功幻觉（文件真的创建了吗？）。

### 失败监控容易走偏的地方

- **只标记崩溃。** 大多数 agent 失败会产出看起来有效的输出。需要内容层面的检查。
- **没有 baseline。** Drift detection 需要 last-known-good；没有它，你无法判断“这正在变差”。
- **过度告警。** 每个失败都发 page。应该聚类并限流。

## 动手实现

`code/main.py` 实现了一个 stdlib failure-mode tagger：

- 一个覆盖五种模式的合成 trace 数据集。
- 每种模式一个 detector 函数（基于 tool calls、outputs、repeat actions 的特征模式）。
- 一个 tagger，给每条 trace 打标签并报告模式分布。

运行：

```text
python3 code/main.py
```

输出：每条 trace 的标签 + 聚合分布，这是 Phoenix trace clustering 所揭示内容的廉价复现。

## 实际使用

- **Phoenix** 用于生产 drift clustering（Lesson 24）。
- **Langfuse** 用于 session replay + annotation。
- **Custom** 用于你的 observability 平台无法检测的领域特定特征。

## 交付成果

`outputs/skill-failure-detector.md` 会生成面向你领域、连接到 trace store 的 failure-mode detectors。

## 练习

1. 添加一个“成功幻觉”检测器：agent 返回成功，但目标状态没有变化。
2. 标注你构建过的产品中的 100 条真实 traces。哪种模式占主导？修复它的成本是什么？
3. 实现一个“cascade radius”指标：给定第 N 步的失败，它影响了多少下游步骤？
4. 阅读 MASFT 的 14 种失败模式。挑出适用于你产品的三种。编写 detectors。
5. 将一个 detector 接入 CI job：如果 >=5% 的 traces 打上某种模式标签，就让 build 失败。

## 关键术语

| 术语 | 人们通常怎么说 | 它实际意味着什么 |
|------|----------------|------------------|
| MASFT | “Multi-agent failure taxonomy” | Berkeley 的 14 模式分类 |
| 级联错误 | “Ripple failure” | 一个早期错误传播到 N 个步骤 |
| 上下文丢失 | “忘了约束” | 长周期轮次丢掉早期轮次事实 |
| 工具误用 | “工具错了 / 参数错了” | 调用合法，但调用方式错误 |
| 成功幻觉 | “伪造完成” | Agent 在 400 上声称成功；状态未变 |
| 范围蔓延 | “越界” | Agent 做了超出请求的事 |
| Instruction-following deviation | “不服从” | 忽略 system prompt 或用户约束 |
| Sub-intention errors | “计划 bug” | 计划执行中的遗漏、重复、乱序 |

## 延伸阅读

- [Cemri et al., MASFT (arXiv:2503.13657)](https://arxiv.org/abs/2503.13657) — 14 种失败模式，3 个类别
- [Microsoft, Taxonomy of Failure Mode in Agentic AI Systems](https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/final/en-us/microsoft-brand/documents/Taxonomy-of-Failure-Mode-in-Agentic-AI-Systems-Whitepaper.pdf) — 风险登记册
- [Arize Phoenix](https://docs.arize.com/phoenix) — 实践中的 drift clustering
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 什么时候更简单的模式可以完全避开某些失败模式
