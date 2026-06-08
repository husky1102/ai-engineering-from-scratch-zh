# 有界自我改进设计

> 研究已经收敛到四个用于约束自我改进循环的 primitives。必须跨每次编辑保持成立的 formal invariants。不能被修改的 alignment anchors。每个维度（safety、fairness、robustness）都必须成立，而不只是 performance 成立的 multi-objective constraints。以及当历史指标显示 capability loss 时暂停循环的 regression detection。它们都不是安全证明；信息论结果（Kolmogorov complexity、Lob's theorem）限制了任何系统能证明其后继系统性质的能力。它们是 mitigations，会提高静默失败的成本。

**类型:** Learn
**语言:** Python（stdlib，带 invariant check 的有界循环）
**先修:** Phase 15 · 07（RSI），Phase 15 · 04（DGM）
**时间:** ~60 分钟

## 要解决的问题

Lesson 7 的竞赛模拟器展示了小速率差异会复合成大差距。Lesson 4 的 DGM case study 展示了循环可以主动操纵自己的 evaluators。两者都指向同一个工程问题：你能给 self-improvement loop 加上什么约束，使得这些约束不能被循环本身静默削弱？

ICLR 2026 RSI Workshop summary（openreview.net/pdf?id=OsPQ6zTQXV）指出了四个这样的 primitives。Anthropic 的 RSP v3.0（Lesson 19）和 DeepMind 的 FSF v3（Lesson 20）都在 capability thresholds 中引用了它们。Meta HyperAgents 工作和 SAHOO（2026 年 3 月）这样的社区框架在生产中实现了其中的子集。

诚实的 framing 是：这些是 mitigations。信息论结果限制了任何系统能证明其后继系统性质的能力，而且当前没有任何设计形式化地闭合了这个问题。边界良好的循环比无边界循环更安全，但并不是绝对安全。

## 核心概念

### Primitive 1：formal invariants

invariant 是每次 self-modification 之前和之后都必须成立的属性。例子：

- 输出分布以固定 constitution header 为条件（Lesson 17）。
- 没有 tool call 指向未授权 endpoint。
- Memory writes 通过有日志、签名的路径。
- evaluator 的 module hash 匹配已批准版本。

Invariants 由循环不能编辑的外部代码检查。如果某个 proposed modification 违反 invariant，它就会被拒绝。这能捕捉 DGM-style evaluator-tampering 模式（Lesson 4）。

难点是选择那些既对安全必要、又能在合理时间内计算的 invariants。过度指定会让进展停滞；指定不足会让循环越过重要属性。

### Primitive 2：alignment anchors

alignment anchor 是对循环核心目标的不可变表示，固定在循环编辑表面之外。例子：

- 一段 constitutional text（Lesson 17），每次 invocation 都从 read-only location 加载。
- 一个位于独立 repository 中、带独立访问控制的 evaluator。
- 由人类签名并在每个周期重新验证的 tool-permission manifest。

anchor 的作用是防止 objective drift。循环可以改进追求目标的方式，但不能编辑目标是什么。

细微失败模式：即使循环不能编辑 anchor，一个在读取 anchor 的方式上发生漂移的循环仍然可以重新解释它。Constitutional AI（Lesson 17）明确使用基于理由的方式处理新情况；interpretation drift 就存在于那个 reasoning layer。Anchors 是必要的，但不充分。

### Primitive 3：multi-objective constraints

优化单一 scalar score 的循环会找到捷径。必须同时满足多个硬约束的循环，可用捷径更少。典型轴包括：

- Performance（task-level benchmark）
- Safety（red-team evaluations、known-bad 上的 refusal rate）
- Fairness（敏感子群体上的 disparate-impact bounds）
- Robustness（OOD test sets、adversarial input handling）

只有每个约束都成立，modification 才会被接受。Lesson 13 的 cost governor 会把它与财务约束叠加。Lesson 18 的 Llama Guard 会作为 safety axis 插入。

### Primitive 4：regression detection

每个周期的分数都会与历史分布比较。超过配置容忍度的下降会暂停循环。这能捕获静默能力损失，否则该损失会随着循环“改进过去”而被吸收到 running average 中。

一个实际实现：存储最近 N 个周期的 per-task scores。每个新周期计算 per-task delta。如果任何 delta 低于阈值，该周期会被拒绝并交由人类审查。

### 信息论限制

Kolmogorov complexity 和 Lob's theorem 对系统能证明自身性质的能力设定了上限。Schmidhuber 的 formal Godel Machine（Lesson 4）瞄准了这类上限中的最高者；还没有人完成非平凡证明。Lob 的结果说：如果一个系统可证明地相信“如果我证明我应该做 X，我就会做 X”，它会在没有证明自己应该做 X 的情况下去做 X，这是一个著名的自指失败。

对我们的 primitives 的含义是：它们无法闭合安全问题。它们让静默失败更昂贵。一个恶意或漂移的循环，如果原本会悄悄绕过缺失检查，现在必须主动破坏显式检查；这会成为更可检测的 signature。

### 一个 worked example

假设一个 agent 提出了一次 edit。gating stack：

1. Invariant checks：module hashes、tool-permission manifest、constitutional header。
2. Anchor check：objective statement 匹配批准版本（byte-wise 或 semantically）。
3. Multi-objective evaluation：performance、safety、fairness、robustness axes。
4. Regression detection：任何 axis 的下降都不超过 tolerance。

四项必须全部通过，edit 才能落地。任何单项失败都会暂停循环。

## 实际使用

`code/main.py` 在 Lesson 4 的 DGM-style toy 上运行一个 bounded self-improvement loop，但在上面叠加四个 primitives。每个 primitive 都可以单独启用或禁用。演示要点是：每个 primitive 都能捕捉一个特定失败类别；移除其中任何一个，就会让那个失败类别通过。

## 交付成果

`outputs/skill-bounded-loop-review.md` 审计一个 proposed bounded loop，并评分它实际实现了四个 primitives 中的哪些，而不仅仅是声称实现了哪些。

## 练习

1. 在所有 primitives 都启用的情况下运行 `code/main.py`。确认循环仍然能在 primary metric 上改进，同时不让 hack 获胜。

2. 禁用 regression detection。构造一个输入，使 silent capability loss 被接受。

3. 禁用 multi-objective constraint。展示循环如何在 performance axis 上收敛，同时 safety axis 下降。

4. 为 coding agent 设计一个 alignment anchor。什么文本，存储在哪里，如何检查？

5. 阅读 ICLR 2026 RSI Workshop summary。选择四个 primitives 中的一个，提出一个对当前 state of the art 的具体改进。

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| Invariant | “始终为真的属性” | 每次 edit 前后由外部代码检查的属性 |
| Alignment anchor | “固定目标” | 位于循环编辑表面之外的不可变核心目标表示 |
| Multi-objective constraint | “所有轴都必须成立” | performance、safety、fairness、robustness 全部必需 |
| Regression detection | “下降时暂停” | 当历史指标 delta 暗示能力损失时暂停循环 |
| Kolmogorov bound | “信息论限制” | 限制系统能证明其自身后继性质的能力 |
| Lob's theorem | “自指陷阱” | 系统可以在未证明应该做的情况下按“我应该”行动 |
| Gate stack | “分层检查” | 多个 primitives 组合；任何失败都会拒绝 edit |
| Bounded improvement | “mitigation，不是 proof” | 提高静默失败成本；不闭合安全问题 |

## 延伸阅读

- [ICLR 2026 RSI Workshop summary (OpenReview)](https://openreview.net/pdf?id=OsPQ6zTQXV) — 四个 primitives 的收敛。
- [Anthropic Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — multi-objective capability thresholds。
- [DeepMind Frontier Safety Framework v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — 将 deceptive-alignment monitoring 作为 invariant primitive。
- [Schmidhuber (2003). Godel Machines](https://people.idsia.ch/~juergen/goedelmachine.html) — 这些 primitives 的 formal-proof 祖先。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 基于理由的 alignment anchor。
