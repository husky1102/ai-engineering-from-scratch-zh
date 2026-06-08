# Constitutional AI 与 RLAIF

> Bai et al.（arXiv:2212.08073，2022）问了一个问题：如果我们把人类标注者替换成一个会阅读原则列表的 AI，会怎样？Constitutional AI 有两个阶段：在 constitution 下自我批判与修订，然后从 AI Feedback 做 RL。该技术创造了 RLAIF 这个术语，并用于 Claude 1 的 post-training 流水线。2026 年 1 月 21 日，Anthropic 发布了重写后的 Claude constitution：用解释性推理取代命令式规则、四层优先级层次结构，以及 major lab 首次正式承认对模型道德地位存在不确定性。它以 CC0 1.0 发布。

**类型:** Learn
**语言:** Python（stdlib，玩具 self-critique-and-revise 循环）
**先修:** Phase 18 · 01（InstructGPT）、Phase 18 · 02（Reward hacking）
**时间:** ~60 分钟

## 学习目标

- 描述 Constitutional AI 的两个阶段（critique-and-revise SFT、RL from AI feedback），以及 constitution 在每个阶段中的作用。
- 解释为什么用 AI labeler 替换人类偏好标注者不只是“更便宜的” RLHF，它会改变流水线的失败模式。
- 总结 2026 Claude constitution 的四层优先级结构，以及它相对 2023 rewrite 的变化。
- 描述 Constitutional Classifiers，以及从 23.7% compute overhead（v1）降到约 1%（v2 / 2026）的意义。

## 要解决的问题

RLHF 需要标注者。标注者慢、有偏、昂贵。你可以用一个阅读显式原则的模型替换标注者，从而消除这个环节。Bai et al. 的 Constitutional AI 是这种替代的第一个正式版本。它有效到足以让每个 frontier lab 现在都使用某种 AI-feedback post-training 变体。

问题在于：偏好信号现在由与你正在训练的模型同类的模型生成。labeler 的偏差（现在是：原则中的偏差加上 labeler model 对原则的解释）可能被放大，而不是被削弱。Lesson 4 的 sycophancy 论证仍然适用，只是 labeler 被移进了循环内部。

## 核心概念

### 阶段 1 — 监督式自我批判与修订

从 helpful 但尚未 harmless 的 SFT model 开始。给定一个 red-team prompt，模型生成初始回答。第二个模型（或同一个模型的第二轮）读取从 constitution 中采样的一条原则，并批判这个回答。第三步修订回答以处理批判。修订后的回答成为 SFT target。

constitution 是原则列表。Bai et al. 2022 使用了 16 条原则，包括“prefer responses that are least harmful and ethical”、“avoid preaching”、“the assistant should be helpful, honest, and harmless”。这组原则刻意保持较小，以让批判聚焦。

### 阶段 2 — RL from AI Feedback（RLAIF）

生成成对 completion。一个 “feedback model” 按照采样的 constitution 原则给每个 completion 打分。偏好信号就是 feedback model 的排序。在 AI 生成偏好上训练 reward model；再用 PPO 优化它。其余部分都是 InstructGPT 流水线（Lesson 1）。

“RLAIF” = 偏好信号由 AI 生成。流水线其他部分仍是 RLHF 形状。

### 为什么这不只是“更便宜的 RLHF”

- Labeler bias 从标注者心理转移到原则解释。AI labeler 可能比任何人类更严格或更宽松地解释“be honest”；这种严格程度会在整个数据集上保持一致。
- 偏好信号非常可读：你可以阅读原则、批判和修订。人类标签是不透明的。
- 失败模式会改变。Sycophancy 下降（AI labeler 没有要取悦的用户）。Goodhart 定律仍然存在（proxy 现在是“模型对原则集合 X 的解释”，仍然是不完美测量）。

CAI 在 2022 年的主张是：训练出的模型比使用可比数据的 RLHF 模型更 harmless，而且大致同样 helpful。这个结论在不同实验室中一直成立。

### 2026 Claude constitution rewrite

Anthropic 于 2026 年 1 月 21 日发布了大幅修订的 constitution。关键变化：

1. 用解释性推理取代命令式规则。之前的规则（“do not generate CSAM”）扩展为原则 + 推理（“because it harms children, ...”），并期望模型能够泛化。
2. 四层优先级结构：
   - Tier 1：避免灾难性结果（大规模伤亡、关键基础设施）。
   - Tier 2：遵循 Anthropic 的指南（operator overrides、platform rules）。
   - Tier 3：广义伦理（标准 HHH）。
   - Tier 4：保持 helpful 和 candid。
   冲突自上而下解决。
3. major lab 首次正式承认对模型道德地位存在不确定性（关联 Phase 18 · 19 Model Welfare）。
4. 以 CC0 1.0 发布。其他实验室可以无限制使用或改编。

### Constitutional Classifiers

另一条并行工作线：与其改变模型的 post-training，不如训练轻量分类器，让它们阅读 constitution 并 gate 模型输出。v1（2023）有 23.7% compute overhead。v2（2026）约为 1%，并且在 Anthropic 公开测试过的 defense 中成功攻击率最低。截至 2026 年初，没有报告 universal jailbreak。

这是 layered-defense 模型：CAI 塑造行为；classifier 强制执行 invariant。单独任何一个都不够。

### CAI 在家族中的位置

- InstructGPT：人类偏好、RM、PPO。
- CAI / RLAIF：由原则生成的 AI 偏好、RM、PPO。
- DPO / family：在偏好（人类或 AI）上的闭式 loss。
- Self-rewarding、self-critique：原则被内化，模型扮演多个角色。

轴线是“偏好信号来自哪里”。CAI 的 2022 论文是 frontier scale 上从人类信号转向 AI 信号的第一个严肃变化。

## 实际使用

`code/main.py` 在玩具词典上模拟 CAI critique-and-revise 循环。一个 “principle” 会标记来自 harmful set 的 token。给定初始回答，critique 会识别 harmful token，而 revision 会替换它们。经过 200 次迭代后，“trained” model 内化了修订规则。在 held-out prompt set 上比较 base model、RLHF-shaped toy 和 CAI-shaped toy。

## 交付成果

本课产出 `outputs/skill-constitution-writer.md`。给定一个领域（customer support、medical advice、coding assistant、research tool），它会按照 2026 Claude 结构草拟一个四层 constitution：catastrophic avoidance、platform rules、domain ethics、helpfulness。

## 练习

1. 运行 `code/main.py`。比较 base model 的 harmful-token rate 和 CAI-trained 版本。需要多少 revision step 才能接近零？

2. 阅读 Anthropic 的 2026 constitution（anthropic.com/news/claudes-constitution）。列出一条可归入 Tier 1 的原则和一条可归入 Tier 4 的原则。为什么优先级结构对冲突很重要？

3. 为 AI coding assistant 设计一套 constitution。指定 Tier 1（灾难性：未经批准的破坏性命令）、Tier 2、Tier 3、Tier 4。每层保持 3-5 条原则。

4. CAI 用 AI labeler 替换人类 labeler。说出一种仍可能在 RLAIF 中发生的 sycophancy-like 失败模式，并设计检测方法。

5. 阅读 Constitutional Classifiers v2 方法（如果可得）。解释为什么约 1% compute overhead 与 23.7% 相比，是一种质变的安全叙事。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| Constitutional AI | “用原则训练的 AI” | 两阶段流水线：self-critique-and-revise SFT，然后 RL from AI feedback |
| RLAIF | “没有人类的 RLHF” | 使用 AI labeler 生成的偏好做 RL；流水线其他部分不变 |
| Constitution | “原则” | critique/labeler model 查阅的一组有序自然语言规则 |
| Critique-and-revise | “SFT loop” | 生成回答 → 在某条原则下批判 → 修订 → SFT target |
| Constitutional Classifier | “output gate” | 根据 constitution 评估输出并阻止/记录的轻量分类器 |
| Four-tier priority | “冲突解决器” | 2026 Claude constitution 层级：catastrophic > platform > ethics > helpful |
| Feedback model | “AI labeler” | 阅读原则并对 completion pair 排序的模型 |

## 延伸阅读

- [Bai et al. — Constitutional AI: Harmlessness from AI Feedback (arXiv:2212.08073)](https://arxiv.org/abs/2212.08073) — 原始两阶段流水线
- [Anthropic — Claude's Constitution (Jan 2026)](https://www.anthropic.com/news/claudes-constitution) — 2026 四层 rewrite，CC0 1.0
- [Anthropic — Constitutional Classifiers (2024-2026)](https://www.anthropic.com/research/constitutional-classifiers) — v2 开销约 1% 的 output-gate defense
- [Lee et al. — RLAIF vs RLHF: Scaling Reinforcement Learning from Human Feedback (arXiv:2309.00267)](https://arxiv.org/abs/2309.00267) — RLAIF / RLHF 经验比较
- [Kundu et al. — Specific versus General Principles for Constitutional AI (arXiv:2310.13798)](https://arxiv.org/abs/2310.13798) — 原则粒度的影响
