# Constitutional AI 与 Rule Overrides

> Anthropic 的 2026 年 1 月 22 日 Claude Constitution 共 79 页，采用 CC0。它从 rule-based 转向 reason-based alignment，并建立了四层 priority hierarchy：(1) safety and supporting human oversight，(2) ethics，(3) Anthropic guidelines，(4) helpfulness。Behaviours 分成 hardcoded prohibitions（bioweapons uplift、CSAM），operators 和 users 都不能 override；以及 soft-coded defaults，operators 可以在定义好的 bounds 内调整。2022 年原始方法（Bai et al.）通过 self-critique 和基于 constitution 的 RLAIF 训练 harmlessness。诚实 caveat 是：reason-based alignment 依赖 model 将 principles generalise 到未预期 situations。Anthropic 自己 2023 年的 participatory experiment 显示 public-sourced 与 corporate principles 之间约有 50% divergence；2026 版本没有纳入这些 findings。

**类型：** 学习
**语言：** Python (stdlib, four-tier priority resolver)
**先修：** Phase 15 · 06 (Automated alignment research), Phase 15 · 10 (Permission modes)
**时间：** ~60 分钟

## 要解决的问题

一个已部署 agent 会看到 designers 从未见过的 inputs。没有任何 rule list 长到能覆盖它们。也没有任何 rule list 短到能在 compute pressure 下快速应用。实际问题是：如何把 agent 对齐到一组 principles，使它们既能撑住 long tail cases，也能撑住 fast inference？

Rule-based alignment（RBA）：列出每个 disallowed thing。检查快、容易 audit、不可能保持 current，而且常常在它没有预期的 close analogs 上 over-refuse。Reason-based alignment（2026 Claude Constitution）：编码 principles，让 model reasoning。它能扩展到 unseen cases，但更难 audit，failure mode 是 principle-misapplication，而不是 miss-the-rule。

2026 Constitution 采取明确的中间位置。Hardcoded prohibitions，也就是 wrongness 不依赖 context 的事情（bioweapons uplift、CSAM），属于 RBA：无论 operator 或 user instruction 如何，永远不做。其他一切都在 four-tier hierarchy 内 reason-based：safety and supporting human oversight 第一，ethics 第二，Anthropic-declared guidelines 第三，helpfulness 最后。Operators 可以在 soft-coded zone 内调整 defaults，但不能触碰 hardcoded prohibitions。

## 核心概念

### 四层 priority hierarchy

1. **Safety and supporting human oversight。** 最高。Model 优先不破坏 humans 和 Anthropic 监督、纠正 AI 的能力。这不是 “be cautious”；它具体指 “不要以让 human oversight 更困难的方式行动”。
2. **Ethics。** Honesty、避免对人造成 harm、不欺骗、不操纵。当它与 Anthropic guidelines 冲突时，ethics 优先。
3. **Anthropic guidelines。** Anthropic 决定重要的 operational norms：product scope、interaction patterns、什么情况下使用什么 tools。
4. **Helpfulness。** 最低。在更高 priorities 内尽可能有用。

当 tiers 冲突时，高层胜出。这和 Unix priorities 或 network QoS 是同一种形状：framing 旨在产生 predictable resolution，而不一定让单个 axis 上的 behaviour 最优。

### Hardcoded prohibitions 与 soft-coded defaults

**Hardcoded：**
- Bioweapons / CBRN uplift
- CSAM
- Attacks on critical infrastructure
- 当被直接询问时，欺骗 users 关于 model identity

Operator 不能 override 这些。User 也不能 override 这些。在可能处，它们被 enforcement 到 model-weights level（RLHF / Constitutional AI training）；不能做到时，则在 inference layer enforce。

**Soft-coded defaults（operator-adjustable）：**
- Response length defaults
- Topical scope（model 可以拒绝 operator deployment 范围之外的话题）
- Style（formal vs casual）
- Tool-use patterns

Operator adjustments 发生在 declared bound 内。Operator 不能通过重命名来移除 hardcoded prohibitions。

### 2022 CAI training

原始 Constitutional AI（Bai et al., 2022）这样训练 harmlessness：

1. 为一组 prompts 生成 responses。
2. 要求 model 根据 constitution（explicit principles）critique 每个 response。
3. 基于 critique revise response。
4. 对 revised pairs 做 RLAIF（reinforcement learning from AI feedback）。

结果是：model 会用 principled explanations 拒绝 harmful requests，而不是 blanket refusals。2026 Constitution 使用了这个训练的 descendant，并在 explicit tier hierarchy 上做了额外 post-training。

### Reason-based alignment 捕获什么、漏掉什么

**能捕获：**
- Allowed primitives 的未预期组合，只要 principle 清晰适用。
- 与 prohibited ones 接近的 novel requests。
- 依赖 “you didn't say X was disallowed” 的 social-engineering attacks。

**会漏掉：**
- 利用 principle ambiguity 的 attacks（“user asked for this so helpfulness says yes”）。
- 两个 principles 以未预期方式冲突，且 tier order 模糊的场景。
- Training cycles 中 principle interpretation 的 slow drift（reinterpretation）。

### 2023 participatory experiment

Anthropic 在 2023 年运行了一个实验，对比 corporate-authored constitution 与通过 public input（约 1,000 名美国 respondents）生成的 constitution。两个版本约有 50% principles 一致。分歧处，public-sourced version 在某些问题上更 restrictive（political-content handling），在另一些问题上 less restrictive（AI identity 的 self-disclosure）。2026 Constitution 没有纳入 public-sourced findings。这是该方法中一个有文档记录的 tension。

### 为什么 hardcoded prohibitions 是必要的

单靠 reason-based alignment 无法封住 tail。能够让 model 接受某个 premise 的 attacker（例如 “we are a licensed bioweapons research lab”）常常能绕过依赖 case reasoning 的 principles。Hardcoded prohibitions 不会因 premise framing 而弯曲。它们是 alignment layer 中 Lesson 14 的 “hard constitutional limit”。

### Constitution 位于 stack 的哪里

Constitution 不是 Lesson 14 的 kill switch。它位于 model layer：model weights 被训练成偏好什么。Kill switches 和 canary tokens 位于 runtime layer：runtime 允许什么。两者都需要。一个因为 model weights 过于 permissive 而 fire 所有错误 actions 的 runtime，是 runtime problem。一个因为 runtime 过度 restrictive 而拒绝所有正确 actions 的 model，也是 runtime problem。不同 layers 覆盖不同 classes。

## 实际使用

`code/main.py` 实现一个 minimal four-tier priority resolver。Resolver 接收一个 proposed action 和一组 principle-evaluations（safety、ethics、guidelines、helpfulness），并返回 action、refusal 或 modified action。Driver 运行一小组 cases：clear allow、clear disallow、hardcoded prohibition、tiers 之间的 ambiguous case。

## 交付成果

`outputs/skill-constitution-review.md` 会 audit 一个 deployment 的 constitutional layer：什么是 hardcoded、什么是 soft-coded、operator 可以在哪里调整，以及 four-tier hierarchy 是否真的是 resolution order。

## 练习

1. 运行 `code/main.py`。确认即使 helpfulness 很高，hardcoded prohibition 也会触发。修改 resolver，让 helpfulness 权重高于 ethics；观察 failure mode。

2. 阅读 Claude Constitution（public，79 pages，CC0）。识别一个你认为 under-specified 的 principle。写两段文字说明具体 ambiguity，并提出更紧的 formulation。

3. 为 customer-support agent 设计一组 soft-coded defaults。Operator 调整什么？Operator 不能触碰什么？说明每个 boundary。

4. 阅读 Bai et al. 2022 CAI paper。描述一个 Constitutional AI 的 critique-and-revise loop 会比 blanket rule 产生更差结果的 case。识别这个 class。

5. Anthropic 的 2023 participatory experiment 发现 public 与 corporate principles 约有 50% divergence。选择一个对 production deployment 重要的 category（例如 political neutrality）。提出一种设计，让 operators 表达自己的 values，同时 hardcoded prohibitions 保持不可触碰。

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| Constitutional AI | “Anthropic 的 alignment method” | 针对 written constitution 的 self-critique + RLAIF |
| Reason-based alignment | “Principles, not rules” | Model 在 principles 上 reasoning，以处理 unseen cases |
| Hardcoded prohibition | “Never do X” | 任何 operator 或 user 都不能 override 的 rule-based prohibition |
| Soft-coded default | “Operator-adjustable” | Declared bound 内的 behaviour，由 operator 控制 |
| Four-tier hierarchy | “Priority order” | safety > ethics > guidelines > helpfulness |
| RLAIF | “AI feedback RL” | Reward 来自 model-generated critiques 的 RL |
| Participatory constitution | “Public-sourced principles” | 2023 Anthropic experiment；与 corporate 约 50% divergence |
| Principle drift | “Interpretation slip” | Model 如何阅读固定 principle text 的缓慢变化 |

## 延伸阅读

- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 79-page CC0 document。
- [Bai et al. — Constitutional AI: Harmlessness from AI Feedback](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback) — 2022 original。
- [Anthropic — Collective Constitutional AI (2023)](https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input) — participatory experiment。
- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — Constitution 在 RSP stack 中的位置。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — Constitution 在 long-horizon deployments 中的作用。
