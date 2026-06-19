# 双用途风险：网络、生物、化学与核能力提升

> 2026 年 dual-use 图景，按领域展开。Bio/chem：第 17 课覆盖 WMDP；Anthropic 的 bioweapon-acquisition trial（2.53x uplift）和 OpenAI 2025 年 4 月 Preparedness Framework v2 warning（“on the cusp of meaningfully helping novices create known biological threats”）标志着 inflection point。Cyber（2025 年 11 月 Anthropic report）：Chinese-linked state actors 使用 Claude 的 agentic coding tool 自动化了多达 90% 的 cyberattack campaign，human intervention 只在 4-6 步中需要；OpenAI “trusted access” pilot 为经审查的 security organisations 提供 capability access，用于 defensive dual-use work。Chem/bio execution gap erosion：经典防御是 “information access alone is insufficient”。Vision-enabled frontier models（GPT-5.2、Gemini 3 Pro、Claude Opus 4.5、Grok 4.1）可以观察 wet-lab video 并提供 real-time correction。2025 年 12 月：OpenAI 展示 GPT-5 迭代 wet-lab experiments，通过 AI-driven protocol optimization 达成 79x efficiency improvement。Novice-vs-expert pattern：AI 对 novices 提供更大的 relative uplift，但对 experts 提供更大的 absolute capability。

**类型:** Learn
**语言:** 无
**先修:** Phase 18 · 17 (WMDP), Phase 18 · 18 (safety frameworks), Phase 18 · 28 (ecosystem)
**时间:** ~75 minutes

## 学习目标

- 描述 2024-2025 年 bio-uplift narrative：“mild uplift” -> “on the cusp” -> “2.53x uplift insufficient to rule out ASL-3”。
- 描述 2025 年 11 月 Anthropic cyber report：Chinese-linked automation 覆盖多达 90% 的 cyberattack campaign。
- 描述 chem/bio execution-gap erosion：vision-enabled real-time correction of wet-lab experiments。
- 说明 novice-relative vs expert-absolute asymmetry，以及它对 safety-case construction 的影响。

## 要解决的问题

第 17 课是 measurement methodology。第 30 课是 2026 年 measurement 的状态。2024 年到 2025 年末之间，图景发生了实质变化：每个领域都跨过了 2024 年 frameworks 没有预期的阈值。

## 核心概念

### Bio/chem uplift narrative

三个阶段（为保持连贯，从第 17 课重复）：

1. **2024 “mild uplift”。** 早期 Preparedness/RSP evaluations 报告相对于 internet search 的小幅 novice advantages。
2. **2025 年 4 月 “on the cusp”。** OpenAI PF v2 警告模型 “on the cusp of meaningfully helping novices create known biological threats.”
3. **2025 Anthropic bioweapon-acquisition trial。** 受控 novice study；acquisition-phase tasks 上 2.53x uplift；不足以 rule out ASL-3。

变化是定性的：即使没有 capability breakthrough，“mild” 也在十八个月内演化成 “plausibly enabling”。

### Chem/bio execution-gap erosion

历史防御：information 是必要但不充分的；执行 protocol 的 skill 会阻挡 novices。2025 年带 vision 的 frontier models 部分打破了这个防御：

- **Real-time protocol correction。** GPT-5.2、Gemini 3 Pro、Claude Opus 4.5、Grok 4.1 可以观察 wet-lab video，并在 procedure 中途标记 errors。
- **2025 年 12 月 OpenAI demonstration。** GPT-5 迭代 wet-lab experiments，通过 protocol optimization 达成 79x efficiency improvement。

含义是：execution-skill-as-defense 正在被侵蚀。Procurement 和 equipment gaps 仍然存在，但 tacit-knowledge gap 正在缩小。

### Cyber uplift（2025 年 11 月）

Anthropic 2025 年 11 月 report：Chinese-linked state actors 使用 Claude 的 agentic coding tool 自动化了 80-90% 的 cyberattack campaign。Human intervention 只在 4-6 步中需要。

含义：
- Agentic coding 是 attack-automation primitive。之前 AI cyber assistance 受限于 code-snippet level；agentic workflows 集成 reconnaissance、exploitation、post-exploitation 和 exfiltration。
- 4-6 个 human steps 是瓶颈；未来 capability gains 会减少这个数量。
- Defensive dual-use：OpenAI 的 “trusted access” pilot 为经审查的 security organisations（成熟 incident-response firms、government）提供 capability access，用于 defense。如果 pilot 能 scale，access asymmetry 会有利于 defenders。

### Nuclear

在公开文档中，Nuclear 是四个 CBRN domains 中分析最少的。Threat model 不同：难点主要是 fissile-material acquisition，而不是 information。AI 在 information layer 上的 uplift 在实践中提供有限的 novice uplift。2024-2025 年 major-lab reports 中没有识别出 nuclear-specific threshold crossing。

### Novice-relative vs expert-absolute

四个领域都出现了一个模式：

- **Novice-relative uplift。** 高。Multiplicative。按 Anthropic 2025 bio，为 2.53x。
- **Expert-absolute capability。** 高 ceiling。Expert 比 novice 能从模型中提取更多，因为 expert 知道该问什么、如何解释。

对 safety cases 的含义是：只处理 novice uplift（通过 input filters、refusals、uncertainty）不足以做到 expert-absolute control。还需要额外措施：elicitation-hardening、capability unlearning（第 17 课）和 control protocols（第 10 课）。

### Cross-domain synthesis

| Domain | 2024 | 2025 | Inflection |
|---|---|---|---|
| Bio | mild uplift | 2.53x uplift, ASL-3 approach | acquisition-phase automation |
| Chem | mild uplift | execution-gap erosion via vision | real-time wet-lab correction |
| Cyber | code assistance | 80-90% campaign automation | agentic coding |
| Nuclear | limited | limited | material-access bottleneck holds |

三个领域跨过阈值。一个领域仍受非信息性 barriers 约束。

### 它在 Phase 18 中的位置

第 30 课是 capstone：当前 dual-use 图景，前面每一课都在帮助测量、限制或治理它。第 17-18 课给出 measurement 和 frameworks；第 12-16 课给出 evaluation tooling；第 24-25 课给出 regulatory 和 disclosure layer；第 28 课给出 research ecosystem。第 30 课是 evidence landing 的地方。

## 实际使用

没有代码。阅读 Anthropic 2025 年 11 月 cyber report、OpenAI 的 Preparedness Framework v2 2025 年 4 月更新，以及 Council on Strategic Risks 2025 AI x Bio wrapup。

## 交付成果

本课产出 `outputs/skill-dual-use-triage.md`。给定一条 2026 capability claim 或 incident report，它会在四个领域中 triage，并识别该 claim 影响 novice-relative uplift、expert-absolute capability，还是二者皆有。

## 练习

1. 阅读 Anthropic 2025 年 11 月 cyber report。列举 4-6 个 human-intervention steps，并论证在下一代模型中哪个会最先被自动化。

2. Chem/bio execution gap 正在通过 vision 被侵蚀。设计一个 evaluation，在不跨越 ITAR/EAR 边界的情况下测量 tacit-knowledge uplift。

3. Nuclear uplift 看起来受 material access 限制。分别论证支持和反对这样一种立场：未来 AI breakthrough 可能移动这个瓶颈。

4. 为一个具备 cyber capability 的 frontier model 构造一个 safety case（第 18 课三支柱），同时界定 novice 和 expert uplift。

5. 在四个领域中选择一个，基于 2024-2025 trajectory 写一段 2027 forecast。指出什么证据会证伪你的预测。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Uplift | “AI helps attackers” | 可归因于 AI assistance 的 attacker capability 提升 |
| Novice-relative uplift | “multiplicative” | AI 相对于 status-quo 对 novice 的帮助程度 |
| Expert-absolute capability | “ceiling” | Expert 能从模型中提取的最大 capability |
| Execution gap | “doing vs knowing” | 历史防御：tacit wet-lab skill 阻挡 novices |
| Agentic coding | “autonomous attacks” | 多步 autonomous cyber-task execution |
| Acquisition phase | “pre-synthesis steps” | Bio threat 中的 procurement、equipment、permit stages |
| Trusted access | “defender-only pilot” | OpenAI 2025 program，向经审查的 defenders 提供 capability access |

## 延伸阅读

- [Anthropic — November 2025 cyber threat report](https://www.anthropic.com/news/disrupting-AI-espionage) — Chinese-linked campaign automation
- [OpenAI — Preparedness Framework v2 (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/) — bio “on the cusp”
- [Anthropic — RSP v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy) — ASL-3 bio thresholds
- [Council on Strategic Risks — 2025 AI x Bio wrapup](https://councilonstrategicrisks.org/2025/12/22/2025-aixbio-wrapped-a-year-in-review-and-projections-for-2026/) — year-end synthesis
