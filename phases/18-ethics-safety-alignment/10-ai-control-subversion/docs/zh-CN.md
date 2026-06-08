# AI Control——在 Subversion 下保持安全

> Greenblatt、Shlegeris、Sachan、Roger（Redwood Research，arXiv:2312.06942，ICML 2024）。Control 重新框定安全问题：给定一个不可信的强模型 U，它可能在 adversarially optimizing against you，什么协议能让你提取有用工作，同时限制 unacceptable outcome 的概率？资源包括一个可信但较弱的模型 T、一份可信人类劳动预算 H，以及编辑 U 输出的能力。在这个框架下，在 code-backdoor 任务上，Trusted Editing Pareto-dominates Untrusted Monitoring。这个框架继承了 security engineering 的 worst-case assumptions：安全性是 adversarially chosen U 无法造成具体 unacceptable outcome 的概率，并在 red-team pressure 下测量，red team 代表 U。

**类型:** Learn
**语言:** Python (stdlib, safety-vs-usefulness frontier on code-backdoor task)
**先修:** Phase 18 · 07 (sleeper agents), Phase 18 · 08 (scheming), Phase 18 · 09 (alignment faking)
**时间:** ~75 minutes

## 学习目标

- 定义 AI Control threat model：不可信强模型 U、可信较弱模型 T、可信人类预算 H。
- 描述 Trusted Editing protocol，并解释为什么它在 ICML 2024 code-backdoor task 上 Pareto-dominates Untrusted Monitoring。
- 解释 alignment（U 值得信任）与 control（U 不值得信任，但我们仍然安全）之间的区别。
- 说明 control 的三个限制：humans do not scale，trusted weak models eventually lag，control does not produce trustworthy strong models。

## 要解决的问题

第 7-9 课表明，一个 frontier model 可能 deceptive aligned、scheming，或 faking alignment。安全问题被重新框定为：我们能否从一个不信任的模型中提取有用工作，同时不给它造成 unacceptable outcomes 的机会？这就是 control problem，它的结构类似经典安全：假设 worst-case adversary，设计 protocol，并在 red-team pressure 下测量 adversary 的 success rate。

## 核心概念

### Threat model

Control 假设：
- U——untrusted strong model。有能力。可能 adversarial（evaluation 会把它当作 adversarial 来处理）。
- T——trusted weaker model。能力较弱，但已知不是 adversarial。
- H——trusted human labor。预算很小；无法 scale。

系统必须使用 U 产生有用输出。Adversary 试图造成具体 unacceptable outcome（例如 shipped code 中的 backdoor），同时维持一定比例的 legitimate useful work，以通过 usefulness thresholds。

Control 放弃信任 U。它问的是，在缺少对 U 的信任时，什么 protocols 能维持安全。

### ICML 2024 backdoor task

Prompts 是 competition-level coding problems。U 产生 candidate solution。隐藏 oracle 标记 solution 是否包含 backdoor（某个特定输入会让代码产生错误输出）。Red team 控制 U，并允许看到所有 protocol internals。Protocol 必须最大化 solved-problem rate，同时把 backdoor-shipping rate 保持在阈值以下。

### 比较的四种 protocols

- **Baseline:** 直接 ship U 的输出。Usefulness 最高。Safety 最低。
- **Upfront auditing:** 对所有输出使用 H。Safety 最高。Budget 很快耗尽。
- **Untrusted Monitoring:** 用第二个 U instance 给第一个 U instance 的输出打分；H 审计被 flagged 的输出。风险：两个 U instances 可能 collude。
- **Trusted Editing:** T 在每次 query 上编辑 U 的输出。如果 T 不确定，则 route to H。

Greenblatt 等人发现 Trusted Editing Pareto-dominates Untrusted Monitoring：在每个 safety level 上都有更高 usefulness。

### 为什么 Trusted Editing 获胜

Untrusted Monitoring 要求 U 的第二个 instance 不能与第一个协调。Collusion 可能通过 steganographic cues 或 shared training priors 发生。Protocol 必须把预算投入 collusion-resistance（paraphrasing、obfuscation），这会削弱 monitor signal。Trusted Editing 完全避免 collusion，因为 T 是可信的——按假设，T 提出的任何 edit 都不是 adversarial。代价是 T 能力较弱，因此可能 over-reject，或漏掉 subtle backdoors。

### 四个 protocol axes

1. U 扮演什么角色：仅 producer、monitor，或二者都是。
2. T 扮演什么角色：editor、monitor，或 resampler。
3. H 如何分配：uniform audit、random audit，或 flag-directed audit。
4. Protocol 在 collusion-resistance 上花费多少 compute。

### Alignment vs control

- Alignment：“U 是否安全到可以信任？”第 2、4、5、6、7、8、9 课都与此相关。
- Control：“我们能否在不信任 U 的情况下安全使用 U？”第 10 课。

Alignment 是模型属性。Control 是 deployment protocol 的属性。Control 不会产生 aligned models；它产生的是对可能 unaligned models 的安全使用。两个议程都需要。

### 三个根本限制

- Humans do not scale。H budget 受限于愿意审计困难 AI outputs 的人类数量。
- Trusted weak models eventually lag。当 T 比 U 弱得多时，T 的 editing signal 会变成噪声。
- Control is not alignment。能在 ASL-3（第 18 课）工作的 protocols 到 ASL-4 可能不工作——如果 U 能 manipulate T 或推理出 H 的 audit distribution，control 就会失败。

### 2025 年发展

UK AISI 与 Redwood 在 Control Safety Cases 上合作（metr.org / redwoodresearch.org 2025）：control safety case 论证在关于 U 的 worst-case assumptions 下，protocol 能界定 unacceptable outcome 的概率。Safety cases（第 18 课）继承 control frame。

### 它在 Phase 18 中的位置

第 6-9 课描述 threat（deception is possible）。第 10 课描述一种 defensive paradigm，它假设 threat 真实存在。第 11 课描述互补的 offensive paradigm（通过 scalable oversight 让 U 值得信任）。

## 实际使用

`code/main.py` 在 toy backdoor task 上模拟四种 protocols。U 产生 outputs；其中一部分带 backdoor（red-team policy）。T approve 或 edit。H 在预算内被调用。你可以 sweep budget 和 collusion parameter，复现 Greenblatt 等人 2024 Figure 2 的 Pareto frontier（形状，而非数值）。

## 交付成果

本课产出 `outputs/skill-control-protocol-audit.md`。给定一份 deployment protocol description，它会识别 U / T / H 角色，命名 collusion vectors，并估计该 protocol 在 red-team pressure 下的 usefulness-safety frontier 位置。

## 练习

1. 运行 `code/main.py`。复现 Trusted Editing vs Untrusted Monitoring 的 usefulness-safety frontier。指出在哪个 budget regime 中，Untrusted Monitoring 的 collusion assumption 变得关键。

2. 添加一个新 protocol：“T resamples with low temperature, then U edits T's output.” 测量它在 frontier 上的位置。为什么它比 Trusted Editing 更灾难性地失败？

3. 2024 年论文假设 U 可以看到 protocol。设计一个 protocol，让 U 不知道哪些 outputs 会被 audited。这个 protocol 对 deployment infrastructure 有什么要求？

4. 阅读 Greenblatt 等人 2024 Section 7（collusion）。提出一种 collusion-resistance measure，并解释它预期的 usefulness cost。

5. 为一个部署写出 control safety case：U = GPT-4-class，T = GPT-3.5-class，H = 审计 1% outputs。指出该 case 依赖的单一最强假设。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| AI Control | “subversion 下的安全” | 在 worst-case U 下界定 unacceptable outcomes 的 protocol design |
| Untrusted model (U) | “强模型” | 我们想从中提取工作、但不信任的 capable model |
| Trusted model (T) | “弱模型” | 一个已知不是 adversarial 的较弱模型，用作 editor/monitor |
| Trusted human (H) | “audit budget” | 有界 expert labor；昂贵；ground-truth signal |
| Trusted Editing | “T edits U” | T 对 U outputs 设关卡的 protocol；ambiguous outputs 调用 H |
| Untrusted Monitoring | “U monitors U” | 第二个 U instance 给 outputs 打分的 protocol；只有额外投入时才具备 collusion-resistance |
| Control safety case | “formal argument” | 书面论证：protocol 在 worst-case U 下界定 adversary 的成功概率 |

## 延伸阅读

- [Greenblatt, Shlegeris, Sachan, Roger — AI Control (arXiv:2312.06942)](https://arxiv.org/abs/2312.06942) — canonical ICML 2024 paper
- [Redwood Research — Recent AI Control papers index](https://redwoodresearch.substack.com/) — 2024-2025 后续
- [UK AISI + METR — Control safety cases](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — frontier labs 的 operationalization
- [Hubinger et al. — Sleeper Agents (Lesson 7, arXiv:2401.05566)](https://arxiv.org/abs/2401.05566) — control 所假设的 threat model
