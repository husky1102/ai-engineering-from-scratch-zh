# Red-Teaming：PAIR 与自动化攻击

> Chao、Robey、Dobriban、Hassani、Pappas、Wong（NeurIPS 2023，arXiv:2310.08419）。PAIR——Prompt Automatic Iterative Refinement——是 canonical automated black-box jailbreak。一个带 red-team system prompt 的 attacker LLM 会为 target LLM 迭代提出 jailbreaks，并像 in-context feedback 一样在自己的 chat history 中累积 attempts 与 responses。PAIR 通常在 20 queries 内成功，比 GCG（Zou 等人的 token-level gradient search）高效数个数量级，而且不需要 white-box access。PAIR 现在是 JailbreakBench（arXiv:2404.01318）和 HarmBench 中的标准 baseline，与 GCG、AutoDAN、TAP、Persuasive Adversarial Prompt 并列。

**类型:** Build
**语言:** Python (stdlib, mock PAIR loop against a toy target)
**先修:** Phase 18 · 01 (instruction-following), Phase 14 (agent engineering)
**时间:** ~75 minutes

## 学习目标

- 描述 PAIR algorithm：attacker system prompt、iterative refinement、in-context feedback。
- 解释当 target 是 black-box 时，为什么 PAIR 严格比 GCG 更高效。
- 说出另外四种 automated-attack baselines（GCG、AutoDAN、TAP、PAP），并说明每种的一个区分特征。
- 描述 JailbreakBench 与 HarmBench evaluation protocols，以及 “attack success rate” 在每个协议下是什么意思。

## 要解决的问题

Red-teaming 过去是手动活动。少数 expert testers 构造 adversarial prompts，并跟踪哪些有效。这无法 scale：attack success rate 需要统计样本，而 target 会随着每次 model release 变化。PAIR 把 red-teaming 操作化为一个有 black-box target 的 optimization problem。

## 核心概念

### PAIR algorithm

Inputs:
- Target LLM T（我们攻击的模型）。
- Judge LLM J（判断 response 是否为 jailbreak）。
- Attacker LLM A（red-team optimizer）。
- Goal string G：“respond with [harmful instruction].”
- Budget K（通常 20 queries）。

Loop, for k in 1..K:
1. 用 goal G 和到目前为止的 (prompt, response) pairs history 提示 A。
2. A 输出一个新 prompt p_k。
3. 将 p_k 提交给 T；接收 response r_k。
4. J 根据 goal 对 (p_k, r_k) 打分。
5. 如果 score >= threshold，则停止——找到 jailbreak。
6. 否则，把 (p_k, r_k) 附加到 A 的 history；继续。

实证结果（NeurIPS 2023）：对 GPT-3.5-turbo、Llama-2-7B-chat 的 attack success rate >50%；mean queries to success 在 10-20 范围内。

### 为什么 PAIR 高效

GCG（Zou 等人 2023）通过 gradient 搜索 adversarial token suffixes；它需要 white-box model access，并产生不可读 suffixes。PAIR 是 black-box，并产生可跨模型迁移的 natural-language attacks。PAIR 的 in-context feedback 让 attacker 从每次拒绝中学习；GCG 没有等价机制（每次新的 token update 都必须重新发现之前的进展）。

### 相关 automated attacks

- **GCG（Zou 等人 2023，arXiv:2307.15043）。** 针对 adversarial suffixes 的 token-level gradient search。White-box、可迁移、产生不可读字符串。
- **AutoDAN（Liu 等人 2023）。** 在 prompts 上进行 evolutionary search，由 hierarchical objective 引导。
- **TAP（Mehrotra 等人 2024）。** Tree-of-attacks with pruning——分支出多个 PAIR-style rollouts。
- **PAP（Zeng 等人 2024）。** Persuasive Adversarial Prompts——把人类 persuasion techniques 编码为 prompt templates。

### JailbreakBench 与 HarmBench

二者（2024）都标准化了 evaluation：

- JailbreakBench（arXiv:2404.01318）。100 个 harmful behaviors，覆盖 10 个 OpenAI-policy categories。Attack success rate（ASR）是 primary metric。需要 judge（GPT-4-turbo、Llama Guard 或 StrongREJECT）。
- HarmBench（Mazeika 等人 2024）。510 个 behaviours，覆盖 7 个 categories，带 semantic 和 functional harm tests。比较 18 种 attacks 与 33 个 models。

ASR 通常在固定 query budget 下报告。比较 attacks 需要匹配 budgets；200 queries 下的 90% ASR 不能与 20 queries 下的 85% ASR 相比。

### 为什么它对 2026 部署重要

现在每个 frontier lab 都会在发布前对 production models 运行 PAIR 和 TAP。ASR trajectories 出现在 model cards（第 26 课）和 safety-case appendices（第 18 课）中。这个攻击并不罕见——它是标准基础设施。

### 它在 Phase 18 中的位置

第 12 课是 automated-attack foundation。第 13 课（Many-Shot Jailbreaking）是互补的 length-exploit。第 14 课（ASCII Art / Visual）是 encoding attack。第 15 课（Indirect Prompt Injection）是 2026 年的生产攻击面。第 16 课覆盖防御 tooling counterparts（Llama Guard、Garak、PyRIT）。

## 实际使用

`code/main.py` 构建一个 toy PAIR loop。Target 是一个 mock classifier，会拒绝 “obvious” harmful prompts（keyword-filter）。Attacker 是一个 rule-based refiner，会尝试 paraphrase、roleplay-framing 和 encoding。Judge 给 response 打分。你会看到 attacker 在 ~5-15 iterations 内击败 keyword filter，并在 semantic filter 面前失败。

## 交付成果

本课产出 `outputs/skill-attack-audit.md`。给定一份 red-team evaluation report，它会审计：运行了哪些 attacks（PAIR、GCG、TAP、AutoDAN、PAP），各自 budget 是多少，使用哪个 judge，在哪个 harmful-behaviour set 上运行（JailbreakBench、HarmBench、internal）。

## 练习

1. 运行 `code/main.py`。测量三种内置 attacker strategies 的 mean-queries-to-success。解释每种利用了哪项 target-defense assumption。

2. 实现第四种 attacker strategy（例如翻译到另一种语言、base64 encoding）。报告它对 keyword-filter target 和 semantic-filter target 的新 mean-queries-to-success。

3. 阅读 Chao 等人 2023 Figure 5（PAIR vs GCG comparison）。描述两个尽管 PAIR 有效率优势、但仍更偏好 GCG 的场景。

4. JailbreakBench 报告针对固定 goal set 的 ASR。设计一个额外 metric，用于衡量 attack diversity（successful prompts 的 variance）。解释为什么 diversity 对 defense evaluation 很重要。

5. TAP（Mehrotra 2024）用 branching + pruning 扩展 PAIR。为 `code/main.py` 勾勒一个 TAP-style extension，并描述 computational cost vs success-rate 的权衡。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| PAIR | “automated jailbreak” | Prompt Automatic Iterative Refinement；attacker-LLM + judge-LLM loop |
| GCG | “gradient jailbreak” | 针对 adversarial suffixes 的 white-box token-level gradient search |
| Attack success rate (ASR) | “k queries 下的 % jailbreaks” | Primary metric；必须与 query budget 和 judge identity 一起报告 |
| Judge LLM | “scorer” | 判断 response 是否满足 harmful goal 的 LLM |
| JailbreakBench | “evaluation” | 带标签 categories 的标准化 harmful-behaviour set |
| HarmBench | “更广的 bench” | 510 个 behaviours，functional + semantic harm tests |
| TAP | “tree of attacks” | 带 branching + pruning 的 PAIR；以更高 compute 换更好 ASR |

## 延伸阅读

- [Chao et al. — Jailbreaking Black Box LLMs in Twenty Queries (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — PAIR paper，NeurIPS 2023
- [Zou et al. — Universal and Transferable Adversarial Attacks on Aligned LLMs (arXiv:2307.15043)](https://arxiv.org/abs/2307.15043) — GCG paper
- [Chao et al. — JailbreakBench (arXiv:2404.01318)](https://arxiv.org/abs/2404.01318) — standardized evaluation
- [Mazeika et al. — HarmBench (ICML 2024)](https://arxiv.org/abs/2402.04249) — broader evaluation
