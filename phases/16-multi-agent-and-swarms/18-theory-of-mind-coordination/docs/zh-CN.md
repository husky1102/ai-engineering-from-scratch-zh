# Theory of Mind 与 Emergent Coordination

> Li et al. (arXiv:2310.10701) 展示了 cooperative text game 中的 LLM agents 会表现出 **emergent high-order Theory of Mind**（ToM）：推理另一个 agent 对第三个 agent beliefs 的 belief，但会因为 context management 和 hallucination 在 long-horizon planning 上失败。Riedl (arXiv:2510.05174) 在 population 中测量 higher-order synergy，发现 **只有** ToM-prompt condition 会产生 identity-linked differentiation 和 goal-directed complementarity；lower-capacity LLMs 只表现出 spurious emergence。也就是说，coordination emergence 是 prompt-conditional 且 model-dependent 的，不是免费的。本课实现一个最小 ToM-aware agent，在有无 ToM prompting 的情况下运行 cooperative task，并按 Riedl 2025 protocol 测量 coordination delta。

**类型：** 学习 + 构建
**语言：** Python (stdlib)
**先修：** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 17 (Generative Agents)
**时间：** ~75 分钟

## 要解决的问题

Multi-agent coordination 常常看起来很神奇：agents 分工、预判彼此、避免重复。通常这种 “emergence” 是 prompt engineering 的 artifact：有人告诉 agents “coordinate”。移除 prompt，就移除 coordination。

Riedl 2025 的发现更严格：在 controlled conditions 下，coordination 只会在 agents 被 prompt 去推理 **other agents' minds**（ToM）时涌现。没有 ToM prompt，即使强模型也会表现出无法经受 statistical controls 的 coordination patterns。这对 production 很重要：团队会交付 “multi-agent coordination” features，但它们 prompt-dependent 且 brittle。

本课把 ToM 作为一种具体能力（推理 beliefs about beliefs），构建一个最小 ToM-aware agent，并衡量真实 coordination 与 prompt dressing 之间的区别。

## 核心概念

### ToM 的含义

发展心理学中：3 岁儿童认为每个人的 inner world 都与自己相同。5 岁儿童理解他人有不同 beliefs。7 岁儿童会推理 beliefs about beliefs（“她认为我认为球在杯子下面”）。这些分别是 zeroth、first 和 second-order ToM。

对 LLM agents 来说，ToM orders 映射为：

- **Zeroth-order:** 没有 others model。agent 只基于自己的 observations 行动。
- **First-order:** agent 拥有每个 other agent 的 beliefs model。“Alice believes X.”
- **Second-order:** agent 建模 recursive beliefs。“Alice believes that Bob believes X.”

Li et al. 2023 发现 first- 和 second-order ToM 会在 cooperative games 中的 LLM agents 身上涌现，但会随着 long horizon 和 unreliable communication 退化。

### 简述 Sally-Anne test

1985 年 false-belief test：Sally 把一颗 marble 放进 basket A，然后离开。Anne 把它移动到 basket B。Sally 回来时会看哪里？拥有 first-order ToM 的孩子会说 basket A（Sally 的 belief 与 reality 不同）。没有 ToM 的孩子会说 basket B。

GPT-4-era LLMs 在直接提出 Sally-Anne-style tests 时能通过。叙事很长、场景变化数次，或问题以间接方式表述时，它们会失败。这就是 2026 年 production LLMs 中 ToM 的实际状态。

### Riedl 的 coordination measurement

Riedl (arXiv:2510.05174) 构建了 population-scale test：N 个 agents、一个 cooperative objective、可变 prompt conditions。测量：

1. **Identity-linked differentiation.** agents 是否随时间发展出稳定 role distinctions？
2. **Goal-directed complementarity.** agents 的 actions 是否彼此 complement（不同 subtasks），而不是 duplicate？
3. **Higher-order synergy.** 一种统计度量，用来判断 group 是否达到了任何 subset 都无法达到的结果。

结果：只有在 ToM prompt condition 下，三个 metrics 都产生高于 baseline 的 signal。没有 ToM prompting 时，moderate-capacity models 的 metrics 接近 chance。Large models 在没有 explicit ToM prompting 时表现出一些 coordination，但效果小于 explicit prompting。

### Coordination illusion

没有 statistical controls，demos 中的 “emergent coordination” 往往反映：

- Prompt engineering 把 coordination 写进去了（system prompts 写着 “work together”）。
- Observer bias（我们看到自己期待的 patterns）。
- 成功 runs 的 post-hoc selection。

没有 measurable signal 却宣传 “emergent coordination” 的 production systems，应该被当成 marketing。先测量，再声称。

### 一个最小 ToM-aware agent

结构：

```text
agent state:
  own_beliefs:    {facts the agent believes}
  other_models:   {other_agent_id -> {beliefs_the_agent_attributes_to_them}}
  actions_last_N: [history of others' actions]

observation update:
  - update own_beliefs from direct observation
  - update other_models[agent_id] from their action + prior beliefs

action selection:
  - enumerate candidate actions
  - for each, predict what each other agent will do next given their modeled beliefs
  - pick action that maximizes joint outcome under those predictions
```

`other_models` attribute 是 ToM state。First-order ToM 只保持一层。Second-order 增加 `other_models[i][other_models_of_j]`：我认为 agent i 认为 agent j 相信什么。

### 为什么 long-horizon 会伤害 ToM

Li et al. 记录了：context limits 会导致 agents 忘记哪个 belief 属于谁。Hallucination 会向 other-agent models 添加 false beliefs。两者都会产生随时间复合的 “I thought he thought X” errors。

论文和 2024-2026 follow-ups 中记录的 mitigations：

- **Explicit ToM state in the prompt.** 结构化格式：`{agent_id: belief_list}`。强制 retrieval 保留 identity-belief binding。
- **Shorter reasoning chains.** 每 turn 更少 ToM updates，减少 compounding hallucination。
- **External ToM store.** 在 LLM context 外维护 model；每 turn 只注入相关部分。

### ToM 在生产中哪里失败

- **Adversarial settings.** ToM 好的 agents 更容易被操纵（你可以建模它们对你的建模，然后 exploit）。
- **Heterogeneous teams.** 模型不同时，适用于一个 opponent 的 ToM model 不会泛化。
- **Ground-truth-dependent tasks.** ToM 关于 beliefs；如果 correctness 依赖 facts，ToM 可能是 distraction。

### 你能实际测量的 coordination

三个实用 signals，可判断团队 coordination 是真实的，而不是 prompt-dressed：

1. **Complementarity over time.** 在 multi-turn task 中，agents 的 actions 是否覆盖 disjoint sub-tasks？
2. **Anticipation.** agent A 在 turn T+1 的 action 是否依赖对 B 在 T+2 的 action 的预测，而且预测最终正确？
3. **Correction.** 当 A 在 turn T 误读 B 的 belief，A 是否在 T+2 前 correction？

这些可以在 logged multi-agent system 中测量。它们是 “coordination” 叙事的实质版本。

## 动手实现

`code/main.py` 实现：

- `ToMAgent`：追踪 own beliefs 和 per-other-agent belief models。
- 一个 cooperative task：三个 agents 必须从三个 boxes 中收集三个 tokens；每个 box 只能放一个 token。agents 不能通信；它们从彼此 actions 推断 intent。
- 两种 configurations：`zeroth_order`（无 ToM）和 `first_order`（带一层 belief model 的 ToM）。
- 200 次 randomized trials 上的 measurement：completion rate、duplication rate（两个 agents targeting 同一个 box）、average turns to completion。

运行：

```text
python3 code/main.py
```

预期输出：zeroth-order agents 的 duplicate effort rate 约 35%，在 10 turns 内完成约 60% trials。First-order ToM agents duplicate 约 5%，完成约 95%。delta 就是可测量 coordination effect。

## 实际使用

`outputs/skill-tom-auditor.md` 是一个 skill，用于 audit multi-agent system 的 “emergent coordination” 声称。检查 prompt dressing、与 control 的 statistical significance，以及 measured complementarity。

## 交付成果

Coordination claims checklist：

- **Control condition.** 你的系统去掉 coordination prompt 的版本。两者都要测。
- **Statistical test.** system 与 control 在 metric 上的差异是否达到 `p < 0.05` 显著？
- **Complementarity measure.** 随时间的 action-disjointness，而不只是 final success。
- **Failure-case log.** agents miscoordinate 时，ToM state 长什么样？
- **Model-capacity disclosure.** 如果效果在 smaller models 上消失，要说出来。

## 练习

1. 运行 `code/main.py`。确认 first-order ToM 将 duplication rate 降低约 7x。当你扩展到 5 agents 和 5 boxes 时，gap 是否仍然存在？
2. 实现 second-order ToM（agent A 建模 B 对 C 的看法）。它相对 first-order 是否改进？在哪些 tasks 上？
3. 向 ToM state 注入一个 **hallucination**：每 turn 随机翻转一个 belief。first-order performance 会退化多少？
4. 阅读 Li et al. (arXiv:2310.10701)。复现 “long-horizon degradation” 发现：turns 从 10 增长到 30 时，你的 first-order ToM performance 如何变化？
5. 阅读 Riedl 2025 (arXiv:2510.05174)。在 simulation logs 上实现 higher-order synergy statistic。没有 ToM prompt condition 时，效果是否存在？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Theory of Mind | “Understanding others' minds” | 建模另一个 agent beliefs 的能力。按 order（0、1、2+）分级。 |
| Sally-Anne test | “The false-belief test” | 1985 developmental psychology；LLMs 能通过 plain versions，但 complex ones 会失败。 |
| First-order ToM | “A believes X” | 建模一个 other 对 facts 的 beliefs。 |
| Second-order ToM | “A believes B believes X” | 更深一层的 recursive modeling。 |
| Identity-linked differentiation | “Stable roles over time” | Riedl 的 metric：roles persist，而不是随机。 |
| Goal-directed complementarity | “Disjoint actions” | agents targeting different subtasks，而不是同一个。 |
| Higher-order synergy | “Group exceeds any subset” | Riedl 用于 real coordination 的 statistical measure。 |
| Coordination illusion | “It looks coordinated” | 没有 measurable signal 的 prompt-dressed coordination appearance。 |

## 延伸阅读

- [Li et al. — Theory of Mind for Multi-Agent Collaboration via Large Language Models](https://arxiv.org/abs/2310.10701) — cooperative games 中的 emergent ToM；long-horizon failure modes
- [Riedl — Emergent Coordination in Multi-Agent Language Models](https://arxiv.org/abs/2510.05174) — population-scale measurement；ToM prompting 是 load-bearing condition
- [Premack & Woodruff — Does the chimpanzee have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-chimpanzee-have-a-theory-of-mind/1E96B02CD9850E69AF20F81FA7EB3595) — ToM concept 的 1978 起源
- [Baron-Cohen, Leslie, Frith — Does the autistic child have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-autistic-child-have-a-theory-of-mind/) — Sally-Anne paper（1985）
