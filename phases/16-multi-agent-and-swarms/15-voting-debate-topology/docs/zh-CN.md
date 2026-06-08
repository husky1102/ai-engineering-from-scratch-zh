# Voting、Self-Consistency 与 Debate Topology

> 最便宜的 aggregation：采样 N 个 independent agents，然后 majority-vote。Wang et al. 2022 的 self-consistency 用一个模型采样 N 次做了这件事。Multi-agent 把它扩展为**异构** agents 来逃离 monoculture：不同 models、不同 prompts、不同 temperatures、不同 contexts。超过 majority vote 后，debate topology 很重要：MultiAgentBench（arXiv:2503.01935, ACL 2025）评估了 star / chain / tree / graph coordination，发现**graph 最适合 research**，但超过 ~4 个 agents 后会出现“coordination tax”。AgentVerse（ICLR 2024）记录了两种 emergent patterns：volunteer behaviors 和 conformity behaviors；conformity 既是功能（找到 consensus），也是风险（groupthink，Lesson 24）。本课绘制 topology space，构建每种 variant，并测量 coordination tax。

**类型：** Learn + Build
**语言：** Python (stdlib)
**先修：** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 14 (Consensus and BFT)
**时间：** ~75 分钟

## 要解决的问题

Debate 可以提高 accuracy（Du et al., arXiv:2305.14325）。它也可能降低 accuracy。Debate 是否有帮助取决于四个结构性选择：

1. 谁和谁说话（topology）。
2. 有多少 rounds（Du 2023：rounds 和 agents 都独立重要）。
3. agents 是否异构（不同 base models 打破 monoculture）。
4. 是否存在 adversarial voice（steel-manning vs. straw-manning）。

很多团队把“run 5 agents and vote”硬接到任务上，结果常常比 single agent 更差。失败不是随机的。它们跟 topology 和 heterogeneity 对齐。本课就是 topology map。

## 核心概念

### Self-consistency，单模型 baseline

Wang et al. 2022（"Self-Consistency Improves Chain of Thought Reasoning"）在 temperature > 0 下对同一模型采样 N 次，并对 reasoning-path answers 做 majority-vote。在 GSM8K 上，N=40 samples 相比 single greedy decode 有显著提升。Self-consistency 是 multi-agent voting 的 single-agent 前身。

限制：self-consistency 使用一个 base model。错误天然相关。如果模型有 systematic bias，全部 N 个 samples 都共享它。

### Multi-agent vote，异构扩展

把 N 个 samples 替换成 N 个*不同* agents。不同 base models（Claude、GPT、Llama）、不同 prompts、不同 tool access。收益：uncorrelated errors。代价：不同 agents 成本不同；协调它们会增加 overhead。

2026 年对 heterogeneous debate 的 canonical name 是 **A-HMAD** — Adversarial Heterogeneous Multi-Agent Debate。这个名称不是普遍采用，但论文会用它指“不同模型进行 debate，从而减少 monoculture collapse 带来的 correlated errors”。

### 四种 topologies

```text
star                chain               tree                graph

    ┌─A─┐           A─B─C─D         ┌──A──┐              A───B
    │   │                           │     │              │ × │
    B   C                           B     C              D───C
    │   │                          / \   / \
    D   E                         D   E F   G           (fully connected)
```

Star：一个 hub，其他所有 agents 只和 hub 说话。等价于没有 back-channel 的 supervisor-worker。
Chain：线性结构，每个 agent 看到前一个 agent 的 output。像 pipeline。
Tree：层级结构，由 hierarchical agent systems 使用（Lesson 06）。
Graph：any-to-any。包括 fully-connected clique 和 arbitrary DAGs。

### Coordination tax（MultiAgentBench）

MultiAgentBench（MARBLE, ACL 2025, arXiv:2503.01935）在包含 research、coding 和 planning 的 task suite 上 benchmarked star、chain、tree、graph。关键测量结果：

- **Graph** topology 在 research tasks 上胜出。Information any-to-any flow；agents 可以互相 critique。
- **Star** 在 fast-answer factual tasks 上胜出。Hub 负责过滤和 consolidation。
- **Chain** 在 stepwise pipelines（staged refinement）上胜出。
- **Coordination tax** 在 graph topology 中超过 ~4 个 agents 后出现。Wall-clock 和 token cost 增长快于 quality。

4-agent ceiling 是经验性的，不是根本限制。它反映了 2026 年 LLM context capacity：每个 agent 的 context 被 peers 的 outputs 填满，而一旦所有人能看到所有人，加入 agent N+1 的 marginal value 就下降。

### Multi-Agent Debate Strategies（"Should we be going MAD?"）

arXiv:2311.17371 是 2023 年关于 MAD strategies 的 survey。被其他工作复现的关键发现：与 self-consistency 在结构上相似的 MAD variants（independent sampling + aggregation），在相同预算下常常弱于 self-consistency。MAD 最有帮助的场景是 agents 真正异构，并且 debate 有 adversarial structure（一个 agent 负责反驳）。

### AgentVerse emergent patterns

AgentVerse（ICLR 2024, https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf）记录了 multi-agent debate 中即使没有显式设计也会出现的两种 behaviors：

- **Volunteer。** agent 主动提供帮助（“I can take the next step”）。有用之处：它把工作分配给对某个 subtask 最 capable 的 agent。
- **Conformity。** agent 调整自己的 stance 来匹配 critic，即使 critic 是错的。这是 debate 版本的 sycophancy（Lesson 14）。

Conformity 说明 debate-until-agreement 会奖励 bullies。Bounded rounds 加 separate judge 可以缓解。

### Heterogeneity：真正推动 accuracy 的旋钮

2024-2026 年 practical literature 中的一个模式是：把 N 个 agents 中的一个换成不同 base model，通常比把 N 增加 1 带来更大的 accuracy bump。直觉是 monoculture：每个新的 independent-error source 都比额外的 correlated sample 更有价值。

极限情况下，heterogeneity 胜过 numerosity。在大多数有清晰 ground truth 的 tasks 上，三个不同 models 胜过五个同一模型副本。

### Jury methods

Sibyl framework（在 Minsky-LLM literature 中被引用）形式化了一个“jury”：一小组 specialized agents 在每个阶段通过 voting refine answers。不同于 plain majority vote，jury 有 roles：一个 agent cross-examines，一个提供 context，一个为 plausibility 打分。Jury methods 处在 plain vote（便宜、monoculture-prone）和 full MAD（昂贵、conformity-prone）之间。

### Vote-with-debate 什么时候占优

- 问题有 ground truth（fact、math、code behavior）。Vote convergence 有意义。
- Agents 可以访问不同 sources 或 tools（heterogeneity 可用）。
- Rounds 有边界（通常 2-3）并且有 separate judge 或 verifier。
- 预算允许 3-5 个 agents。在 graph topology 上超过 5-7 后，coordination tax 占主导。

### Vote-with-debate 什么时候有害

- 问题像 opinion。Agents 会收敛到看起来最自信的答案，而不是最正确的答案。
- 所有 agents 共享一个 base model。Monoculture 让 consensus 失去意义。
- Rounds 无边界。Conformity 每次都会赢。
- 任务很简单。一个 single agent 加 N=5 的 self-consistency 更便宜，也同样准确。

## 动手实现

`code/main.py` 实现：

- `run_star(agents, hub, question)` — hub 轮询每个 worker 并 aggregate。
- `run_chain(agents, question)` — sequential refinement。
- `run_tree(root, children, question)` — depth-2 aggregation 的 hierarchical。
- `run_graph(agents, question, rounds)` — all-to-all debate，bounded rounds。
- 一个 scripted heterogeneity dial：每个 agent 都有 `error_bias`，表示它的 systematic wrongness。
- 一个 measurement harness，在 N=3、5、7 下运行每种 topology，并报告 (accuracy, total_tokens, wallclock_simulated)。

运行：

```text
python3 code/main.py
```

预期输出：topology × N → (accuracy, tokens, latency) 的表格。Graph 在 N=3-5 的 research-style tasks 上胜出；star 在 fast-factual tasks 上胜出；N=7 的 graph 显示 coordination tax（latency 增长快于 accuracy）。

## 实际使用

`outputs/skill-topology-picker.md` 是一个 skill，它读取 task description 并推荐 topology（star / chain / tree / graph）、N（number of agents）、heterogeneity profile（要使用的 base models）和 round bound。

## 交付成果

对任何 ensemble：

- 从使用一个强 base model 的 **self-consistency at N=5** 开始。它是便宜 baseline。
- 如果 accuracy 重要，升级到 **heterogeneous voting at N=3**。测量 delta。
- 只有当 task 有结构（research、multi-step）且 bounded rounds 可行时，才升级到 **debate topology**。
- 始终记录 minority cluster。当某个 minority 持续正确时，你得到了 diversity signal。
- 把 wall-clock 和 tokens 与 accuracy 一起 benchmark。“用 10x cost 换更好 accuracy”是业务决策。

## 练习

1. 运行 `code/main.py`。画出 graph topology 的 coordination-tax curve：accuracy vs N，tokens vs N。曲线在什么 N 出现拐点？
2. 实现 A-HMAD：三个 agents 具有刻意不同的 biases。与 Lesson 14 中 monoculture attack 的 all-same-bias baseline 相比，A-HMAD 表现如何？
3. 给 graph topology 添加一个 "judge" role，它不投票，只为 final consensus 打分。这会改变 emergent conformity behavior 吗？
4. 阅读 AgentVerse paper（ICLR 2024）。找出你的 implementation 最强地展现了哪种 emergent behavior。能否通过 prompt change 引出相反 behavior？
5. 阅读 MultiAgentBench（arXiv:2503.01935）Section 4（topology experiments）。使用你的 harness，在论文中的一个 task 上复现“graph-wins-research”结果。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Self-consistency | “Sample N times, vote” | Wang 2022。Single model，N 个 temperature>0 samples，对 reasoning paths 做 majority vote。 |
| Heterogeneity | “Different models” | 不同 base models 或 prompt families 的 ensemble。打破 monoculture。 |
| MAD | “Multi-agent debate” | agents 在多轮中交换 critiques 的泛称。见 Du 2023。 |
| A-HMAD | “Adversarial Heterogeneous MAD” | 强调 different models + adversarial structure 的 MAD variant。 |
| Topology | “谁和谁说话” | Star、chain、tree、graph。决定 information flow。 |
| Coordination tax | “收益递减” | graph 中超过 ~4 个 agents 后，cost 增长快于 quality。 |
| Volunteer behavior | “主动帮助” | AgentVerse emergent pattern：agent 主动提出承担一步。 |
| Conformity behavior | “压力下同意” | AgentVerse emergent pattern：agent 与 critic 对齐。 |
| Jury | “Small specialized panel” | 带 roles（examiner、context、scorer）的 Sibyl-style ensemble。 |

## 延伸阅读

- [Wang et al. — Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) — single-model baseline
- [Du et al. — Improving Factuality and Reasoning via Multiagent Debate](https://arxiv.org/abs/2305.14325) — agents 和 rounds 都独立重要
- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935) — topology benchmark，显示 graph 最适合 research、chain 适合 pipelines
- [Should we be going MAD?](https://arxiv.org/abs/2311.17371) — MAD-strategy survey；发现相同预算下 MAD 常输给 self-consistency
- [AgentVerse (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf) — volunteer 和 conformity emergent patterns
- [MARBLE repo](https://github.com/ulab-uiuc/MARBLE) — reference benchmark implementation
