# Evaluation and Coordination Benchmarks

> 五个 2025-2026 benchmarks 覆盖 multi-agent evaluation space。**MultiAgentBench / MARBLE**（ACL 2025, arXiv:2503.01935）用 milestone KPIs 评估 star/chain/tree/graph topologies；**graph 最适合 research**，cognitive planning 增加约 3% milestone achievement。**COMMA** 评估 multimodal asymmetric-information coordination；包括 GPT-4o 在内的 state-of-the-art models 很难击败 random baseline。**MedAgentBoard**（arXiv:2505.12371）覆盖四个 medical task categories，经常发现 multi-agent 并不 dominate single-LLM。**AgentArch**（arXiv:2509.10769）benchmark 组合 tool-use + memory + orchestration 的 enterprise agent architectures。**SWE-bench Pro**（[arXiv:2509.16941](https://arxiv.org/abs/2509.16941)）有 41 个 repos、1865 个 problems，覆盖 business apps、B2B services 和 developer tools；frontier models 在 Pro 上约 23%，而 Verified 上 70%+：这是 contamination 的 reality check。Claude Opus 4.7（2026 年 4 月）报告在 Pro 上达到 **64.3%**，带 explicit agent-teams coordination（尚无 Anthropic primary source published：作为 preliminary 对待）；Verdent（agent scaffold）在 Verified 上达到 **76.1% pass@1**（[Verdent technical report](https://www.verdent.ai/blog/swe-bench-verified-technical-report)）。**AAAI 2026 Bridge Program WMAC**（https://multiagents.org/2026/）是 2026 community focal point。本课基于 MARBLE metrics，运行 topology-vs-metric sweep，并固定 “just passing SWE-bench Verified is not evidence of generalization” 规则。

**类型：** 学习
**语言：** Python (stdlib)
**先修：** Phase 16 · 15 (Voting and Debate Topology), Phase 16 · 23 (Failure Modes)
**时间：** ~75 分钟

## 要解决的问题

当论文声称 “our multi-agent system is better” 时，问题是：better than what、on what、measured how？2023-2024 年的 multi-agent evaluation 很混乱：每个人选择自己的 metrics、baselines 和 task sets。2025-2026 benchmarks 引入了结构。

没有 shared benchmarks，你无法有意义地比较两个 multi-agent systems。更糟的是，没有 hold-out benchmarks，frontier models 可能 contaminate。SWE-bench Verified 到 2025 年中已部分进入 training corpora；frontier scores 膨胀；Pro 被设计成 uncontaminated reality check。

本课枚举 2026 年五个 canonical benchmarks，说明每个测量什么，并训练你 skeptical 地阅读 benchmark claims。

## 核心概念

### MultiAgentBench (MARBLE)：ACL 2025

arXiv:2503.01935。在 research、coding 和 planning tasks 上评估四种 coordination topologies（star、chain、tree、graph）。Milestone-based KPIs 追踪 partial progress，而不只是 final success。

测量结果：

- **Graph** topology 最适合 research scenarios；支持 any-to-any critique。
- **Chain** 最适合 stepwise-refinement coding。
- **Star** 最适合 fast-factual consolidation。
- **Coordination tax** 在 graph 超过约 4 个 agents 后出现。
- **Cognitive planning** 在各 topologies 上增加约 3% milestone achievement。

使用场景：你想 apples-to-apples 比较 coordination topologies。MARBLE repo（https://github.com/ulab-uiuc/MARBLE）提供 evaluator。

### COMMA：multimodal asymmetric information

覆盖 agents 有不同 observation modalities，并且必须在不完全 information sharing 下 coordinate 的 tasks。报告结果令人不舒服：包括 GPT-4o 在内的 frontier models 在 COMMA 的 agent-agent collaboration 上很难击败 **random baseline**。信号是 multi-agent modalities 训练和评估都不足：LLMs 能相对处理 single-modality cooperation；multi-modality coordination 会 collapse。

使用场景：你的系统有 multimodal 或 asymmetric-information coordination。COMMA 的 null result 是一个警告：先测量，再声称。

### MedAgentBoard：domain stress test

arXiv:2505.12371。四个 medical task categories：diagnosis、treatment planning、report generation、patient communication。比较 multi-agent vs single-LLM vs conventional rule-based systems。

发现：multi-agent 在大多数 categories 上 **并不** dominate single-LLM。multi-agent advantage 很窄：当 subtasks 明确可分（diagnosis + treatment）时，task decomposition 有帮助；当 coordination overhead 超过 specialization gain（report generation）时，它会伤害。

使用场景：你的 domain 有明确 single-LLM baselines。如果 MedAgentBoard 的 lesson 能泛化，很多 proposed multi-agent systems 都是 over-engineered。

### AgentArch：enterprise architectures

arXiv:2509.10769。带 tool use、memory 和 orchestration 层叠在一起的 enterprise settings。Benchmark 隔离每一层贡献：adding tools 有多少帮助？adding memory 呢？adding multi-agent orchestration 呢？

使用场景：你正在设计 enterprise agent stack，需要证明每一层的价值。AgentArch 帮你避免购买无法测量价值的 features。

### SWE-bench Pro：reality check

arXiv:2509.16941。41 个 repositories、1865 个 problems，覆盖 business apps、B2B services 和 developer tools。设计为对 later training cutoffs **uncontaminated**。Frontier models 在 Pro 上约 23%，而在 Verified 上 70%+。这个 gap 是 contamination signal。

2026 年 4 月 scores：
- Claude Opus 4.7 on Pro：**64.3%**（报告带 explicit agent-teams coordination；尚无 Anthropic primary source published：作为 preliminary 对待）。
- Verdent（agent scaffold）on Verified：**76.1% pass@1**（[technical report](https://www.verdent.ai/blog/swe-bench-verified-technical-report)）。
- 没有 agent scaffolding 的 frontier raw scores on Pro：约 23-35%（[SWE-bench Pro paper](https://arxiv.org/abs/2509.16941)）。

要点：“we beat SWE-bench Verified” 已经不再是 capability 证据。Pro 是当前 gating test。Agent-team scaffolding 在 Pro 上产生 measurable gains（约 30-40 点 delta），这是 2026 年 multi-agent coordination 最强的 empirical arguments 之一。

### AAAI 2026 WMAC

AAAI 2026 Bridge Program：Workshop on Multi-Agent Coordination（https://multiagents.org/2026/）。这是 2026 multi-agent AI research 的 community focal point。Accepted papers 和 workshop proceedings 是评估 new methods 的 canonical venue；production decisions 上应优先相信 WMAC-accepted claims，而不是 arXiv preprints。

### Skeptically 阅读 benchmark claims：2026 checklist

当有人声称 multi-agent result：

1. **Which benchmark, which split?** SWE-bench Verified vs Pro 差别很大。报在错误 split 上的数字没有价值。
2. **Contamination check.** benchmark 是否在 model training cutoff 之后发布？如果不是，谨慎对待。
3. **Baseline comparison.** 对比 single-LLM baseline、random、prior multi-agent work。不是“对比同一系统的 untuned version”。
4. **Statistical significance.** N trials、p-value、confidence interval。Frontier models high-variance；single runs 会误导。
5. **Task diversity.** 一个 task 还是很多？Generalization 对 production 很重要。
6. **Cost disclosure.** 每个 task 的 tokens、wall-clock。20x cost 的 90% solution 是 business decision，不是 capability claim。

### 这些 benchmarks 都没测好的东西

- **Long-horizon coordination.** 多天 wall-clock interaction。目前所有 benchmarks 都短。
- **Adversarial resilience.** 当一个 agent malicious 或 compromised 时会怎样？
- **Drift under deployment.** Benchmarks 是 static；production distributions 会 shift。
- **Cost-normalized performance.** 大多数 benchmarks 报 raw accuracy，而不是 accuracy-per-dollar。

为你真正关心的 axis 构建自己的 internal benchmark，往往是正确选择。

## 动手实现

`code/main.py` 是一个 non-interactive walk-through：

- 模拟 3 个 multi-agent systems 处理 toy task。
- 为每个计算 MARBLE-style milestone metrics。
- 通过从 “training” set 中 withholding tasks 来运行 contamination check。
- 显式与 random baseline 对比。
- 打印 benchmark-claims scorecard。

运行：

```bash
python3 code/main.py
```

预期输出：system scorecard，包含 raw accuracy、milestone achievement、cost-per-task、vs-random baseline delta，以及 contamination-check note。

## 实际使用

`outputs/skill-benchmark-reader.md` 会读取任何 multi-agent benchmark claim 并应用 scrutiny checklist。输出：grade 和 caveats。

## 交付成果

Production evaluation discipline：

- **Build an internal benchmark**，反映你的实际 production distribution。Public benchmarks 提供信息，但不能替代。
- **Include a random baseline** in every comparison。如果你在 coordination task 上无法大幅击败 random，task 可能 ill-posed。
- **Report cost alongside accuracy.** Token cost 和 wall-clock。Ops teams 两者都需要。
- **Rebuild the benchmark quarterly.** Production distribution 会 shift；stale benchmarks 会误导。
- **Avoid published-benchmark overfitting.** 如果团队专门优化 SWE-bench Pro numbers，production 会 regress。

## 练习

1. 运行 `code/main.py`。识别三个 simulated systems 中哪个 cost-per-milestone 最好。它是否也是 raw-accuracy 最高的系统？
2. 阅读 MultiAgentBench（arXiv:2503.01935）。针对你自己的 task domain，决定 MARBLE 会推荐四种 topologies 中哪一种。用论文结果辩护。
3. 阅读 SWE-bench Pro paper。它具体通过什么做到 contamination-resistant？同样 technique 能否应用到你关心的其他 benchmarks？
4. 阅读 COMMA 关于 multimodal coordination 的发现。设计一个你可以加到 internal benchmark 的简单 multimodal coordination task。什么算 useful signal？
5. 将 benchmark-claims checklist 应用于一篇近期 multi-agent paper 的 headline result。你会给这个 claim 什么 grade？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| MARBLE | “MultiAgentBench” | ACL 2025；star/chain/tree/graph topologies，带 milestone KPIs。 |
| COMMA | “Multimodal benchmark” | Multimodal asymmetric-info coordination；frontier models 很难胜过 random。 |
| MedAgentBoard | “Domain stress test” | 四个 medical categories；经常发现 multi-agent 不 dominate single-LLM。 |
| AgentArch | “Enterprise benchmark” | Tools + memory + orchestration layered。 |
| SWE-bench Pro | “Contamination-resistant” | 1865 problems、41 repos；Pro 上约 23% vs Verified 上 70%+（contamination signal）。 |
| Milestone achievement | “Partial credit” | reward progress 的 benchmarks，而不只 reward final success。 |
| Contamination | “Benchmark leaked into training” | 发布后，benchmarks 漂入 training corpora；scores 膨胀。 |
| WMAC | “AAAI 2026 Bridge Program” | Workshop on Multi-Agent Coordination；community focal point。 |

## 延伸阅读

- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935) — 带 milestone KPIs 的 topology benchmark
- [MARBLE repository](https://github.com/ulab-uiuc/MARBLE) — reference implementation
- [MedAgentBoard](https://arxiv.org/abs/2505.12371) — domain stress test；multi-agent 经常不 dominate
- [AgentArch](https://arxiv.org/abs/2509.10769) — enterprise agent architectures
- [SWE-bench leaderboards](https://www.swebench.com/) — frontier models 的 Verified 和 Pro scores
- [AAAI 2026 WMAC](https://multiagents.org/2026/) — 2026 community focal point
