# Multi-Agent Debate 与 Collaboration

> Du et al.（ICML 2024，“Society of Minds”）运行 N 个 model instances，让它们独立提出答案，然后经过 R 轮彼此 critique 以收敛。它提高 factuality、rule-following 和 reasoning。Sparse topology 在 token cost 上优于 full mesh。

**类型:** Learn + Build
**语言:** Python（stdlib）
**先修:** Phase 14 · 12（Workflow Patterns），Phase 14 · 05（Self-Refine and CRITIC）
**时间:** ~60 分钟

## 学习目标

- 解释 debate protocol：N 个 proposers、R 轮、收敛到 shared answer。
- 描述为什么 debate 改进 factuality、rule-following 和 reasoning。
- 解释 sparse topology：不是每个 debater 都需要看到每个其他 debater。
- 在 scripted LLM 上实现 stdlib debate，包含 full-mesh 和 sparse variants；测量 token cost vs accuracy。

## 要解决的问题

Self-Refine（Lesson 05）是一个模型 critique 自己，存在 groupthink 风险。CRITIC（Lesson 05）把 critique grounding 到 external tools，但 tools 不总是可用。Debate 引入第三种模式：multiple instances、cross-critique、通过 disagreement convergence。

## 核心概念

### Society of Minds（Du et al., ICML 2024）

- N 个 model instances 对同一问题独立提出答案。
- 经过 R 轮，每个模型读取其他模型的 proposals 并 critique 它们。
- 模型基于 critiques 更新自己的答案。
- R 轮后，返回 convergent answer。

原始实验因为成本使用 N=3、R=2。在难题（MMLU、GSM8K、Chess Move Validity、biography generation）上，更多 agents 和更多 rounds 会提高 accuracy。

Cross-model combinations 胜过 single-model debates：ChatGPT + Bard together > 任一单独模型。

### Sparse topology

“Improving Multi-Agent Debate with Sparse Communication Topology”（arXiv:2406.11776，2024-2025）显示 full-mesh debate 并不总是最优。Sparse topologies（star、ring、hub-and-spoke）能以更低 token cost 匹配 accuracy。每个 debater 只看到 peers 的一个 subset。

影响：

- Full mesh N=5, R=3 = 5 × 3 = 15 proposals，每个读取 4 peers = 60 critique ops。
- Star N=5, R=3（one hub + 4 spokes）= 15 proposals，spokes 只读 hub = 12 critique ops。

### Debate 何时有帮助

- **Factuality.** N 个 independent proposals，cross-check 减少 hallucination。
- **Rule-following.** Chess move validity，一个模型漏掉规则，其他模型捕捉它。
- **Open-ended reasoning.** 多种 framings 收敛到正确答案。

### Debate 何时有害

- **Latency-sensitive UX.** N × R serial rounds 是你可能没有的 latency。
- **Cost-sensitive scale.** 每个问题 N × R tokens。
- **Simple factual lookups.** 一次 lookup 比五个 debates 便宜。

### 2026 practical instantiations

- **Anthropic orchestrator-workers**（Lesson 12）：带 synthesis step 的 debate 变体。
- **LangGraph supervisor**（Lesson 13）：central router + specialist agents 可以把 debate 实现为一个 node。
- **OpenAI Agents SDK**（Lesson 16）：agents 来回 handoff 进行 iterative critique。
- **Multi-agent evals**：debate + evaluator-optimizer 结合，用于 eval signal。

### 这个 pattern 哪里会出错

- **Convergence collapse.** 所有 agents 收敛到第一个错误答案。用 required disagreement rounds 缓解。
- **Hub failure.** Star topology 中，坏 hub 会污染所有人。轮换 hub 或使用多个 hubs。
- **Prompt homogenization.** 所有 agents 使用相同 prompt，产生相同答案。使用 diverse prompts 和/或 models。

## 动手实现

`code/main.py` 实现 stdlib debate：

- `Debater` class（带 per-debater opinion drift 的 scripted LLM）。
- `FullMeshDebate` 与 `SparseDebate` runners。
- 三个问题：一个 factual、一个 rule-based、一个 reasoning。
- Metrics：convergent answer、rounds to convergence、total critique ops。

运行它：

```text
python3 code/main.py
```

输出：per-protocol accuracy 和 cost；sparse 在 2/3 个问题上以更低 cost 匹配 full mesh。

## 实际使用

- **Anthropic orchestrator-workers** 用于简单 2-3-worker debates。
- **LangGraph** 用于带 checkpointing 的 stateful multi-round debate。
- **Custom** 用于 research 或 specialized correctness guarantees。

## 交付成果

`outputs/skill-debate.md` scaffold 一个 multi-agent debate，带 configurable topology、N、R 和 convergence rule。

## 练习

1. 实现“forced disagreement”规则：round 1 中每个 debater 必须产生 distinct proposal。测量对 convergence speed 的影响。
2. 添加 confidence-weighted aggregation：debaters 返回（answer, confidence）；aggregator 按 confidence 加权。它有帮助吗？
3. 把一个“agent”换成带不同 opinions 的不同 scripted LLM。Heterogeneity 会提高 accuracy 吗？
4. 在你的 3 个问题上测量 full mesh vs sparse 的 token cost。绘制 cost vs accuracy。
5. 阅读 Society of Minds paper。把 toy 移植到 N=5、R=3。什么坏了？什么变好了？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Debate | “Multi-agent critique” | N 个 proposers，R 轮 cross-critique，converge |
| Full mesh | “Everyone reads everyone” | 每个 debater 每轮读取每个 peer |
| Sparse topology | “Limited peer view” | Debaters 只读取 peers 的 subset |
| Hub-and-spoke | “Star topology” | 一个 central debater，N-1 spokes 只读 hub |
| Convergence | “Agreement” | Debaters 收敛到 shared answer |
| Society of Minds | “Du et al. debate paper” | ICML 2024 multi-agent debate method |

## 延伸阅读

- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — canonical multi-agent debate
- [Sparse Communication Topology (arXiv:2406.11776)](https://arxiv.org/abs/2406.11776) — sparse topology results
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — orchestrator-workers as a debate variant
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — single-model self-critique counterpart
