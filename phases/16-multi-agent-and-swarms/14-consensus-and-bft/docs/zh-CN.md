# 面向 Agents 的 Consensus 与 Byzantine Fault Tolerance

> 经典 distributed-systems BFT 遇上随机性的 LLM。2025-2026 年出现了三个研究方向：**CP-WBFT**（arXiv:2511.10400）用 confidence probe 为每票加权；**DecentLLMs**（arXiv:2507.14928）采用 leaderless 方式，用 parallel worker proposals 和 geometric-median aggregation；**WBFT**（arXiv:2505.05103）把 weighted voting 与 Hierarchical Structure Clustering 结合，把节点分成 Core 和 Edge。来自 "Can AI Agents Agree?"（arXiv:2603.01213）的诚实 empirical result 是：今天即使 scalar agreement 也很脆弱，一个 deceptive agent 就能 compromise 一个 Mixture-of-Agents。BFT 必要但不充分。本课构建一个最小 BFT protocol，注入三种 agent-specific attacks（byzantine lie、sycophantic conformity、correlated-error monoculture），并测量每种 consensus variant 如何应对。

**类型：** Learn + Build
**语言：** Python (stdlib)
**先修：** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 13 (Shared Memory)
**时间：** ~75 分钟

## 要解决的问题

你有 N 个 LLM agents，每个都产生一个 answer。它们不一致。Majority vote 选错了，因为两个 agents 是 correlated 的（同一 base model、同一 training data、同一 failure modes）。第三个 agent 碰巧以新颖方式犯错，于是 majority 变成 false majority。

现在加入一个 deceptive agent：它故意撒谎。或者一个 sycophantic agent：它同意最后发言的人。在经典 BFT 中，假设 Byzantine nodes 是一个比例 `f < n/3`，并且行为任意。2026 年的现实是：LLM nodes 即使诚实也有随机性，跨模型存在相关性，还会受彼此输出影响。你不能把它们当作独立的 Bernoulli voters。

经典 BFT（PBFT, 1999）并不是错的，而是不完整。它处理 arbitrary bit-flipping。它不处理“三个诚实 agents 因为共享 training data 而共享一个 hallucination”。本课从 PBFT 的基础出发，再叠加三种 2025-2026 年的 adaptation。

## 核心概念

### 经典 BFT 给你什么

Practical Byzantine Fault Tolerance（Castro & Liskov, OSDI 1999）容忍 `f < n/3` 的 Byzantine nodes。协议有三个 phases（pre-prepare、prepare、commit）和两个 primitives（signed messages、quorum certificates）。目标是在 `n >= 3f + 1` 个 honest-or-malicious nodes 之间对单个 value 达成 agreement。

这些 guarantees 很强，但假设：

1. **Independent faults。** Byzantines 不协同。
2. **Honest nodes 真正诚实。** honest outputs 的 correctness 不是问题；protocol 只对齐 disagreement。
3. **问题有 ground-truth answer。** 对错误事实达成 consensus 仍然是 consensus。

LLM agents 违反了这三点。两个运行同一 base model 的 agents 共享 faults。一个“honest” LLM 仍然会 hallucinate。面对 ambiguous questions，“truth” 是 agents 决定的东西，并没有 external oracle。

### 三种 LLM-specific attacks

**Byzantine lie。** 一个 agent 输出故意错误的 answer。如果 `f < n/3`，经典 BFT 能处理它。

**Sycophantic conformity。** 一个 agent 在 voting 前读取其他人的 answers，并与最后发言者保持一致。不一定 malicious，但会与最响亮的 voice 相关。经典 BFT 不能阻止它，因为这个 agent 通过了每个 signature check。

**Correlated-error monoculture。** 三个 agents 共享一个 base model。它们 hallucinate 同一个错误 answer。majority 是错的。经典 BFT 无济于事，因为三个 agents 都“诚实地”一致。

### 2025-2026 年的 responses

**CP-WBFT**（arXiv:2511.10400）— Confidence-Probed Weighted BFT。每个 voter 给自己的 answer 附加 confidence probe（self-reported probability，或 separate calibration model 的 prediction）。Vote weights 随 confidence 缩放。Reported +85.71% BFT improvement on complete graphs。缓解对象：sycophantic conformity（conforming agents 对它们主动接受的位置往往 confidence 较低）。

**DecentLLMs**（arXiv:2507.14928）— Leaderless。Worker agents 并行提出 proposals，evaluator agents 评分 proposals，final answer 是 scored positions 的 geometric median。当 `f < n/2` 时 robust。缓解对象：Byzantine lie 和 correlated errors（geometric median 对 outliers robust，会拉向 dense cluster，而不是 model-biased average）。

**WBFT**（arXiv:2505.05103）— Weighted BFT with Hierarchical Structure Clustering。Vote weights 由 response quality 加上从历史学到的 trust score 分配。把 agents 聚类为 Core 和 Edge；Core agents 必须先达成 consensus，Edge agents 跟随。缓解对象：scalability（Core consensus 小且快），并部分缓解 monoculture（Core 可以按 diversity 选择）。

### Empirical："Can AI Agents Agree?"（arXiv:2603.01213）

该论文测量多个 frontier models 之间的 scalar agreement（LLM agents 对单个数值达成一致）。发现令人不舒服：

- 即使没有 adversaries，LLM agents 在许多 benchmarks 的 scalar questions 上 disagreement rate 也超过 30%。
- 一个采用 deceptive persona 的 agent 可以把 Mixture-of-Agents consensus 拉离 honest baseline 40+ percentage points。
- Disagreement rates 与 model diversity 相关：heterogeneous ensembles 分歧更多（好处：uncorrelated errors），但漂移更慢（坏处：time-to-agreement 更长）。

结论：BFT 给你对齐 outputs 的 machinery，但不会告诉你对齐后的 output 是否正确。要结合 verification（Phase 16 · 08 role specialization）、diversity（Phase 16 · 15 debate variants）和 evaluator agents（Phase 16 · 24 benchmarks）。

### 核心协议，简化版

面向 LLM agents 的最小 BFT round：

```text
1. task arrives; each agent i produces answer a_i
2. each agent attaches confidence probe c_i in [0, 1]
3. aggregator collects (a_i, c_i) from all n agents
4. aggregator groups by semantic cluster (equivalent answers)
5. aggregator computes weight for each cluster C:
     w(C) = sum_{i in C} c_i
6. winner = cluster with max weight, if max > threshold * sum(c_i)
   else: retry or escalate
7. minority clusters logged with provenance for post-hoc audit
```

semantic clustering step 是 LLM-specific twist。两个 answers “the study reports 4.2%” 和 “4.2% improvement” 属于同一个 cluster。naive string-equality check 会漏掉这一点。生产中使用便宜的 embedding model 或 explicit canonicalization。

### Threshold tuning

`threshold` 参数决定什么时候 accept，什么时候 retry。太低：你接受 weak majorities。太高：你永远无法接受任何东西。经验范围：对 `n=5-7` agents 通常是 0.5-0.67，更小的 `n` 需要更高阈值。低于阈值时，escalate 给 human 或不同的 agent ensemble。

### Consensus 无法帮助的地方

- **Ambiguous questions。** 如果问题没有 ground truth，consensus 是 opinion。应该这样标注。
- **Compound questions。** “Write code and explain it” 是两个 answers。分别投票。
- **Adversarial multi-round。** 如果 agents 可以观察 prior rounds 并 mimic（Du 2023 debate），它们会开始不顾 truth 地相互同意。限制 rounds（通常 2-3）。

## 动手实现

`code/main.py` 实现：

- `AgentVoter` — 带 (answer, confidence) 的 scripted policy。
- `MajorityVote` — 经典 plurality。
- `CPWBFT` — confidence-weighted voting with semantic clustering。
- `DecentLLMs` — 对 scored proposals 做 geometric-median aggregation。
- `Scenario` — 在三种 attack patterns 下运行每个 aggregator。

实现的 attack patterns：

1. `byzantine`：一个 agent 以 high confidence 撒谎。
2. `sycophancy`：一个 agent 复制它看到的第一个 answer，并匹配 confidence。
3. `monoculture`：三个 agents 共享一个错误 answer（correlated error），confidence 中等。

运行：

```text
python3 code/main.py
```

预期输出：一个 (attack, aggregator) -> final answer 表格，并突出正确 answer。Plurality 在 monoculture case 失败。CPWBFT 的 confidence weighting 缓解 sycophancy。DecentLLMs 的 geometric-median 在 monoculture 少于半数人口时拉向 honest cluster。

## 实际使用

`outputs/skill-consensus-designer.md` 为 multi-agent ensemble 设计 consensus protocol：clustering method、weighting、threshold，以及 sub-threshold rounds 的 escalation policy。

## 交付成果

在发布任何 consensus mechanism 前：

- **至少用上面三种 patterns 做 attack-test。** 你的 protocol 应该可预测地失败，而不是静默失败。
- **记录每个 minority cluster** 及其 provenance。Minority clusters 是 correlated errors 的 early-warning system。
- **强制 bounded rounds。** 不要“keep debating until agreement”：那会奖励 sycophancy。
- **把 agreement 与 correctness 分开。** Consensus output 交给 verifier；verifier 独立于 ensemble。
- **Monitor the agreement rate。** 急剧上升意味着 conformity bias；急剧下降意味着 model drift。

## 练习

1. 运行 `code/main.py`。确认 plurality 在 monoculture attack 中失败，但当 monoculture confidence 低于 0.7 时，CPWBFT 会部分缓解。
2. 添加第四种 attack pattern：**silent abstention** — 一个 agent 拒绝回答（"I don't know"）。每个 aggregator 应如何处理 abstentions？实现你的选择。
3. 把 semantic clustering 从 string canonicalization 换成 embedding-similarity（使用任意 open-source embedding model）。sycophancy attack 会发生什么？
4. 阅读 CP-WBFT（arXiv:2511.10400）。实现 confidence-probe calibration step（一个 separate calibration model 检查每个 agent 的 self-reported confidence）。在 monoculture scenario 上测量 accuracy gain。
5. 阅读 "Can AI Agents Agree?"（arXiv:2603.01213）。复现一个简化的 scalar-agreement experiment：三个 agents、一个 scalar question、deceptive-persona prompt。CPWBFT 或 DecentLLMs 能抓住它吗？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| BFT | “Byzantine fault tolerance” | Castro-Liskov 1999 protocol，用于在 `f < n/3` arbitrary faults 下达成 consensus。 |
| Byzantine | “任何坏行为” | 可以撒谎、丢 messages、静默失败的 node：除了安全 crash 以外什么都可能做。 |
| Confidence probe | “你有多确定？” | 附在 vote 上的 self-reported 或 calibrator-predicted probability。 |
| Semantic clustering | “同一个答案，不同措辞” | 计票前对等价 answers 分组。 |
| Geometric median | “Robust center” | 使到 sample points 的距离和最小的点。与 mean 不同，对 outliers robust。 |
| Monoculture | “同一模型，同一失败” | agents 共享 training data 或 base model 时产生的 correlated errors。 |
| Sycophantic conformity | “同意 loud voice” | agent 的 vote 偏向最先/最大声发言的人。 |
| Core/Edge | “Hierarchical BFT” | WBFT split：小型 Core consensus 先达成，Edge nodes 跟随。限制 latency。 |

## 延伸阅读

- [Castro & Liskov — Practical Byzantine Fault Tolerance (OSDI 1999)](https://pmg.csail.mit.edu/papers/osdi99.pdf) — foundation
- [CP-WBFT — Confidence-Probe Weighted BFT](https://arxiv.org/abs/2511.10400) — 按 confidence 做 vote weighting
- [DecentLLMs — leaderless multi-agent consensus](https://arxiv.org/abs/2507.14928) — geometric-median aggregation
- [WBFT — Weighted BFT with Hierarchical Structure Clustering](https://arxiv.org/abs/2505.05103) — 用于 bounded latency 的 Core/Edge split
- [Can AI Agents Agree?](https://arxiv.org/abs/2603.01213) — scalar-agreement fragility 和 deceptive-persona attack
