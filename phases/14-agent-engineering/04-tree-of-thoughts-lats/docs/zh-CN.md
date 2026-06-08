# Tree of Thoughts 与 LATS：刻意搜索

> 单条 chain-of-thought trajectory 没有 backtrack 空间。ToT（Yao et al., 2023）把 reasoning 变成一棵树，并在每个 node 上 self-evaluation。LATS（Zhou et al., 2024）用 Monte Carlo Tree Search 统一 ToT、ReAct 和 Reflexion。Game of 24 从 4%（CoT）到 74%（ToT）；LATS 在 HumanEval 上达到 92.7% pass@1。

**类型:** Build
**语言:** Python（stdlib）
**先修:** Phase 14 · 01（Agent Loop），Phase 14 · 03（Reflexion）
**时间:** ~75 分钟

## 学习目标

- 把 reasoning framing 成 search：nodes 是“thoughts”，edges 是“expansions”，value 是“有多 promising”。
- 用 stdlib 实现 ToT-style BFS tree search，带 self-evaluation scoring。
- 扩展成 toy LATS MCTS loop，包含 select / expand / simulate / backpropagate。
- 判断 search 何时值得 token multiplier（Game of 24、code generation），何时单条 trajectory 足够（simple Q&A）。

## 要解决的问题

Chain-of-thought 是线性行走。如果第一步错了，后续每一步都建立在坏前提上。在 Game of 24（用四个数字通过 + − × ÷ 得到 24）上，GPT-4 CoT 准确率只有 4%。模型很早选错 subexpression，无法恢复。

Reasoning 需要的是提出多个 candidates、评估它们、选择 promising ones，并在出现 dead ends 时 backtrack 的能力。这就是 search。Tree of Thoughts 和 LATS 是两个 canonical formulations。

## 核心概念

### Tree of Thoughts（Yao et al., NeurIPS 2023）

每个 node 是一个 coherent intermediate step（“a thought”）。每个 node 可以扩展出 K 个 child thoughts。LLM 用 scoring prompt self-evaluates 每个 node。Search 探索这棵树：BFS、DFS 或 beam。

```text
                     (root: "find 24 from 4 6 4 1")
                    /               |            \
           ("6 - 4 = 2")    ("4 + 1 = 5")    ("4 * 6 = 24")  <- Score: HIGH
              /   \              |                  |
          ...    ...          ...                finish
```

Self-evaluation 是 load-bearing piece。论文展示三种 variants：`sure / likely / impossible` classification、`1..10` numeric score，以及 candidates 之间投票。三者在 Game of 24 上都大幅超过 CoT（GPT-4 从 4% -> 74%）。

### LATS（Zhou et al., ICML 2024）

LATS 在 MCTS 下统一 ToT、ReAct 和 Reflexion。LLM 扮演三个 roles：

- **Policy**：提出 candidate next actions（ReAct-style）。
- **Value function**：给 partial trajectory 打分（ToT-style self-eval）。
- **Self-reflector**：失败时写 natural-language reflection（Reflexion-style），并用它 reseed future rollouts。

Environment feedback（observations）混入 value function，使 search 由真实 tool results 而不仅是模型意见指导。论文发布时结果：GPT-4 在 HumanEval pass@1 达 92.7%（SOTA），GPT-3.5 在 WebShop average 75.9（接近 gradient-based fine-tuning）。

### MCTS 最小版本

每次 iteration 四个 phases：

1. **Select**：用 UCT（upper confidence bound for trees）从 root 走到 leaf。
2. **Expand**：通过 policy 生成 K 个 children。
3. **Simulate**：从某个 child 用 policy rollout，用 value function（或 environment reward）给 leaf 打分。
4. **Backpropagate**：沿 path 向上更新 visit counts 和 value estimates。

UCT formula：`Q(s, a) + c * sqrt(ln N(s) / N(s, a))`。第一项是 exploitation；第二项是 exploration。按任务调 `c`。

### Cost reality

Search 会让 tokens 爆炸。Game of 24 上的 ToT 使用 CoT 的 100-1000 倍 tokens。LATS 类似。这不是免费的；把 search 留给：

- 单条 trajectory 明显不足的任务（Game of 24、complex code）。
- Wall-clock 不如 correctness 重要的任务。
- 有廉价、可靠 value function 的任务（代码的 unit tests、数学的 explicit target）。

如果你的任务只有一个正确答案且 evaluator 很 noisy，search 经常会变得更糟，它会找到一个“good-scoring”的错误答案。

### 2026 positioning

大多数 production agents 不运行 LATS。它们运行 ReAct 加 tool-grounded verification（CRITIC，Lesson 05）。Search 出现在专业小众场景：

- 把 tests 作为 value function 的 coding agents（HumanEval-style）。
- 探索多个 query paths 的 deep-research agents。
- LangGraph subgraphs 内部的 planning-heavy workflows。

AlphaEvolve（Lesson 11）是 2025 年的极端形式：对代码做 evolutionary search，machine-checkable fitness，frontier gains（56 年来首个 4x4 matmul 改进）。

## 动手实现

`code/main.py` 实现：

- 一个 stylized “pick arithmetic ops” task 上的小型 ToT BFS。
- 同一任务上的 toy LATS MCTS loop（Select / Expand / Simulate / Backpropagate），使用 UCT selection。
- 一个组合 symbolic score 和 self-eval score 的 value function。

运行它：

```text
python3 code/main.py
```

Trace 显示 ToT 用 BFS 每个 node 扩展三个 candidates，相比之下 LATS 通过 MCTS 收敛到最佳 rollout。二者都会打印 token counts。

## 实际使用

LangGraph 把 ToT-style exploration 作为 subgraph patterns 提供；LangChain 团队 2024 年 5 月关于 LATS 的 blog 是参考 tutorial。LlamaIndex 提供 `TreeOfThoughts` agent。对大多数 2026 production agents，这个 pattern 位于 `if task_complexity > threshold: use_search()` gate 后面，见 Lesson 05 的 evaluator-optimizer pattern。

## 交付成果

`outputs/skill-search-policy.md` 会根据 task shape、budget 和 evaluator fidelity，在 linear ReAct、ToT、LATS 和 evolutionary search 之间选择。

## 练习

1. 用 UCT `c=0.1` 与 `c=2.0` 运行 toy LATS。Trace 有什么变化？
2. 把 value function 换成更 noisy 的 scorer（添加 random jitter）。MCTS 仍能找到 best leaf 吗？它能容忍的最小 signal-to-noise 是多少？
3. 实现 beam-search ToT（每层保留 top-k），并与 BFS 比较。Tight token budget 下哪个更好？
4. 阅读 LATS Section 5.1。复现 HumanEval trajectory count：需要多少 rollouts 才达到报告的 pass@1？
5. 阅读 LATS paper 关于“when LATS helps less”的讨论。写一段 decision rule，把 task shape 映射到 search strategy。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Tree of Thoughts | “Branching CoT” | Yao et al.，带 self-evaluation 的 thought nodes tree |
| LATS | “MCTS for LLMs” | Zhou et al.，在 MCTS 下统一 ToT + ReAct + Reflexion |
| UCT | “Upper confidence bound” | 平衡 exploitation（Q）和 exploration（ln N / n）的 select formula |
| Value function | “How good is this state” | Prompted LLM score 或 environment reward；反馈到 backprop |
| Policy | “Action proposer” | ReAct-style generator；发出 candidate next thoughts/actions |
| Rollout | “Simulated trajectory” | 用 policy 从 node 走到 leaf，再用 value 打分 |
| Backpropagate | “Update ancestors” | 把 leaf reward 沿 path 推回去，更新 visit counts 和 Q |
| Search cost | “Token explosion” | Game of 24 上是 CoT 的 100-1000 倍；采用前先预算 |

## 延伸阅读

- [Yao et al., Tree of Thoughts (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) — canonical paper
- [Zhou et al., LATS (arXiv:2310.04406)](https://arxiv.org/abs/2310.04406) — 带 Reflexion feedback 的 MCTS
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — search 的 subgraph patterns
- [AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) — 带 programmatic evaluators 的 evolutionary search
