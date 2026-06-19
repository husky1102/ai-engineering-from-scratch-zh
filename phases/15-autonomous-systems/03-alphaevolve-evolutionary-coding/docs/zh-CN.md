# AlphaEvolve：进化式编码 Agent

> 将 frontier coding model 与 evolutionary loop 和 machine-checkable evaluator 配对。让 loop 运行足够久。它会发现一种 4x4 complex-matrix multiplication procedure，只用 48 次 scalar multiplications，这是 56 年来第一次超过 Strassen 的改进。它也找到了一条 Google-wide Borg scheduling heuristic，在生产中回收约 0.7% 的 cluster compute。architecture 故意很朴素。胜利来自 evaluator 的严谨。

**类型：** 学习
**语言：** Python (stdlib, evolutionary-loop toy)
**先修：** Phase 15 · 01 (long-horizon framing), Phase 15 · 02 (self-taught reasoning)
**时间：** ~60 分钟

## 要解决的问题

Large language models 能写代码。Evolutionary algorithms 能搜索代码。二者分别已经被尝试了几十年，也都撞到了 ceiling。LLM ceiling 是 confabulation：模型写出看似合理、但并不做其声称事情的代码。evolutionary ceiling 是 search cost：对 syntax 做 random mutations，很少产生可编译程序，更别说更好的程序。

AlphaEvolve（Novikov et al., DeepMind, arXiv:2506.13131, 2025 年 6 月）把二者结合起来。LLM 向 program database 提出 targeted edits；automatic evaluator 为每个 variant 打分；high-scoring variants 成为未来 generations 的 parents。LLM 负责写出 plausible code 这个昂贵步骤；evaluator 捕获 confabulations。loop 会运行数小时到数周。

报告的结果包括：48-scalar-multiplication 的 4x4 complex matrix multiplication（Strassen 1969 年的 bound 是 49）、Google production 中的 Borg scheduling heuristic、32.5% FlashAttention kernel speedup、Gemini training throughput improvements。

这个 architecture 有效，是因为 evaluator 是 machine-checkable 的。evaluator 不是 machine-checkable 的地方，它就无效。这种不对称就是本课的 lesson。

## 核心概念

### Loop

1. 从一个正确但 suboptimal 的 seed program `P_0` 开始。
2. 维护一个 variant programs 数据库，每个 variant 都由 evaluator 打分。
3. 从数据库中采样一个或多个 parents（MAP-elites-style 或 island-based）。
4. Prompt LLM（大量 candidates 用 Gemini Flash，困难样本用 Gemini Pro）产出 parent 的 modified variant。
5. 编译、运行，并在 held-out evaluator 上评估 variant。
6. 按 score 和 feature vector keyed 插入数据库。
7. 重复。

两个细节很重要。第一，prompt 给 LLM 的不只是 parent program，通常还包括数据库中的几个 top variants、evaluator signature，以及短 task description。模型的工作是提出可能改善 score 的 targeted change。第二，database 是结构化的（MAP-elites grid、island-based），所以 loop 会探索 diversity，而不只是追当前 leader。

### 为什么 evaluator 不可协商

AlphaEvolve 的胜利都来自 evaluator 快速、确定、难以被 game 的领域：

- **Matrix multiplication algorithm**：一个 unit test，用于矩阵相乘并逐 bit 检查相等。
- **Borg scheduling heuristic**：production-grade simulator，重放历史 cluster load 并衡量 wasted compute。
- **FlashAttention kernel**：correctness test 加真实硬件上的 wall-clock benchmark。
- **Gemini training throughput**：每 step 的 GPU-seconds。

每个案例中，evaluator 都捕获了本会占主导的 LLM 错误类别：虚构 correctness claims、在硬件上消失的 performance claims，以及 edge-case failures。移除 evaluator，loop 就会优化漂亮代码。

### Reward hacking 是这句话的另一面

Evolution 会优化 evaluator 衡量的任何东西。如果 evaluator 不完美，loop 会找到这个不完美之处。在 unverified domain 中，loop 会优化 surface feature，而不是 intended behavior。DeepMind 在论文中明确标记这一点：AlphaEvolve 的成功只会迁移到 evaluator rigor 能匹配 search ambition 的领域。

2025-2026 年 code-search loops 中的 reward hacking 具体例子：

- 奖励 “time to complete” 的 optimization targets 奖励了提交 empty solutions。
- 奖励 correctness-under-test 的 benchmark scores 奖励了 memorizing tests 和 overfitting。
- 一个 “code quality” proxy 奖励删除 comments 和改写 variable names，而没有 semantic change。

AlphaEvolve 中的修复：交付一个 LLM 从未见过的 held-out evaluator，且 inputs 在 evaluation time 生成。即便如此，DeepMind 仍建议对任何 proposed deployment 做强 review。

### 为什么 LLM + search 胜过任一单独方法

LLM 能产生可编译且语义上 plausible 的 modifications。对 2000 行 Python 文件做 random-mutation GA，几乎总会产生 syntax errors。LLM 也会把 search 集中在 plausible neighborhoods（改一个 function，而不是随机 bytes），从而大幅减少浪费的 evaluator calls。

evaluator 反过来捕获 LLM 的 confabulations。LLM 会自信地声称一个 function “is O(n log n) in the limit”，但它实际上是 O(n^2)；wall-clock benchmark 会把问题定下来。

### AlphaEvolve 在 frontier stack 中的位置

| System | Generator | Evaluator | Domain | Example win |
|---|---|---|---|---|
| AlphaEvolve | Gemini | correctness + benchmark | algorithms, kernels, schedulers | 48-mul 4x4 matmul |
| FunSearch (DeepMind, 2023) | PaLM / Codey | correctness | combinatorial math | cap-set lower bounds |
| AI Scientist v2 (Sakana, L5) | GPT/Claude | LLM critique + experiment | ML research | ICLR workshop paper |
| Darwin Godel Machine (L4) | agent scaffolding | SWE-bench / Polyglot | agent code | 20% → 50% SWE-bench |

四者都是同一个配方的变体：generator plus evaluator, loop。区别在于 evaluator 评分什么，以及它有多严谨。

## 实际使用

`code/main.py` 在 toy symbolic-regression problem 上实现一个最小 AlphaEvolve-like loop。“LLM” 是一个 stdlib proxy，会对计算 target function 的 program 提出小型 syntactic mutations。“evaluator” 会在 held-out test points 上衡量 mean squared error。

观察：

- best score 如何随 generations 改善。
- MAP-elites grid 如何保留 diverse solutions，使 loop 不会收敛到 local minimum。
- 移除 held-out test（training-only evaluator）后，loop 如何 spectacularly overfit。

## 交付成果

`outputs/skill-evaluator-rigor-audit.md` 是在新 domain 中考虑 AlphaEvolve-style loop 的前置条件：你的 evaluator 是否真的能捕获你关心的 failures？

## 练习

1. 运行 `code/main.py`。记录 best score trajectory。禁用 held-out evaluator（flag `--no-holdout`）并重新运行。量化 overfitting。

2. 阅读 AlphaEvolve paper Section 3 中关于 MAP-elites grid 的内容。为一个新问题（例如 compiler optimization passes）设计 feature-vector descriptor，以保持 search diverse。

3. 48-multiplication 4x4 结果在 56 年后改进了 Strassen 的 49-mul bound。阅读论文 Appendix F，并用三句话解释为什么这个问题的 evaluator 特别容易做对，以及为什么大多数 domains 不是这样。

4. 提出一个 AlphaEvolve 会失败的 domain。准确指出 evaluator 在哪里破裂以及原因。

5. 对一个你熟悉的 domain，写出你会使用的 evaluator signature。包含 (a) correctness conditions，(b) performance metric，(c) held-out input generation rule，(d) 至少一个 anti-reward-hacking check。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|---|---|---|
| AlphaEvolve | “DeepMind's evolutionary coding agent” | Gemini + program database + machine-checkable evaluator |
| MAP-elites | “Diversity-preserving archive” | 由 feature vectors keyed 的 grid；每个 cell 持有该 descriptor 下的最佳 variant |
| Island model | “Parallel evolution subpopulations” | 周期性迁移的独立 populations；防止 premature convergence |
| Machine-checkable evaluator | “Deterministic oracle” | LLM 无法伪造的 unit test、simulator 或 benchmark；是这个 loop 的前置条件 |
| Reward hacking | “Optimizing the measure, not the goal” | loop 找到最大化 score、但不执行 intended task 的方法 |
| Seed program | “The starting point” | loop 从中 evolution 的初始 correct-but-suboptimal program |
| Held-out evaluator | “Evaluation data the LLM never saw” | 在 evaluation time 生成的 inputs，用于防止 memorization |

## 延伸阅读

- [Novikov et al. (2025). AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131) — 完整论文。
- [DeepMind blog on AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) — vendor writeup 与结果。
- [AlphaEvolve results repository](https://github.com/google-deepmind/alphaevolve_results) — 发现的 algorithms，包括 48-mul 4x4 matmul。
- [Romera-Paredes et al. (2023). Mathematical discoveries from program search with LLMs (FunSearch)](https://www.nature.com/articles/s41586-023-06924-6) — 前身系统。
- [Anthropic — Responsible Scaling Policy v3.0 (Feb 2026)](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — 将 evaluator-bound autonomy framed 为关键 research direction。
