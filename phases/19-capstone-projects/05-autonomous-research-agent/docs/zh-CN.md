# Capstone 05 — 自主研究 Agent（AI-Scientist 类）

> Sakana 的 AI-Scientist-v2 发表了完整论文。Agent Laboratory 运行了实验。Allen AI 分享了轨迹。2026 年的形态是：围绕实验进行 plan-execute-verify 树搜索、受预算约束的成本、沙盒化代码执行、带视觉反馈的 LaTeX 写作器，以及自动化 NeurIPS 风格 reviewer ensemble。本 capstone 要构建这样一个系统，在每篇论文 $30 内端到端运行，并通过 Sakana 记录过的 sandbox-escape 红队测试。

**类型：** Capstone
**语言：** Python (agent + sandbox), LaTeX (output)
**先修：** Phase 2 (ML), Phase 3 (deep learning), Phase 7 (transformers), Phase 10 (LLMs from scratch), Phase 14 (agents), Phase 15 (autonomous), Phase 16 (multi-agent), Phase 18 (safety)
**练习阶段：** P0 · P2 · P3 · P7 · P10 · P14 · P15 · P16 · P18
**时间：** 40 hours

## 要解决的问题

自主研究 agent 在 2026 年跨过了一个门槛。Sakana AI 的 AI-Scientist-v2 发表在 Nature，其生成论文通过了 workshop peer review。ShinkaEvolve (ICLR 2026) 将这条路线扩展到演化假设。AMD 的 Agent Laboratory 交付了可复现轨迹。这些 agent 并不神奇：它们是在候选实验树上运行的 plan-execute-verify 循环，带有成本上限、种子绑定的沙盒和自动化评审。真正的功夫在循环、预算和安全叙事里。

你将通过在一个狭窄领域的 seed idea 上实现一个系统来学习这个循环（例如，对 100M 参数 transformer 做 attention-sparsity ablations）。价值不在于第一次运行就发现新东西。价值在基础设施：tree-search、experiment sandbox、writer-reviewer loop、red-team report。Sakana 团队记录了 sandbox-escape 失败；你的 agent 必须通过同样的红队测试。

## 核心概念

这个 agent 是 best-first tree search。节点是实验规格：(hypothesis, config, code, expected outcome)。expand 步骤通过小编辑提出子节点（替换 optimizer、移动 batch size、ablate 某个组件）。每个子节点都在带硬资源上限的新沙盒里运行。结果反馈到评分函数，根据 (novelty × quality × remaining budget) 对节点排序。树一直增长到预算耗尽，然后把最佳分支写成论文。

writer 是多模态的。它生成 LaTeX draft，编译它，渲染图表，并把渲染后的 PDF 送回 Claude Opus 4.7 的 vision mode，让其批评版式、图的可读性以及 claim-evidence alignment。由五个 LLM judge 组成的 reviewer ensemble 输出 NeurIPS 风格分数（novelty, rigor, clarity, reproducibility, impact）；如果平均分低于阈值，论文会带着 critique 回到 writer。

安全是承重结构。每个实验都在 E2B 或 Daytona sandbox 中运行，没有 network egress，受 wall-clock 限制，并固定资源上限。agent 的 code-generation 步骤会经过 policy layer，阻止逃逸沙盒的 syscalls。red-team report 复现 Sakana 记录过的攻击面（fork bombs、filesystem escapes、LLM-written network calls）。

## 架构

```text
seed idea + domain
      |
      v
  literature search (Semantic Scholar + OpenAlex + FAISS cache)
      |
      v
  LangGraph plan-execute-verify tree
      |
      v
  +--- expand node ----+      per-node sandbox
  |                    |      (E2B / Daytona)
  v                    v      resource caps
  child_1           child_k   no network egress
  |                    |      deterministic seeds
  v                    v
  run experiment       run experiment
  |                    |
  v                    v
  score nodes by (novelty, quality, budget)
      |
      v
  best branch -> LaTeX writer
      |
      v
  compile + vision critique (Opus 4.7 vision)
      |
      v
  reviewer ensemble (5 LLM judges, NeurIPS rubric)
      |
      v
  paper.pdf + review.md + trace.json
```

## 技术栈

- Orchestration: LangGraph with checkpointing and human-approval gates
- Tree search: custom best-first over experiment nodes (AB-MCTS-style from Sakana v2)
- Sandbox: E2B per experiment, Docker-in-Docker fallback; resource caps via cgroups
- Literature: Semantic Scholar Graph API + OpenAlex + local FAISS cache of abstracts
- Writer: LaTeX template + Claude Opus 4.7 (vision mode) for figure critique and layout
- Reviewer: ensemble of 5 judges (Opus 4.7, GPT-5.4, Gemini 3 Pro, DeepSeek R1, Qwen3-Max) with weighted aggregation
- Experiment framework: PyTorch 2.5 for the physical experiments, W&B for logging
- Observability: Langfuse for agent traces, $30 hard budget per paper

## 动手实现

1. **Seed 和领域定界。** 取一个 seed idea（例如，“investigate sparsity patterns in attention maps of sub-1B transformers”）。定义搜索空间：models, datasets, compute budget。

2. **文献遍历。** 查询 Semantic Scholar + OpenAlex，找出 50 篇被引用最多的相关论文；在本地缓存 abstracts；生成 1 页 domain digest。

3. **树脚手架。** 用 seed hypothesis 初始化 root。实现 `expand(node) -> children`，用小编辑 proposal（每个 child 只改一个 config）。将 `score(node)` 实现为加权的 novelty × quality × budget 项。

4. **沙盒包装。** 每个实验都运行 `docker run --network=none --memory=8g --cpus=2 --pids-limit=256 --read-only`（或等价的 E2B policy）。seeds 写入 sandbox；outputs 以 read-only 方式挂载回外部。

5. **Plan-execute-verify 循环。** `plan` 提出 children。`execute` 运行 sandbox，捕获 logs 和 metrics。`verify` 对 metrics 做单元检查（loss 是否下降？ablation 是否隔离了效果？）。失败节点把 failure reason 存到树上。

6. **Writer。** 预算耗尽后，选择最佳分支。用 matplotlib 渲染 figures。把 branch trace 放入上下文，通过 Claude Opus 4.7 生成 LaTeX draft。编译。把编译好的 PDF 送回 Opus 4.7 vision 做 critique。迭代。

7. **Reviewer ensemble。** 五个 judges 按 (novelty, rigor, clarity, reproducibility, impact) 用 NeurIPS 风格 rubrics 给 draft 打分。如果 mean < 4.0/5，则带着 critique 回到 writer。3 次 rewrite 后硬停止。

8. **Red team。** 构建或集成一组瞄准 sandbox 的 adversarial tasks：fork bombs、network exfiltration attempts、filesystem escapes、LLM-written shell metacharacters。确认全部被阻止。写出 findings。

9. **可复现性。** 每篇论文都附带 tree-search trace JSON、seeds、W&B run links、sandbox configs，以及一个可端到端复现的 README。

## 实际使用

```text
$ ai-scientist run --seed "attention sparsity in sub-1B transformers" --budget 30
[lit]    50 papers, digest in 12s
[tree]   expanded 8 nodes, budget 12/30
[exec]   node #3 sparsity=top-8, loss=2.83 (best so far)
[exec]   node #6 sparsity=top-4, loss=3.12 (worse)
[exec]   ...
[tree]   chose branch rooted at node #3 (novelty 0.62, quality 0.81)
[write]  LaTeX draft v1 complete
[vision] critique: figure 2 legend too small, claim-evidence ok
[write]  draft v2 after 3 edits
[review] mean 4.2/5 (novelty 3.9, rigor 4.3, clarity 4.1, repro 4.5, impact 4.2)
[done]   paper.pdf + review.md + trace.json     $28.40 spent
```

## 交付成果

`outputs/skill-ai-scientist.md` 是交付物。给定一个 seed idea、一个 domain 和 $30 budget，它会运行完整 pipeline，并输出一篇可评审论文和一个 reproducibility bundle。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 论文质量 | Against published workshop papers 的 blind rubric review |
| 20 | 实验严谨性 | Baselines、seeds、ablations；每个 claim 都有 results table 中的一个 cell 支撑 |
| 20 | 成本和计算纪律 | 强制 $30/paper ceiling，Langfuse-traced |
| 20 | 安全 | Sandbox red team passes；network policy 和 kill-switch verified |
| 15 | 可复现性 | One-command rerun with identical seeds reproduces the paper |
| **100** | | |

## 练习

1. 在同一领域针对三个不同 seed ideas 运行 pipeline。比较 tree-search 哪些部分重叠。识别重复浪费的 compute。

2. 在执行实验前，为估算超过 $5 的节点添加 human-in-the-loop gate。测量总成本下降了多少。

3. 把 reviewer ensemble 换成单个 judge。用一组 held-out 的 known-bad papers 测量 false-accept rate。

4. 引入 network-exfiltration red team test：agent 写代码尝试 `curl` 外部地址。确认 `--network=none` policy 会阻止它。记录这次尝试。

5. 将你的 tree-search 与 flat random baseline（相同预算，无 expansion strategy）比较。报告 novelty × quality gain。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| Tree search | "AB-MCTS-style expansion" | 在 experiment nodes 上按 novelty×quality×budget score 进行 best-first exploration |
| Sandbox | "Experiment isolation" | 无网络、有 CPU/memory 边界、固定 seeds、read-only inputs 的容器 |
| Vision critique | "Render-then-read" | 将论文编译为 PDF，把 PDF 送回 VLM 做 layout 和 claim-evidence critique |
| Reviewer ensemble | "Automated peer review" | 多个 LLM judges 用 NeurIPS rubric 给论文评分；weighted aggregate 作为 pipeline gate |
| Novelty score | "Is this new?" | 惩罚与 50-paper literature cache 过近的启发式分数 |
| Cost ceiling | "$ budget" | 每篇论文总花费的硬上限；Langfuse counters + pre-run estimates |
| Red team | "Sandbox-escape audit" | 如果 policy 错误就会逃逸 sandbox 的 adversarial tasks |

## 延伸阅读

- [Sakana AI-Scientist-v2 repository](https://github.com/SakanaAI/AI-Scientist-v2) — reference production research agent
- [Sakana AI-Scientist-v1 paper (arXiv:2408.06292)](https://arxiv.org/abs/2408.06292) — original methodology
- [ShinkaEvolve (Sakana ICLR 2026)](https://sakana.ai) — evolutionary extension
- [Agent Laboratory (AMD)](https://github.com/SamuelSchmidgall/AgentLaboratory) — multi-role research-lab framework
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — reference orchestration layer
- [Semantic Scholar Graph API](https://api.semanticscholar.org/) — literature search
- [E2B sandboxes](https://e2b.dev) — reference experiment isolation
- [NeurIPS reviewer guidelines](https://neurips.cc/Conferences/2026/Reviewer-Guidelines) — reviewer ensemble 编码的 rubric
