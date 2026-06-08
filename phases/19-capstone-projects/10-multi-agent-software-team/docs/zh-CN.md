# Capstone 10 — 多 Agent 软件工程团队

> SWE-AF 的 factory architecture、MetaGPT 的 role-based prompting、AutoGen 0.4 的 typed actor graph、Cognition 的 Devin，以及 Factory 的 Droids 都收敛到了同一种 2026 年形态：architect 规划，N 个 coders 在 parallel worktrees 中工作，reviewer 把关，tester 验证。Parallel worktrees 将 wall-clock 转化为 throughput。Shared state 和 handoff protocols 变成失败面。这个 capstone 要构建这支团队，在 SWE-bench Pro 上评估，并报告哪些 handoffs 会失败以及频率如何。

**类型：** Capstone
**语言：** Python / TypeScript (agents), Shell (worktree scripts)
**先修：** Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 15 (autonomous), Phase 16 (multi-agent), Phase 17 (infrastructure)
**练习阶段：** P11 · P13 · P14 · P15 · P16 · P17
**时间：** 40 hours

## 要解决的问题

单 agent coding harnesses 在大型任务上会碰到天花板。不是因为任何单个 agent 弱，而是因为 200k-token context 无法同时容纳 architecture plan、四个并行 codebase slices、reviewer commentary 和 test output。Multi-agent factories 会拆分问题：architect 负责 plan，coders 在 parallel worktrees 中负责 implementation，reviewer 把关，tester 验证。SWE-AF 的 "factory" architecture、MetaGPT 的 roles、AutoGen 的 typed actor graph 这三种说法描述的是同一个形态。

失败面在 handoff。Architect 规划了 coders 无法实现的东西。Coders 产出冲突 diffs。Reviewer 批准了 hallucinated fix。Tester 与仍在写代码的 coder 发生 race。你会构建这样一支团队，在 50 个 SWE-bench Pro issues 上运行，追踪每个 handoff，并发布 post-mortem。

## 核心概念

Roles 是 typed agents。**Architect**（Claude Opus 4.7）读取 issue，写出 plan，并把它分解为带 explicit interfaces 的 subtasks。**Coders**（Claude Sonnet 4.7，N 个 parallel instances，每个在一个 `git worktree` + Daytona sandbox 中）独立实现 subtasks。**Reviewer**（GPT-5.4）读取 merged diff，然后 approve 或请求 specific changes。**Tester**（Gemini 2.5 Pro）在 isolation 中运行 test suite，并带 artifacts 报告 pass/fail。

通信通过 shared task board（file-backed 或 Redis）进行。每个 role 消费自己被允许处理的 tasks。Handoffs 是 A2A-protocol-typed messages。协调关注点：merge-conflict resolution（coordinator role 或自动 three-way merge）、shared-state synchronization（coders 开始后 plan 冻结；replans 是独立 events）、reviewer gatekeeping（reviewer 不能 approve 它自己更改或提出的 changes）。

Token amplification 是隐藏成本。每个 role boundary 都会添加 summary prompts 和 handoff context。一个 40-turn single-agent run 会变成跨四个 roles 的 160 total turns。rubric 特别会将 token efficiency 与 single-agent baseline 对比，因为问题不是 “multi-agent 是否有效”，而是 “它是否按 dollar 取胜”。

## 架构

```text
GitHub issue URL
      |
      v
Architect (Opus 4.7)
   reads issue, produces plan with subtasks + interfaces
      |
      v
Task board (file / Redis)
      |
   +-- subtask 1 ---+-- subtask 2 ---+-- subtask 3 ---+-- subtask 4 ---+
   v                v                v                v                v
Coder A          Coder B          Coder C          Coder D          (4 parallel)
 (Sonnet)         (Sonnet)         (Sonnet)         (Sonnet)
 worktree A       worktree B       worktree C       worktree D
 Daytona          Daytona          Daytona          Daytona
      |                |                |                |
      +--------+-------+-------+--------+
               v
           merge coordinator  (three-way merge + conflict resolution)
               |
               v
           Reviewer (GPT-5.4)
               |
               v
           Tester  (Gemini 2.5 Pro)  -> passes? -> open PR
                                     -> fails?  -> route back to coder
```

## 技术栈

- Orchestration: LangGraph with shared state + per-agent sub-graphs
- Messaging: A2A protocol (Google 2025) for typed inter-agent messages
- Models: Opus 4.7 (architect), Sonnet 4.7 (coders), GPT-5.4 (reviewer), Gemini 2.5 Pro (tester)
- Worktree isolation: `git worktree add` per coder + Daytona sandbox
- Merge coordinator: custom three-way merge + LLM-mediated conflict resolution
- Eval: SWE-bench Pro (50 issues), SWE-AF scenarios, HumanEval++ for unit tests
- Observability: Langfuse with role-tagged spans, per-agent token accounting
- Deployment: K8s with each role as a separate Deployment + HPA on backlog

## 动手实现

1. **Task board。** File-backed JSONL，包含 typed messages：`plan_request`、`subtask`、`diff_ready`、`review_needed`、`test_needed`、`approved`、`rejected`、`replan_needed`。Agents 订阅 tags。

2. **Architect。** 读取 GitHub issue，用 plan template 运行 Opus 4.7，要求 explicit subtask interfaces（files touched、public functions、test impact）。发出一个带 subtasks DAG 的 `plan_request`。

3. **Coders。** N 个 parallel workers，每个从 board claim 一个 subtask。每个都启动一个新的 `git worktree add` branch 加 Daytona sandbox。实现 subtask。发出 `diff_ready`，其中包含 patch + test deltas。

4. **Merge coordinator。** 当所有 coders 完成后，将 N 个 branches three-way merge 到 staging branch。只有存在 file-level overlap 时才使用 LLM-mediated conflict resolution。

5. **Reviewer。** GPT-5.4 读取 merged diff。不能 approve 它 authored 的 diffs。发出 `approved`（no-op）或 `review_feedback`，其中包含 routed back to relevant coder 的 specific change requests。

6. **Tester。** Gemini 2.5 Pro 在 clean sandbox 中运行 test suite。捕获 artifacts。发出带 stacktraces 的 `test_passed` 或 `test_failed`。failed tests loop back 到拥有 failing subtask 的 coder。

7. **Handoff accounting。** 每条跨越 role boundary 的 message 都在 Langfuse 中获得一个 span，记录 payload size 和使用的 model。计算 per-subtask token amplification（coder_tokens + reviewer_tokens + tester_tokens + architect_share / coder_tokens）。

8. **Eval。** 在 50 个 SWE-bench Pro issues 上运行。将 pass@1 和 $-per-solved-issue 与 single-agent baseline（一个 Sonnet 4.7 在单个 worktree 中）比较。

9. **Post-mortem。** 对每个 failed issue，识别 broken handoff（plan too vague、merge conflict、reviewer false-approve、tester flake）。产出 handoff-failure histogram。

## 实际使用

```text
$ team run --issue https://github.com/acme/widget/issues/842
[architect] plan: 4 subtasks (parser, cache, api, migration)
[board]     dispatched to 4 coders in parallel worktrees
[coder-A]   subtask parser  -> 42 lines, tests pass locally
[coder-B]   subtask cache   -> 88 lines, tests pass locally
[coder-C]   subtask api     -> 31 lines, tests pass locally
[coder-D]   subtask migration -> 19 lines, tests pass locally
[merge]     3-way merge: 0 conflicts
[reviewer]  comments on cache (thread pool sizing); routed to coder-B
[coder-B]   revision: 92 lines; submits
[reviewer]  approved
[tester]    all 412 tests pass
[pr]        opened #3382   4 coders, 1 revision, $4.90, 18m
```

## 交付成果

`outputs/skill-multi-agent-team.md` 是交付物。给定 issue URL 和 parallelism level，团队会产出 merge-ready PR，并附带 per-role token accounting。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | Matched 50-issue subset, pass@1 |
| 20 | Parallel speedup | Wall-clock vs single-agent baseline |
| 20 | Review quality | injected-bug probe 上的 false-approval rate |
| 20 | Token efficiency | Total tokens per solved issue vs single-agent |
| 15 | Coordination engineering | Merge-conflict resolution、handoff-failure histogram |
| **100** | | |

## 练习

1. 在 mid-run 的 diff 中注入明显 bug（main body 前额外添加 `return None`）。测量 reviewer 的 false-approve rate。调优 reviewer prompt，直到 false-approval 低于 5%。

2. 缩减为两个 coders（architect + coder + reviewer + tester，coder 顺序运行两个 subtasks）。比较 wall-clock 和 pass rate。

3. 用 single-writer constraint 替换 merge coordinator（subtasks 触碰不相交的 file sets）。测量 architect 的 planning burden。

4. 将 reviewer 从 GPT-5.4 换成 Claude Opus 4.7。测量 false-approval rate 和 token cost delta。

5. 添加第五个 role：documenter（Haiku 4.5）。review 后，它产出 changelog entry。测量 documentation quality 是否值得额外 token spend。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| Parallel worktree | "Isolated branch" | 每个 coder 通过 `git worktree add` 获得一个新的 working tree |
| Task board | "Shared message bus" | agents 订阅的 typed messages 的 file 或 Redis store |
| Handoff | "Role boundary" | 任何从一个 role 的 context 跨到另一个 role context 的 message |
| Token amplification | "Multi-agent overhead" | 同一任务上跨 roles 的 total tokens / single-agent tokens |
| A2A protocol | "Agent-to-agent" | Google 2025 typed inter-agent messages spec |
| Merge coordinator | "Integrator" | 运行 three-way merge 并调解 conflicts 的组件 |
| False approval | "Reviewer hallucination" | reviewer approve 了带 known bugs 的 diff |

## 延伸阅读

- [SWE-AF factory architecture](https://github.com/Agent-Field/SWE-AF) — reference 2026 multi-agent factory
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT) — role-based multi-agent framework
- [AutoGen v0.4](https://github.com/microsoft/autogen) — Microsoft's typed actor framework
- [Cognition AI (Devin)](https://cognition.ai) — reference product
- [Factory Droids](https://www.factory.ai) — alternate reference product
- [Google A2A protocol](https://developers.google.com/agent-to-agent) — inter-agent messaging spec
- [git worktree documentation](https://git-scm.com/docs/git-worktree) — isolation substrate
- [SWE-bench Pro](https://www.swebench.com) — evaluation target
