# Role Specialization——Planner、Critic、Executor、Verifier

> 2026 年最常见的 multi-agent decomposition：一个 agent 规划，一个执行，一个 critique 或 verify。MetaGPT（arXiv:2308.00352）把它形式化为编码进 role prompts 的 SOPs——Product Manager、Architect、Project Manager、Engineer、QA Engineer——遵循 `Code = SOP(Team)`。ChatDev（arXiv:2307.07924）通过 “chat chain” 串起 designer、programmer、reviewer、tester，并使用 “communicative dehallucination”（agents 显式请求缺失细节）。Verifier 是承重角色：Cemri et al.（MAST，arXiv:2503.13657）显示每个 multi-agent failure 都能追溯到缺失或损坏的 verification。PwC 报告称，通过 CrewAI 中的 structured validation loops，accuracy 获得 7× gain（10% → 70%）。

**类型：** 学习 + 构建
**语言：** Python (stdlib)
**先修：** Phase 16 · 04 (Primitive Model), Phase 16 · 05 (Supervisor)
**时间：** ~60 分钟

## 要解决的问题

Generic multi-agent systems 产生 generic output。Group chat 里的三个 coders 会写出三种同样 mediocre 的 code。你可以增加更多 agents，增加更多 rounds，却仍然跨不过 quality threshold。

修复方法不是更多 agents——而是 *不同* agents。分配 distinct roles。给 critic 配 planner 没有的 tools。给 verifier 一个 objective test suite。现在 system 拥有带 grounded correction 的 internal disagreement，而不仅仅是 parallel guessing。

## 核心概念

### 四个 canonical roles

**Planner。** 读取 goal，产生 step list 或 spec。Tools：knowledge retrieval、docs。Output：structured plan。

**Executor。** 一次读取一个 plan step，产生 artifact。Tools：actual work tools（code compiler、shell、API client）。Output：artifact。

**Critic。** 依据 planner's intent 阅读 executor's output。Tools：对 artifact 的 read-only access、static analysis。Output：带 reasons 的 accept/reject。

**Verifier。** 读取 artifact 并运行 deterministic check。Tools：test runner、type checker、schema validator。Output：带 evidence 的 pass/fail。

Critic 是 subjective、opinionated，通常基于 LLM。Verifier 是 objective、deterministic，通常基于 code。它们不是同一个 role。

### MetaGPT 的 SOP pattern

MetaGPT（arXiv:2308.00352）把 software engineering SOPs 编码为 role prompts：

- **Product Manager** 写 PRD。
- **Architect** 产生 system design。
- **Project Manager** 拆分 tasks。
- **Engineer** 实现。
- **QA Engineer** 运行 tests。

每个 role 都有严格 input/output schema。Role prompt 说明这个 role *是什么* 以及它 *必须产出什么*。`Code = SOP(Team)` formulation——deterministic SOPs 把一队 LLMs 变成可预测 pipeline。

### ChatDev 的 communicative dehallucination

ChatDev 增加了一个关键动作：当 executor 需要 plan 里没有的特定 detail 时，它会在继续前明确询问 designer。这会防止经典 LLM failure：看似合理地发明缺失细节。

Implementation：role prompt 包含 “when you need specific information you were not given, ask the relevant role by name before producing output.”

### 为什么 verifier 最重要

Cemri et al.（MAST）追踪了 1642 次 multi-agent execution failures。其中 21.3% 是 verification gaps——system ship 了一个无人检查的 answer。剩下的 79% 经常也能追溯到“有一个 check failed silently 或 never run”。Verification 是承重 role。

PwC 报告称（CrewAI deployments，2025），加入 structured validation loop 后 accuracy 从 10% 提升到 70%。一个 role 带来 7× gain。

### Critic vs verifier

- Critic 是审阅 artifact quality 的 LLM。Subjective。会被 plausible prose 欺骗。
- Verifier 是运行在 artifact 上的 deterministic program。Objective。给出带 evidence 的 pass/fail。

两个都用。Critic 捕捉 verifier 无法表达的 taste issues。Verifier 捕捉 critic 看不见的 bugs，因为它们只在 runtime 出现。

### Anti-pattern

你系统里的每个 role 都是 LLM，而且每个 role 的 output 都是 “looks good to me”。经典 MAST failure mode。至少加入一个 pass/fail 由 code 而不是 LLM 决定的 verifier。

### Framework mappings

- **CrewAI** —— `Agent(role, goal, backstory)` 是 textbook specialization surface。
- **LangGraph** —— nodes 可以有 specialized prompts；edges enforce the pipeline。
- **AutoGen** —— GroupChat 中带 one-word names 的 role-specific ConversableAgents。
- **OpenAI Agents SDK** —— role-specialized Agents 之间的 handoff tools。

## 动手实现

`code/main.py` 实现一个构建 simple Python function 的 4-role pipeline：

- **Planner** 产生 spec。
- **Executor** 生成 code string。
- **Critic**（LLM-simulated）标记 obvious issues。
- **Verifier** 在 sandbox（`exec`）中针对 test case 运行 generated code。

Demo 运行两次：一次 executor 产生 correct code（critic + verifier 都 pass），一次 executor 产生 off-spec code（critic 因为它看起来 plausible 而漏掉 bug，verifier 因为 test fails 而捕捉它）。

运行：

```text
python3 code/main.py
```

## 实际使用

`outputs/skill-role-designer.md` 接收一个 task，并产生 role roster（3-5 roles）、每个 role 的 input/output schema，以及 verifier check。在把 agents 接入 framework 前使用它。

## 交付成果

Checklist：

- **At least one deterministic verifier。** 永远不要 all-LLM。
- **Explicit I/O schema per role。** Planner 返回 spec，不是 prose；executor 读取这个 schema。
- **Communicative dehallucination。** Executor 在 info missing 时必须 ask planner；绝不 invent。
- **Critic/verifier ordering。** 先跑 critic（cheap，捕捉 design issues），再跑 verifier（slow，捕捉 bugs）。
- **Loop budget。** 最多 2 次 critic-executor revision rounds，然后 escalate to human。

## 练习

1. 运行 `code/main.py`，观察 verifier 如何捕捉 critic missed 的 bug。增加一个 static-analysis check（统计 `return` 出现次数）作为额外 verifier。它能捕捉 runtime test 漏掉的什么？
2. 增加第 5 个 role：“requirements analyst”，把 user wish 翻译成 planner-ready spec。哪些 communicative dehallucination requests 应该向上流动到它？
3. 阅读 MetaGPT Section 3（“Agents”）。列出 MetaGPT 5 个 roles 各自的 input/output schema。
4. 阅读 ChatDev 的 chat-chain diagram（arXiv:2307.07924 Figure 3）。识别 communicative dehallucination 在哪里打断本来会无限循环的 loop。
5. PwC 的 7× accuracy gain 来自 verification loops。假设三个加入 verifier 也没有帮助的 tasks——也就是 deterministic checking correctness 不可能或成本过高的场景。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Role specialization | "Different agents, different jobs" | 为 planner/executor/critic/verifier roles 调优的 distinct system prompts。 |
| SOP pattern | "Encoded standard operating procedure" | MetaGPT framing：每个 role 的 strict I/O schemas 把 team 变成 pipeline。 |
| Communicative dehallucination | "Ask before inventing" | ChatDev pattern：executor 在 detail missing 时 ask planner，而不是编造。 |
| Critic | "LLM reviewer" | Subjective、opinionated reviewer。捕捉 taste issues。会被 plausible prose 欺骗。 |
| Verifier | "Deterministic check" | Code-based pass/fail。Test runner、type checker、schema validator。不会被欺骗。 |
| Verification gap | "No one checked" | MAST failures 的 21.3%。Answer shipped without a check that would have caught the bug。 |
| Revision loop | "Critic sends it back" | Critic rejection 触发 executor 带 feedback 重新运行。需要 budget。 |
| All-LLM anti-pattern | "Looks good to me" | 每个 role 都是 LLM，没有 deterministic check。经典 MAST failure。 |

## 延伸阅读

- [Hong et al. — MetaGPT: Meta Programming for Multi-Agent Collaboration](https://arxiv.org/abs/2308.00352) —— SOP-as-role-prompt reference paper
- [Qian et al. — Communicative Agents for Software Development (ChatDev)](https://arxiv.org/abs/2307.07924) —— chat chain + communicative dehallucination
- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) —— MAST taxonomy；verification gaps 是 failures 的 21.3%
- [CrewAI docs — Agent roles](https://docs.crewai.com/en/introduction) —— production role specification surface
