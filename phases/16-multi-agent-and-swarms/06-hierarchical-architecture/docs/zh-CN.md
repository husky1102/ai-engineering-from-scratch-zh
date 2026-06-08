# Hierarchical Architecture 及其失败模式

> Hierarchical 是嵌套的 supervisor。Manager agents 管 sub-managers，sub-managers 管 workers。CrewAI `Process.hierarchical` 是教科书版本：一个 `manager_llm` 动态委派 tasks 并验证 outputs。LangGraph 等价写法是 `create_supervisor(create_supervisor(...))`。当 task 本身就是一张真实 org chart 时，这是自然 pattern。它也是最容易坍缩成 managerial looping 的 pattern——manager agents 分配工作不当、误解 sub-outputs，或无法达成共识。Sequential 往往会赢过它。

**类型：** 学习 + 构建
**语言：** Python (stdlib)
**先修：** Phase 16 · 05 (Supervisor Pattern)
**时间：** ~60 分钟

## 要解决的问题

一旦 supervisor pattern 变得清晰，自然的下一步就是：“如果 workers 自己也是 supervisors 呢？”团队有子团队；公司有部门的部门。Hierarchical architectures 映射了这种结构。

问题是：LLM managers 并不等同于 human managers。Human manager 对自己的 reports 知道什么有稳定 priors。LLM manager 每一轮都从 context 里已有内容重新推理 org。Context 中微小漂移，整棵 tree 就会错误分配工作。

## 核心概念

### 形状

```text
                 Manager
                 ┌─────┐
                 └──┬──┘
           ┌────────┴────────┐
           ▼                 ▼
       Sub-Mgr A         Sub-Mgr B
       ┌─────┐           ┌─────┐
       └──┬──┘           └──┬──┘
         ┌┴──┬──┐          ┌┴──┐
         ▼   ▼  ▼          ▼   ▼
       W1  W2  W3         W4  W5
```

每个 internal node 负责 planning、delegating 和 synthesizing。只有 leaves 做实际工作。

### 它在哪里发光

- **Clear org mapping。** 如果真实 task 是 departmental（“legal review the doc, finance review the doc, engineering review the doc, then summarize for exec”），hierarchy 是显式的。
- **Local summarization。** 每个 sub-manager 会先 synthesize 自己 team 的 output，再让 top manager 看到。Top manager 看到的是三个 sub-manager summaries，而不是十五个 worker outputs。

### 它在哪里破裂

2026 post-mortems 反复发现三种 failure modes：

1. **Task assignment error。** Manager 读取 goal，hallucinate 一个 decomposition，然后委派给错误的 sub-manager。因为 sub-manager 会顺从地处理收到的任务，错误只会在 top synthesis 才浮现——距离 human 本来可以捕捉它的位置已经隔了一层。
2. **Output misinterpretation。** Sub-manager 返回 “unable to verify claim X”。Top manager summarize 成 “claim X not confirmed”。Meaning 在每一层都会漂移。
3. **Consensus loops。** 两个 sub-managers 不同意；top manager 要求它们 reconcile；它们重新向下委派；workers 重新运行；sub-managers 返回略微不同的答案；loop。CrewAI 的 `Process.hierarchical` 用 step limits 防止这种情况，但这个 limit 本身现在成了 hyperparameter。

### 决定性问题

Sequential（linear pipeline）vs hierarchical：你的 task 真的有 independent sub-teams，还是一个假装成 tree 的 linear flow？如果是后者，用 sequential。如果是前者，用 hierarchical，但要为 explicit reconciliation rules 留 budget。

### CrewAI 的实现

`Process.hierarchical` 把一个 manager LLM 接到 specialist crews 上。Manager：

- receives the top-level task,
- assigns subtasks to crews,
- evaluates crew outputs,
- decides whether to accept, re-delegate, or iterate.

Documentation: https://docs.crewai.com/en/introduction（在 Core Concepts 下找 “Hierarchical Process”）。

### LangGraph 的实现

LangGraph 使用嵌套 `create_supervisor` calls。Inner supervisor 有自己的 graph；outer supervisor 把 inner graph 当作 opaque node。与 CrewAI 相比，这对 debugging 更干净（你可以分别 step through each graph），但更难表达 tree 的 dynamic reshaping。

Reference: https://reference.langchain.com/python/langgraph-supervisor.

## 动手实现

`code/main.py` 运行一个 3-level hierarchy：

- top manager：把 task 分成 “engineering” 和 “legal” branches，
- engineering sub-manager：拆成 “frontend” 和 “backend” workers，
- legal sub-manager：一个 worker。

Demo 对比 happy path（所有人一致）和 **perturbed path**，其中 top manager 的 decomposition 把 “legal” 误标为 “finance”，然后观察 error cascade——sub-manager 顺从地做 finance work，top synthesizer 报告 finance findings，原始 legal question 没有被回答。

运行：

```text
python3 code/main.py
```

Output 展示两条路径，并清晰并排比较 “what was asked” vs “what was delivered”。

## 实际使用

`outputs/skill-hierarchy-fitness.md` 评估给定 task 应该使用 hierarchical、sequential 还是 flat supervisor。Inputs：task description、org structure、reconciliation budget。Output：pattern recommendation，以及需要防守的具体 failure modes。

## 交付成果

如果你 ship hierarchical：

- **Cap tree depth at 2。** 三层已经会让大多数 errors 从 observability 中隐藏。
- **Explicit reconciliation budget。** 设置 top manager 必须 commit 前的 max rounds。通常是 2。
- **Provenance on every synthesis。** 每个 node 的 summary 都必须 cite 是哪些 leaf outputs 产生了它。
- **Alert on decomposition drift。** Log manager 每一步的 decomposition；diff against user query。如果 decomposition 不再 cover query，就 fire an alert。

## 练习

1. 运行 `code/main.py` 并比较 happy vs perturbed。经过多少层 manager hand-off，top output 才会完全偏离 user question？
2. 增加第三层（top → sub → sub-sub → worker）。随着 depth 增长，测量 perturbed path 自我修正 vs 完全 diverge 的频率。
3. 在每个 sub-manager 中实现一个 “canary” worker，它总是被原样询问 original user question。使用 canary answer 检测 decomposition drift。当 canary 与 synthesized answer 不一致时，manager 应该如何响应？
4. 阅读 CrewAI 的 `Process.hierarchical` docs。识别 CrewAI 应用的一个具体 guardrail（step limit、manager_llm constraint），并描述它针对哪个 failure mode。
5. 比较 nested LangGraph supervisors 与 CrewAI hierarchical。哪一个让 reconciliation loops 更便宜地被检测？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Hierarchical | "Org chart pattern" | Supervisors over supervisors；只有 leaves 做工作。 |
| Manager LLM | "The boss" | 在 internal node 负责 decomposes、assigns 和 validates 的 LLM。 |
| Decomposition drift | "The boss lost the plot" | Top manager 的 split 不再 cover original question。 |
| Reconciliation loop | "Endless meetings" | Sub-managers 不同意；top re-delegates；workers re-run；loop 直到 budget exhausted。 |
| Depth-2 ceiling | "Don't go deeper than 2 levels" | Empirical guardrail：3+ levels 会坍缩 observability。 |
| Canary question | "Ground truth at every level" | 一个总是被原样询问 original query 的 worker，用来 detect drift。 |
| Provenance chain | "Who said what" | 从每个 synthesis 回溯到产生它的 leaf outputs 的 trace。 |

## 延伸阅读

- [CrewAI introduction — Process.hierarchical](https://docs.crewai.com/en/introduction) —— 带有 manager LLM 的 textbook hierarchical
- [LangGraph supervisor reference](https://reference.langchain.com/python/langgraph-supervisor) —— 通过 `create_supervisor` 实现 nested supervisor
- [Anthropic engineering — Research system](https://www.anthropic.com/engineering/multi-agent-research-system) —— Anthropic 为什么刻意选择 flat supervisor 而不是 hierarchical
- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) —— MAST taxonomy；coordination failures 章节记录 decomposition drift
