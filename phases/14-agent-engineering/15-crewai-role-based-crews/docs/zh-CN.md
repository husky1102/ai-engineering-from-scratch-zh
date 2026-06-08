# CrewAI：基于角色的 Crews 与 Flows

> CrewAI 是 2026 年基于角色的 multi-agent framework。四个 primitives：Agent、Task、Crew、Process。两种顶层形态：Crews（autonomous、role-based collaboration）和 Flows（event-driven、deterministic）。文档说得很直白："for any production-ready application, start with a Flow."

**类型:** 学习 + 构建
**语言:** Python (stdlib)
**先修:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 14 (Actor Model)
**时间:** ~75 分钟

## 学习目标

- 说出 CrewAI 的四个 primitives（Agent、Task、Crew、Process），以及每个 primitive 拥有什么。
- 区分 Sequential、Hierarchical 和计划中的 Consensus process；能为每种 workload 选择一种。
- 区分 Crews（autonomous role-based）与 Flows（event-driven deterministic），并解释文档的生产建议。
- 使用 `@tool` decorator 和 `BaseTool` subclass 接入 tools；推理 structured outputs 与 free text 的取舍。
- 说出 CrewAI 的四种 memory types，以及每种何时值得使用。
- 实现一个 stdlib 三 agent crew（researcher、writer、editor），生成一份 brief。
- 识别三种 CrewAI failure modes：prompt-bloat、manager-LLM tax、brittle handoffs。

## 要解决的问题

采用 multi-agent frameworks 的团队会撞上同一堵墙。"Autonomous collaboration" 在 demo 里听起来很棒。然后客户提交一个 bug，你需要 deterministic replay。或者财务问一次 LLM-routed crew 的运行成本是多少。或者 on-call 需要知道凌晨 3 点哪个 agent 卡住了。

Free-form LLM-routed crews 无法干净地回答这些问题。Pure DAGs 全都能回答，但会失去 brainstorming agent 所需的探索形态。

CrewAI 的拆分坦诚面对了这个取舍。Crews 用于协作式、基于角色、探索性的工作。Flows 用于 event-driven、code-owned、可审计的生产。一个框架，两种形态，按 surface 选择。

## 核心概念

### 四个 primitives

CrewAI 的 surface 很小。记住这里，剩下的都是 config。

- **Agent。** `role + goal + backstory + tools + (optional) llm`。Backstory 是 load-bearing 的。它塑造 tone、judgment，以及 agent 何时停止。Tools 是 agent 能调用的函数（见下文）。
- **Task。** `description + expected_output + agent + (optional) context + (optional) output_pydantic`。可复用的工作单元。`expected_output` 是 contract。`context` 列出上游 tasks，其 outputs 会传入。`output_pydantic` 强制 structured shape。
- **Crew。** Container。拥有 `agents` 列表、`tasks` 列表、`process`，以及可选的 `memory` + `verbose` + `manager_llm` settings。
- **Process。** Execution strategy。Sequential、Hierarchical、Consensus（planned）。选择 run 的形状。

Agents 不能直接看见彼此。Tasks 引用 agents。Crew 排列 tasks。Process 决定谁选择下一项 task。这就是完整 mental model。

> **基于以下版本验证：** CrewAI 0.86（2026-05）。更新版本可能 rename 或 merge process types；依赖具体形态前，请查看 [CrewAI Processes docs](https://docs.crewai.com/concepts/processes)。

### Sequential vs Hierarchical vs Consensus

- **Sequential。** Tasks 按声明顺序运行。Task N 的输出可以作为 `context` 提供给 task N+1。成本最低。最可预测。适用于顺序固定的场景。
- **Hierarchical。** 一个 manager Agent（单独 LLM 调用）在 specialists 之间 routing。CrewAI 会从你的 `manager_llm` config 或默认配置生成 manager。Manager 每轮选择下一项 task，可以拒绝或重新 route。适用于有四个或更多 specialists 且顺序确实依赖先前输出的场景。
- **Consensus。** 计划中，当前 public API 尚未实现。文档为未来 voting-based process 保留了这个名称。今天不要依赖它。

Hierarchical 会在每次 specialist call 之外额外增加一个 per-round LLM call（manager）。五步 run 的 token cost 可能变成三倍。只有真正需要 routing 时才为它付费。

### Crews vs Flows

这是 2026 年文档最先强调的 framing。

- **Crew。** LLM-driven autonomy。框架在 runtime 选择形状。适合：research、brainstorming、first drafts，以及任何 path 本身就是答案一部分的地方。难以 replay。难以 test。便宜地 prototype。
- **Flow。** 你拥有的 event-driven graph。`@start` 标记入口。`@listen(topic)` 标记某个 step，会在另一个 step 发出该 topic 时触发。每个 step 都是普通 Python（内部可以调用 Crew）。适合：production。Observable。Testable。Deterministic。

文档在 2026 年的生产建议：从 Flow 开始。当 autonomy 值得其成本时，把 Crews 作为 Flow steps 内部的 `Crew.kickoff()` calls 折进去。Flow 给你 audit trail，Crew 给你 exploration。组合它们，不要二选一。

### Tool integration

给 Agent 提供 tool 有三种方式。选择最简单且适合的一种。

1. **`@tool` decorator。** 纯函数变成 tools。Signature 是 schema；docstring 是 LLM 看到的 description。最适合一次性 helpers。

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool` subclass。** Class-based tool，带显式 args schema、async support、retries。当 tool 有 state（client、cache）或需要 structured args 时使用。

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **Built-in toolkits。** CrewAI 提供 first-party adapters：`SerperDevTool`、`FileReadTool`、`DirectoryReadTool`、`CodeInterpreterTool`、`RagTool`、`WebsiteSearchTool`。一个 import 即可接好。

Structured outputs 使用 Pydantic。在 Task 上传入 `output_pydantic=MyModel`。CrewAI 会用 model 验证 LLM response，并进行 coercion 或 retry。把它和紧凑的 `expected_output` string 搭配。Free-text outputs 适合 drafts；structured outputs 才是下游 Flows 能消费的东西。

### Memory hooks

CrewAI 内置四种 memory types。它们可以组合：一个 Crew 可以同时启用全部四种。

> **基于以下版本验证：** CrewAI 0.86（2026-05）。近期版本会通过统一的 `Memory` system 路由所有内容，该系统包装这四个 stores。下面的概念模型仍然成立，但更新版本中的 public class surface 可能折叠成单一 `Memory` entry-point；请查看 [CrewAI memory docs](https://docs.crewai.com/concepts/memory) 获取当前 API。

- **Short-term。** 单次 run 内的 conversation buffer。结束后清空。
- **Long-term。** 跨 runs 持久化。存储在 vector DB 中（默认 Chroma，可替换）。按与当前 task 的相似度检索。
- **Entity。** 每个 entity 的 facts。"Customer X is on the enterprise plan." 按 entity key，而不是按 similarity。跨 runs 保留。
- **Contextual。** Assembly-time retrieval。在 Agent 需要时拉取相关 memory，而不是预加载。

通过 `memory=True` 或 per-type config 在 Crew 上启用。背后由你配置的 embeddings provider 支持（默认 OpenAI，可换成本地）。Memory 是 CrewAI 相比更薄框架能证明自己价值的地方之一；pure LangGraph 需要你自己接好这些内容。

### CrewAI 适合的地方

- 三到六个 agents，具名 roles，并且有协作 workflow。Drafting、reviewing、planning、brainstorming。
- LLM 对下一步的判断本身就是价值的一部分时的 routing（Hierarchical）。
- 任何团队更愿意读 `role + goal + backstory`，而不是读 graph definition 的地方。

### CrewAI 不适合的地方

- 带严格顺序的 deterministic DAGs。使用 LangGraph（Lesson 13）。Graph shape 才是正确抽象；CrewAI 的 role framing 会形成摩擦。
- Sub-second latency budgets。Hierarchical 增加 round trips。即使 Sequential 也会串行化包含 backstories 和 prior outputs 的 prompts。
- Single-agent loops。跳过框架；一个 agent loop（Lesson 1）加 tool registry 更短。

Lesson 17（Agent Framework Tradeoffs）用矩阵展开了这些内容。简短版：CrewAI 位于 "collaborative role-based" 角落。

### Dependency shape

独立于 LangChain。Python 3.10 到 3.13。使用 `uv`。Star count：见 [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)（截至 2026-05 的 snapshot）。AWS Bedrock integration 有文档；vendor benchmarks 报告在 QA workloads 上相对 LangGraph 有显著 speedup，但 methodology（dataset、hardware、evaluation metric）未公开，因此只把 framework-vendor numbers 当作方向性信号。

### 这个模式容易出错的地方

- **Prompt-bloat from backstories。** 每个 agent 一份 2000-word backstory，再加五个 agents，第一次 tool call 前 context budget 就烧完了。把 backstories 控制在 200 words 内。跨 agents 复用短语；不要把 house style 重复五遍。
- **Manager-LLM token tax。** Hierarchical process 在每次 specialist call 前都会增加一次 manager LLM call。五 task crew 会变成六次 LLM calls 而不是五次，并且 manager call 携带完整 task list 与 prior outputs。除非 routing 依赖 output，否则切到 Sequential。
- **Brittle handoffs。** Task N 的 `expected_output` 是 "an outline"。Task N+1 把它作为 `context` 读取，并尝试解析三节。LLM 产出了四节。下游 Agent 自由发挥。用 Task N 上的 `output_pydantic` 修复，让 Task N+1 读取 typed object，而不是 free text。
- **Crew-as-prod。** 没有 Flow wrapper 就把 free-form Crew 发到生产。Output variability 高；replay 不可能；on-call 无法 diff bad run 与 good one。用 Flow 包起来。

## 动手实现

`code/main.py` 实现了两种形态的 stdlib 版本，以及一个三 agent crew。

Shape：

- `Agent`、`Task` dataclasses，对应 CrewAI surface。
- `SequentialCrew.kickoff(inputs)` 按声明顺序运行 tasks，并把 outputs 作为 `context` 串起来。
- `HierarchicalCrew.kickoff(topic)` 添加 manager Agent，每轮选择下一个 specialist，遇到 "done" 停止。
- `Flow`，包含 `@start` 和 `@listen(topic)` decorators、一个微型 event loop 和 trace。
- `tool(name)` decorator，镜像 CrewAI 的 `@tool` 形态。
- `Memory`，包含 `short_term`、`long_term`、`entity` stores；mocked similarity 使用 numpy。
- Mock LLM responses 是按 role 加 input prefix 索引的 hardcoded strings。无网络。Deterministic。

具体 demo：researcher、writer、editor crew 产出一份关于 "agent engineering 2026" 的 brief。Researcher 拉取（mocked）sources。Writer 起草。Editor 收紧。同一个 crew 也通过 Flow 运行，以展示 deterministic shape。

运行：

```bash
python3 code/main.py
```

Trace 覆盖：sequential crew 通过 `context` 传递 outputs，hierarchical crew 中 manager 依次选择（researcher、writer、editor，然后 "done"），flow 用显式 topics（`researched`、`drafted`、`edited`）运行同样三步，通过 `@tool` routing 的 tool calls，以及 long-term memory 跨两次 kickoffs 存活。

Crew trace 是流动的；manager 原则上可以重新排序。Flow trace 是固定的。这个选择就是本课的重点。

## 实际使用

- **CrewAI Flow** 用于 production。即使 Flow 只有一步，里面调用 `Crew.kickoff()`。Flow 提供 audit boundary。
- **CrewAI Crew (Sequential)** 用于顺序清晰的协作工作，尤其是 first drafts 和 review loops。
- **CrewAI Crew (Hierarchical)** 用于 routing 依赖 output 且有四个或更多 specialists 的情况。
- **LangGraph**（Lesson 13）用于 explicit state machines、durable resume、strict ordering。
- **AutoGen v0.4**（Lesson 14）用于 actor-model concurrency 和 fault isolation。
- **OpenAI Agents SDK**（Lesson 16）用于 OpenAI-first products，包含 handoffs 和 guardrails。
- **Claude Agent SDK**（Lesson 17）用于 Claude-first products，包含 subagents 和 session store。

## 交付成果

`outputs/skill-crew-or-flow.md` 会为任务选择 Crew 或 Flow，并 scaffold 最小实现。它会 hard reject Crew-without-backstory、Flow-without-explicit-topics，以及 specialists 少于三个的 Hierarchical。

## 常见陷阱

- **Backstory as flavor。** 它塑造 outputs。每个 agent 测试三种 variants；variance 是真实的。选定一个，冻结它。
- **Skipping `expected_output`。** 没有每个 task 的 contract，下游 tasks 会接收 LLM 产出的任何东西。Crew 能跑；audit 会失败。
- **Memory always-on。** Long-term 每次 run 都写入。Vector DB 增长。Retrieval 变噪。把 writes 限定在 fact 持久存在的 tasks 上。
- **Manager prompt drift。** Hierarchical 的 manager prompt 是隐式的。如果 routing 变奇怪，就在 verbose mode 中 dump 出来并阅读。
- **Tool side effects in Crews。** Crew 可能比预期更多次调用 tool。POST、DELETE、payment 属于 Flow step，永远不要放在 Crew tool 里。

## 练习

1. 把 Sequential crew 转成 Flow。数一数 variability 下降的接触点。记录 readability 哪里下降了。
2. 给 crew 添加 entity memory：关于 customer 的 facts 跨 kickoffs 持久化。验证 retrieval 拉到了正确 entity。
3. 实现一个 Hierarchical process：manager 在 writer output 至少有三段之前拒绝 route 给 editor。跟踪 retry。
4. 为（mocked）web search 接一个 `BaseTool` subclass。比较它和 `@tool` decorator 版本的 trace shape。
5. 给 editor task 添加 `output_pydantic=Brief`，其中 `Brief` 包含 `title`、`summary`、`sections`。让 writer task 先输出一次 malformed JSON；验证 CrewAI 的 retry behavior 出现在 trace 中。
6. 阅读 CrewAI 的 docs intro。把 toy 移植到真实 `crewai` API。Stdlib 版本跳过了哪些 guarantees？
7. 把 AgentOps 或 Langfuse（Lesson 24）接到真实 run。Stdlib 版本里你缺了哪些 traces？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Agent | "Persona" | Role + goal + backstory + tools |
| Task | "Unit of work" | Description + expected output + assignee + optional structured output |
| Crew | "Agent team" | Agents + Tasks + Process 的 container |
| Process | "Execution strategy" | Sequential / Hierarchical / Consensus（planned） |
| Flow | "Deterministic workflow" | Event-driven、code-owned、testable |
| Backstory | "Persona prompt" | Agent 的 tone 与 judgment 塑形器 |
| `@tool` | "Function tool" | 把函数转成 Agent 可调用 tool 的 decorator |
| `BaseTool` | "Class tool" | 带 args schema、retries、async support 的 class-based tool |
| Entity memory | "Per-entity facts" | 限定到 customer / account / issue 的 memory |
| Long-term memory | "Cross-run memory" | 在 kickoffs 之间存活的 vector-backed memory |
| Contextual memory | "Just-in-time retrieval" | Agent 需要时才拉取的 memory |
| Manager LLM | "Router agent" | Hierarchical process 中选择下一个 task 的额外 LLM |
| `expected_output` | "Task contract" | 告诉 Agent（也告诉 audit）应返回什么 shape 的 string |

## 延伸阅读

- [CrewAI docs introduction](https://docs.crewai.com/en/introduction)：concepts 和推荐的生产路径
- [CrewAI Flows guide](https://docs.crewai.com/en/concepts/flows)：event-driven shape、`@start`、`@listen`
- [CrewAI tools reference](https://docs.crewai.com/en/concepts/tools)：`@tool`、`BaseTool`、built-in toolkits
- [CrewAI memory](https://docs.crewai.com/en/concepts/memory)：short-term、long-term、entity、contextual
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)：何时 multi-agent 有帮助，何时没有
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)：state-machine alternative
