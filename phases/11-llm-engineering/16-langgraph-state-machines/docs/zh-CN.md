# LangGraph：Agent 的状态机

> 手写的 ReAct loop 是一个 `while True`。用 LangGraph 写的 ReAct loop 是一张可以 checkpoint、interrupt、branch 和 time-travel 的图。agent 没变。变的是围绕它的 harness。

**类型：** 构建
**语言：** Python
**先修：** Phase 11 · 09（Function Calling），Phase 11 · 14（Model Context Protocol）
**时间：** ~75 分钟

## 要解决的问题

你发布了一个 function-calling agent。它能正常工作三轮，然后某处出错：模型尝试了一个返回 500 的工具，用户在任务中途改变主意，或者 agent 在没有人工签署的情况下决定退款。`while True:` loop 没有 hook。你不能暂停它，不能回退它，也不能分叉出“如果模型当时选了另一个工具会怎样”。一旦你把它从 demo 推向生产，agent 就会变成一个黑盒：要么成功，要么失败。

一旦看见它，下一步就很明显。agent 本来就是状态机：system prompt、message history、pending tool calls 以及 next action。把状态机显式化：节点表示“模型思考”“工具运行”“人工批准”，边表示它们之间的条件转移。当图变成显式以后，harness 会自然获得四种能力：checkpointing（在步骤之间保存状态）、interrupts（暂停等待人工）、streaming（流式输出 token 和中间事件）以及 time-travel（回退到之前的状态并尝试另一条分支）。

LangGraph 就是提供这个抽象的库。它不是 LangChain 意义上的 agent framework（“这里有个 AgentExecutor，祝好运”）。它是一个图运行时，拥有一等的 state、一等的 persistence 和一等的 interrupts。agent loop 是你画出来的东西，而不是手写出来的东西。

## 核心概念

![LangGraph StateGraph：节点、边和 checkpointer](../assets/langgraph-stategraph.svg)

一个 `StateGraph` 有三样东西。

1. **State。** 在图中流动的 typed dict（TypedDict 或 Pydantic model）。每个节点都接收完整 state 并返回一个 partial update，LangGraph 会用每个字段对应的 *reducer* 合并它，列表累积用 `operator.add`，默认行为是覆盖。
2. **Nodes。** Python 函数 `state -> partial_state`。每个节点都是一个离散步骤：“调用模型”“运行工具”“总结”。
3. **Edges。** 节点之间的转移。静态边只去一个地方。条件边接收一个 router 函数 `state -> next_node_name`，让图可以基于模型输出分支。

你会编译这张图。Compile 会绑定拓扑，附加 checkpointer（可选，但生产必需），并返回一个 runnable。你用初始 state 和 `thread_id` 调用它。执行的每一步都会持久化一个以 `(thread_id, checkpoint_id)` 为键的 checkpoint。

### 四种超能力

**Checkpointing。** 每个节点转移都会把新 state 写到存储中（测试用 in-memory，生产用 Postgres/Redis/SQLite）。用相同的 `thread_id` 再次调用图即可恢复。图会从暂停处继续。

**Interrupts。** 用 `interrupt_before=["human_review"]` 标记一个节点，执行会在该节点运行前停止。state 会被持久化。你的 API 向用户响应“等待批准”。随后对同一 `thread_id` 发起的请求可携带 `Command(resume=...)` 继续执行。

**Streaming。** `graph.stream(state, mode="updates")` 会在 state delta 发生时产出它们。`mode="messages"` 会流式输出模型节点内部的 LLM token。`mode="values"` 会产出完整快照。你选择要在 UI 中暴露哪一种。

**Time-travel。** `graph.get_state_history(thread_id)` 返回完整 checkpoint log。把任意之前的 `checkpoint_id` 传给 `graph.invoke`，你就能从那里分叉。它很适合调试（“如果模型选了 tool B 会怎样？”），也适合回放生产 trace 的回归测试。

### Reducer 是关键

每个 state 字段都有一个 reducer。多数默认值都可以，一个新值会覆盖旧值。但 message list 需要 `operator.add`，这样新消息会追加，而不是替换。并行边通过 reducer 合并更新。如果两个节点都更新 `messages`，而你忘了 `Annotated[list, add_messages]`，第二个更新会静默获胜，你会丢掉半轮内容。reducer 是这个库唯一微妙的地方；把它做对，其余部分就能组合起来。

### 四个节点里的 ReAct graph

生产 ReAct agent 是四个节点和两条边：

1. `agent`：用当前 message history 调用 LLM。返回 assistant message（其中可能包含 tool_calls）。
2. `tools`：执行最后一条 assistant message 中的任何 tool_calls，并把 tool 结果作为 tool messages 追加。
3. 一条从 `agent` 出发的条件边：如果最后一条消息有 tool_calls，就路由到 `tools`，否则到 `END`。
4. 一条从 `tools` 返回 `agent` 的静态边。

就是这些。你用大约 40 行代码就获得完整 ReAct loop（Thought → Action → Observation → Thought → …），并且自带 checkpointing、interrupts 和 streaming。

### StateGraph vs Send（fanout）

`Send(node_name, state)` 允许一个节点派发并行子图。例子：agent 决定同时查询三个 retriever。每个 `Send` 都会启动一次目标节点的并行执行；它们的输出通过 state reducer 合并。这就是 LangGraph 表达 orchestrator-workers 模式的方式，不需要线程原语。

### Subgraphs

编译后的 graph 可以成为另一张 graph 中的节点。外层 graph 看到的是单个节点；内层 graph 拥有自己的 state 和自己的 checkpoints。这就是团队构建 supervisor-worker agents 的方式：supervisor graph 把用户意图路由到按领域划分的 worker subgraph。

## 动手实现

### 步骤 1：state 和 nodes

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages` 是让 message list 累积而不是覆盖的 reducer。忘记它是最常见的 LangGraph bug。

### 步骤 2：用 thread 运行

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

每个 update 都是一个 dict `{node_name: state_delta}`。你的前端可以把它们流式传到 UI，让用户看到“agent 正在思考……调用 search_web……得到结果……正在回答。”

### 步骤 3：加入 human-in-the-loop interrupt

标记一个节点，让执行在它运行前暂停。

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # pause before every tool call
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] is set. Inspect proposed tool calls.
# If approved:
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# If denied: write a rejection message and resume
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

state、checkpoint 和 thread 都会跨 interrupt 持久存在。除了执行期间，没有任何东西只留在内存里。

### 步骤 4：用 time-travel 调试

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Fork from a prior checkpoint
target = history[3].config  # three steps back
for event in app.stream(None, target, stream_mode="values"):
    pass  # replay from that point forward
```

把 `None` 作为输入会从给定 checkpoint 回放；传入一个值则会先把它作为更新追加到该 checkpoint 的 state，再继续。这就是你无需重跑整段对话就能复现一次糟糕 agent run 的方式。

### 步骤 5：把 checkpointer 换成生产版本

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite、Redis 和 Postgres 都已提供。`MemorySaver` 用于测试。任何需要跨重启持久化的东西都需要真实存储。

## 能力要点

> 你把 agent 构建成图，而不是 `while True` loop。

在使用 LangGraph 之前，先做一个 60 秒设计：

1. **命名节点。** 每个离散决策或有副作用的动作都是节点。“Agent thinks”“tool runs”“reviewer approves”“response streams”。如果你列不出来，任务还不是 agent-shaped。
2. **声明 state。** 最小 TypedDict，并为每个 list 字段配 reducer。不要把所有东西都塞进 `messages`；把任务特定字段（一个工作中的 `plan`、一个 `budget` 计数器、一个 `retrieved_docs` 列表）提升到顶层。
3. **画出边。** 除非下一步取决于模型输出，否则用静态边。每条条件边都需要一个带命名分支的 router 函数。
4. **一开始就选择 checkpointer。** 测试用 `MemorySaver`，其他情况用 Postgres/Redis/SQLite。不要在没有 checkpointer 的情况下发布，因为没有 checkpointer 就没有 resume、interrupt 或 time-travel。
5. **在工具运行前决定 interrupt，而不是运行后。** 审批要放在进入有副作用节点的边上，这样你能在造成损害前取消；验证放在模型输出的边上，这样可以低成本拒绝糟糕调用。
6. **默认 streaming。** UI 用 `mode="updates"`，模型节点内部的 token-level streaming 用 `mode="messages"`，eval 期间的完整快照用 `mode="values"`。

拒绝发布没有 checkpointer 的 LangGraph agent。拒绝发布在副作用*之后*才 interrupt 的 agent。拒绝发布没有把 `add_messages` 作为 reducer 的 `messages` 字段。

## 练习

1. **简单。** 使用 calculator tool 和 web-search tool 实现上面的四节点 ReAct graph。验证对一段两轮对话，`list(app.get_state_history(config))` 至少返回四个 checkpoints。
2. **中等。** 添加一个在 `agent` 之前运行的 `planner` 节点，并把结构化的 `plan: list[str]` 写入 state。让 `agent` 把计划步骤标记为完成。如果 `plan` 在 checkpoint resume 后丢失（reducer 错误），测试应失败。
3. **困难。** 构建一个 supervisor graph，用 `Send` 在三个 subgraphs（`researcher`、`writer`、`reviewer`）之间路由。每个 subgraph 都有自己的 state 和 checkpointer。在外层 graph 上添加 `interrupt_before=["writer"]`，让人工批准 research brief。确认从之前 checkpoint time-travel 时只会重新运行分叉分支。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| StateGraph | “LangGraph 的 graph” | 你在 compile 之前添加节点和边的 builder object。 |
| Reducer | “字段如何合并” | 节点返回某字段更新时应用的函数 `(old, new) -> merged`；默认覆盖，`add_messages` 会追加。 |
| Thread | “一个 conversation ID” | 一个 `thread_id` 字符串，用来限定单个 session 的所有 checkpoints。 |
| Checkpoint | “暂停状态” | 节点转移之后完整 graph state 的持久化快照，以 `(thread_id, checkpoint_id)` 为键。 |
| Interrupt | “暂停给人工处理” | `interrupt_before` / `interrupt_after` 在节点边界停止执行；用 `Command(resume=...)` 恢复。 |
| Time-travel | “从之前步骤分叉” | `graph.invoke(None, config_with_old_checkpoint_id)` 从该 checkpoint 开始向前回放。 |
| Send | “并行 subgraph 派发” | 一个节点可以返回的构造器，用于启动目标节点的 N 次并行执行。 |
| Subgraph | “作为节点的 compiled graph” | 在另一张 graph 中作为节点使用的 compiled StateGraph；保留自己的 state scope。 |

## 延伸阅读

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/)：StateGraph、reducers、checkpointers 和 interrupts 的权威参考。
- [LangGraph concepts: state, reducers, checkpointers](https://langchain-ai.github.io/langgraph/concepts/low_level/)：本课使用的心智模型，直接来自源文档。
- [LangGraph Persistence and Checkpoints](https://langchain-ai.github.io/langgraph/concepts/persistence/)：Postgres/SQLite/Redis stores、checkpoint namespaces 和 thread IDs 的细节。
- [LangGraph Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)：`interrupt_before`、`interrupt_after`、`Command(resume=...)` 以及 edit-state 模式。
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629)：每个 LangGraph agent 都实现的模式；阅读它以理解 reasoning trace 的理由。
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents)：应优先选择哪些 graph shape（chain、router、orchestrator-workers、evaluator-optimizer）以及何时选择。
- Phase 11 · 09（Function Calling）：每个 LangGraph agent 节点都会复用的 tool-call 原语。
- Phase 11 · 14（Model Context Protocol）：通过 MCP adapter 插入 LangGraph `ToolNode` 的外部工具发现。
- Phase 11 · 17（Agent framework tradeoffs）：什么时候选择 LangGraph，而不是 CrewAI、AutoGen 或 Agno。
