# Tool Interface：为什么 Agent 需要结构化 I/O

> 语言模型产生 token。程序执行 action。二者之间的缺口就是 tool interface：一个让模型请求 action、让 host 执行 action 的 contract。2026 年的每个 stack，OpenAI、Anthropic、Gemini 上的 function calling，MCP 的 `tools/call`，A2A 的 task parts，都是同一个四步 loop 的不同编码。本课命名这个 loop，并展示运行它所需的最小机械结构。

**类型:** Learn
**语言:** Python（stdlib，无 LLM）
**先修:** Phase 11（LLM completion APIs）
**时间:** ~45 分钟

## 学习目标

- 解释为什么只能生成文本的 LLM 本身无法对真实世界采取 action。
- 画出四步 tool-call loop（describe → decide → execute → observe），并说明每一步由谁负责。
- 把 tool description 写成三部分：name、JSON Schema input 和 deterministic executor function。
- 区分 pure tools 与 side-effecting tools，并说明这种拆分为什么对安全重要。

## 要解决的问题

LLM 发出的是下一个 token 的概率分布。这就是它全部的输出表面。如果你问聊天模型“what is the weather in Bengaluru right now”，它可以写出看似合理的句子，但它不能拨入天气 API。这个句子可能碰巧正确，也可能已经过期三天。

弥合这个缺口就是 tool interface 的目的。Host program，也就是你的 agent runtime、Claude Desktop、ChatGPT、Cursor 或 custom script，会把一组 callable tools 广告给模型。模型在判断需要 action 时，会发出结构化 payload，命名 tool 及其 arguments。Host 解析 payload，真实运行 tool，并把 result 喂回去。Loop 继续，直到模型判断不再需要调用。

这个 contract 的第一个版本于 2023 年 6 月以 OpenAI 的 `functions` 参数发布。Anthropic 随 Claude 2.1 加入 `tool_use` blocks。Gemini 几个月后加入 `functionDeclarations`。如今每个 provider 都暴露同样的形状：输入是 JSON-Schema-typed tool list，输出是 JSON-payload tool call。Model Context Protocol（2024 年 11 月）把 contract 泛化，使一个 tool registry 可以服务每个模型。A2A（2026 年 4 月，v1.0）为 agent-to-agent delegation 叠加了同样 primitive。

四步 loop 是这一切下面的不变量。Phase 13 的其他内容都是展开。

## 核心概念

### Step one：describe

Host 用三个字段声明每个 tool。

- **Name.** 稳定、机器可读的 identifier。用 `get_weather`，不要用“weather thing”。
- **Description.** 一段自然语言简介。“Use when the user asks about current conditions for a specific city. Do not use for historical data.”
- **Input schema.** 一个 JSON Schema object（draft 2020-12），描述 tool 的 arguments。

模型收到这个列表。现代 provider 会用 provider-specific template 把这些 declaration 序列化进 system prompt，所以调用方只需要处理结构化形式。

### Step two：decide

给定用户消息和可用 tools，模型会选择三种行为之一。

1. **直接用文本回答。** 不调用 tool。
2. **调用一个或多个 tools。** 发出结构化 call objects。在 `parallel_tool_calls: true` 下（OpenAI 和 Gemini 默认，Anthropic opt-in），模型可以在一个 turn 中发出多个 call。
3. **拒绝。** Strict-mode structured outputs 可以产生 typed `refusal` block，而不是 call。

Tool call payload 有三个稳定字段：call `id`、tool `name` 和 JSON `arguments` object。id 的存在是为了让 host 把后续 result 与特定 call 关联起来，这在 parallel call 乱序返回时很重要。

### Step three：execute

Host 收到 call，按声明 schema 验证 arguments，然后运行 executor。无效 arguments 意味着模型幻觉出了字段或使用了错误类型，这是弱模型上非常常见的 failure mode。生产 host 对无效 arguments 通常三选一：fail fast 并把错误暴露给模型、用 constrained parser 修复 JSON，或把 validation error 纳入 prompt 后重试模型。

Executor 本身只是普通代码。Python、TypeScript、shell command、database query 都可以。它产生 result，通常是 string，但也可以是任何 JSON value 或 structured content block（MCP 中的 text、image 或 resource reference）。Result 必须可序列化。

### Step four：observe

Host 把 tool result 追加到 conversation（作为带 matching `id` 的 `tool` role message），然后再次调用模型。模型现在在 context 中有 tool output，可以生成 final answer 或请求更多 call。这个过程持续到模型停止发出 call，或 host 达到 iteration count 的 safety limit。

### Trust split

Tools 有两种与安全相关的风味。

- **Pure.** 只读、确定性、无 side effect。`get_weather`、`search_docs`、`get_current_time`。可以安全地 speculative call。
- **Consequential.** 改变 state、花钱、接触用户数据。`send_email`、`delete_file`、`execute_trade`。必须加 gate。

Meta 2026 年 agent security 的“Rule of Two”说，单个 turn 最多只能组合以下三项中的两项：untrusted input、sensitive data、consequential action。Tool interface 是你执行这条规则的位置：拒绝 call、要求用户确认，或提升 scopes。完整安全章节见 Phase 13 · 15，agent-level permission policies 见 Phase 14 · 09。

### Loop 位于哪里

| Context | 谁 describe | 谁 decide | 谁 execute |
|---------|---------------|-------------|--------------|
| Single-turn function calling（OpenAI/Anthropic/Gemini） | App developer | LLM | App developer |
| MCP | MCP server | LLM via MCP client | MCP server |
| A2A | Agent Card publisher | Calling agent | Called agent |
| Web browser（function-calling agent） | Browser extension / WebMCP | LLM | Browser runtime |

到处都是相同四步。列名会变，结构不变。

### 为什么不只是 prompt 模型输出 JSON？

“Ask the model to reply in JSON”是 function calling 之前的模式。它在 frontier models 上也会失败约 5 到 15%，在小模型上更多。Failure mode 包括缺少括号、trailing commas、hallucinated fields 和 wrong types。然后你需要 JSON repair pass、retry 或 constrained decoder。

Native function calling 有三个优点。第一，provider 端到端用精确 call shape 训练模型，因此 strict mode 下 valid-JSON rate 上升到 98 到 99%。第二，call payload 位于自己的 protocol slot 中，不在 free-text 里，所以 tool call 不会泄漏到用户可见回复。第三，provider 用 constrained decoding 强制 schema compliance（OpenAI strict mode、Anthropic `tool_use`、Gemini `responseSchema`）。输出保证可验证。

Phase 13 · 02 并排走读三家 provider API。Phase 13 · 04 深入 structured outputs。

### Circuit breakers

当模型停止发出 call，或 host 达到最大 turn count 时，loop 终止。生产 host 通常设为 5 到 20 turn。超过这个数，你几乎肯定进入了模型无法退出的循环。Claude Code 默认 20；OpenAI Assistants 默认 10；Cursor agent mode 默认 25。

另一种选择，也就是 unbounded loops，每六个月就会以“agent overnight 花掉 400 美元 API calls”的事后分析出现一次。不要在没有上限的情况下发布。

Phase 14 · 12 深入 error recovery 和 self-healing；Phase 17 覆盖 production rate limits。

### Phase 13 接下来去哪里

- Lessons 02 through 05 打磨 provider-level tool-call surface。
- Lessons 06 through 14 把 loop 泛化成 MCP。
- Lessons 15 through 18 防御 hostile servers、adversarial users 和 unauthenticated remote auth surfaces。
- Lessons 19 through 22 把模式扩展到 agent-to-agent collaboration、observability、routing 和 packaging。
- Lesson 23 使用每个 primitive 交付完整 ecosystem。

所有剩余 lesson 都是这个四步 loop 的展开。把它作为不变量记在脑中。

## 实际使用

`code/main.py` 在没有 LLM 的情况下运行四步 loop。一个假的“decider”函数通过 pattern-matching 用户消息模拟模型；executor、schema validator 和 observe-step harness 是真的。运行它，查看带可打印 intermediate state 的完整 request/response choreography，然后在后续 lesson 中把 fake decider 换成任意真实 provider。

重点看：

- Tool registry 对每个 tool 保存三个字段：name、description、schema，以及 executor reference。
- Validator 是用 stdlib 写的最小 JSON Schema 子集（types、required、enum、min/max）。Phase 13 · 04 会交付更完整版本。
- Loop 把 iteration count 限制为五。生产 agent 也需要这种 circuit breaker。

## 交付成果

本课产出 `outputs/skill-tool-interface-reviewer.md`。给定一个 draft tool definition（name + description + schema + executor outline），该 skill 会审计它是否适合 loop：name 是否机器稳定、description 是否是完整 usage brief、schema 是否正确使用 JSON Schema 2020-12、pure-vs-consequential 分类是否显式。

## 练习

1. 给 `code/main.py` 加第四个 tool，名为 `get_stock_price(ticker)`。把 description 写成“Use when the user asks for a current stock price by ticker. Do not use for historical prices or market summaries.” 运行 harness，并确认 fake decider 会把提到 ticker 的 query 路由到新 tool。

2. 故意破坏 schema validator。传入一个 `arguments` object 缺少 required field 的 call，确认 host 在 execution 前拒绝它。然后传入一个带额外 unknown field 的 call。决定：host 应该 reject 还是 ignore？用安全论证说明。

3. 把 harness 中每个 tool 分类为 pure 或 consequential。给需要的 registry entry 加上 `consequential: true` flag，并修改 loop，让 consequential tool 被选中时打印“would confirm with user”。这就是每个生产 host 都需要的 confirmation gate 形状。

4. 在纸上画出四步 loop，并用你最喜欢的 client（Claude Desktop、Cursor、ChatGPT 或 custom stack）填充上面的 provider-column table。与 Phase 13 · 06 中 MCP-specific variant 交叉对照。

5. 从头到尾阅读 OpenAI function-calling guide。找出一个位于 request 中、但不在本课四步 loop 里的字段。解释它增加了什么，以及为什么它是方便而非必要。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Tool | “A thing the model can call” | name + JSON-Schema-typed input + executor function 的三元组 |
| Function calling | “Native tool use” | Provider-level API 支持发出结构化 tool call，而不是 prose |
| Tool call | “The model's request to act” | 模型发出的带 `id`、`name`、`arguments` 的 JSON payload |
| Tool result | “What the tool returned” | executor 输出，被包进带 matching id 的 `tool` role message |
| Parallel tool calls | “Many calls at once” | 一个 model turn 中的多个 call objects，彼此独立并可按 id 排序 |
| Strict mode | “Guaranteed JSON” | 强制模型输出符合声明 schema 的 constrained decoding |
| Pure tool | “Read-only tool” | 无 side effect；可安全重跑 |
| Consequential tool | “Action tool” | 改变外部 state；需要 gate、audit 或用户确认 |
| Four-step loop | “The tool-call cycle” | describe → decide → execute → observe |
| Host | “Agent runtime” | 持有 tool registry、调用模型并运行 executor 的程序 |

## 延伸阅读

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — OpenAI-style tool declaration 和 call shape 的 canonical reference
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — Claude 的 `tool_use` / `tool_result` block format
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini 中的 `functionDeclarations` 和 parallel-call semantics
- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — tool interface 的 provider-agnostic generalization
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — 每个现代 tool API 使用的 schema dialect
