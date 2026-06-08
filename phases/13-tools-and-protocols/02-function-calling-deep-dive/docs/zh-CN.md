# Function Calling 深入：OpenAI、Anthropic、Gemini

> 三家 frontier provider 在 2024 年收敛到同一个 tool-call loop，然后在其他所有细节上分叉。OpenAI 使用 `tools` 和 `tool_calls`。Anthropic 使用 `tool_use` 和 `tool_result` blocks。Gemini 使用 `functionDeclarations` 和 unique-id correlation。本课并排 diff 三者，避免你把一个 provider 上线的代码迁移到另一个 provider 时坏掉。

**类型:** Build
**语言:** Python（stdlib，schema translators）
**先修:** Phase 13 · 01（the tool interface）
**时间:** ~75 分钟

## 学习目标

- 说出 OpenAI、Anthropic 与 Gemini function-calling payload 的三个 shape differences（declaration、call、result）。
- 在三种 provider format 之间翻译一个 tool declaration，并预测 strict-mode constraints 会在哪里不同。
- 在每个 provider 中使用 `tool_choice` 来强制、禁止或自动选择 tool calls。
- 了解每个 provider 的 hard limits（tool count、schema depth、argument length），以及违反限制时发出的 error signatures。

## 要解决的问题

Function-calling request 的形状因 provider 而异。来自 2026 生产 stack 的三个具体例子：

**OpenAI Chat Completions / Responses API.** 传入 `tools: [{type: "function", function: {name, description, parameters, strict}}]`。模型 response 包含 `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`，其中 `arguments` 是你必须解析的 JSON string。Strict mode（`strict: true`）通过 constrained decoding 强制 schema compliance。

**Anthropic Messages API.** 传入 `tools: [{name, description, input_schema}]`。Response 返回 `content: [{type: "text"}, {type: "tool_use", id, name, input}]`。`input` 已经解析好了（object，不是 string）。你用新的 `user` message 回复，其中包含 `{type: "tool_result", tool_use_id, content}` block。

**Google Gemini API.** 传入 `tools: [{functionDeclarations: [{name, description, parameters}]}]`（嵌套在 `functionDeclarations` 下）。Response 到达为 `candidates[0].content.parts: [{functionCall: {name, args, id}}]`，其中 Gemini 3 及以上的 `id` 对 parallel-call correlation 唯一。你回复 `{functionResponse: {name, id, response}}`。

同一个 loop。不同 field names、不同 nesting、不同 string-vs-object conventions、不同 correlation mechanisms。一个团队在 OpenAI 上写 weather agent，迁到 Anthropic 要花两天处理 plumbing，再迁到 Gemini 又要一天。

本课构建一个 translator，把三种格式统一成一个 canonical tool declaration，并在边缘路由。Phase 13 · 17 会把同样模式泛化成 LLM gateway。

## 核心概念

### 共同结构

每个 provider 都需要五件事：

1. **Tool list.** 每个 tool 的 name、description 和 input schema。
2. **Tool choice.** 强制某个 tool、禁止 tools，或让模型决定。
3. **Call emission.** 命名 tool 和 arguments 的结构化输出。
4. **Call id.** 把 response 关联到正确 call（parallel 时很重要）。
5. **Result injection.** 把 result 绑定回 call 的 message 或 block。

### Shape diffs：逐字段

| Aspect | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| Declaration envelope | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Schema field | `parameters` | `input_schema` | `parameters` |
| Response container | assistant message 上的 `tool_calls[]` | type 为 `tool_use` 的 `content[]` | type 为 `functionCall` 的 `parts[]` |
| Arguments type | stringified JSON | parsed object | parsed object |
| Id format | `call_...`（OpenAI 生成） | `toolu_...`（Anthropic） | UUID（Gemini 3+） |
| Result block | role `tool`，`tool_call_id` | 带 `tool_result`、`tool_use_id` 的 `user` | 带 matching `id` 的 `functionResponse` |
| Force-a-tool | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| Forbid tools | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| Strict schema | `strict: true` | schema-is-schema（总是执行） | request-level `responseSchema` |

### 你实际会撞到的限制

- **OpenAI.** 每 request 128 个 tools。Schema depth 5。Argument string <= 8192 bytes。Strict mode 要求无 `$ref`、无带 overlap 的 `oneOf`/`anyOf`/`allOf`，每个 property 都列入 `required`。
- **Anthropic.** 每 request 64 个 tools。Schema depth 实际上无界，但 practical limit 是 10。没有 strict-mode flag；schema 是 contract，模型通常遵循。
- **Gemini.** 每 request 64 个 functions。Schema types 是 OpenAPI 3.0 subset（与 JSON Schema 2020-12 略有差异）。Gemini 3 开始 parallel calls 有 unique-id。

### `tool_choice` 行为

每家都支持三种模式，只是名字不同。

- **Auto.** 模型选择 tool 或 text。默认。
- **Required / Any.** 模型必须调用至少一个 tool。
- **None.** 模型不得调用 tools。

每家还有一个独特模式：

- **OpenAI.** 按 name 强制特定 tool。
- **Anthropic.** 按 name 强制特定 tool；`disable_parallel_tool_use` flag 分离 single 与 multi。
- **Gemini.** `mode: "VALIDATED"` 会无论模型意图如何，都把每个 response 通过 schema validator。

### Parallel calls

OpenAI 的 `parallel_tool_calls: true`（默认）会在一个 assistant message 中发出多个 call。你运行全部 call，并用 batched tool-role message 回复，每个 `tool_call_id` 一条。Anthropic 历史上是 single-call；`disable_parallel_tool_use: false`（Claude 3.5 起默认）启用 multi。Gemini 2 支持 parallel calls 但没有 stable ids；Gemini 3 加入 UUID，使 out-of-order responses 能干净关联。

### Streaming

三者都支持 streamed tool calls。Wire format 不同：

- **OpenAI.** `tool_calls[i].function.arguments` 的 delta chunks 增量到达。累积到 `finish_reason: "tool_calls"`。
- **Anthropic.** Block-start / block-delta / block-stop events。`input_json_delta` chunks 携带 partial arguments。
- **Gemini.** `streamFunctionCallArguments`（Gemini 3 新增）用 `functionCallId` 发出 chunks，使多个 parallel calls 可以交错。

Phase 13 · 03 深入 parallel + streaming reassembly。本课聚焦 declaration 和 single-call shapes。

### Errors and repair

Invalid-argument errors 看起来也不同。

- **OpenAI（non-strict）.** 模型返回 `arguments: "{bad json}"`，你的 JSON parse 失败，你注入 error message 并重新调用。
- **OpenAI（strict）.** Validation 在 decoding 期间发生；invalid JSON 不可能，但可能出现 `refusal`。
- **Anthropic.** `input` 可能包含 unexpected fields；schema 是 advisory。Server-side validate。
- **Gemini.** OpenAPI 3.0 quirk：object fields 上的 `enum` 会被静默忽略；自行 validate。

### Translator pattern

你代码中的 canonical tool declaration 可以长这样（shape 由你选择）：

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

三个小函数把它翻译成三种 provider shape。`code/main.py` 中的 harness 正是这样做的，然后让 fake tool call 在每个 provider 的 response shape 中 round-trip。无需网络，本课教的是 shape，不是 HTTP。

生产团队会把这个 translator 包进 `AbstractToolset`（Pydantic AI）、`UniversalToolNode`（LangGraph）或 `BaseTool`（LlamaIndex）。Phase 13 · 17 会交付一个 gateway，在任意三家 provider 前暴露 OpenAI-shaped API。

## 实际使用

`code/main.py` 定义一个 canonical `Tool` dataclass，以及三个 translators，输出 OpenAI、Anthropic 和 Gemini declaration JSON。它随后把每种 shape 的手写 provider response 解析成相同 canonical call object，展示表皮下语义相同。运行它，并排 diff 三个 declarations。

重点看：

- 三个 declaration blocks 只在 envelope 和 field names 上不同。
- 三个 response blocks 的差异在 call 所在位置（top-level `tool_calls`、`content[]` block、`parts[]` entry）。
- 一个 `canonical_call()` 函数从三种 response shape 中抽取 `{id, name, args}`。

## 交付成果

本课产出 `outputs/skill-provider-portability-audit.md`。给定一个针对某个 provider 的 function-calling integration，该 skill 会生成 portability audit：它依赖哪些 provider limits、哪些字段需要重命名、迁移到其他 provider 时会坏掉什么。

## 练习

1. 运行 `code/main.py`，验证三种 provider declaration JSON 都序列化了同一个底层 `Tool` object。修改 canonical tool，加一个 enum parameter，并确认只有 Gemini translator 需要处理 OpenAPI quirk。

2. 为每个 provider 增加一个 `ListToolsResponse` parser，从模型在 `list_tools` 或 discovery call 后返回的内容中抽取 tool list。OpenAI 没有原生这种接口；记录这个 asymmetry。

3. 实现 `tool_choice` conversion：把 canonical `ToolChoice(mode="force", tool_name="x")` 映射成三种 provider shape。然后映射 `mode="any"` 和 `mode="none"`。检查本课 diff table。

4. 选择三家 provider 之一，从头到尾阅读其 function-calling guide。找到一个它的 schema spec 中其他两家不支持的字段。候选：OpenAI `strict`、Anthropic `disable_parallel_tool_use`、Gemini `function_calling_config.allowed_function_names`。

5. 写一个 test vector：tool call 的 arguments 违反声明 schema。把它通过每个 provider 的 validator（Lesson 01 的 stdlib validator 可做代理）并记录触发哪些 errors。记录你会在生产中为了 strictness 选择哪个 provider。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Function calling | “Tool use” | 用于结构化 tool-call emission 的 provider-level API |
| Tool declaration | “Tool spec” | Name + description + JSON Schema input payload |
| `tool_choice` | “Force / forbid” | Auto / required / none / specific-name modes |
| Strict mode | “Schema enforcement” | OpenAI flag，约束 decoding 以匹配 schema |
| `tool_use` block | “Anthropic's call shape” | 带 id、name、input 的 inline content block |
| `functionCall` part | “Gemini's call shape” | 包含 name、args 和 id 的 `parts[]` entry |
| Arguments-as-string | “Stringified JSON” | OpenAI 以 JSON string 而不是 object 返回 args |
| Parallel tool calls | “Fan-out in one turn” | 一个 assistant message 中的多个 tool calls |
| Refusal | “Model declines” | Strict-mode-only 的 refusal block，而不是 call |
| OpenAPI 3.0 subset | “Gemini schema quirk” | Gemini 使用类似 JSON Schema 的 dialect，存在小差异 |

## 延伸阅读

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — 包含 strict mode 和 parallel calls 的 canonical reference
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — `tool_use` 和 `tool_result` block semantics
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — parallel calls、unique ids 和 OpenAPI subset
- [Vertex AI — Function calling reference](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — Gemini 的 enterprise surface
- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — strict-mode schema enforcement details
