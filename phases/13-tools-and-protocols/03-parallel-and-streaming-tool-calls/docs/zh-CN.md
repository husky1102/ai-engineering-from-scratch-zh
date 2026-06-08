# Parallel Tool Calls 与工具 Streaming

> 三个独立天气查询如果串行执行，就是三次 round trip。并行执行，总时间会坍缩到最慢的单个 call。每个 frontier provider 现在都能在一个 turn 中发出多个 tool calls。收益真实存在，但 plumbing 很微妙。本课走读两半：parallel fan-out 和 streamed-argument reassembly，重点强调 id-correlation trap。

**类型:** Build
**语言:** Python（stdlib，thread pool + streaming harness）
**先修:** Phase 13 · 02（function calling deep dive）
**时间:** ~75 分钟

## 学习目标

- 解释为什么存在 `parallel_tool_calls: true`，以及何时禁用它。
- 在 parallel fan-out 期间，把 streamed argument chunks 关联到正确的 tool-call id。
- 把 partial `arguments` strings 重新组装成完整 JSON，不要过早解析。
- 运行一个三城市天气 benchmark，演示 sequential vs parallel latency。

## 要解决的问题

没有 parallel calls 时，agent 回答“what is the weather in Bengaluru, Tokyo, and Zurich”会这样做：

```text
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> run executor, reply with result
LLM -> call get_weather(Tokyo)
host -> run executor, reply with result
LLM -> call get_weather(Zurich)
host -> run executor, reply with result
LLM -> final text answer
```

三次 LLM round trip，每次还要支付 executor latency。大约是理想 wall-clock time 的 4 倍。

有 parallel calls 时：

```text
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer
```

一次 LLM round trip。Executor time 是三者最大值，不是总和。OpenAI、Anthropic 和 Gemini 上的生产 benchmark 显示，fan-out workload 的 wall-clock 可减少 60 到 70%。

代价是 correlation complexity。当三个 call 乱序完成时，result 必须携带 matching `tool_call_id`，这样模型才能对齐。当 result streaming 时，你必须先把 partial argument fragments 组装成完整 JSON 再执行。Gemini 3 加入 unique ids，部分原因就是解决真实世界中两个同名 parallel call 无法区分的问题。

## 核心概念

### 启用 parallel

- **OpenAI.** `parallel_tool_calls: true` 默认开启。设为 `false` 强制 serial。
- **Anthropic.** 通过 `disable_parallel_tool_use: false` 启用 parallel（Claude 3.5 及以上默认）。设为 `true` 变 serial。
- **Gemini.** 总是 parallel-capable；`tool_config.function_calling_config.mode = "AUTO"` 让模型决定。

当 tools 有 ordering dependencies（`create_file` then `write_file`）、一个 call 的输出会影响另一个 call 的输入，或 rate limiter 无法承受 fan-out 时，禁用 parallel。

### Id correlation

模型发出的每个 call 都有 `id`。Host 返回的每个 result 都必须包含同一个 id。没有它，result 就有歧义。

- **OpenAI.** 每个 tool-role message 上的 `tool_call_id`。
- **Anthropic.** 每个 `tool_result` block 上的 `tool_use_id`。
- **Gemini.** 每个 `functionResponse` 上的 `id`（Gemini 3 及以上；Gemini 2 通过 name 匹配，同名 parallel call 会坏）。

### 并发运行 calls

Host 在自己的 thread、coroutine 或 remote worker 上运行每个 call 的 executor。最简单的 harness 使用 thread pool；生产使用 asyncio + `asyncio.gather` 或 structured concurrency。Completion order 不可预测，id 才是 identifier。

一个常见 bug：按 call-list order 而不是 completion order 回复 result。这通常也能工作，因为模型只关心 `tool_call_id`，但如果 result 丢失或重复，out-of-order submission 会让调试更困难。更推荐按 completion order 回复，并带显式 id。

### Streaming tool calls

模型 streaming 时，`arguments` 会分块到达。三条 parallel call 的三股 chunk stream 会在 wire 上交错。你需要每个 id 一个 accumulator。

Provider shape：

- **OpenAI.** 每个 chunk 是 `choices[0].delta.tool_calls[i].function.arguments`（partial string）。Chunk 携带 `index`（call list 中的位置）。你按 index 累积，在 id 首次出现时读取它，并在 `finish_reason = "tool_calls"` 时 parse JSON。
- **Anthropic.** Stream events 是 `message_start`，随后每个 block 一个 `content_block_start`，type 为 `tool_use`（包含 id、name、empty input）。`content_block_delta` events 携带 `input_json_delta` chunks。`content_block_stop` 关闭每个 block。
- **Gemini.** `streamFunctionCallArguments`（Gemini 3 及以上）用 `functionCallId` 发出 chunks，因此 call 可以干净交错。Gemini 3 之前，streaming 一次返回一个完整 call。

### Partial JSON 与 parse-early trap

在 `arguments` 完整前不能 parse。类似 `{"city": "Beng` 的 partial JSON 不是合法 JSON，会 raise。正确 gate 是 provider 的 end-of-call signal：OpenAI 的 `finish_reason = "tool_calls"`、Anthropic 的 `content_block_stop`，或 Gemini 的 stream-end event。只有到那时才尝试 `json.loads`。更 robust 的方法是使用 incremental JSON parser，它在结构完成时 yield events；OpenAI streaming guide 推荐这用于展示 live “thinking” indicator 的 UX。Brace-counting 作为 completeness test 不可靠（quoted strings 或 escaped content 中的 braces 会造成 false positives），只应作为 informal debug heuristic。

### Out-of-order completion

```text
call_A: fast API, returns first
call_B: slow API, returns second
call_C: median API, returns third
```

Host reply 仍必须引用 ids：

```text
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

对于 OpenAI 或 Anthropic，reply 中的顺序不影响正确性。Gemini 也接受任意顺序，只要 ids 匹配。

### Benchmark：sequential vs parallel

`code/main.py` 中的 harness 模拟三个 executor，latency 分别为 400、600、800 ms。Sequential 总计 1800 ms。Parallel 是 `max(400, 600, 800) = 800 ms`。差异是常数，不是比例，因此 tool count 越多，节省越大。

真实世界 caveat：parallel calls 会压 downstream APIs。对 rate-limited service 做 10-way fan-out 会失败。Phase 13 · 17 覆盖 gateway-level backpressure；retry semantics 计划放在未来 phase。

### Streaming fan-out wall-clock

如果模型自身 streaming，你可以在某个 call 的 arguments 完成时立刻开始执行，而不是等待全部 calls finalize。这是 OpenAI 文档提到但并非所有 SDK 都暴露的优化。本课 harness 正是这样做的：模拟 stream 一旦 yield 完整 argument object，host 就启动该 call。

## 实际使用

`code/main.py` 分成两半。第一半用 `concurrent.futures.ThreadPoolExecutor` 顺序和并行运行三个模拟 weather calls，并打印 wall-clock time。第二半 replay 一个 fake streaming response，也就是三条 parallel calls 的 `arguments` chunks 在同一 stream 上交错，并用 `StreamAccumulator` 按 id 重新组装。无 LLM、无网络，只有 reassembly logic。

重点看：

- Sequential timer 命中 1.8 秒。Parallel timer 在相同 fake latencies 下命中 0.8 秒。
- Accumulator 通过 per-id buffering 处理乱序 chunks，并只在每个 call 的 JSON 完整后 parse。
- Executor 在某个 id 的 arguments finalize 后立即启动，而不是等所有 streams 结束。

## 交付成果

本课产出 `outputs/skill-parallel-call-safety-check.md`。给定 tool registry，该 skill 会审计哪些 tools 可以安全 parallelize，哪些有 ordering dependencies，哪些会压垮 downstream rate limits，并返回带 per-tool `parallel_safe` flags 的修订 registry。

## 练习

1. 运行 `code/main.py` 并改变模拟 latencies。确认 parallel-to-sequential ratio 近似 `max/sum`（真实运行会因为 thread scheduling、serialization 和 harness overhead 略微偏离理想）。在什么 latency distribution 下 parallel 不再重要？

2. 扩展 accumulator 以处理“call was cancelled mid-stream”场景：丢弃其 buffer 并发出 `cancelled` event。哪个 provider 明确记录了这种情况？检查 Anthropic 的 `content_block_stop` semantics 和 OpenAI 的 `finish_reason: "length"` behavior。

3. 用 `asyncio.gather` 替换 thread pool。Benchmark 二者。只有 executor 做真实 I/O 时，async 才会因为更低 context-switch cost 获得小幅收益。

4. 选择两个不应该 parallelize 的 tools（例如 `create_file` then `write_file`）。给 registry 加一个 `ordering_dependency` graph，并基于该 graph gate parallel fan-out。这是 dependency-aware scheduling 的最小机械结构，未来 agent-engineering phase 会形式化它。

5. 阅读 OpenAI parallel-function-calling 部分和 Anthropic `disable_parallel_tool_use` docs。找出 Anthropic 建议禁用 parallelism 的一种真实 tool 类型。（提示：同一 resource 上的 consequential mutations。）

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Parallel tool calls | “Fan-out in one turn” | 模型在单个 assistant message 中发出多个 tool calls |
| `parallel_tool_calls` | “OpenAI's flag” | 启用或禁用 multi-call emission |
| `disable_parallel_tool_use` | “Anthropic's inverse” | Opt-out flag；默认启用 parallel |
| Tool call id | “Correlation handle” | Result message 必须回显的 per-call identifier |
| Accumulator | “Stream buffer” | 面向 partial `arguments` chunks 的 per-id string buffer |
| Out-of-order completion | “Fastest first” | Parallel calls 以不可预测顺序完成；ids 是粘合剂 |
| Dependency graph | “Ordering constraints” | 某些 tool 的输出会喂给其他 tool 的输入；不能 parallelize |
| Parse-early trap | “JSON.parse exploded” | 尝试解析不完整的 `arguments` string |
| `streamFunctionCallArguments` | “Gemini 3 feature” | 每个 call 带 unique id 的 streamed argument chunks |
| Completion-order reply | “Don't wait for all” | Results 到达即回复，并按 id keyed |

## 延伸阅读

- [OpenAI — Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — 默认行为和 opt-out flag
- [Anthropic — Tool use: implementing tool use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) — `disable_parallel_tool_use` 和 result batching
- [Google — Gemini function calling parallel section](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini 3 起的 id-correlated parallel calls
- [OpenAI — Streaming responses with tools](https://platform.openai.com/docs/api-reference/responses-streaming) — OpenAI streams 的 chunked argument reassembly
- [Anthropic — Streaming messages](https://docs.anthropic.com/en/api/messages-streaming) — 带 `input_json_delta` 的 `content_block_delta`
