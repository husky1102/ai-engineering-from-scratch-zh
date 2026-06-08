# Structured Output：JSON Schema、Pydantic、Zod、Constrained Decoding

> “好好请求模型返回 JSON”即使在 frontier models 上也会有 5 到 15% 的失败率。Structured outputs 用 constrained decoding 关闭这个缺口：模型被字面上阻止发出违反 schema 的 token。OpenAI strict mode、Anthropic schema-typed tool use、Gemini 的 `responseSchema`、Pydantic AI 的 `output_type`、Zod 的 `.parse` 是同一想法的五种 surface forms。本课构建 schema validator 和 strict-mode contract，学习者会在每个生产 extraction pipeline 中使用它们。

**类型:** Build
**语言:** Python（stdlib，JSON Schema 2020-12 subset）
**先修:** Phase 13 · 02（function calling deep dive）
**时间:** ~75 分钟

## 学习目标

- 使用正确 constraints（enum、min/max、required、pattern）为 extraction target 编写 JSON Schema 2020-12。
- 解释 strict mode 与 constrained decoding 为什么提供了不同于“生成后 validate”的保证。
- 区分三种 failure modes：parse error、schema violation、model refusal。
- 交付一个带 typed repair 和 typed refusal handling 的 extraction pipeline。

## 要解决的问题

一个读取 purchase-order email 的 agent 需要把自由文本转换为 `{customer, line_items, total_usd}`。有三种方法。

**方法一：prompt for JSON。** “Reply in JSON with fields customer, line_items, total_usd.” 在 frontier models 上 85 到 95% 时间有效。失败方式有六种：missing brace、trailing comma、wrong types、hallucinated fields、token limit 处截断、泄漏 prose 如“Here is your JSON:”。

**方法二：validate after generation。** 自由生成、parse、按 schema validate，失败则 retry。可靠但昂贵：每次 retry 都要付费，truncation bugs 每次 occurrence 都多花一个 turn。

**方法三：constrained decoding。** Provider 在 decode time 强制 schema。Invalid tokens 会从 sampling distribution 中被 mask 掉。输出保证可 parse，且保证可 validate。Failure 被压缩成一种模式：refusal（模型判断输入不适合 schema）。

每个 2026 frontier provider 都提供某种形式的第三种方法。

- **OpenAI.** `response_format: {type: "json_schema", strict: true}`，如果模型拒绝则 response 中有 `refusal`。
- **Anthropic.** 对 `tool_use` inputs 做 schema enforcement；`stop_reason: "refusal"` 不存在，但 `end_turn` 且无 tool call 是 signal。
- **Gemini.** Request level 的 `responseSchema`；2026 年 Gemini 为选定类型提供 token-level grammar constraints。
- **Pydantic AI.** `output_type=InvoiceModel` 会发出 typed 到 `InvoiceModel` 的 structured `RunResult`。
- **Zod（TypeScript）.** Runtime parser，用 Zod schema 验证 provider output；可与 OpenAI 的 `beta.chat.completions.parse` 搭配。

共同点：只声明一次 schema，端到端执行。

## 核心概念

### JSON Schema 2020-12：lingua franca

每个 provider 都接受 JSON Schema 2020-12。最常用 constructs：

- `type`：`object`、`array`、`string`、`number`、`integer`、`boolean`、`null` 之一。
- `properties`：field name 到 subschema 的 map。
- `required`：必须出现的 field names list。
- `enum`：允许值的 closed set。
- `minimum` / `maximum`（numbers），`minLength` / `maxLength` / `pattern`（strings）。
- `items`：应用到每个 array element 的 subschema。
- `additionalProperties`：`false` 禁止 extra fields（默认因 mode 而异）。

OpenAI strict mode 增加三条要求：每个 property 都必须列入 `required`，每处都要 `additionalProperties: false`，且无 unresolved `$ref`。如果违反，API 会在 request time 返回 400。

### Pydantic：Python binding

Pydantic v2 通过 `model_json_schema()` 从 dataclass-shaped models 生成 JSON Schema。Pydantic AI 包装了这个流程，所以你可以写：

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

agent framework 会在边缘把 schema 翻译成 OpenAI strict mode、Anthropic `input_schema` 或 Gemini `responseSchema`。模型输出会作为 typed `Invoice` instance 返回。Validation errors 会以带 typed error paths 的 `ValidationError` raise。

### Zod：TypeScript binding

Zod（`z.object({customer: z.string(), ...})`）是 TS 等价物。OpenAI 的 Node SDK 暴露 `zodResponseFormat(Invoice)`，它会翻译成 API 的 JSON Schema payload。

### Refusals

Strict mode 不能强迫模型回答。如果输入无法适配 schema（“email was a poem, not an invoice”），模型会发出包含原因的 `refusal` field。你的代码必须把这当成 first-class outcome，而不是 failure。Refusal 也可用作 safety signal：如果模型被要求从 protected-content email 中提取信用卡号，它会返回带 safety reason 的 refusal。

### 开放模型中的 constrained decoding

Open-weights 实现使用三种技术。

1. **Grammar-based decoding**（`outlines`、`guidance`、`lm-format-enforcer`）：从 schema 构建 deterministic finite automaton；每一步 mask 掉会违反 FSM 的 token logits。
2. **Logit masking with a JSON parser**：streaming JSON parser 与模型同步运行；每一步计算 valid-next-token set。
3. **Speculative decoding with a verifier**：廉价 draft model 提议 tokens，verifier 强制 schema。

商业 provider 在幕后选择其中一种。2026 年的 SOTA 对短 structured outputs 比 plain generation 更快，对长输出速度大致相同。

### 三种 failure modes

1. **Parse error.** 输出不是 valid JSON。Strict mode 下不可能发生。Non-strict providers 上仍会发生。
2. **Schema violation.** 输出能 parse，但违反 schema。Strict mode 下不可能发生。在 strict 外很常见。
3. **Refusal.** 模型拒绝。必须作为 typed outcome 处理。

### Retry strategy

当你处在 strict mode 之外（Anthropic tool use、non-strict OpenAI、older Gemini）时，恢复模式是：

```text
generate -> parse -> validate -> if fail, inject error and retry, max 3x
```

通常一次 retry 就够。三次 retry 能捕捉弱模型偶发问题。超过三次说明 schema 糟糕：模型对某些输入无法满足它，prompt 或 schema 需要修。

### Small-model support

Constrained decoding 适用于小模型。带 grammar enforcement 的 3B open model 在 structured tasks 上会优于 raw prompting 的 70B model。这是 structured outputs 对生产重要的主要原因：它把可靠性和模型大小解耦。

## 实际使用

`code/main.py` 用 stdlib 交付一个最小 JSON Schema 2020-12 validator（types、required、enum、min/max、pattern、items、additionalProperties）。它包装一个 `Invoice` schema，并让 fake LLM output 通过 validator，展示 parse error、schema violation 和 refusal paths。生产中可把 fake output 换成任何 provider 的真实 response。

重点看：

- Validator 返回带 path 和 message 的 typed `[ValidationError]` list。这正是你希望暴露给 retry prompt 的形状。
- Refusal branch 不 retry。它 log 并返回 typed refusal。Phase 14 · 09 把 refusals 作为 safety signal。
- `additionalProperties: false` check 会在 adversarial test input 上触发，展示 strict mode 为什么能关上 hallucinated fields 的门。

## 交付成果

本课产出 `outputs/skill-structured-output-designer.md`。给定一个 free-text extraction target（invoices、support tickets、resumes 等），该 skill 会产出 strict-mode-compatible 的 JSON Schema 2020-12 以及镜像它的 Pydantic model，并 stub 出 typed refusal 和 retry handling。

## 练习

1. 运行 `code/main.py`。添加第四个 test case，其中 `total_usd` 是负数。确认 validator 用 `minimum` constraint path 拒绝它。

2. 扩展 validator 支持带 discriminator 的 `oneOf`。常见情况：`line_item` 要么是 product，要么是 service，由 `kind` 标记。Strict mode 这里有微妙规则；检查 OpenAI structured outputs guide。

3. 把同一个 Invoice schema 写成 Pydantic BaseModel，并将 `model_json_schema()` 输出与你手写 schema 比较。找出 Pydantic 默认设置而手写版本省略的一个 field。

4. 测量 refusal rates。构造十个不应可提取的输入（song lyric、math proof、blank email），并用 strict mode 跑真实 provider。统计 refusals vs hallucinated outputs。这是 refusal-aware retries 的 ground truth。

5. 从头到尾阅读 OpenAI structured outputs guide。找出 strict mode 明确禁止、但 plain JSON Schema 允许的一个 construct。然后设计一个非必要使用该 forbidden construct 的 schema，并重构为 strict-compatible。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| JSON Schema 2020-12 | “The schema spec” | 每个现代 provider 使用的 IETF-draft schema dialect |
| Strict mode | “Guaranteed schema” | OpenAI flag，通过 constrained decoding 强制 schema |
| Constrained decoding | “Logit masking” | Decode-time enforcement，mask invalid next-tokens |
| Refusal | “Model declines” | 输入无法适配 schema 时的 typed outcome |
| Parse error | “Invalid JSON” | 输出无法作为 JSON parse；strict 下不可能 |
| Schema violation | “Wrong shape” | Parsed 但违反 types / required / enum / range |
| `additionalProperties: false` | “No extras allowed” | 禁止 unknown fields；OpenAI strict 中必需 |
| Pydantic BaseModel | “Typed output” | 发出并验证 JSON Schema 的 Python class |
| Zod schema | “TypeScript output type” | 用于 provider output validation 的 TS runtime schema |
| Grammar enforcement | “Open-weights constrained decode” | FSM-based logit masking，如 outlines / guidance |

## 延伸阅读

- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — strict mode、refusals 和 schema requirements
- [OpenAI — Introducing structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) — 2024 年 8 月 launch post，解释 decoding guarantee
- [Pydantic AI — Output](https://ai.pydantic.dev/output/) — typed output_type bindings，会序列化到各 provider
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — canonical spec
- [Microsoft — Structured outputs in Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) — enterprise deployment notes 和 strict-mode caveats
