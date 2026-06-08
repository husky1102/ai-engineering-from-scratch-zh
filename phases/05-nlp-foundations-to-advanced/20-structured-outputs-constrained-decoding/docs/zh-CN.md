# Structured Outputs 与 Constrained Decoding

> 向 LLM 要 JSON。大多数时候会得到 JSON。在生产中，“大多数”就是问题。Constrained decoding 通过在 sampling 前编辑 logits，把“大多数”变成“总是”。

**类型：** Build
**语言：** Python
**先修：** Phase 5 · 17 (Chatbots), Phase 5 · 19 (Subword Tokenization)
**时间：** ~60 minutes

## 要解决的问题

一个 classifier prompt LLM："Return one of {positive, negative, neutral}." 模型返回："The sentiment is positive — this review is overwhelmingly favorable because the customer explicitly states that they ..."。你的 parser 崩溃。你的 classifier F1 是 0.0。

Free-form generation 不是合同。它只是建议。Production system 需要合同。

2026 年有三层。

1. **Prompting。** 好好请求。"Return only the JSON object." 在 frontier models 上大约 80% 有效，在较小模型上更低。
2. **Native structured output APIs。** OpenAI `response_format`、Anthropic tool use、Gemini JSON mode。对支持的 schemas 可靠。Vendor-locked。
3. **Constrained decoding。** 在每个 generation step 修改 logits，让模型 *不能* 发出 invalid tokens。构造上 100% valid。适用于任何 local model。

本课会建立对三者的直觉，并说明什么时候该选哪一个。

## 核心概念

![Constrained decoding masking invalid tokens at each step](../assets/constrained-decoding.svg)

**Constrained decoding 如何工作。** 在每个 generation step，LLM 会在完整 vocabulary（约 100k tokens）上产生 logit vector。一个 *logit processor* 位于模型和 sampler 之间。它根据目标 grammar（JSON Schema、regex、context-free grammar）中的当前位置，计算哪些 tokens 是有效的，并把所有 invalid tokens 的 logits 设为 negative infinity。剩余 logits 上的 softmax 只会把 probability mass 放在 valid continuations 上。

2026 年的实现：

- **Outlines。** 将 JSON Schema 或 regex 编译成 finite-state machine。每个 token 都能做 O(1) valid-next-token lookup。基于 FSM，所以 recursive schemas 需要 flattening。
- **XGrammar / llguidance。** Context-free grammar engines。处理 recursive JSON Schema。几乎零 decoding overhead。OpenAI 在他们 2025 年的 structured output implementation 中提到了 llguidance。
- **vLLM guided decoding。** 通过 Outlines、XGrammar 或 lm-format-enforcer backends 内置 `guided_json`、`guided_regex`、`guided_choice`、`guided_grammar`。
- **Instructor。** Pydantic-based wrapper over any LLM。Validation failure 时 retries。Cross-provider，但不修改 logits；它依赖 retries + structured-output-aware prompts。

### The counterintuitive result

Constrained decoding 往往比 unconstrained generation *更快*。两个原因。第一，它缩小了 next-token search space。第二，聪明的实现会跳过 forced tokens 的 token generation（例如 `{"name": "` 这样的 scaffolding；每个 byte 都是确定的）。

### The pitfall that costs you

Field order 很重要。把 `answer` 放在 `reasoning` 前面，模型就会在思考前先承诺一个答案。JSON 是 valid 的。Answer 是错的。没有 validation 能抓住它。

```json
// BAD
{"answer": "yes", "reasoning": "because ..."}

// GOOD
{"reasoning": "... therefore ...", "answer": "yes"}
```

Schema field order 是逻辑，不是格式。

## 动手实现

### Step 1: regex-constrained generation from scratch

完整 standalone FSM implementation 见 `code/main.py`。30 行核心思路：

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM 跟踪我们目前满足了 grammar 的哪些部分。`valid_tokens(state, tokenizer)` 会计算哪些 vocabulary tokens 能推进 FSM，同时不离开 accepting path。

### Step 2: Outlines for JSON Schema

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

零 validation errors。永远如此。FSM 让 invalid output 不可达。

### Step 3: Instructor for provider-agnostic Pydantic

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

机制不同。Instructor 不触碰 logits。它把 schema 格式化进 prompt，解析输出，并在 validation failure 时重试（默认 3 次）。适用于任何 provider。Retries 会增加 latency 和 cost。Cross-provider portability 是卖点。

### Step 4: native vendor APIs

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

Server-side constrained decoding。对支持的 schemas，可靠性与 Outlines 相当。不需要 local model management。会把你锁定到该 vendor。

## Pitfalls

- **Recursive schemas。** Outlines 将 recursion flatten 到固定深度。Tree-structured outputs（nested comments、AST）需要 XGrammar 或 llguidance（基于 CFG）。
- **Huge enums。** 10,000-option enum 编译很慢或超时。切换到 retriever：先预测 top-k candidates，再约束到这些 candidates。
- **Grammar too strict。** 强制 `date: "YYYY-MM-DD"` regex 后，模型无法为缺失日期输出 `"unknown"`。模型会通过编造一个日期来补偿。允许 `null` 或 sentinel。
- **Premature commitment。** 见上面的 field-order pitfall。始终把 reasoning 放在前面。
- **Vendor JSON mode without schema。** 纯 JSON mode 只保证 valid JSON，不保证对你的 use case *有效*。始终提供完整 schema。

## 实际使用

2026 stack：

| Situation | Pick |
|-----------|------|
| OpenAI/Anthropic/Google model, simple schema | Native vendor structured output |
| Any provider, Pydantic workflow, can tolerate retries | Instructor |
| Local model, need 100% validity, flat schema | Outlines (FSM) |
| Local model, recursive schema | XGrammar or llguidance |
| Self-hosted inference server | vLLM guided decoding |
| Batch processing with retries acceptable | Instructor + cheapest model |

## 交付成果

保存为 `outputs/skill-structured-output-picker.md`：

```markdown
---
name: structured-output-picker
description: Choose a structured output approach, schema design, and validation plan.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

Given a use case (provider, latency budget, schema complexity, failure tolerance), output:

1. Mechanism. Native vendor structured output, Instructor retries, Outlines FSM, or XGrammar CFG. One-sentence reason.
2. Schema design. Field order (reasoning first, answer last), nullable fields for "unknown", enum vs regex, required fields.
3. Failure strategy. Max retries, fallback model, graceful `null` handling, out-of-distribution refusal.
4. Validation plan. Schema compliance rate (target 100%), semantic validity (LLM-judge), field-coverage rate, latency p50/p99.

Refuse any design that puts `answer` or `decision` before reasoning fields. Refuse to use bare JSON mode without a schema. Flag recursive schemas behind an FSM-only library.
```

## 练习

1. **Easy.** 不使用 constrained decoding，prompt 一个小型 open-weights model（例如 Llama-3.2-3B）输出 `Review(sentiment, confidence, evidence_span)`。在 100 条 reviews 上测量能 parse as valid JSON 的比例。
2. **Medium.** 在同一个 corpus 上使用 Outlines JSON mode。比较 compliance rate、latency 和 semantic accuracy。
3. **Hard.** 从零实现一个 regex-constrained decoder，用于电话号码（`\d{3}-\d{3}-\d{4}`）。在 1000 个 samples 上验证 0 invalid outputs。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Constrained decoding | 强制 valid output | 在每个 generation step mask invalid-token logits。 |
| Logit processor | 负责约束的东西 | 函数：`(logits, state) -> masked_logits`。 |
| FSM | Finite-state machine | 编译后的 grammar representation；O(1) valid-next-token lookup。 |
| CFG | Context-free grammar | 能处理 recursion 的 grammar；比 FSM 慢但表达力更强。 |
| Schema field order | 重要吗？ | 重要，first field 会提交；始终把 reasoning 放在 answer 前。 |
| Guided decoding | vLLM 对它的命名 | 同一概念，集成进 inference server。 |
| JSON mode | OpenAI 早期版本 | 保证 JSON syntax；不保证 schema match。 |

## 延伸阅读

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) — Outlines 论文。
- [XGrammar paper (2024)](https://arxiv.org/abs/2411.15100) — 快速 CFG-based constrained decoding。
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — inference server 集成。
- [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) — API reference + gotchas。
- [Instructor library](https://python.useinstructor.com/) — 跨 providers 的 Pydantic + retries。
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) — 对 6 个 constrained decoding frameworks 做 benchmark。
