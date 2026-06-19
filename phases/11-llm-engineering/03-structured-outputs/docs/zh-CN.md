# 结构化输出：JSON、Schema 校验与约束解码

> 你的 LLM 返回的是字符串。你的应用需要的是 JSON。这个差距导致崩溃的生产系统，比任何模型幻觉都多。Structured output 是自然语言与 typed data 之间的桥。做对了，你的 LLM 就会变成可靠 API。做错了，你就会在凌晨 3 点用 regex 解析 free-text。

**类型：** Build
**语言：** Python
**先修：** Phase 10, Lessons 01-05 (LLMs from Scratch)
**时间：** ~90 分钟
**相关：** Phase 5 · 20 (Structured Outputs & Constrained Decoding) 覆盖 decoder-level theory（FSM/CFG logit processors、Outlines、XGrammar）。本课聚焦生产 SDK surface（OpenAI `response_format`、Anthropic tool use、Instructor）——如果你想理解 API 之下发生了什么，请先读 Phase 5 · 20。

## 学习目标

- 使用 OpenAI 和 Anthropic API 参数实现 JSON-mode 与 schema-constrained outputs
- 构建一个 Pydantic validation layer，拒绝 malformed LLM outputs，并用 error feedback 重试
- 解释 constrained decoding 如何在 token level 强制 valid JSON，而不依赖 post-processing
- 设计 robust extraction prompts，把 unstructured text 可靠转换为 typed data structures

## 要解决的问题

你向 LLM 提问：“Extract the product name, price, and availability from this text.” 它回答：

```text
The product is the Sony WH-1000XM5 headphones, which cost $348.00 and are currently in stock.
```

这是完全正确的答案。对你的应用来说，它也完全没用。你的 inventory system 需要 `{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true}`。你需要一个带有特定 keys、特定 types、特定 value constraints 的 JSON object。你不需要一句话。

天真的解决方案：在 prompt 里加上 “Respond in JSON”。这在 90% 的情况下有效。剩下 10% 里，模型会把 JSON 包进 markdown code fences，或者加上 “Here's the JSON:” 这样的 preamble，或者因为提前关闭 bracket 而生成 syntactically invalid JSON。你的 JSON parser 崩溃。pipeline 断掉。你加上 try/except 和 retry loop。retry 有时会生成不同数据。现在你不仅有 parsing problem，还多了 consistency problem。

这不是 prompt engineering problem，而是 decoding problem。模型从左到右生成 tokens。每个位置上，它从 100K+ options 的 vocabulary 中选择最可能的下一个 token。任意给定位置上，这些 options 中的大多数都会生成 invalid JSON。如果模型刚输出 `{"price":`，下一个 token 必须是 digit、quote（用于 string）、`null`、`true`、`false` 或 negative sign。其他任何东西都会产生 invalid JSON。没有 constraints，模型可能选一个非常合理的英文单词，但在语法上是灾难性的错误。

## 核心概念

### The Structured Output Spectrum

structured output control 有四个层级，后一层比前一层更可靠。

```mermaid
graph LR
    subgraph Spectrum["Structured Output Spectrum"]
        direction LR
        A["Prompt-based\n'Return JSON'\n~90% valid"] --> B["JSON Mode\nGuaranteed valid JSON\nNo schema guarantee"]
        B --> C["Schema Mode\nJSON + matches schema\nGuaranteed compliance"]
        C --> D["Constrained Decoding\nToken-level enforcement\n100% compliance"]
    end

    style A fill:#1a1a2e,stroke:#ff6b6b,color:#fff
    style B fill:#1a1a2e,stroke:#ffa500,color:#fff
    style C fill:#1a1a2e,stroke:#51cf66,color:#fff
    style D fill:#1a1a2e,stroke:#0f3460,color:#fff
```

**Prompt-based**（“Respond in valid JSON”）：没有强制。模型通常会遵循，但有时不会。可靠性：~90%。failure mode：markdown fences、preamble text、truncated output、wrong structure。

**JSON mode**：API 保证输出是 valid JSON。OpenAI 的 `response_format: { type: "json_object" }` 会启用它。输出可以无错误 parse。但它不一定符合你的 expected schema——可能有 extra keys、wrong types、missing fields。

**Schema mode**：API 接收一个 JSON Schema，并保证输出匹配它。到 2026 年，每个主流 provider 都原生支持：OpenAI 的 `response_format: { type: "json_schema", json_schema: {...} }`（也可通过 `tool_choice="required"`）、Anthropic 带 `input_schema` 的 tool use，以及 Gemini 的 `response_schema` + `response_mime_type: "application/json"`。输出拥有你指定的 exact keys、types 和 constraints。

**Constrained decoding**：generation 期间，在每个 token 位置，decoder 会 mask 掉所有会产生 invalid output 的 tokens。如果 schema 要求 number，而模型准备输出 letter，该 token 的概率会被设为零。模型只能生成会通向 valid output 的 tokens。这就是 OpenAI 的 structured output mode 以及 Outlines、Guidance 等 libraries 在底层实现的东西。

### JSON Schema：契约语言

JSON Schema 是你告诉模型（或 validation layer）输出必须长什么样的方式。每个主流 structured output system 都使用它。

```json
{
  "type": "object",
  "properties": {
    "product": { "type": "string" },
    "price": { "type": "number", "minimum": 0 },
    "in_stock": { "type": "boolean" },
    "categories": {
      "type": "array",
      "items": { "type": "string" }
    }
  },
  "required": ["product", "price", "in_stock"]
}
```

这个 schema 表示：输出必须是一个 object，包含 string 类型的 `product`、非负 number 类型的 `price`、boolean 类型的 `in_stock`，以及可选的 string array `categories`。任何不匹配的输出都会被拒绝。

Schemas 处理困难场景：nested objects、带 typed items 的 arrays、enums（把 string 约束到特定值）、pattern matching（strings 上的 regex），以及 combinators（oneOf、anyOf、allOf，用于 polymorphic outputs）。

### The Pydantic Pattern

在 Python 中，你不会手写 JSON Schema。你定义 Pydantic model，让它为你生成 schema。

```python
from pydantic import BaseModel

class Product(BaseModel):
    product: str
    price: float
    in_stock: bool
    categories: list[str] = []
```

这会生成与上面相同的 JSON Schema。Instructor library（以及 OpenAI 的 SDK）可以直接接受 Pydantic models：传入 model class，拿回 validated instance。如果 LLM output 不匹配，Instructor 会自动 retry。

### Function Calling / Tool Use

同一问题的另一种接口。你不直接要求模型生成 JSON，而是定义带 typed parameters 的 “tools”（functions）。模型输出一个带 structured arguments 的 function call。OpenAI 称之为 “function calling”。Anthropic 称之为 “tool use”。结果相同：structured data。

```mermaid
graph TD
    subgraph ToolUse["Tool Use Flow"]
        U["User: Extract product info\nfrom this review text"] --> M["Model processes input"]
        M --> TC["Tool Call:\nextract_product(\n  product='Sony WH-1000XM5',\n  price=348.00,\n  in_stock=true\n)"]
        TC --> V["Validate against\nfunction schema"]
        V --> R["Structured Result:\n{product, price, in_stock}"]
    end

    style U fill:#1a1a2e,stroke:#0f3460,color:#fff
    style TC fill:#1a1a2e,stroke:#e94560,color:#fff
    style V fill:#1a1a2e,stroke:#ffa500,color:#fff
    style R fill:#1a1a2e,stroke:#51cf66,color:#fff
```

当模型需要选择调用哪个 function，而不只是填参数时，应优先使用 tool use。如果你有 10 种 extraction schemas，模型必须根据输入选择正确的一个，tool use 同时给你 schema selection 和 structured output。

### Common Failure Modes

即使有 schema enforcement，structured outputs 仍可能以微妙方式失败。

**Hallucinated values**：输出匹配 schema，但包含编造数据。文本说 $348，模型却生成 `{"price": 299.99}`。Schema validation 抓不到这个问题——type 正确，value 错误。

**Enum confusion**：你把字段约束为 `["in_stock", "out_of_stock", "preorder"]`。模型输出 `"available"`——语义正确，但不在允许集合里。好的 constrained decoding 会阻止它。prompt-based approaches 不会。

**Nested object depth**：深度嵌套 schemas（4+ levels）会产生更多错误。每一层嵌套都是模型可能丢失结构跟踪的位置。

**Array length**：模型可能在 array 中生成过多或过少 items。Schemas 支持 `minItems` 和 `maxItems`，但并非所有 providers 都在 decoding level 强制执行。

**Optional field omission**：模型会省略 technically optional 但对你的用例 semantically important 的字段。即使数据有时缺失，也应把它们设为 required——强制模型显式生成 `null`。

## 动手实现

### Step 1: JSON Schema Validator

从零构建 validator，检查 Python object 是否匹配 JSON Schema。这就是输出侧用于验证 compliance 的逻辑。

```python
import json

def validate_schema(data, schema):
    errors = []
    _validate(data, schema, "", errors)
    return errors

def _validate(data, schema, path, errors):
    schema_type = schema.get("type")

    if schema_type == "object":
        if not isinstance(data, dict):
            errors.append(f"{path}: expected object, got {type(data).__name__}")
            return
        for key in schema.get("required", []):
            if key not in data:
                errors.append(f"{path}.{key}: required field missing")
        properties = schema.get("properties", {})
        for key, value in data.items():
            if key in properties:
                _validate(value, properties[key], f"{path}.{key}", errors)

    elif schema_type == "array":
        if not isinstance(data, list):
            errors.append(f"{path}: expected array, got {type(data).__name__}")
            return
        min_items = schema.get("minItems", 0)
        max_items = schema.get("maxItems", float("inf"))
        if len(data) < min_items:
            errors.append(f"{path}: array has {len(data)} items, minimum is {min_items}")
        if len(data) > max_items:
            errors.append(f"{path}: array has {len(data)} items, maximum is {max_items}")
        items_schema = schema.get("items", {})
        for i, item in enumerate(data):
            _validate(item, items_schema, f"{path}[{i}]", errors)

    elif schema_type == "string":
        if not isinstance(data, str):
            errors.append(f"{path}: expected string, got {type(data).__name__}")
            return
        enum_values = schema.get("enum")
        if enum_values and data not in enum_values:
            errors.append(f"{path}: '{data}' not in allowed values {enum_values}")

    elif schema_type == "number":
        if not isinstance(data, (int, float)):
            errors.append(f"{path}: expected number, got {type(data).__name__}")
            return
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and data < minimum:
            errors.append(f"{path}: {data} is less than minimum {minimum}")
        if maximum is not None and data > maximum:
            errors.append(f"{path}: {data} is greater than maximum {maximum}")

    elif schema_type == "boolean":
        if not isinstance(data, bool):
            errors.append(f"{path}: expected boolean, got {type(data).__name__}")

    elif schema_type == "integer":
        if not isinstance(data, int) or isinstance(data, bool):
            errors.append(f"{path}: expected integer, got {type(data).__name__}")
```

### Step 2: Pydantic-Style Model to Schema

构建一个最小 class-to-schema converter。定义 Python class，并自动生成它的 JSON Schema。

```python
class SchemaField:
    def __init__(self, field_type, required=True, default=None, enum=None, minimum=None, maximum=None):
        self.field_type = field_type
        self.required = required
        self.default = default
        self.enum = enum
        self.minimum = minimum
        self.maximum = maximum

def python_type_to_schema(field):
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }

    schema = {}

    if field.field_type in type_map:
        schema["type"] = type_map[field.field_type]
    elif field.field_type == list:
        schema["type"] = "array"
        schema["items"] = {"type": "string"}
    elif isinstance(field.field_type, dict):
        schema = field.field_type

    if field.enum:
        schema["enum"] = field.enum
    if field.minimum is not None:
        schema["minimum"] = field.minimum
    if field.maximum is not None:
        schema["maximum"] = field.maximum

    return schema

def model_to_schema(name, fields):
    properties = {}
    required = []

    for field_name, field in fields.items():
        properties[field_name] = python_type_to_schema(field)
        if field.required:
            required.append(field_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }
```

### Step 3: Constrained Token Filter

模拟 constrained decoding。给定 partial JSON string 和 schema，判断当前位置哪些 token categories 是合法的。

```python
def next_valid_tokens(partial_json, schema):
    stripped = partial_json.strip()

    if not stripped:
        return ["{"]

    try:
        json.loads(stripped)
        return ["<EOS>"]
    except json.JSONDecodeError:
        pass

    last_char = stripped[-1] if stripped else ""

    if last_char == "{":
        return ['"', "}"]
    elif last_char == '"':
        if stripped.endswith('":'):
            return ['"', "0-9", "true", "false", "null", "[", "{"]
        return ["a-z", '"']
    elif last_char == ":":
        return [" ", '"', "0-9", "true", "false", "null", "[", "{"]
    elif last_char == ",":
        return [" ", '"', "{", "["]
    elif last_char in "0123456789":
        return ["0-9", ".", ",", "}", "]"]
    elif last_char == "}":
        return [",", "}", "]", "<EOS>"]
    elif last_char == "]":
        return [",", "}", "<EOS>"]
    elif last_char == "[":
        return ['"', "0-9", "true", "false", "null", "{", "[", "]"]
    else:
        return ["any"]

def demonstrate_constrained_decoding():
    partial_states = [
        '',
        '{',
        '{"product"',
        '{"product":',
        '{"product": "Sony"',
        '{"product": "Sony",',
        '{"product": "Sony", "price":',
        '{"product": "Sony", "price": 348',
        '{"product": "Sony", "price": 348}',
    ]

    print(f"{'Partial JSON':<45} {'Valid Next Tokens'}")
    print("-" * 80)
    for state in partial_states:
        valid = next_valid_tokens(state, {})
        display = state if state else "(empty)"
        print(f"{display:<45} {valid}")
```

### Step 4: Extraction Pipeline

把所有组件组合成 extraction pipeline：定义 schema，模拟 LLM 生成 structured output，验证输出，并处理 retries。

```python
def simulate_llm_extraction(text, schema, attempt=0):
    if "headphones" in text.lower() or "sony" in text.lower():
        if attempt == 0:
            return '{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true, "categories": ["audio", "headphones"]}'
        return '{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true}'

    if "laptop" in text.lower():
        return '{"product": "MacBook Pro 16", "price": 2499.00, "in_stock": false, "categories": ["computers"]}'

    return '{"product": "Unknown", "price": 0, "in_stock": false}'

def extract_with_retry(text, schema, max_retries=3):
    for attempt in range(max_retries):
        raw = simulate_llm_extraction(text, schema, attempt)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  Attempt {attempt + 1}: JSON parse error -- {e}")
            continue

        errors = validate_schema(data, schema)
        if not errors:
            return data

        print(f"  Attempt {attempt + 1}: Schema validation errors -- {errors}")

    return None

product_schema = {
    "type": "object",
    "properties": {
        "product": {"type": "string"},
        "price": {"type": "number", "minimum": 0},
        "in_stock": {"type": "boolean"},
        "categories": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["product", "price", "in_stock"],
}
```

### Step 5: Run the Full Pipeline

```python
def run_demo():
    print("=" * 60)
    print("  Structured Output Pipeline Demo")
    print("=" * 60)

    print("\n--- Schema Definition ---")
    product_fields = {
        "product": SchemaField(str),
        "price": SchemaField(float, minimum=0),
        "in_stock": SchemaField(bool),
        "categories": SchemaField(list, required=False),
    }
    generated_schema = model_to_schema("Product", product_fields)
    print(json.dumps(generated_schema, indent=2))

    print("\n--- Schema Validation ---")
    test_cases = [
        ({"product": "Test", "price": 10.0, "in_stock": True}, "Valid object"),
        ({"product": "Test", "price": -5.0, "in_stock": True}, "Negative price"),
        ({"product": "Test", "in_stock": True}, "Missing price"),
        ({"product": "Test", "price": "ten", "in_stock": True}, "String as price"),
        ("not an object", "String instead of object"),
    ]

    for data, label in test_cases:
        errors = validate_schema(data, product_schema)
        status = "PASS" if not errors else f"FAIL: {errors}"
        print(f"  {label}: {status}")

    print("\n--- Constrained Decoding Simulation ---")
    demonstrate_constrained_decoding()

    print("\n--- Extraction Pipeline ---")
    texts = [
        "The Sony WH-1000XM5 headphones are priced at $348 and currently available.",
        "The new MacBook Pro 16-inch laptop costs $2499 but is sold out.",
        "This is a random sentence with no product info.",
    ]

    for text in texts:
        print(f"\n  Input: {text[:60]}...")
        result = extract_with_retry(text, product_schema)
        if result:
            print(f"  Output: {json.dumps(result)}")
        else:
            print(f"  Output: FAILED after retries")
```

## 实际使用

### OpenAI Structured Outputs

```python
# from openai import OpenAI
# from pydantic import BaseModel
#
# client = OpenAI()
#
# class Product(BaseModel):
#     product: str
#     price: float
#     in_stock: bool
#
# response = client.beta.chat.completions.parse(
#     model="gpt-5-mini",
#     messages=[
#         {"role": "system", "content": "Extract product information."},
#         {"role": "user", "content": "Sony WH-1000XM5, $348, in stock"},
#     ],
#     response_format=Product,
# )
#
# product = response.choices[0].message.parsed
# print(product.product, product.price, product.in_stock)
```

OpenAI 的 structured output mode 在内部使用 constrained decoding。模型生成的每个 token 都保证能产生匹配 Pydantic schema 的输出。不需要 retries。不需要 validation。约束被烘焙进 decoding process。

### Anthropic Tool Use

```python
# import anthropic
#
# client = anthropic.Anthropic()
#
# response = client.messages.create(
#     model="claude-opus-4-7",
#     max_tokens=1024,
#     tools=[{
#         "name": "extract_product",
#         "description": "Extract product information from text",
#         "input_schema": {
#             "type": "object",
#             "properties": {
#                 "product": {"type": "string"},
#                 "price": {"type": "number"},
#                 "in_stock": {"type": "boolean"},
#             },
#             "required": ["product", "price", "in_stock"],
#         },
#     }],
#     messages=[{"role": "user", "content": "Extract: Sony WH-1000XM5, $348, in stock"}],
# )
```

Anthropic 通过 tool use 实现 structured output。模型发出一个 tool call，带有匹配 input_schema 的 structured arguments。同样结果，不同 API surface。

### Instructor Library

```python
# pip install instructor
# import instructor
# from openai import OpenAI
# from pydantic import BaseModel
#
# client = instructor.from_openai(OpenAI())
#
# class Product(BaseModel):
#     product: str
#     price: float
#     in_stock: bool
#
# product = client.chat.completions.create(
#     model="gpt-5-mini",
#     response_model=Product,
#     messages=[{"role": "user", "content": "Sony WH-1000XM5, $348, in stock"}],
# )
```

Instructor 包装任意 LLM client，并添加 validation 自动 retries。如果第一次尝试未通过 validation，它会把 errors 作为 context 发回模型，并要求模型修复输出。这适用于任何 provider，不只是 OpenAI。

## 交付成果

本课产出 `outputs/prompt-structured-extractor.md`——一个可复用 prompt template，给定 schema definition 后能从任意文本中提取 structured data。输入 JSON Schema 和 unstructured text，它返回 validated JSON。

它还产出 `outputs/skill-structured-outputs.md`——一个决策框架，根据 provider、reliability requirements 和 schema complexity 选择正确的 structured output strategy。

## 练习

1. 扩展 schema validator，支持 `oneOf`（data 必须恰好匹配多个 schemas 中的一个）。这会处理 polymorphic outputs——例如某个字段既可以是 `Product` object，也可以是形状不同的 `Service` object。

2. 构建一个 “schema diff” 工具，比较两个 schemas，并识别 breaking changes（removed required fields、changed types）与 non-breaking changes（added optional fields、relaxed constraints）。这对生产环境中的 extraction schemas versioning 至关重要。

3. 实现更真实的 constrained decoding simulator。给定 JSON Schema 和 100 tokens 的 vocabulary（letters、digits、punctuation、keywords），逐步遍历 generation，在每个位置 mask invalid tokens。测量每一步中 vocabulary 的有效比例。

4. 构建 extraction eval suite。创建 50 条 product descriptions，并配上人工标注 JSON outputs。在全部 50 条上运行 extraction pipeline，测量 exact match、field-level accuracy 和 type compliance。找出哪些字段最难正确提取。

5. 为 extraction pipeline 添加 “confidence scores”。对每个 extracted field，估计模型置信度（基于 token probabilities，或通过运行 extraction 3 次并测量一致性）。把低置信字段标记给 human review。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| JSON mode | “Returns JSON” | API flag，保证 syntactically valid JSON output，但不强制任何具体 schema |
| Structured output | “Typed JSON” | 匹配具体 JSON Schema 的输出，拥有正确 keys、types 和 constraints |
| Constrained decoding | “Guided generation” | 在每个 token position，mask 掉会产生 invalid output 的 tokens——保证 100% schema compliance |
| JSON Schema | “A JSON template” | 用于描述 JSON data 的 structure、types 和 constraints 的 declarative language（OpenAPI、JSON Forms 等都使用它） |
| Pydantic | “Python dataclasses+” | 定义带 type validation 的 data models 的 Python library，FastAPI 和 Instructor 用它生成 JSON Schemas |
| Function calling | “Tool use” | LLM 输出 structured function invocation（name + typed arguments）而不是 free text——OpenAI 和 Anthropic 都支持 |
| Instructor | “Pydantic for LLMs” | 包装 LLM clients 以返回 validated Pydantic instances 的 Python library，并在 validation failure 时自动 retry |
| Token masking | “Filtering the vocabulary” | generation 期间把特定 token probabilities 设为零，让模型无法生成它们 |
| Schema compliance | “Matches the shape” | 输出拥有每个 required field、正确 types、constraints 范围内的 values，并且没有额外 disallowed fields |
| Retry loop | “Try again until it works” | 把 validation errors 发回模型并要求它修复输出——Instructor 会自动执行，直到 configurable max |

## 延伸阅读

- [OpenAI Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs)——OpenAI API 中基于 JSON Schema 的 constrained decoding 官方文档
- [Willard & Louf, 2023 -- "Efficient Guided Generation for Large Language Models"](https://arxiv.org/abs/2307.09702)——Outlines 论文，描述如何把 JSON Schemas 编译为 finite state machines，用于 token-level constraints
- [Instructor documentation](https://python.useinstructor.com/)——使用 Pydantic validation 和 retries 从任意 LLM 获取 structured outputs 的标准库
- [Anthropic Tool Use Guide](https://docs.anthropic.com/en/docs/tool-use)——Claude 如何通过带 JSON Schema `input_schema` 的 tool use 实现 structured output
- [JSON Schema specification](https://json-schema.org/)——每个主流 structured output system 使用的 schema language 完整规范
- [Outlines library](https://github.com/outlines-dev/outlines)——开源 constrained generation，把 regex 和 JSON Schema 编译为 finite state machines
- [Dong et al., "XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models" (MLSys 2025)](https://arxiv.org/abs/2411.15100)——当前 state-of-the-art grammar engine；pushdown-automaton compilation，以 ~100 ns / token 的速度 mask tokens
- [Beurer-Kellner et al., "Prompting Is Programming: A Query Language for Large Language Models" (LMQL)](https://arxiv.org/abs/2212.06094)——LMQL 论文，把 constrained decoding 表述为带 type 和 value constraints 的 query language
- [Microsoft Guidance (framework docs)](https://github.com/guidance-ai/guidance)——template-driven constrained generation；作为 Outlines 和 XGrammar 的 vendor-agnostic 补充
