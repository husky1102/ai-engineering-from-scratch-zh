# Function Calling 与 Tool Use

> LLM 本身什么都做不了。它们生成文本。这就是全部能力。它们不能查天气、查询数据库、发送邮件、运行代码或读取文件。你见过的每个 "AI agent"，本质上都是一个 LLM 生成 JSON，说明要调用哪个函数 -- 然后你的代码真正去调用它。模型是大脑。工具是双手。Function calling 是连接二者的神经系统。

**类型：** Build
**语言：** Python
**先修：** Phase 11 Lesson 03 (Structured Outputs)
**时间：** ~75 minutes
**相关：** Phase 11 · 14 (Model Context Protocol) -- 当工具需要跨 host 共享时，从 inline function-calling 升级为 MCP server。本课覆盖 inline case；MCP 覆盖 protocol case。

## 学习目标

- 实现 function calling loop：定义 tool schemas、解析模型的 tool-call JSON、执行函数并返回结果
- 用清晰描述和 typed parameters 设计 tool schemas，让模型能够可靠调用
- 构建 multi-turn agent loop，串联多次 function call 来回答复杂查询
- 处理 function calling 边界情况：parallel tool calls、error propagation，以及防止无限 tool loop

## 要解决的问题

你构建了一个 chatbot。用户问："What's the weather in Tokyo right now?"

模型回答："I don't have access to real-time weather data, but based on the season, Tokyo is likely around 15 degrees Celsius..."

这是披着免责声明的幻觉。模型不知道天气。它永远也不会知道。天气每小时都变。模型的训练数据已经是几个月前的。

正确答案需要调用 OpenWeatherMap API，获取当前温度，并返回真实数字。模型不能调用 API。你的代码可以。缺失的一块是结构化协议：让模型说 "我需要用这些参数调用 weather API"，再让你的代码执行它并把结果喂回去。

这就是 function calling。模型输出结构化 JSON，描述要调用哪个函数、使用哪些参数。你的应用执行这个函数。结果回到对话中。模型使用该结果生成最终答案。

没有 function calling，LLM 是百科全书。有了它，LLM 才变成 agent。

## 核心概念

### Function Calling Loop

每次 tool-use 交互都遵循同一个 5 步 loop。

```mermaid
sequenceDiagram
    participant U as User
    participant A as Application
    participant M as Model
    participant T as Tool

    U->>A: "What's the weather in Tokyo?"
    A->>M: messages + tool definitions
    M->>A: tool_call: get_weather(city="Tokyo")
    A->>T: Execute get_weather("Tokyo")
    T->>A: {"temp": 18, "condition": "cloudy"}
    A->>M: tool_result + conversation
    M->>A: "It's 18C and cloudy in Tokyo."
    A->>U: Final response
```

Step 1：用户发送消息。Step 2：模型接收消息以及 tool definitions（描述可用函数的 JSON Schema）。Step 3：模型不直接用文本回答，而是输出一个 tool call -- 一个包含函数名和参数的结构化 JSON object。Step 4：你的代码执行函数并捕获结果。Step 5：结果回到模型，模型现在有真实数据来生成最终答案。

模型从不执行任何东西。它只决定调用什么以及使用什么参数。你的代码才是 executor。

### Tool Definitions：JSON Schema Contract

每个 tool 都由 JSON Schema 定义，它告诉模型该函数做什么、接收哪些参数、以及这些参数必须是什么类型。

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get current weather for a city. Returns temperature in Celsius and conditions.",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {
          "type": "string",
          "description": "City name, e.g. 'Tokyo' or 'San Francisco'"
        },
        "units": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"],
          "description": "Temperature units"
        }
      },
      "required": ["city"]
    }
  }
}
```

`description` 字段至关重要。模型会读取它们来决定何时以及如何使用工具。像 "gets weather" 这样含糊的描述，比 "Get current weather for a city. Returns temperature in Celsius and conditions." 更容易导致错误的工具选择。description 就是工具选择的 prompt。

### Provider Comparison

每个主流 provider 都支持 function calling，但 API surface 不同。

| Provider | API Parameter | Tool Call Format | Parallel Calls | Forced Calling |
|----------|--------------|-----------------|---------------|----------------|
| OpenAI (GPT-5, o4) | `tools` | `tool_calls[].function` | Yes (multiple per turn) | `tool_choice="required"` |
| Anthropic (Claude 4.6/4.7) | `tools` | `content[].type="tool_use"` | Yes (multiple blocks) | `tool_choice={"type":"any"}` |
| Google (Gemini 3) | `function_declarations` | `functionCall` | Yes | `function_calling_config` |
| Open-weight (Llama 4, Qwen3, DeepSeek-V3) | Llama 4 上原生 `tools`；其他模型上用 Hermes 或 ChatML | Mixed | 取决于模型 | Prompt-based 或支持时用 `tool_choice` |

到 2026 年，三家闭源 provider 已经收敛到几乎相同的基于 JSON-Schema 的格式。Llama 4 随模型提供原生 `tools` 字段，形状与 OpenAI 匹配。Open-weight fine-tunes 仍然各不相同 -- Hermes format（NousResearch）是第三方 fine-tunes 最常见的格式。对于跨 host 共享的工具，优先使用 MCP（Phase 11 · 14），而不是 inline function-calling -- server 对所有 host 都相同。

### Tool Choice：Auto、Required、Specific

你可以控制模型何时使用工具。

**Auto**（默认）：模型决定是调用工具还是直接回答。"What's 2+2?" -- 直接回答。"What's the weather?" -- 调用工具。

**Required**：模型必须至少调用一个工具。当你知道用户意图需要工具时使用。它能防止模型猜测，而不是查找真实数据。

**Specific function**：强制模型调用某个特定函数。`tool_choice={"type":"function", "function": {"name": "get_weather"}}` 保证无论查询是什么都会调用 weather tool。用于 routing -- 当上游逻辑已经确定需要哪个工具时。

### Parallel Function Calling

GPT-4o 和 Claude 可以在单轮中调用多个函数。用户问："What's the weather in Tokyo and New York?" 模型会同时输出两个 tool calls：

```json
[
  {"name": "get_weather", "arguments": {"city": "Tokyo"}},
  {"name": "get_weather", "arguments": {"city": "New York"}}
]
```

你的代码执行两者（最好并发），返回两个结果，模型综合成单个回答。这把 round trip 从 2 次降到 1 次。对于每个 query 需要 5-10 次 tool call 的 agent，parallel calling 可以把延迟降低 60-80%。

### Structured Outputs vs Function Calling

Lesson 03 覆盖了 structured outputs。Function calling 使用同一套 JSON Schema 机制，但目的不同。

**Structured outputs**：强制模型按特定形状生成数据。输出就是最终产物。例子：从文本中抽取产品信息为 `{name, price, in_stock}`。

**Function calling**：模型声明执行动作的意图。输出是中间步骤。例子：`get_weather(city="Tokyo")` -- 模型在请求一个动作，而不是生成最终答案。

当你想做数据抽取时，用 structured outputs。当你想让模型与外部系统交互时，用 function calling。

### Security：不可协商的规则

Function calling 是你能给 LLM 的最危险能力。模型会选择执行什么。如果你的 tool set 包含数据库查询，模型会构造查询。如果它包含 shell commands，模型会编写命令。

**Rule 1：永远不要把模型生成的 SQL 直接传给数据库。** 模型可能而且会生成 DROP TABLE、UNION injections，或返回每一行的查询。始终参数化。始终验证。始终使用操作 allowlist。

**Rule 2：Allowlist functions。** 模型只能调用你明确暴露的函数。永远不要构建通用的 "execute any function by name" 工具。如果你有 50 个内部函数，只暴露用户需要的 5 个。

**Rule 3：Validate arguments。** 模型可能传入城市名 `"; DROP TABLE users; --"`。执行前要按期望类型、范围和格式验证每个参数。

**Rule 4：Sanitize tool results。** 如果 tool 返回敏感数据（API keys、PII、internal errors），在送回模型前过滤掉。模型会把 tool results 原样包含进回复中。

**Rule 5：Rate limit tool calls。** 陷入 loop 的模型可能调用工具数百次。设置最大值（每次对话 10-20 次调用是合理的）。打断无限 loop。

### Error Handling

工具会失败。API 会 timeout。数据库会宕机。文件可能不存在。模型需要知道工具何时失败以及为什么失败。

把 errors 作为结构化 tool results 返回，而不是抛出 exception：

```json
{
  "error": true,
  "message": "City 'Toky' not found. Did you mean 'Tokyo'?",
  "code": "CITY_NOT_FOUND"
}
```

模型会读取它，调整参数并重试。模型很擅长从结构化 error messages 中自我修正。它们不擅长从空响应或泛泛的 "something went wrong" 中恢复。

### MCP：Model Context Protocol

MCP 是 Anthropic 的 tool interoperability 开放标准。它不让每个应用定义自己的工具，而是提供通用协议：工具由 MCP servers 提供，由 MCP clients（如 Claude Code、Cursor 或你的应用）消费。

一个 MCP server 可以把工具暴露给任何兼容 client。Postgres MCP server 让任何 MCP-compatible agent 获得数据库访问能力。GitHub MCP server 让任何 agent 获得仓库访问能力。工具定义一次，到处使用。

MCP 之于 function calling，就像 HTTP 之于 networking。它标准化 transport layer，让工具可移植。

## 动手实现

### Step 1: Define the Tool Registry

构建一个 registry，存储 tool definitions 和它们的 implementations。每个 tool 都有一个 JSON Schema definition（模型看到的内容）和一个 Python function（你的代码执行的内容）。

```python
import json
import math
import time
import hashlib


TOOL_REGISTRY = {}


def register_tool(name, description, parameters, function):
    TOOL_REGISTRY[name] = {
        "definition": {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        },
        "function": function,
    }
```

### Step 2: Implement 5 Tools

构建 calculator、weather lookup、web search simulator、file reader 和 code runner。

```python
def calculator(expression, precision=2):
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return {"error": True, "message": f"Invalid characters in expression: {expression}"}
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return {"result": round(float(result), precision), "expression": expression}
    except Exception as e:
        return {"error": True, "message": str(e)}


WEATHER_DB = {
    "tokyo": {"temp_c": 18, "condition": "cloudy", "humidity": 72, "wind_kph": 14},
    "new york": {"temp_c": 22, "condition": "sunny", "humidity": 45, "wind_kph": 8},
    "london": {"temp_c": 12, "condition": "rainy", "humidity": 88, "wind_kph": 22},
    "san francisco": {"temp_c": 16, "condition": "foggy", "humidity": 80, "wind_kph": 18},
    "sydney": {"temp_c": 25, "condition": "sunny", "humidity": 55, "wind_kph": 10},
}


def get_weather(city, units="celsius"):
    key = city.lower().strip()
    if key not in WEATHER_DB:
        suggestions = [c for c in WEATHER_DB if c.startswith(key[:3])]
        return {
            "error": True,
            "message": f"City '{city}' not found.",
            "suggestions": suggestions,
            "code": "CITY_NOT_FOUND",
        }
    data = WEATHER_DB[key].copy()
    if units == "fahrenheit":
        data["temp_f"] = round(data["temp_c"] * 9 / 5 + 32, 1)
        del data["temp_c"]
    data["city"] = city
    return data


SEARCH_DB = {
    "python function calling": [
        {"title": "OpenAI Function Calling Guide", "url": "https://platform.openai.com/docs/guides/function-calling", "snippet": "Learn how to connect LLMs to external tools."},
        {"title": "Anthropic Tool Use", "url": "https://docs.anthropic.com/en/docs/tool-use", "snippet": "Claude can interact with external tools and APIs."},
    ],
    "MCP protocol": [
        {"title": "Model Context Protocol", "url": "https://modelcontextprotocol.io", "snippet": "An open standard for connecting AI models to data sources."},
    ],
    "weather API": [
        {"title": "OpenWeatherMap API", "url": "https://openweathermap.org/api", "snippet": "Free weather API with current, forecast, and historical data."},
    ],
}


def web_search(query, max_results=3):
    key = query.lower().strip()
    for db_key, results in SEARCH_DB.items():
        if db_key in key or key in db_key:
            return {"query": query, "results": results[:max_results], "total": len(results)}
    return {"query": query, "results": [], "total": 0}


FILE_SYSTEM = {
    "data/config.json": '{"model": "gpt-4o", "temperature": 0.7, "max_tokens": 4096}',
    "data/users.csv": "name,email,role\nAlice,alice@example.com,admin\nBob,bob@example.com,user",
    "README.md": "# My Project\nA tool-use agent built from scratch.",
}


def read_file(path):
    if ".." in path or path.startswith("/"):
        return {"error": True, "message": "Path traversal not allowed.", "code": "FORBIDDEN"}
    if path not in FILE_SYSTEM:
        available = list(FILE_SYSTEM.keys())
        return {"error": True, "message": f"File '{path}' not found.", "available_files": available, "code": "NOT_FOUND"}
    content = FILE_SYSTEM[path]
    return {"path": path, "content": content, "size_bytes": len(content), "lines": content.count("\n") + 1}


def run_code(code, language="python"):
    if language != "python":
        return {"error": True, "message": f"Language '{language}' not supported. Only 'python' is available."}
    forbidden = ["import os", "import sys", "import subprocess", "exec(", "eval(", "__import__", "open("]
    for pattern in forbidden:
        if pattern in code:
            return {"error": True, "message": f"Forbidden operation: {pattern}", "code": "SECURITY_VIOLATION"}
    try:
        local_vars = {}
        exec(code, {"__builtins__": {"print": print, "range": range, "len": len, "str": str, "int": int, "float": float, "list": list, "dict": dict, "sum": sum, "min": min, "max": max, "abs": abs, "round": round, "sorted": sorted, "enumerate": enumerate, "zip": zip, "map": map, "filter": filter, "math": math}}, local_vars)
        result = local_vars.get("result", None)
        return {"success": True, "result": result, "variables": {k: str(v) for k, v in local_vars.items() if not k.startswith("_")}}
    except Exception as e:
        return {"error": True, "message": f"{type(e).__name__}: {e}"}
```

### Step 3: Register All Tools

```python
def register_all_tools():
    register_tool(
        "calculator", "Evaluate a mathematical expression. Supports +, -, *, /, parentheses, and decimals. Returns the numeric result.",
        {"type": "object", "properties": {"expression": {"type": "string", "description": "Math expression, e.g. '(10 + 5) * 3'"}, "precision": {"type": "integer", "description": "Decimal places in result", "default": 2}}, "required": ["expression"]},
        calculator,
    )
    register_tool(
        "get_weather", "Get current weather for a city. Returns temperature, condition, humidity, and wind speed.",
        {"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g. 'Tokyo' or 'San Francisco'"}, "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature units, defaults to celsius"}}, "required": ["city"]},
        get_weather,
    )
    register_tool(
        "web_search", "Search the web for information. Returns a list of results with title, URL, and snippet.",
        {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}, "max_results": {"type": "integer", "description": "Maximum results to return", "default": 3}}, "required": ["query"]},
        web_search,
    )
    register_tool(
        "read_file", "Read the contents of a file. Returns the file content, size, and line count.",
        {"type": "object", "properties": {"path": {"type": "string", "description": "Relative file path, e.g. 'data/config.json'"}}, "required": ["path"]},
        read_file,
    )
    register_tool(
        "run_code", "Execute Python code in a sandboxed environment. Set a 'result' variable to return output.",
        {"type": "object", "properties": {"code": {"type": "string", "description": "Python code to execute"}, "language": {"type": "string", "enum": ["python"], "description": "Programming language"}}, "required": ["code"]},
        run_code,
    )
```

### Step 4: Build the Function Calling Loop

这是核心引擎。它模拟模型决定调用哪个 tool，执行 tool，并把结果喂回去。

```python
def simulate_model_decision(user_message, tools, conversation_history):
    msg = user_message.lower()

    if any(word in msg for word in ["weather", "temperature", "forecast"]):
        cities = []
        for city in WEATHER_DB:
            if city in msg:
                cities.append(city)
        if not cities:
            for word in msg.split():
                if word.capitalize() in [c.title() for c in WEATHER_DB]:
                    cities.append(word)
        if not cities:
            cities = ["tokyo"]
        calls = []
        for city in cities:
            calls.append({"name": "get_weather", "arguments": {"city": city.title()}})
        return calls

    if any(word in msg for word in ["calculate", "compute", "math", "what is", "how much"]):
        for token in msg.split():
            if any(c in token for c in "+-*/"):
                return [{"name": "calculator", "arguments": {"expression": token}}]
        if "+" in msg or "-" in msg or "*" in msg or "/" in msg:
            expr = "".join(c for c in msg if c in "0123456789+-*/.() ")
            if expr.strip():
                return [{"name": "calculator", "arguments": {"expression": expr.strip()}}]
        return [{"name": "calculator", "arguments": {"expression": "0"}}]

    if any(word in msg for word in ["search", "find", "look up", "google"]):
        query = msg.replace("search for", "").replace("look up", "").replace("find", "").strip()
        return [{"name": "web_search", "arguments": {"query": query}}]

    if any(word in msg for word in ["read", "file", "open", "cat", "show"]):
        for path in FILE_SYSTEM:
            if path.split("/")[-1].split(".")[0] in msg:
                return [{"name": "read_file", "arguments": {"path": path}}]
        return [{"name": "read_file", "arguments": {"path": "README.md"}}]

    if any(word in msg for word in ["run", "execute", "code", "python"]):
        return [{"name": "run_code", "arguments": {"code": "result = 'Hello from the sandbox!'", "language": "python"}}]

    return []


def execute_tool_call(tool_call):
    name = tool_call["name"]
    args = tool_call["arguments"]

    if name not in TOOL_REGISTRY:
        return {"error": True, "message": f"Unknown tool: {name}", "code": "UNKNOWN_TOOL"}

    tool = TOOL_REGISTRY[name]
    func = tool["function"]
    start = time.time()

    try:
        result = func(**args)
    except TypeError as e:
        result = {"error": True, "message": f"Invalid arguments: {e}"}

    elapsed_ms = round((time.time() - start) * 1000, 2)
    return {"tool": name, "result": result, "execution_time_ms": elapsed_ms}


def run_function_calling_loop(user_message, max_iterations=5):
    conversation = [{"role": "user", "content": user_message}]
    tool_definitions = [t["definition"] for t in TOOL_REGISTRY.values()]
    all_tool_results = []

    for iteration in range(max_iterations):
        tool_calls = simulate_model_decision(user_message, tool_definitions, conversation)

        if not tool_calls:
            break

        results = []
        for call in tool_calls:
            result = execute_tool_call(call)
            results.append(result)

        conversation.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

        for result in results:
            conversation.append({"role": "tool", "content": json.dumps(result["result"]), "tool_name": result["tool"]})

        all_tool_results.extend(results)
        break

    return {"conversation": conversation, "tool_results": all_tool_results, "iterations": iteration + 1 if tool_calls else 0}
```

### Step 5: Argument Validation

构建一个 validator，在执行前根据 JSON Schema 检查 tool call arguments。

```python
def validate_tool_arguments(tool_name, arguments):
    if tool_name not in TOOL_REGISTRY:
        return [f"Unknown tool: {tool_name}"]

    schema = TOOL_REGISTRY[tool_name]["definition"]["function"]["parameters"]
    errors = []

    if not isinstance(arguments, dict):
        return [f"Arguments must be an object, got {type(arguments).__name__}"]

    for required_field in schema.get("required", []):
        if required_field not in arguments:
            errors.append(f"Missing required argument: {required_field}")

    properties = schema.get("properties", {})
    for arg_name, arg_value in arguments.items():
        if arg_name not in properties:
            errors.append(f"Unknown argument: {arg_name}")
            continue

        prop_schema = properties[arg_name]
        expected_type = prop_schema.get("type")

        type_checks = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}
        if expected_type in type_checks:
            if not isinstance(arg_value, type_checks[expected_type]):
                errors.append(f"Argument '{arg_name}': expected {expected_type}, got {type(arg_value).__name__}")

        if "enum" in prop_schema and arg_value not in prop_schema["enum"]:
            errors.append(f"Argument '{arg_name}': '{arg_value}' not in {prop_schema['enum']}")

    return errors
```

### Step 6: Run the Demo

```python
def run_demo():
    register_all_tools()

    print("=" * 60)
    print("  Function Calling & Tool Use Demo")
    print("=" * 60)

    print("\n--- Registered Tools ---")
    for name, tool in TOOL_REGISTRY.items():
        desc = tool["definition"]["function"]["description"][:60]
        params = list(tool["definition"]["function"]["parameters"].get("properties", {}).keys())
        print(f"  {name}: {desc}...")
        print(f"    params: {params}")

    print(f"\n--- Argument Validation ---")
    validation_tests = [
        ("get_weather", {"city": "Tokyo"}, "Valid call"),
        ("get_weather", {}, "Missing required arg"),
        ("get_weather", {"city": "Tokyo", "units": "kelvin"}, "Invalid enum value"),
        ("calculator", {"expression": 123}, "Wrong type (int for string)"),
        ("unknown_tool", {"x": 1}, "Unknown tool"),
    ]
    for tool_name, args, label in validation_tests:
        errors = validate_tool_arguments(tool_name, args)
        status = "VALID" if not errors else f"ERRORS: {errors}"
        print(f"  {label}: {status}")

    print(f"\n--- Tool Execution ---")
    direct_tests = [
        {"name": "calculator", "arguments": {"expression": "(10 + 5) * 3 / 2"}},
        {"name": "get_weather", "arguments": {"city": "Tokyo"}},
        {"name": "get_weather", "arguments": {"city": "Mars"}},
        {"name": "web_search", "arguments": {"query": "python function calling"}},
        {"name": "read_file", "arguments": {"path": "data/config.json"}},
        {"name": "read_file", "arguments": {"path": "../etc/passwd"}},
        {"name": "run_code", "arguments": {"code": "result = sum(range(1, 101))"}},
        {"name": "run_code", "arguments": {"code": "import os; os.system('rm -rf /')"}},
    ]
    for call in direct_tests:
        result = execute_tool_call(call)
        print(f"\n  {call['name']}({json.dumps(call['arguments'])})")
        print(f"    -> {json.dumps(result['result'], indent=None)[:100]}")
        print(f"    time: {result['execution_time_ms']}ms")

    print(f"\n--- Full Function Calling Loop ---")
    test_queries = [
        "What's the weather in Tokyo?",
        "Calculate (100 + 250) * 0.15",
        "Search for MCP protocol",
        "Read the config file",
        "Run some Python code",
        "Tell me a joke",
    ]
    for query in test_queries:
        print(f"\n  User: {query}")
        result = run_function_calling_loop(query)
        if result["tool_results"]:
            for tr in result["tool_results"]:
                print(f"    Tool: {tr['tool']} ({tr['execution_time_ms']}ms)")
                print(f"    Result: {json.dumps(tr['result'], indent=None)[:90]}")
        else:
            print(f"    [No tool called -- direct response]")
        print(f"    Iterations: {result['iterations']}")

    print(f"\n--- Parallel Tool Calls ---")
    multi_city_query = "What's the weather in tokyo and london?"
    print(f"  User: {multi_city_query}")
    result = run_function_calling_loop(multi_city_query)
    print(f"  Tool calls made: {len(result['tool_results'])}")
    for tr in result["tool_results"]:
        city = tr["result"].get("city", "unknown")
        temp = tr["result"].get("temp_c", "N/A")
        print(f"    {city}: {temp}C, {tr['result'].get('condition', 'N/A')}")

    print(f"\n--- Security Checks ---")
    security_tests = [
        ("read_file", {"path": "../../etc/passwd"}),
        ("run_code", {"code": "import subprocess; subprocess.run(['ls'])"}),
        ("calculator", {"expression": "__import__('os').system('ls')"}),
    ]
    for tool_name, args in security_tests:
        result = execute_tool_call({"name": tool_name, "arguments": args})
        blocked = result["result"].get("error", False)
        print(f"  {tool_name}({list(args.values())[0][:40]}): {'BLOCKED' if blocked else 'ALLOWED'}")
```

## 实际使用

### OpenAI Function Calling

```python
# from openai import OpenAI
#
# client = OpenAI()
#
# tools = [{
#     "type": "function",
#     "function": {
#         "name": "get_weather",
#         "description": "Get current weather for a city",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "city": {"type": "string"},
#                 "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
#             },
#             "required": ["city"]
#         }
#     }
# }]
#
# response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[{"role": "user", "content": "Weather in Tokyo?"}],
#     tools=tools,
#     tool_choice="auto",
# )
#
# tool_call = response.choices[0].message.tool_calls[0]
# args = json.loads(tool_call.function.arguments)
# result = get_weather(**args)
#
# final = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {"role": "user", "content": "Weather in Tokyo?"},
#         response.choices[0].message,
#         {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)},
#     ],
# )
# print(final.choices[0].message.content)
```

OpenAI 把 tool calls 作为 `response.choices[0].message.tool_calls` 返回。每个 call 都有一个 `id`，你在返回结果时必须包含它。模型用这个 ID 匹配结果和调用。GPT-4o 可以在单个 response 中返回多个 tool calls -- 遍历并执行全部即可。

### Anthropic Tool Use

```python
# import anthropic
#
# client = anthropic.Anthropic()
#
# response = client.messages.create(
#     model="claude-sonnet-4-20250514",
#     max_tokens=1024,
#     tools=[{
#         "name": "get_weather",
#         "description": "Get current weather for a city",
#         "input_schema": {
#             "type": "object",
#             "properties": {
#                 "city": {"type": "string"},
#                 "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
#             },
#             "required": ["city"]
#         }
#     }],
#     messages=[{"role": "user", "content": "Weather in Tokyo?"}],
# )
#
# tool_block = next(b for b in response.content if b.type == "tool_use")
# result = get_weather(**tool_block.input)
#
# final = client.messages.create(
#     model="claude-sonnet-4-20250514",
#     max_tokens=1024,
#     tools=[...],
#     messages=[
#         {"role": "user", "content": "Weather in Tokyo?"},
#         {"role": "assistant", "content": response.content},
#         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_block.id, "content": json.dumps(result)}]},
#     ],
# )
```

Anthropic 把 tool calls 作为 `type: "tool_use"` 的 content blocks 返回。tool result 放在带有 `type: "tool_result"` 的 user message 中。注意关键差异：Anthropic 用 `input_schema` 定义 tool parameters，而 OpenAI 用 `parameters`。

### MCP Integration

```python
# MCP servers expose tools over a standardized protocol.
# Any MCP-compatible client can discover and call these tools.
#
# Example: connecting to a Postgres MCP server
#
# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client
#
# server_params = StdioServerParameters(
#     command="npx",
#     args=["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"],
# )
#
# async with stdio_client(server_params) as (read, write):
#     async with ClientSession(read, write) as session:
#         await session.initialize()
#         tools = await session.list_tools()
#         result = await session.call_tool("query", {"sql": "SELECT count(*) FROM users"})
```

MCP 把工具实现与工具消费解耦。Postgres server 懂 SQL。GitHub server 懂 API。你的 agent 只需要发现并调用工具 -- 不需要为每个集成写 provider-specific code。

## 交付成果

本课产出 `outputs/prompt-tool-designer.md` -- 一个用于设计 tool definitions 的可复用 prompt template。给它一个工具目标描述，它会产出带 descriptions、types 和 constraints 的完整 JSON Schema definition。

它还产出 `outputs/skill-function-calling-patterns.md` -- 一个用于在生产环境实现 function calling 的决策框架，覆盖 tool design、error handling、security 和 provider-specific patterns。

## 练习

1. **添加第 6 个工具：database query。** 实现一个带内存表的 simulated SQL tool。该工具接收 table name 和 filter conditions（不是 raw SQL）。验证 table name 位于 allowlist 中，且 filter operators 限制为 `=`、`>`、`<`、`>=`、`<=`。以 JSON 返回匹配行。

2. **实现带 error feedback 的 retry。** 当 tool call 失败时（例如 city not found），把 error message 喂回模型决策函数，并让它修正参数。跟踪每次 call 需要多少次 retry。为每个 tool call 设置最多 3 次 retry。

3. **构建 multi-step agent。** 有些查询需要串联 tool calls："Read the config file and tell me what model is configured, then search the web for that model's pricing." 实现一个 loop，持续运行直到模型决定不再需要工具，并把累积结果传入每个 decision step。限制为 10 iterations，防止无限 loop。

4. **测量 tool selection accuracy。** 创建 30 个带 expected tool names 的测试查询。对全部 30 个运行你的 decision function，并测量它选择正确 tool 的比例。识别哪些查询最容易导致工具混淆。

5. **实现 tool call caching。** 如果同一个 tool 在 60 秒内以相同 arguments 被调用，返回 cached result 而不是重新执行。使用以 `(tool_name, frozenset(args.items()))` 为 key 的 dictionary。衡量一段包含 20 个 queries 的对话中的 cache hit rates。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Function calling | "Tool use" | 模型输出结构化 JSON，描述要用特定 arguments 调用的函数 -- 执行它的是你的代码，不是模型 |
| Tool definition | "Function schema" | 描述工具名称、目的、parameters 和 types 的 JSON Schema object -- 模型读取它来决定何时以及如何使用工具 |
| Tool choice | "Calling mode" | 控制模型是否必须调用工具（required）、可以调用工具（auto），或必须调用特定工具（named） |
| Parallel calling | "Multi-tool" | 模型在单轮中输出多个 tool calls，减少 round trips -- GPT-4o 和 Claude 都支持 |
| Tool result | "Function output" | 执行工具的返回值，会作为消息发送回模型，让它能在回复中使用真实数据 |
| Argument validation | "Input checking" | 执行工具前验证模型生成的 arguments 是否匹配期望 types、ranges 和 constraints |
| MCP | "Tool protocol" | Model Context Protocol -- Anthropic 的开放标准，通过 servers 暴露工具，任何兼容 client 都可发现并调用 |
| Agent loop | "ReAct loop" | model-decides-tool、code-executes-tool、result-feeds-back 的迭代循环，直到模型有足够信息回应 |
| Tool poisoning | "Prompt injection via tools" | tool results 中包含操纵模型行为的指令所形成的攻击 -- sanitize all tool outputs |
| Rate limiting | "Call budget" | 设置每次对话 tool calls 的最大数量，防止无限 loop 和失控 API 成本 |

## 延伸阅读

- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling) -- 使用 GPT-4o 进行 tool use 的权威参考，包含 parallel calls、forced calling 和 structured arguments
- [Anthropic Tool Use Guide](https://docs.anthropic.com/en/docs/tool-use) -- Claude 的 tool use 实现，包含 input_schema、multi-tool responses 和 tool_choice configuration
- [Model Context Protocol Specification](https://modelcontextprotocol.io) -- 跨 AI applications 的 tool interoperability 开放标准，采用 server/client architecture
- [Schick et al., 2023 -- "Toolformer: Language Models Can Teach Themselves to Use Tools"](https://arxiv.org/abs/2302.04761) -- 关于训练 LLM 判断何时以及如何调用外部工具的奠基论文
- [Patil et al., 2023 -- "Gorilla: Large Language Model Connected with Massive APIs"](https://arxiv.org/abs/2305.15334) -- 在 1,645 个 API 上 fine-tune LLM 以实现准确 API calls，并减少 hallucination
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) -- 实时 benchmark，比较 GPT-4o、Claude、Gemini 和 open models 的 function calling accuracy
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) -- Thought-Action-Observation loop，也就是每次 tool call 外层的 agent loop；本课结束处，正是 Phase 14 接手处。
- [Anthropic -- Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) -- 从单一 tool-use primitive 构建出的五种可组合模式（prompt chaining、routing、parallelization、orchestrator-workers、evaluator-optimizer）。
