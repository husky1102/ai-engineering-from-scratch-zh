# 工具使用和函数调用

> Toolformer（Schick et al., 2023）开启了 self-supervised tool annotation。Berkeley Function Calling Leaderboard V4（Patil et al., 2025）设定了 2026 年的门槛：40% agentic、30% multi-turn、10% live、10% non-live、10% hallucination。Single-turn 已经基本解决。Memory、dynamic decision-making 和 long-horizon tool chains 还没有。

**类型：** 构建
**语言：** Python (stdlib)
**先修：** Phase 14 · 01 (Agent Loop), Phase 13 · 01 (Function Calling Deep Dive)
**时间：** ~60 分钟

## 学习目标

- 解释 Toolformer 的 self-supervised training signal：只有当执行结果降低 next-token loss 时，才保留 tool annotation。
- 说出 BFCL V4 的五个 evaluation categories，以及每一类衡量什么。
- 用 stdlib 实现带 schema validation、argument coercion 和 execution sandboxing 的 tool registry。
- 诊断 2026 年的三个开放问题：long-horizon tool chaining、dynamic decision-making 和 memory。

## 要解决的问题

早期 tool use 问的是：模型能否预测一个正确的 function call？现代 tool use 问的是：模型能否跨 40 个步骤链式调用工具，带 memory，带 partial observability，能从 tool failures 中恢复，并且不 hallucinate 根本不存在的工具？

Toolformer 建立了基线：模型可以通过 self-supervision 学会何时调用工具。BFCL V4 定义了 2026 年的评估目标。两者之间的差距，就是 production agents 生存的空间。

## 核心概念

### Toolformer（Schick et al., NeurIPS 2023）

想法：让模型用候选 API calls 标注自己的 pretraining corpus。对每个候选调用，执行它。只有当包含 tool result 会降低 surrounding text 的 next-token loss 时，才保留 annotation。然后在过滤后的 corpus 上 fine-tune。

覆盖的工具：calculator、QA system、search engines、translator、calendar。self-supervision signal 只关心工具是否帮助预测文本，不需要 human labels。

规模结果：tool use 会在规模上涌现。较小模型会被 tool annotations 伤害；较大模型会受益。这就是为什么 2026 年 frontier models 已经内置很强的 tool use，而多数 7B models 仍需要显式的 tool-use fine-tuning 才可靠。

### Berkeley Function Calling Leaderboard V4（Patil et al., ICML 2025）

BFCL 是 2026 年事实上的评估。V4 组成：

- **Agentic (40%)** — 完整 agent trajectories：memory、multi-turn、dynamic decisions。
- **Multi-Turn (30%)** — 带 tool chains 的交互式 conversations。
- **Live (10%)** — 用户提交的真实 prompts（更难的 distribution）。
- **Non-Live (10%)** — synthetic test cases。
- **Hallucination (10%)** — 检测什么时候不应该调用工具。

V3 引入了 state-based evaluation：在一串 tool sequence 之后，检查 API 的实际状态（例如“文件是否创建成功？”），而不是匹配 tool calls 的 AST。V4 增加了 web search、memory 和 format sensitivity categories。

2026 年关键发现：single-turn function calling 已经接近解决。失败集中在 memory（跨 turns 携带 context）、dynamic decision-making（基于先前结果选择工具）、long-horizon chains（20+ steps 后 drift）和 hallucination detection（没有合适工具时拒绝调用）。

### Tool schema

每个 provider 都有 schema。细节不同，但形状相同：

```text
name: string
description: string (what it does, when to use it)
input_schema: JSON Schema (properties, required, types, enums)
```

Anthropic 直接使用 `input_schema`。OpenAI 使用 `function.parameters`。二者都接受 JSON Schema。Descriptions 是承重字段，模型会读它们来选择正确工具。糟糕的 tool descriptions 是 wrong-tool-picked failures 的 #1 root cause。

### Argument validation

不要信任任何 tool call。要验证：

1. **Type coercion.** 模型可能在 schema 要 int 时返回字符串 `"5"`。如果无歧义就 coerce；否则 reject。
2. **Enum validation.** 如果 schema 说 `status in {"open", "closed"}`，但模型发出 `"in_progress"`，就带描述性错误 reject。
3. **Required fields.** 缺少 required field -> 立即把 error observation 返回给模型，而不是 crash。
4. **Format validation.** Dates、emails、URLs — 用具体 parsers 验证，而不是 regex。

每次 validation failure 都应该返回 structured observation，让模型可以用正确形状重试。

### Parallel tool calls

现代 providers 支持在一个 assistant turn 中并行 tool calls。循环如下：

1. 模型发出 3 个 tool calls，每个都有不同的 `tool_use_id`。
2. Runtime 执行它们（如果独立，就并行执行）。
3. 每个结果都作为 `tool_result` block 回传，并用 `tool_use_id` 关联。

工程规则：把 correlation IDs 当作承重字段。交换它们，就会得到 wrong-tool-to-wrong-result routing。

### Sandboxing

Tool execution 是 sandbox boundary。细节见 Lesson 09。短版：每个工具都应该声明 read/write surface、network access、timeout、memory cap。通用 `run_shell(cmd)` 是危险信号；具体的 `git_status()` 更安全。

## 动手实现

`code/main.py` 实现一个 production-shape tool registry：

- JSON Schema subset validator（stdlib only）。
- 注册工具时带 description、input schema、timeout 和 executor。
- Argument coercion 和 enum validation。
- 带 correlation IDs 的 parallel tool dispatch。
- Error observations 作为 structured strings。

运行：

```text
python3 code/main.py
```

trace 展示一个 mini agent 在一个 turn 中调用三个工具，其中一个故意 malformed call 会被拒绝，并返回模型可以行动的描述性错误。

## 实际使用

每个 provider 都有自己的 tool schema：Anthropic、OpenAI、Gemini、Bedrock。如果需要 multi-provider，就使用 translation layer（OpenAI Agents SDK、Vercel AI SDK、LangChain tool adapter）。BFCL 是 reference benchmark；如果 tool use 是产品核心，上线前要用它评估你的 agent。

## 交付成果

`outputs/skill-tool-registry.md` 会为给定 task domain 生成 tool catalog、schema 和 registry。它包含 description-quality checks（每个工具的 description 是否告诉模型何时使用它？）。

## 练习

1. 添加一个“no-op”工具，让模型可以显式拒绝使用任何其他工具。在 BFCL-like hallucination test 上测量。
2. 为 int-as-string 和 float-as-string 实现 argument coercion。coercion 从哪里开始掩盖真实 bugs？
3. 添加 per-tool timeout 和 circuit breaker（连续 3 次失败后，60s 内拒绝该工具）。这会如何改变模型的恢复方式？
4. 阅读 BFCL V4 description。选一个 category（例如“multi-turn”），让你的 agent 跑 10 个 example prompts。报告 pass rate。
5. 将 stdlib validator 移植到 Pydantic 或 Zod。Pydantic/Zod 抓到了 toy 漏掉的什么？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Function calling | “Tool use” | 带 validated schema 的 structured-output tool invocation |
| Toolformer | “Self-supervised tool annotation” | Schick 2023 — 保留那些结果会降低 next-token loss 的 tool calls |
| BFCL | “Berkeley Function Calling Leaderboard” | 2026 benchmark：40% agentic、30% multi-turn、10% live、10% non-live、10% hallucination |
| Tool schema | “给模型看的 function signature” | name、description、arguments 的 JSON Schema |
| tool_use_id | “Correlation ID” | 把 tool call 和它的 result 绑定起来；parallel dispatch 必不可少 |
| Hallucination detection | “知道什么时候不该调用” | V4 category：没有合适工具时拒绝调用 |
| Argument coercion | “String-to-int repair” | 对可预测 schema-mismatch 的窄修复；有歧义就 reject |
| Sandboxing | “Tool execution boundary” | 每个工具的 read/write surface、network、timeout、memory cap |

## 延伸阅读

- [Schick et al., Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761) — self-supervised tool annotation
- [Berkeley Function Calling Leaderboard (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html) — 2026 eval benchmark
- [Anthropic, Tool use documentation](https://platform.claude.com/docs/en/agent-sdk/overview) — production tool schema in the Claude Agent SDK
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — function tool type and Guardrails
