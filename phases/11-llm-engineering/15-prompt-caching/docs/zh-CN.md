# Prompt Caching 与 Context Caching

> 你的 system prompt 有 4,000 个 token。你的 RAG context 有 20,000 个 token。每个请求都同时发送两者。你也为两者付费，而且每次都付。Prompt caching 让 provider 在它们那一侧保持这个前缀“热着”，并在复用时只按正常价格的 10% 计费。正确使用时，它能把推理成本降低 50-90%，把首 token 延迟降低 40-85%。

**类型：** 构建
**语言：** Python
**先修：** Phase 11 · 01（Prompt Engineering），Phase 11 · 05（Context Engineering），Phase 11 · 11（Caching and Cost）
**时间：** ~60 分钟

## 要解决的问题

一个 coding agent 在对话的每一轮都把同一个 15,000-token system prompt 发给 Claude。按每百万输入 token 3 美元计算，20 轮仅输入成本就是 0.90 美元，还没算用户自己的实际消息。乘以每天 10,000 场对话，永不改变的文本一天就会带来 9,000 美元账单。

你不能缩短 prompt，否则质量会受损。你也不能不发送它，因为模型每一轮都需要它。唯一的办法，是不要再为 provider 已经见过的前缀付全价。

这个办法就是 prompt caching。Anthropic 在 2024 年 8 月发布了它（2025 年加入 1 小时 extended-TTL 变体），OpenAI 在同年稍晚将其自动化，Google 则随 Gemini 1.5 一起发布了显式 context caching，如今三者都把它作为 frontier model 的一等功能提供。

## 核心概念

![Prompt caching：写一次，读很便宜](../assets/prompt-caching.svg)

**机制。** 当某个请求的前缀匹配最近请求中的一个前缀时，provider 会复用上一次运行得到的 KV-cache，而不是重新编码这些 token。第一次写入时你支付一点写入溢价，之后每次读取都获得大幅折扣。

**2026 年三类 provider 风格。**

| Provider | API 风格 | 命中折扣 | 写入溢价 | 默认 TTL | 最小可缓存 |
|---------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | 内容块上的显式 `cache_control` marker | 输入 90% 折扣 | 25% 附加费 | 5 分钟（可延长到 1 小时） | 1,024 tokens（Sonnet/Opus），2,048（Haiku） |
| OpenAI | 自动前缀检测 | 输入 50% 折扣 | 无 | 最多 1 小时（best-effort） | 1,024 tokens |
| Google（Gemini） | 显式 `CachedContent` API | 按存储计费；读取约为正常价格的 25% | 每 token·hour 收取存储费 | 用户设置（默认 1 小时） | 4,096 tokens（Flash），32,768（Pro） |

**不变量。** 三者都只缓存前缀。如果请求之间有任何 token 不同，第一个不同 token 之后的所有内容都会 miss。把*稳定*部分放在顶部，把*可变*部分放在底部。

### 适合缓存的布局

```text
[system prompt]          <-- cache this
[tool definitions]       <-- cache this
[few-shot examples]      <-- cache this
[retrieved documents]    <-- cache if reused, else don't
[conversation history]   <-- cache up to last turn
[current user message]   <-- never cache (different every time)
```

违反这个顺序，比如把用户消息放在 system prompt 上方，或者把动态检索内容插进 few-shot 之间，缓存就永远不会命中。

### 收支平衡计算

Anthropic 的 25% 写入溢价意味着，一个 cached block 至少要被读取两次，整体才会省钱。1 次写入 + 1 次读取的平均成本是每请求 0.675x（节省 32%）；1 次写入 + 10 次读取的平均成本是 0.205x（节省 80%）。经验法则：只要你预计在 TTL 内复用至少 3 次，就缓存它。

## 动手实现

### 步骤 1：带显式 marker 的 Anthropic prompt caching

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

`cache_control` marker 告诉 Anthropic 把这个 block 存储 5 分钟。窗口内复用会命中；过期后复用会再次写入。

**响应 usage 字段：**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # paid at 1.25x
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # paid at 0.1x
```

在 CI 中检查这两个字段。如果 `cache_read_input_tokens` 在多个请求之间一直为零，你的 cache key 正在漂移。

### 步骤 2：一小时 extended TTL

对长时间运行的批处理作业，5 分钟默认 TTL 可能会在作业之间过期。设置 `ttl`：

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1 小时 TTL 的写入溢价成本是 2 倍（相对 baseline 多 50%，而不是 25%），但任何复用该前缀超过 5 次的批处理都能很快回本。

### 步骤 3：OpenAI 自动缓存

OpenAI 没有什么可配置项。任何超过 1,024 token 且匹配近期请求的前缀，都会自动获得 50% 折扣。

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # long and stable
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # the discounted portion
```

同样适用缓存友好布局规则。有两件事会破坏 OpenAI 的缓存，但不会破坏 Anthropic 的缓存：改变 `user` 字段（它被用作 cache key 的组成部分），以及重新排序 tools。

### 步骤 4：Gemini 显式 context caching

Gemini 把 cache 当作你创建并命名的一等对象：

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Gemini 会按 cache 存活期间的 token·hour 收取存储费，读取约按正常输入费率的 25% 计费。当你要在多天内跨许多会话复用同一个巨大 prompt 时，这就是合适的形状。

### 步骤 5：在生产中衡量命中率

参见 `code/main.py`，里面有一个模拟三家 provider 的记账器，会跟踪 write/read/miss 计数，并计算每 1K 请求的混合成本。用目标命中率作为部署门禁：大多数生产 Anthropic 设置在预热后应该看到超过 80% 的 read fraction。

## 2026 年仍会上线的坑

- **动态时间戳放在顶部。** system prompt 顶部的 `"Current time: 2026-04-22 15:30:02"`。每个请求都会 miss。把时间戳移到 cache breakpoint 下方。
- **Tool 重排序。** 用稳定顺序序列化 tools。部署之间一次 dict 重排就会破坏所有命中。
- **自由文本近重复。** "You are helpful." 与 "You are a helpful assistant."。一个字节差异就是完整 miss。
- **block 太小。** Anthropic 强制要求 1,024-token 下限（Haiku 为 2,048）。更小的 block 会静默不缓存。
- **盲目的成本仪表盘。** 把 “input tokens” 拆分成 cached 与 uncached。否则一次流量下降会看起来像缓存胜利。

## 实际使用

2026 年的缓存技术栈：

| 场景 | 选择 |
|-----------|------|
| 拥有稳定 10k+ system prompt、多轮对话的 agent | Anthropic `cache_control`，5 分钟 TTL |
| 复用前缀 30+ 分钟的批处理作业 | Anthropic，`ttl: "1h"` |
| GPT-5 上的 serverless endpoint，无自定义基础设施 | OpenAI 自动缓存（只需让前缀稳定且足够长） |
| 巨大代码/文档语料的多日复用 | Gemini 显式 `CachedContent` |
| 跨 provider fallback | 保持各 provider 的可缓存前缀布局一致，让任何命中都能生效 |

把它和语义缓存（Phase 11 · 11）结合用于用户消息层：prompt caching 处理 *token-identical* 复用，semantic caching 处理 *meaning-identical* 复用。

## 交付成果

保存 `outputs/skill-prompt-caching-planner.md`：

```markdown
---
name: prompt-caching-planner
description: Design a cache-friendly prompt layout and pick the right provider caching mode.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

Given a prompt (system + tools + few-shot + retrieval + history + user) and a usage profile (requests per hour, TTL needed, provider), output:

1. Layout. Reordered sections with a single cache breakpoint marked; explain which sections are stable, which are volatile.
2. Provider mode. Anthropic cache_control, OpenAI automatic, or Gemini CachedContent. Justify from TTL and reuse pattern.
3. Break-even. Expected reads per write within TTL; net cost vs no-cache with math.
4. Verification plan. CI assertion that cache_read_input_tokens > 0 on the second identical request; dashboard split by cached vs uncached tokens.
5. Failure modes. List the three most likely reasons the cache will miss in this setup (dynamic timestamp, tool reorder, near-duplicate text) and how you will prevent each.

Refuse to ship a cache plan that places a dynamic field above the breakpoint. Refuse to enable 1h TTL without a reuse count that makes the 2x write premium pay back.
```

## 练习

1. **简单。** 用一个 5,000-token system prompt 针对 Claude 跑一段 10 轮对话。先不使用 `cache_control`，再使用它。报告两者的 input-token 账单。
2. **中等。** 写一个测试 harness，给定一个 prompt template 和请求日志，计算每个 provider（Anthropic 5m、Anthropic 1h、OpenAI automatic、Gemini explicit）的预期命中率和美元节省。
3. **困难。** 构建一个布局优化器：给定一个 prompt，以及一组标记为 `stable=True/False` 的字段，在不丢失信息的前提下重写 prompt，把单个 cache breakpoint 放在最大缓存友好位置。在真实 Anthropic endpoint 上验证。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Prompt caching | “让长 prompt 变便宜” | 复用 provider 侧的 KV-cache 来匹配前缀；重复输入 token 可获得 50-90% 折扣。 |
| `cache_control` | “Anthropic marker” | 内容块属性，声明“一直到这里都是可缓存的”；`{"type": "ephemeral"}`。 |
| Cache write | “支付溢价” | 填充 cache 的第一次请求；Anthropic 按约 1.25x 输入费率计费，OpenAI 免费。 |
| Cache read | “折扣” | 匹配前缀的后续请求；按 10%（Anthropic）、50%（OpenAI）、约 25%（Gemini）计费。 |
| TTL | “它能活多久” | cache 保持热状态的秒数；Anthropic 默认 5m（可延长到 1h），OpenAI best-effort 最多 1h，Gemini 由用户设置。 |
| Extended TTL | “1 小时 Anthropic cache” | `{"type": "ephemeral", "ttl": "1h"}`；2 倍写入溢价，但对批处理复用值得。 |
| Prefix match | “为什么我的 cache miss 了” | 只有从开头到 breakpoint 的每个 token 都逐字节相同时，cache 才会命中。 |
| Context caching（Gemini） | “那个显式的” | Google 的命名 cache 对象，按存储计费；最适合大型语料的多日复用。 |

## 延伸阅读

- [Anthropic — Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)：`cache_control`、1h TTL、收支平衡表。
- [OpenAI — Prompt caching](https://platform.openai.com/docs/guides/prompt-caching)：自动前缀匹配。
- [Google — Context caching](https://ai.google.dev/gemini-api/docs/caching)：`CachedContent` API 与存储定价。
- [Anthropic engineering — Prompt caching for long-context workloads](https://www.anthropic.com/news/prompt-caching)：包含延迟数字的原始发布文章。
- Phase 11 · 05（Context Engineering）：在哪里切分 prompt，才能让 cache 落地。
- Phase 11 · 11（Caching and Cost）：把 prompt caching 和用户消息上的 semantic cache 配对。
- [Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102)：prompt caching 暴露给用户的 KV-cache 内存模型；解释为什么重新读取 cached prefix 比重新计算便宜约 10 倍。
- [Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369)：prefill 是 prompt caching 直接跳过的阶段；这篇论文解释为什么 cache hit 会显著降低 TTFT，而 TPOT 不受影响。
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)](https://arxiv.org/abs/2211.17192)：prompt caching 与 speculative decoding、Flash Attention、MQA/GQA 同属改变推理成本曲线的杠杆；想了解另外三者可读这篇。
