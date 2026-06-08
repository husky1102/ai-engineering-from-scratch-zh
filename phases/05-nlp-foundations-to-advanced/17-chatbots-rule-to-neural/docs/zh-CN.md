# 聊天机器人：从规则到神经网络再到 LLM Agents

> ELIZA 用模式匹配回复。DialogFlow 映射 intents。GPT 从 weights 中回答。Claude 运行工具并验证。每个时代都解决了上一代最糟糕的失败。

**类型：** Learn
**语言：** Python
**先修：** Phase 5 · 13 (Question Answering), Phase 5 · 14 (Information Retrieval)
**时间：** ~75 minutes

## 要解决的问题

用户说：“I want to change my flight.” 系统必须弄清楚他们想要什么、缺少什么信息、如何获取信息，以及如何完成动作。然后用户又说：“wait, what if I cancel instead?” 系统必须记住上下文、切换任务，并保留状态。

对一个 ML system 来说，对话很难。输入是开放式的。输出必须在多轮中保持连贯。系统可能需要对真实世界采取行动（改签航班、扣款）。每一步错误都会被用户看见。

Chatbot architectures 经历过四种范式，每一种都是因为上一种失败得太明显而出现。本课会按顺序走过它们。2026 年的生产格局是后两者的混合。

## 核心概念

![Chatbot evolution: rule-based → retrieval → neural → agent](../assets/chatbot.svg)

**Rule-based (ELIZA, AIML, DialogFlow)。** 手写 patterns 匹配用户输入并生成响应。Intent classifiers 路由到预定义 flows。Slot-filling state machines 收集所需信息。在设计好的窄范围内效果极好。一旦超出范围就立刻失败。在不允许 hallucination 的 safety-critical domains（银行认证、航空预订）中仍然在生产使用。

**Retrieval-based。** FAQ 风格系统。编码每一对 (utterance, response)。运行时编码用户消息，检索最近的已存响应。可以把它想成 Zendesk 经典的 "similar articles" 功能。比规则更能处理改写。没有 generation，所以没有 hallucination。

**Neural (seq2seq)。** 在 conversation logs 上训练 encoder-decoder。从零开始生成响应。流畅，但容易输出泛泛回答（"I don't know"）并发生 factual drift。永远不能可靠地保持在主题上。这就是 Google、Facebook 和 Microsoft 在 2016-2019 年的聊天机器人都令人失望的原因。

**LLM agents。** 一个 language model 被包在循环里，能够计划、调用工具并验证结果。它不是带长 prompt 的 chatbot。Agent loop 是：plan → call tool → observe result → decide next step。Retrieval-first grounding (RAG) 让它不容易 hallucinate。Tool calls 让它真的能做事。这就是 2026 年的架构。

这四种范式不是简单的顺序替换。2026 年的生产 chatbot 会路由经过全部四种：rule-based 用于认证和破坏性动作，retrieval 用于 FAQ，neural generation 用于自然措辞，LLM agent 用于模糊的开放式查询。

## 动手实现

### Step 1: rule-based pattern matching

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

20 行里的 ELIZA。Reflection trick（"I feel sad" → "Why do you feel sad"）是 Weizenbaum 1966 年 canonical psychotherapist demo。至今仍然很有教学意义。

### Step 2: retrieval-based (FAQ)

这个说明性片段需要 `pip install sentence-transformers`（会拉入 torch）。本课可运行的 `code/main.py` 改用 stdlib Jaccard similarity，所以课程无需外部依赖即可运行。

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

基于 threshold 的拒答是关键设计选择。如果最佳匹配不够接近，就返回 `None` 并让系统升级处理。

### Step 3: neural generation (baseline)

使用一个小型 instruction-tuned encoder-decoder（FLAN-T5）或 fine-tuned conversational model。2026 年单独用于生产并不可用（contradiction、off-topic drift、factual nonsense），但会在混合系统中负责自然措辞。DialoGPT 风格的 decoder-only models 需要显式 turn separators 和 EOS handling 才能产生连贯回复；FLAN-T5 text2text pipeline 对教学示例来说开箱即用。

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### Step 4: LLM agent loop

2026 年的生产形态：

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

要说清楚三件事。Tools 是 LLM 可以调用的 callable functions。当 LLM 返回 final answer 而不是 tool call 时，循环终止。Step budget 防止在模糊任务上无限循环。

真实生产还会加入：retrieval-first grounding（在每次 LLM call 前注入相关 docs）、guardrails（没有确认就拒绝 destructive actions）、observability（记录每一步）和 evaluations（自动检查 agent behavior 是否保持 on-spec）。

### Step 5: hybrid routing

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

模式是：对任何 destructive 事务使用 deterministic rules，对 canned FAQs 使用 retrieval，对其他所有内容使用 LLM agents。这就是 2026 年 customer-support systems 交付的方式。

## 实际使用

2026 stack：

| Use case | Architecture |
|---------|---------------|
| Booking, payment, authentication | Rule-based state machines + slot filling |
| Customer support FAQs | Retrieval over curated answers |
| Open-ended help chat | LLM agent with RAG + tool calls |
| Internal tools / IDE assistants | LLM agent with tool calls (search, read, write) |
| Companion / character chatbots | Tuned LLM with persona system prompt, retrieval on knowledge |

生产中始终使用 hybrid routing。没有单一架构能很好处理所有请求。Routing layer 本身通常是一个小型 intent classifier。

## Failure modes that still ship

- **Confident fabrication。** LLM agent 声称它完成了某个动作，但实际上没有。缓解：验证 outcomes，记录 tool calls，绝不允许 LLM 在没有成功 tool return 的情况下声称已经做了某事。
- **Prompt injection。** 用户插入覆盖 system prompt 的文本。它在 OWASP Top 10 for LLM Applications 2025 中排名 LLM01。两种形式：direct injection（粘贴到 chat 中）和 indirect injection（隐藏在 agent 会读取的 documents、emails 或 tool outputs 中）。

  Attack rates 会随场景变化。在通用 tool-use 和 coding benchmarks 中，frontier models 的实测 success rates 约为 0.5-8.5%。特定高风险设置（针对 AI coding agents 的 adaptive attacks、脆弱 orchestration）曾达到约 84%。生产 CVEs 包括 EchoLeak (CVE-2025-32711, CVSS 9.3)：Microsoft 365 Copilot 中由攻击者控制邮件触发的 zero-click data-exfiltration flaw。

  缓解：在整个循环中把 user input 当作 untrusted；tool calls 前 sanitize；把 tool outputs 与 main prompt 隔离；使用 Plan-Verify-Execute (PVE) pattern，让 agent 先计划，再在执行前对照计划验证每个动作（这会阻止 tool results 注入新的未计划动作）；destructive actions 需要用户确认；对 tool scopes 应用 least-privilege。

  没有任何 prompt engineering 能完全消除这个风险。需要外部 runtime defense layers（LLM Guard、allowlist validation、semantic anomaly detection）。
- **Scope creep。** Agent 因为一个 tool call 返回了切题但旁支的信息而偏离任务。缓解：收窄 tool contracts；保持 system prompt 聚焦；为 off-task rate 添加 evaluations。
- **Infinite loops。** Agent 不断调用同一个 tool。缓解：step budget、tool-call deduplication、LLM judge 判断“我们是否在取得进展”。
- **Context window exhaustion。** 长对话把最早的 turns 挤出 context。缓解：总结较早 turns，按相似性检索相关 past turns，或使用 long-context model。

## 交付成果

保存为 `outputs/skill-chatbot-architect.md`：

```markdown
---
name: chatbot-architect
description: Design a chatbot stack for a given use case.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

Given a product context (user need, compliance constraints, available tools, data volume), output:

1. Architecture. Rule-based, retrieval, neural, LLM agent, or hybrid (specify which paths go where).
2. LLM choice if applicable. Name the model family (Claude, GPT-4, Llama-3.1, Mixtral). Match to tool-use quality and cost.
3. Grounding strategy. RAG sources, retrieval method (see lesson 14), tool contracts.
4. Evaluation plan. Task success rate, tool-call correctness, off-task rate, hallucination rate on held-out dialogs.

Refuse to recommend a pure-LLM agent for any destructive action (payments, account deletion, data modification) without a structured confirmation flow. Refuse to skip the prompt-injection audit if the agent has write access to anything.
```

## 练习

1. **Easy.** 用上面的 rule-based respond 为咖啡店点单 bot 实现 10 个 patterns。测试边界情况：double orders、modifications、cancellation、unclear intent。
2. **Medium.** 构建一个 hybrid FAQ + LLM fallback。为一个 SaaS product 准备 50 条 canned FAQ entries，用 docs site retrieval 做 LLM fallback。在 100 个真实 support questions 上测量 refusal rate 和 accuracy。
3. **Hard.** 用三个 tools（search、read-user-data、send-email）实现上面的 agent loop。运行包含 50 个 test scenarios 的评估，其中包括 prompt injection attempts。报告 off-task rate、failed task rate，以及任何 injection success。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Intent | 用户想要什么 | Categorical label（book_flight、reset_password）。路由到 handler。 |
| Slot | 一条信息 | Bot 所需的参数（date、destination）。Slot filling 是一系列询问。 |
| RAG | Retrieval plus generation | 检索相关 docs，然后为 LLM response 提供 grounding。 |
| Tool call | 函数调用 | LLM 发出带 name + args 的 structured call。Runtime 执行并返回 result。 |
| Agent loop | Plan, act, verify | 控制器，交错运行 LLM calls 与 tool calls，直到任务完成。 |
| Prompt injection | 用户攻击 prompt | 试图覆盖 system prompt 的恶意输入。 |

## 延伸阅读

- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — 原始 rule-based chatbot 论文。
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) — Google 后期 neural-chatbot 论文，紧接着 LLM agents 时代到来。
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — 命名 agent loop pattern 的论文。
- [Anthropic's guide on building effective agents](https://www.anthropic.com/research/building-effective-agents) — 2024 年的生产 guidance，到 2026 年仍然成立。
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) — prompt-injection 论文。
- [OWASP Top 10 for LLM Applications 2025 — LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) — 让 prompt injection 成为头号安全关注点的排名。
- [AWS — Securing Amazon Bedrock Agents against Indirect Prompt Injections](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) — 实用 orchestration-layer defenses，包括 Plan-Verify-Execute 与 user-confirmation flows。
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) — 来自 indirect prompt injection 的 canonical zero-click data-exfiltration CVE。它说明了为什么有 write-access 的 agents 需要 runtime defenses。
