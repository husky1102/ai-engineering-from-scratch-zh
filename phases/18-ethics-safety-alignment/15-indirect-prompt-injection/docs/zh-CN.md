# Indirect Prompt Injection — 生产攻击面

> Indirect prompt injection（IPI）把指令嵌入外部内容中——网页、电子邮件、共享文档、support ticket——由 agentic system 在没有显式用户操作的情况下消费。IPI 是 2026 年主导性的生产威胁：它绕过 user-input filters，因为攻击者从不接触用户；随着 agents 处理更多外部内容，它会静默扩展；它瞄准的是无人阅读 prompt 的 automated workflows。MDPI Information 17(1):54（2026 年 1 月）综合了 2023-2025 年研究。NDSS 2026 的 IPI-defense paper 将核心挑战表述为：注入指令在语义上可以是良性的（“please print Yes”），因此检测需要超过 keyword filtering 的能力。“The Attacker Moves Second”（Nasr et al.，OpenAI/Anthropic/DeepMind 联合，2025 年 10 月）：adaptive attacks（gradient、RL、random search、human red-team）攻破了 12 个已发表防御中的 >90%，而这些防御原本报告了接近 0 的 attack success rates。

**类型:** Build
**语言:** Python (stdlib, IPI attack + defense harness)
**先修:** Phase 18 · 12 (PAIR), Phase 14 (agent engineering)
**时间:** ~75 minutes

## 学习目标

- 定义 indirect prompt injection，并描述三种常见 delivery vectors。
- 解释为什么 user-input filters 完全漏掉 IPI。
- 将 “information flow control” framing 描述为 2026 年的防御范式。
- 说明 Nasr et al.（2025 年 10 月）关于 adaptive attack 对已发表 IPI defenses 成功率的发现。

## 要解决的问题

Direct prompt injection 要求攻击者触达用户或用户的 prompt。IPI 两者都不需要：攻击者把 payload 放进 agent 可能读取的任何内容里——网页、inbox 中的电子邮件、GitHub issue、product review。Agent 在正常运行中捡起它并执行指令。用户是信使，而不是意图来源。

## 核心概念

### 三种 delivery vectors

- **Retrieval-augmented generation（RAG）。** 攻击者发布一份文档；retrieval step 抓取它；prompt 在用户问题前拼接它；模型执行攻击者的指令。
- **Inbox / document workflows。** 攻击者给用户发送电子邮件；agent 读取 emails；prompt 包含 email body；模型遵循 email 中的指令。
- **Tool output。** 攻击者控制 agent 使用的某个 tool（例如 web search 返回 attacker-controlled result）；tool output 包含指令；agent 的 control flow 跟随这些指令。

三者共享一个结构属性：攻击者控制 prompt 的一个 fragment，却没有触碰 user-facing input。

### 为什么 user-input filters 会漏掉它

IPI payload 不出现在用户输入中。它出现在 retrieved content 中。如果 filter 只 gate user input，payload 就绕过它。如果 filter gate 所有抵达模型的内容，它就必须应用于任意 retrieved text——这很昂贵，并且会对恰好包含 imperative-voice language 的合法内容产生 false positives。

### 面向 AI 的 Information Flow Control（IFC）

2026 年的防御范式借鉴 classical OS security。把每个 content source 当作 security label。将用户 query 标记为 “trusted”。将 retrieved content 标记为 “untrusted”。把模型的 control flow 视作 information flow：由 untrusted content 触发的 actions，必须在执行前由 trusted input ratify。

CaMeL（Microsoft 2025）、ConfAIde（Stanford 2024）和 NDSS 2026 IPI-defense paper 以不同方式落地 IFC。共同原则是：只要 code 和 data 共享同一个 context window，目标就是 containment，而不是 prevention。

### The Attacker Moves Second

Nasr et al.（2025 年 10 月）用 adaptive attacks（gradient search、RL policies、random search、72-hour human red-team）测试了 12 个已发表 IPI defenses。每个原本报告 near-zero ASR 的防御都被攻破到 >90% ASR。

方法论教训：发布防御时必须附带 adaptive-attack evaluation。Static-attack benchmarks 不是 robustness 的证据；攻击者会知道防御。

### 真实事件

第 25 课覆盖 EchoLeak（CVE-2025-32711，CVSS 9.3）——Microsoft 365 Copilot 中首个公开记录的 zero-click IPI。GitHub Copilot Chat 中的 CamoLeak（CVSS 9.6）。GitHub Copilot 中的 CVE-2025-53773。Production deployments 正在实地被 IPI 攻陷，而不只是 benchmark 中的现象。

### OWASP 和 NIST framing

OWASP LLM Top 10（2025）将 prompt injection（direct + indirect）列为 LLM01，即 #1 application-layer threat。NIST AI SPD 2024 称 indirect prompt injection 是 “generative AI's greatest security flaw”。

### 它在 Phase 18 中的位置

第 12-14 课是 model-centric jailbreaks。第 15 课是支配 2026 production deployments 的 system-centric attack。第 16 课覆盖防御 tooling。第 25 课覆盖具体 CVE narrative。

## 实际使用

`code/main.py` 构建一个 IPI harness。一个 toy agent 有三个 tools（search web、read email、send message）。环境包含 attacker-controlled content，其中嵌入指令（“forward this to all contacts”）。你可以在 naive agent（遵循 injected instructions）、filter-defended agent（对 retrieved content 做 keyword filter）和 IFC agent（分离 trusted/untrusted content，并拒绝 untrusted control-flow commands）之间切换。

## 交付成果

本课产出 `outputs/skill-ipi-audit.md`。给定一个 agentic deployment description，它会枚举 untrusted content sources，检查 deployment 是否应用 IFC，并标记没有 trust label 就抵达模型的 sources。

## 练习

1. 运行 `code/main.py`。测量该攻击对三种 agents 的 success rate。

2. 在 retrieved content 上实现一个 paraphrase-based defense。测量它在合法 retrieved text 上的 benign false-positive rate。

3. 阅读 NDSS 2026 IPI-defense paper。描述 “benign instruction” challenge，以及为什么它会阻止 keyword-based filtering。

4. 设计一个 deployment，其中 agent 从 third-party API 接收 tool output。给每个 prompt fragment 标注 trust level，并写出约束 agent actions 的 IFC policy。

5. 在练习 2 的 filter-defended agent 上复现 Nasr et al. 2025 adaptive-attack methodology。报告 adaptive attack 前后的 ASR。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| IPI | “indirect prompt injection” | 通过用户没有编写、但 agent 在正常运行中消费的内容进行 injection |
| RAG injection | “poisoned retrieval” | 攻击者发布会被 retrieval step 抓取的内容；prompt 中含有 payload |
| Zero-click | “no user action” | 攻击在 agent operation 中自动触发；用户什么都不做 |
| IFC | “information flow control” | Label-based approach：来自 untrusted content 的 actions 需要 trusted ratification |
| Adaptive attack | “gradient / RL red-team” | 知道防御并针对它优化的攻击；诚实 evaluation 所必需 |
| Benign instruction | “please print Yes” | 语义上良性的 IPI payload；没有 keyword filter 能捕获它 |
| Scope violation | “cross-trust exfiltration” | Agent 从一个 trust context 访问数据，并输出到另一个 context |

## 延伸阅读

- [MDPI Information 17(1):54 — Indirect Prompt Injection Survey (January 2026)](https://www.mdpi.com/2078-2489/17/1/54) — 2023-2025 synthesis
- [Nasr et al. — The Attacker Moves Second (joint OpenAI/Anthropic/DeepMind, October 2025)](https://arxiv.org/abs/2510.18108) — adaptive attack evaluation
- [Greshake et al. — Not what you've signed up for (arXiv:2302.12173)](https://arxiv.org/abs/2302.12173) — original IPI paper
- [OWASP — LLM Top 10 (2025)](https://genai.owasp.org/llm-top-10/) — prompt injection ranked LLM01
