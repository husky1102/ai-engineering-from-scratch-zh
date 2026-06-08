# Compliance — SOC 2、HIPAA、GDPR、PCI-DSS、EU AI Act、ISO 42001

> Multi-framework coverage 是 2026 年 enterprise deals 的基本门槛。**EU AI Act**：自 2024 年 8 月 1 日起生效。多数 high-risk requirements 于 2026 年 8 月 2 日 enforcement。针对 high-risk-system obligations（Art. 99(4)）的罚款最高 €15M 或 global annual turnover 的 3%；针对 prohibited AI practices（Art. 99(3)）最高 €35M 或 7%。如果服务 EU users，则全球适用。**Colorado AI Act**：2026 年 6 月 30 日生效（由 SB25B-004 从 2026 年 2 月延期）—— high-risk systems 的 impact assessments，以及对 AI decisions 的 appeal right。Virginia 对 credit/employment/housing/education 有类似要求。**SOC 2 Type II**：B2B AI 的事实要求（fintech 需要 Type II，不是 Type I）。**GDPR**：有记录的最大 AI-specific fine 是 Dutch DPA 于 2024 年 9 月对 Clearview AI 罚款 €30.5M；Italy's Garante 于 2024 年 12 月对 OpenAI 罚款 €15M（后来在 2026 年 3 月 appeal 中被 overturned）。Inference 时 real-time PII redaction 是可辩护标准；post-processing cleanup 不够。**HIPAA**：healthcare bound —— 没有 BAA，不能把 PHI 发送给 external AI services。**PCI-DSS**：AI-interaction-layer coverage 需要 configuration + contractual agreements，不是自动适用。**ISO 42001**：新兴 AI governance standard，正与 ISO 27001 一起成为越来越常见的 procurement requirement。Reference profile：OpenAI 维护 SOC 2 Type 2、ISO/IEC 27001:2022、ISO/IEC 27701:2019、GDPR/CCPA/HIPAA（BAA）/FERPA，以及 ChatGPT payment components 的 PCI-DSS。Cross-framework mapping 降低 audit fatigue：access controls 映射到 ISO 27001 A.5.15-5.18、GDPR Art. 32、HIPAA §164.312(a)。

**类型:** 学习
**语言:**（Python optional —— compliance 是 policy + process，不是 code）
**先修:** Phase 17 · 25（Security），Phase 17 · 13（Observability）
**时间:** ~60 分钟

## 学习目标

- 枚举与 LLM products 相关的七个 2026 frameworks，并把每个匹配到 customer segment。
- 引用 EU AI Act enforcement timeline（2024 年 8 月生效；high-risk enforcement 2026 年 8 月）和两级 fine ceiling（high-risk obligations 为 €15M / 3%，prohibited practices 为 €35M / 7%）。
- 解释为什么 post-processing PII cleanup 对 GDPR 来说不够，并说出 real-time inference-layer redaction 是可辩护标准。
- 描述 cross-framework control mapping（例如 access control 映射到 ISO 27001 A.5.15-5.18 + GDPR Art. 32 + HIPAA §164.312(a)）。

## 要解决的问题

一家 enterprise customer 的 procurement 要求 SOC 2 Type II、GDPR、HIPAA BAA、ISO 27001，以及“EU AI Act compliance statement”。你的团队只有 SOC 2 Type I。你离 Type II 还有六个月，而且还没开始 GDPR Article 30 records。

Multi-framework coverage 不是 LLM 问题 —— 它是 enterprise-SaaS 问题，只是带有 LLM-specific overlays。2026 年的 procurement teams 想要的是一个 matrix：每行一个 framework，每列一个 control，而不是一个 PDF。

## 核心概念

### 七个 frameworks

| Framework | Scope | LLM-specific requirement |
|-----------|-------|--------------------------|
| SOC 2 Type II | B2B SaaS baseline | Process controls audited over 6-12 months |
| HIPAA | US healthcare | BAA required；没有 signed agreement，PHI 不能离开 infrastructure |
| GDPR | EU users | Real-time PII redaction；data subject rights；Article 30 records |
| PCI-DSS | Payment data | AI touching payment 时需要 configuration + contracts |
| EU AI Act | Serving EU users | Risk tier classification；high-risk systems：conformity assessment、documentation、logging |
| Colorado AI Act | Serving CO residents | Impact assessments；right to appeal |
| ISO 42001 | AI governance | Emerging；pairs with ISO 27001 |

### EU AI Act timeline

- 2024 年 8 月 1 日：in force。
- 2025 年 2 月 2 日：prohibited-AI practices enforced。
- 2026 年 8 月 2 日：high-risk systems enforced（conformity assessment、documentation、logging）。
- 2027 年 8 月：harmonized legislation 下 products 中的 high-risk systems。

Risk tiers：Unacceptable（banned）、High-risk（conformity + logging）、Limited-risk（transparency）、Minimal-risk（no constraint）。多数 B2B LLM SaaS 是 limited-risk；employment、credit、education、law enforcement、migration、essential services 会触发 high-risk。

Fines（Article 99）：违反 high-risk-system obligations（Art. 99(4)）最高 €15M 或 global annual turnover 的 3%；prohibited AI practices（Art. 99(3)）最高 €35M 或 7%；适用较高者。

### GDPR — real-time redaction 是标准

Post-processing cleanup（LLM 已看过数据后再 redact PII）不是可辩护姿态 —— 模型已经看到了数据。Real-time inference-layer redaction 是 2026 年标准：

- LLM call 前做 entity recognition。
- Consistent tokenization（Mesh approach）保留 semantics。
- 只存 redacted prompts + consented opt-in raw。

近期 enforcement：Dutch DPA 于 2024 年 9 月对 Clearview AI 罚款 €30.5M，是迄今有记录的最大 AI-specific GDPR fine；Italy's Garante 于 2024 年 12 月对 OpenAI 罚款 €15M，是最大的 LLM-specific fine，不过它在 2026 年 3 月 appeal 中被 overturned，且 ruling 仍在 further review。Post-processing claims 在 audit 中失败过。

### HIPAA — BAA 不是可选项

没有 signed Business Associate Agreement，不能把 PHI 发送给 external AI services。三大 hyperscaler LLM platforms（Bedrock、Azure OpenAI、Vertex）都提供 BAAs。OpenAI direct API 提供 BAA。Anthropic direct API 提供 BAA。发送 PHI 前先确认。

### SOC 2 Type II

Type I：controls designed and documented。
Type II：controls operate effectively over 6-12 months。

2026 年 B2B procurement 默认 Type II。Type I 是起点；Type II 才是 gate。

常见 audit drivers：access logs（谁看到了什么）、change management（如何部署）、risk assessments（quarterly）、incident response（是否测试过）。Phase 17 · 25 的 audit log 可以直接复用。

### Cross-framework mapping

一个 access control policy 可以满足多个 framework controls：

| Control | Frameworks |
|---------|-----------|
| Access logging | ISO 27001 A.5.15-5.18、GDPR Art. 32、HIPAA §164.312(a) |
| Change management | ISO 27001 A.8.32、PCI DSS Req. 6、HIPAA breach-notification scope |
| Encryption in transit | ISO 27001 A.8.24、GDPR Art. 32、HIPAA §164.312(e) |
| Secrets management | ISO 27001 A.8.19、PCI DSS Req. 8、SOC 2 CC6.1 |

Compliance tools（Drata、Vanta、Secureframe）会自动化这种 mapping。规模起来后值得花钱。

### ISO 42001 — emerging

2023 年末发布。作为与 ISO 27001 并列的 procurement requirement 正在增长。它是面向 AI governance 的 framework，覆盖 risk management、data quality、transparency、human oversight。

### OpenAI 的 reference profile

OpenAI 维护 SOC 2 Type 2、ISO/IEC 27001:2022、ISO/IEC 27701:2019、GDPR/CCPA/HIPAA（BAA）/FERPA，以及 ChatGPT payment components 的 PCI-DSS。这大致就是 2026 年 enterprise table stakes。

### 你应该记住的数字

- EU AI Act fines：high-risk obligations（Art. 99(4)）最高 €15M / 3%；prohibited practices（Art. 99(3)）最高 €35M / 7%。
- EU AI Act high-risk enforcement：2026 年 8 月 2 日。
- 有记录的最大 AI-specific GDPR fine：€30.5M，Clearview AI（Dutch DPA，2024 年 9 月）。
- 最大 LLM-specific GDPR fine：€15M，OpenAI（Italy's Garante，2024 年 12 月；2026 年 3 月 appeal overturned）。
- SOC 2 Type II window：6-12 months of operated controls。
- Colorado AI Act effective date：2026 年 6 月 30 日（由 SB25B-004 从 2026 年 2 月延期）。

## 实际使用

`code/main.py` 是一个用 Python 写的 compliance-mapping spreadsheet —— 给定 control，列出它满足的 frameworks。

## 交付成果

本课产出 `outputs/skill-compliance-matrix.md`。给定 customer segment 和 geography，它会指定 required frameworks 和 controls。

## 练习

1. 你的第一个 enterprise customer 要求 SOC 2 Type II、HIPAA BAA、EU AI Act statement。赢下这笔 deal 的 minimum viable compliance posture 是什么？
2. 根据 EU AI Act risk tiers 对三个 hypothetical LLM products 分类。到了 high-risk 会改变什么？
3. 你意外把 PHI 发送给了没有 BAA 的 provider。走一遍 incident response。
4. 论证 ISO 42001 对 mid-market AI vendor 来说在 2026 年是否“必要”。
5. 将你的 LLM audit log fields（Phase 17 · 25）映射到至少三个 framework controls。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|------------|----------|
| SOC 2 Type II | “audited controls” | 独立 attested、运行 6-12 months 的 controls |
| HIPAA BAA | “healthcare contract” | Business Associate Agreement；PHI 所必需 |
| GDPR | “EU privacy” | Real-time PII redaction 是可辩护的 2026 标准 |
| EU AI Act | “EU AI rules” | High-risk enforcement 2026 年 8 月；€15M / 3%（high-risk obligations）— €35M / 7%（prohibited practices） |
| Colorado AI Act | “US AI state law” | 2026 年 6 月 30 日生效（由 SB25B-004 延期）；impact assessments |
| ISO 42001 | “AI governance” | 面向 AI risk + transparency 的 emerging framework |
| ISO 27001 | “security ISMS” | Information Security Management System baseline |
| Conformity assessment | “EU AI doc package” | High-risk requirement：docs、testing、logging |
| Cross-framework mapping | “one control, many frames” | 单一 policy 满足多个 framework controls |

## 延伸阅读

- [OpenAI Security and Privacy](https://openai.com/security-and-privacy/) — reference compliance profile.
- [GuardionAI — LLM Compliance 2026: ISO 42001, EU AI Act, SOC 2, GDPR](https://guardion.ai/blog/llm-compliance-guide-iso-42001-eu-ai-act-soc2-gdpr-2026)
- [Dsalta — SOC 2 Type 2 Audit Guide 2026: 10 AI Controls](https://www.dsalta.com/resources/ai-compliance/soc-2-type-2-audit-guide-2026-10-ai-powered-controls-every-saas-team-needs)
- [EU AI Act official text](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) — primary source.
- [Colorado AI Act](https://leg.colorado.gov/bills/sb24-205) — primary source.
- [ISO/IEC 42001:2023](https://www.iso.org/standard/81230.html) — AI management system standard.
