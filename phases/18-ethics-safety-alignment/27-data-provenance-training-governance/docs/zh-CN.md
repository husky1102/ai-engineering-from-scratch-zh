# 数据出处与训练数据治理

> EU AI Act 要求到 2025 年 8 月为 GPAI 建立机器可读的 opt-out 标准（通过 EU Copyright Directive TDM exception）。California AB 2013（2024 年签署）：生成式 AI 训练数据透明度要求开发者发布包含 12 个强制字段的数据集摘要。2025 年 DPA 对 legitimate interest 的立场趋同：Irish DPC（2025 年 5 月 21 日）在 EDPB opinion 之后接受 Meta 在带有防护措施下使用第一方公开 EU/EEA 成人内容训练 LLM；Cologne Higher Regional Court（2025 年 5 月 23 日）驳回 injunction；Hamburg DPA 放弃紧急程序；UK ICO（2025 年 9 月 23 日）对 LinkedIn 的 AI 训练防护措施（透明度、简化 opt-out、延长反对窗口）作出积极监管回应并继续监测，但这不是正式 clearance。Brazilian ANPD（2024 年 7 月 2 日）因信息透明度不足暂停 Meta 的处理；在 Meta 提交合规计划后，该预防措施于 2024 年 8 月 30 日解除。关键不可逆问题：cookie-consent 框架面向实时、可逆跟踪；一旦数据进入模型权重，外科式删除就不可能，没有面向已训练神经网络的实用 GDPR right-to-erasure。合规窗口在收集时。Data Provenance Initiative（dataprovenance.org，Longpre、Mahari、Lee 等，“Consent in Crisis”，2024 年 7 月）：大规模审计显示，随着发布者添加 robots.txt 限制，AI data commons 正在快速衰退。

**类型：** 学习
**语言：** Python (stdlib, 12-field California AB 2013 scaffolding generator)
**先修：** 第 18 阶段 · 第 24 课（regulatory），第 18 阶段 · 第 26 课（cards）
**时间：** 约 60 分钟

## 学习目标

- 描述 California AB 2013 为生成式 AI 训练数据透明度规定的 12 个强制字段。
- 说出 2025 年 DPA 对 legitimate-interest LLM 训练的立场（Irish DPC、UK ICO、Hamburg、Cologne）。
- 描述不可逆问题：为什么 GDPR right-to-erasure 对已训练神经网络没有实用等价物。
- 说出 Data Provenance Initiative 的 “Consent in Crisis” 发现。

## 要解决的问题

训练数据治理是每张 model card（第 26 课）和每项监管义务（第 24 课）的上游。2024-2025 年，监管格局围绕三项原则收敛：opt-out 基础设施、逐数据集披露，以及对公开可得数据的 legitimate-interest 安排。Provider 若在收集时没有合规，下游无法补救。

## 核心概念

### California AB 2013

2024 年签署。对 2022 年 1 月 1 日或之后发布的系统，文档必须在 2026 年 1 月 1 日或之前发布。Section 3111(a) 要求开发者发布用于训练的数据集高层摘要，包含 12 个法定项目：
1. 数据集来源或所有者。
2. 数据集如何推进 AI 系统预期目的的描述。
3. 数据集中的 data points 数量（可接受一般范围；动态数据集可用估计）。
4. data points 类型描述（有标注数据集的 label types；无标注数据集的一般特征）。
5. 数据集是否包含任何受 copyright、trademark 或 patent 保护的数据，或是否完全处于 public domain。
6. 数据集是否购买或授权获得。
7. 数据集是否包含个人信息（按 Cal. Civ. Code §1798.140(v)）。
8. 数据集是否包含 aggregate consumer information（按 Cal. Civ. Code §1798.140(b)）。
9. 开发者进行的清洗、处理或其他修改，以及预期目的。
10. 数据收集的时间段；若仍在持续收集，需要说明。
11. 数据集在开发期间首次使用的日期。
12. 系统是否使用或持续使用 synthetic data generation。

相对于 Gebru 等 2018 datasheets，项目 12（synthetic data）是新的。项目 7（personal information）会触发 Privacy Rights Act (CPRA) 义务。该法规豁免安全 / 完整性、航空器运行，以及仅限联邦的国家安全系统（Section 3111(b)）。

### EU AI Act（第 24 课）与 TDM opt-out

EU Copyright Directive 的 text-and-data-mining exception 允许在公开可得内容上训练，除非权利人选择 opt out。EU AI Act GPAI Code of Practice Copyright 章节要求 GPAI provider 尊重机器可读 opt-out 信号（robots.txt、C2PA “No AI Training” claim 等）。

### 2025 年 DPA 在 legitimate interest 上趋同

Irish DPC（2025 年 5 月 21 日）：在 EDPB opinion 之后，接受 Meta 计划使用第一方公开 EU/EEA 成年用户内容进行训练，前提是有防护措施。Cologne Higher Regional Court（2025 年 5 月 23 日）驳回针对 Meta 的 injunction：opt-out 足够。Hamburg DPA 为了 EU 范围一致性，放弃紧急程序。UK ICO（2025 年 9 月 23 日）对 LinkedIn 在类似防护措施和持续监测下恢复 AI 训练作出积极监管回应，但这不是正式 clearance。

趋同原则：legitimate interest 可以在带有 opt-out 的情况下正当化对公开可得第一方内容的训练。不要求 consent。

### Brazilian ANPD（2024 年 6 月）

因信息透明度不足，暂停 Meta 对巴西用户数据进行 AI 训练的处理。结果不同于 EU DPAs；ANPD 优先考虑透明度，而不是 legitimate-interest 可接受性。

### 不可逆问题

Cookie-consent 是为实时、可逆跟踪设计的。训练数据不同：一旦数据进入模型权重，外科式删除就不可能。彻底补救只有从头重训，而这代价过高。

部分补救：
- **Unlearning。** 近似移除；通过 MIA 测量（第 22 课）。
- **基于 influence function 的定位。** 识别受该数据影响最大的权重；选择性更新。
- **Fine-tune-suppression。** 训练模型拒绝输出源自该数据的内容。

这些都没有完全解决问题。合规窗口在收集时。

### Data Provenance Initiative

dataprovenance.org。Longpre、Mahari、Lee 等 “Consent in Crisis”（2024 年 7 月）：对 AI 训练数据 commons 的大规模审计。发现：发布者正在以加速速度添加 robots.txt 限制。可开放训练的 commons 正在快速收缩。2023 -> 2024 年，top training sources 中约 25% 添加了某种限制。含义：未来训练数据可用性取决于新的获取范式（授权、合成生成、激励参与）。

### 它在第 18 阶段中的位置

第 26 课是模型级文档。第 27 课是数据集级治理。二者共同定义透明度层。第 28 课映射研究这些问题的生态系统。

## 实际使用

`code/main.py` 为一个玩具数据集生成符合 California AB 2013 的 12 字段数据集摘要脚手架。你可以填写这些字段，并观察哪些字段会触发隐私或 copyright 后续义务。

## 交付成果

本课产出 `outputs/skill-provenance-check.md`。给定用于训练的数据集，它会检查 AB 2013 12 字段覆盖、opt-out 基础设施合规、DPA 对齐，以及不可逆风险评估。

## 练习

1. 运行 `code/main.py`。为一个玩具数据集生成 12 字段摘要，并识别哪些字段说明不足。

2. EU Copyright Directive TDM opt-out 是机器可读的。提出一种 opt-out signal 的标准格式，并与 robots.txt 和 C2PA “No AI Training” 比较。

3. 阅读 Data Provenance Initiative 的 “Consent in Crisis”（2024 年 7 月）。描述限制增长最快的三类内容，并论证一个经济后果。

4. 2025 年 DPA 对齐接受 public-content training 的 legitimate interest。构造一个 legitimate interest 不足的场景，并识别 provider 需要的替代法律依据。

5. 勾勒一个训练数据出处 manifest，使其能与 AB 2013 字段和每个数据集的 C2PA-signed provenance chain 组合。识别一个技术障碍和一个法律障碍。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| AB 2013 | “California law” | 生成式 AI 训练数据透明度；12 个强制字段 |
| TDM exception | “text-and-data-mining” | 带 opt-out 的 EU Copyright Directive 训练数据例外 |
| Legitimate interest | “EU basis” | GDPR Article 6 法律依据，可能正当化在 public content 上训练 |
| Opt-out signal | “machine-readable no-train” | robots.txt、C2PA “No AI Training”、TDM.Reservation |
| Irreversibility | “无法 un-train” | 模型权重中的数据无法外科式移除 |
| Unlearning | “近似移除” | 降低模型对特定数据依赖的训练后干预 |
| Consent in Crisis | “DPI audit” | 2024 年 7 月关于 robots.txt 限制加速增长的发现 |

## 延伸阅读

- [California AB 2013](https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240AB2013) — 生成式 AI 训练数据透明度法律
- [EU AI Act + GPAI Code of Practice (Lesson 24)](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai) — Copyright 章节
- [Longpre, Mahari, Lee et al. — Consent in Crisis (dataprovenance.org, July 2024)](https://www.dataprovenance.org/consent-in-crisis-paper) — DPI 审计
- [IAPP — EU Digital Omnibus GDPR amendments (2025)](https://iapp.org/news/a/eu-digital-omnibus-amendments-to-gdpr-to-facilitate-ai-training-miss-the-mark) — 监管背景
