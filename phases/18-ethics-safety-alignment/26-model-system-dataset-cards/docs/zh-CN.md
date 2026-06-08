# 模型卡、系统卡和数据集卡

> 三种文档格式构成了 AI 透明度。Model Cards（Mitchell 等，2019）：模型的营养标签，记录训练数据、量化分组分析、伦理考量、注意事项；只有 0.3% 的 Hugging Face model cards 记录伦理考量（Oreamuno 等，2023）。Datasheets for Datasets（Gebru 等，2018，CACM）：动机、组成、收集过程、标注、分发、维护；类比电子元件 datasheet。Data Cards（Pushkarna 等，Google 2022）：模块化分层细节（telescopic、periscopic、microscopic），作为面向多种读者的边界对象。2024-2025 进展：通过 LLM 自动生成（CardGen，Liu 等，2024）；model-card 细节与 HF 上最高 29% 下载量增长相关（Liang 等，2024）；可验证 attestations（Laminator，Duddu 等，2024）；加入碳 / 水的可持续性报告（Jouneaux 等，2025 年 7 月）；EU/ISO 监管 cards 正在出现。System Cards（Sidhpurwala 2024；Meta 系统级透明度；“Blueprints of Trust” arXiv:2509.20394）：端到端 AI 系统文档，覆盖安全能力、prompt-injection 保护、data-exfiltration 检测、与人类价值一致性。

**类型：** 构建
**语言：** Python (stdlib, model-card + datasheet + system-card generator)
**先修：** 第 18 阶段 · 第 18 课（safety frameworks），第 18 阶段 · 第 24 课（regulatory）
**时间：** 约 60 分钟

## 学习目标

- 描述最初的 Mitchell 等 2019 model card 和 Gebru 等 2018 datasheet。
- 描述 Data Cards 的 telescopic / periscopic / microscopic 分层。
- 描述 System Cards 及其端到端覆盖范围。
- 说出三项 2024-2025 进展（自动生成、可验证 attestations、可持续性报告）。

## 要解决的问题

监管框架（第 24 课）和实验室安全政策（第 18 课）都要求文档。文档格式从面向模型（model cards）演化到面向数据集（datasheets），再到面向系统（system cards）。每一种都处理不同范围的透明度。2024-2025 年的自动化和可验证 attestation 工作，回应了长期存在的采用问题。

## 核心概念

### Model Cards（Mitchell 等 2019）

章节：
- Model details。
- Intended use。
- Factors（用于评估的相关人口统计或环境因素）。
- Metrics。
- Evaluation data。
- Training data。
- Quantitative analyses（按 factors 分组）。
- Ethical considerations。
- Caveats and recommendations。

采用问题：Oreamuno 等 2023 对 Hugging Face model cards 的审计发现，只有 0.3% 记录了 ethical considerations。

### Datasheets for Datasets（Gebru 等 2018）

电子元件 datasheet 类比。章节：
- Motivation（为什么创建该 dataset）。
- Composition（其中包含什么）。
- Collection process（如何组装）。
- Labeling（如适用）。
- Uses（预期、禁止、风险）。
- Distribution。
- Maintenance。

2021 年发表在 CACM。datasheet 是上游文档；model card 依赖 datasheet 的准确性。

### Data Cards（Pushkarna 等，Google 2022）

模块化分层细节。三个缩放级别：
- **Telescopic。** 面向非专家的高层摘要。
- **Periscopic。** 面向 ML practitioners 的中层概览。
- **Microscopic。** 面向审计员的细粒度特征级文档。

边界对象框架：不同读者从同一份文档中抽取不同信息。

### System Cards

范围：端到端 AI 系统，包括 model + safety stack + deployment context。章节通常包括：
- Security capabilities。
- Prompt-injection protection。
- Data-exfiltration detection。
- Alignment with stated human values。
- Incident response。

Sidhpurwala 2024 和 Meta 系统级透明度工作。“Blueprints of Trust”（arXiv:2509.20394）将 System Card 形式化为 Model Cards 在部署层的补充。

### 2024-2025 进展

- **CardGen（Liu 等 2024）。** 通过 LLM 自动生成 model-card；报告称，在标准化 Mitchell 2019 字段上，它比许多人类撰写的 cards 更客观。
- **下载相关性（Liang 等 2024）。** 详细 model cards 与 HF 上最高 29% 更高下载率相关；采用压力现在不仅来自合规，也来自市场。
- **Laminator（Duddu 等 2024）。** 通过硬件 TEE / 密码学签名提供可验证 attestations，让 model card 携带 proof-of-claim，而不只是 claim。
- **可持续性（Jouneaux 等，2025 年 7 月）。** 增加碳、水和计算能源足迹；新兴 ISO 标准。
- **监管 cards。** EU AI Act（第 24 课）GPAI Code of Practice 的 Transparency 章节要求 model cards 作为合规 artifact。

### 它在第 18 阶段中的位置

第 24-25 课是监管和 CVE 层。第 26 课是文档层。第 27 课是训练数据治理，也就是 datasheet 的上游。第 28 课是产出 card 中引用评估的研究生态。

## 实际使用

`code/main.py` 为一个玩具部署生成最小 model card、datasheet 和 system card。每一个都遵循规范章节结构。你可以检查格式并比较三者范围。

## 交付成果

本课产出 `outputs/skill-card-audit.md`。给定一个 model card、datasheet 或 system card，它会审计章节覆盖、数值分组，以及是否存在可验证 attestations。

## 练习

1. 运行 `code/main.py`。检查生成的 cards。识别较弱的章节（只有占位符）并说明什么证据能加强它们。

2. 用两个人口统计群体上的量化分组分析扩展 model card（第 20 课）。

3. 阅读 Oreamuno 等 2023 关于 0.3% 采用率的论文。提出一个能提高 ethical-considerations 采用率的 model card 规范结构变更。

4. Laminator（Duddu 等 2024）使用 TEE 做可验证 attestations。设计一个 model-card 字段，用于承载评估结果的密码学 attestation，并描述 verifier 的角色。

5. 为你过去的某个项目或一个假设部署编写 System Card（System Card，而不是 Model Card）。识别对第三方审计员价值最高的章节。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Model Card | “Mitchell card” | Mitchell 等 2019 面向 ML models 的标准文档 |
| Datasheet | “Gebru datasheet” | Gebru 等 2018 面向 datasets 的标准文档 |
| Data Card | “Pushkarna card” | Google 2022 模块化分层数据文档 |
| System Card | “deployment card” | 包含 safety stack 的端到端 AI 系统文档 |
| Boundary object | “不同读者，一份文档” | Data Cards 框架：同一文档服务多元受众 |
| Verifiable attestation | “Laminator attestation” | 附加到文档声明上的密码学或 TEE 证明 |
| Sustainability field | “碳 / 水足迹” | 2025 年新兴的环境核算字段 |

## 延伸阅读

- [Mitchell et al. — Model Cards for Model Reporting (arXiv:1810.03993, FAT* 2019)](https://arxiv.org/abs/1810.03993) — 经典 model card
- [Gebru et al. — Datasheets for Datasets (CACM 2021, arXiv:1803.09010)](https://arxiv.org/abs/1803.09010) — datasheet 论文
- [Pushkarna et al. — Data Cards (Google 2022)](https://arxiv.org/abs/2204.01075) — 分层数据文档
- [Sidhpurwala et al. — Blueprints of Trust (arXiv:2509.20394)](https://arxiv.org/abs/2509.20394) — System Card 形式化
