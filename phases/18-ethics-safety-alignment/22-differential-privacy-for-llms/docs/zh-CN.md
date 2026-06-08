# LLM 的差分隐私

> DP-SGD 仍然是标准做法：注入噪声的梯度更新提供形式化的 (epsilon, delta) 保证。计算、内存和效用开销都很大；参数高效的 DP 微调（LoRA + DP-SGD）是常见的 2025 配置（ACM 2025）。两类证据彼此紧张：基于 canary 的成员推断（Duan 等，2024）报告对语言模型成功有限；训练数据抽取（Carlini 等，2021；Nasr 等，2025）恢复了大量逐字记忆。解决（arXiv:2503.06808，2025 年 3 月）：差距来自测量对象不同，即插入的 canary 与“最可抽取”数据。新的 canary 设计支持无需 shadow model 的基于 loss 的 MIA，并给出了第一个针对真实数据训练且带现实 DP 保证的 LLM 的非平凡 DP 审计。替代方案：PMixED（arXiv:2403.15638），在推理时通过 next-token 分布上的专家混合进行私有预测；DP 合成数据生成（Google Research 2024）。新兴攻击：通过 LLM Feedback 进行 Differential Privacy Reversal，即置信度分数泄漏。

**类型：** 构建
**语言：** Python (stdlib, DP-SGD noise-injection and ε-δ accountant demonstration)
**先修：** 第 01 阶段 · 第 09 课（information theory），第 10 阶段 · 第 01 课（large-model training）
**时间：** 约 60 分钟

## 学习目标

- 定义 (epsilon, delta)-差分隐私，并说出 DP-SGD 配方。
- 解释 2024-2025 年的紧张关系：canary MIA 与训练数据抽取给出不同图景。
- 描述 PMixED，以及为什么推理时私有预测是 DP 训练的替代方案。
- 描述通过 LLM Feedback 进行 Differential Privacy Reversal 的攻击。

## 要解决的问题

LLM 会记忆。Carlini 等 2021 证明，生产语言模型会按需逐字复现训练文本。DP 是形式化防御：训练时使输出可证明地对任意单个训练样本不敏感。2024-2025 年的证据显示，DP-SGD 是必要的，但部署中的 ε 值可能并不匹配威胁模型。

## 核心概念

### (ε, δ)-差分隐私

如果对任意两个只相差一个样本的数据集，以及任意事件 S，随机算法 M 都满足：
P(M(D) in S) <= e^ε * P(M(D') in S) + δ，
则 M 是 (ε, δ)-DP。

解释：输出分布足够接近（由 ε 参数化），因此任意单个个体的贡献都无法被可靠推断，除非以 δ 的概率发生例外。

### DP-SGD

Abadi 等 2016。标准配方：
1. 采样一个 mini-batch。
2. 计算逐样本梯度。
3. 将每个逐样本梯度裁剪到阈值 C。
4. 对裁剪后的梯度求和，并添加标准差为 σ * C 的 Gaussian noise。
5. 使用带噪声的和更新参数。

隐私成本由 accountant 跟踪（Moments Accountant、Rényi DP accountant）。LLM 文献中报告的 ε 值会随威胁模型、数据敏感性和效用目标而大幅变化；不存在普遍“安全”的默认 ε。已发表示例在某些 LLM 训练设置中大致覆盖 ε ≈ 1–10，但这些只是说明性数值，不是推荐默认值。更低的 ε 通常需要更多噪声，并可能增加效用损失。

### LoRA + DP-SGD

对前沿模型执行完整 DP-SGD 代价过高。LoRA（Hu 等 2022）把梯度更新限制在一个小 adapter 上，从而减少逐样本梯度存储。LoRA + DP-SGD 是常见的 2025 配置。DP 保证适用于 adapter；base model 保持固定。

### 2024-2025 年的紧张关系

两条证据线：

- **Canary MIA（Duan 等 2024）。** 向训练数据插入唯一 canary，测量成员推断攻击者是否能识别它们。报告显示对语言模型成功有限。这暗示 MIA 很难。
- **训练数据抽取（Carlini 2021，Nasr 等 2025）。** 用前缀提示模型；测量它是否恢复训练中的逐字文本。报告显示存在大量记忆。这暗示在相关意义上 MIA 很容易。

2025 年 3 月的解决（arXiv:2503.06808）：二者测量的是不同问题。MIA 在插入的 canary 上问“样本 e 是否在 D 中？”抽取则问“我能从 D 中恢复什么？”对隐私真正重要的是“最可抽取”的样本；canary 会低估这一点，因为它们并未被优化成最可抽取。

新的 canary 设计。无需 shadow model 的基于 loss 的 MIA。首次对真实数据上训练、且带现实 DP 保证的 LLM 进行非平凡 DP 审计。

### DP 训练的替代方案

- **PMixED（arXiv:2403.15638）。** 推理时私有预测。基于 next-token 分布的 mixture of experts；每个 expert 看到训练数据的一个分片；聚合时加入 DP 噪声。完全避开 DP 训练。
- **DP 合成数据生成（Google Research 2024）。** 使用 DP-SGD 做 LoRA 微调，采样合成数据，再在合成数据上训练下游分类器。

二者都绕开了完整 DP 训练的效用成本，但代价是采用不同的威胁模型。

### 通过 LLM Feedback 进行 Differential Privacy Reversal

2025 年新兴攻击。把 DP 训练模型的置信度分数当作 oracle，用于重新识别个体。即使输出本身不泄漏，置信度分布也可能泄漏。

防御：不要暴露置信度，或在暴露前截断 / 量化它们。这是 (ε, δ)-DP 训练之外的额外要求。

### 它在第 18 阶段中的位置

第 20-21 课是偏见 / 公平性。第 22 课是隐私。第 23 课是通过水印实现出处证明。第 27 课覆盖监管所需的数据出处层。

## 实际使用

`code/main.py` 在玩具二分类数据集上模拟 DP-SGD。你可以扫描噪声乘数 σ 和裁剪范数 C，跟踪 (ε, δ) 预算和准确率成本。一个 “canary attack” 会插入唯一训练样本，并测量 log-loss 测试在 DP 前后是否能检测出它。

## 交付成果

本课产出 `outputs/skill-dp-audit.md`。给定对某个语言模型部署的 DP 声明，它会审计：(ε, δ) 数值、使用的 accountant、MIA 评估协议，以及是否评估了置信度暴露向量。

## 练习

1. 运行 `code/main.py`。在 {0.5, 1.0, 2.0} 中扫描 σ，并报告 (ε, δ)-准确率权衡。识别效用崩塌的位置。

2. 实现 canary 插入和 log-loss 测试。测量 σ = 1.0 时 DP-SGD 前后的检测率。

3. 阅读 Nasr 等 2025 关于训练数据抽取的论文。为什么在中等 ε 下抽取成功率不会崩塌？这对把 MIA 当作评估意味着什么？

4. 设计一个完全在推理时运行的 PMixED（arXiv:2403.15638）部署。PMixED 处理的、而 DP-SGD 不处理的威胁模型是什么？

5. 勾勒通过 LLM Feedback 进行 DP Reversal 的攻击。设计一种限制置信度分数泄漏的对策，并估计其部署成本。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| DP | “(ε, δ)-differential privacy” | 形式化隐私：相邻数据集变化下输出分布接近 |
| DP-SGD | “注入噪声的 SGD” | 梯度裁剪 + Gaussian 噪声添加；标准 DP 训练 |
| LoRA + DP-SGD | “高效私有微调” | 在低秩 adapter 上执行 DP-SGD；2025 标准配置 |
| MIA | “membership inference” | 判断某个样本是否在训练数据中的攻击 |
| Canary | “插入的水印样本” | 用于测量 DP 泄漏的唯一训练样本 |
| PMixED | “私有推理混合” | 通过 next-token 分布上的 mixture-of-experts 在推理时实现 DP |
| DP Reversal | “置信度泄漏攻击” | 使用模型置信度作为重新识别 oracle 的攻击 |

## 延伸阅读

- [Abadi et al. — DP-SGD (arXiv:1607.00133)](https://arxiv.org/abs/1607.00133) — 标准 DP 训练算法
- [Carlini et al. — Extracting Training Data (arXiv:2012.07805)](https://arxiv.org/abs/2012.07805) — 经典抽取论文
- [Duan et al. — Canary MIA on LLMs (arXiv:2402.07841, 2024)](https://arxiv.org/abs/2402.07841) — 成功有限的 MIA
- [Kowalczyk et al. — Auditing DP for LLMs (arXiv:2503.06808, March 2025)](https://arxiv.org/abs/2503.06808) — 紧张关系的解决
- [PMixED (arXiv:2403.15638)](https://arxiv.org/abs/2403.15638) — 推理时私有预测
