# Direct Preference Optimization 家族

> Rafailov et al.（2023）指出，RLHF 的最优解可以用偏好数据写成闭式形式，因此你可以跳过显式 reward model，直接优化 policy。这个洞见催生了一个家族：IPO、KTO、SimPO、ORPO、BPO，每个方法都修复 DPO 的一种失败模式。到 2026 年，direct alignment algorithm 在 frontier post-training run 中比 PPO 更常见。但 Lesson 2 的过度优化曲线仍然适用：DAA 没有逃离 Goodhart，只是改变了它咬人的位置。

**类型:** Learn
**语言:** Python（stdlib，六种偏好 loss 比较器）
**先修:** Phase 18 · 01（InstructGPT）、Phase 18 · 02（Reward hacking）、Phase 10 · 08（DPO basics）
**时间:** ~75 分钟

## 学习目标

- 从带 KL 的 RLHF 最优解推导 DPO 闭式形式。
- 说出 IPO、KTO、SimPO、ORPO、BPO 分别修复 DPO 的哪种失败模式。
- 区分 “implicit reward gap” 和 “preference strength”，并解释 IPO 的 identity mapping 为什么重要。
- 解释为什么 Rafailov et al.（NeurIPS 2024）证明：即使没有显式 RM，DAA 仍会过度优化。

## 要解决的问题

RLHF 目标（Lesson 1）：

```text
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

有一个已知最优解：

```text
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

因此 reward 可以由最优 policy 与 reference 的比例隐式定义：

```text
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

把它代入 Bradley-Terry 偏好似然时，partition function `Z(x)` 会抵消，因为它只依赖 `x`。剩下的是仅关于 policy 参数的 loss，不需要 reward model。这就是 DPO。

麻烦在于：推导假设最优解可达、偏好数据在分布内、reference policy 是真正的 mode anchor。这些都不完全成立。每个家族成员都在修复一个被违反的假设。

## 核心概念

### DPO（Rafailov et al., 2023）

```text
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

可能出错的地方：

- implicit reward gap `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)` 无界。一个很小的偏好也可能产生任意大的 gap。
- loss 会把 chosen 和 rejected 的 log-prob 向相反方向推动。只要 rejected 下降得更快，它可以把 chosen 的绝对 log-prob 也推低。这就是 Degraded Chosen Response 现象。
- 分布外偏好（稀有 pair vs 稀有 pair）会产生任意 implicit reward。

### IPO（Azar et al., 2024）

Identity Preference Optimization 把 log-sigmoid 替换为偏好概率上的 identity mapping。loss 变成有界目标上的 squared-error：

```text
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

margin 被 `1/(2 beta)` 限制。Preference strength 与 implicit-reward gap 成比例。不会爆炸。

### KTO（Ethayarajh et al., 2024）

Kahneman-Tversky Optimization 完全放弃成对结构。给定一个单独标注输出和二元 “desirable” 或 “undesirable” 信号，它映射到一个 prospect-theory utility：

```text
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

并对收益和损失使用不同权重（loss aversion）。好处是：你可以使用未成对数据，而这种数据丰富得多。

### SimPO（Meng et al., 2024）

Simple Preference Optimization 让训练信号与生成对齐。完全移除 reference policy，并按长度归一化 log-likelihood：

```text
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

用 margin `gamma` 稳定训练。长度归一化移除了利用 DPO 长度偏置失败模式的动机（更长的 `y_w` 按构造会给出更大的 log-prob gap）。

### ORPO（Hong et al., 2024）

Odds-Ratio Preference Optimization 给标准 SFT negative log-likelihood 加一个偏好项：

```text
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

没有 reference policy，SFT 项就是 regularizer。从 base model 到 aligned model 单阶段训练。不需要单独的 SFT checkpoint。

### BPO（ICLR 2026 submission, OpenReview id=b97EwMUWu7）

它识别了 Degraded Chosen Responses 问题：DPO 保持排序 `y_w > y_l`，但 `y_w` 的绝对 log-prob 可以下降。BPO 加入一个单行修正，惩罚 chosen response 的向下移动。论文报告在 Llama-3.1-8B-Instruct 数学推理上，相对 DPO 准确率 +10.1%。

### 通用结果：DAA 仍然会过度优化

Rafailov et al. “Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms”（NeurIPS 2024）在多个数据集和 KL budget 下用 DPO、IPO、SLiC 训练 policy。gold-reward-vs-KL 曲线有与 Gao et al. 相同的 peak-and-collapse 形状。implicit reward 在训练期间查询分布外样本；KL regularization 无法稳定这一点。

DAA 没有逃离 Goodhart。它们只是把咬人的表面从“reward model 被过度优化”变成“reference policy ratio 被过度优化”。通用修复，也就是更好数据、ensembles、early stopping，对两者都适用。

### 如何选择（2026）

- 如果你有大量成对偏好数据：使用保守 beta 的 DPO；如果长度偏置明显，使用 SimPO。
- 如果你有未成对的二元反馈：KTO。
- 如果你想从 base model 单阶段训练到 aligned model：ORPO。
- 如果你在 DPO 日志中看到 degraded chosen log-probs：BPO。
- 如果 preference strength 差异很大且 DPO 正在饱和：IPO。

每个实验室都会在一组 battery 上跑这五个方法，并按任务选择赢家。数学推理和安全任务的最优方法没有理由相同。

## 实际使用

`code/main.py` 在一个玩具偏好数据集上比较六种 loss（DPO、IPO、KTO、SimPO、ORPO、BPO），其中每个 pair 的真实 preference strength 不同。每种 loss 都在同一个 500-pair 样本上用小型 softmax policy 优化。它会绘制每种方法的最终 win rate、chosen-log-prob drift 和 implicit-reward spread。

## 交付成果

本课产出 `outputs/skill-preference-loss-selector.md`。给定数据集统计（paired vs unpaired、variable vs uniform preference strength、length distribution）和目标（single-stage 或 SFT-then-preference），它会推荐一个 preference loss，并报告它防护的失败模式。

## 练习

1. 运行 `code/main.py`。报告 DPO 和 BPO 的最终 chosen-log-prob drop。BPO 应保留更高的 chosen 绝对概率，请验证。

2. 修改偏好数据，让所有 pair 具有相同 strength。六种方法中哪个最鲁棒？哪个退化？解释 IPO 在这里的优势。

3. 让 rejected response 平均比 chosen 长 2x。不改其他内容，数值展示 DPO 的长度利用以及 SimPO 的修复。

4. Rafailov et al.（NeurIPS 2024）声称 DAA 会过度优化。复现一个单点版本：绘制 chosen-minus-rejected KL divergence，并观察大 beta 下 DPO 的过度优化。

5. 阅读 BPO 论文摘要（OpenReview b97EwMUWu7）。写下 BPO 给 DPO 增加的一行修正。与 `code/main.py` 中的实现核对。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| DPO | “没有 reward model 的 RLHF” | 从 RLHF 闭式最优解推导出的 loss；只含 policy 参数 |
| Implicit reward | “log-ratio” | `beta * log(pi(y\|x) / pi_ref(y\|x))`，DPO 隐含的 reward |
| IPO | “有界 DPO” | 用 identity 替换 log-sigmoid；implicit reward gap 被 `1/(2 beta)` 截断 |
| KTO | “未成对 DPO” | 对单标签使用 prospect-theory utility，并带 loss aversion |
| SimPO | “无 reference 的 DPO” | 长度归一化 log-likelihood + margin；没有 reference policy |
| ORPO | “单阶段 DPO” | NLL + odds-ratio 偏好项；一次从 base model 训练完成 |
| BPO | “保留 chosen 的 DPO” | DPO 加上对降低 chosen response 绝对 log-prob 的惩罚 |
| Degraded Chosen | “chosen 下降” | 只要 rejected 下降更快，DPO 就会降低 chosen log-prob |
| DAA | “direct alignment algorithm” | 任何跳过显式 RM 的 preference-loss 方法 |

## 延伸阅读

- [Rafailov et al. — Direct Preference Optimization (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Azar et al. — A General Theoretical Paradigm to Understand Learning from Human Preferences (AISTATS 2024, arXiv:2310.12036)](https://arxiv.org/abs/2310.12036) — IPO
- [Ethayarajh et al. — KTO: Model Alignment as Prospect Theoretic Optimization (arXiv:2402.01306)](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen — SimPO (NeurIPS 2024, arXiv:2405.14734)](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne — ORPO (EMNLP 2024, arXiv:2403.07691)](https://arxiv.org/abs/2403.07691)
- [BPO — Behavior Preservation Optimization (ICLR 2026 OpenReview b97EwMUWu7)](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov et al. — Scaling Laws for RM Overoptimization in DAAs (NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900)
