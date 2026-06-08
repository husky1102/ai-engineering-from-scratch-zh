# 指令遵循作为对齐信号

> 后续每一种对 RLHF 的批评，都是在反驳这条流水线。研究优化压力如何扭曲代理目标之前，你必须先看清这个代理目标。InstructGPT（Ouyang et al., 2022）定义了参考架构：在指令-回答对上做监督微调，用成对偏好排序训练 reward model，再用带 SFT policy KL 惩罚的 PPO 优化 reward model。1.3B InstructGPT 比 175B GPT-3 更受偏好。正是这个单一结果，让 2026 年每个 frontier lab 仍然交付 RLHF 形状的后训练流水线。

**类型:** Learn
**语言:** Python（stdlib，玩具三阶段流水线）
**先修:** Phase 10 · 06（SFT）、Phase 10 · 07（RLHF）、Phase 10 · 08（DPO）
**时间:** ~45 分钟

## 学习目标

- 说出 InstructGPT 流水线的三个阶段，以及每个阶段使用的 loss。
- 解释为什么一个 1.3B 指令微调模型能在人类偏好评估中击败原始 175B GPT-3。
- 说明第 3 阶段的 KL 惩罚在防止什么，以及去掉它为什么会坍缩成 mode-seeking 行为。
- 描述 alignment tax，以及 Ouyang et al. 用来缓解它的 PPO-ptx。

## 要解决的问题

预训练语言模型会补全文本。它们不会回答问题。问 GPT-3 “write a Python function that reverses a list”，你常常会得到另一个 prompt，因为大部分训练分布是继续接着写的网页文本。模型正在完成自己的工作，只是这个工作本身错了。

每个严肃实验室用来修复它的代理信号都是人类偏好。两个 completion 交给标注者；标注者选择更好的那个；reward model 学习标注者。然后一个 RL 循环把 policy 推向 reward model 打高分的输出。三句话就是完整的 InstructGPT 论点。论文剩下的部分是工程。

## 核心概念

### 阶段 1：监督微调（SFT）

收集 prompt-response 对，其中 response 是一个善意人类会写出的回答。Ouyang et al. 使用了来自标注者和 OpenAI API 的 13k 个 prompt。用标准交叉熵 loss 在这些数据上微调 base model。

SFT 给你的东西：模型现在会回答问题，而不是继续补全问题。它不给你的东西：当多个答案都合理时，哪个答案更受标注者偏好的信号。

### 阶段 2：reward model（RM）

对每个 prompt，从 SFT model 采样 K 个 completion。标注者给它们排序。训练一个 reward model，为任意 prompt-response 对打分，使得对于 `y_w` 优于 `y_l` 的 pair：

```text
L_RM = -log sigmoid(r(x, y_w) - r(x, y_l))
```

这是 Bradley-Terry 成对偏好 loss。RM 通常从 SFT model 初始化，把 LM head 替换为标量 head。

Reward model 很小：6B 对 175B InstructGPT 已经足够。它们也很脆弱，论文第 5 节大部分都在讨论小规模时出现的 reward-hacking 行为。

### 阶段 3：带 KL 惩罚的 PPO

定义目标：

```text
J(pi) = E_{x~D, y~pi(.|x)} [ r(x, y) ] - beta * KL(pi(.|x) || pi_SFT(.|x))
```

用 PPO 最大化它。KL 项让 `pi` 不会漂离 SFT policy 太远。没有它，优化器会找到对抗样本，也就是在 RM 下得分很高的字符串；高分原因不是人类真的偏好它们，而是 RM 从没见过它们。

KL 系数 `beta` 是最重要的 RLHF 超参数。太低：reward hacking。太高：相对 SFT 没有改进。

### Alignment tax

经过 RLHF 后，模型更受人类偏好，但在标准 benchmark（SQuAD、HellaSwag、DROP）上回退。Ouyang et al. 把这称为 alignment tax，并用 PPO-ptx 修复：把预训练梯度混入 RL 目标，让模型不要忘记如何完成那些从未被 reward 覆盖的下游任务。

```text
J_ptx(pi) = J(pi) + gamma * E_{x~D_pretrain} [ log pi(x) ]
```

PPO-ptx 后来成为标准。Anthropic、DeepMind 和 Meta 都使用某种变体。

### 结果

1.3B InstructGPT（SFT + RM + PPO-ptx）被标注者偏好于 175B base GPT-3，比例约为 70%。在来自生产流量的 hidden-test prompt 上，差距更大。这个数字可以读出两件事：

1. 对齐是不同于能力的另一个轴。175B 模型有更多能力；1.3B 模型有更多对齐；标注者偏好已对齐的那个。
2. 能力下限由 base model 设定。你不能靠 RLHF 让 base model 知道它从未见过的事实。

### 为什么这是 Phase 18 的参考点

后续课程里的每一种批评，包括 reward hacking（Lesson 2）、DPO（Lesson 3）、sycophancy（Lesson 4）、CAI（Lesson 5）、sleeper agents（Lesson 7）、alignment faking（Lesson 9），都在反驳这条流水线的某个部分。Reward hacking 攻击阶段 2。DPO 折叠阶段 2 和 3。CAI 替换人类标注者。Sycophancy 显示标注者是一个有偏信号。Alignment faking 显示 policy 可以完全绕开阶段 3。没有先把这条流水线装进脑中，你无法跟上这些批评。

## 实际使用

`code/main.py` 在玩具偏好数据上模拟三个阶段。base “policy” 是动作 {A, B, C} 上的一枚有偏硬币。阶段 1 SFT 在 200 个 prompt 上模仿标注者动作。阶段 2 用 500 个成对排序拟合 Bradley-Terry reward model。阶段 3 运行一个带 SFT policy KL 惩罚的简化 PPO 更新。你可以观察 reward 上升、KL divergence 变大、policy 漂移，也可以关闭 KL 项，在 50 个更新 step 内看到 reward hacking 出现。

观察重点：

- `beta = 0.1` 与 `beta = 0.0` 下的 reward 轨迹。
- 训练 step 中的 KL(pi || pi_SFT)。
- 与标注者偏好相比的最终动作分布。

## 交付成果

本课产出 `outputs/skill-instructgpt-explainer.md`。给定一条 RLHF 流水线描述或论文摘要，它会识别三阶段中哪一阶段被修改、每个阶段使用什么 loss，以及是否存在 KL 惩罚或等价正则项。

## 练习

1. 运行 `code/main.py`。把 `beta = 0.0`，报告 200 个 PPO step 后的动作分布。用一段话解释 mode-seeking 行为。

2. 修改 reward model，让它对动作 B 有 +0.5 偏置（模拟 reward bug）。用 `beta = 0.1` 运行 PPO。KL 惩罚是否阻止了 policy 利用这个偏置？在什么 `beta` 下 exploitation 变得可见？

3. 阅读 Ouyang et al.（arXiv:2203.02155）Figure 1。通过运行 PPO 1、5、20、100 个 step，并测量相对 SFT model 的偏好，复现标注者偏好曲线。

4. 论文 Section 4.3 报告 1.3B InstructGPT 大约 70% 的时候击败 175B GPT-3。为什么这个比例在 hidden production prompt 上会高于标注者自己的 prompt？

5. 在同一偏好数据上用 DPO（Phase 10 · 08）替换 PPO loss。比较最终 policy drift（到 SFT 的 KL）和最终 reward。在匹配 reward 时，哪种方法漂移更远？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| SFT | “instruction tuning” | 阶段 1：在 prompt-response 对上做交叉熵微调 |
| Reward model | “RM” | 在 (prompt, response) 上的标量回归器，用 Bradley-Terry 从成对标签训练 |
| Bradley-Terry | “成对偏好 loss” | -log sigmoid(r_w - r_l)；把成对排序归约为二分类 |
| KL penalty | “正则项” | `beta * KL(pi \|\| pi_SFT)`，让 RL policy 靠近 SFT 锚点 |
| PPO-ptx | “带预训练混合的 PPO” | 给 PPO 目标加入一部分预训练 log-likelihood，以抵消 alignment tax |
| Alignment tax | “RLHF 回退” | RLHF 后在 RLHF 未针对的标准 benchmark 上下降 |
| Labeler preference | “ground truth” | 人类排序样本；RM 是它的统计代理，不是“human values”的代理 |

## 延伸阅读

- [Ouyang et al. — Training language models to follow instructions with human feedback (arXiv:2203.02155)](https://arxiv.org/abs/2203.02155) — InstructGPT 论文，后续每条 RLHF 流水线的基础
- [Stiennon et al. — Learning to summarize from human feedback (arXiv:2009.01325)](https://arxiv.org/abs/2009.01325) — 早期 RLHF 摘要任务前身
- [Christiano et al. — Deep reinforcement learning from human preferences (arXiv:1706.03741)](https://arxiv.org/abs/1706.03741) — 原始偏好式 RL 形式化
- [Bai et al. — Training a Helpful and Harmless Assistant with RLHF (arXiv:2204.05862)](https://arxiv.org/abs/2204.05862) — Anthropic 对 InstructGPT 流水线的 HH 扩展
