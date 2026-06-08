# 近端策略优化（PPO）

> A2C 每次更新后都会丢弃 rollout。PPO 用裁剪后的重要性比率包住策略梯度，让你可以在同一批数据上跑 10+ 个 epoch，而不会让策略爆炸。Schulman 等人（2017）。到 2026 年，它仍然是默认的策略梯度算法。

**类型:** Build
**语言:** Python
**先修:** Phase 9 · 06（REINFORCE），Phase 9 · 07（Actor-Critic）
**时间:** ~75 分钟

## 要解决的问题

A2C（Lesson 07）是 on-policy：梯度 `E_{π_θ}[A · ∇ log π_θ]` 需要从*当前* `π_θ` 采样的数据。做一次更新后，`π_θ` 就变了；你刚用过的数据现在变成了 off-policy。复用它会让梯度产生偏差。

Rollout 很昂贵。在 Atari 上，8 个环境 × 128 步的一次 rollout = 1024 条 transition，还要花十几秒环境时间。一次梯度步之后就把它丢掉，非常浪费。

Trust Region Policy Optimization（TRPO，Schulman 2015）是第一个修复方案：约束每次更新，使旧策略和新策略之间的 KL divergence 保持在 `δ` 以下。理论上干净，但每次更新都需要一次 conjugate-gradient 求解。2026 年已经没人跑 TRPO 了。

PPO（Schulman 等人 2017）用简单的裁剪目标替代硬性的 trust-region 约束。多一行代码。同一个 rollout 上跑十个 epoch。没有 conjugate gradients。理论保证足够好。九年后，它仍然是默认的策略梯度算法，从 MuJoCo 到 RLHF 都在用。

## 核心概念

![PPO 裁剪替代目标：在 1 ± ε 处裁剪 ratio](../assets/ppo.svg)

**重要性比率。**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

这是新策略相对于采集数据的旧策略的 likelihood ratio。`r_t = 1` 表示没有变化。`r_t = 2` 表示新策略采取 `a_t` 的概率是旧策略的两倍。

**裁剪替代目标。**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

两个项：

- 如果 advantage `A_t > 0`，且 ratio 试图增长超过 `1 + ε`，裁剪会让梯度变平：不要把一个好动作推到旧概率以上超过 `+ε`。
- 如果 advantage `A_t < 0`，且 ratio 试图增长超过 `1 - ε`（意思是相对于裁剪后的降低幅度，我们会让一个坏动作更可能发生），裁剪会限制梯度：不要把一个坏动作推到低于 `-ε`。

`min` 处理另一个方向：如果 ratio 已经朝着*有利*方向移动，你仍然能得到梯度（在会伤害你的那一侧不裁剪）。

典型的 `ε = 0.2`。把目标画成 `r_t` 的函数：它是一个分段线性函数，在“好的一侧”有平屋顶，在“坏的一侧”有平地板。

**完整 PPO 损失。**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

和 A2C 一样的 actor-critic 结构。三个系数，通常是 `c_v = 0.5`、`c_e = 0.01`、`ε = 0.2`。

**训练循环。**

1. 在 `N` 个并行环境中各采集 `T` 步，得到 `N × T` 条 transition。
2. 计算 advantages（GAE），并把它们冻结为常量。
3. 把 `π_{θ_old}` 冻结为当前 `π_θ` 的快照。
4. 对每个 minibatch `(s, a, A, V_target, log π_old(a|s))` 跑 `K` 个 epoch：
   - 计算 `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`。
   - 应用 `L^{CLIP}` + value loss + entropy。
   - 做一次梯度步。
5. 丢弃这个 rollout。回到第 1 步。

`K = 10` 和大小为 64 的 minibatch 是一组标准超参数。PPO 很鲁棒：精确数字在 ±50% 范围内通常不太重要。

**KL-penalty 变体。** 原论文提出了一个替代方案，使用自适应 KL penalty：`L = L^{PG} - β · KL(π_θ || π_old)`，并根据观察到的 KL 调整 `β`。裁剪版本后来占主导；KL 变体保留在 RLHF 中（因为那里到 reference policy 的 KL 本来就是你始终想要的单独约束）。

## 动手实现

### Step 1：在 rollout 时捕获 `log π_old(a | s)`

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

快照只在 rollout 时取一次。它在更新 epoch 期间不会变化。

### Step 2：计算 GAE advantages（Lesson 07）

和 A2C 相同。在 batch 内归一化。

### Step 3：裁剪替代目标更新

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

“裁剪 → 零梯度”模式是 PPO 的核心。如果新策略已经在有利方向漂移得太远，更新就会停止。

### Step 4：value 和 entropy

向 critic target 加标准 MSE，并给 actor 加 entropy bonus，和 A2C 相同。

### Step 5：诊断指标

每次更新要观察三件事：

- **Mean KL** `E[log π_old - log π_θ]`。应保持在 `[0, 0.02]`。如果冲过 `0.1`，降低 `K_EPOCHS` 或 `LR`。
- **Clip fraction**：ratio 落在 `[1-ε, 1+ε]` 之外的样本比例。应为 `~0.1-0.3`。如果接近 `~0`，说明裁剪从不触发 → 提高 `LR` 或 `K_EPOCHS`。如果 `~0.5+`，说明你在过拟合 rollout → 降低它们。
- **Explained variance** `1 - Var(V_target - V_pred) / Var(V_target)`。critic 质量指标。随着 critic 学习，应朝 1 上升。

## 常见陷阱

- **Clip coefficient 调错。** `ε = 0.2` 是事实标准。调到 `0.1` 会让更新过于胆小；`0.3+` 会招来不稳定。
- **Epoch 太多。** `K > 20` 经常会让训练不稳定，因为策略会远离 `π_old`。限制 epoch，尤其是大网络。
- **没有 reward normalization。** 大 reward scale 会吞掉裁剪范围。在计算 advantages 前对 rewards 做归一化（running std）。
- **忘记 advantage normalization。** 每个 batch 做零均值/单位标准差归一化是标准做法。跳过它会毁掉 PPO 在多数 benchmark 上的表现。
- **Learning rate 没有衰减。** PPO 受益于线性 LR 衰减到零。常数 LR 往往更差。
- **重要性比率数学错误。** 为了数值稳定，始终用 `exp(log_new - log_old)`，不要用 `new / old`。
- **梯度符号错误。** 最大化 surrogate = *最小化* `-L^{CLIP}`。符号翻转是最常见的 PPO bug。

## 实际使用

PPO 是 2026 年很多领域的默认 RL 算法，范围令人意外：

| 用例 | PPO 变体 |
|------|----------|
| MuJoCo / robotics control | 带 Gaussian policy 的 PPO，GAE(0.95) |
| Atari / discrete games | 带 categorical policy 的 PPO，滚动 128-step rollout |
| LLM 的 RLHF | 对 reference model 加 KL penalty 的 PPO，response 末尾来自 RM 的 reward |
| 大规模游戏 agent | IMPALA + PPO（AlphaStar，OpenAI Five） |
| 推理 LLM | GRPO（Lesson 12）：无 critic 的 PPO 变体 |
| 只有 preference 的数据 | DPO：PPO+KL 的闭式折叠，无 online sampling |

PPO 的*损失形状*（clipped surrogate + value + entropy）是 DPO、GRPO 和几乎所有 RLHF pipeline 的脚手架。

## 交付成果

保存为 `outputs/skill-ppo-trainer.md`：

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## 练习

1. **简单。** 在 4×4 GridWorld 上运行 PPO，设置 `ε=0.2, K=4`。在匹配的环境步数下，和 A2C（每个 rollout 一个 epoch）比较 sample efficiency。
2. **中等。** 扫描 `K ∈ {1, 4, 10, 30}`。绘制 return vs env steps，并跟踪每次更新的 mean KL。在这个任务上，KL 从哪个 `K` 开始爆炸？
3. **困难。** 用自适应 KL penalty 替换裁剪替代目标（如果 `KL > 2·target`，`β` 翻倍；如果 `KL < target/2`，`β` 减半）。比较最终 return、稳定性和无裁剪程度。

## 关键术语

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| Importance ratio | “r_t(θ)” | `π_θ(a\|s) / π_old(a\|s)`；相对于采集数据的策略的偏离程度。 |
| Clipped surrogate | “PPO 的主要技巧” | `min(r·A, clip(r, 1-ε, 1+ε)·A)`；在有利侧越过裁剪后梯度变平。 |
| Trust region | “TRPO / PPO 的意图” | 限制每次更新的 KL，以保证单调改进。 |
| KL penalty | “软 trust region” | 另一种 PPO：`L - β · KL(π_θ \|\| π_old)`。自适应 `β`。 |
| Clip fraction | “裁剪触发频率” | 诊断指标：应为 0.1-0.3；超出说明调参不当。 |
| Multi-epoch training | “数据复用” | 每个 rollout 上跑 K 个 epoch；用方差代价换取 sample efficiency。 |
| On-policy-ish | “基本 on-policy” | PPO 名义上是 on-policy，但 K>1 个 epoch 会安全地使用稍微 off-policy 的数据。 |
| PPO-KL | “另一个 PPO” | KL-penalty 变体；用于 RLHF，因为到 reference 的 KL 本来就是一个约束。 |

## 延伸阅读

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)：论文原文。
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)：TRPO，PPO 的前身。
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990)：对每个 PPO 超参数做消融。
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)：InstructGPT；RLHF 中使用 PPO 的配方。
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)：清晰的现代 PyTorch 讲解。
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl)：许多论文使用的单文件 PPO 参考实现。
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer)：语言模型上 PPO 的生产配方；配合 Lesson 09（RLHF）一起读。
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729)：那篇“37 个代码级优化”的论文；说明哪些 PPO 技巧是承重结构，哪些只是民间经验。
