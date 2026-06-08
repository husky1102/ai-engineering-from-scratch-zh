# Temporal Difference：Q-Learning 与 SARSA

> Monte Carlo 会等到 episode 结束。TD 通过 bootstrap 下一个 value estimate，在每一步之后就更新。Q-learning 是 off-policy 且乐观；SARSA 是 on-policy 且谨慎。二者都只有一行代码。二者也支撑着本阶段每一种 deep-RL 方法。

**类型:** Build
**语言:** Python
**先修:** Phase 9 · 01 (MDPs), Phase 9 · 02 (Dynamic Programming), Phase 9 · 03 (Monte Carlo)
**时间:** ~75 分钟

## 要解决的问题

Monte Carlo 能工作，但它有两个昂贵要求。它需要会 terminate 的 episodes，而且只有最终 return 到手后才更新。如果你的 episode 有 1,000 steps，MC 会等 1,000 steps 才更新任何东西。它 high-variance、low-bias，并且实践中慢。

Dynamic programming 是相反的特征：zero-variance bootstrapped backups，但要求 known model。

Temporal difference (TD) learning 折中二者。从单个 transition `(s, a, r, s')` 形成 one-step target `r + γ V(s')`，然后把 `V(s)` 往它推一点。无 model。无需完整 episodes。从 RHS 上使用近似 `V` 带来 bias，但 variance 比 MC 低得多，而且从第一步就能 online updates。

这是现代 RL，包括 DQN、A2C、PPO、SAC，全部转动的枢纽。Phase 9 剩余内容都是在本 lesson 你会写出的 one-step TD update 之上，叠加 function approximation 和各种 tricks。

## 核心概念

![Q-learning vs SARSA: off-policy max vs on-policy Q(s', a')](../assets/td.svg)

**V 的 TD(0) update：**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

方括号里的量是 TD error `δ = r + γ V(s') - V(s)`。它是 MC 中 `G_t - V(s_t)` 的 online 类比。收敛要求 `α` 满足 Robbins-Monro（`Σ α = ∞`, `Σ α² < ∞`），并且所有 states 被无限次访问。

**Q-learning。** 用于 control 的 off-policy TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

`max` 假设从 `s'` 之后会跟随 *greedy* policy，不管 agent 实际采取了什么 action。这种解耦让 Q-learning 在 agent 通过 ε-greedy exploration 时仍能学习 `Q*`。Mnih et al. (2015) 把它扩展成 Atari 上的 deep Q-learning（Lesson 05）。

**SARSA。** 一种 on-policy TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

名字来自 tuple `(s, a, r, s', a')`。SARSA 使用 agent 接下来 *实际* 采取的动作 `a'`，不是 greedy `argmax`。它会收敛到当前运行中的 ε-greedy `π` 对应的 `Q^π`，并在极限 `ε → 0` 时变成 `Q*`。

**Cliff-walking 差异。** 在经典 cliff-walking task（掉下悬崖 = reward -100）中，Q-learning 会学习沿悬崖边缘的最优路径，但 exploration 时偶尔吃到惩罚。SARSA 会学习离悬崖一步远的更安全路径，因为它把 exploration noise 纳入 Q-value。训练后，当 `ε → 0` 时二者都会达到 optimal。实践中这很重要：如果 deployment 时 exploration 真的还在发生，SARSA 的行为更保守。

**Expected SARSA。** 用 `π` 下的期望值替代 `Q(s', a')`：

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

比 SARSA variance 更低（不采样 `a'`），目标仍然是 on-policy。它通常是现代教材里的默认选择。

**n-step TD 与 TD(λ)。** 在 bootstrapping 前等待 `n` steps，在 TD(0) 和 MC 之间插值。`n=1` 是 TD，`n=∞` 是 MC。TD(λ) 用几何权重 `(1-λ)λ^{n-1}` 对所有 `n` 求平均。大多数 deep-RL 使用 3 到 20 之间的 `n`。

## 动手实现

### Step 1: ε-greedy policy 上的 SARSA

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

八行。和 Q-learning 的 *唯一* 区别就是 target 那一行。

### Step 2: Q-learning

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max` 把 target 与 behavior 解耦。这个符号就是 on-policy 和 off-policy 的差别。

### Step 3: learning curves

跟踪每 100 个 episodes 的 mean return。在简单确定性 GridWorld 上，Q-learning 收敛更快；在 cliff-walking 上，SARSA 更保守。在 `code/main.py` 的 4×4 GridWorld 中，用 `α=0.1, ε=0.1` 时，二者大约 2,000 episodes 后都接近最优。

### Step 4: 与 DP truth 对比

运行 value iteration（Lesson 02）得到 `Q*`。检查 `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`。一个健康的 tabular TD agent 在 4×4 GridWorld 上训练 10,000 episodes 后，通常能落在 `~0.5` 以内。

## 常见陷阱

- **Initial Q values matter。** 乐观初始化（对 negative-reward task 用 `Q = 0`）会鼓励 exploration。悲观初始化可能永远困住 greedy policy。
- **α schedule。** 对 non-stationary problems，constant `α` 很好。衰减 `α_n = 1/n` 有理论收敛，但实践中太慢。把 `α` 固定在 `[0.05, 0.3]`，并监控 learning curve。
- **ε schedule。** 从高值开始（`ε=1.0`），衰减到 `ε=0.05`。“GLIE”（greedy in the limit with infinite exploration）是收敛条件。
- **Q-learning 中的 max bias。** 当 `Q` 有噪声时，`max` operator 会向上偏。导致 overestimation。Hasselt 的 Double Q-learning（Lesson 05 的 DDQN 使用它）用两个 Q tables 修复。
- **Non-terminating episodes。** TD 可以在没有 terminals 的情况下学习，但你要么 cap steps，要么在 cap 处正确处理 bootstrap。标准做法：把 cap 当成 non-terminal，继续 bootstrapping。
- **State hashing。** 如果 states 是 tuples/tensors，使用 hashable key（tuple，不是 list；四舍五入后的 tuple of floats，不是 raw）。

## 实际使用

2026 年的 TD landscape：

| 任务 | 方法 | 原因 |
|------|--------|--------|
| Small tabular environments | Q-learning | 直接学习 optimal policy。 |
| On-policy safety-critical | SARSA / Expected SARSA | Exploration 期间更保守。 |
| High-dimensional state | DQN (Phase 9 · 05) | 带 replay 和 target net 的 neural-net Q-function。 |
| Continuous actions | SAC / TD3 (Phase 9 · 07) | 在 Q-network 上做 TD update；policy net 发出 actions。 |
| LLM RL（reward-model-based） | PPO / GRPO (Phase 9 · 08, 12) | Actor-critic 通过 GAE 得到 TD-style advantage。 |
| Offline RL | CQL / IQL (Phase 9 · 08) | 带 conservative regularization 的 Q-learning。 |

你在 2026 年论文中读到的 “RL”，百分之九十都是 Q-learning 或 SARSA 的某种扩展。先把 tabular update 写到手上，再继续读深层内容。

## 交付成果

保存为 `outputs/skill-td-agent.md`：

```markdown
---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Given a tabular or small-feature environment, output:

1. Algorithm. Q-learning / SARSA / Expected SARSA / n-step variant. One-sentence reason tied to on-policy vs off-policy and variance.
2. Hyperparameters. α, γ, ε, decay schedule.
3. Initialization. Q_0 value (optimistic vs zero) and justification.
4. Convergence diagnostic. Target learning curve, `|Q - Q*|` check if DP is possible.
5. Deployment caveat. How will exploration behave at inference? Is SARSA's conservatism needed?

Refuse to apply tabular TD to state spaces > 10⁶. Refuse to ship a Q-learning agent without a max-bias caveat. Flag any agent trained with ε held at 1.0 throughout (no exploitation phase).
```

## 练习

1. **Easy.** 在 4×4 GridWorld 上实现 Q-learning 和 SARSA。为 2,000 episodes 绘制 learning curves（每 100 episodes 的 mean return）。谁收敛更快？
2. **Medium.** 构建一个 cliff-walking environment（4×12，最后一行是 cliff，reward -100 并 reset 到 start）。比较 Q-learning 和 SARSA 的最终 policies。截取各自走过的 paths。哪个更靠近悬崖？
3. **Hard.** 实现 Double Q-learning。在 noisy-reward GridWorld（per-step reward 加 Gaussian noise σ=5）上，展示 Q-learning 会显著高估 `V*(0,0)`，而 Double Q-learning 不会。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| TD error | “更新信号” | `δ = r + γ V(s') - V(s)`，bootstrapped residual。 |
| TD(0) | “One-step TD” | 每个 transition 后只使用下一个 state estimate 更新。 |
| Q-learning | “Off-policy RL 101” | 对 next-state actions 取 `max` 的 TD update；不管 behavior policy 如何都学习 `Q*`。 |
| SARSA | “On-policy Q-learning” | 使用实际 next action 的 TD update；为当前 ε-greedy π 学习 `Q^π`。 |
| Expected SARSA | “低 variance 的 SARSA” | 用 π 下的 expectation 替代 sampled `a'`。 |
| GLIE | “正确的 exploration schedule” | Greedy in the Limit with Infinite Exploration；Q-learning 收敛所需。 |
| Bootstrapping | “在 target 里使用当前 estimate” | 区分 TD 与 MC 的关键。它带来 bias，但大幅降低 variance。 |
| Maximization bias | “Q-learning overestimates” | 对 noisy estimates 取 `max` 会向上偏；Double Q-learning 可修复。 |

## 延伸阅读

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — 原始论文和收敛证明。
- [Sutton & Barto (2018). Ch. 6 — Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0)、SARSA、Q-learning、Expected SARSA。
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — maximization bias 的修复。
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) — expected SARSA 的动机。
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — 创造 SARSA 这一名称的论文（当时称为 “modified connectionist Q-learning”）。
- [Sutton & Barto (2018). Ch. 7 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) — 把 TD(0) 泛化到 TD(n)，也是从 Q-learning 通向 eligibility traces，以及后来 PPO 中 GAE 的路径。
