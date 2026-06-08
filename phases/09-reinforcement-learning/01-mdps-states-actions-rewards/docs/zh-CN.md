# MDP、状态、动作与奖励

> 马尔可夫决策过程由五样东西组成：状态、动作、转移、奖励、折扣。强化学习里的一切，包括 Q-learning、PPO、DPO、GRPO，都是在这个形状上做优化。学会一次，后面的强化学习就能一路读通。

**类型:** Learn
**语言:** Python
**先修:** Phase 1 · 06 (Probability & Distributions), Phase 2 · 01 (ML Taxonomy)
**时间:** ~45 分钟

## 要解决的问题

你正在写一个国际象棋 bot。或者库存规划器。或者交易 agent。或者训练推理模型的 PPO loop。四个完全不同的领域，却有一个出人意料的事实：它们都能坍缩成同一个数学对象。

监督学习给你 `(x, y)` 样本对，并要求你拟合一个函数。强化学习不给标签，只给一串状态、你采取的动作，以及一个标量奖励。这个走法赢棋了吗？补货决策省钱了吗？交易盈利了吗？LLM 刚生成的 token 是否让 judge 给出了更高奖励？

在把这条流形式化之前，你无法从中学习。“我看到了什么”“我做了什么”“接下来发生了什么”“这件事有多好”，每一样都必须变成可以推理的对象。这个形式化就是马尔可夫决策过程。本阶段的每个 RL 算法，包括最后的 RLHF 和 GRPO loop，都是在这个形状上优化。

## 核心概念

![马尔可夫决策过程：状态、动作、转移、奖励、折扣](../assets/mdp.svg)

**五个对象。**

- **状态** `S`。agent 做决策所需的一切。在 GridWorld 中是格子。在国际象棋中是棋盘。在 LLM 中是 context window 加上任何 memory。
- **动作** `A`。可选项。上/下/左/右移动。下一步棋。发出一个 token。
- **转移** `P(s' | s, a)`。给定状态 `s` 和动作 `a` 后，下一个状态的分布。在国际象棋中是确定性的，在库存中是随机的，在 LLM decoding 中近乎确定。
- **奖励** `R(s, a, s')`。标量信号。赢 = +1，输 = -1。收入减成本。GRPO 中的 log-likelihood ratio 项。
- **折扣** `γ ∈ [0, 1)`。未来奖励相对当前奖励占多大权重。`γ = 0.99` 买到约 100 步的 horizon；`γ = 0.9` 买到约 10 步。

**马尔可夫性质** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`。未来只依赖当前状态。如果不是这样，说明状态表示不完整，这不是方法失败，而是状态失败。

**策略与回报。** 策略 `π(a | s)` 把状态映射到动作分布。回报 `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …` 是未来奖励的折扣和。价值 `V^π(s) = E[G_t | s_t = s]` 是从 `s` 开始、在策略 `π` 下的期望回报。Q-value `Q^π(s, a) = E[G_t | s_t = s, a_t = a]` 是以某个特定动作开始的期望回报。每个 RL 算法都会估计这两个量之一，然后相应地改进 `π`。

**Bellman 方程。** 本阶段所有内容都会用到的 fixed-point 方程：

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

它们把期望回报拆成“这一步的奖励”加上“落点的折扣价值”。递归。本阶段每个算法要么把这个方程迭代到收敛（dynamic programming），要么从中采样（Monte Carlo），要么做一步 bootstrapping（temporal difference）。

## 动手实现

### Step 1: 一个微型确定性 MDP

一个 4×4 GridWorld。agent 从左上角开始，terminal 在右下角，每步奖励 -1，动作 `{up, down, left, right}`。见 `code/main.py`。

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

五行。这就是整个 environment。确定性转移、恒定步惩罚、吸收 terminal 状态。

### Step 2: rollout 一个策略

策略是从状态到动作分布的函数。最简单的策略：均匀随机。

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

运行随机策略 1000 次。对这个 4×4 棋盘，平均回报大约是 -60 到 -80。最优回报是 -6（一路向下向右的直线路径）。缩小这个差距就是 Phase 9 的全部内容。

### Step 3: 通过 Bellman 方程精确计算 `V^π`

对小型 MDP，Bellman 方程就是一个线性系统。枚举状态，应用期望，迭代直到价值停止变化。

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

这就是 iterative policy evaluation。它是 Sutton & Barto 中的第一个算法，也是后续每种 RL 方法的理论基础。

### Step 4: `γ` 是有物理意义的超参数

有效 horizon 约等于 `1 / (1 - γ)`。`γ = 0.9` → 10 步。`γ = 0.99` → 100 步。`γ = 0.999` → 1000 步。

太低，agent 会短视。太高，credit assignment 会变得嘈杂，因为许多早期步骤都要共同承担很远未来奖励的责任。LLM RLHF 通常使用 `γ = 1`，因为 episode 短且有界。控制任务使用 `0.95–0.99`。长 horizon 策略游戏使用 `0.999`。

## 常见陷阱

- **非马尔可夫状态。** 如果你需要最近三次 observation 才能决定，那么“状态”就不只是当前 observation。修复：stack frames（Atari 上的 DQN stacks 4）或使用 recurrent state（在 observations 上用 LSTM/GRU）。
- **稀疏奖励。** 只有胜负的奖励会让大状态空间中的学习几乎不可能。塑造奖励（中间信号）或用 imitation 做 bootstrap（Phase 9 · 09）。
- **奖励 hacking。** 优化代理奖励经常产生病态行为。OpenAI 的 boat-racing agent 一直绕圈收集 powerups，而不是完成比赛。始终从目标 outcome 定义奖励，而不是从 proxy 定义。
- **折扣设错。** 在 infinite-horizon 任务上用 `γ = 1` 会让每个价值都变成无限大。始终用有限 horizon 或 `γ < 1` 来封顶。
- **奖励尺度。** {+100, -100} 与 {+1, -1} 的奖励给出相同的最优策略，但 gradient magnitudes 差别巨大。接入 PPO/DQN 前先把它们 normalize 到接近 `[-1, 1]`。

## 实际使用

2026 年的技术栈会在碰代码之前，把每条 RL pipeline 都化简成 MDP：

| 情境 | 状态 | 动作 | 奖励 | γ |
|-----------|-------|--------|--------|---|
| 控制（locomotion、manipulation） | Joint angles + velocities | Continuous torques | Task-specific shaped | 0.99 |
| 游戏（chess、Go、poker） | Board + history | Legal move | Win=+1 / loss=-1 | 1.0 (finite) |
| 库存 / 定价 | Stock + demand | Order qty | Revenue - cost | 0.95 |
| LLM 的 RLHF | Context tokens | Next token | Reward-model score at end | 1.0 (episode ~200 tokens) |
| 推理的 GRPO | Prompt + partial response | Next token | Verifier 0/1 at end | 1.0 |

先写出这五元组，再写任何 training loop。大多数“RL 不工作”的 bug report，最终都能追溯到纸面上已经坏掉的 MDP formulation。

## 交付成果

保存为 `outputs/skill-mdp-modeler.md`：

```markdown
---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

Given a task (control / game / recommendation / LLM fine-tuning), output:

1. State. Exact feature vector or tensor spec. Justify Markov property.
2. Action. Discrete set or continuous range. Dimensionality.
3. Transition. Deterministic, stochastic-with-known-model, or sample-only.
4. Reward. Function and source. Sparse vs shaped. Terminal vs per-step.
5. Discount. Value and horizon justification.

Refuse to ship any MDP where the state is non-Markovian without explicit mention of frame-stacking or recurrent state. Refuse any reward that was not defined in terms of the target outcome. Flag any `γ ≥ 1.0` on an infinite-horizon task. Flag any reward range >100x the typical step reward as a likely gradient-explosion source.
```

## 练习

1. **Easy.** 在 `code/main.py` 中实现 4×4 GridWorld 和 random-policy rollout。运行 10,000 个 episodes。报告 return 的 mean 和 std。与最优回报 (-6) 对比。
2. **Medium.** 对 uniform-random policy，用 `γ ∈ {0.5, 0.9, 0.99}` 运行 `policy_evaluation`。把每个 `V` 打印为 4×4 grid。解释为什么 terminal 附近的状态价值会随着更大的 `γ` 更快增长。
3. **Hard.** 把 GridWorld 改成随机的：每个动作以概率 `p = 0.1` 滑向相邻方向。重新评估 uniform policy。`V[start]` 会变好还是变差？为什么？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| MDP | “强化学习设置” | 满足马尔可夫性质的 tuple `(S, A, P, R, γ)`。 |
| State | “agent 看到的东西” | 对所选 policy class 下未来 dynamics 的充分统计量。 |
| Policy | “agent 的行为” | 条件分布 `π(a \| s)` 或确定性映射 `s → a`。 |
| Return | “总奖励” | 从当前 step 开始的折扣和 `Σ γ^t r_t`。 |
| Value | “一个状态有多好” | 从 `s` 开始，在 `π` 下的期望回报。 |
| Q-value | “一个动作有多好” | 从 `s` 开始且第一个动作是 `a` 时，在 `π` 下的期望回报。 |
| Bellman equation | “动态规划递归” | 把 value / Q 分解成一步奖励加折扣 successor value 的 fixed-point 分解。 |
| Discount `γ` | “未来 vs 当前” | 对遥远未来奖励的几何权重；有效 horizon `~1/(1-γ)`。 |

## 延伸阅读

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — 经典教材。Ch. 3 覆盖 MDP 和 Bellman equations；Ch. 1 说明每个后续 lesson 背后的 reward hypothesis。
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) — Bellman equation 的源头。
- [OpenAI Spinning Up — Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — 从 deep-RL 角度给出的简洁 MDP 入门。
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — MDP 和精确求解方法的 operations-research 参考。
- [Littman (1996). Algorithms for Sequential Decision Making (PhD thesis)](https://www.cs.rutgers.edu/~mlittman/papers/thesis-main.pdf) — 把 MDP 推导为 dynamic-programming 特例的最清晰版本。
