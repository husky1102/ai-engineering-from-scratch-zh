# Monte Carlo 方法：从完整 Episodes 中学习

> 动态规划需要 model。Monte Carlo 只需要 episodes。运行 policy，观察 returns，取平均。这是 RL 中最简单的想法，也是打开后续所有内容的钥匙。

**类型:** Build
**语言:** Python
**先修:** Phase 9 · 01 (MDPs), Phase 9 · 02 (Dynamic Programming)
**时间:** ~75 分钟

## 要解决的问题

Dynamic programming 很优雅，但它假设你能对每个 state 和 action 查询 `P(s' | s, a)`。现实世界中几乎没有什么东西是这样工作的。机器人无法解析地计算某个 joint torque 之后 camera pixels 的分布。定价算法无法对每一种可能的客户反应积分。LLM 无法枚举一个 token 之后的所有可能续写。

你需要一种只要求能够从 environment *采样* 的方法。运行 policy。得到一条 trajectory `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`。用它来估计 values。这就是 Monte Carlo。

从 DP 到 MC 的转变在哲学上很重要：我们从 *known model + exact backup* 移动到 *sampled rollouts + averaged return*。variance 暴涨，但适用范围也暴涨。这个 lesson 之后的每个 RL 算法，包括 TD、Q-learning、REINFORCE、PPO、GRPO，本质上都是 Monte Carlo estimator，有时在上面叠加 bootstrapping。

## 核心概念

![Monte Carlo: rollout, compute returns, average; first-visit vs every-visit](../assets/monte-carlo.svg)

**核心思想，一行写完：** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)`，其中 `G^{(i)}(s)` 是在 policy `π` 下访问 `s` 之后观测到的 returns。

**First-visit vs every-visit MC。** 给定一个多次访问 state `s` 的 episode，first-visit MC 只统计第一次访问后的 return；every-visit MC 统计所有访问。二者在极限下都是 unbiased。First-visit 更容易分析（iid samples）。Every-visit 每个 episode 使用更多数据，实践中通常收敛更快。

**Incremental mean。** 不存储所有 returns，而是更新 running average：

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

重排：`V_new = V_old + α · (target - V_old)`，其中 `α = 1/n`。把 `1/n` 换成常数 step-size `α ∈ (0, 1)`，你就得到一个能跟踪 `π` 变化的 non-stationary MC estimator。从 MC 到 TD，再到每个现代 RL 算法，整个跳跃都在这一步里。

**探索现在成了问题。** DP 通过枚举触及每个 state。MC 只看到 policy 访问过的 states。如果 `π` 是确定性的，state space 的整片区域永远不会被 sampled，它们的 value estimates 会永远停在零。三个修复方法，按历史顺序：

1. **Exploring starts。** 从随机 (s, a) pair 启动每个 episode。保证 coverage；实践中不现实（你不能把机器人“重置”到任意状态）。
2. **ε-greedy。** 相对于当前 Q 做 greedy，但以概率 `ε` 随机选一个 action。所有 state-action pairs 渐近地都会被 sampled。
3. **Off-policy MC。** 在 behavior policy `μ` 下收集数据，用 importance sampling 学习 target policy `π`。Variance 很高，但它是通往 DQN 这类 replay-buffer 方法的桥。

**Monte Carlo Control。** Evaluate → improve → evaluate，就像 policy iteration，只是 evaluation 是 sampling-based：

1. 运行 `π`，得到一个 episode。
2. 用观测 returns 更新 `Q(s, a)`。
3. 让 `π` 相对于 `Q` 变成 ε-greedy。
4. 重复。

在温和条件下（每个 pair 被无限次访问，`α` 满足 Robbins-Monro），它会以概率 1 收敛到 `Q*` 和 `π*`。

## 动手实现

### Step 1: rollout → (s, a, r) 列表

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

没有 model，只有 `env.reset()` 和 `env.step(s, a)`。接口和 gym environment 一样，只是剥到最小。

### Step 2: 计算 returns（反向 sweep）

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

一遍，`O(T)`。反向递推 `G_t = r_{t+1} + γ G_{t+1}` 避免重复求和。

### Step 3: first-visit MC evaluation

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

三行完成核心工作：第一次访问时把 state 标记为 seen，增加 count，更新 running mean。

### Step 4: ε-greedy MC control（on-policy）

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### Step 5: 与 DP gold standard 对比

随着 episodes → ∞，你的 MC 版 `V^π` estimate 应该与 Lesson 02 的 DP result 一致。实践中：在 4×4 GridWorld 上跑 50,000 个 episodes，通常可以逼近到 DP answer 的 `~0.1` 以内。

## 常见陷阱

- **无限 episodes。** MC 要求 episodes 必须 *terminate*。如果你的 policy 可能永远循环，设置 `max_steps` cap，并把 cap 当成隐式失败。GridWorld 的 random policy 经常 timeout，这很正常，只要确保正确计数。
- **Variance。** MC 使用 full returns。在长 episodes 上 variance 极大，末尾一次不走运的 reward 会以同样幅度移动 `V(s_0)`。TD methods（Lesson 04）通过 bootstrapping 降低这个问题。
- **State coverage。** 新 Q 上的 greedy MC 如果遇到 ties，只会永远尝试一个 action。你 *必须* explore（ε-greedy、exploring starts、UCB）。
- **Non-stationary policies。** 如果 `π` 会变化（如 MC control），旧 returns 来自不同 policy。Constant-α MC 能处理这一点；sample-average MC 不能。
- **Off-policy importance sampling。** 权重 `π(a|s)/μ(a|s)` 会沿 trajectory 相乘。Variance 会随 horizon 爆炸。用 per-decision weighted IS 截断，或切换到 TD。

## 实际使用

2026 年 Monte Carlo methods 的角色：

| 用例 | 为什么用 MC |
|----------|--------|
| Short-horizon games（blackjack、poker） | Episodes 自然 terminate；returns 很干净。 |
| Logged policy 的 offline evaluation | 对 stored trajectories 的 discounted returns 取平均。 |
| Monte Carlo Tree Search (AlphaZero) | 从 tree leaves 发出的 MC rollouts 指导 selection。 |
| LLM RL evaluation | 计算某个 policy 的 sampled completions 的平均 reward。 |
| PPO 中的 baseline estimation | advantage target `A_t = G_t - V(s_t)` 使用 MC `G_t`。 |
| 教学 RL | 最简单且真的能工作的算法，拿掉 bootstrapping 就能看见核心。 |

现代 deep-RL 算法（PPO、SAC）通过 `n`-step returns 或 GAE，在纯 MC（full returns）和纯 TD（one-step bootstrap）之间插值。两个端点都是同一个 estimator 的实例。

## 交付成果

保存为 `outputs/skill-mc-evaluator.md`：

```markdown
---
name: mc-evaluator
description: Evaluate a policy via Monte Carlo rollouts and produce a convergence report with DP-comparison if available.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

Given an environment (episodic, with reset+step API) and a policy, output:

1. Method. First-visit vs every-visit MC. Reason.
2. Episode budget. Target number, variance diagnostic, expected standard error.
3. Exploration plan. ε schedule (if needed) or exploring starts.
4. Gold-standard comparison. DP-optimal V* if tabular; otherwise a bound from a Q-learning / PPO baseline.
5. Termination check. Max-step cap, timeouts, handling of non-terminating trajectories.

Refuse to run MC on non-episodic tasks without a finite horizon cap. Refuse to report V^π estimates from fewer than 100 episodes per state for tabular tasks. Flag any policy with zero-variance actions as an exploration risk.
```

## 练习

1. **Easy.** 实现 4×4 GridWorld 上 uniform-random policy 的 first-visit MC evaluation。运行 10,000 个 episodes。把 `V(0,0)` 随 episode count 的变化画出来，并与 DP answer 对比。
2. **Medium.** 实现 ε-greedy MC control，令 `ε ∈ {0.01, 0.1, 0.3}`。比较训练 20,000 个 episodes 后的 mean return。曲线长什么样？bias-variance tradeoff 在哪里？
3. **Hard.** 实现带 importance sampling 的 *off-policy* MC：在 uniform-random policy `μ` 下收集数据，估计 deterministic optimal policy `π` 的 `V^π`。比较 plain IS、per-decision IS、weighted IS。哪个 variance 最低？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Monte Carlo | “随机采样” | 通过对分布中的 iid samples 取平均来估计 expectations。 |
| Return `G_t` | “未来奖励” | 从 step `t` 到 episode end 的折扣奖励和：`Σ_{k≥0} γ^k r_{t+k+1}`。 |
| First-visit MC | “每个 state 只数一次” | 一个 episode 中只有第一次访问会贡献 value estimate。 |
| Every-visit MC | “使用所有访问” | 每次访问都贡献；略有 bias 但 sample-efficient 更高。 |
| ε-greedy | “Exploration noise” | 以概率 `1-ε` 选择 greedy action；以概率 `ε` 选择 random action。 |
| Importance sampling | “纠正从错误分布采样” | 用 `π(a\|s)/μ(a\|s)` 乘积重加权 returns，从 `μ` 数据估计 `V^π`。 |
| On-policy | “从自己的数据学习” | Target policy = behavior policy。Vanilla MC、PPO、SARSA。 |
| Off-policy | “从别人的数据学习” | Target policy ≠ behavior policy。Importance-sampled MC、Q-learning、DQN。 |

## 延伸阅读

- [Sutton & Barto (2018). Ch. 5 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 经典处理。
- [Singh & Sutton (1996). Reinforcement Learning with Replacing Eligibility Traces](https://link.springer.com/article/10.1007/BF00114726) — first-visit vs every-visit analysis。
- [Precup, Sutton, Singh (2000). Eligibility Traces for Off-Policy Policy Evaluation](http://incompleteideas.net/papers/PSS-00.pdf) — off-policy MC 与 variance control。
- [Mahmood et al. (2014). Weighted Importance Sampling for Off-Policy Learning](https://arxiv.org/abs/1404.6362) — 现代 low-variance IS estimators。
- [Tesauro (1995). TD-Gammon, A Self-Teaching Backgammon Program](https://dl.acm.org/doi/10.1145/203330.203343) — MC/TD self-play 收敛到 superhuman play 的第一个大规模实证展示；也是本阶段后半部分每个 lesson 的概念前身。
