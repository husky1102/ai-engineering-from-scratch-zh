# 动态规划：Policy Iteration 与 Value Iteration

> 动态规划是“作弊版” RL。你已经知道 transition 和 reward functions；只需要迭代 Bellman equation，直到 `V` 或 `π` 不再移动。它是每个采样式方法都想逼近的 benchmark。

**类型:** Build
**语言:** Python
**先修:** Phase 9 · 01 (MDPs)
**时间:** ~75 分钟

## 要解决的问题

你有一个已知 model 的 MDP：对任何 state-action pair，都可以查询 `P(s' | s, a)` 和 `R(s, a, s')`。库存管理器知道 demand distribution。棋盘游戏有确定性 transitions。gridworld 是四行 Python。你有一个 *model*。

Model-free RL（Q-learning、PPO、REINFORCE）是为没有 model 的情况发明的，也就是你只能从 environment 采样。但当你确实有 model 时，就有更快、更好的方法：dynamic programming。Bellman 在 1957 年设计了它们。它们至今仍定义正确性：当人们说“这个 MDP 的 optimal policy”时，意思就是 DP 会返回的 policy。

2026 年你仍然需要它们，原因有三个。第一，RL research 中每个 tabular environment（GridWorld、FrozenLake、CliffWalking）都会用 DP 求出 gold-standard policy。第二，精确 values 能让你 *debug* 采样方法：如果 Q-learning 对 `V*(s_0)` 的估计和 DP answer 差了 30%，你的 Q-learning 有 bug。第三，现代 offline RL 与 planning 方法（MCTS、AlphaZero 的 search、Phase 9 · 10 的 model-based RL）都会在 learned 或 given model 上迭代 Bellman backup。

## 核心概念

![Policy iteration and value iteration, side by side](../assets/dp.svg)

**两个算法，都是 Bellman 上的 fixed-point iteration。**

**Policy iteration。** 在两个步骤之间交替，直到 policy 停止变化。

1. *Evaluation:* 给定 policy `π`，反复应用 `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`，直到收敛，计算 `V^π`。
2. *Improvement:* 给定 `V^π`，让 `π` 相对于 `V^π` 变成 greedy：`π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`。

收敛有保证，因为 (a) 每次 improvement 要么保持 `π` 不变，要么严格提升某些 state 的 `V^π`，(b) 确定性 policies 的空间有限。即便对较大状态空间，通常也会在约 5–20 次 outer iterations 内收敛。

**Value iteration。** 把 evaluation 和 improvement 压缩到一次 sweep。应用 Bellman *optimality* equation：

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

重复直到 `max_s |V_{new}(s) - V(s)| < ε`。最后通过 greedy action 提取 policy。每次 iteration 严格更快，没有 inner evaluation loop，但通常需要更多 iterations 才能收敛。

**Generalized policy iteration (GPI)。** 统一框架。Value function 和 policy 被锁在一个双向改进循环中；任何把二者推向 mutual consistency 的方法（async value iteration、modified policy iteration、Q-learning、actor-critic、PPO）都是 GPI 的实例。

**为什么 `γ < 1` 重要。** Bellman operator 在 sup-norm 下是一个 `γ`-contraction：`||T V - T V'||_∞ ≤ γ ||V - V'||_∞`。Contraction 意味着唯一 fixed point 和几何收敛。去掉 `γ < 1`，保证就没了，你需要 finite horizon 或 absorbing terminal state。

## 动手实现

### Step 1: 构建 GridWorld MDP model

使用 Lesson 01 中同一个 4×4 GridWorld。我们增加一个随机变体：agent 以概率 `0.1` 滑向一个随机垂直方向。

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)` 返回 `(s', r, p)` 的列表。这就是整个 model。

### Step 2: policy evaluation

给定 policy `π(s) = {action: prob}`，迭代 Bellman equation，直到 `V` 停止变化：

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### Step 3: policy improvement

用相对于 `V` 的 greedy policy 替换 `π`。如果 `π` 没有变化，就返回，我们已经到达 optimum。

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### Step 4: 把它们缝合起来

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # arbitrary start
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

4×4 上的典型收敛：4–6 次 outer iterations。输出 `V*(0,0) ≈ -6`，以及一个严格减少步数的 policy。

### Step 5: value iteration（单循环版本）

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

同一个 fixed point，更少的代码行。

## 常见陷阱

- **忘记处理 terminals。** 如果你对 absorbing state 应用 Bellman，它仍会挑一个什么都不改变的 “best action”。用 `if s == terminal: V[s] = 0` guard。
- **Sup-norm vs L2 convergence。** 使用 `max |V_new - V|`，不是 average。理论保证在 sup-norm 上。
- **In-place vs synchronous updates。** 原地更新 `V[s]`（Gauss-Seidel）比分开的 `V_new` dict（Jacobi）收敛更快。生产代码使用 in-place。
- **Policy ties。** 如果两个 actions 的 Q-value 相等，`argmax` 可能在每次 iteration 中用不同方式打破平局，导致 “policy stable” check 震荡。使用稳定 tie-break（固定顺序中的第一个 action）。
- **State-space explosion。** DP 每次 sweep 是 `O(|S| · |A|)`。可工作到约 10⁷ 个 states。再往上，你需要 function approximation（Phase 9 · 05 onwards）。

## 实际使用

2026 年，DP 是 correctness baseline，也是 planner 的 inner loop：

| 用例 | 方法 |
|----------|--------|
| 精确求解小型 tabular MDP | Value iteration（更简单）或 policy iteration（outer steps 更少） |
| 验证 Q-learning / PPO 实现 | 在 toy environment 上与 DP-optimal V* 对比 |
| Model-based RL (Phase 9 · 10) | 在 learned transition model 上做 Bellman backup |
| AlphaZero / MuZero 中的 planning | Monte Carlo Tree Search = async Bellman backup |
| Offline RL (CQL, IQL) | Conservative Q-iteration，即带 OOD action penalty 的 DP |

每当有人说 “the optimal value function”，意思就是 “the DP fixed point”。在论文中看到 `V*` 或 `Q*` 时，请想象这个循环。

## 交付成果

保存为 `outputs/skill-dp-solver.md`：

```markdown
---
name: dp-solver
description: Solve a small tabular MDP exactly via policy iteration or value iteration. Report convergence behavior.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

Given an MDP with a known model, output:

1. Choice. Policy iteration vs value iteration. Reason tied to |S|, |A|, γ.
2. Initialization. V_0, starting policy. Convergence sensitivity.
3. Stopping. Sup-norm tolerance ε. Expected number of sweeps.
4. Verification. V*(s_0) computed exactly. Greedy policy extracted.
5. Use. How this baseline will be used to debug/evaluate sampling-based methods.

Refuse to run DP on state spaces > 10⁷. Refuse to claim convergence without a sup-norm check. Flag any γ ≥ 1 on an infinite-horizon task as a guarantee violation.
```

## 练习

1. **Easy.** 在 4×4 GridWorld 上用 `γ ∈ {0.9, 0.99}` 运行 value iteration。需要多少 sweeps 才能达到 `max |ΔV| < 1e-6`？把 `V*` 打印成 4×4 grid。
2. **Medium.** 在 *stochastic* GridWorld（slip probability `0.1`）上比较 policy iteration 与 value iteration。统计：sweeps、wall-clock time、最终 `V*(0,0)`。按 iterations 谁收敛更快？按 wall-clock 呢？
3. **Hard.** 构建 modified policy iteration：evaluation step 中只运行 `k` 次 sweep，而不是跑到收敛。对 `k ∈ {1, 2, 5, 10, 50}` 绘制 `V*(0,0)` error vs `k`。这条曲线告诉你 evaluation/improvement tradeoff 的什么信息？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Policy iteration | “DP algorithm” | 交替 evaluation (`V^π`) 与 improvement（相对于 `V^π` 的 greedy `π`），直到 policy 停止变化。 |
| Value iteration | “更快的 DP” | 一次 sweep 中应用 Bellman optimality backup；几何收敛到 `V*`。 |
| Bellman operator | “那个递归” | `(T V)(s) = max_a Σ P (r + γ V(s'))`；sup-norm 下的 `γ`-contraction。 |
| Contraction | “DP 为什么收敛” | 任何满足 `\|\|T x - T y\|\| ≤ γ \|\|x - y\|\|` 的 operator 都有唯一 fixed point。 |
| GPI | “一切都是 DP” | Generalized Policy Iteration：任何把 `V` 与 `π` 推向 mutual consistency 的方法。 |
| Synchronous update | “Jacobi-style” | 一次 sweep 中始终使用旧 `V`；易于分析但更慢。 |
| In-place update | “Gauss-Seidel-style” | 使用正在被更新的 `V`；实践中收敛更快。 |

## 延伸阅读

- [Sutton & Barto (2018). Ch. 4 — Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf) — policy iteration 与 value iteration 的经典表述。
- [Bertsekas (2019). Reinforcement Learning and Optimal Control](http://www.athenasc.com/rlbook.html) — contraction-mapping arguments 的严谨处理。
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — modified policy iteration 及其 convergence analysis。
- [Howard (1960). Dynamic Programming and Markov Processes](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) — 原始 policy iteration 论文。
- [Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming](http://www.athenasc.com/ndpbook.html) — 从 DP 到 approximate-DP / deep RL 的桥梁，后续每个 lesson 都会使用。
