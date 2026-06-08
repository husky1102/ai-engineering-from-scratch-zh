# Policy Gradient：从零实现 REINFORCE

> 停止估计 value。直接 parameterize policy，计算 expected return 的 gradient，向上走一步。Williams (1992) 用一个 theorem 写完了它。这就是 PPO、GRPO 和每个 LLM RL loop 存在的原因。

**类型:** Build
**语言:** Python
**先修:** Phase 3 · 03 (Backpropagation), Phase 9 · 03 (Monte Carlo), Phase 9 · 04 (TD Learning)
**时间:** ~75 分钟

## 要解决的问题

Q-learning 和 DQN parameterize 的是 *value* function。你通过 `argmax Q` 选择 actions。这对 discrete actions 和 discrete states 没问题。但当 actions 是 continuous 时会坏掉（你怎么在 10-dimensional torque 上做 `argmax`？），或者当你想要 stochastic policy 时也会坏掉（`argmax` 按构造就是 deterministic）。

Policy gradients 改为 parameterize *policy*。`π_θ(a | s)` 是一个输出 action distribution 的 neural net。从中 sample 来行动。计算 expected return 相对于 `θ` 的 gradient。向上走。没有 `argmax`。没有 Bellman recursion。只是对 `J(θ) = E_{π_θ}[G]` 做 gradient ascent。

REINFORCE theorem（Williams 1992）告诉你这个 gradient 是可计算的：`∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`。运行一个 episode。计算 return。在每个 step 把它乘以 `∇ log π_θ(a | s)`。取平均。Gradient-ascent。完成。

2026 年每个 LLM-RL 算法，包括 PPO、DPO、GRPO，都是 REINFORCE 的细化。把它写到手上，是本阶段剩余内容，以及 Phase 10 · 07（RLHF implementation）和 Phase 10 · 08（DPO）的先决条件。

## 核心概念

![Policy gradient: softmax policy, log-π gradient, return-weighted update](../assets/policy-gradient.svg)

**Policy gradient theorem。** 对任意由 `θ` parameterized 的 policy `π_θ`：

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

其中 `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}` 是从 step `t` 开始的 discounted return。Expectation 在从 `π_θ` sampled 的完整 trajectories `τ` 上取。

**证明很短。** 在 expectation 内对 `J(θ) = Σ_τ P(τ; θ) G(τ)` 求导。使用 `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)`（log-derivative trick）。分解 `log P(τ; θ) = Σ log π_θ(a_t | s_t) + environment terms that do not depend on θ`。Environment terms 消失。两行代数就得到 theorem。

**Variance reduction tricks。** Vanilla REINFORCE 的 variance 很凶，returns 有噪声，`∇ log π` 有噪声，它们的乘积非常有噪声。两个标准修复：

1. **Baseline subtraction。** 把 `G_t` 替换为 `G_t - b(s_t)`，其中 `b(s_t)` 可以是任何不依赖 `a_t` 的 baseline。Unbiased，因为 `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`。典型选择：由 critic 学习的 `b(s_t) = V̂(s_t)` → actor-critic（Lesson 07）。
2. **Reward-to-go。** 把 `Σ_t G_t · ∇ log π_θ(a_t | s_t)` 替换为 `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)`。对给定 action 来说，只有未来 returns 重要；过去 rewards 只贡献 zero-mean noise。

合起来得到：

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

这就是带 baseline 的 REINFORCE，也就是 A2C（Lesson 07）和 PPO（Lesson 08）的直系祖先。

**Softmax policy parameterization。** 对 discrete actions，标准选择是：

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

其中 `f_θ` 是任意输出每个 action score 的 neural net。Gradient 有一个很干净的形式：

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

也就是 taken action 的 score 减去 policy 下的期望值。

**连续 actions 的 Gaussian policy。** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`。`∇ log N(a; μ, σ)` 有 closed form。这就是 Phase 9 · 07 的 SAC 所需的全部。

## 动手实现

### Step 1: softmax policy network

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

对 tabular env 使用 linear policy（每个 action 一个 weight vector）。对 Atari，换成 CNN，并保留 softmax head。

### Step 2: sampling 和 log-probability

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### Step 3: 捕获 log-probs 的 rollout

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### Step 4: REINFORCE update

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

Gradient `∇ log π(a|s) = e_a - π(·|s)`（`a` 的 onehot 减去 probabilities）是 softmax policy gradients 的核心。把它烧进 muscle memory。

### Step 5: baselines

最近 episodes 上 `G` 的 running mean 已经足够降低 variance，让 4×4 GridWorld 跑起来；它大约需要 500 episodes 收敛。把 baseline 升级成 learned `V̂(s)`，你就得到 actor-critic。

## 常见陷阱

- **Exploding gradients。** Returns 可能很大。把 `G` 乘以 `∇ log π` 前，始终在 batch 内 normalize 到 `~N(0, 1)`。
- **Entropy collapse。** Policy 太早收敛到 near-deterministic action，停止探索，然后卡住。修复：在 objective 中加入 entropy bonus `β · H(π(·|s))`。
- **High variance。** Vanilla REINFORCE 需要数千 episodes。Critic baseline（Lesson 07）或 TRPO/PPO 的 trust region（Lesson 08）是标准修复。
- **Sample inefficiency。** On-policy 意味着每个 transition 更新一次后就丢掉。通过 importance sampling 的 off-policy corrections 可以重新利用数据，但代价是 variance（PPO 的 ratio 就是 clipped IS weight）。
- **Non-stationary gradients。** 100 episodes 前的同一个 gradient 使用的是旧 `π`。因此 on-policy methods 会每隔几个 rollouts 就更新。
- **Credit assignment。** 没有 reward-to-go 时，过去 rewards 会贡献噪声。始终使用 reward-to-go。

## 实际使用

2026 年，REINFORCE 很少直接运行，但它的 gradient formula 无处不在：

| 用例 | 派生方法 |
|----------|---------------|
| Continuous control | PPO / SAC with Gaussian policy |
| LLM RLHF | PPO with KL penalty, running on token-level policy |
| LLM reasoning (DeepSeek) | GRPO — REINFORCE with group-relative baseline, no critic |
| Multi-agent | Centralized-critic REINFORCE (MADDPG, COMA) |
| Discrete action robotics | A2C, A3C, PPO |
| Preference-only settings | DPO — REINFORCE rewritten as a preference-likelihood loss, no sampling |

当你在 2026 年 training script 中读到 `loss = -advantage * log_prob`，那就是带 baseline 的 REINFORCE。整篇论文（DPO、GRPO、RLOO）都是这行代码之上的 variance-reduction tricks。

## 交付成果

保存为 `outputs/skill-policy-gradient-trainer.md`：

```markdown
---
name: policy-gradient-trainer
description: Produce a REINFORCE / actor-critic / PPO training config for a given task and diagnose variance issues.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

Given an environment (discrete / continuous actions, horizon, reward stats), output:

1. Policy head. Softmax (discrete) or Gaussian (continuous) with parameter counts.
2. Baseline. None (vanilla), running mean, learned `V̂(s)`, or A2C critic.
3. Variance controls. Reward-to-go on by default, return normalization, gradient clip value.
4. Entropy bonus. Coefficient β and decay schedule.
5. Batch size. Episodes per update; on-policy data freshness contract.

Refuse REINFORCE-no-baseline on horizons > 500 steps. Refuse continuous-action control with a softmax head. Flag any run with `β = 0` and observed policy entropy < 0.1 as entropy-collapsed.
```

## 练习

1. **Easy.** 在 4×4 GridWorld 上用 linear softmax policy 实现 REINFORCE。不使用 baseline 训练 1,000 episodes。绘制 learning curve；测量 variance（returns 的 std）。
2. **Medium.** 加入 running-mean baseline。重新训练。和 vanilla run 比较 sample efficiency 与 variance。baseline 让收敛所需 steps 降低了多少？
3. **Hard.** 加入 entropy bonus `β · H(π)`。扫 `β ∈ {0, 0.01, 0.1, 1.0}`。绘制 final return 和 policy entropy。这个 task 上的 sweet spot 在哪里？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Policy gradient | “直接训练 policy” | `∇J(θ) = E[G · ∇ log π_θ(a\|s)]`；由 log-derivative trick 推导。 |
| REINFORCE | “原始 PG algorithm” | Williams (1992)；Monte Carlo returns 乘以 log-policy gradient。 |
| Log-derivative trick | “Score function estimator” | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`；让 expectations 的 gradients 可处理。 |
| Baseline | “Variance reduction” | 从 `G` 中减去的任何 `b(s)`；unbiased，因为 `E[b · ∇ log π] = 0`。 |
| Reward-to-go | “只有未来 returns 算数” | 使用 `G_t^{from t}`，而不是完整的 `G_0`；正确且 lower-variance。 |
| Entropy bonus | “鼓励 exploration” | `+β · H(π(·\|s))` 项防止 policy collapse。 |
| On-policy | “在刚看到的数据上训练” | Gradient expectation 相对于 current policy，不能直接复用旧数据。 |
| Advantage | “比平均好多少” | `A(s, a) = G(s, a) - V(s)`；REINFORCE-with-baseline 相乘的 signed quantity。 |

## 延伸阅读

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) — 原始 REINFORCE 论文。
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — 带 function approximation 的现代 policy-gradient theorem。
- [Sutton & Barto (2018). Ch. 13 — Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 教材表述。
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) — 配 PyTorch code 的清晰教学 exposition。
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) — variance-reduction 和 natural-gradient 视角，把 REINFORCE 连接到 trust-region family（TRPO、PPO）。
