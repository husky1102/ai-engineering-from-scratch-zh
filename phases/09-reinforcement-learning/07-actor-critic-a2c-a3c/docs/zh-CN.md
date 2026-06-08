# Actor-Critic：A2C 与 A3C

> REINFORCE 很嘈杂。加入一个学习 `V̂(s)` 的 critic，从 return 中减去它，你就得到一个 expectation 相同但 variance 低得多的 advantage。这就是 actor-critic。A2C 同步运行；A3C 在线程间异步运行。二者都是每个现代 deep-RL 方法的 mental model。

**类型:** Build
**语言:** Python
**先修:** Phase 9 · 04 (TD Learning), Phase 9 · 06 (REINFORCE)
**时间:** ~75 分钟

## 要解决的问题

Vanilla REINFORCE 能工作，但 variance 很糟。Monte Carlo returns `G_t` 在 episodes 之间可能波动 10 倍以上。把这些噪声乘以 `∇ log π` 再取平均，会得到一个 gradient estimator，需要数千个 episodes 才能把 policy 移动到 DQN 用少得多 updates 就能达到的距离。

Variance 来自使用 raw returns。如果你减去一个 baseline `b(s_t)`，也就是任何 state 的函数，包括 learned value，expectation 不变且 variance 下降。最好的可处理 baseline 是 `V̂(s_t)`。现在乘以 `∇ log π` 的量就是 *advantage*：

`A(s, a) = G - V̂(s)`

如果某个 action 产生了高于平均的 return，它就是好的；低于平均就是坏的。带 learned critic 的 REINFORCE 就是 *actor-critic*。Critic 给 actor 一个 low-variance teacher。这是 2015 年之后每种 deep-policy method（A2C、A3C、PPO、SAC、IMPALA）。

## 核心概念

![Actor-critic: policy net plus value net, TD residual as advantage](../assets/actor-critic.svg)

**两个 networks，一个 shared loss：**

- **Actor** `π_θ(a | s)`：policy。用于 sample 以行动。用 policy gradient 训练。
- **Critic** `V_φ(s)`：估计从 state 开始的 expected return。训练目标是最小化 `(V_φ(s) - target)²`。

**Advantage。** 两种标准形式：

- *MC advantage:* `A_t = G_t - V_φ(s_t)`。Unbiased，variance 更高。
- *TD advantage:* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`。Biased（使用 `V_φ`），variance 低得多。也叫 *TD residual* `δ_t`。

**n-step advantage。** 在二者之间插值：

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1` 是纯 TD。`n = ∞` 是 MC。大多数实现对 Atari 使用 `n = 5`，对 MuJoCo 上的 PPO 使用 `n = 2048`。

**Generalized Advantage Estimation (GAE)。** Schulman et al. (2016) 提出对所有 n-step advantages 做 exponentially weighted average：

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

其中 `λ ∈ [0, 1]`。`λ = 0` 是 TD（low variance, high bias）。`λ = 1` 是 MC（high variance, unbiased）。`λ = 0.95` 是 2026 年的默认值，调它直到 bias/variance 旋钮落在你想要的位置。

**A2C: synchronous advantage actor-critic。** 在 `N` 个 parallel environments 上收集 `T` steps。为每个 step 计算 advantages。用组合 batch 更新 actor 和 critic。重复。它是 A3C 更简单、更 scalable 的 sibling。

**A3C: asynchronous advantage actor-critic。** Mnih et al. (2016)。启动 `N` 个 worker threads，每个运行一个 env。每个 worker 在自己的 rollout 上本地计算 gradients，然后异步应用到 shared parameter server。不需要 replay buffer，workers 通过运行不同 trajectories 来 decorrelate。A3C 证明了你可以在 CPU 上 scale 训练。2026 年，GPU-based A2C（batched parallel envs）占主导，因为 GPUs 需要 large batches。

**Combined loss。**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

三个 terms：policy-gradient loss、value regression、entropy bonus。`c_v ~ 0.5`、`c_e ~ 0.01` 是 canonical starting points。

## 动手实现

### Step 1: 一个 critic

Linear critic `V_φ(s) = w · features(s)` 用 MSE 更新：

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

在 tabular env 上，critic 几百个 episodes 就能收敛。在 Atari 上，把 linear critic 换成 shared CNN trunk + value head。

### Step 2: n-step advantage

给定长度为 `T` 的 rollout 和一个 bootstrapped final `V(s_T)`：

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns` 是 critic target。`advantages` 是乘以 `∇ log π` 的量。

### Step 3: combined update

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # critic
    critic_update(w, x, target_v, lr_v)

    # actor
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

On-policy，每次 update 一个 rollout，actor 和 critic 使用 separate learning rates。

### Step 4: parallelization（A3C vs A2C）

- **A3C:** spin up `N` threads。每个运行自己的 env 和自己的 forward pass。周期性把 gradient updates push 到 shared master。Master 上不加锁，race 没关系，只是增加噪声。
- **A2C:** 在单个 process 中运行 `N` 个 env instances，把 observations stack 成 `[N, obs_dim]` batch，做 batched forward pass 和 batched backward pass。GPU utilization 更高，deterministic，更容易推理。2026 年默认选择。

我们的 toy code 为清晰起见是 single-threaded；改写成 batched A2C 只需要三行 numpy。

## 常见陷阱

- **Actor gradient 前的 critic bias。** 如果 critic 是 random，它的 baseline 没有信息，你其实在纯噪声上训练。先 warm up critic 几百步，再开启 policy gradient，或使用较慢的 actor learning rate。
- **Advantage normalization。** 在每个 batch 上把 advantages normalize 到 zero-mean/unit-std。几乎无成本，却能大幅稳定训练。
- **Shared trunk。** 对 image inputs，actor 和 critic 使用 shared feature extractor。Separate heads。Shared features 可以从两个 losses 中 free-ride。
- **On-policy contract。** A2C 每条数据只复用一次 update。更多次会让 gradient biased（importance-sampling correction 正是 PPO 增加的东西）。
- **Entropy collapse。** 没有 `c_e > 0` 时，policy 会在几百次 updates 内变得 near-deterministic 并停止 exploration。
- **Reward scale。** Advantage magnitudes 依赖 reward scale。Normalize rewards（例如除以 running-std）以获得跨 tasks 一致的 gradient magnitudes。

## 实际使用

A2C/A3C 在 2026 年很少是最终选择，但它们是后续一切 refinement 的 architecture：

| 方法 | 与 A2C 的关系 |
|--------|----------------|
| PPO | A2C + clipped importance ratio for multi-epoch updates |
| IMPALA | A3C + V-trace off-policy correction |
| SAC (Phase 9 · 07) | Off-policy A2C with a soft-value critic (next lesson) |
| GRPO (Phase 9 · 12) | A2C without the critic — group-relative advantage |
| DPO | A2C collapsed into a preference-ranking loss, no sampling |
| AlphaStar / OpenAI Five | A2C with league training + imitation pre-training |

如果你在 2026 年论文中看到 “advantage”，请想到 actor-critic。

## 交付成果

保存为 `outputs/skill-actor-critic-trainer.md`：

```markdown
---
name: actor-critic-trainer
description: Produce an A2C / A3C / GAE configuration for a given environment, with advantage estimation and loss weights specified.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

Given an environment and compute budget, output:

1. Parallelism. A2C (GPU batched) vs A3C (CPU async) and the number of workers.
2. Rollout length T. Steps per env per update.
3. Advantage estimator. n-step or GAE(λ); specify λ.
4. Loss weights. `c_v` (value), `c_e` (entropy), gradient clip.
5. Learning rates. Actor and critic (separate if using).

Refuse single-worker A2C on environments with horizon > 1000 (too on-policy, too slow). Refuse to ship without advantage normalization. Flag any run with `c_e = 0` and observed entropy < 0.1 as entropy-collapsed.
```

## 练习

1. **Easy.** 在 4×4 GridWorld 上用 MC advantage（`G_t - V(s_t)`）训练 actor-critic。和 Lesson 06 中的 REINFORCE-with-running-mean-baseline 比较 sample efficiency。
2. **Medium.** 切换到 TD-residual advantage（`r + γ V(s') - V(s)`）。测量 advantage batches 的 variance。下降了多少？
3. **Hard.** 实现 GAE(λ)。扫 `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}`。绘制 final return vs sample efficiency。这个 task 的 bias/variance sweet spot 在哪里？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Actor | “Policy net” | `π_θ(a\|s)`，由 policy gradient 更新。 |
| Critic | “Value net” | `V_φ(s)`，通过对 returns / TD targets 做 MSE regression 更新。 |
| Advantage | “比平均好多少” | `A(s, a) = Q(s, a) - V(s)` 或其 estimators。`∇ log π` 的乘数。 |
| TD residual | “δ” | `δ_t = r + γ V(s') - V(s)`；one-step advantage estimate。 |
| GAE | “那个插值旋钮” | n-step advantages 的 exponentially weighted sum，由 `λ` parameterized。 |
| A2C | “Synchronous actor-critic” | 跨 envs 批处理；每个 rollout 做一次 gradient step。 |
| A3C | “Async actor-critic” | Worker threads 把 gradients push 到 shared param server。原始论文；2026 年较少见。 |
| Bootstrap | “在 horizon 处使用 V” | 截断 rollout，加入 `γ^n V(s_{t+n})` 来闭合求和。 |

## 延伸阅读

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — A3C，原始 async actor-critic 论文。
- [Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — GAE。
- [Sutton & Barto (2018). Ch. 13 — Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 基础内容；当 critic 是 neural net 时，请和 Ch. 9 的 function approximation 搭配阅读。
- [Espeholt et al. (2018). IMPALA](https://arxiv.org/abs/1802.01561) — 带 V-trace off-policy correction 的 scalable distributed actor-critic。
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — 值得阅读的 production A2C/PPO implementations。
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms) — two-timescale actor-critic decomposition 的基础收敛结果。
