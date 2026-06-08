# Deep Q-Networks (DQN)

> 2013 年：Mnih 在 raw pixels 上训练了一个 Q-learning network，在七个 Atari games 上击败了所有 classical RL agents。2015 年：扩展到 49 个 games，发表于 Nature，点燃 deep-RL 时代。DQN 是 Q-learning 加上三个让 function approximation 稳定的 tricks。

**类型:** Build
**语言:** Python
**先修:** Phase 3 · 03 (Backpropagation), Phase 9 · 04 (Q-learning, SARSA)
**时间:** ~75 分钟

## 要解决的问题

Tabular Q-learning 需要为每个 (state, action) pair 单独保存一个 Q-value。国际象棋棋盘有约 10⁴³ 个 states。一帧 Atari 是 210×160×3 = 100,800 个 features。Tabular RL 到几千个 states 就死了，更不用说十亿级。

事后看来，修复方法很明显：用 neural network `Q(s, a; θ)` 替换 Q-table。但“事后明显”花了几十年。Naive function approximation 加 Q-learning 会在 “deadly triad” 下发散，也就是 function approximation + bootstrapping + off-policy learning。Mnih et al. (2013, 2015) 识别出三个稳定学习的工程 tricks：

1. **Experience replay** 解相关 transitions。
2. **Target network** 冻结 bootstrap target。
3. **Reward clipping** 归一化 gradient magnitudes。

Atari 上的 DQN 是第一次用单一 architecture 和单一 hyperparameter set，从 raw pixels 解决几十个 control problems。此后所有 “deep-RL”，包括 DDQN、Rainbow、Dueling、Distributional、R2D2、Agent57，都是堆在这三个 tricks 的基底之上。

## 核心概念

![DQN training loop: env, replay buffer, online net, target net, Bellman TD loss](../assets/dqn.svg)

**目标函数。** DQN 在 neural Q-function 上最小化 one-step TD loss：

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = online network，每步由 gradient descent 更新。`θ^-` = target network，定期从 `θ` copy 过来（大约每 10,000 steps）。`D` = 过去 transitions 的 replay buffer。

**三个 tricks，按重要性排序：**

**Experience replay。** 一个容量约 `~10⁶` transitions 的 ring buffer。每个 training step 从中均匀随机采样 minibatch。这打破 temporal correlation（连续 frames 几乎相同），让 network 多次学习罕见的 rewarding transitions，并解相关连续 gradient updates。没有它，带 neural net 的 on-policy TD 会在 Atari 上发散。

**Target network。** 在 Bellman equation 两边使用同一个 network `Q(·; θ)` 会让 target 在每次 update 时移动，也就是“追自己的尾巴”。修复：保留第二个 network `Q(·; θ^-)`，其 weights 冻结。每 `C` steps，copy `θ → θ^-`。这会让 regression target 在数千个 gradient steps 内保持稳定。Soft updates `θ^- ← τ θ + (1-τ) θ^-`（DDPG、SAC 中使用）是更平滑的变体。

**Reward clipping。** Atari reward magnitudes 从 1 到 1000+ 不等。clip 到 `{-1, 0, +1}`，防止任何单个 game 主导 gradient。当 reward magnitude 有意义时，这是错的；但对 Atari 来说没问题，因为只有符号重要。

**Double DQN。** Hasselt (2016) 修复 maximization bias：用 online net *选择* action，用 target net *评估* 它。

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

Drop-in replacement，效果持续更好。默认使用它。

**其他改进（Rainbow, 2017）：** prioritized replay（更多采样 high-TD-error transitions）、dueling architecture（分离 `V(s)` 和 advantage heads）、noisy networks（learned exploration）、n-step returns、distributional Q（C51/QR-DQN）、multi-step bootstrapping。每项增加几个百分点；收益大致可相加。

## 动手实现

这里的代码是 stdlib-only 且 numpy-free。我们在一个微型 continuous GridWorld 上使用手写 single-hidden-layer MLP，因此每个 training step 都在微秒级运行。算法和大规模 Atari DQN 完全相同。

### Step 1: replay buffer

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

Atari 大约用 50,000 capacity；我们的 toy env 5,000 就够。

### Step 2: 一个微型 Q-network（manual MLP）

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

Forward pass：linear → ReLU → linear。这就是整个 net。

### Step 3: DQN update

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

形状就是 Lesson 04 的 Q-learning，只有两个差别：(a) 我们对可微的 `Q(·; θ)` 做 backprop，而不是索引 table；(b) target 使用 `Q(·; θ^-)`。

### Step 4: outer loop

每个 episode 中，在 `Q(·; θ)` 上按 ε-greedy 行动，把 transitions 放进 buffer，采样 minibatch，做一次 gradient step，并定期 sync `θ^- ← θ`。模式如下：

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

在 16-dim one-hot state 的微型 GridWorld 上，agent 约 500 episodes 学会 near-optimal policy。在 Atari 上，把它扩展到 200M frames，并加上 CNN feature extractor。

## 常见陷阱

- **Deadly triad。** Function approximation + off-policy + bootstrapping 可能发散。DQN 用 target net + replay 缓解，不要去掉任何一个。
- **Exploration。** ε 必须衰减，通常在训练前 `~10%` 从 1.0 衰减到 0.01。早期 exploration 不足时，Q-net 会收敛到 local basin。
- **Overestimation。** 对 noisy Q 取 `max` 会向上偏。生产中始终使用 Double DQN。
- **Reward scale。** Clip 或 normalize rewards；gradient magnitude 与 reward magnitude 成正比。
- **Replay buffer coldstart。** buffer 里有几千 transitions 之前不要训练。早期在约 20 个 samples 上的 gradients 会 overfit。
- **Target sync frequency。** 太频繁 ≈ 没有 target net；太不频繁 ≈ stale targets。Atari DQN 使用 10,000 env steps。经验法则：大约每训练 horizon 的 `~1/100` sync 一次。
- **Observation preprocessing。** Atari DQN stacks 4 frames 来让 state Markov。任何带 velocity info 的 env 都需要 frame-stacking 或 recurrent state。

## 实际使用

2026 年，DQN 很少是 state-of-the-art，但仍是参考 off-policy algorithm：

| 任务 | 首选方法 | 为什么不是 DQN？ |
|------|------------------|--------------|
| Discrete-action Atari-like | Rainbow DQN or Muesli | 同一框架，更多 tricks。 |
| Continuous control | SAC / TD3 (Phase 9 · 07) | DQN 没有 policy network。 |
| On-policy / high-throughput | PPO (Phase 9 · 08) | 无 replay buffer；更容易 scale。 |
| Offline RL | CQL / IQL / Decision Transformer | Conservative Q targets，无 bootstrapping blowups。 |
| Large discrete action spaces（recommender） | DQN with action embedding, or IMPALA | 可以；细节装饰很重要。 |
| LLM RL | PPO / GRPO | Sequence-level，不是 step-level；loss 不同。 |

这些 lessons 仍然能迁移。Replay 和 target networks 出现在 SAC、TD3、DDPG、SAC-X、AlphaZero 的 self-play buffer，以及每种 offline RL 方法中。Reward clipping 以 PPO 中的 advantage normalization 继续存在。这个 architecture 是 blueprint。

## 交付成果

保存为 `outputs/skill-dqn-trainer.md`：

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
```

## 练习

1. **Easy.** 运行 `code/main.py`。绘制 per-episode return curve。running mean 超过 -10 需要多少 episodes？
2. **Medium.** 禁用 target network（Bellman target 两边都用 online net）。测量 training instability：return 是否震荡或发散？
3. **Hard.** 添加 Double DQN：用 online net 选择 `argmax a'`，用 target net 评估。对 noisy-reward GridWorld，比较 1,000 episodes 后有无 Double DQN 时 `Q(s_0, best_a)` 相对 true `V*(s_0)` 的 bias。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| DQN | “Deep Q-learning” | 带 neural Q-function、replay buffer、target network 的 Q-learning。 |
| Experience replay | “Shuffled transitions” | 每个 gradient step 从 ring buffer 均匀采样；解相关 data。 |
| Target network | “Frozen bootstrap” | Bellman target 中使用的周期性 Q copy；稳定 training。 |
| Deadly triad | “RL 为什么发散” | Function approximation + bootstrapping + off-policy = 无 convergence guarantee。 |
| Double DQN | “Maximization bias 的修复” | Online net 选择 action，target net 评估它。 |
| Dueling DQN | “V and A heads” | 分解 Q = V + A - mean(A)；输出相同，gradient flow 更好。 |
| Rainbow | “所有 tricks” | DDQN + PER + dueling + n-step + noisy + distributional 合为一体。 |
| PER | “Prioritized Replay” | 按 TD-error magnitude 的比例采样 transitions。 |

## 延伸阅读

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — 开启 deep RL 的 2013 NeurIPS workshop paper。
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Nature 论文，49-game DQN。
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — DDQN。
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) — dueling DQN。
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) — stacked-tricks 论文。
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) — 清晰的现代 exposition。
- [Sutton & Barto (2018). Ch. 9 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) — 教材对 “deadly triad”（function approximation + bootstrapping + off-policy）的处理，DQN 的 target network 和 replay buffer 正是为了驯服它。
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/) — ablation studies 中使用的参考 single-file DQN；适合与本 lesson 的 from-scratch version 对照阅读。
