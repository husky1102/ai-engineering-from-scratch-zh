# 多智能体强化学习

> 单智能体 RL 假设环境是 stationary。把两个正在学习的 agent 放进同一个世界，这个假设就破了：每个 agent 都是另一个 agent 环境的一部分，而且双方都在变化。Multi-agent RL 是一组技巧，用来在 Markov 假设不再成立时让学习收敛。

**类型:** Build
**语言:** Python
**先修:** Phase 9 · 04（Q-learning），Phase 9 · 06（REINFORCE），Phase 9 · 07（Actor-Critic）
**时间:** ~45 分钟

## 要解决的问题

一个学习在房间中导航的机器人是单智能体 RL 问题。一支足球队不是。AlphaStar 对战 StarCraft 对手不是。一个由 bidding agents 组成的 marketplace 不是。两辆车协商四向停车不是。许多 many-on-many 的真实问题都不是。

在每个 multi-agent 设置中，从任意一个 agent 的视角看，其他 agents *就是*环境的一部分。当它们学习并改变行为时，环境变得 non-stationary。Markov property：“next state 只取决于 current state 和我的 action”会被违反，因为 next state 也取决于*其他* agents 选择了什么，而它们的 policies 是移动靶。

这会破坏 tabular convergence proofs（Q-learning 的保证假设环境 stationary）。它也会破坏朴素 deep RL：agents 彼此追逐成环，永远不能收敛到稳定策略。你需要 multi-agent-specific techniques：centralized training / decentralized execution、counterfactual baselines、league play、self-play。

2026 年应用：robot swarms、traffic routing、autonomous vehicle fleets、market simulators、multi-agent LLM systems（Phase 16），以及任何有多个智能玩家的游戏。

## 核心概念

![四种 MARL regime：indep、centralized critic、self-play、league](../assets/marl.svg)

**形式化：Markov Game。** MDP 的推广：states `S`，joint action `a = (a_1, …, a_n)`，transition `P(s' | s, a)`，以及 per-agent rewards `R_i(s, a, s')`。每个 agent `i` 在自己的 policy `π_i` 下最大化自己的 return。如果 rewards 相同，它是 **fully cooperative**。如果是 zero-sum，它是 **adversarial**。如果混合，则是 **general-sum**。

**核心挑战：**

- **Non-stationarity。** 从 agent `i` 的视角看，`P(s' | s, a_i)` 取决于正在变化的 `π_{-i}`。
- **Credit assignment。** 使用共享 reward 时，哪个 agent 导致了它？
- **Exploration coordination。** Agents 必须探索互补策略，而不是冗余地探索同一状态。
- **Scalability。** Joint action space 随 `n` 指数增长。
- **Partial observability。** 每个 agent 只看到自己的 observation；global state 是隐藏的。

**四种主导 regime：**

**1. Independent Q-learning / independent PPO（IQL，IPPO）。** 每个 agent 学自己的 Q 或 policy，把其他 agents 当作环境的一部分。简单，有时有效（尤其是 experience replay 充当平滑的 agent-modeling 技巧时）。理论收敛性：没有。实践中：对 loosely-coupled tasks 可以，对 tightly-coupled tasks 很差。

**2. Centralized training, decentralized execution（CTDE）。** 最常见的现代范式。每个 agent 有自己的 *policy* `π_i`，条件是 local observation `o_i`：部署时标准的 decentralized execution。训练期间，一个 centralized critic `Q(s, a_1, …, a_n)` 以完整 global state 和 joint action 为条件。例子：
- **MADDPG**（Lowe 等人 2017）：每个 agent 有 centralized critic 的 DDPG。
- **COMA**（Foerster 等人 2017）：counterfactual baseline：问“如果我采取 action `a'`，我的 reward 会是多少？”从而隔离我的贡献。
- **MAPPO** / **IPPO** with shared critic（Yu 等人 2022）：带 centralized value function 的 PPO。2026 年 cooperative MARL 的主流方法。
- **QMIX**（Rashid 等人 2018）：value decomposition：`Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))`，使用 monotonic mixing。

**3. Self-play。** 同一个 agent 的两个副本互相对弈。对手的 policy *就是*我过去某个快照的 policy。AlphaGo / AlphaZero / MuZero。OpenAI Five。最适合 zero-sum games；训练信号是对称的。

**4. League play。** self-play 向 general-sum / adversarial environments 的扩展：保留过去和当前 policies 的 population，从 league 中抽样一个 opponent，与其训练。加入 exploiters（专门击败当前最强者）和 main exploiters（专门击败 exploiters）。AlphaStar（StarCraft II）。当游戏允许“rock-paper-scissors”策略循环时需要它。

**Communication。** 允许 agents 互相发送 learned messages `m_i`。适用于 cooperative settings。Foerster 等人（2016）展示了可微 inter-agent communication 可以端到端训练。今天基于 LLM 的 multi-agent systems（Phase 16）本质上是在用自然语言通信。

## 动手实现

本课使用一个 6×6 GridWorld，里面有两个 cooperative agents。它们从相对角落开始，必须到达共享 goal。共享 reward：任一 agent 还在移动时每步 `-1`，两者都到达时 `+10`。见 `code/main.py`。

### Step 1：multi-agent env

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # two agents

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

*Joint* action space 是 `|A|² = 16`。global state 是两个位置。

### Step 2：independent Q-learning

每个 agent 运行自己的 Q-table，以 joint state 为键。每一步：两者都选择 ε-greedy actions，收集 joint transition，每个 agent 用 shared reward 更新自己的 Q。

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

这个任务上能工作，因为 rewards 稠密且一致。在 tightly-coupled tasks 上会失败（例如一个 agent 必须*等待*另一个 agent 的任务）。

### Step 3：centralized Q with decomposed-value update

使用一个 joint actions 上的 Q：`Q(s, a_1, a_2)`。从 shared reward 更新。执行时通过边际化来 decentralize：`π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`。用指数级 joint action space 换一个*正确*的 global view。

### Step 4：简单 self-play（adversarial 2-agent）

同一个 agent，两个角色。训练 agent A 对抗 agent B；每 `K` 个 episodes，把 A 的 weights 复制给 B。对称训练，持续进步。AlphaZero 配方的微缩版。

## 常见陷阱

- **Non-stationary replay。** Independent agents 的 experience replay 比单智能体更糟，因为旧 transitions 是由如今已过时的 opponents 生成的。修复：relabel，或按 recency 加权。
- **Credit assignment ambiguity。** 长 episode 后只有 shared reward；没有清晰方式判断哪个 agent 有贡献。修复：counterfactual baselines（COMA），或 per-agent reward shaping。
- **Policy drift / chasing。** 每个 agent 的 best response 都随着另一个 agent 的更新而变化。修复：centralized critic、较慢 learning rates，或一次冻结一方。
- **通过 coordination 的 reward hacking。** Agents 找到设计者没预料到的 coordinated exploits。Auction agents 收敛到 bid zero。修复：仔细设计 reward、加入行为约束。
- **Exploration redundancy。** 两个 agents 探索相同的 state-action pairs。修复：per-agent entropy bonuses，或 role-conditioning。
- **League cycles。** 纯 self-play 可能陷入 dominance cycle。修复：使用多样 opponents 的 league play。
- **Sample explosion。** `n` 个 agents × state space × joint actions。用 function approximation 近似；使用 factored action spaces（每个 agent 一个 policy output head）。

## 实际使用

2026 年 MARL 应用地图：

| 领域 | 方法 | 备注 |
|------|------|------|
| Cooperative navigation / manipulation | MAPPO / QMIX | CTDE；shared critic + decentralized actors。 |
| Two-player games（chess、Go、poker） | Self-play with MCTS（AlphaZero） | Zero-sum；对称训练。 |
| Complex multiplayer（Dota、StarCraft） | League play + imitation pretraining | OpenAI Five，AlphaStar。 |
| Autonomous-vehicle fleets | CTDE MAPPO / PPO with attention | Partial obs；可变团队规模。 |
| Auction markets | Game-theoretic equilibrium + RL | `n` → ∞ 时使用 mean-field RL。 |
| LLM multi-agent systems（Phase 16） | Natural-language comm + role conditioning | RL loop 位于 agent-planning layer。 |

到 2026 年，MARL 最大的增长领域是基于 LLM 的系统：成群的 language-model agents 进行协商、辩论、构建软件。RL 出现在*trajectory-level* 输出上的 preference optimization，而不是 token-level（Phase 16 · 03）。

## 交付成果

保存为 `outputs/skill-marl-architect.md`：

```markdown
---
name: marl-architect
description: Pick the right multi-agent RL regime (IPPO, CTDE, self-play, league) for a given task.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with `n` agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).
```

## 练习

1. **简单。** 在 2-agent cooperative GridWorld 上训练 independent Q-learning。mean return > 0 需要多少 episodes？绘制 joint learning curve。
2. **中等。** 加入一个“coordination”任务：只有当两个 agents 在同一回合踏上 goal 时，goal 才算达成。Independent Q 还会收敛吗？哪里坏了？
3. **困难。** 为 MAPPO-style training 实现一个 centralized critic，并在 coordination task 上和 independent PPO 比较收敛速度。

## 关键术语

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| Markov game | “Multi-agent MDP” | `(S, A_1, …, A_n, P, R_1, …, R_n)`；每个 agent 有自己的 reward。 |
| CTDE | “Centralized training, decentralized execution” | 训练时使用 joint critic；每个 agent 的 policy 只使用 local obs。 |
| IPPO | “Independent PPO” | 每个 agent 分别运行 PPO。简单 baseline；常常被低估。 |
| MAPPO | “Multi-agent PPO” | 带有以 global state 为条件的 centralized value function 的 PPO。 |
| QMIX | “Monotonic value decomposition” | `Q_tot = f_monotone(Q_1, …, Q_n)` 允许 decentralized argmax。 |
| COMA | “Counterfactual multi-agent” | Advantage = 我的 Q 减去对我的 action 边际化后的 expected Q。 |
| Self-play | “Agent vs past self” | 单个 agent，两个角色；zero-sum games 的标准方法。 |
| League play | “Population training” | 缓存过去 policies，从 pool 中采样 opponents；处理策略循环。 |

## 延伸阅读

- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275)：使用 centralized critic 的 CTDE。
- [Foerster et al. (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926)：用于 credit assignment 的 counterfactual baselines。
- [Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485)：带 monotonicity 的 value decomposition。
- [Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955)：PPO 对 MARL 出人意料地强。
- [Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z)：大规模 league play。
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270)：zero-sum games 中的纯 self-play。
- [Sutton & Barto (2018). Ch. 15 — Neuroscience & Ch. 17 — Frontiers](http://incompleteideas.net/book/RLbook2020.pdf)：包含教材中对 multi-agent settings 的简短处理，以及 CTDE 要解决的 non-stationarity 问题。
- [Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635)：覆盖 cooperative、competitive 和 mixed MARL 以及 convergence results 的综述。
