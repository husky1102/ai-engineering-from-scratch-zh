# MARL：MADDPG、QMIX、MAPPO

> multi-agent coordination 的 reinforcement-learning heritage，在 2026 年仍然影响 LLM-agent systems。**MADDPG**（Lowe et al., NeurIPS 2017, arXiv:1706.02275）引入 Centralized Training, Decentralized Execution（CTDE）：训练期间每个 critic 能看到所有 agents 的 states 和 actions；test time 只运行 local actors。适用于 cooperative、competitive 和 mixed settings。**QMIX**（Rashid et al., ICML 2018, arXiv:1803.11485）是带 monotonic mixing network 的 value-decomposition；per-agent Qs 组合成 joint Q，使 `argmax` 能干净分布式执行，是 StarCraft Multi-Agent Challenge（SMAC）上的 dominant 方法。**MAPPO**（Yu et al., NeurIPS 2022, arXiv:2103.01955）是带 centralized value function 的 PPO；在 particle-world、SMAC、Google Research Football、Hanabi 上 minimal tuning 就 “surprisingly effective”。这些支撑必须 decentrally act 的 agent teams 的 training policies。MAPPO 是 **2026 cooperative-MARL default baseline**。本课从一个小 grid-world toy 构建每种方法，在接触 LLM-agent training 前，把三个 idea 落到肌肉记忆里。

**类型：** 学习
**语言：** Python (stdlib, small NumPy-free implementations)
**先修：** Phase 09 (Reinforcement Learning), Phase 16 · 09 (Parallel Swarm Networks)
**时间：** ~90 分钟

## 要解决的问题

LLM-agent systems 越来越多地训练 inter-agent coordination policies：何时 defer、何时 act、调用哪个 peer。告诉你如何训练这些 policies 的文献是 Multi-Agent Reinforcement Learning（MARL），它早于 LLM wave，并有一小组 dominant algorithms。

没有 pattern vocabulary，读 MARL papers 很痛苦。Centralized training with decentralized execution（CTDE）、value decomposition 和 centralized critics 不是 buzzwords，而是对具体 problems 的具体 answers：

- Independent RL（每个 agent 独自学习）从每个 agent 的视角看是 non-stationary。不好。
- Centralized RL（一个 agent 控制全部）无法 scale，并违反 execution constraints。
- CTDE 取得两者优点：用 global information 训练，用 local policies 部署。

## 核心概念

### 论文使用的三个 environments

- **Particle World (multi-agent particle env).** 简单 2D physics，带 cooperative/competitive tasks。MADDPG 的原始 testbed。
- **StarCraft Multi-Agent Challenge (SMAC).** Cooperative micro-management，partial observation。QMIX 的 testbed。Discrete actions、continuous states。
- **Google Research Football, Hanabi, MPE.** MAPPO baselines。

不同 envs 有不同 action/observation types。algorithms 会相应选择。

### MADDPG（2017）：CTDE pattern

每个 agent `i` 都有 actor `mu_i(o_i)`，把自己的 observation 映射到 action。每个 agent 也有 critic `Q_i(x, a_1, ..., a_n)`，在 training 期间能看到所有 observations 和所有 actions。actor 通过 policy gradient，针对 critic 的 evaluation 更新。

```text
actor update:    grad_theta_i J = E[grad_theta mu_i(o_i) * grad_a_i Q_i(x, a_1..n) at a_i=mu_i(o_i)]
critic update:   TD on Q_i(x, a_1..n) given next-state joint estimate
```

为什么用 CTDE：training time 我们知道每个人的 actions；用它降低每个 critic 的 variance。deploy time 每个 agent 只看到 `o_i` 并调用 `mu_i(o_i)`。

Failure mode：critics 随 N agents 增长（input 包含所有 actions）。没有 approximations 时，超过约 10 个 agents 后不易 scale。

### QMIX（2018）：value decomposition

仅 cooperative。Global reward 是 per-agent Q-values 的一个 monotone function：

```text
Q_tot(tau, a) = f(Q_1(tau_1, a_1), ..., Q_n(tau_n, a_n)),   df/dQ_i >= 0
```

monotonicity 保证 `argmax_a Q_tot` 可以由每个 agent 独立选择 `argmax_{a_i} Q_i` 来计算。这正是你需要的 **decentralized execution property**。training time，一个 mixing network 从 per-agent Qs 产出 `Q_tot`。

QMIX 在 SMAC 上获胜的原因：cooperative StarCraft micro-management 具有 homogeneous agents、local obs、global reward，完美适合 value decomposition。

Failure mode：monotonicity constraint 很 restrictive；有些 tasks 的 reward structures 不是 monotone decomposable（例如一个 agent 为团队牺牲）。Extensions（QTRAN、QPLEX）会放松这一点。

### MAPPO（2022）：被低估的 default

Multi-Agent PPO：带 centralized value function 的 PPO。每个 agent 有自己的 policy；所有 agents 共享（或各自拥有）能看到 full state 的 value functions。Yu et al. 2022 在五个 benchmarks 上将 MAPPO 与 MADDPG、QMIX 及其 extensions 对比，发现：

- MAPPO 在 particle-world、SMAC、Google Research Football、Hanabi、MPE 上匹配或击败 off-policy MARL methods。
- 需要 minimal hyperparameter tuning。
- training 稳定；跨 seeds 可复现。

直到这篇论文前，community 低估了 on-policy MARL。到 2026 年，MAPPO 是 cooperative MARL 的 default baseline；任何新方法都必须击败它。

### 为什么 LLM-agent engineers 应该关心

三个直接用途：

1. **Router training.** meta-agent 选择哪个 sub-agent 处理 task。这是一个 MARL problem，有 N 个 decentralized sub-agents 和一个 centralized router。MAPPO 适合。
2. **Role emergence.** 在 generative-agent simulations 中，训练 agents 随时间 adopt complementary roles，是伪装成别的东西的 MARL problem。QMIX-style value decomposition 通过构造强制 complementarity。
3. **Multi-agent tool use.** 当 agents 共享 tools 并竞争 budget，通过 CTDE 训练它们会产生 deployable local policies，并尊重 resource constraints。

实际 caveat：到 2026 年，大多数 production LLM-agent systems 用 prompt 其 policies，而不是训练它们。当你有 (a) 大量 interaction data，(b) 清晰 reward signal，(c) 投入 training infrastructure 的意愿时，MARL 才进入。

### CTDE 作为 RL 之外的 design pattern

即使不训练，CTDE 也是有用 architecture pattern：

- *design* 期间，假设 full team visibility。
- *runtime* 期间，强制 decentralized execution：每个 agent 只看到 `o_i`。

这个 pattern 会迫使你保持 per-agent state 显式，并提前思考 partial observability。许多 production multi-agent systems 静默假设 shared state everywhere；CTDE discipline 能阻止这一点。

### Non-stationarity problem

多个 agents 同时学习时，每个 agent 的 environment（包含其他 agents' policies）都是 non-stationary。Classical single-agent RL proofs 会破。此课中的 MARL algorithms 都在处理这一点：

- MADDPG：global critic 看到所有 actions，所以其 value estimate 更 stationary。
- QMIX：value decomposition 将 learning 移到 optimality well-defined 的 joint-Q space。
- MAPPO：centralized value function 抑制 others' policy changes 带来的 variance。

在 LLM-agent systems 中，non-stationarity 表现为“我的 agent 上个月能工作，现在 upstream 的另一个 agent 变了，我的开始 misbehave。”用 CTDE 训练 MARL 是原则性修复；prompt-level fixes 更快但更不 durable。

### 本课不覆盖什么

训练真实 networks 是 Phase 09 主题。本课构建 scripted-policy versions，用来展示 CTDE、value-decomposition 和 centralized-value patterns，而不做 gradient updates。目标是在你拿起完整 MARL library（PyMARL、MARLlib、RLlib multi-agent）前内化这些 patterns。

## 动手实现

`code/main.py` 在一个 tiny 2-agent cooperative grid-world 上实现三种 pattern demonstrations：

- Environment：2 个 agents 在 4x4 grid 上，一个 reward pellet。Reward = 任一 agent 到达 pellet 时为 1；task 结束。
- `IndependentAgents`：每个 agent 把其他 agents 当 environment。Baseline。
- `MADDPGStyle`：centralized critic 计算 joint value；actor policies 从中 update。Scripted policy improvement。
- `QMIXStyle`：带 monotone mixer 的 value decomposition。
- `MAPPOStyle`：centralized value function；policies against shared baseline update。

四者运行相同 episodes，并报告 average steps-to-goal。CTDE variants 收敛到比 independent baseline 更短的路径。

运行：

```text
python3 code/main.py
```

预期输出：independent agents 平均约 6 steps；CTDE variants 收敛到约 3.5 steps（4x4 grid 最优为 3）。即使是 scripted policies，pattern difference 也会显现。

## 实际使用

`outputs/skill-marl-picker.md` 是一个 skill，会为给定 multi-agent task 选择 MARL algorithm：cooperative vs competitive、homogeneous vs heterogeneous、action-space type、scale、reward signal。

## 交付成果

MARL 在 production 中很少见。当你确实使用它：

- **Start with MAPPO.** 2022 论文确立了它的 baseline 地位；先复现它能省下数周追逐 fancy methods 的时间。
- **Log every agent's observation and action stream.** 没有 per-agent traces，debugging MARL 几乎无望。
- **Separate training code from execution code.** CTDE 是 discipline；让 execution path 真的只能看到 `o_i`。
- **Reward shaping warning.** MARL 对 reward design 极其敏感。shaping 中一个 coordination bug，agents 就会学会 exploit 它。运行 adversarial tests。
- **For LLM agents**，先考虑 prompt-level policies。只有当 interaction data + reward signal + infrastructure 都存在时，才投入 MARL training。

## 练习

1. 运行 `code/main.py`。测量 independent 与 MAPPO-style agents 之间的 steps-to-goal gap。这个 gap 在 6x6 grid 上会扩大还是缩小？
2. 实现一个 competitive variant：两个 agents、一个 pellet，只有第一个到达者获得 reward。哪种 pattern 能干净处理 competition？历史上是 MADDPG。
3. 阅读 MADDPG（arXiv:1706.02275）Section 3。用你自己的话以 pseudocode 符号化实现 exact critic update rule。
4. 阅读 MAPPO（arXiv:2103.01955）。作者为什么认为 centralized value + PPO 在他们的 benchmarks 上胜过 off-policy MARL？列出三个最强 claims。
5. 将 CTDE 作为 design pattern 应用到一个假想 LLM-agent system（例如 research agent + summarizer + coder）。design time 可用但 runtime 不可用的 joint information 是什么？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| MARL | “Multi-Agent RL” | multi-agent systems 的 reinforcement learning。 |
| CTDE | “Centralized Training, Decentralized Execution” | 用 global info 训练；用 local policies 部署。 |
| MADDPG | “Multi-Agent DDPG” | CTDE，per-agent critic 能看到所有 observations + actions。 |
| QMIX | “Value decomposition” | per-agent Qs 的 monotonic mixing。Cooperative。 |
| MAPPO | “Multi-Agent PPO” | 带 centralized value function 的 PPO。2026 default baseline。 |
| Value decomposition | “Sum of individual Qs” | joint Q 表示为 per-agent Qs 的 monotone function。 |
| Non-stationarity | “Moving targets” | 随其他 agents 学习，每个 agent 的 env 变化。核心 MARL problem。 |
| On-policy / off-policy | “Learn from current / replay” | PPO 是 on-policy（MAPPO）；DDPG 和 Q-learning 是 off-policy。 |
| SMAC | “StarCraft Multi-Agent Challenge” | cooperative micromanagement benchmark；QMIX 的主场。 |

## 延伸阅读

- [Lowe et al. — Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275) — MADDPG；NeurIPS 2017
- [Rashid et al. — QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485) — QMIX；ICML 2018
- [Yu et al. — The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955) — MAPPO；NeurIPS 2022
- [BAIR blog post on MAPPO](https://bair.berkeley.edu/blog/2021/07/14/mappo/) — MAPPO result 的 readable framing
- [SMAC repository](https://github.com/oxwhirl/smac) — StarCraft Multi-Agent Challenge
