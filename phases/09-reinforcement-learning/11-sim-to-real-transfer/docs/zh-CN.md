# Sim-to-Real Transfer

> 在 simulator 中训练出来、到硬件上却失败的 policy，本质上只是记住了 simulator。Domain randomization、domain adaptation 和 system identification 是让 learned controllers 跨过 reality gap 的三件工具。

**类型:** Learn
**语言:** Python
**先修:** Phase 9 · 08（PPO），Phase 2 · 10（Bias/Variance）
**时间:** ~45 分钟

## 要解决的问题

训练真实机器人缓慢、危险且昂贵。一个双足机器人需要数百万个训练 episodes 才能学会走路；真实双足机器人哪怕摔倒一次也会损坏硬件。Simulation 给你无限 reset、确定性可复现、并行环境，并且不会造成物理损伤。

但 simulators 是错的。轴承的摩擦比 MuJoCo 模型更大。摄像头有 simulator 没包含的镜头畸变。电机有延迟、backlash 和 saturation，而 99% 的 sim models 都会跳过。风、灰尘和变化的光照会破坏在无菌渲染上训练的 policy。**Reality gap**：sim distribution 和 real distribution 之间的系统性差异，是机器人 deployed RL 的中心问题。

你需要一个对 *sim-to-real distribution shift* 鲁棒的 policy。三种历史路径：随机化 simulator（domain randomization），用少量真实数据适配 policy（domain adaptation / fine-tuning），或者识别真实系统参数并匹配它们（system identification）。到 2026 年，主流配方把三者都结合起来，并使用大规模并行 simulation（Isaac Sim、Isaac Lab、GPU 上的 Mujoco MJX）。

## 核心概念

![三种 sim-to-real regime：domain randomization、adaptation、system identification](../assets/sim-to-real.svg)

**Domain Randomization（DR）。** Tobin 等人 2017，Peng 等人 2018。在训练期间，随机化每个可能在真实机器人上不同的 sim 参数：质量、摩擦系数、电机 PD gains、传感器噪声、相机位置、光照、纹理、接触模型。policy 学到一个关于“今天处在哪个 sim 中”的条件分布，并在整个范围上泛化。如果真实机器人落在训练 envelope 内，policy 就能工作。

- **优点：** 不需要真实数据。一个配方，许多机器人。
- **缺点：** 过度随机化的训练会产生“通用”但过于谨慎的 policy。太多噪声 ≈ 太多 regularization。

**System Identification（SI）。** 在训练前，把 simulator 的参数拟合到真实世界数据。如果你能测量真实机器人上的 arm-joint friction，就把它插入 sim。然后训练一个期望这些值的 policy。需要访问真实系统，但直接缩小 reality gap。

- **优点：** 精准、低噪声的训练目标。
- **缺点：** 残余 model error 对 policy 不可见；小的未识别效应（例如 motor deadband）仍然会破坏部署。

**Domain Adaptation。** 先在 sim 中训练，再用少量真实数据 fine-tune。两种形式：

- **Real2Sim2Real：** 使用真实 rollouts 学习 residual simulator `f(s, a, z) - f_sim(s, a)`，在修正后的 sim 中训练。用很少真实数据缩小 gap。
- **Observation adaptation：** 训练一个 policy，通过 learned feature extractor（例如 GAN pixel-to-pixel）把 real obs → sim-like obs。controller 保持在 sim 中。

**Privileged learning / teacher-student。** Miki 等人 2022（ANYmal 四足机器人）。在 simulation 中训练一个可以访问 privileged information（ground truth friction、terrain height、IMU drift）的 *teacher*。再蒸馏一个只看到真实传感器 observations 的 *student*。student 学会从 history 中推断 privileged features，并对物理参数保持鲁棒。

**Massively parallel simulation。** 2024-2026。Isaac Lab、Mujoco MJX、Brax 都能在单张 GPU 上运行数千个并行机器人。PPO 配合 4,096 个并行 humanoids，数小时内就能采集数年的经验。当训练分布变宽时，“reality gap”会缩小；当 4,096 个 envs 每个都有不同随机参数时，DR 几乎是免费的。

**2026 年真实世界配方（四足行走示例）：**

1. 使用 massively parallel sim，并 domain-randomized gravity、friction、motor gains、payload。
2. 使用 privileged info（terrain map、body velocity ground truth）训练 teacher policy。
3. 将 teacher 蒸馏为只使用 proprioception（腿部关节编码器）的 student policy。
4. 可选：通过真实 IMU 上的 autoencoder 做 observation adaptation。
5. 部署。在 10+ 个环境中 zero-shot。如果失败，使用 safety-constrained PPO 做几分钟真实世界 fine-tuning。

## 动手实现

本课代码是在带*噪声* transition 的 GridWorld 上演示 domain randomization。我们训练一个 policy，让它在“sim”中经历随机化的 slip probabilities，并在从未训练过的 slip level “real” 上评估。这个形状可以直接映射到 MuJoCo-to-hardware transfer。

### Step 1：参数化 sim

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip` 是 simulator 暴露的参数。在真实机器人中，它可以是 friction、mass、motor gain：任何会在 sim 和 real 之间 shift 的东西。

### Step 2：用 DR 训练

每个 episode 开始时，采样 `slip ~ Uniform[0.0, 0.4]`。训练 PPO / Q-learning / 任意方法。重复很多 episodes。

### Step 3：在“real” slips 上 zero-shot 评估

在 `slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}` 上评估。前四个在训练 support 内；`0.5` 和 `0.7` 在 support 外。DR-trained policy 应该在 support 内接近最优，在 support 外优雅退化。fixed-slip-trained policy 会在训练 slip 外变得脆弱。

### Step 4：和窄训练对比

训练第二个 policy，只使用 `slip = 0.0`。在同一个 `slip` sweep 上评估。你应该看到当 real slip > 0 时，表现立刻灾难性下降。

## 常见陷阱

- **随机化太多。** 在 `slip ∈ [0, 0.9]` 上训练，你的 policy 会过于 risk-averse，甚至不敢尝试最优路径。匹配*预期*真实世界分布，而不是“任何事都可能发生”。
- **随机化太少。** 在很薄的切片上训练，policy 完全无法泛化。使用 adaptive curriculum（Automatic Domain Randomization），随着 policy 改善拓宽分布。
- **参数空间识别错误。** 随机化错误的东西（真实 gap 是 motor delay，却随机化 camera hue），DR 不会有帮助。先 profile 真实机器人。
- **Privileged info leakage。** 如果 teacher 使用 global state 来执行 actions，而不只是 observations，它可能产生 student 无法追上的行为。确保 teacher 的 policy 在给定 observation history 时对 student 是可实现的。
- **Sim-to-sim transfer failure。** 如果你的 policy 对更难的 sim variant 都不鲁棒，它也不会对真实世界鲁棒。部署前始终在 held-out sim variant 上测试。
- **没有真实世界 safety envelope。** 一个在 sim 中有效、也“在 real 中有效”的 policy，如果没有 low-level safety shield，仍然可能损坏硬件。加入 rate limits、torque limits、joint limits，并放在 non-learned controller 中。

## 实际使用

2026 年 sim-to-real stack：

| 领域 | Stack |
|------|-------|
| Legged locomotion（ANYmal、Spot、humanoid） | Isaac Lab + DR + privileged teacher / student |
| Manipulation（dexterous hands、pick-and-place） | Isaac Lab + DR + DR-GAN for vision |
| Autonomous driving | CARLA / NVIDIA DRIVE Sim + DR + real fine-tune |
| Drone racing | RotorS / Flightmare + DR + online adaptation |
| Finger/in-hand manipulation | OpenAI Dactyl（前所未有规模的 DR） |
| Industrial arms | MuJoCo-Warp + SI + small real fine-tune |

对所有尺度的 control，workflow 都一致：尽力拟合 sim，随机化无法拟合的部分，训练巨大的 policies，蒸馏，带 safety shield 部署。

## 交付成果

保存为 `outputs/skill-sim2real-planner.md`：

```markdown
---
name: sim2real-planner
description: Plan a sim-to-real transfer pipeline for a given robot + task, covering DR, SI, and safety.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

Given a robot platform, a task, and access to real hardware time, output:

1. Reality gap inventory. Suspected sources ranked by expected impact (contact, sensing, actuation delay, vision).
2. DR parameters. Exact list, ranges, distribution. Justify each range against real measurements.
3. SI steps. Which parameters to measure; measurement method.
4. Teacher/student split. What privileged info the teacher uses; what obs the student uses.
5. Safety envelope. Low-level limits, emergency stops, backup controller.

Refuse to deploy without (a) a zero-shot sim-variant test, (b) a safety shield, (c) a rollback plan. Flag any DR range wider than 3× measured real variability as likely over-randomized.
```

## 练习

1. **简单。** 在 fixed-slip GridWorld（slip=0.0）上训练 Q-learning agent。在 slip ∈ {0.0, 0.1, 0.3, 0.5} 上评估。绘制 return vs slip。
2. **中等。** 训练一个 DR Q-learning agent，采样 `slip ~ Uniform[0, 0.3]`。在同一 sweep 上评估。在 slip=0.5（out-of-distribution）时，DR 带来了多少收益？
3. **困难。** 实现 curriculum：从 slip=0.0 开始，每当 policy 达到 90% optimal 时拓宽 DR range。测量相对于 fixed DR baseline，达到 slip=0.3 zero-shot 所需的总环境步数。

## 关键术语

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| Reality gap | “Sim-to-real difference” | 训练和部署 physics/sensing 之间的 distribution shift。 |
| Domain randomization (DR) | “Train across random sims” | 训练期间随机化 sim 参数，让 policy 泛化。 |
| System identification (SI) | “Measure real and fit sim” | 估计真实物理参数；设置 sim 与其匹配。 |
| Domain adaptation | “Fine-tune on real data” | sim training 后做少量真实世界 fine-tune；可以适配 obs 或 dynamics。 |
| Privileged info | “Ground truth for teacher” | 只有 sim 拥有的信息；student 必须从 obs history 中推断。 |
| Teacher/student | “Distill privileged -> observable” | teacher 用捷径训练；student 学会在没有这些捷径时模仿。 |
| ADR | “Automatic Domain Randomization” | 随着 policy 改进而拓宽 DR ranges 的 curriculum。 |
| Real2Sim | “Close the gap with real data” | 学习 residual，让 sim 模仿真实 rollouts。 |

## 延伸阅读

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907)：最初的 DR 论文（机器人视觉）。
- [Peng et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537)：用于 dynamics 和四足 locomotion 的 DR。
- [OpenAI et al. (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113)：Dactyl，大规模 ADR。
- [Miki et al. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822)：ANYmal 的 teacher-student。
- [Makoviychuk et al. (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470)：驱动 2025-2026 部署的大规模并行 sim。
- [Akkaya et al. (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113)：ADR curriculum 方法。
- [Sutton & Barto (2018). Ch. 8 — Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf)：支撑现代 sim-to-real pipeline 的 Dyna 框架（用 model 做 planning + rollouts）。
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303)：sim-to-real 方法分类与 benchmark results 综述。
