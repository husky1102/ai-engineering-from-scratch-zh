# LLMs 的 Swarm Optimization（PSO, ACO）

> Bio-inspired optimization 正在 LLM 时代回归。**LMPSO**（arXiv:2504.09247）使用 PSO，其中每个 particle 的 velocity 是 prompt，LLM 生成下一个 candidate；在 structured-sequence outputs（math expressions、programs）上效果很好。**Model Swarms**（arXiv:2410.11163）把每个 LLM expert 当作 model-weight manifold 上的 PSO particle，并报告在 9 个 datasets、12 个 baselines 上仅用 200 instances 就得到 **13.3% average gain**。**SwarmPrompt**（ICAART 2025）混合 PSO + Grey Wolf 做 prompt optimization。**AMRO-S**（arXiv:2603.12933）是 ACO-inspired pheromone specialists，用于 multi-agent LLM routing：**4.7x speedup**、interpretable routing evidence、quality-gated asynchronous update，将 inference 与 learning 解耦。本课在 prompt parameter space 上实现 PSO，并在 agent routing 上实现 ACO，测量为什么这些 classical algorithms 适合 LLM 时代，以及何时不适合。

**类型：** 学习 + 构建
**语言：** Python (stdlib)
**先修：** Phase 16 · 09 (Parallel Swarm Networks), Phase 16 · 14 (Consensus and BFT)
**时间：** ~75 分钟

## 要解决的问题

你有一个 prompt，在 task eval 上得分 62%。你想改进它。朴素做法是 gradient-free manual tweaking，但它扩展得很差。Reinforcement learning 需要 reward signals 和足够 rollouts 来训练。对 prompts 做 backprop 也并不真正可行，因为 prompt 是 discrete string，不是 differentiable parameter。

Classical bio-inspired optimization：PSO 用于 continuous search spaces，ACO 用于 path selection，正是为这种 regime 设计的：gradient-free、population-based、每次 evaluation 便宜。把它们和 LLMs 配对，用于 gradient-free search step，你会得到一个出人意料实用的 optimizer。

同样 patterns 也适用于 multi-agent systems 中的 agent *routing*。ACO-style pheromone trail 会记录哪个 agent 在哪类 task-type 上效果最好，让 router exploit trail，并让 pheromones decay，以便 routes 能重新发现。

## 核心概念

### PSO refresher（Kennedy & Eberhart 1995）

Particle Swarm Optimization：continuous search space 中的 particles population。每个 particle 有 position `x_i` 和 velocity `v_i`。每次 iteration：

```text
v_i <- w * v_i + c1 * r1 * (p_best_i - x_i) + c2 * r2 * (g_best - x_i)
x_i <- x_i + v_i
evaluate fitness(x_i)
update p_best_i if improved
update g_best if global best
```

其中 `p_best` 是 particle 自己的 best，`g_best` 是 swarm 的 best，`w, c1, c2` 是 inertia + cognitive + social weights，`r1, r2` 是 random factors。

### LLM outputs 上的 PSO：LMPSO

arXiv:2504.09247 将 PSO 适配到 LLM-generated structured outputs（math expressions、programs）。每个 particle 是 candidate output。Velocity 是一个 *prompt*，描述如何把 current output 朝 personal/global best 修改。LLM 从 velocity prompt 生成 new output。“inertia” of the velocity 是类似 “make small incremental changes” 的 prompt。

它在这些情况下效果好：
- output 是 structured（parseable、evaluable）。
- fitness 是 automatic（test runs、arithmetic evaluation）。
- population 小（约 10-30 particles），让 total LLM calls 可控。

当 fitness 需要 human review 时，它效果不好：per-iteration cost 会变得 prohibitive。

### Model Swarms

arXiv:2410.11163 将 PSO 从 output layer 移到 *model* layer。每个 “particle” 是一个 expert LLM（parameters）。swarm 通过 gradient-free update 将 parameters 移向 collective best。报告结果：仅用每 iteration 200 instances，就在 9 个 datasets、12 个 baselines 上得到 13.3% average gain。

关键 insight 是：LLM expert models 在 shared parameter manifold（adapter weights、LoRA deltas）中本来就相近。这个 low-dimensional subspace 上的 PSO 便宜且有效。

### ACO refresher（Dorigo 1992）

Ant Colony Optimization：ants traverse graph；每条 path 有 pheromone trail。ant move probabilities 按 pheromone strength 加权。完成 task 的 ants 按 solution quality deposit pheromone。pheromone 随时间 decay。

### AMRO-S：用于 agent routing 的 ACO

arXiv:2603.12933 使用 ACO 做 multi-agent routing。每个 task-type 是 “destination”；每个 agent 是可能 route。产出好 outputs 的 routes 会增强 pheromones。关键贡献：

- **Interpretable routing evidence.** Pheromone strength 是 human-readable signal。
- **Quality-gated asynchronous update.** 只有 quality checks 通过后才更新 pheromones，将 inference 与 learning 解耦。
- multi-agent routing benchmark 上 **4.7x speedup**。

quality gate 很重要：没有它，fast-but-wrong agents 会积累 pheromone，系统会 lock in 到坏 routes。

### 何时为 LLMs 使用 PSO / ACO

**Use PSO when:**
- search space 是 continuous，或能映射到 continuous parameters（prompt embeddings、LoRA weights、numeric generation parameters）。
- fitness 便宜且 automatic。
- population 可以小（10-30）。

**Use ACO when:**
- 你有 routing 或 path-selection problem。
- decisions 会随时间 reinforce（相同 task types 会回来）。
- 你需要 routing decisions 的 interpretable evidence。

**不要在这些情况下使用二者：**
- fitness 需要 human review（每次 iteration 太贵）。
- search space 是 PSO 覆盖不了的 discrete combinatorial 结构（改用 genetic algorithms）。
- real-time decisions 需要 strict latency（PSO/ACO 相对 single-pass heuristics 收敛慢）。

### 为什么 bio-inspired 仍然胜出

Gradient-based methods 需要 differentiable signals。LLM outputs 和 routing decisions 并不 trivially differentiable。Pseudo-gradient methods（reinforcement-learned routers、DPO-style prompt tuners）有效，但需要 expensive training。

PSO 和 ACO 只需要一个 *evaluator* function。如果你能给 candidate output 或 routing decision 打分，就能在这个 space 上优化。这让 applicability bar 低得多。

### Practical limits

- **Population budget.** N particles × T iterations × per-eval cost。LLM evals 如果约 $0.02 / call，一个 20-particle PSO 跑 50 iterations 约 $20。要相应规划。
- **Exploration vs exploitation.** Pheromone decay rate 和 PSO inertia 需要权衡；decay 太快会忘记 solutions，太慢会 stuck on early local optima。
- **Catastrophic drift.** 如果 fitness landscape shifted（新 data distribution），两种 algorithms 都可能先 converge 再 diverge。监控 best-fitness stability。

## 动手实现

`code/main.py` 实现：

- `LMPSO`：numeric prompt parameters（temperature、top_k weights）上的 PSO。每个 particle 的 “LLM generation” 被模拟为 scripted fitness function。运行 30 iterations 并展示 g_best convergence。
- `AMRO_S`：ACO-style routing。3 个 agents、4 种 task types、pheromone matrix、100 个 routed tasks。打印随时间变化的 (task_type → agent choices) distribution，展示 trail formation。
- 对比：同一个 task stream 上的 random routing vs ACO routing。测量 quality 和 latency。

运行：

```text
python3 code/main.py
```

预期输出：
- LMPSO：g_best fitness 在 30 iterations 内从 random 提升到 near-optimal。
- AMRO-S：pheromone table 稳定到每个 task-type 对应的正确 agent；ACO routing 在 quality 上比 random 高约 30-40%，并且因 fewer retries 降低 latency。

## 实际使用

`outputs/skill-swarm-optimizer.md` 帮你在 LLM / agent optimization problems 中选择 PSO、ACO、genetic algorithms 和 gradient-based optimizers。

## 交付成果

- **Start small.** 10-20 particles，20-50 iterations。只有当 convergence curve 显示明确 gain 时再扩大。
- **Log pheromones or g_best per iteration.** 没有 trail 的 swarm optimizers debug 起来很痛苦。
- **Quality-gate updates.** 尤其是 ACO routing：fast-and-wrong agents 不得累积 pheromone。
- **Reset decay on distribution shift.** eval distribution 变化时，aged pheromones 已 stale；临时 reset 或加倍 decay rate。
- **Cap the per-iteration cost.** 发出 cost-per-iteration metric。每 iteration 花 $500 只换来 0.5% gain 的 PSO 不可交付。

## 练习

1. 运行 `code/main.py`。观察 LMPSO convergence。改变 population size：5、10、20、50。time-to-converge 在什么 size 后饱和？
2. 实现一个 “catastrophic drift” 实验：iteration 30 后改变 fitness function。PSO 适应多快？resetting `p_best` 是否有帮助？
3. 给 AMRO-S 添加 quality gate：只在 eval score > 0.7 的 runs 上 deposit pheromone。相对 un-gated version，这如何改变 convergence？
4. 阅读 LMPSO（arXiv:2504.09247）。把论文中的 “velocity as a prompt” 映射回你的 numeric velocity。simulation 丢失了什么，又保留了什么？
5. 阅读 AMRO-S（arXiv:2603.12933）。实现 decoupled “inference fast-path” 和 asynchronous pheromone update。这如何改变 sustained load 下的 system latency？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| PSO | “Particle Swarm Optimization” | Kennedy-Eberhart 1995。Population-based gradient-free optimizer。 |
| ACO | “Ant Colony Optimization” | Dorigo 1992。通过 pheromone trails 做 path/route optimization。 |
| LMPSO | “PSO with LLM generation” | arXiv:2504.09247。Velocity 是 prompt；LLM 产出 candidates。 |
| Model Swarms | “PSO on expert weights” | arXiv:2410.11163。model parameter subspace 上的 gradient-free update。 |
| AMRO-S | “ACO for agent routing” | arXiv:2603.12933。task-type × agent 上的 pheromone matrix。 |
| p_best / g_best | “Personal / global best” | 目前找到的 per-particle 和 swarm-wide best solutions。 |
| Pheromone | “Routing memory” | edge 上的 strength；随时间 decay；按 quality deposit。 |
| Quality-gated update | “Only learn from good runs” | pheromone deposit 以 quality check 为条件。 |
| Catastrophic drift | “Distribution shift” | fitness landscape 变化；旧 p_best 和 pheromones 变 stale。 |

## 延伸阅读

- [Kennedy & Eberhart — Particle Swarm Optimization](https://ieeexplore.ieee.org/document/488968) — 1995 PSO paper
- [Dorigo — Ant Colony Optimization](https://www.aco-metaheuristic.org/about.html) — 1992 ACO foundations
- [LMPSO — Language Model Particle Swarm Optimization](https://arxiv.org/abs/2504.09247) — structured LLM outputs 的 PSO
- [Model Swarms — gradient-free LLM expert optimization](https://arxiv.org/abs/2410.11163) — model-weight subspace 上的 PSO
- [AMRO-S — ant-colony multi-agent routing](https://arxiv.org/abs/2603.12933) — 带 quality gate 的 pheromone-driven routing
