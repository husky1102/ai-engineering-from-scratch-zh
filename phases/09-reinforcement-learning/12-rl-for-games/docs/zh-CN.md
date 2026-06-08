# 游戏强化学习：AlphaZero、MuZero 与 LLM 推理时代

> 1992：TD-Gammon 用纯 TD 在双陆棋中击败人类冠军。2016：AlphaGo 击败李世石。2017：AlphaZero 从零开始统治 chess、shogi 和 Go。2024：DeepSeek-R1 证明同一个配方，在用 GRPO 替代 PPO 后，也适用于 reasoning。游戏是推动本阶段每个突破的 benchmark。

**类型:** Build
**语言:** Python
**先修:** Phase 9 · 05（DQN），Phase 9 · 08（PPO），Phase 9 · 09（RLHF），Phase 9 · 10（MARL）
**时间:** ~120 分钟

## 要解决的问题

游戏拥有 RL 想要的一切。干净的 reward（win/loss）。无限 episodes（self-play resets）。完美 simulation（游戏*就是* simulator）。离散或小型连续 action spaces。强迫 adversarial robustness 的 multi-agent structure。

而且游戏是每个重大 RL 突破的测试场。TD-Gammon（backgammon，1992）。Atari-DQN（2013）。AlphaGo（2016）。AlphaZero（2017）。OpenAI Five（Dota 2，2019）。AlphaStar（StarCraft II，2019）。MuZero（learned model，2019）。AlphaTensor（matrix multiplication，2022）。AlphaDev（sorting algorithms，2023）。DeepSeek-R1（math reasoning，2025）：最新证明 game-RL techniques 能在文本上工作。

本 capstone 通过单一统一视角考察三种里程碑架构：AlphaZero、MuZero 和 GRPO：**self-play + search + policy improvement**。每个都是前一个的泛化；尤其是 GRPO，它把 AlphaZero 的配方应用到 LLM reasoning，把 tokens 当作 actions，把数学验证当作 win signal。

## 核心概念

![AlphaZero ↔ MuZero ↔ GRPO：同一个 loop，不同 environments](../assets/rl-games.svg)

**统一 loop。**

```text
while True:
    trajectory = self_play(current_policy, search)     # play game against self
    policy_target = search.improved_policy(trajectory) # search improves raw policy
    policy_net.update(policy_target, value_target)     # supervised on search output
```

**AlphaZero（2017）。** Silver 等人。给定一个规则已知的游戏（chess、shogi、Go）：

- Policy-value network：一个 tower `f_θ(s) → (p, v)`。`p` 是 legal moves 上的 prior。`v` 是期望 game outcome。
- Monte Carlo Tree Search（MCTS）：在每一步，展开可能 continuations 的树。使用 `(p, v)` 作为 prior + bootstrap。通过 UCB（PUCT）选择 nodes：`a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`。
- Self-play：agent-vs-agent 下棋。在 move `t`，MCTS visit distribution `π_t` 成为 policy training target。
- Loss：`L = (v - z)² - π · log p + c · ||θ||²`。`z` 是 game outcome（+1 / 0 / -1）。

零 human knowledge。零 handcrafted heuristics。一个单一配方，分别在数千万盘 self-play games 后掌握 chess、shogi 和 Go。

**MuZero（2019）。** Schrittwieser 等人。移除规则已知的要求。

- 不使用固定 environment，而是学习一个 *latent dynamics model* `(h, g, f)`：
  - `h(s)`：把 observation 编码为 latent state。
  - `g(s_latent, a)`：预测下一个 latent state + reward。
  - `f(s_latent)`：预测 policy prior + value。
- MCTS 在*学习到的 latent space* 中运行。同样的 search，同样的 training loop。
- 可用于 Go、chess、shogi *以及* Atari：一个算法，不需要规则知识。

**Stochastic MuZero（2022）。** 加入 stochastic dynamics 和 chance nodes；扩展到 backgammon 类游戏。

**Muesli、Gumbel MuZero（2022-2024）。** 对 sample efficiency 和 deterministic search 的改进。

**GRPO（2024-2025）。** DeepSeek-R1 配方。同样是 AlphaZero 形状的 loop，应用到 language-model reasoning：

- “Game”：回答数学 / coding / reasoning problem。“Win” = verifier（test case passes、numerical answer matches）返回 1。
- Policy：LLM。Actions：tokens。State：prompt + response-so-far。
- 没有 critic（PPO-style V_φ）。而是对每个 prompt，从 policy 采样 `G` 个 completions。计算每个 reward。用 **group-relative advantage** `A_i = (r_i - mean_r) / std_r` 作为 REINFORCE-style update 的信号。
- 到 reference policy 的 KL penalty，用于防止 drift（类似 RLHF）。
- 完整 loss：

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

没有 reward model，没有 critic，没有 MCTS。Group-relative baseline 替代三者。在 reasoning benchmarks 上用少得多的 compute 匹配或超过 PPO-RLHF 质量。

**完整 R1 配方。** DeepSeek-R1（DeepSeek 2025）在一篇论文中包含两个模型：

- **R1-Zero。** 从 DeepSeek-V3 base model 开始。没有 SFT。直接用两个 reward components 应用 GRPO：*accuracy reward*（rule-based：最终答案是否能解析成正确数字 / 代码是否通过 unit tests）和 *format reward*（completion 是否把 chain-of-thought 包在 `<think>…</think>` tags 中）。经过数千步，平均 response length 从 ~100 增长到 ~10,000 tokens，math benchmark scores 爬升到接近 o1-preview 的水平。模型从零开始学会 reasoning。缺点：它的 chains of thought 往往不可读、混合语言、缺少风格打磨。
- **R1。** 用四阶段 pipeline 修复 R1-Zero 的可读性问题：
  1. **Cold-start SFT。** 收集几千条带干净格式的 long-CoT demonstrations。在它们上 supervised-finetune base model。这提供一个可读起点。
  2. **Reasoning-oriented GRPO。** 使用 accuracy+format rewards 加一个 *language-consistency* reward 应用 GRPO，防止 code-switching。
  3. **Rejection sampling + SFT round 2。** 从 RL checkpoint 采样 ~600K reasoning trajectories，只保留最终答案正确且 CoT 可读的样本，并与 ~200K non-reasoning SFT examples（writing、QA、self-cognition）结合。再次 fine-tune base。
  4. **Full-spectrum GRPO。** 再做一轮 RL，同时覆盖 reasoning（rule-based rewards）和 general alignment（helpfulness/harmlessness preference-based rewards）。

结果是在 open weights 下匹配 o1 的 AIME 和 MATH-500，并且小到可以蒸馏。同一论文还通过在 R1 reasoning traces 上做 SFT 发布了六个 distilled dense models（Qwen-1.5B 到 Llama-70B）：student 不做 RL。强 RL teacher 的 distillation 在 student scale 上持续优于从零 RL。

**为什么 reasoning 用 GRPO 而不是 PPO。** DeepSeekMath 论文（2024 年 2 月）给出三个原因：（1）不需要训练 value network，内存减半；（2）group baseline 自然处理 reasoning tasks 产生的 sparse end-of-trajectory reward；（3）per-prompt normalization 让不同难度问题的 advantages 可比，而 PPO 的单个 critic 做不到。

**Search-free vs search-based。** 游戏已经分支：

- *长 horizon 的 perfect-information games*（Go、chess）：仍然是 search-based。AlphaZero / MuZero 占主导。
- *LLM reasoning*：生产中尚无 MCTS；使用完整 rollouts 上的 GRPO，inference compute 用 best-of-N。Process reward models（PRMs）暗示 step-level search 可能会被重新加入。

## 动手实现

`code/main.py` 中的代码实现了**微型 GRPO**：一个带多组样本的 bandit。算法和 LLM 上的一样；只是 policy 和 environment 更简单。它教授 *loss* 和 *group-relative advantage*，这是 2025 年的创新。

### Step 1：微型 verifier environment

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

真实 GRPO 中，verifier 会运行 unit tests 或检查数学等价。

### Step 2：policy：每个 prompt 上 K 个 answer tokens 的 softmax

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

等价于以 prompt 为条件的 LLM final-layer output。

### Step 3：group sampling 和 group-relative advantage

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL penalty: pull theta toward reference
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

Group-relative advantage 是 2024 年 DeepSeek 技巧。无需 critic。“Baseline” 是 group mean，normalization 使用 group std。

### Step 4：和 REINFORCE baseline（value-free）比较

同样设置、同样 compute，普通 REINFORCE。GRPO 收敛更快、更稳定。

### Step 5：观察 entropy 和 KL

和 RLHF 一样的诊断：mean KL to reference、policy entropy、reward-over-time。一旦这些稳定，训练就完成了。

## 常见陷阱

- **通过 verifier gaming 的 reward hacking。** GRPO 继承 RLHF 的风险：如果 verifier 错误或可被利用，LLM 会找到 exploit。鲁棒 verifiers（多个 test cases、formal proofs）很重要。
- **Group size 太小。** Group baseline 的方差按 `1/√G` 缩放。低于 `G = 4` 时，advantage signal 很吵；标准选择是 `G = 8` 到 `64`。
- **Length bias。** 不同长度的 LLM completions 有不同 log-probabilities。按 token count 归一化，或使用 sequence-level log-prob，或截断到 max length。
- **纯 self-play cycles。** AlphaZero-style training 在 general-sum games 上可能陷入 dominance loops。通过多样 opponent pools（league play，Lesson 10）缓解。
- **Search-policy mismatch。** AlphaZero 训练 policy 去模仿 search output。如果 policy net 太小，无法表示 search 的 distribution，训练会停滞。
- **Compute floor。** MuZero / AlphaZero 需要巨大 compute。一次 ablation 通常是数百 GPU-hours。学习用的微型 demo 存在（例如 Connect Four 上的 AlphaZero）。
- **Verifier coverage。** 能让 buggy solution 通过的 unit tests 会强化 bug。设计能捕获 edge cases 的 verifiers。

## 实际使用

2026 年 game-RL 版图，按领域：

| 领域 | 主导方法 |
|------|----------|
| Two-player zero-sum board games（Go、chess、shogi） | AlphaZero / MuZero / KataGo |
| Imperfect info card games（poker） | CFR + deep learning（DeepStack、Libratus、Pluribus） |
| Atari / pixel games | Muesli / MuZero / IMPALA-PPO |
| Large multiplayer strategy（Dota、StarCraft） | PPO + self-play + league（OpenAI Five、AlphaStar） |
| LLM math/code reasoning | GRPO（DeepSeek-R1、Qwen-RL、open replications） |
| LLM alignment | DPO / RLHF-PPO（不是 GRPO；verifier 是 preference，不可验证） |
| Robotics | PPO + DR（不是 game-RL，但使用相同的 policy-gradient tools） |
| Combinatorial problems | AlphaZero variants（AlphaTensor、AlphaDev） |

这个*配方*：self-play、search-augmented improvement、policy distillation，横跨 text、pixels 和 physical control。GRPO 是最年轻的实例；还会有更多。

## 交付成果

保存为 `outputs/skill-game-rl-designer.md`：

```markdown
---
name: game-rl-designer
description: Design a game-RL or reasoning-RL training pipeline (AlphaZero / MuZero / GRPO) for a given domain.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Given a target (perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial), output:

1. Environment fit. Known rules? Markov? Stochastic? Multi-agent? Informs AlphaZero vs MuZero vs GRPO.
2. Search strategy. MCTS (PUCT with learned prior), Gumbel-sampled, best-of-N, or none.
3. Self-play plan. Symmetric self-play / league / offline data / verifier-generated.
4. Target signal. Game outcome / verifier reward / preference / learned model. Include robustness plan.
5. Diagnostics. Win rate vs baseline, ELO curve, verifier pass rate, KL to reference.

Refuse AlphaZero on imperfect-info games (route to CFR). Refuse GRPO without a trusted verifier. Refuse any game-RL pipeline without a fixed baseline opponent set (self-play ELO is uncalibrated otherwise).
```

## 练习

1. **简单。** 在 `code/main.py` 中实现 GRPO bandit。训练 2 个 prompts × 每个 4 个 answer tokens。使用 `G=8` 在 < 1,000 次更新内收敛。
2. **中等。** 插入 PPO（clipped）和 vanilla REINFORCE。在同一个 bandit 上和 GRPO 比较 sample efficiency 与 reward variance。
3. **困难。** 扩展到长度为 2 的“reasoning chain”：agent 发出两个 tokens，verifier 给这个 pair 奖励。测量 GRPO 如何处理 two-step sequences 上的 credit assignment。（提示：对*完整 sequence* 计算 group advantage，并传播到两个 token positions。）

## 关键术语

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| MCTS | “Tree search with learned net” | Monte Carlo Tree Search；带 learned `(p, v)` priors 的 UCB1/PUCT selection。 |
| AlphaZero | “Self-play + MCTS” | 训练 policy-value net 去匹配 MCTS visits 和 game outcome。 |
| MuZero | “Learned-model AlphaZero” | 同样 loop，但通过 learned dynamics 在 latent space 中运行。 |
| GRPO | “Critic-free PPO” | Group Relative Policy Optimization；带 group-mean baseline + KL 的 REINFORCE。 |
| PUCT | “AlphaZero's UCB” | `Q + c · p · √N / (1 + N_a)`：在 value estimate 与 prior 之间平衡。 |
| Self-play | “Agent vs past self” | zero-sum 的标准方法；对称训练信号。 |
| League play | “Population-based self-play” | 把 past + current + exploiters 作为 opponents 采样。 |
| Verifier reward | “Verifiable RL” | Reward 来自确定性 checker（tests pass、answer matches）。 |
| Process reward | “PRM” | 给每个 reasoning step 打分，而不仅是最终答案。 |

## 延伸阅读

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270)。
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404)。
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4)。
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z)。
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300)：引入 GRPO 和 group-relative baseline 的论文。
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)：完整四阶段 R1 配方与 R1-Zero ablation。
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400)：大规模 CFR + deep-learning。
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343)：一切开始的论文。
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)：用 custom reward functions 应用 GRPO 的生产参考。
- [Qwen Team (2024). Qwen2.5-Math — GRPO replication](https://github.com/QwenLM/Qwen2.5-Math)：多尺度 R1 配方 open replication。
- [Sutton & Barto (2018). Ch. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf)：self-play、search 和 R1 在 LLM 规模实例化的“designed reward”的教材框架。
