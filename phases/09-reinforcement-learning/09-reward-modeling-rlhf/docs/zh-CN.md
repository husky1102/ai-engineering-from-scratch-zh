# Reward Modeling 与 RLHF

> 人类写不出“好 assistant response”的 reward function，但可以比较两个回答并选出更好的那个。用这些比较拟合 reward model，再用 RL 让语言模型对它优化。Christiano 2017。InstructGPT 2022。这个配方把 GPT-3 变成了 ChatGPT。到 2026 年，它大多正在被 DPO 取代，但心智模型仍然保留。

**类型:** Build
**语言:** Python
**先修:** Phase 5 · 05（Sentiment），Phase 9 · 08（PPO）
**时间:** ~45 分钟

## 要解决的问题

你已经用 next-token-prediction 目标训练了一个语言模型。它能写出语法正确的英文。它也会撒谎、啰嗦，并且该拒绝时不拒绝。更多 pretraining 不能修复这个问题：web text 是问题本身，不是解药。

你想要一个*标量 reward*，表达“对于 instruction X，response A 比 response B 更好”。手写这个 reward function 不可能。“Helpfulness” 不是一个关于 token 的闭式表达式。但人类可以比较两个输出并标记 preference。这可以低成本大规模收集。

RLHF（Christiano 等人 2017；Ouyang 等人 2022）把 preferences 转换成 reward model，然后用 PPO 让 LM 对这个 reward 优化。三个步骤：SFT → RM → PPO。这就是 2023-2025 年 ChatGPT、Claude、Gemini 和其他 aligned-LLM 上线的配方。

到 2026 年，PPO 步骤大多被 DPO（Phase 10 · 08）取代，因为它更便宜，并且在 alignment tuning 上几乎一样好。但 *reward model* 这部分仍然支撑着每个 Best-of-N sampler、每个 RL-from-verifiable-rewards pipeline，以及每个使用 process reward model 的 reasoning model。理解 RLHF，就理解了整个 alignment stack。

## 核心概念

![三阶段 RLHF：SFT、基于成对偏好的 RM 训练、带 KL penalty 的 PPO](../assets/rlhf.svg)

**Stage 1：Supervised Fine-Tuning（SFT）。** 从 pretrained base model 开始。在目标行为的人类示范上 fine-tune（instruction-following responses、helpful replies 等）。结果是一个 `π_SFT` 模型，它*偏向良好行为*，但仍然有无界的 action space。

**Stage 2：Reward Model 训练。**

- 收集 prompt `x` 对应的 response 对 `(y_+, y_-)`，由人类标注为“y_+ 优于 y_-”。
- 训练 reward model `R_φ(x, y)`，让它给 `y_+` 更高的分数。
- 损失：**Bradley-Terry pairwise logistic**：

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ 是 sigmoid。reward 差值隐含 preference 的 log-odds。BT 自 1952 年（Bradley-Terry）以来就是标准方法，也是现代 RLHF 的主流选择。

- `R_φ` 通常从 SFT model 初始化，并在顶部加一个 scalar head。同一个 transformer backbone；单个线性层输出 reward。

**Stage 3：带 KL penalty、针对 RM 的 PPO。**

- 从 `π_SFT` 初始化可训练策略 `π_θ`。保留一个冻结的 *reference* `π_ref = π_SFT`。
- response `y` 末尾的 reward：

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  KL penalty 防止 `π_θ` 任意漂离 `π_SFT`：它是一个 *regularizer*，不是硬 trust region。`β` 通常为 `0.01`-`0.05`。
- 使用这个 reward 运行 PPO（Lesson 08）。Advantages 在 token-level trajectory 上计算，但 RM 只给完整 response 打分。

**为什么需要 KL？** 没有它，PPO 会很乐意找到 reward-hacking 策略：RM 只在 in-distribution completions 上训练过。一个 out-of-distribution response 可能比任何人类写出的 response 分数都更高。KL 让 `π_θ` 保持在 RM 训练时所在的 manifold 附近。它是 RLHF 中最重要的旋钮。

**2026 年状态：**

- **DPO**（Rafailov 2023）：闭式代数把 Stage 2+3 折叠成 preference data 上的单个 supervised loss。没有 RM，没有 PPO。用一小部分 compute 就能在 alignment benchmark 上达到相同质量。Phase 10 · 08 会讲。
- **GRPO**（DeepSeek 2024-2025）：PPO，但用 group-relative baseline 替代 critic，reward 来自 *verifier*（代码运行 / 数学答案匹配），而不是人类训练的 RM。reasoning models 的主流方法。Phase 9 · 12 会讲。
- **Process reward models（PRMs）：** 给 partial solutions（每个 reasoning step）打分，用于 RLHF 和 GRPO 的 reasoning 变体。
- **Constitutional AI / RLAIF：** 用 aligned LLM 生成 preferences，而不是用人类。扩展 preference budget。

## 动手实现

本课使用极小的合成“prompts”和“responses”，用字符串表示。RM 是 bag-of-tokens 表示上的线性 scorer。没有真实 LLM：pipeline 的*形状*重要，而不是规模。见 `code/main.py`。

### Step 1：合成 preference data

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

在真实 RLHF 中，这会被人类标注者替代。形状 `(prompt, preferred_response, rejected_response)` 完全一致。

### Step 2：Bradley-Terry reward model

线性分数：`R(x, y) = w · bag(y)`。训练它最小化 BT pairwise log-loss：

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

几百次更新后，`w` 会给好词 token 赋正权重，给坏词 token 赋负权重。

### Step 3：RM 之上的 PPO-like policy

我们的玩具策略从 vocabulary 中产生单个 token。我们在 RM 下给 token 打分，计算 `log π_θ(token | prompt)`，加上到 reference 的 KL penalty，并应用裁剪 PPO surrogate。

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo-style update on theta, treating reward as the return
    ...
```

### Step 4：监控 KL

每次更新都跟踪 mean `KL(π_θ || π_ref)`。如果它爬过 `~5-10`，说明策略已经远离 `π_SFT`：可能需要提高 `β`，或者 reward hacking 正在开始。这是真实 RLHF 中最重要的诊断指标。

### Step 5：使用 TRL 的生产配方

一旦你理解了玩具 pipeline，下面就是真实库用户会写的同一循环。Hugging Face 的 [TRL](https://huggingface.co/docs/trl) 是参考实现：Stage 2 用 `RewardTrainer`，Stage 3 用 `PPOTrainer`（内置到 reference 的 KL）。

```python
# Stage 2: reward model from pairwise preferences
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# dataset rows: {"prompt", "chosen", "rejected"} — Bradley-Terry format
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# Stage 3: PPO against the RM with KL penalty to the SFT reference
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # frozen

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats includes: mean_kl, clip_frac, value_loss — the three PPO diagnostics
```

库会替你做三件事。`adap_kl_ctrl=True` 实现自适应 β schedule：如果观测 KL 超过 `target_kl`，β 翻倍；如果低于一半，β 减半。reference model 按惯例是冻结的：你绝不能意外让它和 `policy` 共享参数。value head 位于和 policy 相同的 backbone 上（`AutoModelForCausalLMWithValueHead` 会附加一个 scalar MLP head），这就是为什么 TRL 会分别报告 `policy/kl` 和 `value/loss`。

## 常见陷阱

- **过度优化 / reward hacking。** RM 不完美；`π_θ` 会找到高分但糟糕的对抗性 completions。症状：reward 无限上升，而 human eval score 持平或下降。修复：early stop、提高 `β`、扩大 RM 训练数据。
- **Length hacking。** 在 helpful responses 上训练的 RM 往往隐式奖励长度。策略学会填充 response。补救：length-normalized reward，或使用 length-aware RM 的 RLAIF。
- **RM 太小。** RM 至少需要和 policy 一样大。很小的 RM 无法忠实评价 policy 的输出。
- **KL tuning。** β 太低 → drift 和 reward hacking。β 太高 → policy 几乎不变。标准技巧是使用目标为固定 KL per step 的*自适应* β。
- **Preference-data 噪声。** 约 30% 的人类标签是有噪声或模糊的。可以用 agreement-filtered data 训练 RM，或在 BT 上使用 temperature 来校准。
- **Off-policy 问题。** 第一个 epoch 之后，PPO 数据会稍微 off-policy。像 Lesson 08 一样监控 clip fraction。

## 实际使用

2026 年的 RLHF 是分层的：

| 层 | 目标 | 方法 |
|----|------|------|
| Instruction following、helpfulness、harmlessness | Alignment | DPO（Phase 10 · 08）优先于 RLHF-PPO。 |
| Reasoning correctness（math、code） | Capability | 带 verifier reward 的 GRPO（Phase 9 · 12）。 |
| Long-horizon multi-step tasks | Agentic | 对步骤使用 process reward models 的 PPO / GRPO。 |
| Safety / refusal behavior | Safety | 带独立 safety RM 的 RLHF-PPO，或 Constitutional AI。 |
| Best-of-N at inference | Fast alignment | 在 decode time 使用 RM；不需要 policy training。 |
| Reward distillation | Inference compute | 在冻结 LM 顶部训练小型 “reward head”。 |

RLHF 是 2022-2024 年的*核心*方法。到 2026 年，生产 alignment pipeline 是 DPO-first，只有 RM-intensive 或 safety-critical 步骤才使用 PPO。

## 交付成果

保存为 `outputs/skill-rlhf-architect.md`：

```markdown
---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

Given a base LM, a target behavior (alignment / reasoning / refusal / agent), and a preference or verifier budget, output:

1. Stage. SFT? RM? DPO? GRPO? With justification.
2. Preference or verifier source. Humans, AI feedback, rule-based, unit-test-pass, or reward distillation.
3. KL strategy. Fixed β, adaptive β, or DPO (implicit KL).
4. Diagnostics. Mean KL, reward stability, over-optimization guard (holdout human eval).
5. Safety gate. Red-team set, refusal rate, safety RM separate from helpfulness RM.

Refuse to ship RLHF-PPO without a KL monitor. Refuse to use an RM smaller than the target policy. Refuse length-only rewards. Flag any pipeline that does not hold back a blind human-eval set as lacking over-optimization protection.
```

## 练习

1. **简单。** 在 `code/main.py` 中用 500 个合成 preference pairs 训练 Bradley-Terry reward model。在保留的 100 个 pairs 上测量 pairwise accuracy。应超过 90%。
2. **中等。** 用 `β ∈ {0.0, 0.1, 1.0}` 运行玩具 PPO-RLHF 循环。对每个设置，绘制更新过程中的 RM score vs KL-to-reference。哪些运行发生了 reward-hack？
3. **困难。** 在同一 preference data 上实现 DPO（闭式 preference-likelihood loss），并和 RLHF-PPO pipeline 比较使用的 compute 与最终达到的 RM score。

## 关键术语

| 术语 | 大家常说 | 实际含义 |
|------|----------|----------|
| RLHF | “Alignment RL” | 三阶段 SFT + RM + PPO pipeline（Christiano 2017，Ouyang 2022）。 |
| Reward Model (RM) | “打分网络” | 通过 Bradley-Terry 拟合 pairwise preferences 的学习型标量函数。 |
| Bradley-Terry | “Pairwise logistic loss” | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`；标准 RM 目标。 |
| KL penalty | “保持在 reference 附近” | reward 中的 `β · KL(π_θ \|\| π_ref)`；抗 reward-hacking regularizer。 |
| Reward hacking | “Goodhart's law” | Policy 利用 RM 缺陷；症状：reward 上升，human eval 持平。 |
| RLAIF | “AI-labeled preferences” | 标签来自另一个 LM 而不是人类的 RLHF。 |
| PRM | “Process Reward Model” | 给 partial reasoning steps 打分；用于 reasoning pipeline。 |
| Constitutional AI | “Anthropic 的方法” | 由显式规则引导、AI 生成 preferences。 |

## 延伸阅读

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741)：开启 RLHF 的论文。
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)：ChatGPT 背后的配方。
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325)：更早的 summarization RLHF。
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290)：DPO；2026 年后 RLHF 的默认替代。
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)：RLAIF 和 self-critique loop。
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862)：HH 论文。
- [Hugging Face TRL library](https://huggingface.co/docs/trl)：生产级 `RewardTrainer` 和 `PPOTrainer`。阅读 trainer 源码了解 adaptive-KL 和 value-head 细节。
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf)，作者 Lambert、Castricato、von Werra、Havrilla：带图的三阶段 pipeline 标准讲解。
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl)：库本身；`examples/` 中有 Llama、Mistral 和 Qwen 的端到端 RLHF 脚本。
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf)：reward-hypothesis 视角；思考 reward hacking 的必要先修。
