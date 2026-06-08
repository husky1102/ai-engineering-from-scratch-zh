# STaR、V-STaR、Quiet-STaR：Self-Taught Reasoning

> 最小可行的 self-improvement loop 位于 rationale 内部。模型生成一条 chain of thought，保留那些落到正确答案上的，再用它们 fine-tune。这就是 STaR。V-STaR 增加 verifier，让 inference-time selection 更好。Quiet-STaR 把 rationale 下推到每个 token。三者都有效。三者都不是魔法：loop 会保留任何碰巧抵达正确答案的捷径。

**类型：** 学习
**语言：** Python (stdlib, bootstrap-loop simulator)
**先修：** Phase 13 · 01-03 (Reasoning and CoT), Phase 15 · 01 (long-horizon framing)
**时间：** ~60 分钟

## 要解决的问题

教模型推理的直接方法，是收集人类写下的 reasoning traces。这很昂贵、缓慢，而且受限于人类愿意写多少高质量 chain-of-thought。

STaR（Self-Taught Reasoner，Zelikman et al., 2022）问：如果模型自己写 rationales，并用已知答案给它们打分呢？loop 是：

1. 采样一条 reasoning trace 加 answer。
2. 如果 final answer 正确，保留 trace。
3. 在保留下来的 traces 上 fine-tune。
4. 重复。

它有效。GSM8K 和 CommonsenseQA 都在没有新人工标注的情况下得到提升。但这个 loop 内置一个 bias：任何产生正确答案的 rationale 都会被保留，无论推理本身是否 sound。V-STaR（Hosseini et al., 2024）用 learned verifier 修补这一点；Quiet-STaR（Zelikman et al., 2024）把这个想法推广到 per-token internal rationales。

## 核心概念

### STaR：在有效样本上 bootstrap

从一个具备一点弱推理能力的 base model 开始。对每个 training problem，采样 rationale plus answer。如果 answer 匹配 label，就保留这个 (problem, rationale, answer) triple。用保留集合 fine-tune model。重复。

有一个 twist 很关键。如果模型永远无法把某题做对，loop 就无法从它上面学习。STaR 添加了 **rationalization**：对 base model 失败的问题，把 correct answer 作为 hint 注入，并重新 prompt 模型生成一条通向它的 rationale。Rationalized rationales 会被加入 training set。

原始论文结果（Zelikman et al., 2022）：一个 GPT-J base model 通过带 rationalization 的重复 STaR rounds，在 GSM8K 上从 5.8% 提升到 10.7%，绝对提升约 5 个百分点。在 CommonsenseQA 上，STaR-trained GPT-J 6B 达到 72.5%，可比 fine-tuned GPT-3 175B（~73%），而后者是一个大约 30x 的模型，训练在 hand-annotated rationales 上。

### V-STaR：用 DPO 训练 verifier

STaR 丢弃 incorrect rationales。Hosseini et al. (2024) 观察到这些也是数据：每一对 (rationale, “is this correct”) 都能训练 verifier。他们在 correct 和 incorrect solutions 上使用 Direct Preference Optimization 构建 ranker。inference time 时，采样 N 条 rationales，并选择 verifier 的 top choice。

报告的 delta：在 GSM8K 和 MATH 上，相比先前 self-improvement baselines 提升 +4 到 +17 个百分点，其中多数收益来自使用 verifier 做 inference-time selection，而不是进一步 fine-tune generator。

### Quiet-STaR：per-token internal rationales

Zelikman et al. (2024) 问：如果模型学会在每个 token position 生成短 internal rationale，而不只是 problem 和 answer 之间生成，会怎样？Quiet-STaR 训练模型在每个 predicted token 前发出隐藏 “thought”，然后通过 learned weight 把 thought-aware prediction 与 baseline prediction 混合。

结果：Mistral 7B 在没有 task-specific fine-tuning 的情况下，GSM8K absolute zero-shot 从 5.9% 提升到 10.9%，CommonsenseQA 从 36.3% 提升到 47.2%。模型学会了 “when to think”：难 token 获得更长 internal rationales，简单 token 几乎没有。

### 为什么三者共享同一个安全担忧

三种方法都使用 final answer 作为 gradient signal。一条通过 flawed reasoning 抵达正确答案的 rationale：利用 shortcut、猜测，或使用不泛化的 pattern，都会被正向强化。在 in-distribution problems 上，shortcut 有效。在 out-of-distribution problems 上，它会静默破裂。

V-STaR 的 verifier 通过学习 rank rationales 来缓解，但 verifier 训练在同一组 labels 上。它可能学会偏好格式良好的错误推理，而不是诚实的不确定。更安全的设计，是把 STaR-style data 与 (a) process-supervised reward models（奖励 intermediate steps，而不只是 answers）以及 (b) 能打破简单 shortcuts 的 held-out OOD evaluation 结合起来。

### 对比

| Method | Training signal | Inference cost | Data waste | Known failure mode |
|---|---|---|---|---|
| STaR | keep (rationale, answer) if correct | 1x | discards all incorrect rationales | shortcut rationales |
| STaR + rationalization | above + correct-answer hinted retries | 1x | less | rationalized rationales may be implausible |
| V-STaR | STaR + DPO verifier from both classes | Nx (best-of-N) | minimal | verifier can reinforce confident wrongness |
| Quiet-STaR | per-token rationale + mixing weight | 1.5-3x | minimal | still answer-conditioned gradient |

### 它位于 2026 stack 的哪里

STaR 已经不新了。但这个 pattern 在 2025-2026 到处重现。verifiable math problems 上的 RL（DeepSeek-R1、Kimi-k1.5、o1）就是 STaR 的 answer-conditioned gradient signal，被放大了。Process reward models（Lightman et al., 2023；OpenAI 的 “Let's verify step by step”）是 process-supervised alternative。AlphaEvolve（Lesson 3）是用于代码的 STaR，只是用 program evaluator 替代 label。Darwin Godel Machine（Lesson 4）是用于 agent scaffolding 自身的 STaR。

理解 STaR，会让这些系统全部串起来。它是 minimum-viable self-improvement loop。

## 实际使用

`code/main.py` 会在 toy arithmetic task 上运行一个模拟 STaR loop。你可以观察：

- accuracy 如何随 bootstrap rounds 上升。
- shortcuts 如何混入：模拟器包含一个 “lazy” rationale class，它有 40% 概率得到正确答案，但泛化很差。观察 STaR 是否会保留它们。
- verifier（V-STaR 风格）如何在 inference 上提供帮助，但无法完全剪掉 training 中引入的 shortcuts。

## 交付成果

`outputs/skill-star-loop-reviewer.md` 帮你在训练前审计一个 proposed self-taught-reasoning pipeline。

## 练习

1. 运行模拟器。将 shortcut frequency 设为 zero，再设为 0.4。尽管两次都在 training distribution 上达到 >90%，final accuracy 之间会差多少？

2. 给模拟器添加 held-out OOD test。从不同 distribution 采样 problems，并在 in-distribution 和 OOD sets 上评估 bootstrapped model。量化 gap。

3. 阅读 Quiet-STaR paper（arXiv:2403.09629）Section 3。分别用三句话解释 “end-of-thought” token 和 mixing-weight head。

4. 将 STaR 的 keep-if-correct filter 与一个 process-supervised alternative 对比，后者会独立奖励每个 rationale step。识别 labelling cost difference 和 plausible quality difference。

5. 设计一个 evaluation，用来捕获 deployed model 中的 shortcut rationales。它不必完美，但必须打破 STaR loop 会强化的最简单 shortcuts。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|---|---|---|
| STaR | “Self-Taught Reasoner” | 在落到正确答案的 model-generated rationales 上 fine-tune；重复 |
| Rationalization | “Hinted retry” | 注入 correct answer，并对 base model 失败的问题重新 prompt 生成 rationale |
| V-STaR | “Verifier STaR” | 在 correct 和 incorrect rationales 上 DPO-train verifier，并用于 inference-time selection |
| Quiet-STaR | “Per-token rationales” | 在每个 token position 生成 hidden thoughts；与 baseline prediction 混合 |
| Answer-conditioned gradient | “Outcome-based signal” | training loop 奖励 final answers，而不是 reasoning steps |
| Process reward model | “Step-level verifier” | 训练在 per-step correctness 上的 reward model，而不是 outcome；与 STaR 对比 |
| Shortcut rationale | “Right answer, wrong reasoning” | 通过不泛化 pattern 抵达 label 的 rationale；STaR 会保留这些 |

## 延伸阅读

- [Zelikman et al. (2022). STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465) — 原始论文。
- [Hosseini et al. (2024). V-STaR: Training Verifiers for Self-Taught Reasoners](https://arxiv.org/abs/2402.06457) — 添加用于 inference-time selection 的 DPO verifier。
- [Zelikman et al. (2024). Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629) — per-token internal rationales。
- [Lightman et al. (2023). Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — process reward models，另一种 gradient signal。
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — verifiable tasks 上的 RL，把 STaR 扩展到 frontier training。
