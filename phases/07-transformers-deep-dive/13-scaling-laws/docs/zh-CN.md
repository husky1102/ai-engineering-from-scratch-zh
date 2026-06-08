# Scaling Laws

> 2020 年 Kaplan paper 说：模型越大，loss 越低。2022 年 Hoffmann paper 说：你训练得不够。Compute 会进入两个桶：parameters 和 tokens，而怎样切分并不显然。

**类型：** Learn
**语言：** Python
**先修：** Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**时间：** ~45 分钟

## 要解决的问题

当你有 C FLOPs 的 training compute，并想得到最好的模型时，你面对两个旋钮：

1. **多少参数（N）？** 更大的模型，更高 capacity。
2. **多少 training tokens（D）？** 更多数据，更好地使用 capacity。

FLOPs 近似按 `6 × N × D` 缩放。你可以推高 N、降低 D，也可以推高 D、降低 N。哪种更好？

2022 年之前，答案是“用力推 N”。GPT-3（2020）是 175B parameters，在约 300B tokens 上训练。比例约为每个参数 1.7 tokens。Kaplan scaling laws 支持这个结论。

Hoffmann et al.（2022）训练了一小组名为 Chinchilla 的模型，发现了不同结果：optimal ratio 更接近 **每个参数 20 tokens**。GPT-3 训练不足 10×。Chinchilla（70B params，1.4T tokens）用低 2.5× 的 inference cost，在所有 benchmark 上击败 GPT-3（175B，300B tokens）。

2026 年是 Chinchilla 的世界，但有一个重要转折。Llama 3 8B 在 15 trillion tokens 上训练，比例是每个参数 1,875 tokens。超过 Chinchilla-optimal 九十四倍。对于会被大规模使用的模型，inference cost 比 training cost 更重要，所以为了更小、可部署的 footprint 而 over-training（超过 Chinchilla）是 2026 年默认做法。

## 核心概念

![Chinchilla curves: loss vs compute at various N/D ratios](../assets/scaling-laws.svg)

### Hoffmann law

根据 Chinchilla paper，loss 遵循：

```text
L(N, D) = A / N^α + B / D^β + E
```

- `N` = parameters（non-embedding）。
- `D` = training tokens。
- `α ≈ 0.34`，`β ≈ 0.28`（大致对称）。
- `E ≈ 1.69`，不可约 loss ceiling。
- `A ≈ 406`，`B ≈ 411`。

随着你 scale，两个项彼此 trade off。在 fixed compute（C = 6ND）下对 `N` 求导并求解：

```text
N_opt ≈ 0.6 × (C/6)^0.5
D_opt ≈ 0.6 × (C/6)^0.5
D_opt / N_opt ≈ 20
```

Compute-optimal：每个参数 20 tokens。

### 为什么仍要 over-training

Chinchilla-optimal 最小化的是每个 training FLOP 带来的 training loss。但 training cost 只付一次；inference cost 会一直付。

对每月服务一万亿 tokens 的 chatbot 来说，inference 主导总成本。Llama 的做法：训练更小、更久。15T tokens 上的 8B 是深度 inference-optimized：

- 能放进 consumer GPUs。
- Latency 只是 70B Chinchilla-optimal 的一小部分。
- 对多数任务来说，质量足够接近。

DeepMind 2024 paper（“Over-training is the new optimal”）形式化了这一点。对 inference-dominated workloads，正确比例更接近每个参数 100-500 tokens，具体取决于 serving volume。

### Emergence vs smoothness

有人声称：某些能力（arithmetic、multi-step reasoning、chain-of-thought following）会在某个 scale 突然“涌现”。

Schaeffer et al.（2023）认为这是 measurement artifact：emergent metrics 使用 discontinuous scoring（exact match、accuracy at threshold），隐藏了 underlying logits 的 smooth improvement。Continuous metrics（cross-entropy）显示的是 smooth curves。

2026 年的共识是：基于 continuous loss 的预测可靠。Benchmark jumps 往往是 scorer artifacts。用 continuous metrics 来规划预算。

### 2026 图景

Scaling laws 仍然有效，但：

| Factor | Changed how |
|--------|-------------|
| Data quality | 筛选 “good” tokens（Phi-style）会把曲线平移，相当于 >2× effective compute |
| MoE | Total params 与 active FLOPs 解耦；scaling laws 要按 per-active-FLOP 看 |
| Post-training | 某些能力（instruction following, code）受 SFT+RLHF 的影响大于 pretraining |
| Multimodality | Image + text tokens 一起缩放；每个 modality 有单独曲线 |
| Synthetic data | Models 生成 training data；effective compute 可以复合增长 |

Muon optimizer（Kimi Moonlight, 2024）在 matched data 下相对 AdamW 展示了约 2× effective-compute gain。一些 2026 training runs 默认使用 Muon。这会改变 scaling law 的绝对常数，而不是形状。

## 动手实现

见 `code/main.py`。我们实现 Chinchilla loss equation，并在若干 compute budgets 下求解 compute-optimal `(N, D)`。

### Step 1: Chinchilla loss

```python
def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    return A / N ** alpha + B / D ** beta + E
```

在 fixed `C = 6ND` 下，把 `L` 画成 `(N, D)` 上的 contour。找到最小值。

### Step 2: compute-optimal frontier

对从 `1e17` 到 `1e25` FLOPs 的 compute budgets，找到在 `6ND = C` 约束下最小化 loss 的 `(N, D)`。验证比例 `D/N ≈ 20`。

### Step 3: over-training cost

计算训练一个小 10× 的模型（optimal N 的 1/10，optimal D 的 10×）要额外支付多少 loss。报告用它换来的 inference FLOP savings（与 N 成比例）。

### Step 4: compare to real models

把 GPT-3、Chinchilla、Llama 3 8B、DeepSeek-V3（active params）的已知 `(N, D)` pairs 放进去，比较 predicted vs reported loss。

## 实际使用

你大概率不会自己训练 frontier model。但 scaling laws 会告诉你：

1. **你的 fine-tune 是否有足够数据。** 如果 task-specific data 低于 base model 每个参数 20 tokens，预期会在某个 loss floor 饱和。
2. **是否该选更大的 base model。** 如果你把所有预算都花在 inference 上，优先选更小、训练更久的模型。
3. **收益在哪里递减。** 超过 1000× Chinchilla-optimal 后，log-loss 的变化会变成噪声。

**2026 年的 research trajectory：**

- **Data-constrained regime.** Web 上高质量 tokens 有限（过滤后约 5-10 trillion English）。Frontier pretraining 正接近这个 ceiling。Synthetic data、multilingual、multimodal 和 RLHF-scaled fine-tuning 是下一组杠杆。
- **Compute-multiplier tricks.** Muon optimizer、MoE、更好的 data curation：每个都会移动绝对常数，而不是渐近线。
- **Scaling laws for RL.** 开放问题。早期证据显示 RL samples 上也有 power-law，但 exponent 与 pretraining 非常不同。

## 交付成果

见 `outputs/skill-training-budget-estimator.md`。这个 skill 会根据 compute budget、deployment constraints 和 target loss，为新的 training run 选择 `(N, D, hours, GPU)`。

## 练习

1. **Easy.** 运行 `code/main.py`。打印 compute budgets `1e20`、`1e22`、`1e24` 下的 Chinchilla-optimal `(N, D)`。与 real model table 比较。
2. **Medium.** 实现 Hoffmann loss-as-function-of-compute curve。绘制 compute-optimal frontier 的 loss vs `log10(C)`。找出 law 预测下一次 cross-entropy 降低 0.1 会需要 `>10^28` FLOPs 的位置。
3. **Hard.** 在同一 dataset 上训练 5 个 tiny models（100K 到 10M params），拟合你自己的 scaling law。估计 `α` 和 `E`。你的 exponents 与论文中的匹配得多好？

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Parameters (N) | “Model size” | Non-embedding weight count；决定 capacity。 |
| Tokens (D) | “Training data” | 见过的 training tokens 数；决定 parameters 被使用得有多充分。 |
| Compute (C) | “FLOPs spent” | 对标准 transformer，约为 `6 × N × D`。 |
| Chinchilla-optimal | “D/N ≈ 20” | 最小化 pretraining 每 FLOP loss 的比例。 |
| Over-training | “Past Chinchilla” | 花更多 training FLOPs 来节省 inference FLOPs；D/N >> 20。 |
| Irreducible loss | “The floor” | Scaling law 中的 `E` 项；数据本身的 entropy。 |
| Emergent capability | “Sudden jumps at scale” | 往往是 scorer artifact；continuous loss 是平滑的。 |
| Effective compute | “Training-efficiency multiplier” | 更好的 data / optimizer / architecture 会放大一个 FLOP 能走多远。 |

## 延伸阅读

- [Kaplan et al. (2020). Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — 第一篇 scaling law paper；训练不足。
- [Hoffmann et al. (2022). Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — Chinchilla。
- [Schaeffer et al. (2023). Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) — 把 emergence 视为 measurement artifact。
- [Sardana, Frankle (2024). Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws](https://arxiv.org/abs/2401.00448) — 为什么 Llama 的 over-training 对它的 workload 是正确的。
- [Jordan et al. (2024). Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) — 2× compute multiplier。
