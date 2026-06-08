# Flow Matching & Rectified Flows

> Diffusion models 需要 20-50 个 sampling steps，因为它们沿着从 noise 到 data 的弯曲路径行走。Flow matching（Lipman et al., 2023）和 rectified flow（Liu et al., 2022）训练直线路径。路径越直，steps 越少，inference 越快。Stable Diffusion 3、Flux.1 和 AudioCraft 2 都在 2024 年切换到 flow matching。

**类型:** Build
**语言:** Python
**先修:** Phase 8 · 06 (DDPM), Phase 1 · Calculus
**时间:** ~45 minutes

## 要解决的问题

DDPM 的 reverse process 是从 `N(0, I)` 回到 data distribution 的 1000-step stochastic walk。DDIM 把它压缩成 20-50 deterministic steps。你想要更少 steps——理想情况下一个。阻碍在于：求解 reverse process 的 ODE 很 stiff；路径是弯的。

如果你能训练模型，使 noise 到 data 的路径是一条*直线*，那么从 `t=1` 到 `t=0` 的单个 Euler step 就能工作。Flow matching 直接构建这一点：定义从 `x_1 ∼ N(0, I)` 到 `x_0 ∼ data` 的 straight-line interpolation，训练 vector field `v_θ(x, t)` 去匹配它的 time derivative，推理时积分。

Rectified flow（Liu 2022）更进一步：通过 reflow procedure 迭代拉直路径，产生越来越接近线性的 ODE。两次 reflow iterations 后，2-step sampler 就能匹配 50-step DDPM 质量。

## 核心概念

![Flow matching: straight-line interpolation between noise and data](../assets/flow-matching.svg)

### Straight-line flow

定义：

```text
x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]
```

其中 `x_0 ~ data` 且 `x_1 ~ N(0, I)`。沿这条直线的 time derivative 是常数：

```text
dx_t / dt = x_1 - x_0
```

定义 neural vector field `v_θ(x_t, t)`，并训练它匹配这个 derivative：

```text
L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

这就是 **conditional flow matching** loss（Lipman 2023）。训练是 simulation-free：你从不 unroll ODE。只需 sample `(x_0, x_1, t)` 并做 regression。

### Sampling

推理时，沿时间*反向*积分 learned vector field：

```text
x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
```

从 `x_1 ~ N(0, I)` 开始，Euler-step 到 `t=0`。

### Rectified flow (Liu 2022)

Straight-line flow 能工作，但 learned paths *实际上并不直*——它们会弯曲，因为许多 `x_0` 可以映射到同一个 `x_1`。Rectified flow 的 reflow step：

1. 用 random pairings 训练 flow model v_1。
2. 通过从 `x_1` 积分到 landing `x_0`，sample N 对 `(x_1, x_0)`。
3. 在这些 paired examples 上训练 v_2。因为这些 pairs 现在是 “ODE-matched”，它们之间的 straight-line interpolant 会真正更平。
4. 重复。

实践中 2 次 reflow iterations 就能接近线性，支持 2-4 step inference。SDXL-Turbo、SD3-Turbo、LCM 都是 distilled-from-flow-matching models。

### 为什么它在 2024 年赢下图像

三个原因：

1. **Simulation-free training** — 训练时不做 ODE unrolling，实现非常简单。
2. **Better loss geometry** — 直线路径有一致的 signal-to-noise，而 DDPM ε-loss 在 schedule 边缘 SNR 很差。
3. **Faster inference** — 4-8 steps 达到 SDXL-Turbo 质量；配合 consistency distillation 可 1 step。

## Flow matching vs DDPM — the exact connection

带 Gaussian-conditional path 的 flow matching 是带特定 noise schedule 的 diffusion。选择 `x_t = α(t) x_0 + σ(t) x_1` schedule 后，flow matching 会恢复 Stratonovich-reformulated diffusion，其中 `v = α'·x_0 - σ'·x_1`。对于 Gaussian paths，二者代数等价。

Flow matching 增加的是：target 的*清晰性*（普通 velocity）、更干净的 loss，以及实验 non-Gaussian interpolants 的自由。

## 动手实现

`code/main.py` 在 two-mode Gaussian mixture 上实现 1-D flow matching。Vector field `v_θ(x, t)` 是一个 tiny MLP，用 straight-line target 训练。推理时，积分 1、2、4 和 20 个 Euler steps 并比较 sample quality。

### Step 1: training loss

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # backprop + update
```

### Step 2: multi-step inference

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### Step 3: compare step counts

预期 4-step sampler 已经能匹配 20-step quality——这对 latency 很重要。

## Pitfalls

- **Time parameterization.** Flow matching 使用 `t ∈ [0, 1]`，其中 `t=0` 是 data、`t=1` 是 noise。DDPM 使用 `t ∈ [0, T]`，其中 `t=0` 是 data、`t=T` 是 noise。方向相同，尺度不同。论文经常弄错。
- **Schedule choice.** Rectified flow 的 straight line 是 “the” flow-matching schedule，但你可以使用 cosine 或 logit-normal t-sampling（SD3 使用）来获得更好的 scale coverage。
- **Reflow cost.** 为 reflow 生成 paired dataset 等于对每个 sample 完整 inference pass。只有真正需要 1-2 step inference 时才做 reflow。
- **Classifier-free guidance still applies.** 只需把 ε 换成 v 做线性组合：`v_cfg = (1+w) v_cond - w v_uncond`。

## 实际使用

| Use case | 2026 stack |
|----------|-----------|
| Text-to-image, best quality | Flow matching: SD3, Flux.1-dev |
| Text-to-image, 1-4 steps | Distilled flow matching: Flux.1-schnell, SD3-Turbo, SDXL-Turbo |
| Real-time inference | Consistency distillation from a flow-matched base (LCM, PCM) |
| Audio generation | Flow matching: Stable Audio 2.5, AudioCraft 2 |
| Video generation | Flow matching mixed with diffusion (Sora, Veo, Stable Video) |
| Science / physics (particle trajectories, molecules) | Flow matching + equivariant vector field |

当 2025-2026 年论文说 “faster than diffusion” 时，几乎总是 flow matching + distillation。

## 交付成果

保存 `outputs/skill-fm-tuner.md`。Skill 接收 diffusion-style model spec，并将其转换为 flow-matching training config：schedule choice、time sampling distribution（uniform / logit-normal）、optimizer、reflow plan、target step count、eval protocol。

## 练习

1. **Easy.** 运行 `code/main.py`，比较 1-step vs 20-step 相对 true data distribution 的 MSE。
2. **Medium.** 从 uniform `t` sampling 切换到 logit-normal（把 sampling 集中在 mid-t）。模型质量是否提升？
3. **Hard.** 实现一次 reflow iteration：通过积分第一个模型生成 paired (x_0, x_1)，在 pairs 上训练第二个模型，并比较 1-step sample quality。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Flow matching | “Straight-line diffusion” | 训练 `v_θ(x, t)` 沿 interpolant 匹配 `x_1 - x_0`。 |
| Rectified flow | “Reflow” | 迭代拉直 learned flows 的过程。 |
| Velocity field | “v_θ” | 模型输出——`x_t` 应移动的方向。 |
| Straight-line interpolant | “The path” | `x_t = (1-t)·x_0 + t·x_1`；target derivative 很简单。 |
| Euler sampler | “1st order ODE solver” | 最简单 integrator；当路径很直时效果很好。 |
| Logit-normal t | “SD3 sampling” | 把 `t` sampling 集中到 gradients 最强的中间值。 |
| Consistency distillation | “1-step sampler” | 训练 student 把任意 `x_t` 直接映射到 `x_0`。 |
| CFG with velocity | “v-CFG” | `v_cfg = (1+w) v_cond - w v_uncond`；同一个技巧，换了变量。 |

## Production note: Flux.1-schnell is flow matching at its fastest

Flow matching 的 production win 是 Flux.1-schnell——一个 flow-matched DiT，被 distill 到 1-4 inference steps，同时保留 Flux-dev-grade quality。Niels 的 “Run Flux on an 8GB machine” notebook 是 reference deployment recipe：T5 + CLIP encode、quantized MMDiT denoise（schnell 4 steps vs dev 50 steps）、VAE decode。成本核算：

| Variant | Steps | Latency at 1024² on L4 | Total FLOPs (relative) |
|---------|-------|------------------------|------------------------|
| Flux.1-dev (raw) | 50 | ~15 s | 1.0× |
| Flux.1-schnell | 4 | ~1.2 s | 0.08× (12× faster) |
| SDXL-base | 30 | ~4 s | 0.25× |
| SDXL-Lightning 2-step | 2 | ~0.3 s | 0.03× |

Production rule：**flow-matched base + distillation = 2026 年 fast text-to-image 默认方案。** 每个 major vendor 都提供这个组合：SD3-Turbo（SD3 + flow + distillation）、Flux-schnell（Flux-dev + rectified-flow straightening）、CogView-4-Flash。Pure diffusion bases 只存在于 legacy checkpoints。

## 延伸阅读

- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — rectified flow。
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — flow matching。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3，rectified flow at scale。
- [Albergo, Vanden-Eijnden (2023). Stochastic Interpolants](https://arxiv.org/abs/2303.08797) — 覆盖 FM + diffusion 的 general framework。
- [Song et al. (2023). Consistency Models](https://arxiv.org/abs/2303.01469) — diffusion / flow 的 1-step distillation。
- [Sauer et al. (2023). Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042) — turbo variant。
- [Black Forest Labs (2024). Flux.1 models](https://blackforestlabs.ai/announcing-black-forest-labs/) — production 中的 flow matching。
