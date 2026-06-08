# 图像生成：Diffusion Models

> Diffusion model 学的是去噪。训练它从带噪图像中去掉一小点噪声，再把这个过程反向重复一千次，就得到了图像生成器。

**类型：** Build
**语言：** Python
**先修：** Phase 4 Lesson 07 (U-Net), Phase 1 Lesson 06 (Probability), Phase 3 Lesson 06 (Optimizers)
**时间：** ~75 分钟

## 学习目标

- 推导 forward noising process `x_0 -> x_1 -> ... -> x_T`，并解释为什么闭式 `q(x_t | x_0)` 对任意 t 都成立
- 实现 DDPM 风格训练目标：回归每一步加入的噪声；并实现一个从纯噪声走回图像的 sampler
- 构建一个 time-conditioned U-Net（小到可以在 CPU 上训练），能为任意 timestep 预测噪声
- 解释 DDPM 和 DDIM sampling 的区别，以及各自适用场景（Lesson 23 会深入讲 flow matching 和 rectified flow）

## 要解决的问题

GAN 是一次性生成：噪声进，图像出，一次 forward pass。它们很快，但很难训练。Diffusion models 是迭代生成：从纯噪声开始，以小步去噪，图像逐渐浮现。它们很慢，但容易训练。过去五年里，后一个性质占了上风：任何小团队都能训练 diffusion model 并得到合理样本；GAN 训练则是一门要靠多年失败运行才能学会的手艺。

除了训练稳定性，diffusion 的迭代结构也解锁了现代图像生成所做的一切：text conditioning、inpainting、image editing、super-resolution、controllable style。采样循环中的每一步都是注入新约束的位置。正是这个 hook，让 Stable Diffusion、Imagen、DALL-E 3、Midjourney 以及你会使用的每个可控图像模型都基于 diffusion。

本课构建最小 DDPM：forward noising、backward denoising、training loop。下一课（Stable Diffusion）会把它接入生产系统，包括 VAE、text encoder 和 classifier-free guidance。

## 核心概念

### Forward process

取一张图像 `x_0`。加入一小点 Gaussian noise 得到 `x_1`。再多加入一点得到 `x_2`。持续 T 步，直到 `x_T` 几乎与纯 Gaussian noise 无法区分。

```text
q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) * x_{t-1},  beta_t * I)
```

`beta_t` 是一个小方差 schedule，通常在 T=1000 步内从 0.0001 线性增加到 0.02。每一步都会稍微缩小信号并注入新的噪声。

### 闭式跳转

逐步加噪是一个 Markov chain，但数学会折叠起来：你可以一步直接从 `x_0` 采样 `x_t`。

```text
Define alpha_t = 1 - beta_t
Define alpha_bar_t = prod_{s=1..t} alpha_s

Then:
  q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0,  (1 - alpha_bar_t) * I)

Equivalently:
  x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
  where epsilon ~ N(0, I)
```

这个单一方程就是 diffusion 可行的全部原因。训练时你随机选择一个 `t`，直接从 `x_0` 采样 `x_t`，然后一步训练；不需要模拟完整 Markov chain。

### Reverse process

Forward process 是固定的。Reverse process `p(x_{t-1} | x_t)` 是神经网络要学习的东西。Diffusion models 不直接预测 `x_{t-1}`；它们预测第 t 步加入的噪声 `epsilon`，再由数学公式推导出 `x_{t-1}`。

```mermaid
flowchart LR
    X0["x_0<br/>(clean image)"] --> Q1["q(x_t|x_0)<br/>add noise"]
    Q1 --> XT["x_t<br/>(noisy)"]
    XT --> MODEL["model(x_t, t)"]
    MODEL --> EPS["predicted epsilon"]
    EPS --> LOSS["MSE against<br/>true epsilon"]

    XT -.->|sampling| STEP["p(x_{t-1}|x_t)"]
    STEP -.-> XT1["x_{t-1}"]
    XT1 -.->|repeat 1000x| X0S["x_0 (sampled)"]

    style X0 fill:#dcfce7,stroke:#16a34a
    style MODEL fill:#fef3c7,stroke:#d97706
    style LOSS fill:#fecaca,stroke:#dc2626
    style X0S fill:#dbeafe,stroke:#2563eb
```

### 训练损失

每个训练步骤：

1. 采样一张真实图像 `x_0`。
2. 从 [1, T] 均匀采样一个 timestep `t`。
3. 采样噪声 `epsilon ~ N(0, I)`。
4. 计算 `x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon`。
5. 用网络预测 `epsilon_theta(x_t, t)`。
6. 最小化 `|| epsilon - epsilon_theta(x_t, t) ||^2`。

就是这样。神经网络学习在任意 timestep 预测噪声。Loss 是 MSE。没有对抗博弈，没有 collapse，没有 oscillation。

### Sampler（DDPM）

生成时：从 `x_T ~ N(0, I)` 开始，一步一步向后走。

```text
for t = T, T-1, ..., 1:
    eps = model(x_t, t)
    x_{t-1} = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps) + sqrt(beta_t) * z
    where z ~ N(0, I) if t > 1, else 0
return x_0
```

关键在于，虽然一般情况下 reverse conditional 没有已知闭式形式，但对这个特定 Gaussian forward process 来说它有。那些看起来丑的系数正是 Bayes' rule 给出的。

### 为什么是 1000 步

Forward noise schedule 的选择目标是让每一步加入恰好足够的噪声，使 reverse step 近似 Gaussian。步数太少，reverse step 就会远离 Gaussian，网络难以建模。步数太多，采样成本上升且收益递减。T=1000 加线性 schedule 是 DDPM 默认设置。

### DDIM：采样快 20 倍

训练相同，采样改变。DDIM（Song et al., 2020）定义了一个 deterministic reverse process，可以在不重新训练的情况下跳过 timesteps。用 DDIM 采样 50 步能得到接近 1000 步 DDPM 的质量。每个生产系统都会使用 DDIM 或更快的变体（DPM-Solver、Euler ancestral）。

### Time conditioning

网络 `epsilon_theta(x_t, t)` 需要知道它正在去噪哪个 timestep。现代 diffusion models 通过 sinusoidal time embeddings 注入 `t`（与 transformers 中 positional encoding 的想法相同），并在每个 U-Net 层级把它加到 feature maps 上。

```text
t_embedding = sinusoidal(t)
feature_map += MLP(t_embedding)
```

没有 time conditioning，网络就必须从图像本身猜测噪声水平；这虽然能工作，但样本效率低得多。

## 动手实现

### Step 1：Noise schedule

```python
import torch

def linear_beta_schedule(T=1000, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, T)


def precompute_schedule(betas):
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        "sqrt_recip_alphas": torch.sqrt(1.0 / alphas),
    }

schedule = precompute_schedule(linear_beta_schedule(T=1000))
```

预计算一次，在训练和采样时按索引 gather。

### Step 2：Forward diffusion（q_sample）

```python
def q_sample(x0, t, noise, schedule):
    sqrt_a = schedule["sqrt_alphas_cumprod"][t].view(-1, 1, 1, 1)
    sqrt_one_minus_a = schedule["sqrt_one_minus_alphas_cumprod"][t].view(-1, 1, 1, 1)
    return sqrt_a * x0 + sqrt_one_minus_a * noise
```

一行闭式公式。`t` 是一批 timesteps，每张 batch 图像一个。

### Step 3：一个很小的 time-conditioned U-Net

```python
import torch.nn as nn
import torch.nn.functional as F
import math

def timestep_embedding(t, dim=64):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    return emb


class TinyUNet(nn.Module):
    def __init__(self, img_channels=3, base=32, t_dim=64):
        super().__init__()
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, base * 4),
            nn.SiLU(),
            nn.Linear(base * 4, base * 4),
        )
        self.t_dim = t_dim
        self.enc1 = nn.Conv2d(img_channels, base, 3, padding=1)
        self.enc2 = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.mid = nn.Conv2d(base * 2, base * 2, 3, padding=1)
        self.dec1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.dec2 = nn.Conv2d(base * 2, img_channels, 3, padding=1)
        self.time_proj = nn.Linear(base * 4, base * 2)

    def forward(self, x, t):
        t_emb = timestep_embedding(t, self.t_dim)
        t_emb = self.t_mlp(t_emb)
        t_proj = self.time_proj(t_emb)[:, :, None, None]

        h1 = F.silu(self.enc1(x))
        h2 = F.silu(self.enc2(h1)) + t_proj
        h3 = F.silu(self.mid(h2))
        d1 = F.silu(self.dec1(h3))
        d2 = torch.cat([d1, h1], dim=1)
        return self.dec2(d2)
```

两层 U-Net，在 bottleneck 注入 time conditioning。处理真实图像时扩展深度和宽度。

### Step 4：Training loop

```python
def train_step(model, x0, schedule, optimizer, device, T=1000):
    model.train()
    x0 = x0.to(device)
    bs = x0.size(0)
    t = torch.randint(0, T, (bs,), device=device)
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise, schedule)
    pred = model(x_t, t)
    loss = F.mse_loss(pred, noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

这就是完整训练循环。没有 GAN game，没有 specialized loss，只有一次 MSE 调用。

### Step 5：Sampler（DDPM）

```python
@torch.no_grad()
def sample(model, schedule, shape, T=1000, device="cpu"):
    model.eval()
    x = torch.randn(shape, device=device)
    betas = schedule["betas"].to(device)
    sqrt_one_minus_a = schedule["sqrt_one_minus_alphas_cumprod"].to(device)
    sqrt_recip_alphas = schedule["sqrt_recip_alphas"].to(device)

    for t in reversed(range(T)):
        t_batch = torch.full((shape[0],), t, dtype=torch.long, device=device)
        eps = model(x, t_batch)
        coef = betas[t] / sqrt_one_minus_a[t]
        mean = sqrt_recip_alphas[t] * (x - coef * eps)
        if t > 0:
            x = mean + torch.sqrt(betas[t]) * torch.randn_like(x)
        else:
            x = mean
    return x
```

需要 1000 次 forward pass 才能产生一批样本。在真实代码里，你会把它替换成 DDIM 50-step sampler。

### Step 6：DDIM sampler（deterministic，约快 20 倍）

```python
@torch.no_grad()
def sample_ddim(model, schedule, shape, steps=50, T=1000, device="cpu", eta=0.0):
    model.eval()
    x = torch.randn(shape, device=device)
    alphas_cumprod = schedule["alphas_cumprod"].to(device)

    ts = torch.linspace(T - 1, 0, steps + 1).long()
    for i in range(steps):
        t = ts[i]
        t_prev = ts[i + 1]
        t_batch = torch.full((shape[0],), t, dtype=torch.long, device=device)
        eps = model(x, t_batch)
        a_t = alphas_cumprod[t]
        a_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
        x0_pred = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
        sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
        dir_xt = torch.sqrt(1 - a_prev - sigma ** 2) * eps
        noise = sigma * torch.randn_like(x) if eta > 0 else 0
        x = torch.sqrt(a_prev) * x0_pred + dir_xt + noise
    return x
```

`eta=0` 是完全 deterministic（相同噪声输入总会产生相同输出）。`eta=1` 恢复 DDPM。

## 实际使用

生产工作使用 `diffusers`：

```python
from diffusers import DDPMScheduler, UNet2DModel

unet = UNet2DModel(sample_size=32, in_channels=3, out_channels=3, layers_per_block=2)
scheduler = DDPMScheduler(num_train_timesteps=1000)
```

这个库提供现成 schedulers（DDPM、DDIM、DPM-Solver、Euler、Heun）、可配置 U-Nets、text-to-image 和 image-to-image pipelines，以及 LoRA fine-tuning helpers。

研究工作中，`k-diffusion`（Katherine Crowson）拥有最忠实的参考实现和最好的采样变体。

## 交付成果

本课产出：

- `outputs/prompt-diffusion-sampler-picker.md` — 一个 prompt，会根据质量目标、延迟预算和 conditioning 类型选择 DDPM / DDIM / DPM-Solver / Euler。
- `outputs/skill-noise-schedule-designer.md` — 一个 skill：给定 T 和目标 corruption level，生成 linear、cosine 或 sigmoid beta schedule，并附上 signal-to-noise ratio 随时间变化的诊断图。

## 练习

1. **（简单）** 可视化 forward process：取一张图像，绘制 `t in [0, 100, 250, 500, 750, 1000]` 时的 `x_t`。验证 `x_1000` 看起来像纯 Gaussian noise。
2. **（中等）** 在 synthetic-circles 数据集上训练 TinyUNet 20 个 epoch，并采样 16 个圆。比较 DDPM（1000 步）和 DDIM（50 步）采样：它们是否能从同一个 noise seed 产生相似图像？
3. **（困难）** 实现 cosine noise schedule（Nichol & Dhariwal, 2021）：`alpha_bar_t = cos^2((t/T + s) / (1 + s) * pi / 2)`。用 linear 和 cosine schedules 训练同一个模型，并展示 cosine 在低 step count 下给出更好的样本。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| Forward process | “随时间加噪” | 固定 Markov chain，在 T 步内把图像腐蚀成 Gaussian noise |
| Reverse process | “一步步去噪” | 学到的分布，从噪声走回图像 |
| Epsilon prediction | “预测噪声” | 训练目标：`epsilon_theta(x_t, t)` 预测第 t 步加入的噪声 |
| Beta schedule | “噪声量” | T 个小方差组成的序列，定义每一步注入多少噪声 |
| alpha_bar_t | “累计保留因子” | 到时间 t 为止的 (1 - beta_s) 乘积；t 越大，剩余信号越少 |
| DDPM sampler | “Ancestral、stochastic” | 从条件 Gaussian 中采样每个 x_{t-1}；1000 步 |
| DDIM sampler | “Deterministic、fast” | 把采样重写为 deterministic ODE；20-100 步即可达到相似质量 |
| Time conditioning | “告诉模型是哪一个 t” | 把 t 的 sinusoidal embedding 注入 U-Net，让它知道噪声水平 |

## 延伸阅读

- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239) — 让 diffusion 变得实用，并在 FID 上击败 GAN 的论文
- [Improved DDPM (Nichol & Dhariwal, 2021)](https://arxiv.org/abs/2102.09672) — cosine schedule 和 v-parameterisation
- [DDIM (Song, Meng, Ermon, 2020)](https://arxiv.org/abs/2010.02502) — 让实时推理成为可能的 deterministic sampler
- [Elucidating the Design Space of Diffusion (Karras et al., 2022)](https://arxiv.org/abs/2206.00364) — 对每个 diffusion 设计选择的统一视角；当前最佳参考
