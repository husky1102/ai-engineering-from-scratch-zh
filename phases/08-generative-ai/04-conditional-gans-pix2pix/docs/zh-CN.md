# 条件 GAN 与 Pix2Pix

> 2014-2017 年第一个大突破，是控制 GAN 生成什么。附上 label、image 或 sentence。Pix2Pix 做的是 image 版本，在 narrow image-to-image tasks 上，它到现在仍然胜过每个 generic text-to-image model。

**类型：** Build
**语言：** Python
**先修：** Phase 8 · 03 (GANs), Phase 4 · 06 (U-Net), Phase 3 · 07 (CNNs)
**时间：** ~75 分钟

## 要解决的问题

Unconditional GAN 会 sample 任意 faces。做 demo 有用，production 没用。你想要的是：*map a sketch to a photo*、*map a map to an aerial photo*、*map a daytime scene to nighttime*、*colorize a grayscale image*。在所有这些任务里，你拿到 input image `x`，必须输出与它具有语义对应关系的 `y`。每个 `x` 都有许多 plausible `y`。Mean-squared error 会把它们压扁成糊状。Adversarial loss 不会，因为“looks real”是 sharp 的。

Conditional GAN（Mirza & Osindero, 2014）把 condition `c` 作为 input 加到 `G` 和 `D`。Pix2Pix（Isola et al., 2017）专门化了这个 recipe：condition 是一整张 input image，generator 是 U-Net，discriminator 是 *patch-based* classifier（PatchGAN），loss 是 adversarial + L1。即使到 2026 年，这个 recipe 在 narrow image-to-image domains 上仍胜过 from-scratch text-to-image models，因为它训练在 *paired data* 上——你拥有恰好需要的 signal。

## 核心概念

![Pix2Pix: U-Net generator, PatchGAN discriminator](../assets/pix2pix.svg)

**Conditional G.** `G(x, z) → y`。在 Pix2Pix 中，`z` 是 G 内部的 dropout（没有 input noise——Isola 发现 explicit noise 会被忽略）。

**Conditional D.** `D(x, y) → [0, 1]`。Input 是 *pair*（condition, output）。这是关键区别：D 必须判断 `y` 是否与 `x` 一致，而不只是判断 `y` 是否看起来真实。

**U-Net generator.** 带 bottleneck 两侧 skip connections 的 encoder-decoder。对 input 和 output 共享 low-level structure（edges、silhouette）的任务至关重要。没有 skips，high-frequency detail 会消失。

**PatchGAN discriminator.** D 不输出单一 real/fake score，而是输出一个 `N×N` grid，其中每个 cell 判断约 70×70 pixels 的 receptive field。再求平均。这是 Markov random field assumption：realism 是 local 的。训练快得多，参数更少，output 更 sharp。

**Loss.**

```text
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

L1 term 稳定 training，并把 G 推向已知 target。L1 比 L2 给出更 sharp 的 edges（medians，而不是 means）。`λ = 100` 是 Pix2Pix default。

## CycleGAN — when you don't have pairs

Pix2Pix 需要 paired `(x, y)` data。CycleGAN（Zhu et al., 2017）以一个额外 loss 为代价去掉这个要求：*cycle consistency* loss。两个 generators：`G: X → Y` 和 `F: Y → X`。训练它们使 `F(G(x)) ≈ x` 且 `G(F(y)) ≈ y`。这让你无需 paired examples，也能把 horses 翻译成 zebras、summer 翻译成 winter。

到 2026 年，unpaired image-to-image 多数通过 diffusion（ControlNet、IP-Adapter）完成，而不是 CycleGAN，但 cycle-consistency idea 仍存在于几乎每篇 unpaired domain adaptation paper 中。

## 动手实现

`code/main.py` 在 1-D data 上实现了 tiny conditional GAN。Condition `c` 是 class label（0 或 1）。任务：为给定 class 生成来自 conditional distribution 的 sample。

### Step 1: append condition to both G and D inputs

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

One-hot encoding 是最简单的方式。更大的 models 使用 learned embeddings、FiLM modulation 或 cross-attention。

### Step 2: train conditional

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

Generator 必须匹配 *给定 condition* 下的 real distribution，而不是 marginal。

### Step 3: verify per-class output

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## 常见陷阱

- **Condition ignored.** G 学会 marginalize，D 不惩罚，因为 condition signal 太弱。修复：更强地 condition D（early layer，而不是只在 late layer），使用 projection discriminator（Miyato & Koyama 2018）。
- **L1 weight too low.** G 漂移到任意 real-looking outputs，而不是 faithful outputs。对 Pix2Pix-style tasks 从 λ≈100 开始。
- **L1 weight too high.** G 产生 blurry outputs，因为 L1 仍然是 L_p norm。Training 稳定后 anneal down。
- **Ground-truth leakage in D.** 把 `(x, y)` concat 成 D input，而不只是 `y`。没有这个，D 无法检查 consistency。
- **Mode collapse per class.** 每个 class 都可能独立 collapse。运行 class-conditional diversity checks。

## 实际使用

2026 年 image-to-image tasks 状态：

| Task | Best approach |
|------|---------------|
| Sketch → photo, same domain, paired data | Pix2Pix / Pix2PixHD (still fast, still sharp) |
| Sketch → photo, unpaired | ControlNet with a Scribble conditioning model |
| Semantic seg → photo | SPADE / GauGAN2 or SD + ControlNet-Seg |
| Style transfer | Diffusion with IP-Adapter or LoRA; GAN methods are legacy |
| Depth → photo | ControlNet-Depth over Stable Diffusion |
| Super-resolution | Real-ESRGAN (GAN), ESRGAN-Plus, or SD-Upscale (diffusion) |
| Colorization | ColTran, diffusion-based colorizers, or Pix2Pix-color |
| Daytime → nighttime, seasons, weather | CycleGAN or ControlNet-based |

当（a）你有成千上万 paired examples，（b）任务 narrow 且 repeatable，（c）你需要 fast inference 时，Pix2Pix 仍是正确工具。在 generic open-domain tasks 上，diffusion 胜出。

## 交付成果

保存 `outputs/skill-img2img-chooser.md`。Skill 接收 task description、data availability（paired vs unpaired、N samples）以及 latency/quality budget，然后输出：approach（Pix2Pix、CycleGAN、ControlNet variant、SDXL + IP-Adapter）、training data requirements、inference cost，以及 eval protocol（LPIPS、FID、task-specific）。

## 练习

1. **Easy.** 修改 `code/main.py`，加入第三个 class。确认 G 仍把每个 class 的 noise 映射到正确 mode。
2. **Medium.** 在 1-D setting 中用 perceptual-style loss 替换 L1（例如使用一个 small frozen D 作为 feature extractor）。它是否改变 conditional distribution 的 sharpness？
3. **Hard.** 在 1-D setting 中 sketch 一个 CycleGAN：两个 distributions、两个 generators、cycle loss。展示它能在没有 paired data 的情况下学会两者之间的映射。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Conditional GAN | "GAN with labels" | G(z, c)，D(x, c)。两个 networks 都看到 condition。 |
| Pix2Pix | "Image-to-image GAN" | 带 U-Net G 和 PatchGAN D + L1 loss 的 paired cGAN。 |
| U-Net | "Encoder-decoder with skips" | 对称 conv network；skips 保留 high-freq。 |
| PatchGAN | "Local-realism classifier" | D 输出 per-patch score，而不是 global score。 |
| CycleGAN | "Unpaired image translation" | 两个 G + cycle-consistency loss；无需 paired data。 |
| SPADE | "GauGAN" | 用 semantic map normalize intermediate activations；segmentation-to-image。 |
| FiLM | "Feature-wise linear modulation" | 来自 condition 的 per-feature affine transform；cheap conditioning。 |

## 生产备注：Pix2Pix as a latency-bound baseline

当你有 paired data 和 narrow task（sketch → render、semantic map → photo、day → night），Pix2Pix 的 one-shot inference 在 latency 上比 diffusion 快一个数量级。Production comparison 通常是：

| Path | Steps | Typical latency at 512² on a single L4 |
|------|-------|----------------------------------------|
| Pix2Pix (U-Net forward) | 1 | ~30 ms |
| SD-Inpaint or SD-Img2Img | 20 | ~1.2 s |
| SDXL-Turbo Img2Img | 1-4 | ~0.15-0.35 s |
| ControlNet + SDXL base | 20-30 | ~3-5 s |

Pix2Pix 在 static batches 里的 throughput 上胜出（每个 request 都是相同 FLOPs）。Diffusion 在 quality 和 generalization 上胜出。现代做法通常是为 narrow task 交付一个 Pix2Pix-style distilled model，再为 tail inputs 准备 diffusion fallback。

## 延伸阅读

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — cGAN paper。
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — Pix2Pix。
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — CycleGAN。
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) — Pix2PixHD。
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) — SPADE / GauGAN。
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) — projection D。
