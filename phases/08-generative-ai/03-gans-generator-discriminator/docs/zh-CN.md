# GAN：生成器与判别器

> Goodfellow 在 2014 年的技巧是完全跳过 density。两个 networks。一个造 fakes。一个抓 fakes。它们互相对抗，直到 fakes 和 real 无法区分。这不该奏效。它也经常不奏效。但当它奏效时，在 narrow domains 里，samples 仍然是 literature 中最 sharp 的。

**类型：** Build
**语言：** Python
**先修：** Phase 3 · 02 (Backprop), Phase 3 · 08 (Optimizers), Phase 8 · 02 (VAE)
**时间：** ~75 分钟

## 要解决的问题

VAE 产生 blurry samples，因为它们的 MSE decoder loss 对 *mean* image 是 Bayes-optimal——而许多 plausible digits 的 mean 就是 fuzzy digit。你想要一种奖励 *plausibility* 的 loss，而不是奖励与某个 target 的 pixel-wise proximity。Plausibility 没有 closed form。你必须学出来。

Goodfellow 的想法：训练一个 classifier `D(x)` 区分 real images 和 fakes。训练一个 generator `G(z)` 去欺骗 `D`。`G` 的 loss signal 就是 `D` 当前认为某样东西看起来真实的依据。随着 `G` 变好，这个 signal 会更新，追逐一个 moving target。如果两个 networks 都收敛，`G` 就学会了 data distribution，而从未写下 `log p(x)`。

这就是 adversarial training。数学上是一个 minimax game：

```text
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

到 2026 年，GAN 不再是 SOTA generator（diffusion 和 flow matching 吃掉了王冠）。但 StyleGAN 2/3 仍是迄今交付过的最 sharp 的 face models，GAN discriminators 被用作 diffusion training 中的 *perceptual losses*，adversarial training 也驱动了快速 1-step distillations（SDXL-Turbo、SD3-Turbo、LCM），让你能交付 real-time diffusion。

## 核心概念

![GAN training: generator and discriminator in minimax](../assets/gan.svg)

**Generator `G(z)`.** 把 noise vector `z ~ N(0, I)` 映射成 sample `x̂`。形状像 decoder 的 network（dense 或 transposed conv）。

**Discriminator `D(x)`.** 把 sample 映射成 scalar probability（或 score）。Real → 1，fake → 0。

**Loss.** 两个 alternating updates：

- **Train `D`:** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`。对 real=1、fake=0 做 binary cross-entropy。
- **Train `G`:** `loss_G = -log D(G(z))`。这是 Goodfellow 使用的 *non-saturating* 形式（原始 `log(1 - D(G(z)))` 在 `D` confident 时会 saturate 并杀死 gradients）。

**Training loop.** 一步 `D`，一步 `G`。重复。

**Why it works.** 如果 `G` 完美匹配 `p_data`，那么 `D` 最多只能达到随机猜测，到处输出 0.5；`G` 不再得到 gradient。Equilibrium。

**Why it breaks.** Mode collapse（`G` 找到一个 `D` 无法 classify 的 mode，然后永远生成它）、vanishing gradient（`D` 学得太快，`log D` saturates）、training instability（learning rates、batch sizes，什么都可能）。

## 让 GAN 真正可用的变体

| Year | Innovation | Fix |
|------|------------|-----|
| 2015 | DCGAN | Conv/deconv、batch norm、LeakyReLU——第一个稳定 architecture。 |
| 2017 | WGAN, WGAN-GP | 用 Wasserstein distance + gradient penalty 替换 BCE。修复 vanishing gradient。 |
| 2017 | Spectral normalization | Lipschitz-bound discriminator。2026 年的 discriminators 仍在使用。 |
| 2018 | Progressive GAN | 先训练 low-res，再加 layers。第一批 megapixel results。 |
| 2019 | StyleGAN / StyleGAN2 | Mapping network + adaptive instance norm。Fixed-domain photorealism 的 state of the art。 |
| 2021 | StyleGAN3 | Alias-free、translation-equivariant——2026 年仍是 face gold standard。 |
| 2022 | StyleGAN-XL | Conditional、class-aware、更大 scale。 |
| 2024 | R3GAN | 用更强 regularization 重新包装；无需 tricks 即可在 1024² 上工作。 |

## 动手实现

`code/main.py` 在 1-D data 上训练 tiny GAN：两个 Gaussians 的 mixture。Generator 和 discriminator 都是 single-hidden-layer MLP。我们手写 forward、backward 和 minimax loop。目标是亲眼看到两个关键 failure modes（mode collapse + vanishing gradient）如何发生。

### Step 1: non-saturating loss

Vanilla Goodfellow loss `log(1 - D(G(z)))` 会在 D 高置信度地把 G 的 fake classify 为 fake 时趋近 0。此时 G 的 gradient 基本为零——G 无法改进。Non-saturating 形式 `-log D(G(z))` 有相反的渐近行为：当 D 很 confident 时它会爆大，给 G 强信号。

```python
def g_loss(d_fake):
    # maximize log D(G(z))  <=>  minimize -log D(G(z))
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### Step 2: one discriminator step per generator step

```python
for step in range(steps):
    # train D
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # train G
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # fresh fakes
    update_G(fake_batch)
```

给 G 用 fresh fakes，否则 gradients 是 stale 的。

### Step 3: watch for mode collapse

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] mode collapse: one mode is starved")
```

Canonical symptom：两个 real modes 中有一个不再被生成。Discriminator 停止纠正它，因为它再也没被当作 fake 见过。

## 常见陷阱

- **Discriminator too strong.** 把 D 的 learning rate 降低 2-5x，或加入 instance/layer noise。如果 D accuracy 达到 >95%，G 就死了。
- **Generator memorizes a mode.** 给 D inputs 加 noise，使用 minibatch-discriminator layer，或切换到 WGAN-GP。
- **Batch norm leaking statistics.** Real batch + fake batch 通过同一个 BN layer 会混合统计量。改用 instance norm 或 spectral norm。
- **Inception-score gaming.** FID 和 IS 在 low sample counts 下噪声很大。Eval 时使用 ≥10k samples。
- **One-shot sampling is a lie for conditional tasks.** 你仍然需要 CFG scales、truncation tricks 和 re-sampling 才能得到 usable outputs。

## 实际使用

2026 年的 GAN stack：

| Situation | Pick |
|-----------|------|
| Photoreal human faces, fixed pose | StyleGAN3 (sharpest, smallest) |
| Anime / stylized faces | StyleGAN-XL or Stable Diffusion LoRA |
| Image-to-image translation | Pix2Pix / CycleGAN (Phase 8 · 04) or ControlNet (Phase 8 · 08) |
| Fast 1-step text-to-image | Adversarial distillation of diffusion (SDXL-Turbo, SD3-Turbo) |
| Perceptual loss inside a diffusion trainer | Small GAN discriminator on image crops |
| Anything multi-modal, open-ended | Don't — use diffusion or flow matching |

GAN 很 sharp，但 narrow。一旦 domain 打开——photos、任意 text prompts、video——就切换到 diffusion。Adversarial trick 会作为 component（perceptual losses、distillation）继续存在，而不是 standalone generator。

## 交付成果

保存 `outputs/skill-gan-debugger.md`。Skill 接收一次失败的 GAN run（loss curves、sample grid、dataset size），输出一个按可能性排序的原因列表、one-line fixes，以及 rerun protocol。

## 练习

1. **Easy.** 用默认 settings 运行 `code/main.py`。然后设置 `D_LR = 5 * G_LR` 并重新运行。G 的 loss 多快 collapse 成常数？
2. **Medium.** 用 WGAN loss 替换 Goodfellow BCE loss：`loss_D = E[D(fake)] - E[D(real)]`，`loss_G = -E[D(fake)]`，并把 D 的 weights clip 到 `[-0.01, 0.01]`。Training 是否更稳定？比较 wall-clock convergence。
3. **Hard.** 把 1-D example 扩展到 2-D data（环上 8 个 Gaussians 的 mixture）。Track generator 在 steps 1k、5k、10k 捕获了 8 个 modes 中的几个。实现 minibatch discrimination 并重新测量。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Generator | "G" | Noise-to-sample network，`G: z → x̂`。 |
| Discriminator | "D" | Classifier `D: x → [0, 1]`，real vs fake。 |
| Minimax | "The game" | Joint objective 上的 `min_G max_D`。 |
| Non-saturating loss | "The fix" | 对 G 使用 `-log D(G(z))`，而不是 `log(1 - D(G(z)))`。 |
| Mode collapse | "G memorized one thing" | 尽管 data 多样，generator 仍只产生少量不同 outputs。 |
| WGAN | "Wasserstein" | 用 Earth-Mover distance + gradient penalty 替换 BCE；gradient 更平滑。 |
| Spectral norm | "Lipschitz trick" | 约束 D 的 weight norms 来 bound slope；稳定 training。 |
| StyleGAN | "The one that works" | Mapping network + AdaIN；faces 上 best-in-class，2026 年仍然如此。 |

## 生产备注：one-shot inference 是 GAN 留下的优势

GAN 在 open-domain generation 的 sample quality 上不再胜出，但在 inference cost 上仍然胜出。用 production-inference literature 的词汇说，GAN 有：

- **No prefill, no decode stages.** 单次 `G(z)` forward pass。TTFT ≈ total latency。
- **No KV-cache pressure.** 唯一 state 是 weights。Batch size 受 activation memory 限制，而不是 cache。
- **Trivial continuous batching.** 因为每个 request 都消耗相同固定 FLOPs，所以服务器目标 occupancy 上的 static batch 通常最优。不需要 in-flight scheduler。

这就是 GAN distillation（SDXL-Turbo、SD3-Turbo、ADD、LCM）成为 2026 年 fast text-to-image 主导技术的原因：它把一个 20-50-step diffusion pipeline collapse 成 1-4 次 GAN-style forward passes，同时保留 diffusion base 的 distribution。Adversarial loss 作为 training-time knob 存活下来，用来把 slow generators 变成 fast generators。

## 延伸阅读

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — original GAN paper。
- [Radford et al. (2015). Unsupervised Representation Learning with DCGAN](https://arxiv.org/abs/1511.06434) — first stable architecture。
- [Arjovsky, Chintala, Bottou (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875) — WGAN。
- [Miyato et al. (2018). Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957) — SN。
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2。
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3。
- [Sauer et al. (2023). Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) — SDXL-Turbo。
