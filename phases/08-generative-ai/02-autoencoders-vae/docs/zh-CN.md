# 自编码器与变分自编码器（VAE）

> 普通 autoencoder 先压缩再重建。它会记忆。它不会生成。加一个技巧——强制 code 看起来像 Gaussian——你就得到了 sampler。正是 `z = μ + σ·ε` 这个 reparameterization 的单一技巧，让你在 2026 年使用的每个 latent-diffusion 和 flow-matching 图像模型都在输入端带着一个 VAE。

**类型：** Build
**语言：** Python
**先修：** Phase 3 · 02 (Backprop), Phase 3 · 07 (CNNs), Phase 8 · 01 (Taxonomy)
**时间：** ~75 分钟

## 要解决的问题

把一个 784 像素的 MNIST digit 压缩成 16 个数字的 code，然后重建。普通 autoencoder 会在 reconstruction MSE 上表现很好，但 code space 是一团凹凸不平的混乱。随便从 code space 里取一个点，decode 它，得到的是 noise。它没有 sampler。它只是披着生成模型外衣的 compression model。

你真正想要的是：（a）code space 是一个干净、平滑、可采样的分布——比如 isotropic Gaussian `N(0, I)`，（b）decode 任意 sample 都能生成 plausible digit，（c）encoder 和 decoder 仍然压缩得好。三个目标，一个 architecture，一个 loss。

Kingma 的 2013 VAE 通过让 encoder 输出一个 *distribution* `q(z|x) = N(μ(x), σ(x)²)` 来解决这个问题，用 KL penalty 把这个 distribution 拉向 prior `N(0, I)`，然后先从 `q(z|x)` 里 sample `z` 再 decode。Inference 时，丢掉 encoder，sample `z ~ N(0, I)`，decode。KL penalty 迫使 code space 形成结构。

到 2026 年，VAE 很少再作为 standalone 模型交付——原始图像质量已经被 diffusion 超越——但它们是每个 latent-diffusion model（SD 1/2/XL/3、Flux、AudioCraft）的首选 encoder。学会 VAE，你就学会了你使用的每条图像 pipeline 里那层看不见的第一层。

## 核心概念

![Autoencoder vs VAE: the reparameterization trick](../assets/vae.svg)

**Autoencoder.** `z = encoder(x)`，`x̂ = decoder(z)`，loss = `||x - x̂||²`。Code space 没有结构。

**VAE encoder.** 输出两个向量：`μ(x)` 和 `log σ²(x)`。它们定义 `q(z|x) = N(μ, diag(σ²))`。

**Reparameterization trick.** 从 `q(z|x)` sampling 不可微。把 sample 重写为 `z = μ + σ·ε`，其中 `ε ~ N(0, I)`。现在 `z` 是 `(μ, σ)` 加上一个非参数 noise 的确定性函数——gradients 能流过 `μ` 和 `σ`。

**Loss.** Evidence Lower BOund (ELBO)，两项：

```text
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

Reconstruction 把 `x̂` 推向 `x`。KL 把 `q(z|x)` 推向 prior。两者互相 trade off。小 β（<1）= samples 更 sharp，code space 没那么 Gaussian。大 β（>1）= code space 更干净，samples 更 blurry。β-VAE（Higgins 2017）让这个旋钮出名，并开启了 disentanglement research。

**Sampling.** Inference 时：抽取 `z ~ N(0, I)`，forward 通过 decoder。一次 forward pass——不像 diffusion 那样 iterative sampling。

## 动手实现

`code/main.py` 实现了一个不使用 numpy 或 torch 的 tiny VAE。输入是从 8-D 中一个 2-component Gaussian mixture 抽出的 8-dimensional synthetic data。Encoder 和 decoder 是 single hidden-layer MLP。我们实现 tanh activation、forward pass、loss，以及手写 backward pass。不是 production——是 pedagogy。

### Step 1: encoder forward

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

用 `log σ²` 而不是 `σ`，这样 network output 不受约束（对 σ 做 softplus 是陷阱——σ ≈ 0 时 gradients 会死）。

### Step 2: reparameterize and decode

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### Step 3: the ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

因为两个 distributions 都是 Gaussian，所以 KL 有 exact closed form。不要数值积分。2026 年仍有人交付带 monte-carlo KL estimates 的代码——无意义地慢 3x。

### Step 4: generate

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

这就是 generative model。五行。

## 常见陷阱

- **Posterior collapse.** KL term 过于强硬地把 `q(z|x) → N(0, I)`，导致 `z` 不携带关于 `x` 的信息。修复：β-annealing（从 β=0 开始，ramp 到 1）、free bits，或跳过 inactive dimensions 上的 KL。
- **Blurry samples.** Gaussian decoder likelihood 意味着 MSE reconstruction，而 MSE 对 L2 的 Bayes-optimal 解是 mean——一组 plausible digits 的 mean 就是 fuzzy digit。修复：discrete decoder（VQ-VAE、NVAE），或只把 VAE 当 encoder 并在 latents 上叠 diffusion（Stable Diffusion 就是这么做的）。
- **β too large, too early.** 见 posterior collapse。从 β≈0.01 开始并逐步 ramp。
- **Latent dim too small.** MNIST 用 16-D，ImageNet 256² 用 256-D，ImageNet 1024² 用 2048-D。Stable Diffusion 的 VAE 把 512×512×3 压缩成 64×64×4（spatial area 32x downsample factor，channels 也 32x）。

## 实际使用

2026 年的 VAE stack：

| Situation | Pick |
|-----------|------|
| Image-latent encoder for diffusion | Stable Diffusion VAE (`sd-vae-ft-ema`) or Flux VAE |
| Audio-latent encoder | Encodec (Meta), SoundStream, or DAC (Descript) |
| Video latents | Sora's spatiotemporal patches, Latte VAE, WAN VAE |
| Disentangled representation learning | β-VAE, FactorVAE, TCVAE |
| Discrete latents (for transformer modelling) | VQ-VAE, RVQ (ResidualVQ) |
| Continuous latents for generation | Plain VAE, then condition a flow/diffusion model in that latent space |

Latent-diffusion model 就是一个 VAE，中间夹着一个 diffusion model，位于 encoder 和 decoder 之间。VAE 做粗压缩，diffusion model 做重活。Video（VAE + video-diffusion DiT）和 audio（Encodec + MusicGen transformer）也是同一个 pattern。

## 交付成果

保存 `outputs/skill-vae-trainer.md`。

Skill 接收：dataset profile + latent-dim target + downstream use（reconstruction、sampling 或 latent-diffusion input），输出：architecture choice（plain/β/VQ/RVQ）、β schedule、latent dim、decoder likelihood（Gaussian vs categorical），以及 evaluation plan（recon MSE、KL per dim、`q(z|x)` 和 `N(0, I)` 之间的 Fréchet distance）。

## 练习

1. **Easy.** 把 `code/main.py` 中的 `β` 改成 `0.01`、`0.1`、`1.0`、`5.0`。记录最终 reconstruction MSE 和 KL。哪个 β 对你的 synthetic data 是 Pareto-best？
2. **Medium.** 用 Bernoulli likelihood（cross-entropy loss）替换 Gaussian decoder likelihood。在同一 synthetic data 的 binarized version 上比较 sample quality。
3. **Hard.** 把 `code/main.py` 扩展成 mini VQ-VAE：用 K=32 entries 的 codebook 中的 nearest-neighbour lookup 替换连续 `z`。比较 reconstruction MSE，并报告有多少 codebook entries 被使用（codebook collapse 是真实存在的）。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Autoencoder | Encode-decode network | `x → z → x̂`，学习 MSE。不是 generative。 |
| VAE | AE with a sampler | Encoder 输出 distribution，KL penalty 塑造 code space。 |
| ELBO | Evidence lower bound | `log p(x) ≥ recon - KL[q(z\|x) \|\| p(z)]`；当 `q = p(z\|x)` 时 tight。 |
| Reparameterization | `z = μ + σ·ε` | 把 stochastic node 重写成 deterministic + pure noise。让 backprop 可以穿过 sampling。 |
| Prior | `p(z)` | latent 的目标 distribution，通常是 `N(0, I)`。 |
| Posterior collapse | "KL term wins" | Encoder 忽略 `x`，输出 prior；decoder 必须 hallucinate。 |
| β-VAE | Tunable KL weight | `loss = recon + β·KL`。β 越高，越 disentangled 但越 blurry。 |
| VQ-VAE | Discrete latent | 用最近的 codebook vector 替换连续 `z`；支持 transformer modelling。 |

## 生产备注：VAE 是 diffusion server 里最热的路径

在 Stable Diffusion / Flux / SD3 pipeline 里，VAE 每个 request 会被调用两次——一次 encode（如果做 img2img / inpainting），一次 decode。在 1024² 时，decoder pass 往往是整条 pipeline 中 activation-memory peak 最大的单一步骤，因为它把 `128×128×16` latents upsample 回 `1024×1024×3`。两个实践后果：

- **Slice or tile the decode.** `diffusers` 暴露 `pipe.vae.enable_slicing()` 和 `pipe.vae.enable_tiling()`。Tiling 用轻微 seam artifact 换取 `O(tile²)` memory，而不是 `O(H·W)`。对 consumer GPUs 上的 1024²+ 必不可少。
- **bf16 decoder, fp32 numerics for the final resize.** SD 1.x VAE 以 fp32 发布，cast 到 fp16 时在 1024²+ 会 *silently produces NaNs*。SDXL 提供 `madebyollin/sdxl-vae-fp16-fix`——始终优先选择 fp16-fix variant，或使用 bf16。

## 延伸阅读

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — VAE paper。
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) — disentangled β-VAE。
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — VQ-VAE。
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) — state-of-the-art image VAE。
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion；VAE as encoder。
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — Encodec，audio VAE standard。
