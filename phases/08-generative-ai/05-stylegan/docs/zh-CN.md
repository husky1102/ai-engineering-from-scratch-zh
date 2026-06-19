# StyleGAN 风格生成

> 大多数 generators 会把 `z` 同时搅进每一层。StyleGAN 把它拆开：先把 `z` map 到中间的 `w`，再通过 AdaIN 在每个 resolution level *inject* `w`。这一个变化解开了 latent space，并让 photorealistic faces 连续七年成为已解决问题。

**类型：** Build
**语言：** Python
**先修：** Phase 8 · 03 (GANs), Phase 4 · 08 (Normalization), Phase 3 · 07 (CNNs)
**时间：** ~45 分钟

## 要解决的问题

DCGAN 通过一叠 transposed convolutions 把 `z` map 成 image。问题是：`z` 控制一切——pose、lighting、identity、background——全都 entangled 在一起。沿 `z` 的一个轴移动，四者都会变。你无法要求模型“same person, different pose”，因为 representation 不会那样 factor。

Karras et al.（2019, NVIDIA）提出：停止把 `z` 直接喂给 conv layers。用一个 constant `4×4×512` tensor 作为 network input。学习一个 8-layer MLP，把 `z ∈ Z → w ∈ W`。通过 *adaptive instance normalization*（AdaIN）在每个 resolution 注入 `w`：normalize 每个 conv feature map，然后用 `w` 的 affine projections 做 scale 和 shift。为 stochastic detail（skin pores、hair strands）加入 per-layer noise。

结果：`W` 中大致有正交 axes，对应 “high-level style”（pose、identity）和 “fine style”（lighting、color）。你可以在两张 images 之间交换 styles：用 image A 的 `w` 给 low-resolution levels，用 image B 的 `w` 给 high-resolution levels。这开启了 editing、cross-domain stylization，以及整个 “StyleGAN-inversion” 研究线。

## 核心概念

![StyleGAN: mapping network + AdaIN + per-layer noise](../assets/stylegan.svg)

**Mapping network.** `f: Z → W`，一个 8-layer MLP。`Z = N(0, I)^512`。`W` 不被强制为 Gaussian——它学习 data-adapted shape。

**Synthesis network.** 从 learned constant `4×4×512` 开始。每个 resolution block：`upsample → conv → AdaIN(w_i) → noise → conv → AdaIN(w_i) → noise`。Resolutions 翻倍：4、8、16、32、64、128、256、512、1024。

**AdaIN.**

```text
AdaIN(x, y) = y_scale · (x - mean(x)) / std(x) + y_bias
```

其中 `y_scale` 和 `y_bias` 来自 `w` 的 affine projections。先按 feature map normalize，再 restyle。这里的 “Style” 是 feature map 的 first- and second-order statistics。

**Per-layer noise.** 单通道 Gaussian noise 加到每个 feature map 上，并由 learned per-channel factor 缩放。控制 stochastic detail，而不影响 global structure。

**Truncation trick.** Inference 时，sample `z`，计算 `w = mapping(z)`，然后 `w' = ŵ + ψ·(w - ŵ)`，其中 `ŵ` 是许多 samples 上的 mean `w`。`ψ < 1` 用 diversity 换 quality。几乎每个 StyleGAN demo 都使用 `ψ ≈ 0.7`。

## StyleGAN 1 → 2 → 3

| Version | Year | Innovation |
|---------|------|------------|
| StyleGAN | 2019 | Mapping network + AdaIN + noise + progressive growing。 |
| StyleGAN2 | 2020 | Weight demodulation 替代 AdaIN（修复 droplet artifacts）；skip/residual architecture；path-length regularization。 |
| StyleGAN3 | 2021 | Alias-free convolution + equivariant kernels；消除 texture sticking to pixel grid。 |
| StyleGAN-XL | 2022 | Class-conditional，1024²，ImageNet。 |
| R3GAN | 2024 | 用更强 reg 重新包装；在 FFHQ-1024 上用 20x 更少 params 缩小与 diffusion 的差距。 |

到 2026 年，StyleGAN3 仍是以下场景的默认选择：（a）high FPS 的 narrow-domain photorealism，（b）few-shot domain adaptation（用 100 张 images 在新 dataset 上训练，freeze mapping），（c）inversion-based editing（找到能重建 real photo 的 `w`，再 edit 这个 `w`）。对于 open-domain text-to-image，它不是合适工具——diffusion 才是。

## 动手实现

`code/main.py` 在 1-D 中实现 toy "style-GAN lite"：一个 mapping MLP，一个 synthesis function（接受 learned constant vector 并用 `w` 派生的 scale/bias 调制它），以及 per-layer noise。它展示了通过 affine-modulation 注入 `w` 可以匹配或优于把 `z` concat 到 generator input。

### Step 1: mapping network

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### Step 2: adaptive instance normalization

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

Per-feature-map scale 和 bias 来自 `w` 的 linear projection。

### Step 3: per-layer noise

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

Sigma per-channel 是 learnable 的。

## 常见陷阱

- **Droplet artifacts.** StyleGAN 1 因为 AdaIN 把 mean 置零，会在 feature maps 中产生 blobby droplet。StyleGAN 2 的 weight demodulation 通过缩放 convolution weights 修复它。
- **Texture sticking.** StyleGAN 1 和 2 的 textures 跟随 pixel coordinates，而不是 object coordinates（interpolating 时可见）。StyleGAN 3 的 alias-free convolutions 用 windowed sinc filters 修复了这个问题。
- **Mode coverage.** Truncation `ψ < 0.7` 看起来干净，但只从一个 narrow cone 里 sampling；如果需要 diversity，使用 `ψ = 1.0`。
- **Inversion is lossy.** 把 real photo invert 到 `W` 通常通过 optimization 或 encoder（e4e、ReStyle、HyperStyle）完成。结果在许多 iterations 后会漂移。

## 实际使用

| Use case | Approach |
|----------|----------|
| Photoreal human faces (anime, product, narrow) | StyleGAN3 FFHQ / custom fine-tune |
| Face editing from a photo | e4e inversion + StyleSpace / InterFaceGAN directions |
| Face swap / reenactment | StyleGAN + encoder + blending |
| Avatar pipelines | StyleGAN3 w/ ADA for low-data fine-tune |
| Domain adaptation from a few images | Freeze mapping network, fine-tune synthesis |
| Multi-modal or text-conditioned generation | Don't — use diffusion |

对 product-grade demos 来说，如果答案是“photo of a person's face”，StyleGAN 在 inference cost（single forward pass，4090 上 <10ms）和同等 quality bar 下的 sharpness 上胜过 diffusion。

## 交付成果

保存 `outputs/skill-stylegan-inversion.md`。Skill 接收一张 real photo，并输出：inversion method（e4e / ReStyle / HyperStyle）、expected latent loss、editing budget（你可以在 artifacts 出现前在 `W` 中移动多远），以及 known-good editing directions（age、expression、pose）列表。

## 练习

1. **Easy.** 分别以 `adain_on=True` 和 `adain_on=False` 运行 `code/main.py`。比较 fixed latent 与 perturbed latent 的 outputs spread。
2. **Medium.** 实现 mixing regularization：对一个 training batch，计算 `w_a`、`w_b`，在 synthesis 的前半段使用 `w_a`，后半段使用 `w_b`。Decoder 是否学会 disentangled styles？
3. **Hard.** 拿一个 pretrained StyleGAN3 FFHQ model（ffhq-1024.pkl）。通过在 labelled samples 上训练 SVM，找到控制 "smile" 的 `w` direction；报告 identity drift 之前你能推多远。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Mapping network | "The MLP" | `f: Z → W`，8 layers，把 latent geometry 与 data statistics 解耦。 |
| W space | "The style space" | Mapping network 的 output；大致 disentangled。 |
| AdaIN | "Adaptive instance norm" | Normalize feature map，然后用 `w`-projection 做 scale + shift。 |
| Truncation trick | "Psi" | `w = mean + ψ·(w - mean)`，ψ<1 用 diversity 换 quality。 |
| Path-length regularization | "PL reg" | 惩罚 image 对 `w` 单位变化的过大变化；让 `W` 更平滑。 |
| Weight demodulation | "The StyleGAN2 fix" | Normalize conv weights，而不是 activations；消除 droplet artifacts。 |
| Alias-free | "StyleGAN3's trick" | Windowed sinc filters；消除 texture sticking to the pixel grid。 |
| Inversion | "Find w for a real image" | Optimize 或 encode `x → w`，使 `G(w) ≈ x`。 |

## 生产备注：为什么 StyleGAN 在 2026 年仍然交付

4090 上的 StyleGAN3 能在 10 ms 内生成一张 1024² FFHQ face——`num_steps = 1`，没有 VAE decode，没有 cross-attention pass。用 production terms 说，这是任何 image generator 的 floor latency。同 resolution 下，50-step SDXL + VAE-decode pipeline 约 3 秒。这是 **300× gap**，在 narrow-domain products（avatar services、ID document pipelines、stock face generation）中，它会赢在 TCO。

两个 operational consequences：

- **No scheduler, no batcher.** 目标 occupancy 上的 static batch 是最优。Continuous batching（对 LLMs 和 diffusion 必不可少）没有收益，因为每个 request 都使用相同 FLOPs。
- **Truncation `ψ` is the safety knob.** `ψ < 0.7` 从 mapping network range 的 narrow cone 中 sample。这是 serving layer 对 sample variance 拥有的唯一 lever。Peak load 时降低 `ψ`，对 premium users 提高它。

## 延伸阅读

- [Karras et al. (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) — StyleGAN。
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2。
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3。
- [Tov et al. (2021). Designing an Encoder for StyleGAN Image Manipulation](https://arxiv.org/abs/2102.02766) — e4e inversion。
- [Sauer et al. (2022). StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets](https://arxiv.org/abs/2202.00273) — StyleGAN-XL。
- [Huang et al. (2024). R3GAN: The GAN is dead; long live the GAN!](https://arxiv.org/abs/2501.05441) — modern minimal GAN recipe。
