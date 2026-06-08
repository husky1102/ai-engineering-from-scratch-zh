# Latent Diffusion & Stable Diffusion

> 在 512×512 images 上做 pixel-space diffusion 是一场计算层面的战争罪。Rombach et al.（2022）注意到，生成一张 image 不需要全部 786k dimensions——你需要足够捕捉 semantic structure 的维度，并用一个独立 decoder 处理其余部分。在 VAE 的 latent space 里运行 diffusion。这一个想法就是 Stable Diffusion。

**类型：** Build
**语言：** Python
**先修：** Phase 8 · 02 (VAE), Phase 8 · 06 (DDPM), Phase 7 · 09 (ViT)
**时间：** ~75 分钟

## 要解决的问题

512² 的 pixel-space diffusion 意味着 U-Net 跑在 shape 为 `[B, 3, 512, 512]` 的 tensors 上。对一个 500M-param U-Net，每个 sampling step 约 100 GFLOPS。五十步就是每张 image 5 TFLOPS。在十亿张 images 上训练，compute bill 荒谬。

大多数 FLOPs 都花在把 perceptually unimportant details 推过 net——也就是 lossy VAE 本可以压缩掉的 high-frequency texture。Rombach 的想法：先训练一次 VAE（*first stage*），freeze 它，然后完全在 4-channel 64×64 latent space（*second stage*）里运行 diffusion。同一个 U-Net。1/16 的 pixels。约 64x 更少 FLOPs，quality 相近。

这就是 Stable Diffusion recipe。SD 1.x / 2.x 在 `64×64×4` latents 上使用 860M U-Net，SDXL 在 `128×128×4` 上使用 2.6B U-Net，SD3 把 U-Net 换成带 flow matching 的 Diffusion Transformer（DiT）。Flux.1-dev（Black Forest Labs, 2024）交付了一个 12B-param DiT-MMDiT。它们都运行在同一个 two-stage substrate 上。

## 核心概念

![Latent diffusion: VAE compression + diffusion in latent space](../assets/latent-diffusion.svg)

**Two stages, separately trained.**

1. **Stage 1 — VAE.** Encoder `E(x) → z`，decoder `D(z) → x`。目标压缩：每个 spatial axis downsample 8× + 调整 channels，使总 latent size 约为 pixel count 的 1/16。Loss = reconstruction（L1 + LPIPS perceptual）+ KL（小权重，因此 `z` 不会被强制得太 Gaussian，因为我们不需要从 `z` 精确 sampling）。通常还会配合 adversarial loss 训练，让 decoded images 更 sharp。

2. **Stage 2 — diffusion on `z`.** 把 `z = E(x_real)` 当作 data。训练 U-Net（或 DiT）denoise `z_t`。Inference 时：通过 diffusion sample `z_0`，然后 `x = D(z_0)`。

**Text conditioning.** 另外两个 components。Frozen text encoder（SD 1.x 用 CLIP-L，SD 2/XL 用 CLIP-L+OpenCLIP-G，SD3 和 Flux 用 T5-XXL）。Cross-attention injection：每个 U-Net block 接收 `[Q = image features, K = V = text tokens]` 并混合它们。Tokens 是 text 影响 image 的唯一方式。

**The loss function is identical to Lesson 06.** 同一个 DDPM / flow matching 的 noise MSE。你只是换了 data domain。

## Architecture variants

| Model | Year | Backbone | Latent shape | Text encoder | Params |
|-------|------|----------|--------------|--------------|--------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L (77 tokens) | 860M |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 865M |
| SDXL | 2023 | U-Net + refiner | 128×128×4 | CLIP-L + OpenCLIP-G | 2.6B + 6.6B |
| SDXL-Turbo | 2023 | Distilled | 128×128×4 | same | 1-4 step sampling |
| SD3 | 2024 | MMDiT (multimodal DiT) | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 2B / 8B |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B |
| Flux.1-schnell | 2024 | MMDiT distilled | 128×128×16 | T5-XXL + CLIP-L | 12B, 1-4 step |

趋势：用 DiT（latent patches 上的 transformer）替换 U-Net，scale text encoder（T5 在 prompt adherence 上胜过 CLIP），增加 latent channels（4 → 16 给更多 detail headroom）。

## 动手实现

`code/main.py` 在 Lesson 06 的 DDPM 上叠了一个 toy 1-D "VAE"（identity encoder + decoder，用于演示；真实 VAE 会是 conv net），并加入带 classifier-free guidance 的 class conditioning。它展示了同一个 diffusion loss 无论跑在 raw 1-D values 上还是 encoded values 上都有效——这就是关键 insight。

### Step 1: encoder/decoder

```python
def encode(x):    return x * 0.5          # toy "compression" to smaller scale
def decode(z):    return z * 2.0
```

真实 VAE 有训练好的 weights。为了教学，这个 linear map 足以展示 diffusion 在 `z` 上运行，而不关心原始 data space。

### Step 2: diffusion in `z`-space

与 Lesson 06 相同的 DDPM。Net 看到的 data 是 `z = E(x)`。Sampling 出 `z_0` 后，用 `D(z_0)` decode。

### Step 3: classifier-free guidance

Training 时，10% 的时间 drop class label（替换成 null token）。Inference 时，同时计算 `ε_cond` 和 `ε_uncond`，然后：

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0` = no guidance（full diversity），`w = 3` = default，`w = 7+` = saturated / over-sharp。

### Step 4: text conditioning (concept, not code)

用 frozen text encoder output 替换 class label。通过 cross-attention 把 text embedding 喂给 U-Net：

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

这是 class-conditional diffusion model 与 Stable Diffusion 之间唯一实质区别。

## Pitfalls

- **VAE-scale mismatch.** SD 1.x VAE 在 encoding 后应用一个 scaling constant（`scaling_factor ≈ 0.18215`）。忘记它会让 U-Net 在 variance 极端错误的 latents 上训练。每个 checkpoint 都带一个。
- **Text encoder silently wrong.** SD3 需要 T5-XXL 且 >=128 tokens，fallback 到 CLIP-only 是有损的。始终检查 `use_t5=True`，否则 prompt fidelity 会崩。
- **Mixing latent spaces.** SDXL、SD3、Flux 使用不同 VAEs。在 SDXL latents 上训练的 LoRA 无法用于 SD3。Hugging Face diffusers 0.30+ 会拒绝加载 mismatched checkpoints。
- **CFG too high.** `w > 10` 会产生 saturated、oily images，并以 diversity 为代价 over-fit prompt。Sweet spot 是 `w = 3-7`。
- **Negative prompts leaking.** Empty negative prompt 会变成 null token；填了内容的 negative prompt 会变成 `ε_uncond`。它们不是同一个东西；一些 pipelines 会 silently default 到 null。

## 实际使用

2026 年 production stacks：

| Target | Recommended backbone |
|--------|----------------------|
| Narrow domain, paired data, training a model from scratch | SDXL fine-tune (LoRA / full) — fastest to ship |
| Open-domain text-to-image, open weights | Flux.1-dev (12B, Apache / non-commercial) or SD3.5-Large |
| Fastest inference, open weights | Flux.1-schnell (1-4 step, Apache) or SDXL-Lightning |
| Best prompt adherence, hosted | GPT-Image / DALL-E 3 (still), Midjourney v7, Imagen 4 |
| Edit workflows | Flux.1-Kontext (Dec 2024) — natively accepts image + text |
| Research, baseline | SD 1.5 — ancient but well-studied |

## 交付成果

保存 `outputs/skill-sd-prompter.md`。Skill 接收 text prompt + target style，并输出：model + checkpoint、CFG scale、sampler、negative prompt、resolution、optional ControlNet/IP-Adapter combo，以及 per-step QA checklist。

## 练习

1. **Easy.** 用 guidance `w ∈ {0, 1, 3, 7, 15}` 运行 `code/main.py`。记录每个 class 的 mean sample。在哪个 `w` 下，class means 会偏离 real data means？
2. **Medium.** 把 toy linear encoder 替换成带 reconstruction loss 的 tanh-MLP encoder/decoder pair。在新 latents 上重新训练 diffusion。Sample quality 是否改变？
3. **Hard.** 用 diffusers 设置一个真实 Stable Diffusion inference：加载 `sdxl-base`，用 CFG=7 运行 30 Euler steps 并计时。现在切到 `sdxl-turbo`，4 steps 且 CFG=0。同一个 subject，不同 quality——描述改变了什么，以及为什么。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| First stage | "The VAE" | 训练好的 encoder/decoder pair；把 512² 压缩到 64²。 |
| Second stage | "The U-Net" | Latent space 上的 diffusion model。 |
| CFG | "Guidance scale" | `(1+w)·ε_cond - w·ε_uncond`；调节 conditioning strength。 |
| Null token | "Empty prompt embed" | 用于 `ε_uncond` 的 unconditional embed。 |
| Cross-attention | "How text gets in" | 每个 U-Net block 把 text tokens 作为 K 和 V attend。 |
| DiT | "Diffusion Transformer" | 用 latent patches 上的 transformer 替换 U-Net；scales better。 |
| MMDiT | "Multi-modal DiT" | SD3 architecture：text 和 image streams 使用 joint attention。 |
| VAE scaling factor | "Magic number" | 把 latents 除以约 5.4，让 diffusion 在 unit-variance space 中工作。 |

## 生产备注：在 8GB consumer GPU 上运行 Flux-12B

the reference Flux integration 是 canonical 的“我有 consumer GPU，能交付这个吗？”recipe。技巧就是 production inference literature 列出的同一组三个 knobs，只是应用在 diffusion DiT 上：

1. **Staggered loading.** Flux 有三个永远不需要同时存在于 VRAM 的 networks：T5-XXL text encoder（fp32 约 10 GB）、CLIP-L（small）、12B MMDiT，以及 VAE。先 encode prompt，*delete* encoders，load DiT，denoise，*delete* DiT，load VAE，decode。Consumer 8GB GPUs 一次只装得下一 stage。
2. **4-bit quantization via bitsandbytes.** 对 T5 encoder 和 DiT 使用 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)`。Memory 降 8×；按 Aritra 的 benchmarks（notebook 中链接），text-to-image 的 quality drop 几乎不可见。
3. **CPU offload.** `pipe.enable_model_cpu_offload()` 会随着每次 forward pass 推进，自动在 CPU 和 GPU 之间 swap modules。增加 10-20% latency，但让 pipeline 至少能运行。

Memory accounting 是：`10 GB T5 / 8 = 1.25 GB` quantized，`12 B params × 0.5 bytes = ~6 GB` quantized DiT，再加 activations。用 stas00 的术语，这是 TP=1 inference 的 extreme-end——没有 model parallelism，最大 quantization。Production 中你会在 H100 上跑 TP=2 或 TP=4；对 single dev laptop，这就是 recipe。

## 延伸阅读

- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion。
- [Podell et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) — SDXL。
- [Peebles & Xie (2023). Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) — DiT。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3，MMDiT。
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG。
- [Labs (2024). Flux.1 — Black Forest Labs announcement](https://blackforestlabs.ai/announcing-black-forest-labs/) — Flux.1 family。
- [Hugging Face Diffusers docs](https://huggingface.co/docs/diffusers/index) — reference implementation for every checkpoint above。
