# ControlNet, LoRA & Conditioning

> Text 单独作为 control signal 很笨拙。ControlNet 让你 clone 一个 pretrained diffusion model，并用 depth map、pose skeleton、scribble 或 edge image 去 steer 它。LoRA 让你通过训练 1000 万个 parameters 来 fine-tune 一个 2B-parameter model。两者一起，把 Stable Diffusion 从 toy 变成 2026 年每家 agency 都在交付的 image pipeline。

**类型：** Build
**语言：** Python
**先修：** Phase 8 · 07 (Latent Diffusion), Phase 10 (LLMs from Scratch — for LoRA foundation)
**时间：** ~75 分钟

## 要解决的问题

像 "a woman in a red dress walking a dog on a busy street" 这样的 prompt 不会告诉 model dog 在 *哪里*，woman 是 *什么 pose*，或者 street 的 *perspective* 是什么。Text 大约只能钉住你指定一张 image 所需信息的 10%。剩下的是视觉信息，无法高效用文字描述。

为每种 signal（pose、depth、canny、segmentation）从头训练一个新的 conditional model 成本过高。你想保留 2.6B-param SDXL backbone frozen，挂一个读取 conditioning 的小 side-network，让它轻推 backbone 的 intermediate features。这就是 ControlNet。

你还想教 model 学会新 concepts（你的脸、你的 product、你的 style），而不是 retrain full model。你想要一个小 100x 的 delta。这就是 LoRA——low-rank adapters，插进已有 attention weights。

ControlNet + LoRA + text = 2026 年 practitioner's toolkit。大多数 production image pipelines 会在 SDXL / SD3 / Flux base 之上叠 2-5 个 LoRAs、1-3 个 ControlNets，以及一个 IP-Adapter。

## 核心概念

![ControlNet clones the encoder; LoRA adds low-rank deltas](../assets/controlnet-lora.svg)

### ControlNet (Zhang et al., 2023)

取一个 pretrained SD。*Clone* U-Net 的 encoder half。Freeze 原始模型。训练这个 clone 接受额外 conditioning input（edges、depth、pose）。用 *zero-convolution* skip connections（初始化为 zero 的 1×1 convs——从 no-op 开始，学习 delta）把 clone 接回原模型的 decoder half。

```text
SD U-Net decoder:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

Zero-conv init 意味着 ControlNet 从 identity 开始——即使训练前也无害。用标准 diffusion loss 在 1M（prompt, condition, image）triples 上训练。

每种 modality 的 ControlNets 作为小 side models 交付（SDXL 约 360M，SD 1.5 约 70M）。Inference 时可以组合它们：

```text
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA (Hu et al., 2021)

对 model 中任意 linear layer `W ∈ R^{d×d}`，freeze `W` 并添加 low-rank delta：

```text
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

其中 `r << d`。Attention 标准 rank 是 4-16，heavy fine-tunes 用 64-128。新增参数数：`2 · d · r`，而不是 `d²`。对 `d=640` 的 SDXL attention、`r=16`：每个 adapter 20k params，而不是 410k——减少 20x。整个 model 上，一个 LoRA 通常是 20-200MB，而 base 是 5GB。

Inference 时可以 scale LoRA：`W' = W + α · B @ A`。`α = 0.5-1.5` 很常见。多个 LoRAs 会 additively stack（通常 caveat 是它们会以 non-linear ways 交互）。

### IP-Adapter (Ye et al., 2023)

一个 tiny adapter，接受 *image* 作为 conditioning（与 text 并列）。它使用 CLIP image encoder 生成 image tokens，再把它们与 text tokens 一起注入 cross-attention。每个 base model 约 20MB。让你无需 LoRA 就能做 “generate an image in the style of this reference”。

## Composability matrix

| Tool | What it controls | Size | When to use |
|------|------------------|------|-------------|
| ControlNet | Spatial structure (pose, depth, edges) | 70-360MB | Exact layout, composition |
| LoRA | Style, subject, concept | 20-200MB | Personalization, style |
| IP-Adapter | Style or subject from reference image | 20MB | No text can describe the look |
| Textual Inversion | Single concept as a new token | 10KB | Legacy, mostly replaced by LoRA |
| DreamBooth | Full fine-tune on a subject | 2-5GB | Strong identity, high compute |
| T2I-Adapter | Lighter ControlNet alternative | 70MB | Edge devices, inference budget |

ControlNet ≈ spatial。LoRA ≈ semantic。两者一起用。

## 动手实现

`code/main.py` 在 1-D 中模拟这两种 mechanisms：

1. **LoRA.** 一个 pretrained linear layer `W`。Freeze 它。训练 low-rank `B @ A`，使 `W + BA` 匹配一个 target linear layer。展示 `r = 1` 足以完美学习 rank-1 correction。

2. **ControlNet-lite.** 一个 "frozen base" predictor 和一个读取 extra signal 的 "side network"。Side network 的 output 被一个初始化为 zero 的 learnable scalar gate 控制（我们的 zero-conv 版本）。训练并观察 gate ramp up。

### Step 1: LoRA math

```python
def lora(W, A, B, x, alpha=1.0):
    # W is frozen; A, B are the trainable low-rank factors.
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### Step 2: zero-init side network

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate initialized to 0
h = base(x) + gated
```

Step 0 时 output 与 base 完全相同。Early training 会慢慢更新 `gate`——不会 catastrophic drift。

## Pitfalls

- **Over-scaling LoRAs.** `α = 2` 或 `α = 3` 是常见的“make it stronger”hack，会产生 over-stylized / broken outputs。保持 `α ≤ 1.5`。
- **ControlNet weight conflict.** Pose ControlNet 使用 weight 1.0 且 Depth ControlNet 也使用 weight 1.0，通常会 overshoot。Weights 总和 ≈ 1.0 是安全默认值。
- **LoRA on the wrong base.** SDXL LoRAs 在 SD 1.5 上会 silently no-op，因为 attention dimensions 不匹配。Diffusers 0.30+ 会 warning。
- **Textual Inversion drift.** 在一个 checkpoint 上训练的 tokens 在另一个 checkpoint 上会严重 drift。LoRA 更 portable。
- **LoRA weight-merging and storage.** 你可以把 LoRA bake 到 base model weights 里，以获得更快 inference（没有 runtime addition），但会失去 runtime scale `α` 的能力。保留两个 versions。

## 实际使用

| Goal | 2026 pipeline |
|------|---------------|
| Reproduce a brand's art style | LoRA trained on ~30 curated images at rank 32 |
| Put my face in a generated image | DreamBooth or LoRA + IP-Adapter-FaceID |
| Specific pose + prompt | ControlNet-Openpose + SDXL + text |
| Depth-aware composition | ControlNet-Depth + SD3 |
| Reference + prompt | IP-Adapter + text |
| Exact layout | ControlNet-Scribble or ControlNet-Canny |
| Background replace | ControlNet-Seg + Inpainting (Lesson 09) |
| Fast 1-step style | LCM-LoRA on SDXL-Turbo |

## 交付成果

保存 `outputs/skill-sd-toolkit-composer.md`。Skill 接收一个 task（input assets：prompt、optional reference image、optional pose、optional depth、optional scribble），并输出 tool stack、weights，以及 reproducible seed protocol。

## 练习

1. **Easy.** 在 `code/main.py` 中，把 LoRA rank `r` 从 1 变到 4。在什么 rank 下 LoRA 能精确匹配 rank-2 target delta？
2. **Medium.** 分别在两个 target transforms 上训练两个 LoRAs。一起加载它们，并展示 additive interaction。Interaction 什么时候会破坏 linearity？
3. **Hard.** 使用 diffusers stack：SDXL-base + Canny-ControlNet（weight 0.8）+ 一个 style LoRA（α 0.8）+ IP-Adapter（weight 0.6）。随着 stack weights 变化，测量 FID-vs-prompt-adherence trade-off。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| ControlNet | "Spatial control" | Cloned encoder + zero-conv skips；读取 conditioning image。 |
| Zero convolution | "Starts as identity" | 初始化为 zero 的 1×1 conv；ControlNet 从 no-op 开始。 |
| LoRA | "Low-rank adapter" | `W + B @ A`，`r << d`；比 full fine-tune 少 100x params。 |
| rank r | "The knob" | LoRA compression；典型 4-16，heavy personalization 用 64+。 |
| α | "LoRA strength" | LoRA delta 的 runtime scaling。 |
| IP-Adapter | "Reference image" | 通过 CLIP-image tokens 实现的小 image-conditioning adapter。 |
| DreamBooth | "Full subject fine-tune" | 在某个 subject 的约 30 张 images 上训练 full model。 |
| Textual Inversion | "New token" | 只学习新的 word embedding；legacy，基本被替代。 |

## 生产备注：LoRA swaps、ControlNet lanes、multi-tenant serving

真实 text-to-image SaaS 会在同一个 base checkpoint 上服务数百个 LoRAs 和十几个 ControlNets。Serving problem 很像 LLM multi-tenancy（production literature 在 continuous batching 和 LoRAX / S-LoRA 下讨论 LLM case）：

- **Hot-swap LoRAs, do not merge.** 把 `W' = W + α·B·A` merge 到 base 中可以让 per-step inference 快 ~3-5%，但会 freeze `α` 和 base。把 LoRAs 作为 rank-r deltas hot 在 VRAM 中；diffusers 提供 `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])` 做 per-request activation。Swap cost 是 `2 · d · r · num_layers` weights——MB-scale、sub-second。
- **ControlNet as a second attention lane.** Cloned encoder 与 base 并行运行。两个 weight 1.0 的 ControlNets = 每 step 两次额外 forward passes，而不是一次 merged pass。Batch-size headroom 会二次下降。每个 active ControlNet 预算约 ~1.5× step cost。
- **Quantized LoRAs too.** 如果你 quantized base（见 Lesson 07，Flux on 8GB），LoRA delta 也能干净 quantize 到 8-bit 或 4-bit。QLoRA-style loading 让你可以在 4-bit Flux base 上叠 5-10 个 LoRAs 而不爆 memory。

Flux-specific：Niels 的 Flux-on-8GB notebook 把 base quantize 到 4-bit；在这个 quantized base 上 stack 一个 style LoRA（`pipe.load_lora_weights("user/style-lora")`），并使用 `weight_name="pytorch_lora_weights.safetensors"`，仍然可行。这就是 2026 年大多数 SaaS agencies 交付的 recipe。

## 延伸阅读

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — ControlNet。
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA（最初用于 LLMs；移植到 diffusion）。
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — IP-Adapter。
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — 更轻的 ControlNet alternative。
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — DreamBooth。
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter docs](https://huggingface.co/docs/diffusers/training/controlnet) — 上述每个 checkpoint 的 reference pipelines。
