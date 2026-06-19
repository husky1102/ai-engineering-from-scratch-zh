# Inpainting、Outpainting 与 Image Editing

> Text-to-image 会创造新东西。Inpainting 修复旧东西。在生产中，70% 可计费图像工作都是编辑——替换背景、移除 logo、扩展画布、重新生成手部。Inpainting 是 diffusion 真正赚回成本的地方。

**类型:** Build
**语言:** Python
**先修:** Phase 8 · 07 (Latent Diffusion), Phase 8 · 08 (ControlNet & LoRA)
**时间:** ~75 minutes

## 要解决的问题

客户发来一张完美产品照，但背景里有个分散注意力的标牌。你想擦掉标牌，并让其他所有像素保持完全一致。你不能从头运行 text-to-image——结果会有不同颜色、不同光照、不同产品角度。你想只重新生成 masked region，并让重新生成结果尊重周围 context。

这就是 inpainting。变体包括：

- **Inpainting.** 在 mask 内重新生成，保留外部 pixels。
- **Outpainting.** 在 mask 外（或画布之外）重新生成，保留内部。
- **Image editing.** 重新生成整张图，但保持与原图的 semantic 或 structural fidelity（SDEdit、InstructPix2Pix）。

2026 年每个 diffusion pipeline 都有 inpainting mode。Flux.1-Fill、Stable Diffusion Inpaint、SDXL-Inpaint、DALL-E 3 Edit。它们基于同一个原则。

## 核心概念

![Inpainting: mask-aware denoising with context-preserving reinjection](../assets/inpainting.svg)

### 朴素方法（以及为什么它错）

带着 mask 运行标准 text-to-image。每个 sampling step，把 noisy latent 的 unmasked region 替换成 clean image 的 forward-diffused 版本。它能工作……但效果很糟。Boundary artifacts 会渗出，因为模型不知道 masked region 里应该有什么。

### 正确的 inpainting model

训练一个修改版 U-Net，它接收 9 个 input channels，而不是 4 个：

```text
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

额外 channels 是 VAE-encoded source image 的副本，加一个 single-channel mask。训练时，随机 mask 图像区域，训练模型只 denoise masked region，同时把 unmasked region 作为 clean conditioning signal 给它。推理时，模型能“看见” masked region 周围是什么，并生成连贯补全。

SD-Inpaint、SDXL-Inpaint、Flux-Fill 都使用这种 9-channel（或类似）input。Diffusers 中对应 `StableDiffusionInpaintPipeline`、`FluxFillPipeline`。

### SDEdit (Meng et al., 2022) — free editing

把 source image 加噪到某个中间 `t`，然后用新 prompt 从 `t` 反向运行到 0。不需要重新训练。起始 `t` 的选择会在 fidelity 与 creative freedom 之间权衡：

- `t/T = 0.3` → 几乎与 source 相同，只做小风格变化
- `t/T = 0.6` → 中等编辑，保留粗结构
- `t/T = 0.9` → 从近似噪声生成，source preservation 很低

### InstructPix2Pix (Brooks et al., 2023)

在 `(input_image, instruction, output_image)` triples 上 fine-tune diffusion model。推理时同时 condition on input image 和 text instruction（“make it sunset”、“add a dragon”）。有两个 CFG scales：image scale 和 text scale。

### RePaint (Lugmayr et al., 2022)

保留标准 unconditional diffusion model。每个 reverse step 做 resample——偶尔跳回更 noisy 的状态再重新生成。避免 boundary artifacts。适用于没有训练好的 inpainting model 时。

## 动手实现

`code/main.py` 在 5-dimensional data 上实现 toy 1-D inpainting scheme。我们在 5-D mixture data 上训练 DDPM，每个 sample 是来自两个 clusters 之一的 5 个 floats。推理时，我们 “mask” 5 个维度中的 2 个，在每一步注入 unmasked 三个维度的 noisy-forward 版本，并只重新生成 masked dimensions。

### Step 1: 5-D DDPM data

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### Step 2: train denoiser over all 5 dims

标准 DDPM。Net 为 5-D noisy input 输出 5-D noise prediction。

### Step 3: at inference, mask-aware reverse

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # replace unmasked dims with a freshly noised version of the clean source
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...then run the normal reverse step on x_t
```

这是朴素方法，而且在 toy 1-D data 上能工作。真实图像 inpainting 使用 9-channel input，因为 texture coherence 更重要。

### Step 4: outpainting

Outpainting 是 mask 反转后的 inpainting：mask 新的（之前不存在的）canvas，其余部分用原图填充。训练目标完全相同。

## 常见陷阱

- **Seams.** 朴素方法会留下可见边界，因为 gradient info 不能跨 mask 流动。修复：把 mask 膨胀 8-16 pixels，或使用 proper inpainting model。
- **Mask leakage.** 如果 conditioning image 的 unmasked region 低质量或有噪声，它会污染 mask 内的 generation。先 denoise 或轻微 blur。
- **CFG interacts with mask size.** 小 mask 上高 CFG = 饱和 patch。小编辑应降低 CFG。
- **SDEdit fidelity cliff.** 从 `t/T = 0.5` 到 `t/T = 0.6` 可能丢掉主体身份。Sweep 并 checkpoint。
- **Prompt mismatch.** Prompt 应描述*整张*图，而不只是新增内容。用 “A cat sitting on a chair”，不是 “a cat”。

## 实际使用

| Task | Pipeline |
|------|----------|
| Remove object, small mask | SD-Inpaint or Flux-Fill, standard prompt |
| Replace sky | SD-Inpaint + "blue sky at sunset" |
| Extend canvas | SDXL outpaint mode (8px feather) or Flux-Fill with outpaint mask |
| Regenerate hand / face | SD-Inpaint with prompt re-describing the subject + ControlNet-Openpose |
| Change style of one region | SDEdit at `t/T=0.5` on masked region |
| "Make it sunset" | InstructPix2Pix or Flux-Kontext |
| Background replacement | SAM mask → SD-Inpaint |
| Ultra-high-fidelity | Flux-Fill or GPT-Image (hosted) for hardest cases |

SAM（Meta 的 Segment Anything，2023）+ diffusion inpaint 是 2026 年 background-removal pipeline。SAM 2（2024）适用于 video。

## 交付成果

保存 `outputs/skill-editing-pipeline.md`。Skill 接收 original image + edit description + optional mask（或 SAM prompt），输出：mask-generation approach、base model、CFG scales（image + text）、SDEdit-t 或 inpainting mode，以及 QA checklist。

## 练习

1. **Easy.** 在 `code/main.py` 中，把 masked dimensions 的比例从 0.2 改到 0.8。到哪个比例时，inpaint quality（masked dims 中的 residual）等于 unconditional generation？
2. **Medium.** 实现 RePaint：每第 10 个 reverse step，跳回 5 steps（add noise）并重新 denoise。测量它是否减少 mask edge 的 boundary residual。
3. **Hard.** 使用 Hugging Face diffusers 比较：SD 1.5 Inpaint + ControlNet-Openpose vs Flux.1-Fill，在 20 个 face-regeneration tasks 上。分别评分 pose adherence 和 identity preservation。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Inpainting | “Fill the hole” | 在 mask 内重新生成；保留外部 pixels。 |
| Outpainting | “Extend the canvas” | 在画布外重新生成；保留内部。 |
| 9-channel U-Net | “Proper inpainting model” | 以 `noisy \| encoded-source \| mask` 为 input 的 U-Net。 |
| SDEdit | “Img2img with noise level” | 加噪到时间 `t`，再用新 prompt denoise。 |
| InstructPix2Pix | “Text-only edits” | 在 (image, instruction, output) triples 上 fine-tuned 的 diffusion。 |
| RePaint | “No retraining” | Reverse 过程中周期性 re-noise，以减少 seams。 |
| SAM | “Segment Anything” | 通过 clicks 或 boxes 生成 mask；常与 inpaint 搭配。 |
| Flux-Kontext | “Edit with context” | 接收 reference image + instruction 做 edits 的 Flux variant。 |

## Production note: edit pipelines are latency-sensitive

用户编辑图像时期待 sub-5-second round trips。1024² 上 30-step SDXL-Inpaint 在 L4 上约 3-4 s，再加 SAM mask generation（约 200 ms）和 VAE encode/decode（合计约 500 ms）。按生产 framing，这是 TTFT-bound，而不是 throughput-bound——batch 1、低并发、最小化每个阶段：

- **SAM-H is the slow one.** 1024² 上 SAM-H 约 200 ms；SAM-ViT-B 约 40 ms，质量损失很小。SAM 2（video）增加 temporal overhead；不要用于 single-image edits。
- **Skip the encode when possible.** `pipe.image_processor.preprocess(img)` 会 encode 到 latents。如果你已有上一次 generation 的 latents（iterative-edit UIs 中很常见），直接通过 `latents=...` 传入，跳过一次 VAE encode。
- **Mask dilation matters for throughput too.** 小 mask 意味着大部分 U-Net forward pass 被浪费（unmasked pixels 反正会被 clamp）。`diffusers` 的 `StableDiffusionInpaintPipeline` 不管怎样都会运行完整 U-Net；只有 9-channel proper-inpaint variants 能利用 masked compute。
- **Flux-Kontext is the 2025 answer.** 对 `(source_image, instruction)` 做 single forward pass——无需单独 mask，无需 SDEdit noise sweep。在 H100 上约 1.5 s 出一个 edit。架构启示：折叠阶段。

## 延伸阅读

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) — training-free inpainting。
- [Meng et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — SDEdit。
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) — text-instruction editing。
- [Kirillov et al. (2023). Segment Anything](https://arxiv.org/abs/2304.02643) — SAM，mask source。
- [Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) — video SAM。
- [Hertz et al. (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) — attention-level editing。
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) — 2024 tooling。
