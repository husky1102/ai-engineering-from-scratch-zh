# Video Generation

> 图像是 2-D tensor。视频是 3-D tensor。理论相同；compute 难 10-100x。OpenAI 的 Sora（2024 年 2 月）证明了这条路可行。到 2026 年，Veo 2、Kling 1.5、Runway Gen-3、Pika 2.0 和 WAN 2.2 都能从 text 生产 1080p 视频——open-weights stack（CogVideoX、HunyuanVideo、Mochi-1、WAN 2.2）大约落后 12 个月。

**类型:** Build
**语言:** Python
**先修:** Phase 8 · 07 (Latent Diffusion), Phase 7 · 09 (ViT), Phase 8 · 06 (DDPM)
**时间:** ~45 minutes

## 要解决的问题

10-second 1080p video at 24fps 是 240 frames，每帧 1920×1080×3 pixels。每个 clip 的 raw data 约 1.5 GB。Pixel-space diffusion 不可行。你需要：

1. **Spatiotemporal compression.** 一个编码 videos 而不是 frames 的 VAE，把它们编码成 spatial-temporal patches sequence。
2. **Temporal coherence.** 多秒内 frames 需要共享 content、lighting 和 object identity。Net 必须建模 motion。
3. **Compute budget.** 相同 model size 下，video training 比 image 昂贵 10-100x。
4. **Conditioning.** Text、image（first-frame）、audio 或另一个 video。多数 production models 接受四者。

解决这个问题的架构是应用到 spatiotemporal patches 上的 **Diffusion Transformer (DiT)**，用巨大 `(prompt, caption, video)` datasets 训练。Diffusion loss 与 Lesson 06 相同。

## 核心概念

![Video diffusion: patchify, DiT, decode](../assets/video-generation.svg)

### Patchify

用 3D VAE（learned spatiotemporal compression）编码 video。Latent shape 是 `[T_latent, H_latent, W_latent, C_latent]`。切成大小为 `[t_p, h_p, w_p]` 的 patches。对 Sora-style models，`t_p = 1`（per-frame patches）或 `t_p = 2`（每两帧）。一个 10-second 1080p video 压缩后约 20,000-100,000 patches。

### Spatiotemporal DiT

Transformer 处理 flat patches sequence。每个 patch 有 3D positional embedding（time + y + x）。Attention 通常会 factorize：

- **Spatial attention** 在每帧的 patches 内。
- **Temporal attention** 跨 frames、在相同 spatial location 上。
- **Full 3D attention** 贵 16-100x；只在 low resolution 或研究中使用。

### Text conditioning

用 large text encoder 做 cross-attention（Sora 使用 T5-XXL，CogVideoX-5B 使用 T5-XXL）。长 prompts 很重要——Sora 的 training set 有 GPT-generated dense re-captions，平均每个 clip 200 tokens。

### Training

在 spatiotemporal latents 上使用标准 diffusion loss（ε 或 v prediction）。Data：web video + 约 100M curated clips + synthetic text captions。Compute：即使是小型研究运行也需要 10,000+ GPU hours；Sora-scale 是 100,000+。

## 2026 production landscape

| Model | Date | Max duration | Max res | Open weights? | Notable |
|-------|------|--------------|---------|---------------|---------|
| Sora (OpenAI) | 2024-02 | 60s | 1080p | No | First model to show world simulator properties at scale |
| Sora Turbo | 2024-12 | 20s | 1080p | No | Production Sora at 5x faster inference |
| Veo 2 (Google) | 2024-12 | 8s | 4K | No | Highest quality + physics in 2025 |
| Veo 3 | 2025 Q3 | 15s | 4K | No | Native audio and stronger camera control |
| Kling 1.5 / 2.1 (Kuaishou) | 2024-2025 | 10s | 1080p | No | Best human motion in 2025 Q1 |
| Runway Gen-3 Alpha | 2024-06 | 10s | 768p | No | Professional video tools on top |
| Pika 2.0 | 2024-10 | 5s | 1080p | No | Strongest character consistency |
| CogVideoX (THUDM) | 2024 | 10s | 720p | Yes (2B, 5B) | First open 5B-scale video |
| HunyuanVideo (Tencent) | 2024-12 | 5s | 720p | Yes (13B) | Open SOTA late 2024 |
| Mochi-1 (Genmo) | 2024-10 | 5.4s | 480p | Yes (10B) | Most permissively licensed |
| WAN 2.2 (Alibaba) | 2025-07 | 5s | 720p | Yes | Strongest open model mid-2025 |

Open weights 在 image space 中更快缩小差距：到 2026 年中，HunyuanVideo + WAN 2.2 LoRAs 已经支撑大多数 open-source workflows。

## 动手实现

`code/main.py` 模拟核心 spatiotemporal DiT 思路：patchify 一个小型 synthetic video，添加 per-patch position embedding，并用 transformer-style attention over patches 对整个 sequence denoise。没有 numpy；纯 Python。我们展示即使在 1-D 中，当 adjacent-frame patches 共享 denoiser 和 position embeddings 时，temporal coherence 也会出现。

### Step 1: patchify a synthetic 1-D "video"

```python
def make_video(T_frames=8, rng=None):
    # a "video" is a sequence of 1-D values following a smooth trajectory
    base = rng.gauss(0, 1)
    return [base + 0.3 * t + rng.gauss(0, 0.1) for t in range(T_frames)]
```

### Step 2: position embedding per frame

```python
def pos_embed(t, dim):
    return sinusoidal(t, dim)
```

### Step 3: denoiser sees the whole sequence

我们 tiny net 不是独立 denoise 每帧，而是 concatenate 所有 frame values + 它们的 position embeddings，并联合预测所有 frames 的 noise。

### Step 4: temporal coherence test

训练后 sample 一个 video。测量 frame-to-frame delta。如果模型学到了 temporal structure，deltas 会小于独立 sample 每帧。

## Pitfalls

- **Independent per-frame sampling = flicker.** 如果你对每帧单独运行 image diffusion，输出会闪烁，因为每帧 noise 独立。Video diffusion 通过 attention 或 shared noise 耦合 frames 来修复。
- **Naive 3D attention = OOM.** 在 10-second 1080p latent 上 full 3D attention 是数千亿次操作。拆成 spatial + temporal。
- **Data captioning matters more than size.** Sora 相比早期工作的主要升级，是用详细得多的 captions 训练（GPT-4 re-labelled clips，约 10x）。OpenAI technical report 明确说明了这一点。
- **First-frame conditioning.** 多数 production models 也接受一张图作为 first frame。这就是 “image-to-video” mode；训练中包含这个变体。
- **Physics drift.** 长 clips（>10s）会累积微妙不一致。Sliding-window generation + keyframe anchoring 有帮助。

## 实际使用

| Use case | 2026 pick |
|----------|-----------|
| Highest-quality text-to-video, hosted | Veo 3 or Sora |
| Camera-controlled cinematic | Runway Gen-3 with motion brushes |
| Character consistency across clips | Pika 2.0 or Kling 2.1 |
| Open weights, fast fine-tune | WAN 2.2 + LoRA |
| Image-to-video | WAN 2.2-I2V, Kling 2.1 I2V, or Runway |
| Audio-to-video lip sync | Veo 3 (native audio) or a dedicated lip-sync model |
| Video editing | Runway Act-Two, Kling Motion Brush, Flux-Kontext (still-frame) |

在质量相同条件下，video 每秒成本从 2024 到 2026 下降了 20x。

## 交付成果

保存 `outputs/skill-video-brief.md`。Skill 接收 video brief（duration、aspect ratio、style、camera plan、subject consistency、audio），输出：model + hosting、prompt scaffolding（camera language、subject description、motion descriptors）、seed + reproducibility protocol，以及 frame-level QA checklist。

## 练习

1. **Easy.** 在 `code/main.py` 中比较 (a) independent per-frame sampling、(b) joint sequence sampling 的 frame-to-frame delta。报告 deltas 的 mean 和 variance。
2. **Medium.** 添加 first-frame condition：把 frame 0 固定到给定值，并 sample 其余 frames。测量 pinned value 如何传播。
3. **Hard.** 使用 HuggingFace diffusers 在本地 GPU 上运行 CogVideoX-2B。对 6-second clip 的 720p、20 inference steps 计时。Profile spatiotemporal attention 找出 bottleneck。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Video VAE | “3-D VAE” | 把 `(T, H, W, C)` 压缩为 spatiotemporal latent 的 encoder。 |
| Patches | “The tokens” | Latent 的固定大小 3-D blocks；DiT 的 input。 |
| Factorized attention | “Spatial + temporal” | 先在 space 上 attention，再在 time 上 attention；跳过 full 3-D attention。 |
| Image-to-video (I2V) | “Animate this photo” | 模型接收 image + text，并输出从该 image 开始的视频。 |
| Keyframe conditioning | “Anchor frames” | 固定特定 frames 来控制 video arc。 |
| Motion brush | “Directional hint” | 用户在图像上绘制 motion vectors 的 UI input。 |
| Re-captioning | “Dense captions” | 使用 LLM 以详细 prompts 重新标注 training clips。 |
| Flicker | “Temporal artifact” | Frame-to-frame inconsistency；通过 coupled denoising 修复。 |

## Production note: video latents are a memory-bandwidth problem

10-second 1080p clip at 24 fps 是 240 frames × 1920 × 1080 × 3 ≈ 1.5 GB raw pixels。经过 4× video VAE compression（`2 × spatial × 2 × temporal`）后，每个 request 的 latent 约 100 MB。用 spatiotemporal DiT 对它跑 30 steps、batch 1，意味着每步要通过 HBM 移动约 3 GB——bottleneck 是 memory bandwidth，不是 FLOPs。

三个 production knobs，全部来自 production-inference 文献 inference chapter：

- **TP across the DiT.** Text-to-video models 通常 ≥10B params。4 张 H100 上 TP=4 是标准；405B-class models 可用 PP=2 × TP=2。在 all-reduce wall 到来前，每步 latency 随 TP 近似线性下降。
- **Frame batching = continuous batching.** 生成时，video 概念上是由 attention 连接的一批 frames。Continuous batching（in-flight scheduling）适用：如果 architecture 允许 sliding-window generation，就可以在返回 frame `t-1` 时开始渲染 frame `t+1`。
- **Clip-level prefill cache.** 对 image-to-video，first-frame conditioning 类似 LLM 的 prompt prefill：计算一次，在 temporal decoder passes 之间复用。这本质上是 video 的 KV-cache。

## 延伸阅读

- [Brooks et al. (2024). Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) — Sora technical report。
- [Yang et al. (2024). CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072) — CogVideoX。
- [Kong et al. (2024). HunyuanVideo: A Systematic Framework for Large Video Generative Models](https://arxiv.org/abs/2412.03603) — HunyuanVideo。
- [Genmo (2024). Mochi-1 Technical Report](https://www.genmo.ai/blog/mochi) — Mochi-1。
- [Alibaba (2025). WAN 2.2](https://wanvideo.io/) — open SOTA mid-2025。
- [Ho, Salimans, Gritsenko et al. (2022). Video Diffusion Models](https://arxiv.org/abs/2204.03458) — seminal video diffusion paper。
- [Blattmann et al. (2023). Align your Latents (Video LDM)](https://arxiv.org/abs/2304.08818) — Stable Video Diffusion 的祖先。
