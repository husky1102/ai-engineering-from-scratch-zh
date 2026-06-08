# Video-Language Models：Temporal Tokens 与 Grounding

> 视频不是一叠照片。一个 5 秒 clip 有因果顺序、动作动词和事件时间，这是图像模型无法表示的。Video-LLaMA（Zhang et al., 2023 年 6 月）发布了第一个带 audio-visual grounding 的 open video-LLM。VideoChat 和 Video-LLaVA 扩展了这个模式。到 2025 年，Qwen2.5-VL 的 TMRoPE 缩小了与 frontier proprietary models 的差距。每个系统都用不同方式解决 temporal tokens：per clip 的 Q-former、per frame 的 concat-pool、per token 的 TMRoPE。本课阅读这些模式，构建 uniform-vs-dynamic frame sampler，并在 temporal grounding tasks 上评估。

**类型:** Build
**语言:** Python (stdlib, frame sampler + temporal-grounding evaluator)
**先修:** Phase 12 · 08 (LLaVA-OneVision)
**时间:** ~180 minutes

## 学习目标

- 解释为什么 temporal positional encoding 会独立于 vision encoder 改变 video VLM 性能。
- 比较 uniform、dynamic-FPS 与 event-driven frame sampling 在 tokens-per-second 与 grounding accuracy 上的差异。
- 描述 Q-former-per-clip（Video-LLaMA）、pooled-per-frame（Video-LLaVA）与 M-RoPE-per-token（Qwen2.5-VL）设计。
- 说出四个视频 benchmark：VideoMME、TempCompass、EgoSchema、Video-MMMU。

## 要解决的问题

一个 1 分钟视频、30 FPS，就是 1800 帧。每帧 196 个视觉 token（ViT-B at 224），总计 352k tokens，超过任何 2024 时代 LLM context。

有三种 reduction strategies：

1. Subsample frames（依据内容使用 1-8 FPS）。
2. 激进地 pool 每帧 patch tokens（3x3 或 4x4 bilinear pool）。
3. 用 Q-former 压缩：输入一个 16-frame clip，输出 64 tokens。

每种权衡不同。Subsampling 丢失时间细节。Pooling 丢失空间细节。Q-former 两者都丢一点，但节省 tokens。

Temporal position encoding 是另一个轴：模型如何知道第 5 帧在第 6 帧之前？选项包括简单 1D temporal RoPE（Video-LLaMA）、learned temporal embeddings（Video-LLaVA）和 TMRoPE（Qwen2.5-VL，完整 3D）。

## 核心概念

### Video-LLaMA：per clip 的 Q-former + audio branch

Video-LLaMA（2023）是第一个 open video-LLM。架构：

- 16-frame clips at 2 FPS（也就是 8 秒）。
- Per-frame ViT features -> Video Q-former，跨所有 16 帧做 cross-attend -> 32 learned queries -> LLM。
- 并行 audio branch：waveform -> ImageBind audio encoder -> Audio Q-former -> 32 queries -> LLM。

优势：audio-visual joint reasoning。弱点：固定 clip length，无法做任意时间 grounding。

### VideoChat 与 Video-LLaVA

VideoChat 保留 Video-LLaMA 的思路，但去掉 audio 并简化。Video-LLaVA（Lin et al., 2023）在图像和视频帧上训练单个视觉 encoder（“alignment before projection”），得到统一表示。二者都是 frozen-CLIP-encoder + MLP + LLM。

二者都处理不了长视频。它们都是 8-16 frame systems。

### Qwen2.5-VL 与 TMRoPE

Qwen2.5-VL 引入 TMRoPE：Temporal-Modality Rotary Position Embedding。每个 patch token 携带一个 (t, h, w) position，其中 t 是实际 timestamp（不是 frame index）。

它与简单 temporal embedding 的关键差异：

- 绝对时间，而不是 index。模型看到的是“at 4.2 seconds”，不是“at frame 15”。
- Per-token rotation，而不是 per-clip。每个视觉 token 都按自身 timestamp 独立旋转。
- 兼容 dynamic FPS。如果这里按 2 FPS 采样、那里按 4 FPS 采样，TMRoPE 天然处理不均匀间隔。

TMRoPE 让“at what second does the cat jump?”这类问题成为可能。模型可以输出“at 4.2 seconds”。Video-LLaMA 只能说“early in the clip”。

### Frame sampling strategies

Uniform：在 duration 上均匀采样 N 帧。简单，但会丢失 motion peaks。

Dynamic FPS：基于 motion intensity 自适应采样。Optical flow 或 frame differencing 会为高运动片段选择更密集采样。Qwen2.5-VL 训练使用这种方式。

Event-driven：运行轻量 detector，在 action 发生处采样更多。VideoAgent 使用这种方式。

Keyframe + context：在 shot boundaries 采样，再加入几个相邻帧。用于 cinematic content。

### 每帧 pooling

在 1 FPS 且每帧 576 tokens 下，一个 5 分钟 clip 是 172,800 tokens。Qwen2.5-VL-72B 的 128k context 可以接近处理，但很昂贵。

3x3 bilinear pool 把每帧减少到 64 tokens，5 分钟变成 19,200 tokens。这是多数任务的 sweet spot。

对于 agent workflows，如果空间细节不那么重要，可以更激进地 pool（6x6 -> 16 tokens per frame）。

### 四个视频 benchmarks

- VideoMME：综合视频理解，覆盖短、中、长视频。
- TempCompass：细粒度 temporal reasoning，“before” / “after” questions。
- EgoSchema：long-horizon 第一人称视频。
- Video-MMMU：多模态多学科视频问题。

完整 video-VLM evaluation 会覆盖全部四个。它们压力测试不同轴：TempCompass 全是 ordering，EgoSchema 关注 3+ minute reasoning，VideoMME 覆盖不同 durations。

### Grounding output formats

temporal grounding 的输出格式：

- Free text：“The cat jumps around the 4-second mark.” 容易解析但不精确。
- Structured JSON：`{"event": "jump", "start": 4.1, "end": 4.3}`。Qwen2.5-VL 训练这种格式。
- Token-based：特殊 `<time>4.1</time>` tokens 与答案交错。Qwen2.5-VL 的内部格式。

Token-based 对下游使用最准确。Qwen2.5-VL 的 JSON output format 可以直接解析。

### 2026 best practice

2026 年 video VLM 的最佳实践：

- Encoder：带 M-RoPE 或 TMRoPE 的 SigLIP 2（Qwen2.5-VL）。
- Frame sampling：dynamic FPS（依据 motion 在 1-4 之间），带 max-frame cap。
- Per-frame pooling：3x3 bilinear。
- Output：带 time + event fields 的 structured JSON。
- Benchmarks：通用任务用 VideoMME + TempCompass；long-horizon 用 EgoSchema。

## 实际使用

`code/main.py` 包括：

- Uniform 与 dynamic-FPS frame samplers。
- 一个玩具 temporal-grounding evaluator：给定时间 T 处的 “ground truth” event 和 model output，在 tolerance 内评分 accuracy。
- Video-LLaMA（16 frames，Q-former）、Video-LLaVA（8 frames，MLP）、Qwen2.5-VL（dynamic FPS + TMRoPE）的比较。

## 交付成果

本课产出 `outputs/skill-video-vlm-frame-planner.md`。给定一个视频任务（monitoring、action recognition、temporal grounding、summarization），它会选择 frame sampler、pooling factor、output format 和预期 accuracy tier。

## 练习

1. 对一个 3 分钟 cooking demo，选择 uniform 还是 dynamic FPS。用 token count 说明理由。

2. 相比简单 temporal embedding table，TMRoPE 具体新增了什么？

3. 写一个 VLM 可学习发出的 temporal grounding JSON schema。包含 error cases。

4. 阅读 Video-LLaVA Section 3 中的 “Alignment Before Projection”。为什么这比训练独立的 image 与 video encoders 更好？

5. 根据 VideoMME leaderboard，截至 2026 年 top open model 与 top proprietary model 的差距是多少？其中多少可归因于 temporal encoding，多少归因于 base LLM scale？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Temporal grounding | “Time-localized answers” | VLM 为事件发生时间输出具体 timestamp range |
| TMRoPE | “Time-Multimodal RoPE” | 带绝对 timestamps 的 3D rotary position，被 Qwen2.5-VL 使用 |
| Dynamic FPS | “Motion-aware sampling” | 在高运动片段采样更多帧，在静态片段采样更少 |
| Frame pooling | “Spatial compress per frame” | 在进入 LLM 前用 bilinear interpolation 减少每帧 patches |
| Video Q-former | “Clip compressor” | 将 N frames 映射到 K learned queries 的 cross-attention bottleneck |
| VideoMME | “Video bench” | 综合短/中/长视频 benchmark，2500+ samples |

## 延伸阅读

- [Zhang et al. — Video-LLaMA (arXiv:2306.02858)](https://arxiv.org/abs/2306.02858)
- [Li et al. — VideoChat (arXiv:2305.06355)](https://arxiv.org/abs/2305.06355)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Qwen Team — Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Lin et al. — VILA-1.5 (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
