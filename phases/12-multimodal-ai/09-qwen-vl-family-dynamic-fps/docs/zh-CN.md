# Qwen-VL 家族与 Dynamic-FPS Video

> Qwen-VL 家族，也就是 Qwen-VL（2023）、Qwen2-VL（2024）、Qwen2.5-VL（2025）、Qwen3-VL（2025），是 2026 年最有影响力的开放 vision-language model 谱系。每一代都做出一个决定性的架构押注，开放生态其余部分会在十二个月内复制：通过 M-RoPE 实现原生动态分辨率，通过绝对时间对齐实现 dynamic-FPS sampling，ViT 中的 window attention，以及 structured agent output formats。到 Qwen3-VL，recipe 已经稳定：2D-RoPE-ViT encoder，原生 aspect-ratio 输入，MLP projector 接入大型 Qwen3 language base，训练阶段把 OCR、grounding 和 agent behavior 作为一等目标。本课按时间顺序阅读这个家族，让你理解每个旋钮为什么在那里。

**类型:** Learn
**语言:** Python（stdlib，M-RoPE encoder + dynamic-FPS sampler）
**先修:** Phase 12 · 06（patch-n'-pack）
**时间:** ~120 分钟

## 学习目标

- 计算 M-RoPE 的三轴 rotation（temporal、height、width），并解释为什么三者都需要。
- 为视频选择 dynamic-FPS sampling 策略，并推理 tokens-per-second 与 event-detection accuracy 的关系。
- 按顺序说出四代 Qwen-VL 升级以及每一代启用了什么。
- 接线 Qwen2.5-VL-style JSON agent output format，并从 VLM response 中解析 structured tool calls。

## 要解决的问题

Qwen-VL 于 2023 年 8 月发布，直接回应 LLaVA-1.5 和 BLIP-2。Qwen 团队瞄准的差距有三方面：resolution、video 和 structured output。

Resolution：LLaVA-1.5 运行在 336x336。对照片可以，对中文发票或密集电子表格截图没用。Qwen-VL 的第一个创新是 448x448 和 grounded bounding-box output，让模型可以指向物体。

Video：Video-LLaMA 堆叠逐帧 encoder 并把它们喂给 LLM。它适合短片段，不适合时间轴就是信号的多分钟视频。Qwen 团队想要一个理解时间的单一 encoder。

Structured output：LLaVA 输出 free-form text。Agent 需要 JSON。Qwen-VL 在显式 JSON output format 上训练，包括把 bounding-box coordinate 作为文本输出。

每一代 Qwen-VL 都扩展这三条轴之一。

## 核心概念

### Qwen-VL（2023 年 8 月）

第一代：OpenCLIP ViT-bigG/14 作为 encoder（2.5B 参数）、LLama-compatible Q-Former（1-step，256 queries）、Qwen-7B base。贡献：

- 448x448 resolution（当时开放 VLM 的 SOTA）。
- Grounding：在带显式 coordinate-token output 的 image-text pair 上训练。“The cat is at <box>(112, 204), (280, 344)</box>”。
- 从一开始就做中文 + 英文 multilingual training。

当时 benchmark：英语上与 GPT-4V 竞争，中文上占优。grounding supervision 才是真正头条。

### Qwen2-VL（2024 年 9 月）：M-RoPE 与原生分辨率

Qwen2-VL 用原生 dynamic-resolution ViT encoder 替换 fixed-resolution + Q-Former stack。关键变化：

- 原生动态分辨率。ViT 接受任意可被 28 整除的 HxW（patch 14 + 2x spatial merge）。1120x672 图像（40x24 merged patches）产生 960 个 visual token。无 resize，无 tiling，无 thumbnail。
- M-RoPE（Multimodal RoPE）。每个 token 携带 3D 位置（t, h, w），而不是 1D。图像用 t=0，视频用 `t = frame_index`。RoPE 为每个轴用一个频率旋转 query/key。没有 positional embedding table。
- MLP projector。丢掉 Q-Former；在 merged patch token 上使用 2 层 MLP。
- Dynamic FPS 视频。默认以 1-2 FPS 采样视频，但模型接受任意帧数。

结果：Qwen2-VL-7B 在多个多模态 benchmark 上追平 GPT-4o，并在 DocVQA 上击败它（94.5 vs 88.4）。架构变化是决定性一步。

### Qwen2.5-VL（2025 年 2 月）：dynamic FPS + absolute time

Qwen2.5-VL 的大变化是视频。Dynamic FPS 不只是“需要时采样更多帧”。论文形式化了：

- Absolute time tokens。不使用位置索引（frame 0, 1, 2...），而是使用真实 timestamp。“At 0:04, the cat jumps.” 模型看到与 frame token 交错的 `<time>0.04</time>` token。
- Dynamic FPS。慢镜头用 1 FPS，动作场景用 4+ FPS。用户或训练器选择；M-RoPE 适配。
- ViT 中的 window attention。Spatial attention 是 windowed（block 内局部）以提高 throughput；每隔几层加入 global attention。
- 显式 JSON output format。在 tool-call 数据上训练：`"{\"tool\": \"click\", \"coords\": [380, 220]}"`。开箱即 agent-ready。
- MRoPE-v2 scaling。位置按最大输入尺寸缩放，所以 10 分钟视频不会耗尽频率范围。

Benchmark：Qwen2.5-VL-72B 在大多数视频 benchmark 上超过 GPT-4o，在文档上匹配 Gemini 2.0，并创下开放模型 GUI grounding SOTA（ScreenSpot：84% accuracy vs GPT-4o 的 38%）。

### Qwen3-VL（2025 年 11 月）

Qwen3-VL 是巩固而非重造的增量升级：更大的 LLM backbone（Qwen3-72B）、扩展训练数据、改进 OCR、通过 Qwen3 “thinking mode” 实现更强 reasoning。ViT 和 M-RoPE 保持不变。论文关注数据与训练改进，而不是 architecture。

谱系结论：到 2025 年，Qwen-VL architecture 已经稳定。后续世代扩展 compute 和 data，而不是 primitives。

### M-RoPE 数学

经典 RoPE 按位置 `m` 旋转维度 `d` 的 query `q`，使用成对坐标：

```text
q_rot[2i]   = q[2i]   * cos(m * theta_i) - q[2i+1] * sin(m * theta_i)
q_rot[2i+1] = q[2i]   * sin(m * theta_i) + q[2i+1] * cos(m * theta_i)
theta_i     = 10000^(-2i/d)
```

M-RoPE 把 hidden dim 分成三段。假设 `d = 96`。给 temporal 32 dim，height 32 dim，width 32 dim。每段按自己的轴位置旋转。位置为 (t=5, h=10, w=20) 的 patch 会在三段上分别应用 `R_t(5)`、`R_h(10)`、`R_w(20)`。

Text token 使用 `t = text_index, h = 0, w = 0`（或某种 normalized choice），保持兼容。Video frame 使用 `t = frame_time, h = row, w = col`。单图使用 `t = 0`。

好处是：一个 position encoding 处理 text、image 和 video，无需 branching code 或不同 position table。

### Dynamic-FPS sampling logic

给定时长 `T` 秒的视频和目标 token budget `B`：

1. 计算你能承受的最大 FPS：`fps_max = B / (T * tokens_per_frame)`。
2. 从 `{1, 2, 4, 8}` 中选择满足 `fps <= fps_max` 的目标 FPS。
3. 如果 motion 高（optical-flow heuristic 或明确用户请求），选择更高 FPS。如果 motion 低，选择更低。
4. 以选定 FPS 均匀采样；在帧之间插入 `<time>t</time>` token。

Qwen2.5-VL 在训练中隐式学习此逻辑；推理时用户通过 `fps` 参数控制。60 秒动作序列，4 FPS，每帧 81 token，总计 19440 token，可以放进 32k context。

### Structured agent output

Qwen2.5-VL 的 agent training 显式瞄准 structured tool call：

```text
{
  "tool": "mouse_click",
  "coords": [1024, 512],
  "button": "left",
  "modifier": null
}
```

解析是确定性的：对模型输出运行 `JSON.parse`。相比 free-form “click at (1024, 512)” 需要 regex 和歧义处理。这个转变解释了为什么 Qwen2.5-VL 的 ScreenSpot 分数从 Qwen2-VL 的 55% 跳到 84%。

## 实际使用

`code/main.py` 实现：

- 为混合 text、image patches 和 video frames 的 packed sequence 计算 M-RoPE position。
- Dynamic-FPS sampler：给定 (duration, budget, motion_level)，选择 FPS 并输出 frame timestamps。
- 一个 toy Qwen2.5-VL JSON-output parser，用于处理带 coordinate field 的 tool-call response。

运行它，然后把 fixed-FPS 换成 dynamic-FPS 处理 5 分钟视频时感受差异。

## 交付成果

本课产出 `outputs/skill-qwen-vl-pipeline-designer.md`。给定一个视频任务（monitoring、agent、action recognition、accessibility），它会输出 Qwen2.5-VL 配置（frame budget、FPS strategy、window-attention flag、agent-output mode）和 latency estimate。每当你为视频产品部署 Qwen-VL-family model 时使用它。

## 练习

1. 对 hidden 48（每 band 16，base theta 10000）中位置 (t=3, h=5, w=7) 的 patch 计算 M-RoPE rotations。展示每个 band 前三对的 rotation angles。

2. 10 分钟安防摄像头录像，1 FPS 会产生多少帧？384 分辨率、3x pool 下有多少总 token？Qwen2.5-VL 默认 32k context 能处理吗？

3. 为 30 秒网球回合、30 秒菜谱演示、30 秒 UI-agent 录屏分别选择 FPS。用 dynamic-FPS logic 说明理由。

4. Qwen2.5-VL 完全丢掉 Q-Former。为什么简单 MLP 在 2025 年有效，而在 2023 年不行？（提示：data scale 和 encoder quality。）

5. 把三个 Qwen2.5-VL JSON tool-call output 解析成 Python dict。malformed JSON 会失败在哪里，Qwen cookbook 推荐什么 recovery strategy？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| M-RoPE | “Multimodal RoPE” | hidden dim 中带 temporal、height、width band 的 3D rotary position embedding |
| Dynamic FPS | “Smart sampling” | 按视频 motion、duration 和 token budget 为每个视频选择 frame sampling rate |
| Absolute time token | “Timestamp token” | 交错在序列中的 `<time>t</time>`，让模型看到真实秒数而不是 frame index |
| Window attention | “Local attention” | 为速度把 spatial self-attention 限制在小 window；周期性加入 global attention |
| Structured agent output | “JSON mode” | 教 VLM 输出带 coords 和 tool names 的可解析 JSON 的训练监督 |
| min_pixels / max_pixels | “Resolution bounds” | Qwen2.5-VL 每请求控制项，限制总 pixel count，因而限制 token count |
| Grounding | “Point-at-it” | 以文本 token 输出 bounding-box coordinates；从 Qwen-VL v1 开始使用 |

## 延伸阅读

- [Bai et al. — Qwen-VL (arXiv:2308.12966)](https://arxiv.org/abs/2308.12966)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Qwen Team — Qwen3-VL (arXiv:2511.21631)](https://arxiv.org/abs/2511.21631)
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
