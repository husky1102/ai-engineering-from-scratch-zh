# Vision Transformers 与 Patch-Token 原语

> 在任何多模态发生之前，一张图像都必须先变成 transformer 能吃的 token 序列。2020 年的 ViT 论文用 16x16 像素 patches、线性投影和位置 embedding 回答了这个问题。五年之后，每个 2026 年 frontier model（Claude Opus 4.7 的 2576px native、Gemini 3.1 Pro、Qwen3.5-Omni）仍然从这里开始：encoder 从 ViT 变成 DINOv2 再到 SigLIP 2，加入了 register tokens，位置方案变成 2D-RoPE，但这个原语保留下来了。本课端到端阅读 patch-token pipeline，并用 stdlib Python 构建它，让 Phase 12 剩余课程对“visual tokens”有一个具体心智模型。

**类型：** 学习
**语言：** Python（stdlib，patch tokenizer + geometry calculator）
**先修：** Phase 7（Transformers），Phase 4（Computer Vision）
**时间：** ~120 分钟

## 学习目标

- 将一张 HxWx3 图像转换为带正确位置编码的 patch token 序列。
- 对给定（patch size、resolution、hidden dim、depth）的 ViT 计算序列长度、参数量和 FLOPs。
- 说出让 ViT 从 2020 年研究原型走向 2026 年生产系统的三项升级：self-supervised pretraining（DINO / MAE）、register tokens 和 native-resolution packing。
- 为下游任务在 CLS pooling、mean pooling 和 register tokens 之间做选择。

## 要解决的问题

Transformers 处理的是向量序列。文本已经是序列（bytes 或 tokens）。图像是带三个颜色通道的二维像素网格，不是序列。如果你把每个像素都摊平，一张 224x224 RGB 图像会变成 150,528 个 tokens，而在这个长度上做 self-attention 根本不可行（序列长度二次方）。

2020 年前的方法会在前面接一个 CNN feature extractor：ResNet 产生一个 7x7、每个向量 2048 维的 feature map，再把这 49 个 tokens 喂给 transformer。这样可行，但会继承 CNN 的偏置（translation equivariance、local receptive fields），也会失去 transformer 对规模的胃口。

Dosovitskiy 等人（2020）提出了一个直接的问题：如果跳过 CNN 会怎样？把图像切成固定大小的 patches（比如 16x16 像素），把每个 patch 线性投影成一个向量，加上 positional embedding，然后喂给 vanilla transformer。当时这像是异端：没有 convolution 的视觉。只要数据足够多（JFT-300M，后来是 LAION），它就在 ImageNet 上击败 ResNet，并持续改进。

到 2026 年，ViT 原语已经是无可争议的基础。每个 open-weights VLM 的 vision tower 都是某个后代（DINOv2、SigLIP 2、CLIP、EVA、InternViT）。问题不再是“我们该用 patches 吗？”，而是“用什么 patch size、什么 resolution schedule、什么 pretraining objective、什么 positional encoding”。

## 核心概念

### Patches as tokens

给定一张形状为 `(H, W, 3)` 的图像 `x` 和 patch size `P`，你把图像切成 `(H/P) x (W/P)` 个不重叠 patches 的网格。每个 patch 是一个 `P x P x 3` 的像素立方体。把每个立方体摊平成一个 `3 P^2` 向量。应用一个形状为 `(3 P^2, D)` 的共享线性投影 `W_E`，把每个 patch 映射到模型 hidden dimension `D`。

对 ViT-B/16 的标准配置：
- Resolution 224，patch size 16 → grid 14x14 → 196 patch tokens。
- 每个 patch 是 `16 x 16 x 3 = 768` 个像素值，投影到 `D = 768`。
- 添加一个可学习 `[CLS]` token → sequence length 197。

patch projection 在数学上等同于一个 2D convolution，其 kernel size 为 `P`、stride 为 `P`，输出通道数为 `D`。生产代码实际上也是这样实现的：`nn.Conv2d(3, D, kernel_size=P, stride=P)`。“linear projection” 说法是概念视角；kernel 视角更高效。

### Positional embeddings

Patches 本身没有内在顺序，transformer 看到的是一袋向量。早期 ViT 添加可学习 1D positional embedding（每个位置一个 768 维向量，一共 197 个）。它能工作，但会把模型绑定到训练分辨率：推理时如果改变 grid，就必须插值位置表。

现代视觉 backbone 使用 2D-RoPE（Qwen2-VL 的 M-RoPE、SigLIP 2 的默认方案）或 factorized 2D positions。2D-RoPE 会根据 patch 的（row, column）索引旋转 query 和 key 向量，因此模型可以从旋转角度推断相对二维位置。没有 position table。模型能在推理时处理任意 grid size。

### CLS token、pooled output 与 register tokens

图像级表示是什么？三种选择共存：

1. `[CLS]` token。把一个可学习向量前置到 patch sequence。经过所有 transformer blocks 后，CLS token 的 hidden state 就是图像表示。继承自 BERT。原始 ViT、CLIP 使用它。
2. Mean pool。平均 patch tokens 的输出 hidden states。SigLIP、DINOv2 和大多数现代 VLM 使用它。
3. Register tokens。Darcet 等人（2023）观察到，没有显式 sink token 训练的 ViT 会产生高范数“artifact” patches，劫持 self-attention。添加 4-16 个可学习 register tokens 会吸收这部分负载，并提升 dense-prediction 质量（segmentation、depth）。DINOv2 和 SigLIP 2 都带 registers。

这个选择会影响下游任务。CLS 适合分类。对于把 patch tokens 喂进 LLM 的 VLM，你会完全跳过 pooling，因为每个 patch 都会变成 LLM input token。Registers 会在 handoff 前被丢弃（它们是脚手架，不是内容）。

### Pretraining：supervised、contrastive、masked、self-distilled

2020 年 ViT 使用 JFT-300M 上的 supervised classification 预训练。随后很快被替代：

- CLIP（2021）：在 400M 图文对上做 contrastive image-text。Lesson 12.02。
- MAE（2021，He et al.）：mask 75% 的 patches，重构像素。Self-supervised，适用于纯图像。
- DINO（2021）/ DINOv2（2023）：student-teacher self-distillation，无标签、无 captions。2023 年的 DINOv2 ViT-g/14 是最强 purely-visual backbone，也是“dense features” 用例的默认选择。
- SigLIP / SigLIP 2（2023，2025）：使用 sigmoid loss 和 NaFlex 处理 native aspect ratio 的 CLIP。2026 年 open VLMs（Qwen、Idefics2、LLaVA-OneVision）中的主流 vision tower。

你的 pretraining 选择决定 backbone 擅长什么：CLIP/SigLIP 用于与文本做语义匹配，DINOv2 用于 dense visual features，MAE 可作为下游 finetuning 的起点。

### Scaling laws

ViT scaling（Zhai et al. 2022）确立了 ViT 的质量会遵循模型大小、数据大小和计算量中的可预测规律。在固定计算量下：
- 更大的模型 + 更多数据 → 更好质量。
- Patch size 是 sequence length 与 fidelity 之间的杠杆。Patch 14（DINOv2/SigLIP SO400m 的典型值）比 patch 16 给每张图像更多 tokens；对 OCR 和 dense tasks 更好，对速度更差。
- Resolution 是另一个大杠杆。从 224 到 384 再到 512 几乎总有帮助，但 FLOPs 成本是二次方。

ViT-g/14（1B params，patch 14，resolution 224 → 256 tokens）和 SigLIP SO400m/14（400M params，patch 14）是 2026 年 open VLMs 的两种主力 encoder。

### ViT 的参数量

完整计算在 `code/main.py` 中。以 224 分辨率的 ViT-B/16 为例：

```text
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

在加载 checkpoint 之前，先这样粗估每个 ViT。backbone 大小决定任何下游 VLM 的 VRAM 下限。

### 2026 生产配置

2026 年大多数 open VLM 随附的 encoder 是 native resolution（NaFlex）下的 SigLIP 2 SO400m/14。它有：
- 400M 参数。
- Patch size 14，默认 resolution 384 → 每张图像 729 个 patch tokens。
- 图像级任务用 mean pool；VQA 中全部 729 个 patches 流入 LLM。
- 4 个 register tokens，在 LLM handoff 前丢弃。
- 使用 image-level scaling 的 2D-RoPE，以支持 native aspect ratio。

这个配置中的每个决策都能追溯到一篇你可以阅读的论文。

## 实际使用

`code/main.py` 是一个 patch tokenizer 和 geometry calculator。它接收（image H、W、patch P、hidden D、depth L），并报告：

- patching 后的 grid shape 和 sequence length。
- 一个合成 8x8 pixel toy image 的 token sequence（逐步走过 flatten + project 路径）。
- 按 patch embed、position embed、transformer blocks 和 head 拆分的参数量。
- 目标分辨率下每次 forward pass 的 FLOPs。
- ViT-B/16 @ 224、ViT-L/14 @ 336、DINOv2 ViT-g/14 @ 224、SigLIP SO400m/14 @ 384 的对比表。

运行它。把参数量与公开数字对上。调整 patch size 和 resolution，体会 token-count 成本。

## 交付成果

本课产出 `outputs/skill-patch-geometry-reader.md`。给定一个 ViT config（patch size、resolution、hidden dim、depth），它会生成 token-count、parameter-count 和 VRAM 估计，并附上理由。每当你为 VLM 选择 vision backbone 时使用这个 skill，它能防止“tokens 爆炸，把我的 LLM context 填满了”的意外。

## 练习

1. 计算 Qwen2.5-VL 在 native 1280x720 输入、patch size 14 下的 patch-token sequence length。它与 CLS-only representation 相比如何？

2. 一帧 1080p（1920x1080）在 patch 14 下会产生多少 tokens？30 FPS 的 5 分钟视频总共有多少 visual tokens？哪种成本节省最多：pooling、frame sampling，还是 token merging？

3. 用纯 Python 实现 patch tokens 上的 mean pooling。验证对 DINOv2 输出的 196 个 tokens 做 mean-pool，是否匹配当你请求 pooled embedding 时模型 `forward` 返回的结果。

4. 阅读 "Vision Transformers Need Registers"（arXiv:2309.16588）第 3 节。用两句话描述 registers 吸收了什么 artifact，以及为什么这对下游 dense prediction 很重要。

5. 修改 `code/main.py` 以支持 patch-n'-pack：给定一组不同分辨率的图像，生成一个 packed sequence 和 block-diagonal attention mask。到 Lesson 12.06 时用它做验证。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Patch | “16x16 pixel square” | 输入图像中的固定大小不重叠区域；会变成一个 token |
| Patch embedding | “Linear projection” | 将摊平 patch pixels 映射到 D-dim vectors 的共享学习矩阵（或 stride=P 的 Conv2d） |
| CLS token | “Class token” | 前置的可学习向量，其最终 hidden state 表示整张图像；2026 年可选 |
| Register token | “Sink token” | 额外的可学习 tokens，用来吸收 ViT 在预训练中形成的高范数 attention artifacts |
| Position embedding | “Positional info” | 让序列具有顺序感知的 per-position vector 或 rotation；2D-RoPE 是现代默认 |
| Grid | “Patch grid” | 给定 resolution 和 patch size 下的 (H/P) x (W/P) 二维 patch 数组 |
| NaFlex | “Native flexible resolution” | SigLIP 2 特性：单个模型无需重训即可服务多种 aspect ratios 和 resolutions |
| Backbone | “Vision tower” | 预训练图像 encoder，其 patch-token 输出会在 VLM 中喂给 LLM |
| Pooling | “Image-level summary” | 将 patch tokens 转成单个向量的策略：CLS、mean、attention pool 或 register-based |
| Patch 14 vs 16 | “Finer vs coarser grid” | Patch 14 每张图像产生更多 tokens，对 OCR fidelity 更好但更慢；patch 16 是经典默认值 |

## 延伸阅读

- [Dosovitskiy et al. — An Image is Worth 16x16 Words (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929)：原始 ViT。
- [He et al. — Masked Autoencoders Are Scalable Vision Learners (arXiv:2111.06377)](https://arxiv.org/abs/2111.06377)：MAE，self-supervised pretraining。
- [Oquab et al. — DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193)：大规模 self-distillation，无 labels。
- [Darcet et al. — Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588)：register tokens 和 artifact analysis。
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786)：2026 年默认 vision tower。
- [Zhai et al. — Scaling Vision Transformers (arXiv:2106.04560)](https://arxiv.org/abs/2106.04560)：经验 scaling laws。
