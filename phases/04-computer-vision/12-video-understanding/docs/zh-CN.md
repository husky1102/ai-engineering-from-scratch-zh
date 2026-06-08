# 视频理解：Temporal Modeling

> 视频是一串图像，再加上连接它们的物理过程。每个视频模型要么把时间当作额外轴（3D conv），要么当作可 attention 的序列（transformer），要么把它当作提取一次再 pooling 的特征（2D+pool）。

**类型：** Learn + Build
**语言：** Python
**先修：** Phase 4 Lesson 03 (CNNs), Phase 4 Lesson 04 (Image Classification)
**时间：** ~45 分钟

## 学习目标

- 区分三种主要视频建模方法（2D+pool、3D conv、spatio-temporal transformer），并预测它们的成本与准确率权衡
- 在 PyTorch 中实现 frame sampling、temporal pooling 和 2D+pool baseline classifier
- 解释为什么 I3D 的 “inflated” 3D kernels 能很好地从 ImageNet 权重迁移，以及 factorised (2+1)D conv 有何不同
- 读懂标准 action-recognition 数据集和指标：Kinetics-400/600、UCF101、Something-Something V2；clip level 与 video level 的 top-1 accuracy

## 要解决的问题

一个 30 秒、30 fps 的视频有 900 张图像。天真地看，video classification 就是把 image classification 跑 900 次，再做某种聚合。当动作几乎在每一帧都可见时（体育、烹饪、健身视频），这有效；当动作由运动本身定义时，它会严重失败：“把东西从左推到右”在每个静止帧里看起来都只是两个静态对象。

每个视频架构的核心问题是：时间结构在什么时候、以什么方式被建模？答案会驱动其他所有选择：计算成本、预训练策略、是否能复用 ImageNet 权重、模型训练在哪些数据集上。

本课有意比静态图像课程短。核心图像机制已经就位，视频理解主要讲时间故事：sampling、modelling 和 aggregating。

## 核心概念

### 三个架构家族

```mermaid
flowchart LR
    V["Video clip<br/>(T frames)"] --> A1["2D + pool<br/>run 2D CNN per frame,<br/>average over time"]
    V --> A2["3D conv<br/>convolve over<br/>T x H x W"]
    V --> A3["Spatio-temporal<br/>transformer<br/>attention over<br/>(t, h, w) tokens"]

    A1 --> C["Logits"]
    A2 --> C
    A3 --> C

    style A1 fill:#dbeafe,stroke:#2563eb
    style A2 fill:#fef3c7,stroke:#d97706
    style A3 fill:#dcfce7,stroke:#16a34a
```

### 2D + pool

取一个 2D CNN（ResNet、EfficientNet、ViT）。在每个采样帧上独立运行它。对逐帧 embeddings 做 average（或 max-pool，或 attention-pool）。把 pooled vector 送入 classifier。

优点：
- ImageNet 预训练可以直接迁移。
- 最容易实现。
- 便宜：T 帧 * 单图像推理成本。

缺点：
- 无法建模运动。Action = appearances 的聚合。
- Temporal pooling 对顺序不敏感；“开门”和“关门”看起来相同。

使用场景：appearance-heavy tasks、小型视频数据集上的 transfer learning、初始 baselines。

### 3D convolutions

把 2D (H, W) kernels 替换为 3D (T, H, W) kernels。网络同时在空间和时间上卷积。早期家族包括 C3D、I3D、SlowFast。

I3D 技巧：取一个预训练 2D ImageNet 模型，把每个 2D kernel 沿新的时间轴复制来 “inflate”。一个 3x3 2D conv 变成 3x3x3 3D conv。这给 3D 模型提供强预训练权重，而不是从零训练。

优点：
- 直接建模运动。
- I3D inflation 免费提供 transfer learning。

缺点：
- 比 2D 对应模型多 T/8 FLOPs（对时间 kernel 为 3、堆叠 3 次的情况）。
- Temporal kernels 很小；长程运动需要金字塔或 dual-stream approach。

使用场景：运动是信号的 action recognition（Something-Something V2、Kinetics 中 motion-heavy classes）。

### Spatio-temporal transformers

把视频 tokenise 成 space-time patches 的网格，并在所有 patch 上做 attention。TimeSformer、ViViT、Video Swin、VideoMAE。

重要 attention pattern：
- **Joint** — 对 (t, h, w) 做一个大的 attention。对 `T*H*W` 二次方；昂贵。
- **Divided** — 每个 block 两个 attentions：一个 over time，一个 over space。近似线性扩展。
- **Factorised** — time attention 与 space attention 在 blocks 间交替。

优点：
- 在所有主要 benchmark 上都是 SOTA accuracy。
- 可通过 patch inflation 从 image transformers（ViT）迁移。
- 通过 sparse attention 支持 long-context video。

缺点：
- 计算饥渴。
- 需要仔细选择 attention pattern，否则 runtime 会膨胀。

使用场景：大数据集、高保真视频理解、多模态 video+text tasks。

### Frame sampling

一个 10 秒、30 fps 的 clip 有 300 帧；把全部 300 帧喂给任何模型都是浪费。标准策略：

- **Uniform sampling** — 在 clip 中均匀选择 T 帧。2D+pool 默认做法。
- **Dense sampling** — 随机连续 T-frame window。3D convs 常用，因为运动需要相邻帧。
- **Multi-clip** — 从同一个视频采样多个 T-frame windows，分别分类，测试时平均预测。

T 通常是 8、16、32 或 64。更高 T = 更多时间信号，也意味着更多计算。

### 评估

两个层级：
- **Clip-level accuracy** — 模型看到一个 T-frame clip，报告 top-k。
- **Video-level accuracy** — 对每个视频的多个 clips 平均 clip-level predictions；更高也更稳定。

始终报告两者。一个 78% clip / 82% video 的模型高度依赖 test-time averaging；一个 80% / 81% 的模型则在每个 clip 上更稳健。

### 你会遇到的数据集

- **Kinetics-400 / 600 / 700** — 通用 action dataset。40 万 clips；YouTube URLs（许多现在已失效）。
- **Something-Something V2** — 由运动定义的动作（“moving X from left to right”）。2D+pool 无法解决。
- **UCF-101**、**HMDB-51** — 更老、更小，但仍会被报告。
- **AVA** — 时空中的 action *localisation*；比 classification 更难。

## 动手实现

### Step 1：Frame sampler

可用于帧列表（或视频 tensor）的 uniform 和 dense samplers。

```python
import numpy as np

def sample_uniform(num_frames_total, T):
    if num_frames_total <= T:
        return list(range(num_frames_total)) + [num_frames_total - 1] * (T - num_frames_total)
    step = num_frames_total / T
    return [int(i * step) for i in range(T)]


def sample_dense(num_frames_total, T, rng=None):
    rng = rng or np.random.default_rng()
    if num_frames_total <= T:
        return list(range(num_frames_total)) + [num_frames_total - 1] * (T - num_frames_total)
    start = int(rng.integers(0, num_frames_total - T + 1))
    return list(range(start, start + T))
```

两者都返回 `T` 个索引，用来 slice video tensor。

### Step 2：一个 2D+pool baseline

在每帧上运行 2D ResNet-18，average-pool features，然后分类。

```python
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class FramePool(nn.Module):
    def __init__(self, num_classes=400, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.features = nn.Sequential(*(list(backbone.children())[:-1]))  # global avg pool kept
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (N, T, 3, H, W)
        N, T = x.shape[:2]
        x = x.view(N * T, *x.shape[2:])
        feats = self.features(x).view(N, T, -1)
        pooled = feats.mean(dim=1)
        return self.head(pooled)

model = FramePool(num_classes=10)
x = torch.randn(2, 8, 3, 224, 224)
print(f"output: {model(x).shape}")
print(f"params: {sum(p.numel() for p in model.parameters()):,}")
```

1100 万参数，ImageNet 预训练，逐帧运行、平均、分类。在 appearance-heavy tasks 上，这个 baseline 通常距离合适的 3D models 只有 5-10 个点，有时甚至更好，因为它复用了更强的 ImageNet backbone。

### Step 3：I3D 风格 inflated 3D conv

通过沿新的时间轴重复权重，把单个 2D conv 变成 3D conv。

```python
def inflate_2d_to_3d(conv2d, time_kernel=3):
    out_c, in_c, kh, kw = conv2d.weight.shape
    weight_3d = conv2d.weight.data.unsqueeze(2)  # (out, in, 1, kh, kw)
    weight_3d = weight_3d.repeat(1, 1, time_kernel, 1, 1) / time_kernel
    conv3d = nn.Conv3d(in_c, out_c, kernel_size=(time_kernel, kh, kw),
                        padding=(time_kernel // 2, conv2d.padding[0], conv2d.padding[1]),
                        stride=(1, conv2d.stride[0], conv2d.stride[1]),
                        bias=False)
    conv3d.weight.data = weight_3d
    return conv3d

conv2d = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
conv3d = inflate_2d_to_3d(conv2d, time_kernel=3)
print(f"2D weight shape:  {tuple(conv2d.weight.shape)}")
print(f"3D weight shape:  {tuple(conv3d.weight.shape)}")
x = torch.randn(1, 3, 8, 56, 56)
print(f"3D output shape:  {tuple(conv3d(x).shape)}")
```

除以 `time_kernel` 能让激活幅度大致保持恒定；这对第一轮 forward 不破坏 batch-norm statistics 很重要。

### Step 4：Factorised (2+1)D conv

把 3D conv 拆成 2D（spatial）和 1D（temporal）conv。同样的 receptive field，更少参数，在一些 benchmark 上准确率更高。

```python
class Conv2Plus1D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3):
        super().__init__()
        mid_c = (in_c * out_c * kernel_size * kernel_size * kernel_size) \
                // (in_c * kernel_size * kernel_size + out_c * kernel_size)
        self.spatial = nn.Conv3d(in_c, mid_c, kernel_size=(1, kernel_size, kernel_size),
                                 padding=(0, kernel_size // 2, kernel_size // 2), bias=False)
        self.bn = nn.BatchNorm3d(mid_c)
        self.act = nn.ReLU(inplace=True)
        self.temporal = nn.Conv3d(mid_c, out_c, kernel_size=(kernel_size, 1, 1),
                                  padding=(kernel_size // 2, 0, 0), bias=False)

    def forward(self, x):
        return self.temporal(self.act(self.bn(self.spatial(x))))

c = Conv2Plus1D(3, 64)
x = torch.randn(1, 3, 8, 56, 56)
print(f"(2+1)D output: {tuple(c(x).shape)}")
```

完整 R(2+1)D network 与 ResNet-18 相同，只是把每个 3x3 conv 替换为 `Conv2Plus1D`。

## 实际使用

两个库覆盖生产视频工作：

- `torchvision.models.video` — R(2+1)D、MViT、Swin3D，带有预训练 Kinetics weights。API 与图像模型相同。
- `pytorchvideo`（Meta）— model zoo、Kinetics / SSv2 / AVA 的 data loaders、标准 transforms。

对于 Vision-Language video models（video captioning、video QA），使用 `transformers`（`VideoMAE`、`VideoLLaMA`、`InternVideo`）。

## 交付成果

本课产出：

- `outputs/prompt-video-architecture-picker.md` — 一个 prompt，会根据 appearance-vs-motion、dataset size 和 compute budget 选择 2D+pool / I3D / (2+1)D / transformer。
- `outputs/skill-frame-sampler-auditor.md` — 一个 skill，会检查 video pipeline 的 sampler 并标记常见 bug：off-by-one index、`num_frames < T` 时采样不均、缺少 aspect-preserving crop 等。

## 练习

1. **（简单）** 近似计算 T=8 时 FramePool 与 I3D 风格 3D ResNet 的 FLOPs。说明为什么 2D+pool 便宜 3-5 倍。
2. **（中等）** 生成一个合成视频数据集：随机球沿随机方向移动，并按运动方向标注（“left-to-right”、“right-to-left”、“diagonal-up”）。在其上训练 FramePool。展示它接近随机准确率，从而证明 appearance alone 不足以完成 motion tasks。
3. **（困难）** 通过把 ResNet-18 中每个 Conv2d 替换为 `Conv2Plus1D`，构建 R(2+1)D-18。从 ImageNet 预训练 ResNet-18 inflate 第一个 conv 的权重。在练习 2 的运动数据集上训练，并击败 FramePool。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| 2D + pool | “逐帧分类器” | 在每个采样帧上运行 2D CNN，跨时间 average-pool features，然后分类 |
| 3D convolution | “Spatio-temporal kernel” | 在 (T, H, W) 上卷积的 kernel；能原生建模运动 |
| Inflation | “把 2D weights 提升到 3D” | 通过沿新时间轴重复 2D conv 权重来初始化 3D conv 权重，再除以 kernel_T 以保持激活尺度 |
| (2+1)D | “Factorised conv” | 把 3D 拆成 2D spatial + 1D temporal；参数更少，中间多一个 non-linearity |
| Divided attention | “先时间再空间” | 每层有两个 attentions 的 transformer block：一个 over same frame 的 tokens，一个 over same position 的 tokens |
| Clip | “T-frame window” | T 帧组成的采样子序列；视频模型消费的单位 |
| Clip vs video accuracy | “两种 eval 设置” | Clip = 每个视频一个样本，video = 多个 sampled clips 的平均 |
| Kinetics | “视频领域的 ImageNet” | 400-700 个动作类别，30 万+ YouTube clips，标准视频预训练语料 |

## 延伸阅读

- [I3D: Quo Vadis, Action Recognition (Carreira & Zisserman, 2017)](https://arxiv.org/abs/1705.07750) — 介绍 inflation 和 Kinetics 数据集
- [R(2+1)D: A Closer Look at Spatiotemporal Convolutions (Tran et al., 2018)](https://arxiv.org/abs/1711.11248) — factorised conv，至今仍是强 baseline
- [TimeSformer: Is Space-Time Attention All You Need? (Bertasius et al., 2021)](https://arxiv.org/abs/2102.05095) — 第一个强 video transformer
- [VideoMAE (Tong et al., 2022)](https://arxiv.org/abs/2203.12602) — video 的 masked autoencoder pretraining；当前主导预训练配方
