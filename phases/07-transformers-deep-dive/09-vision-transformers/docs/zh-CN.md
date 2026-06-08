# Vision Transformers (ViT)

> 图像是 patch 网格。句子是 token 网格。同一个 transformer 两者都能吃下。

**类型:** Build
**语言:** Python
**先修:** Phase 7 · 05 (Full Transformer), Phase 4 · 03 (CNNs), Phase 4 · 14 (Vision Transformers intro)
**时间:** ~45 minutes

## 要解决的问题

2020 年之前，computer vision 基本意味着 convolutions。ImageNet、COCO 和 detection benchmarks 上的每个 SOTA 都使用 CNN backbone。Transformers 属于语言。

Dosovitskiy et al. (2020) 的 “An Image is Worth 16x16 Words” 证明你可以完全丢掉 convolutions。把图像切成固定大小 patches，把每个 patch 线性投影成 embedding，把这个 sequence 喂给 vanilla transformer encoder。在足够规模下（ImageNet-21k pretraining 或更大），ViT 能匹配或超过 ResNet-based models。

ViT 是 2026 年更广泛模式的开端：一种架构，多种模态。Whisper tokenize audio。ViT tokenize images。Robotics 使用 action tokens。Video 使用 pixel tokens。Transformer 不关心——给它 sequence，它就学习。

到 2026 年，ViT 及其后代（DeiT、Swin、DINOv2、ViT-22B、SAM 3）占据了多数 vision。CNNs 在 edge devices 和 latency-sensitive tasks 上仍然赢。其他地方的 stack 里几乎都有某种 ViT。

## 核心概念

![Image → patches → tokens → transformer](../assets/vit.svg)

### Step 1 — patchify

把 `H × W × C` 图像切成 `N × (P·P·C)` 的 flat patches sequence。典型设置：`224 × 224` 图像，`16 × 16` patches → 196 个 patches，每个 768 个值。

```text
image (224, 224, 3) → 14 × 14 grid of 16x16x3 patches → 196 vectors of length 768
```

Patch size 是杠杆。更小 patches = 更多 tokens、更好 resolution、二次 attention cost。更大 patches = 更粗、更便宜。

### Step 2 — linear embedding

一个 learned matrix 把每个 flat patch 投影到 `d_model`。这等价于 kernel size `P`、stride `P` 的 convolution。在 PyTorch 里它真的就是 `nn.Conv2d(C, d_model, kernel_size=P, stride=P)`——两行实现。

### Step 3 — prepend `[CLS]` token, add positional embeddings

- 前置一个 learnable `[CLS]` token。它的 final hidden state 是用于 classification 的 image representation。
- 加 learnable positional embeddings（ViT-original）或 sinusoidal 2D（后续变体）。
- 2024+ 年 RoPE 扩展到 2D position，有时不再需要显式 embeddings。

### Step 4 — standard transformer encoder

堆叠 L 个 `LayerNorm → Self-Attention → + → LayerNorm → MLP → +` blocks。与 BERT 相同。没有 vision-specific layers。这是论文在教学上的 punchline。

### Step 5 — head

Classification：取 `[CLS]` hidden state → linear → softmax。对 DINOv2 或 SAM，丢弃 `[CLS]`，直接使用 patch embeddings。

### 重要变体

| Model | Year | Change |
|-------|------|--------|
| ViT | 2020 | The original. Fixed patch size, full global attention. |
| DeiT | 2021 | Distillation; trainable on ImageNet-1k only. |
| Swin | 2021 | Hierarchical with shifted windows. Fixed sub-quadratic cost. |
| DINOv2 | 2023 | Self-supervised (no labels). Best general vision features. |
| ViT-22B | 2023 | 22B params; scaling laws apply. |
| SigLIP | 2023 | ViT + language pair, sigmoid contrastive loss. |
| SAM 3 | 2025 | Segment anything; ViT-Large + promptable mask decoder. |

### 为什么它花了一些时间

ViT 需要*大量*数据才能匹配 CNNs，因为它没有 CNN 的 inductive biases（translation invariance、locality）。如果没有 >100M labeled images 或强 self-supervised pretraining，在匹配 compute 下 CNNs 仍会赢。DeiT 在 2021 年用 distillation tricks 修复了这一点；DINOv2 在 2023 年用 self-supervision 永久修复了它。

## 动手实现

见 `code/main.py`。纯 stdlib patchify + linear embedding + sanity checks。不训练——任何现实规模的 ViT 都需要 PyTorch 和数小时 GPU 时间。

### Step 1: fake image

把 24 × 24 RGB 图像表示为 `(R, G, B)` tuples 的 row 列表。我们使用 6×6 patches → 16 patches，每个 108-d embedding vector。

### Step 2: patchify

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

Raster order：按 grid 的 row-major 顺序。每个 ViT 都使用这种 ordering。

### Step 3: linear embed

用随机 `(patch_flat_size, d_model)` matrix 乘每个 flat patch。验证 prepending `[CLS]` 之后 output shape 是 `(N_patches + 1, d_model)`。

### Step 4: count parameters for a realistic ViT

打印 ViT-Base 的参数量：12 layers、12 heads、d=768、patch=16。与 ResNet-50（约 25M）比较。ViT-Base 约 86M。ViT-Large 约 307M。ViT-Huge 约 632M。

## 实际使用

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patches
cls_emb = out[:, 0]                       # image representation
```

**DINOv2 embeddings 是 2026 年 image features 的默认选择。** Freeze backbone，训练一个 tiny head。适用于 classification、retrieval、detection、captioning。Meta 的 DINOv2 checkpoints 在每个非文本 vision task 上都优于 CLIP。

**Patch-size picking.** 小模型使用 16×16（ViT-B/16）。Dense prediction（segmentation）使用 8×8 或 14×14（SAM、DINOv2）。非常大的模型使用 14×14。

## 交付成果

见 `outputs/skill-vit-configurator.md`。这个 skill 会根据 dataset size、resolution 和 compute budget，为新的 vision task 选择 ViT variant 和 patch size。

## 练习

1. **Easy.** 运行 `code/main.py`。验证 patches 数量等于 `(H/P) * (W/P)`，flat patch dimension 等于 `P*P*C`。
2. **Medium.** 实现 2D sinusoidal positional embeddings——对每个 patch 的 `row` 和 `col` 使用两个独立 sinusoidal codes，然后 concatenate。把它们喂给 tiny PyTorch ViT，并在 CIFAR-10 上比较与 learnable positional embeddings 的 accuracy。
3. **Hard.** 构建一个 3-layer ViT（PyTorch），用 4×4 patches 在 1,000 张 MNIST images 上训练。测量 test accuracy。现在在同样 1,000 张 images 上加入 DINOv2 pretraining（简化版：只训练 encoder 从 masked patches 预测 patch embeddings）。Accuracy 是否提升？

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Patch | “The vision-transformer token” | 图像中 `P × P × C` 区域的像素值 flat vector。 |
| Patchify | “Chop + flatten” | 把图像切成 non-overlapping patches，并把每个 patch flatten 成 vector。 |
| `[CLS]` token | “The image summary” | 前置的 learnable token；其最终 embedding 是 image representation。 |
| Inductive bias | “模型的假设” | ViT 的 priors 少于 CNNs；需要更多数据来补上差距。 |
| DINOv2 | “Self-supervised ViT” | 不使用 labels 训练，使用 image augmentation + momentum teacher。2026 年最佳通用 image features。 |
| SigLIP | “CLIP 的继任者” | 用 sigmoid contrastive loss 训练的 ViT + text encoder；匹配 compute 下优于 CLIP。 |
| Swin | “Windowed ViT” | 采用 local attention + shifted windows 的 hierarchical ViT；sub-quadratic。 |
| Register tokens | “2023 trick” | 少量额外 learnable tokens，用来吸收 attention sinks；改善 DINOv2 features。 |

## 延伸阅读

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — ViT 论文。
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT。
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin。
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2。
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — DINOv2 的 register-token fix。
