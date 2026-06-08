# 从零实现卷积

> 卷积就是一个微小的 dense layer，你把它滑过图像，并在每个位置共享同一组权重。

**类型:** Build
**语言:** Python
**先修:** Phase 3 (Deep Learning Core), Phase 4 Lesson 01 (Image Fundamentals)
**时间:** ~75 minutes

## 学习目标

- 只用 NumPy 从零实现 2D convolution，包括 nested-loop 版本和 vectorised `im2col` 版本
- 针对任意 input size、kernel size、padding、stride 组合计算输出空间尺寸，并说明 `(H - K + 2P) / S + 1` 公式为什么成立
- 手工设计 kernel（edge、blur、sharpen、Sobel），并解释每个 kernel 为什么会产生对应的 activation pattern
- 把 convolution 堆叠成 feature extractor，并把 stack depth 与 receptive field size 连接起来

## 要解决的问题

在 224x224 RGB 图像上使用 fully connected layer，每个 neuron 需要 224 * 224 * 3 = 150,528 个输入权重。一个只有 1,000 个 unit 的 hidden layer 已经有 1.5 亿个参数，而这还没学到任何有用东西。更糟的是，这一层完全不知道左上角的狗和右下角的狗是同一种 pattern。它把每个 pixel position 都当作独立对象，而这对图像来说恰好是错的：把猫平移三个像素，不应该迫使网络重新学习这个概念。

图像模型需要两个性质：**translation equivariance**（输入平移时输出也随之平移）和 **parameter sharing**（同一个 feature detector 在所有位置运行）。Dense layer 两者都没有。Convolution 免费给你两者。

Convolution 并不是为 deep learning 发明的。JPEG compression、Photoshop 里的 Gaussian blur、工业视觉中的 edge detection，以及所有发布过的 audio filter，用的都是同一种操作。CNN 从 2012 到 2020 主导 ImageNet 的原因，是 convolution 正好是这类数据的正确先验：邻近数值相互相关，同一种 pattern 可以出现在任何位置。

## 核心概念

### 一个 kernel，滑动起来

2D convolution 会取一个叫 kernel（或 filter）的小权重矩阵，让它滑过输入，并在每个位置计算逐元素乘积之和。这个和会变成一个 output pixel。

```mermaid
flowchart LR
    subgraph IN["Input (H x W)"]
        direction LR
        I1["5 x 5 image"]
    end
    subgraph K["Kernel (3 x 3)"]
        K1["learned<br/>weights"]
    end
    subgraph OUT["Output (H-2 x W-2)"]
        O1["3 x 3 map"]
    end
    I1 --> |"slide kernel<br/>compute dot product<br/>at each position"| O1
    K1 --> O1

    style IN fill:#dbeafe,stroke:#2563eb
    style K fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

一个具体的 3x3 kernel 作用在 5x5 input 上的例子（no padding，stride 1）：

```text
Input X (5 x 5):                Kernel W (3 x 3):

  1  2  0  1  2                   1  0 -1
  0  1  3  1  0                   2  0 -2
  2  1  0  2  1                   1  0 -1
  1  0  2  1  3
  2  1  1  0  1

The kernel slides across every valid 3 x 3 window. Output Y is 3 x 3:

 Y[0,0] = sum( W * X[0:3, 0:3] )
 Y[0,1] = sum( W * X[0:3, 1:4] )
 Y[0,2] = sum( W * X[0:3, 2:5] )
 Y[1,0] = sum( W * X[1:4, 0:3] )
 ... and so on
```

这一个公式，即 **shared weights、locality、sliding window**，就是完整思想。其他一切都是 bookkeeping。

### 输出尺寸公式

给定输入空间尺寸 `H`、kernel size `K`、padding `P`、stride `S`：

```text
H_out = floor( (H - K + 2P) / S ) + 1
```

记住它。你会在每个 architecture 里计算它几十次。

| 场景 | H | K | P | S | H_out |
|----------|---|---|---|---|-------|
| Valid conv, no padding | 32 | 3 | 0 | 1 | 30 |
| Same conv（保持尺寸） | 32 | 3 | 1 | 1 | 32 |
| Downsample by 2 | 32 | 3 | 1 | 2 | 16 |
| Pool 2x2 | 32 | 2 | 0 | 2 | 16 |
| Large receptive field | 32 | 7 | 3 | 2 | 16 |

“Same padding” 的意思是选择 P，使得 S == 1 时 H_out == H。对于奇数 K，P = (K - 1) / 2。也正因如此，3x3 kernel 占据主导地位：它是仍然拥有中心点的最小奇数 kernel。

### Padding

没有 padding 时，每次 convolution 都会缩小 feature map。堆叠 20 层后，你的 224x224 图像会变成 184x184，这会浪费边界上的计算，并让需要 matching shape 的 residual connection 更难处理。

```text
Zero padding (P = 1) on a 5 x 5 input:

  0  0  0  0  0  0  0
  0  1  2  0  1  2  0
  0  0  1  3  1  0  0
  0  2  1  0  2  1  0       Now the kernel can centre on pixel
  0  1  0  2  1  3  0       (0, 0) and still have three rows and
  0  2  1  1  0  1  0       three columns of values to multiply.
  0  0  0  0  0  0  0
```

实践中会遇到的模式：`zero`（最常见）、`reflect`（镜像边缘，避免 generative model 中的硬边界）、`replicate`（复制边缘）、`circular`（环绕，toroidal problem 中使用）。

### Stride

Stride 是滑动的步长。`stride=1` 是默认值。`stride=2` 会把空间维度减半，是 CNN 内部不使用单独 pooling layer 而进行 downsample 的经典方式。每种现代 architecture（ResNet、ConvNeXt、MobileNet）都会在某些位置用 strided conv 替代 max-pool。

```text
Stride 1 on a 5 x 5 input, 3 x 3 kernel:

  starts: (0,0) (0,1) (0,2)        -> output row 0
          (1,0) (1,1) (1,2)        -> output row 1
          (2,0) (2,1) (2,2)        -> output row 2

  Output: 3 x 3

Stride 2 on the same input:

  starts: (0,0) (0,2)              -> output row 0
          (2,0) (2,2)              -> output row 1

  Output: 2 x 2
```

### 多个 input channel

真实图像有三个 channel。RGB input 上的 3x3 convolution 实际上是一个 3x3x3 volume：每个 input channel 各有一个 3x3 slice。在每个空间位置上，你会跨全部三个 slice 相乘求和，并加上 bias。

```text
Input:   (C_in,  H,  W)        3 x 5 x 5
Kernel:  (C_in,  K,  K)        3 x 3 x 3 (one kernel)
Output:  (1,     H', W')       2D map

For a layer that produces C_out output channels, you stack C_out kernels:

Weight:  (C_out, C_in, K, K)   e.g. 64 x 3 x 3 x 3
Output:  (C_out, H', W')       64 x 3 x 3

Parameter count: C_out * C_in * K * K + C_out   (the + C_out is biases)
```

最后一行是你规划模型时会计算的内容。3-channel input 上的 64-channel 3x3 conv 有 `64 * 3 * 3 * 3 + 64 = 1,792` 个参数。很便宜。

### im2col 技巧

Nested loop 易读但很慢。GPU 想要大型 matrix multiply。技巧是：把输入中每个 receptive-field window 展平成一个大矩阵的一列，把 kernel 展平成一行，于是整个 convolution 变成一次 matmul。

```mermaid
flowchart LR
    X["Input<br/>(C_in, H, W)"] --> IM2COL["im2col<br/>(extract patches)"]
    IM2COL --> COLS["Cols matrix<br/>(C_in * K * K, H_out * W_out)"]
    W["Weight<br/>(C_out, C_in, K, K)"] --> FLAT["Flatten<br/>(C_out, C_in * K * K)"]
    FLAT --> MM["matmul"]
    COLS --> MM
    MM --> OUT["Output<br/>(C_out, H_out * W_out)<br/>reshape to (C_out, H_out, W_out)"]

    style X fill:#dbeafe,stroke:#2563eb
    style W fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

每个生产 conv implementation 都是这个思路的某种变体，再加上 cache-tiling trick（direct conv、Winograd、大 kernel 用 FFT conv）。理解 im2col，你就理解了核心。

### Receptive field

单个 3x3 conv 会看 9 个 input pixel。堆叠两个 3x3 conv，第二层中的 neuron 会看 5x5 个 input pixel。三个 3x3 conv 给出 7x7。一般来说：

```text
RF after L stacked K x K convs (stride 1) = 1 + L * (K - 1)

With strides:   RF grows multiplicatively with stride along each layer.
```

“一路 3x3” 有效（VGG、ResNet、ConvNeXt）的根本原因是，两个 3x3 conv 能看到和一个 5x5 conv 相同的输入区域，但参数更少，并且中间多一次 non-linearity。

## 动手实现

### Step 1: Pad 一个 array

从最小 primitive 开始：编写一个在 H x W array 周围补零的函数。

```python
import numpy as np

def pad2d(x, p):
    if p == 0:
        return x
    h, w = x.shape[-2:]
    out = np.zeros(x.shape[:-2] + (h + 2 * p, w + 2 * p), dtype=x.dtype)
    out[..., p:p + h, p:p + w] = x
    return out

x = np.arange(9).reshape(3, 3)
print(x)
print()
print(pad2d(x, 1))
```

Trailing-axes trick `x.shape[:-2]` 意味着同一个函数无需修改，就能作用在 `(H, W)`、`(C, H, W)` 或 `(N, C, H, W)` 上。

### Step 2: 使用 nested loop 实现 2D convolution

这是 reference implementation：慢，但毫不含糊。原则上，这就是 `torch.nn.functional.conv2d` 做的事。

```python
def conv2d_naive(x, w, b=None, stride=1, padding=0):
    c_in, h, w_in = x.shape
    c_out, c_in_w, kh, kw = w.shape
    assert c_in == c_in_w

    x_pad = pad2d(x, padding)
    h_out = (h + 2 * padding - kh) // stride + 1
    w_out = (w_in + 2 * padding - kw) // stride + 1

    out = np.zeros((c_out, h_out, w_out), dtype=np.float32)
    for oc in range(c_out):
        for i in range(h_out):
            for j in range(w_out):
                hs = i * stride
                ws = j * stride
                patch = x_pad[:, hs:hs + kh, ws:ws + kw]
                out[oc, i, j] = np.sum(patch * w[oc])
        if b is not None:
            out[oc] += b[oc]
    return out
```

四重 nested loop（output channel、row、column，再加上对 C_in、kh、kw 的隐式求和）。这是你检查每个更快实现时要对照的 ground truth。

### Step 3: 用手工设计的 kernel 验证

构建一个 vertical Sobel kernel，将它应用到 synthetic step image 上，观察 vertical edge 亮起来。

```python
def synthetic_step_image():
    img = np.zeros((1, 16, 16), dtype=np.float32)
    img[:, :, 8:] = 1.0
    return img

sobel_x = np.array([
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]]
], dtype=np.float32)[None]

x = synthetic_step_image()
y = conv2d_naive(x, sobel_x, padding=1)
print(y[0].round(1))
```

预期 column 7 上有较大的正值（亮度从左到右增加），其他位置都是零。这个 print 是你确认数学正确的 sanity check。

### Step 4: im2col

把输入中每个 kernel-sized window 转成矩阵的一列。对于 `C_in=3, K=3`，每一列有 27 个数字。

```python
def im2col(x, kh, kw, stride=1, padding=0):
    c_in, h, w = x.shape
    x_pad = pad2d(x, padding)
    h_out = (h + 2 * padding - kh) // stride + 1
    w_out = (w + 2 * padding - kw) // stride + 1

    cols = np.zeros((c_in * kh * kw, h_out * w_out), dtype=x.dtype)
    col = 0
    for i in range(h_out):
        for j in range(w_out):
            hs = i * stride
            ws = j * stride
            patch = x_pad[:, hs:hs + kh, ws:ws + kw]
            cols[:, col] = patch.reshape(-1)
            col += 1
    return cols, h_out, w_out
```

它仍然是 Python loop，但现在重活会由一次 vectorised matmul 完成。

### Step 5: 通过 im2col + matmul 实现快速 conv

用一次 matrix multiplication 替换 quadruple loop。

```python
def conv2d_im2col(x, w, b=None, stride=1, padding=0):
    c_out, c_in, kh, kw = w.shape
    cols, h_out, w_out = im2col(x, kh, kw, stride, padding)
    w_flat = w.reshape(c_out, -1)
    out = w_flat @ cols
    if b is not None:
        out += b[:, None]
    return out.reshape(c_out, h_out, w_out)
```

正确性检查：运行两个实现并比较。

```python
rng = np.random.default_rng(0)
x = rng.normal(0, 1, (3, 16, 16)).astype(np.float32)
w = rng.normal(0, 1, (8, 3, 3, 3)).astype(np.float32)
b = rng.normal(0, 1, (8,)).astype(np.float32)

y_naive = conv2d_naive(x, w, b, padding=1)
y_im2col = conv2d_im2col(x, w, b, padding=1)

print(f"max abs diff: {np.max(np.abs(y_naive - y_im2col)):.2e}")
```

`max abs diff` 应该在 `1e-5` 左右；差异来自 floating-point accumulation order，不是 bug。

### Step 6: 一组手工设计的 kernel

五个 filter，展示训练前单个 conv layer 已经能表达什么。

```python
KERNELS = {
    "identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32),
    "blur_3x3": np.ones((3, 3), dtype=np.float32) / 9.0,
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
    "sobel_x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
    "sobel_y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
}

def apply_kernel(img2d, kernel):
    x = img2d[None].astype(np.float32)
    w = kernel[None, None]
    return conv2d_im2col(x, w, padding=1)[0]
```

作用到任意 grayscale image 上时，blur 会柔化，sharpen 会让边缘更清脆，Sobel-x 会点亮 vertical edge，Sobel-y 会点亮 horizontal edge。这些正是 AlexNet 和 VGG 中第一层 trained conv layer 最终学到的 pattern，因为一个好的图像模型无论后续任务是什么，都需要 edge detector 和 blob detector。

## 实际使用

PyTorch 的 `nn.Conv2d` 用 autograd、CUDA kernel 和 cuDNN optimisation 包装了同一个操作。Shape semantics 完全一致。

```python
import torch
import torch.nn as nn

conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
print(conv)
print(f"weight shape: {tuple(conv.weight.shape)}   # (C_out, C_in, K, K)")
print(f"bias shape:   {tuple(conv.bias.shape)}")
print(f"param count:  {sum(p.numel() for p in conv.parameters())}")

x = torch.randn(8, 3, 224, 224)
y = conv(x)
print(f"\ninput  shape: {tuple(x.shape)}")
print(f"output shape: {tuple(y.shape)}")
```

把 `padding=1` 换成 `padding=0`，输出会降到 222x222。把 `stride=1` 换成 `stride=2`，输出会降到 112x112。就是你上面记住的同一个公式。

## 交付成果

本课产出：

- `outputs/prompt-cnn-architect.md` — 一个 prompt：给定 input size、parameter budget 和 target receptive field，设计一组 `Conv2d` layer，并在每一步选对 K/S/P。
- `outputs/skill-conv-shape-calculator.md` — 一个 skill：逐层遍历 network spec，并返回每个 block 的 output shape、receptive field 和 parameter count。

## 练习

1. **(Easy)** 给定一个 128x128 grayscale input 和一组 `[Conv3x3(s=1,p=1), Conv3x3(s=2,p=1), Conv3x3(s=1,p=1), Conv3x3(s=2,p=1)]`，手算每层的 output spatial size 和 receptive field。用 dummy conv 组成的 PyTorch `nn.Sequential` 验证。
2. **(Medium)** 扩展 `conv2d_naive` 和 `conv2d_im2col`，让它们接受 `groups` argument。证明 `groups=C_in=C_out` 会复现 depthwise convolution，并且其 parameter count 是 `C * K * K`，而不是 `C * C * K * K`。
3. **(Hard)** 手写 `conv2d_im2col` 的 backward pass：给定 output gradient，计算 `x` 和 `w` 的 gradient。在相同 input 和 weight 上用 `torch.autograd.grad` 验证。技巧是：im2col 的 gradient 是 `col2im`，而且必须累加重叠 window。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| Convolution | “滑动 filter” | 在每个空间位置用 shared weights 应用的 learnable dot product；数学上是 cross-correlation，但所有人都叫它 convolution |
| Kernel / filter | “feature detector” | shape 为 (C_in, K, K) 的小权重 tensor，它与 input window 的 dot product 产生一个 output pixel |
| Stride | “跳多远” | 连续 kernel placement 之间的步长；stride 2 会把每个空间维度减半 |
| Padding | “边缘上的零” | 添加到 input 周围的额外值，让 kernel 可以以 border pixel 为中心；`same` padding 保持 output size 等于 input size |
| Receptive field | “neuron 看见多少” | 某个 output activation 依赖的 original input patch，会随 depth 和 stride 增长 |
| im2col | “GEMM 技巧” | 把每个 receptive window 重排成 column，让 convolution 变成一次大型 matrix multiply，是每个快速 conv kernel 的核心 |
| Depthwise conv | “每个 channel 一个 kernel” | `groups == C_in` 的 conv，每个 output channel 只从匹配的 input channel 计算得到；是 MobileNet 和 ConvNeXt 的支柱 |
| Translation equivariance | “输入平移，输出平移” | 将 input 平移 k 个 pixel 会让 output 也平移 k 个 pixel 的性质；shared weights 免费带来它 |

## 延伸阅读

- [A guide to convolution arithmetic for deep learning (Dumoulin & Visin, 2016)](https://arxiv.org/abs/1603.07285) — padding/stride/dilation 的权威图解，几乎每门课程都在悄悄借鉴
- [CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/convolutional-networks/) — canonical lecture notes，包含最初的 im2col 解释
- [The Annotated ConvNet (fast.ai)](https://nbviewer.org/github/fastai/fastbook/blob/master/13_convolutions.ipynb) — 从 manual convolution 走到 trained digit classifier 的 notebook
- [Receptive Field Arithmetic for CNNs (Dang Ha The Hien)](https://distill.pub/2019/computing-receptive-fields/) — 关于 receptive field calculation 的 paper-quality 互动讲解
