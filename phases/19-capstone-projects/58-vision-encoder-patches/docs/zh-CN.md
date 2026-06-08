# 视觉编码器 Patches

> 读取 pixels 的 vision model 需要一个面向 pixels 的 tokenizer。Patch embedding 就是这个 tokenizer。把 image 切成方块网格，flatten 每个方块，通过一层 linear layer 投影，然后加入 2D position signal，让 transformer 知道每个方块在原图中的位置。

**类型:** Build
**语言:** Python
**先修:** Phase 19 lessons 30-37 (Track B foundations)
**时间:** ~90 minutes

## 学习目标

- 将一张 image token 化为固定长度的 patch embeddings 序列。
- 实现基于 `Conv2d` 的 patch projection，使其与 unfold-then-linear 的数学等价。
- 构建 deterministic 2D sinusoidal position embedding，让 token order 编码 spatial position。
- 在 synthetic fixture 上验证 patch count、embedding shape，以及 `Conv2d`/unfold equivalence。

## 要解决的问题

transformer 吃的是 vector 序列。image 是 3-channel grid。把每个 pixel 当成 token 会让 sequence length 爆炸：一张 224x224 RGB image 是 150,528 个 tokens，12-layer transformer 在 attention 上承受不起。把 image 当成一个巨大的 flat vector 又会丢掉 locality，而 attention layer 无法从中恢复 locality。encoder front end 的工作，是把 pixel grid 压缩成几百个 tokens，每个 token 概括一个方形区域。

Patch embedding 用一次 linear projection 解决这个问题。把 224x224 image 切成 16x16 patches，会得到 14x14 的 196 个 patches。每个 patch 从 `(3, 16, 16) = 768` 个 pixel values flatten 成一个 vector，然后 linear layer 把它映射到 model 的 hidden dimension。transformer 看到的是 196 个 dimension 为 `hidden`（常见为 768）的 tokens，再加一个 CLS token。这是网络其余部分可以处理的序列。

## 核心概念

```mermaid
flowchart LR
  Image[224x224x3 image] --> Cut[cut into 16x16 patches]
  Cut --> Grid[14x14 grid of patches]
  Grid --> Flatten[flatten each patch]
  Flatten --> Proj[linear projection]
  Proj --> Tokens[196 tokens of dim hidden]
  Tokens --> Pos[add 2D sinusoidal position]
  Pos --> Out[final token sequence]
```

### 为什么用 patches，而不是 pixels

Attention 对 sequence length 是二次复杂度。196-token sequence 每个 head 每层需要 `196 * 196 = 38,416` 个 attention scores；150,528-token sequence 需要 `150,528 * 150,528 = 22.6 billion`。Patches 让 attention compute 减少 590,000 倍，并且单个 16x16 区域已经携带了高级 vision tasks 所需的足够信号。代价是丢失一个 patch 内部的细粒度空间细节，这也是为什么当 fine localization 重要时，下游 multimodal stacks 通常会运行第二条 high-resolution branch。

### 为什么 linear projection 足够

每个 patch 都被视为独立 vector。projection 学到一个 basis：edge detectors、color filters、simple textures。单层 linear layer 很小（ViT-Base 中 `768 * 768 = 589,824` parameters），训练也很快。确实存在更深的 convolutional stems（“hybrid” ViT），但 flat linear projection 是标准做法，大多数现代 open-weight encoders 都使用这个 exact shape。

### `Conv2d` 技巧

没有 padding 的 `Conv2d(in_channels=3, out_channels=hidden, kernel_size=patch_size, stride=patch_size)` 与 unfold-then-linear 产生相同的数值结果，因为每个 output position 都把 patch pixels 与一个 filter 做 dot product。convolution 就是 patch projection；大多数 production codebases 都这样实现，因为它在 GPU 上更快，并且少一次 reshape。

### Position embeddings

tokens 经过 projection 后不携带顺序。2D sinusoidal embedding 给每个 token 一个固定信号，编码它的 `(row, col)` 位置。embedding dimension 的一半用多个 frequencies 的 sin/cos 编码 row position，另一半编码 column position。encoding 是 deterministic 的，因此你可以在不重新训练的情况下切换 resolution，并且它能干净地插值到 model 训练时没见过的 grids。

| Component | Shape | Parameters |
|-----------|-------|------------|
| Patch projection (`Conv2d`) | `(hidden, 3, patch, patch)` | `3 * P * P * hidden + hidden` |
| Position embedding (fixed) | `(num_patches, hidden)` | 0 (computed, not learned) |
| CLS token (learned) | `(1, hidden)` | `hidden` |

对 224 resolution 下的 ViT-Base/16：projection 中有 590,592 个 parameters，CLS token 中有 768 个，sinusoidal position 为零。下一课（59）会在这个 front end 顶部堆叠一个 12-layer transformer。

### 用 equivalence 做 sanity check

patch step 有两种写法：`Conv2d` projection 和显式 unfold-then-linear。给定相同 weights，它们必须产生相同输出。否则 unfold math 就是错的，encoder 的其余部分也建在沙地上。本课 tests 会检查这种 equivalence。

## 动手实现

`code/main.py` 实现：

- `PatchEmbed`，一个包装 `Conv2d` 作为 patch projection 的 `nn.Module`。
- `sinusoidal_2d(grid_h, grid_w, dim)`，一个 stateless function，用来构建 2D position table。
- `VisionFrontEnd`，把 patch embedding、CLS prepend 和 position addition 组合为一次 forward pass。
- `synthesize_image(seed)` helper，它从 `numpy.random` 构建 deterministic 224x224x3 fixture。
- 一个 demo：把一张 fixture image 送入 front end，并打印 output shape、CLS token norm，以及 position embedding 的一行。

运行：

```bash
python3 code/main.py
```

输出：224x224 fixture 会被 tokenized 为 shape `(1, 197, 768)` 的序列。第一个 token 是 CLS；接下来的 196 个是 patch tokens。position embedding norms 在同一行内保持均匀，这是 sinusoidal signature。

## 实际使用

同一个 patch front end 出现在每个现代 vision-language model 中：CLIP ViT-L/14、SigLIP、DINOv2、Qwen-VL family 和 InternVL stack 都从 `Conv2d` patch projection 加 position signal 开始。不同 families 的差异在下游（CLS vs no-CLS pooling、register tokens、patch sizes 14 与 16 的变化、通过 interpolated positions 实现 dynamic resolution）。本课中的 frontend 是所有这些模型站立其上的 substrate。

## 测试

`code/test_main.py` 覆盖：

- patch count 匹配 `(image_size / patch_size) ** 2`
- output shape 匹配 `(batch, num_patches + 1, hidden)`
- 在 small fixture 上，`Conv2d` projection 等于 manual unfold-then-linear
- sinusoidal position table 在多次调用之间 deterministic
- CLS token 在 batch dim 上 broadcast，且没有 leakage

运行：

```bash
python3 -m unittest code/test_main.py
```

## 练习

1. 将 sinusoidal position 替换为 learned `nn.Parameter`，并比较 tiny synthetic classification task 上 first-epoch loss。固定 resolution 下 learned positions 会赢；训练后改变 resolution 时 sinusoidal 会赢。

2. 将 `Conv2d` 换成显式 `nn.Unfold` 加 `nn.Linear`，并断言 outputs 在 float tolerance 内匹配。同一套数学，两种写法。

3. 添加对 non-square patch sizes 的支持（例如 wide-aspect inputs 使用 32x16），并验证 position table 可以处理 non-square grids。

4. 在 batch sizes 1、8、64 下 profile patch step。patch projection 很少是瓶颈；下游 attention layers 才是主导。

5. 在 4-class synthetic shape dataset（circles、squares、triangles、stars）上把 front end 训练成 frozen feature extractor。CLS token output 应该能够线性分离。

## 关键术语

| Term | What it means |
|------|---------------|
| Patch | image 的一个方形子区域，通常是 14x14 或 16x16 |
| Patch embedding | 将一个 flattened patch linear projection 到 hidden dim |
| Sequence length | patch tokenization 后的 token 数量，通常还要加上 CLS |
| Sinusoidal position | 编码 2D grid coordinates 的固定 sin/cos signal |
| CLS token | prepended 到 sequence 前端、作为 pooling head 的 learned vector |

## 延伸阅读

- An Image is Worth 16x16 Words (ViT, 2021)，关于最初的 patch-embed framing。
- Attention Is All You Need (2017)，关于这里适配到 2D 的 sinusoidal position formula。
- DINOv2 paper，关于 register tokens，这是你可以作为 exercise 6 添加的扩展。
