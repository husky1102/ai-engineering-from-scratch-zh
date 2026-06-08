# 任意分辨率视觉：Patch-n'-Pack 与 NaFlex

> 真实图像不是 224x224 的正方形。收据是 9:16，图表是 16:9，医学扫描可能是 4096x4096，手机截图是 9:19.5。2024 年前 VLM 的答案是把一切 resize 到固定正方形，这会丢掉让 OCR、文档理解和高分辨率场景解析有效的信号。NaViT（Google，2023）证明了可以用 block-diagonal masking 把可变分辨率 patch 打包进单个 transformer batch。Qwen2-VL 的 M-RoPE（2024）彻底去掉 absolute positional table。LLaVA-NeXT 的 AnyRes 把高分辨率图像 tile 成 base + sub-images。SigLIP 2 的 NaFlex 变体（2025）现在是开放 VLM 默认 encoder，用于让单个 checkpoint 服务所有 aspect ratio。本课端到端实现 patch-n'-pack。

**类型:** Build
**语言:** Python（stdlib，patch packer + block-diagonal mask）
**先修:** Phase 12 · 01（ViT patches），Phase 12 · 05（LLaVA）
**时间:** ~120 分钟

## 学习目标

- 把一批可变分辨率图像的 patch 打包进一个序列，并构建 block-diagonal attention mask。
- 针对给定任务在 AnyRes tiling（LLaVA-NeXT）、NaFlex（SigLIP 2）和 M-RoPE（Qwen2-VL）之间做选择。
- 在不 resize 的情况下计算 OCR、图表和照片的 token budget。
- 说出 square-resize 的三个 failure mode：文字被压扁、内容被裁切、token 浪费在 padding 上。

## 要解决的问题

Transformer 期待一个序列。Batch 是一叠长度相同的序列。如果你的图像都是 224x224，每次都会得到 196 个 patch token，不需要 padding，任务完成。训练 224，推理 224，再也不用想分辨率。

现实并不配合。文档是竖版（8.5x11 英寸，大约 2:3）。图表截图是横版（16:9）。收据又高又窄（1:3）。医学影像是 2048x2048 或更大。移动设备截图是 1170x2532（0.46:1）。

2024 年前有三种选择，每种都会失败：

1. Resize 到固定正方形（224x224 或 336x336）。挤压会扭曲文字和人脸。下采样会毁掉图表标签和 OCR 内容。直到 LLaVA-1.5 这都是标准做法。
2. Crop 到固定 aspect ratio。你会丢掉大部分图像，而且选择 crop 位置本身就是一个视觉问题。
3. Pad 到最长边。它修复了扭曲，但对竖版图像会把 50% 以上 token 浪费在 padding 上。quadratic attention cost 会花在所有这些 pad token 上。

2024-2025 年的答案是：让 transformer 吃原生分辨率下的 patch，然后弄清楚如何把异构 batch 打包成一个序列而不浪费计算。

## 核心概念

### NaViT 与 patch-n'-pack

NaViT（Dehghani et al., 2023）是证明这个方法可规模化的论文。想法很机械：

1. 对 batch 中每张图像，在选定 patch size（比如 14）下计算原生 patch grid。
2. 把每张图像的 patch flatten 成自己的可变长度序列。
3. 把所有图像的 patch 拼接成 batch 的一个长序列。
4. 构建 block-diagonal attention mask，让图像 A 的 patch 只在图像 A 内 attend。
5. 携带每个 patch 的位置信息（2D RoPE 或 fractional position embeddings）。

三张图像组成的 batch：336x336（576 token）、224x224（256 token）、448x336（768 token），会变成一个 1600-token 序列和一个 1600x1600 的 block-diagonal mask。没有 padding。没有浪费计算。Transformer 可以处理任意 aspect ratio。

NaViT 还引入了训练时的 fractional patch dropping，即跨 batch 随机丢弃 50% patch，这既正则化也加快训练。SigLIP 2 继承了它。

### AnyRes（LLaVA-NeXT）

LLaVA-NeXT 的 AnyRes 是务实替代方案。给定高分辨率图像和一个固定 encoder（CLIP 或 SigLIP at 336），把图像 tile：

1. 从预定义集合中选择最适合图像 aspect ratio 的 grid layout，例如 (1x1)、(1x2)、(2x1)、(1x3)、(3x1)、(2x2) 等。
2. 把完整图像切成该 grid；每个 tile 变成 336x336 crop。
3. 还生成一个 thumbnail：整张图像 resize 到 336x336，作为 global-context token。
4. 每个 tile 都通过冻结的 336-encoder 编码。拼接 tile token + thumbnail token。

对于 672x672 图像，2x2 grid 加 thumbnail：`4 * 576 + 576 = 2880` 个视觉 token。昂贵但有效，LLM 同时看到局部细节和全局上下文。

当 encoder 冻结且只支持一种分辨率时，AnyRes 是首选路线。它会让大图的 token 数爆炸（1344x1344 图像用 4x4 grid 是 9216 + 576 ≈ 9800 token，几乎填满 8k LLM context）。

### M-RoPE（Qwen2-VL）

Qwen2-VL 引入 Multimodal Rotary Position Embedding。不同于 NaViT 的 fractional positions 或 AnyRes 的 tile-and-thumbnail，每个 patch 携带 3D 位置（temporal, height, width）。query/key rotation 处理任意 H、W 和 temporal length。

M-RoPE 原生支持动态分辨率，无需重训。推理时输入任意 HxW 图像，patch embedder 产生 `H/14 x W/14` 个 token，每个 token 获得其 `(t=0, r=row, c=col)` 位置，RoPE 用正确频率旋转 attention，然后完成。Qwen2.5-VL 和 Qwen3-VL 延续了这一点。InternVL3 的 V2PE 是相同思路，只是按模态使用可变 encoding。

不同于 AnyRes，M-RoPE 在原生分辨率下是 `O(H x W / P^2)` 个 token，没有乘法式 tile overhead。不同于 NaViT，它仍然预期每次 forward 是单张图像。跨分辨率 batch 仍需要在上层使用 patch-n'-pack。

### NaFlex（SigLIP 2）

NaFlex 是 SigLIP 2 checkpoint 的 native-flex 模式。单个模型在推理时服务多种 sequence length（256、729、1024 token）。内部在训练时使用 NaViT 风格 patch-n'-pack，并给每个 patch 使用 absolute fractional positions。卖点是：一个 checkpoint，推理时按任务选择 token budget。

语义任务（classification、retrieval）用 256 token。OCR 或图表理解用 1024 token。无需重训。

### Packing mask

Block-diagonal mask 是大多数实现踩坑的地方。对于覆盖图像 `i=0..B-1`、长度为 `n_i` 的 packed sequence（总长度 `N_total`），shape 为 `(N_total, N_total)` 的 mask `M` 在两个索引都落在同一图像 block 内时为 1，否则为 0。可以从 cumulative length list 构建：

```text
offsets = [0, n_0, n_0+n_1, ..., N_total]
M[i, j] = 1 iff there exists b where offsets[b] <= i < offsets[b+1] and offsets[b] <= j < offsets[b+1]
```

PyTorch 中可以用 `torch.block_diag` 一行完成，或显式 gather。FlashAttention 的 variable-length path（`cu_seqlens`）完全跳过 mask，直接用 cumulative-length tensor 在序列内部 attend，对于典型 batch 比 dense mask 快约 10 倍。

### Token budgets

按任务选择策略：

- OCR / documents：1024-4096 token。SigLIP 2 NaFlex at 1024，或 AnyRes 3x3 + thumbnail。
- Charts and UI：384-448 原生下 729-1024 token。Qwen2.5-VL dynamic resolution 加 max pixels cap。
- Natural photos：256-576 token 就够。下游 LLM 已能看到足够信息。只在内容密度高处为 token 付费。
- Video：spatial pooling 后每帧 64-128 token，2-8 FPS。Lesson 12.17 会覆盖。

2026 年的生产规则：选择每任务 max-pixels cap，以原生 aspect ratio 编码直到该 cap，打包 batch，并跳过 padding。Qwen2.5-VL 正是通过 `min_pixels` 和 `max_pixels` 暴露这个旋钮。

## 实际使用

`code/main.py` 为一批具有整数像素坐标的异构图像实现 patch-n'-pack。它会：

- 接收一组 (H, W) 图像尺寸。
- 在 patch size 14 下计算每张图像的 patch sequence length。
- 把它们打包成一个总长度为 `sum(n_i)` 的序列。
- 构建 block-diagonal attention mask（为了清晰使用 dense）。
- 比较 packed cost 与 square-resize、AnyRes tiling 的成本。
- 为混合 batch（收据、图表、截图、照片）打印 token budget 表。

运行它。输出的数字就是每个 2026 开放 VLM 都使用 patch-n'-pack 的原因。

## 交付成果

本课产出 `outputs/skill-resolution-budget-planner.md`。给定一个混合 aspect-ratio 工作负载（OCR、图表、照片、视频帧）和总 token budget，它会选择合适策略（NaFlex、AnyRes、M-RoPE 或 fixed-square），并发出每请求配置。当你为产品规划 VLM 时使用这个 skill，它能避免悄无声息的 10 倍 token 膨胀毁掉延迟预算。

## 练习

1. 一张收据是 600x1500（1:2.5）。patch size 14 下有多少原生分辨率 token？square-resize 到 336 后有多少？实践中哪一种损失更多 OCR 准确率？

2. 为四张图像长度分别为 256、576、729、1024 的 batch 构建 block-diagonal mask。验证 attention matrix 是 2585x2585，并且恰好有 `256^2 + 576^2 + 729^2 + 1024^2` 个非零元素。

3. 对一张 1792x896 图像、patch 14，比较：（a）square-resize 到 336 再编码，（b）AnyRes 2x1 + thumbnail，（c）M-RoPE 原生分辨率。哪个 token 最少？哪个保留最多细节？

4. 实现 fractional patch dropping：给定 packed sequence，均匀随机丢弃 50% token，并相应更新 block-diagonal mask。测量 mask 稀疏度变化。

5. 阅读 Qwen2-VL paper Section 3.2（arXiv:2409.12191）。用两句话描述 `min_pixels` 和 `max_pixels` 控制什么，以及为什么上下界都重要。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| Patch-n'-pack | “NaViT-style packing” | 把来自不同图像的可变长度 patch 序列拼接进一个 batch 维度 |
| Block-diagonal mask | “Packing mask” | 把每张图像的 patch 限制为只 attend 自己，而不是 pack 中邻居的 attention mask |
| AnyRes | “LLaVA-NeXT tiling” | 把高分辨率图像拆成固定尺寸 tile 网格加全局 thumbnail；每个 tile 用固定 encoder 编码 |
| NaFlex | “SigLIP 2 native-flex” | 单个 SigLIP 2 checkpoint 在推理时服务 256/729/1024-token budget，无需重训 |
| M-RoPE | “Multimodal RoPE” | 处理任意 H、W、T 而不需要 position table 的 3D rotary position encoding（time, row, column） |
| cu_seqlens | “FlashAttention packing” | FlashAttention varlen path 用来替代 dense block-diagonal mask 的 cumulative-length tensor |
| min_pixels / max_pixels | “Resolution bounds” | Qwen2.5-VL 的每请求旋钮，用于限制过小或过大输入上的 token count |
| Visual token budget | “How many tokens per image” | 每张图像产生的 patch token 粗略数量；决定 LLM prompt budget 和 attention cost |

## 延伸阅读

- [Dehghani et al. — Patch n' Pack: NaViT (arXiv:2307.06304)](https://arxiv.org/abs/2307.06304)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Laurençon et al. — What matters when building vision-language models? (Idefics2, arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
