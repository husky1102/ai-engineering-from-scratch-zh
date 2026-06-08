# CLIP 与对比式视觉-语言预训练

> OpenAI 的 CLIP（2021）证明了一个足以驱动接下来五年的想法：只用嘈杂的网页图像-标题对和一个 contrastive loss，把 image encoder 与 text encoder 对齐到同一个向量空间。没有 supervised labels。400M 对。得到的 embedding space 可以做 zero-shot classification、image-text retrieval，并作为 vision tower 接入每个 2026 年 VLM。SigLIP 2（2025）用 sigmoid 替代 softmax，并以更低成本扩展到 CLIP 之外。本课从 InfoNCE 到 sigmoid pairwise loss 走一遍数学，并用 stdlib Python 构建训练步骤。

**类型：** 构建
**语言：** Python（stdlib，InfoNCE + sigmoid loss implementations）
**先修：** Phase 12 · 01（ViT patches），Phase 7（Transformers）
**时间：** ~180 分钟

## 学习目标

- 从 mutual information 推导 InfoNCE loss，并实现数值稳定的向量化版本。
- 解释为什么 sigmoid pairwise loss（SigLIP）能扩展到 batch 32768+，且不需要 softmax 所要求的 all-gather 开销。
- 通过构造 text templates（`a photo of a {class}`）并对 cosine similarity 取 argmax，运行 zero-shot ImageNet classification。
- 说出 CLIP / SigLIP 预训练给你的四个杠杆：batch size、temperature、prompt template、data quality。

## 要解决的问题

CLIP 之前的视觉是 supervised 的。收集带标签数据集（ImageNet：1.2M 图像、1000 类），训练 CNN，然后发布。标签昂贵，标签会偏向标注者能达成一致的东西，而且没有 finetuning 时，标签不会迁移到新任务。

图像-标题网页拥有十亿级以上免费但松散标注的 pair。一张 golden retriever 的照片配上 alt text “my dog Max in the park”，携带了监督信号：文本描述图像。问题是：你能把它转成有用的训练吗？

CLIP 的回答是：把图像-标题 pair 当作匹配任务。给定一个包含 N 张图像和 N 条标题的 batch，学习把每张图像与自己的标题匹配，同时区分 N-1 个干扰项。监督信号是“这两样东西属于一起；另外 N-1 个不属于”。没有 class labels。没有人工标注。只有一个 contrastive loss。

得到的 embedding space 能做的远超过 CLIP 的训练目标。ImageNet zero-shot 能工作，是因为 “a photo of a cat” 会嵌入到接近猫图片的位置，即使那些图片从未被显式标成 cat。这就是催生每个 2026 年 VLM 的赌注。

## 核心概念

### 双 encoder

CLIP 有两座 tower：

- Image encoder `f`：ViT 或 ResNet，每张图像输出一个 D-dim 向量。
- Text encoder `g`：小型 transformer，每条 caption 输出一个 D-dim 向量。

两座 tower 都把输出归一化到单位长度。因此 similarity 是 `cos(f(x), g(y)) = f(x)^T g(y)`，因为二者都是 unit-norm。

对包含 N 个（image, caption）pairs 的 batch，构建形状为 `(N, N)` 的 similarity matrix `S`：

```text
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

其中 `tau` 是学习到的 temperature（CLIP 初始化为 0.07；在 log-space 中学习）。

### InfoNCE loss

CLIP 在行和列上使用对称 cross-entropy：

```text
loss_i2t = CE(S, labels=identity)     # each image's positive is its own caption
loss_t2i = CE(S^T, labels=identity)   # each caption's positive is its own image
loss = (loss_i2t + loss_t2i) / 2
```

这就是 InfoNCE。CE 中的 softmax 会迫使每张图像与自己的 caption 匹配得比 batch 中其他 caption 更强。“negatives” 就是所有其他 batch items。更大的 batch = 更多 negatives = 更强信号。CLIP 以 batch 32k 训练；规模很重要。

### Temperature

`tau` 控制 softmax 的锐度。低 tau → 锐利分布，具有 hard negative mining 效果。高 tau → 更柔和，所有样本都有贡献。CLIP 学习 log(1/tau)，并进行 clipping 防止崩塌。SigLIP 2 固定初始 tau，并使用一个 learned bias。

### 为什么 sigmoid 扩展性更好（SigLIP）

Softmax 需要整个 similarity matrix 同步。在分布式训练中，你必须 all-gather 每个 embedding 到每个 replica，然后做 softmax。通信量相对于 world size 是二次方。

SigLIP 用 element-wise sigmoid 替代 softmax：对每一对 `(i, j)`，loss 都是“这是不是匹配 pair？”的二分类。正类标签在对角线上，其余全部是负类。loss 是：

```text
L = -1/N sum over (i, j) [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]
```

如果 `i == j`，则 `y_ij = 1`，否则为 0。每一对的 loss 都是独立的。不需要 all-gather。每块 GPU 计算自己的 local block 并求和。SigLIP 2 能以低成本扩展到 batch 32k-512k，而 CLIP 会需要成比例增加通信。

### Zero-shot classification

给定 N 个 class names，为每个 class 构建一个 text template：

```text
"a photo of a {class}"
```

用 text encoder 嵌入每个 template。用 image encoder 嵌入你的图像。Argmax cosine similarity = predicted class。不在目标类别上训练。

Prompt templates 很重要。CLIP 原论文对每个 class 使用 80 个 templates（plain、artistic、photo、painting 等），并平均 embeddings。ImageNet 提升 3 个点。现代用法通常选择一两个 templates。

### Linear probes 与 finetuning

Zero-shot 是 baseline。Linear probe（在 frozen CLIP features 上训练一个 linear layer 用于目标类别）在 in-domain tasks 上优于 zero-shot。Full finetuning 在 in-domain 上优于 linear probe，但可能损害 zero-shot transfer。三种 regime，对应三种取舍。

### SigLIP 2：NaFlex 与 dense features

SigLIP 2（2025）加入了：
- NaFlex：单个模型处理可变 aspect ratios 和 resolutions。
- 更好的 dense features，用于 segmentation 和 depth estimation，目标是作为 VLM 中的 frozen backbone。
- Multilingual：在 100+ 种语言上训练，而 CLIP 仅限英文。
- 1B param scale，超过 CLIP 的 400M 上限。

在 2026 年 open VLMs 中，SigLIP 2 SO400m/14 是默认 vision tower。CLIP 仍然是纯 image-text retrieval 的默认选择，尤其当具体 LAION-2B 训练分布匹配你的 query pattern 时。

### ALIGN、BASIC、OpenCLIP、EVA-CLIP

ALIGN（Google，2021）：与 CLIP 相同的想法，1.8B pair scale，90% 嘈杂数据。证明 noisy data 能扩展。OpenCLIP（LAION）：在 LAION-400M / 2B 上对 CLIP 的开放复现，多个尺度，是首选 open checkpoint。EVA-CLIP：从 masked image modeling 初始化，是 VLM 的强 backbone。BASIC：Google 的 CLIP+ALIGN hybrid。它们都属于同一家族，只是数据和调参不同。

### Zero-shot ceiling

CLIP 类模型的 ImageNet zero-shot 上限大约在 76%（CLIP-G、OpenCLIP-G）。继续提升需要更大数据（SigLIP 2 达到 80%+）或架构变化（supervised heads、更多参数）。benchmark 正在饱和；真正的价值是下游 VLM 消费的 embedding space。

## 实际使用

`code/main.py` 实现了：

1. 一个 toy dual encoder（基于 hash 的 image features、text char features），让你无需 numpy 就能看到 InfoNCE 的形状。
2. 纯 Python InfoNCE loss（通过 log-sum-exp 实现数值稳定）。
3. Sigmoid pairwise loss，用于对比。
4. 一个 zero-shot classification routine：计算一组 text prompts 的 cosine similarity，并用 argmax 预测。

运行它并观察 loss curve。绝对数字是 toy；形状会匹配真实 CLIP trainer 的输出。

## 交付成果

本课产出 `outputs/skill-clip-zero-shot.md`。给定一组图像（通过路径）和目标类别列表，它会用 CLIP template 构建 text prompts，用指定 checkpoint（例如 `openai/clip-vit-large-patch14`）嵌入两侧，并返回带 similarity scores 的 top-1 / top-5 predictions。这个 skill 会拒绝对 prompt list 之外的类别做断言。

## 练习

1. 手动为 batch size 4 的 pair 实现 InfoNCE。构造 4x4 similarity matrix，运行 softmax，取出对角线，计算 cross-entropy。用这个手算结果验证你的 Python 实现。

2. SigLIP 除 temperature 外还使用 bias 参数 `b`：`S'[i,j] = S[i,j]/tau + b`。当 batch 有很大的类别不平衡（每行 negatives 远多于 positives）时，`b` 扮演什么角色？阅读 SigLIP Section 3（arXiv:2303.15343）。

3. 构建一个 cats vs dogs 的 zero-shot classifier。尝试两个 prompt templates：`a photo of a {class}` 和 `a picture of a {class}`。在 100 张测试图像上测量 accuracy。template ensemble 是否胜过单个 template？

4. 计算一次 512-GPU、batch 32k 运行中，softmax InfoNCE 与 sigmoid pairwise 的通信成本。哪个按 O(N) 扩展，哪个按 O(N^2) 扩展？引用 SigLIP Section 4。

5. 阅读 OpenCLIP scaling-laws 论文（arXiv:2212.07143，Cherti et al.）。从图中复现他们关于 data scaling 的结论：固定 model size 时，ImageNet zero-shot accuracy 与 training data size 之间是什么 log-linear 关系？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| InfoNCE | “Contrastive loss” | 对 batch similarity matrix 做 cross-entropy；每个 item 的 positive 是配对 item，negatives 是其他所有项 |
| Sigmoid loss | “SigLIP loss” | Per-pair binary cross-entropy；没有 softmax、没有 all-gather，在分布式训练中扩展便宜 |
| Temperature | “tau” | 在 softmax/sigmoid 前缩放 logits 的标量；控制分布锐度 |
| Zero-shot | “no-finetune classification” | 用 text prompts 构造 class embeddings，并按 cosine similarity 分类；不在目标类别上训练 |
| Prompt template | “a photo of a ...” | 包裹 class name 的文本脚手架；会影响 zero-shot accuracy 1-5 个点 |
| Dual encoder | “Two-tower” | 一个 image encoder + 一个 text encoder，输出到共享 D-dim 空间 |
| Hard negative | “Tough distractor” | 与 positive 足够相似、让模型必须努力区分的 negative |
| Linear probe | “Frozen + one layer” | 只在 frozen features 上训练一个 linear classifier；用于衡量 feature quality |
| NaFlex | “Native flexible resolution” | SigLIP 2 能力：无需 resizing 即可摄入任意 aspect ratio 和 resolution 的图像 |
| Temperature scaling | “log-parametrized tau” | CLIP 参数化 `log(1/tau)` 以改善梯度行为；通过 clipping 防止 tau 接近零导致崩塌 |

## 延伸阅读

- [Radford et al. — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020)：CLIP 论文。
- [Zhai et al. — Sigmoid Loss for Language Image Pre-Training (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343)：SigLIP。
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786)：multilingual + NaFlex。
- [Jia et al. — ALIGN (arXiv:2102.05918)](https://arxiv.org/abs/2102.05918)：用 noisy web data 扩展。
- [Cherti et al. — Reproducible scaling laws for contrastive language-image learning (arXiv:2212.07143)](https://arxiv.org/abs/2212.07143)：OpenCLIP scaling laws。
