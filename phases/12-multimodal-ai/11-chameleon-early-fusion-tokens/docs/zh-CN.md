# Chameleon 与早期融合的纯 Token 多模态模型

> 到目前为止，我们见过的每个 VLM 都把图像和文本分开处理。视觉 token 来自视觉编码器，流入 projector，然后在 LLM 内部与文本相遇。视觉词表和文本词表从不重叠。Chameleon（Meta，2024 年 5 月）提出了一个问题：如果它们重叠会怎样？训练一个 VQ-VAE，把图像转成来自共享词表的离散 token 序列。现在每份多模态文档都是一个序列：文本 token 与图像 token 交错，只用一个自回归损失。副作用是：模型可以生成混合模态输出，在一次推理调用中交替生成文本和图像 token。本课阅读早期融合的主张，并从头到尾构建一个玩具版本。

**类型:** Build
**语言:** Python (stdlib, VQ-VAE tokenizer + interleaved decoder)
**先修:** Phase 12 · 05, Phase 8 (Generative AI)
**时间:** ~180 minutes

## 学习目标

- 解释为什么共享词表 + 单一损失会改变模型能力边界。
- 描述 VQ-VAE 如何把图像 token 化为与 transformer 的 next-token 目标兼容的离散序列。
- 说出 Chameleon 的训练稳定性技巧：QK-Norm、dropout 放置位置、LayerNorm 顺序。
- 比较 Chameleon 与 BLIP-2 的 Q-Former 方法，并说明何时该选择哪一种。

## 要解决的问题

基于 adapter 的 VLM（LLaVA、BLIP-2、Qwen-VL）把文本和图像当作两种不同的东西。文本 token 经过 `embed(text_token)`；图像经过 `visual_encoder(image) → projector → ... pseudo_tokens`。模型有两条输入路径，在中途才合并。

这带来三个后果：

1. LLM 只能消费图像，不能发出图像。输出只能是文本。
2. 混合模态文档（像文章中交替出现段落和图片）很别扭：你要么在模型外解析多模态输入，要么把多次生成串起来。
3. 分布不匹配。视觉 token 与文本 token 位于 hidden space 的不同区域，会产生微妙的对齐问题。

Chameleon 拒绝这个前提：图像只是来自共享词表的离散 token 序列。在交错文档上训练模型，一个损失、一个自回归 decoder，于是混合模态生成自然解锁。

## 核心概念

### 作为图像 tokenizer 的 VQ-VAE

tokenizer 是一个 vector-quantized variational autoencoder。架构如下：

- Encoder：CNN + ViT，将图像映射为一个空间特征图，例如 32x32 个 dim 256 的特征。
- Codebook：一个由 K 个向量组成的学习词表（Chameleon 使用 8192），同样是 dim 256。
- Quantization：对每个空间特征，按 L2 distance 查找最近的 codebook entry。用整数索引替换连续特征。
- Decoder：CNN，将量化后的特征还原成像素。

训练：VAE reconstruction loss + commitment loss + codebook loss。codebook 索引形成图像的离散字母表。

对 Chameleon 来说，一张图像会变成 32*32 = 1024 个 token，来自大小为 8192 的词表。把它们与文本 token（来自 LLM 的 BPE 词表，比如 32000）拼接。最终词表：40192。transformer 看到的是一个序列、一个损失。

### 共享词表

Chameleon 的词表组合了文本 token、图像 token 和模态分隔符。每个 token 都有一个单一 ID。输入 embedding 层把每个 ID 映射到 D-dim hidden vector。输出 projection 把 hidden 映射回 vocab logits。Softmax 选择下一个 token，不论它属于哪种模态。

分隔符很重要：`<image>` 和 `</image>` 标签包住图像 token 序列。生成时，如果模型发出 `<image>`，下游软件就知道接下来的 1024 个 token 是 VQ 索引，需要送入 decoder 渲染像素。

### 混合模态生成

推理就是在共享词表里做 next-token prediction。示例 prompt："Draw a cat and describe it." Chameleon 发出：

```text
<image> 4821 1029 2891 ... (1024 image tokens) </image>
The cat is orange, sitting on a windowsill...
```

模型自主选择顺序：它可能先生成图像再生成文本，也可能先文本后图像，或者交错生成。同一个 decoder，同一个损失。

相比之下，adapter VLM 的生成只能输出文本。Chameleon 重新打开了“模型输出模态可以是什么”的问题。

### 训练稳定性：QK-Norm、dropout、LayerNorm 顺序

早期融合训练在规模变大时不稳定。Chameleon 论文记录了三个技巧：

- QK-Norm。在 attention 内部，对 query 和 key projection 先做 LayerNorm，再计算 dot product。它能防止深层 logit magnitude 爆炸。多个 2024 年后的大模型都使用了这个技巧。
- Dropout 放置位置。在每次 residual-add 之后都做 dropout，而不只是 attention 和 MLP 之后。当图像 token 的梯度可能占主导时，需要更强的正则化。
- LayerNorm 顺序。residual branch 上使用 Pre-LN（标准做法），并在最后一个 block 的 skip connection 上额外加一个 LN。它能稳定最后一层的梯度流。

没有这些技巧时，34B 参数的 Chameleon 训练在多个 checkpoint 发散。有了这些技巧后，它能收敛。训练 recipe 与架构本身一样，都是贡献的一部分。

### tokenizer 的重建上限

VQ-VAE 是有损的。在 8192 个 codebook entries、每张 512x512 图像 1024 个 token 的设定下，重建 PSNR 上限大约是 26-28 dB。这足以生成可识别图像，但明显差于连续空间 diffusion（Stable Diffusion 3 达到 32+ dB）。

tokenizer 是瓶颈。更好的 tokenizer（MAGVIT-v2、IBQ、SBER-MoVQGAN）能抬高上限。Emu3（Lesson 12.12）仅靠更好的 tokenizer 就达到了 SDXL 质量的生成。

### Chameleon vs BLIP-2 / LLaVA

Chameleon（早期融合，共享词表）：
- 一个损失，一个 decoder。
- 生成混合模态输出。
- tokenizer 是质量上限。
- 昂贵：推理路径中每张生成图像都需要 VQ-VAE decoder。

BLIP-2 / LLaVA（后期融合，独立 tower）：
- 视觉输入，文本输出。
- 复用预训练 LLM。
- 理解任务没有 tokenizer 瓶颈。
- 便宜：单次 forward pass。

按任务选择。如果你需要图像生成，选 Chameleon family。如果你只需要理解，adapter-VLM 更简单，并且复用更多预训练计算。

### Fuyu 和 AnyGPT

Fuyu（Adept，2023）是一种相关方法：完全跳过独立视觉编码器，把原始 image patches 当成 token 一样喂进 LLM 的输入 projection，不使用 tokenizer。它比 Chameleon 更简单，但失去了共享词表带来的输出生成能力。

AnyGPT（Zhan et al., 2024）把 Chameleon 扩展到四种模态：文本、图像、语音、音乐。每种模态都使用同样的 VQ-VAE 技巧，并共享 transformer。Any-to-any generation。Lesson 12.16 会更深入介绍。

## 实际使用

`code/main.py` 构建一个端到端玩具早期融合模型：

- 一个微型 VQ-VAE 风格 quantizer，将 8x8 patches 映射到 codebook 索引（K=16）。
- 一个共享词表：(text ids 0..31) + (image ids 32..47) + (separators 48, 49)。
- 一个玩具自回归 decoder（bigram table），在合成 captions + image-token sequences 上训练。
- 一个 sampling loop，给定 prompt 后发出交替的 text + image tokens。

代码故意把 transformer 保持得很小（bigrams），这样你可以端到端追踪信号流。

## 交付成果

本课产出 `outputs/skill-tokenizer-vs-adapter-picker.md`。给定一个产品规格（只理解，还是理解 + 生成；所需图像质量；成本预算），它会在 Chameleon-family（早期融合）与 LLaVA-family（后期融合）之间做选择，并用定量经验规则说明理由。

## 练习

1. Chameleon 使用 K=8192 个 codebook entries，每张 512x512 图像 1024 个 token。估算相对 24-bit RGB 图像的压缩率。它是有损的吗？有多有损？

2. 按相同 VQ-VAE 密度，一张 4K 图像（3840x2160）会产生多少 image tokens？Chameleon 风格模型能在一次推理调用中生成 4K 图像吗？最先崩的是 context、tokenizer quality，还是 KV cache？

3. 用纯 Python 实现 QK-Norm。给定 64-dim query 和 key，展示 LayerNorm 前后的 dot product。为什么在深层控制 magnitude 很重要？

4. 阅读 Chameleon Section 2.3 中关于训练稳定性的内容。描述论文观察到的 34B 模型在没有 QK-Norm 时的精确失败模式。“norm explosion”的特征是什么？

5. 扩展玩具 decoder，让它在 text-only prompt 下发出 mixed-modality response。测量当训练数据分布为 60% text-first / 40% image-first 时，模型选择 image-first 与 text-first 的频率。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| 早期融合 | “Unified tokens” | 从第一步开始就把图像转换为与 transformer 词表共享的离散 token |
| VQ-VAE | “Image tokenizer” | CNN + ViT + codebook，将图像映射为 transformer 可以预测的整数索引 |
| 共享词表 | “One dictionary” | 覆盖文本 + 图像 + 模态分隔符的单一 token ID 空间 |
| QK-Norm | “Attention stabilizer” | 在 query 和 key 做 dot product 前应用 LayerNorm，防止 norm blowup |
| 混合模态生成 | “Text + image output” | 一次推理中自主产生交错的文本 token 和图像 token |
| Codebook size | “K entries” | VQ-VAE 可量化到的离散向量数量；在压缩与保真度之间权衡 |
| Tokenizer ceiling | “Reconstruction limit” | 解码 VQ tokens 可达到的最佳 PSNR；限制模型图像质量 |

## 延伸阅读

- [Chameleon Team — Chameleon: Mixed-Modal Early-Fusion Foundation Models (arXiv:2405.09818)](https://arxiv.org/abs/2405.09818)
- [Aghajanyan et al. — CM3 (arXiv:2201.07520)](https://arxiv.org/abs/2201.07520)
- [Yu et al. — CM3Leon (arXiv:2309.02591)](https://arxiv.org/abs/2309.02591)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Adept — Fuyu-8B blog (adept.ai)](https://www.adept.ai/blog/fuyu-8b)
