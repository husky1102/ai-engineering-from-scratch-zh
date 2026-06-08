# 从 CLIP 到 BLIP-2：作为模态桥梁的 Q-Former

> CLIP 对齐了图像和文本，但不能生成 captions、回答问题或进行对话。BLIP-2（Salesforce，2023）用一个小型可训练桥梁解决了这个问题：32 个可学习 query vectors 通过 cross-attention 关注 frozen ViT 的 features，然后直接插入 frozen LLM 的输入流。188M 参数的桥梁把一个 11B LLM 连接到 ViT-g/14。直到 2026 年的每个 adapter-based VLM，包括 MiniGPT-4、InstructBLIP、LLaVA 的近亲，都是它的后代。本课阅读 Q-Former 的架构，解释它的两阶段训练，并构建一个把 visual tokens 喂进 frozen text decoder 的 toy version。

**类型：** 构建
**语言：** Python（stdlib，cross-attention + learnable-query demo）
**先修：** Phase 12 · 02（CLIP），Phase 7（Transformers）
**时间：** ~180 分钟

## 学习目标

- 解释为什么在 frozen vision encoder 和 frozen LLM 之间放一个可训练 bottleneck，在成本和稳定性上优于 end-to-end finetuning。
- 实现一个 cross-attention block，让一组固定的 learnable queries 关注外部 image features。
- 走读 BLIP-2 的两阶段预训练：representation（ITC + ITM + ITG），然后 generative（带 frozen decoder 的 LM loss）。
- 将 Q-Former 与 LLaVA 使用的更简单 MLP projector 对比，并论证什么时候各自胜出。

## 要解决的问题

你有一个 frozen ViT，它为每张图像产生 256 个维度为 1408 的 patch tokens。你有一个 frozen 7B LLM，它期望维度为 4096 的 token embeddings。显然的桥梁是一个从 1408 到 4096 的线性层，这确实能工作，但把全部 256 个 patch tokens 喂进 LLM context，会让每张图像多消耗 256 个 tokens。对 batch size 32 的图像来说，仅 visual modality 就会消耗 8192 tokens。

BLIP-2 的问题是：能否把 256-token 图像表示压缩成少得多的 tokens（比如 32），同时保留足够信息，让 LLM 生成 caption、回答问题并对图像推理？并且能否在不触碰 frozen backbones 的情况下训练这个桥梁，把训练成本限制在桥梁参数上？

答案是 Q-Former。32 个可学习 “query” vectors 会 cross-attend 到 ViT 的 patch tokens，产生一个 32-token visual summary 供 LLM 消费。总共 188M 参数。在接触 LLM 之前，它先用 contrastive、matching 和 generative objectives 训练。

## 核心概念

### Learnable queries

Q-Former 的核心技巧是：不是让 LLM 的 text tokens 去关注 image patches，而是引入一组新的 32 个 learnable query vectors `Q`，让*它们*关注 image patches。这些 queries 是模型参数，在训练中学习，并且每张图像都使用同样的 32 个 queries。

经过 cross-attention 后，每个 query 都持有图像的压缩摘要，例如“描述主要物体”“描述背景”“数物体”等。queries 并不会真的专门对应语义标签；它们会学习任何能让下游 losses 下降的编码。

### 架构

Q-Former 是一个小型 transformer（12 层，约 100M params），有两条路径：

1. Query path：32 个 query vectors 经过 self-attention（彼此之间），再对 frozen ViT 的 patch tokens 做 cross-attention，然后经过 FFN。
2. Text path：一个类似 BERT 的 text encoder，与 query path 共享 self-attention 和 FFN 权重。text path 禁用 cross-attention。

训练时两条路径都会运行。queries 和 text 通过共享 self-attention 交互，这意味着对于需要它的任务（ITM、ITG），queries 可以以 text 为条件。VLM handoff 推理时，只有 queries 流过，产生 32 个 visual tokens。

### 两阶段训练

BLIP-2 分两阶段预训练：

Stage 1：representation learning（无 LLM）。三个 losses：
- ITC（image-text contrastive）：在 pooled query tokens 与 text CLS token 之间做 CLIP 风格 contrastive。
- ITM（image-text matching）：二分类器，判断这对 image-text 是否匹配。使用 hard-negative-mined。
- ITG（image-grounded text generation）：基于 text 的 causal LM head，以 queries 为条件。迫使 queries 编码可由文本生成的内容。

只有 Q-Former 训练。ViT frozen。不涉及 LLM。

Stage 2：generative learning。接上一个 frozen LLM（OPT-2.7B 或 Flan-T5-XL 等）。通过一个小型线性层把 32 个 query outputs 投影到 LLM 的 embedding dim。把它们前置到 text prompt。只在拼接后的 prompt + image + caption sequence 上用 LM loss 训练线性投影和 Q-Former。

Stage 2 结束后，Q-Former + projection 就是完整 visual adapter。推理时：image → ViT → Q-Former → linear proj → 前置到 text → frozen LLM 发出输出。

### 参数经济学

BLIP-2 使用 ViT-g/14（1.1B，frozen）+ OPT-6.7B（6.7B，frozen）+ Q-Former（188M，trained）= 总计 8B，其中 188M 参与训练。Q-Former 只占全栈参数的约 2.4%。训练成本反映了这一点：在少量 A100 上跑几天，而不是 end-to-end 训练数周。

质量方面：BLIP-2 在 zero-shot VQA 上匹配或超过 Flamingo-80B，同时小 50 倍。桥梁有效。

### InstructBLIP 与 instruction-aware Q-Former

InstructBLIP（2023）给 Q-Former 扩展了一个额外输入：instruction text 本身。在 cross-attention 时，queries 现在能访问 image patches 和 instruction。queries 可以按 instruction 专门化（“count the cars”“describe the mood”），而不是学习一个固定摘要。在 held-out tasks 上取得 benchmark 提升。

### MiniGPT-4 与 projector-only 方法

MiniGPT-4 保留了 Q-Former，但在冻结其他所有部分的情况下只训练输出线性投影。便宜，但代价是质量：queries 是 BLIP-2 的，不是你的。它适合快速迭代，但不是最佳架构。

### 为什么 LLaVA 选择更简单的做法

LLaVA（2023，Lesson 12.05）用一个普通 2-layer MLP 取代 Q-Former，把每个 ViT patch token 投影到 LLM 空间。对 24x24 grid 来说，每张图像有 576 个 tokens，全部喂给 LLM。压缩更差，但让 LLM 可以关注 raw patches。当时这很有争议；到 2023 年末，它成为主流，因为 visual instruction data（LLaVA-Instruct-150k）证明 MLP 可以训练到保留足够信号。取舍是：LLaVA 的 context 填得更快，但它自然扩展到 multi-image 和 video。

到 2026 年，领域分裂成两边：token budget 重要时（long video、many images）Q-Former 仍然存在；raw quality per token 优先时，MLP projector 占主导。

### Gated cross-attention：祖先 Flamingo

Flamingo（Lesson 12.04）早于 BLIP-2，也使用相同的 cross-attention 思路，但它在每个 frozen LLM layer 上使用，而不是作为单个桥梁。BLIP-2 证明了你可以只压缩到 input layer，而且仍然有效。Gemini 和 Idefics 结合了两者：interleaved input tokens 加上可选 gated cross-attention，用于 in-context few-shot。

### 2026 后代

- Q-Former：BLIP-2、InstructBLIP、MiniGPT-4，以及大多数出于 token budget 考虑的视频语言模型。
- Perceiver resampler：Flamingo 的变体（Lesson 12.04）；Idefics family、Eagle、OmniMAE。
- MLP projector：LLaVA、LLaVA-NeXT、LLaVA-OneVision、Cambrian-1。
- Attention pool：VILA、PaliGemma。

四者都有效。决定性问题是：你受限于 token budget，还是受限于 quality-per-token。

## 实际使用

`code/main.py` 构建了一个 stdlib Q-Former 风格的 cross-attention：

1. 模拟 256 个 image patch tokens（dim 128）。
2. 实例化 32 个 learnable queries（dim 128）。
3. 运行 scaled-dot-product cross-attention（Q 来自 queries，K/V 来自 patches）。
4. 通过线性层投影到 LLM-dim（512）。
5. 输出 32 个 LLM-ready visual tokens。

所有数学都用纯 Python（对向量做嵌套循环）。Toy 但形状正确。attention-weight matrix 会被打印出来，所以你可以看到每个 query 从哪些 patches 拉取信息。

## 交付成果

本课产出 `outputs/skill-modality-bridge-picker.md`。给定一个目标 VLM 配置（vision encoder token count、LLM context budget、deployment constraints、quality target），它会推荐 Q-Former vs MLP vs Perceiver resampler，并为每种 bridge 给出简短理由和 parameter-count 估计。

## 练习

1. 在 PyTorch 中实现 cross-attention block。验证当有 32 个 queries 和 256 个 keys/values 时，attention-weight matrix 是 32 x 256，并且 softmax 后每一行求和为 1。

2. 在 BLIP-2 stage 1 中，Q-Former 同时运行三个 losses：ITC、ITM、ITG。用 pseudo-code 写出每个 forward signature。哪一个需要 text encoder path 激活？

3. 比较参数量：Q-Former（12 layers，768 hidden）vs 2-layer MLP projector（1408 → 4096，两层）。在什么 LLM scale 下，188M Q-Former 的成本会通过训练效率回本？

4. 阅读 BLIP-2 论文（arXiv:2301.12597）Section 3.2 中关于 Q-Former 如何初始化的内容。解释为什么从 BERT-base 初始化（而不是随机初始化）会加速收敛。

5. 对一段 10 分钟视频，以 1 FPS 采样到 60 帧，计算每帧 token 成本：（Q-Former → 32 tokens/frame）vs（MLP projector → 576 tokens/frame）。哪个能放进 128k-token LLM context window？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Q-Former | “Querying transformer” | 带 32 个 learnable query vectors 的小型 transformer，这些 queries 会 cross-attend 到 frozen ViT features |
| Learnable queries | “Soft prompt for vision” | 一组固定参数，作为 cross-attention 的 query 侧；按模型学习，在所有输入之间共享 |
| Cross-attention | “Q from here, K/V from there” | query、key、value 来自不同来源的 attention；queries 由此从 ViT patches 中拉取信息 |
| ITC | “Image-text contrastive” | 应用于 Q-Former pooled queries 与 text CLS 之间的 CLIP 风格 loss |
| ITM | “Image-text matching” | 在 hard-negative-mined pairs 上做二分类；迫使 queries 区分细粒度 mismatch |
| ITG | “Image-grounded text generation” | 以 queries 为条件生成文本的 causal LM loss；迫使 queries 编码 text-decodable 内容 |
| Two-stage pretraining | “Representation then generative” | Stage 1 单独训练 Q-Former（ITC/ITM/ITG）；Stage 2 接入 frozen LLM，只训练 projection + Q-Former |
| Frozen backbone | “Do not finetune” | vision encoder 和 LLM 权重固定；只训练 bridge |
| Projection head | “Linear to LLM dim” | 把 Q-Former output 映射到 LLM embedding dimension 的最终线性层 |
| Perceiver resampler | “Flamingo's version” | 类似 learnable-query cross-attention，由 Flamingo 在每一层使用，而不是作为单个桥梁 |

## 延伸阅读

- [Li et al. — BLIP-2 (arXiv:2301.12597)](https://arxiv.org/abs/2301.12597)：核心论文。
- [Li et al. — BLIP (arXiv:2201.12086)](https://arxiv.org/abs/2201.12086)：带 ITC/ITM/ITG 三件套的前身。
- [Li et al. — ALBEF (arXiv:2107.07651)](https://arxiv.org/abs/2107.07651)：“align before fuse”，stage 1 训练的概念祖先。
- [Dai et al. — InstructBLIP (arXiv:2305.06500)](https://arxiv.org/abs/2305.06500)：instruction-aware Q-Former。
- [Zhu et al. — MiniGPT-4 (arXiv:2304.10592)](https://arxiv.org/abs/2304.10592)：projector-only 方法。
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795)：learnable-query cross-attention 的通用架构。
