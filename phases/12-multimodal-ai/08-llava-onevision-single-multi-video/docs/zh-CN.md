# LLaVA-OneVision：一个模型处理单图、多图与视频

> 在 LLaVA-OneVision（Li et al., 2024 年 8 月）之前，开放 VLM 世界有彼此分离的谱系：单图用 LLaVA-1.5，多图用 Mantis 和 VILA 这类模型，视频用 Video-LLaVA 和 Video-LLaMA。每个模型赢下自己的 benchmark，却在其他场景失效。LLaVA-OneVision 认为，单个 curriculum 可以训练一个模型主导三种场景，并且 emergent task-transfer effects（单图技能迁移到视频，多图推理迁移到单图）会超过专家模型之和。Recipe 看似简单：在各场景中保持 constant 的 visual-token budget，再加上从 single-image 到 OneVision（multi-image）再到 video 的显式 curriculum。本课阅读 budget、curriculum 和 emergent behaviors。

**类型:** Build
**语言:** Python（stdlib，token budget solver + curriculum planner）
**先修:** Phase 12 · 05（LLaVA），Phase 12 · 06（any-resolution）
**时间:** ~180 分钟

## 学习目标

- 设计一个在 single-image、multi-image 和 video 输入之间保持 constant 的 visual-token budget。
- 安排训练 curriculum，把技能从 single-image 迁移到 video，同时避免 catastrophic forgetting。
- 解释当 curriculum 正确时，为什么同参数量下单个模型会击败专家模型。
- 说出 LLaVA-OneVision 报告的三种 emergent capabilities：multi-camera reasoning、set-of-mark prompting、iPhone-screenshot agent。

## 要解决的问题

图像、多图和视频各自以不同方式考验模型。

Single-image 需要高分辨率 token（AnyRes，约 2880 visual token）来捕捉 OCR 和细节。每个 sample budget：1 张图，2880 token。

Multi-image 需要几张中等分辨率图像（每张约 576 token），这样跨图推理才能放进 context。每个 sample budget：4-8 张图，每张 576，总计 2300-4600 token。

Video 需要许多低分辨率帧（pooling 后每帧约 196 token）来捕捉时间动态。每个 sample budget：8-32 帧，每帧 196，总计 1600-6200 token。

如果训练单独模型，你只需选择一个 budget。如果训练一个模型，你需要让 budget 在不同场景间合理缩放，同时不炸掉 context。

OneVision 之前，默认答案是“训练一个场景，忽略其他场景”。Video-LLaVA 用额外训练阶段把 video retrofitted 到 image model 上。LLaVA-NeXT 用 tiling 加入 multi-image 支持。没有一个干净地处理全部三者。

## 核心概念

### OneVision token budget

LLaVA-OneVision 选择约 3000-4000 token/sample 的统一 visual-token budget，并按场景不同分配：

- Single image：AnyRes-9（3x3 tiles + thumbnail），每个 tile 在 384 下有 729 patches，激进 bilinear pooling 2x2 → 每 tile 182。总计：`9 * 182 + 182 = 1820` token。或者 AnyRes-4，每 tile 729，得到 2916 + 729。
- Multi-image：每张图中等分辨率（384，无 tiling），729 token，无 pooling。6 张图 budget → 4374 token。
- Video：32 帧，384 分辨率，激进 3x3 bilinear pool → 每帧 81 token。总计：`32 * 81 = 2592` token。

这种分配保持总 token 近似 constant。LLM 不会看到撑爆 context 的 batch。Encoder 在不同场景中产生不同 geometry，但 LLM 消耗同样的 budget。

### 三阶段 curriculum

LLaVA-OneVision 分三阶段训练：

1. Single-image SFT（stage SI）。所有数据都是 single-image-plus-text。使用高分辨率 AnyRes 输入训练。这教会 perception、OCR 和细粒度理解。使用 LLaVA-NeXT 数据加 OneVision-specific single-image data。
2. OneVision SFT（stage OV）。混合 single-image + multi-image + video（均匀采样帧）。在统一 token budget 上训练。这教模型处理异构 batch shape。不重置权重，从 stage SI 继续。
3. Task transfer（stage TT）。继续使用目标任务混合训练，通常根据产品更偏 multi-image 或 video。可选部署 fine-tune。

关键点：curriculum order 很重要。即使数据相同，先训练 video 或先训练 multi-image 的图像性能也比 single-image-first 更差。论文显式 ablate 了这一点。

### 为什么 curriculum 有效

Single-image 训练建立感知基础。Patch token 携带细粒度视觉特征；LLM 学会把它们与文本整合。Multi-image 和 video 引入结构挑战（哪张图是哪张、先发生了什么），如果没有强感知基础，这些很难学。

如果从头混合训练所有场景，模型会欠拟合 perception（每个 batch 中 single-image 数据有限），同时过拟合 structure（大量 multi-image / video data）。结果是：模型能遵循跨图推理模式，但视觉很浅。

Curriculum ordering 让你先从 stage SI 获得强 perception，再从 stage OV 获得 compositional/temporal reasoning，而不会丢掉任一方。

### 跨场景 emergent skills

LLaVA-OneVision 论文报告了三种 emergent capabilities：

1. Multi-camera reasoning。训练中分别见过 multi-image + video；推理时被要求推理多摄像头驾驶场景。模型正确整合视角，尽管训练中从未见过完全相同格式。
2. Set-of-mark prompting。用户用编号 mark 标注图像中的物体；模型推理“mark 3 相对 mark 7 在做什么”。既没有 mark 训练，也没有 annotation 训练；能力来自 spatial grounding + multi-image reference 的组合。
3. iPhone-screenshot agent。用户提供 iPhone 屏幕截图并要求规划下一次点击。训练中有 UI 截图、用户工作流视频和 before/after 多图对。泛化到 agent 用例。

这些不是训练任务；它们从 curriculum 的组合结构中涌现。

### Visual-token pooling

Token budget 需要 pooling。OneVision 在 2D patch grid 上使用 bilinear interpolation：24x24 = 576 patches 变成 12x12 = 144（2x factor）或 8x8 = 64（3x factor）。Pooling 在 patch-grid space 中完成，而不是 token space，以保留 locality。

每个场景选择哪个 pooling factor 本身也是 hyperparameter。少 pooling = 更多 token = 更丰富表示。多 pooling = 更少 token = 能放入更多帧/图像。

### LLaVA-OneVision-1.5

2025 年 follow-up（LLaVA-OneVision-1.5，arXiv 2509.23661）在 training data、model weights 和 code 上“fully open”。在一些 benchmark 上缩小与 proprietary 的差距，并民主化 recipe。同样 curriculum，更多数据，更好 base LLM。无 architecture change。

### 与 Qwen2.5-VL 对比

Qwen2.5-VL（Lesson 12.09）做了不同选择。它使用 M-RoPE 和 dynamic FPS，而不是固定 pooling。它的 budget 随输入缩放：1 分钟视频比 5 秒视频使用更多 token。LLaVA-OneVision 固定 budget 并缩放 pooling。两者都有效；它们在可配置性与可预测性之间取舍。

## 实际使用

`code/main.py` 是 OneVision-style VLM 的 curriculum 和 budget planner。给定每 sample token budget 和目标场景混合（比如 40% single-image、30% multi-image、30% video），它会：

- 为每个场景分配 resolution、pooling factor 和 frames。
- 检查每个场景都能装进共享 budget。
- 报告预期 token count、LLM FLOPs，以及哪些场景 under-tokenized。
- 打印逐阶段训练 schedule。

用它来规划 OneVision fine-tune，或 sanity-check VLM deployment 的每请求成本。

## 交付成果

本课产出 `outputs/skill-onevision-budget-planner.md`。给定 target task distribution 和 per-sample budget，它会输出 AnyRes factor、per-frame pooling、video frame count 和 curriculum stage weights。训练或 fine-tune unified-scenario VLM 时使用它。

## 练习

1. 你的产品支持 80% single-image、10% multi-image（2-4 张图）、10% video（8-16 帧）。设计 token budget。你会把不做重 multi-image 省下来的额外 budget 放在哪里？

2. 阅读 LLaVA-OneVision Section 4.3（emergent capabilities）。提出 curriculum 很可能解锁但论文未报告的第四种 emergent skill。

3. 交换 curriculum order：先训练 multi-image，再 single-image，再 video。预测哪些 benchmark 会下降，以及为什么。

4. 论文报告的视频 benchmark 每个 sample 只训练 8 帧。这能泛化到推理时 30 秒视频吗？什么先坏掉，token budget 还是 temporal reasoning？

5. 把 24x24 patch bilinear pooling 到 12x12，每个维度是 4x reduction。用 stdlib Python 实现 pooling，并验证每个 2x2 block 的 mean 与 bilinear output 匹配。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| OneVision scenario | “Single-image, multi-image, or video” | unified VLM 处理的三种输入 shape 之一；budget 在它们之间保持 constant |
| Token budget | “How many tokens per sample” | 每个 training / inference sample 中 LLM 看到的总视觉 token，通常 3000-4000 |
| Curriculum | “Training order” | 为 emergent transfer 选择的阶段顺序（single-image → multi-image → video） |
| Bilinear pooling | “Token shrink” | 对 patch grid（2D）应用 bilinear interpolation 以减少 token count，同时保留 locality |
| Emergent skill | “Not trained, still works” | 推理时出现的能力，训练数据中没有匹配任务，来自 curriculum composition |
| AnyRes-k | “k-tile setup” | k 个固定分辨率 sub-tile 加一个 thumbnail，典型 k ∈ {4, 9} |
| Task transfer | “Cross-scenario generalization” | single-image 学到的技能通过 shared backbone 应用于 video（反之亦然） |

## 延伸阅读

- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326)
- [LLaVA-OneVision-1.5: Fully Open Framework (arXiv:2509.23661)](https://arxiv.org/abs/2509.23661)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Lin et al. — VILA (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
