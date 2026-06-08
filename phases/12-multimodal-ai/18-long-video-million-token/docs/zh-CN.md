# 百万 Token 上下文中的长视频理解

> 一段 1 小时、24 FPS 的 4K 视频，在切 patch 并嵌入之后，会产生约 6000 万个 token。一期转录后的 2 小时播客是 30000 个 token。一部长篇蓝光电影，即使用激进的 pooling 压缩，也有数十万个 token。Google 的 Gemini 1.5（2024 年 3 月）用 1000 万 token 上下文开启了这个时代，并能在小时级视频上可靠完成 needle-in-a-haystack 召回。LWM（Liu et al., 2024 年 2 月）展示了 ring attention 的扩展路径。LongVILA 和 Video-XL 进一步扩展了输入吞吐。VideoAgent 则用 agentic retrieval 替代原始上下文。每种方法都是在计算、召回和工程复杂度之间做出的不同取舍。本课会把它们并排阅读。

**类型:** Build
**语言:** Python（stdlib，needle-in-haystack 模拟器 + agentic-retrieval router）
**先修:** Phase 12 · 17（video temporal tokens）
**时间:** ~180 分钟

## 学习目标

- 计算不同 FPS 与 pooling 设置下长视频的视觉 token 总数。
- 解释三条扩展路径：brute context（Gemini 1.5）、ring attention（LWM）、token compression（LongVILA / Video-XL）。
- 从准确率与延迟角度比较 raw-context video VLM 与 agentic-retrieval video VLM（VideoAgent）。
- 为一段 30 分钟视频设计 needle-in-a-haystack 测试，并测量某一分钟位置的召回。

## 要解决的问题

384 原生分辨率下，一帧 Qwen2.5-VL 尺寸的 patch 大约是 729 个 token。经过 3x3 pooling 后，每帧是 81 个 token。30 分钟片段按 1 FPS 采样 = 1800 帧 = 145800 个 token。到 2025 年的开放 VLM 可以处理，但已经很紧。按 2 FPS，则是 291600 个 token，只有最大上下文模型放得下。

一部 2 小时电影按 1 FPS 是 583k token。超过了大多数 2026 年开放模型的能力；需要 Gemini 2.5 Pro，或者更激进地 pooling。

于是出现了三条扩展路径。

## 核心概念

### 路径 1：Brute context（Gemini 1.5、Claude Opus）

把硬件砸向问题。把上下文扩展到数百万 token，在一次 forward pass 中处理全部内容。

Gemini 1.5 Pro 发布时支持 1M token；Gemini 1.5 Ultra 到 10M；Gemini 2.5 Pro 在 2026 年可以可靠处理数小时视频。论文（arXiv:2403.05530）记录了在最高约 9.5M token 时，needle-in-a-haystack 召回达到 99.7%。

工程上：自定义 attention 实现，带内存层级（local + global + sparse），并通过 MoE expert routing 提升长上下文效率。完整细节未公开。也不是开源实现。

### 路径 2：Ring attention（LWM、LongVILA）

Ring attention 把长序列分布到一圈设备上，每个设备持有一个 chunk。跨完整序列的 attention 通过环形模式完成：每个设备把自己的 chunk 发送给下一个设备，计算部分 attention，再聚合结果。

LWM（Liu et al., 2024）用这种方式训练了 1M-token 上下文模型。训练计算随上下文线性扩展，而不是二次扩展：attention 的二次成本被摊到 ring 中的多台设备上。

LongVILA（arXiv:2408.10188）把这个模式适配到 VLM。1400 帧视频，每帧 192 个 token = 268k 上下文，并使用 8 路并行 ring attention 训练。

### 路径 3：Token compression（Video-XL、LongVA）

比 brute context 更便宜：在 LLM 看到序列之前先激进压缩。

Video-XL（arXiv:2409.14485）使用 visual summary token：每个包含 N 帧的 clip 产出一个“summary” token，该 token attend 到这 N 帧。推理时，LLM 每个 clip 只看一个 summary token，从而大幅缩短上下文。

LongVA 通过“long context transfer”技术把 LLM 上下文从 200k 扩到 2M。先在长上下文文本上训练，再通过共享表示迁移到长上下文视频。

Token compression 用特定时间戳处的召回换取可扩展性。模型通常知道发生了什么，但有时会错过精确帧。

### 路径 4：Agentic retrieval（VideoAgent）

不要把完整视频喂给 LLM。相反，把视频当成数据库，用 LLM 去查询它。

VideoAgent（arXiv:2403.10517）：

1. LLM 读取问题。
2. LLM 向 retrieval tool 请求相关 clip（“show me segments with a cat”）。
3. Tool 返回匹配的 clip 时间戳。
4. LLM 通过 VLM 读取这些 clip。
5. LLM 组合答案，或提出后续查询。

这是把 LLM-as-agent 模式应用到长视频上。推理更便宜（只编码相关 clip），工程更难（retrieval 质量变成瓶颈）。

### Needle-in-a-haystack benchmark

标准长上下文测试：在视频随机位置插入一个唯一视觉或文本标记，然后提出一个需要回忆该标记的查询。

指标：在不同视频长度与标记位置上的 Recall@k。

Gemini 2.5 Pro 在最长 90 分钟视频上达到 >99% 召回。开放 72B 模型（Qwen2.5-VL-72B、InternVL3-78B）在 30 分钟时约 85-90%，超过 60 分钟后退化。

如果 retrieval tool 足够好，VideoAgent 在 2 小时以上的视频中可以匹配或超过 raw-context 模型，因为 retrieval 会命中那根针。

### 选择哪条路径

对于 15 分钟片段和 frontier accuracy：开放 72B + native context 通常可用。选择 Qwen2.5-VL-72B。

对于 30 分钟到 1 小时内容：开放模型选 LongVILA 或 Video-XL；闭源选 Gemini 2.5 Pro。质量门槛很关键，frontier 质量仍然偏向闭源。

对于 2 小时以上内容：选择 VideoAgent 或类似 retrieval 模式。另一种方案是先摘要成更小 chunk，再喂分层摘要。

### 2026 年生产模式

实践中的生产长视频 pipeline 通常是混合式：

1. 对整段视频运行 dynamic-FPS sampling + aggressive pooling（得到 100k-token 的全局表示）。
2. 传给 72B VLM 生成全局摘要。
3. 如果用户提出细节问题，就用该摘要作为索引运行 agentic retrieval。

这结合了 brute-context 的全局理解和 retrieval 的局部细节。

## 实际使用

`code/main.py`：

- 计算从 1 分钟到 3 小时视频在不同 FPS + pooling 下的 token budget。
- 模拟 needle-in-a-haystack 运行：在随机时间戳注入 marker，提出问题，给召回打分。
- 包含 agentic-retrieval router 模拟器，用于选择要喂给下游 VLM 的具体 clip。

运行 budget table，感受尺度差距。

## 交付成果

本课产出 `outputs/skill-long-video-strategy-planner.md`。给定视频时长与查询复杂度，它会在 brute-context、compression 与 agentic retrieval 之间选择，并计算延迟与质量预期。

## 练习

1. 一场 45 分钟讲座，1 FPS，每帧 81 个 token。总 token 数是多少？能放进哪些模型的上下文？

2. 设计一个 needle-in-a-haystack 测试：你会在第几分钟注入 marker，精确查询格式是什么？

3. 在 1 小时视频上比较 brute-context Qwen2.5-VL-72B（80k context）与 VideoAgent（Claude 3.5 + retrieval）。谁的召回更好？谁的延迟更低？

4. Ring attention 的内存成本随序列长度线性扩展，也随设备数量线性扩展。解释为什么，以及如果去掉 ring-rotation 阶段会失败在哪里。

5. 阅读 Gemini 1.5 第 5 节关于 needle-in-a-haystack 的内容。论文发现 1M 与 10M token 边界处的召回有什么变化？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Brute context | “Just more tokens” | 把 LLM 上下文扩展到数百万 token；一次 pass 处理全部内容 |
| Ring attention | “LWM-style parallel” | 分布式 attention 模式，每个设备持有一个 chunk 并轮转 |
| Token compression | “Summary tokens” | 在 LLM 之前通过可学习压缩器减少每个 clip 的 token |
| Needle-in-haystack | “NIH test” | 在随机位置插入唯一 marker，测试时要求模型回忆它 |
| Agentic retrieval | “LLM as query planner” | LLM 向 retrieval tool 请求相关 clip，通过 VLM 阅读后组合答案 |
| VideoAgent | “Retrieval pattern for video” | 典型 agentic-retrieval 设计：question -> tool -> clip -> answer |

## 延伸阅读

- [Gemini Team — Gemini 1.5 (arXiv:2403.05530)](https://arxiv.org/abs/2403.05530)
- [Liu et al. — LWM / RingAttention (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Xue et al. — LongVILA (arXiv:2408.10188)](https://arxiv.org/abs/2408.10188)
- [Shu et al. — Video-XL (arXiv:2409.14485)](https://arxiv.org/abs/2409.14485)
- [Wang et al. — VideoAgent (arXiv:2403.10517)](https://arxiv.org/abs/2403.10517)
