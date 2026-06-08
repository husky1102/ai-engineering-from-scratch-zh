# 水印：SynthID、Stable Signature、C2PA

> 三项技术构成了 2026 年 AI 生成内容出处证明的结构。SynthID（Google DeepMind）：图像水印于 2023 年 8 月推出，文本 + 视频于 2024 年 5 月推出（Gemini + Veo），文本水印于 2024 年 10 月通过 Responsible GenAI Toolkit 开源，2025 年 11 月与 Gemini 3 Pro 一起推出统一多媒体检测器。文本水印以难以察觉的方式调整 next-token 采样概率；图像 / 视频水印能经受压缩、裁剪、滤镜和帧率变化。Stable Signature（Fernandez 等，ICCV 2023，arXiv:2303.15435）：微调 latent diffusion decoder，使每个输出都包含固定消息；只保留 10% 内容的裁剪生成图像仍能以 >90% 检出率、FPR<1e-6 被检测。后续 “Stable Signature is Unstable”（arXiv:2405.07145，2024 年 5 月）：微调可以去除水印，同时保持质量。C2PA：带密码学签名、可防篡改的元数据标准（C2PA 2.2 Explainer 2025）。水印与 C2PA 互补：元数据可能被移除，但承载更丰富的出处；水印能跨转码保留，但携带的信息较少。

**类型：** 构建
**语言：** Python (stdlib, token-watermark embed + detect)
**先修：** 第 10 阶段 · 第 04 课（sampling），第 01 阶段 · 第 09 课（information theory）
**时间：** 约 75 分钟

## 学习目标

- 描述 token 级水印（SynthID-text 风格）及其可检测机制。
- 描述 Stable Signature，以及打破它的 2024 年移除攻击。
- 说明 C2PA 的作用，以及为什么它与水印互补。
- 描述关键局限：模型特定信号、改写下的鲁棒性，以及保义攻击（arXiv:2508.20228）。

## 要解决的问题

2023-2024 年，deepfake 和 AI 生成内容大规模进入政治与消费场景。水印是被提出的技术出处信号：在生成时打标，之后再检测。2025 年证据显示，没有任何水印是无条件鲁棒的，但若与 C2PA 元数据分层结合，这个组合能提供可用的出处叙事。

## 核心概念

### 文本水印（SynthID-text 风格）

Kirchenbauer 等 2023 机制，由 Google 产品化：

1. 在每个解码步骤，hash 前 K 个 token，生成 vocabulary 的伪随机划分，分为 “green” 和 “red” 集合。
2. 通过给 green logits 加上 δ，使采样偏向 green 集合。
3. 生成内容包含的 green token 多于随机机会水平。

检测：重新 hash 每个前缀，统计生成中的 green token，并计算 z-score。带水印文本的 z-score >0，人类文本约为 0。

性质：
- 对读者不可感知（δ 足够小，质量损失较小）。
- 在能访问 vocabulary partition function 时可检测。
- 对改写不鲁棒；重写文本会破坏信号。

SynthID-text 于 2024 年 10 月通过 Google 的 Responsible GenAI Toolkit 开源。

### Stable Signature（图像）

Fernandez 等 ICCV 2023。微调 latent diffusion decoder，使每张生成图像都包含嵌入 latent representation 的固定二进制消息。检测通过 neural decoder 从 latent 中解码。裁剪到 10% 内容的图像仍能以 >90% 检出率、FPR<1e-6 被检测。

2024 年 5 月 “Stable Signature is Unstable”（arXiv:2405.07145）：微调 decoder 可以去除水印，同时保持图像质量。生成后的对抗性微调很便宜；该水印的对抗鲁棒性有限。

### SynthID 统一检测器（2025 年 11 月）

与 Gemini 3 Pro 同时推出：一个多媒体检测器，可在一个 API 中读取文本、图像、音频和视频中的 SynthID 信号。统一了 Google 的出处证明栈。

### C2PA

Coalition for Content Provenance and Authenticity。带密码学签名、可防篡改的元数据标准。C2PA 2.2 Explainer（2025）。C2PA manifest 记录出处声明（谁创建、何时创建、做了哪些转换），并由创建者的 key 签名。

与水印互补：
- 元数据可能被移除；水印不容易被移除。
- 元数据很丰富（完整出处链）；水印只携带少量 bits。
- C2PA 依赖平台采用；水印会自动嵌入。

Google 在 Search、Ads 和 “About this image” 中同时整合二者。

### 局限

- **模型特定。** SynthID 只给启用 SynthID 的模型生成内容加水印。来自未启用 SynthID 的模型的生成内容不会带水印，因此“没有 SynthID 信号”并不能证明真实性。
- **改写。** 文本水印无法经受保义改写。
- **转换攻击。** arXiv:2508.20228（2025）展示了会破坏文本水印和许多图像水印的保义攻击。
- **微调移除。** 根据 “Stable Signature is Unstable”，生成后的微调可以移除嵌入水印。

### EU AI Act Article 50

AI 生成内容标注 Transparency Code（第一版草案 2025 年 12 月，第二版草案 2026 年 3 月；根据 [European Commission status page](https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content)，预计 2026 年 6 月定稿）。截至 2026 年 4 月，该 Code 仍为草案，时间线可能变化。监管层要求技术层。Deepfake 必须标注。

### 它在第 18 阶段中的位置

第 22-23 课关注模型发出的内容（私有数据、出处信号）。第 27 课覆盖训练数据治理。第 24 课是要求这些技术措施的监管框架。

## 实际使用

`code/main.py` 构建一个玩具文本水印。Token 是整数 0..N-1；带水印采样偏向由 hash 定义的 green 集合。检测器计算 green-token z-score。你可以观察 1000-token 生成中的检测结果，查看改写如何破坏信号，并测量人类文本上的假阳性率。

## 交付成果

本课产出 `outputs/skill-provenance-audit.md`。给定带出处声明的内容部署，它会审计：水印机制（如果有）、C2PA 签名链（如果有）、每个机制的对抗鲁棒性，以及各模态覆盖范围。

## 练习

1. 运行 `code/main.py`。报告带水印 1000-token 生成与人类撰写文本的 z-score。识别 95% 置信阈值下的假阳性率。

2. 实现一个把 30% token 替换成同义词的改写攻击。重新测量 z-score。

3. 阅读 Kirchenbauer 等 2023 第 6 节关于鲁棒性的内容。为什么文本水印会在改写下失败，而图像水印能经受裁剪？

4. 设计一个使用 SynthID-text + C2PA 元数据的部署。描述消费者看到的出处链。识别每个组件的一个失败模式。

5. 2024 年 “Stable Signature is Unstable” 结果显示微调会移除图像水印。设计一种限制该攻击的部署控制，例如要求对微调 checkpoint 的发布进行签名。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| SynthID | “Google 的水印” | 跨模态出处信号；文本、图像、音频、视频 |
| Token watermark | “Kirchenbauer 风格” | 偏置采样文本水印，可通过 green-token z-score 检测 |
| Stable Signature | “图像水印” | 微调 decoder 的水印；ICCV 2023 |
| C2PA | “元数据标准” | 带密码学签名、可防篡改的出处元数据 |
| 改写鲁棒性 | “重述会不会破坏它” | 文本水印属性；目前有限 |
| 微调移除 | “对抗性去水印” | 通过 decoder 微调移除图像水印的攻击 |
| 跨模态检测器 | “统一 SynthID” | 2025 年 11 月跨模态统一 API |

## 延伸阅读

- [Kirchenbauer et al. — A Watermark for Large Language Models (ICML 2023, arXiv:2301.10226)](https://arxiv.org/abs/2301.10226) — token 水印机制
- [Fernandez et al. — Stable Signature (ICCV 2023, arXiv:2303.15435)](https://arxiv.org/abs/2303.15435) — 图像水印论文
- ["Stable Signature is Unstable" (arXiv:2405.07145)](https://arxiv.org/abs/2405.07145) — 移除攻击
- [Google DeepMind — SynthID](https://deepmind.google/models/synthid/) — 跨模态水印
- [C2PA 2.2 Explainer (2025)](https://c2pa.org/specifications/specifications/2.2/explainer/Explainer.html) — 元数据标准
