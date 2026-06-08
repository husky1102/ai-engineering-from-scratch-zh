# MIO 与 Any-to-Any Streaming 多模态模型

> GPT-4o 交付了一个大多数 open models 无法复制的产品：一个能听语音、看视频，并实时说话回应的 agent。到 2024 年末，open-ecosystem 的答案是 MIO（Wang et al., 2024 年 9 月）。MIO tokenizes 文本、图像、语音和音乐，在交错序列上训练一个 causal transformer，并生成任意模态到任意模态。AnyGPT（Zhan et al., 2024 年 2 月）是 proof of concept；MIO 是 scale-up；Unified-IO 2（Allen AI，2023 年 12 月）是带 vision + action grounding 的近亲。本课阅读 any-to-any pattern：四个 tokenizers、一个 transformer、streaming-friendly decode。

**类型:** Learn
**语言:** Python (stdlib, four-modality token allocator + streaming decode loop)
**先修:** Phase 12 · 11 (Chameleon), Phase 6 (Speech and Audio)
**时间:** ~120 minutes

## 学习目标

- 设计一个共享词表，容纳 text、image、speech 和 music tokens 且不发生 collisions。
- 比较 SEED-Tokenizer（图像）与 SpeechTokenizer residual-VQ（语音）在压缩 + 重建上的权衡。
- 解释构建 any-to-any generation 的四阶段 curriculum。
- 说出三个 open any-to-any recipes 及其主要权衡：MIO、AnyGPT、Unified-IO 2。

## 要解决的问题

统一多模态模型很容易宣称，却很难大规模构建。直到 2024 年，大多数“any-to-any”系统都是流水线式的：vision model → text representation → speech model → audio。每一跳都会丢失信息、增加延迟，并让训练更复杂。GPT-4o 的 demo video 展示了一个 subsecond response 的 single-model alternative；open systems 落后了数月。

工程挑战包括：

- 每种模态都必须有 tokenizer，压缩要足够接近无损以便重建，并以 transformer 能消费的速率产生 token。
- 单个词表必须为空间分配 text（32k+）、image（16k+）、speech（4k+）、music（8k+）。至少需要四万多个 entries。
- 训练数据必须覆盖每种 input-output pair（text→image、image→speech、speech→image 等），或者模型必须能组合泛化。
- 推理必须足够快地 stream output tokens，以达到对话延迟（<500ms time-to-first-audio-byte）。

## 核心概念

### 四种模态的四个 tokenizers

MIO 的 tokenizer stack：

- Text：标准 BPE，vocab ~32000。
- Image：SEED-Tokenizer（2023）——带离散 codebook 的 quantized VAE，4096 entries，每张图 32x32 tokens。
- Speech：SpeechTokenizer residual-VQ（2023）——把 16kHz waveform 编码到 8 个分层 codebooks；第一层是粗 content，后续层添加 prosody 和 speaker identity。
- Music：类似的 residual-VQ（Meta 的 MusicGen / Encodec family），4-8 codebooks。

每种模态都产生 integer tokens。这些 token 在共享词表中获得互不重叠的 ID ranges：

```text
text:   0..31999
image:  32000..36095  (4096 image tokens)
speech: 36096..40191  (4096 speech base tokens, plus residual layers)
music:  40192..48383  (8192 music tokens)
sep:    48384..48390  (<image>, <speech>, <music>, </...>, etc.)
```

总计约 48k vocabulary。input embedding 与 output projection 都覆盖全部词表。

### Streaming decode

语音生成使用 residual-VQ。transformer 预测基础层（layer 0）的 speech tokens；一个 parallel-decoded residual quantizer 预测后续层。每个 layer 0 token 在 16kHz audio 中大约对应 50ms。

streaming pattern：

1. 用户对麦克风说话；real-time audio tokenizer 每 50ms 发出 speech tokens。
2. MIO 边到边消费 token（prompt prefill + incremental forward）。
3. Output tokens 随生成流出；parallel speech decoder 以约 50-150ms 延迟把它们转成 audio samples。
4. Time-to-first-audio-byte：MIO 论文中约 300-500ms，接近 GPT-4o 的约 250ms。

Mini-Omni（arXiv:2408.16725）、GLM-4-Voice（arXiv:2412.02612）和 Moshi（arXiv:2410.00037）是互补的 streaming speech-LLM 设计。尤其 Moshi 在单 GPU 上达到 160ms round-trip。

### 四阶段 curriculum

MIO 的训练 curriculum：

1. Stage 1 — alignment。大规模 modality-pair corpora：text-image、text-speech、text-music。每对模态使用自己的 token vocabulary segment。训练共享词表。
2. Stage 2 — interleaved。多模态交错文档（带图像 + 视频的博客、带 transcripts 的 podcasts 等）。训练跨模态 context。
3. Stage 3 — speech-enhanced。额外音频数据，在不损失文本能力的前提下提升语音质量。
4. Stage 4 — SFT。跨模态 instruction tuning：VQA、captioning、narration、speech-to-speech dialogue。

缺少某一阶段会降低特定能力：跳过 stage 2，模型会失去 cross-modality context；跳过 stage 3，语音会很差。

### Chain-of-visual-thought

MIO 引入 chain-of-visual-thought：模型把中间 image tokens 作为 reasoning step 发出。对于“is the cat climbing a tree?”，模型会：

1. 发出 `<image>` tokens，渲染场景（来自输入图像或 sketch）。
2. 发出分析 sketch 的文本。
3. 发出最终答案。

渲染出的中间图像充当 scratchpad。空间推理任务上的 benchmarks 会提升。这个想法对应文本推理中的 chain-of-thought。

### Any-to-any 竞争者

- AnyGPT（arXiv:2402.12226）：4 种模态（text、image、speech、music），类似设计。
- Unified-IO 2（arXiv:2312.17172）：增加 vision action outputs、depth、normals。任务更多样，规模更小。
- NExT-GPT（arXiv:2309.05519）：LLM + modality-specific diffusion decoders。不是 single-model approach。
- CoDi（arXiv:2305.11846）：composable diffusion；通过 shared latent 做 any-to-any。

MIO 最接近 pure-token any-to-any。AnyGPT 是它的概念祖先。

### 延迟预算

对于对话产品，每个组件的延迟都重要：

- Mic to audio tokens：约 50ms。
- Prefill（audio tokens + history）：8B 模型上约 100ms。
- First output token：约 50ms。
- Parallel residual-VQ + speech decoder：约 100-150ms。

Total time-to-first-audio-byte：最低约 300ms。GPT-4o 声称约 250ms。Moshi 声称 160ms。MIO/AnyGPT 在公开 benchmarks 中约 400-600ms。

### 为什么 any-to-any 仍然很难

即使在 2026 年，open any-to-any models 在两个维度上仍落后于 closed models：

- 语音质量。residual-VQ tokenizer 是有损的；对话语音听起来比 ElevenLabs-class voices 更机械。
- 跨模态推理。要求模型“sing about what you see”仍比纯视觉任务更容易失败。

这些仍是开放研究问题。Qwen3-Omni（Lesson 12.20）是 2025 年最先进的 open attempt。

## 实际使用

`code/main.py`：

- 定义四模态 vocabulary allocation 并打印它。
- 通过 tokenizer router 路由一组 multimodal inputs（text、image、audio-clip、music）。
- 模拟 text-to-speech response 的 streaming decode，并统计 latency。
- 给定 encoder、prefill 和 decoder latencies，计算预期 time-to-first-audio-byte。

## 交付成果

本课产出 `outputs/skill-any-to-any-pipeline-auditor.md`。给定一个对话产品规格（输入模态、输出模态、延迟目标），它会审计 MIO-family 的设计选择，并计算 latency budget。

## 练习

1. 你的产品接受 speech input 并返回 speech output。端到端 latency budget 目标是多少？列出消耗时间的组件。

2. SpeechTokenizer residual-VQ 使用 8 个 codebooks。提出为什么 residual levels 必须 parallel-decoding（相对 sequential），以及它带来什么延迟节省。

3. 你的 vocabulary 有 32k text + 4k image + 4k speech。加入 8k music 和约 10 个 separators。在 hidden dim 4096 下，embedding-matrix 参数成本是多少？

4. Chain-of-visual-thought 会发出中间图像。哪些问题会受益？哪些会被额外 tokens 伤害？

5. 阅读 Moshi（arXiv:2410.00037）。描述它的“inner monologue”技术，并与 MIO 的 chain-of-visual-thought 对比。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Any-to-any | “Multimodal in/out” | 一个能以任意方向接受并发出 text、image、speech 和 music 的单一模型 |
| Residual-VQ | “Speech tokenizer stack” | 多 codebook tokenization，每层添加信息；base layer 是 content，后续层是 prosody |
| SEED-Tokenizer | “Image codes” | MIO 使用的离散图像 tokenizer，codebook 有 4096 entries |
| Chain-of-visual-thought | “Visual scratchpad” | 模型在最终答案前生成一张中间图像作为 reasoning step |
| Time-to-first-audio-byte | “TTFAB” | 从用户语音到第一个音频输出的延迟；<500ms 才有对话感 |
| Four-stage curriculum | “Training recipe” | Alignment -> interleaved -> speech-enhanced -> SFT，按此顺序 |

## 延伸阅读

- [Wang et al. — MIO (arXiv:2409.17692)](https://arxiv.org/abs/2409.17692)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Lu et al. — Unified-IO 2 (arXiv:2312.17172)](https://arxiv.org/abs/2312.17172)
- [Wu et al. — NExT-GPT (arXiv:2309.05519)](https://arxiv.org/abs/2309.05519)
- [Tang et al. — CoDi (arXiv:2305.11846)](https://arxiv.org/abs/2305.11846)
