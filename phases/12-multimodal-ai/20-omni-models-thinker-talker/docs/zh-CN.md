# Omni Models：Qwen2.5-Omni 与 Thinker-Talker 拆分

> GPT-4o 在 2024 年 5 月的产品演示之所以具有冲击力，不是因为底层模型本身，而是因为产品形态：一个语音界面，你说话，模型看到摄像头看到的内容，并在 250ms 内用语音回应。开放生态在 2024 年余下时间和 2025 年一路追赶这个产品表面。Qwen2.5-Omni（2025 年 3 月）是开放设计的参考：一个 Thinker（大型文本生成 transformer）加一个 Talker（并行语音生成 transformer），由 streaming speech tokens 连接。Mini-Omni 简化了它，Moshi 追平了它的延迟，GLM-4-Voice 把它扩展到中文。本课会读懂 Thinker-Talker 架构，以及让实时流式对话成立的 latency budget。

**类型:** Build
**语言:** Python（stdlib，streaming pipeline latency simulator + VAD loop）
**先修:** Phase 12 · 19（audio-LLMs），Phase 12 · 16（any-to-any）
**时间:** ~180 分钟

## 学习目标

- 把推理 pipeline 拆成 Thinker（文本推理）和 Talker（语音合成），并解释为什么并行 streaming 有效。
- 逐组件计算一次对话交互的 time-to-first-audio-byte（TTFAB）budget。
- 描述 TMRoPE 如何在 Thinker 内对 vision、audio 与 text 做时间对齐 position encoding。
- 说出三种实时对话模式：half-duplex、turn-taking、full-duplex。

## 要解决的问题

实时语音助手必须快速完成很多事：

1. 听到用户。实时 speech tokenization，并用 voice activity detection（VAD）判断用户何时说完。
2. 可选地看见。摄像头输入以 2-4 FPS 流式送入 Thinker，并与音频并行。
3. 思考。基于会话历史组合回答。
4. 说话。合成 audio tokens，解码成 waveform，流式传到用户扬声器。

每一步都会增加延迟。要有对话感，总 round-trip 必须 < 500ms；低于这个阈值，用户才不容易注意到卡顿。GPT-4o 声称约 250ms。Moshi 约 160ms。Qwen2.5-Omni 约 350-500ms。

每个组件都必须 streaming。不能“先 batch 全部内容再 decode”。

## 核心概念

### Thinker 与 Talker

Qwen2.5-Omni 的分解：

- Thinker：7B-80B 的文本生成 transformer。消费交错的 text + image + audio tokens。输出表示要说什么的 text tokens。
- Talker：更小的语音生成 transformer（200M-1B）。消费 Thinker 的 text output tokens 与最近的 speech-context tokens。输出离散 speech tokens（residual-VQ indices）。
- Speech decoder：一个 streaming waveform decoder（SNAC、MoVQGAN family），把 speech tokens 实时转换成 audio samples。

这种分离很重要。Thinker 必须足够大，才有好的 reasoning。Talker 可以很小，因为它的任务是局部的：把文本转换成 speech tokens。更大的 Talker 不一定更有表达力；它只是更慢。

二者并行运行：

1. Thinker 发出 text token t_i。
2. Talker 通过 streaming 消费 t_i，并发出 speech tokens s_i、s_{i+1}、...、s_{i+k}。
3. Speech decoder 消费陆续到来的 speech tokens，并发出 audio samples。
4. 当 Thinker 到达 text token t_{i+3} 时，Talker 已经为 t_0..t_{i+2} 流式输出了音频。

### TMRoPE：时间对齐的多模态位置

Thinker 需要整合 image frames（比如 4 FPS 到达）、audio frames（每秒 50 帧到达）和会话历史中的 text。朴素序列顺序（所有 image，再所有 audio，再 text）会丢失时间对齐。

TMRoPE 为每个 token 分配绝对时间戳。Vision token 在 t=2.3s。Audio token 在 t=2.32s。用户说出的 text token “stop” 在 t=2.35s。RoPE 按时间戳旋转 attention；模型会把它们看作时间上并发。

这是让“他一边挥手一边说 hello”成立的基础设施：模型能在同一个概念时刻看到视频帧和音频。

### Streaming speech synthesis

Speech tokens 必须 streaming。Mini-Omni（Xie & Wu, 2024）提出“language models can hear, talk while thinking in streaming”：Thinker output tokens 与 Talker output tokens 在同一序列中交错。Thinker 一提交下一个 text token，Talker 就启动。没有 batch 边界。

Moshi（Défossez et al., 2024 年 10 月）是最快的开放实现。在单张 A100 上 TTFAB 为 160ms。架构：单个 7B transformer，在交替位置发出 text 与 speech tokens，并通过“inner monologue”把 thinking stream 与 speaking stream 分离。这本质上是把 Thinker + Talker 融成一个模型，再配合细致训练。

### VAD 与 turn-taking

Voice activity detection 在输入侧运行。两种模式：

- Half-duplex：用户说话，模型倾听。模型说话，用户倾听。通过 VAD silence detection（约 200ms）完成清晰交接。
- Full-duplex：双方可以同时说话。模型可以 backchannel（“uh-huh”）或打断。难很多。Moshi 支持这一点。

Qwen2.5-Omni 默认支持 half-duplex，通过 silence threshold 做 turn-taking。Full-duplex 需要应用层处理。

### Qwen3-Omni（2025 年 11 月）

后继版本。Qwen3-80B Thinker、更大的 Talker、改进的 TMRoPE-v2。延迟接近 GPT-4o 的 250ms。开放权重。在 OmniBench 上与 Gemini 2.0 Live 有竞争力。

### 生产 latency budget

对于一次典型 streaming 交互：

- Mic -> audio tokens：40-80ms。
- Prefill（prompt + history）：7B 时 100-200ms，70B 时长得多。
- 第一个 Thinker text token：40ms。
- Talker 处理第一个 text token：20ms。
- 第一个 speech tokens commit：40ms。
- Residual-VQ decode：30ms。
- Speech waveform decode：50-80ms。

总 TTFAB：7B 时 320-510ms，70B 时 600-900ms。Frontier quality 通常意味着 70B+；这就是 frontier latency gap 的来源。

### Token-rate 数学

对于 16kHz 语音和 50 Hz base speech tokens，你每秒输出需要 50 个 speech tokens。Talker 必须发出 ≥50 tok/s 才能跟上。在 H100 上，典型 LLM throughput 为 30-80 tok/s，一个小型（200-300M）Talker 足够快；7B Talker 会落后。

这就是为什么存在小型专用 Talker，而不是“直接用主模型”。

## 实际使用

`code/main.py`：

- 用 mock token-emission rate 模拟 Thinker-Talker pipeline。
- 为可配置的模型尺寸和 mic sample rate 计算 TTFAB。
- 用 VAD silence threshold 演示 half-duplex turn-taking。

## 交付成果

本课产出 `outputs/skill-omni-streaming-budget.md`。给定实时语音产品的目标 TTFAB 和功能集（vision-in、bilingual、full-duplex），它会选择 Qwen2.5-Omni、Qwen3-Omni、Moshi 或 Mini-Omni，并确定 Thinker/Talker 尺寸。

## 练习

1. 你的目标 TTFAB 是 300ms。在 7B Thinker 与 300M Talker 上，写出每个组件的延迟。

2. Qwen2.5-Omni 使用 TMRoPE。描述当用户从 t=1s 开始说话、摄像头在 t=1.2s 捕捉到一个手势时，模型看到的内容。

3. Full-duplex 支持要求模型边听边输出音频。提出一种能教会它这一点的训练数据格式。

4. 阅读 Moshi 论文第 4 节。描述“inner monologue”分离，以及它为什么避免了 Thinker-Talker 拆分。

5. 计算 throughput budget：Talker 必须以多快速度发出 tokens，才能跟上 16kHz 语音在 50 base-layer tokens/sec 下的输出？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Thinker | “Reasoning brain” | 生成“要说什么”的大型文本生成 transformer |
| Talker | “Speech-generating mouth” | 从 Thinker 文本生成离散 speech tokens 的小型 transformer |
| TTFAB | “Latency budget” | Time-to-first-audio-byte：从用户语音结束到第一个 audio sample 输出 |
| TMRoPE | “Time-aligned RoPE” | 在 vision、audio、text 上使用绝对时间戳的 position encoding |
| Half-duplex | “Turn-taking” | 用户与模型轮流说话；VAD silence 检测用户说完 |
| Full-duplex | “Simultaneous” | 模型可以同时说话和倾听；支持 backchannel |
| Inner monologue | “Moshi separation” | thinking-stream 与 speaking-stream 在单模型中交错的设计 |

## 延伸阅读

- [Xu et al. — Qwen2.5-Omni (arXiv:2503.20215)](https://arxiv.org/abs/2503.20215)
- [Qwen Team — Qwen3-Omni (arXiv:2509.17765)](https://arxiv.org/html/2509.17765v1)
- [Xie & Wu — Mini-Omni (arXiv:2408.16725)](https://arxiv.org/abs/2408.16725)
- [Défossez et al. — Moshi (arXiv:2410.00037)](https://arxiv.org/abs/2410.00037)
- [Zeng et al. — GLM-4-Voice (arXiv:2412.02612)](https://arxiv.org/abs/2412.02612)
