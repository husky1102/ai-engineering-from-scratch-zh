# Audio-Language Models：从 Whisper 到 Audio Flamingo 3 的弧线

> Whisper（Radford et al., 2022 年 12 月）基本定型了语音识别：68 万小时弱监督多语言语音、一个简单的 encoder-decoder transformer，以及一个让后续每个 ASR 发布都必须引用它的 benchmark。但识别不是推理。问“这段录音里有哪些乐器”、“说话者表达了什么情绪”，或者“第 3 分钟发生了什么”，需要的是音频理解，不只是转录。Qwen-Audio、SALMONN、LTU 和 NVIDIA 的 Audio Flamingo 3（AF3，2025 年 7 月）逐步搭起了这套 stack：保留 Whisper 级 encoder，接上 Q-former，用 audio-text instruction data 训练，再加入 chain-of-thought reasoning。本课会走读这条弧线。

**类型:** Build
**语言:** Python（stdlib，log-Mel spectrogram + audio Q-former skeleton）
**先修:** Phase 6（Speech and Audio），Phase 12 · 03（Q-Former）
**时间:** ~180 分钟

## 学习目标

- 从 waveform 计算 log-Mel spectrogram：windowing、FFT、filter banks、log transform。
- 比较 encoder 选项：Whisper encoder、BEATs、AF-Whisper hybrid。说明各自什么时候胜出。
- 构建 audio Q-former：N 个可学习 query 对 spectrogram patch 做 cross-attention。
- 解释 cascaded（Whisper-then-LLM）与 end-to-end audio-LLM training：为什么 end-to-end 更适合扩展到推理。

## 要解决的问题

语音识别已经被 Whisper 解决。OCR-of-audio 成了 commodity。但“commodity”只到转录为止。如果模型无法对听到的内容推理，包括时间、说话者、情绪、音乐结构、环境声音，光有转录无法驱动产品功能。

三条明显路线：

1. Cascade：Whisper 转录，LLM 基于 transcript 推理。适合纯语音场景。遇到音乐、环境音、多说话者重叠、情绪就会失败。

2. End-to-end audio-LLM：audio encoder 直接把 audio tokens 喂给 LLM，跳过转录。保留声学信息（情绪、说话者、环境）。需要新的训练数据。

3. Hybrid：audio encoder + text decoder，既能转录又能推理。Qwen-Audio 与 Audio Flamingo 选择这条路。

## 核心概念

### Log-Mel spectrogram：输入特征

每个 audio encoder 都从同一种特征开始：log-Mel spectrogram。

1. 重采样到 16 kHz。
2. 用 25ms window、10ms hop 做 short-time Fourier transform。
3. 取 FFT 结果的 magnitude。
4. 应用 Mel filter banks（通常是 80 个按 0-8000 Hz log 间隔排布的 filter），映射到感知频率。
5. 做 log compression（log(1 + x)）以压缩动态范围。

结果：形状为 (T, 80) 的 2D array，其中 T 是时间帧数。对于 100 Hz frame rate 下的 30 秒 clip：形状是 (3000, 80)。

### Whisper 的 encoder

Whisper 的 encoder 是一个 12 层 ViT-style transformer，把 log-Mel spectrogram 当作时间帧序列处理。输出：每个时间帧一个 hidden-state vector。

对于 ASR，Whisper 的 decoder 是一个 cross-attention transformer，在 encoder output 条件下生成 text tokens。标准 encoder-decoder。

对于 ALM（audio-LLM），你希望把 encoder output 作为另一个 LLM 的输入。模式是：冻结 Whisper encoder，训练 Q-former，LLM 冻结或微调。

### BEATs 与音频专用 encoder

Whisper 训练数据以语音为主。它在音乐和环境音上较弱。

BEATs（Chen et al., 2022）是在 AudioSet 上训练的 self-supervised transformer。在同等参数量下，它比 Whisper 更擅长捕捉音乐和环境声音。

AF-Whisper（Audio Flamingo 3 的 hybrid）：把 Whisper + BEATs 特征 concat 成音频输入。Whisper 承载语言信号，BEATs 承载声学信号。

### Audio Q-former

与 BLIP-2 的 visual Q-former 是同一个模式。固定数量的可学习 query（常见为 32 或 64 个）对 audio encoder 的输出帧做 cross-attention。这些 query 变成供 LLM 消费的 audio tokens。

训练 alignment stage：只训练 Q-former，在 audio-text pair（AudioCaps、Clotho）上用 contrastive + captioning loss。Instruction stage：端到端，解冻 LLM，在 instruction data 上训练。

### 这条弧线：SALMONN、Qwen-Audio、AF3

SALMONN（Tang et al., 2023）：Whisper + BEATs + Q-former + LLaMA。第一个具备认真推理能力的开放 audio-LLM。在 MMAU benchmark 上 composite 约 0.55。

Qwen-Audio（Chu et al., 2023）：架构类似，训练数据更丰富，并面向多轮对话调优。MMAU 约 0.60。

LTU — Listen, Think, Understand（Gong et al., 2023）：显式 reasoning data，关注音频 clip 上的 chain-of-thought。更小但更聚焦。

Audio Flamingo 3（Goel et al., 2025 年 7 月）：当前开放 SOTA。8B LLM backbone（Qwen2 7B）、Whisper-large encoder concat BEATs、64-query Q-former，在 100 万以上 audio-text instruction pair 上训练。MMAU 0.72，在一些子任务上追平 proprietary frontier。

AF3 还引入了 audio 的 on-demand chain-of-thought：模型可以选择性地在 final answer 之前输出 thinking tokens（“let me identify the instruments first: ...”）。开启 thinking 后，复杂推理任务准确率提升 3-5 个点。

### Cascaded vs end-to-end

Cascaded pipeline：

1. Whisper 转录 audio → text。
2. LLM 基于 text 推理。

对于“summarize this podcast”这类任务非常好。失败场景：
- “What's the mood of this song?” —— mood 在声音里，不在文字里。
- “Who is speaking, Alice or Bob?” —— 需要 speaker identification。
- “At what second does the explosion happen?” —— 文本里丢失了 temporal grounding。
- “Is this real or generated audio?” —— deepfake detection 需要声学特征。

End-to-end 保留声学信号。Qwen-Audio 和 AF3 可以原生处理音乐、环境与情绪。

### 2026 年生产配方

对于新的音频理解产品：

- 如果目标是转录、没有音乐、没有情绪推断：用 cascaded。
- 如果涉及音乐、情绪、多说话者或复杂音频推理：用 AF3 / Qwen-Audio-family。

Cascaded 更便宜、更简单。End-to-end 能力更强。

### MMAU：音频推理 benchmark

MMAU（Massive Multimodal Audio Understanding）是 2024-2025 年的音频推理 benchmark：

- 10000 个 audio-text QA pair，覆盖语音、音乐、环境声音。
- 覆盖 classification、temporal reasoning、causal reasoning、open-ended QA。
- 测试 cascaded pipeline 系统性漏掉的内容。

开放 SOTA（AF3）为 0.72；proprietary frontier 约 0.78（Gemini 2.5 Pro、Claude Opus 4.7）。这个差距小于 VideoMME 的 open-vs-closed delta，说明 audio-LLM 正在成熟。

## 实际使用

`code/main.py`：

- 用 stdlib 实现 log-Mel spectrogram 计算：windowing、naive DFT、Mel filter-bank。
- Audio Q-former skeleton：给定 encoder output frame，计算 Q、K、V、attention，并发出 N 个 token。
- 在 toy task 上比较 cascaded 与 end-to-end。

## 交付成果

本课产出 `outputs/skill-audio-llm-pipeline-picker.md`。给定一个音频任务（transcription、music tagging、emotion inference、multi-speaker diarization、environment classification），它会选择 cascaded、end-to-end AF3 或 hybrid。

## 练习

1. 对一个 16kHz、25ms window、10ms hop、80 个 Mel bin 的 30 秒 clip，计算 log-Mel spectrogram 维度。48kHz 时会怎样变化？

2. 为什么 Whisper 在音乐上表现较差？BEATs 捕捉了哪些 Whisper 没有捕捉的音频特征？

3. 64 个 query 的 audio Q-former 与 32 个 query 相比：什么任务复杂度下 64 值得？32 又为哪些任务节省计算？

4. 阅读 AF3 第 4 节关于 on-demand thinking 的内容。提出三类 chain-of-thought 最有帮助的音频任务。

5. 使用 AF3 的输出实现一个最小 diarization pipeline。你如何标记 speaker changes？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Log-Mel spectrogram | “Mel features” | Mel filter banks 之后的 log-magnitude value 组成的 2D（time, frequency）array |
| Audio Q-former | “Audio Perceiver” | 从 audio encoder output 到固定长度 query、再喂给 LLM 的 cross-attention bottleneck |
| Cascaded | “ASR-then-LLM” | Whisper 先转录、text LLM 再推理的 pipeline；会丢失声学信息 |
| End-to-end | “Audio-LLM” | 音频特征通过 Q-former 直接进入 LLM；保留声学信号 |
| BEATs | “Audio AudioSet encoder” | 在 AudioSet 上训练的 SSL transformer；擅长音乐 + 环境声音 |
| MMAU | “Audio reasoning bench” | 覆盖语音、音乐、环境的 10k QA pair；2024 eval standard |
| On-demand thinking | “Audio CoT” | 模型可以在 final answer 前选择性输出 reasoning tokens，使准确率提升 3-5 点 |

## 延伸阅读

- [Radford et al. — Whisper (arXiv:2212.04356)](https://arxiv.org/abs/2212.04356)
- [Chu et al. — Qwen-Audio (arXiv:2311.07919)](https://arxiv.org/abs/2311.07919)
- [Goel et al. — Audio Flamingo 3 (arXiv:2507.08128)](https://arxiv.org/abs/2507.08128)
- [Tang et al. — SALMONN (arXiv:2310.13289)](https://arxiv.org/abs/2310.13289)
- [Gong et al. — LTU (arXiv:2305.10790)](https://arxiv.org/abs/2305.10790)
