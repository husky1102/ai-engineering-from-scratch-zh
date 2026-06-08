# 神经音频编解码器：EnCodec、SNAC、Mimi、DAC 与语义-声学拆分

> 2026 年的音频生成几乎全是 tokens。EnCodec、SNAC、Mimi 和 DAC 把连续 waveforms 转成 transformer 可以预测的离散 sequences。semantic-vs-acoustic token split，也就是 first-codebook as semantic、rest as acoustic，是音频领域自 Transformer 以来最重要的架构变化。

**类型：** Learn
**语言：** Python
**先修：** Phase 6 · 02 (Spectrograms), Phase 10 · 11 (Quantization), Phase 5 · 19 (Subword Tokenization)
**时间：** ~60 minutes

## 要解决的问题

语言模型处理离散 tokens。音频是连续的。如果你想为 speech / music 构建 LLM-style model，比如 MusicGen、Moshi、Sesame CSM、VibeVoice、Orpheus，你首先需要一个 **neural audio codec**：一个把音频离散化成小词表 tokens 的 learned encoder，以及一个匹配的 decoder 用来重建 waveform。

已经出现两大家族：

1. **Reconstruction-first codecs** - EnCodec、DAC。优化 perceptual audio quality。Tokens 是 "acoustic" 的：它们捕获一切，包括 speaker identity、timbre、background noise。
2. **Semantic-first codecs** - Mimi（Kyutai）、SpeechTokenizer。强制第一个 codebook 编码 linguistic / phonetic content（常通过 WavLM 蒸馏）。后续 codebooks 是 acoustic detail。

2024-2026 年的洞见：**当你试图从文本生成时，纯 reconstruction codec 会给出模糊语音。** codec tokens 上的 LLM 必须在同一个 codebook 中同时学习 language structure 和 acoustic structure，这无法扩展。把它们分离，也就是 semantic codebook 0、acoustic codebooks 1-N，正是 Moshi 和 Sesame CSM 得以工作的原因。

## 核心概念

![Four codec landscape: EnCodec, DAC, SNAC (multi-scale), Mimi (semantic+acoustic)](../assets/codec-comparison.svg)

### 核心技巧：Residual Vector Quantization (RVQ)

不是使用一个巨大 codebook（要获得好质量会需要数百万 codes），所有现代 audio codecs 都使用 **RVQ**：一串小 codebooks。第一个 codebook 量化 encoder output；第二个量化 residual；依此类推。每个 codebook 有 1024 codes。8 个 codebooks = 1024^8 = 10^24 的有效词表。

Inference 时，decoder 对每个 frame 中选中的所有 codes 求和来重建。

### 2026 年重要的四个 codecs

**EnCodec（Meta，2022）。** Baseline。Waveform 上的 encoder-decoder，RVQ bottleneck。24 kHz，可用 32 codebooks，默认 4 codebooks @ 1.5 kbps。使用 `1D conv + transformer + 1D conv` 架构。MusicGen 使用它。

**DAC（Descript，2023）。** RVQ with L2-normalized codebooks、periodic activation functions、improved losses。所有开源 codec 中 reconstruction fidelity 最高，有时 12 codebooks 下的语音与原始语音难以区分。44.1 kHz full-band。

**SNAC（Hubert Siuzdak，2024）。** Multi-scale RVQ，粗 codebooks 的 frame rate 低于细 codebooks。实际上分层建模音频：约 12 Hz 的粗 "sketch" 加 50 Hz 的细节。Orpheus-3B 使用它，因为这种分层结构很适合 LM-based generation。

**Mimi（Kyutai，2024）。** 2026 年的 game-changer。12.5 Hz frame rate（极低），8 codebooks @ 4.4 kbps。Codebook 0 是 **从 WavLM 蒸馏而来**，训练目标是预测 WavLM 的 speech-content features。Codebooks 1-7 是 acoustic residuals。这个拆分支撑 Moshi（Lesson 15）和 Sesame CSM。

### Frame rates 对语言建模很重要

更低 frame rate = 更短 sequence = 更快 LM。

| Codec | Frame rate | 1 s = N frames | Good for |
|-------|-----------|----------------|---------|
| EnCodec-24k | 75 Hz | 75 | music, general audio |
| DAC-44.1k | 86 Hz | 86 | high-fidelity music |
| SNAC-24k (coarse) | ~12 Hz | 12 | AR-LM efficient |
| Mimi | 12.5 Hz | 12.5 | streaming speech |

在 12.5 Hz 下，10 秒 utterance 只有 125 codec frames，transformer 可以轻松预测。

### Semantic vs acoustic tokens

```text
frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]
```

- **Semantic token（Mimi 中的 codebook 0）。** 编码说了什么：phonemes、words、content。通过 auxiliary prediction loss 从 WavLM 蒸馏而来。
- **Acoustic tokens（codebooks 1-7）。** 编码 timbre、speaker identity、prosody、background noise、fine detail。

AR LM 先预测 semantic token（以 text 为条件），再预测 acoustic tokens（以 semantic + speaker reference 为条件）。这个 factorization 解释了为什么现代 TTS 可以 zero-shot-clone voices：semantic model 处理 content；acoustic model 处理 timbre。

### 2026 reconstruction quality（bits per sec，bitrate 越低越好）

| Codec | Bitrate | PESQ | ViSQOL |
|-------|---------|------|--------|
| Opus-20kbps | 20 kbps | 4.0 | 4.3 |
| EnCodec-6kbps | 6 kbps | 3.2 | 3.8 |
| DAC-6kbps | 6 kbps | 3.5 | 4.0 |
| SNAC-3kbps | 3 kbps | 3.3 | 3.8 |
| Mimi-4.4kbps | 4.4 kbps | 3.1 | 3.7 |

像 Opus 这样的传统 codecs 在 per bit perceptual quality 上仍然胜出。Neural codecs 胜在 **discrete tokens**（Opus 不产生）和 **generative-model quality**（LM 能用这些 tokens 做什么）。

## 动手实现

### Step 1：用 EnCodec encode

```python
from encodec import EncodecModel
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # kbps

wav = torch.randn(1, 1, 24000)
with torch.no_grad():
    encoded = model.encode(wav)
codes, scale = encoded[0]
# codes: (1, n_codebooks, n_frames), dtype=int64
```

6 kbps 下 `n_codebooks=8`。每个 code 为 0-1023（10-bit）。

### Step 2：decode 并测量 reconstruction

```python
with torch.no_grad():
    wav_recon = model.decode([(codes, scale)])

from torchaudio.functional import compute_deltas
import torch.nn.functional as F

mse = F.mse_loss(wav_recon[:, :, :wav.shape[-1]], wav).item()
```

### Step 3：semantic-acoustic split（Mimi-style）

```python
from moshi.models import loaders
mimi = loaders.get_mimi()

with torch.no_grad():
    codes = mimi.encode(wav)  # shape (1, 8, frames@12.5Hz)

semantic = codes[:, 0]
acoustic = codes[:, 1:]
```

Semantic codebook 0 与 WavLM 对齐。你可以训练一个 text-to-semantic transformer，它的词表比 direct-to-audio 小得多。然后一个单独的 acoustic-to-waveform decoder 以 speaker reference 为条件。

### Step 4：为什么 codec tokens 上的 AR LM 能工作

对于一段 10 s speech clip，在 Mimi 的 12.5 Hz × 8 codebooks 下：

```text
N_tokens = 10 * 12.5 * 8 = 1000 tokens
```

1000 tokens 对 transformer 来说只是很小的 context。现代 GPU 上，一个 256M-parameter transformer 可以在毫秒级生成 10 秒语音。

## 实际使用

把问题映射到 codec：

| 任务 | Codec |
|------|-------|
| General music generation | EnCodec-24k |
| Highest-fidelity reconstruction | DAC-44.1k |
| AR LM over speech (TTS) | SNAC 或 Mimi |
| Streaming full-duplex speech | Mimi (12.5 Hz) |
| Sound-effect library with text | EnCodec + T5 condition |
| Fine-grained audio editing | DAC + inpainting |

经验法则：**如果你在构建 generative model，从 Mimi 或 SNAC 开始。如果你在构建 compression pipeline，使用 Opus。**

## 常见陷阱

- **Codebooks 太多。** 增加 codebooks 会线性提高 fidelity，但也会线性增加 LM sequence length。停在 8-12。
- **Frame-rate mismatch。** 在 12.5 Hz Mimi 上训练 LM，然后在 50 Hz EnCodec 上 fine-tune，会静默失败。
- **假设所有 codebooks 相等。** 在 Mimi 中，codebook 0 承载 content；丢掉它会摧毁可懂度。丢掉 codebook 7 几乎不可察觉。
- **把 reconstruction quality 当成唯一指标。** 如果 semantic structure 很差，一个 codec 即使 reconstruction 很棒，也可能对 LM-based generation 没用。

## 交付成果

保存为 `outputs/skill-codec-picker.md`。为给定 generative 或 compression task 选择 codec。

## 练习

1. **Easy。** 运行 `code/main.py`。它实现一个 toy scalar + residual quantizer，并测量增加 codebooks 时的 reconstruction error。
2. **Medium。** 安装 `encodec`，在 held-out speech clip 上比较 1、4、8、32 codebooks。绘制 PESQ 或 MSE vs bitrate。
3. **Hard。** 加载 Mimi。Encode 一个 clip。把 codebook 0 替换为随机 integers；decode。然后同样替换 codebook 7。比较两种 corruption：codebook 0 corruption 应该摧毁 intelligibility；codebook 7 corruption 应该几乎不改变任何东西。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| RVQ | Residual quantization | 一串小 codebooks；每个量化前一个 residual。 |
| Frame rate | Codec speed | 每秒多少 token-frames。更低 = 更快 LM。 |
| Semantic codebook | Codebook 0 (Mimi) | 从 SSL features 蒸馏而来的 codebook；编码 content。 |
| Acoustic codebooks | 其他全部 | Timbre、prosody、noise、fine detail。 |
| PESQ / ViSQOL | Perceptual quality | 与 MOS 相关的 objective metrics。 |
| EnCodec | Meta codec | RVQ baseline；MusicGen 使用。 |
| Mimi | Kyutai codec | 12.5 Hz frame rate；semantic-acoustic split；支撑 Moshi。 |

## 延伸阅读

- [Défossez et al. (2023). EnCodec](https://arxiv.org/abs/2210.13438) - RVQ baseline。
- [Kumar et al. (2023). Descript Audio Codec (DAC)](https://arxiv.org/abs/2306.06546) - 最高保真开源。
- [Siuzdak (2024). SNAC](https://arxiv.org/abs/2410.14411) - multi-scale RVQ。
- [Kyutai (2024). Mimi codec](https://kyutai.org/codec-explainer) - semantic-acoustic split，WavLM distillation。
- [Borsos et al. (2023). AudioLM](https://arxiv.org/abs/2209.03143) - two-stage semantic/acoustic paradigm。
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) - original streamable RVQ codec。
