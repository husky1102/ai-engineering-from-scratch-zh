# 文本转语音（TTS）：从 Tacotron 到 F5 和 Kokoro

> ASR 把语音反转成文本；TTS 把文本反转成语音。2026 年的栈分为三段：text → tokens、tokens → mel、mel → waveform。每一段都有能在笔记本上运行的默认模型。

**类型：** Build
**语言：** Python
**先修：** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 09 (Seq2Seq), Phase 7 · 05 (Full Transformer)
**时间：** ~75 minutes

## 要解决的问题

你有一个字符串："Please remind me to water the plants at 6 pm." 你需要生成一段 3 秒音频：听起来自然，有正确的韵律（停顿、重音），把 "plants" 中的元音读对，并且能在 CPU 上 300 ms 内运行，以支持实时语音助手。你还需要切换声音、处理代码混合输入（"remind me at 6 pm, daijoubu?"），并且不要在姓名发音上出丑。

现代 TTS pipeline 大致如下：

1. **文本前端。** 规范化文本（日期、数字、email），转换成音素或 subword tokens，预测韵律特征。
2. **声学模型。** Text → mel spectrogram。Tacotron 2 (2017)、FastSpeech 2 (2020)、VITS (2021)、F5-TTS (2024)、Kokoro (2024)。
3. **Vocoder。** Mel → waveform。WaveNet (2016)、WaveRNN、HiFi-GAN (2020)、BigVGAN (2022)、2024+ 的 neural codec vocoders。

到 2026 年，acoustic + vocoder 的分界已经被端到端 diffusion 和 flow-matching 模型模糊了。但三段式心智模型仍然适合调试。

## 核心概念

![Tacotron, FastSpeech, VITS, F5/Kokoro side-by-side](../assets/tts.svg)

**Tacotron 2 (2017)。** Seq2seq：char-embedding → BiLSTM encoder → location-sensitive attention → autoregressive LSTM decoder 输出 mel frames。慢（AR），长文本上容易抖。今天仍常被引用为 baseline。

**FastSpeech 2 (2020)。** 非自回归。Duration predictor 输出每个 phoneme 分配多少 mel frames。一次前向，速度比 Tacotron 快 10×。损失一部分自然度（monotonic alignment），但到处都能上线。

**VITS (2021)。** 使用 variational inference，把 encoder + flow-based duration + HiFi-GAN vocoder 端到端联合训练。质量高，单模型。2022-2024 年主导开源 TTS。变体：YourTTS（multi-speaker zero-shot）、XTTS v2（2024，Coqui）。

**F5-TTS (2024)。** 基于 flow matching 的 diffusion transformer。自然韵律，5 秒参考音频即可 zero-shot voice cloning。位于 2026 开源 TTS leaderboard 顶部。335M params。

**Kokoro (2024)。** 小模型（82M），CPU 可运行，实时场景下同类最佳 English TTS。封闭词表、仅英文，apache-2.0。

**OpenAI TTS-1-HD、ElevenLabs v2.5、Google Chirp-3。** 商业 state of the art。ElevenLabs v2.5 的情绪标签（"[whispered]"、"[laughing]"）和角色声音在 2026 年主导有声书制作。

### Vocoder 演进

| 时代 | Vocoder | 延迟 | 质量 |
|-----|---------|---------|---------|
| 2016 | WaveNet | 只能离线 | 发布时 SOTA |
| 2018 | WaveRNN | ~realtime | 好 |
| 2020 | HiFi-GAN | 100× realtime | 接近人类 |
| 2022 | BigVGAN | 50× realtime | 跨说话人/语言泛化 |
| 2024 | SNAC, DAC (neural codecs) | 与 AR models 集成 | 离散 tokens，bit-efficient |

到 2026 年，大多数 "TTS" 模型都是从文本到 waveform 的端到端模型；mel spectrogram 是内部表示。

### 评估

- **MOS (Mean Opinion Score)。** 1-5 分，众包标注。仍是黄金标准；慢得让人痛苦。
- **CMOS (Comparative MOS)。** A-vs-B 偏好。每条标注的置信区间更紧。
- **UTMOS、DNSMOS。** 无参考的神经 MOS 预测器。用于 leaderboards。
- **CER (Character Error Rate) via ASR。** 将 TTS 输出送入 Whisper，计算相对输入文本的 CER。是可懂度的 proxy。
- **SECS (Speaker Embedding Cosine Similarity)。** voice-cloning 质量指标。

LibriTTS test-clean 上的 2026 数字：

| Model | UTMOS | CER (via Whisper) | Size |
|-------|-------|-------------------|------|
| Ground truth | 4.08 | 1.2% | - |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |
| Parler-TTS Large | 3.76 | 2.8% | 2.3B |

## 动手实现

### Step 1：phonemize 输入

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

Phonemes 是通用桥梁。不要把 raw text 喂给低于 VITS 级别质量的东西。

### Step 2：运行 Kokoro（2026 CPU 默认选择）

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = American English
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 tensor, sr=24000
```

离线运行，单文件，82M params。

### Step 3：用 F5-TTS 做 voice cloning

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

传入 5 秒参考片段 + 它的转录；F5 会克隆韵律和音色。

### Step 4：从零理解 HiFi-GAN vocoder

太大了，不适合塞进一份 tutorial script，但形状是：

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 upsample blocks, total 256x to go from mel-rate to audio-rate
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> waveform
```

训练：adversarial（短窗口上的 discriminator）+ mel-spectrogram reconstruction loss + feature-matching loss。已经商品化了，使用 `hifi-gan` repo 或 nvidia-NeMo 的 pretrained checkpoints。

### Step 5：完整 pipeline（pseudocode）

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## 实际使用

2026 年的栈：

| 场景 | 选择 |
|-----------|------|
| 实时英文语音助手 | Kokoro (CPU) 或 XTTS v2 (GPU) |
| 5 s 参考音频 voice cloning | F5-TTS |
| 商业角色声音 | ElevenLabs v2.5 |
| 有声书旁白 | ElevenLabs v2.5 或 XTTS v2 + fine-tune |
| 低资源语言 | 在 5-20 h 目标语言数据上训练 VITS |
| 表现力 / 情绪标签 | ElevenLabs v2.5 或 StyleTTS 2 fine-tune |

截至 2026 年的开源领先者：**F5-TTS 质量最佳，Kokoro 效率最佳**。除非你是历史学家，否则不要伸手去拿 Tacotron。

## 常见陷阱

- **没有 text normalizer。** "Dr. Smith" 读成 "Doctor" 还是 "Drive"？"2026" 读成 "twenty twenty six" 还是 "two zero two six"？在 phonemizer 之前规范化。
- **OOV proper nouns。** "Ghumare" → "ghyu-mair"？为 unknown tokens 发布 fallback grapheme-to-phoneme model。
- **Clipping。** Vocoder 输出很少 clipping，但 inference 时 mel scaling mismatch 可能超过 ±1.0。始终 `np.clip(wav, -1, 1)`。
- **Sample-rate mismatch。** Kokoro 输出 24 kHz；你的下游 pipeline 期望 16 kHz → resample，否则会 aliasing。

## 交付成果

保存为 `outputs/skill-tts-designer.md`。为给定声音、延迟和语言目标设计 TTS pipeline。

## 练习

1. **Easy。** 运行 `code/main.py`。它会从 toy vocab 构建 phoneme dictionary，估计每个 phoneme 的 duration，并打印一个假的 "mel" schedule。
2. **Medium。** 安装 Kokoro，用 voice `af_bella` 和 `am_adam` 合成同一句话。比较音频时长和主观质量。
3. **Hard。** 录制你自己的 5 秒参考片段。用 F5-TTS 克隆它。报告参考音频和克隆输出之间的 SECS。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Phoneme | 声音单位 | 抽象声音类别；英语中有 39 个（ARPABet）。 |
| Duration predictor | 每个 phoneme 持续多久 | 非 AR 模型输出；每个 phoneme 对应整数 frames。 |
| Vocoder | Mel → waveform | 把 mel-spec 映射到 raw samples 的神经网络。 |
| HiFi-GAN | 标准 vocoder | 基于 GAN；主导 2020-2024。 |
| MOS | 主观质量 | 来自人工评分者的 1-5 mean opinion score。 |
| SECS | Voice-clone metric | target 与 output speaker embedding 之间的 cosine similarity。 |
| F5-TTS | 2024 开源 SOTA | Flow-matching diffusion；zero-shot cloning。 |
| Kokoro | CPU English leader | 82M-param model, Apache 2.0。 |

## 延伸阅读

- [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) - seq2seq baseline。
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) - 端到端 flow-based。
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) - 当前开源 SOTA。
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) - 2026 年仍在上线的 vocoder。
- [Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) - 2024 CPU-friendly English TTS。
