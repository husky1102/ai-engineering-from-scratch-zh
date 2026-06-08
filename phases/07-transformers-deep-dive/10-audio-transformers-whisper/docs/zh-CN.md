# Audio Transformers — Whisper Architecture

> 音频是一张“频率随时间变化”的图像。Whisper 是一个吃 mel spectrograms 并吐出文字的 ViT。

**类型:** Learn
**语言:** Python
**先修:** Phase 7 · 05 (Full Transformer), Phase 7 · 08 (Encoder-Decoder), Phase 7 · 09 (ViT)
**时间:** ~45 minutes

## 要解决的问题

在 Whisper（OpenAI, Radford et al. 2022）之前，state-of-the-art automatic speech recognition (ASR) 意味着 wav2vec 2.0 和 HuBERT——self-supervised feature extractors 加 fine-tuned head。质量高，但数据管线昂贵，且对 domain 脆弱。Multilingual speech recognition 需要按 language family 拆分模型。

Whisper 做了三个赌注：

1. **Train on everything.** 从互联网上抓取跨 97 种语言的 680,000 小时 weakly-labeled audio。没有干净 academic corpus。没有 phoneme labels。
2. **Multi-task single model.** 一个 decoder 通过 task tokens 联合训练 transcription、translation、voice activity detection、language ID 和 timestamping。
3. **Standard encoder-decoder transformer.** Encoder 消费 log-mel spectrograms。Decoder 自回归产生 text tokens。没有 vocoder、没有 CTC、没有 HMM。

结果：Whisper large-v3 对 accents、noise 和没有干净 labeled data 的语言都很鲁棒。它是 2026 年每个 open-source voice assistant 以及多数商业语音助手的默认 speech front-end。

## 核心概念

![Whisper pipeline: audio → mel → encoder → decoder → text](../assets/whisper.svg)

### Step 1 — resample + window

音频采样率 16 kHz。Clip/pad 到 30 seconds。计算 log-mel spectrogram：80 mel bins、10 ms stride → 约 3,000 frames × 80 features。这就是 Whisper 看到的“input image”。

### Step 2 — convolutional stem

两个 kernel 3、stride 2 的 Conv1D layers 把 3,000 frames 降到 1,500。在不增加太多参数的情况下把 sequence length 减半。

### Step 3 — encoder

一个 24-layer（large）transformer encoder，处理 1,500 timesteps。Sinusoidal positional encoding、self-attention、GELU FFN。产生 1,500 × 1,280 hidden states。

### Step 4 — decoder

一个 24-layer transformer decoder。它从 BPE vocabulary 自回归产生 tokens；该 vocabulary 是 GPT-2 vocabulary 的超集，并加入了一些 audio-specific special tokens。

### Step 5 — task tokens

Decoder prompt 以 control tokens 开始，告诉模型要做什么：

```text
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

或：

```text
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

模型按这个约定训练。你通过 prefix 控制 task。这是 2026 年 instruction-tuning 的语音版本。

### Step 6 — output

Beam search（width 5）加 log-prob threshold。当 `<|notimestamps|>` token 不存在时，timestamps 会按每 0.02 seconds audio 预测一次。

### Whisper sizes

| Model | Params | Layers | d_model | Heads | VRAM (fp16) |
|-------|--------|--------|---------|-------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB (4-layer decoder) |

Large-v3-turbo (2024) 把 decoder 从 32 layers 削到 4。Decoding 快 8×，WER 回退小于 1 point。这种 decode speed unlock 正是 Whisper-turbo 成为 2026 年 real-time voice agents 默认选择的原因。

### Whisper 不做什么

- 不做 diarization（谁在说话）。这件事要搭配 pyannote。
- 原生不做 real-time streaming——30-second window 是固定的。现代 wrappers（`faster-whisper`、`WhisperX`）通过 VAD + overlap 外接 streaming。
- 如果没有外部 chunking，就没有超过 30 s 的 long-form context。实践中效果很好，因为人类语音转写很少需要长程 context。

### 2026 landscape

| Task | Model | Notes |
|------|-------|-------|
| English ASR | Whisper-turbo, Moonshine | Moonshine is 4× faster on edge |
| Multilingual ASR | Whisper-large-v3 | 97 languages |
| Streaming ASR | faster-whisper + VAD | 150 ms latency targets achievable |
| TTS | Piper, XTTS-v2, Kokoro | Encoder-decoder pattern, but Whisper-shaped |
| Audio + language | AudioLM, SeamlessM4T | Text tokens + audio tokens in one transformer |

## 动手实现

见 `code/main.py`。我们不训练 Whisper——我们构建 log-mel spectrogram pipeline + task-token prompt formatter。这些才是你在生产中实际会触碰的部分。

### Step 1: synthesize audio

生成 1-second、440 Hz、16 kHz 采样的 sine wave。16,000 samples。

### Step 2: log-mel spectrogram (simplified)

完整 mel spectrogram 需要 FFT。我们做一个简化 framing + per-frame energy 版本，在不依赖 `librosa` 的情况下展示 pipeline：

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

Frame = 25 ms，hop = 10 ms。匹配 Whisper 的 windowing。Per-frame energy 在教学上代替 mel bins。

### Step 3: pad to 30 s

Whisper 总是处理 30-second chunks。把 spectrogram pad（或 clip）到 3,000 frames。

### Step 4: build the prompt tokens

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

这就是完整 task-control surface。一个 4-token prefix。

## 实际使用

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

更快、OpenAI-compatible：

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**2026 年何时选择 Whisper：**

- 用一个模型做 multilingual ASR。
- 对 noisy、diverse audio 做 robust transcription。
- 研究 / prototype ASR——最快起点。

**何时选择其他方案：**

- Edge 上 ultra-low latency streaming——匹配质量下 Moonshine 胜过 Whisper。
- 需要 <200 ms 的 real-time conversational AI——使用 dedicated streaming ASR。
- Speaker diarization——Whisper 不做这个；外接 pyannote。

## 交付成果

见 `outputs/skill-asr-configurator.md`。这个 skill 会为新的 speech application 选择 ASR model、decoding parameters 和 preprocessing pipeline。

## 练习

1. **Easy.** 运行 `code/main.py`。确认 16 kHz、10 ms hop 的 1-second signal frame count 约为 100。30 seconds 约为 3,000 frames。
2. **Medium.** 使用 `numpy.fft` 构建完整 log-mel spectrogram。验证 80 mel bins 与 `librosa.feature.melspectrogram(n_mels=80)` 在 numerical error 内匹配。
3. **Hard.** 实现 streaming inference：把 audio 切成 10 s windows，2 s overlap，对每个 chunk 运行 Whisper，合并 transcripts。在 5-minute podcast sample 上测量相对 single-pass 的 word-error rate。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Mel spectrogram | “Audio image” | 2D 表示：一个轴是 frequency bins，另一个轴是 time frames；每个 cell 是 log-scaled energy。 |
| Log-mel | “Whisper 看到的东西” | 经过 log 的 mel spectrogram；近似人类对 loudness 的感知。 |
| Frame | “One time slice” | 25 ms samples window；以 10 ms stride 重叠。 |
| Task token | “Prompt prefix for speech” | Decoder prompt 中 `<\|transcribe\|>` / `<\|translate\|>` 这类 special tokens。 |
| Voice activity detection (VAD) | “Find the speech” | 移除 silence 的 gate；大幅降低成本。 |
| CTC | “Connectionist Temporal Classification” | 经典 ASR loss，用于 alignment-free training；Whisper 不使用它。 |
| Whisper-turbo | “Small decoder, full encoder” | large-v3 encoder + 4-layer decoder；decoding 快 8×。 |
| Faster-whisper | “The production wrapper” | CTranslate2 reimplementation；int8 quantization；比 OpenAI reference 快 4×。 |

## 延伸阅读

- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper 论文。
- [OpenAI Whisper repo](https://github.com/openai/whisper) — reference code + model weights。阅读 `whisper/model.py`，可以在约 400 行内自顶向下看到 Conv1D stem + encoder + decoder。
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — Steps 5–6 中描述的 beam-search + task-token logic 在这里；500 行，完全可读。
- [Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) — 前身；在某些设置中仍是 SOTA features。
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — production wrapper，比 reference 快 4×。
- [Jia et al. (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) — 2024 年 edge-friendly ASR，Whisper-shaped 但更小。
- [HuggingFace blog — "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper) — canonical fine-tuning recipe，包含 mel spectrogram preprocessor 和 token-timestamp handling。
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — 完整实现（encoder、decoder、cross-attention、generation），对应本课 architecture diagram。
