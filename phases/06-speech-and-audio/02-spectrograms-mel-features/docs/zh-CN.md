# 频谱图、Mel 尺度与音频特征

> 神经网络不太擅长直接消费 raw waveform。它们消费 spectrogram。更进一步，它们更擅长消费 mel spectrogram。2026 年的每个 ASR、TTS 和 audio classifier，成败都取决于这个预处理选择。

**类型：** Build
**语言：** Python
**先修：** Phase 6 · 01（Audio Fundamentals）
**时间：** ~45 分钟

## 要解决的问题

拿一段 10 秒、16 kHz 的 clip。它有 160,000 个 float，全都在 `[-1, 1]` 中，和标签 “dog barking” 或 “the word cat” 几乎完全不相关。raw waveform 包含信息，但形式并不方便模型抽取。两个相同 phoneme 若相隔 100 ms 发出，其 raw sample 会完全不同。

Spectrogram 解决了这个问题。它会压缩人类感知忽略的时间细节（microsecond jitter），同时保留感知关注的结构（哪些频率有能量，以及它们在约 10-25 ms 时间窗上的变化）。

Mel spectrogram 进一步推进。人类以对数方式感知 pitch：100 Hz vs 200 Hz 听起来和 1000 Hz vs 2000 Hz 有“同样的距离”。mel scale 会扭曲 frequency axis 以匹配这种感知。从 2010 到 2026，mel-scaled spectrogram 一直是 speech ML 中最重要的单一特征。

## 核心概念

![Waveform to STFT to mel spectrogram to MFCC ladder](../assets/mel-features.svg)

**STFT（Short-Time Fourier Transform）。** 将 waveform 切成重叠 frame（典型值：25 ms window、10 ms hop = 16 kHz 下 400 samples / 160 samples）。给每个 frame 乘以 window function（Hann 是默认选择；Hamming 有略不同的 tradeoff）。对每个 frame 做 FFT。把 magnitude spectrum 堆叠成形状为 `(n_frames, n_freq_bins)` 的矩阵。这就是 spectrogram。

**Log-magnitude。** Raw magnitude 跨越 5-6 个数量级。取 `log(|X| + 1e-6)` 或 `20 * log10(|X|)` 来压缩 dynamic range。每条生产 pipeline 都使用 log-magnitude，而不是 raw magnitude。

**Mel scale。** Hz 中的频率 `f` 通过 `m = 2595 * log10(1 + f / 700)` 映射为 mel `m`。这个映射在 1 kHz 以下大致线性，在 1 kHz 以上大致对数。覆盖 0-8 kHz 的 80 个 mel bin 是标准 ASR input。

**Mel filterbank。** 一组三角形 filter，在 mel scale 上等间距排列。每个 filter 是相邻 FFT bin 的加权和。用 filterbank matrix 乘 STFT magnitude，就能通过一次 matmul 得到 mel spectrogram。

**Log-mel spectrogram。** `log(mel_spec + 1e-10)`。Whisper 的输入。Parakeet 的输入。SeamlessM4T 的输入。2026 年通用音频 frontend。

**MFCCs。** 对 log-mel spectrogram 应用 DCT（type II），保留前 13 个 coefficient。它会去相关特征并进一步压缩。在 CNN/Transformer 直接处理 raw log-mel 追上来之前，MFCC 一直是主导特征，约到 2015 年。它仍用于 speaker recognition（x-vectors、ECAPA）。

**Resolution trade。** 更大的 FFT = 更好的 frequency resolution，但更差的 time resolution。25 ms / 10 ms 是 audio-ML 默认；音乐常用 50 ms / 12.5 ms；瞬态检测（drum hit、plosive）常用 5 ms / 2 ms。

## 动手实现

### Step 1: frame the waveform

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

一段 10 秒、16 kHz 的 clip，在 `frame_len=400, hop=160` 时会产生 998 个 frame。

### Step 2: Hann window

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

在 FFT 之前做逐元素相乘。它会消除非零端点截断造成的 spectral leakage。

### Step 3: STFT magnitude

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

生产中使用 `torch.stft` 或 `librosa.stft`（FFT-backed、vectorized）。这里的 loop 是教学用；它可以在 `code/main.py` 的短 clip 上运行。

### Step 4: mel filterbank

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

`n_fft=400` 时，覆盖 0-8 kHz 的 80 个 mel 会给出一个 `(80, 201)` 矩阵。将 `(n_frames, 201)` 的 STFT magnitude 乘以它的转置，就得到 `(n_frames, 80)` 的 mel spectrogram。

### Step 5: log-mel

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

常见替代形式：`librosa.power_to_db`（reference-normalized dB）、`10 * log10(power + eps)`。Whisper 使用更复杂的 clip + normalize 例程（见 Whisper 的 `log_mel_spectrogram`）。

### Step 6: MFCCs

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

对每个 log-mel frame 应用 DCT，并保留前 13 个 coefficient。这就是 MFCC matrix。第一个 coefficient 通常会被丢弃（它编码整体能量）。

## 实际使用

2026 年的 stack：

| 任务 | 特征 |
|------|----------|
| ASR（Whisper、Parakeet、SeamlessM4T） | 80 log-mels，10 ms hop，25 ms window |
| TTS acoustic model（VITS、F5-TTS、Kokoro） | 80 mels，5-12 ms hop，用于精细时间控制 |
| Audio classification（AST、PANNs、BEATs） | 128 log-mels，10 ms hop |
| Speaker embedding（ECAPA-TDNN、WavLM） | 80 log-mels 或 raw-waveform SSL |
| Music（MusicGen、Stable Audio 2） | EnCodec discrete tokens（不是 mels） |
| Keyword spotting | tiny device 上的 40 MFCCs |

经验法则：**如果你做的不是音乐，先从 80 log-mels 开始。** 任何偏离都需要拿出证据。

## 2026 年仍会被带进生产的坑

- **Mel count mismatch。** 训练用 80 mels，推理用 128 mels。沉默失败。训练端和推理端都要记录 feature shape。
- **上游 sample-rate mismatch。** 22.05 kHz 下计算的 mels 看起来和 16 kHz 不一样。先修正 SR，再做 featurization。
- **dB vs log。** Whisper 期望 log-mel，不是 dB-mel。有些 HF pipeline 会自动检测；你的 custom code 不会。
- **Normalization drift。** 训练时 per-utterance normalization，推理时 global normalization。这是会让 WER 翻倍的生产 bug。
- **Padding leakage。** 在 clip 末尾 zero-padding 会让尾部 frame 出现平坦 spectrum。使用对称 padding 或 replicate。

## 交付成果

保存为 `outputs/skill-feature-extractor.md`。这个 skill 会为给定模型目标选择 feature type、mel count、frame/hop 和 normalization。

## 练习

1. **Easy.** 运行 `code/main.py`。它会合成一个 chirp（频率从 200 → 4000 Hz sweep），并打印每个 frame 的 argmax mel bin。绘图（可选）并确认它匹配 sweep。
2. **Medium.** 用 `n_mels` in `{40, 80, 128}` 和 `frame_len` in `{200, 400, 800}` 重新运行。测量时间轴上的 sharp-peak bandwidth。哪种组合最能解析 chirp？
3. **Hard.** 实现 `power_to_db`，并在 AudioMNIST 上比较 tiny CNN classifier 使用以下特征时的 ASR accuracy：（a）raw log-mel，（b）带 `ref=max` 的 dB-mel，（c）MFCC-13 + delta + delta-delta。报告 top-1 accuracy。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Frame | 一个 slice | 送进一次 FFT 的 25 ms waveform chunk。 |
| Hop | Stride | 相邻 frame 之间的 sample 数；10 ms 是 ASR 默认。 |
| Window | Hann/Hamming 那个东西 | 逐点 multiplier，把 frame 边缘 taper 到零。 |
| STFT | Spectrogram generator | Framed + windowed FFT；产生 time × frequency matrix。 |
| Mel | 扭曲后的 frequency | 对数感知尺度；`m = 2595·log10(1 + f/700)`。 |
| Filterbank | 矩阵 | 将 STFT 投影到 mel bin 的三角形 filter。 |
| Log-mel | Whisper 的输入 | `log(mel_spec + eps)`；2026 年已标准化。 |
| MFCC | 老派特征 | log-mel 的 DCT；13 个 coeff，去相关。 |

## 延伸阅读

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) —— MFCC 论文。
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) —— 原始 mel scale。
- [OpenAI — Whisper source, log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) —— 阅读参考实现。
- [librosa feature extraction docs](https://librosa.org/doc/main/feature.html) —— `mfcc`、`melspectrogram` 和 hop/window 的参考。
- [NVIDIA NeMo — audio preprocessing](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) —— Parakeet + Canary model 的生产级 pipeline。
