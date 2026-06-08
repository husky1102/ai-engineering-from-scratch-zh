# 音频基础：波形、采样、傅里叶变换

> Waveform 是原始信号。Spectrogram 是表示方式。Mel feature 是适合 ML 的形式。每条现代 ASR 和 TTS pipeline 都会沿着这架梯子往上走，而第一阶就是理解 sampling 和 Fourier。

**类型：** Learn
**语言：** Python
**先修：** Phase 1 · 06（Vectors & Matrices），Phase 1 · 14（Probability Distributions）
**时间：** ~45 分钟

## 要解决的问题

麦克风产生的是“压力随时间变化”的信号。神经网络消费的是 tensor。两者之间夹着一整套约定；一旦违反，就会产生沉默的 bug：模型训练看似正常，但 WER 翻倍；TTS 上线后带 hiss；voice cloning system 记住了麦克风，而不是说话人。

语音系统里的每个 bug 都能追溯到三个问题之一：

1. 数据是以什么 sample rate 录制的，而模型期望什么？
2. 信号是否发生 aliasing？
3. 你是在 raw sample 上操作，还是在 frequency representation 上操作？

这些问题答对了，Phase 6 的其余内容就可控。答错了，就算 Whisper-Large-v4 也会输出垃圾。

## 核心概念

![Waveform, sampling, DFT, and frequency bins visualized](../assets/audio-fundamentals.svg)

**Waveform。** `[-1.0, 1.0]` 中的一维 float 数组。按 sample number 建索引。要转换为秒，用 sample rate 相除：`t = n / sr`。16 kHz 下 10 秒音频是一组 160,000 个 float。

**Sampling rate（sr）。** 每秒有多少 sample。2026 年常见 rate：

| Rate | 用途 |
|------|-----|
| 8 kHz | 电话、legacy VOIP。Nyquist 在 4 kHz，会杀掉辅音。ASR 避免使用。 |
| 16 kHz | ASR 标准。Whisper、Parakeet、SeamlessM4T v2 都消费 16 kHz。 |
| 22.05 kHz | 旧模型的 TTS vocoder training。 |
| 24 kHz | 现代 TTS（Kokoro、F5-TTS、xTTS v2）。 |
| 44.1 kHz | CD audio、音乐。 |
| 48 kHz | 电影、专业音频、高保真 TTS（VALL-E 2、NaturalSpeech 3）。 |

**Nyquist-Shannon。** `sr` 的 sample rate 可以无歧义表示最高 `sr/2` 的频率。`sr/2` 边界就是 *Nyquist frequency*。高于 Nyquist 的能量会发生 *aliasing*：被折叠回较低频率，并污染信号。降采样之前一定要 low-pass filter。

**Bit depth。** 16-bit PCM（signed int16，范围 ±32,767）是通用交换格式。音乐常用 24-bit，内部 DSP 常用 32-bit float。`soundfile` 这样的库会读取 int16，但暴露 `[-1, 1]` 中的 float32 array。

**Fourier Transform。** 任何有限信号都可以看作不同频率正弦波的和。Discrete Fourier Transform（DFT）会对 `N` 个 sample 计算 `N` 个 complex coefficient——每个 frequency bin 一个。`bin k` 映射到频率 `k · sr / N` Hz。Magnitude 是该频率上的 amplitude，angle 是 phase。

**FFT。** Fast Fourier Transform：当 `N` 是 2 的幂时，用于 DFT 的 `O(N log N)` 算法。每个音频库底层都会使用 FFT。16 kHz 下的 1024-sample FFT 会给出 512 个可用 frequency bin，覆盖 0-8 kHz，分辨率为 15.6 Hz。

**Framing + window。** 我们不会对整段 clip 做 FFT。我们把它切成重叠的 *frame*（通常 25 ms，hop 10 ms），给每个 frame 乘上 window function（Hann、Hamming）以消除边缘不连续，再对每个 frame 做 FFT。这就是 Short-Time Fourier Transform（STFT）。Lesson 02 会从这里接上。

## 动手实现

### Step 1: read a clip and plot the waveform

`code/main.py` 只使用 stdlib `wave` 模块，让 demo 保持零依赖。生产中你会使用 `soundfile` 或 `torchaudio.load`（两者都返回 `(waveform, sr)` tuple）：

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### Step 2: synthesize a sine wave from first principles

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

16 kHz 下 1 秒的 440 Hz sine（concert A）是 16,000 个 float。用 `wave.open(..., "wb")` 以 16-bit PCM encoding 写出。

### Step 3: compute the DFT by hand

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

`O(N²)`——对于 `N=256` 的正确性确认没问题，对真实音频完全不可用。真实代码会调用 `numpy.fft.rfft` 或 `torch.fft.rfft`。

### Step 4: find the dominant frequency

Magnitude peak index `k_star` 映射到频率 `k_star * sr / N`。在 440 Hz sine 上运行时，应该返回位于 bin `440 * N / sr` 的峰。

### Step 5: demonstrate aliasing

以 10 kHz 采样一个 7 kHz sine（Nyquist = 5 kHz）。7 kHz tone 高于 Nyquist，会折叠到 `10 − 7 = 3 kHz`。FFT peak 会出现在 3 kHz。这是经典 aliasing demo，也是每个 DAC/ADC 都配有 brick-wall low-pass filter 的原因。

## 实际使用

2026 年你真正会交付的 stack：

| 任务 | Library | 原因 |
|------|---------|-----|
| 读写 WAV/FLAC/OGG | `soundfile`（libsndfile wrapper） | 最快、稳定、返回 float32。 |
| Resample | `torchaudio.transforms.Resample` 或 `librosa.resample` | 内置正确 anti-aliasing。 |
| STFT / Mel | `torchaudio` 或 `librosa` | GPU-friendly；PyTorch ecosystem。 |
| Real-time streaming | `sounddevice` 或 `pyaudio` | 跨平台 PortAudio binding。 |
| 检查文件 | `ffprobe` 或 `soxi` | CLI、快速、报告 sr/channels/codec。 |

决策规则：**先匹配 sample rate，再匹配其他任何东西**。Whisper 期望 16 kHz mono float32。给它传 44.1 kHz stereo，你会得到看起来像模型 bug 的垃圾。

## 交付成果

保存为 `outputs/skill-audio-loader.md`。这个 skill 会帮助你检查音频输入是否符合下游模型预期，并在不符合时正确 resample。

## 练习

1. **Easy.** 在 16 kHz 下合成 1 秒的 220 Hz + 440 Hz + 880 Hz 混合信号。运行 DFT。确认三个 peak 位于预期 bin。
2. **Medium.** 录制一段 48 kHz 的 3 秒人声 WAV。用 `torchaudio.transforms.Resample`（带 anti-aliasing）降采样到 16 kHz；再用 naive decimation（每三个 sample 取一个）降到 16 kHz。对两者做 FFT。aliasing 出现在哪里？
3. **Hard.** 只使用 `math` 和 Step 3 中的 DFT 从零构建 STFT。Frame size 400，hop 160，Hann window。用 `matplotlib.pyplot.imshow` 绘制 magnitude。这就是 Lesson 02 的 spectrogram。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Sample rate | 每秒多少 sample | ADC 测量信号的频率，单位 Hz。 |
| Nyquist | 可表示的最高频率 | `sr/2`；高于它的能量会 alias 回低频。 |
| Bit depth | 每个 sample 的分辨率 | `int16` = 65,536 个 level；`float32` = `[-1, 1]` 中的 24-bit precision。 |
| DFT | sequence 的 Fourier transform | `N` 个 sample → `N` 个 complex frequency coefficient。 |
| FFT | 快速 DFT | 要求 `N` 为 2 的幂的 `O(N log N)` 算法。 |
| Bin | Frequency column | `k · sr / N` Hz；resolution = `sr / N`。 |
| STFT | spectrogram 的底层机制 | 沿时间做 framed + windowed FFT。 |
| Aliasing | 奇怪的 frequency ghost | 高于 Nyquist 的能量镜像到较低 bin。 |

## 延伸阅读

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) —— sampling theorem 背后的论文。
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) —— 免费的经典 DSP 教科书。
- [librosa docs — audio primer](https://librosa.org/doc/latest/tutorial.html) —— 带代码的实践 walkthrough。
- [Heinrich Kuttruff — Room Acoustics (6th ed.)](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781482260434) —— 理解真实世界音频为什么不是干净 sinusoid 的参考书。
- [Steve Eddins — FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) —— 10 分钟理清 frequency bin 直觉。
