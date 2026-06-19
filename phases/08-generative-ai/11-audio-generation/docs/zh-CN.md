# 音频生成

> Audio 是 16-48 kHz 的 1-D signal。5 秒 clip 是 80-240k samples。没有 transformer 会直接 attend 这么长的 sequence。2026 年每个生产音频模型的方案相同：neural codec（Encodec、SoundStream、DAC）把 audio 压缩成 50-75 Hz 的 discrete tokens，然后 transformer 或 diffusion model 生成 tokens。

**类型:** Build
**语言:** Python
**先修:** Phase 6 · 02 (Audio Features), Phase 6 · 04 (ASR), Phase 8 · 06 (DDPM)
**时间:** ~45 minutes

## 要解决的问题

三类 audio generation tasks：

1. **Text-to-speech.** 给定 text，生成 speech。干净语音是 narrow-band，并且有强 phonetic structure——transformer-over-tokens 已经很好地解决了它。VALL-E（Microsoft）、NaturalSpeech 3、ElevenLabs、OpenAI TTS。
2. **Music generation.** 给定 prompt（text、melody、chord progression、genre），生成 music。分布宽得多。MusicGen（Meta）、Stable Audio 2.5、Suno v4、Udio、Riffusion。
3. **Audio effects / sound design.** 给定 prompt，生成 ambient sound 或 Foley。AudioGen、AudioLDM 2、Stable Audio Open。

三者都运行在同一个 substrate 上：neural audio codec + token-AR 或 diffusion generator。

## 核心概念

![Audio generation: codec tokens + transformer or diffusion](../assets/audio-generation.svg)

### Neural audio codecs

Encodec（Meta, 2022）、SoundStream（Google, 2021）、Descript Audio Codec（DAC, 2023）。Convolutional encoder 把 waveform 压缩为 per-timestep vector；residual vector quantization (RVQ) 把每个 vector 转换成 K 个 codebook indices 的 cascade。Decoder 反向重建。24 kHz audio at 2 kbps，使用 8 个 RVQ codebooks at 75 Hz = 600 tokens/sec。

```text
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8
```

### Two generative paradigms on top

**Token-autoregressive.** 把 RVQ tokens flatten 成 sequence，运行 decoder-only transformer。MusicGen 使用 “delayed parallel”，以 per-stream offsets 并行发出 K 个 codebook streams。VALL-E 从 text prompt + 3-second voice sample 生成 speech tokens。

**Latent diffusion.** 把 codec tokens 打包为 continuous latents，或用 categorical diffusion 建模。Stable Audio 2.5 在 continuous audio latents 上使用 flow matching。AudioLDM 2 使用 text-to-mel-to-audio diffusion。

2024-2026 趋势：flow matching 在 music 上胜出（推理更快、samples 更干净），而 token-AR 仍主导 speech，因为它天然 causal 且易于 streaming。

## Production landscape

| System | Task | Backbone | Latency |
|--------|------|----------|---------|
| ElevenLabs V3 | TTS | Token-AR + neural vocoder | ~300ms first token |
| OpenAI GPT-4o audio | Full-duplex speech | End-to-end multimodal AR | ~200ms |
| NaturalSpeech 3 | TTS | Latent flow matching | Non-streaming |
| Stable Audio 2.5 | Music / SFX | DiT + flow matching on audio latents | ~10s for 1-minute clip |
| Suno v4 | Full songs | Undisclosed; token-AR suspected | ~30s per song |
| Udio v1.5 | Full songs | Undisclosed | ~30s per song |
| MusicGen 3.3B | Music | Token-AR on Encodec 32kHz | Real-time |
| AudioCraft 2 | Music + SFX | Flow matching | ~5s for 5s clip |
| Riffusion v2 | Music | Spectrogram diffusion | ~10s |

## 动手实现

`code/main.py` 模拟核心思想：在 synthetic “audio token” sequences 上训练 tiny next-token transformer，这些 sequences 来自两种不同 “styles”（style A 是 alternating low and high tokens，style B 是 monotonic ramp）。按 style condition 并 sample。

### Step 1: synthetic audio tokens

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "speech-like": alternating
        return [i % vocab_size for i in range(length)]
    # "music-like": ramp
    return [(i * 3) % vocab_size for i in range(length)]
```

### Step 2: train a tiny token predictor

一个按 style conditioned 的 bigram-style predictor。重点是模式：codec tokens → cross-entropy training → autoregressive sampling。

### Step 3: sample conditionally

给定 style token 和 starting token，从预测分布中 sample next token。持续 20-40 tokens。

## 常见陷阱

- **Codec quality caps output quality.** 如果 codec 无法忠实表示一种 sound，再好的 generator 也帮不上。DAC 是当前 open best。
- **RVQ error accumulation.** 每个 RVQ layer 建模前一层 residual。Layer 1 的错误会传播。对高层使用 temperature 0 sampling 有帮助。
- **Musical structure.** 30 seconds tokens 在 75 Hz 下是 20k+ tokens。Transformers 很难处理。MusicGen 使用 sliding window + prompt continuation；Stable Audio 使用更短 clips + crossfading。
- **Artifacts at boundaries.** Generated clips 之间的 crossfading 需要小心 overlap-add。
- **Clean-data appetite.** Music generators 需要数万小时 licensed music。Suno / Udio RIAA lawsuit（2024）让这一点浮出水面。
- **Voice cloning ethics.** 3-second sample 加 text prompt 足够让 VALL-E / XTTS / ElevenLabs clone 一个 voice。每个 production model 都需要 abuse detection + opt-out lists。

## 实际使用

| Task | 2026 stack |
|------|------------|
| Commercial TTS | ElevenLabs, OpenAI TTS, or Azure Neural |
| Voice cloning (consent-verified) | XTTS v2 (open) or ElevenLabs Pro |
| Background music, fast | Stable Audio 2.5 API, Suno, or Udio |
| Music with lyrics | Suno v4 or Udio v1.5 |
| Sound effects / Foley | AudioCraft 2, ElevenLabs SFX, or Stable Audio Open |
| Real-time voice agent | GPT-4o realtime or Gemini Live |
| Open-weights music research | MusicGen 3.3B, Stable Audio Open 1.0, AudioLDM 2 |
| Dubbing / translation | HeyGen, ElevenLabs Dubbing |

## 交付成果

保存 `outputs/skill-audio-brief.md`。Skill 接收 audio brief（task、duration、style、voice、license），输出：model + hosting、prompt format（genre tags、style descriptors、structural markers）、codec + generator + vocoder chain、seed protocol，以及 eval plan（MOS / CLAP score / CER for TTS / user A/B）。

## 练习

1. **Easy.** 运行 `code/main.py` 并显式设置 style。验证 generated sequences 匹配该 style 的 pattern。
2. **Medium.** 添加 delayed parallel decoding：模拟 2 条必须保持 1 step offset 的 token streams。训练 joint predictor。
3. **Hard.** 使用 HuggingFace transformers 在本地运行 MusicGen-small。用三个不同 prompts 生成 10-second clip；A/B 比较 style adherence。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Codec | “Neural compression” | Audio encoder / decoder；典型输出是 50-75 Hz tokens。 |
| RVQ | “Residual VQ” | K 个 quantizers 的 cascade；每个建模前一层的 residual。 |
| Token | “One codec symbol” | 指向 codebook 的 discrete index；典型大小 1024 或 2048。 |
| Delayed parallel | “Offset codebooks” | 用 staggered offsets 发出 K 个 token streams，以减少 sequence length。 |
| Flow matching | “The 2024 win for audio” | Diffusion 的 straighter-path 替代；sampling 更快。 |
| Voice prompt | “3-second sample” | 引导 cloned voice 的 speaker embedding 或 token prefix。 |
| Mel spectrogram | “The visual” | Log-magnitude perceptual spectrogram；很多 TTS systems 会用。 |
| Vocoder | “Mel to wave” | 把 mel spectrograms 转回 audio 的 neural component。 |

## Production note: audio is a streaming problem

Audio 是用户期待*随着生成到达*而不是一次性全部到达的输出模态。生产术语里，这意味着 TPOT（Time Per Output Token）很重要，因为目标 throughput 是用户的聆听速度，不是阅读速度。对于以约 75 tokens/second tokenize 的 16kHz audio（Encodec），服务器必须为每个用户生成 ≥75 tokens/sec，才能保持 playback smooth。

两个架构后果：

- **Flow-matching audio models cannot stream trivially.** Stable Audio 2.5 和 AudioCraft 2 一次性 render 固定 clip length。要 streaming，需要 chunk clip 并 overlap boundaries——类似 sliding-window diffusion——相比 codec AR model 增加 100-300ms latency overhead。

如果产品是 “live voice chat” 或 “real-time music continuation”，选择 codec AR path。如果是 “submit 后 render 一个 30-second clip”，flow-matching 在质量和总 latency 上胜出。

## 延伸阅读

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — codec standard。
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — 第一个广泛使用的 neural audio codec。
- [Kumar et al. (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) — DAC。
- [Wang et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) — VALL-E。
- [Copet et al. (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) — MusicGen。
- [Liu et al. (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) — AudioLDM 2。
- [Stability AI (2024). Stable Audio 2.5](https://stability.ai/news/introducing-stable-audio-2-5) — 使用 flow matching 的 2025 text-to-music。
