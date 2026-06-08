# 音乐生成：MusicGen、Stable Audio、Suno 与授权地震

> 2026 年音乐生成：商业端由 Suno v5 和 Udio v4 主导；开源端由 MusicGen、Stable Audio Open 和 ACE-Step 领先。技术问题基本已经解决。法律问题（Warner Music $500M settlement、UMG settlement）在 2025-2026 年重塑了整个领域。

**类型：** Build
**语言：** Python
**先修：** Phase 6 · 02 (Spectrograms), Phase 4 · 10 (Diffusion Models)
**时间：** ~75 minutes

## 要解决的问题

Text → 一段 30 秒到 4 分钟的音乐片段，包含歌词、人声和结构。三个子问题：

1. **Instrumental generation。** 像 "lo-fi hip-hop drums with warm keys" 这样的文本 → audio。MusicGen、Stable Audio、AudioLDM。
2. **Song generation（含 vocals + lyrics）。** "Country song about rainy Texas nights" → full song。Suno、Udio、YuE、ACE-Step。
3. **Conditional / controllable。** 延展已有片段、重新生成 bridge、切换 genre、stem-separate 或 inpaint。Udio 的 inpainting + stem separation 是 2026 年要对标的功能。

## 核心概念

![Music generation: token-LM vs diffusion, the 2026 model map](../assets/music-generation.svg)

### Neural-codec tokens 上的 token LM

Meta 的 **MusicGen**（2023，MIT）及许多衍生模型：以 text/melody embeddings 为条件，自回归预测 EnCodec tokens（32 kHz，4 codebooks），再用 EnCodec decode。300M - 3.3B params。强 baseline；超过 30 秒会吃力。

**ACE-Step**（开源，4B XL 于 2026 年 4 月发布）把这一路线扩展到基于歌词条件的 full-song generation。它是开源社区最接近 Suno 的东西。

### Mels 或 latents 上的 diffusion

**Stable Audio (2023)** 和 **Stable Audio Open (2024)**：压缩音频上的 latent diffusion。擅长 loops、sound design、ambient textures。不擅长结构化 full songs。

**AudioLDM / AudioLDM2**：通过 T2I-style latent diffusion 做 text-to-audio，并泛化到音乐、音效、语音。

### Hybrid（生产）- Suno、Udio、Lyria

Closed weights。很可能是 AR codec LM + diffusion-based vocoder，并带有专门的 voice / drum / melody heads。Suno v5 (2026) 是 ELO 1293 的质量领先者。Udio v4 增加了 inpainting + stem separation（bass、drums、vocals 可单独下载）。

### 评估

- **FAD (Fréchet Audio Distance)。** 使用 VGGish 或 PANNs features，计算 generated 与 real audio distribution 的 embedding-level distance。越低越好。MusicGen small 在 MusicCaps 上为 4.5 FAD；SOTA 约 3.0。
- **Musicality（主观）。** 人类偏好。Suno v5 ELO 1293 领先。
- **Text-audio alignment。** prompt 与 output 之间的 CLAP score。
- **Musicality artifacts。** Off-beat transitions、vocal-phrase drift、超过 30 s 后结构丢失。

## 2026 模型地图

| Model | Params | Length | Vocals | License |
|-------|--------|--------|--------|---------|
| MusicGen-large | 3.3B | 30 s | no | MIT |
| Stable Audio Open | 1.2B | 47 s | no | Stability non-commercial |
| ACE-Step XL (Apr 2026) | 4B | &gt; 2 min | yes | Apache-2.0 |
| YuE | 7B | &gt; 2 min | yes, multilingual | Apache-2.0 |
| Suno v5 (closed) | ? | 4 min | yes, ELO 1293 | commercial |
| Udio v4 (closed) | ? | 4 min | yes + stems | commercial |
| Google Lyria 3 (closed) | ? | real-time | yes | commercial |
| MiniMax Music 2.5 | ? | 4 min | yes | commercial API |

## 法律格局（2025-2026）

- **Warner Music vs Suno settlement。** $500M。WMG 现在对 Suno 上的 AI-likeness、music rights 和 user-generated tracks 拥有监督权。Udio 上也有类似的 UMG settlement。
- **EU AI Act** + **California SB 942**：AI-generated music 必须披露。
- **Riffusion / MusicGen** 使用 MIT，没有合规包袱，但也没有商业人声。

安全上线模式：

1. 只生成 instrumental（MusicGen、Stable Audio Open、MIT/CC0 outputs）。
2. 使用商业 APIs（Suno、Udio、ElevenLabs Music）并获得 per-generation license。
3. 在自有或已授权 catalog 上训练（大多数企业最后都会走到这里）。
4. 为 generations 添加 watermarks + metadata。

## 动手实现

### Step 1：用 MusicGen 生成

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

三个尺寸：`small`（300M，快）、`medium`（1.5B）、`large`（3.3B）。Small 足够回答 "does the idea land."

### Step 2：melody conditioning

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody 接收 chromagram，在切换 timbre 的同时保留 tune。适合 "give me this melody as a string quartet."

### Step 3：FAD evaluation

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

计算 VGGish-embedding distance。适合 genre-level regression tests；不能替代人类听众。

### Step 4：加入 LLM-music workflow

结合 Lessons 7-8 的思路：

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## 实际使用

| 目标 | Stack |
|------|-------|
| Instrumental sound design | Stable Audio Open |
| Game / adaptive music | Google Lyria RealTime (closed) |
| Full songs with vocals（商业） | Suno v5 或 Udio v4，带 explicit license |
| Full songs with vocals（开源） | ACE-Step XL 或 YuE |
| Short ad jingle | MusicGen melody-conditioned on a hummed reference |
| Music-video background | MusicGen + Stable Video Diffusion |

## 2026 年仍会被带上线的陷阱

- **Copyright-laundering prompts。** "Song in the style of Taylor Swift" - 商业 Suno/Udio 现在会过滤这些，开源模型不会。添加你自己的 filter list。
- **超过 30 s 后重复 / drift。** AR models 会 loop。Crossfade 多次生成，或使用 ACE-Step 获得结构一致性。
- **Tempo drift。** 模型会偏离 BPM。在 prompt 中使用 BPM tags，并用 librosa 的 `beat_track` 做 post-filter。
- **Vocal intelligibility。** Suno 很出色；开源模型的词往往糊。如果歌词重要，使用商业 API 或 fine-tune。
- **Mono output。** 开源模型生成 mono 或 fake-stereo。用合适的 stereo reconstruction 升级（ezst、Cartesia 的 stereo diffusion）。

## 交付成果

保存为 `outputs/skill-music-designer.md`。为 music-gen deployment 选择模型、license strategy、length / structure plan 和 disclosure metadata。

## 练习

1. **Easy。** 运行 `code/main.py`。它会用 ASCII symbols 生成一个 "generative" chord progression + drum pattern，也就是音乐生成的简笔画。如果愿意，可以用任何 MIDI renderer 播放。
2. **Medium。** 安装 `audiocraft`，用 MusicGen-small 在 4 个 genre prompts 上生成 10 秒 clips，并相对 reference genre set 测量 FAD。
3. **Hard。** 使用 ACE-Step（或 MusicGen-melody），用不同 timbre prompts 生成同一 tune 的三个变体。计算与 prompt 的 CLAP similarity 来验证 alignment。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| FAD | Audio FID | real 与 generated 的 embedding distributions 之间的 Fréchet distance。 |
| Chromagram | 作为 pitches 的 melody | 12-dim per-frame vector；melody conditioning 的输入。 |
| Stems | Instrument tracks | 分离出的 bass / drums / vocals / melody WAV。 |
| Inpainting | 重新生成一个 section | mask 一个 time window；模型只重新生成那一段。 |
| CLAP | Text-audio CLIP | 对比式 audio-text embedding；评估 text-audio alignment。 |
| EnCodec | Music codec | Meta 的 neural codec，MusicGen 使用；32 kHz，4 codebooks。 |

## 延伸阅读

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) - 开源 autoregressive benchmark。
- [Evans et al. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) - sound-design 默认选择。
- [ACE-Step](https://github.com/ace-step/ACE-Step) - 开源 4B full-song generator，2026 年 4 月。
- [Suno v5 platform docs](https://suno.com) - 商业质量领先者。
- [AudioLDM2](https://arxiv.org/abs/2308.05734) - 音乐 + 音效的 latent diffusion。
- [WMG-Suno settlement coverage](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/) - 2025 年 11 月 precedent。
