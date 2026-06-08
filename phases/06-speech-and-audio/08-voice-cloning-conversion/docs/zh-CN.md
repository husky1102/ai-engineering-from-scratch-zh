# 语音克隆与语音转换

> Voice cloning 用别人的声音朗读你的文本。Voice conversion 把你的声音改写成别人的声音，同时保留你说了什么。二者都依赖同一个分解：把 speaker identity 与 content 分开。

**类型：** Build
**语言：** Python
**先修：** Phase 6 · 06 (Speaker Recognition), Phase 6 · 07 (TTS)
**时间：** ~75 minutes

## 要解决的问题

到 2026 年，5 秒音频片段已经足以用消费级 GPU 生成任何人声音的高质量克隆。ElevenLabs、F5-TTS、OpenVoice v2、VoiceBox 都提供 zero-shot 或 few-shot cloning。这个技术既是福音（无障碍 TTS、配音、辅助语音），也是武器（诈骗电话、政治 deepfakes、IP 盗用）。

两个紧密相关的任务：

- **Voice cloning（TTS 侧）：** text + 5 秒参考声音 → 该声音的 audio。
- **Voice conversion（speech 侧）：** source audio（A 说 X）+ B 的参考声音 → B 说 X 的 audio。

二者都把 waveform 分解成（content, speaker, prosody），再把一个来源的 content 与另一个来源的 speaker 重新组合。

你在 2026 年上线时必须满足的关键约束：**在欧盟（AI Act，2026 年 8 月可执行）和加州（AB 2905，2025 年生效），watermarking 和 consent gates 是法律要求**。你的 pipeline 必须输出不可听水印，并拒绝非同意克隆。

## 核心概念

![Voice cloning vs conversion: factorize, swap speaker, recombine](../assets/voice-cloning.svg)

**Zero-shot cloning。** 把一段 5 秒片段传给一个在数千说话人上训练过的模型。Speaker encoder 把片段映射到 speaker embedding；TTS decoder 以该 embedding 和文本为条件。

使用者：F5-TTS (2024)、YourTTS (2022)、XTTS v2 (2024)、OpenVoice v2 (2024)。

**Few-shot fine-tuning。** 录制目标声音 5-30 分钟。对 base model 做一小时 LoRA-fine-tune。质量会从 "okay" 跳到 "indistinguishable"。Coqui 和 ElevenLabs 都支持这种模式；社区也把它用于 F5-TTS。

**Voice conversion (VC)。** 两大家族：

- **Recognition-synthesis。** 运行类似 ASR 的模型抽取 content representation（例如 soft phoneme posteriors、PPGs），再用 target speaker embedding 重新合成。对语言和口音更稳健。KNN-VC (2023)、Diff-HierVC (2023) 使用这一类。
- **Disentanglement。** 训练一个 autoencoder，在 bottleneck 的 latent space 中分离 content、speaker 和 prosody。Inference 时替换 speaker embedding。质量较低但更快。AutoVC (2019)、VITS-VC variants 使用这一类。

**基于 neural codec 的 cloning（2024+）。** VALL-E、VALL-E 2、NaturalSpeech 3、VoiceBox：把音频视为来自 SoundStream / EnCodec 的离散 tokens，在 codec tokens 上训练大型 autoregressive 或 flow-matching model。短 prompts 上的质量可与 ElevenLabs 相比。

### 伦理不是事后补丁

**Watermarking。** PerTh (Perth) 和 SilentCipher (2024) 在音频中不可感知地嵌入约 16-32 bit ID。能经受 re-encoding、streaming 和常见编辑。已达到 production-ready open source。

**Consent gates。** 必须把每个 cloned output 与可验证的 consent record 配对。"I, Rohit, on 2026-04-22, authorize this voice for X purpose." 存入 tamper-evident log。

**Detection。** AASIST、RawNet2 和 Wav2Vec2-AASIST 都可作为 detector。ASVspoof 2025 challenge 发布的 EER 显示，state-of-the-art detectors 对 ElevenLabs、VALL-E 2 和 Bark 输出达到 0.8-2.3%。

### 数字（2026）

| Model | Zero-shot? | SECS (target sim) | WER (intel.) | Params |
|-------|-----------|--------------------|--------------|--------|
| F5-TTS | Yes | 0.72 | 2.1% | 335M |
| XTTS v2 | Yes | 0.65 | 3.5% | 470M |
| OpenVoice v2 | Yes | 0.70 | 2.8% | 220M |
| VALL-E 2 | Yes | 0.77 | 2.4% | 370M |
| VoiceBox | Yes | 0.78 | 2.1% | 330M |

SECS > 0.70 对大多数听众来说通常已与目标难以区分。

## 动手实现

### Step 1：用 recognition-synthesis 分解（`main.py` 中的 code-only demo）

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

概念上很简单；实现体量主要在 `tts_model` 和 speaker encoder 中。

### Step 2：用 F5-TTS 做 zero-shot clone

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

参考转录必须与参考音频完全匹配；不匹配会破坏 alignment。

### Step 3：用 KNN-VC 做 voice conversion

```python
import torch
from knnvc import KNNVC  # 2023 model, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC 运行 WavLM，为 source 和 target pool 抽取 per-frame embeddings，然后把每个 source frame 替换成 pool 中最近的邻居。非参数方法，用一分钟目标语音就能工作。

### Step 4：嵌入 watermark

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # returns payload bytes
```

约 32 bits payload，经过 MP3 re-encode 和轻微噪声后仍可检测。

### Step 5：consent gate

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Signed consent required"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## 实际使用

2026 年的栈：

| 场景 | 选择 |
|-----------|------|
| 5 秒 zero-shot clone，开源 | F5-TTS 或 OpenVoice v2 |
| 商业生产 cloning | ElevenLabs Instant Voice Clone v2.5 |
| Voice conversion（改写） | KNN-VC 或 Diff-HierVC |
| 多说话人 fine-tune | StyleTTS 2 + speaker adapter |
| 跨语言 cloning | XTTS v2 或 VALL-E X |
| Deepfake detection | Wav2Vec2-AASIST |

## 常见陷阱

- **参考转录错位。** F5-TTS 和类似模型要求 reference text 与 reference audio 完全匹配，包括标点。
- **混响参考音频。** Echo 会毁掉 clone。干声、近距离麦克风录制。
- **情绪不匹配。** "cheerful" 的训练参考会让所有东西都变成 cheerfully cloned。让参考情绪匹配目标用途。
- **语言泄漏。** 克隆英语说话人后让模型说法语，往往还是带着口音；使用 cross-lingual models（XTTS、VALL-E X）。
- **没有 watermark。** 2026 年 8 月起在欧盟法律上无法上线。

## 交付成果

保存为 `outputs/skill-voice-cloner.md`。设计一个带 consent gate + watermark + quality target 的 cloning 或 conversion pipeline。

## 练习

1. **Easy。** 运行 `code/main.py`。它会通过计算两个 "speakers" 在 swap 前后的 cosine，演示 speaker-embedding swap。
2. **Medium。** 使用 OpenVoice v2 克隆你自己的声音。测量 reference 和 clone 之间的 SECS。通过 Whisper 测量 CER。
3. **Hard。** 对 20 个 clones 应用 SilentCipher watermark，让它们经过 128 kbps MP3 encode+decode，再检测 payload。报告 bit-accuracy。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Zero-shot clone | 5 秒就够 | Pretrained model + speaker embedding；不训练。 |
| PPG | Phonetic posteriorgram | 用作 language-agnostic content rep 的 per-frame ASR posteriors。 |
| KNN-VC | Nearest-neighbor conversion | 把每个 source frame 替换成最近的 target-pool frame。 |
| Neural codec TTS | VALL-E style | EnCodec/SoundStream tokens 上的 AR model。 |
| Watermark | 不可听签名 | 嵌入音频的 bits，可经受 re-encode。 |
| SECS | Cloning fidelity | target 与 clone speaker embeddings 之间的 cosine。 |
| AASIST | Deepfake detector | Anti-spoof model；检测合成语音。 |

## 延伸阅读

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) - 开源 SOTA zero-shot cloning。
- [Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) and [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) - neural-codec TTS。
- [Qian et al. (2019). AutoVC](https://arxiv.org/abs/1905.05879) - 基于 disentanglement 的 voice conversion。
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) - 基于 retrieval 的 VC。
- [SilentCipher (2024) - Audio Watermarking](https://github.com/sony/silentcipher) - production-ready 32-bit audio watermark。
- [ASVspoof 2025 results](https://www.asvspoof.org/) - detector vs synthesizer arms race，2026 更新。
