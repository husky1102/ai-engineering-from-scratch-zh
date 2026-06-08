# 音频评估：WER、MOS、UTMOS、MMAU、FAD 与开放排行榜

> 你无法发布无法衡量的东西。本课列出 2026 年每类音频任务的指标：ASR（WER、CER、RTFx）、TTS（MOS、UTMOS、SECS、WER-on-ASR-round-trip）、audio-language（MMAU、LongAudioBench）、music（FAD、CLAP）和 speaker（EER）。还包括用于对比的排行榜。

**类型：** Learn
**语言：** Python
**先修：** Phase 6 · 04, 06, 07, 09, 10; Phase 2 · 09 (Model Evaluation)
**时间：** ~60 分钟

## 要解决的问题

每个音频任务都有多个指标，每个指标测量不同轴。用错指标，就是把一个在 dashboard 上很好看、在生产中很糟糕的模型发布出去。2026 年规范列表：

| Task | Primary | Secondary |
|------|---------|-----------|
| ASR | WER | CER · RTFx · first-token latency |
| TTS | MOS / UTMOS | SECS · WER-on-ASR-round-trip · CER · TTFA |
| Voice cloning | SECS (ECAPA cosine) | MOS · CER |
| Speaker verification | EER | minDCF · FAR / FRR at operating point |
| Diarization | DER | JER · speaker confusion |
| Audio classification | top-1 · mAP | macro F1 · per-class recall |
| Music generation | FAD | CLAP · listening panel MOS |
| Audio language model | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| Streaming S2S | latency P50/P95 | WER · MOS |

## 核心概念

![音频评估矩阵：metrics vs tasks vs 2026 leaderboards](../assets/eval-landscape.svg)

### ASR 指标

**WER（Word Error Rate）。** `(S + D + I) / N`。评分前小写化、去标点、规范化数字。使用 `jiwer` 或 OpenAI 的 `whisper_normalizer`。&lt; 5% = 人类同等朗读语音。

**CER（Character Error Rate）。** 同一公式，字符级。用于普通话、粤语等词分割不明确的声调语言。

**RTFx（inverse real-time factor）。** 每 wall-clock second 处理的音频秒数。越高越好。Parakeet-TDT 达到 3380×。Whisper-large-v3 约 30×。

**First-token latency。** 从音频输入到第一个 transcript token 的 wall-clock。对 streaming 至关重要。Deepgram Nova-3：约 150 ms。

### TTS 指标

**MOS（Mean Opinion Score）。** 1-5 人类评分。黄金标准但很慢。每个 sample 收集 20+ listeners，每个 model 100+ samples。

**UTMOS（2022-2026）。** 学习式 MOS predictor。在标准 benchmarks 上与 human MOS 的相关性约 0.9。F5-TTS：UTMOS 3.95；ground truth：4.08。

**SECS（Speaker Encoder Cosine Similarity）。** 用于 voice cloning。参考音频与克隆输出之间的 ECAPA embedding cosine。&gt; 0.75 = 可识别的 clone。

**WER-on-ASR-round-trip。** 对 TTS output 运行 Whisper，并针对 input text 计算 WER。捕捉 intelligibility regressions。2026 SOTA：&lt; 2% CER。

**TTFA（time-to-first-audio）。** Wall-clock latency。Kokoro-82M：约 100 ms；F5-TTS：约 1 s。

### Voice-cloning-specific

**SECS + MOS + CER** 三元组。克隆高 SECS 但低 MOS 意味着音色对但不自然；相反则意味着自然但说话人不对。

### Speaker verification

**EER（Equal Error Rate）。** False Accept Rate 等于 False Reject Rate 的阈值。VoxCeleb1-O 上的 ECAPA：0.87%。

**minDCF（min Detection Cost）。** 某个选择的 operating point（通常 FAR=0.01）上的加权成本。比 EER 更贴近生产。

### Diarization

**DER（Diarization Error Rate）。** `(FA + Miss + Confusion) / total_speaker_time`。漏检语音 + 误报语音 + speaker-confusion，分别作为比例。AMI meetings：DER ~10-20% 是现实水平。pyannote 3.1 + Precision-2 commercial：在录制良好的音频上 DER &lt;10%。

**JER（Jaccard Error Rate）。** DER 的替代指标，对短 segment bias 更稳健。

### Audio classification

Multi-label：所有类别上的 **mAP（mean Average Precision）**。AudioSet：BEATs-iter3 为 0.548 mAP。

Multi-class exclusive：**top-1、top-5 accuracy**。Speech Commands v2：99.0% top-1（Audio-MAE）。

Imbalanced：**macro F1** + **per-class recall**。报告 per-class，aggregate accuracy 会隐藏哪些类别失败。

### Music generation

**FAD（Fréchet Audio Distance）。** 真实音频与生成音频的 VGGish-embedding 分布距离。MusicGen-small 在 MusicCaps 上为 4.5。MusicLM：4.0。越低越好。

**CLAP Score。** 使用 CLAP embeddings 的文本-音频对齐分数。&gt; 0.3 = 合理对齐。

**Listening panel MOS。** 对消费级音乐仍是最终裁决。Suno v5 在 TTS Arena 上的 ELO 为 1293（来自成对人类偏好）。

### Audio-language benchmarks

**MMAU（Massive Multi-Audio Understanding）。** 1 万个 audio-QA pairs。

**MMAU-Pro。** 1800 个 hard items，四类：speech / sound / music / multi-audio。4-way 随机概率为 25%。Gemini 2.5 Pro 总体约 60%；所有模型在 multi-audio 上约 22%。

**LongAudioBench。** 多分钟 clips，带 semantic queries。Audio Flamingo Next 超过 Gemini 2.5 Pro。

**AudioCaps / Clotho。** Captioning benchmarks。SPICE、CIDEr、FENSE metrics。

### Streaming speech-to-speech

**Latency P50 / P95 / P99。** 从用户语音结束到第一个可听响应的 wall-clock。Moshi：200 ms；GPT-4o Realtime：300 ms。

**WER / MOS** 在输出上计算。

**Barge-in responsiveness。** 从用户打断到 assistant mute 的时间。目标 &lt; 150 ms。

### 2026 排行榜

| Leaderboard | Tracks | URL |
|------------|--------|-----|
| Open ASR Leaderboard (HF) | English + multilingual + long-form | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena (HF) | English TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT, ELO from paired votes | `artificialanalysis.ai/speech` |
| MMAU-Pro | LALM reasoning | `mmaubenchmark.github.io` |
| SpeakerBench / VoxSRC | Speaker recognition | `voxsrc.github.io` |
| MMAU music subset | Music LALM | (within MMAU) |
| HEAR benchmark | Self-supervised audio | `hearbenchmark.com` |

## 动手实现

### Step 1：带 normalization 的 WER

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### Step 2：TTS round-trip WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### Step 3：用于 voice cloning 的 SECS

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### Step 4：用于 music generation 的 FAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### Step 5：用于 speaker verification 的 EER（和 Lesson 6 相同代码）

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## 实际使用

为每次部署配套一个固定 eval harness，并在每次模型更新时运行。三条基本规则：

1. **评分前先 normalize。** Lowercase、punctuation-strip、number-expand。报告 normalization rule。
2. **报告分布，不只报告平均值。** 延迟用 P50/P95/P99。分类用 per-class recall。MMAU 用 per-category。
3. **运行一个规范公共 benchmark。** 即使生产数据不同，在 Open ASR / TTS Arena / MMAU 上报告也能让审阅者做 apples-to-apples 对比。

## 常见陷阱

- **UTMOS extrapolation。** 在 VCTK-style clean speech 上训练；对 noisy / cloned / emotional audio 打分很差。
- **MOS panel bias。** 20 个 Amazon Mechanical Turk 工人 ≠ 20 个目标用户。高风险场景要付费找领域 panel。
- **FAD 依赖 reference set。** 跨模型比较时必须使用同一个 reference distribution。
- **Aggregate WER。** 总体 5% WER 可能掩盖 accented speech 上 30% WER。按人口统计 slice 报告。
- **公共 benchmark 饱和。** 大多数 frontier models 已接近标准 benchmarks 上限。构建反映你流量的内部 held-out set。

## 交付成果

保存为 `outputs/skill-audio-evaluator.md`。为任意音频模型发布选择指标、benchmarks 和 reporting format。

## 练习

1. **Easy。** 运行 `code/main.py`。在 toy inputs 上计算 WER / CER / EER / SECS / FAD-ish / MMAU-ish。
2. **Medium。** 构建一个 TTS round-trip WER harness。让你的 Kokoro 或 F5-TTS output 通过 Whisper。对 50 个 prompts 计算 WER。标记 WER &gt; 10% 的 prompts。
3. **Hard。** 在 MMAU-Pro speech + multi-audio subsets（各 50 items）上评测你的 Lesson 10 LALM choice。报告 per-category accuracy，并和 published number 对比。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| WER | ASR score | Normalize 后词级 `(S+D+I)/N`。 |
| CER | Character WER | 用于声调语言或 char-level systems。 |
| MOS | Human opinion | 1-5 评分；20+ listeners × 100 samples。 |
| UTMOS | ML MOS predictor | 学习式模型；与 human MOS 相关性约 0.9。 |
| SECS | Voice-clone similarity | reference 与 clone 的 ECAPA cosine。 |
| EER | Speaker verif score | FAR = FRR 的阈值。 |
| DER | Diarization score | (FA + Miss + Confusion) / total。 |
| FAD | Music-gen quality | VGGish embeddings 上的 Fréchet distance。 |
| RTFx | Throughput | 每 wall-clock second 处理的音频秒数。 |

## 延伸阅读

- [jiwer](https://github.com/jitsi/jiwer) — 带 normalization utilities 的 WER/CER library。
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152) — 学习式 MOS predictor。
- [Fréchet Audio Distance (Kilgour et al. 2019)](https://arxiv.org/abs/1812.08466) — music-gen standard。
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 2026 live rankings。
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) — human-vote TTS leaderboard。
- [MMAU-Pro benchmark](https://mmaubenchmark.github.io/) — LALM reasoning leaderboard。
- [HEAR benchmark](https://hearbenchmark.com/) — audio SSL benchmarks。
