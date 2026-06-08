# 语音反欺骗与音频水印：ASVspoof 5、AudioSeal、WaveVerify

> 语音克隆的发展速度超过了防御。2026 年生产级语音系统需要两样东西：一个将真实与伪造语音分类的检测器（AASIST、RawNet2），以及一个能承受压缩和编辑的水印（AudioSeal）。二者都要发布，否则就不要发布语音克隆。

**类型：** Build
**语言：** Python
**先修：** Phase 6 · 06 (Speaker Recognition), Phase 6 · 08 (Voice Cloning)
**时间：** ~75 分钟

## 要解决的问题

三类相关防御：

1. **Anti-spoofing / deepfake detection。** 给定一个音频 clip，它是合成的还是真实的？ASVspoof benchmarks（ASVspoof 2019 → 2021 → 5）是黄金标准。
2. **Audio watermarking。** 在生成音频中嵌入不可感知信号，检测器之后可以提取。AudioSeal（Meta）和 WavMark 是开放选项。
3. **Authenticated provenance。** 对音频文件 + 元数据进行加密签名。C2PA / Content Authenticity Initiative。

检测处理不合作的对手。水印处理合规，AI 生成音频应当能被识别为 AI 生成。2026 年二者都必需。

## 核心概念

![Anti-spoofing vs watermarking vs provenance — 三层防御](../assets/spoofing-watermark.svg)

### ASVspoof 5：2024-2025 benchmark

相较之前版本的最大变化：

- **众包数据**（不是干净录音棚数据）— 更真实的条件。
- **约 2000 名说话人**（之前约 100）。
- **32 种攻击算法。** TTS + voice conversion + adversarial perturbation。
- **两个 track。** Countermeasure（CM）独立检测；Spoofing-robust ASV（SASV）用于生物识别系统。

ASVspoof 5 上的 state-of-the-art：约 7.23% EER。较老的 ASVspoof 2019 LA 上：0.42% EER。真实世界部署：对 in-the-wild clips 预期 5-10% EER。

### AASIST 与 RawNet2：检测模型家族

**AASIST**（2021，持续更新到 2026）。基于 spectral features 的 graph-attention。当前 ASVspoof 5 countermeasure task 的 SOTA。

**RawNet2。** 原始 waveform 上的卷积 front-end + TDNN backbone。更简单的 baseline；微调后仍有竞争力。

**NeXt-TDNN + SSL features。** 2025 变体：ECAPA-style + WavLM features + focal loss。在 ASVspoof 2019 LA 上达到 0.42% EER。

### AudioSeal：2024 年水印默认选择

Meta 的 **AudioSeal**（2024 年 1 月，v0.2 2024 年 12 月）。关键设计：

- **Localized。** 在 16 kHz sample resolution（1/16000 s）逐帧检测水印。
- **Generator + detector 联合训练。** Generator 学会嵌入不可听信号；detector 学会在增广后找出它。
- **Robust。** 承受 MP3 / AAC 压缩、EQ、速度偏移 ±10%、噪声混合 +10 dB SNR。
- **Fast。** Detector 以 485× realtime 运行；比 WavMark 快 1000×。
- **Capacity。** 16-bit payload（可编码 model ID、generation timestamp、user ID）可嵌入每个 utterance。

### WavMark

AudioSeal 之前的开放 baseline。可逆神经网络，32 bits/sec。问题：

- 同步 brute-force 很慢。
- 可被 Gaussian noise 或 MP3 compression 移除。
- 不适合实时。

### WaveVerify（2025 年 7 月）

解决 AudioSeal 的弱点，尤其是时间操纵（反转、变速）。使用 FiLM-based generator + Mixture-of-Experts detector。在标准攻击上与 AudioSeal 竞争；能处理时间编辑。

### 对手利用的缺口

来自 AudioMarkBench：“under pitch shift, all watermarks show Bit Recovery Accuracy below 0.6, indicating near-complete removal。” **Pitch-shift 是通用攻击。** 2026 年没有任何水印对激进 pitch modification 完全稳健。这就是为什么你需要 detection（AASIST）与 watermarking 并用。

### C2PA / Content Authenticity Initiative

这不是 ML 技术，而是一种 manifest format。音频文件携带关于创建工具、作者、日期的加密签名元数据。Audobox / Seamless 使用它。适合 provenance；但如果坏人重新编码并剥离元数据，它什么也做不了。

## 动手实现

### Step 1：简单 spectral-feature detector（toy）

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

合成语音通常有异常平坦的高频能量。生产检测器使用 AASIST，而不是这个。但直觉成立。

### Step 2：AudioSeal embed + detect

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: float in [0, 1] — probability of watermark presence
# decoded_payload: 16 bits; match against embedded payload
```

### Step 3：evaluation：EER

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### Step 4：生产集成

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

每次生成都携带：（1）水印，（2）签名 manifest，（3）符合留存政策的审计日志。

## 实际使用

| Use case | Defense |
|----------|---------|
| 发布 TTS / voice cloning | 每个输出都嵌入 AudioSeal（不可协商） |
| 生物识别 voice unlock | AASIST + ECAPA ensemble；liveness challenge |
| 呼叫中心欺诈检测 | 对 20% incoming calls 采样运行 AASIST |
| Podcast authenticity | 上传时 C2PA signing；若 AI-generated 则加 AudioSeal |
| 研究 / 训练 detectors | ASVspoof 5 train/dev/eval sets |

## 常见陷阱

- **有水印但检测器从不运行。** 没意义。把 detector 放进 CI。
- **检测没有校准。** 在 ASVspoof LA 上训练的 AASIST 会过拟合；真实世界准确率下降。针对你的领域校准。
- **Pitch-shift 缺口。** 激进 pitch shift 会移除大多数水印。准备 detection fallback。
- **Metadata strip-and-rehost。** C2PA 可被重新编码轻易绕过。始终把加密防御和感知防御（水印）放在一起。
- **把 liveness 当 detection。** 要求用户说随机短语。能防 replay attacks，但不能防实时克隆。

## 交付成果

保存为 `outputs/skill-spoof-defender.md`。为 voice-gen deployment 选择检测模型、水印、provenance manifest 和 operational playbook。

## 练习

1. **Easy。** 运行 `code/main.py`。在合成音频上使用 toy detector + toy watermark embed/detect。
2. **Medium。** 安装 `audioseal`，在 TTS output 中嵌入 16-bit payload，再重新解码。用噪声损坏音频并测量 Bit Recovery Accuracy。
3. **Hard。** 在 ASVspoof 2019 LA 上微调 RawNet2 或 AASIST。测量 EER。在一组留出的 F5-TTS-generated clips 上测试，观察 OOD detection 如何退化。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| ASVspoof | benchmark | 双年 challenge；2024 = ASVspoof 5。 |
| CM (countermeasure) | Detector | 分类器：真实语音 vs synthetic / converted。 |
| SASV | Speaker verif + CM | 集成的 biometric + spoof detection。 |
| AudioSeal | Meta watermark | Localized，16-bit payload，比 WavMark 快 485×。 |
| Bit Recovery Accuracy | Watermark survival | 攻击后恢复出的 payload bit 比例。 |
| C2PA | Provenance manifest | 关于创建 / 作者身份的加密元数据。 |
| AASIST | Detector family | 基于 graph-attention 的 anti-spoofing SOTA。 |

## 延伸阅读

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) — 当前 benchmark。
- [Defossez et al. (2024). AudioSeal](https://arxiv.org/abs/2401.17264) — 默认水印方案。
- [Chen et al. (2025). WaveVerify](https://arxiv.org/abs/2507.21150) — 面向 temporal attacks 的 MoE detector。
- [Jung et al. (2022). AASIST](https://arxiv.org/abs/2110.01200) — SOTA detection backbone。
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) — robustness evaluation。
- [C2PA specification](https://c2pa.org/specifications/specifications/) — provenance manifest format。
