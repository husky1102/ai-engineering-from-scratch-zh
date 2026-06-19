# 说话人识别与验证

> ASR 问的是“他们说了什么？” Speaker recognition 问的是“谁说的？” 数学看起来一样——embedding 加 cosine——但每个生产决策都取决于一个 EER 数字。

**类型：** Build
**语言：** Python
**先修：** Phase 6 · 02（Spectrograms & Mel），Phase 5 · 22（Embedding Models）
**时间：** ~45 分钟

## 要解决的问题

用户说出 passphrase。你想知道：这是不是他们声称的那个人（*verification*，1:1），或者是不是 enrollment bank 中的第一个人（*identification*，1:N）？或者都不是——这是否是未知说话人（*open-set*）？

2018 年前：GMM-UBM + i-vectors。EER 合理，但对 channel shift（phone vs laptop）和情绪很脆弱。2018-2022：x-vectors（用 angular margin 训练的 TDNN backbone）。2022+：ECAPA-TDNN 和 WavLM-large embeddings。到 2026 年，这个领域由三个模型和一个指标主导。

这个指标是 **EER**——Equal Error Rate。设置决策 threshold，使 False Accept Rate = False Reject Rate。交叉点就是 EER。每篇论文、每个 leaderboard、每次采购评审都使用它。

## 核心概念

![Enrollment + verification pipeline with embedding + cosine + EER](../assets/speaker-verification.svg)

**Pipeline。** Enrollment：录制目标说话人 5-30 秒音频；计算固定维度 embedding（ECAPA-TDNN 为 192-d，WavLM-large 为 256-d）。Verification：得到 test utterance embedding；计算 cosine similarity；与 threshold 比较。

**ECAPA-TDNN（2020，2026 年仍占主导）。** Emphasized Channel Attention, Propagation and Aggregation - Time-Delay Neural Network。由带 squeeze-excitation 的 1D conv block、multi-head attention pooling 组成，之后接 linear layer 得到 192-d。在 VoxCeleb 1+2（2,700 speakers，1.1M utterances）上用 Additive Angular Margin loss（AAM-softmax）训练。

**WavLM-SV（2022+）。** 用 AAM loss fine-tune 预训练 WavLM-large SSL backbone。质量更高但更慢——300+ MB vs 15 MB。

**x-vector（baseline）。** TDNN + statistics pooling。经典；在 CPU / edge 上仍然有用。

**AAM-softmax。** 在 angular space 中给正确 class 加 margin `m` 的标准 softmax：`cos(θ + m)`。强制 inter-class angular separation。典型值是 `m=0.2`，scale `s=30`。

### Scoring

- **Cosine** between enrollment and test embeddings。基于 threshold 决策。
- **PLDA（Probabilistic LDA）。** 将 embedding 投影到 latent space，在其中 same-speaker vs different-speaker 有 closed-form likelihood ratio。在 cosine 之上增加它，可降低 10-20% EER。2020 年前是标准做法；现在只用于 closed-set setup。
- **Score normalization。** `S-norm` 或 `AS-norm`：用一组 imposter mean/std cohort 归一化每个 score。对 cross-domain eval 至关重要。

### 你应该知道的数字（2026）

| Model | VoxCeleb1-O EER | Params | Throughput (A100) |
|-------|-----------------|--------|-------------------|
| x-vector (classic) | 3.10% | 5 M | 400× RT |
| ECAPA-TDNN | 0.87% | 15 M | 200× RT |
| WavLM-SV large | 0.42% | 316 M | 20× RT |
| Pyannote 3.1 segmentation + embedding | 0.65% | 6 M | 100× RT |
| ReDimNet (2024) | 0.39% | 24 M | 100× RT |

### Diarization

多说话人 clip 中的 “Who spoke when”。Pipeline：VAD → segment → embed each segment → cluster（agglomerative 或 spectral）→ smooth boundaries。现代 stack：`pyannote.audio` 3.1，它把 speaker segmentation + embedding + clustering 包在一次调用后面。2026 年 AMI 上的 SOTA DER 约为 15%（2022 年是 23%）。

## 动手实现

### Step 1: toy embedding from MFCC statistics

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

离 SOTA 很远——仅用于教学。`code/main.py` 将它用作 synthetic speaker data 上的 proof-of-concept。

### Step 2: cosine similarity + threshold

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### Step 3: EER from similarity pairs

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

返回 `(eer, threshold_at_eer)`。两个都要报告。

### Step 4: production with SpeechBrain

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# enroll: average the embeddings of 3-5 clean samples
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# verify
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA typical threshold; tune on your data
```

### Step 5: diarize with pyannote

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## 实际使用

2026 年的 stack：

| 场景 | 选择 |
|-----------|------|
| Closed-set 1:1 verification, edge | ECAPA-TDNN + cosine threshold |
| Open-set verification, cloud | WavLM-SV + AS-norm |
| Diarization（meetings、podcasts） | `pyannote/speaker-diarization-3.1` |
| Anti-spoofing（replay / deepfake detection） | AASIST 或 RawNet2 |
| Tiny embedded（KWS + enrollment） | Titanet-Small（NeMo） |

## 常见陷阱

- **Channel mismatch。** 在 VoxCeleb（web video）上训练的模型 ≠ phone-call audio。始终在目标 channel 上评估。
- **Short utterances。** 低于 3 秒 test audio 时，EER 会急剧恶化。
- **Enrollment with noise。** 一个 noisy enrollment 会污染 anchor。使用 ≥3 个 clean sample 并取平均。
- **Fixed threshold across conditions。** 始终在目标领域的 held-out dev set 上调 threshold。
- **Cosine on non-normalized embeddings。** 先做 L2-normalize；否则 magnitude 会主导。

## 交付成果

保存为 `outputs/skill-speaker-verifier.md`。选择 model、enrollment protocol、threshold-tuning plan 和 fraud safeguard。

## 练习

1. **Easy.** 运行 `code/main.py`。它会构建 synthetic "speakers"（不同 tone profile）、enroll，并在 100-pair trial list 上计算 EER。
2. **Medium.** 在 30 条 VoxCeleb1 utterance（5 speakers × 每人 6 条）上使用 SpeechBrain ECAPA。比较 cosine vs PLDA 的 EER。
3. **Hard.** 使用 `pyannote.audio` 构建完整 enroll → diarize → verify pipeline。在 AMI dev set 上评估 DER。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| EER | headline metric | False Accept = False Reject 时的 threshold。 |
| Verification | 1:1 | “这是 Alice 吗？” |
| Identification | 1:N | “谁在说话？” |
| Open-set | 可能未知 | Test set 可以包含未 enrollment 的说话人。 |
| Enrollment | 注册 | 计算说话人的 reference embedding。 |
| AAM-softmax | loss | 带 additive angular margin 的 softmax；强制 cluster separation。 |
| PLDA | 经典 scoring | Probabilistic LDA；embedding 之上的 likelihood-ratio scoring。 |
| DER | Diarization metric | Diarization Error Rate——miss + false alarm + confusion。 |

## 延伸阅读

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) —— 经典 deep-embedding 论文。
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) —— 2020-2026 年的主导架构。
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) —— 用于 SV 和 diarization 的 SSL backbone。
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) —— production diarization + embedding stack。
- [VoxCeleb leaderboard (updated 2026)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) —— 跨模型的当前 EER 排名。
