# 音频分类：从 MFCC 上的 k-NN 到 AST 和 BEATs

> 从 “dog barking vs siren” 到 “这是什么语言”，都属于 audio classification。特征是 mels。架构每十年变化一次。评估始终离不开 AUC、F1 和 per-class recall。

**类型：** Build
**语言：** Python
**先修：** Phase 6 · 02（Spectrograms & Mel），Phase 3 · 06（CNNs），Phase 5 · 08（CNNs & RNNs for Text）
**时间：** ~75 分钟

## 要解决的问题

你拿到一段 10 秒 clip。你想知道：“它是什么？” 城市声音（siren、drill、dog）、speech command（yes/no/stop）、language ID（en/es/ar）、speaker emotion（angry/neutral），或环境声音（indoor/outdoor、babble）。这些全都是 *audio classification*。到 2026 年，baseline 架构已经很成熟：log-mel → CNN 或 Transformer → softmax。

核心难点不是网络，而是数据。音频数据集有残酷的 class imbalance、强烈的 domain shift（clean vs noisy）和 label noise（谁决定这是 “urban babble” 还是 “restaurant noise”？）。80% 的问题在于 curation、augmentation 和 evaluation，而不是把 CNN 换成 Transformer。

## 核心概念

![Audio classification ladder: k-NN on MFCCs to AST to BEATs](../assets/audio-classification.svg)

**MFCC 上的 k-NN（1990s baseline）。** 对每个 clip 展平 MFCC，计算它和 labeled bank 的 cosine similarity，返回 top K 的多数票。在干净、小型数据集（Speech Commands、ESC-50）上出奇地强。不需要 GPU。

**Log-mel 上的 2D CNN（2015-2019）。** 把 `(T, n_mels)` log-mel 当作图像。应用 ResNet-18 或 VGG 风格网络。对 time axis 做 global mean pool。对 class 做 softmax。到 2026 年，它仍是多数 kaggle 竞赛的 baseline。

**Audio Spectrogram Transformer，AST（2021-2024）。** 将 log-mel patchify（例如 16×16 patch），添加 position embedding，送入 ViT。在 AudioSet 上是 supervised learning 的 state of the art（mAP 0.485）。

**BEATs 和 WavLM-base（2024-2026）。** 在数百万小时音频上做 self-supervised pretraining。用你原本需要的 1-10% supervised data 就能 fine-tune 到目标任务。到 2026 年，这是 non-speech audio 的默认起点。BEATs-iter3 在 AudioSet 上比 AST 高 1-2 mAP，同时只用 1/4 compute。

**Whisper-encoder 作为 frozen backbone（2024）。** 取 Whisper 的 encoder，丢掉 decoder，接一个 linear classifier。对 language ID 和简单 event classification 来说，几乎无需 audio augmentation 就接近 SOTA。这是“免费午餐” baseline。

### Class imbalance 才是真挑战

ESC-50：50 个 class，每类 40 个 clip——平衡、简单。UrbanSound8K：10 个 class，10:1 不平衡。AudioSet：632 个 class，有 100,000:1 的 long tail。有效技术：

- 训练时做 balanced sampling（评估时不要）。
- Mixup：把两个 clip（及其 label）线性插值作为 augmentation。
- SpecAugment：随机 mask time 和 frequency band。简单，但关键。

### Evaluation

- Multiclass exclusive（Speech Commands）：top-1 accuracy、top-5 accuracy。
- Multiclass multi-label（AudioSet、UrbanSound 风格）：mean average precision（mAP）。
- 严重不平衡：per-class recall + macro F1。

你应该知道的 2026 年数字：

| Benchmark | Baseline | SOTA 2026 | Source |
|-----------|----------|-----------|--------|
| ESC-50 | 82%（AST） | 97.0%（BEATs-iter3） | BEATs paper（2024） |
| AudioSet mAP | 0.485（AST） | 0.548（BEATs-iter3） | HEAR leaderboard 2026 |
| Speech Commands v2 | 98%（CNN） | 99.0%（Audio-MAE） | HEAR v2 results |

## 动手实现

### Step 1: featurize

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### Step 2: fixed-length summary

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

简单但很强：沿时间计算 mean + variance，为 13-coef MFCC 得到 26 维固定 embedding。瞬间运行。在 ESC-50 上，直到 2017 年还打赢过 state-of-the-art NN baseline。

### Step 3: k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### Step 4: upgrade to CNN on log-mels

In PyTorch:

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

3M 参数。单张 RTX 4090 上约 10 分钟可在 ESC-50 上训练完成。准确率 80%+。

### Step 5: the 2026 default — fine-tune BEATs

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

BEATs 可通过 `beats` library 使用 `microsoft/BEATs-base`；transformers API 的形状相同。

## 实际使用

2026 年的 stack：

| 场景 | 起点 |
|-----------|-----------|
| Tiny dataset（<1000 clips） | MFCC mean 上的 k-NN（你的 baseline）+ audio augmentation |
| Medium dataset（1K-100K） | BEATs 或 AST fine-tune |
| Large dataset（>100K） | 从零训练或 fine-tune Whisper-encoder |
| Real-time、edge | 40-MFCC CNN，quantized to int8（KWS-style） |
| Multi-label（AudioSet） | 带 BCE loss + mixup + SpecAugment 的 BEATs-iter3 |
| Language ID | MMS-LID、SpeechBrain VoxLingua107 baseline |

决策规则：**从 frozen backbone 开始，而不是从 fresh model 开始**。Fine-tuning 一个 BEATs head 能在数小时内得到 95% 的 SOTA，而不是花数周。

## 交付成果

保存为 `outputs/skill-classifier-designer.md`。为给定 audio classification task 选择 architecture、augmentation、class-balance strategy 和 eval metric。

## 练习

1. **Easy.** 运行 `code/main.py`。它会在 4-class synthetic dataset（不同 pitch 的 pure tone）上训练 k-NN MFCC baseline。报告 confusion matrix。
2. **Medium.** 用 [mean, var, skew, kurtosis] 替换 `summarize`。在同一 synthetic dataset 上，4-moment pooling 是否优于 mean+var？
3. **Hard.** 使用 `torchaudio`，在 ESC-50 fold 1 上训练 2D CNN。报告 5-fold cross-validation accuracy。加入 SpecAugment（time mask = 20，freq mask = 10）并报告 delta。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| AudioSet | 音频界的 ImageNet | Google 的 2M-clip、632-class、weakly-labeled YouTube dataset。 |
| ESC-50 | 小型分类 benchmark | 50 类 × 40 个 environmental sound clip。 |
| AST | Audio Spectrogram Transformer | log-mel patch 上的 ViT；2021 SOTA。 |
| BEATs | Self-supervised audio | Microsoft model；截至 2026 年 iter3 领先 AudioSet。 |
| Mixup | 成对 augmentation | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`。 |
| SpecAugment | Mask-based augmentation | 将 spectrogram 的随机 time 和 frequency band 置零。 |
| mAP | 主要 multi-label 指标 | 跨 class 和 threshold 的 mean average precision。 |

## 延伸阅读

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) —— 2021-2024 年的标杆架构。
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) —— 2024+ 默认方案。
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) —— 主导性的 audio augmentation。
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) —— 仍然常用的 50-class benchmark。
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) —— 632-class YouTube taxonomy；仍是 gold standard。
