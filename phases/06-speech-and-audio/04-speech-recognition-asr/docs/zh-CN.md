# 语音识别（ASR）：CTC、RNN-T、Attention

> 语音识别是在每个 timestep 上做 audio classification，再由一个懂英语和 silence 的 sequence model 粘起来。CTC、RNN-T 和 attention 是三种实现方式。选一种，并理解为什么。

**类型：** Build
**语言：** Python
**先修：** Phase 6 · 02（Spectrograms & Mel），Phase 5 · 08（CNNs & RNNs for Text），Phase 5 · 10（Attention）
**时间：** ~45 分钟

## 要解决的问题

你有一段 10 秒、16 kHz 的 clip。你想要一个字符串："turn on the kitchen lights"。挑战在结构上：audio frame 和 character 不是一一对齐的。单词 "okay" 可能持续 200 ms，也可能持续 1200 ms。silence 会给 utterance 加标点。有些 phoneme 比其他 phoneme 更长。output token 数量事先未知。

三种表述能解决这个问题：

1. **CTC（Connectionist Temporal Classification）。** 输出每个 frame 上的 token probability，其中包括一个特殊的 *blank*。decode 时折叠重复项和 blank。Non-autoregressive、快。wav2vec 2.0、MMS 使用它。
2. **RNN-T（Recurrent Neural Network Transducer）。** Joint network 在给定 encoder frame 和 previous tokens 时预测 next token。可流式。Google on-device ASR、NVIDIA Parakeet 使用它。
3. **Attention encoder-decoder。** Encoder 将 audio 压缩为 hidden state，decoder 通过 cross-attention 自回归生成 token。Whisper、SeamlessM4T 使用它。

到 2026 年，LibriSpeech test-clean 上的 SOTA WER 是 1.4%（Parakeet-TDT-1.1B，NVIDIA）和 1.58%（Whisper-Large-v3-turbo）。差异很小；部署差异巨大。

## 核心概念

![Three ASR formulations: CTC, RNN-T, attention-encoder-decoder](../assets/asr-formulations.svg)

**CTC intuition。** 让 encoder 输出 `T` 个 frame-level distribution，覆盖 `V+1` 个 token（V 个 char + blank）。对于长度为 `U < T` 的目标字符串 `y`，任何折叠后等于 `y` 的 frame alignment 都算数。CTC loss 会对所有这类 alignment 求和。Inference：逐 frame argmax，折叠重复项，移除 blank。

优点：non-autoregressive、streamable、zero lookahead。缺点：*conditional independence assumption*——每个 frame prediction 彼此独立，因此没有内部 language model。可通过 beam search 或 shallow fusion 接外部 LM 修正。

**RNN-T intuition。** 添加一个嵌入 token history 的 *predictor* network，以及一个把 predictor state 和 encoder frame 组合成 `V+1` joint distribution 的 *joiner*（`+1` 是 null / no-emit）。它显式建模了 CTC 忽略的 conditional dependence。因为每一步只依赖过去 frame 和过去 token，所以可流式。

优点：streamable + internal LM。缺点：训练更复杂、更吃内存（3D loss lattice）；RNN-T loss kernel 本身就是一个独立库类别。

**Attention encoder-decoder。** Encoder（6-32 个 transformer layer）处理 log-mel frame。Decoder（6-32 个 transformer layer）通过 cross-attention 访问 encoder output，自回归生成 token。没有 alignment 约束——attention 可以看音频中的任何位置。除非限制 attention（chunked Whisper-Streaming，2024），否则不可流式。

优点：offline ASR 质量最高，容易用标准 seq2seq tooling 训练。缺点：autoregressive latency 与输出长度成正比；没有工程处理就无法 streaming。

### WER: the one number

**Word Error Rate** = `(S + D + I) / N`，其中 S=substitutions，D=deletions，I=insertions，N=reference word count。它等价于 word level 的 Levenshtein edit distance。越低越好。WER 高于 20% 通常不可用；低于 5% 对 read speech 来说接近 human-parity。标准 benchmark 上的 2026 年数字：

| Model | LibriSpeech test-clean | LibriSpeech test-other | Size |
|-------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B params |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

这些系统都基于 encoder-decoder 或 RNN-T。纯 CTC 系统（wav2vec 2.0）在 test-clean 上约为 1.8-2.1%。

## 动手实现

### Step 1: greedy CTC decode

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: list of per-frame probability vectors
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

两条规则：折叠连续重复项，丢弃 blank。示例：`a a _ _ a b b _ c` → `a a b c`。

### Step 2: beam-search CTC

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

生产中使用带 LM fusion 的 prefix tree beam search；这里是概念骨架。

### Step 3: WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### Step 4: inference against Whisper

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

这是 2026 年最强通用 ASR 的一行调用。在 24 GB GPU 上大约以 20× realtime 运行。

### Step 5: streaming with Parakeet or wav2vec 2.0

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

Streaming ASR 需要 chunked encoder attention 和 carryover state；请使用支持它的库（Parakeet 用 NeMo，`transformers` pipeline 用 `chunk_length_s`）。

## 实际使用

2026 年的 stack：

| 场景 | 选择 |
|-----------|------|
| English、offline、最高质量 | Whisper-large-v3-turbo |
| Multilingual、robust | SeamlessM4T v2 |
| Streaming、low latency | Parakeet-TDT-1.1B 或 Riva |
| Edge、mobile、<500 ms latency | Whisper-Tiny quantized 或 Moonshine（2024） |
| Long-form | 带 VAD-based chunking 的 Whisper（WhisperX） |
| Domain-specific（medical、legal） | Fine-tune wav2vec 2.0 + domain LM fusion |

## 2026 年仍会被带进生产的坑

- **没有 VAD。** 在 silence 上运行 Whisper 会产生 hallucination（"Thanks for watching!"）。一定要用 VAD gate。
- **Character vs word vs subword WER。** 报告 normalization（lowercase、去 punctuation）之后的 word-level WER。
- **Language ID drift。** Whisper 的 auto LID 会把噪声音频误路由到 Japanese 或 Welsh；你知道语言时强制 `language="en"`。
- **Long clips without chunking。** Whisper 有 30 秒 window。任何更长音频都使用 `chunk_length_s=30, stride=5`。

## 交付成果

保存为 `outputs/skill-asr-picker.md`。为给定 deployment target 选择 model、decoding strategy、chunking 和 LM fusion。

## 练习

1. **Easy.** 运行 `code/main.py`。它会 greedily decode 一个手写 CTC output，并计算与 reference 的 WER。
2. **Medium.** 正确实现 Step 2 的 prefix-tree beam search（考虑 blank merge rule）。在 10 个样本的 synthetic dataset 上与 greedy 比较。
3. **Hard.** 在 [LibriSpeech test-clean](https://www.openslr.org/12) 上使用 `whisper-large-v3-turbo`。计算前 100 条 utterance 的 WER。与公开数字比较。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| CTC | blank-token loss | 对所有 frame-to-token alignment 边缘化；non-AR。 |
| RNN-T | streaming loss | CTC + next-token predictor；处理 word-order。 |
| Attention enc-dec | Whisper-style | Encoder + cross-attending decoder；最佳 offline quality。 |
| WER | 你报告的数字 | word level 的 `(S+D+I)/N`。 |
| Blank | 空 | CTC 中表示 “no emission this frame” 的特殊 token。 |
| LM fusion | 外部 language model | 在 beam search 中加入加权 LM log-prob。 |
| VAD | silence gate | Voice activity detector；裁剪 non-speech。 |

## 延伸阅读

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) —— CTC 论文。
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) —— RNN-T 论文。
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) —— 2022 年经典论文；v3-turbo 扩展于 2024 年。
- [NVIDIA NeMo — Parakeet-TDT card](https://huggingface.co/nvidia/parakeet-tdt-1.1b) —— 2026 Open ASR Leaderboard leader。
- [Hugging Face — Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) —— 覆盖 25+ 模型的实时 benchmark。
