# Whisper：架构与微调

> Whisper 是一个 30 秒 window 的 transformer encoder-decoder，在 680k 小时 multilingual weakly-supervised audio-text pair 上训练。一个架构，多个任务，覆盖 99 种语言并对噪声鲁棒。它是 2026 年的参考 ASR。

**类型：** Build
**语言：** Python
**先修：** Phase 6 · 04（ASR），Phase 5 · 10（Attention），Phase 7 · 05（Full Transformer）
**时间：** ~75 分钟

## 要解决的问题

OpenAI 于 2022 年 9 月发布 Whisper。它是第一个作为 commodity 交付的 ASR model：粘贴音频，得到文本，99 种语言，对噪声鲁棒，还能在笔记本电脑上跑。到 2024 年，OpenAI 发布了 Large-v3 和 Turbo variant；到 2026 年，Whisper 已经是 podcast transcription、voice assistant、YouTube subtitle 等所有场景的默认 baseline。

但 Whisper 不是一个可以永远当黑盒使用的 pipeline。Domain shift 会杀死它——技术术语、说话人口音、proper noun、短 clip、silence。你需要知道：

1. 它内部到底是什么。
2. 如何正确给它 chunked、streaming 或 long-form audio。
3. 什么时候 fine-tune，以及怎么 fine-tune。

## 核心概念

![Whisper encoder-decoder, tasks, chunked inference, fine-tune](../assets/whisper.svg)

**Architecture。** 标准 transformer encoder-decoder。

- Input：30 秒 log-mel spectrogram，80 mels，10 ms hop → 3000 frames。更短的 clip 会 zero-pad，更长的 clip 会 chunk。
- Encoder：conv-downsample（stride 2）+ `N` 个 transformer block。Large-v3：32 layers、1280-dim、20 heads。
- Decoder：`N` 个 transformer block，包含 causal self-attn + 到 encoder output 的 cross-attn。尺寸与 encoder 相同。
- Output：覆盖 51,865-token vocab 的 BPE tokens。

Large-v3 有 1.55B 参数。Turbo 使用 4-layer decoder（从 32 层缩到 4 层），以 <1% WER 损失换来 8× latency 降低。

**Prompt format。** Whisper 是一个 multitask model，通过 decoder prompt 中的 special token 来控制：

```text
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>` —— language tag；强制 translation-vs-transcription 行为。
- `<|transcribe|>` 或 `<|translate|>` —— 对任意语言输入转写原文，或翻译成 English output。
- `<|notimestamps|>` —— 跳过 word-level timestamp（更快）。

Prompt 让一个模型完成许多任务。把 `<|en|>` 改成 `<|fr|>`，它就会转写法语。

**30-second window。** 一切都被固定在 30 秒。更长 clip 需要 chunk；更短 clip 会 padding。Window 不是原生 streaming 的——这就是 WhisperX、Whisper-Streaming 和 faster-whisper 存在的原因。

**Log-mel normalization。** `(log_mel - mean) / std`，其中 stats 来自 Whisper 自己的训练语料。你*必须*使用 Whisper 的 preprocessing（`whisper.audio.log_mel_spectrogram`），而不是 `librosa.feature.melspectrogram`。

### 2026 年的 variants

| Variant | Params | Latency (A100) | WER (LibriSpeech-clean) |
|---------|--------|----------------|------------------------|
| Tiny | 39M | 1× realtime | 5.4% |
| Base | 74M | 1× | 4.1% |
| Small | 244M | 1× | 3.0% |
| Medium | 769M | 1× | 2.7% |
| Large-v3 | 1.55B | 2× | 1.8% |
| Large-v3-turbo | 809M | 8× | 1.58% |
| Whisper-Streaming (2024) | 1.55B | streaming | 2.0% |

### Fine-tuning

2026 年的经典 workflow：

1. 收集 10-100 小时目标领域音频，以及对齐 transcript。
2. 使用带 `generate_with_loss` callback 的 `transformers.Seq2SeqTrainer`。
3. 参数高效：在 attention layer 的 `q_proj`、`k_proj`、`v_proj` 上使用 LoRA，可将 GPU memory 降低 4×，WER 成本 <0.3。
4. 如果你有 <10 小时数据，freeze encoder。只 tune decoder。
5. 使用 Whisper 自己的 tokenizer 和 prompt format；永远不要换 tokenizer。

社区结果：在 20 小时 medical dictation 上 fine-tune Medium，可将 medical vocabulary 的 WER 从 12% 降到 4.5%。在 4 小时 Icelandic 上 fine-tune Turbo，可将 WER 从 18% 降到 6%。

## 动手实现

### Step 1: run Whisper out of the box

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # prevents runaway repetition
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

你应该总是覆盖的关键默认值：`temperature=0.0`（sampling 默认是 0.0 → 0.2 → 0.4 … fallback chain）、`condition_on_previous_text=False`（防止 cascading hallucination problem），以及 `no_speech_threshold=0.6`（silence detection）。

### Step 2: chunked long-form

```python
# whisperx is the 2026 reference for long-form with word-level timestamps
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX 增加了：（1）Silero VAD gating，（2）通过 wav2vec 2.0 做 word-level alignment，（3）通过 `pyannote.audio` 做 diarization。它是 2026 年 production transcription 的主力。

### Step 3: fine-tune with LoRA

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> ~3M trainable / 809M total
```

然后使用标准 Trainer loop。每 1000 step checkpoint 一次。用 held-out 上的 WER 评估。

### Step 4: inspect what each layer learns

```python
# Grab cross-attention weights during decode to see what the decoder attends to.
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: layer × head × step × src_len
```

用 heatmap 可视化——你会看到 decoder step 扫过 encoder frame 时出现的 diagonal alignment。那条对角线就是 Whisper 对 word timestamp 的理解。

## 实际使用

2026 年的 stack：

| 场景 | 选择 |
|-----------|------|
| General English、offline | 通过 `whisperx` 使用 Large-v3-turbo |
| Mobile / edge | Whisper-Tiny quantized（int8）或 Moonshine |
| Multilingual long-form | Large-v3 via `whisperx` + diarization |
| Low-resource language | 用 LoRA fine-tune Medium 或 Turbo |
| Streaming（2 s latency） | Whisper-Streaming 或 Parakeet-TDT |
| Word-level timestamps | WhisperX（通过 wav2vec 2.0 做 forced alignment） |

`faster-whisper`（CTranslate2 backend）是 2026 年最快的 CPU+GPU inference runtime——比 vanilla 快 4×，输出相同。

## 2026 年仍会被带进生产的坑

- **Silence 上的 hallucinated text。** Whisper 的训练 captions 包含 "Thanks for watching!"、"Subscribe!"、song lyrics。调用前一定要 VAD-gate。
- **`condition_on_previous_text` cascade。** 一次 hallucination 会污染后续 window。除非需要 chunk 间 fluency，否则设为 `False`。
- **Short-clip padding。** 2 秒 clip padding 到 30 秒，会在尾部 silence 中 hallucinate。使用 `pad=False` 或 VAD-gate。
- **错误的 mel stats。** 使用 librosa 的 mels 而不是 Whisper 的，会产生近乎随机的输出。使用 `whisper.audio.log_mel_spectrogram`。

## 交付成果

保存为 `outputs/skill-whisper-tuner.md`。为给定领域设计 Whisper fine-tune 或 inference pipeline。

## 练习

1. **Easy.** 运行 `code/main.py`。它会 tokenize 一个 Whisper-style prompt，计算 decoded shape budget，并打印 10 分钟 clip 的 chunk schedule。
2. **Medium.** 安装 `faster-whisper`，转写一个 10 分钟 podcast，与人工 transcript 比较 WER。尝试 `language="auto"` vs 强制 `language="en"`。
3. **Hard.** 使用 HF `datasets`，选择一种 Whisper 表现吃力的语言（例如 Urdu），用 2 小时数据对 Medium 做 2 个 epoch 的 LoRA fine-tune，并报告 WER delta。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| 30-sec window | Whisper 的限制 | 硬 input cap；更长音频需要 chunk。 |
| SOT | Start-of-transcript | `<\|startoftranscript\|>` 启动 decoder prompt。 |
| Timestamps token | Temporal alignment | 51k vocab 中每 0.02 s offset 都是一个 special token。 |
| Turbo | 快速 variant | 4 个 decoder layer，快 8×，WER regression <1%。 |
| WhisperX | Long-form wrapper | VAD + Whisper + wav2vec alignment + diarization。 |
| LoRA fine-tune | 高效 tuning | 给 attention 添加 low-rank adapter；训练约 0.3% 参数。 |
| Hallucination | silent failure | Whisper 从 noise/silence 生成流利英语。 |

## 延伸阅读

- [Radford et al. (2022). Whisper paper](https://arxiv.org/abs/2212.04356) —— 原始 architecture 和 training recipe。
- [OpenAI (2024). Whisper Large-v3-turbo release](https://github.com/openai/whisper/discussions/2363) —— 4-layer decoder，8× speedup。
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747) —— long-form、word-aligned、diarized。
- [Systran — faster-whisper repo](https://github.com/SYSTRAN/faster-whisper) —— CTranslate2-backed，快 4×。
- [HuggingFace — Whisper fine-tune tutorial](https://huggingface.co/blog/fine-tune-whisper) —— 经典 LoRA / full-FT walkthrough。
