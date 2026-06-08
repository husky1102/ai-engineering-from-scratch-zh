# 音频语言模型：Qwen2.5-Omni、Audio Flamingo、GPT-4o Audio

> 2026 年 audio-language models 能对语音、环境声和音乐进行推理。Qwen2.5-Omni-7B 在 MMAU-Pro 上追平 GPT-4o Audio。Audio Flamingo Next 在 LongAudioBench 上超过 Gemini 2.5 Pro。开源与闭源之间的差距基本闭合，除了 multi-audio tasks：在那里所有人都接近随机。

**类型：** Learn
**语言：** Python
**先修：** Phase 6 · 04 (ASR), Phase 12 · 03 (Vision-Language Models), Phase 7 · 10 (Audio Transformers)
**时间：** ~45 minutes

## 要解决的问题

你有 5 秒音频：狗叫，有人喊 "stop!"，随后沉默。有用的问题跨越多个轴：

- **Transcription。** "What was said?" - ASR 领域。
- **Semantic reasoning。** "Is the person in danger?" - 需要联合理解狗叫 + 喊叫 + 沉默。
- **Music reasoning。** "What instruments play the melody?"
- **Long-audio retrieval。** "Where in this 90-minute lecture did the instructor explain gradient descent?"

一个能用同一个 prompt 回答所有这些问题的模型，就是 **audio-language model**（LALM / ALM）。它不同于纯 ASR：LALMs 生成自由形式自然语言答案，而不只是 transcript。

## 核心概念

![Audio-language model: audio encoder + projector + LLM decoder](../assets/alm-architecture.svg)

### 三组件模板

每个 2026 LALM 都有同一副骨架：

1. **Audio encoder。** Whisper encoder · BEATs · CLAP · WavLM · 或每个模型自定义的 encoder。
2. **Projector。** Linear 或 MLP，把 audio-encoder features 桥接到 LLM 的 token embedding space。
3. **LLM。** 基于 Llama / Qwen / Gemma 的 decoder。接收交错的 text + audio tokens；生成文本。

训练：

- **Stage 1。** 冻结 encoder + LLM；只在 ASR / captioning data 上训练 projector。
- **Stage 2。** 在 instruction-following audio tasks（QA、reasoning、music understanding）上做 full / LoRA fine-tune。
- **Stage 3（可选）。** Voice-in / voice-out 增加 speech decoder。Qwen2.5-Omni 和 AF3-Chat 这样做。

### 2026 模型地图

| Model | Backbone | Audio encoder | Output modality | Access |
|-------|----------|---------------|-----------------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | Custom + Whisper | text + speech | Apache-2.0 |
| Qwen3-Omni | Qwen3 | Custom | text + speech | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | text | NVIDIA non-commercial |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | text | NVIDIA non-commercial |
| SALMONN | Vicuna | Whisper + BEATs | text | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | text | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | text | Apache-2.0 |
| Gemini 2.5 Flash/Pro (closed) | Gemini | proprietary | text + speech | API |
| GPT-4o Audio (closed) | GPT-4o | proprietary | text + speech | API |

### Benchmark 现实检查（2026）

**MMAU-Pro。** 1800 个 QA pairs，覆盖 speech / sound / music / mixed。包含 multi-audio subset。

| Model | Overall | Speech | Sound | Music | Multi-audio |
|-------|---------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | - | - | - | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | - | - | - | - |
| Audio Flamingo Next | SOTA on LongAudioBench | - | - | - | - |

**multi-audio 这一列对所有人都很难看。** 4 选一 multiple choice 的随机概率 = 25%；大多数模型就在这个水平附近。LALMs 仍然难以比较两段 clips。

### 2026 年 LALMs 在哪里有用

- **Call-center recordings 的合规审计。** "Did the agent mention the required disclosure?"
- **无障碍。** 向 deaf users 描述 sound events（不只是 transcription）。
- **内容审核。** 检测 violent language + threatening tone + background context。
- **Podcast / meeting chaptering。** 做语义摘要，而不只是 speaker turns。
- **音乐 catalog 分析。** "Find all tracks with a B-section key change."

### 它们还不适合哪里

- 细粒度音乐理论（低于 chord-level）。
- 长对话上的 speaker-attributed reasoning（超过 10 分钟后退化）。
- Multi-audio comparison（22-26% 只是略高于随机）。
- Real-time streaming reasoning（大多数是 offline batch inference）。

## 动手实现

### Step 1：查询 Qwen2.5-Omni

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### Step 2：projector pattern

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

就是这样。Projector 通常只有 1-3 个 linear layers。用 ASR pairs（audio → transcript）训练它，就是 Stage-1 pretext task。

### Step 3：benchmark MMAU / LongAudioBench

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"Accuracy: {correct / len(mmau['test']):.3f}")
```

分别报告 per-category（speech / sound / music / multi-audio）。聚合数字会隐藏模型失败的位置。

## 实际使用

| 任务 | 2026 pick |
|------|-----------|
| Free-form audio QA（开源） | Qwen2.5-Omni-7B |
| 长音频最佳开源 | Audio Flamingo Next |
| 最佳闭源 | Gemini 2.5 Pro |
| Voice-in / voice-out agent | Qwen2.5-Omni 或 GPT-4o Audio |
| Music reasoning | Audio Flamingo 3 或 2（music-specialized AF-CLAP） |
| Call-center audit | 通过 API 使用 Gemini 2.5 Pro，并对你的 policy docs 做 RAG |

## 常见陷阱

- **过度信任 multi-audio。** 如果你的任务需要 "which clip has X"，random-chance-level performance 是真实存在的。
- **长音频退化。** 超过 10 分钟后，大多数模型的 speaker attribution 会崩。先做 diarize（Lesson 6），再 summarize。
- **沉默上的幻觉。** 使用 Whisper encoder 的 LALMs 会继承同样的 Whisper-style issue。用 VAD-gate。
- **Benchmark cherry-picking。** Vendor blog posts 会突出 best-case categories。自己跑 MMAU-Pro multi-audio subset。

## 交付成果

保存为 `outputs/skill-alm-picker.md`。为给定 audio-understanding task 选择 LALM + benchmark subset + output-modality（text vs speech）。

## 练习

1. **Easy。** 运行 `code/main.py`，观察一个 toy projector pattern + fake LALM routing，把 (audio-embedding, text-tokens) → output tokens。
2. **Medium。** 在 100 个 MMAU-Pro speech items 上打分 Qwen2.5-Omni-7B。与论文报告的数字比较。
3. **Hard。** 构建一个最小 audio-captioning baseline：BEATs encoder + 2-layer projector + frozen Llama-3.2-1B。只在 AudioCaps 上 fine-tune projector。与 SALMONN 在 Clotho-AQA 上比较。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| LALM | Audio ChatGPT | Audio encoder + projector + LLM decoder。 |
| Projector | Adapter | 把 audio features 映射到 LLM embedding space 的小型 MLP。 |
| MMAU | The benchmark | 横跨 speech、sound、music 的 10k audio-QA pairs。 |
| MMAU-Pro | 更难的 MMAU | 1800 个 multi-audio / reasoning-heavy questions。 |
| LongAudioBench | Long-form eval | 带 semantic queries 的多分钟 clips。 |
| Voice-in / voice-out | Speech-native | 模型摄入 speech 并发出 speech，不绕过 text。 |

## 延伸阅读

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) - reference architecture。
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) - speech-in-speech-out。
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) - 开源 long-audio leader。
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) - LongAudioBench SOTA。
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) - dual-encoder pioneer。
- [MMAU-Pro leaderboard](https://mmaubenchmark.github.io/) - live 2026 rankings。
