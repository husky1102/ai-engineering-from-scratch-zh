# 音频语言模型：Qwen2.5-Omni、Audio Flamingo、GPT-4o Audio

> 2026 年的音频语言模型能同时对语音、环境声和音乐进行推理。Qwen2.5-Omni-7B 在 MMAU-Pro 上追平 GPT-4o Audio。Audio Flamingo Next 在 LongAudioBench 上超过 Gemini 2.5 Pro。开源与闭源之间的差距基本闭合，除了多音频任务：在那里所有模型都接近随机。

**类型：** 学习
**语言：** Python
**先修：** 第 6 阶段 · 04（ASR），第 12 阶段 · 03（视觉语言模型），第 7 阶段 · 10（音频 Transformer）
**时间：** ~45 分钟

## 要解决的问题

你有一段 5 秒音频：狗叫，有人喊“停！”，随后沉默。有用的问题横跨多个维度：

- **转写。** “说了什么？”这是 ASR 领域。
- **语义推理。** “这个人有危险吗？”需要联合理解狗叫、喊叫和沉默。
- **音乐推理。** “哪些乐器在演奏旋律？”
- **长音频检索。** “这场 90 分钟讲座里，老师在哪里解释了梯度下降？”

一个能用同一个提示词回答所有这些问题的模型，就是 **音频语言模型**（LALM / ALM）。它不同于纯 ASR：LALM 生成自由形式的自然语言答案，而不只是转写文本。

## 核心概念

![音频语言模型：音频编码器 + 投影器 + LLM 解码器](../assets/alm-architecture.svg)

### 三组件模板

每个 2026 LALM 都有同一副骨架：

1. **音频编码器。** Whisper 编码器、BEATs、CLAP、WavLM，或每个模型自定义的编码器。
2. **投影器。** 线性层或 MLP，把音频编码器特征桥接到 LLM 的 token 嵌入空间。
3. **LLM。** 基于 Llama / Qwen / Gemma 的解码器。它接收交错的文本与音频 token，并生成文本。

训练：

- **阶段 1。** 冻结编码器和 LLM；只在 ASR / 音频描述数据上训练投影器。
- **阶段 2。** 在遵循指令的音频任务（QA、推理、音乐理解）上做全量或 LoRA 微调。
- **阶段 3（可选）。** 语音输入 / 语音输出会增加语音解码器。Qwen2.5-Omni 和 AF3-Chat 采用这种做法。

### 2026 模型地图

| 模型 | 骨干模型 | 音频编码器 | 输出模态 | 访问方式 |
|-------|----------|---------------|-----------------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | 自定义 + Whisper | 文本 + 语音 | Apache-2.0 |
| Qwen3-Omni | Qwen3 | 自定义 | 文本 + 语音 | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | 文本 | NVIDIA 非商业许可 |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | 文本 | NVIDIA 非商业许可 |
| SALMONN | Vicuna | Whisper + BEATs | 文本 | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | 文本 | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | 文本 | Apache-2.0 |
| Gemini 2.5 Flash/Pro（闭源） | Gemini | 专有 | 文本 + 语音 | API |
| GPT-4o Audio（闭源） | GPT-4o | 专有 | 文本 + 语音 | API |

### 基准现实检查（2026）

**MMAU-Pro。** 1800 个 QA 对，覆盖语音、声音、音乐和混合任务。包含多音频子集。

| 模型 | 总体 | 语音 | 声音 | 音乐 | 多音频 |
|-------|---------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | - | - | - | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | - | - | - | - |
| Audio Flamingo Next | LongAudioBench 最优 | - | - | - | - |

**多音频这一列对所有模型都很难看。** 4 选一选择题的随机概率是 25%；大多数模型就在这个水平附近。LALM 仍然很难比较两段音频片段。

### 2026 年 LALMs 在哪里有用

- **呼叫中心录音的合规审计。** “坐席是否提到了必需披露事项？”
- **无障碍。** 向聋人用户描述声音事件（不只是转写）。
- **内容审核。** 检测暴力语言、威胁语气和背景上下文。
- **播客 / 会议分章。** 做语义摘要，而不只是说话人轮次切分。
- **音乐目录分析。** “找出所有 B 段发生转调的曲目。”

### 它们还不适合哪里

- 细粒度音乐理论（低于和弦级别）。
- 长对话中的说话人归因推理（超过 10 分钟后会退化）。
- 多音频比较（22-26% 只是略高于随机）。
- 实时流式推理（大多数模型仍是离线批处理推理）。

## 动手实现

### 步骤 1：查询 Qwen2.5-Omni

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "你听到了哪些声音，发生了什么？"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### 步骤 2：投影器模式

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

就是这样。投影器通常只有 1-3 个线性层。用 ASR 数据对（音频 → 转写文本）训练它，就是阶段 1 的预训练任务。

### 步骤 3：评测 MMAU / LongAudioBench

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

分别报告每个类别（语音 / 声音 / 音乐 / 多音频）的结果。聚合数字会隐藏模型失败的位置。

## 实际使用

| 任务 | 2026 年推荐选择 |
|------|-----------|
| 自由形式音频问答（开源） | Qwen2.5-Omni-7B |
| 长音频最佳开源 | Audio Flamingo Next |
| 最佳闭源 | Gemini 2.5 Pro |
| 语音输入 / 语音输出智能体 | Qwen2.5-Omni 或 GPT-4o Audio |
| 音乐推理 | Audio Flamingo 3 或 2（面向音乐特化的 AF-CLAP） |
| 呼叫中心审计 | 通过 API 使用 Gemini 2.5 Pro，并对你的政策文档做 RAG |

## 常见陷阱

- **过度信任多音频能力。** 如果你的任务需要回答“哪段音频有 X”，接近随机的性能是真实存在的。
- **长音频退化。** 超过 10 分钟后，大多数模型的说话人归因会崩。先做说话人分离（第 6 课），再做摘要。
- **沉默上的幻觉。** 使用 Whisper 编码器的 LALM 会继承类似 Whisper 的问题。使用 VAD 做门控。
- **基准挑樱桃。** 厂商博客会突出最好看的类别。你应该自己跑 MMAU-Pro 的多音频子集。

## 交付成果

保存为 `outputs/skill-alm-picker.md`。为给定的音频理解任务选择 LALM、基准子集和输出模态（文本或语音）。

## 练习

1. **简单。** 运行 `code/main.py`，观察一个玩具投影器模式和假的 LALM 路由如何把（音频嵌入，文本 token）变成输出 token。
2. **中等。** 在 100 个 MMAU-Pro 语音样本上评测 Qwen2.5-Omni-7B。与论文报告的数字比较。
3. **困难。** 构建一个最小音频描述 baseline：BEATs 编码器 + 2 层投影器 + 冻结的 Llama-3.2-1B。只在 AudioCaps 上微调投影器。与 SALMONN 在 Clotho-AQA 上比较。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| LALM | 音频版 ChatGPT | 音频编码器 + 投影器 + LLM 解码器。 |
| Projector | 适配器 | 把音频特征映射到 LLM 嵌入空间的小型 MLP。 |
| MMAU | 核心基准 | 横跨语音、声音、音乐的 10k 个音频问答对。 |
| MMAU-Pro | 更难的 MMAU | 1800 个多音频 / 重推理问题。 |
| LongAudioBench | 长音频评测 | 带语义查询的多分钟音频片段。 |
| Voice-in / voice-out | 原生语音 | 模型摄入语音并发出语音，不绕过文本。 |

## 延伸阅读

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) - 参考架构。
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) - 语音输入、语音输出。
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) - 开源长音频领先模型。
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) - LongAudioBench SOTA。
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) - 双编码器先驱。
- [MMAU-Pro leaderboard](https://mmaubenchmark.github.io/) - 2026 年实时排名。
