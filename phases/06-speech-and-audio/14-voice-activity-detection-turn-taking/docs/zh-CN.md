# 语音活动检测与轮次接管：Silero、Cobra 与 Flush 技巧

> 每个语音智能体的成败都取决于两个决策：用户现在是否在说话，以及用户是否已经说完？VAD 回答第一个问题。轮次检测（VAD + 静音延迟 + 语义端点模型）回答第二个问题。任意一个做错，你的助手要么打断用户，要么一直说个不停。

**类型：** Build
**语言：** Python
**先修：** Phase 6 · 11 (Real-Time Audio), Phase 6 · 12 (Voice Assistant)
**时间：** ~45 分钟

## 要解决的问题

语音智能体在每个 20 ms chunk 上都会做三个不同的决策：

1. **这一帧是语音吗？** — VAD。二分类，逐帧判断。
2. **用户开始了新的 utterance 吗？** — 起音检测。
3. **用户说完了吗？** — 端点检测（turn-end）。

朴素答案（能量阈值）在任何噪声下都会失败：交通声、键盘声、人群嘈杂声。2026 年的答案是：Silero VAD（开放、深度学习模型）+ 轮次检测模型（语义端点）+ 经过 VAD 校准的静音延迟。

## 核心概念

![VAD 级联：energy → Silero → turn-detector → flush trick](../assets/vad-turn-taking.svg)

### 三层 VAD 级联

**第 1 层：energy gate。** 最便宜。用 -40 dBFS 的 RMS 阈值过滤。能滤掉明显静音，但任何超过阈值的噪声都会触发。

**第 2 层：Silero VAD**（2020-2026，MIT）。100 万参数。基于 6000+ 种语言训练。单 CPU 线程上每个 30 ms chunk 约 1 ms 跑完。在 5% FPR 下 TPR 为 87.7%。这是开源默认选择。

**第 3 层：语义轮次检测器。** LiveKit 的 turn-detection 模型（2024-2026）或你自己的小分类器。它区分“句中暂停”和“已经说完”。使用语言上下文（语调 + 最近词语），而不只是静音。

### 关键参数及其默认值

- **Threshold。** Silero 输出概率；在 &gt; 0.5（默认）或 &gt; 0.3（敏感）时分类为语音。阈值越低，首词被截断越少，误报越多。
- **Minimum speech duration。** 拒绝短于 250 ms 的语音，通常是咳嗽或椅子噪声。
- **Silence hangover（端点检测）。** VAD 回到 0 后，等待 500-800 ms 再声明轮次结束。太短 → 打断用户。太长 → 感觉迟钝。
- **Pre-roll buffer。** 在 VAD 触发前保留 300-500 ms 音频。防止 “hey” 被截断。

### Flush 技巧（Kyutai 2025）

流式 STT 模型有 look-ahead 延迟（Kyutai STT-1B 为 500 ms，STT-2.6B 为 2.5 s）。通常你必须在语音结束后再等这么久才能拿到转写。Flush 技巧：当 VAD 触发语音结束时，**向 STT 发送 flush 信号**，强制它立刻输出。STT 以约 4× 实时速度处理，所以 500 ms 缓冲会在约 125 ms 内完成。

端到端：125 ms VAD + flush STT = 对话级延迟。

### 2026 VAD 对比

| VAD | TPR @ 5% FPR | Latency | License |
|-----|--------------|---------|---------|
| WebRTC VAD (Google, 2013) | 50.0% | 30 ms | BSD |
| Silero VAD (2020-2026) | 87.7% | ~1 ms | MIT |
| Cobra VAD (Picovoice) | 98.9% | ~1 ms | commercial |
| pyannote segmentation | 95% | ~10 ms | MIT-ish |

Silero 是正确默认选择。Cobra 是合规 / 准确性升级项。2026 年生产中不该再只有 energy-only VAD。

## 动手实现

### Step 1：energy gate

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### Step 2：Python 中的 Silero VAD

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### Step 3：turn-end 状态机

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### Step 4：flush 技巧骨架

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

STT（Kyutai、Deepgram、AssemblyAI）必须支持 flush 才能工作。Whisper streaming 不支持，它是基于 block 的，并且总是等待 chunk。

## 实际使用

| Situation | VAD choice |
|-----------|-----------|
| 开放、快速、通用 | Silero VAD |
| 商业呼叫中心 | Cobra VAD |
| 端侧（手机） | Silero VAD ONNX |
| 研究 / diarization | pyannote segmentation |
| 零依赖后备 | WebRTC VAD（legacy） |
| 需要高质量轮次结束 | Silero + LiveKit turn-detector 分层 |

经验法则：除非你真的没有其他选择，否则不要发布 energy-only VAD。

## 常见陷阱

- **固定阈值。** 安静环境有效，嘈杂环境失败。要么在设备上校准，要么切换到 Silero。
- **静音延迟太短。** 智能体会在句中打断。500-800 ms 是对话语音的甜点区。
- **延迟太长。** 感觉迟钝。用目标用户做 A/B test。
- **没有 pre-roll buffer。** 用户音频的前 200-300 ms 会丢失。始终保留滚动 pre-roll。
- **忽略语义端点。** “Hmm, let me think...” 包含长暂停。用户讨厌思路中途被打断。使用 LiveKit 的 turn-detector 或类似模型。

## 交付成果

保存为 `outputs/skill-vad-tuner.md`。为某个工作负载选择 VAD 模型、阈值、hangover、pre-roll 和轮次检测策略。

## 练习

1. **Easy。** 运行 `code/main.py`。它会模拟语音 + 静音 + 语音 + 咳嗽序列，并测试三层 VAD。
2. **Medium。** 安装 `silero-vad`，处理一段 5 分钟录音，调节阈值以同时最小化首词截断和误触发。报告 precision/recall。
3. **Hard。** 构建一个迷你 turn-detector：Silero VAD + 基于最后 10 个词 embedding 的 3 层 MLP（使用 sentence-transformers）。在手工标注的 turn-end 数据集上训练。F1 比 Silero-only 提升 10%。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| VAD | 语音检测器 | 逐帧二分类：这是语音吗？ |
| Turn detection | 端点检测 | VAD + silence-hangover + semantic endpoint。 |
| Silence hangover | 语音后等待 | 在声明轮次结束前等待的时间；500-800 ms。 |
| Pre-roll | 语音前缓冲 | 在 VAD 触发前保留 300-500 ms 音频。 |
| Flush trick | Kyutai hack | VAD → flush-STT → 125 ms，而不是 500 ms 延迟。 |
| Semantic endpoint | “他们是想停了吗？” | 查看词语而不只是静音的 ML 分类器。 |
| TPR @ FPR 5% | ROC 点 | 标准 VAD benchmark；Silero 为 87.7%，WebRTC 为 50%。 |

## 延伸阅读

- [Silero VAD](https://github.com/snakers4/silero-vad) — 参考级开放 VAD。
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/) — 商业准确率领先者。
- [Kyutai — Unmute + flush trick](https://kyutai.org/stt) — 低于 200 ms 的工程技巧。
- [LiveKit — turn detection](https://docs.livekit.io/agents/logic/turns/) — 生产中的语义端点。
- [WebRTC VAD](https://webrtc.googlesource.com/src/) — legacy baseline。
- [pyannote segmentation](https://github.com/pyannote/pyannote-audio) — diarization 级分割。
