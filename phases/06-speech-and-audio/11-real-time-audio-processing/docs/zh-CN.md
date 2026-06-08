# 实时音频处理

> Batch pipelines 处理一个文件。Real-time pipelines 必须在下一个 20 milliseconds 到来之前处理完当前 20 milliseconds。每个 conversational AI、broadcast studio 和 telephony bot 都由这个 latency budget 决定生死。

**类型：** Build
**语言：** Python
**先修：** Phase 6 · 02 (Spectrograms), Phase 6 · 04 (ASR), Phase 6 · 07 (TTS)
**时间：** ~75 minutes

## 要解决的问题

你想要一个感觉鲜活的语音助手。人类会话 turn-taking latency 约为 230 ms（silence-to-response）。超过 500 ms 就会显得机械；超过 1500 ms 就会显得坏掉。2026 年完整 **hear → understand → respond → speak** loop 的预算是：

| Stage | Budget |
|-------|--------|
| Mic → buffer | 20 ms |
| VAD | 10 ms |
| ASR (streaming) | 150 ms |
| LLM (first token) | 100 ms |
| TTS (first chunk) | 100 ms |
| Render → speaker | 20 ms |
| **Total** | **~400 ms** |

Moshi（Kyutai，2024）实现了 200 ms full-duplex。GPT-4o-realtime（2024）约为 320 ms。2022 年上线的 cascaded pipelines 还在 2500 ms。10× 改善来自三项技术：（1）处处 streaming，（2）用 partial results 做 asynchronous pipelining，（3）interruptible generation。

## 核心概念

![Streaming audio pipeline with ring buffer, VAD gate, interruption](../assets/real-time.svg)

**Frame / chunk / window。** 实时音频以固定大小 blocks 流动。常见选择：20 ms（16 kHz 下 320 samples）。下游所有东西都必须跟上这个 cadence。

**Ring buffer。** 固定大小 circular buffer。Producer thread 写入新 frames，consumer thread 读取。避免 hot path 中的 allocations。大小 ≈ maximum-latency × sample-rate；2 秒 16 kHz ring = 32,000 samples。

**VAD (Voice Activity Detection)。** 没人说话时 gate downstream work。Silero VAD 4.0 (2024) 在 CPU 上每 30 ms frame <1 ms。`webrtcvad` 是更老的替代。

**Streaming ASR。** 随着音频到达而输出 partial transcripts 的模型。Parakeet-CTC-0.6B 在 streaming mode（NeMo，2024）下，以 320 ms latency 达到 2-5% WER。Whisper-Streaming（Macháček et al., 2023）把 Whisper 分块，获得约 2 s latency 的近 streaming。

**Interruption。** 当用户在助手说话时开口，你必须（a）检测 barge-in，（b）停止 TTS，（c）丢弃剩余 LLM output。所有这些要在 100 ms 内完成，否则用户会觉得助手“听不见”。

**WebRTC Opus transport。** 20 ms frames，48 kHz，adaptive bitrate 8-128 kbps。浏览器和移动端标准。LiveKit、Daily.co、Pion 是 2026 年构建 voice apps 的栈。

**Jitter buffer。** 网络 packets 会乱序 / 延迟到达。Jitter buffer 负责重排和平滑；太小 → 可听空洞，太大 → 延迟。典型 60-80 ms。

### 常见坑点

- **Thread contention。** Python 的 GIL + heavy models 可能饿死 audio thread。使用 C-callback audio library（sounddevice、PortAudio），并让 Python 离开 hot path。
- **Sample-rate conversion latency。** Pipeline 内部 resampling 会增加 5-20 ms。要么 upfront resample，要么使用 zero-latency resampler（PolyPhase、`soxr_hq`）。
- **TTS priming。** 即使 Kokoro 这样的快速 TTS，第一次请求也有 100-200 ms warm-up。缓存模型，并在第一个真实 turn 之前用 dummy run 预热。
- **Echo cancellation。** 没有 AEC，TTS output 会重新进入麦克风，让 ASR 识别 bot 自己的声音。WebRTC AEC3 是开源默认选择。

## 动手实现

### Step 1：ring buffer

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

Capacity 决定最大 buffering latency。16 kHz 下 32,000 samples = 2 s。

### Step 2：VAD gate

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

生产中替换为 Silero VAD：

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### Step 3：streaming ASR

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### Step 4：interruption handler

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # barge-in
        # then feed to streaming ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

关键在 async I/O 和可取消的 TTS streaming。音频轨上调用 WebRTC peerconnection.stop() 是 canonical way。

## 实际使用

2026 年的栈：

| Layer | Pick |
|-------|------|
| Transport | LiveKit (WebRTC) 或 Pion (Go) |
| VAD | Silero VAD 4.0 |
| Streaming ASR | Parakeet-CTC-0.6B 或 Whisper-Streaming |
| LLM first-token | Groq, Cerebras, vLLM-streaming |
| Streaming TTS | Kokoro 或 ElevenLabs Turbo v2.5 |
| Echo cancel | WebRTC AEC3 |
| End-to-end native | OpenAI Realtime API 或 Moshi |

## 常见陷阱

- **为了安全 buffer 500 ms。** buffer *就是* 你的 latency floor。缩小它。
- **没有 pinning threads。** Audio callback 运行在比 UI 更低优先级的 thread 上 = 负载下出现 glitches。
- **TTS chunks 太小。** 小于 200 ms 的 chunks 会让 vocoder artifacts 变得可听。320 ms chunks 是甜点区。
- **没有 jitter buffer。** 真实网络会 jittery；没有 smoothing 就会有 pops。
- **Single-shot error handling。** Audio pipelines 必须 crash-proof。一个 exception 就会杀死 session。

## 交付成果

保存为 `outputs/skill-realtime-designer.md`。设计一个 real-time audio pipeline，并给出每个 stage 的具体 latency budgets。

## 练习

1. **Easy。** 运行 `code/main.py`。模拟 ring buffer + energy VAD；为一个假的 10 秒 stream 打印 stage latencies。
2. **Medium。** 使用 `sounddevice` 构建 passthrough loop，以 20 ms frames 处理你的麦克风，并在每个 frame 打印 VAD state。
3. **Hard。** 用 `aiortc` 构建 full duplex echo test：browser → WebRTC → Python → WebRTC → browser。用 1 kHz pulse 测量 glass-to-glass latency。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Ring buffer | The circular queue | 用于 audio frames 的固定大小、lock-free（或 SPSC-locked）FIFO。 |
| VAD | Silence gate | 标记 speech vs non-speech 的模型或 heuristic。 |
| Streaming ASR | Real-time STT | 音频到达时输出 partial text；有界 lookahead。 |
| Jitter buffer | Network smoother | 重排 out-of-order packets 的 queue；典型 60-80 ms。 |
| AEC | Echo cancellation | 减去 speaker-to-mic feedback path。 |
| Barge-in | User interrupt | 系统检测 mid-TTS 用户语音；必须取消 playback。 |
| Full duplex | Simultaneous both ways | 用户和 bot 可以同时说话；Moshi 是 full duplex。 |

## 延伸阅读

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) - chunked near-streaming Whisper。
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) - full-duplex 200 ms latency。
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) - production audio agent orchestration。
- [Silero VAD repo](https://github.com/snakers4/silero-vad) - sub-1 ms VAD, Apache 2.0。
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) - open source 下的 echo cancellation。
