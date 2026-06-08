# 构建语音助手 Pipeline：Phase 6 Capstone

> 把 lessons 01-11 的所有内容缝合起来。构建一个会听、会推理、会说话回应的语音助手。到 2026 年，这已经是一个已解决的工程问题，而不是研究问题；但集成细节决定它能不能上线。

**类型：** Build
**语言：** Python
**先修：** Phase 6 · 04, 05, 06, 07, 11; Phase 11 · 09 (Function Calling); Phase 14 · 01 (Agent Loop)
**时间：** ~120 minutes

## 要解决的问题

构建一个端到端助手：

1. 捕获麦克风输入（16 kHz mono）。
2. 检测用户语音的开始/结束。
3. 流式转录。
4. 把 transcript 传给能调用 tools（timer、weather、calendar）的 LLM。
5. 把 LLM 文本流式送入 TTS。
6. 把 audio 播放回给用户。
7. 如果用户在回应中途打断，则停止。

Latency target：用户说完后 800 ms 内拿到第一个 TTS audio byte，运行在 laptop CPU 上。Quality target：不漏词、不在沉默上 hallucinated subtitles、不发生 voice cloning leakage、不让 prompt injection 成功。

## 核心概念

![Voice assistant pipeline: mic → VAD → STT → LLM+tools → TTS → speaker](../assets/voice-assistant.svg)

### 七个组件

1. **Audio capture。** Mic → 16 kHz mono → 20 ms chunks。Python 中通常是 `sounddevice`，生产中是 native AudioUnit/ALSA/WASAPI。
2. **VAD（Lesson 11）。** Silero VAD @ threshold 0.5, min speech 250 ms, silence hang-over 500 ms。发出 "start" 和 "end" 信号。
3. **Streaming STT（Lesson 4-5）。** Whisper-streaming、Parakeet-TDT 或 Deepgram Nova-3（API）。Partial + final transcripts。
4. **带 tool calling 的 LLM。** GPT-4o / Claude 3.5 / Gemini 2.5 Flash。为 tools 提供 JSON schema。流式输出 tokens。
5. **Streaming TTS（Lesson 7）。** Kokoro-82M（最快开源）或 Cartesia Sonic（商业）。在 20 个 LLM tokens 之后启动 TTS。
6. **Playback。** Speaker out；低带宽网络用 opus-encode。
7. **Interruption handler。** 如果 VAD 在 TTS playback 期间触发，停止 playback，取消 LLM，重新启动 STT。

### 你一定会遇到的三个 failure modes

1. **First-word clip。** VAD 开始得稍晚。用户的 "hey" 丢了。Start threshold 用 0.3，而不是 0.5。
2. **Mid-response interrupt confusion。** 用户打断后 LLM 还在生成；assistant 和用户抢话。把 VAD → cancel-LLM 接起来。
3. **Silence hallucination。** Whisper 在 silent warm-up frames 上输出 "Thanks for watching"。始终 VAD-gate。

### 2026 生产参考栈

| Stack | Latency | License | Notes |
|-------|---------|---------|-------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | commercial API | 2026 行业默认 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 ms | mostly open | DIY-friendly |
| Moshi (full-duplex) | 200-300 ms | CC-BY 4.0 | 单模型；架构不同，lesson 15 |
| Vapi / Retell (managed) | 300-500 ms | commercial | 最快上线；customization 有限 |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | offline | open | Privacy / edge |

## 动手实现

### Step 1：带 chunking 的麦克风捕获（pseudocode）

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### Step 2：VAD-gated turn capture

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### Step 3：streaming STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### Step 4：LLM loop 内的 tool calling

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### Step 5：interruption handling

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## 实际使用

查看 `code/main.py`，里面有一个 runnable simulation，把七个组件都用 stub models 接起来。这样即使没有硬件，你也能看到 pipeline 的形状。真实实现时，把 stubs 替换为：

- `silero-vad`（`pip install silero-vad`）
- `deepgram-sdk` 或 `openai-whisper`
- `openai`（`gpt-4o`）或 `anthropic`
- `kokoro` 或 `cartesia`
- `sounddevice` 用于 I/O

## 常见陷阱

- **永久记录 PII。** Full-turn audio 在大多数司法辖区都是 PII。30 天保留，静态加密。
- **没有 barge-in。** 用户会打断。你的 assistant 必须停止说话。
- **TTS 阻塞。** Synchronous TTS 会阻塞 event loop。使用 async 或单独 thread。
- **没有 tool-call error handling。** Tools 会失败。LLM 必须拿到 error + 重试一次，然后 graceful degrade。
- **过度激进的 hallucination filters。** 过滤过度，assistant 会反复说 "I can't help with that." 过滤不足，它什么都敢说。用 held-out set 校准。
- **没有 wake-word option。** Always-listening 是隐私风险。添加 wake-word gate（Porcupine 或 openWakeWord）。

## 交付成果

保存为 `outputs/skill-voice-assistant-architect.md`。给定 budget + scale + language + compliance constraints，产出完整 stack spec。

## 练习

1. **Easy。** 运行 `code/main.py`。它用 stub modules 模拟一个完整 turn end-to-end，并打印 per-stage latency。
2. **Medium。** 把 STT stub 替换为预录 `.wav` 上的真实 Whisper model。测量 WER 和 end-to-end latency。
3. **Hard。** 添加 tool calling：实现 `get_weather`（任意 API）和 `set_timer`。让 LLM 通过 tools 路由，并验证当用户说 "set a 5 minute timer" 时触发正确函数，spoken reply 会确认。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Turn | A user + assistant round-trip | 一个 VAD-bounded user speech + 一个 LLM-TTS response。 |
| Barge-in | Interruption | 用户在 assistant 说话时开口；assistant 停止。 |
| Wake word | "Hey assistant" | 短关键词 detector；Porcupine、Snowboy、openWakeWord。 |
| End-pointing | Turn ending | VAD + min-silence decision，判断用户已经说完。 |
| Pre-roll | Pre-speech buffer | 保留 VAD 触发前 200-400 ms audio，避免 first-word clip。 |
| Tool call | Function invocation | LLM 发出 JSON；runtime dispatch；result 回流到 in-loop。 |

## 延伸阅读

- [LiveKit - voice agent quickstart](https://docs.livekit.io/agents/) - production-grade reference。
- [Pipecat - voice agent examples](https://github.com/pipecat-ai/pipecat) - DIY-friendly framework。
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) - managed voice-native path。
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) - full-duplex reference（Lesson 15）。
- [Porcupine wake-word](https://picovoice.ai/products/porcupine/) - wake-word gating。
- [Anthropic - tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) - LLM function calling。
