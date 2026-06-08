# Voice Agents：Pipecat 与 LiveKit

> Voice agents 是 2026 年的一等 production category。Pipecat 提供 Python frame-based pipeline（VAD → STT → LLM → TTS → transport）。LiveKit Agents 通过 WebRTC 把 AI models 连接到 users。Premium stacks 的 production latency target 落在 450-600ms end-to-end。

**类型:** Learn
**语言:** Python（stdlib）
**先修:** Phase 14 · 01（Agent Loop），Phase 14 · 12（Workflow Patterns）
**时间:** ~60 分钟

## 学习目标

- 描述 Pipecat 的 frame-based pipeline：DOWNSTREAM（source→sink）和 UPSTREAM（control）。
- 说出 canonical voice pipeline stages，以及 Pipecat 支持哪些 transports。
- 解释 LiveKit Agents 的两种 voice agent classes（MultimodalAgent、VoicePipelineAgent）以及各自适用场景。
- 总结 2026 production latency expectations，以及它们如何驱动 architecture choices。

## 要解决的问题

Voice agents 不是在 text loop 上 bolted on TTS。Latency budgets 很残酷（约 600ms），partial audio 是默认，turn detection 本身是模型，transports 从 telephony SIP 到 WebRTC。你要么构建 frame-based pipeline（Pipecat），要么依赖平台（LiveKit）。

## 核心概念

### Pipecat（pipecat-ai/pipecat）

- Python frame-based pipeline framework。
- `Frame` → `FrameProcessor` chain。
- 两个 flow directions：
  - **DOWNSTREAM**：source → sink（audio in, TTS out）。
  - **UPSTREAM**：feedback and control（cancellation、metrics、barge-in）。
- `PipelineTask` 用 events（`on_pipeline_started`、`on_pipeline_finished`、`on_idle_timeout`）和 observers 管理 lifecycle，用于 metrics/tracing/RTVI。

典型 pipeline：

```text
VAD (Silero) → STT → LLM (context alternates user/assistant) → TTS → transport
```

Transports：Daily、LiveKit、SmallWebRTCTransport、FastAPI WebSocket、WhatsApp。

Pipecat Flows 加入 structured conversations（state machines）。Pipecat Cloud 是 managed runtime。

### LiveKit Agents（livekit/agents）

- 通过 WebRTC 把 AI models 连接到 users。
- Key concepts：`Agent`、`AgentSession`、`entrypoint`、`AgentServer`。
- 两种 voice agent classes：
  - **MultimodalAgent**：通过 OpenAI Realtime 或等价物直接处理 audio。
  - **VoicePipelineAgent**：STT → LLM → TTS cascade；提供 text-level control。
- 通过 transformer model 做 semantic turn detection。
- Native MCP integration。
- 通过 SIP 支持 telephony。
- LiveKit Inference 提供 50+ models 且无需 API keys；plugins 提供另外 200+。

### Commercial platforms

Vapi（optimized premium stack 上约 450-600ms）和 Retell（180 个 test calls 上约 600ms end-to-end）构建在这些之上。当你想要 managed voice stack 且没有 WebRTC team 时，选择平台。

### 这个 pattern 哪里会出错

- **No barge-in handling.** 用户打断；agent 继续说话。Pipecat 需要 UPSTREAM cancel frames，LiveKit 中有等价机制。
- **STT confidence ignored.** 低置信 transcript 被当成 gospel 喂给 LLM。按 confidence gate，或请求确认。
- **TTS mid-sentence cutoff.** Pipeline 在 utterance 中途 cancel 时，TTS 需要知道，否则会截断 audio。
- **Latency budget ignored.** 每个组件增加 50-200ms。上线前先把 chain 总和算出来。

### 2026 典型 latencies

- VAD：20-60ms
- STT partial：100-250ms
- LLM first token：150-400ms
- TTS first audio：100-200ms
- Transport RTT：30-80ms

End-to-end 450-600ms 是 premium。800-1200ms 常见。任何 >1500ms 都感觉坏了。

## 动手实现

`code/main.py` 是 frame-based toy pipeline，包含：

- `Frame` types（audio、transcript、text、tts_audio、control）。
- 带 `process(frame)` 的 `Processor` interface。
- 五阶段 pipeline（VAD → STT → LLM → TTS → transport），使用 scripted processors。
- 一个 UPSTREAM cancel frame，演示 barge-in。

运行它：

```text
python3 code/main.py
```

Trace 显示正常 flow，以及一次停止 TTS mid-utterance 的 barge-in cancel。

## 实际使用

- **Pipecat** 用于 full control：custom processors、Python-first、pluggable providers。
- **LiveKit Agents** 用于 WebRTC-first deployments 和 telephony。
- **Vapi / Retell** 用于无 WebRTC team 的 hosted voice agents。
- **OpenAI Realtime / Gemini Live** 用于 direct audio-in/audio-out（MultimodalAgent）。

## 交付成果

`outputs/skill-voice-pipeline.md` scaffold 一个 Pipecat-shaped voice pipeline，包含 VAD + STT + LLM + TTS + transport 以及 barge-in handling。

## 练习

1. 给 toy pipeline 添加 metrics observer：统计每个 stage 每秒 frames。Latency 积累在哪里？
2. 实现 confidence-gated STT：低于 threshold 时，请求“could you repeat that?”
3. 添加 semantic turn detection：简单规则，如果 transcript 以“?”结尾，则 end of turn。
4. 阅读 Pipecat transport docs。把 stdlib transport 换成 SmallWebRTCTransport config（stub）。
5. 在同一 query 上测量 OpenAI Realtime 与 STT+LLM+TTS cascade。Text-level control 带来什么 latency cost？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Frame | “Event” | Pipeline 中 typed unit of data（audio、transcript、text、control） |
| Processor | “Pipeline stage” | 带 process(frame) 的 handler |
| DOWNSTREAM | “Forward flow” | Source to sink：audio in，speech out |
| UPSTREAM | “Feedback flow” | Control：cancel、metrics、barge-in |
| VAD | “Voice activity detection” | 检测用户是否在说话 |
| Semantic turn detection | “Smart end-of-turn” | 基于模型判断用户是否说完 |
| MultimodalAgent | “Direct audio agent” | Audio in，audio out；中间没有 text |
| VoicePipelineAgent | “Cascade agent” | STT + LLM + TTS；text-level control |

## 延伸阅读

- [Pipecat docs](https://docs.pipecat.ai/getting-started/introduction) — frame-based pipeline、processors、transports
- [LiveKit Agents docs](https://docs.livekit.io/agents/) — WebRTC + voice primitives
- [Vapi](https://vapi.ai/) — managed voice platform
- [Retell AI](https://www.retellai.com/) — managed voice、latency-benchmarked
