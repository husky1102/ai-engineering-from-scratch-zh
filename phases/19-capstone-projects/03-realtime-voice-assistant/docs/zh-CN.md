# 综合项目 03：实时语音助手（ASR 到 LLM 到 TTS）

> 一个感觉对的 voice agent 需要 end-to-end latency 低于 800ms，知道你什么时候停止说话，能处理 barge-in，并且能在不 stalling 的情况下调用 tool。Retell、Vapi、LiveKit Agents 和 Pipecat 在 2026 年都达到了这条线。它们的形态相同：streaming ASR、turn-detector、streaming LLM、streaming TTS，全部通过 WebRTC 串接，并在每一跳都有 aggressive latency budgets。构建一个，测量 WER、MOS 和 false-cutoff rate，并在 packet loss 下运行它。

**类型:** Capstone
**语言:** Python (agent + pipeline), TypeScript (web client)
**先修:** Phase 6 (speech and audio), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 17 (infrastructure)
**练习阶段:** P6 · P7 · P11 · P13 · P14 · P17
**时间:** 30 hours

## 要解决的问题

Voice 是 2025-2026 年变化最快的 AI UX category。技术 ceiling 每个季度都在下降。OpenAI Realtime API、Gemini 2.5 Live、Cartesia Sonic-2、ElevenLabs Flash v3、LiveKit Agents 1.0 和 Pipecat 0.0.70 都让 sub-800ms first-audio-out 触手可及。门槛不只是 latency。它是 interaction feel：不打断用户，不被打断，能从 mid-sentence interruption 中恢复，能在对话中调用 tool 而不让 audio stall，能在 jittery mobile networks 下存活。

你无法通过拼接三个 REST calls 达到这个效果。架构必须是端到端 pipelined streaming。构建之后，failure modes 会变得可见：针对 phone audio 调好的 VAD 被背景电视触发，turn-detector 等待永远不会出现的 punctuation，TTS 在发出声音前 buffer 400ms。本 capstone 是在 load 下逐一修复这些问题，并发布一份 latency-and-quality report。

## 核心概念

Pipeline 有五个 streaming stages：**audio in**（来自 browser 或 PSTN 的 WebRTC）、**ASR**（Deepgram Nova-3 或 faster-whisper 的 streaming partial transcripts）、**turn detection**（VAD 加一个读取 partial transcripts 完成度线索的小 turn-detector model）、**LLM**（一旦判断 turn complete 就开始 streaming tokens）、**TTS**（在第一个 LLM token 后约 200ms 内 streaming audio out）。

三个 cross-cutting concerns。**Barge-in**：当用户在 agent 说话时开始说话，TTS cancel，ASR 立即接管。**Tool use**：mid-conversation function calls（weather、calendar）必须通过 side channel 运行，不能 stall audio；如果 latency 超过 300ms，agent 会 pre-fill 一个 acknowledgement token（“one second...”）。**Backpressure**：在 packet loss 下，partial transcripts 会被 hold，VAD 提高 speech-gate threshold，agent 避免盖过未确认的消息。

Measurement bar 是定量的。15 dB SNR 的 Hamming VAD benchmark 上 WER 低于 8%。100 个 measured calls 上 first-audio-out p50 低于 800ms。False-cutoff rate 低于 3%。TTS 的 MOS 高于 4.2。单个 g5.xlarge 上 50 concurrent calls。这些数字就是 deliverable。

## 架构

```text
browser / Twilio PSTN
        |
        v
   WebRTC / SIP edge
        |
        v
  LiveKit Agents 1.0  (or Pipecat 0.0.70)
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5         turn-detector     side-channel
(Deepgram         (Silero)          (LiveKit)        tools
 Nova-3 /         speech-gate    completion score    (weather,
 Whisper-v3)      per 20ms        on partials        calendar)
   |                   |              |
   +--------+----------+--------------+
            v
        LLM (streaming)
     GPT-4o-realtime / Gemini 2.5 Flash /
     cascaded Claude Haiku 4.5
            |
            v
        TTS streaming
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     audio back to caller
            |
            v
   OpenTelemetry voice traces -> Langfuse
```

## 技术栈

- Transport: LiveKit Agents 1.0 (WebRTC) plus Twilio PSTN gateway; Pipecat 0.0.70 as the alternate framework
- ASR: Deepgram Nova-3 (streaming, sub-300ms first partial) or faster-whisper Whisper-v3-turbo self-hosted
- VAD: Silero VAD v5 plus the LiveKit turn-detector (small transformer that reads partial transcripts)
- LLM: OpenAI GPT-4o-realtime for tight integration, Gemini 2.5 Flash Live, or cascaded Claude Haiku 4.5 (streaming completions, separate audio path)
- TTS: Cartesia Sonic-2 (lowest first-byte), ElevenLabs Flash v3, or open-source Orpheus for self-host
- Tools: FastMCP side-channel for weather/calendar/booking; agent pre-emits filler if tool takes >300ms
- Observability: OpenTelemetry voice spans, Langfuse voice traces with audio replay
- Deployment: single g5.xlarge (24GB VRAM) for self-hosted Whisper + Orpheus; hosted APIs for lowest latency

## 动手实现

1. **WebRTC session。** 启动一个 LiveKit room 和一个 streaming microphone audio 的 web client。在 server 上，attach 一个加入 room 的 agent worker。

2. **ASR streaming。** 把 20ms PCM frames 送入 Deepgram Nova-3（或 GPU 上的 faster-whisper）。订阅 partial 和 final transcripts。记录 per-partial latency。

3. **VAD and turn detector。** 在 frame stream 上运行 Silero VAD v5。在 speech-end event 上，对最新 partial transcript 运行 LiveKit turn-detector。只有当 VAD 判断 silence 持续 500ms 且 turn-detector completion score > 0.6 时，才 commit 为 “turn complete”。

4. **LLM stream。** Turn complete 后，用 running conversation 加 final transcript 启动 LLM call。Streaming tokens out。第一个 token 出现时，交给 TTS。

5. **TTS stream。** Cartesia Sonic-2 把 audio chunks streaming 回来。第一个 chunk 必须在第一个 LLM token 后 200ms 内离开 server。把 chunks emit 到 LiveKit room；client 通过 WebRTC jitter buffer 播放。

6. **Barge-in。** 当 VAD 在 TTS 播放期间检测到新的 user speech，立即 cancel TTS stream，丢弃剩余 LLM output，并重新 arm ASR。发布一个 `tts_canceled` span。

7. **Tool side channel。** 把 weather 和 calendar 注册为 function-calling tools。调用时并发执行；如果 300ms 内未 resolve，让 LLM 输出 “one second, let me check” 作为 filler；tool 返回后继续。

8. **Eval harness。** 录制 100 calls。计算 WER（对 held-out transcript）、false-cutoff rate（用户 mid-sentence 时 TTS 被 cancelled）、first-audio-out p50、TTS MOS（human 或 NISQA）以及 jitter-loss test（drop 3% packets）。

9. **Load test。** 用 synthetic caller 在单个 g5.xlarge 上驱动 50 concurrent calls。测量 sustained first-audio-out p95。

## 实际使用

```text
caller: "what is the weather in tokyo tomorrow"
[asr  ] partial @280ms: "what is the"
[asr  ] partial @540ms: "what is the weather"
[turn ] completion score 0.82 at @820ms; commit
[llm  ] first token @960ms
[tool ] weather.tokyo tomorrow -> 68/52 partly cloudy @1140ms
[tts  ] first audio-out @1040ms: "Tokyo tomorrow will be partly cloudy..."
turn latency: 1040ms user-stop -> audio-out
```

## 交付成果

`outputs/skill-voice-agent.md` 是 deliverable。给定一个 domain（customer support、scheduling 或 kiosk），它会启动一个 LiveKit agent，带调到 measurement bar 的 ASR/VAD/LLM/TTS pipeline。Rubric：

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | End-to-end latency | p50 first-audio-out under 800ms across 100 recorded calls |
| 20 | Turn-taking quality | False-cutoff rate under 3% on the Hamming VAD benchmark |
| 20 | Tool-use correctness | Mid-conversation tool calls that return the right data without stalling audio |
| 20 | Reliability under packet loss | WER and turn-taking stability with 3% packet drop injected |
| 15 | Eval harness completeness | Reproducible measurements with public config |
| **100** | | |

## 练习

1. 把 Deepgram Nova-3 换成 g5.xlarge 上的 faster-whisper v3 turbo。测量 latency 和 WER gap。指出 CPU-vs-GPU decisions 在哪里重要。

2. 添加一个 interruption-arbitration policy：当用户在 tool call 期间 barge in，agent 做什么？比较三种 policies（hard cancel、finish-tool-then-stop、queue next turn）。

3. 运行 adversarial turn-detector test：让用户在句中长暂停。调 VAD silence threshold 和 turn-detector score threshold，在不超过 900ms 的情况下最小化 false-cutoff。

4. 通过 Twilio 在 PSTN 上部署同一个 agent。比较 PSTN first-audio-out 与 WebRTC。解释 jitter-buffer 和 codec differences。

5. 为非英语语言（Japanese、Spanish）添加 voice activity detection。测量 Silero VAD v5 false-trigger rate 与 language-specific fine-tunes 的差异。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Turn detection | “End of utterance” | 给定 VAD silence 和 partial transcript 后，判断用户是否说完的 classifier |
| Barge-in | “Interruption handling” | VAD 检测到新的 user speech 时，在 playback 中途 cancel TTS |
| First-audio-out | “Latency” | 从用户停止说话到第一个 audio packet 离开 server 的时间 |
| VAD | “Speech gate” | 把 audio frames 分类为 speech vs silence 的模型；Silero VAD v5 是 2026 默认 |
| Jitter buffer | “Audio smoothing” | Client-side buffer，短暂持有 packets 以吸收 network variance |
| Filler | “Acknowledgment token” | Tool 变慢时 agent 发出的短语，用于避免沉默 |
| MOS | “Mean opinion score” | 感知 speech quality rating；NISQA 是 automated proxy |

## 延伸阅读

- [LiveKit Agents 1.0](https://github.com/livekit/agents) — reference WebRTC agent framework
- [Pipecat](https://github.com/pipecat-ai/pipecat) — alternate Python-first streaming agent framework
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — integrated speech models 的 reference
- [Deepgram Nova-3 documentation](https://developers.deepgram.com/docs) — streaming ASR reference
- [Silero VAD v5](https://github.com/snakers4/silero-vad) — VAD reference model
- [Cartesia Sonic-2](https://docs.cartesia.ai) — low-latency TTS reference
- [Retell AI architecture](https://docs.retellai.com) — production voice agent architecture
- [Vapi.ai production stack](https://docs.vapi.ai) — alternate production reference
