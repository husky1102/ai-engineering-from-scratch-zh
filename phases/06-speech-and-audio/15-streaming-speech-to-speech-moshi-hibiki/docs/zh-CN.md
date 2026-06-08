# 流式语音到语音：Moshi、Hibiki 与全双工对话

> 2024-2026 年重新定义了语音 AI。Moshi 发布了一个能以 200 ms 延迟同时听和说的单一模型。Hibiki 逐 chunk 做 speech-to-speech translation。二者都放弃了 ASR → LLM → TTS pipeline，转向基于 Mimi codec token 的统一全双工架构。这是新的参考设计。

**类型：** Learn
**语言：** Python
**先修：** Phase 6 · 13 (Neural Audio Codecs), Phase 6 · 11 (Real-Time Audio), Phase 7 · 05 (Full Transformer)
**时间：** ~75 分钟

## 要解决的问题

Lessons 11 + 12 构建的每个语音智能体都有一个约 300-500 ms 的基础延迟下限：VAD 触发，STT 处理，LLM 推理，TTS 生成。每个阶段都有自己的最低延迟。你可以调优和并行化，但 pipeline 形状限制了上限。

Moshi（Kyutai，2024-2026）提出了一个不同的问题：如果没有 pipeline 呢？如果一个模型直接接收音频并持续输出音频，而文本只是中间的“inner monologue”，不是必需阶段呢？

答案是 **full-duplex speech-to-speech**。理论延迟 160 ms（80 ms Mimi frame + 80 ms acoustic delay）。单张 L4 GPU 上实际延迟 200 ms。这是最佳流水线式语音智能体的一半。

## 核心概念

![Moshi 架构：两个并行 Mimi 流 + inner-monologue text](../assets/moshi-hibiki.svg)

### Moshi 架构

**输入。** 两个 Mimi codec 流，二者都是 12.5 Hz × 8 codebooks：

- Stream 1：用户音频（Mimi 编码，持续到达）
- Stream 2：Moshi 自己的音频（由 Moshi 生成）

**Transformer。** 一个 70 亿参数 Temporal Transformer 同时处理这两个流和一个文本 “inner monologue” 流。在每个 80 ms step，它会：

1. 消费最新的用户 Mimi token（8 个 codebooks）。
2. 消费最近的 Moshi Mimi token（8 个 codebooks，即已经生成的内容）。
3. 生成下一个 Moshi text token（inner monologue）。
4. 通过一个小 Depth Transformer 生成下一个 Moshi Mimi token（8 个 codebooks）。

三个流：用户音频、Moshi 音频、Moshi 文本，并行运行。Moshi 可以在自己说话时听见用户；可以在用户打断时打断自己；可以 back-channel（“mhm”）而不破坏自己的主 utterance。

**Depth transformer。** 在一帧内部，8 个 codebooks 不是并行预测的，它们有 codebook 间依赖。一个小型 2 层 “depth transformer” 会在 80 ms 内按顺序预测它们。这是 AR codec LM 的标准分解方式（VALL-E、VibeVoice 也使用）。

### 为什么 inner-monologue text 有帮助

没有显式文本时，模型必须在声学流中隐式建模语言。Moshi 的洞察是：强制它在音频旁边同时发出 text token。文本流本质上就是 Moshi 正在说的话的转写。这提升语义连贯性，让替换语言模型 head 更容易，并且免费给你 transcript。

### Hibiki：流式语音到语音翻译

同一架构，基于翻译对训练。源音频输入，目标语言音频持续输出。Hibiki-Zero（2026 年 2 月）消除了对词级对齐训练数据的需要，使用句级数据 + GRPO 强化学习进行延迟优化。

最初支持四个语言对；使用约 1000 小时数据即可适配到新语言。

### 更广泛的 Kyutai 技术栈（2026）

- **Moshi** — 全双工对话（法语优先，英语支持很好）
- **Hibiki / Hibiki-Zero** — 同声语音翻译
- **Kyutai STT** — 流式 ASR（500 ms 或 2.5 s look-ahead）
- **Kyutai Pocket TTS** — 100M 参数 TTS，可在 CPU 上运行（2026 年 1 月）
- **Unmute** — 在公共服务器上组合这些能力的完整 pipeline

L40S GPU 上吞吐：64 个并发 session，3× real-time。

### Sesame CSM：表亲

Sesame CSM（2025）使用类似想法：Llama-3 backbone + Mimi codec head。但 CSM 是单向的（接收 context + text，产生 speech），而不是 full-duplex。它是市场上最佳 “voice presence” TTS；但和 Moshi 的全双工能力并不完全相同。

### 2026 性能数字

| Model | Latency | Use case | License |
|-------|---------|----------|---------|
| Moshi | 200 ms (L4) | full-duplex English / French dialogue | CC-BY 4.0 |
| Hibiki | 12.5 Hz framerate | French ↔ English streaming translation | CC-BY 4.0 |
| Hibiki-Zero | same | 5 language-pairs, no aligned data | CC-BY 4.0 |
| Sesame CSM-1B | 200 ms TTFA | context-conditioned TTS | Apache-2.0 |
| GPT-4o Realtime | ~300 ms | closed, OpenAI API | commercial |
| Gemini 2.5 Live | ~350 ms | closed, Google API | commercial |

## 动手实现

### Step 1：接口

Moshi 暴露一个 WebSocket server，接收 80 ms 的 Mimi 编码音频 chunk，并返回 80 ms 的 Mimi 编码音频 chunk。双向。持续不断。

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### Step 2：全双工循环

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

两个方向同时运行。Python asyncio 或 Rust futures 是标准传输方式。

### Step 3：训练目标（概念性）

对每个 80 ms frame `t`：

- 输入：`user_mimi[0..t]`、`moshi_mimi[0..t-1]`、`moshi_text[0..t-1]`
- 预测：`moshi_text[t]`，然后是 `moshi_mimi[t, codebook_0..7]`

文本先于音频预测（inner monologue）；音频在 depth transformer 内按 codebook 顺序预测。

### Step 4：Moshi 赢在哪里，不赢在哪里

Moshi 赢在：

- 便宜硬件上端到端低于 250 ms。
- 自然的 back-channel 和打断。
- 不需要 pipeline glue code。

Moshi 不赢在：

- Tool calling（没有为此训练；你需要单独的 LLM path）。
- 长推理（Moshi 是 8B 左右的对话模型，不是 Claude/GPT-4）。
- 小众主题的事实准确性。
- 大多数生产企业用例（2026 年仍使用 pipelines）。

## 实际使用

| Situation | Pick |
|-----------|------|
| 最低延迟语音陪伴 | Moshi |
| 实时翻译通话 | Hibiki |
| 语音 demo / research | Moshi, CSM |
| 带工具的企业智能体 | Pipeline（Lesson 12），不是 Moshi |
| 上下文中的自定义语音 TTS | Sesame CSM |
| 任意语言的 speech-to-speech | GPT-4o Realtime 或 Gemini 2.5 Live（commercial） |

## 常见陷阱

- **有限的 tool calling。** Moshi 是对话模型，不是 agent framework。和 pipeline 结合来使用工具。
- **特定声音条件控制。** Moshi 使用单个训练 persona；克隆声音是另一次单独训练。
- **语言覆盖。** 法语 + 英语很优秀；其他语言有限。Hibiki-Zero 有帮助，但你仍需要训练数据。
- **资源成本。** 一个完整 Moshi session 占用一个 GPU slot；不是便宜的共享租户部署模式。

## 交付成果

保存为 `outputs/skill-duplex-pipeline.md`。为一个语音智能体工作负载选择 pipeline 或 full-duplex 架构，并说明理由。

## 练习

1. **Easy。** 运行 `code/main.py`。它会符号化模拟 two-stream + inner-monologue 架构。
2. **Medium。** 从 HuggingFace 拉取 Moshi，运行 server，测试一次对话。测量从用户语音结束到 Moshi 响应开始的 wall-clock latency。
3. **Hard。** 拿你的 Lesson 12 pipeline agent，与 Moshi 在 20 条匹配测试 utterance 上比较 P50 latency。写下 pipeline 在架构上仍然胜出的场景。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Full-duplex | 同时听和说 | 同一个模型上两个音频流同时活跃。 |
| Inner monologue | 模型的文本流 | Moshi 在输出音频旁边同时发出 text token。 |
| Depth transformer | codebook 间预测器 | 在一个 80 ms frame 内预测 8 个 codebooks 的小 transformer。 |
| Mimi | Kyutai 的 codec | 12.5 Hz × 8 codebooks；semantic+acoustic；驱动 Moshi。 |
| Streaming S2S | 现场 audio → audio | 逐 chunk 翻译 / 对话，没有 pipeline stages。 |
| Back-channeling | “Mhm” 反应 | Moshi 可以发出小的确认声，而不破坏自己的轮次。 |

## 延伸阅读

- [Défossez et al. (2024). Moshi — speech-text foundation model](https://arxiv.org/html/2410.00037v2) — 论文。
- [Kyutai Labs (2026). Hibiki-Zero](https://arxiv.org/abs/2602.12345) — 无对齐数据的流式翻译。
- [Sesame (2025). Crossing the uncanny valley of voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) — CSM spec。
- [Kyutai — Moshi repo](https://github.com/kyutai-labs/moshi) — 安装 + server。
- [OpenAI — Realtime API](https://platform.openai.com/docs/guides/realtime) — 闭源商业 peer。
- [Kyutai — Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling) — 底层 STT/TTS framework。
