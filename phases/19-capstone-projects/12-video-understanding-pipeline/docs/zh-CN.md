# 综合项目 12 — 视频理解管线（场景、问答、搜索）

> Twelve Labs 将 Marengo + Pegasus 产品化。VideoDB 发布了面向视频的 CRUD API。AI2 的 Molmo 2 发布了开放 VLM checkpoint。Gemini 长上下文可以原生处理数小时视频。TimeLens-100K 在规模化场景下定义了时间定位。2026 年的管线形态已经稳定：场景分割、逐场景 caption + embedding、转写对齐、多向量索引，以及返回 `(start, end)` 时间戳和帧预览的查询。这个综合项目要摄取 100 小时视频，跑公开基准，并衡量计数和动作问题上的幻觉。

**类型:** Capstone
**语言:** Python（pipeline），TypeScript（UI）
**先修:** Phase 4（CV），Phase 6（speech），Phase 7（transformers），Phase 11（LLM engineering），Phase 12（multimodal），Phase 17（infrastructure）
**覆盖阶段:** P4 · P6 · P7 · P11 · P12 · P17
**时间:** 30 小时

## 要解决的问题

长视频问答是 2026 年规模下最吃带宽的多模态问题。Gemini 2.5 Pro 可以原生读取 2 小时视频，但要把 100 小时视频摄取成可查询语料库，仍然需要场景级索引。生产形态会组合场景分割（TransNetV2 或 PySceneDetect）、用 VLM 做逐场景 caption（Gemini 2.5、Qwen3-VL-Max 或 Molmo 2）、转写对齐（带词级时间戳的 Whisper-v3-turbo），以及把 caption、帧 embedding 和转写并排存储的多向量索引。查询管线返回 `(start, end)` 时间戳和帧预览。

基准包括公开的 ActivityNet-QA、NeXT-GQA，以及你自己的 100 题自定义集合。计数类和动作类问题上的幻觉是已知困难失败类型；本综合项目会明确测量它。

## 核心概念

摄取时有三条管线并行运行。**场景分割**把视频切成场景。**VLM captioning**为每个场景生成 caption，并从关键帧生成帧 embedding。**ASR 对齐**产生词级时间戳。三条流通过 `(scene_id, time range)` 连接起来。每个场景在多向量索引（Qdrant）里获得三类向量：caption embedding、keyframe embedding、transcript embedding。

查询时，自然语言问题会同时打到三类向量；结果用 RRF 合并；一个 TimeLens 风格的时间定位 adapter 会在 top scene 内细化 `(start, end)` 窗口。VLM synthesizer（Gemini 2.5 Pro 或 Qwen3-VL-Max）接收 query + top scenes + cropped frames，并用带引用的时间戳和帧预览回答。

幻觉测量很重要。计数问题（“有多少人进入房间？”）和动作类问题（“厨师是在搅拌前倒入的吗？”）出了名不可靠。把它们的准确率和描述性问题分开报告。

## 架构

```text
video file / URL
      |
      v
PySceneDetect / TransNetV2  (scene segmentation)
      |
      +--- per-scene keyframe --- VLM caption + frame embedding
      |                            (Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2)
      |
      +--- audio channel --- Whisper-v3-turbo ASR + word timestamps
      |
      v
multi-vector Qdrant: {caption_emb, keyframe_emb, transcript_emb}
      |
query:
  dense queries against all three -> RRF merge -> top-k scenes
      |
      v
TimeLens / VideoITG temporal grounding (refine start/end within scene)
      |
      v
VLM synth: query + top scenes + frame previews
      |
      v
answer + (start, end) timestamps + frame thumbs + citations
```

## 技术栈

- 场景分割：TransNetV2（2024-26 的 state-of-the-art）或 PySceneDetect
- ASR：通过 faster-whisper 使用带词级时间戳的 Whisper-v3-turbo
- VLM captioner + answerer：Gemini 2.5 Pro、Qwen3-VL-Max 或 Molmo 2
- 时间定位：基于 TimeLens-100K 训练的 adapter 或 VideoITG
- 索引：支持多向量的 Qdrant（caption / frame / transcript）
- UI：Next.js 15，配 HTML5 video player 和场景缩略图
- Eval：ActivityNet-QA、NeXT-GQA、自定义 100 题人工标注集合
- 幻觉基准：带人工标签的计数和动作类型子集

## 动手实现

1. **摄取遍历器。** 接收 YouTube URL 或本地 MP4。必要时降采样到 720p。持久化 `{video_id, file_path}`。

2. **场景分割。** 运行 TransNetV2 或 PySceneDetect，生成 `[{scene_id, start_ms, end_ms, keyframe_path}]`。目标 100 小时：约 6k-8k 个场景。

3. **ASR pass。** 在音频上运行 Whisper-v3-turbo；导出词级时间戳；切成逐场景 transcript slices。

4. **VLM captioning。** 对每个场景，用 keyframe 和短 caption template 调用 Gemini 2.5 Pro（或 Qwen3-VL-Max）。产出 caption + frame embedding。

5. **多向量索引。** Qdrant collection 使用三个 named vectors。Payload：`{video_id, scene_id, start_ms, end_ms, keyframe_url}`。

6. **查询。** 自然语言问题触发三次 dense queries；用 reciprocal rank fusion 合并；`top-k=5` 个场景。

7. **时间定位。** 在 top scene 上运行 TimeLens 风格 adapter，把场景内的 `(start, end)` 窗口细化。

8. **VLM synth。** 用 query + top-3 scene clips（图像或短片段）+ transcripts 调用 Gemini 2.5 Pro。要求给出 `(video_id, start_ms, end_ms)` citations。

9. **Eval。** 运行 ActivityNet-QA 和 NeXT-GQA。构建 100 题自定义集合。报告 overall accuracy + per-class breakdown（counting、action、descriptive）。

## 实际使用

```text
$ video-qa ask --url=https://youtube.com/watch?v=X "how many cars pass the intersection in the first minute?"
[scene]    23 scenes detected
[asr]      transcript complete, 4m12s
[index]    69 vectors written (23 scenes x 3)
[query]    top scene: scene 3 [01:32-01:54], confidence 0.84
[ground]   refined window: [00:12-00:58]
[synth]    gemini 2.5 pro, 1.4s
answer:    5 cars pass the intersection between 00:12 and 00:58.
citations: [scene 3: 00:12-00:58]
          [frame preview at 00:14, 00:27, 00:44, 00:51, 00:57]
```

## 交付成果

`outputs/skill-video-qa.md` 是交付物。给定 YouTube URL 或上传的视频后，管线会索引场景，并用带时间戳的引用回答问题。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 时间定位 IoU | 在留出 grounding set 上计算 intersection-over-union |
| 20 | QA 准确率 | NeXT-GQA 和自定义 100 题 |
| 20 | 摄取吞吐 | 每美元处理的视频小时数 |
| 20 | UI 和引用 UX | 时间戳链接、缩略图条、jump-to-frame |
| 15 | 幻觉率 | 分开统计计数和动作类型准确率 |
| **100** | | |

## 练习

1. 在 captioning pass 中把 Gemini 2.5 Pro 换成 Qwen3-VL-Max。在人工评分的 50-scene sample 上报告 caption quality delta。

2. 把逐场景 frame embedding 从 multi-vector 降为一个 pooled vector。测量 retrieval regression。

3. 构建一个 “counting strict” 模式：synthesizer 提取每个被计数实例及其时间戳，用户点击验证。衡量 user-verification 是否减少幻觉。

4. Benchmark ingest cost：跨三种 VLM 选择比较 hours-of-video-per-dollar。选出 sweet spot。

5. 添加带 speaker diarization 的 transcript：在音频上运行 pyannote speaker diarization，并嵌入逐说话人 transcript。演示 “what did Alice say about X?” 查询。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| 场景分割 | “Shot detection” | 在镜头边界把视频切成场景 |
| 多向量索引 | “Caption + frame + transcript” | 每种表示都有 named vectors 的 Qdrant collection |
| 时间定位 | “When exactly did it happen” | 为查询答案细化 `(start, end)` 窗口 |
| 帧 embedding | “Visual representation” | 关键帧的向量 embedding；用于场景视觉相似度 |
| RRF fusion | “Reciprocal rank fusion” | 跨多个排序列表的合并策略；经典 hybrid-retrieval 技巧 |
| 计数幻觉 | “Miscount” | VLM 在 “how many X” 问题上的已知失败模式 |
| ActivityNet-QA | “Video-QA benchmark” | 长视频 QA 准确率基准 |

## 延伸阅读

- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — 开放 VLM checkpoint
- [TimeLens (CVPR 2026)](https://github.com/TencentARC/TimeLens) — 规模化时间定位
- [Gemini Video long-context](https://deepmind.google/technologies/gemini) — hosted reference
- [VideoDB](https://videodb.io) — 面向视频的 CRUD API reference
- [Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io) — commercial reference
- [TransNetV2](https://github.com/soCzech/TransNetV2) — 场景分割模型
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — 经典开放替代方案
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467) — reference eval benchmark
