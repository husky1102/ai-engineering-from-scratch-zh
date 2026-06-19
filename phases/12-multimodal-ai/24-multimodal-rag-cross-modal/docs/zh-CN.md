# Multimodal RAG 与 Cross-Modal Retrieval

> Vision-native document RAG 只是一个切片。生产级 multimodal RAG 更广：跨 text、images、audio 和 video retrieval，用于 trip planning（“find me a quiet vegan brunch with natural light”）、medical triage（“what injury matches this photo + these notes”）、e-commerce（“outfits similar to this selfie, in my size”）和 field service（“diagnose this engine sound plus photo of the part”）等 workflow。2025 年的三篇 survey（Abootorabi et al.、Mei et al.、Zhao et al.）把子问题规范化为：cross-modal retrieval、retrieval fusion、generation grounding、multimodal evaluation。本课会阅读这些 survey，并设计一个生产 pipeline。

**类型:** Build
**语言:** Python（stdlib，cross-modal retriever with fusion + grounded generator）
**先修:** Phase 12 · 23（ColPali），Phase 11（RAG basics）
**时间:** ~180 分钟

## 学习目标

- 设计 cross-modal retrieval：text → image、image → text、audio → video 等。
- 比较三种 fusion strategy：score fusion、attention-based fusion、MoE fusion。
- 解释 generation grounding：当 source 是多种 modality 混合时，“cite your sources”意味着什么。
- 说出 2025 年三篇典型 multimodal RAG survey 及其子问题 taxonomy。

## 要解决的问题

单模态 RAG 已经是成熟模式：embed query，embed chunks，retrieve，再塞给 LLM。Multimodal RAG 需要：

1. 多个 retrieval heads（每种 modality 都需要 compatible space 中的 embeddings）。
2. 跨 modality 融合 retrieval results。
3. 能引用跨 modality sources 的 generation grounding。
4. 覆盖 cross-modal signal 的 evaluation metrics。

2025 年的 survey 都得出了相同 taxonomy。

## 核心概念

### Cross-modal retrieval

给定 modality A 的 query，检索 modality B 的 documents。三种模式：

1. Shared embedding space。CLIP 与 CLAP 在共享空间中生成 text + image / text + audio embeddings。可以直接跨 modality 做 cosine similarity。受限于 CLIP-trained pairs。

2. Per-modality encoder + translation。Text encoder + image encoder + 一个小型 translator module，在空间之间映射。Gupta et al. 的 Sen2Sen 和其他 2024 年设计。灵活，但增加复杂度。

3. VLM as encoder。使用 VLM 的 hidden states 作为 retrieval representation。VLM 支持的任何 modality 都能用。质量更高，成本也更高。

选择：text+image 用 CLIP / SigLIP 2；text+audio 用 CLAP；frontier quality 的 cross-modal 用 VLM-hidden-states。

### Fusion strategies

你检索到了 10 个结果：5 张图片、3 段文本、2 个 audio clips。如何合并？

Score fusion（最便宜）。每种 modality 有自己的 retriever，每个 retriever 返回 scores。先在 modality 内 normalize scores，再求和。简单，经常有效。

Attention-based fusion。把所有 retrieved items concat，让一个小型 attention network 给它们加权。需要训练。

MoE fusion。Gating network 路由到 modality-specific experts。不同 query type 会走不同路线，例如 visual question 会给 images 更高权重。

生产默认：score fusion，并轻微偏向 query 的 dominant modality。如果 A/B 显示领域内有明显收益，再升级到 MoE。

### Generation grounding

LLM 应该引用每个 claim 由哪个 retrieved item 支撑。对于 multimodal：

- Text source：标准 citation `[1]`。
- Image source：`[img 3]`，加短 caption。
- Audio：`[audio 2 at 0:34]`。

用 grounding-aware data 训练 generator：训练目标中的每个 claim 都标注 source index。推理时，模型会自然发出 citations。

### 2025 年 survey

Abootorabi et al.（arXiv:2502.08826，“Ask in Any Modality”）：multimodal RAG taxonomy。覆盖 retrieval、fusion、generation。覆盖面最广。

Mei et al.（arXiv:2504.08748，“A Survey of Multimodal RAG”）：关注 sub-task benchmarks 与 failure modes。对 evaluation design 有用。

Zhao et al.（arXiv:2503.18016）：vision-focused survey。对 ColPali-family work 写得很强。

读完三篇，就能掌握截至 2025 年春季的 state of the art。大多数子问题仍然开放。

### MuRAG：奠基论文

MuRAG（Chen et al., 2022）是第一篇 multimodal RAG。它从 multimodal KB 检索 image + text，再生成答案。在 VLM 浪潮之前证明了可行性。现代系统（REACT、VisRAG、M3DocRAG）都建立在它之上。

### 一个生产 trip-planner 示例

Query：“find me a quiet vegan brunch with natural light.”

Pipeline：

1. Decompose query。“quiet” → audio/review keyword；“vegan brunch” → menu item；“natural light” → image feature。
2. 按 modality 检索：
   - Text retrieval on reviews：“vegan brunch, quiet ambiance.”
   - Image retrieval on restaurant photos：“natural light, airy.”
   - Audio retrieval on ambient-sound clips：“low decibel, no music.”
3. Fuse scores。每家餐厅有一个 composite score。
4. Top-k restaurants → VLM generator with all evidence → 带 citations 的 answer。

这已经远超 text-RAG。每种 modality 都加入了文本单独缺失的信号。

### Agentic multimodal RAG

Multi-hop：如果第一次 retrieval 没有返回高置信度答案，LLM 会重写查询并再次检索。Phase 14 的 Agentic RAG 模式也适用于这里。示例：

- Retrieve initial top-10 → LLM asks “too noisy, filter for <40 dB” → re-retrieve。
- Retrieve images → LLM sees one has a menu → retrieve the menu text → answer。

它增加复杂度，但能处理 single-shot retrieval 无法处理的查询。

### 评估

Cross-modal evaluation 仍不成熟。常见 proxy：

- 每种 modality 的 Recall@k。
- Fused top-k accuracy。
- Human-judged end-to-end satisfaction。
- Task-specific（bookings completed、purchases made）。

没有覆盖全部 modality 的标准 benchmark。大多数论文在 domain-specific tasks 上评估。

## 实际使用

`code/main.py`：

- 三个 mock retrievers（text、image、audio），运行在共享 restaurant corpus 上。
- Score fusion，用可配置 weights 组合 modality scores。
- Generator stub，发出带 citations 的 final answer。
- 一个简单 agentic loop，在 confidence 低时重写 query。

## 交付成果

本课产出 `outputs/skill-multimodal-rag-designer.md`。给定带 multimodal query flow 的 product spec，它会设计 retrievers、fusion、generator 与 evaluation。

## 练习

1. 提出一个 medical-triage multimodal RAG：query = injury photo + text symptoms。哪些 modality 从哪些 KB 检索？

2. Score fusion 是简单 weighted sum。它有什么 failure mode 是 MoE fusion 可以避免的？

3. 阅读 Abootorabi et al. 的 taxonomy（第 3 节）。三个典型子问题是什么，它们如何映射到你选择的产品？

4. 为 trip-planner multimodal RAG 设计 eval spec。哪些 metrics 覆盖 image recall、audio recall 与 composite correctness？

5. Agentic multi-hop RAG 每轮往返都有 latency tax。在什么查询难度下，准确率收益值得这份延迟？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Cross-modal retrieval | “Query one modality, retrieve another” | Text query 检索 images；image query 检索 text；需要 shared space 或 translator |
| Score fusion | “Combine scores” | 对每种 modality 的 retrieval scores 做 weighted sum；最简单的 fusion |
| MoE fusion | “Modality-routed experts” | Gating network 针对每个 query 选择该信任哪种 modality 的 scores |
| Grounded generation | “Cite your sources” | 答案中的每个 claim 都带有 source index |
| MuRAG | “First multimodal RAG” | 2022 年确立 multimodal RAG pattern 的论文 |
| Agentic multi-hop | “Reformulate and retry” | 当 first-pass confidence 低时，LLM 重新查询 retrievers |

## 延伸阅读

- [Abootorabi et al. — Ask in Any Modality (arXiv:2502.08826)](https://arxiv.org/abs/2502.08826)
- [Mei et al. — A Survey of Multimodal RAG (arXiv:2504.08748)](https://arxiv.org/abs/2504.08748)
- [Zhao et al. — Vision RAG Survey (arXiv:2503.18016)](https://arxiv.org/abs/2503.18016)
- [Chen et al. — MuRAG (arXiv:2210.02928)](https://arxiv.org/abs/2210.02928)
- [Liu et al. — REACT (arXiv:2301.10382)](https://arxiv.org/abs/2301.10382)
