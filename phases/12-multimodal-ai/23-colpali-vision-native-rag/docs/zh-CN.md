# ColPali 与 Vision-Native Document RAG

> 传统 RAG 会把 PDF 解析成文本，切分成 chunks，嵌入 chunks，再存入 vectors。每一步都会丢信号：OCR 丢掉 chart data，chunking 切断 table rows，text embeddings 忽略 figures。ColPali（Faysse et al., 2024 年 7 月）问了一个更简单的问题：为什么要抽取文本？直接通过 PaliGemma 嵌入 page image，用 ColBERT-style late interaction 做 retrieval，并保留文档携带的全部 layout、figures、fonts 与 formatting signal。公开 benchmark 显示：在视觉丰富文档上，端到端准确率比 text-RAG 高 20-40%。ColQwen2、ColSmol 与 VisRAG 扩展了这个模式。本课会读懂 vision-native RAG thesis，并构建一个小型 ColPali-like indexer。

**类型:** Build
**语言:** Python（stdlib，multi-vector indexer + MaxSim scorer）
**先修:** Phase 11（LLM Engineering — RAG basics），Phase 12 · 05（LLaVA）
**时间:** ~180 分钟

## 学习目标

- 解释 bi-encoder retrieval（每个 document 一个 vector）与 late-interaction retrieval（每个 document 多个 vectors）的差异。
- 描述 ColBERT 的 MaxSim 操作，以及 ColPali 如何把它从 text tokens 推广到 image patches。
- 构建一个小型 ColPali-like indexer：page → patch embeddings → 对 query-term embeddings 做 MaxSim → top-k pages。
- 在 invoices / financial reports 用例上比较 ColPali + Qwen2.5-VL generator 与 text-RAG + GPT-4。

## 要解决的问题

PDF 上的 Text-RAG 会丢掉文档的大部分内容。财报的 Q3 revenue growth 通常在图表里；医疗报告的 findings 在标注图像里；法律合同的 signature block 是 layout fact，而不是 text fact。

Text-RAG pipeline：

1. PDF → text，通过 OCR / pdftotext。
2. Text → 300-500 token chunks。
3. Chunk → bi-encoder embedding（一个 vector）。
4. User query → embedding → cosine similarity → top-k chunks。
5. Chunks + query → LLM。

五个有损步骤。Charts 没有被捕捉。Tables 被 chunks 切断。Multi-column layout 被压平。Figure annotations 消失。

ColPali 的修复：跳过 OCR，直接嵌入 page image。使用 ColBERT-style late interaction 做 retrieval，让模型在 query time attend 到细粒度 patches。

## 核心概念

### ColBERT（2020）

ColBERT（Khattab & Zaharia, arXiv:2004.12832）是一种文本检索方法。它不是每个 document 一个 vector，而是每个 token 一个 vector。查询时：

- Query tokens 有自己的 embeddings（N_q vectors）。
- Document tokens 有 embeddings（N_d vectors，通常缓存）。
- Score = 对每个 query token，取所有 document token 中 cosine similarity 最大值，再求和：Σ_i max_j cos(q_i, d_j)。

这就是 MaxSim 操作。每个 query token 会“选择”最匹配的 document token。最终分数是这些选择的总和。

优点：recall 强，能处理 term-level semantics。缺点：每个 document 有 N_d vectors，存储昂贵。

### ColPali

ColPali（Faysse et al., arXiv:2407.01449）把 ColBERT 模式应用到图像。

- 每页由 PaliGemma（ViT + language）编码成 patch embeddings：每页 N_p vectors。
- 每个用户 query（text）编码成 query-token embeddings：N_q vectors。
- Score = Σ_i max_j cos(q_i, p_j)，即在 query-text-tokens 与 page-image-patches 上做 MaxSim。
- 按总分检索 top-k pages。

Document-ingestion time：用 PaliGemma 嵌入每一页，存储所有 patch embeddings。Query time：嵌入 query tokens，对全部已存 page embeddings 计算 MaxSim，返回 top-k pages。

优点：在视觉丰富文档上端到端比 text-RAG 高 20-40%。每个 patch-vector 捕捉局部 layout 与 content。

缺点：每页 N_p patches × 4-byte floats × D-dim vectors = 存储增长很快。可用 PQ / OPQ quantization 缓解。

### ColQwen2 与 ColSmol

ColQwen2（illuin-tech, 2024-2025）把 PaliGemma 换成 Qwen2-VL。Base encoder 更好，retrieval 更好。

ColSmol 是面向 local / edge 使用的小规模变体。约 1B params 的 ColSmol retriever 可以在消费级 GPU 上运行。

### VisRAG

VisRAG（Yu et al., arXiv:2410.10594）是另一种变体：它不在 patches 上做 MaxSim，而是用 VLM 把每页 pool 成单个 vector，再做 bi-encoder retrieve。Indexing 更快，存储更小，recall 更弱。

质量与成本取舍：追求质量用 ColPali，追求规模用 VisRAG。

### M3DocRAG

M3DocRAG（Cho et al., arXiv:2411.04952）把 multimodal retrieval 扩展到 multi-page multi-document reasoning。它跨文档检索 pages，并为 VLM 组合 multi-page context。

### ViDoRe：benchmark

ColPali 的配套 benchmark。Visual Document Retrieval Evaluation。任务包括 financial reports、scientific papers、administrative documents、medical records、manuals。指标：nDCG@5。

ColPali-v1 在 ViDoRe 上约 80% nDCG@5；同一批文档上的 text-RAG 约 50-60%。

### 端到端 RAG pipeline

对于 vision-native RAG：

1. Ingest：PDF → page images → PaliGemma encoding → 存储全部 patch embeddings。
2. Query：user text → query-token embeddings → 对全部 indexed pages 做 MaxSim → top-k pages。
3. Generate：top-k page images + query → VLM（Qwen2.5-VL 或 Claude）→ answer。

全程没有 OCR。Figures、charts、fonts、layout 都流入答案。

### Storage math

一份 50 页财报，每页 729 个 patches，embedding 为 128-dim：

- ColPali：50 * 729 * 128 * 4 bytes = 约 18 MB raw，PQ 后约 4 MB。
- Text-RAG：50 chunks * 768-dim * 4 bytes = 约 150 kB。

ColPali 每份文档的存储约为 30x。规模化时，OPQ / PQ 可把它降到约 5-10x，通常可以接受。

### Text-RAG 仍然胜出的场景

- 没有 layout signal 的纯文本文档（wiki articles、chat logs）。Text-RAG 更简单，存储更便宜。
- 存储成本主导的数百万页 archives。
- 严格监管要求 retrieval 同时提供可抽取 OCR text。

对于 2026 年的其他几乎所有场景：financial reports、scientific papers、legal contracts、medical records、UX documentation，vision-native RAG 胜出。

## 实际使用

`code/main.py`：

- Toy patch encoder：把一个“page”（小型 feature vector grid）映射成 patch embeddings array。
- MaxSim scorer：计算 query token embedding set 与 page patch set 之间的 ColBERT-style score。
- 索引 5 个 toy pages，运行 3 个 queries，返回带 scores 的 top-k。

## 交付成果

本课产出 `outputs/skill-vision-rag-designer.md`。给定一个 document-RAG 项目，它会选择 ColPali / ColQwen2 / VisRAG / text-RAG，并估算存储。

## 练习

1. 一份 200 页年报，每页 729 个 patches，128-dim emb，4-byte floats。计算 raw storage 与 PQ-compressed（8x）storage。

2. MaxSim 是 Σ_i max_j cos(q_i, p_j)。这个 sum 捕捉了简单 mean similarity 捕捉不到的什么？

3. ColPali 把 pages 索引为 patch sets。如果我们改成 word level 索引（像 ColBERT），会发生什么变化？取舍是什么？

4. 为 1M-page corpus 设计端到端 pipeline，查询 latency budget 为 500ms。选择 ColQwen2 / VisRAG，并说明理由。

5. 阅读 M3DocRAG（arXiv:2411.04952）。描述 multi-page attention pattern，以及它与 single-page ColPali retrieval 的区别。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Late interaction | “ColBERT-style” | 使用 per-token 或 per-patch embeddings + MaxSim 的 retrieval，而不是单个 doc vector |
| MaxSim | “Max-over-patches” | 对每个 query token，选择相似度最高的 document token；对 query 求和 |
| Bi-encoder | “Single-vector” | 每个 document 一个 vector；更快但丢失粒度 |
| Multi-vector | “Many-vectors-per-doc” | 每个 document / page 存 N_p vectors；存储成本上升但 recall 提升 |
| Patch embedding | “Page feature” | 来自 VLM encoder 的每个 image patch 一个 vector，并按页缓存 |
| ViDoRe | “Vision doc bench” | ColPali 面向 visual document retrieval 的 benchmark suite |
| PQ quantization | “Product quantization” | 在缩小存储约 8x 的同时保持 vector similarity 的压缩方式 |

## 延伸阅读

- [Faysse et al. — ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [Khattab & Zaharia — ColBERT (arXiv:2004.12832)](https://arxiv.org/abs/2004.12832)
- [Yu et al. — VisRAG (arXiv:2410.10594)](https://arxiv.org/abs/2410.10594)
- [Cho et al. — M3DocRAG (arXiv:2411.04952)](https://arxiv.org/abs/2411.04952)
- [illuin-tech/colpali GitHub](https://github.com/illuin-tech/colpali)
