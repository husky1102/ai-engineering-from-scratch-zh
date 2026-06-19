# 综合项目 04：多模态文档问答（视觉优先 PDF、表格与图表）

> 2026 年 document-QA frontier 已从 OCR-then-text 转向 vision-first late interaction。ColPali、ColQwen2.5 和 ColQwen3-omni 把每个 PDF page 当作 image，用 multi-vector late interaction 嵌入，并让 query 直接 attend to patches。在 financial 10-Ks、scientific papers 和 handwritten notes 上，这种模式大幅超过 OCR-first。端到端构建这个 pipeline，在 10k pages 上运行，并发布与 OCR-then-text 的 side-by-side 对比。

**类型:** Capstone
**语言:** Python (pipeline), TypeScript (viewer UI)
**先修:** Phase 4 (computer vision), Phase 5 (NLP), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 17 (infrastructure)
**练习阶段:** P4 · P5 · P7 · P11 · P12 · P17
**时间:** 30 hours

## 要解决的问题

企业拥有大量 OCR pipelines 会搞坏的 PDFs：带 rotated tables 的 scanned 10-Ks、密集 equations 的 scientific papers、只有作为 image 才有意义的 charts、handwritten annotations。把这些当作 text-first 意味着丢掉一半信号。2026 年答案是对 raw page images 做 late-interaction multi-vector retrieval。ColPali（Illuin Tech）提出了它；ColQwen2.5-v0.2 和 ColQwen3-omni 推高了 accuracy。在 ViDoRe v3 上，vision-first retrieval 明显超过 OCR-then-text——且差距在 charts、tables 和 handwriting 上更大。

权衡是 storage 和 latency。ColQwen embedding 每页约 2048 个 patch vectors，而不是单个 1024-dim vector。Raw storage 会膨胀。DocPruner（2026）带来 50% pruning，且没有可测量的 accuracy loss。你将 index 10k pages，测量 ViDoRe v3 nDCG@5，把 answers 控制在 2s 内，并与 OCR-then-text baseline 直接比较。

## 核心概念

Late interaction 意味着每个 query token 都与每个 patch token 打分，并对每个 query token 取最大 score 再求和。你得到细粒度匹配，而不需要单个 pooled vector。Multi-vector index（Vespa、Qdrant multi-vector 或 AstraDB）存储 per-patch embeddings，并在 retrieval time 运行 MaxSim。

Answerer 是 vision-language model，它接收 query 和 top-k retrieved pages 的 images，并写出带 evidence regions（bounding boxes 或 page references）的 answer。Qwen3-VL-30B、Gemini 2.5 Pro 和 InternVL3 是 2026 frontier choices。对于 equations 和 scientific notation，OCR fallback（Nougat、dots.ocr）会作为可选 text channel 拼接进来。

Evaluation 是二维矩阵。一个轴是 content type（plain text paragraphs、dense tables、bar/line charts、handwritten notes、equations）。另一个轴是 retrieval approach（vision-first late interaction vs OCR-then-text vs hybrid）。每个 cell 都有 nDCG@5 和 answer accuracy。Report 就是 deliverable。

## 架构

```text
PDFs -> page renderer (PyMuPDF, 180 DPI)
           |
           v
  ColQwen2.5-v0.2 embed (multi-vector per page, ~2048 patches)
           |
           +------> DocPruner 50% compression
           |
           v
   multi-vector index (Vespa or Qdrant multi-vector)
           |
query ----+----> retrieve top-k pages (MaxSim)
           |
           v
  VLM answerer: Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    inputs: query + top-k page images + optional OCR text
           |
           v
  answer with cited page numbers + evidence regions
           |
           v
  Streamlit / Next.js viewer: highlighted boxes on source page
```

## 技术栈

- Page rendering: PyMuPDF (fitz) at 180 DPI, portrait-normalized
- Late-interaction model: ColQwen2.5-v0.2 or ColQwen3-omni (vidore team on Hugging Face)
- Index: Vespa with multi-vector field, or Qdrant multi-vector, or AstraDB with MaxSim
- Pruning: DocPruner 2026 policy (keep high-variance patches, 50% compression at < 0.5% accuracy loss)
- OCR fallback (equations / dense tables): dots.ocr or Nougat
- VLM answerer: Qwen3-VL-30B self-hosted or Gemini 2.5 Pro hosted; InternVL3 as fallback
- Evaluation: ViDoRe v3 benchmark, M3DocVQA for multi-page reasoning
- Viewer UI: Next.js 15 with canvas overlay for evidence regions

## 动手实现

1. **Ingest。** 遍历一个 10k PDF pages corpus，覆盖 10-Ks、scientific papers 和 scanned documents。把每页 render 为 1536x2048 PNG。持久化 `{doc_id, page_num, image_path}`。

2. **Embed。** 在每个 page image 上运行 ColQwen2.5-v0.2。输出形状为约 2048 个 dim 128 的 patch embeddings。应用 DocPruner，保留最高信号的一半。写入 Vespa multi-vector field 或 Qdrant multi-vector。

3. **Query。** 对每个 incoming query，用 query tower embed（token-level embeddings）。对 index 运行 MaxSim：对每个 query token，在 page patch embeddings 上取 max dot-product，再求和。返回 top-k pages。

4. **Synthesize。** 用 query 和 top-5 page images 调用 Qwen3-VL-30B。Prompt：“Answer using only the supplied pages. Cite each claim by (doc_id, page) and name the region (figure, table, paragraph).”

5. **Evidence regions。** Post-process answer 以抽取 cited regions。如果 VLM 输出 bounding boxes（Qwen3-VL 会），就在 viewer 中把它们 render 为 overlays。

6. **OCR fallback。** 对被识别为 equation-dense 的 pages（基于 image variance 的 heuristic），运行 Nougat 或 dots.ocr，并把 OCR text 作为额外 channel 与 image 一起传入。

7. **Eval。** 运行 ViDoRe v3（retrieval nDCG@5）和 M3DocVQA（multi-page QA accuracy）。还要在同一 corpus 上用同一 synthesizer 运行 OCR-then-text pipeline。产出一个 content-type × approach matrix。

8. **UI。** 先做 Streamlit prototype；再做 Next.js 15 production viewer，带 page-by-page evidence-region overlay。

## 实际使用

```text
$ doc-qa ask "what was the 2024 operating margin change for segment EMEA?"
[retrieve]   top-5 pages in 320ms (ColQwen2.5, MaxSim, Vespa)
[synth]      qwen3-vl-30b, 1.4s, cited (form-10k-2024, p. 88) + (..., p. 92)
answer:
  EMEA operating margin moved from 18.2% to 16.8%, a 140bp decline.
  cited: 10-K-2024.pdf p.88 (Table 4, Segment Operating Margin)
         10-K-2024.pdf p.92 (MD&A, Operating Performance)
[viewer]     open with highlighted bounding boxes overlaid on p.88 Table 4
```

## 交付成果

`outputs/skill-doc-qa.md` 描述 deliverable：一个针对特定 corpus 调优的 vision-first multimodal document QA system，并在 ViDoRe v3 上与 OCR-then-text baseline 对比评估。

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | ViDoRe v3 / M3DocVQA accuracy | Benchmark numbers vs OCR-text baseline and published leaderboard |
| 20 | Evidence-region grounding | Fraction of cited regions that actually contain the answer span |
| 20 | Storage and latency engineering | DocPruner compression ratio, index p95, answer p95 |
| 20 | Multi-page reasoning | Accuracy on a hand-labeled 100-question multi-page set |
| 15 | Source-inspection UX | Viewer clarity, overlay fidelity, side-by-side comparison tools |
| **100** | | |

## 练习

1. 在同一 corpus 上测量 ColQwen2.5-v0.2 vs ColQwen3-omni。哪些 pages 一个做对而另一个漏掉？向 index 添加 “content class” tag，用于按类型 route。

2. 激进 prune embeddings（75%、90%）。找到 compression cliff：ViDoRe nDCG@5 低于 OCR baseline 的点。

3. 构建 hybrid：并行运行 OCR-then-text 和 ColQwen，用 RRF 融合，再用 cross-encoder rerank。Hybrid 是否超过单独任一种？它在哪里帮助最大？

4. 把 Qwen3-VL-30B 换成更小的 VLM（Qwen2.5-VL-7B）。测量 accuracy-per-dollar curve。

5. 添加 handwritten-note support。Render handwriting corpus，用 ColQwen embed，测量 retrieval。与 handwriting OCR pipeline 对比。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Late interaction | “ColPali-style retrieval” | Query tokens 独立地对 page patches 打分；MaxSim 聚合 |
| Multi-vector | “Per-patch embedding” | 每个 document 有很多 vectors，而不是一个 pooled vector |
| MaxSim | “Late-interaction scoring” | 对每个 query token，在 document vectors 上取 max similarity；再求和 |
| DocPruner | “Patch compression” | 2026 pruning，在几乎不损失 accuracy 的情况下保留 50% patches |
| ViDoRe v3 | “Document-retrieval benchmark” | 2026 年测量 visual-document retrieval 的标准 |
| Evidence region | “Cited bounding box” | Source page 上定位 answer span 的 bbox |
| OCR fallback | “Equation channel” | 与 vision 并行使用的 text pipeline，服务于 equation- 或 table-heavy pages |

## 延伸阅读

- [ColPali (Illuin Tech) repository](https://github.com/illuin-tech/colpali) — reference late-interaction doc retrieval
- [ColPali paper (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449) — foundational method paper
- [ColQwen family on Hugging Face](https://huggingface.co/vidore) — production-ready checkpoints
- [M3DocRAG (Adobe)](https://arxiv.org/abs/2411.04952) — multi-page multimodal RAG baseline
- [Vespa multi-vector tutorial](https://docs.vespa.ai/en/colpali.html) — reference serving stack
- [Qdrant multi-vector support](https://qdrant.tech/documentation/concepts/vectors/#multivectors) — alternate index
- [AstraDB multi-vector](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) — alternate managed index
- [Nougat OCR](https://github.com/facebookresearch/nougat) — equation-capable OCR fallback
