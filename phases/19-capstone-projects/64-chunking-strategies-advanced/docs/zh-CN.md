# Chunking Strategies, Compared

> Chunking 决定了你的 retriever 永远能召回什么。边界切错了，下游没有 embedding model、reranker 或 LLM 能修复损伤。

**类型:** Build
**语言:** Python
**先修:** Phase 11 lessons 04 (embeddings), 06 (RAG), 07 (advanced RAG); Phase 19 Track B foundations (lessons 20-29)
**时间:** ~90 minutes

## 学习目标
- 从零实现五种 chunking strategies：fixed-window、sentence、recursive-split、semantic clustering 和 structural markdown headers。
- 在带 gold-labeled answer spans 的 fixture corpus 上测量 recall@k，并解释为什么一种策略赢在 prose，另一种策略赢在 technical documents。
- 读取 chunk-length distribution，并识别每种策略注入的 failure modes：orphan sentences、mid-symbol cuts、header-only chunks、semantic drift。
- 不运行 benchmark，仅通过检查三项属性为新 corpus 选择默认值：document type、average paragraph length，以及格式是否携带 explicit structure。

## 要解决的问题

每条 RAG pipeline 都从把源文档切成 pieces 开始：小到 embedding model 能容纳，大到每片都承载一个自包含 idea。在哪里切，不是 hyperparameter。它是 retriever 永远能返回什么的上限。

一个问 “what does the budget abort threshold look like” 的 query，只有在包含 abort threshold 的 chunk 可被触达时才可能成功。如果 fixed-window splitter 把 threshold value 从周围 context 中切走，embedding 会移动到另一个 cluster，BM25 score 下降，rerankers 看到 noise，LLM 生成的 answer 就会错。2024 年论文 “LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs” 测得，纯粹由 chunking choice 导致的 retrieval recall 绝对差异达到 35 percent。2025 年关于 contextual chunk headers 的后续工作缩小了差距，但没有消除它。

本课把五种策略并排构建，在带 gold-labeled answer spans 的 fixture corpus 上运行，并让你自己读取 recall numbers。

## 核心概念

```mermaid
flowchart LR
  Doc[Source Document] --> S1[Fixed Window]
  Doc --> S2[Sentence]
  Doc --> S3[Recursive Split]
  Doc --> S4[Semantic Cluster]
  Doc --> S5[Structural Markdown]
  S1 --> Chunks1[Chunks]
  S2 --> Chunks2[Chunks]
  S3 --> Chunks3[Chunks]
  S4 --> Chunks4[Chunks]
  S5 --> Chunks5[Chunks]
  Chunks1 --> Index[Embedding Index]
  Chunks2 --> Index
  Chunks3 --> Index
  Chunks4 --> Index
  Chunks5 --> Index
  Index --> Eval[Recall@k vs Gold Spans]
```

### Fixed-window

暴力 baseline。每 N 个 characters 切一次。可以选择 overlap，让在位置 N 被切断的句子完整出现在从 N - overlap 开始的 chunk 中。快速、确定性、边界很糟。把它当 control，而不是 default。

### Sentence

用 regex 或简单 state machine 在 sentence boundaries 上 split。把一个或多个 sentences 打包进 chunk，直到接近目标 character budget。不会从 word 中间切开。仍会从 paragraph 和 section 中间切。它是很多早期 RAG pipelines 的默认值，对于没有其他结构的 prose 是合理选择。

### Recursive split

2023 年前后流行库推广的 hierarchy strategy。先尝试最强 separator（double newline、paragraph），再 fallback 到下一级（single newline），再到 sentences，再到 characters。当 chunk 符合 budget 时递归终止。它在结构不一致的文档上很强，因为能按 region 自适应。

### Semantic clustering

Embed 每个 sentence。把共享 topic centroid 的连续 sentences 聚成一组。只要 running similarity to the centroid 低于 threshold，就切开。边界反映含义，而不是 characters。构建更慢，也依赖 embedding model，但能抵抗 paragraph 内 topic 切换的文档。

### Structural markdown headers

对携带 explicit structure 的文档（markdown、reStructuredText、RFC-style numbered sections），在 heading boundaries 处切。每个 chunk 是 heading 加上它下面直到下一个同级或更高级 heading 的全部内容。每个 topic 的 chunks 最小，但只有 corpus 格式良好时才可用。

### recall@k 如何衡量 boundary choice

gold-labeled query 携带 source document 中 answer span 的精确 character offsets。chunking 后，你问：retriever 返回的 top-k chunks 中，有没有任何一个与 gold span 重叠？有，则该 query 的 recall@k 是 1。没有，则是 0。对 query set 平均。对每个 strategy 运行同一个 evaluation，分差会展示哪种 boundary policy 能经受你的 corpus。

## 动手实现

`code/main.py` 实现：

- `fixed_window(text, size, overlap)` - baseline。
- `sentence_chunks(text, target)` - 简单 sentence packer。
- `recursive_split(text, separators, target)` - hierarchical recursion。
- `semantic_chunks(text, similarity_threshold)` - 基于 deterministic mock embedding 的 centroid-based clustering。
- `structural_markdown(text)` - header-aware splitter。
- `mock_embed(text, dim)` - hash-based embedding，让 loop 能 offline 运行。
- `DenseIndex` - Phase 19 Track B 的 hybrid retrieval lesson 中使用的同样形状。
- `eval_recall(strategy, corpus, queries, k)` - comparison loop。
- 一个 `main()`：在 fixture corpus 上运行所有 strategy，并打印 recall@k table。

运行：

```bash
python3 code/main.py
```

输出是一个小表，每个 strategy 一行、每个 k 一列。Sentence 在 structured fixture 上失利。Structural-markdown 赢在 markdown fixture 上。Recursive 在 mixed fixture 上表现稳健，因为 recursion 会自适应。Semantic clustering 赢在没有有用 structural cues 的 prose fixture 上。

## 表格不会隐藏的失败模式

**Orphan sentences。** Sentence packing 产出的 chunks 可能错过 topic sentence。embedding 随后指向错误 cluster。

**Mid-symbol cuts。** Fixed-window 在 code 或 YAML 内部会把 identifier 切成两半。两半都会 embed 成 noise。

**Header-only chunks。** Structural markdown 会发出只包含 `## Title` 的 chunk。过滤掉它们，或附加下一个 chunk 的第一段。

**Semantic drift。** 当 corpus 主题非常统一时，semantic clustering 会切得太少。一个 5000-character chunk 会把许多具体答案打包成一个 diffuse embedding。把 semantic 与 hard character cap 结合。

**Stale embeddings。** Semantic clustering 使用 embedding model。如果你改变 model，就也改变 chunks。将 chunk model 与 retrieval model 分别 pin，或一起 rebuild index。

## 不运行 benchmark 时如何选择默认值

三个属性决定新 corpus 的默认 chunker。

| Property | Value | Default |
|----------|-------|---------|
| Document type | Prose with no structure | Recursive split, target 800 |
| Document type | Markdown / RFC / API docs | Structural markdown |
| Document type | Code | AST-aware (out of scope; see Phase 19 lesson 02) |
| Paragraph length | Long, single topic | Sentence, target 500 |
| Paragraph length | Short, mixed topics | Semantic, threshold 0.6 |

拿不准时，选择 recursive split。它是最强的单策略 baseline。

## 实际使用

生产模式：

- 在发版新 pipeline 前运行 eval；不要信任 library 默认策略。
- 每次改变 embedding model 或 corpus mix 时重新运行 eval；赢家依赖 corpus。
- 在每个 chunk 的 metadata 中持久化 strategy name，便于之后归因 regressions。

## 交付成果

lesson 69 中 Track F end-to-end RAG system 会使用这里选择的 chunker 作为第一阶段。lesson 68 中的 eval harness 会读取与本课 `eval_recall` 返回形状相同的 recall@k。选择在你的 corpus 上获胜的 strategy，并把它向前传递。

## 练习

1. 添加第六种策略：使用 `tiktoken` 而不是 character counts 的 token-window。在同一个 fixture 上与 fixed-window 比较。
2. 向 prose fixture 注入 30 percent 的 code blocks。重新运行表格。解释为什么除了 structural markdown 外，每种 strategy 都会丢 recall。
3. 用你项目真实 provider 的 embedding 替换 deterministic embedding。测量 semantic-clustering recall delta。报告策略之间的分差是变宽还是变窄。
4. 给每个 chunk 添加一个 `summary` field：一句话 centroid description。把 summary 附加到 chunk body 后重新运行 eval。测量 recall lift。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Recall@k | “我们拿到正确 chunk 了吗？” | top-k chunks 中任意一个与 gold answer span 重叠的 queries 比例 |
| Chunk overlap | “Sliding window” | 把上一个 chunk 的最后 N characters 重新包含到下一个 chunk 中 |
| Structural splitter | “Header-aware chunks” | 在 H1/H2/H3 boundaries 切分；heading text 是 chunk 的一部分 |
| Semantic chunker | “Topic-aware chunks” | Embed sentences，按 centroid similarity 聚类，并在 drift 时切开 |
| Centroid drift | “Topic shift” | running mean 与 next sentence 的 cosine similarity 下降越过 threshold |

## 延伸阅读

- [LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs (arXiv 2406.15319)](https://arxiv.org/abs/2406.15319)
- [Anthropic, Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [LlamaIndex, Chunking strategies for production RAG](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/)
- Phase 11 lesson 06 - RAG fundamentals
- Phase 11 lesson 07 - advanced RAG
- Phase 19 lesson 65 - 对这里产出的 chunks 进行 ranking 的 hybrid retrieval
- Phase 19 lesson 68 - 在生产中为 strategy choice 打分的 eval harness
