# Capstone 02——Codebase 上的 RAG（Cross-Repo Semantic Search）

> 到 2026 年，每个严肃工程组织都会运行一个理解语义而不仅是字符串的内部 code search。Sourcegraph Amp、Cursor 的 codebase answers、Augment 的 enterprise graph、Aider 的 repomap、Pinterest 的 internal MCP——形态相同。Ingest 多个 repos，用 tree-sitter parse，嵌入 function- 和 class-level chunks，hybrid-search，re-rank，并用 citations 回答。本 capstone 要求你构建一个系统，能处理 10 个 repos 中的 2M lines of code，并且在每次 git push 的 incremental re-indexing 中存活。

**类型:** Capstone
**语言:** Python (ingestion), TypeScript (API + UI)
**先修:** Phase 5 (NLP foundations), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 13 (tools), Phase 17 (infrastructure)
**练习阶段:** P5 · P7 · P11 · P13 · P17
**时间:** 30 hours

## 要解决的问题

到 2026 年，每个 frontier coding agent 都会带 codebase retrieval layer，因为 context windows alone 无法解决 cross-repo questions。Claude 的 1M-token context 有帮助；但不能消除 ranked retrieval 的需求。对 raw chunks 做 naive cosine search，会在 generated code、monorepo duplication，以及 rarely-imported symbols 的长尾上污染结果。Production answer 是对 AST-aware chunks 做 hybrid（dense + BM25）search，配 re-ranker，并由 symbol references graph 支撑。

你要通过 indexing 一个真实 fleet 来学习这些，而不是一个 tutorial repo，并测量 MRR@10、citation faithfulness 和 incremental freshness。Failure modes 是基础设施层面的：一个 100k-file monorepo，一次 retouches 半数文件的 push，一个需要跨四个 repos 才能正确回答的 query。

## 核心概念

AST-aware ingestion pipeline 用 tree-sitter parse 每个文件，抽取 function 和 class nodes，并在 node boundaries 而不是固定 token windows 处 chunk。每个 chunk 得到三种 representation：dense embedding（Voyage-code-3 或 nomic-embed-code）、sparse BM25 terms，以及短 natural-language summary。Summary 增加第三种可检索 modality——用户问 “how is X authorized”，summary 提到 “authz”，即使代码里只有 `check_permission`。

Retrieval 是 hybrid。一个 query 同时触发 dense 和 BM25 searches，合并 top-k，并把 union 交给 cross-encoder re-ranker（Cohere rerank-3 或 bge-reranker-v2-gemma-2b）。Re-ranked list 进入 long-context synthesizer（带 prompt caching 的 Claude Sonnet 4.7，或 self-hosted Llama 3.3 70B），并要求每个 claim 都用 file 和 line range 引用。没有 citations 的 answers 会被 post-filter 拒绝。

Incremental freshness 是基础设施问题。Git push 触发 diff：哪些 files changed，哪些 symbols changed。只有 affected chunks 会重新 embed。Affected cross-file symbol edges（imports、method calls）会重新计算。Index 可以保持一致，而不用每次 commit 都重新处理 2M lines。

## 架构

```text
git push --> webhook --> ingest worker (LlamaIndex Workflow)
                           |
                           v
             tree-sitter parse + AST chunk
                           |
            +--------------+----------------+
            v              v                v
          dense        BM25 index       summary (LLM)
        (Voyage / bge)  (Tantivy)        (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      symbol graph (Neo4j / kuzu)
                            |
  query --> LangGraph agent (retrieve -> rerank -> synth)
                            |
                            v
                 Claude Sonnet 4.7 1M context
                            |
                            v
                 answer + file:line citations
```

## 技术栈

- Parsing: tree-sitter with 17 language grammars (Python, TS, Rust, Go, Java, C++, etc.)
- Dense embeddings: Voyage-code-3 (hosted) or nomic-embed-code-v1.5 (self-host), bge-code-v1 fallback
- Sparse index: Tantivy (Rust) with BM25F, field-weighted on symbol name vs body
- Vector DB: Qdrant 1.12 with hybrid search, or pgvector + pgvectorscale for teams under 50M vectors
- Chunk summary model: Claude Haiku 4.5 or Gemini 2.5 Flash, prompt-cached
- Re-ranker: Cohere rerank-3 or bge-reranker-v2-gemma-2b self-hosted
- Orchestration: LlamaIndex Workflows for ingestion, LangGraph for query agent
- Synthesizer: Claude Sonnet 4.7 (1M context) with prompt caching
- Symbol graph: Neo4j (managed) or kuzu (embedded) for import and call edges
- Observability: Langfuse spans per retrieval + synthesis step

## 动手实现

1. **Ingestion walker。** 在每个 push hook 上迭代 git history。收集 changed files。对每个 file，用 tree-sitter parse，抽取 function 和 class nodes 及其 full source span。输出 chunk records `{repo, path, start_line, end_line, symbol, body}`。

2. **Chunk summarizer。** 把 chunks 批量送入 Haiku 4.5 calls，并对 system preamble 使用 prompt caching。Prompt：“Summarize this function in one sentence, naming its public contract and side effects.” 把 summary 与 chunk 一起存储。

3. **Embedding pool。** 两个 parallel queues：dense（Voyage-code-3 batch 128）和 summary（同一模型，但输入 summary string）。向 Qdrant 写入 vectors，payload 为 `{repo, path, start_line, end_line, symbol, kind}`。

4. **BM25 index。** Field-weighted Tantivy index：symbol name weight 4，symbol body weight 1，summary weight 2。它既支持 “find the function named X”，也支持 “find the function that does X” queries。

5. **Symbol graph。** 对每个 chunk，记录 edges：imports（this file uses symbol Y from repo Z）、calls（this function calls method M on class C）、inheritance。存储在 kuzu 中。Query time 用它把 retrieval 扩展到 repo boundaries 之外。

6. **Query agent。** LangGraph 有三个 nodes。`retrieve` 并行触发 dense + BM25，并按 (repo, path, symbol) deduplicate。`rerank` 在 top-50 上运行 cross-encoder，并保留 top-10。`synth` 调用 Claude Sonnet 4.7，把 reranked chunks 放入 context，缓存 system prompt，并要求 file:line citations。

7. **Citation enforcement。** Parse model output；任何没有 `(repo/path:start-end)` anchor 的 claim 都会被 flagged，用于 re-ask 或 drop。只把 cited-only answer 返回给用户。

8. **Incremental re-index。** 每个 webhook 上计算 symbol-level diff。只重新 embed text changed 的 chunks。对 imports changed 的 chunks 重新计算 symbol edges。测量：2M-LOC fleet 中一次 50-file push 在 60 秒内完成 re-index。

9. **Eval。** 标注 100 个 cross-repo questions，并给出 gold file:line answers。测量 MRR@10、nDCG@10、citation faithfulness（带可验证 anchors 的 claims 比例）以及 p50/p99 latency。

## 实际使用

```text
$ code-rag ask "how is S3 multipart abort wired into our retry budget?"
[retrieve]  12 chunks dense + 7 chunks bm25, 16 unique after dedup
[rerank]    top-5 kept (cohere rerank-3)
[synth]     claude-sonnet-4.7, cache hit rate 68%, 2.1s
answer:
  Multipart aborts are triggered by `AbortMultipartOnFail` in
  services/uploader/retry.go:122-148, which decrements the per-bucket
  retry budget defined in config/budgets.yaml:34-51 ...
  citations: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

## 交付成果

Deliverable skill `outputs/skill-codebase-rag.md`。给定一个 repos corpus，它会启动 ingestion pipeline、hybrid index 和 query agent，并为任何 cross-repo question 返回 cited answer。Rubric：

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Retrieval quality | MRR@10 and nDCG@10 on a 100-question held-out set |
| 20 | Citation faithfulness | Fraction of answer claims with verifiable file:line anchors |
| 20 | Latency and scale | p95 query latency at 10k QPS on the indexed corpus size |
| 20 | Incremental indexing correctness | Time from git push to searchable on a 50-file commit |
| 15 | UX and answer formatting | Citation clickability, snippet previews, follow-up affordance |
| **100** | | |

## 练习

1. 把 Voyage-code-3 替换为 self-hosted nomic-embed-code。测量 MRR@10 delta。报告在启用 re-ranking 后差距是否缩小。

2. 向 corpus 注入 20% generated code（LLM-produced boilerplate）并重新评估。观察 retrieval poisoning。向 payload 添加 `"generated"` flag，并 down-weight 这些 hits。

3. 在你的 corpus size 下 benchmark Qdrant hybrid search vs pgvector + pgvectorscale。报告 batch size 1 时的 p99。

4. 添加 sampling-based drift check：每周重新运行 100-question eval。若 MRR@10 drop > 5%，则 alert。

5. 扩展到 cross-language symbol resolution：一个 Python function 通过 gRPC 调用 Go service。用 symbol graph 连接它们。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| AST-aware chunking | “Function-level splits” | 在 tree-sitter node boundaries 处切分代码，而不是固定 token windows |
| Hybrid search | “Dense + sparse” | 并行运行 BM25 和 vector search，合并 top-k，再 rerank |
| Cross-encoder rerank | “Second-stage rank” | 把每个 (query, candidate) pair 一起打分的模型，比 cosine 更准确 |
| Prompt caching | “Cached system prompt” | 2026 Claude / OpenAI feature，可对重复 prefix tokens 最高打 90% 折扣 |
| Symbol graph | “Code graph” | 跨 files 和 repos 的 imports、calls、inheritance edges |
| Citation faithfulness | “Grounded answer rate” | 用户可通过点击 anchor 并阅读 referenced span 来验证的 claims 比例 |
| Incremental re-index | “Push-to-search time” | 从 git push 到 changed symbols 可被 query 的 wall-clock |

## 延伸阅读

- [Sourcegraph Amp](https://ampcode.com) — production cross-repo code intelligence
- [Sourcegraph Cody RAG architecture](https://sourcegraph.com/blog/how-cody-understands-your-codebase) — 本 capstone 的 reference deep-dive
- [Aider repo-map](https://aider.chat/docs/repomap.html) — tree-sitter ranked repo view
- [Augment Code enterprise graph](https://www.augmentcode.com) — commercial symbol-graph RAG
- [Qdrant hybrid search docs](https://qdrant.tech/documentation/concepts/hybrid-queries/) — reference implementation
- [Voyage AI code embeddings](https://docs.voyageai.com/docs/embeddings) — Voyage-code-3 details
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank) — cross-encoder reference
- [Pinterest MCP internal search](https://medium.com/pinterest-engineering) — internal-platform reference
