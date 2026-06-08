# 混合记忆：Vector + Graph + KV（Mem0）

> Mem0（Chhikara et al., 2025）把 memory 视为三个并行 stores：vector 负责 semantic similarity，KV 负责快速 fact lookup，graph 负责 entity-relationship reasoning。检索时由 scoring layer 融合三者。这是 2026 年 external memory 的 production standard。

**类型：** 构建
**语言：** Python (stdlib)
**先修：** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Letta Blocks)
**时间：** ~75 分钟

## 学习目标

- 解释为什么单一 store（只有 vector、只有 graph 或只有 KV）不足以支撑 agent memory。
- 说出 Mem0 的三个 parallel stores，以及每个 store 优化什么。
- 描述 Mem0 的 fusion scoring：relevance、importance、recency，以及为什么它是 weighted sum，而不是 hierarchy。
- 用 stdlib 实现一个 toy three-store memory：`add()` 写入全部三个 stores，`search()` 融合结果。

## 要解决的问题

一个 store 会在三类 query 中至少错一类：

- **Semantic similarity** — “what did we discuss about agent drift last week?” Vector 胜出；KV 和 graph 会漏掉。
- **Fact lookup** — “what is the user's phone number?” KV 胜出；vector 浪费，graph 过度。
- **Relationship reasoning** — “which customers share the same billing entity?” Graph 胜出；vector 和 KV 无法回答。

Production agents 在一次 session 中会发出全部三类查询。Single-store memory 对其中两类总是不合适。Mem0 的贡献是把三者接到同一个 `add`/`search` surface 后面，并用 scoring function 融合它们。

## 核心概念

### 三个 stores 并行

Mem0（arXiv:2504.19413, April 2025）在 `add(text, user_id, metadata)` 时：

1. 从 text 中抽取 candidate facts（LLM-driven step）。
2. 把每个 fact 写入 vector store（embedding），用于 semantic search。
3. 把每个 fact 写入 KV store，以 (user_id, fact_type, entity) 为 key，用于 O(1) lookup。
4. 把每个 fact 写入 graph store（Mem0g），作为 typed edges，用于 relationship queries。

在 `search(query, user_id)` 时：

1. Vector store 按 embedding cosine 返回 top-k。
2. KV store 按 query-derived (user_id, type, entity) 返回 direct hits。
3. Graph store 返回从 query entities 可达的 subgraph。
4. Scoring layer 融合三者。

### Fusion scoring

```text
score = w_relevance * relevance(q, record)
      + w_importance * importance(record)
      + w_recency * recency(record)
```

- **Relevance** — vector cosine、KV exact match、graph path weight。
- **Importance** — 写入时打 tag 或学习得到（某些 facts 更重要：names、IDs、policies）。
- **Recency** — 距离上次 write 或 read 的时间做 exponential decay。

Weights 按产品调优。Chat agents 提高 `w_recency`；compliance agents 提高 `w_importance`；retrieval agents 提高 `w_relevance`。

### Mem0g 和 temporal reasoning

Mem0g 增加 conflict detector。当新 fact 与已有 edge 矛盾时，已有 edge 会被标记为 invalid，但不会删除。Temporal queries（“what was the user's city in March?”）会遍历 valid-at-time subgraph。

这是 Letta invalidation pattern 泛化出的 compliance-grade behavior。

### Benchmark numbers

Mem0 paper 报告（2025）：

- **LoCoMo**（long-form conversation memory）：91.6
- **LongMemEval**（long-horizon episodic memory）：93.4
- **BEAM 1M**（1M-token memory benchmark）：64.1

对比 baselines（full-context 128k LLM、flat vector store、flat KV）都落后 10+ points。Benchmarks 本身不能决定选择，operational shape 才能；但这些数字说明 fusion design 不是 rounding error。

### Scope taxonomy

Mem0 按 scope 拆分 memory：

- **User memory** — 跨 sessions 持久化，以 `user_id` 为 key。
- **Session memory** — 在一个 thread 内持久化。
- **Agent memory** — per-agent instance state。

每次 write 都选择一个 scope。Retrieval 可以跨 scopes 查询，并对每个 scope 设置权重。不经思考地混合 scopes，就是“assistant 把 Bob 的项目告诉 Alice”这类 incident 的来源。

### 这个模式容易出错的地方

- **Embedding drift.** Vector results 在最初一百个 queries 看起来正确，但随着 corpus 增长会退化。给 top-N-used records 增加 periodic re-embedding。
- **KV schema creep.** `(user_id, type, entity)` 看起来简单，直到每个团队都添加自己的 `type`。每季度审计 type set。
- **Graph explosion.** 一个 noisy extractor 每条 message 添加 50 条 edges。对每个 `add` call 限制 graph writes；丢弃 low-confidence edges。

## 动手实现

`code/main.py` 用 stdlib 实现 three-store pattern：

- `VectorStore` — naive token-overlap similarity，作为 embedding stand-in。
- `KVStore` — 以 `(user_id, fact_type, entity)` 为 key 的 dict。
- `GraphStore` — typed edges（subject、relation、object、valid）。
- `Mem0` — top-level facade，带 `add()`、`search()`、fusion scoring 和 scope-aware retrieval。
- 一个 multi-user、multi-session conversation 的 worked trace。

运行：

```text
python3 code/main.py
```

output 展示三条不同 recall paths 以及 fused top-k。调整 `main()` 顶部的 scoring weights，观察 ranking 如何变化。

## 实际使用

- **Mem0（Apache 2.0）** — production-ready。可用 Postgres + Qdrant + Neo4j self-host，也可用 managed cloud。
- **Letta** — three-tier core/recall/archival；自带 vector 和 graph backends。
- **Zep** — 带 temporal KG 和 fact extraction 的 commercial alternative。
- **Custom builds** — 当你需要精确控制 extractor（compliance）或 fusion weights（recency 主导的 voice agents）时使用。

## 交付成果

`outputs/skill-hybrid-memory.md` 会生成 three-store memory scaffold，包含 fusion scorer、scope taxonomy 和 temporal invalidation。

## 练习

1. 用真实 embedding model（sentence-transformers、Ollama、OpenAI embeddings）替换 toy vector similarity。在 synthetic long conversation 上测量 recall@10。1000 次 writes 后 ranking 会 drift 吗？
2. 添加 temporal query：`search(query, as_of=timestamp)`。只返回在该时间或之前 valid 的 records。哪个 store 需要最多工作？
3. 实现 conflict detector：如果 incoming fact 和 graph edge 矛盾，就 invalidate old edge，并记录二者。在“user lives in Berlin” -> “user lives in Lisbon”上测试。
4. 扩展 fusion scorer，加入 `user_feedback` 维度（对 retrieved records 点赞）。如何防止 gaming（agent 只返回它已经喜欢的 records）？
5. 阅读 Mem0 docs（`docs.mem0.ai`）。把 toy 移植成 `mem0` client calls。在同样 20 个 test queries 上比较 retrieval quality。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Hybrid memory | “Vector plus graph plus KV” | 三个 stores 并行写入，检索时融合 |
| Fact extraction | “Memory ingestion” | 把 text 拆成 (entity, relation, fact) tuples 的 LLM step |
| Fusion scoring | “Relevance ranking” | relevance、importance、recency 的 weighted sum |
| Scope | “Memory namespace” | user / session / agent — 决定谁能看到什么 |
| Mem0g | “Memory graph” | 带 temporal validity 的 typed edges，用于 relationship queries |
| Temporal invalidation | “Soft delete” | 将被矛盾事实覆盖的 edges 标记为 invalid；永不删除 |
| Embedding drift | “Retrieval rot” | corpus 增长导致 vector quality 退化；定期 re-embed |

## 延伸阅读

- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) — the original paper
- [Mem0 docs](https://docs.mem0.ai/platform/overview) — production API, SDKs, managed cloud
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — the virtual-context predecessor
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) — the three-tier sibling design
