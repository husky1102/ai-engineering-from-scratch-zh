# RAG 的分块策略

> 分块配置对检索质量的影响，和嵌入模型选择一样大（Vectara NAACL 2025）。分块做错了，再多 reranking 也救不回来。

**类型：** Build
**语言：** Python
**先修：** Phase 5 · 14 (Information Retrieval), Phase 5 · 22 (Embedding Models)
**时间：** ~60 minutes

## 要解决的问题

你把一份 50 页合同放进 RAG 系统。用户问：“终止条款是什么？”检索器返回了封面。为什么？因为模型是在 512-token chunk 上训练的，而终止条款在第 20 页，跨过分页符被切开，并且周围没有能把它和查询联系起来的局部关键词。

修复方法不是“买一个更好的嵌入模型”。修复方法是分块。多大？要不要 overlap？在哪里切？要不要带周边上下文？

2026 年 2 月的 benchmark 给出了一些意外结果：

- Vectara 的 2026 研究：recursive 512-token chunking 以 69% → 54% accuracy 击败 semantic chunking。
- Natural Questions 上的 SPLADE + Mistral-8B：overlap 没有提供任何可测量收益。
- Context cliff：上下文约 2,500 tokens 时，响应质量会急剧下降。

“显而易见”的答案（semantic chunking、20% overlap、1000 tokens）往往是错的。本课会为六种策略建立直觉，并告诉你什么时候该用哪一种。

## 核心概念

![Six chunking strategies visualized on one passage](../assets/chunking.svg)

**固定分块。** 每 N 个字符或 token 切一次。最简单的 baseline。会在句子中间切开。压缩好，一致性差。

**递归分块。** LangChain 的 `RecursiveCharacterTextSplitter`。先尝试按 `\n\n` 切，再按 `\n`，再按 `.`，最后按空格。退化路径干净。2026 年默认选择。

**语义分块。** 嵌入每个句子。计算相邻句子的余弦相似度。低于阈值处切分。保留主题一致性。更慢；有时会产生 40-token 的极小片段，伤害检索。

**句子分块。** 按句子边界切分。每句一个 chunk，或 N 句窗口。在成本只有一小部分的情况下，直到约 5k tokens 都能接近 semantic chunking。

**父文档分块。** 存小的 child chunks 用于检索，同时存更大的 parent chunk 用于上下文。按 child 检索；返回 parent。它能优雅退化：差的 child chunk 仍会返回合理的 parent。

**Late chunking（2024）。** 先在 token 级嵌入整篇文档，再把 token embeddings 池化为 chunk embeddings。保留跨 chunk 上下文。适合长上下文 embedder（BGE-M3、Jina v3）。计算成本更高。

**Contextual retrieval（Anthropic, 2024）。** 给每个 chunk 前置一个由 LLM 生成的、说明它在文档中位置的摘要（“This chunk is section 3.2 of the termination clauses...”）。在 Anthropic 自己的 benchmark 中，检索提升 35-50%。索引成本高。

### 击败所有默认值的规则

让 chunk size 匹配查询类型：

| 查询类型 | Chunk size |
|------------|-----------|
| Factoid（“CEO 的名字是什么？”） | 256-512 tokens |
| Analytical / multi-hop | 512-1024 tokens |
| Whole-section comprehension | 1024-2048 tokens |

这是 NVIDIA 的 2026 benchmark。chunk 应该足够大，能包含答案和局部上下文；又要足够小，让检索器的 top-K 聚焦答案，而不是上下文噪声。

## 动手实现

### Step 1: fixed and recursive chunking

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### Step 2: semantic chunking

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

在你的领域上调 `threshold`。太高 → 碎片化。太低 → 一个巨大 chunk。

### Step 3: parent-document

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

关键洞见：对 parents 去重。多个 children 可以映射到同一个 parent；全部返回会浪费上下文。

### Step 4: contextual retrieval (Anthropic pattern)

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

索引 contextualized chunks。在查询时，检索会从额外的周边信号中受益。

### Step 5: evaluate

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

永远 benchmark。你的语料上的“最佳”策略，可能和任何 blog post 都不一样。

## 常见陷阱

- **只用 factoid queries 评估分块。** Multi-hop queries 会暴露完全不同的赢家。使用按查询类型分层的 eval set。
- **Semantic chunking 没有最小尺寸。** 会产生伤害检索的 40-token 片段。始终强制 `min_tokens`。
- **把 overlap 当成惯例照搬。** 2026 年研究发现 overlap 通常没有收益，还会让索引成本翻倍。测量，不要假设。
- **没有 min/max 约束。** 5 tokens 或 5000 tokens 的 chunks 都会破坏检索。要 clamp。
- **跨文档分块。** 绝不要让一个 chunk 跨两个文档。始终先按文档分块，再合并。

## 实际使用

2026 年的栈：

| 场景 | 策略 |
|-----------|----------|
| 第一次构建，语料未知 | Recursive, 512 tokens, no overlap |
| Factoid QA | Recursive, 256-512 tokens |
| Analytical / multi-hop | Recursive, 512-1024 tokens + parent-document |
| 重交叉引用（合同、论文） | Late chunking or contextual retrieval |
| 对话语料 | Turn-level chunks + speaker metadata |
| 短文本（tweets、reviews） | One document = one chunk |

从 recursive 512 开始。在 50-query eval set 上测量 recall@5。再从那里调。

## 交付成果

保存为 `outputs/skill-chunker.md`：

```markdown
---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
```

## 练习

1. **Easy.** 用 fixed(512, 0)、recursive(512, 0) 和 recursive(512, 100) 对一份 20 页文档分块。比较 chunk counts 和 boundary quality。
2. **Medium.** 在 5 篇文档上构建 30-query eval set。测量 recursive、semantic 和 parent-document 的 recall@5。谁赢了？它和 blog posts 一致吗？
3. **Hard.** 实现 contextual retrieval。测量它相对 baseline recursive 的 MRR improvement。报告 index cost（LLM calls）与 accuracy gain。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Chunk | 文档片段 | 会被嵌入、索引和检索的子文档单元。 |
| Overlap | 安全边距 | 相邻 chunks 共享的 N 个 tokens；在 2026 benchmark 中通常没用。 |
| Semantic chunking | 智能分块 | 在相邻句子嵌入相似度下降处切分。 |
| Parent-document | 两级检索 | 检索小 children，返回更大的 parents。 |
| Late chunking | 嵌入后再分块 | 在 token 级嵌入整篇文档，再池化为 chunk vectors。 |
| Contextual retrieval | Anthropic 的技巧 | 索引前给每个 chunk 前置 LLM 生成的摘要。 |
| Context cliff | 2500-token 墙 | RAG 中约 2.5k context tokens 处观察到的质量下降（Jan 2026）。 |

## 延伸阅读

- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — 生产默认选择。
- [Vectara (2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070) — chunking 和 embedding choice 一样重要。
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — late chunking 论文。
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — 使用 LLM 生成的 context prefixes 带来 35-50% 检索提升。
- [NVIDIA 2026 chunk-size benchmark — Premai summary](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — 按查询类型选择 chunk size。
