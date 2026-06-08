# Information Retrieval 与 Search

> BM25 精确但脆弱。Dense 撒网很广但会漏掉 keywords。Hybrid 是 2026 年默认选择。其他都是 tuning。

**类型:** Build
**语言:** Python
**先修:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 04 (GloVe, FastText, Subword)
**时间:** ~75 minutes

## 要解决的问题

用户输入 “what happens if someone lies to get money”，期待找到真正覆盖这个问题的 statute：“Section 420 IPC.” keyword search 会完全错过（没有共享 vocabulary）。如果 embeddings 没有在 legal text 上训练，semantic search 也会错过。真实 search 必须同时处理二者。

IR 是每个 RAG system、每个 search bar、每个 docs site fuzzy lookup 底下的 pipeline。2026 年在 production 中有效的 architecture 不是单一方法，而是一串互补方法，每一层都捕捉前一层的 failures。

本课构建每个 piece，并说明每个 piece 捕捉哪类 failures。

## 核心概念

![Hybrid retrieval: BM25 + dense + RRF + cross-encoder rerank](../assets/retrieval.svg)

四层。按需选择。

1. **Sparse retrieval (BM25).** 快，在 exact matches 上精确，在 semantics 上很差。运行在 inverted index 上。百万 documents 上每 query 低于 10ms。能正确处理 statute references、product codes、error messages、named entities。
2. **Dense retrieval.** 将 query 和 documents encode 成 vectors。Nearest neighbor search。捕捉 paraphrases 和 semantic similarity。会漏掉差一个 character 的 exact keyword matches。用 FAISS 或 vector DB 每 query 50-200ms。
3. **Fusion.** 合并 sparse 和 dense 的 ranked lists。Reciprocal Rank Fusion（RRF）是简单默认值，因为它忽略 raw scores（不同尺度），只使用 rank positions。当你知道某个 signal 在 domain 中占主导时，weighted fusion 也是选项。
4. **Cross-encoder rerank.** 取 fusion 的 top-30。运行 cross-encoder（query + document 一起输入，给每对打分）。保留 top-5。Cross-encoders 每对更慢，但比 bi-encoders 准确得多。只在 top-30 上运行，摊平成本。

三路 retrieval（BM25 + dense + learned-sparse，如 SPLADE）在 2026 benchmarks 中超过两路，但需要 learned-sparse indexes 基础设施。对多数 teams，两路加 cross-encoder rerank 是 sweet spot。

## 动手实现

### Step 1: 从零实现 BM25

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

两个参数值得知道。`k1=1.5` 控制 term-frequency saturation；越高，term repetition 权重越大。`b=0.75` 控制 length normalization；0 忽略 document length，1 完全 normalize。默认值来自 Robertson 原始论文的建议，很少需要 tuning。

### Step 2: 用 bi-encoder 做 dense retrieval

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

L2-normalize embeddings，使 dot product 等于 cosine。`all-MiniLM-L6-v2` 是 384-dim、快速，并且对多数 English retrieval 足够强。multilingual work 使用 `paraphrase-multilingual-MiniLM-L12-v2`。追求最高 accuracy，则用 `bge-large-en-v1.5` 或 `e5-large-v2`。

### Step 3: Reciprocal Rank Fusion

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

`k=60` constant 来自原始 RRF paper。更高 `k` 会抹平 rank differences 的贡献；更低 `k` 让 top ranks 主导。60 是公开默认值，很少需要 tuning。

### Step 4: hybrid search + rerank

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

三个 stages 组合。BM25 找 lexical matches。Dense 找 semantic matches。RRF 不需要 score calibration，就能合并两个 rankings。Cross-encoder 用 query-document pairs 一起重打 top-30 分数，捕捉 bi-encoder 漏掉的 fine-grained relevance。保留 top-5。

### Step 5: evaluation

| Metric | Meaning |
|--------|---------|
| Recall@k | 在 correct document 存在的 queries 中，它出现在 top-k 的频率 |
| MRR (Mean Reciprocal Rank) | 第一个 relevant document 的 1/rank 的平均值 |
| nDCG@k | 考虑 relevance gradations，而不只是 binary relevant/not |

对 RAG 来说，retriever 的 **Recall@k** 是最重要数字。如果正确 passage 不在 retrieved set 中，reader 无法回答。

Debugging tip：对失败 queries，diff sparse 和 dense rankings。如果其中一个找到正确 document 而另一个没有，你遇到的是 vocabulary mismatch（fix：加上 missing half）或 semantic ambiguity（fix：更好的 embeddings 或 reranker）。

## 实际使用

2026 年 stack：

| Scale | Stack |
|-------|-------|
| 1k-100k docs | In-memory BM25 + `all-MiniLM-L6-v2` embeddings + RRF。不需要单独 DB。 |
| 100k-10M docs | FAISS 或 pgvector 用于 dense + Elasticsearch / OpenSearch 用于 BM25。并行运行。 |
| 10M+ docs | Qdrant / Weaviate / Vespa / Milvus with hybrid support。Cross-encoder rerank top-30。 |
| Best-quality frontier | Three-way (BM25 + dense + SPLADE) + ColBERT late-interaction reranking |

无论选择什么，都要为 evaluation 留预算。benchmark end-to-end RAG accuracy 前，先 benchmark retrieval recall。reader 无法修复 retriever 漏掉的东西。

### 2026 production RAG 的 hard-won lessons

- **80% 的 RAG failures 来自 ingestion 和 chunking，而不是 model。** Teams 花几周换 LLM、调 prompts，而 retrieval 每三个 query 就静默返回错误 context。先修 chunking。
- **Chunking strategy 比 chunk size 更重要。** Fixed-size splits 会打断 tables、code 和 nested headers。Sentence-aware 是默认；semantic 或 LLM-based chunking 对 technical docs 和 product manuals 值得投入。
- **Parent-doc pattern.** retrieve 小的 “child” chunks 以获得 precision。当同一个 parent section 中的多个 children 出现时，换入 parent block 以保留 context。这能稳定提升 answer quality，无需 retraining。
- **k_rerank=3 通常最优。** 超过它的每个 extra chunk 都增加 token cost 和 generation latency，却不提升 answer quality。如果 k=8 仍然优于 k=3，你的 reranker 表现不足。
- **HyDE / query expansion.** 从 query 生成 hypothetical answer，embed 它，再 retrieve。弥合短 questions 与长 documents 之间的 phrasing gap。不训练即可免费提升 precision。
- **Context budget under 8K tokens.** 持续打到这个上限意味着 reranker threshold 太松。
- **Version everything.** Prompts、chunking rules、embedding model、reranker。任何 drift 都会静默破坏 answer quality。CI 用 faithfulness、context precision 和 unanswered-question rate 做 gates，在用户看到前拦截 regressions。
- **Three-way retrieval（BM25 + dense + learned-sparse like SPLADE）在 2026 benchmarks 中超过 two-way**，尤其适合 proper nouns 与 semantics 混合的 queries。基础设施支持 SPLADE indexes 时上线它。

根据 2026 industry measurements，合适 retrieval design 可减少 70-90% hallucinations。多数 RAG performance gains 来自更好的 retrieval，而不是 model fine-tuning。

## 交付成果

保存为 `outputs/skill-retrieval-picker.md`：

```markdown
---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.
```

## 练习

1. **Easy.** 在 500-document corpus 上实现上面的 `hybrid_search`。测试 20 个 queries。比较 BM25-only、dense-only 和 hybrid 的 recall at 5。
2. **Medium.** 添加 MRR calculation。对每个有已知 correct document 的 test query，找出 correct doc 在 BM25、dense 和 hybrid rankings 中的 rank。报告每个的 MRR。
3. **Hard.** 使用 MultipleNegativesRankingLoss（Sentence Transformers）在你的 domain 上 fine-tune dense encoder。用 500 query-document pairs 构建 training set。比较 fine-tune 前后的 recall。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| BM25 | Keyword search | Okapi BM25。按 term frequency、IDF 和 length 为 documents 打分。 |
| Dense retrieval | Vector search | 将 query + doc encode 成 vectors，寻找 nearest neighbors。 |
| Bi-encoder | Embedding model | 独立 encode query 和 doc。query time 快。 |
| Cross-encoder | Reranker model | 一起 encode query + doc。慢但准确。 |
| RRF | Rank fusion | 通过求和 `1/(k + rank)` 合并两个 rankings。 |
| Recall@k | Retrieval metric | relevant doc 出现在 top-k 中的 queries 比例。 |

## 延伸阅读

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)——BM25 的权威处理。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906)——DPR，canonical bi-encoder。
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720)——缩小与 dense 差距的 learned-sparse retriever。
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)——RRF paper。
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832)——late-interaction retrieval。
