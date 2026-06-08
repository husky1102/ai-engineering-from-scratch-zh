# Capstone 08 — 面向受监管垂直领域的生产级 RAG Chatbot

> Harvey、Glean、Mendable 和 LlamaCloud 在 2026 年都运行着相同的生产形态。用 docling 或 Unstructured 摄取文档，用 ColPali 处理视觉内容。Hybrid search。用 bge-reranker-v2-gemma 重新排序。用 Claude Sonnet 4.7 synthesis，并通过 prompt caching 达到 60-80% hit rate。用 Llama Guard 4 和 NeMo Guardrails 防护。用 Langfuse 和 Phoenix 观察。用 RAGAS 在 200-question golden set 上评分。在一个受监管领域（legal、clinical、insurance）构建它；capstone 的目标是通过 golden set、red team 和 drift dashboard。

**类型：** Capstone
**语言：** Python (pipeline + API), TypeScript (chat UI)
**先修：** Phase 5 (NLP), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 17 (infrastructure), Phase 18 (safety)
**练习阶段：** P5 · P7 · P11 · P12 · P17 · P18
**时间：** 30 hours

## 要解决的问题

受监管领域 RAG（legal contracts、clinical trial protocols、insurance policies）是 2026 年最常交付的生产形态，因为 ROI 明显，风险也具体。Harvey（Allen & Overy）为 legal 构建了它。Mendable 交付 developer-docs 版本。Glean 覆盖 enterprise search。模式是：高保真 ingest，用 hybrid retrieve + rerank，带 citation enforcement 和 prompt caching 的 synthesis，多层 safety guard，并持续监控 drift。

难点不在模型。难点在 jurisdiction-aware compliance（HIPAA, GDPR, SOC2）、citation-level auditability、成本控制（当 hit rate 高时，prompt caching 带来 60-90% discount）、通过 RAGAS faithfulness 做 hallucination detection，以及当 source documents 更新但 index 没跟上时的 drift detection。本 capstone 要你在一个 200-question golden set 和 red-team suite 上交付完整系统。

## 核心概念

pipeline 有两侧。**Ingestion**：docling 或 Unstructured 解析 structured documents；ColPali 处理 visually rich documents；chunks 获得 summaries、tags 和 role-based access labels。vectors 进入 pgvector + pgvectorscale（低于 50M vectors）或 Qdrant Cloud；sparse BM25 并行运行。**Conversation**：LangGraph 处理 memory 和 multi-turn；每个 query 都运行 hybrid retrieval，用 bge-reranker-v2-gemma-2b rerank，用 Claude Sonnet 4.7 synthesis（prompt-cached），让 output 通过 Llama Guard 4 和 NeMo Guardrails，并输出 citation-anchored response。

eval stack 有四层。**Golden set**（200 个带 citations 的 labeled Q/A）用于 correctness。**Red team**（jailbreaks、PII extraction attempts、off-domain questions）用于 safety。**RAGAS** 为每一轮自动计算 faithfulness / answer relevance / context precision。**Drift dashboard**（Arize Phoenix）每周观察 retrieval quality 和 hallucination score。

Prompt caching 是成本杠杆。Claude 4.5+ 和 GPT-5+ 支持缓存 system prompts + retrieved context。在 60-80% hit rate 下，per-query cost 会下降 3-5x。pipeline 必须围绕稳定前缀设计（system prompt + reranked context first），才能达到高 cache hit rates。

## 架构

```text
documents (contracts, protocols, policies)
      |
      v
docling / Unstructured parse + ColPali for visuals
      |
      v
chunks + summaries + role-labels + jurisdiction tags
      |
      v
pgvector + pgvectorscale  +  BM25 (Tantivy)
      |
query + role + jurisdiction
      |
      v
LangGraph conversational agent
   +--- retrieve (hybrid)
   +--- filter by role + jurisdiction
   +--- rerank (bge-reranker-v2-gemma-2b or Voyage rerank-2)
   +--- synthesize (Claude Sonnet 4.7, prompt cached)
   +--- guard (Llama Guard 4 + NeMo Guardrails + Presidio output PII scrub)
   +--- cite + return
      |
      v
eval:
  RAGAS faithfulness / answer_relevance / context_precision (online)
  Langfuse annotation queue (sampled)
  Arize Phoenix drift (weekly)
  red team suite (pre-release)
```

## 技术栈

- Ingestion: Unstructured.io or docling for structured documents; ColPali for visually-rich PDFs
- Vector DB: pgvector + pgvectorscale under 50M vectors; Qdrant Cloud otherwise
- Sparse: Tantivy BM25 with field weights
- Orchestration: LlamaIndex Workflows (ingestion) + LangGraph (conversation)
- Re-ranker: bge-reranker-v2-gemma-2b self-hosted or Voyage rerank-2 hosted
- LLM: Claude Sonnet 4.7 with prompt caching; fallback Llama 3.3 70B self-hosted
- Eval: RAGAS 0.2 online, DeepEval for hallucination and jailbreak suites
- Observability: Langfuse self-hosted with annotation queue; Arize Phoenix for drift
- Guardrails: Llama Guard 4 input/output classifier, NeMo Guardrails v0.12 policy, Presidio PII scrub
- Compliance: role-based access labels on chunks; jurisdiction tags for GDPR/HIPAA

## 动手实现

1. **Ingestion。** 用 Unstructured 或 docling 解析你的 corpus（认真构建时为 1000-10000 documents）。对 scanned / visual-heavy pages，通过 ColPali 路由。产出带 summaries、role-labels、jurisdiction tags 的 chunks。

2. **Index。** Dense embeddings（Voyage-3 或 Nomic-embed-v2）进入 pgvector + pgvectorscale。通过 Tantivy 建 BM25 side-index。Role 和 jurisdiction filters 作为 payload。

3. **Hybrid retrieve。** 先按 role+jurisdiction filter；然后并行 dense + BM25；用 reciprocal rank fusion 合并；top-20 交给 reranker；top-5 交给 synth。

4. **用 prompt caching 做 synthesis。** System prompt + static policies 放进 cache header；reranked context 作为 cache extension；user question 作为 uncached suffix。steady state 目标是 60-80% cache hit rate。

5. **Guardrails。** input 上跑 Llama Guard 4；NeMo Guardrails rails 阻止 off-domain questions 或 policy-forbidden topics；Presidio scrub output 中意外出现的 PII；citation enforcement post-filter。

6. **Golden set。** 由 domain expert 标注 200 个 Q/A pairs，包含 (answer, citations)。按 exact-citation match、answer correctness、faithfulness（RAGAS）给 agent 评分。

7. **Red team。** 50 个 adversarial prompts：jailbreaks（PAIR, TAP）、PII exfiltration attempts、off-domain、cross-jurisdiction leaks。用 pass/fail 和 severity 评分。

8. **Drift dashboard。** Arize Phoenix 每周跟踪 retrieval quality（nDCG, citation faithfulness）。如果下降 5%，触发 alert。

9. **Cost report。** Langfuse：prompt-caching hit rate、tokens per query、按阶段拆分的 $/query。

## 实际使用

```text
$ chat --role=analyst --jurisdiction=GDPR
> what is the data-retention obligation for EU user profiles under our contract?
[retrieve]  hybrid top-20 filtered to GDPR + analyst-role
[rerank]    top-5 kept
[synth]     claude-sonnet-4.7, cache hit 74%, 0.8s
answer:
  The contract (Section 12.4, Master Services Agreement dated 2024-03-11)
  obligates EU user profile deletion within 30 days of termination per GDPR
  Article 17. The DPA amendment (DPA-v2.1, Section 5) extends this to 14 days
  for "restricted" category data.
  citations: [MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

## 交付成果

`outputs/skill-production-rag.md` 描述交付物。一个面向受监管领域的 chatbot，带 compliance labels 部署，通过 rubric，并由 live drift monitoring 观察。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | RAGAS faithfulness + answer relevance | golden set（200 Q/A）上的 online scores |
| 20 | Citation correctness | 带可验证 source anchors 的 answers 比例 |
| 20 | Guardrail coverage | Llama Guard 4 pass rate + jailbreak suite results |
| 20 | Cost / latency engineering | Prompt-cache hit rate、p95 latency、$/query |
| 15 | Drift monitoring dashboard | 带 weekly retrieval-quality trend 的 Phoenix live dashboard |
| **100** | | |

## 练习

1. 在另一个 jurisdiction 下构建第二个 corpus slice（例如，在 GDPR 旁边加入 HIPAA）。用 20-question cross-jurisdiction probe 展示 role+jurisdiction filtering 能防止 cross-leak。

2. 测量一周 production traffic 的 prompt-cache hit rate。识别哪些 queries 破坏 cache prefix。重构。

3. 添加带 10k-token summary buffer 的 multi-turn memory。测量随着 conversation 增长，faithfulness 是否下降。

4. 将 Claude Sonnet 4.7 换成 self-hosted Llama 3.3 70B。测量 $/query 和 faithfulness delta。

5. 添加 "unsure" mode：如果 top reranked scores 低于阈值，agent 说 "I do not have confident citations"，而不是回答。测量 false-confidence reduction。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| Prompt caching | "Cached system + context" | Claude/OpenAI feature：命中时 cached prefix tokens 享受 60-90% discount |
| RAGAS | "RAG evaluator" | 对 faithfulness、answer relevance、context precision 的自动评分 |
| Golden set | "Labeled eval" | 200+ expert-labeled Q/A with citations；ground truth |
| Jurisdiction tag | "Compliance label" | 附加到 chunks 的 GDPR/HIPAA/SOC2 scope；由 retrieval filter 强制执行 |
| Citation faithfulness | "Grounded answer rate" | claims 由可检索 source spans 支撑的比例 |
| Drift | "Retrieval quality decay" | nDCG 或 citation score 的 weekly change；alert threshold 5% |
| Red team | "Adversarial eval" | pre-release jailbreak、PII extraction、off-domain probes |

## 延伸阅读

- [Harvey AI](https://www.harvey.ai) — reference legal production stack
- [Glean enterprise search](https://www.glean.com) — reference RAG at enterprise scale
- [Mendable documentation](https://mendable.ai) — developer-docs RAG reference
- [LlamaCloud Parse + Index](https://docs.llamaindex.ai/en/stable/examples/llama_cloud/llama_parse/) — managed ingestion
- [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — cost-lever reference
- [RAGAS 0.2 documentation](https://docs.ragas.io/) — canonical RAG eval framework
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — reference drift observability
- [Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 safety classifier
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — policy rail framework
