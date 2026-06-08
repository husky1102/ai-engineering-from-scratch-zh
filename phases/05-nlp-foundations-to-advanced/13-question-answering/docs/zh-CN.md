# 问答系统

> 三类系统塑造了现代 QA。Extractive 找 spans。Retrieval-augmented 把它们 grounded 到 documents。Generative 生成 answers。每个现代 AI assistant 都是三者的混合。

**类型:** Build
**语言:** Python
**先修:** Phase 5 · 11 (Machine Translation), Phase 5 · 10 (Attention Mechanism)
**时间:** ~75 minutes

## 要解决的问题

用户输入 “When did the first iPhone launch?” 并期望得到 “June 29, 2007.” 不是 “Apple's history is long and varied.” 也不是孤零零的 “2007”。而是直接、grounded、correct 的 answer。

过去十年中，三种 architectures 主导了 QA。

- **Extractive QA.** 给定 question 和已知包含答案的 passage，找出 passage 中 answer span 的 start 和 end indices。SQuAD 是 canonical benchmark。
- **Open-domain QA.** passage 不给定。先 retrieve 相关 passage，再 extract 或 generate answer。这是今天每个 RAG pipeline 的基石。
- **Generative / Closed-book QA.** large language model 从 parametric memory 回答。没有 retrieval。inference 最快，对 facts 最不可靠。

2026 年的趋势是 hybrid：retrieve 最好的几个 passages，然后 prompt generative model 基于这些 passages 回答。这就是 RAG，lesson 14 会深入讲 retrieval half。本课构建 QA half。

## 核心概念

![QA architectures: extractive, retrieval-augmented, generative](../assets/qa.svg)

**Extractive.** 用 transformer（BERT family）一起 encode question 和 passage。训练两个 heads，预测 answer 的 start 和 end token indices。Loss 是 valid positions 上的 cross-entropy。Output 是 passage 中的一个 span。按构造不会 hallucinate，也按构造无法处理 passage 无法回答的问题。

**Retrieval-augmented (RAG).** 两个 stages。首先，retriever 从 corpus 中找到 top-`k` passages。然后，reader（extractive 或 generative）使用这些 passages 产生 answer。retriever-reader split 让二者可以独立训练和评估。现代 RAG 通常还在二者之间加入 reranker。

**Generative.** decoder-only LLM（GPT、Claude、Llama）从 learned weights 回答。没有 retrieval step。对 common knowledge 很强，对 rare 或 recent facts 可能灾难性失败。hallucination rate 与 fact 在 pretraining data 中的 frequency 负相关。

## 动手实现

### Step 1: 使用 pretrained model 做 extractive QA

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2` 在 SQuAD 2.0 上训练，SQuAD 2.0 包含 unanswerable questions。默认情况下，`question-answering` pipeline 即便模型的 null score 获胜，也会返回最高分 span——它*不会*自动返回 empty answer。要得到显式 “no answer” 行为，需要在 pipeline call 中传 `handle_impossible_answer=True`：pipeline 只有在 null score 超过每个 span score 时才返回 empty answer。不管哪种方式，都要始终检查 `score` field。

### Step 2: retrieval-augmented pipeline（sketch）

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

两阶段 pipeline。Dense retriever（Sentence-BERT）通过 semantic similarity 找到 relevant passages。Extractive reader（RoBERTa-SQuAD）从 combined top passages 中抽取 answer span。适用于小 corpus。百万文档 corpus 需要 FAISS 或 vector database。

### Step 3: generative with RAG

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

prompt pattern 很重要。明确告诉模型基于 context 回答，并在 context 不充分时返回 “I don't know”，相比 naive prompting 可将 hallucination rates 降低 40-60%。更复杂的 patterns 会添加 citations、confidence scores 和 structured extraction。

### Step 4: 反映真实世界的 evaluation

SQuAD 使用 **Exact Match (EM)** 和 **token-level F1**。EM 是 normalization（lowercase、strip punctuation、remove articles）后的严格匹配——prediction 要么完全匹配，要么得 0。F1 按 prediction 与 reference 的 token overlap 计算，并给 partial credit。二者都会低估 paraphrases：“June 29, 2007” vs “June 29th, 2007” 通常得到 0 EM（ordinal 打破 normalization），但因 overlapping tokens 仍会获得可观 F1。

对 production QA：

- **Answer accuracy**（LLM-judged 或 human-judged，因为 metrics 无法捕捉 semantic equivalence）。
- **Citation accuracy.** cited passage 是否真的支持 answer？可以通过 generated citations 与 retrieved passages 之间的 string match 自动检查。
- **Refusal calibration.** 当 retrieved passages 不包含 answer 时，system 是否正确说 “I don't know”？测量 false confidence rate。
- **Retrieval recall.** 评估 reader 之前，先测 retriever 是否把正确 passage 放入 top-`k`。reader 无法修复 missing passage。

### RAGAS：2026 production eval framework

`RAGAS` 专为 RAG systems 设计，是 2026 年的 shipping default。它在不需要 gold references 的情况下打四个维度分：

- **Faithfulness.** answer 中每个 claim 是否来自 retrieved context？通过 NLI-based entailment 衡量。你的主要 hallucination metric。
- **Answer relevance.** answer 是否回答了 question？通过从 answer 生成 hypothetical questions 并与 real question 比较来衡量。
- **Context precision.** retrieved chunks 中有多少实际 relevant？低 precision = prompt 中有噪声。
- **Context recall.** retrieved set 是否包含所有必要信息？低 recall = reader 不可能成功。

Reference-free scoring 让你能在 live production traffic 上评估，而无需 curated gold answers。对 exact-match metrics 没用的 open-ended questions，在上层再加 LLM-as-judge。

`pip install ragas`。接入你的 retriever + reader。每个 query 得到四个 scalars。对 regressions 发 alert。

## 实际使用

2026 年 stack。

| Use case | Recommended |
|---------|-------------|
| Given passage, find answer span | `deepset/roberta-base-squad2` |
| Over a fixed corpus, closed-book not acceptable | RAG: dense retriever + LLM reader |
| Real-time over a document store | RAG with hybrid (BM25 + dense) retriever + reranker (lesson 14) |
| Conversational QA (follow-up questions) | LLM with conversation history + RAG on each turn |
| Highly factual, regulated domains | Extractive over an authoritative corpus; never generative alone |

Extractive QA 在 2026 年不时髦，因为 RAG with LLMs 能处理更多情况。但它仍会在需要 literal quotation 的场景中上线：legal research、regulatory compliance、audit tools。

## 交付成果

保存为 `outputs/skill-qa-architect.md`：

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

## 练习

1. **Easy.** 在 10 段 Wikipedia passages 上设置上面的 SQuAD extractive pipeline。手写 10 个 questions。测量 answer correct 的频率。如果 passages 和 questions 干净，应该有 7-9 个正确。
2. **Medium.** 添加 refusal classifier。当 top retrieval score 低于 threshold（例如 0.3 cosine）时，返回 “I don't know”，而不是调用 reader。在 held-out set 上调 threshold。
3. **Hard.** 在你选择的 10,000-document corpus 上构建 RAG pipeline。用 RRF fusion 实现 hybrid retrieval（BM25 + dense，见 lesson 14）。测量有无 hybrid step 时的 answer accuracy。记录哪些 question types 获益最多。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Extractive QA | Find the answer span | 在给定 passage 中预测 answer 的 start 和 end indices。 |
| Open-domain QA | QA over a corpus | 没有给定 passage；必须先 retrieve 再 answer。 |
| RAG | Retrieve then generate | Retrieval-augmented generation。Retriever + reader pipeline。 |
| SQuAD | Canonical benchmark | Stanford Question Answering Dataset。EM + F1 metrics。 |
| Hallucination | Made-up answer | 不被 retrieved context 支持的 reader output。 |
| Refusal calibration | Know when to shut up | system 在无法回答时正确说 “I don't know”。 |

## 延伸阅读

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250)——benchmark paper。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906)——DPR，QA 的 canonical dense retriever。
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)——命名 RAG 的论文。
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997)——全面 RAG survey。
