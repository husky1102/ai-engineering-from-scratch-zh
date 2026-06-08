# 文本摘要

> Extractive systems 告诉你 document 说了什么。Abstractive systems 告诉你作者想表达什么。不同任务，不同陷阱。

**类型:** Build
**语言:** Python
**先修:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 11 (Machine Translation)
**时间:** ~75 minutes

## 要解决的问题

一篇 2,000-word news article 出现在你的 feed 中。你需要 120 words 抓住它。你可以从 article 中挑出三句最重要的句子（extractive），也可以用自己的话重写内容（abstractive）。二者都叫 summarization。它们是完全不同的问题。

Extractive summarization 是 ranking problem。给每个 sentence 打分，返回 top-`k`。output 总是 grammatical，因为它是 verbatim 提取的。风险是漏掉分散在文章各处的内容。

Abstractive summarization 是 generation problem。transformer 以 input 为条件生成新 text。output 流畅且压缩，但可能 hallucinate source 中不存在的 facts。风险是自信地编造。

本课会构建二者，并指出各自拥有的 failure mode。

## 核心概念

![Extractive TextRank vs abstractive transformer](../assets/summarization.svg)

**Extractive.** 将 article 视为 graph：nodes 是 sentences，edges 是 similarities。在 graph 上运行 PageRank（或类似算法），按 sentences 与其他所有内容连接的程度打分。最高分 sentences 是 summary。经典实现是 **TextRank**（Mihalcea and Tarau, 2004）。

**Abstractive.** 在 document-summary pairs 上 fine-tune transformer encoder-decoder（BART、T5、Pegasus）。inference 时，模型读取 document，并通过 cross-attention 逐 token 生成 summary。Pegasus 特别使用 gap-sentence pretraining objective，因此不需要太多 fine-tuning 就很擅长 summarization。

用 **ROUGE**（Recall-Oriented Understudy for Gisting Evaluation）评估。ROUGE-1 和 ROUGE-2 衡量 unigram 和 bigram overlap。ROUGE-L 衡量 longest common subsequence。越高越好，但 40 ROUGE-L 是 “good”，50 是 “exceptional”。每篇论文都会报告三者。使用 `rouge-score` package。

## 动手实现

### Step 1: TextRank（extractive）

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

有两点值得命名。similarity function 使用 log-normalized word overlap，这是原始 TextRank variant。TF-IDF vectors 的 cosine 也可以。damping factor 0.85 和 iteration count 是 PageRank defaults。

### Step 2: 用 BART 做 abstractive

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN 在 CNN/DailyMail corpus 上 fine-tuned。它开箱生成 news-style summaries。对其他 domains（scientific papers、dialog、legal），使用对应 Pegasus checkpoint，或在你的 target data 上 fine-tune。

### Step 3: ROUGE evaluation

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

始终使用 stemming。没有它，“running” 和 “run” 会被算作不同 words，ROUGE 会低估。

### Beyond ROUGE（2026 summarization eval）

ROUGE 已经主导 summarization metric 二十年，但在 2026 年单独使用并不够。NLG papers 的大规模 meta-analysis 显示：

- **BERTScore**（contextual embedding similarity）到 2023 年持续扩大使用，现在多数 summarization papers 都会与 ROUGE 一起报告。
- **BARTScore** 将 evaluation 视为 generation：给定 source，按 pretrained BART 给 summary 的 likelihood 打分。
- **MoverScore**（contextual embeddings 上的 Earth Mover's Distance）在 2025 summarization benchmarks 中到达 top spot，因为它比 ROUGE 更能捕捉 semantic overlap。
- **FactCC** 和 **QA-based faithfulness** 在 2021-2023 年常见，现在常被 **G-Eval** 替代（一个 GPT-4 prompt chain，用 chain-of-thought reasoning 为 coherence、consistency、fluency、relevance 打分）。
- **G-Eval** 和类似 LLM-judge approaches 在 rubrics 设计良好时，与 human judgment 约 80% 一致。

Production recommendation：报告 ROUGE-L 做 legacy comparison，BERTScore 做 semantic overlap，G-Eval 做 coherence 和 factuality。用 50-100 条 human-labeled summaries 做校准。

### Step 4: factuality problem

Abstractive summaries 容易 hallucination。Extractive summaries 的 hallucination risk 低得多，因为 output 是从 source verbatim 提取的；不过如果 source sentences 被去上下文化、过时，或乱序引用，它们仍可能误导。这是 production systems 在 compliance-adjacent content 上仍偏好 extractive methods 的最大原因。

需要命名的 Hallucination types：

- **Entity swap.** Source 写 “John Smith.” Summary 写 “John Brown.”
- **Number drift.** Source 写 “25,000.” Summary 写 “25 million.”
- **Polarity flip.** Source 写 “rejected the offer.” Summary 写 “accepted the offer.”
- **Fact invention.** Source 没有提 CEO。Summary 写 CEO 批准了。

有效的 evaluation approaches：

- **FactCC.** 一个训练在 source sentence 与 summary sentence entailment 上的 binary classifier。预测 factual/not-factual。
- **QA-based factuality.** 向 QA model 提问，答案在 source 中。如果 summary 支持不同答案，就 flag。
- **Entity-level F1.** 比较 source 与 summary 中的 named entities。只出现在 summary 中的 entities 可疑。

对任何 factuality 重要的 user-facing 内容（news、medical、legal、financial），extractive 是更安全的默认选择。Abstractive 需要在 loop 中加入 factuality check。

## 实际使用

2026 年 stack：

| Use case | Recommended |
|---------|-------------|
| News, 3-5 sentence summary, English | `facebook/bart-large-cnn` |
| Scientific papers | `google/pegasus-pubmed` or a tuned T5 |
| Multi-document, long-form | Any LLM with 32k+ context, prompted |
| Dialog summarization | `philschmid/bart-large-cnn-samsum` |
| Extractive, low hallucination risk by construction | TextRank or `sumy`'s LSA / LexRank |

如果 compute 不是约束，long context LLMs 在 2026 年通常会胜过 specialized models。tradeoff 是 cost 和 reproducibility；specialized models 的 outputs 更一致。

## 交付成果

保存为 `outputs/skill-summary-picker.md`：

```markdown
---
name: summary-picker
description: Pick extractive or abstractive, named library, factuality check.
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

Given a task (document type, compliance requirement, length, compute budget), output:

1. Approach. Extractive or abstractive. Explain in one sentence why.
2. Starting model / library. Name it. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed`, or an LLM prompt.
3. Evaluation plan. ROUGE-1, ROUGE-2, ROUGE-L (use rouge-score with stemming). Plus factuality check if abstractive.
4. One failure mode to probe. Entity swap is the most common in abstractive news summarization; flag samples where source entities do not appear in summary.

Refuse abstractive summarization for medical, legal, financial, or regulated content without a factuality gate. Flag input over the model's context window as needing chunked map-reduce summarization (not just truncation).
```

## 练习

1. **Easy.** 在 5 篇 news articles 上运行 TextRank。将 top-3 sentences 与 reference summary 比较。测量 ROUGE-L。你应该在 CNN/DailyMail-style articles 上看到 30-45 ROUGE-L。
2. **Medium.** 实现 entity-level factuality：从 source 和 summary 中抽取 named entities（spaCy），计算 summary 中 source entities 的 recall，以及 summary entities 相对 source 的 precision。high precision + low recall 表示安全但简短；low precision 表示 hallucinated entities。
3. **Hard.** 在 50 篇 CNN/DailyMail articles 上比较 BART-large-CNN 与 LLM（Claude 或 GPT-4）。报告 ROUGE-L、factuality（by entity F1）和 cost per summary。记录各自胜出的地方。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Extractive | Pick sentences | verbatim 返回 source 中的 sentences。不会 hallucinate。 |
| Abstractive | Rewrite | 以 source 为条件生成新 text。可能 hallucinate。 |
| ROUGE | Summary metric | system output 与 reference 之间的 N-gram / LCS overlap。 |
| TextRank | Graph-based extractive | 在 sentence similarity graph 上做 PageRank。 |
| Factuality | Is it right | summary claims 是否被 source 支持。 |
| Hallucination | Made-up content | source 不支持的 summary content。 |

## 延伸阅读

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/)——extractive canonical paper。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461)——BART paper。
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777)——Pegasus 和 gap-sentence objective。
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/)——ROUGE paper。
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661)——factuality landscape paper。
