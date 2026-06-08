# Natural Language Inference：Textual Entailment

> "t entails h" 意味着一个读到 t 的人会得出 h 为真的结论。NLI 是预测 entailment / contradiction / neutral 的任务。表面无聊，生产中却很承重。

**类型：** Learn
**语言：** Python
**先修：** Phase 5 · 05 (Sentiment Analysis), Phase 5 · 13 (Question Answering)
**时间：** ~60 minutes

## 要解决的问题

你构建了一个 summarizer。它生成了 summary。你怎么知道这个 summary 没有包含 hallucination？

你构建了一个 chatbot。它回答了 "yes"。你怎么知道这个答案受到 retrieved passage 支持？

你需要按主题分类 10,000 篇新闻文章。你没有 training labels。你能复用一个模型吗？

这三个问题都可以化约为 Natural Language Inference。NLI 问的是：给定一个 premise `t` 和一个 hypothesis `h`，`h` 是被 `t` 蕴含、被反驳，还是 neutral（无关）？

- **Hallucination check：** `t` = source document，`h` = summary claim。Not entailment = hallucination。
- **Grounded QA：** `t` = retrieved passage，`h` = generated answer。Not entailment = fabrication。
- **Zero-shot classification：** `t` = document，`h` = verbalized label（"This is about sports"）。Entailment = predicted label。

一个任务，三种生产用途。这就是为什么每个 RAG evaluation framework 都会在底层带一个 NLI model。

## 核心概念

![NLI: three-way classification, premise vs hypothesis](../assets/nli.svg)

**三个标签。**

- **Entailment。** `t` → `h`。"The cat is on the mat" entails "There is a cat."
- **Contradiction。** `t` → ¬`h`。"The cat is on the mat" contradicts "There is no cat."
- **Neutral。** 任一方向都无法推断。"The cat is on the mat" is neutral to "The cat is hungry."

**不是 logical entailment。** NLI 是 *natural* language inference，也就是典型人类读者会推断什么，而不是严格逻辑。"John walked his dog" 在 NLI 中 entails "John has a dog"，但严格的一阶逻辑只有在你把 possession 公理化后才会承认这一点。

**Datasets。**

- **SNLI** (2015)。570k human-annotated pairs，image captions 作为 premises。领域较窄。
- **MultiNLI** (2017)。覆盖 10 种 genres 的 433k pairs。2026 年的标准训练语料。
- **ANLI** (2019)。Adversarial NLI。人类专门编写用来击破已有模型的 examples。更难。
- **DocNLI, ConTRoL** (2020–21)。Document-length premises。测试 multi-hop 与 long-range inference。

**The architecture。** Transformer encoder（BERT、RoBERTa、DeBERTa）读取 `[CLS] premise [SEP] hypothesis [SEP]`。`[CLS]` representation 输入一个 3-way softmax。在 MNLI 上训练，在 held-out benchmarks 上评估，在 in-distribution pairs 上得到 90%+ accuracy。

**Zero-shot via NLI。** 给定一个 document 和 candidate labels，把每个 label 转成 hypothesis（"This text is about sports"）。计算每个的 entailment probability。选择最大值。这就是 Hugging Face 的 `zero-shot-classification` pipeline 背后的机制。

## 动手实现

### Step 1: run a pretrained NLI model

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # return all labels; replaces deprecated return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

对于 production NLI，`facebook/bart-large-mnli` 和 `microsoft/deberta-v3-large-mnli` 是 open defaults。DeBERTa-v3 位于 leaderboards 前列。

### Step 2: zero-shot classification

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

默认 template 是 "This example is about {label}."。可用 `hypothesis_template` 自定义。不需要 training data。不需要 fine-tuning。开箱即用。

### Step 3: faithfulness check for RAG

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

这是 RAGAS faithfulness 的核心。把 generated answer 拆成 atomic claims。将每个 claim 与 retrieved context 对照检查。报告被 entail 的比例。

### Step 4: hand-rolled NLI classifier (conceptual)

Stdlib-only toy 见 `code/main.py`：通过 lexical overlap + negation detection 比较 premise 与 hypothesis。它无法与 transformer models 竞争，但展示了任务形状：两个 texts 输入，3-way label 输出，loss = `{entail, contradict, neutral}` 上的 cross-entropy。

## Pitfalls

- **Hypothesis-only shortcuts。** Models 只看 hypothesis 就能在 SNLI 上以约 60% 预测 label，因为 "not"、"nobody"、"never" 与 contradiction 相关。它是检测 label leakage 的强 baseline。
- **Lexical overlap heuristic。** Subsequence heuristic（“每个 subsequence 都被 entailed”）能通过 SNLI，但会在 HANS/ANLI 上失败。使用 adversarial benchmarks。
- **Document-length degradation。** Single-sentence NLI models 在 document-length premises 上会掉 20+ F1。长 context 使用 DocNLI-trained models。
- **Zero-shot template sensitivity。** "This example is about {label}" vs "{label}" vs "The topic is {label}" 可能让 accuracy 摆动 10+ points。调优 template。
- **Domain mismatch。** MNLI 在 general English 上训练。Legal、medical 和 scientific text 需要 domain-specific NLI models（例如 SciNLI、MedNLI）。

## 实际使用

2026 stack：

| Use case | Model |
|---------|-------|
| General-purpose NLI | `microsoft/deberta-v3-large-mnli` |
| Fast / edge | `cross-encoder/nli-deberta-v3-base` |
| Zero-shot classification (lightweight) | `facebook/bart-large-mnli` |
| Document-level NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| Multilingual | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| Hallucination detection in RAG | NLI layer inside RAGAS / DeepEval |

2026 meta-pattern：NLI 是文本理解的 duct tape。只要你需要判断“Does A support B?” 或 “Does A contradict B?”，在再调用另一个 LLM 之前，先考虑 NLI。

## 交付成果

保存为 `outputs/skill-nli-picker.md`：

```markdown
---
name: nli-picker
description: Pick an NLI model, label template, and evaluation setup for a classification / faithfulness / zero-shot task.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

Given a use case (faithfulness check, zero-shot classification, document-level inference), output:

1. Model. Named NLI checkpoint. Reason tied to domain, length, language.
2. Template (if zero-shot). Verbalization pattern. Example.
3. Threshold. Entailment cutoff for the decision rule. Reason based on calibration.
4. Evaluation. Accuracy on held-out labeled set, hypothesis-only baseline, adversarial subset.

Refuse to ship zero-shot classification without a 100-example labeled sanity check. Refuse to use a sentence-level NLI model on document-length premises. Flag any claim that NLI solves hallucination — it reduces it; it does not eliminate it.
```

## 练习

1. **Easy.** 在覆盖全部三类的 20 个手写 (premise, hypothesis, label) triples 上运行 `facebook/bart-large-mnli`。测量 accuracy。加入 adversarial "subsequence heuristic" traps（"I did not eat the cake" vs "I ate the cake"），看看是否会被击破。
2. **Medium.** 在 100 条 AG News headlines 上比较 zero-shot template `"This text is about {label}"`、`"The topic is {label}"` 和 `"{label}"`。报告 accuracy swing。
3. **Hard.** 构建一个 RAG faithfulness checker：atomic-claim decomposition + 每个 claim 做 NLI。在 50 个带 gold context 的 RAG-generated answers 上评估。测量相对 hand labels 的 false-positive 和 false-negative rates。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| NLI | Natural Language Inference | 对 premise-hypothesis relationship 的 3-way classification。 |
| RTE | Recognizing Textual Entailment | NLI 的旧名称；同一个任务。 |
| Entailment | "t implies h" | 给定 t，典型读者会得出 h 为真的结论。 |
| Contradiction | "t rules out h" | 给定 t，典型读者会得出 h 为假的结论。 |
| Neutral | "undecided" | 从 t 到 h 任一方向都没有推断。 |
| Zero-shot classification | 把 NLI 当 classifier | 把 labels verbalize 成 hypotheses，选择最大 entailment。 |
| Faithfulness | 答案是否被支持？ | 在 (retrieved context, generated answer) 上做 NLI。 |

## 延伸阅读

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — SNLI。
- [Williams, Nangia, Bowman (2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) — MultiNLI。
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) — ANLI benchmark。
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) — NLI-as-classifier。
- [He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) — 2026 年的 NLI workhorse。
