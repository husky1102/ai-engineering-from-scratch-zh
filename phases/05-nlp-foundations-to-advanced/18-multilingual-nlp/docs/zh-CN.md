# 多语言 NLP

> 一个模型，100+ 种语言，其中大多数几乎没有训练数据。Cross-lingual transfer 是 2020 年代的实用奇迹。

**类型：** Learn
**语言：** Python
**先修：** Phase 5 · 04 (GloVe, FastText, Subword), Phase 5 · 11 (Machine Translation)
**时间：** ~45 minutes

## 要解决的问题

英语有数十亿个标注样本。乌尔都语有几千个。Maithili 几乎没有。任何服务全球用户的实用 NLP system，都必须能在长尾语言上工作，而这些语言通常不存在 task-specific training data。

Multilingual models 通过同时在多种语言上训练一个模型来解决这个问题。共享表示让模型能把在高资源语言中学到的技能迁移到低资源语言上。用 English sentiment analysis fine-tune 模型后，它开箱就能在 Urdu 上给出出人意料地好的 sentiment predictions。这就是 zero-shot cross-lingual transfer，它重塑了 NLP 面向世界交付的方式。

本课会命名这些权衡、canonical models，以及一个常常绊倒多语言新团队的决策：为 transfer 选择 source language。

## 核心概念

![Cross-lingual transfer via shared multilingual embedding space](../assets/multilingual.svg)

**Shared vocabulary。** Multilingual models 使用在所有目标语言文本上训练的 SentencePiece 或 WordPiece tokenizer。Vocabulary 是共享的：同一个 subword unit 表示相关语言中的同一个 morpheme。英语和意大利语里的 `anti-` 得到同一个 token。

**Shared representation。** 一个在多语言 masked language modeling 上预训练的 transformer，会学到不同语言中语义相似的句子产生相似 hidden states。mBERT、XLM-R 和 NLLB 都表现出这一点。英语 "cat" 的 embeddings 会聚在法语 "chat" 与西班牙语 "gato" 附近，full-sentence embeddings 也是如此。

**Zero-shot transfer。** 在一种语言（通常是英语）的 labeled data 上 fine-tune 模型。Inference 时，在模型支持的任何其他语言上运行它。不需要 target-language labels。对类型学相关的语言结果强，对距离远的语言结果弱。

**Few-shot fine-tuning。** 在目标语言中添加 100-500 个 labeled examples。分类任务的 accuracy 会跳到 English baseline 的 95-98%。这是 multilingual NLP 中性价比最高的单一杠杆。

## The models

| Model | Year | Coverage | Notes |
|-------|------|----------|-------|
| mBERT | 2018 | 104 languages | 训练于 Wikipedia。第一个实用 multilingual LM。低资源语言上较弱。 |
| XLM-R | 2019 | 100 languages | 训练于 CommonCrawl（远大于 Wikipedia）。设定 cross-lingual baseline。Base 270M，Large 550M。 |
| XLM-V | 2023 | 100 languages | XLM-R 加 1M-token vocabulary（对比 250k）。低资源语言上更好。 |
| mT5 | 2020 | 101 languages | 用于 multilingual generation 的 T5 architecture。 |
| NLLB-200 | 2022 | 200 languages | Meta 的 translation model；包含 55 种 low-resource languages。 |
| BLOOM | 2022 | 46 languages + 13 programming | 多语言训练的 open 176B LLM。 |
| Aya-23 | 2024 | 23 languages | Cohere 的 multilingual LLM。Arabic、Hindi、Swahili 上较强。 |

按 use case 选择。Classification 使用 XLM-R-base 是理性的默认值。Generation tasks 根据 translation vs open generation 选择 mT5 或 NLLB。LLM-style work 则搭配 Aya-23 或 Claude，并使用显式 multilingual prompting。

## The source-language decision (2026 research)

大多数团队默认用英语作为 fine-tuning source。近期研究（2026）显示这通常是错的。

Language similarity 比 raw corpus size 更能预测 transfer quality。对 Slavic targets，German 或 Russian 往往优于 English。对 Indic targets，Hindi 往往优于 English。**qWALS** similarity metric（2026，基于 World Atlas of Language Structures features）量化了这一点。**LANGRANK**（Lin et al., ACL 2019）是一个独立、更早的方法，它结合 linguistic similarity、corpus size 和 genetic relatedness 来为候选 source languages 排名。

实践规则：如果你的 target language 有一个类型学上接近的 high-resource relative，先尝试在那个语言上 fine-tune，再与 English fine-tune 对比。

## 动手实现

### Step 1: zero-shot cross-lingual classification

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

一个模型，三种语言，同一个 API。XLM-R 在 NLI data 上训练，通过 entailment trick 能很好迁移到 classification。

### Step 2: multilingual embedding space

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

翻译会落在 embedding space 中相近的位置。另一个不同的英文句子会落得更远。这就是 cross-lingual retrieval、clustering 和 similarity 能工作的原因。

### Step 3: few-shot fine-tuning strategy

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

对于 100-500 个 target-language examples，`num_train_epochs=5` 和 `learning_rate=2e-5` 是安全默认值。更高的 learning rates 会导致 multilingual alignment 坍缩，最后得到一个 English-only model。

## Evaluation that actually works

- **Per-language accuracy on held-out sets。** 不要聚合。Aggregate 会隐藏长尾。
- **Benchmark against monolingual baseline。** 对数据足够的语言，从零训练的 monolingual model 有时会超过 multilingual one。要测试。
- **Entity-level tests。** 目标语言中的 named entities。Multilingual models 对远离 Latin 的 scripts 往往 tokenization 较弱。
- **Cross-lingual consistency。** 两种语言表达相同含义时，应产生相同 prediction。测量 gap。

## 实际使用

2026 stack：

| Task | Recommended |
|-----|-------------|
| Classification, 100 languages | XLM-R-base (~270M) fine-tuned |
| Zero-shot text classification | `joeddav/xlm-roberta-large-xnli` |
| Multilingual sentence embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Translation, 200 languages | `facebook/nllb-200-distilled-600M`（见 lesson 11） |
| Generative multilingual | Claude, GPT-4, Aya-23, mT5-XXL |
| Low-resource language NLP | XLM-V 或在相关 high-resource language 上做 domain-specific fine-tune |

如果 performance 重要，始终为 target language 的 fine-tuning 预留预算。Zero-shot 是起点，不是最终答案。

### The tokenization tax (what goes wrong for low-resource languages)

Multilingual models 在所有语言之间共享一个 tokenizer。这个 vocabulary 在一个由 English、French、Spanish、Chinese、German 主导的 corpus 上训练。对 dominant set 之外的任何语言，三种税会悄悄叠加：

- **Fertility tax。** Low-resource language text 会比 English 切成多得多的 tokens per word。一个 Hindi 句子可能需要等价 English 句子的 3-5 倍 tokens。这 3-5 倍会吞掉你的 context window、training efficiency 和 latency。
- **Variant recovery tax。** 每一个 typo、diacritic variant、Unicode normalization mismatch 或 case variation 都会在 embedding space 中变成 cold-start unrelated sequence。模型无法学习 native speaker 认为显而易见的 orthographic correspondences。
- **Capacity spillover tax。** 税 1 和 2 会消耗 context positions、layer depth 和 embedding dimensions。留给 actual reasoning 的容量，会系统性地小于同一个模型给 high-resource language 的容量。

实践症状是：你的模型在 Hindi 上正常训练，loss curve 看起来对，eval perplexity 看起来合理，production outputs 却有微妙错误。Morphology 在句中坍缩。Rare inflections 始终无法恢复。**你不能靠 data-scale 走出一个坏 tokenizer。**

缓解：选择对目标语言覆盖良好的 tokenizer（XLM-V 的 1M-token vocabulary 是直接修复）；训练前在 held-out target text 上验证 tokenization fertility；对真正长尾的 scripts 使用 byte-level fallback（SentencePiece `byte_fallback=True`、GPT-2-style byte-level BPE），确保永远没有 OOV。

## 交付成果

保存为 `outputs/skill-multilingual-picker.md`：

```markdown
---
name: multilingual-picker
description: Pick source language, target model, and evaluation plan for a multilingual NLP task.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Given requirements (target languages, task type, available labeled data per language), output:

1. Source language for fine-tuning. Default English; check LANGRANK or qWALS if target language has a typologically close high-resource language.
2. Base model. XLM-R (classification), mT5 (generation), NLLB (translation), Aya-23 (generative LLM).
3. Few-shot budget. Start with 100-500 target-language examples if available. Zero-shot only if labeling is infeasible.
4. Evaluation plan. Per-language accuracy (not aggregate), cross-lingual consistency, entity-level F1 on non-Latin scripts.

Refuse to ship a multilingual model without per-language evaluation — aggregate metrics hide long-tail failures. Flag scripts with low tokenization coverage (Amharic, Tigrinya, many African languages) as needing a model with byte-fallback (SentencePiece with byte_fallback=True, or byte-level tokenizer like GPT-2).
```

## 练习

1. **Easy.** 在 English、French、Hindi 和 Arabic 中，每种语言各取 10 个句子，运行 zero-shot classification pipeline。报告每种语言的 accuracy。你应该看到 French 很强，Hindi 尚可，Arabic 有波动。
2. **Medium.** 使用 `paraphrase-multilingual-MiniLM-L12-v2` 在一个小型混合语言语料库上构建 cross-lingual retriever。用 English 查询，检索任意语言的 documents。测量 recall@5。
3. **Hard.** 对一个 Hindi classification task 比较 English-source 与 Hindi-source fine-tuning。在两种 regime 下都使用 500 个 target-language examples 进行 few-shot fine-tuning。报告哪个 source 产生更好的 Hindi accuracy，以及好多少。这是 LANGRANK thesis 的迷你版。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Multilingual model | 一个模型，多种语言 | 跨语言共享 vocabulary 和 parameters。 |
| Cross-lingual transfer | 在一种语言上训练，在另一种语言上运行 | 在 source 上 fine-tune，在没有 target-language labels 的情况下在 target 上评估。 |
| Zero-shot | 没有 target-language labels | 不在 target language 上 fine-tune 的 transfer。 |
| Few-shot | 少量 target labels | 用于 fine-tuning 的 100-500 个 target-language examples。 |
| mBERT | 第一个 multilingual LM | 在 Wikipedia 上预训练的 104-language BERT。 |
| XLM-R | 标准 cross-lingual baseline | 在 CommonCrawl 上预训练的 100-language RoBERTa。 |
| NLLB | Meta 的 200-language MT | No Language Left Behind。包含 55 种 low-resource languages。 |

## 延伸阅读

- [Conneau et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) — XLM-R 论文。
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) — 开启 cross-lingual transfer 研究线的分析论文。
- [Costa-jussà et al. (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672) — NLLB-200 论文。
- [Üstün et al. (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) — Aya，Cohere 的 multilingual LLM。
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65) — qWALS / LANGRANK source-language 论文。
