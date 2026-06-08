# 命名实体识别

> 把名字抽出来。听起来很容易，直到你遇到歧义边界、嵌套实体和领域黑话。

**类型：** Build
**语言：** Python
**先修：** 第 5 阶段 · 02（BoW + TF-IDF），第 5 阶段 · 03（Word Embeddings）
**时间：** 约 75 分钟

## 要解决的问题

"Apple sued Google over its iPhone search deal in the US." 五个实体：Apple（ORG）、Google（ORG）、iPhone（PRODUCT）、search deal（也许是）、US（GPE）。一个好的 NER 系统会抽出所有实体，并给出正确类型。一个差的系统会漏掉 iPhone，把水果 Apple 和公司 Apple 混淆，还把 "US" 标成 PERSON。

NER 是每条结构化抽取 pipeline 下面的主力。简历解析、合规日志扫描、病历匿名化、搜索查询理解、聊天机器人响应的 grounding、法律合同抽取。你很少直接看见它，但你总是在依赖它。

本课会从经典路线（rule-based、HMM、CRF）走到现代路线（BiLSTM-CRF，然后 transformers）。每一步都解决前一步的某个具体限制。这种模式就是本课的重点。

## 核心概念

**BIO tagging**（或 BILOU）把实体抽取变成序列标注问题。给每个 token 标上 `B-TYPE`（实体开头）、`I-TYPE`（实体内部）或 `O`（不在任何实体内）。

```text
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

多 token 实体会串起来：`New B-GPE`、`York I-GPE`、`City I-GPE`。理解 BIO 的模型可以抽取任意 span。

架构演进：

- **Rule-based。** Regex + gazetteer lookup。对已知实体 precision 高，对新实体零覆盖。
- **HMM。** Hidden Markov Model。给定 tag 的 token emission probability，tag-to-tag transition probability。用 Viterbi decode。在标注数据上训练。
- **CRF。** Conditional Random Field。像 HMM，但它是判别式的，所以可以混入任意特征（词形、大小写、邻近词）。在 2026 年，低资源部署里它仍是经典生产主力。
- **BiLSTM-CRF。** 用神经特征替代手工特征。LSTM 双向读取句子，顶部 CRF 层强制 tag 序列一致。
- **Transformer-based。** Fine-tune BERT，并加 token-classification head。准确率最高。计算量最大。

## 动手实现

### 第 1 步：BIO tagging helpers

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### 第 2 步：手工特征

对经典（非神经）NER 来说，特征就是全部。好用的特征包括：

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` 返回 `xXxxxx`。`word_shape("USA-2024")` 返回 `XXX-dddd`。大小写模式对专有名词是高信号。

### 第 3 步：简单 rule-based + dictionary baseline

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

生产 gazetteers 往往有从 Wikipedia 和 DBpedia 抓取的数百万条目。覆盖率很好。消歧（公司 Apple vs 水果 apple）很糟。这就是统计模型获胜的原因。

### 第 4 步：CRF 这一步（草图，不是完整实现）

如果没有概率论基础，50 行从零实现完整 CRF 并不会让人更明白。这里改用 `sklearn-crfsuite`：

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` 和 `c2` 是 L1 与 L2 regularization。`all_possible_transitions=True` 让模型学习非法序列（例如 `O` 后接 `I-ORG`）是不太可能的，这就是 CRF 如何在你不手写约束的情况下强制 BIO 一致性。

### 第 5 步：BiLSTM-CRF 增加了什么

特征变成学习出来的。输入：token embeddings（GloVe 或 fastText）。LSTM 从左到右、从右到左读取句子。拼接后的 hidden states 进入 CRF 输出层。CRF 仍然强制 tag-sequence consistency；LSTM 用学习特征替代手工特征。

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

CRF 层使用 `torchcrf.CRF`（pip install pytorch-crf）。除非你有数万条标注句子，否则它相对手工 CRF 的提升可测量，但比你预期的小。

## 实际使用

spaCy 开箱即用地提供生产级 NER。

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```text
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

注意 `iPhone` 被标成了 `ORG`，而不是 `PRODUCT`。spaCy 的 small model 对产品实体覆盖较弱。large model（`en_core_web_lg`）会更好。transformer model（`en_core_web_trf`）还会更好。

Hugging Face 做 BERT-based NER：

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```text
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` 会把连续的 B-X、I-X tokens 合并成 span。没有它，你会得到 token-level labels，然后必须自己合并。

### LLM-based NER（2026 选项）

Zero-shot 和 few-shot LLM NER 如今在许多领域已经能和 fine-tuned models 竞争，而且在标注数据稀缺时明显更好。

- **Zero-shot prompting。** 给 LLM 一个实体类型列表和示例 schema。要求 JSON 输出。开箱即用；在新领域上准确率中等。
- **ZeroTuneBio-style prompting。** 把任务拆成候选抽取 → 含义解释 → 判断 → 复查。多阶段 prompt（不是 one-shot）会显著提升生物医学 NER 准确率。同样模式也适用于法律、金融和科学领域。
- **Dynamic prompting with RAG。** 每次推理调用都从一个小型标注 seed set 中检索最相似的标注样例，动态构建 few-shot prompt。2026 benchmark 中，这能把 GPT-4 生物医学 NER F1 相比静态 prompting 提升 11-12%。
- **Per-entity-type decomposition。** 对长文档来说，一次调用抽取所有实体类型会随着长度增长而丢 recall。每个实体类型单独跑一次抽取。推理成本更高，但准确率显著更高。这是临床笔记和法律合同中的标准模式。

截至 2026 年的生产建议：在收集训练数据之前，先做一个 LLM zero-shot baseline。很多时候 F1 已经足够好，你永远不需要 fine-tune。

### 经典 NER 仍然获胜的地方

即使 LLM 可用，经典 NER 仍会在这些情况下胜出：

- 延迟预算低于 50ms。
- 你有数千个标注样本，并且需要 98%+ F1。
- 领域 ontology 稳定，预训练 CRF 或 BiLSTM 能很好迁移。
- 监管约束要求 on-prem、非生成式模型。

### 它会在哪里崩掉

- **Domain shift。** 在 CoNLL 上训练的 NER 用到法律合同上，表现可能比 gazetteer 更差。要在你的领域上 fine-tune。
- **Nested entities。** "Bank of America Tower" 同时是 ORG 和 FACILITY。标准 BIO 无法表示重叠 span。你需要 nested NER（multi-pass 或 span-based models）。
- **Long entities。** "United States Federal Deposit Insurance Corporation." Token-level 模型有时会把它拆开。使用 `aggregation_strategy` 或后处理。
- **Sparse types。** 医学 NER 标签如 DRUG_BRAND、ADVERSE_EVENT、DOSE。通用模型完全不知道。Scispacy 和 BioBERT 是那里起步点。

## 交付成果

保存为 `outputs/skill-ner-picker.md`：

```markdown
---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.
```

## 练习

1. **简单。** 实现 `bio_to_spans`（`spans_to_bio` 的逆操作），并在 10 个句子上验证 round-trip consistency。
2. **中等。** 在 CoNLL-2003 English NER 数据集上训练上面的 sklearn-crfsuite CRF。用 `seqeval` 报告 per-entity F1。典型结果：约 84 F1。
3. **困难。** 在一个领域特定 NER 数据集（医学、法律或金融）上 fine-tune `distilbert-base-cased`。和 spaCy small model 对比。记录 data leakage checks，并写下让你意外的发现。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| NER | 抽取名字 | 用类型（PERSON、ORG、GPE、DATE 等）标注 token spans。 |
| BIO | 标注方案 | `B-X` 开始，`I-X` 延续，`O` 表示外部。 |
| BILOU | 更好的 BIO | 增加 `L-X`（last）和 `U-X`（unit），边界更干净。 |
| CRF | 结构化分类器 | 建模标签之间的 transitions，而不只是 emissions。强制有效序列。 |
| Nested NER | 重叠实体 | 一个 span 和它的子 span 是不同实体。BIO 无法表达这一点。 |
| Entity-level F1 | 正确的 NER 指标 | 预测 span 必须和真实 span 完全匹配。Token-level F1 会夸大准确率。 |

## 延伸阅读

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — BiLSTM-CRF 论文。Canonical。
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — 引入后来成为标准的 token-classification 模式。
- [spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities) — `Doc.ents` 和 `Span` 上每个属性的实践参考。
- [seqeval](https://github.com/chakki-works/seqeval) — 正确的指标库。永远使用它。
