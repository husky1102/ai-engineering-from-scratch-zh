# 共指消解

> “She called him. He did not answer. The doctor was at lunch.” 三个指代，两个对象，而且没人被点名。共指消解要弄清楚谁是谁。

**类型：** Learn
**语言：** Python
**先修：** Phase 5 · 06 (NER), Phase 5 · 07 (POS & Parsing)
**时间：** ~60 minutes

## 要解决的问题

从一篇 300 词文章中抽取 Apple Inc. 的所有提及。文章写着 “Apple” 时很容易。难的是它写着 “the company”、“they”、“Cupertino's technology giant” 或 “Jobs's firm”。如果不把这些提及解析到同一个实体，你的 NER pipeline 会漏掉 60-80% 的提及。

共指消解会把所有指向同一个真实世界实体的表达链接到一个 cluster 中。它是表层 NLP（NER、parsing）和下游语义任务（IE、QA、summarization、KG）之间的胶水。

为什么它在 2026 年重要：

- 摘要：“The CEO announced...” vs “Tim Cook announced...”：摘要应该说出 CEO 的名字。
- 问答：“Who did she call?” 需要解析 “she”。
- 信息抽取：知识图谱里同时有 “PER1 founded Apple” 和 “Jobs founded Apple” 两条独立记录，这是错的。
- 多文档 IE：跨多篇关于同一事件的文章合并 mentions，就是跨文档共指。

## 核心概念

![Coreference clustering: mentions → entities](../assets/coref.svg)

**任务。** 输入：一篇文档。输出：mentions（spans）的聚类，其中每个 cluster 指向一个实体。

**Mention 类型。**

- **命名实体。** “Tim Cook”
- **名词性提及。** “the CEO”, “the company”
- **代词性提及。** “he”, “she”, “they”, “it”
- **同位语。** “Tim Cook, Apple's CEO,”

**架构。**

1. **基于规则（Hobbs, 1978）。** 使用语法规则，基于句法树进行代词消解。很好的 baseline。在代词上出人意料地难以击败。
2. **Mention-pair classifier。** 对每一对 mentions（m_i, m_j）预测它们是否共指。通过传递闭包聚类。2016 年前的标准方法。
3. **Mention-ranking。** 对每个 mention，为候选 antecedents 排名（包括“没有 antecedent”）。选择最高分。
4. **基于 span 的端到端模型（Lee et al., 2017）。** Transformer encoder。枚举所有长度上限内的候选 spans。预测 mention scores。为每个 span 预测 antecedent probability。贪心聚类。现代默认方案。
5. **生成式（2024+）。** 提示 LLM：“List every pronoun in this text and its antecedent.” 简单案例效果很好，在长文档和罕见 referents 上挣扎。

**评估指标。** 标准指标有五个（MUC、B³、CEAF、BLANC、LEA），因为没有单个指标能完整捕获聚类质量。通常报告前三个指标平均值作为 CoNLL F1。2026 年 CoNLL-2012 上的 state-of-the-art 约为 83 F1。

**已知困难案例。**

- 指向数页前引入实体的 definite descriptions。
- Bridging anaphora（“the wheels” → 前文提到的一辆 car）。
- 中文、日语等语言中的 zero anaphora。
- Cataphora（代词在 referent 前出现）：“When **she** walked in, Mary smiled.”

## 动手实现

### Step 1: pretrained neural coreference (AllenNLP / spaCy-experimental)

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # experimental model
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

在更长文档上，你会得到类似结果：
- Cluster 1: [Apple, The company, they]
- Cluster 2: [new products]

### Step 2: rule-based pronoun resolver (teaching)

见 `code/main.py` 中仅使用 stdlib 的实现：

1. 抽取 mentions：命名实体（大写 spans）、代词（dict lookup）、definite descriptions（“the X”）。
2. 对每个代词，查看前 K 个 mentions，并按以下方式打分：
   - gender/number agreement（heuristic）
   - recency（越近越好）
   - syntactic role（优先 subjects）
3. 链接到得分最高的 antecedent。

它无法和神经模型竞争，但展示了搜索空间，以及端到端模型必须做出的决策。

### Step 3: using LLMs for coreference

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

要注意两种失败模式。第一，LLM 会过度合并（把指向两个不同人的 “him” 和 “her” 合并）。第二，LLM 会在长文档中静默漏掉 mentions。务必用 span-offset checks 验证。

### Step 4: evaluation

标准 conll-2012 脚本计算 MUC、B³、CEAF-φ4，并报告平均值。对于内部 eval，先在标注测试集上从 span-level precision 和 recall 开始，再加上 mention-linking F1。

## 常见陷阱

- **Singleton explosion。** 有些系统把每个 mention 都报告成自己的 cluster。B³ 比较宽松，MUC 会惩罚这一点。始终检查三个指标。
- **长上下文中的代词。** 文档超过 2,000 tokens 时性能会下降约 15 F1。要谨慎 chunk。
- **性别假设。** 硬编码的 gender rules 会在非二元 referents、组织、动物上出错。使用学习模型或中性打分。
- **LLM 在长文档上漂移。** 单次 API 调用无法可靠地跨 50+ 段落聚类 mentions。使用 sliding-window + merge。

## 实际使用

2026 年的栈：

| 场景 | 选择 |
|-----------|------|
| 英文，单文档 | `en_coreference_web_trf`（spaCy-experimental）或 AllenNLP neural coref |
| 多语言 | 在 OntoNotes 或 Multilingual CoNLL 上训练的 SpanBERT / XLM-R |
| 跨文档事件共指 | 专门的端到端模型（2025–26 SOTA） |
| 快速 LLM baseline | GPT-4o / Claude 搭配 structured-output coref prompt |
| 生产对话系统 | 基于规则的 fallback + neural primary + critical slots 人工审核 |

2026 年可上线的集成模式：先运行 NER，再运行 coref，把 coref clusters 合并进 NER entities。下游任务看到的是每个 cluster 一个实体，而不是每个 mention 一个实体。

## 交付成果

保存为 `outputs/skill-coref-picker.md`：

```markdown
---
name: coref-picker
description: Pick a coreference approach, evaluation plan, and integration strategy.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

Given a use case (single-doc / multi-doc, domain, language), output:

1. Approach. Rule-based / neural span-based / LLM-prompted / hybrid. One-sentence reason.
2. Model. Named checkpoint if neural.
3. Integration. Order of operations: tokenize → NER → coref → downstream task.
4. Evaluation. CoNLL F1 (MUC + B³ + CEAF-φ4 average) on held-out set + manual cluster review on 20 documents.

Refuse LLM-only coref for documents over 2,000 tokens without sliding-window merge. Refuse any pipeline that runs coref without a mention-level precision-recall report. Flag gender-heuristic systems deployed in demographically diverse text.
```

## 练习

1. **Easy.** 在 5 个手写段落上运行 `code/main.py` 中的基于规则 resolver。对照 ground truth 测量 mention-link accuracy。
2. **Medium.** 在一篇新闻文章上使用 pretrained neural coref model。把 clusters 和你自己的人工标注比较。它在哪里失败？
3. **Hard.** 构建 coref-enhanced NER pipeline：先 NER，再通过 coref clusters 合并。测量它相对 NER-only 在 100 篇文章上的 entity-coverage improvement。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Mention | 一个指代 | 指向某个实体的文本 span（名称、代词、名词短语）。 |
| Antecedent | “it” 指的东西 | 后续 mention 与之共指的较早 mention。 |
| Cluster | 该实体的所有 mentions | 全部指向同一真实世界实体的 mentions 集合。 |
| Anaphora | 后向指代 | 后出现的 mention 指向更早的 mention（“he” → “John”）。 |
| Cataphora | 前向指代 | 更早的 mention 指向后面的 referent（“When he arrived, John...”）。 |
| Bridging | 隐式指代 | “I bought a car. The wheels were bad.”（那辆车的 wheels。） |
| CoNLL F1 | 排行榜上的数字 | MUC、B³、CEAF-φ4 F1 分数的平均值。 |

## 延伸阅读

- [Jurafsky & Martin, SLP3 Ch. 26 — Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — 经典教材章节。
- [Lee et al. (2017). End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) — 基于 span 的端到端方法。
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) — 改善 coref 的预训练。
- [Pradhan et al. (2012). CoNLL-2012 Shared Task](https://aclanthology.org/W12-4501/) — benchmark。
- [Hobbs (1978). Resolving Pronoun References](https://www.sciencedirect.com/science/article/pii/0024384178900064) — 经典规则方法。
