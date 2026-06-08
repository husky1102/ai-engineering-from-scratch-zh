# 关系抽取与知识图谱构建

> NER 找到了实体。实体链接把它们锚定到知识库。关系抽取找到它们之间的边。知识图谱就是节点、边和它们的 provenance 之和。

**类型：** Build
**语言：** Python
**先修：** Phase 5 · 06 (NER), Phase 5 · 25 (Entity Linking)
**时间：** ~60 minutes

## 要解决的问题

分析师读到：“Tim Cook became CEO of Apple in 2011.” 四个事实：

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

Relation Extraction（RE）把自由文本转成结构化 triples `(subject, relation, object)`。跨语料聚合后，你就有了知识图谱。再聚合和查询，你就有了支撑 RAG、analytics 或 compliance audits 的推理基底。

2026 年的问题：LLMs 抽取关系时很积极。太积极了。它们会 hallucinate 源文本不支持的 triples。没有 provenance，你就无法区分真实 triples 和看似合理的虚构内容。2026 年答案是 AEVS 风格的 anchor-and-verify pipelines。

## 核心概念

![Text → triples → knowledge graph](../assets/relation-extraction.svg)

**Triple form。** `(subject_entity, relation_type, object_entity)`。关系来自闭合 ontology（Wikidata properties、FIBO、UMLS），或开放集合（OpenIE 风格，什么都可以）。

**三种抽取方法。**

1. **基于规则 / pattern。** Hearst patterns：“X such as Y” → `(Y, isA, X)`。再加手写 regex。脆弱、精确、可解释。
2. **监督分类器。** 给定句子中的两个 entity mentions，从固定集合中预测 relation。用 TACRED、ACE、KBP 训练。2015–2022 年标准方法。
3. **生成式 LLM。** 提示模型输出 triples。开箱即用。需要 provenance，否则会幻觉出看似合理的垃圾。

**AEVS（Anchor-Extraction-Verification-Supplement, 2026）。** 当前的 hallucination-mitigation framework：

- **Anchor。** 用精确位置识别每个 entity span 和 relation-phrase span。
- **Extract。** 生成链接到 anchor spans 的 triples。
- **Verify。** 把每个 triple element 匹配回源文本；拒绝任何不受支持的内容。
- **Supplement。** 覆盖率 pass 确保没有 anchored span 被漏掉。

Hallucinations 会大幅下降。它需要更多计算，但可审计。

**Open-vs-closed 取舍。**

- **闭合 ontology。** 固定 property list（例如 Wikidata 的 11,000+ properties）。可预测、可查询、难以编造。
- **Open IE。** 任意动词短语都可以成为 relation。高召回，低精度。查询很乱。

生产 KG 通常混合使用：用 open IE 做发现，然后把 relations canonicalize 到闭合 ontology，再合并进主图。

## 动手实现

### Step 1: pattern-based extraction

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

完整玩具 extractor 见 `code/main.py`。Hearst patterns 仍会在领域专用 pipelines 中上线，因为它们可调试。

### Step 2: supervised relation classification

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL 是 seq2seq relation extractor：输入文本，输出 triples，且已经是 Wikidata property ids。它在 distant-supervision data 上 fine-tuned。标准开源权重 baseline。

### Step 3: LLM-prompted extraction with anchoring

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

验证每个返回的 span 是否匹配源文本。拒绝任何 `text[start:end] != triple_entity` 的结果。这是最小形式的 AEVS “verify” 步骤。

### Step 4: canonicalize onto a closed ontology

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # drop unmapped open relations or route to manual review
```

Canonicalization 往往占工程工作的 60-80%。要为它留预算。

### Step 5: build a small graph and query

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

这是每个 RAG-over-KG 系统的原子。用 RDF triple stores（Blazegraph、Virtuoso）、property graphs（Neo4j）或 vector-augmented graph stores 来扩展它。

## 常见陷阱

- **RE 之前要先 coreference。** “He founded Apple”——RE 需要知道 “he” 是谁。先运行 coref（第 24 课）。
- **Entity canonicalization。** “Apple Inc” 和 “Apple” 必须解析到同一个 node。先做 entity linking（第 25 课）。
- **幻觉 triples。** LLMs 会输出源文本不支持的 triples。强制 span verification。
- **Relation canonicalization drift。** Open IE relations 不一致（“was born in,” “came from,” “is a native of”）。折叠到 canonical ids，否则 graph 无法查询。
- **Temporal errors。** “Tim Cook is CEO of Apple”——现在为真，2005 年为假。许多关系都有时间边界。使用 qualifiers（Wikidata 中的 `P580` start time、`P582` end time）。
- **Domain mismatch。** REBEL 在 Wikipedia 上训练。法律、医学和科学文本通常需要领域 fine-tuned RE models。

## 实际使用

2026 年的栈：

| 场景 | 选择 |
|-----------|------|
| 快速生产，通用领域 | REBEL or LlamaPred with Wikidata canonicalization |
| 领域专用（biomed, legal） | SciREX-style domain fine-tune + custom ontology |
| LLM-prompted, audited output | AEVS pipeline: anchor → extract → verify → supplement |
| 高吞吐新闻 IE | Pattern-based + supervised hybrid |
| 从零构建 KG | Open IE + manual canonicalization pass |
| Temporal KG | Extract with qualifiers (start/end time, point in time) |

集成模式：NER → coref → entity linking → relation extraction → ontology mapping → graph load。每个阶段都是潜在质量门。

## 交付成果

保存为 `outputs/skill-re-designer.md`：

```markdown
---
name: re-designer
description: Design a relation extraction pipeline with provenance and canonicalization.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Given a corpus (domain, language, volume) and downstream use (KG-RAG, analytics, compliance), output:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Reason tied to precision vs recall target.
2. Ontology. Closed property list (Wikidata / domain) or open IE with canonicalization pass.
3. Provenance. Every triple carries source char-span + doc id. Non-negotiable for audit.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. Precision / recall on 200 hand-labelled triples + hallucination-rate on LLM-extracted sample.

Refuse any LLM-based RE pipeline without span verification (source provenance). Refuse open-IE output flowing into a production graph without canonicalization. Flag pipelines with no temporal qualifier on time-bounded relations (employer, spouse, position).
```

## 练习

1. **Easy.** 在 5 个新闻文章句子上运行 `code/main.py` 中的 pattern extractor。手工检查 precision。
2. **Medium.** 在同样句子上使用 REBEL（或小 LLM）。比较 triples。哪个 extractor 的 precision 更高？recall 更高？
3. **Hard.** 构建 AEVS pipeline：用 LLM 抽取 + 对源文本验证 spans。在 50 个 Wikipedia 风格句子上，测量 verify step 前后的 hallucination rate。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Triple | Subject-relation-object | `(s, r, o)` tuple，是 KG 的原子单元。 |
| Open IE | 抽取一切 | 开放词表关系短语；高召回、低精度。 |
| Closed ontology | 固定 schema | 有边界的关系类型集合（Wikidata、UMLS、FIBO）。 |
| Canonicalization | 全部规范化 | 把表面名称 / 关系映射到 canonical ids。 |
| AEVS | 有依据的抽取 | Anchor-Extraction-Verification-Supplement pipeline（2026）。 |
| Provenance | 真值来源链接 | 每个 triple 都携带 doc id + char-span 指向来源。 |
| Distant supervision | 便宜标签 | 将文本与已有 KG 对齐来创建训练数据。 |

## 延伸阅读

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) — distant-supervision 论文。
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) — seq2seq RE 主力模型。
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) — 联合 IE。
- [AEVS — Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) — 2026 hallucination-mitigation 设计。
- [Wikidata SPARQL tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — canonical graph queries。
