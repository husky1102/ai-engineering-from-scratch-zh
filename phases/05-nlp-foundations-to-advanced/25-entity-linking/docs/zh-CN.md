# 实体链接与消歧

> NER 找到了 “Paris”。实体链接要决定：Paris, France？Paris Hilton？Paris, Texas？Paris（特洛伊王子）？没有链接，你的知识图谱就会一直含糊。

**类型：** Build
**语言：** Python
**先修：** Phase 5 · 06 (NER), Phase 5 · 24 (Coreference Resolution)
**时间：** ~60 minutes

## 要解决的问题

一句话写着：“Jordan beat the press.” 你的 NER 把 “Jordan” 标成 PERSON。很好。但它是*哪个* Jordan？

- Michael Jordan（篮球）？
- Michael B. Jordan（演员）？
- Michael I. Jordan（Berkeley ML 教授，是的，这种混淆在 ML 论文里真实存在）？
- Jordan（国家）？
- Jordan（希伯来语名）？

Entity linking（EL）会把每个 mention 解析到知识库中的唯一条目：Wikidata、Wikipedia、DBpedia，或你的领域 KB。两个子任务：

1. **Candidate generation。** 给定 “Jordan”，哪些 KB 条目是可能的？
2. **Disambiguation。** 给定上下文，哪个 candidate 是正确的？

两个步骤都可学习，也都有 benchmark。组合 pipeline 已经稳定了十年；变化的是 disambiguator 的质量。

## 核心概念

![Entity linking pipeline: mention → candidates → disambiguated entity](../assets/entity-linking.svg)

**Candidate generation。** 给定 mention surface form（“Jordan”），在 alias index 中查找 candidates。Wikipedia alias dictionaries 覆盖大多数命名实体：“JFK” → John F. Kennedy、Jacqueline Kennedy、JFK airport、JFK（movie）。典型索引每个 mention 返回 10-30 个 candidates。

**Disambiguation：三种方法。**

1. **Prior + context（Milne & Witten, 2008）。** `P(entity | mention) × context-similarity(entity, text)`。效果好，速度快，无需训练。
2. **基于嵌入（ESS / REL / Blink）。** 编码 mention + context。编码每个 candidate 的 description。选择最大余弦。2020-2024 年默认方案。
3. **生成式（GENRE, 2021；LLM-based, 2023+）。** 逐 token 解码实体的 canonical name。受限于合法实体名 trie，因此输出保证是有效 KB id。

**端到端 vs pipeline。** 现代模型（ELQ、BLINK、ExtEnD、GENRE）在一次 pass 中运行 NER + candidate generation + disambiguation。Pipeline 系统仍然主导生产，因为你可以替换组件。

### 两个测量值

- **Mention recall（candidate gen）。** gold mentions 中，正确 KB 条目出现在 candidate list 中的比例。这是整个 pipeline 的下限。
- **Disambiguation accuracy / F1。** 给定正确 candidates，top-1 有多经常是正确的。

始终同时报告两者。一个在 80% candidate recall 上有 99% disambiguation 的系统，本质上是 80% pipeline。

## 动手实现

### Step 1: build an alias index from Wikipedia redirects

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Wikipedia alias data：约 18M 对（alias, entity）。从 Wikidata dumps 下载。存成 inverted index。

### Step 2: context-based disambiguation

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Jaccard overlap 只是玩具方法。替换为嵌入上的 cosine similarity（transformer 版本见 `code/main.py` step-2）。

### Step 3: embedding-based (BLINK-style)

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

索引时，为每个 KB entity 嵌入一次。查询时，把 mention + context 嵌入一次，对 candidate pool 做 dot-product，选择最大值。

### Step 4: generative entity linking (concept)

GENRE 会逐字符解码实体的 Wikipedia title。Constrained decoding（见第 20 课）确保只能输出有效 title。它和 KB-backed trie 紧密集成。现代后继是 REL-GEN，以及带 structured output 的 LLM-prompted EL。

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

结合 whitelist（Outlines `choice`），这是 2026 年最容易上线的 EL pipeline。

### Step 5: evaluate on AIDA-CoNLL

AIDA-CoNLL 是标准 EL benchmark：1,393 篇 Reuters 文章、34k mentions、Wikipedia entities。报告 in-KB accuracy（`P@1`）和 out-of-KB NIL-detection rate。

## 常见陷阱

- **NIL handling。** 有些 mentions 不在 KB 中（新兴实体、冷门人物）。系统必须预测 NIL，而不是猜一个错误实体。单独测量。
- **Mention boundary errors。** 上游 NER 漏掉部分 span（“Bank of America” 只标成 “Bank”）。EL recall 会下降。
- **Popularity bias。** 训练系统会过度预测高频实体。ML 论文中的 “Michael I. Jordan” 往往会被链接到篮球 Jordan。
- **Cross-lingual EL。** 把中文文本中的 mentions 映射到英文 Wikipedia entities。需要多语言 encoder 或翻译步骤。
- **KB staleness。** 新公司、事件、人物不在去年的 Wikipedia dump 里。生产 pipeline 需要 refresh loop。

## 实际使用

2026 年的栈：

| 场景 | 选择 |
|-----------|------|
| 通用英文 + Wikipedia | BLINK or REL |
| Cross-lingual, KB = Wikipedia | mGENRE |
| LLM-friendly, few mentions/day | Prompt Claude/GPT-4 with candidate list + constrained JSON |
| 领域专用 KB（医疗、法律） | Custom BERT with KB-aware retrieval + fine-tune on domain AIDA-style set |
| 极低延迟 | Exact-match prior only (Milne-Witten baseline) |
| Research SOTA | GENRE / ExtEnD / generative LLM-EL |

2026 年可上线的生产模式：NER → coref → 对每个 mention 做 EL → 把 clusters 折叠成每个 cluster 一个 canonical entity。输出：文档中每个实体一个 KB id，而不是每个 mention 一个。

## 交付成果

保存为 `outputs/skill-entity-linker.md`：

```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

## 练习

1. **Easy.** 在 `code/main.py` 中对 10 个歧义 mentions（Paris、Jordan、Apple）实现 prior+context disambiguator。手工标注正确实体。测量 accuracy。
2. **Medium.** 用 sentence transformer 编码 50 个歧义 mentions。嵌入每个 candidate 的 description。比较 embedding-based disambiguation 和 Jaccard context overlap。
3. **Hard.** 构建一个 1k-entity 领域 KB（例如你公司中的员工 + 产品）。端到端实现 NER + EL。在 100 个 held-out sentences 上测量 precision 和 recall。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Entity linking (EL) | 链到 Wikipedia | 把一个 mention 映射到唯一 KB entry。 |
| Candidate generation | 它可能是谁？ | 为一个 mention 返回可能 KB entries 的 shortlist。 |
| Disambiguation | 选对那个 | 用上下文给 candidates 打分，选择赢家。 |
| Alias index | 查找表 | 从 surface form → candidate entities 的映射。 |
| NIL | 不在 KB 中 | 显式预测没有 KB entry 匹配。 |
| KB | Knowledge base | Wikidata、Wikipedia、DBpedia，或你的领域 KB。 |
| AIDA-CoNLL | Benchmark | 带 gold entity links 的 1,393 篇 Reuters 文章。 |

## 延伸阅读

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) — foundational prior+context approach。
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) — 基于 embedding 的主力方法。
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) — 带 constrained decoding 的 generative EL。
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) — benchmark 论文。
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) — 开源生产栈。
