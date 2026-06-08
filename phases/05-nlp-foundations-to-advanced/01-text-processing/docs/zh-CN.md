# 文本处理：分词、词干提取与词形还原

> 语言是连续的。模型是离散的。预处理就是中间的桥。

**类型：** Build
**语言：** Python
**先修：** 第 2 阶段 · 14（朴素贝叶斯）
**时间：** 约 45 分钟

## 要解决的问题

模型不能直接读懂 "The cats were running."。它读的是整数。

每个 NLP 系统一开始都会面对同样三个问题：一个词从哪里开始？词根是什么？什么时候应该把 "run"、"running"、"ran" 当成同一个东西，什么时候又应该把它们当成不同的东西？

分词错了，模型就会从垃圾里学习。如果你的 tokenizer 把 `don't` 当成一个 token，却把 `do n't` 当成两个，训练分布就被拆开了。如果你的 stemmer 把 `organization` 和 `organ` 压到同一个词干，主题建模就会崩。如果你的 lemmatizer 需要词性上下文，而你没有传进去，动词就会被当成名词处理。

本课会从零构建这三步预处理，然后展示 NLTK 和 spaCy 如何完成同样的工作，让你看清其中的取舍。

## 核心概念

三个操作。每个都有自己的任务，也都有自己的失效模式。

**Tokenization（分词）**把字符串切成 token。这里的 "Token" 是故意保持宽泛的，因为正确粒度取决于任务。经典 NLP 常用词级 token。transformer 常用 subword。没有空格分隔的语言可能用字符级 token。

**Stemming（词干提取）**用规则砍掉后缀。快、激进、粗糙。`running -> run`。`organization -> organ`。第二个例子就是它的失效模式。

**Lemmatization（词形还原）**利用语法知识把词还原到词典形式。更慢，更准确，需要查表或形态分析器。`ran -> run`（需要知道 "ran" 是 "run" 的过去式）。`better -> good`（需要知道比较级形式）。

经验法则：当速度重要且你能容忍噪声时用 stemming（搜索索引、粗粒度分类）。当意义重要时用 lemmatization（问答、语义搜索、任何用户会阅读的输出）。

## 动手实现

### 第 1 步：一个 regex 词级 tokenizer

最简单可用的 tokenizer 会按非字母数字字符切分，同时把标点保留为独立 token。它不完美，也不是最终方案，但一行就能跑。

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

三个模式按优先级排列：带可选内部撇号的单词（`don't`、`it's`）、纯数字、任何单个非空白且非字母数字字符作为独立 token（标点）。

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

要注意的失效模式：`3pm` 会被切成 `['3', 'pm']`，因为我们在字母片段和数字片段之间做了交替。对多数任务已经够用。URL、邮箱、hashtag 都会断开。生产环境中，要把这些模式加在通用模式之前。

### 第 2 步：Porter stemmer（只实现 step 1a）

完整 Porter 算法有五个阶段的规则。单独 step 1a 就覆盖了最常见的英文后缀，也足以教会这种模式。

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

规则要从上往下读。`ies -> i` 这条规则解释了为什么 `ponies -> poni`，而不是 `pony`。真正的 Porter 会在 step 1b 里继续修正。规则之间会竞争。前面的规则获胜。顺序比任何单条规则都更重要。

### 第 3 步：基于查表的 lemmatizer

真正的 lemmatization 需要形态学。一个适合教学的版本可以用小型 lemma 表加 fallback。

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

最后一个例子是关键教学点。`watched` 不在我们的表里，而 fallback 只处理 `ing`。真实的 lemmatization 会覆盖 `ed`、不规则动词、比较级形容词、带音变的复数（`children -> child`）。这就是为什么生产系统会使用 WordNet、spaCy 的 morphologizer，或完整形态分析器。

### 第 4 步：把它们串起来

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

缺失的一块是 POS tagger。第 5 阶段 · 07（POS Tagging）会构建一个。现在先默认所有 token 都是 `NOUN`，并明确承认这个限制。

## 实际使用

NLTK 和 spaCy 都提供了生产级版本。各自只需要几行。

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` 会处理缩写、Unicode，以及你的 regex 漏掉的边界情况。`PorterStemmer` 会跑完全部五个阶段。`WordNetLemmatizer` 需要把 NLTK 的 Penn Treebank 词性标签翻译成 WordNet 的缩写集合。上面那段转换胶水，正是大多数教程跳过的部分。

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```text
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy 把整条 pipeline 藏在 `nlp(text)` 后面。分词、POS tagging、lemmatization 都会运行。大规模场景下比 NLTK 更快。开箱即用的准确率也更高。代价是你很难随意替换单个组件。

### 什么时候选哪个

| 场景 | 选择 |
|-----------|------|
| 教学、研究、替换组件 | NLTK |
| 生产、多语言、速度重要 | spaCy |
| Transformer pipeline（反正会用模型自己的 tokenizer） | 使用 `tokenizers` / `transformers`，跳过经典预处理 |

### 几乎没人提醒你的两个失效模式

大多数教程讲完算法就停了。但真实预处理 pipeline 里有两件事一定会咬人，而且几乎没人覆盖。

**可复现性漂移。** NLTK 和 spaCy 会在版本之间改变 tokenization 和 lemmatizer 行为。spaCy 2.x 里产出 `['do', "n't"]` 的内容，3.x 里可能产出 `["don't"]`。你的模型是在一个分布上训练的。推理时却跑在另一个分布上。准确率悄悄下降，没人知道原因。要在 `requirements.txt` 里固定库版本。写一个预处理回归测试，冻结 20 个样例句子的期望 tokenization。每次升级都运行它。

**训练 / 推理不一致。** 训练时使用激进预处理（小写、停用词移除、stemming），部署时却喂原始用户输入，然后看着性能坠崖。这是生产 NLP 最常见的失败。如果训练时做了预处理，推理时必须运行完全相同的函数。把预处理作为模型包里的函数交付，而不是作为一个由服务团队重写的 notebook cell。

## 交付成果

一个可复用 prompt，帮助工程师在不读三本教材的情况下选择预处理策略。

保存为 `outputs/prompt-preprocessing-advisor.md`：

```markdown
---
name: preprocessing-advisor
description: Recommends a tokenization, stemming, and lemmatization setup for an NLP task.
phase: 5
lesson: 01
---

You advise on classical NLP preprocessing. Given a task description, you output:

1. Tokenization choice (regex, NLTK word_tokenize, spaCy, or transformer tokenizer). Explain why.
2. Whether to stem, lemmatize, both, or neither. Explain why.
3. Specific library calls. Name the functions. Quote the POS-tag translation if NLTK is involved.
4. One failure mode the user should test for.

Refuse to recommend stemming for user-visible text. Refuse to recommend lemmatization without POS tags. Flag non-English input as needing a different pipeline.
```

## 练习

1. **简单。** 扩展 `tokenize`，让 URL 保持为单个 token。测试：`tokenize("Visit https://example.com today.")` 应该产出一个 URL token。
2. **中等。** 实现 Porter step 1b。如果一个词包含元音且以 `ed` 或 `ing` 结尾，就移除它。处理双辅音规则（`hopping -> hop`，不是 `hopp`）。
3. **困难。** 构建一个 lemmatizer：用 WordNet 作为查找表，但当 WordNet 没有条目时 fallback 到你的 Porter stemmer。拿带标签语料分别对比纯 WordNet 和纯 Porter，衡量准确率。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Token | 一个词 | 模型消费的任何单位。可以是词、subword、字符或 byte。 |
| Stem | 词根 | 基于规则剥离后缀的结果。不一定是真实单词。 |
| Lemma | 词典形式 | 你会拿去查词典的形式。需要语法上下文才能正确计算。 |
| POS tag | 词性 | NOUN、VERB、ADJ 这类类别。准确 lemmatize 需要它。 |
| Morphology | 词形变化规则 | 词如何根据时态、数量、格而改变形式。Lemmatization 依赖它。 |

## 延伸阅读

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — 原始论文，五页，至今仍是最清晰的解释。
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) — 一个真实 pipeline 是如何串起来的。
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) — 你还没想过的 tokenization 边界情况。
