# 机器翻译

> Translation 是三十年来资助 NLP research 的任务，而且现在仍在继续付费。

**类型:** Build
**语言:** Python
**先修:** Phase 5 · 10 (Attention Mechanism), Phase 5 · 04 (GloVe, FastText, Subword)
**时间:** ~75 minutes

## 要解决的问题

模型读取一种语言的 sentence，并生成另一种语言的 sentence。长度会变。词序会变。一些 source words 会映射到多个 target words，反过来也一样。Idioms 拒绝一一对应。“I miss you” 在 French 中是 “tu me manques”——字面意思是 “you are lacking to me”。没有 word-level alignment 能保住这种关系。

Machine translation 是迫使 NLP 发明 encoder-decoders、attention、transformers，并最终走向整个 LLM paradigm 的任务。每一次进步都来自翻译质量可测，而人类与机器之间的差距顽固存在。

本课跳过历史课，教授 2026 年的工作 pipeline：pretrained multilingual encoder-decoder（NLLB-200 或 mBART）、subword tokenization、beam search、BLEU 和 chrF evaluation，以及仍会未被发现地进入 production 的几个 failure modes。

## 核心概念

![MT pipeline: tokenize → encode → decode with attention → detokenize](../assets/mt-pipeline.svg)

现代 MT 是在 parallel text 上训练的 transformer encoder-decoder。encoder 用 source language 的 tokenization 读取 source。decoder 通过 cross-attention（lesson 10）使用 encoder output，并一次生成一个 subword。Decoding 使用 beam search 来避开 greedy-decoding trap。output 会被 detokenized、detruecased，并与 reference 对比打分。

三个 operational choices 决定真实 MT quality。

- **Tokenizer.** 在 mixed-language corpus 上训练的 SentencePiece BPE。跨 languages 共享 vocabulary，是 NLLB 支持 zero-shot pairs 的原因。
- **Model size.** NLLB-200 distilled 600M 能放进 laptop。NLLB-200 3.3B 是公开的 production default。54.5B 是 research ceiling。
- **Decoding.** general content 用 beam width 4-5。用 length penalty 避免过短 output。需要 terminology consistency 时用 constrained decoding。

## 动手实现

### Step 1: pretrained MT call

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

这里有三件事很重要。`src_lang` 告诉 tokenizer 应用哪种 script 和 segmentation。`forced_bos_token_id` 告诉 decoder 生成哪种 language。二者都是 NLLB-specific tricks；mBART 和 M2M-100 使用各自 conventions，不能互换。

### Step 2: BLEU and chrF

BLEU 衡量 output 与 reference 之间的 n-gram overlap。四个 reference n-gram sizes（1-4）、precisions 的 geometric mean，以及用于惩罚过短 output 的 brevity penalty。分数在 [0, 100]。常用。但解释起来很烦：30 BLEU 是 “usable”；40 是 “good”；50 是 “exceptional”；低于 1 BLEU 的差异都是噪声。

chrF 衡量 character-level F-score。对 morphologically rich languages 更敏感，因为 BLEU 会低估匹配。通常与 BLEU 一起报告。

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

始终使用 `sacrebleu`。它会标准化 tokenization，使分数能跨 papers 比较。自己手写 BLEU computation，是产生误导性 benchmarks 的方式。

### 三层 evaluation hierarchy（2026）

现代 MT evaluation 使用三类互补 metrics。上线时至少使用两类。

- **Heuristic**（BLEU、chrF）。快速、reference-based、可解释、对 paraphrase 不敏感。用于 legacy comparison 和 regression detection。
- **Learned**（COMET、BLEURT、BERTScore）。在 human judgment 上训练的 neural models；比较 translation 与 source/reference 的 semantic similarity。自 2023 年以来，COMET 与 MT research 的关联最高，并且是 2026 年质量重要场景中的 production default。
- **LLM-as-judge**（reference-free）。提示大型模型从 fluency、adequacy、tone、cultural appropriateness 角度给 translations 打分。如果 rubric 设计得好，GPT-4-as-judge 与 human agreement 约 80%。用于没有 reference 的 open-ended content。

2026 年实用 stack：`sacrebleu` 用于 BLEU 和 chrF，`unbabel-comet` 用于 COMET，一个 prompted LLM 用于最终 human-facing signal。在 production data 上信任任何 metric 之前，都要用 50-100 个 human-labeled examples 校准它。

Reference-free metrics（COMET-QE、BLEURT-QE、LLM-as-judge）让你能在没有 reference 的情况下评估 translations，这对 reference translations 不存在的 long-tail language pairs 很重要。

### Step 3: production 中会坏掉什么

上面的工作 pipeline 会在 80% 的时间里生成流畅翻译，并在剩下 20% 中静默失败。命名 failure modes：

- **Hallucination.** 模型发明 source 中不存在的内容。常见于陌生 domain vocabulary。症状：output 很流畅，但声称 source 没有陈述的事实。缓解：对 domain terms 做 constrained decoding，对 regulated content 做 human review，监控 output 是否远长于 input。
- **Off-target generation.** 模型翻译成错误 language。NLLB 在 rare language pairs 上出乎意料地容易这样。缓解：验证 `forced_bos_token_id`，并始终用 language-ID model 检查 output。
- **Terminology drift.** “Sign up” 在文档 1 中变成 “s'inscrire”，在文档 2 中变成 “créer un compte”。对 UI text 和 user-facing strings，一致性比 raw quality 更重要。缓解：glossary-constrained decoding 或 post-edit dictionary。
- **Formality mismatch.** French “tu” vs “vous”，Japanese politeness levels。模型会选择 training 中更常见的 form。对 customer-facing content，这通常是错的。缓解：如果模型支持，用 formality token 做 prompt prefix，或在 formal-only corpora 上 fine-tune 小模型。
- **Length explosion on short input.** 很短 input sentences 往往产生过长 translations，因为 length penalty 在低于约 5 个 source tokens 时会突然失效。缓解：按 source length 比例设置 hard max-length cap。

### Step 4: 为 domain fine-tuning

Pretrained models 是 generalists。Legal、medical 或 game-dialog translation 可以从 domain parallel data fine-tuning 中明显获益。recipe 并不 exotic：

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

几千条 high-quality parallel examples 胜过几十万 noisy web-scraped ones。training data quality 是最大的 production lever。

## 实际使用

2026 年 MT production stack：

| Use case | Recommended starting point |
|---------|---------------------------|
| Any-to-any, 200 languages | `facebook/nllb-200-distilled-600M`（laptop）或 `nllb-200-3.3B`（production） |
| English-centric, high quality, 50 languages | `facebook/mbart-large-50-many-to-many-mmt` |
| Short runs, cheap inference, English-French/German/Spanish | Helsinki-NLP / Marian models |
| Latency-critical browser-side | ONNX-quantized Marian（~50 MB） |
| Maximum quality, willing to pay | GPT-4 / Claude / Gemini with translation prompts |

截至 2026 年，LLMs 已经在若干 language pairs 上超过 specialized MT models，尤其是 idiomatic content 和 long context。tradeoff 是 per-token cost 和 latency。当 context length、stylistic consistency 或通过 prompting 做 domain adaptation 比 throughput 更重要时，选择 LLM。

## 交付成果

保存为 `outputs/skill-mt-evaluator.md`：

```markdown
---
name: mt-evaluator
description: Evaluate a machine translation output for shipping.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Given a source text and a candidate translation, output:

1. Automatic score estimate. BLEU and chrF ranges you would expect. State whether a reference is available.
2. Five-point human-verifiable check list: (a) content preservation (no hallucinations), (b) correct language, (c) register / formality match, (d) terminology consistency with glossary if provided, (e) no truncation or length explosion.
3. One domain-specific issue to probe. E.g., for legal: named entities and statute citations. For medical: drug names and dosages. For UI: placeholder variables `{name}`.
4. Confidence flag. "Ship" / "Ship with review" / "Do not ship". Tie to the severity of issues found in step 2.

Refuse to ship a translation without a language-ID check on output. Refuse to evaluate without a reference unless the user explicitly opts in to reference-free scoring (COMET-QE, BLEURT-QE). Flag any content over 1000 tokens as likely needing chunked translation.
```

## 练习

1. **Easy.** 使用 `nllb-200-distilled-600M` 将一个 5-sentence English paragraph 翻译成 French，再翻回 English。测量 round-trip 与 original 有多接近。你应该看到 semantic preservation，但 word-choice drift。
2. **Medium.** 使用 `fasttext lid.176` 或 `langdetect` 实现 translation outputs 的 language-ID check。接入 MT call，在返回前捕获 off-target generations。
3. **Hard.** 在你选择的 5,000-pair domain corpus 上 fine-tune `nllb-200-distilled-600M`。在 held-out set 上测量 fine-tuning 前后的 BLEU。报告哪些 sentence types 改善，哪些 regression。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| BLEU | Translation score | 带 brevity penalty 的 N-gram precision。[0, 100]。 |
| chrF | Character F-score | Character-level F-score。对 morphologically rich languages 更敏感。 |
| NMT | Neural MT | 在 parallel text 上训练的 transformer encoder-decoder。2017+ 默认选择。 |
| NLLB | No Language Left Behind | Meta 的 200-language MT model family。 |
| Constrained decoding | Controlled output | 强制特定 tokens 或 n-grams 出现 / 不出现在 output 中。 |
| Hallucination | Invented content | 不被 source 支持的 model output。 |

## 延伸阅读

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672)——NLLB paper。
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/)——为什么 `sacrebleu` 是报告 BLEU 的唯一正确方式。
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/)——chrF paper。
- [Hugging Face MT guide](https://huggingface.co/docs/transformers/tasks/translation)——实用 fine-tuning walkthrough。
