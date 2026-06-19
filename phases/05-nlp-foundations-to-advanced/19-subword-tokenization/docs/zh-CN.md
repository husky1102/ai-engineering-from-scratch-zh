# 子词分词：BPE、WordPiece、Unigram、SentencePiece

> Word tokenizers 会被未见词卡住。Character tokenizers 会让序列长度膨胀。Subword tokenizers 取中间路线。每个现代 LLM 都运行在其中一种之上。

**类型：** Learn
**语言：** Python
**先修：** Phase 5 · 01 (Text Processing), Phase 5 · 04 (GloVe / FastText / Subword)
**时间：** ~60 minutes

## 要解决的问题

你的 vocabulary 有 50,000 个词。用户输入 "untokenizable"。你的 tokenizer 返回 `[UNK]`。模型现在对这个词没有任何信号。更糟的是：你语料库中第 90 百分位的文档有 40 个 rare words，这意味着每篇文档丢失 40 bits 信息。

Subword tokenization 解决这个问题。常见词保持单个 tokens。稀有词分解为有意义的片段：`untokenizable` → `un`, `token`, `izable`。训练数据覆盖所有内容，因为任何字符串最终都是 bytes 序列。

2026 年，每个 frontier LLM 都使用三种算法之一（BPE、Unigram、WordPiece），并由三种库之一包装（tiktoken、SentencePiece、HF Tokenizers）。不先选定一种，你无法交付 language model。

## 核心概念

![BPE vs Unigram vs WordPiece, character-by-character](../assets/subword-tokenization.svg)

**BPE (Byte-Pair Encoding)。** 从 character-level vocabulary 开始。统计每个相邻 pair。把最频繁的 pair 合并成一个新 token。重复直到达到目标 vocabulary size。主导算法：GPT-2/3/4、Llama、Gemma、Qwen2、Mistral。

**Byte-level BPE。** 同一个算法，但作用在 raw bytes（256 个 base tokens）上，而不是 Unicode characters。保证零 `[UNK]` tokens，任何 byte sequence 都能编码。GPT-2 使用 50,257 个 tokens（256 bytes + 50,000 merges + 1 special）。

**Unigram。** 从巨大 vocabulary 开始。为每个 token 分配 unigram probability。迭代裁剪那些移除后最小幅度增加 corpus log-likelihood 的 tokens。Inference 时是概率式的：可以 sample tokenizations（对通过 subword regularization 做 data augmentation 很有用）。T5、mBART、ALBERT、XLNet、Gemma 使用它。

**WordPiece。** 合并那些能最大化 training corpus likelihood 的 pairs，而不是 raw frequency。BERT、DistilBERT、ELECTRA 使用它。

**SentencePiece vs tiktoken。** SentencePiece 是直接在 raw Unicode text 上 *训练* vocabularies（BPE 或 Unigram）的库，把 whitespace 编码为 `▁`。tiktoken 是 OpenAI 针对预构建 vocabularies 的快速 *encoder*；它不训练。

经验法则：

- **Training a new vocabulary：** SentencePiece（multilingual，无 pre-tokenization）或 HF Tokenizers。
- **Fast inference against GPT vocab：** tiktoken（cl100k_base、o200k_base）。
- **Both：** HF Tokenizers，一个库同时覆盖 training + serving。

## 动手实现

### Step 1: BPE from scratch

见 `code/main.py`。循环如下：

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

这个算法编码了三个事实。`</w>` 标记词尾，让 "low"（suffix）和 "lower"（prefix）保持不同。Frequency weighting 让高频 pairs 更早胜出。Merge list 是有序的；inference 会按训练顺序应用 merges。

### Step 2: encode with the learned merges

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

朴素实现是 O(n·|merges|)。生产实现（tiktoken、HF Tokenizers）使用 merge-rank lookup 与 priority queues，运行时间接近线性。

### Step 3: SentencePiece in practice

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995, # lower for CJK (e.g. 0.9995 for English, 0.995 for Japanese)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

注意：不需要 pre-tokenization，space 编码为 `▁`，`character_coverage` 控制对 rare characters 的保留程度，或把它们映射到 `<unk>` 的激进程度。

### Step 4: tiktoken for OpenAI-compatible vocabs

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

只做 encoding。速度快（Rust backend）。与 GPT-4/5 tokenization 精确匹配，可用于 byte-counting、cost estimation 和 context-window budgeting。

## Pitfalls that still ship in 2026

- **Tokenizer drift。** 用 vocab A 训练，却用 vocab B 部署。Token IDs 不同，模型输出垃圾。在 CI 中检查 `tokenizer.json` hash。
- **Whitespace ambiguity。** BPE 中 "hello" 与 " hello" 会产生不同 tokens。始终显式指定 `add_special_tokens` 与 `add_prefix_space`。
- **Multilingual undertraining。** English-heavy corpora 会产生把 non-Latin scripts 切成 5-10 倍更多 tokens 的 vocabularies。同一个 prompt 在 GPT-3.5 上用 Japanese/Arabic 会贵 5-10 倍。o200k_base 部分修复了这一点。
- **Emoji splits。** 单个 emoji 可能占 5 个 tokens。做 context budgeting 时要对 emoji handling 做 checkpoint。

## 实际使用

2026 stack：

| Situation | Pick |
|-----------|------|
| Training a monolingual model from scratch | HF Tokenizers (BPE) |
| Training a multilingual model | SentencePiece (Unigram, `character_coverage=0.9995`) |
| Serving an OpenAI-compatible API | tiktoken (`o200k_base` for GPT-4+) |
| Domain-specific vocab (code, math, protein) | 在 domain corpus 上训练 custom BPE，并与 base vocab 合并 |
| Edge inference, small model | Unigram（较小 vocabularies 效果更好） |

Vocabulary size 是 scaling decision，不是常数。粗略启发式：<1B params 用 32k，1-10B 用 50-100k，multilingual/frontier 用 200k+。

## 交付成果

保存为 `outputs/skill-bpe-vs-wordpiece.md`：

```markdown
---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
```

## 练习

1. **Easy.** 在 `code/main.py` 的 tiny corpus 上训练一个 500-merge BPE。编码三个 held-out words。有多少正好生成 1 个 token，有多少生成 >1 个 token？
2. **Medium.** 比较 100 个 English Wikipedia sentences 在 `cl100k_base`、`o200k_base` 和你用 vocab=32k 训练的 SentencePiece BPE 之间的 token counts。报告每一种的 compression ratio。
3. **Hard.** 用 BPE、Unigram 和 WordPiece 在同一个 corpus 上训练。把每一种用于一个小型 sentiment classifier，测量 downstream accuracy。这个选择是否让 F1 变化超过 1 point？

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| BPE | Byte-Pair Encoding | 贪心合并最频繁的 character pairs，直到达到目标 vocab size。 |
| Byte-level BPE | 永远没有 unknown tokens | 在 raw 256 bytes 上做 BPE；GPT-2 / Llama 使用它。 |
| Unigram | 概率式 tokenizer | 使用 log-likelihood 从一个大 candidate set 中裁剪；T5、Gemma 使用。 |
| SentencePiece | 处理 whitespace 的那个 | 在 raw text 上训练 BPE/Unigram 的库；space 编码为 `▁`。 |
| tiktoken | 快的那个 | OpenAI 针对预构建 vocabs 的 Rust-backed BPE encoder。不训练。 |
| Merge list | 魔法数字 | 有序的 `(a, b) → ab` merges 列表；inference 按顺序应用。 |
| Character coverage | 多稀有才算太稀有？ | Tokenizer 必须覆盖的 training corpus 字符比例；~0.9995 很典型。 |

## 延伸阅读

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — BPE 论文。
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) — Unigram 论文。
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) — 这个库。
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) — 简洁参考。
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) — cookbook + encoding list。
