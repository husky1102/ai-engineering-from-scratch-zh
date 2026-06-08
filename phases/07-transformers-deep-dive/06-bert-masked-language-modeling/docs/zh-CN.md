# BERT — Masked Language Modeling

> GPT 预测下一个词。BERT 预测缺失的词。只差一句话，却塑造了接下来半个十年的所有 embedding 形态。

**类型:** Build
**语言:** Python
**先修:** Phase 7 · 05 (Full Transformer), Phase 5 · 02 (Text Representation)
**时间:** ~45 minutes

## 要解决的问题

2018 年时，每个 NLP 任务——sentiment、NER、QA、entailment——都要在自己的标注数据上从头训练自己的模型。还没有一个预训练好的“理解英语”的 checkpoint 可供 fine-tune。ELMo (2018) 证明了可以用 bidirectional LSTM 预训练 contextual embeddings；它有帮助，但泛化有限。

BERT (Devlin et al. 2018) 问了一个问题：如果我们拿一个 transformer encoder，在互联网上的每个句子上训练它，并强迫它根据左右两侧 context 预测缺失词，会怎样？然后你只需要在 downstream task 上 fine-tune 一个 head。参数效率带来的震撼非常大。

结果是：18 个月内，BERT 及其变体（RoBERTa、ALBERT、ELECTRA）统治了当时存在的每个 NLP leaderboard。到 2020 年，地球上每个搜索引擎、内容审核管线和 semantic-search 系统里都有一个 BERT。

到 2026 年，encoder-only models 仍然是 classification、retrieval 和 structured extraction 的正确工具——它们每 token 运行速度比 decoders 快 5–10×，其 embeddings 是每个现代 retrieval stack 的骨干。ModernBERT（2024 年 12 月）用 Flash Attention + RoPE + GeGLU 把架构推到 8K context。

## 核心概念

![Masked language modeling: pick tokens, mask them, predict originals](../assets/bert-mlm.svg)

### The training signal

取一个句子：`the quick brown fox jumps over the lazy dog`。

随机 mask 15% 的 tokens：

```text
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

训练模型在 masked positions 预测原始 token。因为 encoder 是 bidirectional，在位置 1 预测 `[MASK]` 时可以使用位置 2+ 的 `brown fox jumps`。这就是 GPT 做不到的事情。

### The BERT mask rules

在被选中用于预测的 15% tokens 中：

- 80% 被替换为 `[MASK]`。
- 10% 被替换为随机 token。
- 10% 保持不变。

为什么不总是用 `[MASK]`？因为 `[MASK]` 在 inference time 从不出现。如果训练时 100% masked positions 都是 `[MASK]`，模型会期待看到 `[MASK]`，从而在 pretraining 和 fine-tuning 之间产生 distribution shift。10% random + 10% unchanged 让模型保持诚实。

### Next Sentence Prediction (NSP) — 以及为什么它被丢弃

原始 BERT 还训练 NSP：给定两个句子 A 和 B，预测 B 是否跟在 A 后面。RoBERTa (2019) 做了 ablation，证明 NSP 有害无益。现代 encoders 会跳过它。

### 2026 年变化：ModernBERT

2024 年 ModernBERT 论文用 2026 primitives 重建了 block：

| Component | Original BERT (2018) | ModernBERT (2024) |
|-----------|----------------------|-------------------|
| Positional | Learned absolute | RoPE |
| Activation | GELU | GeGLU |
| Normalization | LayerNorm | Pre-norm RMSNorm |
| Attention | Full dense | Alternating local (128) + global |
| Context length | 512 | 8192 |
| Tokenizer | WordPiece | BPE |

并且不同于 2018 stack，它原生支持 Flash Attention。在 sequence length 8K 时，inference 比 DeBERTa-v3 快 2–3×，GLUE 分数还更好。

### 2026 年仍然选择 encoder 的用例

| Task | Why encoder beats decoder |
|------|---------------------------|
| Retrieval / semantic search embeddings | Bidirectional context = better embedding quality per token |
| Classification (sentiment, intent, toxicity) | One forward pass; no generation overhead |
| NER / token labeling | Per-position output, natively bidirectional |
| Zero-shot entailment (NLI) | Classifier head on top of encoder |
| Reranker for RAG | Cross-encoder scoring, 10x faster than LLM rerankers |

## 动手实现

### Step 1: masking logic

见 `code/main.py`。函数 `create_mlm_batch` 接收 token IDs 列表、vocab size 和 mask probability。返回 input IDs（已应用 masks）和 labels（只有 masked positions 有标签，其余为 -100——PyTorch 的 ignore index convention）。

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: keep original
    return input_ids, labels
```

### Step 2: run MLM prediction on a tiny corpus

在 20 个词的 vocabulary、200 个句子上训练 2-layer encoder + MLM head。不做 gradient——只做 forward-pass sanity checks。完整训练需要 PyTorch。

### Step 3: compare mask types

展示三路规则如何让模型在没有 `[MASK]` 的时候仍然可用。分别在 unmasked sentence 和 masked sentence 上预测。两者都应该产生合理的 token distributions，因为模型在训练中见过两种模式。

### Step 4: fine-tune head

把 MLM head 替换成 toy sentiment dataset 上的 classification head。只训练 head；encoder 冻结。这就是每个 BERT 应用遵循的模式。

## 实际使用

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**Embedding models 是 fine-tuned BERT。** `sentence-transformers` 中的 `all-MiniLM-L6-v2` 这类模型，就是用 contrastive loss 训练的 BERT。Encoder 相同，改变的是 loss。

**Cross-encoder rerankers 也是 fine-tuned BERT。** 它们在 `[CLS] query [SEP] doc [SEP]` 上做 pair-classification。query 和 doc 之间的 bidirectional attention 正是 cross-encoders 相比 biencoders 有质量优势的原因。

**2026 年什么时候不要选 BERT。** 任何 generative 任务。Encoder 没有合理的自回归产 token 方法。还有：任何小于 1B params 且小 decoder 能以更大灵活性匹配质量的场景（Phi-3-Mini、Qwen2-1.5B）。

## 交付成果

见 `outputs/skill-bert-finetuner.md`。这个 skill 会为新的 classification 或 extraction task 界定 BERT fine-tune（backbone choice、head spec、data、eval、stopping）。

## 练习

1. **Easy.** 运行 `code/main.py`，打印 10,000 tokens 上的 mask distribution。确认约 15% 被选中，其中约 80% 变成 `[MASK]`。
2. **Medium.** 实现 whole-word masking：如果一个词被 tokenized 成 subwords，要么 mask 所有 subwords，要么一个都不 mask。在 500-sentence corpus 上测量这是否提升 MLM accuracy。
3. **Hard.** 在来自公共数据集的 10,000 个句子上训练 tiny（2-layer, d=64）BERT。对 SST-2 sentiment fine-tune `[CLS]` token。与参数匹配的 decoder-only baseline 比较——谁赢？

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| MLM | “Masked language modeling” | 训练信号：随机把 15% tokens 替换为 `[MASK]`，预测原始 token。 |
| Bidirectional | “两边都看” | Encoder attention 没有 causal mask——每个位置都能看到其他所有位置。 |
| `[CLS]` | “The pooler token” | 加在每个 sequence 前面的特殊 token；其最终 embedding 用作 sentence-level representation。 |
| `[SEP]` | “Segment separator” | 分隔成对 sequences（例如 query/doc、sentence A/B）。 |
| NSP | “Next sentence prediction” | BERT 的第二个 pretraining task；RoBERTa 证明它没用，2019 年后被丢弃。 |
| Fine-tuning | “适配任务” | 基本冻结 encoder；在上面训练一个小 head 来做 downstream task。 |
| Cross-encoder | “A reranker” | 同时接收 query 和 doc 作为输入、输出 relevance score 的 BERT。 |
| ModernBERT | “2024 refresh” | 用 RoPE、RMSNorm、GeGLU、alternating local/global attention、8K context 重建的 encoder。 |

## 延伸阅读

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — 原始论文。
- [Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — 如何正确训练 BERT；终结 NSP。
- [Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) — replaced-token detection 在匹配 compute 下胜过 MLM。
- [Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) — ModernBERT 论文。
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) — canonical encoder reference。
