# T5、BART — Encoder-Decoder Models

> Encoders 负责理解。Decoders 负责生成。把它们重新合在一起，就得到一个为 input → output 任务而生的模型：translate、summarize、rewrite、transcribe。

**类型:** Learn
**语言:** Python
**先修:** Phase 7 · 05 (Full Transformer), Phase 7 · 06 (BERT), Phase 7 · 07 (GPT)
**时间:** ~45 minutes

## 要解决的问题

Decoder-only GPT 和 encoder-only BERT 各自为了不同目标精简了 2017 年架构。但很多任务天然就是 input-output：

- Translation: English → French。
- Summarization: 5,000-token article → 200-token summary。
- Speech recognition: audio tokens → text tokens。
- Structured extraction: prose → JSON。

对这些任务，encoder-decoder 是最干净的匹配。Encoder 产生 source 的 dense representation。Decoder 生成输出，并在每一步 cross-attend 到这个 representation。训练是在输出侧做 shift-by-one。Loss 和 GPT 相同，只是条件里多了 encoder output。

两篇论文定义了现代 playbook：

1. **T5** (Raffel et al. 2019)。“Text-to-Text Transfer Transformer。”每个 NLP 任务都重构为 text-in、text-out。同一架构、同一 vocabulary、同一 loss。用 masked span prediction 预训练（corrupt input 中的 spans，在 output 中 decode 它们）。
2. **BART** (Lewis et al. 2019)。“Bidirectional and Auto-Regressive Transformer。”Denoising autoencoder：用多种方式 corrupt input（shuffle、mask、delete、rotate），要求 decoder 重构原文。

到 2026 年，encoder-decoder 格式仍然活在输入结构重要的地方：

- Whisper（speech → text）。
- Google 的 translation stack。
- 一些具有 distinct context-and-edit structures 的 code-completion / repair models。
- Flan-T5 及其用于 structured reasoning tasks 的变体。

Decoder-only 赢得了聚光灯，但 encoder-decoder 从未消失。

## 核心概念

![Encoder-decoder with cross-attention](../assets/encoder-decoder.svg)

### The forward loop

```text
source tokens ─▶ encoder ─▶ (N_src, d_model)  ──┐
                                                 │
target tokens ─▶ decoder block                   │
                 ├─▶ masked self-attention       │
                 ├─▶ cross-attention ◀───────────┘
                 └─▶ FFN
                ↓
              next-token logits
```

关键点：encoder 每个输入只运行一次。Decoder 自回归运行，但每一步都 cross-attend 到*同一个* encoder output。缓存 encoder output 对长输入来说是免费的加速。

### T5 pretraining — span corruption

随机选择输入 spans（平均长度 3 tokens，总计 15%）。把每个 span 替换为唯一 sentinel：`<extra_id_0>`、`<extra_id_1>` 等。Decoder 只输出被 corrupt 的 spans，并带上对应 sentinel prefix：

```text
source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy
```

这个信号比预测整个 sequence 更便宜。在 T5 论文的 ablation 中，它能与 MLM（BERT）和 prefix-LM（UniLM）竞争。

### BART pretraining — multi-noise denoising

BART 尝试五种 noising functions：

1. Token masking。
2. Token deletion。
3. Text infilling（mask 一个 span，decoder 插入正确长度）。
4. Sentence permutation。
5. Document rotation。

Text infilling + sentence permutation 的组合产生了最佳 downstream numbers。Decoder 总是重构原文。BART 的输出是完整 sequence，而不只是被 corrupt 的 spans——所以 pretraining compute 高于 T5。

### Inference

与 GPT 相同的 autoregressive generation。Greedy / beam / top-p sampling 都适用。Beam search（width 4–5）是 translation 和 summarization 的标准选择，因为输出分布比 chat 更窄。

### 2026 年何时选择各变体

| Task | Encoder-decoder? | Why |
|------|------------------|-----|
| Translation | Yes, usually | Clear source sequence; fixed output distribution; beam search works |
| Speech-to-text | Yes (Whisper) | Input modality differs from output; encoder shapes audio features |
| Chat / reasoning | No, decoder-only | No persistent "input" — the conversation is the sequence |
| Code completion | Usually no | Decoder-only with long context wins; code models like Qwen 2.5 Coder are decoder-only |
| Summarization | Either works | BART, PEGASUS beat earlier decoder-only baselines; modern decoder-only LLMs match them |
| Structured extraction | Either | T5 is clean because "text → text" absorbs any output format |

自约 2022 年以来的趋势是：decoder-only 接管了过去 encoder-decoder 拥有的任务，因为 (a) instruction-tuned decoder-only LLMs 可以通过 prompting 泛化到任何任务，(b) 一种架构比两种更容易扩展，(c) RLHF 假设有 decoder。Encoder-decoder 仍保留在输入模态不同（speech、images）或 beam search 质量重要的地方。

## 动手实现

见 `code/main.py`。我们为 toy corpus 实现 T5-style span corruption——这是本课最有用的单个组件，因为它出现在此后几乎所有 encoder-decoder pretraining recipe 中。

### Step 1: span corruption

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """Pick spans summing to ~mask_rate of tokens. Return (corrupted_input, target)."""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

Target format 是 T5 约定：`<sent0> span0 <sent1> span1 ...`。Corrupted input 会在 span 位置把 unchanged tokens 与 sentinel tokens 交织起来。

### Step 2: verify round-trip

给定 corrupted input 和 target，重构原句。如果你的 corruption 可逆，forward pass 就定义良好。这是 sanity check——真实训练不会这么做，但测试很便宜，而且能抓住 span bookkeeping 的 off-by-one bugs。

### Step 3: BART noising

五个函数：`token_mask`、`token_delete`、`text_infill`、`sentence_permute`、`document_rotate`。组合其中两个并展示结果。

## 实际使用

HuggingFace reference：

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

T5 trick：任务名进入 input text。同一个模型能处理几十个任务，因为每个任务都是 text-in、text-out。到 2026 年，instruction-tuned decoder-only models 已经泛化了这一模式，但 T5 最先把它 codify。

## 交付成果

见 `outputs/skill-seq2seq-picker.md`。这个 skill 会根据 input-output structure、latency 和 quality targets，在 encoder-decoder 与 decoder-only 之间为新任务做选择。

## 练习

1. **Easy.** 运行 `code/main.py`，对 30-token sentence 应用 span corruption，验证把 non-sentinel source tokens 与 decoded target spans 拼接起来能复原原文。
2. **Medium.** 实现 BART 的 `text_infill` noise：用单个 `<mask>` token 替换 random spans，decoder 必须推断正确的 span length 和 contents。展示一个例子。
3. **Hard.** 在 tiny English → pig-Latin corpus（200 pairs）上 fine-tune `flan-t5-small`。在 held-out 50-pair set 上测量 BLEU。与在相同数据和相同 compute 下 fine-tune `Llama-3.2-1B` 比较。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Encoder-decoder | “Seq2seq transformer” | 两个 stacks：面向输入的 bidirectional encoder，以及带 cross-attention 的 causal decoder。 |
| Cross-attention | “source 和 target 对话的地方” | Decoder 的 Q × encoder 的 K/V。Encoder 信息进入 decoder 的唯一位置。 |
| Span corruption | “T5 的 pretraining trick” | 用 sentinel tokens 替换 random spans；decoder 输出这些 spans。 |
| Denoising objective | “BART 的游戏” | 对 input 应用 noise function，训练 decoder 重构 clean sequence。 |
| Sentinel token | “`<extra_id_N>` placeholder” | 在 source 中标记 corrupted spans、并在 target 中重新标记它们的特殊 token。 |
| Flan | “Instruction-tuned T5” | 在 >1,800 tasks 上 fine-tuned 的 T5；让 encoder-decoder 在 instruction-following 上保持竞争力。 |
| Beam search | “Decoding strategy” | 每一步保留 top-k partial sequences；translation/summarization 的标准做法。 |
| Teacher forcing | “Training-time input” | 训练时向 decoder 喂真实 previous output token，而不是 sampled token。 |

## 延伸阅读

- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — T5。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) — BART。
- [Chung et al. (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — Flan-T5。
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper，2026 年 canonical encoder-decoder。
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) — reference implementation。
