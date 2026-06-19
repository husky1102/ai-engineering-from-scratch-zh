# 序列到序列模型

> 两个 RNNs 假装自己是翻译器。它们撞上的 bottleneck，就是 attention 存在的原因。

**类型:** Build
**语言:** Python
**先修:** Phase 5 · 08 (CNNs + RNNs for Text), Phase 3 · 11 (PyTorch Intro)
**时间:** ~75 minutes

## 要解决的问题

Classification 将 variable-length sequence 映射到单个 label。Translation 将 variable-length sequence 映射到另一个 variable-length sequence。输入和输出可能处在不同 vocabularies，甚至不同 languages 中，而且长度不一定对应。

seq2seq architecture（Sutskever, Vinyals, Le, 2014）用一个刻意简单的 recipe 解决了它。两个 RNNs。一个读取 source sentence 并产生 fixed-size context vector。另一个读取该 vector，并逐 token 生成 target sentence。就是 lesson 08 中写过的同一类代码，只是换了一种连接方式。

它值得学习有两个原因。第一，context-vector bottleneck 是 NLP 中最有教学价值的 failure。它解释了 attention 和 transformers 擅长什么。第二，training recipe（teacher forcing、scheduled sampling、inference 时的 beam search）仍适用于每个现代 generation system，包括 LLMs。

## 核心概念

**Encoder.** 一个读取 source sentence 的 RNN。它的 final hidden state 是 **context vector**——整个 input 的 fixed-size summary。理论上，除了 source 本身，它什么都不该丢。

**Decoder.** 另一个用 context vector 初始化的 RNN。每一步它接收前一个生成 token 作为 input，并产生 target vocabulary 上的 distribution。sample 或 argmax 选择 next token。把它再喂回去。重复直到产生 `<EOS>` token 或达到 max length。

**Training:** 每个 decoder step 上的 cross-entropy loss，沿 sequence 求和。标准 backprop through time 穿过两个 networks。

**Teacher forcing.** 训练期间，decoder 在 step `t` 的 input 是 position `t-1` 的*ground-truth* token，而不是 decoder 自己的前一个 prediction。这会稳定训练；没有它，早期错误会 cascade，模型永远学不会。inference 时，你必须使用模型自己的 predictions，所以 train/inference distribution gap 总是存在。这个 gap 叫 **exposure bias**。

**The bottleneck.** encoder 从 source 学到的一切，都必须被挤进那个 context vector。长句会丢细节。Rare words 会被模糊。Reordering（chat noir vs. black cat）必须被记住，而不是被计算出来。

Attention（lesson 10）通过让 decoder 查看*每一个* encoder hidden state，而不只是最后一个，来修复这一点。这就是完整卖点。

## 动手实现

### Step 1: encoder

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` 的 shape 是 `[batch, seq_len, hidden_dim]`——每个 input position 一个 hidden state。`hidden` 的 shape 是 `[1, batch, hidden_dim]`——final step。Lesson 08 说“classification 时对 outputs 做 pool”。这里我们保留 last hidden state 作为 context vector，并忽略 per-step outputs。

### Step 2: decoder

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

Decoder 一次调用一步。Input：一批 single tokens 和 current hidden state。Output：next token 的 vocabulary logits，以及 updated hidden state。

### Step 3: 使用 teacher forcing 的 training loop

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

有两个值得命名的旋钮。`ignore_index=0` 会跳过 padding tokens 上的 loss。`teacher_forcing_ratio` 是每一步使用 true token 而不是 model prediction 的概率。从 1.0（full teacher forcing）开始，在训练中 anneal 到约 0.5，以缩小 exposure-bias gap。

### Step 4: inference loop（greedy）

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

Greedy decoding 每一步都选择最高概率 token。它可能跑偏：一旦提交某个 token，就不能撤回。**Beam search** 会保留 top-`k` partial sequences，并在最后选择 score 最高的 complete one。Beam width 3-5 是标准选择。

### Step 5: bottleneck 演示

在 toy copy task 上训练模型：source `[a, b, c, d, e]`，target `[a, b, c, d, e]`。增加 sequence length。观察 accuracy。

```text
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

单个 GRU hidden state 无法无损记忆 40-token input。信息存在于每个 encoder step 中，但 decoder 只看到 last state。Attention 会直接修复这个问题。

## 实际使用

PyTorch 有 `nn.Transformer` 和基于 `nn.LSTM` 的 seq2seq templates。Hugging Face 的 `transformers` library 提供训练在数十亿 tokens 上的完整 encoder-decoder models（BART、T5、mBART、NLLB）。

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

现代 encoder-decoders 将 RNNs 替换为 transformers。高层形状（encoder、decoder、逐 token generate）与 2014 seq2seq paper 完全相同。每个 block 内部的机制不同。

### 什么时候还该用 RNN-based seq2seq

对新项目来说，几乎永远不该用。具体例外：

- Streaming translation，需要一次 consume 一个 input token，并保持 bounded memory。
- On-device text generation，其中 transformer memory cost 过高。
- 教学。理解 encoder-decoder bottleneck，是理解 transformers 为什么胜出的最快路径。

### Exposure bias 及其缓解

- **Scheduled sampling.** 训练期间 anneal teacher forcing ratio，让模型学会从自己的错误中恢复。
- **Minimum risk training.** 用 sentence-level BLEU score 训练，而不是 token-level cross-entropy。更接近你真正想要的东西。
- **Reinforcement learning fine-tuning.** 用 metric 奖励 sequence generator。现代 LLM RLHF 中会用到。

三者仍然适用于 transformer-based generation。

## 交付成果

保存为 `outputs/prompt-seq2seq-design.md`：

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## 练习

1. **Easy.** 实现 toy copy task。训练一个 GRU seq2seq，input-output pairs 中 target 等于 source。测量 length 5、10、20 时的 accuracy。复现 bottleneck。
2. **Medium.** 添加 beam width 3 的 beam search decoding。在小 parallel corpus 上相对 greedy 测量 BLEU。记录 beam search 在哪里胜出（通常是最后几个 tokens），哪里没有差异。
3. **Hard.** 在 10k-pair paraphrase dataset 上 fine-tune `facebook/bart-base`。在 held-out inputs 上比较 fine-tuned model 的 beam-4 output 与 base model。报告 BLEU 并挑选 10 个 qualitative examples。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Encoder | Input RNN | 读取 source。产生 per-step hidden states 和 final context vector。 |
| Decoder | Output RNN | 由 context vector 初始化。一次生成一个 target token。 |
| Context vector | The summary | final encoder hidden state。Fixed size。attention 要解决的 bottleneck。 |
| Teacher forcing | Use true tokens | 训练时喂入 ground-truth previous token。稳定学习。 |
| Exposure bias | Train/test gap | 模型在 true tokens 上训练，从未练习过从自己的错误中恢复。 |
| Beam search | Better decoding | 每一步保留 top-k partial sequences，而不是 greedy 地提交。 |

## 延伸阅读

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)——原始 seq2seq paper。四页。
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)——介绍 GRU 和 encoder-decoder framing。
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)——attention paper。学完本课后立刻读。
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)——可构建的 seq2seq + attention 代码。
