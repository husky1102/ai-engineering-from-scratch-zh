# 多 Token 预测（MTP）

> 从 GPT-2 到 Llama 3，每个自回归 LLM 都在每个位置上训练一个 loss：预测下一个 token。DeepSeek-V3 在每个位置上加了第二个 loss：预测再后一个 token。额外的 14B 参数（在一个 671B 模型上）通过梯度流蒸馏回主模型，训练好的 MTP heads 又在推理时被改造成 speculative-decoding drafter，acceptance 超过 80%。1.8× 生成吞吐几乎白送。本课会构建 DeepSeek 技术报告中的 sequential MTP module，计算 loss 和 shared-head 参数布局，并解释为什么 MTP 保留了 causal chain，而 Gloeckle et al. 最初的 parallel MTP 打破了它。

**类型:** Build
**语言:** Python (stdlib)
**先修:** Phase 10 · 04 (pre-training a mini GPT), Phase 10 · 15 (speculative decoding)
**时间:** ~60 分钟

## 学习目标

- 说明 MTP 训练目标，并推导跨 prediction depth 的 joint loss。
- 解释 Gloeckle et al. 的 parallel MTP heads（2024）和 DeepSeek-V3 的 sequential MTP modules 之间的区别，以及为什么 sequential design 会保留 causal chain。
- 计算给预训练运行添加 MTP modules 带来的参数和内存开销。
- 从零实现一个 MTP module：shared embedding、per-depth transformer block、projection，以及 shared output head。

## 要解决的问题

Next-token prediction 是标准的 LLM 训练目标。每个 hidden state 都被监督去预测一件事：紧跟在后面的 token。这是一个出人意料地弱的信号。序列中的大部分信息都会延伸到一个 token 之外——结构、连贯性、事实性、算术流。模型必须通过在数万亿 token 上累积大量 one-token signals 来学到这些。

MTP 问的是：如果每个 hidden state 同时被监督去预测多个未来 token，会怎样？Gloeckle et al.（Meta, 2024）展示了这会有帮助。他们的实现是在 backbone 顶部放多个独立 output heads，每个 head 预测不同 offset。并行、简单，但这些 head 看到的是同一个 hidden state，没有任何层级式 refinement——预测之间也没有 causal chaining，因此不能用于 speculative decoding。

DeepSeek-V3（2024 年 12 月）把 MTP 重新设计成 sequential modules，在每个 prediction depth 上保留 causal chain。模型先从 `h_i^(0)` 预测 `t+1`，再从新的 hidden state `h_i^(1)` 预测 `t+2`；这个新状态把 `h_i^(0)` 和 `E(t+1)` embedding 组合起来，依此类推。每个 depth 都有自己的小 transformer block。Shared embedding 和 shared output head 让参数开销保持温和。在 DeepSeek-V3 的规模上，MTP modules 带来 14B 额外参数，叠加在 671B 主模型权重上。这个 2% 开销换来了更密集的训练信号，以及推理时现成的 speculative-decoding draft。

本课会从零构建一个 MTP module 和 D-depth loss。数学很干净。实现约 150 行。

## 核心概念

### Sequential MTP recipe

DeepSeek-V3 在主模型顶部添加 `D` 个 MTP modules。每个 module `k`（`k = 1..D`）预测 depth `k` 的 token，也就是在给定到位置 `i` 的 prefix 时预测 `t_{i+k}`。

Module `k` 包含：

- 一个 transformer block `T_k`，有自己的 attention 和 MLP。
- 一个 projection matrix `M_k`，把上一 depth 的 hidden state 和下一 depth ground-truth token 的 embedding 组合起来。
- Shared embedding `E`（与主模型相同）。
- Shared output head `Out`（与主模型相同）。

训练时，对到位置 `i` 的 prefix，per-depth hidden state 为：

```text
h_i^(0) = main model backbone at position i
h_i^(k) = T_k( M_k * concat(RMSNorm(h_i^(k-1)), RMSNorm(E(t_{i+k}))) )   for k >= 1
```

Per-depth prediction 为：

```text
logits_{i+k} = Out(h_i^(k-1))   for k = 1..D
```

Per-depth loss 是对 ground-truth `t_{i+k}` 的 cross-entropy：

```text
L_k = CE(logits_{i+k}, t_{i+k})
```

跨 depth 的 joint loss：

```text
L_MTP = (lambda / D) * sum_{k=1..D} L_k
```

`lambda` 是一个较小的 weighting factor——DeepSeek-V3 在训练前 10% 使用 0.3，之后使用 0.1。总训练 loss 是 `L_main + L_MTP`。

### 为什么是 sequential，而不是 parallel

Gloeckle 最初的 parallel MTP 有 D 个 output heads，每个都直接应用在 `h_i^(0)` 上。每个 head 都从同一个 backbone hidden state 预测 `t_{i+k}`。这样可以训练，但这些预测并不互相条件化。你不能用 `head_1` 的输出帮助 `head_2`——这些 head 是并行发射的。

DeepSeek-V3 的 sequential design 用 `h_i^(k-1)` 加上真实 next-token embedding `E(t_{i+k})` 来构建 `h_i^(k)`。这保留了 causal chain：要预测 `t_{i+k+1}`，depth `k+1` 的 module 会看到 `t_{i+k}` 上有什么。这在结构上等同于自回归 decoder 消费自己输出的方式——使 MTP modules 可以直接作为 speculative-decoding drafters 使用。

推理时：把 `h_i^(k-1)` 和 draft 出来的 `t_{i+k}` 喂给 module `k+1`，得到对 `t_{i+k+1}` 的预测。重复。它正是一个 EAGLE-style draft，只是把训练好的 MTP module 当成 draft network。DeepSeek-V3 报告第一个 MTP module 的 acceptance 超过 80%，加速约 1.8×。

### 参数核算

对一个 hidden 为 `h`、词表为 `V` 的模型：

- 主模型：数十亿参数，加一个大小为 `V * h` 的 output head。
- Shared output head：复用主模型的 head。无额外参数。
- Shared embedding：复用主模型的 embedding。无额外参数。
- 每个 MTP module：
  - Projection `M_k`：`(2h) * h = 2h^2`。
  - Transformer block `T_k`：attention（MHA 约 `4h^2`）加 MLP（SwiGLU ratio 8/3 时通常约 `8h^2`）。每个 block 约 `12h^2`。

每个 module 总额外参数：`~14h^2`。对 DeepSeek-V3 的 `h = 7168`，D = 1 module：纸面上是 `~14 * 7168^2 = ~720M` 参数。DeepSeek-V3 报告 14B——差异主要来自 MTP module 中的 expert layers 也采用 MoE。

### Speculative-decoding 的回报

预训练期间，MTP modules 会让训练慢约 10%（更多 forward compute，额外 loss）。回报有两方面：

1. 更密集的训练信号。每个 hidden state 看到 D+1 个监督目标。在 DeepSeek-V3 的 ablations 中，MMLU、GSM8K、MATH、HumanEval 上都有一致的几个百分点提升。

2. 推理时免费的 speculative decoding draft。MTP module 已经被训练来预测未来几个 token。改造成 draft network 后，它能达到 80%+ acceptance rates。在这个水平上，N=3 或 N=5 的 spec decoding 会带来 1.8× 吞吐。10% 的训练时成本会在第一次规模化推理时就回本。

### 与 EAGLE 的关系

EAGLE 在预训练后单独训练一个小 draft model。MTP 把 draft 烘进预训练。两种方法在 accept rate 上趋近，但管线不同：

| Dimension | EAGLE-3 | MTP (DeepSeek-V3) |
|-----------|---------|------------------|
| When trained | Post-pre-training | During pre-training |
| Backward-compatible with existing weights | Yes | No (need to re-train) |
| Draft params | 1-2 transformer layers | 1 transformer block + projection |
| Acceptance rate | 0.88-0.92 | 0.80+ at depth 1 |
| Benefit beyond speedup | Speculative decoding only | Denser training signal + speedup |

## 动手实现

`code/main.py` 会端到端构建一个 MTP module：shared embedding、projection、transformer block、shared output head。随后在一段短合成序列上计算 per-depth cross-entropy loss，并按组件打印参数数量。32 个 token 的 toy vocabulary 让数字保持可读。

### Step 1: shared embedding table

一个 `vocab_size x hidden` 表同时被主模型和每个 depth 的每个 MTP module 使用。不是第二份拷贝——字面上就是同一个 tensor。

### Step 2: per-depth combination

```python
def combine(prev_hidden, next_token_embed, M_k):
    # concat along feature dim, then project down to hidden
    concat = rms_norm(prev_hidden) + rms_norm(next_token_embed)  # vector addition stand-in
    projected = matvec(M_k, concat)
    return projected
```

真实 DeepSeek-V3 会把两个 RMSNorm 后的向量 concat 成 `[2h]`，再用一个 `h x 2h` 矩阵投影。这个 toy 为了 stdlib 简洁，用向量加法代替。

### Step 3: depth k 的 transformer block

Self-attention 加 MLP。在 toy 中，一个单层 linear attention block 和一个 SwiGLU MLP 保持结构可见，同时不需要 numpy。

### Step 4: shared output head

复用主模型的 output projection。输出词表上的 logits。

### Step 5: per-depth loss

softmax(logits) 相对 offset `k` 处 ground-truth token 的 cross-entropy。用 `lambda / D` scaling factor 跨 depth 聚合。

### Step 6: 参数核算

打印总参数量、shared（embedding, head）数量，以及每个 module 的额外参数量。展示 MTP extra 相对主模型大小的比例。

## 实际使用

MTP 已集成进 DeepSeek-V3（2024 年 12 月）和 DeepSeek-R1 系列。推理时：

- DeepSeek 自己的 serving stack 可以开箱即用地把 MTP modules 当作 speculative decoders。
- 截至 2026 年 4 月，vLLM 和 SGLang 都有 DeepSeek-V3 MTP 的集成路径。
- AMD 的 ROCm SGLang tutorial 展示了一个具体的 MTP speculative-decoding config，并在 V3 checkpoint 上测得 1.8× speedup。

什么时候在新的预训练运行中使用 MTP：

- 你控制完整预训练管线，并希望提前存入更密集的训练信号。
- 你知道模型将被规模化服务，并希望免费获得 speculative decoding。
- 你的 hidden size 至少是 4096。在 1B 规模，开销比收益更疼。

什么时候不该用：

- Fine-tuning 一个已有的预训练 dense model。MTP module 没有被训练过。
- 研究模型中你想要干净 baseline 做对比。MTP 会改变架构。

## 交付成果

本课产出 `outputs/skill-mtp-planner.md`。给定一个预训练运行规格（model size、data、compute），它会返回一个集成 MTP 的计划：depth 数 D、`lambda` schedule、memory overhead，以及推理时 speculative-decoding 的 wiring。

## 练习

1. 运行 `code/main.py`。展示当合成信号变强时，per-depth loss 单调下降。把合成数据改成固定模式，并验证 depth-1 和 depth-2 losses 都会收敛。

2. 计算一个 dense 70B 模型（hidden 8192，80 layers）使用 D=1 MTP module 的参数开销。与 DeepSeek-V3 报告的 14B overhead 对比。解释为什么 DeepSeek 的数字更高：MTP transformer block 继承了相同的 MoE 结构，从而膨胀了 per-module parameter count。

3. 在 toy 中实现 D=2：添加第二个 MTP module，接收 h^(1) 并预测 `t_{i+2}`。验证 joint loss 和参数核算与 DeepSeek 论文的 equations 19-21 匹配。

4. 把 toy 切换到 parallel MTP（Gloeckle-style）：在主 hidden state 顶部添加 D 个 output heads，每个预测不同 offset。测量同一个合成信号上每个 depth 的 losses 与 sequential version 如何比较。Sequential version 对 k > 1 应该产生更低的 depth-k loss，因为它条件化在中间预测上。

5. 把训练好的 MTP module 当作 EAGLE-style draft：推理时调用 module k 提议 `t_{i+k}`。在 held-out sequence 上测量这些 draft tokens 相对主模型预测的 acceptance rate。如果 toy 上超过 50%，你就复现了 MTP-as-draft 的经验性质。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MTP module | “额外 loss block” | 一个小 transformer block 加 projection，用来预测主模型当前位置前方 `k` 个位置的 token |
| Prediction depth | “哪个 offset” | 整数 `k`，表示 module `k` 根据到位置 `i` 的 prefix 预测 `t_{i+k}` |
| Parallel MTP | “Gloeckle-style” | 同一个 backbone hidden state 上的 D 个独立 heads，没有条件链 |
| Sequential MTP | “DeepSeek-V3 style” | 每个 module 条件化在上一 depth 的 hidden state 加下一个 token 的 embedding 上；保留 causal chain |
| Shared output head | “复用主 head” | MTP modules 调用主模型的 LM head，而不是独立 output projection |
| Shared embedding | “复用主表” | 同一个 vocabulary embedding table 到处使用；没有重复参数 |
| Projection matrix M_k | “合并 hidden + next-token” | 一个 `h x 2h` 线性层，把上一 hidden state 和目标 token embedding 折叠成下一 depth 的输入 |
| Joint loss L_MTP | “平均后的额外 losses” | per-depth cross-entropy losses 的算术均值，并由 `lambda` 缩放 |
| Acceptance rate at depth 1 | “MTP draft 多常正确” | D=1 MTP module 的 top-1 prediction 等于主模型 top-1 prediction 的比例；DeepSeek-V3 上 80%+ |
| Lambda weighting | “额外 loss 重要性” | Per-depth scaling factor；DeepSeek-V3 训练开始用 0.3，之后用 0.1 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — 完整 sequential MTP 描述（Section 2.2），包括 joint-loss equations 和推理时 1.8× speedup
- [Gloeckle et al. — Better & Faster Large Language Models via Multi-token Prediction (arXiv:2404.19737)](https://arxiv.org/abs/2404.19737) — DeepSeek 设计改进的 parallel MTP baseline
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — 685B total（671B main + 14B MTP），部署说明
- [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) — MTP 所适配的 speculative-decoding framework
- [Li et al. — EAGLE-3 (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — EAGLE 的 2025 draft architecture，MTP 竞争的 counterpart
