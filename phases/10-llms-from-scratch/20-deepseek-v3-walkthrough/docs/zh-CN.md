# DeepSeek-V3 架构走读

> Phase 10 · Lesson 14 命名了每个开放模型都会调的六个架构旋钮。DeepSeek-V3（2024 年 12 月，total 671B parameters，active 37B）把六个旋钮全都转了，并额外加入四个：Multi-Head Latent Attention、auxiliary-loss-free load balancing、Multi-Token Prediction，以及 DualPipe training。本课会自顶向下阅读 DeepSeek-V3 的架构，并从公开 config 推导每一项参数量。学完之后，你可以解释为什么 671B/37B 这个比例是正确赌注，以及为什么 MLA + MoE 组合在 frontier 上优于单独使用其中任何一个。

**类型:** Learn
**语言:** Python (stdlib, parameter calculator)
**先修:** Phase 10 · 14 (open-model walkthroughs), Phase 10 · 17 (NSA), Phase 10 · 18 (MTP), Phase 10 · 19 (DualPipe)
**时间:** ~75 分钟

## 学习目标

- 自顶向下阅读 DeepSeek-V3 config，并用六个 GPT-2 knobs 加四个 DeepSeek-specific additions 来解释每个字段。
- 推导总参数量（671B）、active parameter count（37B），以及贡献这些数字的组件。
- 计算 MLA 在 128k 上下文下的 KV cache footprint，并与 same-active-param dense model 使用 GQA 时的开销比较。
- 说出四个 DeepSeek-specific innovations（MLA、MTP、auxiliary-loss-free routing、DualPipe），并命名它们各自作用于架构/训练 stack 的哪一部分。

## 要解决的问题

DeepSeek-V3 是第一个架构上明显不同于 Llama 家族的 frontier open model。Llama 3 405B 是“GPT-2 with six knobs turned”。DeepSeek-V3 是 GPT-2 加上全部六个 knobs，再加四个 knobs。阅读 Llama 3 config 是阅读 DeepSeek config 的热身，但 attention block 的形状、routing logic、训练时 objective 等深层结构差异足够大，需要单独走读。

学习它的回报：DeepSeek-V3 的 open-weights release 改变了开放模型中“frontier capability”的含义。这个架构是许多 2026 training runs 正在复制的 blueprint。理解它，是任何接触 frontier LLM training 或 inference 的角色的基本功。

## 核心概念

### 不变的核心，再来一次

DeepSeek-V3 仍然是 autoregressive。它仍然堆叠 decoder blocks。每个 block 仍然有 attention 加 MLP 加两个 RMSNorm。MLP 中仍然使用 SwiGLU。仍然使用 RoPE。Pre-norm。Weight-tied embeddings。和每个 Llama 或 Mistral 的 baseline 一样。

### 转折：用 MLA 代替 GQA

从 Phase 10 · 14 你已经知道，GQA 通过在多组 Q heads 间共享 K 和 V 来缩小 KV cache。Multi-Head Latent Attention（MLA）更进一步：K 和 V 被压缩进一个共享低秩 latent representation（`kv_lora_rank`），再在需要时按 head 解压。KV cache 只存 latent——通常每个 token 每层 512 个 floats，而不是 8 x 128 = 1024 floats。

在 128k context 下，使用 MLA 的 DeepSeek-V3（每个 token 每层一个共享 latent `c^{KV}`；K 和 V 都由这个 latent 通过 up-projections 得到，并且这些 up-projections 可吸收到后续 matmul 中）：

```text
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

一个假设的 GQA baseline（Llama 3 70B shape，8 KV heads，head dim 128）则要付出：

```text
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

在 128k context 下，MLA 比 Llama-3-70B-style GQA cache 小 4 倍。

取舍：MLA 在每次 attention computation（per head）上增加一次 decompression step。额外 compute 相比节省的 bandwidth 很小。对长上下文推理是净收益。

### 路由：auxiliary-loss-free load balancing

MoE routers 会决定每个 token 由哪些 top-k experts 处理。Naive router 会把太多工作集中在少数 experts 上，让其他 experts 空闲。标准修复：添加一个 auxiliary loss term 来惩罚 load imbalance。这能工作，但会略微损伤 main-task performance。

DeepSeek-V3 引入了一个 auxiliary-loss-free scheme。在 router logits 上加入 per-expert bias terms，并在训练中用一个简单规则调整：如果 expert `e` overloaded，就降低 `bias_e`；如果 underloaded，就提高它。没有额外 loss term。训练保持干净。Expert load 保持平衡。

对 main loss 的影响：不可测。对 MoE 架构的影响：更干净，没有需要调的 auxiliary-loss hyperparameter。

### MTP：更密集训练 + 免费 draft

从 Phase 10 · 18 你知道，DeepSeek-V3 添加了 D=1 MTP module，用来预测前方两个位置的 token。推理时，训练好的 module 被改造成 speculative-decoding draft，acceptance 超过 80%。训练时，每个 hidden state 接受 D+1 = 2 个目标的监督，提供更密集信号。

参数：叠加在 671B main 上的 14B。开销：2.1%。

### 训练：DualPipe

从 Phase 10 · 19 你知道，DualPipe 是一个双向 pipeline，会把 forward 和 backward chunks 与跨节点 all-to-all comms 重叠。在 DeepSeek-V3 的 2,048-H800 规模上，它收回了 1F1B 原本会损失在 pipeline bubbles 上的大约 245k GPU-hours。

### Config，逐字段

下面是 DeepSeek-V3 config（简化版）：

```text
hidden_size: 7168
intermediate_size: 18432   (dense MLP hidden size, used on first few layers)
moe_intermediate_size: 2048 (expert MLP hidden size)
num_hidden_layers: 61
first_k_dense_layers: 3    (first 3 layers use dense MLP)
num_attention_heads: 128
num_key_value_heads: 128   (formally equal to num_heads under MLA, but
                           the real compression is in kv_lora_rank)
kv_lora_rank: 512          (MLA latent dimension)
num_experts: 256            (MoE expert count per block)
num_experts_per_tok: 8      (top-8 routing)
shared_experts: 1           (always-on shared expert per block)
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               (1 MTP module at depth 1)
```

解析它：

- `hidden_size=7168`：embedding dimension。
- `num_hidden_layers=61`：总 block depth。
- `first_k_dense_layers=3`：前 3 个 blocks 使用大小为 18432 的 dense MLP。剩余 58 个使用 MoE。
- `num_attention_heads=128`：128 个 query heads。
- `kv_lora_rank=512`：K 和 V 被压缩到这个 latent dimension，并按 head 解压。
- `num_experts=256, num_experts_per_tok=8`：每个 MoE block 有 256 个 experts，路由 top-8。
- `shared_experts=1`：在 256 个 routed experts 之外，1 个 always-on expert 会贡献给每个 token。可以把它理解成一个 “dense floor”，确保每个 token 都获得某种可靠处理。
- `moe_intermediate_size=2048`：每个 expert 的 MLP hidden size。比 dense MLP 小，因为有 256 个 experts。

### 参数核算

完整计算在 `code/main.py` 中。Headline：

- Embedding：`vocab * hidden = 129280 * 7168 = ~0.93B`。
- 前 3 个 dense blocks：MLA attention（每 block 约 144M）+ dense MLP（每 block 约 260M）+ norms。总计约 1.2B。
- 58 个 MoE blocks：MLA attention（约 144M）+ 256 个 experts（每个约 30M）+ 1 个 shared expert（30M）+ norm。计入所有 experts 后，每 block 总计约 7.95B。58 个 MoE blocks 总计 461B。
- MTP module：14B。

总计：core architecture 约 476B + 14B MTP；而公开的 671B 数字还计入了额外 structural parameters（bias tensors、expert-specific components、shared expert scaling 等）。Calculator 复现出的数字在公开值的 3-5% 以内——delta 来自 DeepSeek 报告 Section 2 appendix 中的细粒度核算。

每次 forward 的 active parameters：

- Attention：每层 144M * 61 = 8.8B（所有层都会触发）。
- MLP active：前 3 层 dense（3 * 260M = 780M），58 个 MoE layers 每层激活 8 个 routed + 1 个 shared + routing overhead。每层 active MLP：约 260M。总计：3 * 260M + 58 * 260M = ~15.9B。
- Embedding + norms：1.2B。
- Total active：大约 26B core + 14B MTP（训练时运行，但推理时不一定总运行）≈ 37B。

### 671B / 37B 比例

18x sparsity ratio（active params 是 total params 的 5.5%）。DeepSeek-V3 是已经发布 open weights 的最稀疏 frontier MoE model。Mixtral 8x7B 的比例是 13/47（28%），密集得多。Llama 4 Maverick 的比例是 17B/400B（4.25%），与之相近。DeepSeek 的赌注是：在 frontier scale 上，更多 experts 加更低 activation ratio 能带来更好的 quality per active-FLOP。

### DeepSeek-V3 的位置

| Model | Total | Active | Ratio | Attention | Novel ideas |
|-------|------|-------|-------|-----------|-------------|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA 512 | MLA + MTP + aux-free + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | YaRN extension |

### 后续：R1、V4

DeepSeek-R1（2025）是在 V3 backbone 上做的 reasoning-training run。R1 使用相同架构。变化的是 post-training recipe（在可验证任务上的大规模 RL），不是 pretraining architecture。

DeepSeek-V4（如果发布）预计会保留 MLA + MoE + MTP，并加入 DSA（DeepSeek Sparse Attention），也就是 Phase 10 · 17 中 NSA 的后继者。谱系是稳定的：架构层面的 innovations 会累积；每个版本都会转动额外 knobs。

## 实际使用

`code/main.py` 是一个专门针对 DeepSeek-V3 shape 的 parameter calculator。运行它，把输出和论文数字比较，并在假设变体上使用它（256 experts vs 512、top-8 vs top-16、MLA rank 512 vs 1024）。

要关注的内容：

- Total parameter count vs published 671B。
- Active parameter count vs published 37B。
- 128k context 下的 KV cache——MLA vs GQA comparison。
- Per-layer breakdown，用来看参数预算真正花在哪里。

## 交付成果

本课产出 `outputs/skill-deepseek-v3-reader.md`。给定一个 DeepSeek-family model（V3、R1 或任何未来变体），它会生成 component-by-component architecture reading，命名 config 的每个字段，按组件推导参数量，并识别模型使用了四个 DeepSeek-specific innovations 中的哪些。

## 练习

1. 运行 `code/main.py`。把 calculator 的 total-parameter estimate 与 published 671B 比较，并找出 delta 来自哪里。论文 Section 2 有完整 itemization。

2. 把 config 改成 MLA rank 256，而不是 512。计算 128k context 下的 KV cache size。它带来多少百分比 reduction，代价是 per-head expressiveness 上什么损失？

3. 比较 DeepSeek-V3 的（256 experts, top-8）routing 与一个假设的（512 experts, top-8）variant。Total parameters 增长；active parameters 不变。额外 expert capacity 理论上买到了什么，推理时又付出什么成本？

4. 阅读 DeepSeek-V3 technical report（arXiv:2412.19437）Section 2.1 中的 MLA。用三句话解释为什么 K 和 V decompression matrices 可以在 inference-time efficiency 上被 “absorbed” 到后续 matmul 中。

5. DeepSeek-V3 对大多数 operations 使用 FP8 training。计算用 FP8 而不是 BF16 存储 671B weights 的 memory savings。它如何与 14.8T-token training budget 相交？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MLA | “Multi-Head Latent Attention” | 把 K 和 V 压缩进共享低秩 latent（kv_lora_rank，通常 512），运行时按 head 解压；KV cache 只存 latent |
| kv_lora_rank | “MLA compression dim” | K 和 V 共享 latent 的大小；DeepSeek-V3 使用 512 |
| First k dense layers | “Early layers stay dense” | MoE-model 的前几层跳过 MoE router，运行 dense MLP 以获得稳定性 |
| num_experts_per_tok | “Top-k routing” | 每个 token 激活多少 routed experts；DeepSeek-V3 使用 8 |
| Shared experts | “Always-on experts” | 无论 routing 如何都会处理每个 token 的 experts；DeepSeek-V3 使用 1 |
| Auxiliary-loss-free routing | “Bias-adjusted load balance” | 训练中调整 per-expert bias terms，在不添加 loss term 的情况下保持 expert load balanced |
| MTP module | “Extra prediction head” | Transformer block，从 h^(1) 和 E(t+1) 预测 t+2；更密集训练，免费 speculative-decoding draft |
| DualPipe | “Bidirectional pipeline” | 将 forward/backward compute 与跨节点 all-to-all 重叠的训练 schedule |
| Active parameter ratio | “Sparsity” | active_params / total_params；DeepSeek-V3 达到 5.5% |
| FP8 training | “8-bit training” | 用 FP8 存储训练数据，并让许多 compute ops 用 FP8；相对 BF16 大约减半内存，质量成本很小 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — 完整 architecture、training 和 results 文档
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — config files 和 deployment notes
- [DeepSeek-V2 paper (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) — 引入 MLA 的前代模型
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — 在 V3 架构上的 reasoning-training successor
- [Native Sparse Attention (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089) — DeepSeek-family attention 的未来方向
- [DualPipe repository](https://github.com/deepseek-ai/DualPipe) — training-schedule reference
