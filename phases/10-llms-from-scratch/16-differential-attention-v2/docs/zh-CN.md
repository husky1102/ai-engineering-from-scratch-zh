# 差分注意力（V2）

> Softmax attention 会把少量 probability 分散到每个不匹配 token 上。超过 100k tokens 时，这些噪声会累积并淹没信号。Differential Transformer（Ye et al., ICLR 2025）通过把 attention 计算成两个 softmax 的差来修复它，减去共享 noise floor。DIFF V2（Microsoft，January 2026）是 production-stack rewrite：decode latency 匹配 baseline Transformer，不需要 custom kernels，兼容 FlashAttention。本课从 V1 到 V2 端到端讲解，并给出一个可用 stdlib Python 运行的 difference operation toy implementation。

**类型:** Build
**语言:** Python (stdlib)
**先修:** Phase 7 · 02 (self-attention), Phase 7 · 15 (attention variants), Phase 10 · 14 (architecture walkthrough)
**时间:** ~60 minutes

## 学习目标

- 精确说明为什么 softmax attention 有 noise floor，以及为什么它会随 context length 增长。
- 推导 differential attention formula，并解释为什么 subtraction 会抵消共享 noise component，同时保留 signal。
- 走过 V1-to-V2 diff：哪些变快了、哪些变简单了、哪些变稳定了，以及为什么每个变化都是 production pre-training 必需的。
- 用纯 Python 从零实现 differential attention，并在 synthetic signal-plus-noise query 上经验验证 noise-cancellation property。

## 要解决的问题

标准 softmax attention 有一个数学性质，在规模上会变成运维头疼。对 query `q`，attention weights 是 `softmax(qK^T / sqrt(d))`。Softmax 永远不能产生精确零 -- 每个不匹配 token 都会得到一些正质量。这个 residual mass 是噪声，并且会随 context length 缩放。在 128k tokens 下，即使每个不匹配 token 只得到 0.001% 的 probability，127,999 个合起来也会贡献总量的大约 12%。模型必须学习绕开一个随 context 增长的 noise floor。

经验上，这表现为 attention-head interference：long-context RAG 中 hallucinated citations、100k-token retrieval tasks 上的 lost-in-the-middle failures，以及 needle-in-haystack benchmarks 在 32k 之后的细微 accuracy degradation。Differential Transformer 论文（arXiv:2410.05258, ICLR 2025）测到了差距：DIFF Transformers 相比同尺寸 baselines，有更低 perplexity、更高 long-context accuracy 和更少 hallucinations。

DIFF V1 有三个问题让它进不了 frontier pre-training pipelines。它的 value cache 在每个 decode step 必须加载两次，它需要 custom CUDA kernels 导致 FlashAttention compatibility 破裂，并且它的 per-head RMSNorm 会在 70B-plus scale 的 long-run training 中不稳定。DIFF V2（Microsoft unilm blog，January 20, 2026）修复了全部三点。本课讲解两个版本、构建 difference operator，并在 toy query 上 benchmark noise cancellation。

## 核心概念

### softmax 的 noise floor

对于 query `q` 和 keys `K = [k_1, ..., k_N]`，attention weights 是：

```text
w_i = exp(q . k_i / sqrt(d)) / sum_j exp(q . k_j / sqrt(d))
```

没有任何 `w_i` 会是零。如果 `k_i` 与 `q` 完全无关，score `q . k_i` 也不是 0 -- 它会围绕 0 波动，variance 为 `||q||^2 / d`。经过 softmax normalization 后，每个无关 token 仍然会对 weighted sum 贡献 `O(1/N)`。无关 tokens 的总贡献是 `O((N-1)/N) = O(1)` -- 不是小量。

模型想要的是类似 hard top-k 的东西：matching tokens 上高权重，其他地方接近零。Softmax 太平滑，不能直接做到这一点。

### differential idea

把每个 head 的 Q 和 K projections 分成两个：Q = (Q_1, Q_2)，K = (K_1, K_2)。计算两张 attention maps：

```text
A_1 = softmax(Q_1 K_1^T / sqrt(d))
A_2 = softmax(Q_2 K_2^T / sqrt(d))
```

输出：

```text
DiffAttn = (A_1 - lambda * A_2) V
```

subtraction 会抵消两张 maps 共享的任何 noise distribution。如果两张 maps 在 127k 个无关 tokens 上都有大致 uniform weight（random initialization 时会这样），这些会相互抵消。signal -- 对少数真正相关 tokens 的 peaked weight -- 只有在两张 maps 中以相同 magnitude 出现时才会抵消，而模型一旦训练后不会这样。

`lambda` 是每个 head 的 learnable scalar，parameterized as `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`。它可以为负。`lambda_init` 默认是类似 0.8 的小正数。

### 为什么这像有头的降噪

想象两个嘈杂麦克风记录同一个声音。二者都拾取说话者加 correlated background noise。一个减去另一个，共享噪声就会下降。声音会保留下来，因为两路信号在 phase 或 amplitude 上差异足够大，不会完全抵消。per-head `lambda` 学到的正是这个平衡。

### V1 vs V2：diff

V1 保持 parameter count 等于 baseline Transformer。为了每个 head 得到两个 queries，它把 head dimension 减半。这损害了 head expressiveness，更痛的是，它让每个 head 的 value cache 减半。Decode 每步必须加载 value cache 两次（每个 softmax branch 一次）。结果：虽然 parameter count 匹配，但 decode 比 baseline 更慢。

V2 把 query heads 数量加倍，并保持 KV heads 不变（从 up-projection 借参数）。head dimension 保持与 baseline 相同。subtraction 后，extra dimension 会被 projected back down，以匹配 baseline Transformer 的 O_W projection。三件事同时发生：

1. Decode speed 匹配 baseline（KV cache 只加载一次）。
2. FlashAttention 无需改变即可运行（不需要 custom kernel）。
3. decode 时 arithmetic intensity 上升（每个从 HBM 加载的 byte 对应更多 compute）。

V2 还移除了 V1 用来稳定 subtraction 的 per-head RMSNorm。在 70B-class pre-training scales 下，这个 RMSNorm 会让训练后期不稳定。V2 用更简单的 initialization scheme 替代它，在没有额外 module 的情况下保持 training stable。

### 什么时候使用它

| Workload | Benefit |
|----------|---------|
| Long-context RAG (64k+) | Cleaner attention maps, fewer hallucinated citations |
| Needle-in-haystack benchmarks | Substantial accuracy lift past 32k |
| Multi-document QA | Less cross-document interference |
| Code completion at 8k | Marginal, not worth the architecture change |
| Short chat (< 4k) | Essentially indistinguishable from baseline |

价值会随 context length 增长。4k tokens 时 noise floor 小到 standard attention 足够好。128k 时它已经在伤害你。

### 它如何与其他 2026 knobs 叠加

| Feature | Compatible with DIFF V2? |
|---------|------------------------|
| GQA | Yes (V2 increases Q heads, not KV heads) |
| MLA (DeepSeek) | Yes in principle, no published paper combining them |
| MoE | Yes (attention is independent of MLP block) |
| RoPE | Yes (unchanged) |
| YaRN / long-context scaling | Yes (exactly where DIFF helps most) |
| FlashAttention | Yes in V2 (was no in V1) |
| Speculative decoding | Yes (attention change is invisible to the spec-decode loop) |

## 动手实现

`code/main.py` 用纯 Python 实现 differential attention。一个有已知 signal-plus-noise 结构的 toy query 让你可以直接测量 noise-cancellation ratio。

### Step 1：standard softmax attention

Stdlib matrix ops：lists of lists、manual matmul、带 max subtraction 做 numerical stability 的 softmax。

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### Step 2：把 Q、K 分成两半

V1 style：把 head dimension 减半。V2 style：保留 head dimension，并把 heads 数量加倍。toy implementation 为教学清晰使用 V1 -- 数学相同，只有 bookkeeping 不同。

### Step 3：两个 softmax branches + subtraction

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

注意：output weights 可以为负。这没问题 -- value cache 仍然处理 signed contributions。后续 V projection 会吸收符号。

### Step 4：noise cancellation measurement

构建一个长度为 1024 的 synthetic sequence。把 signal token 放在已知位置，其余填充 noise。计算 (a) standard softmax attention 在 signal position 上的 weight 和 (b) differential attention weight。测量二者的 signal-to-noise ratio。DIFF attention 会稳定地产生更高 signal-to-noise ratio，幅度通常是 3x-10x，取决于两个 branches 被训练到多大程度的差异。

### Step 5：V1 vs V2 parameter accounting

给定 config（hidden=4096, heads=32, d_head=128），打印：

- Baseline Transformer：Q、K、V 各自大小为 `hidden * hidden`，MLP 为 4 * hidden。
- DIFF V1：Q、K 各自大小为 `hidden * hidden`，V 大小为 `hidden * hidden`（不变），内部 head dim 减半。添加 per-head `lambda` parameters（O(heads * d_head)）。
- DIFF V2：Q 大小为 `2 * hidden * hidden`，K 大小为 `hidden * hidden`，V 大小为 `hidden * hidden`。Extra dim 会在 O_W 前 projected back down。添加相同 `lambda` parameters。

toy 会测量 V2 的 extra parameter cost（每个 attention block 大约额外 `hidden * hidden`）并打印它。

## 实际使用

截至 2026 年 4 月，DIFF V2 还没有在每个 production inference server 中发布，但 vLLM 和 SGLang 的集成正在进行。与此同时，这个模式已经出现在：

- Microsoft internal long-context production models。
- 多个面向 256k-plus context 的 open model training runs 的 research replications。
- 在 alternate layers 上结合 DIFF attention 与 sliding-window attention 的 hybrid architectures。

2026 年什么时候该用它：

- 从零训练面向 64k-plus effective context 的新模型。从一开始就加入 differential attention；以后重训会很昂贵。
- fine-tuning 一个 long-context model，且 lost-in-the-middle failures 主导你的 eval。Q projections 上的 LoRA 可以近似 DIFF structure。

什么时候不要用：

- 你在服务一个 long-context performance 稳定的 pre-trained dense model。对现有 weights 来说，retraining cost 很少能回本。
- 你的 context 总是在 16k 以下。Noise floor 可以忽略。

## 交付成果

本课产出 `outputs/skill-diff-attention-integrator.md`。给定 model architecture、target context length、hallucination profile 和 training budget，它会为新 pre-training run 或 LoRA fine-tune 生成添加 differential attention 的 integration plan。

## 练习

1. 运行 `code/main.py`。验证 synthetic query 上 differential attention 报告的 signal-to-noise ratio 高于 standard softmax attention。改变 noise amplitude，展示 standard attention 变得不可用的 crossover point。

2. 为 7B-class model（hidden=4096, heads=32, d_head=128, 32 layers）计算从 baseline 到 DIFF V1、从 baseline 到 DIFF V2 的 parameter-count delta。展示哪些 components 增加了参数，哪些保持不变。

3. 阅读 DIFF V1 paper（arXiv:2410.05258）的 Section 3 和 DIFF V2 Hugging Face blog 的 Section 2。用两句话解释为什么 V1 per-head RMSNorm 是必要的，以及为什么 V2 可以移除它而不导致 training divergence。

4. 实现一个 ablation：用 `lambda = 0`（纯第一个 softmax）和 `lambda = 1`（完全 subtraction）计算 differential attention。在 synthetic query 上，测量 signal-to-noise 如何随 sweep 变化。识别最大化 signal-to-noise 的 `lambda`。

5. 把 toy 扩展到 GQA + DIFF V2。选择 8 KV heads 和 32 Q heads。展示 KV cache size 与相同 (8, 32) configuration 的 baseline GQA model 匹配。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Differential attention | "Two softmaxes minus each other" | 把 Q、K 分成两半，计算两张 softmax maps，从第一张减去第二张（按 lambda 缩放），再乘以 V |
| Noise floor | "The non-zero tail of softmax" | softmax 分配给每个无关 token 的 O(1/N) weight，在长上下文中加总为 O(1) |
| lambda | "The subtraction scale" | 每个 head 的 learnable scalar，parameterized as `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`；可以为负 |
| DIFF V1 | "The ICLR 2025 version" | 原始 Differential Transformer；减半 head dim 以保持 parameter count，需要 custom kernel，decode 更慢 |
| DIFF V2 | "The January 2026 fix" | 在保持 KV heads 的同时把 Q heads 加倍；decode speed 匹配 baseline，并兼容 FlashAttention |
| Per-head RMSNorm | "The V1 stabilizer" | V1 在 difference 后应用的额外 norm；V2 移除它以避免 late-training instability |
| Signal-to-noise ratio | "How much attention is wasted" | true signal position 上的 weight 与 unrelated positions 平均 weight 的比率 |
| Lost in the middle | "Long-context failure mode" | 长上下文中，中间文档的 retrieval accuracy 下降的经验现象 -- DIFF attention 会缓解它 |
| Arithmetic intensity | "FLOPs per byte loaded" | V2 在 decode 时通过每次 KV load 加倍 queries 来提升的比率；对 memory-bound decode 很重要 |

## 延伸阅读

- [Ye et al. — Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) — 原始论文，包含 noise-cancellation theory 和 long-context ablations
- [Microsoft unilm — Differential Transformer V2 (Hugging Face blog, January 2026)](https://huggingface.co/blog/microsoft/diff-attn-v2) — production-stack rewrite，匹配 baseline decode，兼容 FlashAttention
- [Understanding Differential Transformer Unchains Pretrained Self-Attentions (arXiv:2505.16333)](https://arxiv.org/abs/2505.16333) — 关于 subtraction 为什么能恢复 pretrained attention structure 的理论分析
- [Shared DIFF Transformer (arXiv:2501.17900)](https://arxiv.org/html/2501.17900) — parameter-sharing variant
- [Vaswani et al. — Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) — DIFF 所减去的 baseline Transformer
- [Liu et al. — Lost in the Middle (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172) — DIFF attention 面向的 long-context benchmark
