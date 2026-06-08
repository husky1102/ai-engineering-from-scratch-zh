# Speculative Decoding 与 EAGLE-3

> Phase 7 · Lesson 16 证明了数学：Leviathan rejection rule 会精确保留 verifier 的 distribution。本课是 2026 年生产 speculative decoding 的 training-stack 视角。EAGLE-3 把 draft model 从廉价近似变成了一个专门构建的小网络，在 verifier 自己的 hidden states 上训练，然后加入 training-time test loop，对齐 train 和 inference distributions。结果：端到端 3× 到 6.5× speedup，chat 上 accepted per-token rates 超过 0.9，没有 distributional tradeoff。2026 年每个生产 inference stack 都默认发布它。

**类型:** Build
**语言:** Python (stdlib)
**先修:** Phase 7 · 16 (speculative decoding math), Phase 10 · 12 (inference optimization)
**时间:** ~75 minutes

## 学习目标

- 用一句话陈述 Leviathan theorem，并证明 speculative loop 产生的 samples 与 verifier 分布完全相同。
- 走过从 vanilla spec-decoding（Leviathan 2023）到 EAGLE、EAGLE-2、EAGLE-3 的两年演进，并说出每一步移除了哪个精确限制。
- 根据 acceptance rate `α` 和 draft-to-verifier cost ratio `c` 计算 expected speedup，并为每种 regime 选择最优 draft length `N`。
- 从零实现完整 speculative loop：draft、verify、从 residual reject-sample、rejection 时回滚 KV cache、full acceptance 时发出 bonus token。

## 要解决的问题

在 70B 模型上做 autoregressive decoding，一张 H100 也许只有 35 tokens per second。GPU 远远没有饱和。Memory bandwidth 是天花板：每个 token 都要从 HBM 加载 70B weights，做一步算术，然后产生一个 float。compute units 大部分时间都闲着。

Speculative decoding 把这件事变成你真的能解决的 throughput problem。一个廉价 draft 用 `N` 次小 forward passes 提出 `N` 个 tokens。verifier 在 prefix 加全部 `N` 个 drafts 上运行一次。如果 verifier 在位置 `i` 的 distribution 与 draft 一致（以我们会精确定义的统计意义），我们接受；否则拒绝，并从 residual distribution 采样修正。一次大模型 forward 可以产生最多 `N+1` 个 accepted tokens，而不是一个。

真正重要的 theorem 是 Leviathan, Kalman, Matias (ICML 2023)：output distribution 与直接从 verifier 采样产生的结果完全相同。不是近似。是完全相同。这是 speculative decoding 可以进入生产的全部原因 -- 它是纯 latency optimization，没有质量 tradeoff。

Phase 7 · Lesson 16 给你的是数学。本课给你的是 training stack。一个好 draft 带来的 speedup 比 cheap draft 多 2×。EAGLE、EAGLE-2 和 EAGLE-3（Li et al., 2024-2025）把 "draft = 同家族更小模型" 变成了一门精确工程纪律。2026 年生产 inference servers 默认使用 EAGLE-3。

## 核心概念

### 不变量：Leviathan rejection sampling

令 `p(t)` 为给定某个 prefix 时 draft 对 next token 的 distribution，`q(t)` 为 verifier 的 distribution。采样一个 draft token `d ~ p`。以 `min(1, q(d) / p(d))` 的概率接受。若 reject，则从 residual distribution `(q - p)_+ / ||(q - p)_+||_1` 采样。得到的 samples 按 `q` 分布。这与 `p` 有多差无关 -- 越差就越常 reject，但 output 仍然精确。

把 `N` 个这样的调用背靠背堆起来，对 `prefix + d_1 + ... + d_N` 做一次 verifier forward pass。verifier 同时返回 `q_1, q_2, ..., q_{N+1}`。从左到右走。第一次在位置 `j` reject 时，从 `residual(q_j, p_j)` 采样并停止。全部接受时，从 `q_{N+1}` 采样一个 bonus token。

### 什么决定 speedup

令 `α` 为每个 drafted token 的 expected acceptance rate。令 `c = cost(draft) / cost(verifier)` 为 cost ratio。每次 verifier forward 的 expected accepted tokens 数量是：

```text
E[accepted] = (1 - α^(N+1)) / (1 - α)
```

每个 accepted token 的 expected total wall time 是 `(N * c + 1) / E[accepted]`。对 `N` 最小化，你就得到 sweet spot。对 `α = 0.8, c = 0.05`：最优 `N` 大约是 5-7，speedup 是 3.2×。对 `α = 0.95, c = 0.02`：最优 `N` 大约是 8-10，speedup 推到 5×。

最大的杠杆是 `α`。在固定 `N = 5` 时，从 `α = 0.6`（vanilla draft）到 `α = 0.9`（EAGLE-3），每次 verifier forward 的 expected accepted tokens 会从 2.2 变成 4.1。同一个 verifier 获得几乎 2× throughput。

### 两年演进

**Vanilla speculative (Leviathan, 2023).** Draft model 是同一家族中独立训练的更小 LLM。容易接线，`α ≈ 0.6`，speedup 最多约 2×。

**EAGLE-1 (Li et al., 2024).** Draft 是一个 tiny transformer -- 通常一两层 -- 它以 verifier 的 last-layer hidden state 为输入，并直接预测 next token。因为 draft 看到了 verifier 的 feature representation，它的 distribution 更接近 verifier。`α` 上升到 0.7-0.8。

**EAGLE-2 (Li et al., 2024).** 添加 dynamic draft tree：不是提出单条 `N` tokens 序列，而是提出一个小 candidate tree，用 verifier 在一次 forward pass（tree attention）中为每个 candidate 打分，并沿 highest-probability path 走。Draft length 变成每步自适应。每个 accepted-path token 的 `α` 上升到 0.85 以上。

**EAGLE-3 (Li et al., 2025, NeurIPS).** 又做了两个改变。第一，完全去掉 feature-prediction loss -- EAGLE-1/2 训练 draft 去匹配 verifier 的 hidden states，这会限制更多数据带来的收益。EAGLE-3 直接在 token prediction 上训练。第二，training-time test (TTT)：在 draft 训练期间，把 draft 自己之前的 predictions 多步反馈为 inputs，就像 inference 时那样。这对齐了 train 和 test distributions，并阻止 error accumulation。测得 speedup：chat 上最高 6.5×，SGLang 在 H100 上 batch 64 时 throughput improvement 38%。

### KV cache rollback

Verification 会在一次 pass 中把 verifier 的 KV cache 扩展 `N` 个 entries。如果 rejection 发生在位置 `j`，那么位置 `j-1` 之后的 cache contents 已经错了。两种常见实现：写入 scratch buffer 并在 acceptance 时 commit（vLLM、TensorRT-LLM），或保留 physical KV cache 加 logical length，并在 reject 时 truncate。无论哪种方式，rollback cost 都是每层每 head 的 bytes，相比 forward-pass cost 可忽略。

对 EAGLE-2 tree search，verifier 用尊重 tree topology 的 non-causal mask 运行 attention。工程上有点细碎，但计算本身就是一个带 custom mask 的标准 flash-attention call。

### 2026 年的 Draft architectures

| Strategy | Draft type | `α` | Speedup | Training cost |
|----------|-----------|-----|---------|---------------|
| Vanilla | Separate small LLM | 0.55-0.70 | 1.8-2.3× | None (reuse existing small model) |
| Medusa | Extra LM heads on verifier | 0.65-0.75 | 2-3× | ~1B SFT tokens |
| EAGLE-1 | 1-layer transformer on hidden states | 0.70-0.80 | 2.5-3× | ~60B tokens |
| EAGLE-2 | EAGLE-1 + dynamic draft tree | 0.80-0.88 | 3-4× | ~60B tokens |
| EAGLE-3 | Multi-layer feature fusion + TTT | 0.88-0.92 | 3.5-6.5× | ~60-200B tokens |
| Lookahead | No draft (Jacobi iteration) | N/A | 1.3-1.6× | None |

2026 年生产中：vLLM 和 SGLang 在可用时默认使用 EAGLE-3，否则使用 EAGLE-2。TensorRT-LLM 为 Meta 和 NVIDIA public models 提供最快 Medusa path。llama.cpp 为 CPU deployments 发布 vanilla draft。

## 动手实现

见 `code/main.py`。这是完整的 Leviathan speculative loop，包含所有部分：draft-of-N、verifier parallel pass、per-position rejection、residual sampling、bonus token、KV rollback，以及经验验证 output distribution 匹配直接从 `q` 采样。

### Step 1：rejection rule

```python
def accept(q_prob, p_prob, u):
    if p_prob <= 0:
        return True
    return u < min(1.0, q_prob / p_prob)
```

### Step 2：residual distribution

```python
def residual(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0:
        return list(q)
    return [r / s for r in raw]
```

### Step 3：完整 speculative step

`spec_step` function 会从 `p` draft `N` 个 tokens，然后在一次并行 `q` evaluation 中全部验证。对每个 drafted token，它应用 rejection rule；第一次 reject 时，它从 residual 采样 correction。如果全部接受，它从 `q_{N+1}` 发出一个 bonus token。

### Step 4：KV rollback bookkeeping

simulator 为每个 worker 跟踪 logical `kv_length`。接受 `k` 个 drafts 时，`kv_length += k`。如果在位置 `j` reject，cache 已经写过了 `j`，但 logical length 会设置为 `prefix_length + j + 1` -- 即 correction token 之后一位。后续 reads 会 truncate 到 logical length。

### Step 5：Leviathan check

运行 50,000 个 speculative steps。统计 accepted tokens 的 empirical distribution。与 50,000 个来自 `q` 的 direct samples 比较。chi-square statistic 应该远低于 critical value。theorem 在实践中通过。

### Step 6：speedup vs. α

通过用不同 amplitudes 把 `p` 从 `q` 扰开来 sweep draft quality。测量 `α`，然后绘制 expected tokens per verifier call 随 `α` 和 `N` 变化的函数。代码会打印一张表，展示 EAGLE-3-class draft quality（`α ≈ 0.9`）如何解锁每次 verifier call 4-5 个 tokens。

## 实际使用

使用 EAGLE-3 的 production-level `vllm serve`：

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --speculative-config '{
    "model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "num_speculative_tokens": 5,
    "method": "eagle3"
  }'
```

根据 EAGLE-3 论文，SGLang 在 H100 上 batch 64 时使用 EAGLE-3，相比 batch-64 vanilla decoding，throughput 大约多 1.38×。

什么时候该用 speculative decoding：

- 任何 p50 latency 比 peak throughput 更重要的 interactive chat workload。
- Code generation 和 structured output（JSON、SQL）。因为 target distribution 高度可预测，`α` 高于 0.9。
- Long-form generation（数千 tokens）。amortized speedup 会持续发挥作用。

什么时候不要用：

- 非常小的模型（< 3B）。draft 并不比 verifier 便宜多少。
- 极小 batch-1 CPU deployments。draft model 的 memory overhead 可能不值得。
- `α` 会崩掉的 very-high-temperature creative sampling。

## 交付成果

本课产出 `outputs/skill-eagle3-tuner.md`。给定 inference workload（model、batch size、target latency、task profile），它会推荐 speculative-decoding strategy 和 tuning parameters（draft family、`N`、tree depth、temperature-aware switching）。

## 练习

1. 运行 `code/main.py`。确认 Leviathan distribution check 在 50,000 samples 上的 chi-square statistic 低于 95% critical value。

2. 在 `α` 固定为 0.9、`c` 固定为 0.04 时，把 `N` 从 1 sweep 到 10。绘制 expected tokens per verifier call 和 actual wall time per token。找到最小化 wall time 的 `N`。解释曲线形状。

3. 修改代码以模拟 EAGLE-2 tree search：每一步 draft 提出形状为 `[2, 2, 2]` 的 tree（八条 candidate paths）。verifier 运行一次，highest-probability accepted path 胜出。计算每个 leaf 的 `α` 和每次 verifier call 的 total tokens。与 equivalent compute 下的 linear-chain spec-decoding 比较。

4. 为两个 concurrent sequences 实现 batched KV rollback simulator。Sequence A 的所有 drafts 都被接受；sequence B 在位置 2 reject。展示正确的 `kv_length` 会按 sequence 更新，并且没有浪费工作。

5. 阅读 EAGLE-3 论文 Section 4（Training-Time Test）。用两句话解释为什么没有 TTT 的 naive draft training 会遭遇 exposure bias，以及为什么训练时把 draft 自己的 predictions 喂回去能修复它。把它连接到 seq2seq 中的 scheduled-sampling literature。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Leviathan rule | "min(1, q over p)" | 以 `min(1, q(d)/p(d))` 概率做 Bernoulli accept/reject；如果 reject 时从 residual 采样，就精确保留 verifier distribution |
| Residual distribution | "(q minus p) plus, normalized" | `(q - p)_+` clamp 到零并 renormalize -- rejection 时应该采样的正确 distribution |
| Acceptance rate α | "how often the draft is right" | rejection rule 下的 expected per-token Bernoulli-success probability；支配所有 speedup math |
| EAGLE-1 | "hidden-state draft" | 以 verifier 的 last-layer hidden state 为条件的 tiny transformer draft（Li et al., 2024） |
| EAGLE-2 | "dynamic draft tree" | EAGLE-1 加 candidate continuations tree，在一次 verifier pass 中用 tree attention 打分 |
| EAGLE-3 | "training-time test" | 去掉 feature-prediction loss，直接训练 token prediction，并在训练中让 draft 喂入自己的 outputs |
| Training-time test (TTT) | "exposure bias fix" | 训练期间 autoregressively 运行 draft，让 train 和 test input distributions 匹配 -- scheduled sampling 的直接对应物 |
| KV rollback | "undo rejected drafts" | rejection 后把 verifier 的 KV cache 重置到 accepted-prefix length 的 bookkeeping |
| Bonus token | "the free one" | 当所有 `N` drafts 都接受时，从 `q_{N+1}` 额外采样一个，不增加 verifier cost |
| Tree attention | "verify many candidates at once" | 带 non-causal mask 的 attention，尊重 draft tree topology；一次 forward pass 计算 tree 中每个 node 的 `q_i` |

## 延伸阅读

- [Leviathan, Kalman, Matias — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192, ICML 2023)](https://arxiv.org/abs/2211.17192) — foundational paper 和 equivalence theorem
- [Chen et al. — Accelerating Large Language Model Decoding with Speculative Sampling (arXiv:2302.01318)](https://arxiv.org/abs/2302.01318) — 并行独立提出的方法，证明清晰
- [Li et al. — EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) — EAGLE-1，hidden-state-conditioned draft
- [Li et al. — EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — dynamic tree search
- [Li et al. — EAGLE-3: Scaling up Inference Acceleration via Training-Time Test (arXiv:2503.01840, NeurIPS 2025)](https://arxiv.org/abs/2503.01840) — 2026 年生产默认方案
- [Cai et al. — Medusa: Multiple Decoding Heads (arXiv:2401.10774)](https://arxiv.org/abs/2401.10774) — 另一种 draft-free approach
- [vLLM Speculative Decoding documentation](https://docs.vllm.ai/en/latest/features/spec_decode.html) — 连接好所有 strategies 的 canonical production reference
