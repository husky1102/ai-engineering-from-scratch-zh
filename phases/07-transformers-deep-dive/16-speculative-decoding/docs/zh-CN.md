# Speculative Decoding — Draft, Verify, Repeat

> Autoregressive decoding 是串行的。每个 token 都要等待前一个 token。Speculative decoding 打破这条链：便宜模型先 draft N 个 tokens，昂贵模型用一次 forward pass 全部 verify。draft 对了的时候，你为 N 次 generation 只付了一次 big forward。

**类型：** Build
**语言：** Python
**先修：** Phase 7 · 07 (GPT Causal LM), Phase 7 · 12 (KV Cache & Flash Attention)
**时间：** ~60 分钟

## 要解决的问题

一个 70B LLM 在 H100 上 sample 一个 token 约需 30 ms。一个 3B draft model 约需 3 ms。如果我们让 3B draft 提前生成 5 个 tokens，然后让 70B *一次* verify 全部 5 个，总耗时是 `5×3 + 30 = 45 ms`，最多接受 5 个 tokens；而直线生成需要 `5×30 = 150 ms`。这就是 speculative-decoding 的完整主张：用少量额外 GPU memory（draft model）换取 2-4× 更低的 decode latency。

诀窍是必须保留分布。由 Leviathan et al.（2023）以及 Chen et al. 同期提出的 speculative sampling，保证输出 sequence 与大模型单独生成时 **同分布**。没有质量 tradeoff。只是更快。

四类 draft-verifier pair 主导 2026 inference：

1. **Vanilla speculative（Leviathan 2023）.** 独立 draft model（例如 Llama 3 1B）+ verifier（例如 Llama 3 70B）。
2. **Medusa（Cai 2024）.** verifier 上的多个 decoding heads 并行预测位置 `t+1..t+k`。不需要独立 draft model。
3. **EAGLE family（Li 2024, 2025）.** 复用 verifier hidden states 的轻量 draft；比 vanilla 更高 acceptance rate；典型 3-4×。
4. **Lookahead decoding（Fu 2024）.** Jacobi iteration；完全不需要 draft model。Self-speculation。小众但 dependency-free。

2026 年每个 production inference stack 都默认交付 speculative decoding。vLLM、TensorRT-LLM、SGLang 和 llama.cpp 至少都支持 vanilla + EAGLE-2。

## 核心概念

### Core algorithm

给定 verifier `M_q` 和更便宜的 draft `M_p`：

1. 令 `x_1..x_k` 为已经 decoded 的 prefix。
2. **Draft**：使用 `M_p` 自回归地提出 `d_{k+1}, d_{k+2}, ..., d_{k+N}`，并记录 draft probabilities `p_1..p_N`。
3. **并行 verify**：让 `M_q` 在 `x_1..x_k, d_{k+1}, ..., d_{k+N}` 上运行一次，得到位置 `k+1..k+N+1` 的 verifier probabilities `q_1..q_{N+1}`。
4. **从左到右 accept/reject 每个 draft token**：对每个 `i`，以概率 `min(1, q_i(d_i) / p_i(d_i))` 接受。
5. 在位置 `j` 第一次拒绝时：从 normalized 的 “residual” distribution `(q_j - p_j)_+` 中 sample `t_j`。`j` 之后所有 drafts 都丢弃。
6. 如果全部 `N` 个都接受：从 `q_{N+1}` 中再 sample 一个额外 token `t_{N+1}`（免费的 bonus token）。

Residual distribution trick 是保持输出与 `M_q` 从头 sampling 完全同分布的数学洞见。

### 什么决定 speedup

令 `α` = 每个 draft token 的 expected acceptance rate。令 `c` = draft-to-verifier cost ratio。每一步：

- Naive generation 每个 token 调一次 big-model。
- 当 `α` 很高时，speculative 每 `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` 个 tokens 调一次 big-model。

经验法则：在 `α = 0.75`、`N = 5` 时，big-model calls 少约 3×。Draft cost 是 5× cheap。总 wall-clock 下降约 2.5×。

**α 取决于：**

- Draft 对 verifier 的近似程度。同 family / 同 training data 会显著提升 α。
- Decoding strategy。Greedy draft 对 greedy verifier：高 α。Temperature sampling：更难匹配；acceptance 会下降。
- Task type。Code 和 structured output 接受更多（可预测）；free-form creative writing 接受更少。

### Medusa：没有 draft model 的 drafts

Medusa 用 verifier 上的额外 output heads 替代 draft model。在位置 `t`：

```text
shared trunk → hidden h_t
    ├── head_0: predict token at t+1  (standard LM head)
    ├── head_1: predict token at t+2
    ├── head_2: predict token at t+3
    ├── head_3: predict token at t+4
```

每个 head 输出自己的 logits。Inference 时，你从每个 head sample 得到 candidate sequence，然后用一次 forward pass 和 tree-attention scheme 来同时 verify 所有 candidate continuations。

优点：没有第二个模型。缺点：增加 trainable parameters；需要一次 supervised fine-tuning stage（约 1B tokens）；acceptance rate 比带好 draft 的 vanilla speculative 略低。

### EAGLE：复用 hidden states 的更好 draft

EAGLE-1/2/3（Li et al., 2024-2025）把 draft model 做成一个 tiny transformer（通常 1 layer），输入 verifier 的 last-layer hidden states。因为 draft 看到了 verifier 的 feature representation，它的预测与 verifier 的 output distribution 强相关。Acceptance rates 从约 0.6（vanilla）升到 0.85+。

EAGLE-3（2025）加入了对 candidate continuations 的 tree search。vLLM 和 SGLang 将 EAGLE-2/3 作为 Llama 3/4 和 Qwen 3 的默认 spec pathway。

### KV cache dance

Verification 会把 `N` 个 draft tokens 一次性喂给 verifier。这会把 verifier 的 KV cache 扩展 `N` 项。如果某些 drafts 被拒绝，你必须把 cache 回滚到已接受 prefix length。

Production implementations（vLLM 的 `--speculative-model`，TensorRT-LLM 的 LookaheadDecoder）用 scratch KV buffers 处理这个问题。先写入，acceptance 后 commit。概念上不难，但细节很繁琐。

## 动手实现

见 `code/main.py`。我们用以下组件实现 core speculative-sampling algorithm（rejection step + residual distribution）：

- 一个 “big model”，它是 hand-coded distribution 上的 deterministic-softmax（这样可以分析地验证 acceptance math）。
- 一个 “draft model”，它是 big model 的 perturbation。
- 一个 acceptance / rejection loop，生成与 direct sampling 相同的 marginal distribution。

### Step 1: rejection step

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` 是 uniform random number。`q_prob` 是 verifier 对 drafted token 的 probability。`p_prob` 是 draft model 的 probability。Leviathan theorem 说，这个 Bernoulli decision 加上 rejection 时从 residual sample，会精确保留 verifier distribution。

### Step 2: residual distribution

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

逐元素从 `q` 中减去 `p`，把负值 clamp 到零，再 renormalize。任何 rejection 发生时都从这里 sample。

### Step 3: one speculative step

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

五个都接受 → 一个 bonus → 一次 verifier pass 生成六个 tokens。

### Step 4: measure acceptance rate

在不同 draft-quality levels 上运行 10,000 个 speculative steps。绘制 acceptance rate vs. draft 与 verifier distributions 之间的 KL divergence。你应该看到清晰的单调关系。

### Step 5: verify distribution equivalence

经验验证：speculative loop 产生的 token histogram 应该匹配从 verifier direct sampling 得到的 histogram。实际中这就是 Leviathan theorem。Chi-square test 会确认它在 sampling error 内。

## 实际使用

Production：

```bash
# vLLM with EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# vLLM with vanilla draft model
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

截至 2026 年中，TensorRT-LLM 拥有最快的 Medusa path。`faster-whisper` 为 Whisper-large 封装了带小 draft 的 speculative decoding。

**选择 draft：**

| Strategy | When to pick | Speedup |
|----------|--------------|---------|
| Vanilla draft (1B/3B Llama family) | Fast prototype, no training | 1.8-2.3× |
| Medusa heads | You can fine-tune the verifier | 2-3× |
| EAGLE-2 / 3 | Production, max speed | 3-4× |
| Lookahead | No draft, no training, no extra params | 1.3-1.6× |

**什么时候不要 spec-decode：**

- 1-5 tokens 的 single-sequence generation。Overhead 会主导。
- Wildly creative / high-temperature sampling（α 会下降）。
- Memory-constrained deployments（draft model 增加 VRAM）。

## 交付成果

见 `outputs/skill-spec-decode-picker.md`。这个 skill 会为新的 inference workload 选择 speculative decoding strategy（vanilla / Medusa / EAGLE / lookahead）和 tuning parameters（N, draft temperature）。

## 练习

1. **Easy.** 运行 `code/main.py`。确认 speculative token distribution 与 verifier 的 direct-sample distribution 在 50,000 tokens 上匹配，chi-square p > 0.05。
2. **Medium.** 把 speedup（tokens per big-model forward）画成 `N` 的函数，分别使用 `α = 0.5, 0.7, 0.85`。找出每个 α 的 optimal `N`。（Hint: expected tokens per verify call = `(1 - α^{N+1}) / (1 - α)`。）
3. **Hard.** 实现一个 tiny Medusa：使用 Lesson 14 的 capstone GPT，增加 3 个额外 LM heads，预测位置 t+2、t+3、t+4。在 tinyshakespeare 上用 joint multi-head loss 训练。比较它与通过截断同一模型得到的 vanilla draft 的 acceptance rates。
4. **Hard.** 实现 rollback：从一个 10-token prefix KV cache 开始，喂入 5 个 draft tokens，模拟在位置 3 reject。验证下一轮迭代时你的 cache reads 正确匹配 “prefix + first 2 accepted drafts”。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Draft model | “便宜的那个” | 提出 candidate tokens 的小模型；通常比 verifier 便宜 10-50×。 |
| Verifier | “大的那个” | 我们要保留其分布的 target model；每个 speculative step 运行一次。 |
| Acceptance rate (α) | “draft 有多常对” | verifier 接受 draft 的 per-token probability。典型 0.7-0.9。 |
| Residual distribution | “rejection fallback” | `(q - p)_+` normalized；rejection 时从这里 sample 会保留 verifier distribution。 |
| Bonus token | “免费的那个” | 当全部 N 个 drafts 都接受时，从 verifier 的 next-step distribution 再 sample 一个。 |
| Medusa | “Draft-less speculative” | verifier 上的多个 LM heads 并行预测位置 t+1..t+k。 |
| EAGLE | “Hidden-state draft” | 以 verifier 的 last-layer hidden states 为条件的 tiny transformer draft。 |
| Lookahead decoding | “Jacobi iteration” | 使用 fixed-point iteration 的 self-speculation；没有 draft model。 |
| Tree attention | “一次 verify 多个 candidates” | 同时考虑多个 draft continuations 的 branching verification。 |
| KV rollback | “撤销 rejected drafts” | Scratch KV buffer；接受时 commit，拒绝时 discard。 |

## 延伸阅读

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — core algorithm 和 equivalence theorem。
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — 同期提出；清晰的 Bernoulli-rejection proof。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — Medusa paper；tree-attention verification。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1；hidden-state-conditioned draft。
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — EAGLE-2；dynamic tree depth。
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) — EAGLE-3。
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — lookahead，无 draft 方法。
- [vLLM docs — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) — canonical production reference，四种策略都已接入。
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE) — EAGLE-1/2/3 的 reference code。
