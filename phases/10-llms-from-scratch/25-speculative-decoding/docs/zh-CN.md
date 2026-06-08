# Speculative Decoding 与 EAGLE

> 一个 frontier LLM 生成一个 token，需要对数十亿参数做一次完整 forward pass。这个 forward pass 其实严重 over-provisioned：大多数时候，一个小得多的模型可以正确猜出接下来的 3-5 个 tokens，而大模型只需要 *verify* 这个猜测。猜对时，你用一次 forward 的价格得到了 5 个 tokens。Speculative decoding（Leviathan et al. 2023）把这件事做成了精确算法，而 EAGLE-3（2025）把 acceptance rates 推到每次 verify 约 4.5 个 tokens——在输出分布匹配的情况下实现 4-5x speedup。

**类型:** Build
**语言:** Python (with numpy)
**先修:** Phase 10 Lesson 12 (Inference Optimization), Phase 10 Lesson 04 (Pre-training Mini-GPT)
**时间:** ~75 分钟

## 要解决的问题

70B 级模型在 H100 上的 decode throughput 通常是 40-80 tokens/second。每个 token 都需要一次完整 forward pass，从 HBM 读取所有模型权重。你不能在不改变输出的情况下让模型变小。你也不能把 batch size 增加到超过内存。你卡住了——除非你能让模型每次 forward pass 输出不止一个 token。

自回归生成看起来天然是串行的：`x_{t+1} = sample(p(· | x_{1:t}))`。但这里有一个并发机会。如果你有一个廉价 predictor 说“接下来 4 个 tokens 大概是 [a, b, c, d]”，你就可以在一次 **big model 的单次 forward pass** 中 verify 全部 5 个位置，并接受最长匹配 prefix。

Leviathan, Kalai, Matias（2023，“Fast Inference from Transformers via Speculative Decoding”）通过一个聪明的 accept/reject rule 让这件事精确成立，并保留 target model 的 sampling distribution。同样的输出分布，快 2-4×。

## 核心概念

### Two-Model Setup

- **Target model** `M_p`：你真正想从中采样的大、慢、高质量模型。分布：`p(x)`。
- **Draft model** `M_q`：小、快、质量较低的模型。分布：`q(x)`。通常小 5-30×。

每一步：

1. Draft model 自回归提议 `K` 个 tokens：`x_1, x_2, ..., x_K ~ q`。
2. Target model 对所有 `K+1` 个位置并行运行一次 forward pass，为每个 proposed token 产生 `p(x_k)`。
3. 用下面的 modified rejection-sampling rule 从左到右 accept/reject 每个 token。接受最长匹配 prefix。
4. 如果任何 token 被 rejected，就从 corrected distribution 中采样 replacement 并停止。否则从 `p(· | x_1...x_K)` 采样一个 bonus token。

如果 draft 与 target 完全匹配，你每次 target-forward 能得到 K+1 个 tokens。如果 draft 在位置 1 就错了，你只得到 1 个 token。

### 精确性规则

Speculative decoding **在分布上可证明等价于从 p 采样**。Rejection rule：

```text
For each drafted token x_t:
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        accept x_t
    else:
        sample replacement from residual: (p - q)+ / ||(p - q)+||_1
        stop
```

其中 `(p - q)+` 表示 pointwise difference 的正部。当 draft 和 target 一致（`p ≈ q`）时，acceptance 接近 1。当它们不一致时，residual distribution 被构造为让整体 sample 仍然精确等于 `p`。

**Greedy case.** 对 temperature=0 sampling，只需检查 `argmax(p) == x_t`。如果是，accept；如果不是，输出 `argmax(p)` 并停止。

### 期望加速

如果 draft model 的 token-level acceptance rate 是 `α`，每次 target-forward pass 期望产生的 tokens 数是：

```text
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = draft length, α in [0, 1]
```

在 `α = 0.8, K = 4` 时：`(1 - 0.8^5)/(1 - 0.8) = 3.36` tokens per forward。一次 target forward 大约花费 `cost_q * K + cost_p`（K 个 draft steps 加一次 target verify）。如果 `cost_p >> cost_q * K`，吞吐 speedup ratio 就是 `3.36× / 1 = 3.36×`。

唯一真正的参数是 `α`，它完全取决于 draft-target alignment。一个好 draft 就是一切。

### 训练 Draft：Distillation

随机小模型是糟糕的 draft。标准 recipe 是从 target distill：

1. 选择一个小架构（70B target 用约 1B，7B target 用约 500M）。
2. 在大文本语料上运行 target model；存储它的 next-token distributions。
3. 用 KL divergence 训练 draft 去匹配 target 的 distribution（不是 ground-truth tokens）。

结果：`α` 在 coding 上通常 0.6-0.8，在自然语言聊天上 0.7-0.85。生产中 speedups 通常 2-3×。

### EAGLE：Tree Drafting + Feature Reuse

Li, Wei, Zhang, Zhang（2024，“EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty”）观察到 standard speculative decoding 的两个低效：

1. Draft 会做 K 个串行 steps，每个都是 full-stack。但 draft 可以复用最近一次 verify 时 target 的 features（hidden states）——target 已经算出了 rich representations，draft 正在从零重推它们。
2. Draft 输出一条 linear chain。如果 draft 能输出一个候选 *tree*（每个 node 多个 guesses），target 的单次 forward pass 就可以通过 tree attention mask 并行 verify 多条 candidate paths，并选择最长 accepted branch。

EAGLE-1 变化：
- Draft input = target 在位置 t 的 final hidden state，而不是 raw tokens。
- Draft architecture = 1 transformer decoder layer（不是独立小模型）。
- Output = depth 4-6、每 depth K = 4-8 candidates 的 tree。

EAGLE-2（2024）加入 dynamic tree topology：draft 不确定时 tree 变宽，确定时保持较窄。在不增加 verify cost 的情况下提高 `α_effective`。

EAGLE-3（Li et al. 2025，“EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test”）移除了固定 top-layer feature dependency，并用新的 “test-time simulation” loss 训练 draft——draft 被训练去匹配 target test-time distribution 的 outputs，而不是 teacher-forced training distribution。Acceptance rate 从 0.75（EAGLE-2）升到 0.82（EAGLE-3），mean tokens/verify 从 3.0 升到 4.5。

### Tree Attention Verification

当 draft 输出一棵 tree 时，target model 会用一个 **tree attention mask** 在单次 forward pass 中 verify 它——这是一个编码 tree topology 而不是纯线性序列的 causal mask。每个 token 只关注它在 tree 中的 ancestors。Verify pass 仍然是一次 forward、一次 matmul；topological mask 只多花少量 KV entries。

```text
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

如果 `a, b` 是竞争的 first-token candidates，而 `c, d, e, f` 是 second-token candidates，所有六个位置都会在一次 forward pass 中被 verified。输出是任意 accepted path 中最长的 prefix。

### 什么时候赢，什么时候不赢

**赢：**
- Chat / completion 中可预测文本（code、common English、structured output）。`α` 高。
- Decode 阶段 GPU compute 未被充分使用的设置（memory-bound phase）。Tree drafting 会利用可用 FLOPs。

**输 / 没收益：**
- 高随机性输出（高 temperature creative writing）。`α` 会掉向 `1/|vocab|`。
- 很高 concurrency 的 batch serving——batching 已经填满 FLOPs，几乎没有 tree verification 空间。
- Target models 非常小，此时 draft 并没有小多少。

Production shops 通常报告 chat 上 2-3× wall-clock speedup，code generation 上 3-5×，creative writing 上接近零。

## 动手实现

`code/main.py`：

- 一个 reference `speculative_decode(target, draft, prompt, K, temperature)`，实现 exact rejection rule，并验证它保留 target distribution（empirical KL < 0.01 vs plain target sampling）。
- 一个 EAGLE-style tree drafter，用 top-p branching 构建 depth-K tree。
- 一个 tree attention mask builder，为 verifier 产生正确 causal pattern。
- 一个 acceptance-rate harness，在 tiny LM 上同时运行两者（从 GPT-2-medium target distill 一个 GPT-2-small）。

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """One round of speculative decoding. Returns list of accepted tokens."""
    # 1. Draft K tokens
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. Target computes p at every drafted position + 1 extra
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. Accept/reject left-to-right
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. All K accepted → sample bonus token from target
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## 实际使用

- **vLLM** 和 **SGLang** 提供 first-class speculative decoding。Flags：`--speculative_model`、`--num_speculative_tokens`。EAGLE-2/3 通过 `--spec_decoding_algorithm eagle` flag 支持。
- **NVIDIA TensorRT-LLM** 原生支持 Medusa 和 EAGLE trees。
- **Reference draft models**：`Qwen/Qwen3-0.6B-spec`（Qwen3-32B 的 drafts）、`meta-llama/Llama-3.2-1B-Instruct-spec`（70B 的 drafts）。
- **Medusa heads**（Cai et al. 2024，“Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads”）：不使用 draft model，而是在 target 自身上添加 K 个 parallel prediction heads。部署更简单，acceptance 略低于 EAGLE。

## 交付成果

本课产出 `outputs/skill-speculative-tuning.md`——一个 skill，用来 profile target model 的 workload，并选择：draft model、K（draft length）、tree width、temperature，以及何时回退到 plain decode。

## 练习

1. 实现 exact rejection rule 并做 empirical verification。通过 `speculative_decode` 和 plain target sampling 各运行 10K samples；计算两个 output distributions 的 TV distance。应小于 0.01。

2. 计算 speedup formula。给定固定 `α` 和 `K`，绘制每次 target-forward 的 expected tokens。找出 α ∈ {0.5, 0.7, 0.9} 时的最优 K。

3. 训练一个 tiny draft。取一个 124M GPT-2 target，并在 100M tokens 上用 KL loss distill 一个 30M GPT-2 draft。测量 held-out text 上的 `α`。期望：0.6-0.7。

4. 实现 EAGLE-style tree drafting。不要 draft chain，而是让 draft 在每个 depth 输出 top-3 branches。构建 tree attention mask。验证 target 会接受最长正确 branch。

5. 测量 failure modes。在 temperature=1.5（高随机性）下运行 speculative decode。展示 α 崩塌，并且算法由于 draft overhead 比 plain decode 更慢。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Target model | “大模型” | 你想从中采样的慢速高质量模型（p distribution） |
| Draft model | “Speculator” | 小而快的 predictor（q distribution）；小 5-30x |
| K / draft length | “Look-ahead” | 每次 verify pass speculative 的 token 数量 |
| α / acceptance rate | “Hit rate” | Draft proposal 被接受的 per-token probability |
| Exact rejection rule | “Accept test” | 保留 target distribution 的 r < p/q 比较 |
| Residual distribution | “Corrected p-q” | (p - q)+ / ||(p - q)+||_1，rejection 时从中采样的分布 |
| Tree drafting | “Branching speculation” | Draft 输出候选 tree，用 tree-structured attention mask 在一次 pass 中 verify |
| Tree attention mask | “Topological mask” | 编码 tree topology 的 causal mask，让每个 node 只关注其 ancestors |
| Medusa heads | “Parallel heads” | Target 自身上的 K 个额外 prediction heads；没有独立 draft model |
| EAGLE feature reuse | “Hidden-state draft” | Draft input 是 target 的最后 hidden state，而不是 raw tokens，从而缩小 draft |
| Test-time simulation loss | “EAGLE-3 training” | 训练 draft 去匹配 target test-time distribution 的 outputs，而不是 teacher forcing |

## 延伸阅读

- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — exact rejection rule 和理论 speedup analysis
- [Chen, Borgeaud, Irving et al., 2023 — "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) — DeepMind 的同期 speculative-sampling paper
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) — draft model 的 parallel-heads alternative
- [Li, Wei, Zhang, Zhang, 2024 — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077) — feature reuse 和 tree drafting
- [Li et al., 2024 — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858) — dynamic tree topology
- [Li et al., 2025 — "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840) — train-time test-time matching
- [Fu, Haotian, Peng et al., 2024 — "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057) — Jacobi/lookahead decoding，一个无需 speculator 的 alternative
