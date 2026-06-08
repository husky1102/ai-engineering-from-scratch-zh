# 生产中的 EAGLE-3 Speculative Decoding

> Speculative decoding 把快速 draft model 与 target model 配对。Draft 提出 K 个 tokens；target 在一次 forward 中验证；accepted tokens 等于免费。2026 年，EAGLE-3 是 production-grade variant，它在 target model 的 hidden states 上训练 draft head，而不是在 raw tokens 上训练，把 general chat 上的 acceptance rate alpha 推到 0.6-0.8 区间。正确问题不是“draft 多快”，而是“我的 traffic 上 alpha 是多少？”如果 alpha 降到约 0.55 以下，在 high concurrency 下 speculative decoding 会变成净负收益，因为每个 rejected draft 都会花掉第二次 target forward。本课教你先测 alpha，再翻 flag。

**类型：** Learn
**语言：** Python (stdlib, toy acceptance-rate simulator)
**先修：** Phase 17 · 04 (vLLM Serving Internals), Phase 10 · 18 (Multi-Token Prediction)
**时间：** ~60 minutes

## 学习目标

- 说出 speculative decoding 的三代，并解释 EAGLE-3 相比 EAGLE-2 和 classic draft model 改变了什么。
- 定义 acceptance rate alpha，根据 alpha 和 K（draft length）计算 expected speedup，并识别 target concurrency 下的 break-even alpha。
- 解释为什么 speculative decoding 在 vLLM 2026 中是 opt-in（不是 default），以及为什么不测 alpha 就开启它是 production anti-pattern。
- 写一个 measurement plan：使用哪个 benchmark、哪种 prompt distribution、哪个 concurrency point、用哪个 metric 做 gate。

## 要解决的问题

Decode 是 memory-bound。在 H100 上运行 Llama 3.3 70B FP8 时，每个 decoded token 会读取约 140 GB/s 的 weights，并输出一个 token。GPU compute 在 decode 期间几乎空闲，bottleneck 是 HBM bandwidth，而不是 matmul throughput。

Speculative decoding 利用这个缺口。用便宜的 draft model 生成 K 个 candidate tokens，然后让 target model 在一次 forward pass 中验证全部 K 个。每个 verified token 实际上都是免费的（被摊入 target 原本无论如何都要做的 batch-of-K forward）。

Classic draft-model approach 使用同 family 的更小模型（例如 Llama 3.2 1B 为 Llama 3.3 70B drafting）。它能工作，但 acceptance rate 一般，因为小模型 distribution 偏离 target。EAGLE、EAGLE-2、EAGLE-3 则直接在 target model 的 internal states 上训练 light draft head，因此 draft distribution 更贴近 target。这就是 alpha 从 draft-model 的 0.4 提升到 EAGLE-3 的 0.6-0.8 的原因。

问题是：EAGLE-3 在 vLLM 2026 中是 opt-in。必须显式设置 `speculative_config`。没有 flag，就没有 acceleration。不在真实 traffic 上测 alpha 就打开它的团队，常常看到 tail latency 变差，而不是变好。

## 核心概念

### Speculative decoding 实际买到什么

没有 spec decode 时，每个 token 的 cost 是一次 target forward。使用 draft length K 和 acceptance alpha 的 spec decode 时，expected tokens per target forward 是 `1 + K * alpha`。Speedup 是 `(1 + K * alpha) / (1 + epsilon)`，其中 epsilon 是 draft-plus-verify overhead。对于 K=5、alpha=0.7：`(1 + 5*0.7) / (1 + 0.1) = 4.5 / 1.1 = 4.1x`。真实世界数字集中在 2-3x，因为 production traffic 上 alpha 很少这么高，而且 high batch size 下 epsilon 会增长。

### 为什么 alpha 是唯一重要指标

Rejected tokens 不会消失，它们会迫使第一个 rejected token 再做一次 target forward。当 alpha 降到 0.4 的 workload 上，你要支付 draft overhead、verification 和 re-roll。High concurrency（比如 256 concurrent）下，decode batch 已经足够大，“target alone” 和 “target with verify” 之间的 memory-bandwidth gap 会缩小。大多数 2026 hardware 上，alpha 低于 0.55 时，spec decode 会变成净负收益。

Alpha 随 workload 变化。在 ShareGPT-style general chat 上，用 ShareGPT 训练的 EAGLE-3 可达到 0.6-0.8。在 domain-specific traffic（code、medical、legal）上，使用 general data 训练的 draft head 会降到 0.4-0.6。训练 domain-specific draft head 可以恢复 alpha；相比 target finetuning，这是一个轻量、快速的训练任务。

### EAGLE generations 一览

- **Classic draft model**：同 family 的小模型。Alpha 0.3-0.5。Infrastructure 简单：加载两个 models，draft 每次 target forward 前运行 K 次 forward。
- **EAGLE-1 (2024)**：在 target hidden states（last layer）上训练的 single draft head。Alpha 约 0.5-0.6。Target 上方增加少量 parameter overhead。
- **EAGLE-2 (2025)**：adaptive draft length 和 tree-based drafts（在一次 target pass 中验证多个 branches）。Alpha 约 0.6-0.7。Draft scheduler 更复杂。
- **EAGLE-3 (2025-2026)**：draft head 在 multiple target layers 上训练（不只 last layer），alignment 更好。General chat 上 alpha 约 0.6-0.8。

### 2026 年生产配方

1. 先发布 plain target model。测量 target concurrency 下的 baseline TTFT、ITL、throughput。
2. 通过 vLLM `speculative_config` 启用 EAGLE-3 draft。重新运行 benchmark。
3. 记录 acceptance rate alpha。vLLM V1 把它报告为 `spec_decode_metrics.accepted_tokens_per_request`。除以 requested draft length 即得到 alpha。
4. 如果 production traffic distribution 上 alpha < 0.55，禁用 spec decode，或训练 domain-specific EAGLE-3 draft。
5. 在 production concurrency 下重新运行。确认 P99 ITL 没有变差。

### 生产陷阱：P99 tail

Mean ITL 会随 spec decode 下降。如果不调优，P99 可能变差。Rejected drafts 会触发 two-pass sequence（draft + verify-fail + reroll）。Full batch 下，这两个 passes 会串行化。看 P99 ITL，不要只看 P50。

### EAGLE-3 已部署在哪里

Google 于 2025 年在 AI Overviews 中部署了 speculative decoding（质量相同，响应更快）。vLLM V1 以 `speculative_config` 作为 documented interface；V1 中的 N-gram GPU speculative decoding 是 compatible with chunked prefill 的 variant。SGLang 支持 EAGLE-3，作为 prefix-heavy workloads 的 recommended draft path。

### 一行 break-even math

Expected speedup：`S(alpha, K) = (1 + K*alpha) / (1 + verify_overhead)`。令 `S = 1` 解得 alpha：`alpha_breakeven = verify_overhead / K`。对于 typical verify_overhead ~0.15 和 K=5：`alpha_breakeven = 0.03`。但这是 raw decode math。在 high concurrency 下，verify overhead 上升，decode batch 已经把 memory reads 跨 sequences 摊销，因此 effective alpha_breakeven 在实践中会升到约 0.45-0.55。

### 什么时候不要用 speculative decoding

- Batch-1 offline generation，latency 不重要。使用 plain target。
- 很短的 outputs（少于 50 tokens）。Draft overhead 和 verify cost 占主导。
- 没有 domain-trained draft head 的 specialized domains。Alpha 太低。
- vLLM v0.18.0 加 draft-model spec decode 加 `--enable-chunked-prefill`。这个组合无法 compile。文档例外是 V1 中的 N-gram GPU spec decode。

## 实际使用

`code/main.py` 在一系列 alpha values 和 draft lengths K 下模拟有无 speculative decoding 的 decode loop。它打印 break-even alpha、measured speedup 和 tail behavior。用多个 (alpha, K) 组合运行它，精确观察 speculative decoding 在哪里不再划算。

## 交付成果

本课产出 `outputs/skill-eagle3-rollout.md`。给定 target model、traffic distribution description 和 concurrency target，它会生成 staged EAGLE-3 rollout plan：benchmark baseline、enable config、measure alpha、gate on alpha >= 0.55、watch P99 ITL。

## 练习

1. 运行 `code/main.py`。K=5 时，要达到 2x speedup 需要什么 alpha？3x speedup 呢？它对 verify_overhead 有多敏感？
2. 假设 production traffic 是 70% general chat、30% code。General chat 使用 ShareGPT 训练的 EAGLE-3 达到 alpha 0.7；code 达到 alpha 0.4。Blended alpha 是多少，spec decode 是否 net-positive？
3. 阅读 vLLM `speculative_config` 文档。说出三种 modes（draft model、EAGLE、N-gram），以及哪一种 compatible with chunked prefill。
4. 启用 EAGLE-3 后你看到 mean ITL 下降 25%，但 P99 ITL 上升 15%。诊断并提出 mitigation。
5. 计算 Llama 3.3 70B 的 EAGLE-3 draft head memory cost。它与运行 Llama 3.2 1B 作为 classic draft 相比如何？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Speculative decoding | "draft plus verify" | 用便宜模型提出 K 个 tokens，并在一次 target forward 中验证全部 K 个 |
| Acceptance rate alpha | "spec accept rate" | Target 接受 draft tokens 的比例；唯一重要指标 |
| Draft length K | "spec k" | 每次 target forward 前 draft 提出的 tokens 数；典型 4-8 |
| Verify overhead epsilon | "spec overhead" | 相比 plain target forward，verify-and-reroll 的额外 cost；随 batch 增长 |
| EAGLE-3 | "latest EAGLE" | 2025-2026 variant；在 multiple target layers 上训练 draft head；general chat alpha 0.6-0.8 |
| `speculative_config` | "vLLM spec config" | vLLM V1 中的 explicit opt-in；没有 default 就没有 acceleration |
| N-gram spec decode | "N-gram draft" | GPU-side draft，使用 prompt 中的 N-gram lookups；chunked-prefill-compatible |
| Break-even alpha | "no-op alpha" | Spec decode 速度收益为零的 alpha；要在 production concurrency 下观察 |
| Rejected-draft two-pass | "reroll cost" | Draft reject 时需要两次 target forwards；驱动 P99 tail |

## 延伸阅读

- [vLLM — Speculative Decoding docs](https://docs.vllm.ai/en/latest/features/spec_decode/) — 关于 V1 中 `speculative_config` 和 chunked-prefill compatibility 的权威来源。
- [vLLM Speculative Config API](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/) — 精确 field set。
- [EAGLE paper (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) — 原始 EAGLE draft-head formulation。
- [EAGLE-2 paper (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — adaptive drafts 和 trees。
- [UC Berkeley EECS-2025-224](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html) — 使用 speculative decoding 的 efficient LLM system。
- [BentoML — Speculative Decoding](https://bentoml.com/llm/inference-optimization/speculative-decoding) — production rollout checklist。
