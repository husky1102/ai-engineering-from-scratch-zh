# 推理指标：TTFT、TPOT、ITL、Goodput、P99

> 四个指标决定一个 inference deployment 是否真的可用。TTFT 是 prefill 加 queue 加 network。TPOT（等价于 ITL）是每个 token 的 memory-bound decode cost。End-to-end latency 是 TTFT 加上 TPOT 乘以输出长度。Throughput 是整个 fleet 聚合后的 tokens per second。但对产品真正重要的是 goodput，也就是同时满足每个 SLO 的请求比例。高 throughput 但低 goodput，意味着你正在处理那些无法准时到达用户的 tokens。2026 年 TRT-LLM 上 Llama-3.1-8B-Instruct 的参考数字：mean TTFT 162 ms、mean TPOT 7.33 ms、mean E2E 1,093 ms。始终报告 P50、P90、P99，不要只报 mean。还要注意测量陷阱：GenAI-Perf 在 ITL 计算中排除 TTFT，LLMPerf 则包含它；两个工具会对同一次运行给出不同 TPOT。

**类型：** Learn
**语言：** Python (stdlib, toy percentile calculator and goodput reporter)
**先修：** Phase 17 · 04 (vLLM Serving Internals)
**时间：** ~60 minutes

## 学习目标

- 精确定义 TTFT、TPOT、ITL、E2E、throughput 和 goodput，并说出每个指标测量的是哪个组件。
- 解释为什么 mean 是 LLM serving 的错误统计量，以及如何阅读 P50/P90/P99。
- 构造一个 SLO multi-constraint（例如 TTFT<500 ms AND TPOT<15 ms AND E2E<2 s），并用它计算 goodput。
- 说出两个会在同一次运行上对 TPOT 给出不同结果的 benchmark tools，并解释原因。

## 要解决的问题

“我们的 throughput 是每秒 15,000 tokens。”所以呢？如果 40% 的请求超过了 2 秒 end-to-end，用户已经放弃了会话。单看 throughput 无法告诉你产品是否可用。

Inference 有多个 latency 轴，每个轴失败方式都不同。Prefill 受 compute 限制，并随 prompt length 扩展。Decode 受 memory 限制，并随 batch size 扩展。Queuing delay 是运营问题。Network 是物理距离问题。你需要为每个部分设置不同指标，需要 percentiles，还需要一个单一复合指标来回答“用户是否得到预期体验”：这就是 goodput。

## 核心概念

### TTFT：time to first token

`TTFT = queue_time + network_request + prefill_time`

当 prompts 很长时，prefill 占主导。在 H100 上运行 Llama-3.3-70B FP8，一个 32k prompt 需要约 800 ms 的纯 prefill。Queue time 是负载下 scheduler behavior 的结果。Network request 是包含 TLS 的 wire time。TTFT 是用户在任何内容 streaming back 之前看到的 latency。

### TPOT / ITL：inter-token latency

同一个量有很多名字。`TPOT`（time per output token）、`ITL`（inter-token latency）、`decode latency per token` 都是一回事。它表示第一个 token 之后，相邻 streamed tokens 之间的时间。

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

在同一套 Llama-3.3-70B H100 stack 上，启用 chunked prefill 时 TPOT mean 约为 7 ms。没有 chunked prefill 时，如果邻近 sequence 正在做长 prefill，TPOT 可能飙到 50 ms。看 P99，不要看 mean。

### E2E latency

`E2E = TTFT + TPOT * output_tokens + network_response`

对于长输出（>500 tokens），E2E 由 TPOT 主导。对于长 prompt 短输出，E2E 由 TTFT 主导。报告 E2E 时要按 output length 分组。

### Throughput

`throughput = total_output_tokens / elapsed_time`

这是聚合指标。它告诉你 fleet efficiency，但不告诉你单个请求是否健康。

### Goodput：你真正关心的指标

`goodput = fraction of requests meeting (TTFT <= a) AND (TPOT <= b) AND (E2E <= c)`

SLO 是 multi-constraint。只有每条约束都满足，请求才是“good”。Goodput 就是这个比例。60% goodput 下的高 throughput 是失败。目标是以较低 throughput 换取 99% goodput。

2026 年，goodput 是 MLPerf Inference v6.0 submissions 和 AI platform providers 内部 SLA tracking 使用的指标。

### 为什么 mean 是错误统计量

LLM latency distributions 是右偏的。一个 decode batch 里，如果有一个邻居正在做长 prefill，可能会有 500 个 tokens 的 TPOT 约为 7 ms，另有 20 个 tokens 的 TPOT 约为 60 ms。Mean TPOT 是 9 ms。P99 TPOT 是 65 ms。用户会经常撞上 P99，这就是他们离开的原因。

始终报告三元组（P50、P90、P99）。对用户体验而言，P99 才是你要优化的指标。

### 参考数字：2026 年 TRT-LLM 上的 Llama-3.1-8B-Instruct

- mean TTFT: 162 ms
- mean TPOT: 7.33 ms
- mean E2E: 1,093 ms
- P99 TPOT: 取决于 chunked-prefill 配置，约 10-25 ms。

这些是 NVIDIA 发布的参考点。它们会随着 model size（70B 会显示 3-5x）、hardware（H100 vs B200 约 3x）和负载变化。

### 测量陷阱

2026 年最常用的两个 benchmark tools 会在同一次运行上对 TPOT 给出不同结果：

- **NVIDIA GenAI-Perf**：在 ITL 计算中排除 TTFT。ITL 从 token 2 开始。
- **LLMPerf**：包含 TTFT。ITL 从 token 1 开始。

对于一个 TTFT 500 ms、100 output tokens、总 decode 700 ms 的请求，GenAI-Perf 报告 `ITL = 700/99 = 7.07 ms`，LLMPerf 报告 `ITL = 1200/100 = 12.00 ms`。工具选择会改变数字。

始终声明使用哪个工具。始终发布定义。

### 构造一个 SLO

2026 年面向消费者的 70B chat model，一个合理 SLO 是：

- TTFT P99 <= 800 ms.
- TPOT P99 <= 25 ms.
- E2E P99 <= 3 s for <300-token outputs.
- Goodput target >= 99%.

Enterprise SLOs 会收紧 TTFT（200-400 ms）并放宽 E2E。重点是把它们写下来，测量三者，并把 goodput 作为单一复合指标跟踪。

### 如何测量

- 运行真实流量或逼真的 synthetic（LLMPerf 配合 `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`）。
- benchmark run 目标设为 peak concurrency 的 2x。
- 运行 30-50 iterations，对合并样本取 percentiles。
- 发布时附上 tool name、tool version、model、hardware、concurrency、prompt distribution。

## 实际使用

`code/main.py` 是一个 toy goodput calculator。生成 synthetic latency distribution，应用 SLO，然后计算 goodput。它还展示同一条 trace 上 GenAI-Perf 和 LLMPerf 的 TPOT 差异。

## 交付成果

本课产出 `outputs/skill-slo-goodput-gate.md`。给定 workload 和 SLO，它会生成 CI/CD-ready benchmark recipe，用 goodput 而不是 throughput 来 gate deploys。

## 练习

1. 运行 `code/main.py`。生成带 1% tail spike 的 distribution。当你把 P99 TPOT 从 30 ms 收紧到 15 ms 时，goodput 如何变化？
2. 供应商报价说“Llama 3.3 70B H100 上 15,000 tok/s”。在相信它之前，你要问哪三个问题？
3. 为什么 chunked prefill 能保护 P99 TPOT，却不保护 mean TPOT？
4. 为 voice assistant 构造一个 consumer SLO（第一个 token 被听到，而不是被读到）。哪个指标最能被用户感知？
5. 阅读 LLMPerf README 和 GenAI-Perf docs。找出工具之间还有哪三个指标定义不一致。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| TTFT | “time to first token” | Queue + network + prefill；长 prompts 下由 prefill 主导 |
| TPOT | “time per output token” | 第一个 token 之后，每个 token 的 memory-bound decode cost |
| ITL | “inter-token latency” | 多数工具中与 TPOT 相同（并非全部，见 GenAI-Perf） |
| E2E | “end to end” | TTFT + TPOT * output_len；再加 response-side network |
| Throughput | “tok/s” | Fleet efficiency；没有 latency percentiles 就没有意义 |
| Goodput | “SLO-met rate” | 同时满足每个 SLO constraint 的请求比例 |
| P99 | “tail” | 100 次中最差 1 次的 latency；用户体验指标 |
| SLO multi-constraint | “the joint” | 三个 latency bounds 的 AND；任意一个违反即请求失败 |
| GenAI-Perf vs LLMPerf | “the tool trap” | 工具在 ITL 是否包含 TTFT 上定义不同 |

## 延伸阅读

- [NVIDIA NIM — LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — TTFT、ITL、TPOT 的 canonical definition。
- [Anyscale — LLM Serving Benchmarking Metrics](https://docs.anyscale.com/llm/serving/benchmarking/metrics) — alternative definitions 和 measurement recipe。
- [BentoML — LLM Inference Metrics](https://bentoml.com/llm/inference-optimization/llm-inference-metrics) — real deployments 上的 applied measurement。
- [LLMPerf](https://github.com/ray-project/llmperf) — 基于 Ray 的 open-source benchmark。
- [GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c++/perf_analyzer/genai-perf/README.html) — NVIDIA 的 benchmark tool。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 业内认可的 goodput-based benchmark。
