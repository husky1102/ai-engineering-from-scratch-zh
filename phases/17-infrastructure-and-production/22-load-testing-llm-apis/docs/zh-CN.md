# LLM APIs 的 Load Testing — 为什么 k6 和 Locust 会撒谎

> 传统 load testers 不是为 streaming responses、可变 output lengths、token-level metrics 或 GPU saturation 设计的。两个陷阱会咬到大多数团队。GIL trap：Locust 的 token-level measurement 在 Python GIL 下运行 tokenization，在高并发时会与 request generation 竞争；tokenization backlog 随后抬高报告出来的 inter-token latency —— 瓶颈在你的 client，不在 server。Prompt-uniformity trap：循环里的 identical prompts 只测试 token distribution 上的一个点；真实流量有可变长度和多样 prefix matches。LLMPerf 用 `--mean-input-tokens` + `--stddev-input-tokens` 修复这一点。2026 年工具映射：LLM-specialized（GenAI-Perf、LLMPerf、LLM-Locust、guidellm）用于 token-level accuracy；**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）** — streaming-aware、Kubernetes-native，通过 TestRun/PrivateLoadZone CRDs 做 distributed，最适合 CI/CD gates；Vegeta 用于 Go constant-rate saturation；Locust 2.43.3 只有配合 LLM-Locust extension 才适合 streaming。Load patterns：steady-state、ramp、spike（autoscaling test）、soak（memory leaks）。

**类型:** 构建
**语言:** Python（stdlib，玩具 realistic-prompt generator + latency collector）
**先修:** Phase 17 · 08（Inference Metrics），Phase 17 · 03（GPU Autoscaling）
**时间:** ~75 分钟

## 学习目标

- 解释两个会让 generic load testers 在 LLM APIs 上撒谎的反模式（GIL trap、prompt-uniformity trap）。
- 为给定目的选择工具：LLMPerf（benchmark run）、k6 + streaming extension（CI gate）、guidellm（large-scale synthetic）、GenAI-Perf（NVIDIA reference）。
- 设计四种 load patterns（steady、ramp、spike、soak），并说出每种能捕获的 failure mode。
- 用 input tokens 的 mean + stddev 构建 realistic prompt distribution，而不是固定长度。

## 要解决的问题

你用 k6 在 500 个 concurrent users 下测试了 LLM endpoint。它撑住了。你发版。生产环境只有 200 个真实用户时服务就倒了 —— P99 TTFT 爆炸，GPUs 打满。

发生了两件事。第一，k6 发送了 500 个 identical prompts —— 你的 request-coalescing 和 prefix caching 让它看起来像是在处理 500 个 concurrent decodes，实际却只是在处理一个。第二，k6 不会以人眼体验 streaming responses 的方式追踪 inter-token latency；它看到的是一个 HTTP connection，而不是 500 个 token 以不同间隔抵达。

LLMs 的 load testing 是一门独立纪律。

## 核心概念

### GIL trap（Locust）

Locust 使用 Python，并在 client-side 的 GIL 下运行 tokenization。在高并发下，tokenizer 会排在 request generation 后面。报告出来的 inter-token latency 包含 client-side tokenization backlog。你以为 server 慢，其实是 test harness 慢。

修复：LLM-Locust extension 将 tokenization 移到单独进程，或者使用 compiled-language harness（k6、使用 tokenizers.rs 的 LLMPerf）。

### Prompt-uniformity trap

所有已知 load testers 都允许你配置一个 prompt。在 10,000 次迭代的 loop test 中，每次发送完全相同的 prompt。Server 每次都看到相同 prefix —— prefix cache hits 接近 100%，throughput 看起来很好。

修复：从 prompt distribution 中采样。LLMPerf 使用 `--mean-input-tokens 500 --stddev-input-tokens 150` —— 多样长度、多样内容。

### 四种 load patterns

1. **Steady-state** — 以 constant RPS 持续 30-60 分钟。捕获：baseline performance regressions。
2. **Ramp** — 在 15 分钟内把 RPS 从 0 线性提升到目标值。捕获：capacity breakpoint、warm-up anomalies。
3. **Spike** — 突然提升到 3-10x RPS 持续 2 分钟，然后回落。捕获：autoscaling latency、queue saturation、cold-start impact。
4. **Soak** — steady-state 持续 4-8 小时。捕获：memory leaks、connection-pool drift、observability overflow。

### 2026 工具映射

**LLMPerf**（Anyscale）— Python，但 tokenization 由 Rust 支撑。Mean/stddev prompts。Streaming-aware。Performance runs 的默认最佳选择。

**NVIDIA GenAI-Perf** — NVIDIA 的 reference。使用 Triton client；metric coverage 全面。注意它的 ITL 不包含 TTFT；LLMPerf 的包含。同一 server 上，这两个工具会产生不同 TPOT。

**LLM-Locust**（TrueFoundry）— 修复 GIL trap 的 Locust extension。熟悉的 Locust DSL + streaming metrics。

**guidellm** — large-scale synthetic benchmarking。

**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**:
- k6 本身（Go、compiled、无 GIL）加入了 streaming-aware metrics。
- k6 Operator 使用 TestRun / PrivateLoadZone CRDs 做 Kubernetes-native distributed testing。
- 最适合 CI/CD gates 和 SLA testing。

**Vegeta** — Go，比 k6 更简单。Constant-rate HTTP saturation。不懂 LLM，但适合 gateway / rate-limit testing。

**Locust 2.43.3 stock** — 对 LLM 有 GIL trap。只有配合 LLM-Locust extension 才使用。

### CI 中的 SLA gate

在 PR 上运行 k6：

- baseline RPS 下每个 30-50 iterations。
- Gate：P50/P95 TTFT、5xx < 5%、TPOT 低于 threshold。
- 违反即 break the build。

### Realistic prompt distribution

从真实 traffic samples 构建（如果有），或者从公开 distributions 构建（例如 chat 用 ShareGPT prompts，code 用 HumanEval）。把 mean + stddev 喂给 LLMPerf。无论如何都避免 loop-with-one-prompt。

### 你应该记住的数字

- k6 Operator 1.0 GA：2025 年 9 月。
- k6 v2026.1.0：streaming-aware metrics。
- 典型 LLMPerf run：concurrency X 下 100-1000 requests。
- 典型 CI gate：每个 PR 30-50 iterations。
- 四种 patterns：steady、ramp、spike、soak。

## 实际使用

`code/main.py` 模拟带 realistic prompt distribution 的 load test，测量 effective TPOT，并演示 uniform-prompt trap。

## 交付成果

本课产出 `outputs/skill-load-test-plan.md`。给定 workload 和 SLA，它会选择工具并设计四种 load patterns。

## 练习

1. 运行 `code/main.py`。比较 uniform 与 realistic distribution —— 差距在哪里？
2. 为 CI gate 编写 k6 script：100 concurrent 下 TTFT P95 < 800 ms，runtime 5 分钟。
3. 你的 soak test 显示 memory 每小时增长 50 MB。说出三个原因，以及用于区分它们的 instrumentation。
4. Spike test 从 10 RPS 到 100 RPS。如果 Karpenter + vLLM production-stack 已就位（Phase 17 · 03 + 18），预期 recovery time 是多少？
5. GenAI-Perf 在同一 server 上报告 TPOT=6ms；LLMPerf 报告 TPOT=11ms。解释原因。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|------------|----------|
| LLMPerf | “the LLM harness” | Anyscale benchmark tool，streaming-aware |
| GenAI-Perf | “NVIDIA tool” | NVIDIA reference harness |
| LLM-Locust | “Locust for LLMs” | 修复 GIL trap 的 Locust extension |
| guidellm | “synthetic benchmark” | Large-scale synthetic tool |
| k6 Operator | “K8s k6” | 基于 CRD 的 distributed k6 |
| GIL trap | “Python client overhead” | Tokenization backlog 抬高报告 latency |
| Prompt-uniformity trap | “single-prompt lie” | 使用相同 prompt 的 loop 命中 cache，抬高 throughput |
| Steady-state | “constant load” | 持续 N 分钟的平坦 RPS |
| Ramp | “linear up” | 在一段时间内从 0 到 target |
| Spike | “burst test” | 突然乘倍后再回落 |
| Soak | “long test” | 用数小时检测 leaks |

## 延伸阅读

- [TianPan — Load Testing LLM Applications](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — Load Testing LLMs 2026](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — Introduction to LLM Inference Benchmarking](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)
