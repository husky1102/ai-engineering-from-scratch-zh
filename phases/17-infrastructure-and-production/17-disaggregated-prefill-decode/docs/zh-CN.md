# Disaggregated Prefill/Decode：NVIDIA Dynamo 与 llm-d

> Prefill 是 compute-bound；decode 是 memory-bound。把二者运行在同一 GPU 上会浪费一种资源。Disaggregation 将它们拆到独立 pools，并通过 NIXL（RDMA/InfiniBand 或 TCP fallback）在二者之间传输 KV cache。NVIDIA Dynamo（GTC 2025 announce, 1.0 GA）位于 vLLM/SGLang/TRT-LLM 之上：其 Planner Profiler + SLA Planner 会自动 rate-match prefill:decode ratios 以满足 SLOs。NVIDIA 发布的 throughput gains 大致在这个范围：developer.nvidia.com（2025-06）展示了 GB200 NVL72 + Dynamo 上 DeepSeek-R1 MoE 在 medium-latency regime 中约 6x improvement，Dynamo product page（developer.nvidia.com, undated）宣称 GB300 NVL72 + Dynamo vs Hopper 上最高 50x MoE throughput。“30x” 数字是 full-stack Blackwell + Dynamo + DeepSeek-R1 reports 的 community aggregate；我们没有找到一个 primary source 精确写着 30x，所以把它当作 directional claim。llm-d（Red Hat + AWS）是 Kubernetes-native：prefill / decode / router 作为 independent Services，带 per-role HPA。llm-d 0.5 添加 hierarchical KV offloading、cache-aware LoRA routing、UCCL networking、scale-to-zero。Economics：多个 customer disclosures 的 internal rollup 表明，在 constant SLA 下从 colocated serving 切换到 Dynamo disaggregated，$2M-class inference spend 可节省 30-40%（即 $600-800K/year）；具体 $2M→$600-800K 是 internal composite，不是单个 published case study：把它当 order-of-magnitude anchor，不要当 reference citation。短 prompts（<512 tokens、short output）不值得付 transfer cost。

**类型：** 学习
**语言：** Python (stdlib, toy disaggregated-vs-colocated simulator)
**先修：** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 08 (Inference Metrics)
**时间：** ~75 分钟

## 学习目标

- 解释 prefill 和 decode 为什么有不同 optimal GPU allocations，并量化 colocation 下的 waste。
- 画出 disaggregated architecture：prefill pool、decode pool、通过 NIXL 传输 KV、router。
- 说出 disaggregation **不**划算的条件（short prompts、short outputs）。
- 区分 NVIDIA Dynamo（stack-above）与 llm-d（Kubernetes-native），并将每个匹配到 operational context。

## 要解决的问题

你在 8 张 H100 上运行 Llama 3.3 70B。mixed workload（long prompts + short outputs）下，GPUs 在 decode 时 idle，因为大部分 compute 花在 prefill 上。另一个 workload（short prompts + long outputs）则相反。colocated prefill + decode 意味着两边都 over-provision。

预算影响：20-40% GPU time 浪费在错误资源上。你在购买 H100 compute 去跑 memory-bound decode，或购买 H100 HBM bandwidth 去跑 compute-bound prefill。二者都是昂贵浪费。

Disaggregation 将 prefill 和 decode 拆到单独 pools，并按各自 bottleneck sizing。KV cache 通过 high-bandwidth interconnect 从 prefill pool 传到 decode pool。

## 核心概念

### 为什么 bottlenecks 不同

**Prefill**：对完整 input prompt 一次 forward 跑 transformer。Matrix multiplications dominate；compute-bound。H100 FP8 提供约 2000 TFLOPS useful throughput。Batch efficiency 好：一次 forward 处理许多 tokens。

**Decode**：一次生成一个 token，每次 iteration 都读取完整 weights。Memory-bandwidth-bound。HBM3 提供约 3 TB/s。Batch efficiency 只有在 high concurrency 下才好：weights read 被 batch 摊薄。

把二者 colocate：你购买同时 optimized for both 的 GPUs。H100 两者都擅长，但不管哪种用法成本相同。scale 下，你希望 prefill pool 用 H100 / compute-heavy；decode pool 用 H200 / memory-heavy，或 aggressive quantization。

### Architecture

```text
            ┌──────────────┐
  Request → │    Router    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ (prompt only)                  │
            ┌──────────────┐    KV cache    ┌───────▼──────┐
            │ Prefill pool │ ─── NIXL ────► │ Decode pool  │
            │  (compute)   │                │  (memory)    │
            └──────────────┘                └──────┬───────┘
                                                   │ tokens
                                                   ▼
                                                 Client
```

NIXL 是 NVIDIA 的 inter-node transport。有 RDMA/InfiniBand 时使用它，否则 TCP fallback。Transfer latency 是真实成本：70B FP8 上 4K-token prompt 的 KV cache 通常 20-80 ms。这就是为什么 short prompts 不值得 disaggregation：transfer tax 超过 savings。

### Dynamo vs llm-d

**NVIDIA Dynamo**（GTC 2025 announce, 1.0 GA）：
- 位于 vLLM、SGLang、TRT-LLM 之上作为 orchestrator。
- Planner Profiler 测量 workload，SLA Planner 自动配置 prefill:decode ratios。
- Rust core，Python extensibility。
- Throughput gains：NVIDIA 报告 GB200 NVL72 + Dynamo 上 DeepSeek-R1 MoE 在 medium-latency regime 中 6x（developer.nvidia.com, 2025-06）；community reports 中 full Blackwell + Dynamo + DeepSeek-R1 stacks 的 “up to 30x” 缺少单个 primary source，应作为 directional。
- GB300 NVL72 + Dynamo：Dynamo product page（developer.nvidia.com, undated）称相对 Hopper 最高 50x MoE throughput。

**llm-d**（Red Hat + AWS，Kubernetes-native）：
- Prefill / decode / router 作为 independent Kubernetes Services。
- per-role HPA，signals 为 queue depth（prefill）/ KV utilization（decode）。
- `topologyConstraint packDomain: rack` 将 prefill+decode cliques pack 到同 rack，以获得 high-bandwidth KV transfer。
- llm-d 0.5（2026）：hierarchical KV offloading、cache-aware LoRA routing、UCCL networking、scale-to-zero。

如果你想要 managed stack-above orchestrator，用 Dynamo。如果你想要 Kubernetes-native primitives，并且已经投入 CNCF ecosystem，用 llm-d。

### Economics

Internal composite（不是单个 published case study：order-of-magnitude anchor）：

- colocated serving 年 inference spend $2M。
- 切换到 Dynamo disaggregated。
- 相同 request volume，相同 P99 latency SLA。
- reported savings：$600K-$800K/year（30-40% reduction）。
- 无新 hardware。

我们从多个 customer disclosures 综合这个数字，而不是从单个 citable case study 获取；最接近的 published data point 是 Baseten 的 2x faster TTFT / 61% higher throughput with Dynamo KV routing（baseten.co, 2025-10），以及 VAST + CoreWeave 预测在 40-60% KV hit rate 下 tokens/$ 增加 60-130%（vastdata.com, 2025-12）。savings 来自 right-sizing each pool；prefill-heavy workloads（带 8K+ prefixes 的 RAG）比 balanced ones 更受益。

### 何时不要 disaggregate

- Prompts < 512 tokens 且 outputs < 200 tokens：transfer tax dominates gain。
- Small cluster（< 4 GPUs）：pool diversity 不足。
- 团队无法运行两个 GPU pools 并做 per-role scaling：Dynamo 有帮助，但不是 trivial。
- 无 RDMA fabric：TCP transfer tax 更重。

### Router 与 Phase 17 · 11 集成

Disaggregated routers 是 KV-cache-aware（Phase 17 · 11）。request 会落到持有其 prefix 的 decode pool；如果没有 match，它走 prefill → decode。Hit rate 与 disaggregation 会 compound：cache-aware router 决定是否甚至需要新的 prefill。

### MoE on Blackwell 才是真正数字所在

GB300 NVL72 + Dynamo 显示相对 Hopper baselines 50x MoE throughput。MoE expert routing 在 prefill 上 compute-heavy，但在 decode 上 memory-heavy（expert caches），所以 disaggregation 是 double win。2026 frontier model serving 以 MoE 为主（DeepSeek-V3、future GPT-5 variants）。

### 你应该记住的数字

Benchmark numbers 会漂移：NVIDIA 和 inference stack 每季度都会发布更新结果。引用前重新检查。

- DeepSeek-R1 on GB200 NVL72 + Dynamo：medium-latency regime 中相对 baseline 约 6x throughput（developer.nvidia.com, 2025-06）；full Blackwell + Dynamo stacks 上的 community “up to 30x” claims 是没有单个 primary source 的 directional aggregates。
- GB300 NVL72 + Dynamo：相对 Hopper 最高 50x MoE throughput（developer.nvidia.com, undated）。
- Savings anchor（internal composite，不是单个 case study）：constant SLA 下，$2M annual spend 可节省 $600-800K/year。
- Disaggregation threshold：prompts >512 tokens + outputs >200 tokens。
- KV transfer via NIXL：70B FP8 上 4K-prompt KV 为 20-80 ms。

## 实际使用

`code/main.py` 模拟 colocated vs disaggregated serving。报告 throughput、cost per request 和 prompt-length crossover。

## 交付成果

本课产出 `outputs/skill-disaggregation-decider.md`。给定 workload 和 cluster，判断是否 disaggregate。

## 练习

1. 运行 `code/main.py`。什么 prompt length 下 disaggregation 胜过 colocation？
2. 为 P99 prefix length 8K、output 300 的 RAG service 设计 prefill pool 和 decode pool。
3. Dynamo vs llm-d：为一家 pure-Kubernetes shop 且没有 Python runtime preference 的团队选择一个。
4. 计算 KV transfer cost：70B FP8 上 4K prefill 约 500 MB KV。RDMA 100 GB/s 下 transfer = 5 ms；TCP 10 GB/s = 50 ms。哪个会影响你的 SLA？
5. MoE expert routing 会改变 KV access patterns。每个 token 激活不同 experts 的 MoE 中，disaggregation 行为如何？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Disaggregated serving | “split prefill/decode” | 为每个 phase 使用 separate GPU pools |
| NIXL | “NVIDIA transport” | Dynamo 的 inter-node KV transfer（RDMA/TCP） |
| NVIDIA Dynamo | “the orchestrator” | vLLM/SGLang/TRT-LLM 的 stack-above coordinator |
| llm-d | “Kubernetes native” | Red Hat + AWS K8s disaggregated stack |
| Planner Profiler | “Dynamo auto-config” | 测量 workload，配置 pool ratios |
| SLA Planner | “Dynamo policy” | 自动 rate-match prefill:decode 以满足 SLOs |
| `packDomain: rack` | “llm-d topology” | 将 prefill+decode pack 在同 rack，以加速 KV |
| UCCL | “unified collective” | llm-d 0.5 用于 scale-to-zero 的 networking layer |
| MoE expert routing | “expert per token” | DeepSeek-V3 pattern；disaggregation 有帮助 |

## 延伸阅读

- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM Disaggregated Serving blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 release notes](https://github.com/llm-d/llm-d/releases)
