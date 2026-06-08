# 带 LMCache KV Offloading 的 vLLM Production Stack

> vLLM 的 production-stack 是 reference Kubernetes deployment：router、engines 和 observability 都接好。LMCache 是 KV-offloading layer，会把 KV cache 从 GPU memory 中抽出，并在 queries 和 engines 之间复用（先 CPU DRAM，再 disk/Ceph）。vLLM 0.11.0 KV Offloading Connector（2026 年 1 月）通过 Connector API（v0.9.0+）让它 asynchronous 且 pluggable。Offload latency 不面向用户。LMCache 即使没有 shared prefixes 也有价值：当 GPU 用完 KV slots 时，preempted requests 可以从 CPU restore，而不是 recomputing prefill。4 个 a3-highgpu-4g 上 16x H100（80GB HBM）的 published benchmarks：当 KV cache 超过 HBM 时，native CPU offload 和 LMCache 都显著提高 throughput；低 KV footprint 下，所有 configs 都匹配 baseline，只有小 overhead。

**类型：** 学习
**语言：** Python (stdlib, toy KV-spill simulator)
**先修：** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 06 (SGLang/RadixAttention)
**时间：** ~60 分钟

## 学习目标

- 画出 vLLM production-stack layers：router、engines、KV offload、observability。
- 解释 KV Offloading Connector API（v0.9.0+），以及 0.11.0 asynchronous path 如何隐藏 offload latency。
- 量化 LMCache CPU-DRAM 何时有帮助（KV > HBM）vs 何时增加 overhead（KV 小到能放进 HBM）。
- 给定 deployment constraints，在 native vLLM CPU offload 和 LMCache connector 之间选择。

## 要解决的问题

你的 vLLM serving 显示 GPUs 在 concurrency 上升时达到 100% HBM，并发生 preemption events。Requests 被 evicted、requeued，然后你在一分钟内对同一个 2K-token prompt 重新 prefill 四次。GPU compute 花在 redundant prefills 上；goodput 远低于 raw throughput。

增加 GPUs 的成本是线性的。增加 HBM 不可能。但 CPU DRAM 很便宜：一个 socket 有 512 GB+，latency 比 HBM 差几个数量级，但对 “temporarily warm” KV cache 足够。

LMCache 将 KV cache 抽到 CPU DRAM，使 preempted requests 快速恢复，并让跨 engines 的 repeated prefixes 能共享 cache，而不是每个 engine 重新 prefill。

## 核心概念

### vLLM production-stack

`github.com/vllm-project/production-stack` 是 reference Kubernetes deployment：

- **Router**：cache-aware（Phase 17 · 11）。消费 KV events。
- **Engines**：vLLM workers。每 GPU 一个，或每 TP/PP group 一个。
- **KV cache offload**：LMCache deployment 或 native connector。
- **Observability**：Prometheus scrape、Grafana dashboards、OTel traces。
- **Control plane**：service discovery、config、rolling updates。

以 Helm chart + operator 交付。

### KV Offloading Connector API（v0.9.0+）

vLLM 0.9.0 引入 Connector API，用于 pluggable KV cache backends。engine 将 blocks offload 到 connector；connector 存储它们（RAM、disk、object storage、LMCache）。request 需要 block 时，connector 将它 load back。

vLLM 0.11.0（2026 年 1 月）添加 asynchronous offload path：common case 下，offload 可以后台发生，让 engine 不阻塞。end-to-end latency 和 throughput 仍依赖 workload shape、KV cache hit rate 和 system pressure；vLLM 自己的 notes 指出 custom-kernel offload 在 low hit rates 下可能 degrade throughput，并且 async scheduling 与 speculative decoding 存在已知 interaction issues。

### Native CPU offload vs LMCache

**Native vLLM CPU offload**：engine-local。将 KV blocks 存在 host RAM。实现快，无 network hop。不跨 engines。

**LMCache connector**：cluster-scale。将 blocks 存在 shared LMCache server（CPU DRAM + Ceph/S3 tier）。任意 engine 都可访问 blocks。有 16x H100 benchmarks published。

单个 engine 有 HBM pressure 时选 native。多个 engines 共享 prefixes 时选 LMCache（带 common system prompts 的 RAG、带 shared templates 的 multi-tenant）。

### Benchmark behavior

跨 4 个 a3-highgpu-4g 的 16x H100（80 GB HBM）测试：

- Low KV footprint（short prompts、low concurrency）：所有 configs 匹配 baseline，LMCache 增加约 3-5% overhead。
- Moderate footprint：LMCache 开始在跨 engines prefix reuse 上有帮助。
- KV exceeds HBM：native CPU offload 和 LMCache 都显著改善 throughput；LMCache gain 更大，因为 cross-engine sharing。

### LMCache 何时 decisive

- Multi-tenant serving，其中 system prompts 跨 tenants 共享。
- RAG，其中 document chunks 跨 queries 重复。
- 同一 base 上的 fine-tuned variants（LoRA），base-model KV reuse 减少 redundant work。
- Preemption-heavy workloads：从 CPU restore 比 re-prefill 更便宜。

### 何时不要 enable

- Small HBM pressure：你支付 overhead，却无 benefit。
- Short contexts（<1K tokens）：transfer time > re-prefill。
- Single-tenant single-prompt workload：没有 reuse 可捕获。

### 与 disaggregated serving 集成

Phase 17 · 17 disaggregated serving + LMCache 会 compound：prefill pool 到 decode pool 的 KV transfers 如果未使用，会落入 LMCache；subsequent queries 从 LMCache 拉取。Phase 17 · 11 cache-aware router 可以 route 到 local 或 LMCache-shared cache 匹配的 engine。

### 你应该记住的数字

- vLLM 0.9.0：Connector API shipped。
- vLLM 0.11.0（Jan 2026）：asynchronous offload path；end-to-end latency impact 取决于 workload、KV hit rate 和 system pressure（不是绝对 guarantee）。
- 16x H100 benchmark：KV footprint 超过 HBM 时，LMCache 有帮助。
- Small HBM pressure：3-5% overhead，无 benefit。

## 实际使用

`code/main.py` 模拟有无 LMCache 的 preemption-heavy workload。报告 avoided re-prefills、throughput gain 和 break-even HBM utilization。

## 交付成果

本课产出 `outputs/skill-vllm-stack-decider.md`。给定 workload shape 和 vLLM deployment，决定 native vs LMCache vs neither。

## 练习

1. 运行 `code/main.py`。什么 HBM utilization 下 LMCache 开始划算？
2. 一个 tenant 每小时 200 queries 共享一个 6K-token system prompt。计算每个 tenant 的 expected LMCache savings。
3. LMCache server 是 single point of failure。设计 HA strategy（replicas、fallback to native）。
4. LMCache 存到 spinning disk 上的 Ceph。70B FP8 上 4K-token KV（500 MB）的 read time vs re-prefill 是多少？
5. 论证 vLLM 0.11.0 asynchronous path 是否 “free”：overhead 藏在哪里？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Production-stack | “the reference deployment” | vLLM 的 Kubernetes Helm chart + operator |
| Connector API | “KV backend interface” | vLLM 0.9.0+ pluggable KV store interface |
| Native CPU offload | “engine-local spill” | 将 KV 存储在 same engine 的 host RAM |
| LMCache | “cluster KV cache” | 位于 CPU DRAM + disk 上的 cross-engine KV cache server |
| 0.11.0 async | “non-blocking offload” | 隐藏在 engine stream 后的 offload |
| Preemption | “evict to make room” | HBM full 时的 KV cache shuffle |
| Prefix reuse | “same system prompt” | 多个 queries 共享开头；cache hit |
| Ceph tier | “disk tier” | cache hierarchy 中 DRAM 下方的 durable storage |

## 延伸阅读

- [vLLM Blog — KV Offloading Connector (Jan 2026)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack) — Helm chart + operator。
- [LMCache for Enterprise-Scale LLM Inference (arXiv:2510.09665)](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache) — Connector implementation。
- [vLLM 0.11.0 release notes](https://github.com/vllm-project/vllm/releases) — asynchronous path details。
