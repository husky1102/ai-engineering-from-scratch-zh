# Multi-Region LLM Serving 与 KV Cache Locality

> 对 cached LLM inference 来说，round-robin load balancing 是有害的。没有落到持有其 prefix 的节点上的请求，会支付完整 prefill cost：长 prompt 上 P50 约 800 ms，而 cache hit 约 80 ms。2026 年的生产模式是 cache-aware router（Rust 写的 vLLM Router、llm-d router），它消费 KV-cache events，并基于 prefix-hash match 路由。近期研究（GORGO）把 cross-region network latency 显式放入 routing objective。商业 “cross-region inference” 产品（Bedrock cross-region inference、GKE multi-cluster gateways）把 inference 视为黑盒：它们处理 availability，而不是 TTFT。JPMorgan 和 Mayo Clinic 在 2024 年 11 月演练 us-east-1 failover，约 22 分钟恢复。DR 的现实是：32% 的 LLM DR failures 来自团队备份了 weights，却忘了 tokenizer files 或 quantization configs。

**类型：** Learn
**语言：** Python (stdlib, toy prefix-cache-aware router simulator)
**先修：** Phase 17 · 04 (vLLM Serving), Phase 17 · 06 (SGLang RadixAttention)
**时间：** ~60 minutes

## 学习目标

- 解释为什么 round-robin load balancing 会破坏 cached inference，并量化 TTFT penalty。
- 画出 cache-aware router：inputs（KV-cache events）、algorithm（prefix-hash match）、tie-breaker（GPU utilization）。
- 说出 32% LLM DR failure driver（missing tokenizer files / quantization configs），并给出三文件 DR checklist。
- 区分商业 cross-region offerings（Bedrock CRI、GKE Multi-Cluster Gateway）与 KV-aware routing。

## 要解决的问题

你的服务运行在 us-east-1、us-west-2 和 eu-west-1。你在前面放了一个 ALB 做 round-robin。生产中的 prefix cache hit rate 降到 8%。TTFT P50 翻了三倍。vLLM logs 显示每个请求都在支付完整 prefill cost。

Round-robin 对 stateless services 是最优的。LLM inference 天生就是 stateful：KV cache 编码了模型已经看过的一切。盲目路由就是路由到错误 cache。

另外，你的团队有 DR plan。你把 model weights cross-region 备份到 S3。区域 outage 发生；你尝试 failover；replica 拒绝启动。你忘了 tokenizer.json、quantization config 和 RoPE scaling config 都在另一个没有同步的 bucket 里。

Multi-region LLM serving 是 cache problem、routing problem 和 DR-hygiene problem，不是 load-balancer problem。

## 核心概念

### Cache-aware routing

请求携带 prompt 到达。Router 对 prefix 做 hash（比如前 512 tokens）；它询问每个 replica：“你缓存了这个 prefix 吗？”Replicas 在分配和驱逐 blocks 时，通过 pub/sub channel 发布 KV-cache events。Router 选择命中的 replica；如果没人命中，则回退到基于 GPU-util 的 tie-breaker。

**vLLM Router**（Rust，2026 production-stack）：订阅 `kv.cache.block_added` events，维护 prefix-hash → replica index，用 O(1) lookup 做路由。没有匹配时回退到 least-queue-depth。

**llm-d router**：同样模式，Kubernetes-native。通过 ControlPlane API 发布 events。

**SGLang RadixAttention**（Phase 17 · 06）是 intra-replica 的等价物。Cross-replica routing 严格位于它的上游。

### 数字

2K-token prompt、Llama 3.3 70B FP8、H100 上的 TTFT P50：
- Cache hit（同一 replica，prefix resident）：~80 ms。
- Cache miss（cold prefill）：~800 ms。

10x 差距。如果你的 router 在 replicas 之间达到 60-80% 的 prefix cache hit，你就能在 N-replica capacity 上接近 single-replica performance。如果只有 10%，你接近 naive scaling。

### Cross-region 有新的约束：network latency

Inter-region RTT：
- us-east-1 ↔ us-west-2: ~65 ms。
- us-east-1 ↔ eu-west-1: ~75 ms。
- us-east-1 ↔ ap-southeast-1: ~220 ms。

如果 routing 把来自 us-east-1 的请求送到 ap-southeast-1 上的 hot prefix，节省的 prefill（800 → 80 ms）会被 440 ms round-trip 抵消。GORGO（2026 research）把这点显式化：联合最小化 `prefill_time + network_latency`，而不是只最小化 prefill。通常答案是保持 regional routing，除非 prefix 极大、达到 multi-MB 级以至于 prefill 占主导。

### 商业 “cross-region inference” 对这里没有帮助

AWS Bedrock cross-region inference 会在 capacity pressure 下自动把请求路由到其他区域。它优化 availability，不优化 TTFT，并把 inference 视为黑盒。GKE Multi-Cluster Gateway 也是一样：service-level failover，没有 KV cache 意识。

即使用这些产品，你仍然需要 app-layer cache-aware router。它们处理“us-east-1 起火”的情况。Cache-aware routing 处理 TTFT 情况。

### DR hygiene：32% missing-files problem

广泛引用的 2026 年统计：32% 的 LLM DR failures 是因为团队备份了 weights，却忘了：

- `tokenizer.json` 或 `tokenizer.model`
- Quantization configs（`quantize_config.json`、AWQ scales、GPTQ zero-points）
- Model-specific configs（RoPE scaling、attention masks、chat templates）
- Engine config（`vllm_config.yaml`、sampling defaults、LoRA adapter manifests）

修复方式是最小三文件 DR manifest：

1. HF model repo 下的所有文件（weights + configs + tokenizer）。
2. Engine-specific serving config。
3. Deployment manifest（K8s YAML、Dockerfile、dependency lock）。

另外：每季度进行 DR drill。2024 年 11 月 JPMorgan 的 us-east-1 drill 能达到 22 分钟恢复，只是因为 playbook 已经排练过。

### Data residency 是正交问题

EU customer PHI 不能离开 EU。如果你的 cache-aware router 为了 prefix match 把 Paris-originated request 发到 us-east-1，无论 TTFT 增益如何，你都违反了 GDPR。优化 cache 之前，先按 residency boundary 分区 routers。

### 你应该记住的数字

- Cache hit vs miss TTFT gap: ~10x（2K prompt 上 80 ms vs 800 ms）。
- Inter-region RTT US-EU: ~75 ms。
- DR failure: 32% 漏掉 tokenizer/quant configs。
- JPMorgan us-east-1 failover Nov 2024: 22 minutes（30-min SLA）。

## 实际使用

`code/main.py` 会在 multi-region workload 上模拟三种 routing strategies（round-robin、cache-aware regional、cache-aware global）。它报告 cache hit rate、TTFT P50/P99 和 cross-region bill。

## 交付成果

本课产出 `outputs/skill-multi-region-router.md`。给定 regions、residency constraints 和 SLA，它会设计 routing plan。

## 练习

1. 运行 `code/main.py`。在 75 ms RTT 下，prompt length 达到多少时 cross-region routing 才会胜过 local-only routing？
2. 你的 cache hit rate 从 70% 降到 12%。诊断三个可能原因，并指出能确认每个原因的 observables。
3. 为一个在 vLLM 中 serving、带 5 个 LoRA adapters、AWQ-quantized 的 70B 模型设计 DR manifest。列出每个 file 和 config。
4. 论证 Bedrock cross-region inference 对一个有严格 TTFT SLOs 的 fintech 是否“足够”。引用具体行为。
5. 一个 Paris-origin request 在 us-east-1 匹配到 prefix。你会路由过去吗？写出 policy。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Cache-aware routing | “smart LB” | 基于 prefix-hash match 路由到持有 KV cache 的 replica |
| KV-cache events | “cache pub-sub” | Replicas 发布 block add/evict；router 建索引 |
| Prefix hash | “cache key” | 前 N tokens 的 hash，用作 router lookup |
| GORGO | “cross-region routing research” | arXiv 2602.11688；把 network latency 作为显式项 |
| Cross-region inference | “Bedrock CRI” | AWS 产品；availability failover，而非 TTFT awareness |
| DR manifest | “the backup list” | restore 所需的每个文件，不只是 weights |
| Data residency | “GDPR boundary” | 哪个 region 可以看到 user data 的法律约束 |
| RTT | “round-trip time” | Network latency；US-EU 75 ms，US-APAC 220 ms |
| LLM-aware LB | “cache-hit LB” | 作为产品类别的 cache-aware router |

## 延伸阅读

- [BentoML — Multi-cloud and cross-region inference](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference)
- [arXiv — GORGO (2602.11688)](https://arxiv.org/html/2602.11688v1) — 带 network latency 项的 cross-region KV-cache reuse。
- [TianPan — Multi-Region LLM Serving Cache Locality](https://tianpan.co/blog/2026-04-17-multi-region-llm-serving-data-residency-routing)
- [AWS Bedrock Cross-Region Inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) — availability failover documentation。
- [vLLM Production Stack Router](https://github.com/vllm-project/production-stack) — cache-aware router source。
