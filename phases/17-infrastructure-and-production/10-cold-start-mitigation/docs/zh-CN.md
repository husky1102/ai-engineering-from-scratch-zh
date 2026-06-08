# Serverless LLMs 的 Cold Start Mitigation

> 一个 20 GB model image 从冷态到 serving，7B 需要 5-10 分钟，70B 需要 20+ 分钟。在真正 serverless 的世界里，这不是 warm-up，而是 outage。缓解手段运行在五个层次：pre-seeded node images（AWS 上的 Bottlerocket、dual-volume arch）、model streaming（NVIDIA Run:ai Model Streamer、vLLM 原生支持）、GPU memory snapshots（Modal checkpoints，restart 最多快 10x）、warm pools（`min_workers=1`）、tiered loading（ServerlessLLM 的 NVMe→DRAM→HBM pipeline，latency 降低 10-200x），以及移动 input tokens（KB）而不是 KV cache（GB）的 live migration。Modal 发布的 cold starts 下限为 2-4s；Baseten 默认 5-10s，pre-warming 可做到 sub-second。本课教你测量、预算并叠加这五层。

**类型：** Learn
**语言：** Python (stdlib, toy cold-start path simulator)
**先修：** Phase 17 · 02 (Inference Platform Economics), Phase 17 · 03 (GPU Autoscaling)
**时间：** ~60 minutes

## 学习目标

- 枚举 cold-start mitigation 的五层，并为每层说出一个工具或模式。
- 将 70B 模型的总 cold-start time 计算为 (node provision) + (weights download) + (weights load into HBM) + (engine init)。
- 解释为什么 live migration 传输 input tokens（KB）而不是 KV cache（GB），以及代价是什么（recomputation）。
- 说出 warm-pool 取舍（为 idle GPU 付费，或接受 cold-start tail），以及 `min_workers > 0` 变成必选的 SLA 阈值。

## 要解决的问题

你的 serverless LLM endpoint 在夜间 scale to zero。早上 8 点流量暴涨。第一个请求需要等待：

1. Karpenter provision 一个 GPU node：45-60s。
2. Container 拉取带 weights 的 30 GB image：120-300s。
3. Engine 将 weights 加载进 HBM：45-120s，取决于 model size 和 storage speed。
4. vLLM 或 TRT-LLM 初始化 CUDA graphs、KV cache pool、tokenizer：10-30s。

总计：220-510s（约 3-8 分钟），才返回第一个 token。你的 SLA 是 2s。你上线一个 warm-pool（`min_workers=1`），问题看起来消失了，但现在你要为一张 idle GPU 24x7 付费。如果服务有 5 个产品，每个都有一个 warm replica，那就是 5 × 24 × 30 = 3,600 GPU-hours/month，无论是否有任何用户调用。

Cold-start mitigation 的目标是：尽量保留 serverless economics，同时接近 always-on 的 latency。

## 核心概念

### Layer 1：pre-seeded node images（Bottlerocket）

在 AWS 上，Bottlerocket 的 dual-volume architecture 将 OS 与 data 分离。用已经 pre-pulled container image 的 data volume 制作 snapshot，并在你的 `EC2NodeClass` 中引用 snapshot ID。新节点启动时，weights 已经在 local NVMe 上，步骤 2 和步骤 3 的一部分消失。它与 Karpenter 原生配合。典型节省：大模型每次 cold start 节省 2-4 分钟。

GCP 上的等价做法：带 pre-baked container layers 的 custom VM images。Azure 上：使用同样模式的 managed disk snapshots。

### Layer 2：model streaming（Run:ai Model Streamer）

不是等整个文件加载完才回答第一个请求，而是逐层把 weights stream 到 GPU memory 中，只要第一个 transformer block resident 就开始处理。NVIDIA Run:ai Model Streamer 在 vLLM 2026 中原生提供。支持 S3、GCS 和 local NVMe。它通过将 I/O 与 compute setup 重叠，把大模型的 weight-load time 大约减半。

### Layer 3：GPU memory snapshots（Modal）

Modal 在第一次加载后对 GPU state（weights、CUDA graphs、KV cache region）做 checkpoint。后续 restarts 直接 deserialize into HBM，比重新初始化快 10x。这最接近“2 秒启动一张 warm GPU”。取舍是：snapshots 与 GPU topology 绑定，所以如果 Karpenter 把你迁移到不同 SKU，就需要重新 checkpoint。

### Layer 4：warm pools（min_workers=1）

最简单的缓解：始终保持一个 replica ready。成本是一个 GPU 的 hourly rate 24x7。这个算术对小模型很残酷（你每小时花 $0.85-$1.50 来避免 30s cold start），对大模型比较友好（每小时 $4 避免 5 分钟 cold start）。Warm pools 成为必需的 SLA 阈值：通常是 70B+ 模型上的 TTFT P99 < 60s。

### Layer 5：tiered loading（ServerlessLLM）

ServerlessLLM 把存储视为层级：NVMe（快且大）、DRAM（中等但可分层）、HBM（小但即时）。Weights 会 pre-loaded 到 DRAM，并按需 load into HBM。论文报告相对 naive disk-to-HBM，cold loads 的 latency reduction 为 10-200x。生产采用仍处早期，但已有与 vLLM 的 integrations。

### Layer 6：live migration（bonus pattern）

当节点不可用（spot eviction、node drain）时，传统模式是 cold-start 另一个 replica，然后 drain request queue。Live migration 会把 input tokens（kilobytes）移动到已经加载模型的 destination，并在 destination 上 recompute KV cache。相比通过网络传输 GB 级 KV cache，recomputation 更便宜。适用于 disaggregated deployments。

### Warm-pool 算术

对于 P99 TTFT SLA 为 2s 的服务，问题不是“要不要 warm pool”，而是“需要多少 warm replicas，以及哪些路径获得它们”。

- High-value interactive paths（live chat、voice agent）：`min_workers=1-2`。
- Background batch paths（nightly classification）：接受 scale-to-zero，可容忍 5-10 分钟 cold start。
- Premium tier：每个 tenant 用 `min_workers` 提供 dedicated capacity。

### 先测量再优化

一台 fresh node 上 70B 模型的 cold-start anatomy（示例）：

| Phase | Time | Mitigation |
|-------|------|-----------|
| Node provision | 50s | Bottlerocket + pre-seeded image, warm pool |
| Image pull | 180s | Pre-seeded data volume (eliminate) |
| Weights to HBM | 75s | Model streamer (halve); GPU snapshot (eliminate) |
| Engine init | 20s | Persistent CUDA graph cache |
| First forward | 3s | Min inherent latency |
| **Total cold** | **328s** | |
| **Total with mitigations** | **~15s** | 22x reduction |

### 你应该记住的数字

- Modal cold start: 2-4s（with GPU snapshots）。
- Baseten default cold start: 5-10s；pre-warming 后 sub-second。
- Raw 70B cold start: 3-8 分钟。
- Run:ai Model Streamer: ~2x weight-load speedup。
- ServerlessLLM tiered loading: 10-200x latency reduction（paper numbers）。

## 实际使用

`code/main.py` 会对有无每项 mitigation 的 cold-start path 建模。它报告 total cold-start time、warm-pool cost，以及 warm pool 比承受 cold-start tax 更划算的 break-even request rate。

## 交付成果

本课产出 `outputs/skill-cold-start-planner.md`。给定 SLA、model size 和 traffic shape，它会选择要叠加哪些 mitigations。

## 练习

1. 运行 `code/main.py`。计算超过哪个 break-even request rate 后，warm replica 会比通过额外 SLO 请求丢失来支付 cold-start tax 更便宜。
2. 你部署了一个 13B 模型，P99 TTFT SLA 为 3s。选择能达到它的 minimum mitigation stack（层数最少）。
3. Bottlerocket pre-seeding 消除了 image pull，但 weights 仍需从 snapshot 加载到 HBM。如果 snapshot-backed NVMe 读取速度为 7 GB/s，计算 70B 模型的 wall-clock。
4. 你的 serverless provider 提供 GPU snapshots（Modal），团队拒绝的理由是“snapshots leak PII”。为双方论证：真实风险是什么，缓解措施是什么（ephemeral snapshots、encryption、namespace isolation）？
5. 设计 tiered warm-pool policy：paid users、trial users 和 batch workloads 分别需要多少 warm replicas？展示计算。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Cold start | “the big pause” | Fresh replica 上从请求到第一个 token 的时间 |
| Warm pool | “always-on minimum” | `min_workers >= 1`，保持至少一个 replica ready |
| Pre-seeded image | “baked AMI” | Container weights 已经预驻留的 node image |
| Bottlerocket | “AWS node OS” | AWS container-optimized OS，支持 dual-volume snapshot |
| Model streamer | “streaming load” | 将 weights I/O 与 compute setup 重叠 |
| GPU snapshot | “checkpoint to HBM” | 序列化 post-load GPU state；restart 时反序列化 |
| Tiered loading | “NVMe + DRAM + HBM” | 存储层级；按需加载 |
| Live migration | “move tokens” | 传输 input（KB），在 destination 上 recompute KV |
| `min_workers` | “warm replicas” | Serverless minimum keep-alive count |
| Scale-to-zero | “full serverless” | Idle 时无成本；接受完整 cold-start tax |

## 延伸阅读

- [Modal — Cold start performance](https://modal.com/docs/guide/cold-start) — Modal 发布的 benchmarks 和 checkpoint architecture。
- [AWS Bottlerocket](https://github.com/bottlerocket-os/bottlerocket) — pre-seeded data volume snapshot pattern。
- [NVIDIA Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer) — 将 weights load 与 compute setup 重叠。
- [Baseten — Cold-start mitigation](https://www.baseten.co/blog/cold-start-mitigation/) — pre-warming playbook。
- [ServerlessLLM paper (USENIX OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/fu) — tiered loading design。
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — disaggregated deployments 的 live migration。
