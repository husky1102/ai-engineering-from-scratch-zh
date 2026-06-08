# Kubernetes 上的 GPU Autoscaling：Karpenter、KAI Scheduler、Gang Scheduling

> 三层，不是一层。Karpenter 动态 provision nodes（低于一分钟，比 Cluster Autoscaler 快 40%）。KAI Scheduler 处理 gang scheduling、topology awareness 和 hierarchical queues，避免 7-of-8 partial allocation trap：七个 nodes 等待并消耗成本，只差一个 GPU。Application-level autoscalers（NVIDIA Dynamo Planner、llm-d Workload Variant Autoscaler）基于 inference-specific signals 扩缩：queue depth、KV cache utilization，而不是 CPU/DCGM duty cycle。经典 HPA 陷阱在于 `DCGM_FI_DEV_GPU_UTIL` 是 duty-cycle measurement：100% 可能是 10 个 requests，也可能是 100 个。vLLM 会预分配 KV cache memory，因此 memory 永远不会触发 scale-down。本课教你组合三层，并避免默认 Karpenter `WhenEmptyOrUnderutilized` policy 在推理中途终止运行中的 GPU jobs。

**类型：** Learn
**语言：** Python (stdlib, toy queue-depth autoscaler simulator)
**先修：** Phase 17 · 02 (Inference Platform Economics), Phase 17 · 04 (vLLM Serving Internals)
**时间：** ~75 minutes

## 学习目标

- 画出三层 autoscaling（node provisioning、gang scheduling、application-level），并说出每层使用的工具。
- 解释为什么 `DCGM_FI_DEV_GPU_UTIL` 是 vLLM 的错误 HPA signal，并说出两个替代信号（queue depth、KV cache utilization）。
- 描述 gang scheduling 以及 KAI Scheduler 防止的 partial-allocation failure mode（8 个 GPU 中 7 个 idle）。
- 说出会终止 running GPU jobs 的 Karpenter consolidation policy（`WhenEmptyOrUnderutilized`），并给出 2026 年安全替代方案。

## 要解决的问题

你的团队在 Kubernetes 上发布了一个 LLM-serving service。你把 HPA signal 设置为 `DCGM_FI_DEV_GPU_UTIL`。业务时段里 service 固定在 100% utilization。HPA 从不 scale up，因为它已经认为你满载了。你手动增加一个 replica；TTFT 降低。HPA 仍然不扩容。这个 signal 在骗你。

另一个问题是，你使用 Cluster Autoscaler 管理 nodes。凌晨 2 点，一个 1M-token prompt 到达；cluster 花 3 分钟 provision 一个 node，请求超时。

再另一个问题是，你部署一个需要跨 2 个 nodes 使用 8 个 GPUs 的 70B model。Cluster 中有 7 个 GPUs 空闲，另有 1 个分散在 3 个 nodes 上。Cluster Autoscaler 为缺失的 1 个 GPU provision 一个 node。七个 nodes 等 4 分钟，一边烧钱一边等待 Kubernetes 把最后一个 GPU 拉起来。

三层，三个不同 failure modes。2026 年的 GPU-aware autoscaling 不是“打开 HPA”。它是组合 node provisioning、gang scheduling 和 application-signal autoscaling。

## 核心概念

### Layer 1：node provisioning (Karpenter)

Karpenter 观察 pending pods，并在约 45-60 秒内 provision nodes（Cluster Autoscaler 对 GPU nodes 通常需要 90-120 秒）。它会根据 `NodePool` constraint 动态选择 instance types；如果你的 pod 需要 8 个 H100 且 cluster 没有匹配 node，Karpenter 会直接 provision 一个，而不是扩展现有 group。

**Consolidation trap**：Karpenter 默认的 `consolidationPolicy: WhenEmptyOrUnderutilized` 对 GPU pools 很危险。它会终止 running GPU node，把 pods 迁移到更便宜、尺寸更合适的 instance。对于 inference workloads，这意味着驱逐运行中的 requests，并在新 node 上重新加载 70B model。损失是数分钟 capacity 加 request failures。

GPU pools 的安全设置：

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

这允许 Karpenter 在一小时后 consolidate 真正空的 nodes，但永不驱逐 running job。

### Layer 2：gang scheduling (KAI Scheduler)

KAI Scheduler（项目名曾为 “Karp”，后改名）处理 default kube-scheduler 不处理的内容：

**Gang scheduling**：all-or-nothing scheduling。需要 8 个 GPUs 的 distributed inference pod 要么 8 个一起启动，要么一个都不启动。没有它，你会遇到 partial-allocation trap：8 个 pods 中 7 个启动，随后无限等待并烧钱。

**Topology awareness**：知道哪些 GPUs 共享 NVLink、哪些位于同一 rack、哪些之间有 InfiniBand。相应地放置 pods。DeepSeek-V3 67B tensor-parallel workload 必须留在一个 NVLink domain 内；KAI Scheduler 会遵守这一点。

**Hierarchical queues**：多个 teams 以 priority 和 quota 竞争同一个 GPU pool。Team A 的 production pinch 只有在 priority rules 允许时才会被 Team B 的 training job preempt。

KAI 与 kube-scheduler 并行部署，作为 secondary scheduler；你用 annotation 指定 workloads 使用它。Ray 和 vLLM production-stack 都已集成。

### Layer 3：application-level signals

**HPA trap**：`DCGM_FI_DEV_GPU_UTIL` 是 duty-cycle metric，它测量 GPU 在每个采样间隔是否在工作。100% utilization 可能意味着 10 个 concurrent requests，也可能意味着 100 个；无论如何 GPU 都是 busy。基于 duty cycle 扩缩就是盲目扩缩。

更糟的是，vLLM 和类似 engines 会预分配 KV cache memory（最高到 `--gpu-memory-utilization`）。即使只有一个 request，memory usage 也会接近 90%。基于 memory 的 HPA 永远不会 scale down。

**2026 年替代信号**：

- Queue depth（等待 prefill 的 requests 数）。
- KV cache utilization（分配给 active sequences 的 blocks 比例）。
- Per-replica P99 TTFT（你的 SLA signal）。
- Goodput（每秒满足所有 SLOs 的 requests）。

NVIDIA Dynamo Planner 和 llm-d Workload Variant Autoscaler 消费这些 signals 并扩缩 replicas。它们会完全替代 LLM serving 中的 HPA。

### 何时使用什么

| Scale decision | Tool |
|----------------|------|
| Add/remove nodes | Karpenter |
| Schedule multi-GPU jobs | KAI Scheduler |
| Add/remove replicas | Dynamo Planner / llm-d WVA (or custom HPA on queue depth) |
| Choose GPU type | Karpenter NodePool |
| Preempt low-priority | KAI Scheduler queues |

### Disaggregated prefill/decode 会让一切更复杂

如果运行 disaggregated prefill/decode（Phase 17 · 17），你会有两类 pod，它们的 scaling triggers 不同：prefill pods 按 queue depth 扩缩，decode pods 按 KV cache pressure 扩缩。llm-d 会把这些暴露为单独的 `Services`，并为每个 role 配置 HPA。不要试图在二者前面放一个 single HPA。

### Cold start 在这里也很重要

Cold-start mitigation（Phase 17 · 10）正是 node provisioning time 变得用户可见的地方。Karpenter 的 45-60 秒 warm-up，加上 20GB model load，再加 engine init，意味着 from-zero request 需要 2-5 分钟。为 SLO-critical paths 保留 warm pool（`min_workers=1`），或者在 application layer 使用 Modal-style checkpointing。

### 你应该记住的数字

- Karpenter node provisioning：约 45-60s；Cluster Autoscaler 约 90-120s（GPU nodes）。
- KAI Scheduler 防止 partial-allocation waste，即 7-of-8 trap。
- `DCGM_FI_DEV_GPU_UTIL` 作为 HPA signal：失效；使用 queue depth 或 KV utilization。
- Karpenter `WhenEmptyOrUnderutilized`：终止 running GPU jobs。Inference 中使用 `WhenEmpty + consolidateAfter: 1h`。

## 实际使用

`code/main.py` 在 bursty GPU workload 上模拟三层 autoscaler。比较 naive HPA（duty cycle）、queue-depth HPA 和 KAI-gang-scheduled scaling。报告 unmet requests、idle-GPU minutes 和 composite score。

## 交付成果

本课产出 `outputs/skill-gpu-autoscaler-plan.md`。给定 cluster topology、workload shape 和 SLO，它会设计一个三层 autoscaling plan。

## 练习

1. 运行 `code/main.py`。在 bursty workload 下，naive duty-cycle HPA 会丢掉多少个 queue-depth HPA 能抓住的 requests？差异来自哪里？
2. 为一个服务 Llama 3.3 70B FP8 on H100 SXM5 的 cluster 设计 Karpenter NodePool。指定 `capacity-type`、`disruption.consolidationPolicy`、`consolidateAfter`，以及一个让 non-GPU workloads 远离这些 nodes 的 taint。
3. 你的团队报告 deployments 卡在 Pending，因为 “GPUs available but pod won't schedule”。诊断一下：这是 Karpenter、kube-scheduler，还是 KAI Scheduler？哪些 metrics 能确认？
4. 为 disaggregated prefill pods 选择一个 autoscale signal，并为 decode pods 选择另一个。说明二者理由。
5. 计算 `WhenEmptyOrUnderutilized` consolidation trap 在一个 24x7 production service 上的成本：该 service 平均每天 60 次 request-dropping events，P99 TTFT > 10s。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Karpenter | "the node provisioner" | Kubernetes node autoscaler；sub-minute provisioning |
| Cluster Autoscaler | "the old scaler" | Kubernetes node autoscaler predecessor；更慢，基于 group |
| KAI Scheduler | "the GPU scheduler" | 用于 gang + topology + queues 的 secondary scheduler |
| Gang scheduling | "all or nothing" | 原子化调度 N 个 pods，或全部 defer |
| Topology awareness | "rack-aware" | 基于 NVLink/IB/rack placement 放置 pods |
| `DCGM_FI_DEV_GPU_UTIL` | "GPU utilization" | Duty-cycle metric；不是 LLM 的 scaling signal |
| Queue depth | "waiting requests" | Prefill-bound scaling 的正确 HPA signal |
| KV cache utilization | "memory pressure" | Decode-bound scaling 的正确 HPA signal |
| Consolidation | "Karpenter consolidation" | 为迁移到更便宜 instance type 而终止 node |
| `WhenEmpty + 1h` | "safe consolidation" | 不驱逐 running GPU jobs 的 policy |

## 延伸阅读

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler) — design docs 和 configuration examples。
- [Karpenter Disruption Controls](https://karpenter.sh/docs/concepts/disruption/) — consolidation policy semantics 和 GPU-safe defaults。
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — Dynamo Planner scaling signals。
- [Ray docs — KAI Scheduler for RayClusters](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) — Ray integration pattern。
- [AWS EKS Compute and Autoscaling Best Practices](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) — managed-Kubernetes-specific guidance。
- [llm-d GitHub](https://github.com/llm-d/llm-d) — Workload Variant Autoscaler design。
