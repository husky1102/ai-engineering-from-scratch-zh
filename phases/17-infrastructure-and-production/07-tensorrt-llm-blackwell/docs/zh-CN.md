# Blackwell 上使用 FP8 和 NVFP4 的 TensorRT-LLM

> TensorRT-LLM 只服务 NVIDIA 生态，但它在 Blackwell 上确实胜出。在 GB200 NVL72 搭配 Dynamo 编排时，SemiAnalysis InferenceX 在 2026 年 Q1-Q2 对一个 120B 模型测得每百万 tokens $0.012，而 H100 + vLLM 为 $0.09/M，形成 7x 的经济差距。这个栈叠加了三种浮点制度：FP8 对 KV cache 和 attention kernels 仍然关键，因为它们需要 FP8 的动态范围；NVFP4（4-bit microscaling）处理权重和激活；multi-token prediction (MTP) 与 disaggregated prefill/decode 又在此基础上增加 2-3x。Day-0 model support 可以直接加载 FP4 weights，无需训练后转换。2026 年工程团队要注意的代价是：TRT-LLM 是封闭的 NVIDIA 栈，所以采用它就是用可移植性换吞吐量。投入前先根据你的模型和硬件组合算清楚。

**类型：** Learn
**语言：** Python (stdlib, toy FP8/NVFP4 memory and cost calculator)
**先修：** Phase 17 · 04 (vLLM Serving Internals), Phase 10 · 13 (Quantization)
**时间：** ~75 minutes

## 学习目标

- 解释为什么即使权重使用 NVFP4，FP8 对 KV cache 和 attention 仍然关键。
- 计算 frontier model 在 BF16、FP8 和 NVFP4 下的 HBM 占用，并推理节省来自哪里。
- 说出 TRT-LLM 利用的 Blackwell 专属特性（day-0 FP4、MTP、disaggregated serving、all-to-all primitives）。
- 判断 TRT-LLM 的 NVIDIA 锁定何时值得用来换取相对 Hopper 上 vLLM 的 7x 成本差距。

## 要解决的问题

2026 年推理经济性的前沿问题是“每美元能跑多少 tokens”。答案取决于四层叠加选择：硬件代际（Hopper H100/H200 vs Blackwell B200/GB200）、精度（BF16 → FP8 → NVFP4）、serving engine（vLLM vs SGLang vs TRT-LLM）和编排方式（plain vs disaggregated vs Dynamo）。

在 Hopper + vLLM 上，一个 120B MoE 约为每百万 tokens $0.09。在 Blackwell + TRT-LLM + Dynamo 上，同一个模型约为 $0.012，便宜 7x。差距一部分来自硬件（Blackwell 的单 GPU LLM 吞吐量比 Hopper 高 11-15x）。另一部分来自软件栈：FP4 weights、MTP draft、disaggregated prefill/decode，以及用于 MoE expert communication 的 NVLink 5 all-to-all。

你无法在 NVIDIA 栈之外复现这一点。这就是取舍：用可移植性换经济性。理解哪些栈选择贡献了差距中的哪一部分，就是本课重点。

## 核心概念

### 为什么 FP8 仍是 KV cache 的底线

2026 年一个常见错误是：以为 NVFP4 可以用于所有地方。事实并非如此。KV cache 需要 FP8（8-bit floating point），因为它存储 attention keys 和 values，这些值跨越很宽的动态范围。把 KV 量化到 FP4 会造成灾难性的准确率损失：分布尾部被截掉，attention scores 随之崩塌。FP8 的 exponent bits 给了 KV cache 所需的范围。

NVFP4（2025-2026）适用于权重和激活。Microscaling 的意思是：每个权重块都有自己的 scale factor，因此小块可以覆盖不同动态范围，不会遭遇 per-tensor scale 的损失。对激活而言，FP4 能撑住，是因为同一层内的 activations 通常范围较小。

典型 Blackwell 配置：

- Weights: NVFP4 (4-bit microscaling).
- Activations: NVFP4.
- KV cache: FP8.
- Attention accumulator: FP32 (softmax stability).

### TRT-LLM 使用的 Blackwell 专属 primitives

- **Day-0 FP4 weights**：模型提供方直接发布 FP4 weights；TRT-LLM 无需训练后转换即可加载。FP4 不需要 AWQ / GPTQ 步骤。
- **Multi-token prediction (MTP)**：思路与 EAGLE（Phase 17 · 05）相同，但集成到 TRT-LLM build 中。
- **Disaggregated serving**：prefill 和 decode 使用独立 GPU pools，KV cache 通过 NVLink 或 InfiniBand 传输。思路与 Dynamo（Phase 17 · 20）相同。
- **All-to-all communication primitives**：相比 Hopper，NVLink 5 将 MoE expert communication latency 降低 3x。TRT-LLM 的 MoE kernels 针对此做了调优。
- **NVFP4 + MXFP8 microscaling**：Blackwell Tensor Cores 上硬件加速的 scale-factor handling。

### 你应该记住的数字

- HGX B200 通过 TRT-LLM 在 GPT-OSS-120B 上达到 $0.02/M tokens。
- GB200 NVL72 通过 Dynamo（编排 TRT-LLM）达到 $0.012/M tokens。
- H100 + vLLM 在可比 workload 上约为 $0.09/M tokens。
- TRT-LLM 更新在三个月内带来 2.8x 吞吐提升（2026）。
- Blackwell 相比 Hopper 的单 GPU LLM 吞吐量为 11-15x。
- MLPerf Inference v6.0（2026 年 4 月）：Blackwell 主导所有提交任务。

### FP4 实际付出的质量成本

NVFP4 很激进。在 reasoning-heavy workloads（chain-of-thought、math、带长上下文的 code-gen）上，FP4 weights 的退化很明显。Per-block calibration 可以缓解但不能消除。发布 reasoning models 的团队常常使用 FP8 weights + FP4 activations 作为折中，或者继续在 H200 上全程使用 FP8。

规则：投入 NVFP4 weights 之前，一定要先在你的 eval set 上验证任务质量。

### 为什么这是 NVIDIA-lock 决策

TRT-LLM 是 C++ + CUDA + closed-source kernels。模型需要为特定 GPU SKU 编译。没有 AMD，没有 Intel，也没有 ARM。如果你的基础设施策略是多供应商，TRT-LLM 对由 TRT-LLM serving 的 tier 来说就不成立，不过你仍然可以在混合硬件上用 vLLM serving。若你已经是 NVIDIA-only，7x 差距足以为这种锁定买单。

### 2026 年实用配方

对于年度推理账单超过 $100M 的团队，继续跑在 Hopper + vLLM 上等于留下 7-10x 的优化空间。把成本主导型 workloads 迁移到 Blackwell + TRT-LLM + Dynamo。把实验 tier 留在 H100 + vLLM 上，以保留模型迭代速度。每个 NVFP4-converted model 上线前都要验证质量。

### Disaggregation 的额外收益

TRT-LLM 的 disaggregated serving（分离 prefill 和 decode pools）会在 Phase 17 · 20 深入讲解。在 Blackwell 上，乘数会继续叠加：FP4 weights × MTP speedup × disaggregated placement × cache-aware routing。7x 这个数字假设使用了完整栈。

## 实际使用

`code/main.py` 会为三种栈计算模型的 HBM footprint、decode throughput（memory-bound regime）和 $/M-tokens：H100 + BF16 + vLLM、H100 + FP8 + vLLM、B200 + NVFP4/FP8 + TRT-LLM。运行它，观察复合效应，以及每个变化分别贡献了差距中的多少。

## 交付成果

本课产出 `outputs/skill-trtllm-blackwell-advisor.md`。给定 workload、model size 和 annual token volume，它会判断 Blackwell + TRT-LLM 栈是否值得付出 NVIDIA-lock。

## 练习

1. 运行 `code/main.py`。对于一个 120B MoE 且 active parameters 为 30% 的模型，计算 H100 BF16、H100 FP8 和 B200 NVFP4/FP8 上受 memory-bandwidth 限制的 decode throughput。最大的跃迁来自哪里？
2. 某客户每年在 H100 + vLLM 上花费 $2M。考虑 7x 经济差距，为了在 12 个月内摊销迁移到 TRT-LLM 的成本，他们需要购买多少 Blackwell GPUs 才能达到 break-even？
3. 你看到 NVFP4 weight conversion 后 MATH 准确率下降 3 个点。说出两条恢复路径：quality-first（保留 FP8 weights）和 cost-first（用 in-domain data 做 calibration）。
4. 阅读 MLPerf v6.0 inference results。哪个任务的 Blackwell-over-Hopper 差距最小，为什么？
5. 计算一个 405B 模型在 NVFP4 weights + FP8 KV cache 且 128k context 下需要多少 HBM。它能放进单个 GB200 NVL72 node 吗？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| FP8 | “eight-bit float” | 8-bit floating point；因动态范围用于 KV cache 和 attention |
| NVFP4 | “four-bit micro” | NVIDIA 的 4-bit microscaling FP format；Blackwell 上用于 weights 和 activations |
| MXFP8 | “MX eight” | Microscaling FP8 variant；在 Blackwell Tensor Cores 上有硬件加速 |
| Day-0 FP4 | “ship FP4 weights” | 模型提供方发布已经是 FP4 的 weights；无需 post-train conversion step |
| MTP | “multi-token prediction” | TRT-LLM 集成的 speculative-decoding draft（Phase 17 · 05） |
| Disaggregated serving | “split prefill/decode” | Prefill 和 decode 位于独立 GPU pools；KV 通过 NVLink/IB 传输 |
| All-to-all | “MoE expert comm” | 将 tokens 路由到 expert GPUs 的通信模式；NVLink 5 降低 3x |
| InferenceX | “SemiAnalysis inference bench” | 2026 年业内认可的 cost-per-token benchmark |

## 延伸阅读

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — 2026 年 4 月 MLPerf 结果。
- [NVIDIA — MoE Inference on Blackwell](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) — NVLink 5 all-to-all 与 MoE kernels。
- [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html) — 官方 engine 文档。
- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) — TRT-LLM 之上的 disaggregated orchestration。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 发布 Blackwell 数字的 benchmark suite。
