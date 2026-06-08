# 综合项目 14 — Speculative-Decoding Inference Server

> vLLM 0.7 中的 EAGLE-3 在真实流量上带来 2.5-3x 吞吐。P-EAGLE（AWS 2026）把 parallel speculation 推得更远。SGLang 的 SpecForge 规模化训练 draft heads。Red Hat 的 Speculators hub 为常见开放模型发布了对齐 draft。TensorRT-LLM 让 speculative decoding 成为 NVIDIA 上的一等能力。2026 年生产 serving stack 是 vLLM 或 SGLang，配 EAGLE-family drafts、FP8 或 INT4 quantization，并基于 queue-wait 做 HPA。这个综合项目要以 2.5x+ baseline throughput 服务两个开放模型，并给出完整的 tail-latency report。

**类型:** Capstone
**语言:** Python（serving），C++ / CUDA（kernel inspection），YAML（configs）
**先修:** Phase 3（deep learning），Phase 7（transformers），Phase 10（LLMs from scratch），Phase 17（infrastructure）
**覆盖阶段:** P3 · P7 · P10 · P17
**时间:** 30 小时

## 要解决的问题

Speculative decoding 在 2026 年已经商品化。EAGLE-3 draft heads 基于目标模型的 hidden states 训练，并预测后续 N 个 tokens；目标模型在一次 pass 中验证。60-80% 的 acceptance rates 会转化为 2-3x 的端到端吞吐。vLLM 0.7 原生集成了它。SGLang + SpecForge 给你训练管线。Red Hat 的 Speculators 发布了 Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B 的对齐 drafts。

工艺重点在 serving operations，而不在模型。Acceptance rate 会随流量分布漂移（ShareGPT vs code vs domain data）。拒绝时的 tail latency 比没有 speculation 更差；你必须报告多个 batch size 下的 p99，而不能只报告 steady-state tokens/sec。每 1M tokens 的成本对比 Anthropic / OpenAI API，是可信度杠杆。

## 核心概念

Speculative decoding 有两层。**Draft** 模型（EAGLE-3 head、ngram 或更小的 target-aligned model）每一步提出 k 个候选 tokens。**Target** 模型在一次 pass 中验证所有 k 个；任何被接受的前缀都会替代 greedy path。Acceptance rate 取决于 draft-target alignment 和输入分布。

EAGLE-3 在大多数流量上优于 ngram drafts。P-EAGLE 运行 parallel speculation，构建更深的 draft trees。权衡在于：拒绝时的 P99 latency 更高，因为 verify pass 更大。Serving config 必须报告按 batch-size 分桶的 latency，才能暴露这一点。

部署在 Kubernetes 上。vLLM 0.7 每个 GPU 或 tensor-parallel shard 运行一个 replica。HPA 基于 queue-wait 而不是 CPU 自动扩缩。FP8（Marlin）和 INT4（AWQ）quantization 让 GPU memory 落在 H100 / H200 的容量内。端到端报告包含 throughput、acceptance rate、batch 1/8/32 的 p50/p99，以及 $/1M tokens。

## 架构

```text
request ingress
    |
    v
vLLM server (0.7) or SGLang (0.4)
    |
    +-- draft: EAGLE-3 heads | P-EAGLE parallel | ngram fallback
    +-- target: Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     quantized FP8-Marlin or INT4-AWQ
    |
    v
verify pass: batch k draft tokens through target
    |
    v (accept prefix; resample for rejected suffix)
    v
token stream back to client
    |
    v
Prometheus metrics: throughput, acceptance rate, queue wait, latency p50/p99
    |
    v
HPA on queue-wait metric
```

## 技术栈

- Serving：vLLM 0.7 或 SGLang 0.4
- Speculative methods：EAGLE-3 draft heads、P-EAGLE parallel speculation、ngram fallback
- Draft training：SpecForge（SGLang）或 Red Hat Speculators
- Target models：Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B
- Quantization：FP8（Marlin）、INT4 AWQ
- Deployment：Kubernetes + NVIDIA device plugin；基于 queue-wait metric 的 HPA
- Eval：ShareGPT、MT-Bench-v2、GSM8K、HumanEval，用于 domain-spread acceptance measurement
- Reference：TensorRT-LLM speculative decoding，作为 vendor baseline

## 动手实现

1. **Target model prep。** 选择 Llama 3.3 70B。通过 Marlin quantize 到 FP8。用 vLLM 0.7 在 1xH100（或 2x tensor-parallel）上部署。

2. **Draft source。** 从 Red Hat Speculators 拉取 aligned EAGLE-3 draft head（或通过 SpecForge 训练一个）。加载进 vLLM 的 speculative-decoding config。

3. **Baseline numbers。** 在 speculation 前：batch 1/8/32 的 tokens/s、p50/p99 latency、GPU utilization。发布结果。

4. **启用 EAGLE-3。** 翻转 config；重新跑同一 benchmark。报告 speedup、acceptance rate、p99 tail-latency delta。

5. **P-EAGLE。** 启用 parallel speculation；测量更深 draft tree vs serial EAGLE-3。报告 P-EAGLE 从有益到有害的 inflection。

6. **Domain traffic。** 让 ShareGPT vs HumanEval vs domain-specific traffic 通过同一个 server。测量每种分布的 acceptance rate。识别 drafts 何时漂移。

7. **Second target model。** 在 Qwen3-Coder-30B MoE 上运行同一管线。Draft 更棘手（MoE routing noise）。报告结果。

8. **K8s HPA。** 在 K8s 下部署，并让 HPA 跟踪 `queue_wait_ms`。演示负载翻三倍时的 scale-out。

9. **Cost comparison。** 在同一 eval 上计算 $/1M tokens，并与 Anthropic Claude Sonnet 4.7 和 OpenAI GPT-5.4 对比。发布。

## 实际使用

```text
$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[serve]     vLLM 0.7, Llama 3.3 70B FP8, EAGLE-3 active
[decode]    bs=8, accepted_tokens_per_step=3.2, acceptance_rate=0.76
[latency]   first-token 42ms, full-response 980ms (620 tokens)
[cost]      $0.34 per 1M output tokens at sustained throughput
```

## 交付成果

`outputs/skill-inference-server.md` 描述交付物。一个经过测量的 serving stack，包含 speculative decoding、完整 benchmark report 和 K8s deployment。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | Measured speedup vs baseline | 两个模型上在质量匹配时达到 2.5x+ throughput |
| 20 | realistic traffic 上的 acceptance rate | Per-distribution acceptance-rate report |
| 20 | P99 tail-latency discipline | batch 1/8/32 下有无 speculation 的 p99 |
| 20 | Ops | K8s deploy、基于 queue-wait 的 HPA、rollout smooth |
| 15 | Write-up and methodology | 清楚解释改了什么以及为什么 |
| **100** | | |

## 练习

1. 测量 draft 比 target 落后一个版本时的 acceptance-rate degradation（例如 Llama 3.3 -> 3.4 drift）。构建 monitoring alert。

2. 实现 ngram-fallback：如果 EAGLE-3 acceptance 低于阈值，切换到 ngram drafts。报告 reliability improvement。

3. 运行受控 MoE 实验：同一个 Qwen3-Coder-30B，在注入 routing noise 和不注入时对比。测量 draft acceptance sensitivity。

4. 扩展到 H200（141 GB）。报告获得的 model-size-per-replica headroom，以及是否能服务未量化的 Llama 3.3 70B。

5. 在相同 H100 hardware 上 benchmark TensorRT-LLM speculative decoding。报告它相对 vLLM 的优势场景。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| Draft model | “Speculator” | 提出 N 个 tokens 供 target 验证的小模型 |
| EAGLE-3 | “2026 draft architecture” | 基于 target hidden states 训练的 draft head；约 75% acceptance |
| P-EAGLE | “Parallel speculation” | 在一次 target pass 中验证的 draft branches 树 |
| Acceptance rate | “Hit rate” | 无需 resampling 就被接受的 drafted tokens 比例 |
| Quantization | “FP8 / INT4” | 低精度 weights，用于在 GPU memory 中容纳更多模型 |
| Queue wait | “HPA metric” | 请求在 inference starts 前等待在 pending queue 中的时间 |
| Speculators hub | “Aligned drafts” | Red Hat Neural Magic 为常见开放模型提供的 EAGLE drafts hub |

## 延伸阅读

- [vLLM EAGLE and P-EAGLE documentation](https://docs.vllm.ai) — reference serving stack
- [P-EAGLE (AWS 2026)](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/) — parallel speculative decoding paper + integration
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — draft-head training pipeline
- [Red Hat Speculators](https://github.com/neuralmagic/speculators) — aligned draft hub
- [TensorRT-LLM speculative decoding](https://nvidia.github.io/TensorRT-LLM/) — vendor alternative
- [Fireworks.ai serving architecture](https://fireworks.ai/blog) — commercial reference
- [EAGLE-3 paper (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — method paper
- [vLLM repository](https://github.com/vllm-project/vllm) — code and benchmarks
