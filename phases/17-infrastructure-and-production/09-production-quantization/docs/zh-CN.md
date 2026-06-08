# 生产量化：AWQ、GPTQ、GGUF K-quants、FP8、MXFP4/NVFP4

> Quantization format 不是一个通用选择，它是 hardware、serving engine 和 workload 的函数。GGUF Q4_K_M 或 Q5_K_M 属于 CPU 与 edge，由 llama.cpp 和 Ollama 交付。需要在同一个 base 上做 multi-LoRA 时，GPTQ 在 vLLM 内胜出。AWQ 搭配 Marlin-AWQ kernels，在 7B 级模型上达到约 741 tok/s，并在 INT4 下拥有最佳 Pass@1，是 2026 年 datacenter production 默认选项。FP8 在 Hopper、Ada 和 Blackwell 上保持中间地带：近乎无损，支持广泛。NVFP4 和 MXFP4（Blackwell microscaling）更激进，需要 per-block validation。两个陷阱经常咬到团队：calibration dataset 必须匹配部署域；KV cache 与 weight quantization 是分开的，AWQ 课里“我的模型现在只有 4 GB”会忘掉 production batch sizes 下 10-30 GB 的 KV cache。

**类型：** Learn
**语言：** Python (stdlib, toy memory and throughput comparison across formats)
**先修：** Phase 10 · 13 (Quantization foundations), Phase 17 · 04 (vLLM Serving Internals)
**时间：** ~75 minutes

## 学习目标

- 说出六种 production quantization formats 及其在 2026 年的甜点区。
- 根据 hardware（CPU vs GPU、Hopper vs Blackwell）、engine（vLLM、TRT-LLM、llama.cpp）和 workload（routine chat、reasoning、multi-LoRA）选择格式。
- 计算所选格式节省的 weight memory，以及未被影响的 KV cache。
- 说出会让量化模型在 domain traffic 上退化的 calibration-dataset 陷阱。

## 要解决的问题

Quantization 会减少 memory 和 HBM bandwidth，而这正是 decode 所需要的。一个 FP16 70B 模型有 140 GB weights。把 weights 量化到 INT4（AWQ 或 GPTQ）后，模型变为 35 GB，可以放进一张 H100，并为 KV cache 留出空间；这很重要，因为在 128 concurrent sequences、2k context 下，仅 KV cache 就有 20-30 GB。

但 quantization 不是免费的。激进量化会损害质量，尤其在 reasoning-heavy tasks 上。不同格式适配不同 engines。不同硬件原生支持不同 precisions。2026 年的格式动物园是真实存在的，你不能照抄别人的选择，必须基于自己的 stack 决定。

## 核心概念

### 六种格式

| Format | Bits | Sweet spot | Engines |
|--------|------|-----------|---------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU、edge、laptops | llama.cpp, Ollama |
| GPTQ | 4-8 | vLLM 上的 Multi-LoRA | vLLM, TGI |
| AWQ | 4 | Datacenter GPU production | vLLM (Marlin-AWQ), TGI |
| FP8 | 8 | Hopper/Ada/Blackwell datacenter | vLLM, TRT-LLM, SGLang |
| MXFP4 | 4 | Blackwell multi-user | TRT-LLM |
| NVFP4 | 4 | Blackwell multi-user | TRT-LLM |

### GGUF：CPU/edge 默认选项

GGUF 是文件格式，不是严格意义上的量化方案；它在一个容器里打包 K-quant variants（Q2_K、Q3_K_M、Q4_K_M、Q5_K_M、Q6_K、Q8_0）。Q4_K_M 和 Q5_K_M 是 production defaults，在 4-5 bits 下接近 BF16 质量。它是 CPU 或 edge serving 的最佳选择，因为 llama.cpp 是目前最快的 CPU inference engine。

在 vLLM 中的 throughput penalty：7B 上约 93 tok/s，这种格式并未针对 GPU kernels 优化。只有当 deployment target 是 CPU/edge 时使用 GGUF，其他情况不要用。

### GPTQ：vLLM 中的 multi-LoRA

GPTQ 是一种带 calibration pass 的 post-training quantization algorithm。Marlin kernels 让它在 GPU 上很快（相对 non-Marlin GPTQ 有 2.6x speedup）。7B 上约 712 tok/s。

独特优势：GPTQ-Int4 支持 vLLM 中的 LoRA adapters。如果你在 serving 一个 base model 外加 10-50 个 fine-tuned variants（每个都是 LoRA），GPTQ 就是路径。截至 2026 年初，NVFP4 还不支持 LoRA。

### AWQ：datacenter GPU 默认选项

Activation-aware Weight Quantization。在量化过程中保护约 1% 最显著的 weights。Marlin-AWQ kernels：相比 naive 实现有 10.9x speedup。7B 上约 741 tok/s，在 INT4 formats 中 Pass@1 最佳。

新的 GPU serving 默认选择 AWQ，除非你需要 multi-LoRA（GPTQ）或激进的 Blackwell FP4（NVFP4）。

### FP8：可靠中间地带

8-bit floating point。近乎无损。支持广泛。Hopper Tensor Cores 原生加速 FP8，Blackwell 继承了这一点。当质量不可妥协（reasoning、medical、code-gen）时，FP8 是 2026 年的安全默认。内存节省只有 INT4 的一半，但质量风险低得多。

### MXFP4 / NVFP4：Blackwell 激进选项

Microscaling FP4。每个权重块都有自己的 scale factor。它很激进，但在 Blackwell Tensor Cores 上有硬件加速。相对 FP8 将 bytes per token 减半，这是 Phase 17 · 07 中的经济收益来源。

注意事项：
- 尚无 LoRA support（2026 年初）。
- 在 reasoning-heavy workloads 上质量下降明显。
- 每个模型都要在你的 eval set 上验证。

### Calibration 陷阱

AWQ 和 GPTQ 需要 calibration dataset，通常是 C4 或 WikiText。对于 domain models（code、medical、legal），如果用通用 web text 做 calibration，算法会错误判断应保护哪些 weights。HumanEval 上的 Pass@1 可能下降好几个点。

修复方式：用 in-domain data 做 calibration。通常几百条 domain samples 就够了。上线前要在 eval set 上测试。

### KV cache 陷阱

AWQ 会把 weights 缩到 4 bits。KV cache 是独立的，并保持 FP16/FP8。对于一个使用 AWQ 的 70B 模型：

- Weights: ~35 GB (INT4 from 140 GB).
- KV cache at 128 concurrent × 2k context: ~20 GB.
- Activations: ~5 GB.
- Total: ~60 GB — fits on H100 80GB.

天真地说“我把模型量化到 4 GB 了”，会忘掉另外 30-50 GB。要整体预算 HBM。

另外，KV cache quantization（FP8 KV 或 INT8 KV）是另一项选择，有自己的取舍；它直接影响 attention accuracy，不是免费收益。

### AWQ INT4 对 reasoning 有风险

Chain-of-thought、math、长上下文 code-gen：这些任务会明显受到激进量化影响。AWQ INT4 在 MATH 上会损失约 3-5 points。对于 reasoning-heavy workloads，发布 FP8 或 BF16，接受内存成本。

### 2026 选择指南

- CPU/edge serve: GGUF Q4_K_M。结束。
- GPU serve、routine chat、no LoRA: AWQ。
- GPU serve、multi-LoRA: GPTQ with Marlin。
- Reasoning workload: FP8。
- Blackwell datacenter、质量已验证: NVFP4 + FP8 KV。
- 模糊不清：对每种候选格式跑 1,000-sample eval。

## 实际使用

`code/main.py` 会为不同 model sizes 下的六种格式计算 memory footprint（weights + KV + activations）和 relative throughput。它展示 KV cache 在哪里占主导、weight compression 在哪里有收益，以及 FP8 在哪里是安全选择。

## 交付成果

本课产出 `outputs/skill-quantization-picker.md`。给定 hardware、model size、workload type 和 quality tolerance，它会选择一种格式，并生成 calibration/validation plan。

## 练习

1. 运行 `code/main.py`。对于一个 70B 模型，在 128 concurrent、2k context 下计算每种格式的 total HBM。哪种格式可以让你放进一张 H100 80GB？
2. 你有一个 7B coding model。选择一种格式并说明理由。如果你对 quality tolerance 的判断错了，恢复路径是什么？
3. 计算为 medical domain model 校准 AWQ 所需的 calibration-dataset size。为什么更多数据不总是更好？
4. 阅读 Marlin-AWQ kernel paper 或 release notes。用三句话解释为什么 AWQ 在 7B 上达到 741 tok/s，而 raw GPTQ 约为 712。
5. 什么时候适合把 AWQ weights 与 FP8 KV cache 组合，而不是让 KV 保持 BF16？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| GGUF | “llama.cpp format” | 打包 K-quant variants 的文件格式；CPU/edge 默认 |
| Q4_K_M | “Q4 K M” | 4-bit K-quant medium；production GGUF 默认 |
| GPTQ | “gee pee tee q” | 带 calibration 的 post-train INT4；在 vLLM 中支持 LoRA |
| AWQ | “a w q” | Activation-aware INT4；Marlin kernels；INT4 下最佳 Pass@1 |
| Marlin kernels | “fast INT4 kernels” | Hopper 上用于 INT4 的 custom CUDA kernels；10x speedup |
| FP8 | “eight-bit float” | Hopper/Ada/Blackwell 上的安全默认 precision |
| MXFP4 / NVFP4 | “microscaling four” | Blackwell 4-bit FP，带 per-block scale factors |
| Calibration dataset | “cal data” | 用来选择 quantization parameters 的输入文本；必须匹配 domain |
| KV cache quantization | “KV INT8” | 与 weights 独立的选择；影响 attention accuracy |

## 延伸阅读

- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — comparative benchmarks。
- [Jarvis Labs — vLLM Quantization Complete Guide](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) — 按格式列出的 throughput numbers。
- [PremAI — GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/) — format-by-format picking。
- [vLLM docs — Quantization](https://docs.vllm.ai/en/latest/features/quantization/index.html) — supported formats and flags。
- [AWQ paper (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978) — 原始 AWQ formulation。
- [GPTQ paper (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323) — 原始 GPTQ formulation。
