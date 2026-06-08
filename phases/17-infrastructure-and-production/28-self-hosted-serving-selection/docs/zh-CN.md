# 自托管 Serving 选型 — llama.cpp、Ollama、TGI、vLLM、SGLang

> 到 2026 年，自托管推理主要由四类引擎主导。选择时看硬件、规模和生态。**llama.cpp** 在 CPU 上最快，模型支持最广，对量化和线程控制最完整。**Ollama** 是开发笔记本上的一行命令安装方案，比 llama.cpp 慢约 15-30%（Go + CGo + HTTP 序列化），在接近生产的负载下吞吐差距可达 3x。**TGI 于 2025 年 12 月 11 日进入维护模式**，之后只修 bug，原始吞吐比 vLLM 慢约 10%，但历史上可观测性和 HF 生态集成最强。维护状态让它成为有风险的长期押注；新项目默认选择 SGLang 或 vLLM 更稳。**vLLM** 是通用生产默认项，v0.15.1（2026 年 2 月）加入 PyTorch 2.10、RTX Blackwell SM120、H200 优化。**SGLang** 是 agentic 多轮 / 前缀复用密集场景的专家，已有 400,000+ GPU 投入生产（xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS）。硬件约束：仅 CPU → 只能 llama.cpp。AMD / 非 NVIDIA → 只能 vLLM（TRT-LLM 锁定 NVIDIA）。2026 年流水线模式：dev = Ollama，staging = llama.cpp，prod = vLLM 或 SGLang。全程使用同一套 GGUF/HF 权重。

**类型:** Learn
**语言:** Python（stdlib，引擎决策树遍历器）
**先修:** 覆盖引擎的所有 Phase 17 课程（04、06、07、09、18）
**时间:** ~45 分钟

## 学习目标

- 在给定硬件（CPU / AMD / NVIDIA Hopper / Blackwell）、规模（1 个用户 / 100 / 10,000）和工作负载（通用聊天 / agent / 长上下文）时选择引擎。
- 说出 2026 年 TGI 的维护模式状态（2025 年 12 月 11 日）以及它为什么让新项目更偏向 vLLM 或 SGLang。
- 描述 dev/staging/prod 流水线，并在全程使用同一套 GGUF 或 HF 权重。
- 解释为什么“仅 CPU”会强制选择 llama.cpp，而“AMD”会排除 TRT-LLM。

## 要解决的问题

你的团队启动一个新的自托管 LLM 项目。一位工程师说用 Ollama，另一位说用 vLLM，第三位说“ TGI 不是开箱即用吗？”三个人在不同语境下都对，但没有任何一个答案适用于所有场景。

到 2026 年，选型树的顺序很重要：硬件第一，规模第二，工作负载第三。而且一个具体的 2025 年事件，也就是 TGI 在 12 月 11 日进入维护模式，改变了新项目的默认选择。

## 核心概念

### 五个引擎

| 引擎 | 最适合 | 备注 |
|--------|----------|-------|
| **llama.cpp** | CPU / 边缘 / 最小依赖 / 最广模型支持 | CPU 上最快，控制最完整 |
| **Ollama** | 开发笔记本、单用户、一行命令安装 | 比 llama.cpp 慢 15-30%；生产吞吐差距 3x |
| **TGI** | HF 生态、受监管行业 | **2025 年 12 月 11 日进入维护模式** |
| **vLLM** | 通用生产、100+ 用户 | 2026 年广泛生产默认项；v0.15.1 2026 年 2 月 |
| **SGLang** | Agentic 多轮、前缀密集工作负载 | 400,000+ GPU 投入生产 |

### 硬件优先决策

**仅 CPU** → llama.cpp。Ollama 也能工作，但更慢。没有其他引擎在 CPU 上有竞争力。

**AMD GPU** → vLLM（AMD ROCm 支持）。SGLang 也能工作。TRT-LLM 锁定 NVIDIA，因此排除。

**NVIDIA Hopper（H100 / H200）** → vLLM、SGLang 或 TRT-LLM。三者都是第一梯队。

**NVIDIA Blackwell（B200 / GB200）** → TRT-LLM 是吞吐领先者（Phase 17 · 07）。vLLM 和 SGLang 紧随其后。

**Apple Silicon（M-series）** → llama.cpp（Metal）。Ollama 封装了这一路径。

### 规模其次决策

**1 个用户 / 本地开发** → Ollama。一条命令，几秒出首 token。

**10-100 个用户 / 小团队** → vLLM 单 GPU。

**100-10k 个用户 / 生产** → vLLM production-stack（Phase 17 · 18）或 SGLang。

**10k+ 用户 / 企业** → vLLM production-stack + disaggregated（Phase 17 · 17）+ LMCache（Phase 17 · 18）。

### 工作负载第三决策

**通用聊天 / Q&A** → vLLM 作为广义默认项胜出。

**Agentic 多轮（工具、规划、记忆）** → SGLang 的 RadixAttention（Phase 17 · 06）占优。

**有大量前缀复用的 RAG** → SGLang。

**代码生成** → vLLM 足够好；SGLang 在缓存上略优。

**长上下文（128K+）** → vLLM + chunked prefill；SGLang + tiered KV。

### TGI 维护陷阱

Hugging Face TGI 于 2025 年 12 月 11 日进入维护模式，之后只修 bug。历史上：一流可观测性、最佳 HF 生态集成（model cards、安全工具），原始吞吐略落后 vLLM。

对 2026 年的新项目：默认避开 TGI。现有 TGI 部署可以继续运行，但最终应该迁移。SGLang 和 vLLM 是更稳的默认项。

### 流水线模式

Dev（Ollama）→ staging（llama.cpp）→ prod（vLLM）。全程使用同一套 GGUF 或 HF 权重。工程师在笔记本上快速迭代；staging 镜像生产量化；prod 是 serving 目标。

### Ollama 注意事项

Ollama 非常适合开发。它不适合共享生产：Go HTTP 序列化带来开销，并发管理比 vLLM 简单，OpenTelemetry 支持滞后。把 Ollama 用在它闪光的地方，也就是一个用户、一条命令，然后在共享场景切换到 vLLM。

### 自托管 vs 托管是另一个决策

Phase 17 · 01（托管 hyperscalers）和 · 02（推理平台）覆盖托管方案。本课假设你已经决定自托管。自托管理由包括：数据驻留、自定义 fine-tune、大规模总拥有成本、托管平台没有的领域模型。

### 你应该记住的数字

- TGI 维护模式：2025 年 12 月 11 日。
- vLLM v0.15.1：2026 年 2 月；PyTorch 2.10；Blackwell SM120 支持。
- SGLang 生产足迹：400,000+ GPU。
- Ollama 相对 llama.cpp 的吞吐差距：慢 15-30%；生产负载下 3x。

## 实际使用

`code/main.py` 是一个决策树遍历器：给定硬件 + 规模 + 工作负载，选择一个引擎并解释原因。

## 交付成果

本课产出 `outputs/skill-engine-picker.md`。给定约束后，它会选择引擎并写出迁移计划。

## 练习

1. 用你的硬件 / 规模 / 工作负载运行 `code/main.py`。输出符合你的直觉吗？
2. 你的基础设施是 12 张 H100 和 8 张 MI300X AMD。选什么引擎？为什么 TRT-LLM 不在表内？
3. 一个团队想在 2026 年继续用 TGI，因为“这是我们熟悉的”。论证迁移理由。
4. 从 Ollama dev 到 vLLM prod：量化、配置和可观测性会发生什么变化？
5. RAG 产品的 P99 前缀长度为 8K，且租户间高复用。选择一个引擎，并把它与 Phase 17 · 11 + 18 组合成栈。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| llama.cpp | “CPU 那个” | 最广模型支持，CPU 上最快 |
| Ollama | “笔记本那个” | 一行命令安装，开发级吞吐 |
| TGI | “HF 的 serving” | 自 2025 年 12 月起进入维护模式 |
| vLLM | “默认项” | 2026 年广泛生产基线 |
| SGLang | “agentic 那个” | 前缀密集，RadixAttention |
| TRT-LLM | “NVIDIA 锁定” | Blackwell 吞吐领先，仅 NVIDIA |
| GGUF | “llama.cpp 格式” | 打包的 K-quant 变体 |
| Production-stack | “vLLM K8s” | Phase 17 · 18 参考部署 |
| Pipeline pattern | “dev→stage→prod” | 同一权重上的 Ollama → llama.cpp → vLLM |

## 延伸阅读

- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morph — llama.cpp vs Ollama 2026](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai — Comprehensive LLM Inference Engine Comparison](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI — 10 Best vLLM Alternatives 2026](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [TGI maintenance announcement](https://github.com/huggingface/text-generation-inference) — 发布说明。
- [vLLM v0.15.1 release notes](https://github.com/vllm-project/vllm/releases)
