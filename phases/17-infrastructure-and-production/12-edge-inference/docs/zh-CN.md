# Edge Inference：Apple Neural Engine、Qualcomm Hexagon、WebGPU/WebLLM、Jetson

> Edge 的核心约束是 memory bandwidth，而不是 compute。Mobile DRAM 约为 50-90 GB/s；datacenter HBM3 超过 2-3 TB/s，差距为 30-50x。Decode 是 memory-bound 的，所以这个差距具有决定性。2026 年，格局分成四类。Apple M4/A18 Neural Engine 峰值 38 TOPS，采用 unified memory（没有 CPU↔NPU copy）。Qualcomm Snapdragon X Elite / 8 Gen 4 Hexagon 达到 45 TOPS。WebGPU + WebLLM 在 M3 Max 上以约 41 tok/s 运行 Llama 3.1 8B (Q4)（约为 native 的 70-80%）；17.6k GitHub stars、OpenAI-compatible API、约 70-75% mobile coverage。NVIDIA Jetson Orin Nano Super (8GB) 可放下 Llama 3.2 3B / Phi-3；AGX Orin 通过 vLLM 以约 40 tok/s 运行 gpt-oss-20b；Jetson T4000（JetPack 7.1）为 AGX Orin 的 2x。TensorRT Edge-LLM 支持 EAGLE-3、NVFP4、chunked prefill，并由 Bosch、ThunderSoft、MediaTek 在 CES 2026 展示。

**类型：** Learn
**语言：** Python (stdlib, toy bandwidth-bound decode simulator)
**先修：** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 09 (Production Quantization)
**时间：** ~60 minutes

## 学习目标

- 解释为什么 mobile LLM inference 是 memory-bandwidth-bound，compute 反而是次要因素。
- 枚举四类 edge targets（Apple ANE、Qualcomm Hexagon、WebGPU/WebLLM、NVIDIA Jetson），并把每类匹配到 use case。
- 说出 2026 年 WebGPU coverage gap（Firefox Android 正在追赶）和 Safari iOS 26 落地情况。
- 为每个 target 选择 quantization format（ANE 用 Core ML INT4 + FP16，Hexagon 用 QNN INT8/INT4，browser 用 WebGPU Q4，Jetson Thor 用 NVFP4）。

## 要解决的问题

客户想要一个 on-device chatbot：voice-first、private-by-default、offline 可用。在 MacBook Pro M3 Max 上，Llama 3.1 8B Q4 约 55 tok/s，没问题。在 iPhone 16 Pro 上，同一个模型约 3 tok/s，不行。在搭载 Snapdragon 8 Gen 3 的中端 Android 上，约 7 tok/s。在 Chrome Android v121+ 通过 WebGPU 跑浏览器端，取决于设备为 4-8 tok/s。

Throughput variance 不是移植问题。它是 bandwidth gap × quantization format × NPU 是否可从 user-space 访问的结果。2026 年的 edge inference 是四个不同问题，需要四套不同解决方案。

## 核心概念

### Bandwidth 才是真正天花板

Decode 每生成一个 token 都会读取完整 weights。一个 Q4 的 7B 模型是 3.5 GB。以 50 GB/s 读取 3.5 GB 需要 70 ms，理论天花板约为 14 tok/s。在 90 GB/s（high-end mobile DRAM）下，天花板变为约 25 tok/s。在低于这个数字时，再多 compute 也帮不上忙。

Datacenter HBM3 以 3 TB/s 读取同样的 3.5 GB 只需 1.2 ms，天花板是 830 tok/s。同一个模型、同一份 weights。不同的是 memory subsystem。

### Apple Neural Engine（M4 / A18）

- 最高 38 TOPS。Unified memory（CPU 和 ANE 共享同一池）意味着没有 copy overhead。
- 通过 Core ML + `.mlmodel` compiled models 访问，或通过 PyTorch 使用 Metal Performance Shaders（MPS）。
- Llama.cpp Metal backend 使用 MPS，不直接使用 ANE；native ANE 需要 Core ML conversion。
- 2026 年 iOS apps 的最佳实践路径：Core ML with INT4 weights + FP16 activations。

### Qualcomm Hexagon（Snapdragon X Elite / 8 Gen 4）

- 最高 45 TOPS。集成在 SoC 的 CPU 和 GPU 旁边，但拥有独立 memory domain。
- QNN（Qualcomm Neural Network）SDK 和 AI Hub 提供从 PyTorch/ONNX 的 conversion。
- Chat templates、Llama 3.2、Phi-3 都作为 AI Hub 上的一等 artifacts 发布。

### Intel / AMD NPUs（Lunar Lake、Ryzen AI 300）

- 40-50 TOPS。Software 落后 Apple/Qualcomm；OpenVINO 正在改进，但仍偏小众。
- 最适合 Windows ARM copilot apps；在 AMD/Intel desktops 上适合 local-first native 应用。

### WebGPU + WebLLM

- 通过 WebGPU compute shaders 在浏览器中运行模型；无需安装。
- M3 Max 上 Llama 3.1 8B Q4 约 41 tok/s，约为同一 backend native 的 70-80%。
- WebLLM 有 17.6k GitHub stars；OpenAI-compatible JS API；Apache 2.0。
- 2026 coverage：Chrome Android v121+、Safari iOS 26 GA，Firefox Android 仍在追赶。总体约 70-75% mobile coverage。

### NVIDIA Jetson family

- Orin Nano Super (8GB)：可以容纳 Llama 3.2 3B、Phi-3，并有不错 tok/s。
- AGX Orin：通过 vLLM 以约 40 tok/s 运行 gpt-oss-20b。
- Thor / T4000（JetPack 7.1）：2x AGX Orin performance，支持 EAGLE-3 和 NVFP4。
- TensorRT Edge-LLM（2026）支持 EAGLE-3 speculative decoding、NVFP4 weights、chunked prefill，也就是把 datacenter optimizations 移植到 edge。

### 每个 target 的 Quantization choice

| Target | Format | Notes |
|--------|--------|-------|
| Apple ANE | INT4 weights + FP16 activations | Core ML conversion path |
| Qualcomm Hexagon | QNN INT8 / INT4 | AI Hub converters |
| WebGPU / WebLLM | Q4 MLC (q4f16_1) | Use `mlc_llm convert_weight` + compiled `.wasm`; GGUF is not supported |
| Jetson Orin Nano | Q4 GGUF or TRT-LLM INT4 | Memory-bound |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Edge-LLM path |

### Edge 上的 long-context 陷阱

Llama 3.1 的 128K context 是 datacenter feature。在一台 8 GB RAM 的手机上，4 GB model + 32K tokens 的 2 GB KV cache + OS overhead = OOM。Edge deployments 通常把 context 保持在 4K-8K，除非接受激进 KV quantization（Q4 KV）。

### Voice 是 killer app

Voice agents 对 latency 敏感（first token < 500 ms）。Local inference 完全消除 network latency。再结合 speech-to-text（Whisper Turbo variants 可在 edge 上运行），edge inference 就成为 production-quality voice loop。

### 你应该记住的数字

- Apple M4 / A18 ANE: 38 TOPS。
- Qualcomm Hexagon SD X Elite: 45 TOPS。
- WebLLM M3 Max: Llama 3.1 8B Q4 上约 41 tok/s。
- AGX Orin: 通过 vLLM 在 gpt-oss-20b 上约 40 tok/s。
- Datacenter-edge bandwidth gap: 30-50x。
- WebGPU mobile coverage: ~70-75%（Firefox Android lagging）。

## 实际使用

`code/main.py` 用 bandwidth-bound math 计算不同 edge targets 的 theoretical decode throughput ceilings。它会与 observed benchmarks 对比，并指出瓶颈何时是 bandwidth 而不是 compute。

## 交付成果

本课产出 `outputs/skill-edge-target-picker.md`。给定 platform（iOS/Android/browser/Jetson）、model 和 latency/memory budget，它会选择 quantization format 与 conversion pipeline。

## 练习

1. 运行 `code/main.py`。对于 Snapdragon 8 Gen 3（约 77 GB/s bandwidth）上的 Q4 7B 模型，计算 decode ceiling。与 observed 6-8 tok/s 对比：runtime 是否高效？
2. Android 上的 WebGPU 需要 Chrome v121+。为旧浏览器设计 fallback：通过同一个 OpenAI-compatible API 使用 server-side。
3. 你的 iOS app 需要 4K-context streaming。哪种 model/format combination 能让你在 iPhone 16 上保持低于 4 GB active memory？
4. Jetson AGX Orin 以 40 tok/s 运行 gpt-oss-20b。Jetson Nano 只能放下 3B。如果你的产品同时面向两者，如何统一 inference stack？
5. 论证“WebLLM 在 2026 年是否 production-ready”。引用 coverage、performance 和 Firefox Android gap。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| ANE | “Apple neural engine” | M-series 和 A-series 中的 on-device NPU；unified memory |
| Hexagon | “Qualcomm NPU” | Snapdragon NPU；通过 QNN SDK 访问 |
| WebGPU | “browser GPU” | W3C-standardized browser GPU API；Chrome/Safari 2026 |
| WebLLM | “browser LLM runtime” | MLC-LLM project；Apache 2.0；OpenAI-compatible JS |
| Jetson | “NVIDIA edge” | Orin Nano / AGX / Thor / T4000 family |
| TRT Edge-LLM | “edge TensorRT” | TensorRT-LLM 的 2026 edge port；EAGLE-3 + NVFP4 |
| Unified memory | “shared pool” | CPU 和 NPU 看到同一 RAM；无 copy overhead |
| Bandwidth-bound | “memory limited” | Decode 受读取 weights 的 bytes/sec 限制 |
| Core ML | “Apple conversion” | Apple 用于 ANE-native models 的 framework |
| QNN | “Qualcomm stack” | Qualcomm Neural Network SDK |

## 延伸阅读

- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/) — landscape and benchmarks。
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/) — Orin / AGX / Thor。
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) — 2026 edge port announcement。
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2) — design and benchmarks。
- [Apple Core ML](https://developer.apple.com/documentation/coreml) — ANE-native conversion。
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — Hexagon 的 pre-converted models。
