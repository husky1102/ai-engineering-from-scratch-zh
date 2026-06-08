# Capstone 07 — 端到端微调 Pipeline（Data to SFT to DPO to Serve）

> 一个基于你自己的数据训练的 8B model，在你自己的偏好上做 DPO 对齐，量化，speculative-decoded，并以可衡量的 $/1M tokens 提供服务。2026 年的开放栈是 Axolotl v0.8、TRL 0.15、用于迭代的 Unsloth、用于量化的 GPTQ/AWQ/GGUF、以及带 EAGLE-3 的 vLLM 0.7。这个 capstone 要可复现地跑完整个 pipeline：YAML 进入，served endpoint 输出，并在 2026 Model Openness Framework 下发布 model card。

**类型：** Capstone
**语言：** Python (pipeline), YAML (configs), Bash (scripts)
**先修：** Phase 2 (ML), Phase 3 (DL), Phase 7 (transformers), Phase 10 (LLMs from scratch), Phase 11 (LLM engineering), Phase 17 (infrastructure), Phase 18 (safety)
**练习阶段：** P2 · P3 · P7 · P10 · P11 · P17 · P18
**时间：** 35 hours

## 要解决的问题

2026 年，每个严肃的 AI team 都会随时保留一条 fine-tuning pipeline。不是因为他们会发布 frontier base model，而是因为下游适配才是可衡量收益所在：domain SFT、基于 labeled preferences 的 DPO、用于 speculative decoding 的 distilled drafts、用 EAGLE-3 serving。Axolotl v0.8 处理 multi-GPU SFT configs。TRL 0.15 处理 DPO 和 GRPO。Unsloth 让你能快速做 single-GPU iteration。带 EAGLE-3 的 vLLM 0.7 能在不损失质量的情况下把 decode throughput 提升 2-3x。工具已经可用；真正的功夫在 YAMLs、data hygiene 和 eval discipline。

你会将一个 8B base（Llama 3.3、Qwen3 或 Gemma 3）在 task-specific data 上跑 SFT 然后 DPO，量化用于 serving，并用 lm-evaluation-harness、RewardBench-2、MT-Bench-v2 和 MMLU-Pro 衡量增益。你会在 2026 Model Openness Framework 下产出 model card。重点是可复现性：一个命令端到端重跑整个 pipeline。

## 核心概念

pipeline 有五个阶段。**Data**：dedup（MinHash / Datatrove）、quality filter（Nemotron-CC style classifier）、PII scrub、针对 public benchmark contamination 的 split-hygiene check。**SFT**：Axolotl YAML、8xH100 上的 ZeRO-3、cosine schedule、packed sequences、2-3 epochs。**DPO or GRPO**：TRL config、1 epoch、preference pairs 可以来自 human-labeled 或 model-judged、beta tuning。**Quantize**：GPTQ + AWQ + GGUF，提供部署灵活性。**Serve**：带 EAGLE-3 speculative heads 的 vLLM 0.7（或带 SpecForge 的 SGLang）、K8s deployment、基于 queue-wait 的 HPA。

ablations 是交付物：在三个 task-specific benchmarks 上比较 SFT-only vs SFT+DPO vs SFT+GRPO。Serving metrics：batch 1 / 8 / 32 下的 tokens/s、EAGLE-3 acceptance rate、$/1M tokens。Safety eval：Llama Guard 4 pass rate。Model card：bias evaluations、reproducibility seeds、data licensing。

## 架构

```text
raw data (HF datasets + internal)
    |
    v
Datatrove dedup + Nemotron-CC quality filter + PII scrub
    |
    v
split hygiene (MMLU-Pro contamination check)
    |
    v
Axolotl SFT config (YAML)  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO config       ---> 4xH100, 1 epoch
    |
    v
GPTQ + AWQ + GGUF quantize
    |
    v
vLLM 0.7 + EAGLE-3 speculative decoding
    |
    v
K8s deployment, HPA on queue-wait
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
model card (2026 MOF) + safety eval (Llama Guard 4)
```

## 技术栈

- Data: Datatrove for dedup, Nemotron-CC classifier for quality, Presidio for PII
- Base: Llama 3.3 8B, Qwen3 14B, or Gemma 3 12B
- SFT: Axolotl v0.8 with ZeRO-3, Flash Attention 3, packed sequences
- Preference tuning: TRL 0.15 for DPO or GRPO; Unsloth for single-GPU iteration
- Quantization: GPTQ (Marlin), AWQ, GGUF via llama.cpp
- Serving: vLLM 0.7 with EAGLE-3 speculative decoding (or SGLang 0.4 + SpecForge)
- Eval: lm-evaluation-harness, RewardBench-2, MT-Bench-v2, MMLU-Pro
- Safety eval: Llama Guard 4, ShieldGemma-2
- Infrastructure: Kubernetes + NVIDIA device plugin, HPA on queue-wait metric
- Observability: W&B for training, Langfuse for inference

## 动手实现

1. **Data pipeline。** 在 raw corpus 上运行 Datatrove dedup。应用 Nemotron-CC-style quality classifier。Presidio scrub PII。用明确 seed 写出 train/val splits。

2. **Contamination check。** 对每个 validation split，计算其与 MMLU-Pro、MT-Bench-v2、RewardBench-2 test sets 的 MinHash。拒绝任何 overlap。

3. **Axolotl SFT。** 使用包含 ZeRO-3、FA3、sequence packing 的 YAML。在 8xH100 上训练 2-3 epochs。记录到 W&B。

4. **TRL DPO / GRPO。** 取 SFT checkpoint，在 preference pairs 上跑一个 epoch 的 DPO（或用 math/code 上可验证 reward 做 GRPO）。扫 beta。

5. **Quantize。** 产出三种 quants：GPTQ-INT4-Marlin、AWQ-INT4、GGUF-Q4_K_M for llama.cpp。记录 size 和 nominal throughput。

6. **用 speculative decoding 服务。** vLLM 0.7 config 使用通过 Red Hat Speculators 训练的 EAGLE-3 draft heads。测量 batch 1 / 8 / 32 下的 acceptance rate 和 tail latency。在相同 eval 上报告相对 Anthropic / OpenAI 的 $/1M tokens。

7. **Eval matrix。** 在 base、SFT-only、SFT+DPO、SFT+GRPO 上运行 lm-eval-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro。产出表格。

8. **Safety eval。** dev set 上的 Llama Guard 4 pass rate。ShieldGemma-2 output filter。

9. **Model card。** MOF 2026 template：data、training、eval、safety、license，以及带 YAMLs 和 commit SHAs 的 reproducibility section。

## 实际使用

```text
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k deduped, 12k filtered, 280k accepted (seed=7)
[SFT]     3 epochs, 8xH100, 6h12m, val loss 1.42 -> 1.03
[DPO]     1 epoch, beta=0.08, 4xH100, 1h40m
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 acceptance 0.74, p99 126ms @ bs=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    model-card.md generated under 2026 MOF
```

## 交付成果

`outputs/skill-finetuning-pipeline.md` 描述交付物。单个命令会从 data 到 SFT 到 DPO 到 quant 到 serve 到 eval 跑完整流程，并输出 model card + served endpoint。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | Eval delta vs base | 在 target tasks 上测量 gain（MMLU-Pro、MT-Bench-v2、task-specific） |
| 20 | Pipeline reproducibility | 一个命令用 identical seeds 端到端重跑 |
| 20 | Data hygiene | Dedup rate、PII scrub coverage、contamination check green |
| 20 | Serving efficiency | bs=1/8/32 下的 tokens/s、EAGLE-3 acceptance rate、$/1M tokens |
| 15 | Model card + safety eval | 2026 MOF completeness + Llama Guard 4 pass rate |
| **100** | | |

## 练习

1. 在同一个 task-specific benchmark 上运行 SFT-only vs SFT+DPO vs SFT+GRPO。报告哪种 preference method 获胜，以及优势有多大。

2. 将 Llama 3.3 8B 换成 Qwen3 14B。在 matched quality 下测量 $/1M tokens。

3. 测量 EAGLE-3 acceptance rate 在 domain data 与 generic ShareGPT 上的差异。报告 delta 以及它对 latency budgets 意味着什么。

4. 注入 1% contamination（把 MMLU-Pro answers 泄漏进 training data），然后重跑 eval。观察 MMLU-Pro accuracy 不真实地跳升。构建一个能抓住它的 contamination-check CI gate。

5. 添加 LoRA SFT 作为 full fine-tune 的替代方案。在 10x lower memory 下测量 quality gap。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| Axolotl | "SFT trainer" | 用 YAML 驱动的统一 trainer，支持 SFT、DPO 和 distillation |
| TRL | "Preference tuner" | Hugging Face library，用于 LLMs 上的 DPO、GRPO、PPO |
| GRPO | "Group-relative policy optimization" | DeepSeek R1 的 RL recipe，使用 verifiable rewards |
| EAGLE-3 | "Speculative decoding draft" | 预测未来 N 个 tokens 的 draft heads；vLLM 用 target model 验证 |
| MOF | "Model Openness Framework" | 2026 年标准，用于按 data、code、license 给 model releases 分级 |
| Contamination check | "Split hygiene" | 基于 MinHash 检测 test-set leakage into training |
| Acceptance rate | "EAGLE / MTP metric" | target model 接受 drafted tokens 的比例 |

## 延伸阅读

- [Axolotl documentation](https://axolotl-ai-cloud.github.io/axolotl/) — reference SFT / DPO trainer
- [TRL documentation](https://huggingface.co/docs/trl) — DPO and GRPO reference implementations
- [Unsloth](https://github.com/unslothai/unsloth) — single-GPU iteration reference
- [DeepSeek R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — GRPO methodology
- [vLLM + EAGLE-3 documentation](https://docs.vllm.ai) — reference serving stack
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — alternate speculative-decoding trainer
- [Model Openness Framework 2026](https://isocpp.org/) — open-release grading standard
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — canonical eval runner
