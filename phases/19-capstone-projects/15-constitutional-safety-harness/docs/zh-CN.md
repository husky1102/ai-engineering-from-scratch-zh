# 综合项目 15 — Constitutional Safety Harness + Red-Team Range

> Anthropic 的 Constitutional Classifiers、Meta 的 Llama Guard 4、Google 的 ShieldGemma-2、NVIDIA 的 Nemotron 3 Content Safety，以及覆盖多语言的 X-Guard，定义了 2026 年的 safety-classifier stack。garak、PyRIT、NVIDIA Aegis 和 promptfoo 成为标准 adversarial evaluation tools。NeMo Guardrails v0.12 将它们接入生产管线。这个综合项目把所有东西连起来：围绕目标应用的分层 safety harness、运行 6+ attack families 的 autonomous red-team agent，以及一次 constitutional self-critique run，产出可测量的 harmlessness delta。

**类型:** Capstone
**语言:** Python（safety pipeline，red team），YAML（policy configs）
**先修:** Phase 10（LLMs from scratch），Phase 11（LLM engineering），Phase 13（tools），Phase 14（agents），Phase 18（ethics, safety, alignment）
**覆盖阶段:** P10 · P11 · P13 · P14 · P18
**时间:** 25 小时

## 要解决的问题

2026 年 LLM safety 的前沿不在于 classifiers 是否有效（大体有效），而在于如何把它们正确组合到 production app 周围，同时避免 over-refusing 或留下明显漏洞。Llama Guard 4 处理英文 policy violations。X-Guard（132 种语言）处理 multilingual jailbreak。ShieldGemma-2 捕获 image-based prompt injection。NVIDIA Nemotron 3 Content Safety 覆盖企业类别。Anthropic 的 Constitutional Classifiers 是另一种 approach，用在 training 而不是 serving 期间。

攻击演化同样重要。PAIR 和 TAP 自动化 jailbreak discovery。GCG 运行 gradient-based suffix attacks。Multi-turn 和 code-switch attacks 利用 agent memory。任何已部署的 LLM 都需要 red-team range：garak 和 PyRIT 是 canonical drivers；还需要记录 mitigations 和带 CVSS 评分的 findings。

你将加固一个目标应用（一个 8B instruction-tuned model，或来自其他综合项目的 RAG chatbot），对它运行 6+ attack families，并生成 before/after harmlessness measurement。

## 核心概念

Safety pipeline 有五层。**Input sanitize**：移除 zero-width chars、decode base64/rot13、normalize Unicode。**Policy layer**：NeMo Guardrails v0.12 rails（off-domain、toxicity、PII extraction）。**Classifier gate**：输入上用 Llama Guard 4，非英文用 X-Guard，图像输入用 ShieldGemma-2。**Model**：目标 LLM。**Output filter**：输出上用 Llama Guard 4、Presidio PII scrub，并在适用时强制 citation。**HITL tier**：被标记为高风险的输出进入 Slack queue。

Red-team range 在 scheduler 上运行。PAIR 和 TAP 自主发现 jailbreaks。GCG 运行 gradient-based suffix attacks。ASCII / base64 / rot13 encoding attacks。Multi-turn attacks（persona adoption、memory exploitation）。Code-switch attacks（英语混合斯瓦希里语或泰语）。每次运行都会生成带 CVSS scoring 和 disclosure timeline 的 structured findings file。

Constitutional-self-critique run 是 training-time intervention。取 1k 个 harmful-attempt prompts，让模型草拟 response，根据书面 constitution（do-not-harm rules）批判它，并在 critique loop 上重新训练。用 held-out eval 测量 before/after harmlessness delta。

## 架构

```text
request (text / image / multilingual)
      |
      v
input sanitize (strip zero-width, decode, normalize)
      |
      v
NeMo Guardrails v0.12 rails (off-domain, policy)
      |
      v
classifier gate:
  Llama Guard 4 (English)
  X-Guard (multilingual, 132 langs)
  ShieldGemma-2 (image prompts)
  Nemotron 3 Content Safety (enterprise)
      |
      v (allowed)
target LLM
      |
      v
output filter: Llama Guard 4 + Presidio PII + citation check
      |
      v
HITL tier for flagged outputs

parallel:
  red-team scheduler
    -> garak (classic attacks)
    -> PyRIT (orchestrated red team)
    -> autonomous jailbreak agent (PAIR + TAP)
    -> GCG suffix attacks
    -> multilingual / code-switch
    -> multi-turn persona adoption

output: CVSS-scored findings + disclosure timeline + before/after harmlessness delta
```

## 技术栈

- Safety classifiers：Llama Guard 4、ShieldGemma-2、NVIDIA Nemotron 3 Content Safety、X-Guard
- Guardrail framework：NeMo Guardrails v0.12 + OPA
- Red-team drivers：garak（NVIDIA）、PyRIT（Microsoft Azure）、NVIDIA Aegis、promptfoo
- Jailbreak agents：PAIR（Chao et al., 2023）、Tree-of-Attacks（TAP）、GCG suffix
- Constitutional training：Anthropic-style self-critique loop + SFT on critiques
- PII scrub：Presidio
- Target：一个 8B instruction-tuned model，或其他综合项目中的 RAG chatbot

## 动手实现

1. **Target setup。** 在 vLLM 上启动一个 8B instruction-tuned model（或复用其他综合项目中的 RAG chatbot）。这是待测 app。

2. **Safety pipeline wrap。** 围绕目标接入五层 pipeline。验证每一层都可单独观测（Langfuse 中每层一个 span）。

3. **Classifier coverage。** 加载 Llama Guard 4、X-Guard（multilingual）、ShieldGemma-2（image）。在一个小型 labeled set 上运行每个 classifier，建立 baselines。

4. **Red-team scheduler。** 调度 garak、PyRIT、一个 PAIR agent、一个 TAP agent、一个 GCG runner、一个 multi-turn attacker 和一个 code-switch attacker。每个运行在单独 queue 上。

5. **Attack suite。** 六个 attack families：(1) PAIR automated jailbreak，(2) TAP tree-of-attacks，(3) GCG gradient suffix，(4) ASCII / base64 / rot13 encoding，(5) multi-turn persona，(6) multilingual code-switch。报告每类 success rate。

6. **Constitutional self-critique。** 策划 1k 个 harmful-attempt prompts。对每个 prompt，target 草拟 response。Critic LLM 根据书面 constitution（“do no harm,” “cite evidence,” “refuse illegal requests”）评分。Critic 反对的 prompts 会被重写；target 在 critique-improved pairs 上 fine-tune。用 held-out eval 测量 before/after harmlessness。

7. **Over-refusal measurement。** 在 benign prompt suite（例如 XSTest）上跟踪 false-positive rate。目标必须在 benign questions 上保持有帮助。

8. **CVSS scoring。** 对每个成功 jailbreak，按 CVSS 4.0 打分（attack vector、complexity、impact）。生成 disclosure timeline 和 mitigation plan。

9. **Range automation。** 上述所有流程都通过 cron 运行；findings 写入 queue；over-refusal regression alerts 发到 Slack。

## 实际使用

```text
$ safety probe --model=target --family=PAIR --budget=50
[attacker]   PAIR agent running on target
[attack]     attempt 1/50: disguise query as academic research ... blocked
[attack]     attempt 2/50: appeal to roleplay ... blocked
[attack]     attempt 3/50: chain-of-thought coax ... SUCCEEDED
[finding]    CVSS 4.8 medium: roleplay bypass on target
[range]      7 successes out of 50 (14% success rate)
```

## 交付成果

`outputs/skill-safety-harness.md` 是交付物。一个 production-grade layered safety pipeline，加上可复现的 red-team range，并包含 before/after harmlessness deltas。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | Attack-surface coverage | 运行 6+ attack families，覆盖 2+ languages |
| 20 | True-positive / false-positive trade-off | Attack block rate vs XSTest benign pass rate |
| 20 | Self-critique delta | held-out eval 上的 before/after harmlessness |
| 20 | Documentation and disclosure | 带 timeline 的 CVSS-scored findings |
| 15 | Automation and repeatability | 所有内容通过 cron 和 alerts 运行 |
| **100** | | |

## 练习

1. 在 RAG chatbot 上运行 garak 的 prompt-injection plugin，并比较有无 output-filter layer 时的 attack success rate。

2. 添加第七个 attack family：通过 retrieved documents 进行 indirect prompt injection。测量需要的额外防御。

3. 实现 “refuse-with-help” 模式：当 guardrail 阻止时，target 提供一个更安全的相关答案，而不是平铺拒绝。测量 XSTest delta。

4. Multilingual coverage gap：找出 X-Guard 表现较差的一种语言。提出面向它的 fine-tune dataset。

5. 在 30B model 上运行 constitutional self-critique，并测量 delta 是否随规模扩大。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| Layered safety | “Defense in depth” | 输入、gate、输出、HITL 上的多层 guardrails |
| Llama Guard 4 | “Meta's safety classifier” | 2026 年 reference input/output content classifier |
| PAIR | “Jailbreak agent” | Chao et al. 关于 LLM-driven jailbreak discovery 的论文 |
| TAP | “Tree-of-Attacks” | PAIR 的 tree-search 变体 |
| GCG | “Greedy coordinate gradient” | Gradient-based adversarial suffix attack |
| Constitutional self-critique | “Anthropic-style training” | Target drafts -> critic scores -> rewrite -> retrain |
| XSTest | “Benign probe set” | 用于 over-refusal regression 的 benchmark |
| CVSS 4.0 | “Severity score” | Safety findings 的标准 vulnerability scoring |

## 延伸阅读

- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — training-time reference
- [Meta Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 input/output classifier
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b) — image + multimodal safety
- [NVIDIA Nemotron 3 Content Safety](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/) — enterprise reference
- [X-Guard (arXiv:2504.08848)](https://arxiv.org/abs/2504.08848) — 132-language multilingual safety
- [garak](https://github.com/NVIDIA/garak) — NVIDIA red-team toolkit
- [PyRIT](https://github.com/Azure/PyRIT) — Microsoft red-team framework
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — rail framework
- [PAIR (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — jailbreak agent paper
