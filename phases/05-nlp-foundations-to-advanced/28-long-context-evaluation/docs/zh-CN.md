# 长上下文评估：NIAH、RULER、LongBench、MRCR

> Gemini 3 Pro 宣称 10M tokens 上下文。在 1M tokens 时，8-needle MRCR 掉到 26.3%。标称 ≠ 可用。长上下文评估告诉你正在上线的模型到底有多少真实容量。

**类型：** Learn
**语言：** Python
**先修：** Phase 5 · 13 (Question Answering), Phase 5 · 23 (Chunking Strategies)
**时间：** ~60 minutes

## 要解决的问题

你有一份 200 页合同。模型声称有 1M-token context。你把合同贴进去，问：“终止条款是什么？”模型回答了，但它答的是封面内容，因为终止条款在 120k tokens 深处，已经超过了模型实际会关注的位置。

这就是 2026 年的 context-capacity gap。规格表说 1M 或 10M。现实说其中 60-70% 才可用，而且“可用”取决于任务。

- **Retrieval（single needle in haystack）：** frontier models 在标称最大值前接近完美。
- **Multi-hop / aggregation：** 大多数模型超过约 128k 后急剧退化。
- **Reasoning over dispersed facts：** 最先失败的任务。

长上下文评估会测量这些轴。本课会说明这些 benchmark 的名字、它们实际测量什么，以及如何为你的领域构建自定义 needle test。

## 核心概念

![NIAH baseline, RULER multi-task, LongBench holistic](../assets/long-context-eval.svg)

**Needle-in-a-Haystack（NIAH, 2023）。** 在长上下文的受控深度放入一个事实（“the magic word is pineapple”）。询问模型是否能取回它。扫过 depth × length。原始长上下文 benchmark。Frontier models 现在能把它跑满；它是必要但不充分的 baseline。

**RULER（Nvidia, 2024）。** 4 类下的 13 种任务类型：retrieval（single / multi-key / multi-value）、multi-hop tracing（variable tracking）、aggregation（common word frequency）、QA。可配置 context length（4k 到 128k+）。它能揭示那些跑满 NIAH、却在 multi-hop 上失败的模型。2024 版本中，17 个声称 32k+ context 的模型里，只有一半能在 32k 保持质量。

**LongBench v2（2024）。** 503 个 multiple-choice questions、8k-2M 词 contexts、六类任务：single-doc QA、multi-doc QA、long in-context learning、long dialogue、code repo、long structured data。用于真实世界长上下文行为的生产 benchmark。

**MRCR（Multi-Round Coreference Resolution）。** 大规模多轮共指。8-needle、24-needle、100-needle variants。暴露模型在注意力退化前能同时处理多少事实。

**NoLiMa。** “Non-lexical needle。” needle 和 query 没有字面重叠；retrieval 需要一步语义推理。比 NIAH 更难。

**HELMET。** 拼接许多文档，从任意一个文档提问。测试 selective attention。

**BABILong。** 把 bAbI reasoning chains 嵌入无关 haystacks。测试 reasoning-in-a-haystack，而不只是 retrieval。

### 实际应该报告什么

- **Advertised context window。** 规格表数字。
- **Effective retrieval length。** 在某个阈值（例如 90%）通过 NIAH 的长度。
- **Effective reasoning length。** 在该阈值通过 multi-hop 或 aggregation 的长度。
- **Degradation curve。** Accuracy vs context length，按任务类型分别绘图。

给你的规格表两个数字：retrieval-effective 和 reasoning-effective。通常 reasoning-effective 是标称窗口的 25-50%。

## 动手实现

### Step 1: a custom NIAH for your domain

见 `code/main.py`。骨架：

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # Repeat filler until long enough to fill the haystack body.
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

扫过 `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}。画 heatmap。这就是你的目标模型的 NIAH card。

### Step 2: a multi-needle variant

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

像 “What are the three magic words?” 这样的问题要求取回全部三个。Single-needle success 不能预测 multi-needle success。

### Step 3: multi-hop variable tracing (RULER-style)

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

答案需要串联三个赋值。Frontier models 在 128k 处通常会降到 50-70% accuracy。

### Step 4: LongBench v2 on your stack

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

报告 per-category accuracy。Aggregate scores 会隐藏巨大的任务级差异。

## 常见陷阱

- **只用 NIAH 评估。** 在 1M tokens 上通过 NIAH，并不能说明 multi-hop 能力。始终运行 RULER 或自定义 multi-hop test。
- **均匀深度采样不足。** 许多实现只测 depth=0.5。要测 depth=0、0.25、0.5、0.75、1.0；“lost in the middle” 效应是真实的。
- **与 filler 有词法重叠。** 如果 needle 和 filler 共享关键词，retrieval 会变得平凡。使用 NoLiMa 风格的不重叠 needles。
- **忽略延迟。** 1M-token prompts 需要 30-120 秒 prefill。和 accuracy 一起测量 time-to-first-token。
- **Vendor-self-reported numbers。** OpenAI、Google、Anthropic 都发布自己的分数。始终在你的 use case 上独立重跑。

## 实际使用

2026 年的栈：

| 场景 | Benchmark |
|-----------|-----------|
| 快速 sanity check | Custom NIAH at 3 depths × 3 lengths |
| 生产模型选择 | RULER（13 tasks）at your target length |
| 真实世界 QA 质量 | LongBench v2 single-doc-QA subset |
| Multi-hop reasoning | BABILong or custom variable-tracing |
| 对话 / dialogue | MRCR 8-needle at your target length |
| 模型升级回归 | Fixed in-house NIAH + RULER harness, run on every new model |

生产经验法则：在你计划使用的长度上跑过 NIAH + 1 个 reasoning task 之前，永远不要相信 context window。

## 交付成果

保存为 `outputs/skill-long-context-eval.md`：

```markdown
---
name: long-context-eval
description: Design a long-context evaluation battery for a given model and use case.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

Given a target model, target context length, and use case, output:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. Depths 0, 0.25, 0.5, 0.75, 1.0 at each length.
3. Metrics. Retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. Effective retrieval length (90% pass) and effective reasoning length (70% pass). Report both.
5. Regression. Fixed harness, rerun on every model upgrade, surface deltas.

Refuse to trust a context window from the model card alone. Refuse NIAH-only evaluation for any multi-hop workload. Refuse vendor self-reported long-context scores as independent evidence.
```

## 练习

1. **Easy.** 构建一个 3 个深度（0.25、0.5、0.75）× 3 个长度（1k、4k、16k）的 NIAH。在任意模型上运行。把 pass rate 画成 3×3 heatmap。
2. **Medium.** 添加 3-needle variant。测量每个长度下全部 3 个 needle 的 retrieval。和同长度 single-needle pass rate 比较。
3. **Hard.** 构建一个嵌入 64k filler 的 variable-tracing task（X1 → X2 → X3，3 hops）。跨 3 个 frontier models 测量 accuracy。报告每个模型的 effective reasoning length。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| NIAH | Needle in haystack | 在 filler 中植入一个事实，要求模型取回。 |
| RULER | 加强版 NIAH | 覆盖 retrieval / multi-hop / aggregation / QA 的 13 种任务类型。 |
| Effective context | 真实容量 | accuracy 仍保持在阈值以上的长度。 |
| Lost in the middle | 深度偏差 | 模型对长输入中间位置的内容关注不足。 |
| Multi-needle | 一次多个事实 | 多个植入点；测试 attention juggling，而不只是 retrieval。 |
| MRCR | Multi-round coref | 8、24 或 100-needle coreference；暴露 attention saturation。 |
| NoLiMa | Non-lexical needle | Needle 和 query 没有字面 tokens 重叠；需要 reasoning。 |

## 延伸阅读

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — 原始 NIAH repo。
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — multi-task benchmark。
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — 真实世界 long-context eval。
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — 更难的 needles。
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — reasoning-in-haystack。
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — depth-bias 论文。
