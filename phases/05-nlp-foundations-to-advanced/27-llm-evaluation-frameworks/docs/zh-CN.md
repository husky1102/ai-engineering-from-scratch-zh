# LLM 评估：RAGAS、DeepEval、G-Eval

> Exact-match 和 F1 会漏掉语义等价。人工审核无法规模化。LLM-as-judge 是生产答案，但要有足够校准，才能相信数字。

**类型：** Build
**语言：** Python
**先修：** Phase 5 · 13 (Question Answering), Phase 5 · 14 (Information Retrieval)
**时间：** ~75 minutes

## 要解决的问题

你的 RAG 系统回答：“June 29th, 2007.”
gold reference 是：“June 29, 2007.”
Exact Match 得 0。F1 得约 75%。人类会给 100%。

现在把它乘以 10,000 个测试用例。再乘以 retriever、chunking、prompt 或 model 的每一次变更。你需要一个 evaluator：理解含义、能低成本规模化运行、不对回归说谎，并且能暴露正确的失败模式。

2026 年有三个框架主导这个问题。

- **RAGAS。** Retrieval-Augmented Generation ASsessment。四个 RAG metrics（faithfulness、answer-relevance、context-precision、context-recall），带 NLI + LLM-judge 后端。研究支撑，轻量。
- **DeepEval。** 面向 LLMs 的 pytest。G-Eval、task-completion、hallucination、bias metrics。原生适配 CI/CD。
- **G-Eval。** 一种方法（也是 DeepEval metric）：带 chain-of-thought、自定义 criteria 和 0-1 score 的 LLM-as-judge。

三者都依赖 LLM-as-judge。本课会建立对该方法以及其信任层的直觉。

## 核心概念

![Four evaluation dimensions, LLM-as-judge architecture](../assets/llm-evaluation.svg)

**LLM-as-judge。** 用一个 LLM 根据 rubric 给输出打分，替代静态指标。给定 `(query, context, answer)`，提示 judge LLM：“Score 0-1 on faithfulness.” 返回分数。

为什么它有效：LLMs 能以极低成本近似人类判断。GPT-4o-mini 每个 scored case 约 ~$0.003，使 1000-sample regression eval run 低于 $5。

为什么它会静默失败：

1. **Judge bias。** Judges 偏爱更长答案、来自同模型家族的答案、匹配 prompt 风格的答案。
2. **JSON parsing failures。** 坏 JSON → NaN score → 被静默排除在 aggregate 之外。RAGAS 用户知道这种痛。用 try/except + 显式 failure mode 做 gate。
3. **模型版本漂移。** 升级 judge 会改变每个 metric。冻结 judge model + version。

**RAG 四指标。**

| Metric | Question | Backend |
|--------|----------|---------|
| Faithfulness | 答案中的每个 claim 是否来自 retrieved context？ | NLI-based entailment |
| Answer relevance | 答案是否回应了问题？ | 从答案生成 hypothetical questions；与真实问题比较 |
| Context precision | retrieved chunks 中有多少比例相关？ | LLM-judge |
| Context recall | retrieval 是否返回了所有需要的信息？ | LLM-judge against gold answer |

**G-Eval。** 定义自定义 criterion：“Did the answer cite the correct source?” 框架会自动扩展成 chain-of-thought evaluation steps，然后给 0-1 分。适合 RAGAS 没覆盖的领域专用质量维度。

**Calibration。** 在没有与人类标签对齐之前，永远不要相信原始 judge score。运行 100 个手工标注示例。画 judge vs human。计算 Spearman rho。如果 rho < 0.7，你的 judge rubric 需要继续打磨。

## 动手实现

### Step 1: faithfulness with NLI (RAGAS-style)

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` is any callable: prompt str -> generated str.
# Example: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

把答案拆成原子 claims。用 NLI 检查每个 claim 是否被 retrieved context 支持。Faithfulness = 被支持的比例。

### Step 2: answer relevance

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: any model implementing .encode(texts, normalize_embeddings=True) -> ndarray
# e.g., encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

如果答案暗示的问题和实际提问不同，relevance 会下降。

### Step 3: G-Eval custom metric

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

evaluation steps 就是 rubric。显式步骤比隐式 “score 0-1” prompts 更稳定。

### Step 4: CI gate

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

作为 pytest 文件发布。每个 PR 都运行。出现回归时阻止合并。

### Step 5: toy eval from scratch

见 `code/main.py`。Faithfulness（答案 claims 与 context 的重叠）和 relevance（答案 tokens 与问题 tokens 的重叠）的仅 stdlib 近似。不可用于生产。它展示的是形状。

## 常见陷阱

- **没有 calibration。** 一个与人类标签相关性 0.3 的 judge 就是噪声。上线前要求校准运行。
- **Self-evaluation。** 用同一个 LLM 生成和评判会把分数抬高 10-20%。用不同模型家族做 judge。
- **Pairwise judging 中的位置偏差。** Judges 偏爱先出现的选项。始终随机化顺序，并双向运行。
- **Raw aggregate 隐藏失败。** 平均分 0.85 常常掩盖 5% 灾难性失败。始终检查 bottom quantile。
- **Golden dataset rot。** 未版本化的 eval sets 随时间漂移，会破坏纵向比较。每次变更都给 dataset 打 tag。
- **LLM cost。** 规模化时，judge calls 会主导成本。使用满足 calibration threshold 的最便宜模型：GPT-4o-mini、Claude Haiku、Mistral-small。

## 实际使用

2026 年的栈：

| 使用场景 | 框架 |
|---------|-----------|
| RAG quality monitoring | RAGAS（4 metrics） |
| CI/CD regression gates | DeepEval + pytest |
| 自定义领域 criteria | G-Eval within DeepEval |
| Online live-traffic monitoring | RAGAS with reference-free mode |
| Human-in-the-loop spot checks | LangSmith or Phoenix with annotation UI |
| Red-teaming / safety eval | Promptfoo + DeepEval |

典型栈：RAGAS 做 monitoring，DeepEval 做 CI，G-Eval 做新维度。三者都跑；它们的分歧很有用。

## 交付成果

保存为 `outputs/skill-eval-architect.md`：

```markdown
---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

## 练习

1. **Easy.** 在 10 个带已知 hallucinations 的 RAG examples 上使用 RAGAS。验证 faithfulness metric 能抓到每个 hallucination。
2. **Medium.** 手工把 50 个 QA answers 按 correctness 标成 0-1。用 G-Eval 打分。测量 judge 和 human 之间的 Spearman rho。
3. **Hard.** 用 DeepEval 构建 pytest CI gate。故意让 retriever 回归。验证 gate 失败。通过对最低 10% 做 threshold check 添加 bottom-quantile alerting。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| LLM-as-judge | 用 LLM 打分 | 提示 judge model 根据 rubric 给输出打 0-1 分。 |
| RAGAS | RAG metric library | 开源 eval framework，包含 4 个 reference-free RAG metrics。 |
| Faithfulness | 答案有依据吗？ | 答案 claims 中，被 retrieved context 蕴含的比例。 |
| Context precision | 检索 chunks 相关吗？ | top-K chunks 中实际有用的比例。 |
| Context recall | retrieval 找全了吗？ | gold-answer claims 中被 retrieved chunks 支持的比例。 |
| G-Eval | 自定义 LLM judge | Rubric + chain-of-thought eval steps + 0-1 score。 |
| Calibration | 信任但验证 | judge score 和 human score 之间的 Spearman correlation。 |

## 延伸阅读

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — RAGAS 论文。
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — G-Eval 论文。
- [DeepEval docs](https://deepeval.com/docs/metrics-introduction) — 开源生产栈。
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — 偏差、校准、限制。
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — 集成 RAGAS、DeepEval、Phoenix 的统一框架。
