# 对话状态跟踪

> “我想找一家北边的便宜餐厅……其实改成中等价位……再加上意大利菜。” 三轮对话，三次状态更新。DST 让 slot-value 字典保持同步，预订才会真正成功。

**类型：** Build
**语言：** Python
**先修：** Phase 5 · 17（Chatbots），Phase 5 · 20（Structured Outputs）
**时间：** ~75 分钟

## 要解决的问题

在面向任务的对话系统中，用户目标会被编码成一组 slot-value 对：`{cuisine: italian, area: north, price: moderate}`。每一次用户发言都可能新增、修改或删除一个 slot。系统必须读取整段对话，并正确输出当前状态。

只要错一个 slot，系统就会订错餐厅、排错航班，或扣错银行卡。DST 是“用户说了什么”和“后端执行什么”之间的铰链。

为什么即使在 2026 年有了 LLM，它仍然重要：

- 合规敏感领域（银行、医疗、航空预订）需要确定性的 slot 值，而不是自由形式生成。
- Tool-use agent 在调用 API 之前仍然需要 slot resolution。
- 多轮修正比看起来更难：“其实不是，改成 Thursday。”

现代流水线是：经典 DST 概念 + LLM extractor + structured-output guardrail。

## 核心概念

![DST: dialog history → slot-value state](../assets/dst.svg)

**任务结构。** schema 定义 domain（restaurant、hotel、taxi）及其 slot（cuisine、area、price、people）。每个 slot 可以为空，可以填入封闭集合中的值（price: {cheap, moderate, expensive}），也可以是自由形式值（name: "The Copper Kettle"）。

**两种 DST 表述。**

- **Classification。** 对每个 `(slot, candidate_value)` 对预测 yes/no。适合封闭词表 slot。2020 年前的标准做法。
- **Generation。** 给定对话，将 slot value 作为自由文本生成。适合开放词表 slot。现代默认方案。

**指标。** Joint Goal Accuracy（JGA）——*所有* slot 都正确的 turn 占比。全对才算对。到 2026 年，MultiWOZ 2.4 leaderboard 顶部大约在 83%。

**架构。**

1. **Rule-based（slot regex + keyword）。** 对窄领域是很强的 baseline。可调试。
2. **TripPy / BERT-DST。** 使用 BERT 编码的 copy-based generation。LLM 之前的标准。
3. **LDST（LLaMA + LoRA）。** 使用 domain-slot prompt 的 instruction-tuned LLM。在 MultiWOZ 2.4 上达到 ChatGPT 级质量。
4. **Ontology-free（2024-26）。** 跳过 schema；直接生成 slot name 和 value。能处理开放领域。
5. **Prompt + structured output（2024-26）。** LLM + Pydantic schema + constrained decoding。5 行代码，足够生产使用。

### 经典失败模式

- **跨轮指代。** “就选第一个吧。” 需要解析是哪个选项。
- **覆盖还是追加。** 用户说 “add Italian”。是替换 cuisine，还是追加？
- **隐式确认。** “OK cool” —— 这是否接受了系统给出的 booking？
- **修正。** “Actually make it 7 pm.” 必须更新时间，同时不清空其他 slot。
- **指向上一条系统话语的 coreference。** “Yes, that one.” 哪个 “that”？

## 动手实现

### Step 1: rule-based slot extractor

见 `code/main.py`。Regex + synonym dictionary 可以覆盖窄领域中 70% 的典型话语：

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

一旦超出规范词表就很脆弱。它适合确定性的 slot confirmation。

### Step 2: state update loop

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

三个不变量：

- 永远不要重置用户没有触碰的 slot。
- 显式否定（“never mind the cuisine”）必须清空。
- 用户修正（“actually...”）必须覆盖，而不是追加。

### Step 3: LLM-driven DST with structured output

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic 保证得到有效的 state object。没有 regex，没有 schema mismatch，也没有幻觉 slot。

### Step 4: JGA evaluation

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

校准问题：系统在多少比例的 turn 上能把所有 slot 全部做对？对 MultiWOZ 2.4 而言，2026 年顶级系统约为 80-83%。你的窄词表 in-domain 系统应该超过这个水平，否则 LLM baseline 就会胜出。

### Step 5: handling correction

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

检测到修正时，覆盖最近更新的 slot，而不是追加。没有 LLM 帮助时，这件事很难做对。现代模式是：始终让 LLM 从历史重新生成完整 state，而不是增量更新——这会自然处理修正。

## Pitfalls

- **完整历史再生成成本。** 每一轮都让 LLM 重新生成 state，总 token 成本是 O(n²)。限制 history 长度，或汇总更早的 turn。
- **Schema drift。** 事后添加新 slot 会破坏旧训练数据。给 schema 版本化。
- **大小写敏感。** "Italian" vs "italian" vs "ITALIAN" —— 到处都要 normalize。
- **隐式继承。** 如果用户之前已经指定 “for 4 people”，后来请求不同时间时不应清空 people。始终传入完整 history。
- **自由形式 vs 封闭集合。** name、time、address 需要自由形式 slot；cuisine 和 area 是封闭集合。schema 中要混合两者。

## 实际使用

2026 年实际会交付的 stack：

| 场景 | 方法 |
|-----------|----------|
| 窄领域（一两个 intent） | Rule-based + regex |
| 宽领域，有标注数据 | LDST（在 MultiWOZ 风格数据上训练的 LLaMA + LoRA） |
| 宽领域，无标签，production-ready | LLM + Instructor + Pydantic schema |
| 语音 / voice | ASR + normalizer + LLM-DST |
| 多领域 booking flow | 带 per-domain Pydantic model 的 schema-guided LLM |
| 合规敏感 | Rule-based primary，带 confirmation flow 的 LLM fallback |

## 交付成果

保存为 `outputs/skill-dst-designer.md`：

```markdown
---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.
```

## 练习

1. **Easy.** 在 `code/main.py` 中为 3 个 slot（cuisine、area、price）构建 rule-based state tracker。用 10 段手写 dialogue 测试。测量 JGA。
2. **Medium.** 在同一数据集上使用 Instructor + Pydantic + small LLM。比较 JGA。检查最难的 turn。
3. **Hard.** 同时实现两者并做路由：rule-based primary；当 rule-based 输出的高置信 slot 少于 2 个时使用 LLM fallback。测量组合 JGA 和每轮 inference cost。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| DST | Dialogue state tracking | 跨对话轮维护 slot-value dict。 |
| Slot | 用户意图单元 | 后端需要的命名参数（cuisine、date）。 |
| Domain | 任务领域 | Restaurant、hotel、taxi —— slot 集合。 |
| JGA | Joint Goal Accuracy | 每个 slot 都正确的 turn 占比。全对才算对。 |
| MultiWOZ | benchmark | 多领域 WOZ 数据集；标准 DST 评估。 |
| Ontology-free DST | 无 schema | 直接生成 slot name 和 value，没有固定列表。 |
| Correction | "Actually..." | 覆盖已填 slot 的 turn。 |

## 延伸阅读

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) —— 经典 benchmark。
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) —— 用于 DST 的 LLaMA + LoRA instruction tuning。
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) —— copy-based DST 主力方法。
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) —— 基于 EM 的无监督 TOD。
- [MultiWOZ leaderboard](https://github.com/budzianowski/multiwoz) —— 经典 DST 结果。
