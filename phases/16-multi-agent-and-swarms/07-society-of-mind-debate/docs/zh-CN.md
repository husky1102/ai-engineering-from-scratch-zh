# Society of Mind 与 Multi-Agent Debate

> Minsky 1986 年的前提——intelligence 是一个 specialists 的 society——每隔十年就会被重新发现一次。2023 年，Du et al. 把它变成一个具体算法：多个 LLM instances 提出答案，读取彼此的答案，critique，并 update。经过 N rounds，它们收敛到一个 consensus，在六个 reasoning 和 factuality tasks 上超过 zero-shot CoT 与 reflection。两个发现最重要：**multiple agents** 和 **multiple rounds** 都独立贡献收益。Society 胜过 single-agent monologue；multi-round exchange 胜过 one-shot voting。

**类型：** 学习 + 构建
**语言：** Python (stdlib)
**先修：** Phase 16 · 04 (Primitive Model)
**时间：** ~60 分钟

## 要解决的问题

Self-consistency——从一个 model 采样多次并取 majority answer——是你能加上的最便宜 reasoning improvement。它有效，但很快会饱和。你把 samples 加倍，可能仍然看不到另一次有意义的提升。

Debate 打破这种饱和。不是从一个 model 得到 N 个 independent samples，而是让 N 个 agents 读取彼此的 reasoning 并 revise。Samples 之间的 correlation 下降（它们不再是 i.i.d.），而 convergence point 往往在 i.i.d. voting 自信地错掉的地方是正确的。

## 核心概念

### Du et al. 2023 算法

来自 arXiv:2305.14325（ICML 2024）：

1. 每个 N agents 都为 question 产生一个 initial answer。
2. 对 round r = 2..R：每个 agent 都会看到其他 agents 在 round r-1 的 answers，并被要求 “considering these, give your updated answer.”
3. R rounds 后，对 final answers 做 majority-vote。

这篇 paper 在 MMLU、GSM8K、biographies、MATH 和 factuality benchmarks 上测试。Debate 持续超过 CoT 和 Self-Reflection。

### 两个独立旋钮

同一篇 paper 的 ablations：

- **Agent count alone**（1 round，N 的 majority vote）在大多数 tasks 上超过 single-agent，但会 plateau。
- **Round count alone**（1 agent 看见自己的 prior reasoning）几乎没有帮助——这是 reflection 的已知弱点。
- **Both together** 产生大的 jumps。多个 agents 之间的 multi-round exchange 驱动收益。

### 为什么它有效

两种机制：

1. **Exposure to disagreement。** 当一个 agent 看到另一个 agent 的 reasoning chain 得出不同 conclusion，它必须 justify 或 update。不管哪种，round r+1 的 context 都比 round r 更丰富。
2. **Correlated error reduction。** 在 self-consistency 中，所有 samples 来自同一个 model，所以 errors 会 correlate——你会 average into a confidently wrong answer。Different models 或 different seeds 会 decorrelate。Different *debated views* 会进一步 decorrelate。

### Heterogeneous debate

A-HMAD 和相关 follow-ups 为不同 agents 使用 *different base models*。Llama + Claude + GPT debating 能降低 monoculture collapse（Lesson 26），因为一个 model family 的 correlated errors 不会被其他 family 共享。

缺点：弱 model 参与 debate 可能把 consensus 拉向它的 wrong answer（见 “Should we be going MAD?”, arXiv:2311.17371）。

### NLSOM——129-agent 扩展

Zhuge et al.（“Mindstorms in Natural Language-Based Societies of Mind,” arXiv:2305.17066）把这个想法扩展到 129-member societies。结果：specialization 和 self-organization 随规模涌现，system 在 visual question answering 等 tasks 上超过 single-agent。

### Failure modes

- **Sycophancy cascade。** 所有 agents 都服从听起来最 confident 的 agent。Debate 坍缩成最响亮的声音。为 adversarial roles 做 prompting（“one agent must argue the counter-position”）会有帮助。
- **Topic drift。** 多轮 debates 会偏离 original question。缓解：每一轮重新注入 question。
- **Compute blowup。** N agents × R rounds = N·R LLM calls，而且每次 call 的 context 都在增长。一个 5-agent、5-round debate 是 25 次 calls，且 context 逐渐增长。每题成本可能超过单次 CoT call 的 10×。

## 动手实现

`code/main.py` 在一道 math question 上运行一个 3-agent × 3-round debate，其中每个 agent 都从不同（可能错误）的 answer 开始。Agents 是 scripted——每个 agent 通过按 scripted confidence 加权 averaging neighbors' answers 来 “updates”。Round-by-round log 中可以看见 convergence。

Demo 展示两个关键效果：

- 单轮 exchange 会让 agents 更接近 correct answer。
- 超过 round 2 的额外 rounds 会出现 diminishing returns（匹配 Du et al. 的 plateau）。

运行：

```text
python3 code/main.py
```

## 实际使用

`outputs/skill-debate-configurator.md` 为新 task 配置 debate：number of agents、number of rounds、heterogeneity（same model vs mixed）、role assignment（symmetric vs one-adversarial）。它还会在运行前估算 token cost。

## 交付成果

如果你 ship debate：

- **Cap rounds at 3。** Du et al. 显示 3 rounds 捕获大部分收益。更多是成本，不是质量。
- **Cap agents at 5。** 超过 5 时，context bloat 和 cost 会占主导。
- **Heterogeneous by default。** Pool 中至少有两个 different base models。
- **Adversarial slot。** 一个 agent 被 prompted to disagree regardless。打断 sycophancy。
- **Log every round。** 隐藏 intermediate rounds 的 debate systems 无法 debug 或 audit。

## 练习

1. 运行 `code/main.py`，然后把 round count 设为 5，观察 diminishing returns。到第几轮 additional convergence 停止？
2. 增加第四个带 adversarial role 的 agent：总是 disagree with the current majority。这会破坏还是改善 convergence？
3. Plot（print）每轮的 agreement score（majority answer 上的 agents fraction）。它什么时候达到 1.0？这是否等价于 “correct”？
4. 阅读 Du et al. Section 4 ablations。用这份 code 复现 “agents-only” vs “rounds-only” vs “both” 结果。
5. 阅读 “Should we be going MAD?”（arXiv:2311.17371），列出 round-robin 之外的两种 debate variants——例如 judge-led、chain-of-debate、adversarial。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Society of Mind | "Minsky's idea" | Intelligence 作为 interacting specialists；1986 framing 现在通过 LLM debate 被 operationalized。 |
| Multi-agent debate | "Agents argue" | N agents propose、critique each other、revise over R rounds，然后 majority-vote。 |
| Consensus | "They agree" | 不是 epistemic truth——只是 fraction-on-majority-answer。可能自信地错误。 |
| Rounds | "Exchange steps" | 一轮 = 每个 agent 读取其他 agents 并更新一次。 |
| Heterogeneous debate | "Mix model families" | 使用不同 base models 来 decorrelate errors。 |
| Sycophancy cascade | "Everyone agrees with the loud one" | Debate failure：agents 不管 correctness，都服从最 confident agent。 |
| NLSOM | "129-agent society" | Natural-language society of mind；Zhuge et al. 的 scaled version。 |
| Correlated error | "Same model, same bug" | Self-consistency 为什么会 saturate；跨 different views 的 debate 会 decorrelate。 |

## 延伸阅读

- [Du et al. — Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325) —— reference paper，ICML 2024
- [Zhuge et al. — Mindstorms in Natural Language-Based Societies of Mind](https://arxiv.org/abs/2305.17066) —— 129-agent NLSOM
- [Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs](https://arxiv.org/abs/2311.17371) —— benchmarks debate variants
- [Debate project page](https://composable-models.github.io/llm_debate/) —— Du et al. 的 code、demos 和 ablation details
