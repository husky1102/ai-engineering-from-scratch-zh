# Negotiation 与 Bargaining

> Agents 会协商 resources、prices、task allocations 和 terms。2026 年的 benchmark set 已经很清楚：NegotiationArena（arXiv:2402.05863）显示 LLMs 可以通过 persona manipulation（“desperation”）把 payoffs 提升约 20%；"Measuring Bargaining Abilities"（arXiv:2402.15813）显示 buyer 比 seller 更难，而且规模无济于事——他们的 **OG-Narrator**（deterministic offer generator + LLM narrator）把 deal rate 从 26.67% 推到 88.88%；Large-Scale Autonomous Negotiation Competition（arXiv:2503.06416）运行了约 180k 次 negotiations，并发现 **chain-of-thought-concealing** agents 通过向 counterpart 隐藏 reasoning 获胜；Bhattacharya et al. 2025 基于 Harvard Negotiation Project metrics 的排名显示 Llama-3 最 effective，Claude-3 aggressive，GPT-4 fairest。本课实现 Contract Net Protocol（FIPA ancestor，Lesson 02），接入 LLM-style buyer/seller，运行 OG-Narrator-style decomposition，并测量每个 structural choice 如何改变 deal rate。

**类型：** Learn + Build
**语言：** Python (stdlib)
**先修：** Phase 16 · 02 (FIPA-ACL Heritage), Phase 16 · 09 (Parallel Swarm Networks)
**时间：** ~75 分钟

## 要解决的问题

两个 agents 需要就一个价格达成一致。如果只靠 pure language prompts 放任它们自己谈，2024-2026 年的 LLMs 在严格参数化的 bargains 上 close deals 的比例出奇地低（arXiv:2402.15813 中约 27%）。Scale 不能修复这个问题：GPT-4 在结构上并不比 GPT-3.5 更擅长 bargaining；它只是更擅长 bargaining 的*语言*。

根本问题是 LLMs 混淆了两个工作：决定 offer 和叙述 offer。OG-Narrator 把两者分开：deterministic offer generator 计算 numeric moves；LLM 只负责 narrate。Deal rate 跳到约 89%。

这呼应了经典 multi-agent 发现：把 mechanism 与 communication layer 解耦会赢。Contract Net Protocol（FIPA, 1996; Smith, 1980）是参考 task-market mechanism。把 LLM 插入 narration slot，你就得到一个现代 LLM-powered task market。

## 核心概念

### Contract Net，一段话说明

Smith 1980 的 Contract Net Protocol：一个 **manager** 广播一个 **call for proposals (cfp)**；**bidders** 用包含其 offers 的 **propose** messages 响应；manager 选择 winner，向 winner 发送 **accept-proposal**，向 losers 发送 **reject-proposal**。winner 执行工作。可选 message：**refuse**（bidder 拒绝 propose）。FIPA 将其 codified 为 `fipa-contract-net` interaction protocol。

### 为什么 OG-Narrator 胜出

"Measuring Bargaining Abilities of Language Models"（arXiv:2402.15813）观察到：

- LLMs 经常违反 bargaining rules（以荒谬价格 offer、忽略对方的 ZOPA）。
- 它们 anchoring 很差（接受糟糕的 first offers；counter-offer 使用象征性金额而不是 strategic amounts）。
- 单靠 scale 不能修复这些问题。更大的 models 生成更 plausible 的语言，但 strategic error 类似。

OG-Narrator decomposition：

```text
           ┌──────────────────┐        ┌──────────────────┐
  state  → │ offer generator  │ price → │  LLM narrator    │ → message
           │  (deterministic) │        │  (writes the     │
           │                  │        │   human-style    │
           └──────────────────┘        │   accompaniment) │
                                       └──────────────────┘
```

offer generator 是经典 negotiation strategy：Rubinstein bargaining model、Zeuthen strategy，或一个简单的 tit-for-tat over price。LLM 负责 narrate。message 包含 deterministic price 和 natural-language framing。

Deal rate 跳升，因为：
- Prices 留在 bargaining zone 内。
- Anchors 是 strategic 的，不是 emotional 的。
- LLM 做它擅长的事：writing。

### NegotiationArena findings

arXiv:2402.05863 提供 canonical benchmark。Headline findings：

- LLMs 可以通过采用 personas（“I am desperate to sell this by Friday”）把 payoffs 提高约 20%：persona manipulation 是真实 tactic。
- Fair/cooperative agents 会被 adversarial agents exploit；防御需要 explicit counter-posturing。
- Symmetric pair-ups 在约 40% benchmark scenarios 中收敛到 inequitable outcomes。

这不是“LLMs 是糟糕 negotiators”。而是“LLMs negotiate too much like humans，包括可被利用的部分”。

### Chain-of-thought concealment

Large-Scale Autonomous Negotiation Competition（arXiv:2503.06416）在许多 LLM strategies 上运行了约 180k 次 negotiations。赢家会向 counterparts 隐藏 reasoning：

- 如果一个 agent 把 “I will only go to $75; my reservation price is $70” 打印进 publicly visible scratchpad，对手会读到它。
- 赢家私下计算 strategy；output channel 只包含 offer 和 minimum required narration。

这是经典 game theory（Aumann 1976 关于 rationality and information）在 2026 年的回响：revealing your private valuation costs payoff。LLMs 不会直觉到这一点，常常愉快地把自己的 reservations 打进会被 counterpart 看到的 reasoning traces。

Engineering takeaway：把 private-scratchpad context 与 public-message context 分开。不是可选项。

### Bhattacharya et al. 2025 — model rankings

在 Harvard Negotiation Project metrics（principled negotiation、BATNA respect、interest reciprocity）上：

- **Llama-3** 最 effective 于达成 bargains（deal rate + payoff）。
- **Claude-3** 是最 aggressive 的 negotiator（high anchors、late concessions）。
- **GPT-4** 最 fair（pairings 之间 payoff variance 最小）。

这是 2025 年的 snapshot。重点不是哪个 model 在 2026 年 4 月获胜，而是不同 base models 具有持久 negotiation styles。Heterogeneous ensembles（Lesson 15）会把它纳入 diversity source。

### 通过 Contract Net + LLM 做 task allocation

Contract Net 在现代 LLM multi-agent 中的复用方式：

1. Manager agent 把 task 分解成 units。
2. 向 worker agents 广播带 task description 的 `cfp`。
3. 每个 worker 返回一个 offer：`(price, eta, confidence)`，其中 price 可以是 tokens、compute units 或 dollars。
4. Manager 选择 winners（根据 task 可选单个或多个）并 award。
5. Rejected workers 可以自由 bid 其他 tasks。

这能很好地扩展到 100+ workers，因为 coordination 是 broadcast-and-respond，而不是 synchronous chat。Production usage：Microsoft Agent Framework 的 orchestration patterns，以及一些 LangGraph implementations。

### LLM-Stakeholders Interactive Negotiation

NeurIPS 2024（https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf）引入了带 **secret scores** 和 **minimum-acceptance thresholds** 的 multi-party scorable games。每个 stakeholder 拥有 private utilities；LLM 必须从 messages 推断它们。这是 two-party bargaining 向 N-party coalition formation 的泛化。它与具有异构 worker capabilities 的 production task markets 相关。

### Narration-vs-mechanism rule

在所有 2024-2026 negotiation benchmarks 中，一条一致的工程规则是：

> Let the LLM narrate. Do not let the LLM compute the offer.

如果 offer 需要是数字（price、ETA、quantity），就从 negotiation state 确定性生成它，再让 LLM 产生 framing。如果 offer 需要是 proposal structure（task decomposition、role assignment），可以让 LLM 起草，但发送前要用 schema validate 并 constraint-check。

## 动手实现

`code/main.py` 实现：

- `ContractNetManager`, `ContractNetTask`, `Bid` — manager + bidders，broadcast cfp、collect proposals、award。
- `og_narrator_bargain(state, rng)` — OG-Narrator buyer：deterministic Zeuthen-style concession toward the midpoint。
- `seller_response(state, rng)` — deterministic seller counter-offer policy（两种 style 的 structural ground truth）。
- `naive_llm_bargain(state, rng)` — 模拟 all-LLM bargainer：以高 variance 选择 prices，常常在 ZOPA 外。
- Measurement：1000 trials 的 deal rate，每次 trial 重新采样 reservation prices。

运行：

```text
python3 code/main.py
```

预期输出：naive-LLM deal rate ~65-75%；OG-Narrator deal rate ~85-95%；15-25 点差距就是把 offer-generation 从 narration 中分解出来的 structural advantage。另有一个 Contract Net task-market allocation example，包含三个 bidders 和一个 task。

## 实际使用

`outputs/skill-bargainer-designer.md` 设计 bargaining protocol：谁生成 offers（deterministic 或 LLM）、谁负责 narrate、private scratchpads 如何与 public messages 分离，以及如何 monitor deal rate。

## 交付成果

Production bargaining checklist：

- **Separate scratchpad。** Private state 永远不能进入 counterpart 的 context。这不可谈判。
- **Deterministic offer generation。** Prices、quantities、ETAs：compute，不要 prompt。
- **Validate all incoming offers** against a schema。在 protocol boundary reject out-of-ZOPA offers。
- **Bound rounds。** 最多 3-5 rounds；deadlock 时 escalate 给 mediator。
- **Measure deal rate and payoff variance** continuously。deal rate 下降是症状，常常来自 prompt drift 或 counterpart-side attack。
- **Log all rejected proposals** with the deterministic rationale。对 Contract Net managers 来说，losing bidders 需要理解原因。

## 练习

1. 运行 `code/main.py`。确认 OG-Narrator 在 deal rate 上胜过 naive-LLM。差距是多少？
2. 实现 **persona-based payoff improvement**（arXiv:2402.05863）：buyer 只在 narration 中采用 “desperate to buy this week” persona，offer generator 不变。deal rate 或 payoff 会改变吗？
3. 实现 chain-of-thought **concealment**：维护一个不会传给 counterpart 的 private scratchpad string。如果你不小心泄露它会怎样（通过交换 channels 来模拟）？
4. 把 Contract Net 扩展成带 reserve price 的 N-bidder auction。当 bids 都超过 reserve 时，manager 如何在 lowest-price 和 highest-quality 之间决策？你选择哪条 award rule，为什么？
5. 阅读 Bhattacharya et al. 2025 关于 Harvard Negotiation Project metrics 的工作。实现两个具有不同 styles（aggressive vs fair）的 bargainers。测量 symmetric 和 asymmetric pairings 下的 payoff variance。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Contract Net | “Task market” | Smith 1980, FIPA 1996。cfp + propose + accept/reject。canonical task-market。 |
| ZOPA | “Zone of possible agreement” | buyer max 与 seller min 的重叠区域。区域外 offers 无法 close。 |
| BATNA | “Best alternative to a negotiated agreement” | deal 失败时的 fallback。设定你的 reservation price。 |
| OG-Narrator | “Offer generator + narrator” | decomposition：deterministic offer，LLM narration。 |
| Zeuthen strategy | “Risk-minimizing concession” | 根据 risk limits 做 concession 的经典 offer-generator。 |
| Rubinstein bargaining | “Alternating-offer equilibrium” | 带 discounting 的 infinite-horizon bargaining game-theoretic model。 |
| CoT concealment | “Hide your reasoning” | arXiv:2503.06416 中赢家保留 private scratchpads；public channel 只展示 offer。 |
| Persona manipulation | “Emotional posturing” | arXiv:2402.05863：通过 desperation/urgency personas 获得约 20% payoff gain。 |

## 延伸阅读

- [NegotiationArena](https://arxiv.org/abs/2402.05863) — benchmark；persona manipulation 和 exploitation findings
- [Measuring Bargaining Abilities of Language Models](https://arxiv.org/abs/2402.15813) — OG-Narrator 和 buyer-harder-than-seller 结果
- [Large-Scale Autonomous Negotiation Competition](https://arxiv.org/abs/2503.06416) — 约 180k negotiations；chain-of-thought concealment wins
- [LLM-Stakeholders Interactive Negotiation (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf) — 带 secret utilities 的 multi-party scorable games
- [Smith 1980 — The Contract Net Protocol](https://ieeexplore.ieee.org/document/1675516) — classical mechanism，IEEE Transactions on Computers
