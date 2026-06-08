# Agent Economies、Token Incentives、Reputation

> Long-horizon autonomous agents（METR 的 1-hour 到 8-hour work-curve）需要 economic agency。正在出现的 **5-layer stack** 是：**DePIN**（physical compute）→ **Identity**（W3C DIDs + reputation capital）→ **Cognition**（RAG + MCP）→ **Settlement**（account abstraction）→ **Governance**（Agentic DAOs）。Production agent-incentive networks 包括 **Bittensor**（TAO subnets reward task-specific models）、**Fetch.ai / ASI Alliance**（ASI-1 Mini LLM + FET token）和 **Gonka**（transformer-based PoW，将 compute 重新分配到 productive AI tasks）。Academic work：AAMAS 2025 的 decentralized LaMAS 使用 **Shapley-value credit attribution** 公平奖励 contributing agents；Google Research “Mechanism design for large language models” 提出带 monotone aggregation 的 **token auctions**，使用 second-price payment。本课构建一个最小 agent marketplace，把 Shapley-value credit attribution 应用到 multi-agent pipeline，并运行 second-price token auction，让 game-theory machinery 具体落地。

**类型：** 学习
**语言：** Python (stdlib)
**先修：** Phase 16 · 16 (Negotiation and Bargaining), Phase 16 · 09 (Parallel Swarm Networks)
**时间：** ~75 分钟

## 要解决的问题

当 agents 共同产生价值但需要被 individually rewarded 时，multi-agent systems 会复杂起来。Classical mechanisms：equal split、last-contributor-takes-all，要么不公平，要么可被 game。通过 Shapley values 进行 coalition-based rewarding，构造上公平但计算昂贵。2025-2026 literature 推出了有用 approximations：Shapley sampling、monotone aggregation auctions，以及从 confirmed contributions 累积的 on-chain reputation。

除了 credit attribution，这个领域已经转向实际 economic agents：Bittensor TAO 奖励 mining compute 来 fine-tune subnet-specific models，Fetch.ai/ASI 用 FET tokens 奖励 ASI-1 Mini LLM usage，Gonka 将 transformer proof-of-work 转向 productive AI tasks。能 autonomous transact 的 agents 今天已经存在；问题是如何 align incentives。

本课把 agent economies 视为一个特定 problem family：credit attribution、mechanism design 和 reputation，并用最小数学构建每一个，让 ideas 粘住。

## 核心概念

### 5-layer agent-economy stack

1. **DePIN (physical compute).** 租用 GPU、storage、bandwidth 的 decentralized infrastructure。Bittensor subnets、Render Network、Akash。不是 agent-specific；agents 使用它。
2. **Identity.** W3C Decentralized Identifiers（DIDs）给每个 agent 一个独立于任何 platform 的 durable ID。Reputation accrues to DID。Agent Network Protocol（ANP）使用 DID 作为 discovery layer。
3. **Cognition.** agent 的 reasoning loop：LLM + RAG + MCP。这是其他 phases 构建的内容。
4. **Settlement.** Account abstraction（ERC-4337）让 agents 能从自己的 balances 支付 gas，而无需持有 ETH。agents 可以为 services、彼此或 compute 付款。
5. **Governance.** Agentic DAOs：humans *and* agents 对 protocol changes 投票的 governance structures，voting power 与 reputation 绑定。

不是每个 production system 都使用五层。Bittensor 使用 1、2，部分使用 3 和 4，不使用 5。OpenAI agents 除了 3 之外都不用。这个 stack 是 reference map，不是 requirement。

### Bittensor、Fetch.ai、Gonka：实际运行的是什么

**Bittensor (TAO).** Subnets 是 specialized tasks（language modeling、image generation、forecasting）。Miners 提交 model outputs。Validators rank them；stake-weighted scoring 分发 TAO rewards。每个 subnet 有自己的 evaluation。经济 lesson：按 task-specific output quality 支付，而不是按 compute used 支付。

**Fetch.ai / ASI Alliance.** ASI-1 Mini LLM 运行在 Fetch.ai network 上；users 用 FET tokens 为 inference 付费。agents-as-peers narrative 在这里更强：Fetch 上的 agent 可以调用另一个 agent 处理 task，并用 FET 支付。

**Gonka.** Transformer proof-of-work：“work” 是 transformer 的 forward passes。Miners 通过运行有已知 correct outputs（来自 training data）的 inference tasks 获利。Resource-productive PoW 替代 hash-based PoW。

截至 2026 年 4 月，三者都是 production-grade。Payoff distribution 不同。Bittensor 按 subnet validators 的相对 quality 奖励；Fetch 按 paying users 衡量的 utility 奖励；Gonka 奖励 verifiable inference work。

### Shapley-value credit attribution

三个 agents 协作完成 task。output score 是 0.8。谁贡献了什么？

Shapley value：满足四个 axioms（efficiency、symmetry、linearity、null）的唯一 credit allocation。对 agent `i`：

```text
shapley(i) = (1/N!) * sum over all orderings O of (v(S_i_O ∪ {i}) - v(S_i_O))
```

其中 `S_i_O` 是 ordering `O` 中位于 `i` 之前的 agents set。实践中：枚举所有 permutations，记录每个 agent 在每个 permutation 中的 marginal contribution，取平均。

N=3 agents 时有 6 个 permutations。N=10 时有 3.6M，所以实践中会 sample orderings，而不是 enumerate。

### 用于 aggregation 的 Second-price auction

Google Research（“Mechanism design for large language models”）提出用 second-price token auctions 聚合 LLM outputs。设定：N 个 agents 各自提出 completion；每个 agent 对被选中有 private value。auctioneer 选择 highest-value proposal，并支付 *second-highest* value。在 monotone aggregation 下（value 取决于选中哪个 proposal，而不是 bid 了多少），这是 truthful 的：agents 会 bid true value。

这对 LLM systems 很重要：你可以把 completion tasks 外包给多个 agents，它们有不同 pricing；auction 选择 best 并公平支付，agents 没有 incentive 去 misreport。

### Reputation capital

一个 DID-bound reputation score 从 confirmed contributions 中累积。简单 update rule：

```text
rep(i, t+1) = alpha * rep(i, t) + (1 - alpha) * contribution_quality(i, t)
```

decay factor `alpha` 接近 1。Reputation：

- 对 routing decisions 便宜可读（“send hard tasks to high-rep agents”）。
- 难以伪造（随时间累积，绑定 DID）。
- 可以被 slashed：verification 失败的 contributions 会扣减。

### AAMAS 2025 decentralized LaMAS

LaMAS proposal（AAMAS 2025）结合：DID identity、Shapley-value credit attribution 和 simple auction mechanism。关键 claim：decentralizing credit attribution step 让系统 auditable，并免疫 single-point manipulation。

### 经济机制在哪里崩塌

- **Price oracle manipulation.** 如果 credit function 可被 game，agents 会 game 它。每个 mechanism 都需要 adversarial test。
- **Sybil attacks.** 一个 operator 生成 N 个 fake agents 来膨胀自己的 contribution。DIDs 会减缓但不能阻止；reputation cost-to-forge 是 mitigation。
- **Verification cost.** Credit attribution 的公平性只和 verifier 一样好。如果 verification 便宜（small LLM），它能被 game；如果昂贵（human panel），系统无法 scale。
- **Regulatory overhang.** Agent economies 与 financial regulation 相交。截至 2026 年，Bittensor、Fetch 和 Gonka 在一些 jurisdictions 都处于 legal gray areas。

### Agent economies 什么时候有意义

- **Open networks with heterogeneous operators.** 没有单一团队控制全部 agents。
- **Verifiable outputs.** 没有 verification，credit attribution 就是猜测。
- **Long-horizon workflows.** One-shot tasks 不会从 reputation accumulation 中受益。
- **Tokenized payments are legally viable** in your jurisdiction。

在 closed corporate systems 中，economics 会让位给更简单的 allocation（managers 分配工作，metrics 是 internal）。economics literature 主要适用于 open networks。

## 动手实现

`code/main.py` 实现：

- `shapley(value_fn, agents)`：small N 下通过 enumeration 做 exact Shapley computation。
- `second_price_auction(bids)`：truthful mechanism；winner pays second-highest。
- `Reputation`：带 exponential decay 和 slashing 的 DID-bound reputation。
- Demo 1：三个 agents 协作，exact Shapley 归因 credit。
- Demo 2：五个 agents 为 task slot bidding；second-price auction 选择 winner + payment。
- Demo 3：100 轮 task assignment 到具有 heterogeneous rep 的 agents；rep-weighted routing 胜过 random。

运行：

```text
python3 code/main.py
```

预期输出：每个 agent 的 Shapley values；展示 truthful-bid equilibrium 的 auction result；rep-weighted routing 在 warmup 后展示相对 random 的 10-20% quality gain。

## 实际使用

`outputs/skill-economy-designer.md` 设计一个最小 agent economy：identity layer、credit attribution mechanism、payment mechanism、reputation rule 的选择。

## 交付成果

在 2026 年运行 agent economy：

- **Start with reputation, not tokens.** Reputation 实现便宜且单独就有价值；tokens 添加 legal 和 economic complexity。
- **Verify before you reward.** 没有 independent verification step，绝不 distribute credit。Self-reported quality 会累积 sybil games。
- **Shapley-sample, not Shapley-exact.** sample 100-1000 orderings；exact enumeration 无法 scale。
- **Cap decay factor and floor reputation.** unbounded decay 会清空 legitimate contributors；太慢的 decay 会奖励 stale high-rep agents。
- **Audit mechanisms adversarially.** opening network 前运行 red-team scenarios。每个 mechanism 都有 game theory；你想先找到 holes，而不是等 attackers 找到。

## 练习

1. 运行 `code/main.py`。确认 Shapley values sum to total value（efficiency axiom）。改变 value function；Shapley allocations 是否按预期方向变化？
2. 实现 Shapley *sampling*（K 个 orderings 上的 Monte Carlo）。K 如何影响 approximation accuracy？与 N=4 的 exact 结果对比。
3. 在 auction 前实现 coalition-forming step：agents 可以 merge 成 teams 并作为 unit bidding。哪些 coalitions 会形成？outcome 是否比 individual bidding Pareto-better？
4. 阅读 Google Research mechanism-design post。识别一个被违反就会破坏 truthfulness 的 assumption。这个 failure mode 在 LLM setting 中会是什么样？
5. 阅读 AAMAS 2025 decentralized LaMAS paper。在 synthetic task 上为 10 agents 实现它们的 Shapley step。exact computation 要多久？100 draws 的 sampling 有多接近？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| DePIN | “Decentralized physical infrastructure” | Token-incentivized compute/storage/bandwidth。Bittensor、Akash、Render。 |
| DID | “Decentralized identifier” | portable IDs 的 W3C spec。Agent reputation 绑定到 DID，而不是 platform。 |
| ERC-4337 | “Account abstraction” | 可以 sponsor gas 的 contract accounts，使 agent payments 成为可能。 |
| Shapley value | “Fair credit attribution” | 满足 efficiency、symmetry、linearity、null 的唯一 allocation。 |
| Second-price auction | “Vickrey auction” | Truthful mechanism：winner pays second-highest bid。与 monotone aggregation 兼容。 |
| Reputation capital | “Accumulated quality score” | 来自 confirmed contributions 的 DID-bound score；随时间 decay。 |
| Agentic DAO | “Agents + humans govern” | 以 agent voters 为 first-class、voting power 绑定 reputation 的 DAO。 |
| TAO / FET / GPU credits | “Token denominations” | Bittensor TAO、Fetch.ai FET、各种 DePIN tokens。 |

## 延伸阅读

- [The Agent Economy](https://arxiv.org/abs/2602.14219) — 5-layer agent-economy stack 的 2026 survey
- [Google Research — Mechanism design for large language models](https://research.google/blog/mechanism-design-for-large-language-models/) — 带 monotone aggregation 的 token auctions
- [AAMAS 2025 — decentralized LaMAS](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p2896.pdf) — Shapley-value credit attribution
- [Bittensor TAO documentation](https://docs.bittensor.com/) — subnet structure and reward distribution
- [Fetch.ai / ASI Alliance](https://fetch.ai/) — ASI-1 Mini LLM and FET token
- [W3C Decentralized Identifiers (DIDs) spec](https://www.w3.org/TR/did-core/) — identity foundation
