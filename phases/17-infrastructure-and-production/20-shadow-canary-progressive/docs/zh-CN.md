# LLMs 的 Shadow Traffic、Canary Rollout 与 Progressive Deployment

> LLM rollouts 结合了软件部署中最难的部分：没有 unit tests、failure modes 分散、signals 延迟。顺序是：(1) shadow mode：将 prod requests duplicate 到 candidate model，log、compare，zero user impact；能捕获明显 distribution issues，但不是 quality guarantee；(2) canary rollout：progressive traffic shift 10% → 25% → 50% → 75% → 100%，每一步都有 gates；追踪 latency percentiles、cost/request、error/refusal rate、output length distribution、user-feedback rate；(3) 稳定性确认后，对 distinct alternatives 做 A/B testing。Non-determinism 不可约：因为 GPU FP non-associativity 加 batch-size variance，相同 inputs 的 run-to-run accuracy variation 最高可达 15%。Cost 是 variable，不是 constant：一个好 20% 的 model 每 call 可以贵 3x。Rollback speed 是决定性的：如果 rollback 需要 redeploy，你太慢了。Policy 位于 config/flags；model 位于 registry，带 pinned digests；rollback = 数秒内 flip policy + revert threshold + pin old model。

**类型：** 学习
**语言：** Python (stdlib, toy canary-progression simulator)
**先修：** Phase 17 · 13 (Observability), Phase 17 · 21 (A/B Testing)
**时间：** ~60 分钟

## 学习目标

- 区分 shadow mode（zero-impact compare）、canary（live traffic progressive）和 A/B（stability-confirmed comparison）。
- 枚举五个 LLM-specific canary metrics（latency、cost/request、error/refusal、output-length distribution、user feedback）。
- 解释为什么 LLM non-determinism（最高 15%）会改变 rollout 中 “stable” 的含义。
- 设计一个 rollback path，用 seconds（policy flip）而不是 hours（redeploy）。

## 要解决的问题

你交付了一个新 model。Offline evals 显示 3% accuracy gain。你在 production 中直接打开。24 小时内，cost 上升 40%，user thumbs-down 上升 8%，三个 customer tickets 报告 “weird answers”。你 roll back。redeploy 花 3 小时。你的周末毁了。

这一切都可以避免。Shadow mode 会在任何用户看到之前捕获 40% cost spike。Canary 会在 10% 阶段发现 thumbs-down moved 并停止。Policy-flag rollback 只需要 30 秒。discipline 填补了 “offline evals look good” 和 “real users are happy” 之间的 gap。

## 核心概念

### Shadow mode

Candidate 接收与 production 相同的 requests；outputs 会被 logged，不返回给 users。Zero user impact。记录：

- Output content（与 production diff）。
- Token counts（cost delta）。
- Latency。
- Refusal 和 error。

能捕获：cost blow-ups、length regressions、obvious refusal changes、hard errors。不能捕获：用户会感知到的 quality delta。Shadow 是 smoke test，不是 quality test。

### Canary rollout

带 gates 的 progressive traffic shift。典型 progression：1% → 10% → 25% → 50% → 75% → 100%。每一步 gate 5 个 metrics：

1. **Latency percentiles**：P50、P95、P99。Breach：canary P99 > 1.5x baseline。
2. **Cost per request**：blended $。Breach：>20% above baseline。
3. **Error / refusal rate**：5xx 加 explicit refusals。Breach：2x baseline。
4. **Output length distribution**：mean + P99。Breach：distributional shift。
5. **User-feedback rate**：thumbs-down / ticket filings。Breach：1.5x baseline。

### Non-determinism 是新的 variance

相同 inputs 产出非相同 outputs。原因：

- GPU FP non-associativity（floating-point reduction order 随 batch 变化）。
- Batch-size variance（同一 prompt 在 batch of 128 vs batch of 16 中）。
- Sampling（temperature > 0）。

测量结果：相同 eval sets 上 run-to-run accuracy variation 最高可达 15%。“Stable” 在 rollout 中意味着 metrics 位于 expected variance 内，而不是与 baseline 完全相同。将 gates 设置在 noise floor 之上。

### Cost 是 variable

一个好 20% 的 model 可能每 call 贵 3x。Cost/request 是五个 gates 之一。交付一个 “better” model 但破坏 unit economics，是 rollback case。

### Rollback 是武器

- Policy flag（feature flag system）：在 config 中 flip percentage；耗时 seconds。
- Model pinning（registry digest）：pinned model 不会 auto-upgrade。
- Rollback = revert flag + set pinned digest to previous。Seconds，不是 hours。

如果你的 stack 需要 redeploy 才能 rollback，在 rolling 前先修它。

### Tooling

**Argo Rollouts** / **Flagger**：Kubernetes progressive delivery controllers。与 Istio/Linkerd weighted routing 集成。

**Istio weighted routing**：service-mesh-level traffic split。

**KServe / Seldon Core**：带 built-in canary 的 model serving。

**Feature flags**：LaunchDarkly、Flagsmith、Unleash。Policy-level flip，无需 redeploy。

### Metrics cadence

Canary gates 每 5-15 分钟检查一次，取决于 traffic volume。1% traffic 且 10 req/min 时，每个 window 有 50-150 data points：latency 足够，但 user feedback 很 noisy。10% 会给约 10x 更多。Progressions 应在每一步暂停足够久，以积累足够 samples。

### A/B step 是 optional

如果新 model 明显不同（different behavior、different cost curve、different tone），canary 通过后在 50% 做 A/B test。如果只是 improved version，canary gates 通过后直接到 100%。

### 你应该记住的数字

- Canary progression：1% → 10% → 25% → 50% → 75% → 100%。
- Non-determinism ceiling：相同 inputs 上 run-to-run variance 最高 15%。
- 五个 canary metrics：latency、cost、error/refusal、output length、user feedback。
- Cost gate：>20% above baseline 是 breach。
- Rollback：seconds，不是 hours。

## 实际使用

`code/main.py` 模拟带 injected regressions 的 canary rollout。报告 rollout 在哪个 stage halt，以及哪个 gate triggered。

## 交付成果

本课产出 `outputs/skill-rollout-runbook.md`。给定 candidate model、baseline 和 risk tolerance，设计 shadow→canary→100% plan。

## 练习

1. 运行 `code/main.py`。注入 25% cost regression。canary 会在哪个 stage halt？
2. 新 model offline accuracy gain 为 3%，但 cost/request 是 +18%。是否 ship？取决于 policy：写出两条路径。
3. 设计一个 end-to-end 低于 60 秒的 rollback。列出所需 infrastructure。
4. Non-determinism 在你的 eval 上显示 ±7%。设置 canary gates，避免 false-alarm。你使用什么 multipliers？
5. Shadow mode 在 canary 前捕获 40% cost spike。写出会 fire 的 alert rule。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Shadow mode | “duplicate to new” | zero-impact send-to-candidate for logging |
| Canary | “progressive traffic” | 带 gates 的渐进式 user-exposed rollout |
| Gates | “rollout checks” | 阻止 progression 的 metric thresholds |
| Non-determinism | “LLM variance” | 不可约的 run-to-run differences |
| Policy flag | “flag flip rollback” | config-level rollback，seconds not hours |
| Model pin | “registry digest” | model version 的 immutable reference |
| Argo Rollouts | “K8s progressive” | Kubernetes-native canary/rollback controller |
| KServe | “inference K8s” | 带 canary primitives 的 model serving |
| Istio weighted | “mesh split” | Service-mesh traffic splitter |

## 延伸阅读

- [TianPan — Releasing AI Features Without Breaking Production](https://tianpan.co/blog/2026-04-09-llm-gradual-rollout-shadow-canary-ab-testing)
- [MarkTechPost — Safely Deploying ML Models](https://www.marktechpost.com/2026/03/21/safely-deploying-ml-models-to-production-four-controlled-strategies-a-b-canary-interleaved-shadow-testing/)
- [APXML — Advanced LLM Deployment Patterns](https://apxml.com/courses/mlops-for-large-models-llmops/chapter-4-llm-deployment-serving-optimization/advanced-llm-deployment-patterns)
- [Argo Rollouts docs](https://argo-rollouts.readthedocs.io/)
- [Flagger docs](https://docs.flagger.app/)
