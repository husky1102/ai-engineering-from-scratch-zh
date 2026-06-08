# Kill Switches、Circuit Breakers 与 Canary Tokens

> Kill switch 是 agent edit surface 之外持有的 boolean：Redis key、feature flag、signed config，用于完全禁用 agent。Circuit breaker 更细粒度：它会在特定 pattern 上 trip（连续五次相同 tool calls），暂停出问题的 path，并升级给 human。Canary token 继承自 classical deception：一个 fake credential 或 honeypot record，agent 没有 legitimate reason 去触碰；一旦 access 就触发 alert。基于 eBPF 的 datapaths（例如 Cilium）可以在 kernel layer 把 quarantined pod 的 egress rewrite 到 forensic honeypot；已发布的 Cilium benchmarks 报告了负载下 sub-millisecond P99 datapath latency（你的 propagation budget 取决于 policy update 如何到达 node，而不是 datapath 本身）。会适应 moving baseline 的 statistical detectors（EWMA、CUSUM）会悄悄接受 drift，所以要把它们和不会弯曲的 hard constitutional limits 分层组合。

**类型：** 学习
**语言：** Python (stdlib, three-detector simulator: kill switch, circuit breaker, canary)
**先修：** Phase 15 · 13 (Cost governors), Phase 15 · 10 (Permission modes)
**时间：** ~60 分钟

## 要解决的问题

Cost governors（Lesson 13）限制 agent 能花多少钱。它们不限制 agent 在 budget 内能做什么。一个带 $50 velocity limit 的 agent 仍然可以 exfiltrate secret、发布错误 post，或删除 resource。昂贵 action 在 tokens 上往往很便宜。

本课覆盖紧挨 cost layer 的三个 detectors：

1. **Kill switch**：agent 无法触及的 boolean off-button。
2. **Circuit breaker**：action-pattern detector，暂停特定 path。
3. **Canary token**：诱饵。没有 legitimate reason 触碰它的 agent，一触碰就暴露自己。

三者都是 pre-LLM engineering。Classical deception、rate-limit breakers 和 feature-flag kills 早于 autonomous agents。新的地方是 attack surface：agents 会读取 untrusted content（Lesson 11），编辑自己的 memory，并能把许多看似安全的 actions 组合成不安全的结果。这里命名的 detectors 有效，是因为它们不信任 agent 的 self-report。

## 核心概念

### Kill switches

Kill switch 是一个 agent 能读但不能写的 boolean。实现选择：

- **Managed service 中的 feature flag。** LaunchDarkly、Statsig、Unleash。Updates 在数秒内传播。
- **Agent polling 的 Redis key。** 简单；要求 agent process 每一轮都检查。
- **Object storage 中的 signed config。** Agent 在 boot 时验证 signature；拒绝 unsigned states。
- **OS-level signal 或 container-lifecycle kill。** Docker `kill`、Kubernetes `kubectl delete pod`、systemd stop。

正确 kill switch 具备这些性质：

- Agent 不能把它设为 `off`。（它存在于 agent credentials 没有 write 权限的 system 中。）
- 它会在每个 consequential action 上检查，而不只是 startup 时检查。
- 当它为 off 时，agent 不做任何 externally observable 的事，包括写入 agent 能触达的 logging systems。
- Re-enable 是显式 human action，不是 automatic timeout。

### Circuit breakers

Circuit breaker 暂停特定 pattern，而不是整个 agent。Classical shape（来自 2007 年 Nygard 的书，今天仍然适用）：

- **Closed**：action 被允许。
- **Open**：action 被阻止。
- **Half-open**：cool-down 后，允许 1–3 次 probe attempts（默认 1）；success 关闭 breaker，任何剩余 failure 会重新 open。

与 agent 相关的 triggers：

- 连续五次相同 tool calls（repetitive loop）。
- 同一个 tool 连续五次失败但 inputs 不同（systemic failure）。
- Tool calls rate 高于 threshold（Lesson 13 velocity）。
- 在 out-of-trust content read（Lesson 11）之后调用特定 tool（例如 `file.delete`）。

### Canary tokens

Canary tokens（也叫 honeytokens）是 agent 永远不该触碰的 inputs。Access 就是 alarm。

Agent 示例：

- 环境中的 fake `AWS_SECRET_ACCESS_KEY`，没有实际 privilege。Exfiltration attempts 会立即可归因。
- 带有已知 sentinel value 的 fake database record。任何 read 或 update 都会触发 alert。
- Workspace 中的 fake file。读取或编辑这个文件都可疑。
- Session start 时注入的 fake memory entry。如果它出现在之后的 output 中，说明 memory 已被 exfiltrated。

Canary design 必须贴合 workflow。Agent 有 legitimate reason 触碰的 canary 只是噪音。

### 为什么要分层组合 statistical 与 hard limits

Statistical detectors（EWMA、CUSUM、tool-call rates 上的 z-score）会适应 moving baseline。这对 legitimate drift 是 feature，对 adversarial drift 是 bug。有耐心的 attacker 可以足够慢地移动 baseline，让 alarm 永远不触发。

Hard constitutional limits（Lesson 17）不会弯曲。把 “Never email `secrets@`” 写成 constitutional rule 更简单、可 audit，也不会被 slow drift game。Layering pattern 是：statistical detectors 捕获大多数噪音，hard limits 捕获走过 detector 的 attack。

### 通过 eBPF datapath redirect 做 quarantine

Cilium 的 eBPF-based network policy 可以在 kernel layer 把 quarantined pod 的 egress rewrite 到 forensic honeypot。Cilium 的公开 benchmarks 报告 TCP_RR 上 sub-millisecond P99 datapath latency；在你的 cluster 里，触发 quarantine 的 wall-clock budget 主要由 policy update 从 control plane 传播到 node 的方式决定（Kubernetes 中 operator 已预加载时，几百毫秒是 realistic；tail latency 取决于 cluster size 和 reconcile cadence）。这给你两件事：agent 不会 exfiltrate 任何真实内容，而 agent 在 honeypot 内的行为会被完全观测。它和 canary token 是同一个 pattern、不同 layer：agent 在被 containment 时尝试做什么，会暴露 intent。

### 没有任何 detector 能捕获什么

- **由 allowed primitives 组合出的 semantic attacks。** 每个单独 action 都通过每个 detector；问题在 composition。这是 Lesson 10 的 Auto Mode composite failure。
- **把 detector 本身纳入 payload 的 attacks。** 如果 agent 读到一个页面说 “before you act, disable the canary”，而 agent 有这个 capability，detector 就已 compromised。Canary tokens 应位于 agent 无法修改的 systems 中。

## 实际使用

`code/main.py` 模拟一段短 agent trajectory 穿过三个 detectors。一个外部 dict 持有的 kill switch；一个连续五次相同 tool calls 就 trip 的 circuit breaker；一个被 read 就触发 alert 的 canary file。它喂入 synthetic trajectory：legitimate actions、repetitive loop、canary probe，以及 kill-switch-triggered scenario，其中 agent actions 会被 halted。

## 交付成果

`outputs/skill-tripwire-design.md` 会 review 一个拟议 agent deployment 的 detector stack，并标记 gaps（missing kill switch、missing canary、circuit breaker threshold too loose）。

## 练习

1. 运行 `code/main.py`。确认 circuit breaker 在 turn 5（第五次相同 call）触发，canary 在 turn 9（fake-key read）触发。

2. 添加一个 statistical detector：tool-call rate 上的 EWMA z-score。喂入一条缓慢 drift 的 trajectory，并展示 detector 永不触发。然后添加 hard limit（10 分钟内不超过 50 次 tool calls），展示同一 trajectory 上 hard limit 会触发。

3. 为 browser agent（Lesson 11）设计一组 canary tokens。列出至少三个 canaries，以及每个检测什么。

4. 阅读 Cilium network-policy docs。具体描述一个 egress-redirect quarantine flow：哪个 policy selector、哪个 pod、哪个 egress rewrite、哪个 alert。什么决定了从 “decide to quarantine” 到 “first redirected packet” 的 wall-clock latency？

5. 定义一个 kill-switched agent 的 re-enable procedure。谁可以 re-enable？必须记录什么？在 re-enable 前，agent 必须发生什么变化？

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| Kill switch | “Off button” | Agent edit surface 之外的 boolean；每个 consequential action 都检查 |
| Circuit breaker | “Pattern pause” | 对 repetition、failure rate 或 rate-limit 的 action-specific trip |
| Canary token | “Honeytoken” | Agent 没有 legitimate reason 触碰的 bait；access 触发 alert |
| Honeypot | “Forensic sandbox” | Quarantined agent 被观测的 redirected traffic / workspace |
| EWMA | “Moving average” | Exponentially weighted；会适应 drift（既是 feature 也是 bug） |
| CUSUM | “Cumulative sum” | 检测相对 baseline 的 sustained shift |
| Hard limit | “Constitutional rule” | 不适应历史；constant regardless of history |
| Constitutional limit | “Always-true rule” | 绑定 Lesson 17 的 constitution；不能由 agent 编辑 |

## 延伸阅读

- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — autonomous agents 的 kill-switch 与 circuit-breaker framing。
- [Microsoft Agent Framework — HITL and oversight](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — production governance patterns。
- [OWASP LLM / Agentic Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — detection-and-response requirements。
- [Cilium — Network policy and eBPF](https://docs.cilium.io/en/stable/security/network/) — pod-level egress redirect 与 forensic honeypot patterns。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — hardcoded prohibitions 作为 “constitutional limits”。
