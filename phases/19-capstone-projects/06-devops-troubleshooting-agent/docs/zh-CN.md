# Capstone 06 — 面向 Kubernetes 的 DevOps 故障排查 Agent

> AWS 的 DevOps Agent 已 GA，Resolve AI 发布了 K8s playbooks，NeuBird 演示了 semantic monitoring，Metoro 把 AI SRE 绑定到 per-service SLO。生产形态已经稳定：alert webhook 触发，agent 读取 telemetry，遍历 K8s objects 的图，排序 root-cause hypotheses，并发布带 approval buttons 的 Slack brief。默认 read-only。每个 remediation 都由人类 gate。本 capstone 就是这个 agent，用 20 个 synthetic incidents 评估，并在三个 shared cases 上与 AWS 的 Agent 对比。

**类型：** Capstone
**语言：** Python (agent), TypeScript (Slack integration)
**先修：** Phase 11 (LLM engineering), Phase 13 (tools and MCP), Phase 14 (agents), Phase 15 (autonomous), Phase 17 (infrastructure), Phase 18 (safety)
**练习阶段：** P11 · P13 · P14 · P15 · P17 · P18
**时间：** 30 hours

## 要解决的问题

2025-2026 年的 SRE 叙事变成了：“AI agents triage incidents, humans approve remediations.” AWS DevOps Agent、Resolve AI、NeuBird、Metoro、PagerDuty AIOps 都在生产中交付了这种形态。agent 读取 Prometheus metrics、Loki logs、Tempo traces、kube-state-metrics，以及 K8s objects 的 knowledge graph。它会在五分钟内产出带 telemetry citations 的 ranked root-cause hypothesis。没有通过 Slack 的明确人类审批，它永远不会执行 destructive commands。

大部分难点在 scoping 和 safety，而不是 reasoning。agent 需要一个默认 read-only 的 RBAC surface、一个 hardened MCP tool server，以及每条 considered vs executed command 的 audit logs。它需要知道自己何时超出能力范围并升级处理。并且它必须足够便宜，不能让 OOM-kill cascades 产生 $5k 的 agent 账单。

## 核心概念

agent 在 knowledge graph 上运行。节点是 K8s objects（Pods, Deployments, Services, Nodes, HPAs, PVCs）加上 telemetry sources（Prometheus series, Loki streams, Tempo traces）。边编码 ownership（Pod -> ReplicaSet -> Deployment）、scheduling（Pod -> Node）和 observation（Pod -> Prometheus series）。图由 kube-state-metrics sync 保持新鲜，并在每次 alert 时重新采样。

alert 触发时，agent 从 affected object 开始做 root-cause。它沿边遍历，拉取相关 telemetry slices（last 15 minutes），并起草 hypothesis。hypothesis 按 evidence 排序：有多少 telemetry citations 支持、证据多新、具体程度多高。top-3 hypotheses 会进入 Slack，附带 graph-path visualizations 和 remediation actions 的 approval buttons。

remediation 受到 gate。默认允许的 actions 都是 read-only。destructive actions（scaling down, rolling back, deleting Pods）需要 Slack approval；ArgoCD rollback hooks 需要 agent 从不持有的 auth token。audit log 会记录 agent *considered* 的每条 command，而不只是 executed 的命令，这样 review process 能捕捉 near-misses。

## 架构

```text
PagerDuty / Alertmanager webhook
           |
           v
     FastAPI receiver
           |
           v
   LangGraph root-cause agent
           |
           +---- read-only MCP tools ----+
           |                             |
           v                             v
   K8s knowledge graph              telemetry slices
     (Neo4j / kuzu)              Prometheus, Loki, Tempo
   ownership + scheduling          last 15m, scoped
           |
           v
   hypothesis ranking (evidence weight)
           |
           v
   Slack brief + approval buttons
           |
           v (approved)
   ArgoCD rollback hook / PagerDuty escalate
           |
           v
   audit log: considered vs executed, every command
```

## 技术栈

- Observability sources: Prometheus, Loki, Tempo, kube-state-metrics
- Knowledge graph: Neo4j (managed) or kuzu (embedded) of K8s objects + telemetry edges
- Agent: LangGraph with per-tool allow-list, read-only by default
- Tool transport: FastMCP over StreamableHTTP; separate server for destructive tools behind approval gate
- Models: Claude Sonnet 4.7 for root-cause reasoning, Gemini 2.5 Flash for log summarization
- Remediation: ArgoCD rollback webhook, PagerDuty escalate, Slack approval card
- Audit: append-only structured log (considered, executed, approved, outcome)
- Deployment: K8s deployment with its own narrow RBAC role; separate namespace

## 动手实现

1. **图摄取。** 每 30s 将 kube-state-metrics sync 到 Neo4j/kuzu。Nodes: Pod, Deployment, Node, Service, PVC, HPA。Edges: OWNED_BY, SCHEDULED_ON, EXPOSES, MOUNTS, SCALES。Telemetry overlay edges: OBSERVED_BY（一个 Pod 被一个 Prometheus series 观测）。

2. **Alert receiver。** FastAPI endpoint 接受 PagerDuty 或 Alertmanager webhooks。提取 affected object(s) 和 SLO breach。

3. **Read-only tool surface。** 通过 FastMCP 包装 kubectl、Prometheus query、Loki logql、Tempo traceql。每个 tool 都只有很窄的 RBAC verb（"get", "list", "describe"）。默认 server 中没有 "delete"、"exec"、"scale"。

4. **Root-cause agent。** LangGraph 有三个 nodes：`sample` 拉取 last-15-minutes telemetry slice，`walk` 查询 graph 中的 neighboring objects，`hypothesize` 起草带 telemetry citations 的 ranked root-cause candidates。

5. **Evidence scoring。** 每个 hypothesis 都有一个 score = recency * specificity * graph-path length inverse * citation count。返回 top-3。

6. **Slack brief。** 发布一个 attachment，其中包含 hypothesis、graph-path visualization（server-side 渲染的 subgraph image），以及最多一个 remediation action 的 approval buttons。

7. **Remediation gate。** destructive tools（scale down, roll back, delete）存在第二个 MCP server 上，位于 approval token 之后。只有 Slack card 被人类批准后，agent 才能调用它们。

8. **Audit log。** Append-only JSONL：对每个 candidate command，记录它是否被 considered、是否被 executed、谁 approved。每天发送到 S3。

9. **Synthetic incident suite。** 构建 20 个 scenarios：OOMKill cascade、DNS flap、HPA thrash、PVC fill、noisy neighbor、faulty sidecar、bad ConfigMap rollout、certificate rotation、image-pull backoff 等。按 root-cause accuracy 和 time-to-hypothesis 给 agent 评分。

## 实际使用

```text
webhook: alert.pagerduty.com -> checkout-api SLO breach, error rate 14%
[graph]   affected: Deployment checkout-api (3 Pods, Node ip-10-2-3-4)
[walk]    neighbors: ReplicaSet checkout-api-abc, Service checkout-api,
           recent rollout 14m ago
[sample]  prometheus error_rate 14%, up-trend; loki 500s on /api/v2/pay
[hypo]    #1 bad rollout: latest image checkout-api:v2.41 fails /healthz
          citations: deploy.yaml (rev 42), prometheus errorRate, loki 500 stack
[slack]   [ROLL BACK to v2.40]  [ESCALATE]  [IGNORE]
          (approval required; agent does not roll back unilaterally)
```

## 交付成果

`outputs/skill-devops-agent.md` 是交付物。给定一个 K8s cluster 和 alert source，agent 会产出 ranked root-cause hypotheses，并提供 Slack-gated remediation flow。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | RCA accuracy on scenario suite | Across 20 synthetic incidents，≥80% correct root cause |
| 20 | 安全 | audit log 中 destructive-action guard 从不在没有 Slack approval 时触发 |
| 20 | Time-to-hypothesis | 从 alert 到 Slack brief 的 p50 低于 5 minutes |
| 20 | 可解释性 | 每个 hypothesis 都有 graph paths 和 telemetry citations |
| 15 | Integration completeness | PagerDuty、Slack、ArgoCD、Prometheus 端到端工作 |
| **100** | | |

## 练习

1. 在 AWS 的 DevOps Agent demo 过的同三个 incidents 上运行你的 agent。发布 side-by-side。报告 agent 发生分歧的位置。

2. 添加一个 "near-miss" audit，用来标记 agent *considered* 的任何如果没有 approval 就会 destructive 的 command。测量一周内的 near-miss rate。

3. 将 hypothesis model 从 Claude Sonnet 4.7 换成 self-hosted Llama 3.3 70B。测量 RCA accuracy delta 和 dollar per incident。

4. 构建 causal filter：区分 correlated telemetry spikes 和真正的 root cause。用 20-scenario labels 训练一个小 classifier。

5. 添加 rollback dry-run：针对有相同 manifest 的 staging cluster 执行 ArgoCD rollback。先在 live cluster 中验证 rollback plan，再显示 Slack approval button。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| K8s knowledge graph | "Cluster graph" | Nodes = K8s objects + telemetry series；edges = ownership, scheduling, observation |
| Read-only-by-default | "Scoped RBAC" | Agent 的 service account 只有 get/list/describe verbs；destructive verbs 位于 approval 后的独立 server |
| Audit log | "Considered vs executed" | 每个 candidate command 的 append-only record，记录是否运行、谁 approved |
| Hypothesis ranking | "Evidence score" | Recency × specificity × graph-path length inverse × citation count |
| Slack approval card | "HITL gate" | 带 remediation buttons 的 interactive Slack message；agent 只有在人类点击后才能继续 |
| Telemetry citation | "Evidence pointer" | 支撑某个 claim 的 Prometheus query、Loki selector 或 Tempo trace URL |
| MTTR | "Time to resolution" | 从 alert fire 到 SLO recovery 的 wall-clock |

## 延伸阅读

- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — canonical 2026 reference
- [Resolve AI K8s troubleshooting](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai) — competitor reference
- [NeuBird semantic monitoring](https://www.neubird.ai) — semantic-graph approach
- [Metoro AI SRE](https://metoro.io) — SLO-first production framing
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) — cluster-state source
- [LangGraph](https://langchain-ai.github.io/langgraph/) — reference agent orchestrator
- [FastMCP](https://github.com/jlowin/fastmcp) — Python MCP server framework
- [ArgoCD rollback](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/) — gated remediation target
