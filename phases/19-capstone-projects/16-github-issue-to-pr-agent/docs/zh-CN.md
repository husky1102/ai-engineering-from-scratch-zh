# 综合项目 16 — GitHub Issue-to-PR Autonomous Agent

> AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud 和 Google Jules 都发布了同一种 2026 产品形态：给 issue 打标签，然后得到 PR。在云 sandbox 中运行 agent，验证 tests pass，并发布带 rationale 的 review-ready PR。难点是自动复现 repo 的 build environment、防止 credential leakage、强制 per-repo budgets，并确保 agent 不能 force-push。这个综合项目构建 self-hosted 版本，并在 cost 和 pass rate 上与 hosted alternatives 对比。

**类型:** Capstone
**语言:** Python（agent），TypeScript（GitHub App），YAML（Actions）
**先修:** Phase 11（LLM engineering），Phase 13（tools），Phase 14（agents），Phase 15（autonomous），Phase 17（infrastructure）
**覆盖阶段:** P11 · P13 · P14 · P15 · P17
**时间:** 30 小时

## 要解决的问题

异步云 coding agent 是一个独立于交互式 coding agents（综合项目 01）的产品类别。UX 是一个 GitHub label。你给 issue 标记 `@agent fix this`，worker 在云 sandbox 中启动，clone repo、运行 tests、编辑文件、验证，并在 PR body 中带上 agent rationale 打开 PR。没有交互循环，没有 terminal。AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud、Google Jules 和 Factory Droids 都趋同到这个形态。

工程挑战很具体：环境复现（agent 必须从零构建 repo，而不能依赖 cached dev image）、flaky tests（必须重跑或隔离）、credential scoping（使用具备最小 fine-grained permissions 的 GitHub App）、每个 repo 每日 budget enforcement，以及 no-force-push policy。这个综合项目会测量 pass rate、cost、安全性，并与 hosted alternatives 对比。

## 核心概念

触发器是 GitHub webhook（issue label 或 PR comment）。Dispatcher 把 work 入队到 ECS Fargate 或 Lambda。Worker 把 repo 拉进 Daytona 或 E2B sandbox，并使用从 repo 推断出的 generic Dockerfile（language、framework）。Agent 运行 mini-swe-agent 或 SWE-agent v2 loop，对接 Claude Opus 4.7 或 GPT-5.4-Codex。它不断迭代：read code、propose fix、apply patch、run tests。

Verification 是 gating step。打开 PR 之前，必须在 sandbox 中通过完整 CI。计算 coverage delta；如果负向超过阈值，PR 仍会打开，但会标记 `needs-review`。Agent 把 rationale 发布为 PR description，并附上一个 reviewer 可通过 `@agent` ping 继续跟进的 thread。

Safety 通过两个不同的 GitHub surface 做 scoping：App 提供短期 installation token，具备 `workflows: read` 和窄 repo contents/PR scopes；branch protection（而不是 app permissions）强制 “no direct writes to `main`” 和 “no force-push”；app 永远不会加入 bypass list。对 `.github/workflows` 的 path-scoped read-only access 并不是 GitHub App 的真实 primitive，所以 agent 的 file edits allow-list 必须在 worker 中强制这一点。每个 repo 每日 budget ceilings 在 dispatcher 处强制（例如每个 repo 每天最多 5 个 PR、每个 PR $20）。

## 架构

```text
GitHub issue labeled `@agent fix` or PR comment
            |
            v
    GitHub App webhook -> AWS Lambda dispatcher
            |
            v
    ECS Fargate task (or GitHub Actions self-hosted runner)
       - pull repo
       - infer Dockerfile (language, package manager)
       - Daytona / E2B sandbox with target runtime
       - clone -> git worktree -> agent branch
            |
            v
    mini-swe-agent / SWE-agent v2 loop
       Claude Opus 4.7 or GPT-5.4-Codex
       tools: ripgrep, tree-sitter, read/edit, run_tests, git
            |
            v
    verify CI passes in-sandbox + coverage delta check
            |
            v (verified)
    git push + open PR via GitHub App
       PR body = rationale + diff summary + trace URL
       label: needs-review
            |
            v
    operator reviews; can @-mention agent for follow-ups
```

## 技术栈

- Trigger：具备 fine-grained token 的 GitHub App；webhook receiver 通过 Lambda 或 Fly.io
- Worker：ECS Fargate task（或 GitHub Actions self-hosted runner）
- Sandbox：每个 task 一个 Daytona devcontainer 或 E2B sandbox
- Agent loop：mini-swe-agent baseline 或 SWE-agent v2，使用 Claude Opus 4.7 / GPT-5.4-Codex
- Retrieval：tree-sitter repo-map + ripgrep
- Verification：full CI in-sandbox + coverage delta gate
- Observability：Langfuse，per-PR trace archive 链接在 PR body 中
- Budget：per-repo daily dollar ceiling；每个 repo 每天最多 PR 数

## 动手实现

1. **GitHub App。** Fine-grained installation token：issues read+write、pull_requests write、contents read+write、workflows read。Branch protection（唯一能做到这一点的 surface）强制 “no direct push to `main`” 和 “no force-push”；app 不在 bypass list。Worker 通过 proposed diff 上的 allow-list check 强制 “no writes under `.github/workflows`”，因为 GitHub App permissions 不是 path-scoped。

2. **Webhook receiver。** Lambda function 接收 issue label / PR comment webhooks。按 label `@agent fix this` 过滤。入队到 SQS。

3. **Dispatcher。** 从 SQS 弹出 tasks。强制 per-repo per-day budget。用 repo URL、issue body 和 fresh Daytona sandbox 启动 ECS Fargate task。

4. **Environment inference。** 检测语言（Python、Node、Go、Rust）和 package manager（uv、pnpm、go mod、cargo）。如果没有 Dockerfile，则即时生成一个。

5. **Agent loop。** mini-swe-agent 或 SWE-agent v2，配 Claude Opus 4.7。Tools：ripgrep、tree-sitter repo-map、read_file、edit_file、run_tests、git。硬限制：$20 cost、30 min wall-clock、30 agent turns。

6. **Verification。** Loop 结束后，在 sandbox 中运行完整 test suite。通过 jacoco / coverage.py 计算 coverage delta。如果 CI red：停止，不打开 PR。如果 coverage 下降超过 2%：打开 PR 并加 `needs-review` label。

7. **PR posting。** Push agent branch。通过 GitHub API 打开 PR，包含：title、rationale、diff summary、trace URL、cost、turns。

8. **Credential hygiene。** Worker 使用短期 GitHub App installation token。归档前 scrub logs 中的 secrets。

9. **Eval。** 30 个不同难度的 seeded internal issues。测量 pass rate、PR quality（diff size、style、coverage）、cost、latency。与 Cursor Background Agents 和 AWS Remote SWE Agents 在相同 issues 上对比。

## 实际使用

```text
# on github.com
  - user labels issue #842 with `@agent fix this`
  - PR #1903 appears 14 minutes later
  - body:
    > Fixed NPE in widget.dedupe() caused by null comparator entry.
    > Added regression test widget_test.go::TestDedupeNullComparator.
    > Coverage delta: +0.12%
    > Turns: 7  Cost: $1.80  Trace: langfuse:...
    > Label: needs-review
```

## 交付成果

`outputs/skill-issue-to-pr.md` 是交付物。一个 GitHub App + async cloud worker，把已标记的 issues 转成 review-ready PR，并带 bounded cost 和 scoped credentials。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 30 个 issues 上的 pass rate | 端到端成功（CI green + coverage OK） |
| 20 | PR quality | Diff size、coverage delta、style conformance |
| 20 | 每个 resolved issue 的 cost 和 latency | 每个 PR 的 $ 和 wall-clock |
| 20 | Safety | Scoped token、per-repo budget、no force-push、credential hygiene |
| 15 | Operator UX | Rationale comments、retry affordance、@-mention follow-up |
| **100** | | |

## 练习

1. 添加 “fix flaky test” 模式：label `@agent stabilize-flake TestX` 会在 sandbox 中运行测试 50 次，并提出一个稳定它的最小改动。

2. 在三个 shared issues 上与 Cursor Background Agents 比较 cost。报告哪些工具在哪些场景胜出。

3. 实现 budget dashboard：per-repo per-day cost、per-user cost。对 anomaly 告警。

4. 构建 “dry-run” 模式：打开一个 draft PR 而不运行 CI，让 reviewer 低成本查看 plan。

5. 添加 retention policy：超过 7 天未 merge 的 PR branches 自动删除。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| GitHub App | “Scoped bot identity” | 具备 fine-grained permissions 和短期 installation token 的 App |
| Async cloud agent | “Background agent” | 在云 sandbox 中运行的非交互 worker，而不是 terminal |
| Environment inference | “Dockerfile synthesis” | 检测 language + package manager，缺失时生成 Dockerfile |
| Verification | “CI-in-sandbox” | 打开 PR 前在 worker 内运行完整 test suite |
| Coverage delta | “Coverage preservation” | 从 base 到 agent branch 的 test coverage % 变化 |
| Per-repo budget | “Daily ceiling” | Dispatcher 强制执行的 dollar 和 PR-count cap |
| Rationale | “PR body explanation” | Agent 对改了什么和为什么的总结；PR body 中必需 |

## 延伸阅读

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — canonical async cloud agent reference
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) — CLI reference
- [Cursor Background Agents](https://docs.cursor.com/background-agent) — commercial alternative
- [OpenAI Codex (cloud)](https://openai.com/codex) — hosted competitor
- [Google Jules](https://jules.google) — Google 的 hosted version
- [Factory Droids](https://www.factory.ai) — alternate commercial reference
- [GitHub App documentation](https://docs.github.com/en/apps) — scoped bot identity
- [Daytona cloud sandboxes](https://daytona.io) — reference sandbox
