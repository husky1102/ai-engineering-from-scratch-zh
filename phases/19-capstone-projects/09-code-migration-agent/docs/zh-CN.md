# Capstone 09 — 代码迁移 Agent（Repo-Level Language / Runtime Upgrade）

> Amazon 的 MigrationBench（Java 8 到 17）和 Google 的 App Engine Py2-to-Py3 migrator 设定了 2026 年的标杆。Moderne 的 OpenRewrite 以规模化方式做 deterministic AST rewrites。Grit 用 codemod-style DSL 瞄准同一个问题。生产模式将两者结合：一个用于安全 rewrites 的 deterministic substrate，加上一层处理 ambiguous cases 的 agent，一个用于 per-branch builds 的 sandbox，以及一个在 PR 打开前变绿的 test harness。这个 capstone 要迁移 50 个真实 repos，并发布 pass rate 和 failure taxonomy。

**类型：** Capstone
**语言：** Python (agent), Java / Python (targets), TypeScript (dashboard)
**先修：** Phase 5 (NLP), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 15 (autonomous), Phase 17 (infrastructure)
**练习阶段：** P5 · P7 · P11 · P13 · P14 · P15 · P17
**时间：** 30 hours

## 要解决的问题

大规模代码迁移是 2026 年 coding agents 最干净的生产应用之一。ground truth 很明显（迁移后 test suite 是否通过？），收益真实（Java-8 fleet migration 是 headcount-scale project），benchmark 公开（MigrationBench 50-repo subset）。Moderne 的 OpenRewrite 处理 deterministic side。agent layer 处理 OpenRewrite recipes 无法覆盖的一切：ambiguous rewrites、build-system drift、long-tail syntax、transitive dependency breakage。

你将构建一个 agent，它接收 Java 8 repo（或 Python 2 repo），产出一个 green-CI migrated branch。你会测量 pass rate、test-coverage preservation、cost per repo，并构建 failure taxonomy。与 deterministic-only baseline 的 side-by-side 会告诉你 agent 的价值究竟在哪里。

## 核心概念

pipeline 有两层。**deterministic substrate**（Java 用 OpenRewrite，Python 用 libcst）安全地运行大部分 mechanical rewrites：imports、method signatures、null-safety edits、try-with-resources、deprecated API replacements。它很快，并产生可审计 diffs。**agent layer**（OpenAI Agents SDK 或 LangGraph，使用 Claude Opus 4.7 和 GPT-5.4-Codex）处理 recipes 无法处理的情况：build-file upgrades（Maven/Gradle/pyproject）、transitive dependency conflicts、test flakes、custom annotations。

每个 repo 都有一个安装好 target runtime 的 Daytona sandbox。agent 迭代：run build、classify failures、apply fix、rerun。硬限制：每个 repo 30 minutes、$8、20 agent turns。如果所有 tests 通过且 coverage delta 不为负，就打开 PR。否则，该 repo 会带 evidence 被归入一个 failure class。

failure taxonomy 是交付物。跨 50 个 repos，什么坏了？Transitive deps？Custom annotations？Build tool version？与迁移无关的 test flakes？每个 class 都有 count 和 exemplar diff。未来的 recipe authors 可以瞄准前三个。

## 架构

```text
target repo
      |
      v
OpenRewrite / libcst deterministic recipes
   (safe, fast, auditable, ~70-80% of fixes)
      |
      v
Daytona sandbox per branch
      |
      v
agent loop (Claude Opus 4.7 / GPT-5.4-Codex):
   - run build -> capture failures
   - classify failures (build, test, lint)
   - apply fix (patch or retry recipe)
   - rerun
   - budget: 30 min, $8, 20 turns
      |
      v
test + coverage delta gate
      |
      v (passed)
open PR
      |
      v (failed)
file under failure class + attach repro
```

## 技术栈

- Deterministic substrate: OpenRewrite (Java) or libcst (Python)
- Agent: OpenAI Agents SDK or LangGraph over Claude Opus 4.7 + GPT-5.4-Codex
- Sandbox: Daytona devcontainers per branch, pre-installed target runtime (Java 17 / Python 3.12)
- Build systems: Maven, Gradle, uv (Python)
- Benchmarks: Amazon MigrationBench 50-repo subset (Java 8 to 17), Google App Engine Py2-to-Py3 repos
- Test harness: parallel runner, coverage via Jacoco (Java) or coverage.py (Python)
- Observability: Langfuse + trace bundle per repo with every diff chunk
- Dashboard: failure-taxonomy dashboard with per-class counts and exemplar diffs

## 动手实现

1. **Recipe pass。** 先运行 OpenRewrite（Java）或 libcst（Python）recipes。捕捉 70-80% 的 mechanical migrations。提交为 "recipe" commit。

2. **Build trial。** Daytona sandbox：安装 target runtime，运行 build。如果 green，跳到 tests。如果 red，交给 agent。

3. **Agent loop。** LangGraph 配置 tools：`run_build`、`read_file`、`edit_file`、`run_test`、`git_diff`。Agent 对 failure 分类（dep, syntax, test, build-tool），并应用 targeted fix。重新运行。

4. **Budget caps。** 每个 repo 30 minutes wall-clock、$8 cost、20 agent turns。任何超限都会停止，并以当前 diff 归入 "budget_exhausted"。

5. **Test + coverage gate。** build 变 green 后，运行 test suite。将 coverage 与 base repo 对比。如果 coverage 下降超过 2%，归入 "coverage_regression"。

6. **PR open。** 成功后，push branch，打开 PR，附带 diff、已应用 recipes 摘要，以及 agent authored 的 commits。

7. **Failure taxonomy。** 对每个 failed repo，打上 class：`dep_upgrade_required`、`build_tool_drift`、`custom_annotation`、`test_flake`、`syntax_edge_case`、`budget_exhausted`。构建 dashboard。

8. **50-repo run。** 在 MigrationBench subset 上执行。报告 per-class pass rate、cost-per-repo、coverage-preservation，并与 deterministic-only baseline 比较。

## 实际使用

```text
$ migrate legacy-java-service --target java17
[recipe]   27 rewrites applied (JUnit 4->5, HashMap initializer, try-with-resources)
[build]    FAIL: cannot find symbol sun.misc.BASE64Encoder
[agent]    turn 1 classify: removed_jdk_api
[agent]    turn 2 apply: sun.misc.BASE64Encoder -> java.util.Base64
[build]    OK
[tests]    412/412 passing; coverage 84.1% -> 84.3%
[pr]       opened #1841  cost=$3.20  turns=4
```

## 交付成果

`outputs/skill-migration-agent.md` 是交付物。给定一个 repo，它先执行 deterministic recipes，然后运行 agent loop 产出 green migrated branch，或把 repo 归入 taxonomy class。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | MigrationBench pass rate | 50-repo subset pass@1 |
| 20 | Test-coverage preservation | Mean coverage delta vs base |
| 20 | Cost per migrated repo | passing runs 上的 $/repo |
| 20 | Agent / deterministic-tool integration | OpenRewrite 处理的 fixes 与 agent authored fixes 的比例 |
| 15 | Failure analysis write-up | 带 exemplars 的 taxonomy completeness |
| **100** | | |

## 练习

1. 只用 OpenRewrite（无 agent）运行 migrate pipeline。将 pass rate 与完整 pipeline 对比。识别只有 agent 才能带来差异的 cases。

2. 实现 "lint-clean" check：迁移后运行 style linter（Java 用 spotless，Python 用 ruff）。如果出现新的 lint errors，就让 PR 失败。测量 coverage-preserved-but-style-regressed rate。

3. 添加 "minimal-diff" optimizer：agent 的 branch 通过 tests 后，用第二次 pass 修剪不必要 changes。报告 diff-size reduction。

4. 扩展到第三种迁移：Node 18 到 Node 22。复用 sandbox wrapping；把 recipe layer 换成 custom codemod。

5. 将 time-to-first-green-build（TTFGB）作为 UX metric 来测量。目标：p50 低于 10 minutes。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| Deterministic substrate | "Recipe engine" | OpenRewrite / libcst：带 safety guarantees 的 declarative AST rewrites |
| Codemod | "Code-modifying program" | 机械修改 source code 的 rewrite rule |
| Build drift | "Tool version skew" | major versions 之间微妙的 Maven / Gradle / uv 行为变化 |
| Failure class | "Taxonomy bucket" | repo 未迁移成功的 labeled reason：dep、syntax、test、build-tool、budget |
| Coverage delta | "Coverage preservation" | 从 base 到 migrated branch 的 test coverage % 变化 |
| Agent turn | "Tool-call round" | agent loop 中的一个 plan -> act -> observe cycle |
| Budget exhaustion | "Hit the ceiling" | repo 消耗完 30-min / $8 / 20-turn limit 仍未通过 |

## 延伸阅读

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — canonical 2026 benchmark
- [Moderne.io OpenRewrite platform](https://www.moderne.io) — deterministic substrate reference
- [OpenRewrite documentation](https://docs.openrewrite.org) — recipe authoring
- [Grit.io](https://www.grit.io) — alternate codemod DSL
- [OpenAI sandboxed migration cookbook](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent) — Agents SDK reference
- [Google App Engine Py2 to Py3 migrator](https://cloud.google.com/appengine) — alternate migration benchmark
- [libcst](https://github.com/Instagram/LibCST) — Python deterministic substrate
- [Daytona sandboxes](https://daytona.io) — reference per-branch sandbox
