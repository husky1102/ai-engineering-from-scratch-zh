# i18n 收尾后续工作

Status: planned
Source: human
Created: 2026-06-20

## Issue I-01: 测验题数契约漂移
Priority: medium
Kind: docs
Status: planned

### Feedback
AGENTS.md 要求每个 `quiz.json` 正好 6 题，但当前中英文测验中有 207 套不符合 6 题契约。需要单独处理，不能混在本次 i18n/local deploy 收尾中扩大范围。

### Notes
2026-06-19 盘点结果：338 套测验中，131 套为 6 题，118 套为 5 题，60 套为 7 题，29 套为 8 题；`quiz.zh-CN.json` 与英文题数分布一致，所以 i18n 校验通过但与仓库 quiz 合约不一致。按阶段分布：Phase 0 有 7 套非 6 题，Phase 1 有 22 套，Phase 2 有 18 套，Phase 3 有 13 套，Phase 4 有 28 套，Phase 5 有 29 套，Phase 7 有 1 套，Phase 10 有 11 套，Phase 11 有 16 套，Phase 14 有 38 套，Phase 16 有 2 套，Phase 17 有 1 套，Phase 19 有 21 套。

### User Choices
2026-06-20: User accepted recommended option A. Keep upstream-compatible quiz content during the current closeout and open a separate plan/review path for the 207 non-6-question quizzes. Do not silently add, delete, or rewrite quiz questions inside unrelated i18n work.

### Plan
2026-06-22: Planned in `../plan/P-20260622-01-post-i18n-followups.md` as an inventory-and-remediation-design slice. The plan does not rewrite quiz content.

## Issue I-02: 公开维护文档汉化
Priority: medium
Kind: docs
Status: planned

### Feedback
README 与 ROADMAP 已汉化，但面向贡献者或 GitHub 流程的公开元文档仍为英文，影响中文 fork 的维护与协作体验。

### Notes
2026-06-19 盘点到的根目录英文元文档包括 `CHANGELOG.md`、`CODE_OF_CONDUCT.md`、`CONTRIBUTING.md`、`FORKING.md`、`LESSON_TEMPLATE.md`、`SPONSORS.md`；GitHub 模板包括 `.github/ISSUE_TEMPLATE/bug_report.md`、`.github/ISSUE_TEMPLATE/new_lesson_proposal.md`、`.github/PULL_REQUEST_TEMPLATE.md`。`.claude/skills/check-understanding/SKILL.md` 与 `.claude/skills/find-your-level/SKILL.md` 也是英文，但属于 agent skill 内部说明，不在首轮公开维护文档范围内。

### User Choices
2026-06-20: User accepted recommended option A. Translate public contributor-facing docs and GitHub templates first. Do not translate all internal agent skill metadata in the same follow-up unless a later plan explicitly expands scope.

### Plan
2026-06-22: Planned in `../plan/P-20260622-01-post-i18n-followups.md` with scope limited to root public maintainer docs and GitHub templates.

## Issue I-03: fork 同步检查清单
Priority: low
Kind: docs
Status: planned

### Feedback
当前仓库是中文 fork，并保留 `upstream` 原仓库 remote；后续从 upstream 同步可能重新引入英文源变更和 i18n stale，需要明确维护流程。

### Notes
2026-06-19 事实：`origin` 指向 `husky1102/ai-engineering-from-scratch-zh`，`upstream` 指向 `rohitg00/ai-engineering-from-scratch.git`；本地 `main` 相对 `origin/main` ahead 9 commits，未落后。需要把手动 sync 后的 i18n inventory、validate、README/ROADMAP/title drift 检查写成可执行 checklist，而不是现在就自动化跨 remote 同步。

### User Choices
2026-06-20: User accepted recommended option A. Create a manual upstream sync checklist for this fork. Do not freeze upstream sync, and do not build a sync automation script until a later plan proves the manual checklist is insufficient.

### Plan
2026-06-22: Planned in `../plan/P-20260622-01-post-i18n-followups.md` as a manual checklist in contributor-facing docs, not automation.

## Issue I-04: .cowork 版本管理策略
Priority: low
Kind: docs
Status: planned

### Feedback
`.cowork/` 当前同时包含 durable plan/review 文件和本地执行状态文件，需要明确哪些应进入 git，哪些应保持本地队列，避免误提交运行时状态或丢失可复用决策记录。

### Notes
2026-06-19 事实：已跟踪 `.cowork/plan/P-20260618-01-i18n-local-deploy-finish.md` 与 `.cowork/review/R-20260618-01-i18n-local-deploy-finish.md`；`.cowork/current.md` 为 untracked 本地状态。`.gitignore` 目前有 `cowork/`，但没有 `.cowork/`，因此不会自动忽略 `.cowork/current.md`。

### User Choices
2026-06-20: User accepted recommended option A. Track durable plan/review records, but ignore local runtime state such as `.cowork/current.md` and future history/session scratch files. Implementing the ignore/exclude details should be a separate follow-up, not an incidental change inside unrelated work.

### Plan
2026-06-22: Planned in `../plan/P-20260622-01-post-i18n-followups.md` to add targeted repo ignore rules and documentation without ignoring durable `.cowork/plan` or `.cowork/review` records.
