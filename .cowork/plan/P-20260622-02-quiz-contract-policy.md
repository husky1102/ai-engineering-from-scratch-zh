# 测验契约兼容策略计划

Status: done
Review: ../review/R-20260622-02-quiz-contract-remediation.md
Created: 2026-06-22
Approved: 2026-06-22 by user choice: "B,本项目主要工作为汉化，不要新增事端"

## Issue I-01: 保留既有测验并更新契约说明
Priority: medium
Status: done

### Review Problem
AGENTS.md 写着 `quiz.json` 应正好 6 题，但当前中文 fork 从上游继承了 207 套非 6 题测验。用户明确选择 B：本项目主要工作是汉化，不要新增补题、裁题或内容重写事端；保留现有测验内容，把这类漂移作为 advisory 兼容现实处理。

### Scope
更新贡献者和 agent 可见的维护说明，使 canonical 6 题结构适用于新增课程或主动重做的课程；既有上游兼容测验可以保留，`audit_lessons.py` 的 `A006`/`A007` 题数与 stage 分布提示保持 advisory，不作为默认阻断。同步更新 doIt review/plan 记录和必要的 README/CONTRIBUTING 说明。

### Out of Scope
不修改任何 `quiz.json` 或 `quiz.zh-CN.json` 内容；不新增、删除、裁剪或重写测验题；不把 `A006`/`A007` 改成 blocking；不做大规模 schema migration；不改变课程正文或翻译内容。

### Steps
1. 修改 `AGENTS.md` 的 quiz 契约文字，明确 6 题结构是新课和主动维护时的 canonical target，既有上游兼容题数漂移暂不要求本中文 fork 批量修复。
2. 在 `CONTRIBUTING.md` 或 README 的贡献验证说明中补充 audit 输出语义：blocking issue 会使默认检查失败，quiz 题数/stage 分布 drift 当前作为 advisory 暴露。
3. 检查 `scripts/audit_lessons.py` 保持现状：`A006`/`A007` 继续是 warnings，默认 audit 仍 0 exit；不调整脚本逻辑，除非文档语义需要注释澄清。
4. 更新 review/plan 的 Findings/Progress/Verification 日志，确认没有 quiz 内容文件被改动。
5. 运行验证并单独提交本策略更新。

### Verification
- `git diff --name-only` confirms no `quiz.json` or `quiz.zh-CN.json` files changed.
- `python3 scripts/audit_lessons.py`
- `python3 scripts/check_readme_counts.py`
- `python3 /Users/lolita/.codex/skills/doit-queue-auditor/scripts/queue_auditor.py --root .`

### Risks
- 如果文档写得过松，未来新增课程可能继续扩大 schema drift；必须保留 canonical target。
- 如果文档写得过硬，会重新暗示需要批量改 207 套测验，违背用户选择。

### Findings Log
- 2026-06-22: `scripts/audit_lessons.py` already keeps quiz count/stage drift as advisory warnings: `A006` and `A007` are added through `audit.warn`, so default audit remains exit 0 while `--strict` can still fail on warnings.
- 2026-06-22: The default audit blocking rules still reject invalid JSON, missing question structure, legacy `q/choices/answer` question keys, invalid options, and invalid `correct` indexes; this keeps renderer-breaking schema problems distinct from inherited question-count drift.

### Progress Log
- 2026-06-22: Updated `AGENTS.md` to describe the 6-question structure as the canonical target for new or deliberately reworked quizzes, while preserving existing upstream-compatible 5/7/8-question quizzes as advisory drift.
- 2026-06-22: Updated README contribution validation text so contributors understand blocking failures versus advisory quiz count/stage warnings.
- 2026-06-22: No `quiz.json` or `quiz.zh-CN.json` files were modified.

### Verification Log
- 2026-06-22: `git diff --name-only | rg 'quiz(\\.zh-CN)?\\.json$' || true` returned no paths.
- 2026-06-22: `python3 scripts/audit_lessons.py` returned `503 lesson(s) checked, 0 issue(s), 786 advisory warning(s)`.
- 2026-06-22: `python3 scripts/check_readme_counts.py` returned `README.md counts match catalog.json totals.`
- 2026-06-22: `python3 /Users/lolita/.codex/skills/doit-queue-auditor/scripts/queue_auditor.py --root .` returned `Errors: 0`, `Warnings: 0`.
