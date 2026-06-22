# i18n 收尾后续计划

Status: approved
Review: ../review/R-20260620-01-post-i18n-followups.md
Created: 2026-06-22
Approved: 2026-06-22 by user reply: "同意"

## Issue I-01: 测验题数契约漂移
Priority: medium
Status: approved

### Review Problem
AGENTS.md 要求每个 `quiz.json` 正好 6 题，但当前 338 套测验中有 207 套不是 6 题。用户选择是保持当前收尾不直接补/裁题，并为这批漂移单独建立处理路径，避免在无范围确认时大规模改写测验内容。

### Scope
只做盘点与处理设计：复核非 6 题测验清单、阶段分布、英文/中文题数一致性、`audit_lessons.py` 的 advisory 行为，并生成一个后续可审批的分阶段 remediation 说明或 doIt review/plan。

### Out of Scope
不新增、删除、裁剪或重写任何 `quiz.json` / `quiz.zh-CN.json` 题目；不改变 `audit_lessons.py` 的 blocking/advisory 语义；不把 207 套测验混入公开文档汉化提交。

### Steps
1. 运行或编写一次只读统计，确认非 6 题测验数量、题数分布、阶段分布，以及英文/中文 quiz 题数是否仍一致。
2. 检查 `scripts/audit_lessons.py` 与 `scripts/i18n_validate.py`，记录当前 6 题契约是 advisory 还是 blocking，以及 i18n 校验为何仍通过。
3. 生成一个独立的后续处理说明，列出推荐分批顺序、每批验收命令、是否需要新增题/裁题的用户确认边界，并明确当前计划不改测验内容。
4. 更新本 plan 的 Findings/Progress 日志，若生成新的 doIt review 或维护文档，记录路径和下一步确认边界。
5. 运行验证后单独提交本 issue 的盘点/计划产物。

### Verification
- `python3 scripts/audit_lessons.py`
- `python3 scripts/i18n_validate.py`
- 重新运行 quiz 题数统计并确认输出与计划记录一致。

### Risks
- 直接修 207 套 quiz 会改变课程内容和中英文同步范围，必须留到后续明确批准。
- 如果上游后续同步改变 quiz 数量，盘点结果会过期，需要在执行时重新统计。

## Issue I-02: 公开维护文档汉化
Priority: medium
Status: approved

### Review Problem
README 与 ROADMAP 已汉化，但根目录维护文档和 GitHub 模板仍为英文，会让中文 fork 的贡献、提案、PR 和行为规范入口割裂。

### Scope
汉化公开维护者/贡献者可见文档：`CHANGELOG.md`、`CODE_OF_CONDUCT.md`、`CONTRIBUTING.md`、`FORKING.md`、`LESSON_TEMPLATE.md`、`SPONSORS.md`、`.github/ISSUE_TEMPLATE/bug_report.md`、`.github/ISSUE_TEMPLATE/new_lesson_proposal.md`、`.github/PULL_REQUEST_TEMPLATE.md`。

### Out of Scope
不翻译 `.claude/skills/`、`.agents/`、`.codex/` 或内部 agent skill 元数据；不改变 GitHub workflow 行为；不改课程正文、README 课程表、ROADMAP 状态或 lesson 文件。

### Steps
1. 建立文档 feedback loop：逐个读取公开维护文档和模板，记录需要保留的链接、占位符、checkbox、标题层级、代码块和贡献流程术语，形成修改前结构基线。
2. 将正文汉化为简体中文，保留必要英文专名、命令、文件路径、badge、URL、模板字段和法律/行为规范含义。
3. 对涉及 fork 维护或贡献流程的交叉引用保持一致；如果与 Issue I-03 的 upstream checklist 共用 `FORKING.md`，先完成基础汉化再加入 checklist。
4. 检查所有 fenced code block 仍有语言标记，Markdown 链接和模板 checkbox 未被破坏。
5. 运行验证后单独提交本 issue 的文档汉化。

### Verification
- `python3 scripts/audit_lessons.py`
- `python3 scripts/check_readme_counts.py`
- `python3 scripts/link_check.py --path CHANGELOG.md --path CODE_OF_CONDUCT.md --path CONTRIBUTING.md --path FORKING.md --path LESSON_TEMPLATE.md --path SPONSORS.md --path .github/ISSUE_TEMPLATE/bug_report.md --path .github/ISSUE_TEMPLATE/new_lesson_proposal.md --path .github/PULL_REQUEST_TEMPLATE.md --cache 0 --timeout 10`
- 手工抽查 GitHub issue/PR 模板的 checkbox、占位符和标题层级。

### Risks
- 行为规范和许可/赞助相关措辞不应被意译到改变含义。
- 链接检查可能受网络限制；若外链验证不可用，需要记录本地结构检查作为替代证据。

## Issue I-03: fork 同步检查清单
Priority: low
Status: approved

### Review Problem
当前仓库是中文 fork，并保留 `upstream` 原仓库 remote；同步 upstream 后可能重新引入英文源变更、README/ROADMAP 标题漂移和 i18n stale，需要一个人工可执行 checklist。

### Scope
在公开维护文档中加入手动 upstream sync checklist，优先放入 `FORKING.md`；内容覆盖 fetch/merge、冲突处理、i18n inventory/validate、README/ROADMAP/title drift、site build 生成文件处理和提交边界。

### Out of Scope
不实现自动同步脚本；不修改 git remotes；不执行真实 upstream merge；不改变 CI workflow。

### Steps
1. 建立 checklist feedback loop：读取当前 `FORKING.md`、README 的本地网页说明、AGENTS.md 的 CI/生成文件规则和 `.github/workflows/curriculum.yml`，提取同步后必须跑的命令并形成修改前结构基线。
2. 在 `FORKING.md` 中加入中文 fork 的手动 upstream sync checklist，明确 `origin`/`upstream` 角色、冲突处理、i18n 与站点验证、生成文件不要随 PR 提交的规则。
3. 标注何时需要重新跑 `scripts/i18n_inventory.py`、`scripts/i18n_validate.py`、`scripts/audit_lessons.py`、`scripts/check_readme_counts.py` 和 `node site/build.js`。
4. 记录网络/远程操作是人工步骤，不在文档更新执行期间自动运行。
5. 运行验证后单独提交本 issue 的 checklist 更新。

### Verification
- `python3 scripts/check_readme_counts.py`
- `python3 scripts/audit_lessons.py`
- `python3 scripts/link_check.py --path FORKING.md --cache 0 --timeout 10`
- 手工确认 checklist 没有要求提交 `site/data.js`、`catalog.json`、`site/content/`、`site/sitemap.xml` 或 `site/llms.txt`。

### Risks
- 过度自动化会超出用户选择；本 issue 只写人工流程。
- 如果远程名称或 CI 行为后来变化，checklist 需要重新复核。

## Issue I-04: .cowork 版本管理策略
Priority: low
Status: approved

### Review Problem
`.cowork/` 同时包含可复用的 durable plan/review 记录和本地运行时状态。用户选择是跟踪 durable plan/review，忽略 `current.md`、`.doit/`、history/session scratch 等本地状态。

### Scope
更新仓库级忽略规则和说明，使 `.cowork/plan/*.md`、`.cowork/review/*.md` 可继续作为 durable 记录提交，同时忽略 `.cowork/current.md`、`.cowork/.doit/`、`.cowork/history/`、`.cowork/externalQA/`、`.cowork/knowledge/` 等本地或临时状态。

### Out of Scope
不删除现有 `.cowork` durable 文件；不重写历史 plan/review；不清理用户本机 `.git/info/exclude`；不改变 doIt skill 包源码。

### Steps
1. 检查当前 `.gitignore`、`.git/info/exclude` 和已跟踪 `.cowork` 文件，确认 repo 规则与本地 exclude 的差异。
2. 在 `.gitignore` 中增加针对 `.cowork` 运行时状态的精确规则，不添加会屏蔽 `.cowork/plan/` 或 `.cowork/review/` 的宽泛规则。
3. 在合适的维护文档中说明 durable doIt 记录与本地 runtime state 的区别，以及新 review/plan 若被本地 exclude 屏蔽时需要显式暂存。
4. 验证 `.cowork/current.md` 与 `.cowork/.doit/` 被忽略，而 `.cowork/plan/*.md` 和 `.cowork/review/*.md` 仍可被跟踪。
5. 运行验证后单独提交本 issue 的 ignore/docs 更新。

### Verification
- `git check-ignore -v .cowork/current.md .cowork/.doit/index.json`
- `git check-ignore -v .cowork/plan/P-keep.md .cowork/review/R-keep.md` 应不因仓库 `.gitignore` 命中；若本地 `.git/info/exclude` 命中，记录为本机状态而非 repo 规则。
- `python3 scripts/check_readme_counts.py`
- `python3 scripts/audit_lessons.py`

### Risks
- 本机 `.git/info/exclude` 已可能包含 `/.cowork/`，会影响 `git status` 显示但不代表仓库级规则；执行时要区分本地状态和可提交规则。
- 忽略规则写得过宽会让后续 durable review/plan 难以提交。
