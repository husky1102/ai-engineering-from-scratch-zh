# 测验题数契约修复路径

Status: new
Source: human
Created: 2026-06-22

## Issue I-01: 为 207 套非 6 题测验制定修复批次
Priority: medium
Kind: docs
Status: new

### Feedback
AGENTS.md 的课程契约要求 `quiz.json` 正好 6 题，但当前仓库有 207 套英文测验不是 6 题。此前用户选择是保持当前 i18n 收尾不直接补题或裁题，并单独建立后续处理路径。

### Notes
2026-06-22 只读复核结果：338 套 `quiz.json` 中，题数分布为 5 题 118 套、6 题 131 套、7 题 60 套、8 题 29 套，非 6 题合计 207 套。阶段分布为 Phase 0: 7、Phase 1: 22、Phase 2: 18、Phase 3: 13、Phase 4: 28、Phase 5: 29、Phase 7: 1、Phase 10: 11、Phase 11: 16、Phase 14: 38、Phase 16: 2、Phase 17: 1、Phase 19: 21。英文 `quiz.json` 与中文 `quiz.zh-CN.json` 题数不一致项为 0。

`scripts/audit_lessons.py` 中 `REQUIRED_QUIZ_QUESTION_COUNT = 6`，非 6 题当前作为 `A006` advisory warning 报告；默认 audit 仍返回 0，只有 `--strict` 才会因 warning 失败。`scripts/i18n_validate.py` 校验英文与中文题数一致性，不要求题数必须为 6，所以当前输出为 `docs 503/503, quiz 338/338, 0 issue(s)`。

推荐后续批次：
1. 先处理 118 套 5 题测验：为每套补 1 题，并同步补入 `quiz.zh-CN.json`。
2. 再处理 60 套 7 题测验：逐套判断是合并、删除低价值题，还是拆分为后续练习；需要用户确认裁题原则。
3. 最后处理 29 套 8 题测验：优先检查 Phase 5 是否有成组历史模板问题，再决定裁题或迁移额外题目。
4. 每批都必须运行 `python3 scripts/audit_lessons.py`、`python3 scripts/i18n_validate.py`，并抽查被改课程的中英文题目、答案索引和 stage 分布。

2026-06-22 追加复核：338 套英文 quiz 中，顶层 object schema 为 283 套、旧 list schema 为 55 套。非 6 题的模式高度集中：
- 118 套 5 题：集中在 Phase 00/01/02/03/04/07/10/11/16，典型 stage 分布为 `pre:2, post:3`，缺少 `check` 阶段。
- 60 套 7 题：集中在 Phase 14/17/19，典型 stage 分布为 `pre:2, check:3, post:2`，比契约多 1 个 `pre`。
- 29 套 8 题：全部在 Phase 05，典型 stage 分布为 `pre:2, check:3, post:3`，比契约多 1 个 `pre` 和 1 个 `post`。

这说明修复不只是题数问题，还牵涉旧 schema、stage 分布和内容取舍：5 题批次需要新增 `check` 题；7/8 题批次需要决定是裁掉额外题、迁移为练习材料，还是修改课程契约允许更多题。

### User Choices
Pending: 需要用户确认是否启动实际 quiz remediation，以及 7/8 题测验采用“裁题到 6 题”“保留内容但调整契约”或“把额外题迁移到练习/扩展材料”的处理原则。
