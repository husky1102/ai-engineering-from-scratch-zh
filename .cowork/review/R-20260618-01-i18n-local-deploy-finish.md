# 汉化与本地部署收尾计划

Status: planned
Source: human
Created: 2026-06-18

## Issue I-01: Pyodide 本地自托管
Priority: high
Kind: enhancement
Status: done

### Feedback
当前站点其余资源已经可以本地构建与浏览，但 `site/runner-pyodide.js` 仍从 jsdelivr CDN 拉取 Pyodide，浏览器内运行 Python 需要联网，不满足完全离线目标。

### Notes
原盘点给出的目标是新增 vendor 脚本、将 Pyodide 路径改为本地可配置、确保构建/部署纳入本地产物，并用断网运行课内代码作为验收。

2026-06-19: 已完成本地 Pyodide self-hosting 收尾：runner 默认使用 `site/vendor/pyodide/v0.26.4/full/`，vendor runtime 通过脚本下载并由 `.gitignore` 排除，CLI 静态服务器补齐 `.wasm`/`.zip`/`.whl` MIME。当前 production `lesson.html` 不加载 runner，浏览器验证改用同源临时 smoke 页完成。

### User Choices
用户要求先将 `.cowork/plan/` 下的收尾计划整理为 doIt 规范形式。本次仅创建 proposed 计划，不执行 Pyodide 改造。

## Issue I-02: 本地化剩余课程标题
Priority: high
Kind: docs
Status: done

### Feedback
115 篇 `docs/zh-CN.md` 的 H1 仍为英文，另有约 889 个可翻译英文小标题散落在 407 课，README 与 ROADMAP 后续翻译也需要统一标题来源。

### Notes
原盘点建议先形成 503 课中文标题映射，再统一替换 `zh-CN.md`、README、ROADMAP 与重建后的 `site/data.js`，避免链接列表和课程正文标题不一致。

2026-06-19: 已将剩余英文 H1 降为 0，并让 `site/build.js` 可从各课 `zh-CN.md` 读取中文 `nameZh`。后续 README/ROADMAP 翻译应继续复用这些 H1 作为课程标题来源。

### User Choices
用户要求先将 `.cowork/plan/` 下的收尾计划整理为 doIt 规范形式。本次仅创建 proposed 计划，不执行标题翻译。

## Issue I-03: 翻译 ROADMAP.md
Priority: high
Kind: docs
Status: done

### Feedback
`ROADMAP.md` 仍为英文，并被 `site/build.js` 解析；翻译时必须保留状态字形、链接、表格结构和阶段解析不变量。

### Notes
原盘点给出结构 gate：`✅`=524、`🚧`=1、`⬚`=2、`](`=307、阶段标题 20、表格行 543、H1-H3 标题 21。

2026-06-19: ROADMAP 已汉化，结构 gate 保持不变；`audit_lessons.py` 和 `site/build.js` 通过。课程标题来自 I-02 后的 `zh-CN.md` H1。

### User Choices
用户要求先将 `.cowork/plan/` 下的收尾计划整理为 doIt 规范形式。本次仅创建 proposed 计划，不执行 ROADMAP 翻译。

## Issue I-04: 翻译 README.md
Priority: high
Kind: docs
Status: planned

### Feedback
`README.md` 仍为英文，是项目门面；翻译时必须保留课程链接、徽章、HTML 横幅、锚点、表格结构以及计数脚本可解析性。

### Notes
原盘点给出结构 gate：课程链接行 503、全部 `](` 530、shields 徽章 6、`<img` 12、`| ---` 56、包含 `lessons-503`，并要求 `check_readme_counts.py` 与 `site/build.js` 通过。

### User Choices
用户要求先将 `.cowork/plan/` 下的收尾计划整理为 doIt 规范形式。本次仅创建 proposed 计划，不执行 README 翻译。

## Issue I-05: 刷新 stale 译文与测验
Priority: medium
Kind: docs
Status: planned

### Feedback
`i18n_inventory.py` 显示 `docs:stale=257`、`quiz:stale=10`，需要逐篇复核英文源变化、补小改动并刷新 manifest。

### Notes
原盘点强调这是复核补译而非重翻；完成后目标是 `i18n_inventory.py` 报 `docs:stale=0, quiz:stale=0`，且 `i18n_validate.py` 仍为 0 issue。

### User Choices
用户要求先将 `.cowork/plan/` 下的收尾计划整理为 doIt 规范形式。本次仅创建 proposed 计划，不执行 stale 刷新。

## Issue I-06: 收尾决策项
Priority: low
Kind: investigation
Status: planned

### Feedback
还存在若干需要用户拍板的低优先事项：上游 5 题测验缺陷如何处理、贡献者向元文档是否汉化、fork 与上游同步策略、`.cowork/` 是否纳入 git 或 gitignore。

### Notes
这些决策会影响后续变更范围和维护策略，不能在没有明确用户选择时直接执行。

### User Choices
用户要求先将 `.cowork/plan/` 下的收尾计划整理为 doIt 规范形式。本次仅创建 proposed 计划，不替用户决定这些事项。
