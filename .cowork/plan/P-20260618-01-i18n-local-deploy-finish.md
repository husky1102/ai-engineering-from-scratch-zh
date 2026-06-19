# 汉化与本地部署收尾计划

Status: approved
Review: ../review/R-20260618-01-i18n-local-deploy-finish.md
Created: 2026-06-18
Approved: 2026-06-19 by user instruction: "按 plan 执行"

## Issue I-01: Pyodide 本地自托管
Priority: high
Status: done

### Review Problem
当前站点可以本地构建和浏览中文内容，但浏览器内 Python 运行器仍依赖 jsdelivr CDN 加载 Pyodide，无法达到完全离线部署目标；用户已要求先把该收尾任务整理为 proposed 计划。

### Scope
只处理 Pyodide 本地 vendor、运行器本地路径配置、构建/部署纳入方式，以及离线运行验证所需文档或脚本。

### Out of Scope
不改课程内容、翻译内容、非 Python 运行器、无关 CLI 行为，也不在未确认体积策略前提交大型 vendor 产物。

### Steps
1. 检查 `site/runner-pyodide.js`、`site/build.js`、`bin/aefs.mjs` 和现有静态资源镜像逻辑，确认 Pyodide 加载路径和部署产物边界。
2. 新增 `scripts/vendor_pyodide.sh`，下载 Pyodide v0.26.4 到 `site/vendor/pyodide/`，并明确核心 runtime 与常用 wheel 的取舍。
3. 修改 `site/runner-pyodide.js`，让 Pyodide base URL 可配置并默认使用本地相对路径，同时设置本地 `indexURL`。
4. 调整构建或部署路径，使 `site/vendor/` 在本地部署时可被服务；如不提交 vendor 产物，则补充 `.gitignore` 与文档说明。
5. 运行构建、CLI 测试和本地站点冒烟；验证通过后按仓库要求为本阶段单独提交一次。

### Verification
- `node site/build.js`
- `npm run test:cli`
- 启动 `node bin/aefs.mjs --no-open --no-terminal --port 8731` 后访问含 Python 运行器的课程页，确认页面 HTTP 200 且本地 Pyodide 资源可加载。
- 断网或拦截网络后点击运行 Python 示例，确认没有 CDN 请求且代码能执行。

### Risks
- Pyodide 与 wheel 体积较大，提交 vendor 文件或仅提供下载脚本需要用户确认。
- 运行器路径变更可能影响非本地部署方式，需要保留可配置回退。
- 完全离线验证依赖浏览器网络面板或等价证据，执行时应记录 fresh evidence。

### Findings Log
- 2026-06-19: `lesson.html` currently does not load `runner-pyodide.js`; existing tests assert that browser code runner UI is absent. I-01 therefore removes the stale CDN dependency in the runner files and validates via a temporary same-origin smoke page instead of a production lesson click.

### Progress Log
- 2026-06-19: Downloaded Pyodide v0.26.4 core and numpy wheel into ignored `site/vendor/pyodide/v0.26.4/full/` for local validation. Browser smoke verified `print(2 + 3)` and `numpy` both output `5`.

## Issue I-02: 本地化剩余课程标题
Priority: high
Status: proposed

### Review Problem
115 篇 `docs/zh-CN.md` 的 H1 仍为英文，且 README/ROADMAP 也需要使用同一套中文课程标题；如果直接翻译顶层文档，会造成三处标题不一致。

### Scope
建立 503 课中文标题映射，替换剩余英文 H1，并同步供 README 与 ROADMAP 翻译使用；可顺手润色明确可翻译的小标题。

### Out of Scope
不大规模重写课程正文，不改代码、quiz 结构、课程目录 slug，也不翻译应保留的专有名词、API 名称或代码步骤标题。

### Steps
1. 扫描全部 `phases/*/*/docs/zh-CN.md`，生成英文 H1 和可疑英文小标题清单，并与已有中文 H1 合并为标题映射。
2. 人工复核 115 个新增中文 H1，保持术语一致，并优先复用课程正文里已经稳定的中文表达。
3. 按映射替换各课 `zh-CN.md` 的 H1，并有选择地翻译 `Pitfalls`、`Evaluation`、`Failure modes` 等非专有小标题。
4. 运行 i18n 校验和标题残留统计，必要时修正误翻或结构漂移。
5. 重建 `site/data.js`；验证通过后按仓库要求为本阶段单独提交一次。

### Verification
- `python3 scripts/i18n_validate.py`
- 重新运行英文 H1 统计，目标是剩余英文 H1 接近 0，且残留项均有专有名词保留理由。
- `node site/build.js`，并抽查 `site/data.js` 中课程标题为中文。

### Risks
- 误翻专有名词或代码步骤标题会降低教学准确性。
- 批量替换可能破坏 Markdown 结构，应分阶段扫描和校验。
- README/ROADMAP 翻译必须复用该标题映射，否则会重新产生标题漂移。

## Issue I-03: 翻译 ROADMAP.md
Priority: high
Status: proposed

### Review Problem
`ROADMAP.md` 仍为英文，并被 `site/build.js` 解析；翻译若改变状态字形、表格行或链接结构，会破坏站点进度数据。

### Scope
翻译 ROADMAP 的散文、阶段标题、说明文字和课程标题，并保留所有状态字形、链接目标、表格结构和解析不变量。

### Out of Scope
不改变课程状态、阶段数量、课程顺序、链接目标、站点解析逻辑或 README 内容。

### Steps
1. 记录翻译前结构计数：`✅`、`🚧`、`⬚`、`](`、阶段标题、表格行和 H1-H3 数量。
2. 按阶段分块翻译 ROADMAP 文案，课程标题复用 Issue I-02 的中文标题映射。
3. 每个阶段翻译后运行结构计数对比，发现漂移立即定位修正。
4. 运行课程审计和站点构建，确认 ROADMAP 解析没有新增问题。
5. 验证通过后按仓库要求为本阶段单独提交一次。

### Verification
- 结构计数保持：`✅`=524、`🚧`=1、`⬚`=2、`](`=307、阶段标题=20、表格行=543、H1-H3=21。
- `python3 scripts/audit_lessons.py`
- `node site/build.js`

### Risks
- 状态字形或表格管线被误改会破坏站点进度解析。
- 批量翻译容易让课程标题与正文 H1 不一致，应依赖 Issue I-02 的映射。
- `audit_lessons.py` 可能包含上游遗留 advisory，执行时需要区分新增问题与既有问题。

## Issue I-04: 翻译 README.md
Priority: high
Status: proposed

### Review Problem
`README.md` 仍为英文，是项目门面；它包含 503 行课程链接、徽章和 HTML 横幅，结构漂移会影响计数脚本和站点 URL 推导。

### Scope
翻译 README 正文、标题、说明和表头，并保留课程链接目标、徽章、HTML、锚点、表格结构和计数脚本可解析性。

### Out of Scope
不修改课程状态、课程数量、链接 slug、徽章计数语义、CI 自动计数逻辑或非 README 文件，除非重建 `site/data.js` 所需。

### Steps
1. 记录翻译前 README 结构计数：课程链接行、全部 `](`、shields 徽章、`<img`、表头分隔行和 `lessons-503` 徽章。
2. 按章节分块翻译 README，课程标题复用 Issue I-02 的中文标题映射，并保持每条课程链接目标不变。
3. 每块翻译后运行结构 gate，确保表格、HTML 和链接数量没有漂移。
4. 运行 README 计数检查和站点构建，确认课程 URL 仍能从链接行推导。
5. 验证通过后按仓库要求为本阶段单独提交一次。

### Verification
- 结构计数保持：课程链接行=503、全部 `](`=530、shields 徽章=6、`<img`=12、`| ---`=56，并包含 `lessons-503`。
- `python3 scripts/check_readme_counts.py`
- `node site/build.js`
- `grep -c 'tree/main/phases/' site/data.js` 输出大于 0。

### Risks
- CI 会在 main 上自动同步 README 计数；翻译必须兼容该脚本。
- 课程链接文字和 `zh-CN.md` H1 不一致会影响学习者导航体验。
- HTML 横幅和徽章不应被 Markdown 翻译工具误改。

## Issue I-05: 刷新 stale 译文与测验
Priority: medium
Status: proposed

### Review Problem
`i18n_inventory.py` 报 `docs:stale=257`、`quiz:stale=10`，说明英文源在 manifest 记录后有变化；需要复核并补译小改动，而不是全量重翻。

### Scope
处理 manifest 中 stale 的文档和测验：对比英文源变化、补译必要差异、更新对应 manifest 状态，并保持结构校验通过。

### Out of Scope
不重翻 current 文档，不改英文源内容，不补未被用户批准的上游测验缺陷，也不借机重写课程结构。

### Steps
1. 运行 `python3 scripts/i18n_inventory.py`，导出 docs 和 quiz 的 stale 清单并按改动大小排序。
2. 对每个 stale 项比较当前 `en.md` 或 quiz 与 manifest 对应源版本，判断是无需正文变化的小漂移还是需要补译的内容变化。
3. 将必要新增或改动翻译补入 `zh-CN.md` 或 `quiz.zh-CN.json`，保持题目选项和答案索引与英文结构一致。
4. 重新生成或刷新 `i18n/manifest.jsonl` 中对应记录，使 stale 项回到 current。
5. 运行 i18n inventory 与 validate gate；验证通过后按仓库要求为本阶段单独提交一次。

### Verification
- `python3 scripts/i18n_inventory.py` 目标输出包含 `docs:stale=0` 和 `quiz:stale=0`。
- `python3 scripts/i18n_validate.py` 目标为 0 issue。
- 必要时运行 `node site/build.js`，确认站点内容镜像和 `site/data.js` 未出现解析异常。

### Risks
- 257 篇文档体量较大，适合分批执行并保留每批验证证据。
- manifest 刷新若未对应真实补译，会掩盖翻译过期问题。
- 上游 5 题测验缺陷属于 Issue I-06 决策项，不应在本 issue 中擅自补题。

## Issue I-06: 收尾决策项
Priority: low
Status: proposed

### Review Problem
还有若干低优先但会影响范围的事项需要用户拍板：上游 5 题测验缺陷、贡献者向元文档汉化、fork 与上游同步策略、`.cowork/` 是否纳入 git 或 gitignore。

### Scope
逐项收集当前事实、列出可选处理方式、记录用户选择，并把已确认的后续工作拆成新的 review 或 plan issue。

### Out of Scope
不在没有明确用户选择时补题、翻译额外元文档、修改同步策略、提交或忽略 `.cowork/`。

### Steps
1. 汇总 207 套 5 题测验中的具体缺陷范围，列出保持上游一致与补足 6 题两种选项的影响。
2. 列出仍为英文的贡献者向元文档，并区分学习者可见文档和维护者内部文档。
3. 梳理 fork 同步上游后的 i18n stale 维护流程，判断是否需要脚本或文档化流程。
4. 明确 `.cowork/` 是否应纳入 git、加入 `.gitignore`，或保持仅本地队列。
5. 将用户明确选择后的工作分别转入新的 review/plan；本决策 issue 本身不直接执行实现变更。

### Verification
- 每个决策项都有用户明确选择记录。
- 若生成后续 plan，必须包含对应 review issue 和 `### User Choices`。
- 未确认的事项保持 proposed 或 blocked，不标记 done。

### Risks
- 这些事项会改变维护策略，擅自执行会扩大当前收尾范围。
- `.cowork/` 是否纳入版本管理与本次计划文件本身相关，执行前需要避免把队列文件意外当作产品文件。
- 测验补题会同时影响中英文内容和 i18n 校验，需单独规划。
