# Fork 指南

这套课程采用 MIT 许可。你可以自由 fork，并按自己的需要改造。下面是更稳妥的做法。

## 面向团队

想把它用作内部培训？Fork 后按团队需要定制：

1. Fork 仓库。
2. 移除团队不需要的阶段。
3. 加入公司内部示例和数据。
4. 把内部工具集成到 outputs 中。
5. 保留来源说明——这有助于社区继续成长。

## 面向学校与大学

想把它用作课程材料？

1. Fork 仓库。
2. 把阶段映射到你的学期安排。
3. 为练习添加评分 rubrics。
4. 加入自己的作业和考试。
5. 考虑把改进贡献回 upstream。

## 面向训练营

运行付费 bootcamp？MIT 许可允许这样做。

1. Fork 并按 cohort 时间线重新组织。
2. 添加视频内容、直播课程和导师支持。
3. 代码和文档都可以在此基础上继续构建。
4. 考虑赞助项目或贡献改进。

## 面向其他编程语言

想用另一门编程语言教授这套课程？

1. Fork 仓库。
2. 用你的语言重新实现代码示例。
3. 保持课程结构和文档结构。
4. 提交 PR，把你的 fork 链接加入主 README。

## 保持 fork 更新

如果你维护的是这个中文 fork，推荐把 `origin` 保持为中文仓库，把 `upstream` 指向英文原仓库。同步时先在单独分支上做，确认中文内容、i18n 状态和生成文件边界都干净后再发 PR。

```bash
git remote add upstream https://github.com/rohitg00/ai-engineering-from-scratch.git

git fetch upstream
git merge upstream/main
```

### 中文 fork 同步 checklist

1. 开一个同步分支，不要直接在 `main` 上试错：

   ```bash
   git switch -c sync/upstream-YYYYMMDD
   git fetch upstream
   git merge --no-edit upstream/main
   ```

2. 解决冲突时优先保留中文 fork 的本地化入口；如果冲突涉及课程英文源、README、ROADMAP 或 glossary，逐文件确认链接、状态字形和表格结构没有漂移。

3. 重新检查 i18n 状态：

   ```bash
   python3 scripts/i18n_inventory.py
   python3 scripts/i18n_validate.py
   ```

   如果 inventory 发现 stale 项，只补译源文变化对应的小差异；不要用刷新 manifest 掩盖真实过期内容。

4. 重新检查课程和 README 计数：

   ```bash
   python3 scripts/audit_lessons.py
   python3 scripts/check_readme_counts.py
   ```

   PR 中的 README count drift 是 advisory；`main` 上的 workflow 会自动修复计数。只有当你手动改了 README 结构或课程链接时，才需要在本分支修 README。

5. 如涉及站点解析、README、ROADMAP 或 glossary，运行站点构建做本地验收：

   ```bash
   node site/build.js
   ```

   这个命令会重建 `site/data.js`、`site/content/`、`site/sitemap.xml` 和 `site/llms.txt` 等产物。除非当前任务明确要求，否则不要把这些生成文件提交到 PR；`site/data.js` 会在 main 上由 CI 重建。

6. 提交前确认没有误带生成文件或本地状态：

   ```bash
   git status --short
   git diff --check
   ```

   不要提交 `catalog.json`、`site/content/`、`site/sitemap.xml`、`site/llms.txt`、`i18n/manifest.jsonl` 或本地运行时状态。

7. 每个独立阶段单独提交。课程目录变更遵守“一课一提交”；维护文档、同步 checklist、i18n 修复和生成文件处理也应分开提交，方便 review 和回滚。

## 来源说明

MIT 许可不强制要求署名，但我们很感谢你保留来源：

```text
Based on AI Engineering from Scratch
https://github.com/rohitg00/ai-engineering-from-scratch
```
