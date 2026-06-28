# README 本地网页 Quick Start 完成

Status: done
Review: ../review/R-20260622-01-readme-quick-start.md
Plan: ../plan/P-20260622-03-readme-quick-start.md
Completed: 2026-06-28

## Issue I-01: README 本地网页 Quick Start 完成

### Review Summary
README 开始学习区域补齐中文 fork 本地网页 Quick Start，覆盖课程根目录注册、本地站点启动、单课代码运行和可选 Pyodide vendor 下载说明。

### User Choice
用户明确要求新增为单独 review，并根据仓库当前 CLI 与站点实际情况直接更新 README 与 Quick Start。

### Changes Made
更新 README 的开始学习区，说明 aefs / learnAI 注册课程根目录、启动本地网页、端口参数、单课代码运行方式，以及离线 Python runner 所需的 Pyodide vendor 下载。

### Verification
记录的验证包括：python3 scripts/check_readme_counts.py；node --test cli/tests/*.test.mjs；curl -I http://127.0.0.1:49200/ against node bin/aefs.mjs --port 49200 --no-open --no-terminal。

### Files Changed
- README.md

### Follow-ups
- None
