# README 本地网页 Quick Start

Status: done
Source: human
Created: 2026-06-22

## Issue I-01: README 本地网页 Quick Start
Priority: medium
Kind: docs
Status: done

### Feedback
用户要求新增一条单独 doIt review，整理 README 的开始学习/Quick Start 区域，并根据仓库当前 CLI 与站点实际情况，补上本地网页注册和启动指令。更新内容应帮助中文 fork 用户 clone 后能注册本地课程根目录、启动本地网页，并知道何时需要下载本地 Pyodide vendor。

### Notes
本地检查显示 package.json 暴露 aefs 与 learnAI 两个 bin；cli/main.mjs 支持 aefs config set-root <path>、aefs config get-root、aefs doctor、aefs [--root <path>] [--port <number>] [--no-open] [--no-terminal] [--rebuild]；默认静态站点端口从 4173 开始寻找可用端口。scripts/vendor_pyodide.sh 会下载 Pyodide v0.26.4 到 site/vendor/pyodide/v0.26.4/full/，供浏览器 Python runner 离线使用。

2026-06-22: 已更新 README 的“开始学习”区，新增中文 fork 本地网页 Quick Start、`aefs`/`learnAI` 注册根目录方式、启动参数、单课代码运行方式，以及可选 Pyodide vendor 下载说明。

### User Choices
2026-06-22: 用户明确要求新增为单独 review，并根据实际情况直接更新 README 与 Quick Start，补上本地网页注册和启动指令。

### Verification
- `python3 scripts/check_readme_counts.py`
- `node --test cli/tests/*.test.mjs`
- `curl -I http://127.0.0.1:49200/` against `node bin/aefs.mjs --port 49200 --no-open --no-terminal`
