# 开发环境

> 你的工具会塑造你的思考。一次搭好，并且搭对。

**类型：** Build
**语言：** Python, TypeScript, Rust
**前置要求：** 无
**时间：** ~45 分钟

## 学习目标

- 从零搭建 Python 3.11+、Node.js 20+ 和 Rust 工具链
- 配置虚拟环境和包管理器，让构建可以复现
- 使用 CUDA/MPS 验证 GPU 访问，并运行一次测试张量运算
- 理解四层技术栈：系统、包、运行时、AI 库

## 要解决的问题

你将用 Python、TypeScript、Rust 和 Julia 学习 200 多节 AI 工程课程。如果你的环境是坏的，每一节课都会变成和工具链搏斗，而不是学习本身。

大多数人会跳过环境配置。然后他们花几个小时调试导入错误、版本冲突和缺失的 CUDA 驱动。我们要把这件事认真地、一次性做好。

## 核心概念

一个 AI 工程环境有四层：

```mermaid
graph TD
    A["4. AI/ML 库\nPyTorch、JAX、transformers 等"] --> B["3. 语言运行时\nPython 3.11+、Node 20+、Rust、Julia"]
    B --> C["2. 包管理器\nuv、pnpm、cargo、juliaup"]
    C --> D["1. 系统基础\nOS、shell、git、编辑器、GPU 驱动"]
```

我们自底向上安装。每一层都依赖它下面的那一层。

## 动手实现

### 步骤 1：系统基础

检查你的系统并安装基础工具。

```bash
# macOS
xcode-select --install
brew install git curl wget

# Ubuntu/Debian
sudo apt update && sudo apt install -y build-essential git curl wget

# Windows (use WSL2)
wsl --install -d Ubuntu-24.04
```

### 步骤 2：使用 uv 配置 Python

我们使用 `uv`，它比 pip 快 10-100 倍，并且会自动处理虚拟环境。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv python install 3.12

uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

uv pip install numpy
```

验证：

```python
import sys
print(f"Python {sys.version}")

import numpy as np
print(f"NumPy {np.__version__}")
a = np.array([1, 2, 3])
print(f"Vector: {a}, dot product with itself: {np.dot(a, a)}")
```

### 步骤 3：使用 pnpm 配置 Node.js

用于 TypeScript 课程（智能体、MCP 服务器、Web 应用）。

```bash
curl -fsSL https://fnm.vercel.app/install | bash
fnm install 22
fnm use 22

npm install -g pnpm

node -e "console.log('Node', process.version)"
```

### 步骤 4：Rust

用于对性能要求很高的课程（推理、系统）。

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

rustc --version
cargo --version
```

### 步骤 5：Julia（可选）

用于那些 Julia 特别擅长的重数学课程。

```bash
curl -fsSL https://install.julialang.org | sh

julia -e 'println("Julia ", VERSION)'
```

### 步骤 6：GPU 配置（如果你有 GPU）

```bash
# NVIDIA
nvidia-smi

# Install PyTorch with CUDA
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

没有 GPU？没问题。大多数课程都可以在 CPU 上运行。对于训练量很大的课程，可以使用 Google Colab 或云端 GPU。

### 步骤 7：验证全部环境

运行验证脚本：

```bash
python phases/00-setup-and-tooling/01-dev-environment/code/main.py
```

默认情况下，验证器会打印缺失工具的诊断并退出 0，因为这节课经常在环境尚未完成时运行。需要让 shell 脚本或 CI gate 因缺少必需工具而失败时，加上 `--strict`。

## 实际使用

你的环境现在已经可以支撑本课程中的每一节课。下面是各语言的使用位置：

| 语言 | 使用位置 | 包管理器 |
|----------|---------|-----------------|
| Python | 第 1-12 阶段（ML、DL、NLP、Vision、Audio、LLMs） | uv |
| TypeScript | 第 13-17 阶段（工具、智能体、群体、基础设施） | pnpm |
| Rust | 第 12、15-17 阶段（对性能要求很高的系统） | cargo |
| Julia | 第 1 阶段（数学基础） | Pkg |

## 交付成果

本课产出一个验证脚本，任何人都可以运行它来检查自己的环境配置。

查看 `outputs/prompt-env-check.md`，其中有一个提示词，可以帮助 AI 助手诊断环境问题。

## 练习

1. 运行验证脚本并修复所有失败项
2. 为本课程创建一个 Python 虚拟环境，并安装 PyTorch
3. 用四种语言分别写一个 "hello world"，并逐一运行
