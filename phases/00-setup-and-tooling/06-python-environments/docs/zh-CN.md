# Python 环境

> 依赖地狱是真实存在的。虚拟环境就是解药。

**类型：** Build
**语言：** Shell
**前置要求：** 第 0 阶段，第 01 课
**时间：** ~30 分钟

## 学习目标

- 使用 `uv`、`venv` 或 `conda` 创建隔离的虚拟环境
- 编写带可选依赖组的 `pyproject.toml`，并生成 lockfile 来保证可复现性
- 诊断并修复常见陷阱：全局安装、混用 pip/conda、CUDA 版本不匹配
- 为存在依赖冲突的项目实施按阶段划分的环境策略

## 要解决的问题

你为一个微调项目安装了 PyTorch 2.4。下周，另一个项目需要 PyTorch 2.1，因为它的 CUDA 构建被固定了。你在全局环境里升级，于是第一个项目坏了。你降级，于是第二个项目坏了。

这就是依赖地狱。它在 AI/ML 工作中经常发生，原因包括：

- PyTorch、JAX 和 TensorFlow 都会发布各自的 CUDA bindings
- 模型库会固定特定的框架版本
- 全局 `pip install` 会覆盖之前已经存在的包
- CUDA 11.8 builds 无法配合 CUDA 12.x drivers 使用（反之亦然）

解决办法是：每个项目都有自己隔离的环境，以及属于它自己的包。

## 核心概念

```mermaid
graph TD
    subgraph without["Without virtual environments"]
        SP[System Python] --> T24["torch 2.4.0 (CUDA 12.4)\nProject A needs this"]
        SP --> T21["torch 2.1.0 (CUDA 11.8)\nProject B needs this"]
        SP --> CONFLICT["CONFLICT: only one\ntorch version can exist"]
    end

    subgraph with["With virtual environments"]
        PA["Project A (.venv/)"] --> PA1["torch 2.4.0 (CUDA 12.4)"]
        PA --> PA2["transformers 4.44"]
        PB["Project B (.venv/)"] --> PB1["torch 2.1.0 (CUDA 11.8)"]
        PB --> PB2["diffusers 0.28"]
    end
```

## 动手实现

### 选项 1：uv venv（推荐）

`uv` 是最快的 Python 包管理器（比 pip 快 10-100 倍）。它在一个工具里处理虚拟环境、Python 版本和依赖解析。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv python install 3.12

cd your-project
uv venv
source .venv/bin/activate
```

安装包：

```bash
uv pip install torch numpy
```

一步创建带有 `pyproject.toml` 的项目：

```bash
uv init my-ai-project
cd my-ai-project
uv add torch numpy matplotlib
```

### 选项 2：venv（内置）

如果你无法安装 `uv`，Python 自带 `venv`：

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

pip install torch numpy
```

它比 `uv` 慢，但在所有已安装 Python 的地方都能用。

### 选项 3：conda（需要时使用）

Conda 会管理非 Python 依赖，比如 CUDA toolkits、cuDNN 和 C libraries。在这些情况下使用它：

- 你需要特定的 CUDA toolkit 版本，但不想把它安装到整个系统
- 你在共享集群上，无法安装系统包
- 某个库的安装说明写着 “use conda”

```bash
# Install miniconda (not the full Anaconda)
curl -LsSf https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b

conda create -n myproject python=3.12
conda activate myproject

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

一条规则：如果你在某个环境里使用 conda，那这个环境里的所有包都用 conda 管理。把 `pip install` 混进 conda env 会造成依赖冲突，而且调试起来很痛苦。

### 本课程：按阶段划分的策略

你可以为整门课创建一个环境。但不要这样做。不同阶段需要不同的依赖，有时这些依赖还会互相冲突。

策略：

```text
ai-engineering-from-scratch/
├── .venv/                    <-- shared lightweight env for phases 0-3
├── phases/
│   ├── 04-neural-networks/
│   │   └── .venv/            <-- PyTorch env
│   ├── 05-cnns/
│   │   └── .venv/            <-- same PyTorch env (symlink or shared)
│   ├── 08-transformers/
│   │   └── .venv/            <-- might need different transformer versions
│   └── 11-llm-apis/
│       └── .venv/            <-- API SDKs, no torch needed
```

`code/env_setup.sh` 中的脚本会为本课程创建基础环境。

## pyproject.toml 基础

每个 Python 项目都应该有一个 `pyproject.toml`。它用一个文件替代了 `setup.py`、`setup.cfg` 和 `requirements.txt`。

```toml
[project]
name = "ai-engineering-from-scratch"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "matplotlib>=3.8",
    "jupyter>=1.0",
    "scikit-learn>=1.4",
]

[project.optional-dependencies]
torch = ["torch>=2.3", "torchvision>=0.18"]
llm = ["anthropic>=0.39", "openai>=1.50"]
```

然后安装：

```bash
uv pip install -e ".[torch]"    # base + PyTorch
uv pip install -e ".[llm]"     # base + LLM SDKs
uv pip install -e ".[torch,llm]" # everything
```

## 锁文件

lockfile 会把每个依赖（包括传递依赖）都固定到精确版本。这能保证可复现性：任何人从 lockfile 安装，都会得到完全相同的包。

```bash
# uv generates uv.lock automatically when using uv add
uv add numpy

# pip-tools approach
uv pip compile pyproject.toml -o requirements.lock
uv pip install -r requirements.lock
```

把你的 lockfile 提交到 git。别人 clone repo 后，从 lockfile 安装，就会得到一致的版本。

## 常见错误

### 1. 安装到全局环境

```bash
pip install torch  # BAD: installs to system Python

source .venv/bin/activate
pip install torch  # GOOD: installs to virtual environment
```

检查你的包会安装到哪里：

```bash
which python       # should show .venv/bin/python, not /usr/bin/python
which pip           # should show .venv/bin/pip
```

### 2. 混用 pip 和 conda

```bash
conda create -n myenv python=3.12
conda activate myenv
conda install pytorch -c pytorch
pip install some-other-package   # BAD: can break conda's dependency tracking
conda install some-other-package # GOOD: let conda manage everything
```

如果你必须在 conda 内使用 pip（有些包只在 pip 上发布），先安装所有 conda 包，再最后安装 pip 包。

### 3. 忘记激活

```bash
python train.py           # uses system Python, missing packages
source .venv/bin/activate
python train.py           # uses project Python, packages found
```

你的 shell prompt 应该显示环境名称：

```text
(.venv) $ python train.py
```

### 4. 把 .venv 提交到 git

```bash
echo ".venv/" >> .gitignore
```

虚拟环境通常有 200MB-2GB。它们是本地环境，不能在不同机器之间移植。应该提交 `pyproject.toml` 和 lockfile。

### 5. CUDA 版本不匹配

```bash
nvidia-smi                # shows driver CUDA version (e.g., 12.4)
python -c "import torch; print(torch.version.cuda)"  # shows PyTorch CUDA version

# These must be compatible.
# PyTorch CUDA version must be <= driver CUDA version.
```

## 实际使用

运行 setup script 来创建你的课程环境：

```bash
bash phases/00-setup-and-tooling/06-python-environments/code/env_setup.sh
```

这会在 repo root 创建一个 `.venv`，并安装、验证核心依赖。

## 练习

1. 运行 `env_setup.sh`，确认所有检查都通过
2. 创建第二个虚拟环境，在其中安装不同版本的 numpy，并确认两个环境是隔离的
3. 为一个同时需要 PyTorch 和 Anthropic SDK 的项目编写 `pyproject.toml`
4. 故意在全局环境里安装一个包（不激活 venv），观察它被安装到哪里，然后卸载它

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| 虚拟环境 | “一个 venv” | 一个隔离目录，包含 Python 解释器和包，并与系统 Python 分离 |
| Lockfile | “固定的依赖” | 列出每个包及其精确版本的文件，保证跨机器安装结果一致 |
| pyproject.toml | “新的 setup.py” | 标准的 Python 项目配置文件，替代 setup.py/setup.cfg/requirements.txt |
| 传递依赖 | “依赖的依赖” | Package B 依赖 C；如果你安装依赖 B 的 A，那么 C 就是 A 的传递依赖 |
| CUDA 不匹配 | “我的 GPU 不能用了” | PyTorch 编译所用的 CUDA 版本与你的 GPU 驱动支持的版本不同 |
