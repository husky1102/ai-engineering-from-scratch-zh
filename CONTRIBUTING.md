# 贡献指南

课程、翻译、修复、产物都欢迎贡献。一个 pull request 只做一类贡献，可以让 review 更快，也能让贡献统计和署名更准确。

## 重要：README 和 ROADMAP 会喂给网站

`site/build.js` 会解析 `README.md`、`ROADMAP.md` 和 `glossary/terms.md`，生成 `site/data.js`。任何改动这些文件的 pull request 都必须保持下面两类结构不变：

- Phase 标题可以是 `### Phase N: Name \`X lessons\``，也可以是 `<details><summary><b>Phase N — Name</b> ... <code>X lessons</code> ... <em>Description</em></summary>`。
- 课程表保持列形状 `| # | Lesson | Type | Lang |`（综合项目表可用 `| # | Project | Combines | Lang |`）。`Lang` 列可以写纯文本（`Python, TypeScript`），也可以保留旧的语言 emoji 标记（`🐍 🟦 🦀 🟣 ⚛️`）；解析器会把两者视为等价。
- `ROADMAP.md` 的阶段标题和课程行必须保留状态字形（`✅`、`🚧`、`⬚`）。不要替换成文字——解析器依赖这些精确字符。

编辑这些文件后运行 `node site/build.js`；如果结构安全，`git diff site/data.js` 应该只看到时间戳变化。

## 贡献方式

### 1. 新增课程

每节课位于 `phases/XX-phase-name/NN-lesson-name/`，目录结构如下：

```text
NN-lesson-name/
├── code/           至少一个可运行实现
├── notebook/       用于实验的 Jupyter notebook（可选）
├── docs/
│   └── en.md       课程文档（必需）
└── outputs/        本课产出的 prompts、skills 或 agents（如适用）
```

**课程文档格式**（`en.md`）：

```markdown
# Lesson Title

> One-line motto — the core idea in one sentence.

## The Problem

Why does this matter? What can't you do without this?

## The Concept

Explain with diagrams, visuals, and intuition. Code comes later.

## Build It

Step-by-step implementation from scratch.

## Use It

Now use a real framework or library to do the same thing.

## Ship It

The prompt, skill, agent, or tool this lesson produces.

## Exercises

1. Exercise one
2. Exercise two
3. Challenge exercise
```

### 2. 新增翻译

在任意课程的 `docs/` 目录中新建语言文件：

```text
docs/
├── en.md    （英文，始终必需）
├── zh.md    （中文）
├── ja.md    （日文）
├── es.md    （西班牙文）
├── hi.md    （印地文）
└── ...
```

保持与英文版本相同的结构。翻译正文，不翻译代码。

### 3. 新增产物

如果某节课应产出可复用 prompt、skill、agent 或 MCP server：

1. 在该课程的 `outputs/` 目录中创建产物。
2. 在顶层 `outputs/` 索引中添加引用。

**Prompt 格式：**

```markdown
---
name: prompt-name
description: What this prompt does
phase: 14
lesson: 01
---

[System prompt or template here]
```

**Skill 格式：**

```markdown
---
name: skill-name
description: What this skill teaches
version: 1.0.0
phase: 14
lesson: 01
tags: [agents, loops]
---

[Skill content here]
```

### 4. 修复 bug 或改进已有课程

- 修复无法运行的代码
- 改进解释
- 添加更好的图示
- 更新过时信息

### 5. 新增练习或项目

欢迎更多练习和项目，尤其是能把多个阶段连接起来的内容。

## 准则

- **代码必须能运行。** 每个代码文件都应能在列出的依赖下无错误执行。
- **代码文件不写注释。** 代码应自解释，解释放在文档里。
- **使用最适合任务的语言。** 不要在 TypeScript 或 Rust 更合适时强行使用 Python。
- **先从零构建。** 先用第一性原理实现概念，再展示框架版本。
- **保持实用。** 理论服务于实践，而不是反过来。
- **不要 AI 味填充。** 像人一样写作，直接，删掉废话。

## Pull Request 流程

1. Fork 仓库。
2. 创建 feature branch（`git checkout -b add-lesson-phase3-gradient-descent`）。
3. 完成你的修改。
4. 确认所有代码都能运行。
5. 提交 pull request，并写清楚描述。

## 行为准则

见 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。保持友善、乐于帮助，并提出建设性意见。

## 风格

- 文字直接，删掉填充句。匹配手册语气，不写营销文案。
- 标题里不要放装饰性 emoji。`Lang` 列的语言 emoji 是唯一例外，因为解析器会映射它们。
- 代码要能用课程列出的依赖直接运行。
- 先从零构建，再展示框架版本。
