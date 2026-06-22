# 课程模板

创建新课程时使用这个模板。复制目录结构，然后填入内容。

## 目录结构

```text
NN-lesson-name/
├── code/
│   ├── main.py            （主实现）
│   ├── main.ts            （TypeScript 版本，如适用）
│   ├── main.rs            （Rust 版本，如适用）
│   └── main.jl            （Julia 版本，如适用）
├── notebook/
│   └── lesson.ipynb       （用于实验的 Jupyter notebook）
├── docs/
│   └── en.md              （课程文档）
└── outputs/
    ├── prompt-*.md        （本课产出的 prompts）
    └── skill-*.md         （本课产出的 skills）
```

## 文档格式（docs/en.md）

```markdown
# [Lesson Title]

> [One-line motto — the core idea that sticks]

**Type:** Build | Learn
**Languages:** Python, TypeScript, Rust, Julia (list what's used)
**Prerequisites:** [List prior lessons needed]
**Time:** ~[estimated time] minutes

## The Problem

[2-3 paragraphs. What can't you do without this? Why should you care?
Make it concrete — show a scenario where not knowing this hurts.]

## The Concept

[Explain with diagrams and intuition. No code yet.
Use ASCII diagrams, tables, or link to visuals in the web app.
Build mental models before implementation.]

## Build It

[Step-by-step implementation from scratch.
Start with the simplest version, then add complexity.
Every code block should be runnable on its own.]

### Step 1: [Name]

[Explanation]

    [code block]

### Step 2: [Name]

[Explanation]

    [code block]

[...continue...]

## Use It

[Now show how frameworks/libraries do the same thing.
Compare your from-scratch version to the library version.
This proves the concept and introduces practical tools.]

## Ship It

[What reusable artifact does this lesson produce?
Could be a prompt, a skill, an agent, an MCP server, or a tool.
Include it here and save it in the outputs/ folder.]

## Exercises

1. [Easy — reinforce the core concept]
2. [Medium — apply it to a different problem]
3. [Hard — extend or combine with prior lessons]

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| [term] | [common misconception] | [actual definition] |

## Further Reading

- [Resource 1](url) — [why it's worth reading]
- [Resource 2](url) — [why it's worth reading]
```

## 代码文件准则

- 代码必须无错误运行。
- 不写注释——代码应自解释。
- 使用最适合主题的语言。
- 如果有依赖，包含 `requirements.txt` 或等价文件。
- 从最简单版本开始，再逐步增加复杂度。
- 每个函数和类都应有清晰目的。

## 产物文件格式

### Prompts

```markdown
---
name: prompt-name
description: What this prompt does
phase: [phase number]
lesson: [lesson number]
---

[Prompt content]
```

### Skills

```markdown
---
name: skill-name
description: What this skill teaches
version: 1.0.0
phase: [phase number]
lesson: [lesson number]
tags: [relevant, tags]
---

[Skill content]
```
