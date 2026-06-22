# 变更日志

这里记录课程的新变化，最近的更新排在最前。

格式大致遵循 [Keep a Changelog](https://keepachangelog.com/)。每条记录都会说明涉及的阶段、课程以及变化内容，方便学习者直接跳到对应差异。

## [Unreleased]

### Added
- `scripts/scaffold-lesson.sh`：脚手架脚本，会创建带完整目录结构的 `phases/NN-phase/NN-lesson/`，并用 `LESSON_TEMPLATE.md` 预填 `docs/en.md` 骨架。
- `.github/PULL_REQUEST_TEMPLATE.md`：贡献者 checklist（代码可运行、代码文件不写注释、先从零构建再用框架、每课原子提交、ROADMAP 行使用 Markdown 链接）。
- `.github/ISSUE_TEMPLATE/bug_report.md` 与 `new_lesson_proposal.md`：用于 bug 报告和新课程提案的结构化入口。
- 本 `CHANGELOG.md`。

## 2026-04 — Phase 4：计算机视觉完成

### Added
- Phase 4 全部 28 节课程，覆盖图像基础、多模态视觉（VLMs）、3D、视频和自监督学习。
- `ROADMAP.md` 中的 Phase 4 行已用 Markdown 链接指向课程目录，网站可以正确展示这些课程。

### Fixed
- Phase 4 跨 15+ 节课做了一轮精度修订：
  - `phase-4/02`：shape calculator 明确 adaptive pool、flatten 和 linear 的 RF/stride 处理。
  - `phase-4/03`：backbone selector 描述列出全部覆盖家族；为 OCR、医疗、工业场景补充 head 指南。
  - `phase-4/04`：classification diagnostics 为每种 failure mode 使用量化阈值；为未定义指标声明 `n/a`；补少于 3 类的 guard。
  - `phase-4/06`：detection metric reader 使用 `AP@0.5`（不是 `mAP@0.5`）；per-class recall 声明为 optional；anchor designer 澄清 stride truncation 和 single-anchor-per-level 路径。
  - `phase-4/10`：sampler picker 声明 `unet_forward_ms` 为输入；ControlNet guard 提升为 rule 0。
  - `phase-4/14`：ViT inspector 与 refusal rule 对齐——port attempts 只审计，不背书。
  - `phase-4/24`：open-vocab stack picker 增加明确规则优先级和 license-filter 语义；concept designer 解决 step-5/rule-80 冲突。
  - `phase-4/25`：VLM docs `_merge` 在 placeholder mismatch 时抛出描述性 `ValueError`；CMER 内部做 normalization。
  - `phase-4/27`：`synthetic_frames` 将 GT boxes 裁剪到 frame H/W。
  - `phase-4/28`：`rope_3d` 校验 dim split；从 DiT block 示例中移除未使用的 `F` import。

## 2026-Q1 及更早

### Added
- Phase 0（环境搭建与工具链）：全部 12 节课。
- Phase 1（数学基础）：全部 22 节课。
- Phase 2（机器学习基础）：全部 18 节课。
- Phase 3（深度学习核心）：核心课程覆盖 perceptron、backprop 和 optimizers。
- 内置 Claude Code skills：`find-your-level`（定位测验）与 `check-understanding`（每阶段测验）。
- 网站 `aiengineeringfromscratch.com`：目录、单课页面、路线图和 277 个术语的 glossary。
- 20 个阶段的初始脚手架（`phases/00-*` 到 `phases/19-*`）。
- `LESSON_TEMPLATE.md`、`CONTRIBUTING.md`、`ROADMAP.md`、`README.md`。

[Unreleased]: https://github.com/rohitg00/ai-engineering-from-scratch/compare/HEAD...HEAD
