# Benchmarks：WebArena 与 OSWorld

> WebArena 在四个 self-hosted apps 上测试 web-agent capability。OSWorld 在 Ubuntu、Windows、macOS 上测试 desktop-agent capability。发布时（2023-2024），二者都显示 best-in-class agents 与 humans 之间存在巨大差距。差距正在缩小，但 failure modes 没变。

**类型:** Learn
**语言:** Python（stdlib）
**先修:** Phase 14 · 19（SWE-bench, GAIA）
**时间:** ~60 分钟

## 学习目标

- 描述 WebArena 的四个 self-hosted apps，以及为什么 execution-based evaluation 很重要。
- 解释为什么 OSWorld 使用真实 OS screenshots，而不是 accessibility APIs。
- 说出两个主要 OSWorld failure modes：GUI grounding 和 operational knowledge。
- 总结 OSWorld-G 与 OSWorld-Human 在 base benchmark 之上增加了什么。

## 要解决的问题

Generalist agents 可以调用 tools。它们能否跨 20 次点击驱动浏览器完成购物 checkout？能否只用键盘和鼠标配置 Linux box？WebArena 和 OSWorld 回答这些问题。

## 核心概念

### WebArena（Zhou et al., ICLR 2024）

- 812 个 long-horizon tasks，横跨四个 self-hosted web apps：购物网站、论坛、GitLab-like dev tool、business CMS。
- 加上 utilities：map、calculator、scratchpad。
- Evaluation 通过 gym APIs 做 execution-based：订单是否下单、issue 是否关闭、CMS page 是否更新。
- 发布时：最佳 GPT-4 agent 成功率 14.41%，human 78.24%。

Self-hosted framing 很重要，benchmark 不 flaky，因为 target apps 被 pin 且可复现。

### Extensions

- **VisualWebArena**：visually grounded tasks，成功依赖图像理解（screenshots 作为 first-class observations）。
- **TheAgentCompany**（2024 年 12 月）：加入 terminal + coding；更像真实 remote-work environment。

### OSWorld（Xie et al., NeurIPS 2024）

- 横跨 Ubuntu、Windows、macOS 的 369 个真实 computer tasks。
- 对真实 applications 进行 free-form keyboard and mouse control。
- 1920×1080 screenshots 作为 observation。
- 发布时：最佳模型 12.24%，human 72.36%。

### Primary failure modes

1. **GUI grounding.** Pixel → element mapping。模型难以在 1920×1080 中可靠定位 UI elements。
2. **Operational knowledge.** 哪个 menu 有 setting、哪个 keyboard shortcut、哪个 preference pane。这是人类多年积累的知识长尾。

### Follow-ups

- **OSWorld-G**：564-sample grounding suite + Jedi training set。把 grounding 与 planning 分解，使你可以分别测量。
- **OSWorld-Human**：人工 curated gold action trajectories。显示 top agents 使用了必要步数的 1.4-2.7 倍（trajectory-efficiency gap）。

### 为什么重要

Claude computer use、OpenAI CUA、Gemini 2.5 Computer Use（Lesson 21）都在 WebArena 和 OSWorld 形状的 workloads 上训练。Benchmark 是目标；production models 是交付的答案。

### Benchmarking 常见错误

- **Screenshot-only evals.** OSWorld 是 screenshot-driven；如果在 OSWorld 上评估使用 DOM 或 accessibility APIs 的 agent，就会错过 grounding challenge。
- **Ignoring trajectory length.** 只看 success-rate 会错过 OSWorld-Human 揭示的 1.4-2.7x step inefficiency。
- **Stale self-hosted apps.** WebArena 的 apps pin 到特定 versions；不重新 curate 就升级会破坏可比性。

## 动手实现

`code/main.py` 实现一个 toy web-agent harness：

- 一个最小 “shopping app” state machine：list_items、add_to_cart、checkout。
- 3 个 tasks 的 gold trajectories。
- 一个 scripted agent 尝试每个 task。
- Execution-based evaluator（state check）和 trajectory-efficiency metric（steps vs gold）。

运行它：

```text
python3 code/main.py
```

输出：per-task success rate 和 trajectory efficiency，映射 OSWorld-Human 方法论。

## 实际使用

- **WebArena Verified**：在 internal cluster 上 self-hosted，用于 continuous evaluation。
- **OSWorld**：在 VM fleet 中用于 desktop agents。
- **Computer-use agents**（Lesson 21）：Claude、OpenAI CUA、Gemini 都在类似 workloads 上训练。
- **Your own product flows**：为 top 20 tasks 捕捉 gold trajectories；每周运行 agents。

## 交付成果

`outputs/skill-web-desktop-harness.md` 构建带 execution-based eval 和 trajectory efficiency metric 的 web/desktop agent harness。

## 练习

1. 用第二个 app（论坛）扩展 toy harness。编写 3 个 tasks 和 gold trajectories。
2. 添加 per-task trajectory-efficiency reporting。在你的 toy 上，agent 是 gold 的 1x、2x 还是 3x？
3. 实现一个“distractor”tool，也就是 gold trajectory 从不使用的 tool。Scripted agent 会被诱惑吗？
4. 阅读 OSWorld-G。你会如何在自己的 evals 中分离 grounding failures 与 planning failures？
5. 阅读 WebArena apps README。升级某个 pinned app version 时会坏掉什么？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| WebArena | “Web agent benchmark” | 4 个 self-hosted apps 上的 812 tasks；gym-style evaluation |
| VisualWebArena | “Visual WebArena” | Visually grounded WebArena；screenshots 是 observations |
| OSWorld | “Desktop agent benchmark” | 真实 Ubuntu/Windows/macOS 上的 369 tasks |
| GUI grounding | “Pixel-to-element mapping” | 模型在 1920x1080 中定位 UI elements |
| Operational knowledge | “OS know-how” | 哪个 menu、shortcut、preference pane |
| OSWorld-G | “Grounding suite” | 564 个 grounding-only samples + training set |
| OSWorld-Human | “Gold trajectories” | 用来测 efficiency 的人工 expert action sequences |
| Trajectory efficiency | “Steps over gold” | Agent step count 除以 human minimum |

## 延伸阅读

- [Zhou et al., WebArena (arXiv:2307.13854)](https://arxiv.org/abs/2307.13854) — four-app web benchmark
- [Xie et al., OSWorld (arXiv:2404.07972)](https://arxiv.org/abs/2404.07972) — cross-OS desktop benchmark
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — Claude 的 benchmark-shaped capability
- [OpenAI, Computer-Using Agent](https://openai.com/index/computer-using-agent/) — OSWorld 和 WebArena numbers
