# AI Scientist v2：工作坊级自主研究

> Sakana 的 AI Scientist v2（Yamada et al., arXiv:2504.08066）运行完整研究循环：假设、代码、实验、图表、论文撰写、投稿。它是第一个让生成论文通过 ICLR 2025 workshop 同行评审的系统。独立评估（Beel et al.）发现，42% 的实验因编码错误失败，文献综述也经常把既有概念误标为新颖。Sakana 自己的文档警告说，该代码库会执行 LLM 编写的代码，并建议使用 Docker 隔离。这幅图景的两半，正是本课重点。

**类型:** Learn
**语言:** Python（stdlib，研究循环状态机玩具模型）
**先修:** Phase 15 · 03（AlphaEvolve），Phase 15 · 04（DGM）
**时间:** ~60 分钟

## 要解决的问题

研究是一类开放式任务。不同于 AlphaEvolve 的算法搜索，或 DGM 受基准限制的自我修改，研究成果没有机器可检查的正确性标准。论文由审稿人判断，不由单元测试判断。这让循环更难闭合；一旦闭合也更有价值，因为研究正是复合式进步发生的地方。

AI Scientist v1（Sakana, 2024）通过从人工编写的模板开始来闭合循环。LLM 在固定脚手架内填充实验。AI Scientist v2（Yamada et al., 2025）用带有视觉语言模型批评循环的 agentic tree search 移除了模板要求。系统会生成想法、实现实验、产出图表、撰写论文，并根据审稿反馈迭代。

同行评审结论：一篇 v2 生成论文被 ICLR 2025 workshop 接收（并披露了来源）。独立评估结论：系统离可靠还很远。两者都是真的。

## 核心概念

### 架构

1. **想法生成。** LLM 基于主题和既有文献提出研究想法。v1 使用模板；v2 在假设空间中使用 agentic search。
2. **新颖性检查。** 文献检索步骤检查该想法是否已经发表。这正是 Beel et al. 的评估发现误标的步骤：既有方法经常被归类为新颖。
3. **实验计划。** agent 起草实验协议并编写代码。
4. **执行。** 代码在沙箱中运行。失败会反馈到重试循环。在 Beel et al. 的测量中，42% 的实验在这一阶段因编码错误失败。
5. **图表生成。** 视觉语言模型读取生成图表，并重写图表以提升清晰度。这是 v2 的关键技术新增点。
6. **论文撰写。** LLM 起草论文，并与内部审稿器迭代。
7. **可选：投稿。** 论文提交到某个 venue。

### workshop 接收结果意味着什么

一篇 v2 生成论文通过了 ICLR 2025 workshop 的同行评审。作者向 program committee 披露了论文来源。接收是一个数据点；它不是声称该系统“会做研究”的许可证。

重要背景：workshop 论文门槛低于主会论文。同行评审有噪声；在任意一天，小比例投稿都会被接收。一次成功是概念验证，不是可靠性声明。Nature 2026 论文记录的是端到端循环，而且该论文自身由人类研究者共同署名；它不是“系统写了一篇 Nature 论文”。

### 独立评估发现了什么

Beel et al.（arXiv:2502.14297）做了一次外部评估。核心发现：

- **实验失败。** 42% 的实验因编码错误失败（错误 imports、shape mismatches、未定义变量）。重试循环捕获了一部分，但不是全部。
- **新颖性误标。** 文献检索步骤经常把既有概念标为新颖。这是研究版本的幻觉。
- **呈现质量差距。** 视觉语言图表批评产生了发表级视觉效果，掩盖了底层实验弱点。

最后一项发现对本阶段最重要。一个能产出可信外观、却没有做出可信研究的系统，比明显失败的系统更危险，而不是更安全。评估必须触及底层主张，不能停在图表层面。

### 沙箱逃逸担忧

Sakana 自己的仓库 README 警告：

> 由于该软件会执行 LLM 生成的代码，我们无法保证安全。存在危险 packages、失控网络访问以及生成非预期进程的风险。请自行承担风险，并考虑使用 Docker 隔离。

这就是未验证领域中 autonomy 的操作形态。LLM 编写代码；代码运行；代码可以做该进程被允许做的任何事。如果没有对文件系统、网络和进程动作进行硬限制的沙箱，任何自导向研究 agent 都可能外传数据、烧掉计算资源，或重写自己。

AlphaEvolve 的沙箱故事更容易，因为它的 evaluator 很紧。AI Scientist v2 的循环会用开放目标运行开放式代码。这就是为什么它需要更强隔离（Docker 是最低要求；seccomp / gVisor 更好），并且在任何投稿离开系统之前都需要人工审查。

### v2 在 frontier stack 中的位置

| System | Target | Output kind | Evaluator | Known failure |
|---|---|---|---|---|
| AlphaEvolve | algorithms | code | unit + benchmark | 受 evaluator 严谨程度限制 |
| DGM | agent scaffolding | code | SWE-bench | reward hacking |
| AI Scientist v2 | research papers | text + code + figures | peer review（弱） | 实验失败、误标、润色掩盖弱点 |

在三者中，v2 的自动 evaluator 最弱、输出表面最宽、通往公共 artifacts 的路径最短。操作控制（沙箱、审查、披露）承担了大部分安全工作。

## 实际使用

`code/main.py` 将 v2 循环模拟为一个状态机：idea → novelty check → experiment → figure → writeup → review → accept-or-iterate。每个状态都有一个可配置失败概率，取自 Beel et al. 的发现。运行 N 个循环并计数：

- 有多少想法到达 submission。
- 有多少 submissions 会带有被润色论文掩盖的关键实验缺陷。
- 重试预算如何在质量和产出率之间取舍。

## 交付成果

`outputs/skill-ai-scientist-sandbox-review.md` 是一个双关卡审查清单，用于任何研究循环 agent 产物在离开沙箱之前的检查。

## 练习

1. 使用默认参数运行 `code/main.py`。有多大比例的循环运行产出一篇“clean”论文？有多大比例产出一篇图表批评把实验失败缺陷润色过去的论文？

2. 默认值已经使用 Beel et al. 的 42% / 25%。用 `--experiment-failure 0.20 --novelty-mislabel 0.10` 重新运行，再用 `--experiment-failure 0.60 --novelty-mislabel 0.40` 运行。polished-but-flawed 占比在两次运行之间如何变化？

3. 阅读 Sakana 的 AI Scientist v2 仓库 README 中关于沙箱要求的部分。说出你会为多日自主运行添加的两个额外限制（Docker 之外）。

4. 阅读 Beel et al. 第 4 节关于 presentation-quality gap 的内容。设计一个额外 evaluator，用来抓住外观润色但实验有缺陷的论文。

5. 为 research-agent 输出提出一种人工审查协议，它要比“每篇论文都由一名 PhD 阅读”更可扩展。指出瓶颈，并围绕瓶颈设计。

## 关键术语

| Term | What people say | What it actually means |
|---|---|---|
| AI Scientist v1 | “Sakana 的模板化研究 agent” | 将实验填入固定 scaffold |
| AI Scientist v2 | “无模板研究 agent” | 带 VLM 图表批评的 agentic tree search |
| Agentic tree search | “分支式研究 agent” | 并行扩展多个实验计划；由内部 critic 剪枝 |
| Vision-language critique | “VLM 对图表润色” | 多模态模型读取图表并重写以提升清晰度 |
| Literature retrieval | “新颖性检查” | 搜索既有工作以确认想法新颖性；已有误标记录 |
| Polish masking | “漂亮论文，破损研究” | 呈现质量超过实验质量；隐藏弱点 |
| Sandbox escape | “LLM 代码逃出沙箱” | agent 执行的代码做了循环设计者没有意图允许的事 |

## 延伸阅读

- [Yamada et al. (2025). The AI Scientist-v2](https://arxiv.org/abs/2504.08066) — paper。
- [Sakana blog on the Nature 2026 publication](https://sakana.ai/ai-scientist-nature/) — 带同行评审背景的供应商总结。
- [Beel et al. (2025). Independent evaluation of The AI Scientist](https://arxiv.org/abs/2502.14297) — 外部评估数字。
- [Sakana AI Scientist v1 paper](https://arxiv.org/abs/2408.06292) — 模板化前身。
- [Anthropic — Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — 开放式研究 agent 的更广泛框架。
