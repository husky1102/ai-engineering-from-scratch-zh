# Scalable Oversight 与 Weak-to-Strong Generalization

> Burns 等人（OpenAI Superalignment，“Weak-to-Strong Generalization”，2023）提出了一个 superalignment problem 的代理任务：使用较弱模型产生的 labels 来 fine-tune 一个强模型。如果强模型能够从不完美的 weak supervision 中正确泛化，那么当前 human-scale alignment methods 也许可以扩展到 superhuman systems。Scalable oversight 与 W2SG 是互补的。Scalable oversight（debate、recursive reward modeling、task decomposition）提升 overseer 的有效能力，使其能够跟上被监督模型。W2SG 确保强模型能从 overseer 提供的任何不完美监督中正确泛化。Debate Helps W2SG（arXiv:2501.13124，2025 年 1 月）把二者结合起来。

**类型:** Learn
**语言:** Python (stdlib, W2SG gap simulator)
**先修:** Phase 18 · 01 (instruction-following), Phase 18 · 10 (AI Control), Phase 09 (RL foundations)
**时间:** ~60 minutes

## 学习目标

- 定义 scalable oversight 与 weak-to-strong generalization，并解释二者如何互补。
- 描述 Burns 等人 2023 的实验设置：使用来自 GPT-2 的 labels 来 fine-tune GPT-4。
- 解释 performance gap recovered（PGR）metric，以及它测量什么。
- 说明三种主要 scalable-oversight mechanisms（debate、recursive reward modeling、task decomposition），并给出每种的一项优势。

## 要解决的问题

Phase 18 到目前为止的每一种 alignment technique 都假设 overseer 能够评估模型行为。当模型达到超人水平时，overseer 就是薄弱环节。Superalignment 问题是：较弱的 supervisor 能否可靠地产生更强且 aligned 的模型？

Burns 等人把它缩减成一个可操作化的实证设置：用 weak 监督 strong，测量 strong model 的能力有多少能在 weak supervision 下存活。这不是 superalignment 的解决方案——它是一个让我们可以衡量进展的 proxy。

## 核心概念

### W2SG：Burns 等人的设置

- Weak model：GPT-2 class。
- Strong model：GPT-4 class。
- Target：任务上的 strong GPT-4 ceiling。

流程：
1. 获取 weak model 在任务上的 zero-shot predictions。
2. 在 weak-labeled data 上 fine-tune strong model。
3. 测量 fine-tuned strong model 的 accuracy。

Baselines：weak model 自身 accuracy，strong model 在 gold labels 上的 supervised ceiling。

Gap metric：Performance Gap Recovered (PGR) = (fine-tuned - weak) / (ceiling - weak)。PGR 为 1.0 表示 weak supervision 完全关闭 gap；PGR 为 0 表示 weak supervision 没有帮助。

### Burns 等人的实证发现

在 NLP、chess-puzzle 和 reward-modeling tasks 上，PGR 持续为正：strong model 能够泛化到 weak supervisor 的错误之外。具体任务的 PGR 从 ~20% 到 ~80% 不等。Strong model 会从不完美 labels 中 “理解” 目标任务，并用自己的 pre-trained priors 填补空白。

Burns 等人指出的限制：
- 这里的 weak-to-strong gap 是 capability，而不是 alignment。Weak model 是 aligned；真正的 superhuman model 也许能被人类 aligned，但 capability gap 会大得多。
- Strong model 的 generalization 可能更多来自 task priors，而不是恢复 ground-truth intent。
- Eliciting strong model's latent knowledge 才是真问题；PGR 衡量的是一种具体 operationalization。

### Scalable oversight：三种机制

- **Debate（Irving 等人 2018）。** 两个 U instances 论证相反立场；weak judge 作出判断。假设是：找到一个有说服力的真答案，比找到一个有说服力的假答案更容易。2024-2025 年近期工作（Khan 等人、Michael 等人）显示 debate 有时有帮助，有时没有，且是否有帮助取决于 task structure。
- **Recursive Reward Modeling（Leike 等人 2018）。** U 帮助人类为 U+1 训练 reward model。Overseer 的有效能力随着模型增长。
- **Task Decomposition（Christiano、Shlegeris、Amodei 2018）。** 把困难任务递归地分解成 human 可以检查的 sub-tasks。假设任务可分解。

每种机制都对 task structure 或 intermediate components 的 alignment 作出假设。

### 为什么 scalable oversight 与 W2SG 互补

Scalable oversight 提高 overseer 的有效信号质量。
W2SG 从 overseer 能提供的任何不完美信号中关闭 gap。

Lang 等人——Debate Helps Weak-to-Strong Generalization（arXiv:2501.13124）把二者结合起来：debate protocol 提供更好的 weak labels，然后 strong model 在这些 labels 上训练。论文报告了 NLP tasks 上的 PGR gains。

### 组织层面的戏剧性变化

OpenAI 的 Superalignment team 在 Jan Leike 离职去 Anthropic 后，于 2024 年 5 月解散。这个议程（scalable oversight、W2SG、automated alignment research）继续在 Anthropic 和学术实验室中推进——MATS（第 28 课）、Redwood（第 10 课）、Apollo（第 8 课）、METR（第 28 课）。组织结构变了；研究问题没有变。

### 它在 Phase 18 中的位置

第 6-10 课描述 threat，以及在假设 U 不可信时的 defensive paradigm。第 11 课是 offensive paradigm：让 overseer 足够强，从而验证 U 的 alignment。第 12-16 课随后转向 adversarial evaluation 的实践 tooling。

## 实际使用

`code/main.py` 在 synthetic task 上模拟一次 W2SG fine-tune。Weak labeler 有 70% accuracy，带 structured errors；strong model 在 gold labels 上有 95% ceiling。你会在 weak labels 上 fine-tune strong model，测量 PGR，并与 strong-on-gold 和 weak-alone 比较。

## 交付成果

本课产出 `outputs/skill-w2sg-pgr.md`。给定一份 oversight setup description，它会识别 weak supervisor、strong model、supervision quality，并计算（或请求）PGR。它还会标记该 claim 是 “weak can supervise strong” 还是 “weak + oversight mechanism can supervise strong”。

## 练习

1. 运行 `code/main.py`。报告 weak_accuracy = 0.60、0.70、0.80 时的 PGR。解释 PGR curve 的形状。

2. 修改 weak labeler，让它有 structured error（例如在某个特定 input class 上总是错误）。PGR 会增加、减少，还是保持不变？解释原因。

3. 阅读 Burns 等人 2023 Section 4.3（NLP tasks）。复现 “confidence auxiliary loss” 的直觉：当 strong model 比 weak labels 更自信时，谁会赢？

4. 为 software-engineering task 设计一个结合 debate 和 task decomposition 的 scalable-oversight protocol。分别说出每个组件的一个 failure mode，并解释组合如何处理或未能处理它们。

5. 说明什么证据会证伪 “weak-to-strong generalization 是通往 superalignment 的可行路径” 这一 claim。请具体说明你需要看到的 empirical signature。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Scalable oversight | “让 overseer 更强” | 提升 overseer 评估更强模型能力的机制 |
| W2SG | “weak supervises strong” | 在 weak labels 上 fine-tuning strong model，并测量恢复的 capability |
| PGR | “performance gap recovered” | (fine-tuned - weak) / (ceiling - weak)；1.0 = 完全关闭，0 = 没有帮助 |
| Debate | “两个 U instances 辩论” | Scalable oversight 机制，其中 weak judge 在两个 U defenders 之间选择 |
| RRM | “recursive reward modeling” | U 帮助训练 U+1 的 reward model；overseer capability 跟随 U |
| Task decomposition | “human 检查 sub-tasks” | 把困难任务递归地拆成 human 可以验证的 sub-tasks |
| Superalignment | “对齐超人 AI” | 关注对齐人类无法直接评估的模型的研究议程 |

## 延伸阅读

- [Burns et al. — Weak-to-Strong Generalization (OpenAI 2023)](https://openai.com/index/weak-to-strong-generalization/) — W2SG paper
- [Irving, Christiano, Amodei — AI safety via debate (arXiv:1805.00899)](https://arxiv.org/abs/1805.00899) — debate mechanism
- [Leike et al. — Scalable agent alignment via reward modeling (arXiv:1811.07871)](https://arxiv.org/abs/1811.07871) — recursive reward modeling
- [Khan et al. — Debating with More Persuasive LLMs Leads to More Truthful Answers (arXiv:2402.06782)](https://arxiv.org/abs/2402.06782) — 2024 年 stronger debaters 的 debate 实证研究
- [Lang et al. — Debate Helps Weak-to-Strong Generalization (arXiv:2501.13124)](https://arxiv.org/abs/2501.13124) — 2025 年 debate + W2SG 组合
