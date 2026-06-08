# CAIS、CAISI 与 Societal-Scale Risk

> Center for AI Safety（CAIS，旧金山，2022 年由 Hendrycks 和 Zhang 创立）发布 four-risk framework：malicious use、AI races、organizational risks、rogue AIs，以及 2023 年 5 月由数百名教授和公司领导签署的 extinction risk statement。CAIS 2026 releases：用于 frontier-model evaluation 的 AI Dashboard、Remote Labor Index（与 Scale AI）、Superintelligence Strategy Paper、AI Frontiers newsletter。另一个不同实体：NIST Center for AI Standards and Innovation（CAISI）：面向美国政府的 voluntary agreements，以及聚焦 cyber、bio 和 chemical-weapons risks 的 unclassified capability evaluations。CAIS 将 organizational risk 标记为四个 top-level risks 之一：safety culture、rigorous audits、multi-layered defenses 和 information security 是基础，但经常被拿来与 deployment speed 交换。California SB-53 如果签署，将成为美国第一个 state-level catastrophic-risk regulation。

**类型：** 学习
**语言：** Python (stdlib, four-risk inventory and mitigation matcher)
**先修：** Phase 15 · 19 (RSP), Phase 15 · 20 (PF + FSF)
**时间：** ~45 分钟

## 要解决的问题

Lessons 19 和 20 覆盖了 lab-internal scaling policies。Lesson 21 覆盖了 independent capability evaluation。本课覆盖第三种视角：塑造 public discussion 和 catastrophic AI risk regulatory baseline 的 civil society 和 government organizations。

两个不同实体很重要。CAIS 是 non-profit research org，发布思考 AI risk 的 frameworks，并协调 public statements。CAISI 是 NIST 内部的 US-government center，运行与 labs 的 voluntary agreements 和 unclassified capability evaluations。名字押韵；使命并不重叠。从业者应该都知道。

实际内容：CAIS 的 four-risk framework 是文献中引用最广的 societal-scale-risk taxonomy。Safety culture 和 organizational risk 是这四者之一，而且是从业者最能直接控制的一个。SB-53（California）如果签署，会成为美国第一个 state-level catastrophic-risk regulation；该 bill 的 framing 很重要，因为在美国科技政策中，state-level regulation 历史上常常先于 federal action。

## 核心概念

### CAIS：Center for AI Safety

- Founded：2022 年，旧金山，由 Dan Hendrycks 和同事创立（“Zhang” 名字指一位早期 collaborator，而不是当前 co-founder；当前 leadership 见 CAIS website）。
- Status：501(c)(3) non-profit。
- Notable 2023 output：extinction risk statement，由数百名 researchers 和 CEOs 共同签署。原文：“Mitigating the risk of extinction from AI should be a global priority alongside other societal-scale risks such as pandemics and nuclear war.”
- 2026 outputs：AI Dashboard for frontier-model evaluation、Remote Labor Index（与 Scale AI）、Superintelligence Strategy Paper、AI Frontiers newsletter。

### Four-risk framework

CAIS 的 framework 将 catastrophic AI risk 分为四个 top-level categories：

1. **Malicious use**：bad actor 使用 AI 造成伤害（bioweapons synthesis、disinformation、cyberattacks）。
2. **AI races**：labs、companies 或 nations 之间的 competitive pressure 推动 deployment 越过安全点。
3. **Organizational risks**：internal lab dynamics（safety-culture failures、insufficient audit、under-resourced security）导致 bad deployment。
4. **Rogue AIs**：足够 capable 的 AI 追求与 human welfare 冲突的 goals。

这不是唯一 taxonomy；它是最常被引用的。categories 并不 mutually exclusive：一个在 race 中把 audit 换成速度的 organization 产出的 rogue AI，可以同时属于四类。

### Organizational risk 位于哪里

四个 categories 中，organizational risk 对 practitioners 最 actionable。一个 lab 的 safety culture、audit rigor、defense layering 和 information security，会决定其模型交付时是否真的带着 Lessons 10-18 中的 controls，还是那些 controls 只是没人验证过的 checklist items。

具体 organizational-risk levers：

- **Safety culture**：team members 是否能在不付出 career cost 的情况下 escalate concern？CAIS surveys 发现这是其他 levers 的强 predictor。
- **Rigorous audits**：external 和 internal。Internal-only audits 会产出 optimistic reports。
- **Multi-layered defenses**：没有单一 layer 足够（Phase 15 的 running theme）。
- **Information security**：model weights leaking、eval data leaking、monitor-bypass techniques leaking。Lesson 19 中的 RAND SL-4 是一个 specific standard。

### CAISI：Center for AI Standards and Innovation

- 在 NIST 内运行。
- 与 frontier labs 运行 voluntary agreements。
- 发布聚焦 cyber、bio 和 chemical-weapons risks 的 unclassified capability evaluations。
- 与 CAIS 不同；acronyms 碰撞；检查 URL（nist.gov）以确认你读的是哪一个。

CAISI 的角色是 METR private lab engagements（Lesson 21）的 public、government-facing counterpart。CAISI reports 是 unclassified；METR reports 经常是 NDA-gated。从业者同时阅读两者，会获得更完整图景。

### California SB-53

California Senate bill（2025-2026 session）处理 frontier models 的 catastrophic risk。draft 中的关键 provisions：

- specific capability thresholds，会触发 state-level obligations。
- AI lab employees 的 whistleblower protections。
- catastrophic failures 的 incident reporting requirements。

如果签署，它会成为美国第一个 state-level catastrophic-risk regulation。无论签署状态如何，该 bill 的 framing 都会塑造其他 state legislatures 如何处理这个问题。California practitioners 应该追踪 bill status；其他地区的 practitioners 也应该阅读它，以理解美国 state-level regulation 很可能长什么样。

### Societal-scale risk 不是 single-layer problem

Phase 15 的 running theme：defense in depth，也适用于 societal layer。没有单个 organization、regulation 或 framework 能关闭 catastrophic risk。只有在以下条件同时成立时，ecosystem 才会运转：

- Labs 交付 scaling policies（Lessons 19, 20）。
- External evaluators 产出 measurements（Lesson 21）。
- Civil society 跟踪并 publicize（CAIS）。
- Government 运行 voluntary programs 和 baseline regulation（CAISI、SB-53）。
- Practitioners 构建 multi-layered controls（Lessons 10-18）。

这是本阶段的最终 synthesis：前面每一课都是一个 stack 中的一层；stack 的 completeness 比任何单层 strength 更重要。

## 实际使用

`code/main.py` 实现一个小型 risk-inventory tool。给定 proposed deployment，它会根据 four-risk categories 标记 deployment，并返回 mitigation checklist。它是 framework 的 reading aid，不是 human judgment 的替代品。

## 交付成果

`outputs/skill-societal-risk-review.md` 会 review 一个 deployment 的 societal-scale-risk posture：它触及四个 categories 中哪些、有哪些 mitigations、organizational-risk exposure 是什么。

## 练习

1. 运行 `code/main.py`。输入三个不同 scales 的 synthetic deployments。确认 four-risk tags 符合你的预期；识别一个 tool under- 或 over-tags 的 case。

2. 完整阅读 CAIS four-risk paper。选择一个 risk category，并写两段说明你认为该 category 中最重要的 2026 development。

3. 阅读 California SB-53 当前 draft。识别一个你认为会 strengthen catastrophic-risk posture 的 provision，以及一个你认为会 weaken 的 provision。分别辩护。

4. 选择一个你知道的 production AI deployment（自己的或公开的）。按 organizational-risk sub-levers 给它打分：safety culture、audit rigor、multi-layered defenses、information security。哪个最弱？把它补到同等水平需要什么成本？

5. 勾勒一个反映额外一年 capability 和额外一年 deployment experience 的 2028 版 four-risk framework。你会添加、删除或重组什么？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|---|---|---|
| CAIS | “Center for AI Safety” | Non-profit；four-risk framework；2023 extinction statement |
| CAISI | “US government AI safety” | NIST Center；voluntary agreements；unclassified evals |
| Four-risk framework | “CAIS's taxonomy” | malicious use、AI races、organizational risks、rogue AIs |
| Malicious use | “Bad actor uses AI” | Bioweapons、disinformation、cyberattacks |
| AI races | “Competitive pressure” | Labs/companies/nations 推动 deployment 越过 safety |
| Organizational risk | “Lab internal failure” | Safety culture、audit、defenses、infosec |
| Rogue AI | “Misaligned agent” | capable AI 追求与 human welfare 冲突的 goals |
| California SB-53 | “State-level regulation” | 2025-2026 bill；如果签署，将成为美国第一个 state catastrophic-risk regulation |

## 延伸阅读

- [Center for AI Safety](https://safe.ai/) — four-risk framework 的 institutional home。
- [CAIS — AI Risks that Could Lead to Catastrophe](https://safe.ai/ai-risk) — four-risk paper。
- [CAIS — May 2023 statement on extinction risk](https://safe.ai/statement-on-ai-risk) — 简短 joint statement。
- [NIST CAISI](https://www.nist.gov/caisi) — 面向政府的 AI standards and innovation center。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 将 lab-level commitments 连接到 societal-scale framing。
