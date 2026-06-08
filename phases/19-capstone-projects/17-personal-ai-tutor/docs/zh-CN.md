# 综合项目 17 — Personal AI Tutor（自适应、多模态、带记忆）

> Khanmigo（Khan Academy）、Duolingo Max、Google LearnLM / Gemini for Education、Quizlet Q-Chat 和 Synthesis Tutor 都在 2026 年规模化发布了 adaptive multimodal tutoring。共同形态是 Socratic policy（绝不直接倾倒答案）、每次交互后更新的 learner model（Bayesian knowledge tracing 风格）、voice + text + photo-math input、curriculum graph retrieval、spaced-repetition scheduling，以及面向年龄适宜内容的硬 safety filters。这个综合项目要发布一个 subject-specific tutor（K-12 algebra 或 intro Python），用 10 名学习者运行两周 efficacy study，并通过 content-safety audit。

**类型:** Capstone
**语言:** Python（backend，learner model），TypeScript（web app），SQL（curriculum graph via Postgres + Neo4j）
**先修:** Phase 5（NLP），Phase 6（speech），Phase 11（LLM engineering），Phase 12（multimodal），Phase 14（agents），Phase 17（infrastructure），Phase 18（safety）
**覆盖阶段:** P5 · P6 · P11 · P12 · P14 · P17 · P18
**时间:** 30 小时

## 要解决的问题

Adaptive tutoring 曾经是 ed-tech research niche。到 2026 年，它已经是 consumer product。Khanmigo 部署到了美国多数学区。Duolingo Max 达到数千万 MAUs。Google 的 LearnLM / Gemini for Education 为 Google Classroom 中的 tutoring 提供能力。Quizlet Q-Chat 与 flashcards 并列。Synthesis Tutor 作为 tutor-for-curious-kids 爆红。共同要素包括：multimodal input（type、speak、photograph equations）、Socratic pedagogy（先提问，后解释）、每次交互后更新的 learner model，以及严格的 age-appropriate safety。

你将为一个具体 cohort 构建其中一种 tutor。测量标准是真实 efficacy study：10 名学习者两周内的 pre-test 和 post-test scores。Voice loop 必须感觉自然（综合项目 03 的 sub-stack）。Memory 必须尊重 privacy。Safety filter 必须通过面向 K-12 的 COPPA-aware red-team。

## 核心概念

四个组件。**Tutor policy** 是 Socratic loop：当学习者索要答案时，policy 会问一个 leading question；当他们答对时，进入下一个 concept；当他们卡住时，给出 scaffolded hint。**Learner model** 是 Bayesian knowledge tracing（或一个简单变体），会在每次交互后更新每个 curriculum node 的 mastery probability。**Curriculum graph** 是带 prerequisite edges 的 Neo4j concepts；policy 遍历图来选择下一个 concept。**Memory** 是 episodic + semantic store（agentmemory 风格），保存过往 interactions、mistakes 和 preferences。

UX 是多模态的。Text input 用于 typed answers。Voice input 通过 LiveKit + Whisper（复用综合项目 03）。Photo input 通过 dots.ocr 或 PaliGemma 2 处理 math problems。Voice output 通过 Cartesia Sonic-2。Safety 使用 Llama Guard 4 加 age-appropriate filter（阻止 adult content、violence、self-harm），并使用 COPPA-aware memory retention policy。

Efficacy study 是交付物。10 名学习者，pre-test 和 post-test，两周。报告 learning gain delta 和 confidence interval。与 non-adaptive baseline（同样内容线性呈现，不使用 tutor policy）对比。

## 架构

```text
learner device
  |
  +-- text         -> web app
  +-- voice        -> LiveKit Agents (ASR + TTS)
  +-- photo math   -> dots.ocr / PaliGemma 2
       |
       v
  tutor policy (LangGraph)
       - Socratic decision head
       - next-concept chooser (curriculum graph walk)
       - hint scaffolder
       - mastery update
       |
       v
  learner model (BKT / item-response theory)
       - per-concept mastery probability
       - spaced-repetition scheduler (SM-2 or FSRS)
       |
       v
  memory (agentmemory-style)
       - episodic: every interaction
       - semantic: learned mistakes, preferences
       - retention policy: COPPA / GDPR aware
       |
       v
  curriculum graph (Neo4j)
       - prerequisite edges
       - OER content attached
       |
       v
  safety:
    Llama Guard 4 + age-appropriate filter
    memory access guarded by learner ID scope
```

## 技术栈

- Subject choice：K-12 algebra 或 intro Python（选一个深入）
- Tutor policy：LangGraph over Claude Sonnet 4.7（with prompt caching）
- Learner model：Bayesian knowledge tracing（classic）或用于 spacing 的 FSRS
- Curriculum graph：概念 + prerequisite edges + OER content 的 Neo4j
- Memory：agentmemory-style persistent vector + episodic + semantic store
- Voice：LiveKit Agents 1.0 + Cartesia Sonic-2（复用综合项目 03 sub-stack）
- Photo math：dots.ocr 或 PaliGemma 2，用于 equation recognition
- Safety：Llama Guard 4 + custom age-appropriate filter
- Eval：Bloom-level question generation、pre/post test harness、efficacy study tooling

## 动手实现

1. **Curriculum graph。** 构建一个包含 50-150 个 concept nodes 的 Neo4j（例如 K-12 algebra，从 “number line” 到 “quadratic formula”），带 prerequisite edges。给每个 node 附上 OER content（Open Textbook、OpenStax）。

2. **Learner model。** 用 priors 初始化 Bayesian knowledge tracing：guess、slip、learn-rate。每次交互后更新 per-concept mastery。按 learner 持久化。

3. **Tutor policy。** LangGraph 节点：`read_signal`（学习者答案是 correct / partial / stuck?）、`select_concept`（遍历 curriculum graph，选择最高优先级 concept）、`scaffold`（Socratic prompt）、`update_mastery`。

4. **Memory。** 每次交互写入 episodic store。Mistakes 和 preferences 提升为 semantic memory。COPPA-aware retention policy：1 年后自动删除，parent-accessible。

5. **Voice path。** LiveKit Agents worker 接入 tutor policy。ASR 使用 Whisper-v3-turbo。TTS 使用 Cartesia Sonic-2。支持 barge-in（复用综合项目 03 mechanics）。

6. **Photo-math path。** 上传或拍摄 image；运行 dots.ocr 或 PaliGemma 2 识别 equation；作为 structured input 送入 tutor。

7. **Safety。** 每个 model output 都经过 Llama Guard 4 + age-appropriate filter（阻止 self-harm、adult content、violence）。Memory access 按 learner ID 做 scoping；提供 parental access surface 以删除数据。

8. **Efficacy study。** 10 名学习者，pre-test（标准化 30-question baseline），两周 tutor interaction（每周 3 次 session），post-test。与 10 名学习者的 non-adaptive baseline cohort 在相同内容上对比。

9. **Weekly progress reports。** 对每个 learner，自动生成 PDF summary，包含 topics explored、mastery trajectories 和 recommended next steps。

## 实际使用

```text
learner: "I don't understand why 3x + 6 = 12 means x = 2"
[signal]   stuck
[concept]  'isolating variables' (prerequisite: addition-subtraction-equality)
[scaffold] "what number would you subtract from both sides to start?"
learner: "6"
[signal]   correct
[mastery]  addition-subtraction-equality: 0.62 -> 0.77
[concept]  continue 'isolating variables'
[scaffold] "great. now what is 3x / 3 equal to?"
```

## 交付成果

`outputs/skill-ai-tutor.md` 是交付物。一个 subject-specific adaptive tutor，具备 multimodal input、learner model、memory、safety 和经过测量的 efficacy。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | Learning gain delta | 10-learner 两周 study 中的 pre/post-test delta |
| 20 | Socratic fidelity | transcript samples 上的 rubric score |
| 20 | Multimodal UX | Voice + photo + text coherence end to end |
| 20 | Safety + privacy posture | Llama Guard 4 pass rate + COPPA-aware retention |
| 15 | Curriculum breadth and graph quality | Concept coverage + prerequisite graph consistency |
| **100** | | |

## 练习

1. 在有和没有 adaptive learner model（random concept order）的情况下运行 efficacy study。报告 delta。预期 adaptive 会赢，但大小才是有趣数字。

2. 添加 multimodal probe：同一个 concept question 分别以 text、voice 和 photo 呈现。测量学习者是否在偏好的 modality 下更快收敛。

3. 构建 parent dashboard：topics practiced、mastery trajectories、upcoming concepts、safety events（任何 guardrail hits）。COPPA-aligned。

4. 添加 language-switch mode：tutor 接受 Spanish input，并用 Spanish 教学。测量 X-Guard coverage。

5. 压测 memory privacy：验证 learner A 即使通过 voice-clip re-ingest attack，也不能看到 learner B 的数据。记录 attempted access 并告警。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| Socratic policy | “Ask, do not dump” | Tutor 问 leading question，而不是直接给答案 |
| Bayesian knowledge tracing | “BKT” | 用于每个 concept mastery probability 的经典 learner-model equations |
| FSRS | “Free Spaced Repetition Scheduler” | 2024 spaced-repetition scheduler，优于 SM-2 |
| Curriculum graph | “Concept DAG” | 带 prerequisite edges 的 concepts Neo4j |
| Episodic memory | “Per-interaction log” | 每次交互都会存储，供之后检索 |
| Semantic memory | “Learned pattern store” | 从 episodic 压缩提升出的 mistakes 和 preferences |
| COPPA | “Kids privacy law” | 限制收集 13 岁以下儿童数据的美国法律 |

## 延伸阅读

- [Khanmigo (Khan Academy)](https://www.khanmigo.ai) — reference consumer K-12 tutor
- [Duolingo Max](https://blog.duolingo.com/duolingo-max/) — reference language-learning tutor
- [Google LearnLM / Gemini for Education](https://blog.google/technology/google-deepmind/learnlm) — hosted reference model
- [Quizlet Q-Chat](https://quizlet.com) — alternate reference
- [Synthesis Tutor](https://www.synthesis.com) — startup reference
- [FSRS algorithm](https://github.com/open-spaced-repetition/fsrs4anki) — spaced-repetition scheduler
- [Bayesian Knowledge Tracing](https://en.wikipedia.org/wiki/Bayesian_knowledge_tracing) — learner-model classic
- [LiveKit Agents](https://github.com/livekit/agents) — voice stack
