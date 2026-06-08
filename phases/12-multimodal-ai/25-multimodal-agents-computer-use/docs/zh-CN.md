# 多模态 Agent 与 Computer-Use（Capstone）

> 2026 年的前沿产品是多模态 agent：它能阅读截图、点击按钮、导航网页 UI、填写表单，并端到端完成工作流。SeeClick 和 CogAgent（2024）证明了 GUI-grounding primitive。Ferret-UI 加入移动端。ChartAgent 引入面向图表的 visual tool-use。VisualWebArena 和 AgentVista（2026）是前沿模型追逐的 benchmark，而即使 Gemini 3 Pro 和 Claude Opus 4.7 在 AgentVista 的 hard tasks 上也只有约 30%。这个 capstone 汇总 Phase 12 的全部线索：perception（高分辨率 VLM）、reasoning（带 tool use 的 LLM）、grounding（coordinate output）、long-horizon memory 和 evaluation。

**类型:** Capstone
**语言:** Python（stdlib，action schema + agent loop skeleton）
**先修:** Phase 12 · 05（LLaVA），Phase 12 · 09（Qwen-VL JSON），Phase 14（Agent Engineering）
**时间:** ~240 分钟

## 学习目标

- 设计一个多模态 agent loop：perceive → reason → act → observe → repeat。
- 构建 GUI grounding output schema（click coordinates、type text、scroll、drag），使 VLM 可以以 JSON 形式输出。
- 比较 screenshot-only agents、accessibility-tree agents 与 hybrid agents。
- 在一个小型 VisualWebArena slice 上设置多模态 agent benchmark evaluation。

## 要解决的问题

一个订票网站工作流：“find me a flight to Tokyo for April 15, aisle seat under $800, book it.”

多模态 agent 需要：

1. 获取浏览器截图。
2. 把截图 + URL + goal 解析成 plan。
3. 发出结构化 action：click（at x,y）、type “Tokyo”（at element E）、scroll down、select（radio button）。
4. 把 action 应用到浏览器。
5. 观察新状态（下一张截图）。
6. 重复直到任务完成。

每一步都是一个多模态 VLM 调用。VLM 输出必须是可解析 JSON。错误会跨步骤累积，所以 recovery 很重要。

## 核心概念

### GUI grounding：primitive

GUI grounding 是：给定一张截图和一条自然语言指令，输出要点击的 `(x, y)` 坐标（或其他 action）。

SeeClick（arXiv:2401.10935）是第一个大规模开放结果：在 synthetic + real GUI data 上 fine-tune VLM，以 plain text token 输出坐标。可用。

CogAgent（arXiv:2312.08914）加入 1120x1120 高分辨率编码，用于 dense UIs。分数：web navigation 上约 84%。

Ferret-UI（arXiv:2404.05719）聚焦 mobile UIs，并与 iOS accessibility data 集成。

输出格式通常是 JSON：

```json
{"action": "click", "x": 384, "y": 220, "element_desc": "Search button"}
```

`element_desc` 帮助 recovery：如果截图之间坐标漂移，semantic hint 让系统可以重新 ground。

### Action schemas

典型 action schema 有 6-10 种 action types：

- `click`: (x, y)
- `type`: (text, x?, y?)
- `scroll`: (direction, amount)
- `drag`: (x0, y0, x1, y1)
- `select`: (option_index)
- `hover`: (x, y)
- `navigate`: (url)
- `wait`: (ms)
- `done`: (success, explanation)

Agent 每步发出一个 action。Browser wrapper 执行并返回新状态。

### Screenshot-only vs accessibility-tree

两种输入模式：

- Screenshot-only：完整图像，无结构信息。最通用，适用于任何 app。
- Accessibility tree：结构化 DOM / iOS accessibility info。grounding 可靠得多；适用于 tree 可用的地方。
- Hybrid：两者都用，tree 作为原子 action 的可靠 grounder，screenshot 作为语义上下文。

生产 agent 在可行时使用 hybrid。浏览器自动化（Selenium + accessibility）总是有 tree；桌面 app 有时有。

### Long-horizon memory

20 步工作流会产生 20 张截图。VLM context 很快填满。三种压缩策略：

- Summary-chain：每 5 步总结已经发生的事情，丢弃旧截图。
- Skip-frame：保留第一张、最后一张，以及每第 3 张截图。
- Tool-recorded log：执行 action，保留已做事项的文本 log；不再重新查看旧截图。

Claude 的 computer-use API 使用 log pattern。更简单，也更可靠。

### Visual tool use

ChartAgent（arXiv:2510.04514）为图表理解引入 visual tool use：crop、zoom、OCR、调用外部 detection。Agent 可以输出“crop to region (100, 200, 300, 400) then call OCR”作为 tool call。Tool 返回文本；VLM 继续推理。

这个模式可以泛化：set-of-mark prompting、region annotation 和 external detection tools 都适合同一个“输出 tool call，接收 structured response”的 schema。

### 2026 benchmarks

- ScreenSpot-Pro。约 1k web screenshots 上的 GUI grounding。开放 SOTA Qwen2.5-VL-72B 约 85%。Frontier 约 90%。
- VisualWebArena。端到端 web tasks（shop、forum、classifieds）。开放 SOTA 约 20%。Gemini 3 Pro 约 27%。
- AgentVista（arXiv:2602.23166）。2026 年最难 benchmark。横跨 12 个领域的真实工作流。Frontier models 得分 27-40%；open models 10-20%。
- WebArena / WebShop。更老的 benchmark；已被 frontier 饱和。

### 为什么仍然很难

Agent performance bottlenecks：

1. 细尺度 visual grounding。“Click the small X”在移动分辨率下经常失败。
2. Long-horizon planning。10 个 action 后，agent 会偏离目标。
3. Error recovery。点击失败（错按钮）时，检测 + recovery 很少出现在训练数据里。
4. Cross-page context。在 tab 间跳转或长表单中会丢失 state。

研究方向：memory architectures、explicit replanning、multimodal verification（用 screenshot match 验证 action success）。

### Capstone build-it

Capstone 任务：构建一个 computer-use agent，它会：

1. 读取订票网站 mock page 的 HTML + screenshot。
2. 规划多步序列：search → select → fill form → submit。
3. 发出匹配 action schema 的 JSON actions。
4. 在固定 10-task slice 上 evaluation。

本课提供 scaffold code，很容易扩展成真实浏览器。

## 实际使用

`code/main.py` 是 capstone scaffold：

- Action schema JSON definition（10 actions）。
- 作为 dict 的 mock browser state。
- Agent loop skeleton：接收 state，发出 action，应用，循环。
- 10-task mini-benchmark（synthetic pages），用于测量 end-to-end success rate。
- action 失败时的 error-recovery hook。

## 交付成果

本课产出 `outputs/skill-multimodal-agent-designer.md`。给定一个 computer-use product（domain、action set、evaluation target），它会设计完整 agent loop、memory strategy、grounding mode 和预期 benchmark score。

## 练习

1. 用 `screenshot_region` tool（crop + zoom）扩展 action schema。哪些任务会受益？

2. 阅读 AgentVista（arXiv:2602.23166）。描述最难的 task category，以及为什么 frontier models 仍会失败。

3. Long-horizon memory compression：设计一个 summary-chain，live 保留 ≤4 张 screenshot，可记录任意数量 log。

4. 构建 error-recovery hook：action failure（button not found）时，agent 下一步做什么？

5. 在 10 个 web tasks 上比较 screenshot-only Claude 4.7 与 hybrid screenshot + accessibility-tree Qwen2.5-VL。哪些任务谁赢？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| GUI grounding | “Click coordinates” | 模型为截图上的指令目标输出 (x,y) |
| Action schema | “Tool definitions” | 有效 actions（click、type、scroll、drag）的 JSON 描述 |
| Accessibility tree | “Structured DOM” | 来自 browser/iOS API 的机器可读 UI hierarchy |
| Hybrid agent | “Screenshot + tree” | 同时使用图像和结构化信息；比任一单独方式更可靠 |
| Visual tool use | “Zoom/crop/detect” | Agent 在 plan 中途调用外部 vision tools（OCR、detection） |
| Summary-chain | “Memory compression” | 用周期性文本总结替代长 screenshot history |
| VisualWebArena | “E2E web bench” | 2024 年端到端 web tasks benchmark |
| AgentVista | “2026 hard bench” | 12-domain 真实工作流；即使 Gemini 3 Pro 也约 30% |

## 延伸阅读

- [Cheng et al. — SeeClick (arXiv:2401.10935)](https://arxiv.org/abs/2401.10935)
- [Hong et al. — CogAgent (arXiv:2312.08914)](https://arxiv.org/abs/2312.08914)
- [You et al. — Ferret-UI (arXiv:2404.05719)](https://arxiv.org/abs/2404.05719)
- [ChartAgent (arXiv:2510.04514)](https://arxiv.org/abs/2510.04514)
- [Koh et al. — VisualWebArena (arXiv:2401.13649)](https://arxiv.org/abs/2401.13649)
- [AgentVista (arXiv:2602.23166)](https://arxiv.org/abs/2602.23166)
