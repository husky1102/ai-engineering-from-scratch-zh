# 文档与图表理解

> 文档不是照片。PDF、科学论文、发票或手写表单都有 layout、tables、diagrams、footnotes、headers 和 semantic structure，而普通图像理解无法捕捉这些内容。VLM 之前的 stack 是一条 pipeline：Tesseract OCR + LayoutLMv3 + table-extraction heuristics。VLM 浪潮用 OCR-free models 取代了它：Donut（2022）、Nougat（2023）、DocLLM（2023）可以直接输出 structured markup。到 2026 年，frontier 几乎就是“把 page image 以 2576px native 喂给 Claude Opus 4.7”，structured-markup output 自然就能得到。本课会走读 document AI 的三个时代。

**类型:** Build
**语言:** Python（stdlib，layout-aware document parser skeleton）
**先修:** Phase 12 · 05（LLaVA），Phase 5（NLP）
**时间:** ~180 分钟

## 学习目标

- 解释 document AI 的三个时代：OCR pipeline、OCR-free、VLM-native。
- 描述 LayoutLMv3 的三条输入流：text、layout（bbox）、image patches，以及 unified masking。
- 比较 Donut（OCR-free，image → markup）、Nougat（scientific paper → LaTeX）、DocLLM（layout-aware generative）、PaliGemma 2（VLM-native）。
- 为新任务选择文档模型（invoices、scientific papers、handwritten forms、Chinese receipts）。

## 要解决的问题

“Understand this PDF”看似简单，其实很难。信息存在于：

- Text content（90% 的信号）。
- Layout（headers、footnotes、sidebars、two-column format）。
- Tables（rows、columns、merged cells）。
- Figures and diagrams。
- Handwritten annotations。
- Fonts and typography（title vs body）。

原始 OCR 会倾倒文本，并丢掉其余部分。一个关心发票的系统需要知道“Total: $1,245”来自右下角，而不是来自脚注。

## 核心概念

### 时代 1：OCR pipeline（2021 年前）

经典 stack：

1. PDF → 每页 image。
2. Tesseract（或商业 OCR）抽取文本，并附带每个词的 bounding box。
3. Layout analyzer 识别 blocks（header、table、paragraph）。
4. Table structure recognizer 解析 tables。
5. Domain rules + regex 抽取字段。

它适合干净的印刷文本。遇到手写、倾斜扫描、复杂表格、非英语文字就会崩。每种失败模式都需要一条自定义 exception path。

### TrOCR（2021）

TrOCR（Li et al., arXiv:2109.10282）用 transformer encoder-decoder 取代 Tesseract 经典 CNN-CTC，并在合成 + 真实 text images 上训练。它在手写和多语言文本上明显胜出。它仍然是一条 pipeline（detector 再 TrOCR 再 layout），但 OCR 这一步大幅改善。

### 时代 2：OCR-free（2022-2023）

第一批 OCR-free 模型说：完全跳过 detection，把 image pixels 直接映射到 structured output。

Donut（Kim et al., arXiv:2111.15664）：
- Encoder-decoder transformer，encoder 是 Swin-B。
- 输出可以是 form understanding 的 JSON、summarization 的 markdown，或任何 task-specific schema。
- 没有 OCR，没有 layout，没有 detection。

Nougat（Blecher et al., arXiv:2308.13418）：
- 专门在科学论文上训练。
- 输出 LaTeX / markdown。
- 处理 equations、multi-column layout、figures。
- 每个 arXiv-parser 都会调用的模型。

这些是 specialist，不是 generalist。Donut 处理科学论文会失败；Nougat 处理发票会失败。

### LayoutLMv3（2022）

另一条路线。LayoutLMv3（Huang et al., arXiv:2204.08387）保留 OCR，但加入 layout understanding：

- 三条输入流：OCR text tokens、每个 token 的 2D bounding boxes、image patches。
- 跨全部三种 modality 的 masked training objective（masked text、masked patches、masked layout）。
- Downstream：classification、entity extraction、table QA。

LayoutLMv3 是基于 OCR 的 document understanding 的顶峰。在 forms 与 invoices 上很强。需要上游 OCR。在标准化文档 benchmark 上是 VLM 之前的最佳准确率。

### DocLLM（2023）

DocLLM（Wang et al., arXiv:2401.00908）是 LayoutLM 的 generative sibling。它基于 layout tokens 生成 free-form answers。更适合文档 QA；仍然依赖 OCR input。

### 时代 3：VLM-native（2024+）

2024 年，VLM 已经足够好，可以完全取代 pipeline。把完整页面图像以高分辨率喂给 VLM，提出问题，得到答案。

- LLaVA-NeXT 336-tile AnyRes 适合小型文档。
- Qwen2.5-VL dynamic-resolution 原生处理 2048+ pixels。
- Claude Opus 4.7 支持 2576px documents。
- PaliGemma 2（2025 年 4 月）专门针对 documents + handwriting 训练。

VLM-native 与 OCR-pipeline 的差距迅速缩小。到 2026 年，VLM-native 在这些任务上胜出：

- Scene text（手写 + 印刷，混合文字）。
- 含 merged cells 的复杂 tables。
- 嵌入文本中的 math equations。
- 带文字标注的 figures。

OCR pipelines 仍然胜出的场景：

- 超大规模纯扫描 workload，其中 per-page latency 很关键。
- Pipeline reliability（确定性失败 vs VLM hallucinations）。
- 需要可审计 OCR output 的 regulated environments。

### Claude 4.7 / GPT-5 frontier

在 2576-pixel native input 下，frontier VLM 能以接近人类准确率做 document understanding。2026 年初的 benchmark 数字：

- DocVQA：Claude 4.7 约 95.1，PaliGemma 2 约 88.4，Nougat 约 77.3，pipelined LayoutLMv3 约 83。
- ChartQA：Claude 4.7 约 92.2，GPT-4V 约 78。
- VisualMRC：Claude 4.7 约 94。

闭源模型差距主要来自 resolution 与 base-LLM scale。开放 7B 模型落后几分，但正在追赶。

### 数学公式与 LaTeX output

科学论文需要方程的精确 LaTeX output。Nougat 就是在这上面训练的。使用 LaTeX targets 训练的 VLM（Qwen2.5-VL-Math、Nougat derivatives）能生成可用 LaTeX。没有显式 LaTeX 训练时，VLM 会产出可读但不精确的转录。

2026 年的科学论文 pipeline：先在 PDF 上跑 Nougat，再对 tricky pages 用 VLM。

### 手写

这仍然是最难的子任务。印刷 + 手写混合（医生笔记、填写过的表单）是 OCR pipeline 在成本上仍然胜过 VLM 的地方。Handwritten-only VLM 正在进步（Claude 4.7、PaliGemma 2）。

### 2026 年配方

对于新的 document-AI 项目：

- 大规模纯印刷发票：LayoutLMv3 + rules，成本高效。
- 混合文档（scientific + handwritten + forms）：VLM-native（PaliGemma 2 或 Qwen2.5-VL）。
- 完整 arXiv ingestion：Nougat 处理数学，VLM 处理 figures。
- Regulatory：OCR pipeline + VLM validator 做交叉检查。

## 实际使用

`code/main.py`：

- 一个 toy layout-aware tokenizer：给定 (text, bbox) pairs，生成 LayoutLMv3-style input。
- 一个 Donut-style task schema generator：为表单生成 JSON template。
- 比较 OCR-pipeline、Donut、Nougat 与 VLM-native 每页的 token budgets。

## 交付成果

本课产出 `outputs/skill-document-ai-stack-picker.md`。给定一个 document-AI 项目（domain、scale、quality、regulatory），它会在 OCR pipeline、OCR-free specialist 与 VLM-native 之间选择。

## 练习

1. 你的项目每天处理 1000 万张发票。哪个 stack 能在不损失准确率的前提下最小化 cost-per-page？

2. 为什么 LayoutLMv3 在 form QA 上超过 pure-CLIP-VLM，但在 scene-text 上落后？bbox stream 放弃了什么？

3. Nougat 生成 LaTeX。提出一个 VLM-native output 在 LaTeX fidelity 上超过 Nougat 的测试用例，以及一个 Nougat 胜出的用例。

4. 阅读 PaliGemma 2 论文（Google，2024）。相比 PaliGemma 1，提升文档准确率的关键 training-data addition 是什么？

5. 设计一个 regulatory-safe hybrid：OCR pipeline 作为 primary，VLM 作为 secondary cross-check。你如何解决二者分歧？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| OCR pipeline | “Tesseract-style” | 分阶段 stack：detect -> OCR -> layout -> rules；确定性强但脆弱 |
| OCR-free | “Donut-style” | 跳过显式 OCR 的 image-to-output transformer；单模型 |
| Layout-aware | “LayoutLM” | 输入包含每个 token 的 bbox coordinates；跨 modality 做 unified masking |
| VLM-native | “Frontier VLM” | 直接把 page image 以高分辨率喂给 Claude/GPT/Qwen VLM；无 pipeline |
| DocVQA | “Doc benchmark” | Document VQA 标准；最常被引用的分数 |
| Markup output | “LaTeX / MD” | 结构化输出格式，而不是 free-form text；支持 downstream automation |

## 延伸阅读

- [Li et al. — TrOCR (arXiv:2109.10282)](https://arxiv.org/abs/2109.10282)
- [Blecher et al. — Nougat (arXiv:2308.13418)](https://arxiv.org/abs/2308.13418)
- [Huang et al. — LayoutLMv3 (arXiv:2204.08387)](https://arxiv.org/abs/2204.08387)
- [Kim et al. — Donut (arXiv:2111.15664)](https://arxiv.org/abs/2111.15664)
- [Wang et al. — DocLLM (arXiv:2401.00908)](https://arxiv.org/abs/2401.00908)
