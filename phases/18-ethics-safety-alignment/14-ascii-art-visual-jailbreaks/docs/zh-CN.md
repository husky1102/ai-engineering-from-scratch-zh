# ASCII Art 和视觉越狱

> Jiang、Xu、Niu、Xiang、Ramasubramanian、Li、Poovendran，“ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs”（ACL 2024，arXiv:2402.11753）。在 harmful request 中遮蔽与安全相关的 tokens，用相同字母的 ASCII-art 渲染替换它们，再发送这个伪装 prompt。GPT-3.5、GPT-4、Gemini、Claude、Llama-2 都无法稳定识别 ASCII-art tokens。该攻击绕过 PPL（perplexity filters）、Paraphrase defenses 和 Retokenization。相关方向：ViTC benchmark 衡量对 non-semantic visual prompts 的识别；StructuralSleight 将其泛化到 Uncommon Text-Encoded Structures（trees、graphs、nested JSON），作为一类 encoding attacks。

**类型:** Build
**语言:** Python (stdlib, ArtPrompt token-masking harness)
**先修:** Phase 18 · 12 (PAIR), Phase 18 · 13 (MSJ)
**时间:** ~60 minutes

## 学习目标

- 描述 ArtPrompt attack：word-identification step、ASCII-art substitution、final cloaked prompt。
- 解释为什么标准防御（PPL、Paraphrase、Retokenization）会在 ArtPrompt 上失败。
- 定义 ViTC，并描述它衡量的内容。
- 将 StructuralSleight 描述为对任意 Uncommon Text-Encoded Structures 的泛化。

## 要解决的问题

通过 paraphrase 和 roleplay 的攻击（第 12 课）以及通过 long context 的攻击（第 13 课）作用在 text-level pattern 上。ArtPrompt 作用在 recognition level：模型没有解析 forbidden token。它解析的是由字符渲染出的图像。Safety filter 看到的是无害 punctuation。模型看到的是一个词。

## 核心概念

### ArtPrompt，两步

Step 1. Word Identification。给定 harmful request，攻击者使用 LLM 识别与安全相关的词（例如 “how to make a bomb” 中的 “bomb”）。

Step 2. Cloaked Prompt Generation。用每个被识别词的 ASCII-art 渲染替换它（由字符组成字母形状的 7x5 或 7x7 block）。模型收到的是一组 punctuation 和 spaces，足够强的模型可以把它识别为单词；safety filter 只看到网格。

结果：GPT-4、Gemini、Claude、Llama-2、GPT-3.5 都会失败。在其 benchmark subset 上，attack success rate 超过 75%。

### 为什么标准防御会失败

- **PPL（perplexity filter）。** ASCII art 的 perplexity 很高，但所有新颖输入也是如此。能阻断 ArtPrompt 的 threshold 也会阻断合法的 structured input。
- **Paraphrase。** 对 prompt 做 paraphrase 会破坏 ASCII art。实践中，paraphrase LLM 往往会保留或重建这段 art。
- **Retokenization。** 以不同方式切分 tokens，并不会改变模型通过视觉识别 letter shapes 这一事实。

底层问题是：safety filters 位于 token level 或 semantic level；ArtPrompt 作用在 visual recognition level。

### ViTC benchmark

识别 non-semantic visual prompts。它衡量模型阅读 ASCII-art、wingdings 和其他 non-text-semantic visual content 的能力。ArtPrompt 的有效性与 ViTC accuracy 相关：模型越擅长阅读 visual text，ArtPrompt 在它身上越有效。这是 capability-safety tradeoff。

### StructuralSleight

泛化 ArtPrompt：Uncommon Text-Encoded Structures（UTES）。Trees、graphs、nested JSON、CSV-in-JSON、diff-style code blocks。如果某种结构在 training safety data 中罕见，但模型可以解析，它就能隐藏 harmful content。

防御含义：安全必须泛化到模型能够解析的各种 structured representations。这个集合很大，而且还在增长。

### Image-modality analog

Visual LLMs（GPT-5.2、Gemini 3 Pro、Claude Opus 4.5、Grok 4.1）扩展了 attack surface。带真实图像的 ArtPrompt-style attacks 比 ASCII-art analogs 更强，因为 image encoders 会产生更丰富的信号。

### 它在 Phase 18 中的位置

第 12-14 课描述三种正交攻击向量：iterative refinement（PAIR）、context length（MSJ）和 encoding（ArtPrompt/StructuralSleight）。第 15 课从 model-centric attacks 转向 system-boundary attacks（indirect prompt injection）。第 16 课描述防御 tooling response。

## 实际使用

`code/main.py` 构建一个 toy ArtPrompt。你可以用 ASCII-art glyphs 伪装 harmful query 中的特定 words，验证 cloaked string 能通过 keyword filter，并且（可选）用简单 recognizer 把 cloaked string 解码回来。

## 交付成果

本课产出 `outputs/skill-encoding-audit.md`。给定一份 jailbreak-defense report，它会枚举已覆盖的 encoding attack families（ASCII art、base64、leet-speak、UTF-8 homoglyph、UTES），以及捕获每一类攻击的 defense layer。

## 练习

1. 运行 `code/main.py`。验证 cloaked string 可以通过一个简单 keyword filter。报告所需的 character-level change。

2. 实现第二种 encoding：对同一个 target word 使用 base64。比较它与 ArtPrompt 的 filter-bypass rate 和 recovery difficulty。

3. 阅读 Jiang et al. 2024 Section 4.3（five-model results）。提出一个原因，解释为什么 Claude 在同一 benchmark 上的 ArtPrompt-resistance 高于 Gemini。

4. 设计一个 pre-generation defense，用于检测 prompt 中 ASCII-art-shaped regions。测量它在合法 code、tables 和 mathematical notation 上的 false-positive rate。

5. StructuralSleight 列出 10 种 encoding structures。草拟一个能处理全部 10 种结构的 generalized defense，并估计每个 defended prompt 的 compute cost。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| ArtPrompt | “the ASCII-art attack” | 用 ASCII-art renderings 遮蔽 safety words 的两步 jailbreak |
| Cloaking | “hide the word” | 将 forbidden token 替换成模型能读到但 filter 读不到的 visual representation |
| UTES | “uncommon structure” | Uncommon Text-Encoded Structure — tree、graph、nested JSON 等，用于 smuggle content |
| ViTC | “visual-text capability” | 衡量模型读取 non-semantic visual encoding 能力的 benchmark |
| Perplexity filter | “PPL defense” | 拒绝 high perplexity prompts；会失败，因为合法 structured input 也会高分 |
| Retokenization | “tokenizer shift defense” | 用不同 tokenizer 预处理 prompt；会失败，因为 recognition 是视觉层面的 |
| Homoglyph | “lookalike characters” | 看起来与 Latin letters 相同的 Unicode characters；可绕过 substring checks |

## 延伸阅读

- [Jiang et al. — ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753) — ASCII-art jailbreak paper
- [Li et al. — StructuralSleight (arXiv:2406.08754)](https://arxiv.org/abs/2406.08754) — UTES generalization
- [Chao et al. — PAIR (Lesson 12, arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — 互补的 iterative attack
- [Anil et al. — Many-shot Jailbreaking (Lesson 13)](https://www.anthropic.com/research/many-shot-jailbreaking) — 互补的 length attack
