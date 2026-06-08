# Open-Weight VLM Recipes：真正重要的是什么

> 2024-2026 年的 open-weight VLM 文献是一片 ablation table 森林。Apple 的 MM1 测试了 image encoder、connector 和 data mix 的 13 种组合。Allen AI 的 Molmo 证明详细人工 caption 胜过 GPT-4V distillation。Cambrian-1 跑了 20 多个 encoder 对比。Idefics2 形式化了五轴 design space。Prismatic VLMs 在受控 benchmark 上比较了 27 种训练 recipe。在所有噪音之中，有一小组结果跨论文成立：image encoder 比 connector architecture 更重要，data mixture 比二者都更重要，而详细人工 caption 在相同 token count 下胜过 distilled synthetic data。本课阅读这些表，这样你就不用自己读了。

**类型:** Learn + lab
**语言:** Python（stdlib，ablation table parser + recipe picker）
**先修:** Phase 12 · 05（LLaVA baseline）
**时间:** ~180 分钟

## 学习目标

- 说出五轴 VLM design space：image encoder、connector、LLM、data mix、resolution schedule。
- 阅读 MM1 / Idefics2 / Cambrian-1 ablation table，并预测哪个旋钮会移动某个 benchmark。
- 给定 compute budget 和 task mix，为新 VLM 选择 recipe（encoder、connector、data、resolution）。
- 解释为什么详细人工 caption 在相同 token count 下胜过 GPT-4V distillation。

## 要解决的问题

已经存在数百个 open-weight VLM。“好”和“state-of-the-art”之间的大部分差距不在 architecture，而在 data、resolution schedule 和 encoder choice。知道模型表现不佳时先转哪个旋钮，可以省掉一次 500 万 GPU-hour 的错误。

2023 年浪潮（LLaVA-1.5、InstructBLIP、MiniGPT-4）依赖 caption-pair pretraining + LLaVA-Instruct-150k。不错的 baseline。MMMU 大约封顶在 35%。

2024 年浪潮（MM1、Idefics2、Molmo、Cambrian-1、Prismatic VLMs）进行了穷尽式 ablation。结果既意外又实用。

## 核心概念

### 五轴 design space

Idefics2（Laurençon et al., 2024）命名了这些轴：

1. Image encoder。CLIP ViT-L/14、SigLIP SO400m/14、DINOv2 ViT-g/14、InternViT-6B。Encoder 在 patch size、resolution 和 pretraining objective 上不同。
2. Connector。MLP（2-4 层）、Q-Former（32 queries + cross-attn）、Perceiver Resampler（64 queries）、C-Abstractor（convolutional + bilinear pooling）。
3. Language model。Llama-3 8B / 70B、Mistral 7B、Phi-3、Gemma-2、Qwen2.5。LLM size 是主要参数成本。
4. Training data。Caption pairs（CC3M、LAION）、interleaved（OBELICS、MMC4）、instruction（LLaVA-Instruct、ShareGPT4V、PixMo、Cauldron）。
5. Resolution schedule。Fixed 224/336/448、AnyRes、native dynamic。训练中 ramp 或保持 constant。

每个生产 VLM 都在每个轴上做选择。MMMU 分数的大部分方差由轴 1、4、5 解释，而不是由你选择了哪个 connector 解释。

### Axis 1：encoder > connector

MM1 Section 3.2 显示：从 CLIP ViT-L/14 换到 SigLIP SO400m/14，会让 MMMU 增加 3+ 分。把 connector 从 MLP 换到 Perceiver Resampler，增加不到 1 分。Idefics2 复现了这一点：同样 token count 下，SigLIP > CLIP，Q-Former ≈ MLP ≈ Perceiver。

Cambrian-1 的“Cambrian Vision Encoders Match-Up”（Tong et al., 2024）在 vision-centric benchmark（CV-Bench）上跑了 20 多个 encoder。leaderboard 顶部混合了 DINOv2 和 SigLIP；CLIP 位于中游；ImageBind 和 ViT-MAE 更低。CLIP ViT-L 到 DINOv2 ViT-g/14 的差距在 CV-Bench 上约 5-7 分。

2026 年开放 VLM 的默认 encoder 是 SigLIP 2 SO400m/14，用于 semantic + dense features；如果需要 segmentation/grounding，有时会拼接 DINOv2 ViT-g/14 features（Cambrian 的“Spatial Vision Aggregator”这样做）。

### Axis 2：connector design 基本打平

MM1、Idefics2、Prismatic 和 MM-Interleaved 都得出同样结论：在固定 visual-token count 下，connector architecture 几乎不重要。相同 token budget 下，对 mean-pooled patch 使用 2 层 MLP，表现与 32-query Q-Former 相差不到 1 分。

真正重要的是 token count。更多视觉 token = 更多 LLM compute = 性能提升，直到某一点后收益递减。每图 64 token 对 OCR 太少。576-1024 token 是大多数开放 VLM 的甜点区。2048+ 只对文档和图表有帮助。

Q-Former vs MLP 是成本问题，不是质量问题：Q-Former 无论图像分辨率如何都把 token 限制在 32-64；MLP 发出所有 patch token。高分辨率输入下，Q-Former 节省 LLM context；低分辨率下，差异只是噪音。

### Axis 3：LLM size 决定天花板

在几乎所有 VLM 论文中，把 LLM 从 7B 翻到 13B 都可靠地让 MMMU 增加 2-4 分。到 70B 时，大多数 benchmark 接近饱和。VLM 的多模态推理上限就是 LLM 的文本推理上限，视觉 encoder 只能喂它，不能替它推理。

这就是为什么 Qwen2.5-VL-72B 和 Claude Opus 4.7 在 MMMU-Pro 与 ScreenSpot-Pro 上碾压：语言大脑很大。7B VLM 不能靠巧妙 connector design 替代 70B VLM。

### Axis 4：data，详细人工 caption 胜过 distillation

Molmo + PixMo（Deitke et al., 2024）是每个人都该读的 2024 结果。Allen AI 让人工标注者用 1-3 分钟的 dense speech-to-text 描述图像，得到 71.2 万张 dense-captioned images。训练数据里没有任何 GPT-4V distillation。

Molmo-72B 在 11/11 个 benchmark 上击败 Llama-3.2-90B-Vision。差异不在 architecture，而在 caption quality。详细人工 caption 每张图像包含的信息量是短 web caption 的 5-10 倍，并且在 GPT-4V distillation 容易幻觉的地方保持事实 grounded。

ShareGPT4V（Chen et al., 2023）和 Cauldron（Idefics2）用混合 human + GPT-4V caption 遵循了同样 playbook。趋势很清楚：对 2026 frontier 来说，caption density > caption quantity > distillation convenience。

### Axis 5：resolution 及其 schedule

Idefics2 的 ablation：384 -> 448 增加 1-2 分。448 -> 980 加 image splitting（AnyRes）在 OCR benchmark 上再增加 3-5 分。平坦分辨率训练在中等准确率处 plateau；resolution ramping（从 224 开始，以 448 或 native 结束）训练更快，最终更高。

Cambrian-1 跑了 resolution vs tokens trade-off：在固定 compute 下，你可以选择低分辨率更多 token，或高分辨率更少 token。OCR 上高分辨率胜出；通用场景理解上低分辨率更多 token 胜出。

2026 年生产 recipe：Stage 1 用 384 fixed 训练，Stage 2 对 OCR-heavy 任务使用最高 1280 的 dynamic resolution。

### Prismatic 受控对比

Prismatic VLMs（Karamcheti et al., 2024）是控制所有轴的论文。同一个 13B LLM、同一份 instruction data、同一套 evaluation，只一次改变一个轴。结果：

- 每图 visual-token count 解释约 60% 方差。
- Encoder choice 解释约 20%。
- Connector architecture 解释约 5%。
- 其他所有因素（data mix、scheduler、LR）解释剩余约 15%。

这是粗略分解，但它是文献中对“我应该先 ablate 什么”的最干净回答。

### 2026 年 picker

基于证据，2026 年新项目的默认 open-VLM recipe：

- Encoder：SigLIP 2 SO400m/14，原生分辨率 + NaFlex；如果需要 segmentation/grounding，则拼接 DINOv2 ViT-g/14 作为 dense features。
- Connector：patch token 上的 2 层 MLP。除非你受到 token 限制，否则跳过 Q-Former。
- LLM：Qwen2.5 / Llama-3.1 / Gemma 2，成本优先选 7B，质量优先选 70B，按目标延迟选择。
- Data：PixMo + ShareGPT4V + Cauldron，再补充任务特定 instruction data。
- Resolution：dynamic（长边 min 256、max 1280 pixels）。
- Schedule：Stage 1 alignment（projector-only），Stage 2 full fine-tune，Stage 3 task-specific fine-tune。

这些默认值中的每一个都能追溯到本课末尾引用论文中的实测 ablation。

## 实际使用

`code/main.py` 是一个 ablation table parser 和 recipe picker。它编码 MM1 与 Idefics2 ablation table（压缩版），并允许你查询：

- “给定 budget X 和 task Y，哪个 recipe 胜出？”
- “如果我在 7B Llama 上把 SigLIP 换成 CLIP，预期 MMMU delta 是多少？”
- “为了得到 80% 置信答案，应该先 ablate 哪个轴？”

输出是带有预期 benchmark delta 和“ablate first”建议的 ranked recipe list。

## 交付成果

本课产出 `outputs/skill-vlm-recipe-picker.md`。给定 target task mix、compute budget 和 latency target，它会输出完整 recipe（encoder、connector、LLM、data mix、resolution schedule），并附上支撑每个选择的 ablation 引用。它能阻止工程师每次启动新 VLM 项目时重新发明 Idefics2 ablation table。

## 练习

1. 阅读 MM1 Section 3.2。对固定 2B LLM、5000 万图像 budget，哪个 encoder 胜出？到 13B LLM 时答案会反转吗？为什么？

2. Cambrian-1 发现 DINOv2 + SigLIP 拼接在 vision-centric benchmark 上优于任一单独 encoder，但在 MMMU 上不增加信号。预测哪些 benchmark 会提升，哪些保持不变。

3. 你的目标是一个运行在 2B LLM 上的移动 UI agent。选择 encoder、connector、resolution 和 data mix。用具体 ablation table 为每个选择辩护。

4. Molmo 发布 4B 和 72B 模型。4B 与 closed 7B VLM 竞争力相当；72B 在 11/11 个 benchmark 上击败 Llama-3.2-90B-Vision。这对 LLM-size plateau hypothesis 说明了什么？

5. 设计一个 ablation table，在 7B VLM 上隔离 data-mix quality 与 encoder quality。最少需要多少次训练运行？提出四个轴设置。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| Ablation | “Turning one knob” | 训练多次 run，每次只改变一个 design-space axis，其他全部保持不变 |
| Connector | “Bridge” / “projector” | 把 vision encoder 输出映射到 LLM token space 的可训练模块（MLP、Q-Former、Perceiver） |
| Detailed human caption | “Dense caption” | 多句人工描述（通常 80-300 token），比 web alt text 更丰富 |
| Distillation | “GPT-4V captions” | 由更强 proprietary VLM 生成的训练数据；方便但容易继承幻觉 |
| AnyRes / dynamic res | “High-res path” | 通过 tiling 或 M-RoPE 输入大于 encoder 原生分辨率的图像的策略 |
| Resolution ramp | “Curriculum” | 从低分辨率开始并逐步提高的训练 schedule，加速 alignment learning |
| Vision-centric bench | “CV-Bench / BLINK” | 强调细粒度视觉感知，而不是语言重推理的 evaluation |
| PixMo | “Molmo's data” | Allen AI 的 71.2 万张 dense-captioned image 数据集；人工语音转录成 dense caption |

## 延伸阅读

- [McKinzie et al. — MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611)
- [Laurençon et al. — Idefics2 / What matters building VLMs (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Deitke et al. — Molmo and PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)
- [Tong et al. — Cambrian-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)
