# InternVL3：原生多模态预训练

> InternVL3 之前的每个开放 VLM 都遵循同样的三步 recipe：取一个在万亿级文本 token 上训练的文本 LLM，接上 vision encoder，然后 fine-tune 连接处。这有效，但有 alignment debt：文本 LLM 已经把全部预训练预算花在纯文本上，并不原生理解视觉 token。当你事后加入 vision，LLM 必须重新学习如何把视觉输入和文本推理关联起来，同时不能忘记文本。InternVL3（Zhu et al., 2025 年 4 月）拒绝 post-hoc 方法：一次预训练运行，从第一步开始交错 text 和 multimodal。结果是在开放 78B 参数下 MMMU-Pro 匹配 Gemini 2.5 Pro。本课阅读 native pretraining 的理由，以及采用它后会改变什么。

**类型:** Learn
**语言:** Python（stdlib，training-corpus mixer）
**先修:** Phase 12 · 05，Phase 12 · 07（recipes）
**时间:** ~120 分钟

## 学习目标

- 解释为什么 post-hoc VLM training 会积累 alignment debt，并引用三个可测症状（catastrophic forgetting、answer drift、visual-text inconsistency）。
- 描述 InternVL3 的 native pretraining corpus mix，以及 text : interleaved : caption 比例为什么重要。
- 对比 V2PE（variable visual position encoding）与 Qwen2-VL 的 M-RoPE。
- 说出 Visual Resolution Router（ViR）和 Decoupled Vision-Language（DvD）部署优化。

## 要解决的问题

Post-hoc VLM training 是默认做法。LLaVA、BLIP-2、Qwen-VL、Idefics 都取一个已经预训练好的 LLM（Llama、Vicuna、Qwen、Mistral）并加入 vision。训练阶段通常如下：

1. Frozen LLM + frozen vision encoder + trainable projector，在 caption pair 上训练以对齐 embedding。
2. 解冻 LLM，在 instruction data（LLaVA-Instruct、ShareGPT4V）上训练。
3. 可选 task-specific fine-tune。

会出现三个 alignment debt 症状：

- Catastrophic forgetting。Post-hoc VLM 忘记纯文本技能。GSM8K 分数下降 5-10 分。Hellaswag 分数下降。纯文本 agent 回退。
- Answer drift。同一个视觉问题的小措辞变化会得到不同答案。Vision encoder 与 LLM 的连接弱于 LLM 自有 token 之间的绑定。
- Visual-text inconsistency。VLM 可以正确描述图像，然后在回答问题时与自己的描述矛盾。视觉 token 不像文本 token 那样参与 LLM 的内部一致性检查。

这些症状已有充分记录。MM1.5 Section 4 量化了它们。LLaVA-OneVision 的 ablation 也暗示了它们。Native pretraining 是答案。

## 核心概念

### Native multimodal pretraining

InternVL3 从头在原生多模态语料上训练。混合比例是：

- 40% text-only data（FineWeb、Proof-Pile-2 等）
- 35% interleaved image-text data（OBELICS、MMC4-style）
- 20% paired image-caption data
- 5% video-text data

从第一个梯度步骤开始，vision token、text token 和跨模态交互都参与同一个 loss。没有 alignment pretraining，没有 projector freezing stage，也没有需要恢复的 catastrophic forgetting。

Base model 的训练是单阶段。Instruction tuning 随后进行，但 base model 已经把视觉 token 当作一等公民理解。

### V2PE（variable visual position encoding）

Qwen2-VL 使用固定轴分配的 M-RoPE。InternVL3 引入 V2PE：position encoding 随模态类型（text、image、video）变化，并带可学习 scaling。实践中：

- Text tokens 得到 1D position（text index）。
- Image patches 得到 2D position（row, col）。
- Video frames 得到 3D position（time, row, col）。

三者共享同一个 RoPE frequency base，但每个 band 的 hidden-dim allocation 是可学习参数，而不是固定 split。这样可以在预训练期间自由权衡 temporal 与 spatial frequency resolution。

V2PE 的 ablation claim：同 compute 下比 M-RoPE 在视频 benchmark 上高 1-2 分。不算革命，但更干净。

### Visual Resolution Router（ViR）

部署优化。不是所有图像都需要全分辨率编码。一张只有单个低细节物体的照片，如果按 1280px 原生编码就是浪费 token。ViR 是一个小分类器，在编码前预测回答问题所需的最低分辨率。

Routing 有三档：low-res（256 token）、medium（576）、high（2048+）。生产流量中 60% query 使用 low 或 medium 已足够。净效果：在同等质量下 throughput 提高 2-3 倍。

### Decoupled Vision-Language deployment（DvD）

服务大型 VLM 时，vision encoder 每张图像运行一次，而 LLM 对每个输出 token 自回归运行。两个组件瓶颈不同（vision = conv + attention 的 GPU memory bandwidth；LLM = KV cache）。DvD 把它们拆到不同 GPU 上，中间 streaming。

对于 8B + 400M encoder 模型，DvD 相比共置大约让单节点 throughput 翻倍。

### 单阶段 vs 多阶段质量

InternVL3 的主要 benchmark claim：78B 参数匹配 Gemini 2.5 Pro 的 MMMU-Pro。38B 匹配 GPT-4o。8B 领先开放 8B leaderboard。全部基于 single-stage pretrain + instruction-tune recipe。

Alignment-debt hypothesis 可测：相对于视觉 benchmark 增益，InternVL3-8B 在 text benchmark（MMLU、GSM8K）上丢失的分数少于 Qwen2.5-VL-7B。模型更像 generalist，因为训练是一整块，而不是两块。

### InternVL3.5 与 InternVL-U

InternVL3.5（2025 年 8 月）扩展 recipe。同样 native-pretrain approach，更多数据，更多参数。MMMU 改进是增量的。

InternVL-U（2026）加入 unified generation，也就是在同一 backbone 上叠加 MMDiT heads 做 image output。“U”代表“Understanding + generation”，追赶 Transfusion-style unified models（Lesson 12.13）。同一个 native-pretrain backbone 同时支持 understanding 和 generation heads。

### Native pretraining 的取舍

Native pretraining 不是免费的：

- Compute。从头训练新 VLM 的成本等同于训练文本 LLM，数百万 GPU-hours。Post-hoc adaptation 复用既有 LLM 权重，节省大部分成本。
- Data。大规模 interleaved image-text corpora 稀缺。OBELICS 有 1.41 亿 documents；MMC4 有 5.71 亿。纯文本可达 15T tokens。Multimodal pretraining data scarcity 是硬约束。
- Base-LLM reuse。Native pretraining 放弃了以后 drop in 新 LLM 的选项。Post-hoc 可通过只重训 adapter 把 Llama-3.1 换成 Llama-4。

InternVL3 的赌注是：alignment debt 比 reuse loss 更糟。Benchmark 支持这个 claim。生产成本会阻止未来实验室廉价复制。Post-hoc VLM 仍会存在，因为对大多数项目来说它更便宜。

## 实际使用

`code/main.py` 是 training-corpus mixer 和 ViR router simulator。它会：

- 接收目标语料混合（%text、%interleaved、%caption、%video），并计算每个模态的预期 step 数。
- 在一批 query 上模拟 ViR routing（分布：50% low-detail、30% medium、20% high-detail），并报告平均 token count。
- 给定 encoder 与 LLM FLOPs，报告 DvD throughput estimates。
- 打印 post-hoc 与 native pretraining 在参数、compute、data 和预期 alignment-debt symptoms 上的并排对比。

## 交付成果

本课产出 `outputs/skill-native-vs-posthoc-auditor.md`。给定一个 VLM training plan，它会审计应该走 native 还是 post-hoc，标记 alignment-debt risk，并推荐 corpus mix。当你评估新的 open-VLM 项目并需要选择训练策略时使用它。

## 练习

1. 估算 InternVL3-8B（native pretrain）与 LLaVA-OneVision-7B（post-hoc）之间的 compute delta。GPU-hour 比例大约是多少？差距由什么解释？

2. InternVL3 报告 40% text / 35% interleaved / 20% caption / 5% video。如果目标任务偏 video-heavy，提出新的比例，并论证为什么 base model 仍需要大量 text 和 caption data。

3. 阅读 MM1.5 Section 4 关于 forgetting 的内容。说出 post-hoc training 回退最大的具体 benchmark。回退损失了多少？

4. ViR 把 60% 流量路由到 low-resolution encoding。它会误路由哪些 query（需要 high-res 却发到 low-res）？提出三种 router-failure mode。

5. DvD 把 vision 和 LLM 拆到不同 GPU。在哪种 traffic pattern 下，DvD 会降低而不是提高 throughput？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|------------------------|
| Native multimodal pretraining | “From scratch together” | Text + image + video tokens 从第 1 步开始参与 loss，而不是后来接上去 |
| Alignment debt | “Post-hoc penalty” | 由把 vision 接到冻结 LLM 上造成的 text skills 和 answer consistency 可测回退 |
| V2PE | “Variable visual pos encoding” | 每模态可学习 position encoding allocation；InternVL3 的 M-RoPE 后继 |
| ViR | “Resolution router” | 编码前按 query 选择所需最低分辨率的小分类器，以节省 inference tokens |
| DvD | “Decoupled deployment” | Vision encoder 在一张 GPU，LLM 在另一张，通过 stream handoff；大型 VLM throughput 翻倍 |
| InternVL-U | “Unified understanding + generation” | 2026 follow-up，在 native-pretrain backbone 上加入 image-generation heads |
| Interleaved corpus | “OBELICS / MMC4” | 文本和图像按自然阅读顺序排列的 documents；native pretraining 的原料 |

## 延伸阅读

- [Chen et al. — InternVL 1 (arXiv:2312.14238)](https://arxiv.org/abs/2312.14238)
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
- [InternVL3.5 (arXiv:2508.18265)](https://arxiv.org/abs/2508.18265)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Zhang et al. — MM1.5 (arXiv:2409.20566)](https://arxiv.org/abs/2409.20566)
