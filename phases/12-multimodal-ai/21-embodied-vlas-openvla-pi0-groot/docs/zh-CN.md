# Embodied VLAs：RT-2、OpenVLA、π0、GR00T

> 第一次有模型从网站上读菜谱并在厨房机器人上执行，是 RT-2（Google DeepMind，2023 年 7 月）。RT-2 把 action 离散化成 text tokens，把 VLM 在 web data 与 robot-action data 上共同 fine-tune，并证明 web-scale vision-language knowledge 可以迁移到机器人控制。OpenVLA（2024 年 6 月）发布了开放 7B reference。Physical Intelligence 的 π0 系列（2024-2025）加入了 flow-matching action experts。NVIDIA 的 GR00T N1（2025 年 3 月）为大规模 humanoid robots 交付了 dual-system（System 1 / System 2）控制。VLA primitive，也就是 vision-language-action、一个会看、会读、会行动的单一模型，是本阶段理解模型与 Phase 15 autonomous systems 之间的桥梁。

**类型:** Learn
**语言:** Python（stdlib，action tokenizer + VLA inference skeleton）
**先修:** Phase 12 · 05（LLaVA），Phase 15（Autonomous Systems，referenced）
**时间:** ~180 分钟

## 学习目标

- 描述 action tokenization：discrete bin encoding（RT-2）、FAST efficient action tokens、continuous flow-matching actions（π0）。
- 解释为什么在 web + robot data 上 co-fine-tuning 能保留 general-knowledge transfer，让模型适应新任务。
- 在同一个 robot task 上比较 OpenVLA（开放 7B Llama+VLM）、π0（flow-matching）与 GR00T N1（dual-system）。
- 说出 Open X-Embodiment dataset，以及它作为 RT-X training corpus 的作用。

## 要解决的问题

一个能根据自然语言指令做家务的机器人，从 1970 年代起就是研究目标。2020 年代的答案是 vision-language-action（VLA）模型。它使用与 VQA 相同的 VLM 架构，但输出不是文本，而是 actions（joint torques、end-effector poses、discrete commands）。

VLA 特有的挑战：

1. Action space 是连续的（joint angles、forces），且高维（7-DOF arm + 3-DOF gripper = 10 dims at 30 Hz）。
2. 机器人专用训练数据稀缺。Open X-Embodiment 约有 1M trajectories；web text-image 有 5B+。
3. Control frequency 很关键。30 Hz control loop 意味着每个 action 只有 33ms budget。
4. Safety。错误 action 会损坏硬件、伤害人，或破坏财产。

## 核心概念

### Action tokenization（RT-2）

RT-2 的技巧：把每个 joint target 表示成量化 text token。把归一化的 [-1, 1] 范围离散化成 256 个 bin，并把每个 bin 映射到 vocabulary ID。一个 10-DOF action 在每个 control step 变成 10 个 token。

在混合数据上 co-fine-tune 一个 PaLM-X VLM：

- Web image-text pairs（captioning、VQA）。
- Robot demonstrations，action 作为 tokens。

模型看到“pick up the red cube”（language）→ image（vision）→ 10-token action sequence（discretized joint targets）。Web pretraining 保留 general-knowledge transfer：RT-2 可以遵循“move towards the fast-moving object”，尽管“fast-moving”不在训练数据中。

RT-2 论文中的 inference 为 3-5 Hz，受限于 VLM autoregressive decode。

### OpenVLA：开放 7B reference

OpenVLA（Kim et al., 2024 年 6 月）是开放权重版 RT-2 等价物。7B Llama backbone，DINOv2 + SigLIP dual vision encoder，并在 256 个 bin 上做 action tokenization。

它在 Open X-Embodiment（跨 22 个机器人、970k trajectories）上训练。提供 LoRA fine-tuning 支持，可适配新机器人。

Inference：在 A100 上配合 quantization 为 4-5 Hz。足以做慢速 manipulation，但不适合高频控制。

### FAST tokenizer：更快的 action decode

Pertsch et al.（2024）指出 discrete-bin tokenization 效率低：多数 action 聚集在 bin-space 的一个小区域。FAST（Frequency-domain Action Sequence Tokenizer）通过 DCT 压缩 action sequence，并量化系数。

一个 30-step action trajectory 会变成约 10 个 FAST tokens，而不是 300 个 discrete-bin tokens。Inference 加速 3-5x，且不损失质量。

### π0 与 flow-matching actions

Physical Intelligence 的 π0（Black et al., 2024 年 10 月）用 flow-matching action expert 替代 discrete action tokens：

- 一个小型 action transformer 读取 VLM 的 hidden states，并通过 rectified flow 输出连续 50-step action sequence。
- Action head 用 flow-matching loss 训练；VLM pretraining 保持不变。
- Inference：完整 action sequence 在约 5 个 denoising steps 内发出，实际可达 50 Hz control。

π0 的主张：在广泛 manipulation tasks 上超过 OpenVLA 和 Octo。连续 action 形式保留了离散化会破坏的平滑性。

π0.5 与 π0-FAST 是增量升级。π0-FAST 把 FAST tokenization 与 flow matching 结合。

### GR00T N1：面向 humanoid 的 dual-system

NVIDIA 的 GR00T N1（2025 年 3 月）面向 humanoid robots（>30 DOF，全身）：

- System 2：大型 VLM，读取场景 + instruction，以约 1 Hz 生成高层 subgoals。
- System 1：小型 action-head transformer，在 subgoals 条件下生成低层 50-100 Hz joint commands。

这种拆分对应 Kahneman 的 fast-and-slow thinking：System 2 规划，System 1 执行。好处是：慢速 VLM 规模规划不会阻塞快速控制；System 1 保持小模型以满足延迟。

GR00T N1.7（2025 年末）改进了 data scaling。GR00T 使用来自 Omniverse 的 sim-to-real data fine-tune。

### Open X-Embodiment

训练数据。RT-X（2023 年 10 月）汇集了 22 个数据集，覆盖 22 个机器人上的 1M trajectories。Open X-Embodiment 是大家都在用的语料：

- ALOHA / Bridge V2 / Droid / RT-2 Kitchen / Language Table。
- 每个样本：(robot state, camera views, instruction, action sequence)。
- 训练卫生：统一 action space、归一化 joint ranges、resize cameras。

OpenVLA 和 π0 都在 Open X-Embodiment 上训练。与任意具体机器人的 domain gap 通过 100-1000 条任务专用 demo 上的 LoRA fine-tuning 弥合。

### Co-fine-tuning vs robot-only

Co-fine-tuning 混合 web VQA data 与 robot trajectories。比例很关键：VQA 太多，模型会忘记 actions；robot data 太多，模型会丢失 general knowledge。

RT-2 的比例：约 1:1。OpenVLA：web-to-robot 约 0.5:1。π0：类似。精确比例是需要按数据集大小调优的 hyperparameter。

Robot-only training 会产生任务专用模型，在 out-of-distribution instructions 上失败。Co-fine-tuning 决定了模型是只能执行“pick up the red cube（demo 中出现过）”，还是能执行“pick up the third largest object from the left（新的表达）”。

### Safety 与 action limits

每个生产 VLA 都带有：

- Hard joint limits（不能超过规格施加 torque）。
- Velocity limits（soft clipping）。
- Workspace bounds（end-effector 不能离开桌面）。
- Human-in-the-loop approval，用于 novel tasks。

这些作为 control-layer checks 位于 VLA 外部。VLA 的输出是建议，不是命令。

## 实际使用

`code/main.py`：

- 实现 256-bin action tokenization 与 de-tokenization。
- 草拟一个基于 DCT + quantization 的 FAST tokenizer。
- 比较（discrete-bin、FAST、continuous-flow）在每个 action step 上的 token-count。
- 打印 RT-2 → OpenVLA → π0 → GR00T 的 lineage summary。

## 交付成果

本课产出 `outputs/skill-vla-action-format-picker.md`。给定一个 robot task（manipulation、navigation、humanoid whole-body），它会在 discrete-bin + RT-2、FAST + OpenVLA、flow-matching + π0 或 dual-system + GR00T 之间选择。

## 练习

1. 一个 10-DOF arm，control rate 为 30 Hz。256 bin 的 discrete-bin tokenization 每秒发出多少 token？7B VLM 能跟上吗？

2. FAST tokenization 把 30-step trajectories 压到约 10 个 token。如果 trajectory 有高频运动（例如 drumming），用户会失去什么？

3. π0 的 flow-matching head 用约 5 个 steps 完成 denoise。把 throughput 与 OpenVLA 在 4-5 Hz 下的 autoregressive decode 做比较。

4. GR00T 的 System 1 / System 2 拆分映射到 Kahneman。提出另一种可能有助于 bipedal walking 的拆分（System 3?）。

5. 阅读 Open X-Embodiment 第 4 节关于 dataset curation 的内容。列出三条防止 domain leakage 的 curation rules。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| VLA | “Vision-language-action” | 接收 image + instruction 并输出 action commands 的模型 |
| Action tokenization | “Discrete bins” | 把连续 joint targets 按每维 256 个 bin 量化，每个 bin 对应一个 vocab ID |
| FAST tokenizer | “Frequency action tokens” | DCT + quantize，把 30-step trajectories 压缩为约 10 个 token |
| Co-fine-tune | “Mix web + robot” | 把 web VQA data 与 robot demos 一起训练，以保留 general knowledge |
| Flow-matching action head | “π0 continuous output” | 通过 rectified flow 输出 50-step action sequence 的小型 transformer |
| System 1 / System 2 | “Dual-system control” | 大 VLM 慢速规划，小 action head 快速行动；GR00T pattern |
| Open X-Embodiment | “RT-X dataset” | 跨机器人的 1M-trajectory 数据集；训练语料 |

## 延伸阅读

- [Brohan et al. — RT-2 (arXiv:2307.15818)](https://arxiv.org/abs/2307.15818)
- [Kim et al. — OpenVLA (arXiv:2406.09246)](https://arxiv.org/abs/2406.09246)
- [Black et al. — π0 (arXiv:2410.24164)](https://arxiv.org/abs/2410.24164)
- [NVIDIA — GR00T N1 (arXiv:2503.14734)](https://arxiv.org/abs/2503.14734)
- [Open X-Embodiment Collab — RT-X (arXiv:2310.08864)](https://arxiv.org/abs/2310.08864)
