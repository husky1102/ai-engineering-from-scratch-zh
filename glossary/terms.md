# AI 工程术语表

## A

### 智能体（Agent）
- **What people say:** "一个能自己思考、自主行动的 AI"
- **What it actually means:** 一个 while 循环：由 LLM 决定下一步调用哪个工具，执行它，观察结果，然后不断重复
- **Why it's called that:** 借自哲学——“agent（行动者）”指任何能在世界中行动的实体。在 AI 里，它不过就是“LLM + 工具 + 循环”

### 注意力（Attention）
- **What people say:** "AI 是怎么聚焦到重要部分的"
- **What it actually means:** 一种机制：每个 token 都对其他所有 token 的 value 做加权求和，权重由它们之间的相关程度决定（通过 query 与 key 向量的点积得到）
- **Why it's called that:** 2017 年的论文《Attention Is All You Need》以人类的选择性注意力作类比，给它起了这个名字

### 对齐（Alignment）
- **What people say:** "让 AI 变安全"
- **What it actually means:** 让 AI 系统的行为符合人类意图、价值观和偏好的技术挑战，包括设计者未曾预料到的边缘情况

### 自回归（Autoregressive）
- **What people say:** "AI 一个词一个词地生成"
- **What it actually means:** 一种模型：以之前所有 token 为条件预测下一个 token，再把该预测作为输入喂回去预测下一步。GPT、LLaMA 和 Claude 都是自回归的。

### 激活函数（Activation Function）
- **What people say:** "层与层之间那个非线性的东西"
- **What it actually means:** 在每个线性层之后施加的函数，用来引入非线性。没有它，无论堆叠多少线性层都会塌缩成单个线性变换。ReLU、GELU 和 SiLU 最常见。它的选择直接影响训练时梯度能否顺畅流动。

### Adam（优化器）
- **What people say:** "默认的优化器"
- **What it actually means:** 自适应矩估计（Adaptive Moment Estimation）。把动量（一阶矩）与逐参数的自适应学习率（二阶矩）结合起来，并对前期步数做偏差校正。在多数任务上几乎不用怎么调就能工作得很好。

### AdamW
- **What people say:** "更好的 Adam"
- **What it actually means:** 带解耦权重衰减的 Adam。在标准 Adam 中，L2 正则会被逐参数的自适应学习率缩放，而这并不是你想要的。AdamW 把权重衰减直接作用在权重上，与梯度统计量无关。训练 transformer 的默认优化器。

### Autograd（自动微分）
- **What people say:** "自动求梯度"
- **What it actually means:** 一套记录张量运算、并通过反向模式微分自动计算梯度的系统。PyTorch 的 autograd 在运行时动态构建计算图，而 JAX 用函数变换（grad）实现。正是它让反向传播变得实用——你只写前向过程，框架替你算出所有导数。

## B

### 批大小（Batch Size）
- **What people say:** "一次处理多少个样本"
- **What it actually means:** 在更新权重前，一次前向/反向传播所处理的训练样本数量。批越大，梯度估计越稳定，但占用显存越多。常见取值：训练时 32-512，推理时更大。批大小与学习率相互影响——批翻倍，学习率也翻倍（线性缩放法则）。

### 反向传播（Backpropagation）
- **What people say:** "神经网络是怎么学习的"
- **What it actually means:** 一种算法：通过沿网络反向应用链式法则，计算出每个权重对误差贡献了多少，然后按比例调整权重
- **Why it's called that:** 误差从输出向输入、逐层向后传播

## C

### 上下文窗口（Context Window）
- **What people say:** "AI 能记住多少东西"
- **What it actually means:** 单次 API 调用能容纳的最大 token 数（输入 + 输出）。它不是记忆——而是一个固定大小的缓冲区，每次调用都会清空重置

### 思维链（Chain of Thought / CoT）
- **What people say:** "让 AI 一步一步思考"
- **What it actually means:** 一种提示技巧：要求模型展示其推理步骤。这能提升多步问题的准确率，因为每一步都会作为条件影响下一个 token 的生成

### CNN（卷积神经网络）
- **What people say:** "处理图像的 AI"
- **What it actually means:** 一种使用卷积运算（在输入上滑动滤波器）来检测局部模式的神经网络。层层堆叠卷积可以检测越来越复杂的特征：边缘、纹理、物体。

### CUDA
- **What people say:** "GPU 编程"
- **What it actually means:** NVIDIA 的并行计算平台。让你能在数千个 GPU 核心上同时运行矩阵运算。PyTorch 和 TensorFlow 底层都用它。

### 分块（Chunking）
- **What people say:** "把文档切成小段"
- **What it actually means:** 在嵌入以供检索之前，把文本切成片段。块大小决定了搜索结果的粒度。太小：丢失上下文。太大：稀释相关性。常见策略：带重叠的定长切分、按句子切分，或语义切分。典型块大小：256-512 token，带 10-20% 的重叠。

### 对比学习（Contrastive Learning）
- **What people say:** "通过比较来学习"
- **What it actually means:** 通过在嵌入空间中把相似的样本对拉近、把不相似的样本对推远来训练。CLIP 就用了这一点：匹配的图文对 vs 不匹配的图文对。

### 余弦相似度（Cosine Similarity）
- **What people say:** "两个向量有多相似"
- **What it actually means:** 两个向量夹角的余弦：dot(a, b) / (||a|| * ||b||)。取值从 -1（方向相反）到 1（方向相同）。忽略大小，只关心方向。是嵌入和语义搜索的标准相似度度量。

### 交叉熵（Cross-Entropy）
- **What people say:** "分类用的损失"
- **What it actually means:** 衡量两个概率分布之间的差异。分类任务：-sum(y_true * log(y_pred))。语言模型：正确的下一个 token 的负对数概率。越低越好。困惑度（perplexity）就是 exp(交叉熵)。

## D

### 数据增强（Data Augmentation）
- **What people say:** "造出更多训练数据"
- **What it actually means:** 对已有数据制作修改过的副本（旋转图像、加噪声、改写文本），在不采集新数据的前提下增加训练集的多样性。能减少过拟合。

### 解码器（Decoder）
- **What people say:** "负责输出的那部分"
- **What it actually means:** 在 transformer 中，解码器使用因果（带掩码的）自注意力，使每个位置只能关注更早的位置。GPT 是 decoder-only，BERT 是 encoder-only，T5 是 encoder-decoder。

### 扩散模型（Diffusion Model）
- **What people say:** "从噪声生成图像的 AI"
- **What it actually means:** 一种被训练来逆转“逐步加噪”过程的模型——它学会预测并去除噪声；在生成时从纯噪声出发，迭代地去噪

### DPO（直接偏好优化）
- **What people say:** "更简单的 RLHF"
- **What it actually means:** 一种完全跳过奖励模型的训练方法——它直接优化语言模型，使其在成对的人类偏好中更倾向于更好的那个回复

### Dropout
- **What people say:** "随机关闭一些神经元"
- **What it actually means:** 训练时，随机把一部分激活值置零。迫使网络不依赖任何单个神经元。推理时关闭。简单却有效的正则化方法。

## E

### 特征值（Eigenvalue）
- **What people say:** "PCA 里那个数学玩意儿"
- **What it actually means:** 对矩阵 A，特征值 lambda 满足 Av = lambda*v（v 为某向量）。它告诉你矩阵在那个方向上把向量缩放了多少。大的特征值 = 数据中方差大的方向。

### 嵌入（Embedding）
- **What people say:** "把词变成数字的某种 AI 魔法"
- **What it actually means:** 一种学习得到的映射，把离散项（词、图像、用户）映射到连续空间中的稠密向量，让相似的项最终彼此靠近
- **Why it's called that:** 这些项被“嵌入（embed）”到一个几何空间里，在那里距离是有意义的

### 编码器（Encoder）
- **What people say:** "负责输入的那部分"
- **What it actually means:** 在 transformer 中，编码器使用双向自注意力，使每个位置都能关注所有位置。BERT 是 encoder-only。擅长理解类任务（分类、命名实体识别），但不擅长生成。

### 轮次（Epoch）
- **What people say:** "把数据过一遍"
- **What it actually means:** 字面意思就是如此。完整地遍历一次训练集中的每个样本。多个 epoch = 多次看到数据。更多 epoch 可能提升学习效果，但有过拟合风险。

## F

### 特征（Feature）
- **What people say:** "数据里的一列"
- **What it actually means:** 数据中一个可度量的单独属性。在经典 ML 中，你手工设计特征；在深度学习中，网络从原始数据中自动学到特征。

### 少样本（Few-Shot）
- **What people say:** "先给 AI 几个例子"
- **What it actually means:** 在让模型执行任务之前，于提示中加入少量输入-输出示例，通常 3-5 个。模型据此进行模式匹配，理解期望的格式与行为。与零样本（无示例）和微调（成千上万示例烧进权重）相对。

### 微调（Fine-tuning）
- **What people say:** "拿你的数据去训练 AI"
- **What it actually means:** 从一个预训练模型的权重出发，在更小的、任务特定的数据集上继续训练。只更新已有权重，并不会从零添加新知识

### 函数调用（Function Calling）
- **What people say:** "能使用工具的 AI"
- **What it actually means:** 一种让 LLM 请求执行外部函数的结构化方式。你用 JSON Schema 描述定义工具，模型输出一个结构化 JSON 对象，指明调用哪个函数、传什么参数，你的代码执行它，结果再返回给模型。它与智能体不是一回事——函数调用是机制，智能体是那个循环。

## G

### 护栏（Guardrails）
- **What people say:** "AI 的安全过滤器"
- **What it actually means:** 围绕 LLM 的输入/输出校验层，用来检测并拦截有害内容、提示注入尝试、PII（个人隐私信息）泄露或离题回复。通常是一条流水线：输入过滤 -> LLM -> 输出过滤。可以基于规则（正则、关键词表）或基于模型（给安全性打分的分类器）。

### GPT
- **What people say:** "ChatGPT"或"那个 AI"
- **What it actually means:** Generative Pre-trained Transformer（生成式预训练 Transformer）——一种用 decoder-only transformer、在大规模文本语料上训练、预测下一个 token 的特定架构
- **Why it's called that:** Generative（生成文本）、Pre-trained（先在大数据上训练一次，再做适配）、Transformer（所用架构）

### GAN（生成对抗网络）
- **What people say:** "两个 AI 互相打架"
- **What it actually means:** 一个生成器网络试图造出逼真的数据，而一个判别器网络试图分辨真假。它们一起训练：生成器越来越会骗过判别器，判别器越来越会识破假货。

### 梯度（Gradient）
- **What people say:** "斜率"
- **What it actually means:** 一个由偏导数组成的向量，指向函数上升最陡的方向。在 ML 中，你沿梯度的反方向走（梯度下降）来最小化损失。

### 梯度下降（Gradient Descent）
- **What people say:** "AI 是怎么变好的"
- **What it actually means:** 一种优化算法，朝着最能陡峭地降低损失函数的方向调整参数，就像在高维地形里往山下走

## H

### 超参数（Hyperparameter）
- **What people say:** "你要调的设置"
- **What it actually means:** 在训练前设定、用来控制训练过程本身的值：学习率、批大小、层数、dropout 比例。与模型参数（权重）不同，这些不是从数据中学到的。

### 幻觉（Hallucination）
- **What people say:** "AI 在撒谎"或"在瞎编"
- **What it actually means:** 模型生成了听起来合理、但并未扎根于其训练数据或给定上下文的文本——它在做模式补全，而不是事实检索

## I

### 推理（Inference）
- **What people say:** "运行 AI"
- **What it actually means:** 用训练好的模型对新数据做预测。不发生任何权重更新。这正是你在生产中做的事：送入输入，得到输出。

### 归纳偏置（Inductive Bias）
- **What people say:** 没听说过
- **What it actually means:** 内建在模型架构中的假设。CNN 假设局部模式重要（卷积）。RNN 假设顺序重要（顺序处理）。Transformer 假设任何东西都可能与任何东西相关（注意力）。合适的偏置能让模型用更少的数据更快学会。

### JAX
- **What people say:** "Google 的 ML 框架"
- **What it actually means:** 一个兼容 NumPy 的库，额外提供自动微分（grad）、JIT 编译（jit）、自动向量化（vmap）和多设备并行（pmap）。与 PyTorch 的面向对象风格不同，JAX 是纯函数式的——没有隐藏状态，没有原地修改。Google DeepMind 用它做了 AlphaFold、Gemini 和大规模研究。

## K

### KV 缓存（KV Cache）
- **What people say:** "让推理更快"
- **What it actually means:** 在自回归生成过程中，缓存之前 token 的 key 和 value 矩阵，这样每一步就不必重新计算它们。以显存换速度。是快速 LLM 推理的关键。

## L

### 潜在空间（Latent Space）
- **What people say:** "隐藏的表示"
- **What it actually means:** 一个压缩的、学习得到的表示空间，相似的输入映射到邻近的点。自编码器、VAE 和扩散模型都在潜在空间中工作。它比输入维度更低，却抓住了重要的结构。

### 学习率（Learning Rate）
- **What people say:** "AI 学得多快"
- **What it actually means:** 一个控制梯度下降步长的标量。太高：越过最小值并发散。太低：收敛太慢或卡住。最重要的单个超参数。

### LLM（大语言模型）
- **What people say:** "AI"或"大脑"
- **What it actually means:** 一种基于 transformer 的神经网络，被训练来预测序列中的下一个 token，拥有数十亿参数，在互联网规模的文本数据上训练而成

### LoRA（低秩适配）
- **What people say:** "高效微调"
- **What it actually means:** 不去更新所有权重，而是在原权重旁插入小的低秩矩阵。只训练这些小矩阵，把显存占用降低 10-100 倍

### 损失函数（Loss Function）
- **What people say:** "AI 错得有多离谱"
- **What it actually means:** 一个衡量预测输出与真实输出之间差距的函数。训练就是最小化这个函数。回归用 MSE，分类用交叉熵，嵌入用对比损失。损失函数的选择定义了模型眼中“好”的含义。

## M

### 混合精度（Mixed Precision）
- **What people say:** "提速的训练技巧"
- **What it actually means:** 前向传播和大多数运算用 float16（更快、更省显存），但梯度累加和权重更新保留 float32（更精确）。能获得 2 倍加速，而精度损失可忽略。

### MoE（专家混合）
- **What people say:** "只有部分模型在运行"
- **What it actually means:** 一种含有许多“专家”子网络的模型，由路由机制把每个输入只送给少数几个专家。整个模型很大，但每次前向都很便宜，因为大多数专家被跳过了。Mixtral 和 GPT-4 都用了它。

### MCP（模型上下文协议）
- **What people say:** "一种让 AI 使用工具的方式"
- **What it actually means:** 一个开放协议（基于 stdio/HTTP 的 JSON-RPC），用来标准化 AI 应用连接外部数据源和工具的方式，并为工具、资源和提示提供带类型的 schema

## N

### NaN（非数值）
- **What people say:** "训练崩了"
- **What it actually means:** 一个表示未定义结果（0/0、inf-inf）的浮点值。训练中出现 NaN 损失通常意味着：学习率太高、梯度爆炸、对零取对数，或除以零。训练失败时永远第一个要查的东西。

### 归一化（Normalization）
- **What people say:** "缩放数据"
- **What it actually means:** 把数值调整到一个标准范围。批归一化（batch norm）在一个批内归一化，层归一化（layer norm）在特征维度上归一化。两者都能稳定训练并允许更高的学习率。

## O

### 过拟合（Overfitting）
- **What people say:** "模型把数据背下来了"
- **What it actually means:** 模型在训练数据上表现好，但在未见过的数据上表现差。它学到的是噪声，而非信号。解决办法：更多数据、正则化（dropout、权重衰减）、早停、数据增强、更简单的模型。

### 优化器（Optimizer）
- **What people say:** "更新权重的那个东西"
- **What it actually means:** 一种用梯度来更新模型参数的算法。SGD 最简单，Adam 最常用。每种优化器有不同特性：收敛速度、显存占用、对超参数的敏感度。

## P

### 参数（Parameter）
- **What people say:** "模型大小"
- **What it actually means:** 模型中一个可学习的值，通常是权重或偏置。“7B 参数”意味着 70 亿个可学习的数。每个 float32 参数占 4 字节，所以 7B 参数 = 仅权重就要 28GB 显存。

### 困惑度（Perplexity）
- **What people say:** "模型有多困惑"
- **What it actually means:** 平均交叉熵损失的指数。越低越好。困惑度为 10 意味着模型的不确定程度，相当于每一步都在 10 个 token 中均匀地随机挑选。

### 精确率与召回率（Precision & Recall）
- **What people say:** "准确度指标"
- **What it actually means:** 精确率 = 你标记出来的项里，有多少是对的。召回率 = 所有正确的项里，你找到了多少。两者此消彼长：抓住每一封垃圾邮件（高召回）意味着更多误报（低精确）。F1 分数是两者的调和平均。误报代价高时看精确率，漏报代价高时看召回率。

### 提示工程（Prompt Engineering）
- **What people say:** "用对的方式跟 AI 说话"
- **What it actually means:** 设计输入文本以可靠地产生期望的输出——包括系统提示、少样本示例、格式说明和思维链触发语

### 提示注入（Prompt Injection）
- **What people say:** "用文字黑掉 AI"
- **What it actually means:** 一种攻击：输入中的恶意文本覆盖了系统提示或指令。直接注入：用户输入“忽略之前的指令”。间接注入：被检索到的文档里藏着指令。它相当于 LLM 版的 SQL 注入。没有彻底的解决方案——防御是多层的输入校验、输出过滤和权限隔离。

## Q

### QLoRA
- **What people say:** "更便宜的 LoRA"
- **What it actually means:** 量化版 LoRA。把冻结的基座模型权重保留为 4 位精度（NF4 格式），同时以 16 位训练 LoRA 适配器。相比标准 LoRA 再省 3-4 倍显存。一个用 LoRA 需要 14GB 的 7B 模型，用 QLoRA 只要 4-6GB。在多数基准上质量与全量微调相差不到 1%。

## R

### RAG（检索增强生成）
- **What people say:** "能搜索的 AI"
- **What it actually means:** 一种模式：你从知识库中检索相关文档（用嵌入相似度），把它们塞进提示，让 LLM 基于该上下文作答
- **Why it's called that:** Retrieval（检索文档）+ Augmented（加入提示）+ Generation（LLM 写出答案）

### RLHF（基于人类反馈的强化学习）
- **What people say:** "他们是怎么让 AI 变得有用的"
- **What it actually means:** 一条训练流水线：(1) 收集人类对模型输出的偏好，(2) 在这些偏好上训练一个奖励模型，(3) 用 PPO 优化 LLM 去产生更高奖励的输出

### 量化（Quantization）
- **What people say:** "把模型变小"
- **What it actually means:** 把模型权重的精度从 float32（4 字节）降到 int8（1 字节）或 int4（0.5 字节）。用很小的精度损失换取 4-8 倍的显存减少和更快的推理。GPTQ、AWQ 和 GGUF 是常见格式。

### ReLU
- **What people say:** "激活函数"
- **What it actually means:** 修正线性单元（Rectified Linear Unit）：f(x) = max(0, x)。最简单的非线性激活。计算快，对正值不饱和。因为有效又便宜而被到处使用。变体：LeakyReLU、GELU、SiLU。

### ROUGE
- **What people say:** "摘要评测指标"
- **What it actually means:** Recall-Oriented Understudy for Gisting Evaluation。衡量生成文本与参考文本之间的重叠。ROUGE-1 计算一元词（unigram）匹配，ROUGE-2 计算二元词（bigram）匹配，ROUGE-L 找最长公共子序列。计算便宜，但只衡量表面相似度——两句意思相同但用词不同的话，得分会很低。

## S

### 语义搜索（Semantic Search）
- **What people say:** "理解含义的智能搜索"
- **What it actually means:** 按含义而非关键词匹配来找文档。把查询和所有文档嵌入到同一个向量空间，然后返回嵌入与查询嵌入最接近的文档。“payment failed”能找到“transaction declined”，哪怕它们没有共同的词。由嵌入模型 + 向量数据库驱动。

### 流式输出（Streaming）
- **What people say:** "看着回复一个词一个词地冒出来"
- **What it actually means:** LLM 在 token 生成的同时就发送它们，而不是等完整回复生成完。使用 Server-Sent Events（SSE）或 WebSocket 协议。把首 token 的感知延迟从几秒降到几毫秒。对生产级聊天界面至关重要。每个 chunk 含有一个 delta（部分 token 或词）。

### 自注意力（Self-Attention）
- **What people say:** "模型怎么决定关注什么"
- **What it actually means:** 每个 token 计算出 query、key、value 向量。两个 token 间的注意力权重 = 它们 query 与 key 的点积，经缩放并 softmax。输出 = value 向量的加权和。让每个 token 都能看到其他每个 token。

### SFT（监督微调）
- **What people say:** "教模型遵循指令"
- **What it actually means:** 在（指令, 回复）对上微调一个预训练模型。模型学会在给定指令时生成对应回复。这正是把一个基座模型变成聊天模型的过程。

### Softmax
- **What people say:** "把数字变成概率"
- **What it actually means:** softmax(x_i) = exp(x_i) / sum(exp(x_j))。把一个任意实数向量变成一个概率分布（全为正、加和为 1）。用于分类头、注意力权重，以及任何你需要概率的地方。

### 蜂群（Swarm）
- **What people say:** "一群 AI 智能体像蜜蜂一样协作"
- **What it actually means:** 多个智能体共享状态、通过消息传递进行协调，复杂行为从简单的个体规则中涌现，而非来自中央控制

## T

### 系统提示（System Prompt）
- **What people say:** "AI 的指令"
- **What it actually means:** 对话开头的一条特殊消息，用来设定模型的行为、人设和约束。在用户消息之前被处理。多数 UI 中对用户不可见。它定义模型该做什么、不该做什么，以及语气、格式偏好和领域聚焦。与用户提示不同——系统提示由开发者设定。

### 张量（Tensor）
- **What people say:** "一个多维数组"
- **What it actually means:** 深度学习框架中的基本数据结构。0 维张量是标量，1 维是向量，2 维是矩阵，3 维及以上是张量。在 PyTorch 和 JAX 中，张量会记录其计算历史以供自动微分，并可存放在 CPU 或 GPU 上。神经网络所有的输入、输出、权重和梯度都是张量。

### Token（词元）
- **What people say:** "一个词"
- **What it actually means:** 由 BPE 这类分词器产生的子词单元（英文中通常 3-4 个字符）。“unbelievable”可能是 3 个 token：“un” + “believ” + “able”

### 温度（Temperature）
- **What people say:** "创造力开关"
- **What it actually means:** 一个在 softmax 之前对 logits 做除法的标量。Temperature=1 是默认。越高 = 分布越平 = 输出越随机。越低 = 分布越尖 = 越确定。Temperature=0 是 argmax（总是选最可能的 token）。

### 迁移学习（Transfer Learning）
- **What people say:** "用一个预训练模型"
- **What it actually means:** 把在一个任务上训练好的模型适配到另一个任务。前面的层学到的是通用特征（边缘、句法模式），这些可以迁移。只有后面的层需要任务特定训练。这就是为什么你能把 BERT 微调到任何 NLP 任务。

### Transformer
- **What people say:** "现代 AI 背后的架构"
- **What it actually means:** 一种神经网络架构，用自注意力（让每个位置都能关注其他每个位置）取代循环来处理序列，从而实现大规模并行
- **Why it's called that:** 它通过注意力层把输入表示“变换（transform）”为输出表示

## U

### 欠拟合（Underfitting）
- **What people say:** "模型没在学"
- **What it actually means:** 模型太简单，抓不住数据中的模式。训练损失一直很高。解决办法：更多参数、更多层、更长训练、更弱的正则、更好的特征。

## V

### VAE（变分自编码器）
- **What people say:** "一种生成模型"
- **What it actually means:** 一种自编码器，通过强制编码器输出服从高斯分布，来学习一个平滑的潜在空间。你可以从该分布采样并解码以生成新数据。重参数化技巧让它能通过反向传播训练。

### 向量数据库（Vector Database）
- **What people say:** "给 AI 用的特殊数据库"
- **What it actually means:** 一种为存储向量（浮点数的稠密数组）并执行快速近似最近邻搜索而优化的数据库。它是相似度搜索、RAG 和推荐系统中的核心操作。

## W

### 权重（Weight）
- **What people say:** "模型学到的东西"
- **What it actually means:** 模型参数矩阵中的单个数。一个输入大小 768、输出大小 3072 的线性层有 768*3072 = 2,359,296 个权重。训练就是调整每个权重以最小化损失函数。

### 权重衰减（Weight Decay）
- **What people say:** "正则化"
- **What it actually means:** 在损失函数中加入一个与权重大小成正比的惩罚项。等价于 L2 正则化。防止权重变得过大。典型取值：0.01-0.1。

## Z

### 零样本（Zero-Shot）
- **What people say:** "不需要训练"
- **What it actually means:** 把模型用在它未被显式训练过的任务上，且提示中没有任何任务特定示例。模型从预训练中泛化而来。之所以可行，是因为大模型见过足够多的样式，能应付新的任务格式。
