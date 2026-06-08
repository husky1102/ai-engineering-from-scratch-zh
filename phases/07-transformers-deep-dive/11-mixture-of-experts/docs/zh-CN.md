# Mixture of Experts (MoE)

> 一个稠密 70B transformer 会为每个 token 激活所有参数。一个 671B MoE 每个 token 只激活 37B 参数，却在各项 benchmark 上胜出。稀疏性是这个十年最重要的 scaling 思想。

**类型：** Build
**语言：** Python
**先修：** Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**时间：** ~45 分钟

## 要解决的问题

稠密 transformer 在推理时的 FLOPs 等于它的参数量（forward pass 再乘以 2）。把稠密模型变大后，每个 token 都要付完整账单。到 2024 年，前沿模型已经撞上了 compute wall：想显著变聪明，就需要每个 token 指数级更多的 FLOPs。

Mixture of Experts 打破了这个绑定。把每个 FFN 替换成 `E` 个独立 expert，加一个为每个 token 选择 `k` 个 expert 的 router。总参数量 = `E × FFN_size`。每个 token 的激活参数量 = `k × FFN_size`。典型的 2026 配置：`E=256`，`k=8`。存储随 `E` 缩放，计算随 `k` 缩放。

2026 年的前沿几乎完全是 MoE：DeepSeek-V3（671B 总参数 / 37B 激活参数）、Mixtral 8×22B、Qwen2.5-MoE、Llama 4、Kimi K2、gpt-oss。在 Artificial Analysis 的独立 leaderboard 上，前 10 个 open-source model 全都是 MoE。

## 核心概念

![MoE layer: router selects k of E experts per token](../assets/moe.svg)

### FFN 替换

稠密 transformer block：

```text
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE block：

```text
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # pick k of E per token
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

每个 expert 都是一个独立 FFN（通常是 SwiGLU）。router 是一个单独的 linear layer。每个 token 选择自己的 `k` 个 expert，并得到它们输出的 gated mixture。

### 负载均衡问题

如果 router 把 90% 的 token 都送进 expert 3，其他 expert 就会饿死。人们尝试过三种修复：

1. **辅助负载均衡损失**（Switch Transformer, Mixtral）。加入一个与 expert 使用方差成比例的惩罚。有效，但会增加一个 hyperparameter 和第二个 gradient signal。
2. **Expert capacity + token dropping**（早期 Switch）。每个 expert 最多处理 `C × N/E` 个 token；溢出的 token 跳过这一层。会伤害质量。
3. **无辅助损失均衡**（DeepSeek-V3）。加入一个 learned per-expert bias，用来移动 router 的 top-k 选择。bias 在训练损失之外更新。不惩罚主目标。这是 2024 年的大解锁。

DeepSeek-V3 的做法：每个 training step 后，对每个 expert 检查它的使用量是高于还是低于目标。用 `±γ` 微调 bias。选择时使用 `scores + bias`。用于 gating 的 expert probabilities 仍然使用未改动的原始 `scores`。这把 routing 和 expression 解耦。

### Shared experts

DeepSeek-V2/V3 也把 expert 分成 *shared* 和 *routed*。每个 token 都经过所有 shared experts。Routed experts 通过 top-k 选择。Shared experts 捕获通用知识；routed experts 专门化。V3 运行 1 个 shared expert，再从 256 个 routed expert 中选 top-8。

### Fine-grained experts

经典 MoE（GShard, Switch）：每个 expert 都和完整 FFN 一样宽。`E` 较小（8-64），`k` 较小（1-2）。

现代 fine-grained MoE（DeepSeek-V3, Qwen-MoE）：每个 expert 更窄（1/8 FFN size）。`E` 更大（256+），`k` 也更大（8+）。总参数量相同，但组合数量增长快得多。`C(256, 8) = 400 trillion` 个可能的每-token “experts”。质量上升，延迟保持平坦。

### 成本画像

每个 token、每层：

| Config | Active params / token | Total params |
|--------|-----------------------|--------------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B (dense) | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2 (MoE) | ~32B | 1T |

DeepSeek-V3 几乎在所有 benchmark 上都击败 Llama 3 70B（dense），同时每个 token 的 **active FLOPs 更少**。更多参数 = 更多知识。更多 active FLOPs = 每个 token 更多计算。MoE 把二者解耦。

### 代价：memory

不管哪些 expert 被触发，所有 expert 都常驻 GPU。一个 671B 模型需要约 1.3 TB VRAM 来存 fp16 weights。前沿 MoE 部署需要 expert parallelism：把 experts 分片到多张 GPU 上，再跨网络 route tokens。延迟主要由 all-to-all communication 主导，而不是 matmul。

## 动手实现

见 `code/main.py`。一个纯 stdlib 的紧凑 MoE layer，包含：

- `n_experts=8` 个 SwiGLU-ish experts（每个一个 linear，仅作示意）
- top-k=2 routing
- softmax-normalized gating weights
- 通过 per-expert bias 实现无辅助损失均衡

### Step 1: router

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # softmax over ORIGINAL scores of the chosen experts
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

Bias 影响选择，不影响 gate weight。这就是 DeepSeek-V3 的技巧：bias 修正负载不均衡，却不转向模型预测。

### Step 2: 让 100 个 token 通过 router

追踪哪些 experts 被触发、触发了多少次。没有 bias 时，使用分布会偏斜。加入 bias update loop（对过度使用的 expert 用 `-γ`，对使用不足的 expert 用 `+γ`）后，使用量会在几轮内收敛到接近均匀分布。

### Step 3: 参数量对比

打印一个 MoE 配置的 “dense equivalent”。DeepSeek-V3 形状：256 routed + 1 shared，8 active，d_model=7168。总参数量惊人。active count 只有稠密 Llama 3 70B 的七分之一。

## 实际使用

HuggingFace 加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026 年的 production inference：vLLM 原生支持 MoE routing。SGLang 拥有最快的 expert-parallel 路径。两者都会自动处理 top-k selection 和 expert parallelism。

**什么时候选 MoE：**
- 你想用更低的每-token 推理成本获得前沿质量。
- 你有 VRAM / expert-parallel infrastructure。
- 你的 workload 是 token-heavy（chat, code），不是 context-heavy（long docs）。

**什么时候不要选 MoE：**
- Edge deployment：你要为任何 active FLOP 支付完整存储成本。
- Latency-critical single-user serving：expert routing 会增加 overhead。
- Small models（<7B）：MoE 的质量优势只会在超过某个 compute threshold（约 6B active params）后出现。

## 交付成果

见 `outputs/skill-moe-configurator.md`。这个 skill 会根据 parameter budget、training tokens 和 deployment target，为新的 MoE 选择 E、k 以及 shared-expert layout。

## 练习

1. **Easy.** 运行 `code/main.py`。观察 auxiliary-loss-free bias update 如何在 50 次迭代中抹平 expert usage。
2. **Medium.** 用 hash-based router（确定性、不学习）替换 learned router。比较 quality 和 balance。为什么 learned router 更好？
3. **Hard.** 实现 GRPO-style “rollout-matched routing”（DeepSeek-V3.2 trick）：记录 inference 期间哪些 experts 被触发，在 gradient computation 期间强制使用相同 routing。测量它对一个 toy policy-gradient setup 的影响。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Expert | “很多 FFN 中的一个” | 一个独立 feed-forward network；参数专用于 FFN 计算中的一个 sparse slice。 |
| Router | “The gate” | 一个很小的 linear layer，为每个 token 相对每个 expert 打分；top-k selection。 |
| Top-k routing | “每个 token 有 k 个 active experts” | 每个 token 的 FFN 计算正好经过 k 个 expert，并由 gate 加权。 |
| Auxiliary loss | “Load-balance penalty” | 额外 loss term，用来惩罚偏斜的 expert usage。 |
| Auxiliary-loss-free | “DeepSeek-V3 的技巧” | 只通过 router selection 上的 per-expert bias 来平衡；没有额外 gradient。 |
| Shared expert | “Always on” | 每个 token 都会经过的额外 expert；捕获通用知识。 |
| Expert parallelism | “按 expert 分片” | 把不同 experts 分发到不同 GPU；跨网络 route tokens。 |
| Sparsity | “Active params < total params” | 比率 `k × expert_size / (E × expert_size)`；DeepSeek-V3 为 37/671 ≈ 5.5%。 |

## 延伸阅读

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — 这个想法的起点。
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — Switch，经典 MoE。
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mixtral 8×7B。
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MLA + auxiliary-loss-free MoE + MTP。
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) — 基于 bias 的 balancing paper。
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) — 本课 router 使用的 fine-grained + shared-expert split。
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) — 最初的 shared-expert paper。
