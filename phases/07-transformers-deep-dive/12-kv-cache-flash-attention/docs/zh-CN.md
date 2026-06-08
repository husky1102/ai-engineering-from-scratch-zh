# KV Cache, Flash Attention & Inference Optimization

> Training 是并行且 FLOP-bound 的。Inference 是串行且 memory-bound 的。瓶颈不同，技巧也不同。

**类型：** Build
**语言：** Python
**先修：** Phase 7 · 02 (Self-Attention), Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**时间：** ~75 分钟

## 要解决的问题

一个 naive autoregressive decoder 生成 `N` 个 token 要做 `O(N²)` 工作：每一步都会重新计算整个 prefix 上的 attention。对一个 4K-token response 来说，这是 16M 次 attention operations，其中大多数都是冗余的。prefix token 的每个 hidden state 一旦算出就是确定的；你只需要让新 token 的 query 去 attend 前面所有 token 的 cached keys 和 values。

除此之外，attention 本身会移动大量数据。标准 attention 会物化一个 N×N score matrix、N×d softmax output、N×d final output：对 HBM 来说读写太多。对 N≥2K，attention 在 FLOP-bound 之前就先变成 memory-bound。经典 attention kernels 对现代 GPU 的利用率低了 4-10×。

两个都来自 Dao et al. 的优化，把前沿 inference 从“慢”推到“快”：

1. **KV cache.** 存储每个 prefix token 的 K 和 V vectors。每个新 token 的 attention 变成一个 query 对 cached keys。Inference 从每个 generation step 的 `O(N²)` 降到 `O(N)`。
2. **Flash Attention.** 对 attention computation 做 tiling，让完整 N×N matrix 永远不落到 HBM。softmax + matmul 全部在 SRAM 中完成。在 A100 上获得 2-4× wall-clock speedup；在带 FP8 的 H100 上获得 5-10×。

到 2026 年，两者已经是通用默认项。每个 production inference stack（vLLM, TensorRT-LLM, SGLang, llama.cpp）都假设它们存在。每个 frontier model 都带着 Flash Attention enabled 出货。

## 核心概念

![KV cache growth and Flash Attention tiling](../assets/kv-cache-flash-attn.svg)

### KV cache math

每个 decoder layer、每个 token、每个 head：

```text
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K and V
```

对一个 7B 模型，32 layers、32 heads、d_head=128、fp16：

```text
per token per layer = 2 * 128 * 2 = 512 bytes
per token (32 layers) = 16 KB
per 32K context = 512 MB
```

对 Llama 3 70B（80 layers、d_head=128、GQA with 8 KV heads）：

```text
per token per layer = 2 * 8 * 128 * 2 = 4096 bytes (4 KB)
per 32K context = 10.4 GB
```

这 10 GB 就解释了为什么 Llama 3 70B 在 128K context、batch size 1 时，仅 KV cache 就需要占掉一张 40 GB A100 的大部分空间。

**GQA 是 KV-cache 的胜利点。** 使用 64 heads 的 MHA 会是 32 GB。MLA 还能进一步压缩。

拖动维度，观察 cache size 如何变化。把 sequence length 或 batch 调高，看看它多快会超过单张 GPU：

```figure
kv-cache-sizer
```

### Flash Attention：tiling trick

标准 attention：

```text
S = Q @ K^T          (HBM read, N×N, HBM write)
P = softmax(S)       (HBM read, HBM write)
O = P @ V            (HBM read, HBM write)
```

三次 HBM round trips。在 H100 上，HBM bandwidth 是 3 TB/s；SRAM 是 30 TB/s。与把所有东西留在 on-chip 相比，每一次 HBM trip 都是约 10 倍的 slowdown。

Flash Attention：

```text
for each block of Q (tile size ~128 × 128):
    load Q_tile into SRAM
    for each block of K, V:
        load K_tile, V_tile into SRAM
        compute S_tile = Q_tile @ K_tile^T     (SRAM)
        running softmax aggregation             (SRAM)
        accumulate into O_tile                  (SRAM)
    write O_tile to HBM
```

每个 tile 一次 HBM trip。总 memory footprint 从 `O(N²)` 降到 `O(N)`。Backward pass 不存储某些 forward pass 的值，而是在需要时重算它们：又是一次 memory win。

**数值技巧。** Running softmax 在 tile 之间维护 `(max, sum)`，所以最终 normalization 是精确的。这不是近似：Flash Attention 计算的输出与标准 attention bit-identical（除 fp16 non-associativity 外）。

**版本演进：**

| Version | Year | Key change | Speedup on reference hardware |
|---------|------|-----------|-------------------------------|
| Flash 1 | 2022 | Tiled SRAM kernel | 2× on A100 |
| Flash 2 | 2023 | Better parallelism, causal-first ordering | 3× on A100 |
| Flash 3 | 2024 | Hopper asynchrony, FP8 | 1.5-2× on H100 (~740 TFLOPs FP16) |
| Flash 4 | 2026 | Blackwell 5-stage pipeline, software exp2 | Inference-first (forward only initially) |

Flash 4 发布时只支持 forward-pass。Training 仍使用 Flash 3。Flash 4 对 GQA 和 varlen 的支持仍在推进中（2026 年中）。

### Speculative decoding：另一个 latency win

便宜模型提出 N 个 token。大模型并行验证全部 N 个。如果 verification 接受 k 个 token，你只为 k 次 generation 支付了 1 次 big-model forward pass。对 code 和 prose，典型 k=3-5。

2026 默认项：
- **EAGLE 2 / Medusa.** 集成 draft heads，共享 verifier 的 hidden states。无质量损失，2-3× speedup。
- **Speculative decoding with draft model.** 在 consumer hardware 上 2-4× speedup。
- **Lookahead decoding.** Jacobi iteration；不需要 draft model。小众但免费。

### Continuous batching

经典 batched inference：等待最慢的 sequence 结束，然后开始新 batch。当短 response 提前完成时会浪费 GPU。

Continuous batching（最早由 Orca 交付，现在在 vLLM、TensorRT-LLM、SGLang 中使用）：旧请求一结束，就把新请求换进 batch。对典型 chat workloads，吞吐提升 5-10×。

### PagedAttention：把 KV cache 当作 virtual memory

vLLM 的招牌功能。KV cache 以 16-token blocks 分配；page table 把 logical positions 映射到 physical blocks。这让你能在 parallel samples（beam search, parallel sampling）之间共享 KV，为 prompt caching 热切换 prefixes，并整理 memory fragmentation。相比 naive contiguous allocation，吞吐提升 4×。

## 动手实现

见 `code/main.py`。我们实现：

1. 一个 naive `O(N²)` incremental decoder。
2. 一个 `O(N)` KV-cached decoder。
3. 一个模拟 Flash Attention running-max algorithm 的 tiled softmax。

### Step 1: KV cache

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

很简单：把每个 token 的 K、V vectors 追加到 per-layer、per-head lists 中。

### Step 2: tiled softmax

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention-style softmax(qK^T)V with running max/sum."""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

输出与一次性 `softmax(qK) V` bit-identical，但任意时刻的 working set 是一个 `tile × d_head` block，而不是完整的 `N × d_head`。

### Step 3: 在 100-token generation 上比较 naive vs cached decoding

统计 attention operations。Naive：`O(N²)` = 5050。Cached：`O(N)` = 100。代码会打印两者。

## 实际使用

```python
# HuggingFace transformers auto-enables KV cache on decoder-only generate().
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # use FA3 if Hopper
    torch_dtype="bfloat16",
)
# generate() uses KV cache automatically
```

vLLM production：

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

跨请求 prefix caching 是 2026 年的大收益：相同的 system prompt、few-shot examples 或 long context document 会在多次调用之间复用 KV。对带重复 tool prompts 的 agent workloads，prefix caching 经常带来 5× throughput gain。

## 交付成果

见 `outputs/skill-inference-optimizer.md`。这个 skill 会为新的 inference deployment 选择 attention implementation、KV cache strategy、quantization 和 speculative decoding。

## 练习

1. **Easy.** 运行 `code/main.py`。确认 naive 和 cached decoders 产生相同输出；注意 op-count 差异。
2. **Medium.** 实现 prefix caching：给定一个 prompt P 和若干 completions，先对 P 运行一次 forward pass 填充 KV cache，然后按 completion 分支。测量相对每个 completion 都重新 encode P 的 speedup。
3. **Hard.** 实现一个 toy PagedAttention：KV cache 使用固定 16-token blocks 和 free-list。sequence 完成时，把 blocks 归还到 pool。模拟 1,000 个长度不同的 chat completions。比较 memory fragmentation 与 contiguous allocation。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| KV cache | “让 decoding 变快的技巧” | 存储每个 prefix token 的 K 和 V；新 query attend 到它们，而不是重算。 |
| HBM | “GPU main memory” | High Bandwidth Memory；H100 上 80 GB，B200 上 192 GB。~3 TB/s bandwidth。 |
| SRAM | “On-chip memory” | Per-SM fast memory，H100 上每个 SM 约 256 KB。~30 TB/s bandwidth。 |
| Flash Attention | “Tiled attention kernel” | 不在 HBM 中物化 N×N 就计算 attention。 |
| Continuous batching | “No-wait batching” | 不清空 batch 就把完成的 sequences 换出、新 sequences 换入。 |
| PagedAttention | “vLLM 的招牌” | KV cache 用 page table 分配在固定 blocks 中；消除 fragmentation。 |
| Prefix caching | “复用长 prompts” | 在请求之间为共享 prefix 缓存 KV；对 agents 是重大 cost cut。 |
| Speculative decoding | “Draft + verify” | 便宜 draft model 提出 tokens；大模型一次 pass 验证 k 个。 |

## 延伸阅读

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 1。
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Flash 2。
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Flash 3。
- [FlashAttention-4 release notes (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention) — Blackwell 5-stage pipeline 和 software-exp2 trick；阅读 repo README 以了解本课提到的 forward-only launch caveats。
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM paper。
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — spec decoding。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — 本课引用的 integrated-draft approach 的 EAGLE-1/2 paper。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — 与 EAGLE 一起被引用的 Medusa approach。
- [vLLM docs — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) — 关于 16-token block 和 page-table design 的 canonical deep dive。
