# vLLM 服务内部机制：PagedAttention、Continuous Batching、Chunked Prefill

> vLLM 在 2026 年的主导地位来自三个会复合叠加的 defaults，而不是单一技巧。PagedAttention 始终开启。Continuous batching 会在 decode iterations 之间把新 requests 注入 active batch。Chunked prefill 会切分 long prompts，让 decode tokens 永不饥饿。三者全开时，一张 H100 SXM5 上的 Llama 3.3 70B FP8，在 128 concurrent 下可达到 2,200-2,400 tok/s，约比 vLLM 自身 default 高 25%，是 naive PyTorch loop 的 3-4x。本课会以你能画图解释的层级阅读 scheduler 和 attention kernel，并以 `code/main.py` 中的 toy continuous batcher 结束，它会像 vLLM 一样调度 prefill 和 decode。

**类型：** Learn
**语言：** Python (stdlib, toy continuous batching scheduler)
**先修：** Phase 17 · 01 (Model Serving), Phase 11 (LLM Engineering)
**时间：** ~75 minutes

## 学习目标

- 把 PagedAttention 解释为 KV cache allocator：blocks、block tables，以及为什么 production load 下 fragmentation 保持在 4% 以下。
- 在 iteration level 画出 continuous batching：finished sequences 如何离开 batch，new ones 如何加入而不 draining。
- 用一句话描述 chunked prefill，并说出它保护哪个 latency metric（提示：是 TTFT tail，不是 mean throughput）。
- 说出 2026 年 vLLM v0.18.0 中会咬到一次性开启所有 optimization 团队的 gotcha。

## 要解决的问题

Naive PyTorch serve loop 一次处理一个 request：tokenize、prefill、decode until EOS、return。一个用户时可行；一百个用户时，就成了一队耐心等待的人。显而易见的修复是 static batching，但它会把每个 request pad 到窗口中最长 prompt，把每次 decode pad 到最长 expected output，并让整个 batch 卡在最慢 sequence 上。你为从未使用的 padding 付费，fast requests 等 slow ones。

vLLM 同时解决三个问题。PagedAttention 阻止 KV cache fragmentation 像经典 contiguous allocation 那样吃掉 60-80% GPU memory。Continuous batching 让 requests 在每个 decode iteration 之间加入和离开 batch，因此 batch 始终充满真实工作。Chunked prefill 把 32k-token prompt 切成约 512-token slices，与 decode 交错，让一个 long prompt 不会冻结 GPU 上的每个 decode token。

2026 年的生产默认是三者全开。你需要理解每个做什么，因为 failure modes 全都在 scheduler 上，而不是 model 上。

## 核心概念

### PagedAttention 作为 virtual memory system

KV cache 对每个 sequence 的大小是 `num_layers × 2 × num_heads × head_dim × seq_len × bytes_per_element`。对于 Llama 3.3 70B、8192 tokens，BF16 下每个 sequence 约为 1.25 GB。如果你为每个 request 预留 8192 slots，但平均 request 只用 1500 tokens，就会浪费约 82% 已预留 HBM。Classic batching 要为这种浪费买单。

PagedAttention 借鉴 OS virtual memory。KV cache 对每个 sequence 不再 contiguous。它被分配为固定大小 blocks（默认 16 tokens）。每个 sequence 有一个 block table，把逻辑 token positions 映射到物理 block IDs。当 sequence 超过已分配 blocks 时，再加一个 block。完成后，blocks 返回 pool。

Fragmentation 从 60-80%（classic）下降到 4% 以下（PagedAttention）。你不需要用 flag 开启 PagedAttention，它是 vLLM 提供的唯一 allocator。可调旋钮是 `--gpu-memory-utilization`（默认 0.9），告诉 vLLM 在加载 weights 和 activations 后，为 KV blocks 预留多少 HBM。

### Iteration level 的 continuous batching

旧式 “dynamic batching” 会等待一个窗口（比如 10 ms）填满 batch，然后运行 prefill + decode + decode + decode，直到每个 sequence 完成。Fast sequences 早早离开，却在 GPU 处理 slow ones 时闲置等待。

Continuous batching 在每个 decode step 之间操作。把 running sequences 的集合称为 `RUNNING` list。每次 iteration：

1. `RUNNING` 中刚到 EOS 或 max_tokens 的 sequence 被移除。
2. Scheduler 查看 waiting queue。如果有 free KV blocks，就接纳 new sequences（prefill 或 resumed）。
3. Forward pass 在当前 `RUNNING` 上运行，每个 sequence 发出一个新 token。

Batch size 不会 pad 到固定数字。Output 位置不同的 sequences 共享一个 fused forward。2026 年 vLLM 中，这叫 `V1 scheduler`。关键 invariant：scheduler 每个 decode iteration 运行一次，而不是每个 request 运行一次。

### Chunked prefill 保护 TTFT tail

Prefill 是 compute-bound。在一张 H100 上，Llama 3.3 70B 的 32k-token prompt 需要约 800 ms 的纯 prefill。当 prefill 运行时，batch 中所有其他 sequences 的 decode tokens 都在等待。在 serving loop 中，一个 long prompt 的 first-token latency (TTFT) 会变成几十个其他用户的 inter-token latency (ITL) 抖动。

Chunked prefill 把 prefill 切成固定大小 chunks（默认 512 tokens），并把每个 chunk 作为一个 unit 调度。Chunks 之间，scheduler 可以让 decode sequences 前进一步。你用一小点 absolute prefill latency 损失（每 chunk 几 ms）换取低得多的 decode-time jitter。在已发布 benchmarks 中，mixed load 下 P99 ITL 从约 50 ms 降到约 15 ms。

### 三个 defaults 会相互作用

三个功能彼此假设对方存在。PagedAttention 给 scheduler 一个细粒度 KV resource 可供权衡。Continuous batching 需要这个细粒度 resource，这样接纳 new sequence 时不会触发全局 reshuffle。Chunked prefill 是 scheduler 在同一个 `RUNNING` list 上做出的决策，它是另一个 scheduler policy，而不是独立系统。

你不需要知道每个 flag。你需要知道 scheduler 优化什么：在 KV-block budget 下最大化 goodput，同时受 chunked prefill slicing 约束。

### 2026 年 v0.18.0 gotcha

在 vLLM v0.18.0 中，不能把 `--enable-chunked-prefill` 与 draft-model speculative decoding（`--speculative-model`）组合使用。文档例外是 V1 scheduler 中的 N-gram GPU speculative decoding。那些不读 release notes 就把所有 flag 打开的团队会在启动时遇到 run-time error，而不是 soft regression。如果 speculative gain 值得你启用 chunked prefill，就重新审视选择；2026 年的正确答案通常是 EAGLE-3 without chunked prefill，而不是不会 compile 的 draft model plus chunked prefill。

### 你应该记住的数字

- Llama 3.3 70B FP8、H100 SXM5、128 concurrent、三者全开：2,200-2,400 tok/s。
- 同一模型，default vLLM（no chunked prefill）：约 1,800 tok/s。
- 同一模型，naive PyTorch forward loop：约 600 tok/s。
- PagedAttention 下 production load 的 KV fragmentation waste：<4%。
- Mixed load 下 P99 ITL：with chunked prefill 约 15 ms，without 约 50 ms。

### Scheduler 长什么样

```text
while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # schedule prefill chunks + decode in one batch
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # e.g. 512 tokens
        else:
            batch.append(decode_one_token(s))     # 1 token

    run_forward(batch)                            # one fused GPU call
```

`code/main.py` 正是这个 loop 的 stdlib Python 版本，使用 fake token counts 和 fake forward latency。运行它会展示 chunked prefill 如何在 long prefill 期间让 decode sequences 保持活跃。

## 实际使用

`code/main.py` 模拟一个 vLLM-style scheduler，可切换 features。运行它观察：

- `NAIVE` mode：一次一个 request，无 batching。
- `STATIC` mode：pad and wait，classic batching。
- `CONTINUOUS` mode：iteration-level admission and release。
- `CONTINUOUS + CHUNKED` mode：prefill slices 与 decode 交错。

输出会显示 total throughput（tokens per virtual second）、TTFT mean 和 P99 ITL。`CONTINUOUS + CHUNKED` 行在 mixed traffic 上应占优。

## 交付成果

本课产出 `outputs/skill-vllm-scheduler-reader.md`。给定 serving config（batch size、KV memory utilization、chunked prefill size、speculative config），它会生成 scheduler diagnosis，指出三个 defaults 中哪一个正在成为 bottleneck，以及该调什么。

## 练习

1. 运行 `code/main.py`。在 mixed short and long requests workload 上比较 `STATIC` 与 `CONTINUOUS`。Throughput gap 来自哪里：prefill efficiency、decode efficiency，还是 tail latency？
2. 修改 toy scheduler，加入 `--max-num-batched-tokens`。对于运行 Llama 3.3 70B FP8 的 H100，正确值是多少？（提示：它是 KV block size 和 free blocks 数量的函数，而不是 raw HBM。）
3. 重新阅读 vLLM v0.18.0 release notes。哪些 flag 组合互斥？列出来。
4. 计算 1,000 requests trace 的 KV cache fragmentation waste：mean 1,500 output tokens，std 600 tokens，在 (a) 8192 max 的 contiguous per-request allocation，(b) 16-token blocks 的 PagedAttention 下分别是多少。
5. 用一段话解释为什么 chunked prefill 帮助 P99 ITL，但单独看并不提高 throughput。实践中的 throughput win 来自哪里？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| PagedAttention | "the KV trick" | KV cache 的 fixed-size block allocator；fragmentation <4% |
| Block table | "the page table" | 从 logical token position 到 physical KV block 的 per-sequence map |
| Continuous batching | "dynamic batching, but right" | 每个 decode iteration 做 admit/release decisions |
| Chunked prefill | "prefill splitting" | 把 long prefill 切成 512-token slices，与 decode 交错 |
| TTFT | "first token time" | Prefill + queue + network；long prompts 下由 prefill 主导 |
| ITL | "inter-token latency" | 连续 decode tokens 之间的时间；由 batch size 主导 |
| Goodput | "throughput that meets SLO" | 每个 request 仍满足 TTFT 和 ITL targets 的 tokens/sec |
| V1 scheduler | "the new scheduler" | vLLM 的 2026 scheduler；N-gram spec decode 是 chunked-prefill-compatible path |
| `--gpu-memory-utilization` | "the memory knob" | 加载 weights 和 activations 后，为 KV blocks 预留的 HBM 比例 |

## 延伸阅读

- [vLLM documentation — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/) — 关于 chunked-prefill 和 speculative-decoding compatibility 的官方来源。
- [vLLM Release Notes (NVIDIA)](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html) — 2026 release cadence 和 version-specific behavior。
- [vLLM Blog — PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) — 仍然定义 allocator 思维方式的原始文章。
- [PagedAttention paper (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180) — fragmentation analysis 和 scheduler design。
- [Aleksa Gordic — Inside vLLM](https://www.aleksagordic.com/blog/vllm) — 带 flame graphs 的详细 V1 scheduler walkthrough。
