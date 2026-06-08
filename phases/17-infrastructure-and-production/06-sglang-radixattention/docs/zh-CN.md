# 面向 Prefix-Heavy Workloads 的 SGLang 与 RadixAttention

> SGLang 把 KV cache 视为 first-class、可复用资源，并存储在 radix tree 中。vLLM 按 FCFS（first-come, first-served）调度 requests，而 SGLang 的 cache-aware scheduler 优先处理 shared prefixes 更长的 requests，实际上是在做 depth-first radix traversal，让 hot branches 保持驻留在 HBM 中。在 Llama 3.1 8B、ShareGPT-like 1K prompts 上，SGLang 达到约 16,200 tok/s，而 vLLM 约 12,500，优势约 29%。在 prefix-heavy RAG workloads 上，优势可达 6.4x。在 voice-cloning-shaped workloads 上，cache hit rate 超过 86%。2026 年，它部署在 xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS 的 400,000+ GPUs 上。Gotcha 是：当 prefix ordering 不一致时，6.4x 这个数字会消失；ordering 是工程师的杠杆。

**类型：** Learn
**语言：** Python (stdlib, toy radix-tree cache + cache-aware scheduler)
**先修：** Phase 17 · 04 (vLLM Serving Internals), Phase 14 (Agentic RAG)
**时间：** ~75 minutes

## 学习目标

- 画出 RadixAttention：prefixes 如何存储在 radix tree 中，以及 KV blocks 如何在同一 branch 下的 sequences 间共享。
- 解释 cache-aware scheduling，以及为什么 FCFS 对 prefix-heavy traffic 是错误的。
- 给定 prefix-cache hit rate 和 prompt length distribution，计算 expected speedup。
- 说出让 6.4x 数字变成现实而不是损失收益的 prompt-ordering discipline。

## 要解决的问题

经典 serving 把每个 request 的 prompt 当作 opaque。即使 5,000 个 RAG requests 都以相同 2,000-token system prompt 加相同 retrieval preamble 开头，vLLM 也会 prefill 这个 2,000-token prefix 5,000 次。GPU 一遍又一遍做同样的工作。

观察是：agentic 和 RAG workloads 中的 prompts 几乎总是共享 long prefixes。System prompt、tool schemas、few-shot examples、retrieval headers、conversation history 都会跨 requests 重复。如果把该 prefix 的 KV cache 存储一次并复用，就不需要再次 prefill。

RadixAttention 正是这么做的。Tokens 在 radix tree 中索引；每个 node 拥有从 root 到该 node 路径上 token sequence 对应的 KV blocks。新 request 进入时遍历 tree：任何 token 匹配的 node 都复用该 node 的 KV blocks。Prefill cost 变成与“新增” suffix 成正比，而不是与完整 prompt 成正比。

挑战在 scheduling。如果两个 requests 共享 2,000-token prefix，第三个只共享同一 prefix 的 200 tokens，你希望一起服务两个 long-shared requests，让 long prefix 保持在 HBM 中。FCFS 做的是相反的事，它服务先到者，可能在下一个 long-prefix request 命中前就驱逐 hot branch。

## 核心概念

### Radix tree 作为 KV index

Radix tree（compact trie）存储 token sequences。每个 node 拥有一个 token range 和该 range 计算得到的 KV blocks。Children 会把 sequence 延伸一个或多个 tokens。

```text
root
 |- "You are a helpful assistant..."  (2,000 tokens, 124 KV blocks)
      |- "Context: <doc A>..."        (500 tokens, 31 blocks)
           |- "Question: Alice..."    (80 tokens, 5 blocks)
           |- "Question: Bob..."      (95 tokens, 6 blocks)
      |- "Context: <doc B>..."        (520 tokens, 33 blocks)
```

一个新 request 带着 system prompt + “Context: <doc A>” + “Question: Carol” 进入。Scheduler 遍历：system prefix 匹配（复用 124 blocks），doc-A branch 匹配（复用 31 blocks），然后只为 “Question: Carol”（4 blocks）分配新 blocks。Prefill cost：4 blocks 的新 tokens。没有 tree 时：160 blocks。Prefill 节省约 40x。

### Cache-aware scheduling

Radix-tree-backed reuse 如果 cache churn，就没有意义。两个关键 policies：

1. **Depth-first dispatch**。从 queue 中选择下一个 request 时，优先选择与当前 running set 处在同一 branch 的 requests。这会把 hot branch pin 住。
2. **Branch-level LRU，不是 block-level LRU**。驱逐整个 branches（从 shortest-used leaves 开始），而不是单个 blocks，让 cache shape 匹配 radix shape。

FCFS 违反两者。共享 2,000 tokens 的 request 排在共享 50 tokens 的 request 后面，然后 2,000-token branch 为接纳 50-token branch 而被驱逐。

### 你应该记住的 benchmark 数字

- Llama 3.1 8B、H100、ShareGPT 1K prompts：SGLang 约 16,200 tok/s，vLLM 约 12,500（约 29% 优势）。
- Prefix-heavy RAG（相同 system + 相同 doc，不同 question）：SGLang 最高 6.4x。
- Voice cloning workloads：86.4% prefix-cache hit rate。
- SGLang customers 的生产 hit rates：50-99%，取决于 prompt discipline。
- 2026 年部署在 400,000+ GPUs 上。

### Ordering gotcha

6.4x 这个数字依赖一致的 prompt-template ordering。如果你的 client 在某些 requests 中按 `[system, tools, context, history, question]` 构造 prompts，又在另一些中按 `[system, context, tools, history, question]` 构造，tree 无法找到 shared prefix。对人类来说看似共享 prefix，对 radix tree 来说是两个不同 sequences。

工程师的杠杆：你的 prompt template 就是 cache key。固定顺序。把所有 immutable 内容（system、tools、schemas）放在最前面。把 retrieval context 放在后面。把 user question 放在最后。不要把 dynamic content 交织进 prefix。

研究中的真实案例：把 dynamic content 移出 cacheable prefix，让一次 deployment 在一次改动中从 7% cache hit rate 提升到 74%。

### RadixAttention 何时胜出，何时不胜

胜出：
- RAG（相同 retrieval preamble，不同 question）。
- Agents（相同 tool schemas，不同 query）。
- 带 long system prompt 的 chat。
- 带 repeated preambles 的 voice / vision workloads。

不胜（回到 vLLM-level throughput）：
- Unique prompts 的 single-shot generation（code completion、没有 system prompt 的 open-ended chat）。
- 每个 request 都把 unique content 交织进 prefix 的 dynamic prompts。

### 为什么这是 scheduler problem，而不只是 kernel problem

你可以把 KV reuse 实现成 kernel trick。SGLang 的 insight 是：只有 scheduler 让 hot branch 保持驻留时，reuse 才有收益。Naive “reuse if available” policy 会在 mixed load 下让 cache churn。Radix-tree-indexed scheduler 才是把 kernel trick 变成 29% production edge 的东西。

### 与 vLLM 的相互关系

这两个系统并非严格竞争者。2026 年，vLLM 增加了 prefix caching（`--enable-prefix-caching`）和 cache-aware router（Rust 写的 vLLM Router）。差距缩小但没有完全消失，因为 SGLang 的整个 stack 都是 radix-first；vLLM 是 grafted it on。对于以 prefix reuse 为主的 workloads，SGLang 仍是默认选择。对于没有强 prefix patterns 的 general-purpose serving，vLLM 仍然持平或更好。

## 实际使用

`code/main.py` 实现了一个 toy radix-tree KV cache，加上两种 scheduler policies：FCFS 和 cache-aware。它让同一 workload 通过两者运行，报告 prefix-cache hit rate 和 throughput delta。然后运行一个 “scrambled ordering” workload，展示 6.4x 如何 collapse。

## 交付成果

本课产出 `outputs/skill-radix-scheduler-advisor.md`。给定 workload description（prompt-template shape、retrieval pattern、number of concurrent tenants），它会生成 prompt-ordering prescription 和是否采用 SGLang 的 go/no-go。

## 练习

1. 运行 `code/main.py`。在同一 workload 上比较 FCFS 和 cache-aware。Delta 来自哪里：prefill savings、decode savings，还是 queue delay？
2. 修改 workload，让 prompts 随机排列 `[system, tools, context]`。重新运行。Hit rate 会发生什么？为什么？
3. 计算在 Llama 3.1 8B 上，把 2,000-token system prompt 作为一个 radix branch 保持 resident 的 HBM cost。与没有 prefix reuse 的 16-sequence batch 成本比较。
4. 阅读 SGLang RadixAttention paper。用三句话解释为什么在 prefix-heavy load 下，tree-shaped LRU eviction 胜过 block-shaped LRU。
5. 一个客户报告 cache hit rate 只有 8%。说出三个可能原因，以及你会为每个原因运行的 diagnostic。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| RadixAttention | "the SGLang thing" | KV cache indexed as a radix tree，让 shared prefixes 复用 blocks |
| Radix tree | "compact trie" | 每个 node 拥有一个 token range 及其 KV blocks 的 tree |
| Cache-aware scheduler | "hot-branch-first" | 优先选择共享 resident branch 的 requests 的 scheduler |
| Prefix-cache hit rate | "how much of your prompt was free" | 从 reused KV blocks 服务的 prompt tokens 比例 |
| FCFS | "first-come first-served" | 破坏 prefix locality 的默认 scheduling |
| Branch-level LRU | "evict the leaf" | 与 radix shape 匹配的 eviction policy |
| Prompt template ordering | "the cache key" | Prompt 的 component order 决定 tree 能共享什么 |
| System prompt pinning | "resident prefix" | 保持 immutable system portion pinned，避免 eviction thrash |

## 延伸阅读

- [SGLang GitHub](https://github.com/sgl-project/sglang) — source and docs。
- [SGLang documentation](https://sgl-project.github.io/) — RadixAttention 和 scheduling details。
- [SGLang paper — Efficiently Programming Large Language Models (arXiv:2312.07104)](https://arxiv.org/abs/2312.07104) — design reference。
- [LMSYS blog — SGLang with RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/) — benchmark numbers 和 scheduler rationale。
- [vLLM — Prefix Caching](https://docs.vllm.ai/en/latest/features/prefix_caching.html) — vLLM 自己的 radix-like implementation，用于比较。
