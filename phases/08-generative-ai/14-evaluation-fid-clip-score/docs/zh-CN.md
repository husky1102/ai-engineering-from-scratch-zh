# 生成模型评估：FID、CLIP Score 与人类偏好

> 每个 generative model leaderboard 都会引用 FID、CLIP score，以及来自 human-preference arena 的 win rate。每个数字都有一个能被有心研究者利用的 failure mode。如果你不知道这些 failure modes，就分不清真实改进和刷榜运行。

**类型:** Build
**语言:** Python
**先修:** Phase 8 · 01 (Taxonomy), Phase 2 · 04 (Evaluation Metrics)
**时间:** ~45 minutes

## 要解决的问题

Generative model 的评价维度是 *sample quality* 和 *conditioning adherence*。两者都没有 closed-form measure。你的模型必须 render 10,000 张 images；必须有东西给它们打分；你必须信任这些数字能跨 model families、resolutions、architectures 比较。三种 metrics 挺过了 2014-2026 的考验：

- **FID (Fréchet Inception Distance).** 在 Inception network feature space 中，real 与 generated 两个 distributions 之间的距离。越低越好。
- **CLIP score.** Generated image 的 CLIP-image embedding 与 prompt 的 CLIP-text embedding 之间的 cosine similarity。越高越好。衡量 prompt adherence。
- **Human preference.** 在同一 prompt 上让两个模型 head-to-head，由 humans（或 GPT-4-class model）选择更好者，聚合为 Elo score。

你还会看到：IS（inception score，基本退役）、KID、CMMD、ImageReward、PickScore、HPSv2、MJHQ-30k。每个都修正了前一个的某种失败。

## 核心概念

![FID, CLIP, and preference: three axes, different failure modes](../assets/evaluation.svg)

### FID — sample quality

Heusel et al. (2017)。步骤：

1. 为 N 张 real images 和 N 张 generated images 提取 Inception-v3 features（2048-D）。
2. 对每个 pool 拟合 Gaussian：计算 mean `μ_r, μ_g` 和 covariance `Σ_r, Σ_g`。
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`。

解释：feature space 中两个 multivariate Gaussians 的 Fréchet distance。越低 = distributions 越相似。

Failure modes：
- **Biased on small N.** FID 是 feature distribution 上的 mean-squared——小 N 会低估 covariance，给出虚低 FID。始终使用 N ≥ 10,000。
- **Inception-dependent.** Inception-v3 在 ImageNet 上训练。远离 ImageNet 的 domains（faces、art、text images）会产生无意义 FID。使用 domain-specific feature extractor。
- **Gaming.** Overfit Inception prior 可以让 FID 变低，却没有视觉质量提升。用 CMMD（见下）对抗。

### CLIP score — prompt adherence

Radford et al. (2021)。对于 generated image + prompt：

```text
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

在 30k generated images 上取平均 → 可在模型间比较的 scalar。

Failure modes：
- **CLIP's own blind spots.** CLIP 的 compositional reasoning 很弱（“a red cube on a blue sphere” 经常失败）。模型可能 CLIP score 很高，却没有真正遵循复杂 prompts。
- **Short prompt bias.** Short prompts 在自然图像中有更多 CLIP-image matches。Longer prompts 的 CLIP scores 会机械性更低。
- **Prompt gaming.** 在 prompt 中加入 “high quality, 4k, masterpiece” 会抬高 CLIP score，但不会改善 image-text binding。

CMMD（Jayasumana et al., 2024）修复了其中一些问题：使用 CLIP features 而非 Inception，使用 maximum-mean discrepancy 而非 Fréchet。它更擅长检测细微质量差异。

### Human preference — the ground truth

选择一组 prompts。用 model A 和 model B 生成。把 pairs 展示给 humans（或强 LLM judge）。把 wins 聚合成 Elo 或 Bradley-Terry score。Benchmarks：

- **PartiPrompts (Google)**：1,600 diverse prompts，12 categories。
- **HPSv2**：107k human annotations，广泛用作 automated proxy。
- **ImageReward**：137k prompt-image preference pairs，MIT-licensed。
- **PickScore**：在 Pick-a-Pic 2.6M preferences 上训练。
- **Chatbot-Arena-style image arenas**：https://imagearena.ai/ 等。

Failure modes：
- **Judge variance.** Non-experts 与 experts 的偏好不同。两者都用。
- **Prompt distribution.** Cherry-picked prompts 会偏向某个 family。始终 document。
- **LLM-judge reward hacking.** GPT-4-judge 会被漂亮但错误的 outputs 欺骗。用 human triangulate。

## Use together

Production eval report 应包括：

1. 在 10-30k samples 上，相对 held-out real distribution 计算 FID（sample quality）。
2. 在同一批 samples 上，对 prompts 计算 CLIP score / CMMD（adherence）。
3. 在 blinded arena 中相对上一版模型的 win rate（overall preference）。
4. Failure mode analysis：随机抽取 50 个 outputs，标记 known issues（hand anatomy、text rendering、consistent object count）。

单个 metric 都是谎言。三个互相印证的 metrics + qualitative review 才是 claim。

## 动手实现

`code/main.py` 在 synthetic “feature vectors” 上实现 FID、CLIP-score-like 和 Elo aggregation（我们用 4-D vectors 代替 Inception features）。你会看到：

- 在 small N 和 large N 上计算 FID——展示 bias。
- 作为 feature pools cosine similarity 的 “CLIP score”。
- 来自 synthetic preference stream 的 Elo update rule。

### Step 1: FID in four lines

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### Step 2: CLIP-style cosine-similarity

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### Step 3: Elo aggregation

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## 常见陷阱

- **FID at N=1000.** N<10k 时这个 heuristic 不可靠。报告 low-N FID 的论文是在 gaming。
- **Comparing FID across resolutions.** Inception 的 299×299 resize 会改变 feature distribution。只在 matched resolution 下比较。
- **Reporting one seed.** 最少运行 3 个 seeds。报告 std。
- **CLIP score inflation via negative prompts.** 有些 pipelines 会通过过拟合 prompt 来提高 CLIP。检查 visual saturation。
- **Elo bias from prompt overlap.** 如果两个模型都在训练中见过 benchmark prompt，Elo 没意义。使用 held-out prompt sets。
- **Human eval paid-crowd skew.** Prolific、MTurk annotators 偏年轻 / tech-friendly。与招募来的 art/design experts 混合。

## 实际使用

2026 年 production eval protocol：

| Pillar | Minimum | Recommended |
|--------|---------|-------------|
| Sample quality | FID on 10k vs held-out real | + CMMD on 5k + FID on subset per category |
| Prompt adherence | CLIP score on 30k | + HPSv2 + ImageReward + VQA-style question answering |
| Preference | 200 blinded pairs vs baseline | + 2000 paired human + LLM-judge + Chatbot Arena |
| Failure analysis | 50 hand-flagged | 500 hand-flagged + automated safety classifier |

四个 pillars 都在一份报告里 = claim。任何单个 pillar = marketing。

## 交付成果

保存 `outputs/skill-eval-report.md`。Skill 接收 new model checkpoint + baseline，输出完整 eval plan：sample sizes、metrics、failure-mode probes、sign-off criteria。

## 练习

1. **Easy.** 运行 `code/main.py`。在相同 synthetic distributions 上比较 N=100 与 N=1000 的 FID。报告 bias magnitude。
2. **Medium.** 从 synthetic CLIP-style features 实现 CMMD（公式见 Jayasumana et al., 2024）。比较它与 FID 对质量差异的 sensitivity。
3. **Hard.** 复现 HPSv2 setup：从 Pick-a-Pic 子集取 1000 image-prompt pairs，在 preferences 上 fine-tune 一个 small CLIP-based scorer，并测量它与 held-out set 的 agreement。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| FID | “Fréchet Inception Distance” | Real vs gen Inception features 的 Gaussian fits 之间的 Fréchet distance。 |
| CLIP score | “Text-image similarity” | CLIP image embedding 与 text embedding 之间的 cosine similarity。 |
| CMMD | “FID's replacement” | CLIP-feature MMD；bias 更低，没有 Gaussian assumption。 |
| IS | “Inception score” | Exp KL(p(y|x) || p(y))；在现代模型上相关性很差，已退役。 |
| HPSv2 / ImageReward / PickScore | “Learned preference proxies” | 在 human preferences 上训练的小模型；用作 automatic judges。 |
| Elo | “Chess rating” | Pairwise wins 的 Bradley-Terry aggregation。 |
| PartiPrompts | “The benchmark prompt set” | Google curated 的 1,600 prompts，跨 12 categories。 |
| FD-DINO | “Self-sup replacement” | 使用 DINOv2 features 的 FD；更适合 out-of-ImageNet domains。 |

## Production note: evaluation is an inference workload too

在 10k samples 上运行 FID 意味着生成 10k 张 images。对 1024² 上的 50-step SDXL base，在单张 L4 上就是约 11 小时 single-request inference。Evaluation budgets 是真实成本，framing 正是 offline-inference scenario（最大化 throughput，忽略 TTFT）：

- **Batch hard, forget latency.** Offline eval = static batching，使用内存能容纳的最大 batch size。在 80GB H100 上用 `pipe(...).images` 和 `num_images_per_prompt=8`，wall-clock 比 single-request 快 4-6×。
- **Cache the real features.** 对 real reference set 做 Inception（FID）或 CLIP（CLIP-score、CMMD）feature extraction 只运行*一次*，存成 `.npz`。不要每次 eval 重算。

对 CI / regression gates：每个 PR 在 500-sample subset 上运行 FID + CLIP score（约 30 min）；nightly 运行完整 10k FID + HPSv2 + Elo。

## 延伸阅读

- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — FID 论文。
- [Jayasumana et al. (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603) — CMMD。
- [Radford et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — CLIP。
- [Wu et al. (2023). HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) — HPSv2。
- [Xu et al. (2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) — ImageReward。
- [Yu et al. (2023). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) — PartiPrompts。
- [Stein et al. (2023). Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675) — failure-mode survey。
