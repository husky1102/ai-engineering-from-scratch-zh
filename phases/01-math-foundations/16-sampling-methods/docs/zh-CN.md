# 采样方法

> 采样是 AI 探索可能性空间的方式。

**类型：** 构建
**语言：** Python
**先修：** Phase 1，第 06-07 课（概率，贝叶斯定理）
**时间：** ~120 分钟

## 学习目标

- 只使用均匀随机数，从零开始实现 inverse CDF、rejection sampling 和 importance sampling
- 为语言模型 token 生成构建 temperature、top-k 和 top-p（nucleus）sampling
- 解释 reparameterization trick，以及它为什么能让 VAE 中的采样参与反向传播
- 运行 Metropolis-Hastings MCMC，从未归一化的目标分布中采样

## 要解决的问题

语言模型处理完你的 prompt 后，会产生一个包含 50,000 个 logits 的向量。词表中的每个 token 对应一个 logit。现在它必须选出一个。怎么选？

如果它总是选择概率最高的 token，每次回答都会一模一样。确定、单调、无聊。如果它均匀随机选择，输出就会变成胡言乱语。答案位于这两个极端之间，而这个“中间地带”由采样控制。

采样并不只用于文本生成。强化学习通过采样轨迹来估计 policy gradient。VAE 通过从学到的分布中采样并让梯度穿过随机性，学习 latent representation。Diffusion model 通过采样噪声并迭代去噪来生成图像。Monte Carlo 方法估计没有闭式解的积分。MCMC 算法探索无法枚举的高维后验分布。

每一个生成式 AI 系统都是一个采样系统。采样策略决定输出的质量、多样性和可控性。本课会从零构建每一种主要采样方法：从均匀随机数开始，一直到支撑现代 LLM 和生成模型的技术。

## 核心概念

### 为什么采样重要

采样在 AI 和机器学习中承担四种基础角色：

**生成。** 语言模型、diffusion model 和 GAN 都通过采样产生输出。采样算法直接控制创造性、连贯性和多样性。Temperature、top-k 和 nucleus sampling 是工程师每天都会调整的旋钮。

**训练。** Stochastic gradient descent 采样 mini-batch。Dropout 采样要停用的神经元。Data augmentation 采样随机变换。Importance sampling 通过重新加权样本，降低强化学习（PPO、TRPO）中的 gradient variance。

**估计。** ML 中的许多量没有闭式解：数据分布上的期望损失、energy-based model 的 partition function、Bayesian inference 中的 evidence。Monte Carlo estimation 通过对样本求平均来近似所有这些量。

**探索。** MCMC 算法在 Bayesian inference 中探索后验分布。Evolutionary strategies 采样参数扰动。Thompson sampling 在 bandit 问题中平衡探索与利用。

核心挑战是：你只能直接从简单分布（uniform、normal）中采样。对于其他一切分布，你都需要一种方法，把简单样本转换成来自目标分布的样本。

### 均匀随机采样

每一种采样方法都从这里开始。均匀随机数生成器会在 [0, 1) 中产生数值，其中任意等长子区间都有相同概率。

```text
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    for 0 <= a <= b <= 1

Properties:
  E[U] = 0.5
  Var(U) = 1/12
```

要从 n 个元素的离散集合中均匀采样，生成 U 并返回 floor(n * U)。要从连续区间 [a, b] 中采样，计算 a + (b - a) * U。

关键洞见是：一个均匀随机数恰好包含了从任意分布中生成一个样本所需的随机性。技巧在于找到正确的变换。

### Inverse CDF 方法（Inverse Transform Sampling）

累积分布函数（CDF）把数值映射为概率：

```text
F(x) = P(X <= x)

Properties:
  F is non-decreasing
  F(-inf) = 0
  F(+inf) = 1
  F maps the real line to [0, 1]
```

inverse CDF 把概率映射回数值。如果 U ~ Uniform(0, 1)，那么 X = F_inverse(U) 服从目标分布。

```text
Algorithm:
  1. Generate u ~ Uniform(0, 1)
  2. Return F_inverse(u)

Why it works:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**Exponential distribution 示例：**

```text
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

Solve F(x) = u for x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Since (1 - U) and U have the same distribution:
  x = -ln(u) / lambda
```

当你能写出 F_inverse 的闭式表达式时，这个方法非常完美。对于 normal distribution，没有闭式 inverse CDF，所以我们会使用其他方法（Box-Muller，或数值近似）。

**离散版本：** 对于离散分布，把 CDF 构造成累计和，生成 U，然后找到累计和第一次超过 U 的索引。第 06 课中的 `sample_categorical` 就是这样工作的。

### Rejection Sampling

当你无法反转 CDF，但可以在差一个常数的情况下计算目标 PDF 时，可以使用 rejection sampling。

```text
Target distribution: p(x)  (can evaluate, possibly unnormalized)
Proposal distribution: q(x)  (can sample from)
Bound: M such that p(x) <= M * q(x) for all x

Algorithm:
  1. Sample x ~ q(x)
  2. Sample u ~ Uniform(0, 1)
  3. If u < p(x) / (M * q(x)), accept x
  4. Otherwise, reject and go to step 1

Acceptance rate = 1/M
```

边界 M 越紧，接受率越高。在低维（1-3 维）中，rejection sampling 表现很好。在高维中，接受率会指数级下降，因为 proposal 的大部分体积都会被拒绝。这就是 rejection sampling 的维度灾难。

**示例：从 truncated normal 中采样。** 在截断区间上使用 uniform proposal。包络 M 是该区间内 normal PDF 的最大值。

**示例：从半圆中采样。** 在外接矩形中均匀提议点。如果点落在半圆内部，就接受。这也是 Monte Carlo 计算 pi 的方式：接受率等于面积比 pi/4。

### Importance Sampling

有时你并不需要来自目标分布 p(x) 的样本。你需要估计 p(x) 下的期望，而你手里有来自另一个分布 q(x) 的样本。

```text
Goal: estimate E_p[f(x)] = integral of f(x) * p(x) dx

Rewrite:
  E_p[f(x)] = integral of f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

where w(x) = p(x) / q(x)  are the importance weights.

Estimator:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    where x_i ~ q(x)
```

这在强化学习中非常关键。在 PPO（Proximal Policy Optimization）中，你用旧策略 pi_old 收集轨迹，但想优化新策略 pi_new。importance weight 是 pi_new(a|s) / pi_old(a|s)。PPO 会裁剪这些权重，防止新策略偏离旧策略太远。

importance sampling 估计器的方差取决于 q 与 p 的相似程度。如果 q 与 p 差得很远，少数样本会得到巨大的权重并主导估计。Self-normalized importance sampling 会除以权重和，以缓解这个问题：

```text
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### Monte Carlo Estimation

Monte Carlo estimation 通过对随机样本求平均来近似积分。大数定律保证它会收敛。

```text
Goal: estimate I = integral of g(x) dx over domain D

Method:
  1. Sample x_1, ..., x_N uniformly from D
  2. I ~ (Volume of D / N) * sum(g(x_i))

Error: O(1 / sqrt(N))   regardless of dimension
```

误差率与维度无关。这就是为什么在基于网格的积分不可行的高维空间里，Monte Carlo 方法占据主导地位。

**估计 pi：**

```text
Sample (x, y) uniformly from [-1, 1] x [-1, 1]
Count how many fall inside the unit circle: x^2 + y^2 <= 1
pi ~ 4 * (count inside) / (total count)
```

**估计期望：**

```text
E[f(X)] ~ (1/N) * sum(f(x_i))    where x_i ~ p(x)

The sample mean converges to the true expectation.
Variance of the estimator = Var(f(X)) / N
```

### Markov Chain Monte Carlo（MCMC）：Metropolis-Hastings

MCMC 构造一条 Markov chain，使它的 stationary distribution 是目标分布 p(x)。经过足够多步后，来自这条链的样本就（近似）来自 p(x)。

```text
Target: p(x)  (known up to a normalizing constant)
Proposal: q(x'|x)  (how to propose the next state given the current state)

Metropolis-Hastings algorithm:
  1. Start at some x_0
  2. For t = 1, 2, ..., T:
     a. Propose x' ~ q(x'|x_t)
     b. Compute acceptance ratio:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Accept with probability min(1, alpha):
        - If u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - Otherwise: x_{t+1} = x_t
  3. Discard first B samples (burn-in)
  4. Return remaining samples
```

对于对称 proposal（q(x'|x) = q(x|x')），这个比值会简化为 p(x')/p(x)。这就是原始的 Metropolis algorithm。

**为什么有效。** 接受规则保证 detailed balance：位于 x 并移动到 x' 的概率，等于位于 x' 并移动到 x 的概率。Detailed balance 意味着 p(x) 是这条链的 stationary distribution。

**实践注意事项：**
- Burn-in：丢弃链到达平衡之前的早期样本
- Thinning：每隔 k 个样本保留一个，以降低 autocorrelation
- Proposal scale：太小则链移动缓慢（高接受率、慢探索）；太大则大多数 proposal 被拒绝（低接受率、停在原地）
- 高维中 Gaussian proposal 的最优接受率大约是 0.234

### Gibbs Sampling

Gibbs sampling 是多变量分布中 MCMC 的一个特例。它不会一次性在所有维度上提议移动，而是每次从条件分布中更新一个变量。

```text
Target: p(x_1, x_2, ..., x_d)

Algorithm:
  For each iteration t:
    Sample x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Sample x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Sample x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

Gibbs sampling 要求你能从每个条件分布 p(x_i | x_{-i}) 中采样。对许多模型来说这很直接：
- Bayesian networks：条件分布来自图结构
- Gaussian mixtures：条件分布是 Gaussian
- Ising models：每个 spin 的条件分布只依赖它的邻居

接受率永远是 1（每个 proposal 都会被接受），因为从精确条件分布中采样会自动满足 detailed balance。

**局限。** 当变量高度相关时，Gibbs sampling 混合得很慢，因为一次只更新一个变量，无法沿分布做大幅的对角移动。

### Temperature Sampling（用于 LLM）

语言模型会为词表中的每个 token 输出 logits z_1, ..., z_V。Softmax 会把它们转换成概率。Temperature 会在 softmax 之前重新缩放 logits：

```text
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: standard softmax (original distribution)
T -> 0:  argmax (deterministic, always picks highest logit)
T -> inf: uniform (all tokens equally likely)
T < 1.0: sharpens the distribution (more confident, less diverse)
T > 1.0: flattens the distribution (less confident, more diverse)
```

**为什么有效。** 用 T < 1 除 logits 会放大 logits 之间的差异。如果 z_1 = 2 且 z_2 = 1，除以 T = 0.5 会得到 z_1/T = 4 和 z_2/T = 2，让差距变大。经过 softmax 后，最高 logit 的 token 会获得大得多的概率份额。

**实践中：**
- T = 0.0：greedy decoding，最适合事实型 Q&A
- T = 0.3-0.7：略有创造性，适合代码生成
- T = 0.7-1.0：平衡，适合一般对话
- T = 1.0-1.5：创意写作、头脑风暴
- T > 1.5：越来越随机，很少有用

Temperature 不会改变哪些 token 是可能的。它改变的是分配给每个 token 的 probability mass。

### Top-k Sampling

Top-k sampling 把候选集合限制为概率最高的 k 个 token，然后重新归一化，并从这个受限集合中采样。

```text
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Keep only the top k tokens
  4. Renormalize: p_i' = p_i / sum(p_j for j in top-k)
  5. Sample from the renormalized distribution

k = 1:  greedy decoding
k = V:  no filtering (standard sampling)
k = 40: typical setting, removes long tail of unlikely tokens
```

Top-k 会阻止模型选择词表分布长尾中极不可能的 token（拼写错误、无意义内容）。问题在于：无论上下文如何，k 都是固定的。当模型很有把握时（某个 token 有 95% 概率），k = 40 仍然允许 39 个替代项。当模型不确定时（概率分散在 1000 个 token 上），k = 40 又会切掉一些合理选项。

### Top-p（Nucleus）Sampling

Top-p sampling 会动态调整候选集合大小。它不保留固定数量的 token，而是保留累计概率超过 p 的最小 token 集合。

```text
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Find smallest k such that sum of top-k probabilities >= p
  4. Keep only those k tokens
  5. Renormalize and sample

p = 0.9:  keeps tokens covering 90% of probability mass
p = 1.0:  no filtering
p = 0.1:  very restrictive, nearly greedy
```

当模型很有把握时，nucleus sampling 只保留很少的 token（也许 2-3 个）。当模型不确定时，它会保留很多 token（也许 200 个）。这种自适应行为就是 nucleus sampling 通常能生成比 top-k 更好文本的原因。

**常见组合：**
- Temperature 0.7 + top-p 0.9：优秀的通用设置
- Temperature 0.0（greedy）：最适合确定性任务
- Temperature 1.0 + top-k 50：Fan et al.（2018）原始论文设置

Top-k 和 top-p 可以组合使用。先应用 top-k，再在剩余集合上应用 top-p。

### Reparameterization Trick（用于 VAE）

Variational autoencoders（VAE）的学习方式是：把输入编码成 latent space 中的一个分布，从该分布中采样，再把样本解码回来。问题是：你无法通过采样操作做反向传播。

```text
Standard sampling (not differentiable):
  z ~ N(mu, sigma^2)

  The randomness blocks gradient flow.
  d/d_mu [sample from N(mu, sigma^2)] = ???
```

Reparameterization trick 会把随机性与参数分离：

```text
Reparameterized sampling:
  epsilon ~ N(0, 1)          (fixed random noise, no parameters)
  z = mu + sigma * epsilon   (deterministic function of parameters)

  Now z is a deterministic, differentiable function of mu and sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradients flow through mu and sigma.
```

它之所以有效，是因为 N(mu, sigma^2) 与 mu + sigma * N(0, 1) 具有相同分布。关键洞见是：把随机性移到一个不含参数的来源（epsilon）里，然后把样本表达为参数的可微变换。

**在 VAE training loop 中：**
1. Encoder 为每个输入输出 mu 和 log(sigma^2)
2. 采样 epsilon ~ N(0, 1)
3. 计算 z = mu + sigma * epsilon
4. 解码 z 来重构输入
5. 通过第 4、3、2、1 步反向传播（可行，因为第 3 步是可微的）

没有 reparameterization trick，VAE 就无法用标准反向传播训练。正是这个洞见让 VAE 变得实用。

### Gumbel-Softmax（可微的 Categorical Sampling）

Reparameterization trick 适用于连续分布（Gaussian）。对于离散 categorical distribution，我们需要另一种方法。Gumbel-Softmax 提供了对 categorical sampling 的可微近似。

**Gumbel-Max trick（不可微）：**

```text
To sample from a categorical distribution with log-probabilities log(p_1), ..., log(p_k):
  1. Sample g_i ~ Gumbel(0, 1) for each category
     (g = -log(-log(u)), where u ~ Uniform(0, 1))
  2. Return argmax(log(p_i) + g_i)

This produces exact categorical samples.
```

**Gumbel-Softmax（可微近似）：**

```text
Replace the hard argmax with a soft softmax:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (temperature) controls the approximation:
  tau -> 0:  approaches a one-hot vector (hard categorical)
  tau -> inf: approaches uniform (1/k, 1/k, ..., 1/k)
  tau = 1.0: soft approximation
```

Gumbel-Softmax 产生的是离散样本的连续松弛。输出是一个概率向量（soft one-hot），而不是 hard one-hot。梯度会穿过 softmax。在训练的 forward pass 中，你可以使用 "straight-through" estimator：forward pass 使用 hard argmax，backward pass 使用 soft Gumbel-Softmax 的梯度。

**应用：**
- VAE 中的离散 latent variables
- Neural architecture search（选择离散操作）
- Hard attention mechanisms
- 带离散动作的强化学习

### Stratified Sampling

标准 Monte Carlo sampling 可能会因为随机性，在样本空间中留下空隙。Stratified sampling 通过把空间划分成 strata 并从每个 strata 中采样，强制实现均匀覆盖。

```text
Standard Monte Carlo:
  Sample N points uniformly from [0, 1]
  Some regions may have clusters, others gaps

Stratified sampling:
  Divide [0, 1] into N equal strata: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Sample one point uniformly within each stratum
  x_i = (i + u_i) / N   where u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

与标准 Monte Carlo 相比，stratified sampling 的方差总是更低或相等：

```text
Var(stratified) <= Var(standard Monte Carlo)

The improvement is largest when f(x) varies smoothly.
For piecewise-constant functions, stratified sampling is exact.
```

**应用：**
- 数值积分（quasi-Monte Carlo）
- 训练数据划分（确保每个 fold 中的类别平衡）
- 带分层的 importance sampling（组合两种技术）
- NeRF（Neural Radiance Fields）沿 camera rays 使用 stratified sampling

### 与 Diffusion Models 的连接

Diffusion models 通过采样过程生成图像。forward process 会在 T 步内向图像添加 Gaussian noise，直到它变成纯噪声。reverse process 学习去噪，一步步恢复原始图像。

```text
Forward process (known):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  where epsilon ~ N(0, I)

  After T steps: x_T ~ N(0, I)  (pure noise)

Reverse process (learned):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  where z ~ N(0, I)

  Each denoising step is a sampling step.
```

与本课方法的连接：
- 每个去噪步骤都使用 reparameterization trick（采样噪声，应用确定性变换）
- noise schedule {alpha_t} 控制一种 temperature annealing
- 训练使用 Monte Carlo estimation 来近似 ELBO（evidence lower bound）
- Diffusion models 中的 ancestral sampling 是一条 Markov chain（每一步只依赖当前状态）

整个图像生成过程都是迭代采样：从噪声开始，每一步都在学到的去噪模型条件下，采样一个噪声稍少的版本。

## 动手实现

### Step 1：Uniform and inverse CDF sampling

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

生成 10,000 个 exponential samples，并验证均值是 1/lambda。

### Step 2：Rejection sampling

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

使用 rejection sampling 从 truncated normal distribution 中抽样。通过绘制样本直方图验证形状。

### Step 3：Importance sampling

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

使用 uniform proposal 估计 normal distribution 下的 E[X^2]。与已知答案（mu^2 + sigma^2）比较。

### Step 4：Monte Carlo estimation of pi

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### Step 5：Metropolis-Hastings MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

从 bimodal distribution（两个 Gaussian 的 mixture）中采样。可视化这条链的轨迹。

### Step 6：Gibbs sampling

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### Step 7：Temperature sampling

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

展示 temperature 如何改变一组 token logits 的输出分布。

### Step 8：Top-k and top-p sampling

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### Step 9：Reparameterization trick

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

演示梯度可以穿过 reparameterized sample，但不能穿过 direct sampling。

### Step 10：Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

展示降低 temperature 如何让输出接近 one-hot vector。

包含所有可视化的完整实现位于 `code/sampling.py`。

## 实际使用

使用 NumPy 和 SciPy 时，生产版本如下：

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```

对于大规模 MCMC，可以使用专门的库：
- PyMC：使用 NUTS（adaptive HMC）进行完整的 Bayesian modeling
- emcee：ensemble MCMC sampler
- NumPyro/JAX：GPU 加速的 MCMC

你已经从零构建过这些方法。现在你知道库调用背后在做什么了。

## 练习

1. 为 Cauchy distribution 实现 inverse CDF sampling。CDF 是 F(x) = 0.5 + arctan(x)/pi。生成 10,000 个样本，并把直方图与真实 PDF 画在一起。注意它的 heavy tails（远离中心的极端值）。

2. 使用 rejection sampling 从 Beta(2, 5) distribution 中生成样本，proposal 使用 Uniform(0, 1)。把接受的样本与真实 Beta PDF 画在一起。理论接受率是多少？

3. 使用 Monte Carlo，用 1,000、10,000 和 100,000 个样本估计 sin(x) 从 0 到 pi 的积分。比较每个级别的误差。验证误差按 O(1/sqrt(N)) 缩放。

4. 实现 Metropolis-Hastings，从二维分布 p(x, y) 中采样，其中 p(x, y) 正比于 exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2)。绘制样本和链轨迹。尝试不同的 proposal standard deviation。

5. 构建一个完整的文本生成 demo：给定一个包含 10 个词的词表和对应 logits，分别使用 (a) greedy、(b) temperature=0.7、(c) top-k=3、(d) top-p=0.9 生成长度为 20 个 token 的序列。比较 5 次运行之间的输出多样性。

## 关键术语

| 术语 | 人们常说 | 它真正的含义 |
|------|----------|--------------|
| Sampling | “抽取随机值” | 按照概率分布生成值。所有生成式 AI 背后的机制 |
| Uniform distribution | “所有都一样可能” | [a, b] 中每个值都有相同概率密度 1/(b-a)。所有采样方法的起点 |
| Inverse CDF | “概率变换” | F_inverse(U) 把 uniform sample 转换成来自任何已知 CDF 分布的样本。精确且高效 |
| Rejection sampling | “提议并接受/拒绝” | 从简单 proposal 中生成样本，并以与 target/proposal 比例成正比的概率接受。精确，但会浪费样本 |
| Importance sampling | “重新加权样本” | 使用来自 q(x) 的样本，并用 p(x)/q(x) 为每个样本加权，估计 p(x) 下的期望。是 RL 中 PPO 的核心 |
| Monte Carlo | “平均随机样本” | 把积分近似为样本平均。无论维度如何，误差都是 O(1/sqrt(N)) |
| MCMC | “会收敛的随机游走” | 构造一条 stationary distribution 为目标分布的 Markov chain。Metropolis-Hastings 是基础算法 |
| Metropolis-Hastings | “接受上坡，有时也接受下坡” | 提议移动，并基于密度比接受。Detailed balance 保证收敛到目标分布 |
| Gibbs sampling | “一次一个变量” | 固定其他变量，从每个变量的条件分布中更新它。接受率 100% |
| Temperature | “置信度旋钮” | 在 softmax 之前用 T 除 logits。T<1 会锐化（更自信），T>1 会展平（更多样） |
| Top-k sampling | “保留最好的 k 个” | 把除 k 个最高概率 token 之外的所有 token 清零，重新归一化，然后采样。候选集合大小固定 |
| Nucleus sampling (top-p) | “保留可能的那些” | 保留累计概率超过 p 的最小 token 集合。候选集合大小自适应 |
| Reparameterization trick | “把随机性移到外面” | 写成 z = mu + sigma * epsilon，其中 epsilon ~ N(0,1)。让采样变得可微。对 VAE 训练至关重要 |
| Gumbel-Softmax | “软 categorical sampling” | 使用 Gumbel noise + 带 temperature 的 softmax，对 categorical sampling 做可微近似 |
| Stratified sampling | “强制覆盖” | 把样本空间划分成 strata，并从每个 strata 中采样。方差总是低于 naive Monte Carlo |
| Burn-in | “预热期” | 在链到达 stationary distribution 之前丢弃的初始 MCMC 样本 |
| Detailed balance | “可逆性条件” | p(x) * T(x->y) = p(y) * T(y->x)。这是 p 成为 Markov chain stationary distribution 的充分条件 |
| Diffusion sampling | “迭代去噪” | 从噪声开始，通过学到的去噪步骤生成数据。每一步都是一次条件采样操作 |

## 延伸阅读

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - MCMC 基础的详细教程
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - Gumbel-Softmax 原始论文
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - nucleus（top-p）sampling 论文
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - 引入 reparameterization trick 的 VAE 论文
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - DDPM 将采样连接到图像生成
