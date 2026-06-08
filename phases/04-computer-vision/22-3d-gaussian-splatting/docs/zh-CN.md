# 从零实现 3D Gaussian Splatting

> 场景是一团由数百万个 3D Gaussian 组成的云。每一个 Gaussian 都有位置、朝向、尺度、不透明度，以及一个依赖观察方向的颜色。把它们 rasterise，再反向传播穿过 rasterisation，就完成了。

**类型:** Build
**语言:** Python
**先修:** Phase 4 Lesson 13 (3D Vision & NeRF), Phase 1 Lesson 12 (Tensor Operations), Phase 4 Lesson 10 (Diffusion basics optional)
**时间:** ~90 minutes

## 学习目标

- 解释为什么到 2026 年，3D Gaussian Splatting 已经取代 NeRF，成为照片级真实 3D 重建的生产默认方案
- 说出每个 Gaussian 的六类参数（position、rotation quaternion、scale、opacity、spherical harmonics colour、optional feature），以及每类贡献多少个浮点数
- 从零实现一个使用 `alpha` compositing 的 2D Gaussian splatting rasterizer，然后说明 3D 情况如何投影到同一个循环
- 使用 `nerfstudio`、`gsplat` 或 `SuperSplat` 从 20-50 张照片重建一个场景，并导出到 `KHR_gaussian_splatting` glTF 扩展或 OpenUSD 26.03 的 `UsdVolParticleField3DGaussianSplat` schema

## 要解决的问题

NeRF 把场景存成一个 MLP 的权重。每个渲染像素都要沿着一条 ray 查询数百次 MLP。训练要花数小时，渲染要花数秒，而且权重无法直接编辑：如果你想移动场景里的一把椅子，就必须重新训练。

3D Gaussian Splatting（Kerbl、Kopanas、Leimkühler、Drettakis，SIGGRAPH 2023）替代了这一整套做法。场景是一个显式的 3D Gaussians 集合。渲染是在 GPU 上以 100+ fps 运行的 rasterisation。训练只需要几分钟。编辑是直接的：平移一部分 Gaussians，就等于移动了那把椅子。到 2026 年，Khronos Group 已经批准了用于 Gaussian splats 的 glTF 扩展，OpenUSD 26.03 发布了 Gaussian splat schema，Zillow 和 Apartments.com 用它们渲染房产，而大多数新的 3D 重建研究论文都是核心 3DGS 思想的变体。

心理模型很简单，但数学里有足够多的活动部件，以至于大多数介绍会从 rasterisation 开始，然后跳过投影和 spherical harmonics。本课会把整件事搭完整：先做 2D 版本，再扩展到 3D。

## 核心概念

### 一个 Gaussian 携带什么

一个 3D Gaussian 是空间中的参数化 blob，包含这些属性：

```text
position         mu         (3,)    centre in world coordinates
rotation         q          (4,)    unit quaternion encoding orientation
scale            s          (3,)    log-scales per axis (exponentiated at render time)
opacity          alpha      (1,)    post-sigmoid opacity [0, 1]
SH coefficients  c_lm       (3 * (L+1)^2,)   view-dependent colour
```

Rotation + scale 构成一个 3x3 covariance：`Sigma = R S S^T R^T`。这就是 Gaussian 在 3D 中的形状。Spherical harmonics 让颜色随观察方向变化：specular highlights、细微的 sheen、view-dependent glow，而不必存储每个视角的 textures。使用 SH degree 3 时，每个颜色通道有 16 个 coefficients，也就是每个 Gaussian 仅颜色就需要 48 个 floats。

一个场景通常有 1-5 million 个 Gaussians。每个大约存 60 个 floats（3 + 4 + 3 + 1 + 48 + misc）。一个五百万 Gaussian 的场景大约是 240 MB，比带 per-point texture 的等价 point cloud 小得多，也比高分辨率重渲染的 NeRF MLP weights 小一个数量级。

### Rasterisation，而不是 ray marching

```mermaid
flowchart LR
    SCENE["Millions of 3D Gaussians<br/>(position, rotation, scale,<br/>opacity, SH colour)"] --> PROJ["Project to 2D<br/>(camera extrinsics + intrinsics)"]
    PROJ --> TILES["Assign to tiles<br/>(16x16 screen-space)"]
    TILES --> SORT["Depth-sort<br/>per tile"]
    SORT --> ALPHA["Alpha-composite<br/>front-to-back"]
    ALPHA --> PIX["Pixel colour"]

    style SCENE fill:#dbeafe,stroke:#2563eb
    style ALPHA fill:#fef3c7,stroke:#d97706
    style PIX fill:#dcfce7,stroke:#16a34a
```

五个步骤，全都对 GPU 友好。没有每个 pixel 一次 MLP query。单块 RTX 3080 Ti 可以用 147 fps 渲染 6 million splats。

### 投影步骤

世界位置为 `mu`、3D covariance 为 `Sigma` 的 3D Gaussian，会投影成屏幕位置为 `mu'`、2D covariance 为 `Sigma'` 的 2D Gaussian：

```text
mu' = project(mu)
Sigma' = J W Sigma W^T J^T          (2 x 2)

W = viewing transform (rotation + translation of camera)
J = Jacobian of the perspective projection at mu'
```

2D Gaussian 的 footprint 是一个 ellipse，ellipse 的轴是 `Sigma'` 的 eigenvectors。该 ellipse 内的每个 pixel 都会收到这个 Gaussian 的贡献，权重为 `exp(-0.5 * (p - mu')^T Sigma'^-1 (p - mu'))`。

### Alpha-compositing 规则

对一个 pixel 来说，覆盖它的 Gaussians 会按 back-to-front 排序（或等价地按 front-to-back 排序并使用反向公式）。颜色使用与 1980 年代以来所有半透明 rasteriser 相同的方程进行 compositing：

```text
C_pixel = sum_i alpha_i * T_i * c_i

T_i = prod_{j < i} (1 - alpha_j)       transmittance up to i
alpha_i = opacity_i * exp(-0.5 * d^T Sigma'^-1 d)   local contribution
c_i = eval_SH(SH_i, view_direction)    view-dependent colour
```

这就是 **NeRF volumetric render 的同一个方程**，只是这里遍历的是显式稀疏的 Gaussians 集合，而不是沿 ray 的密集 samples。这个等价关系解释了为什么渲染质量能匹配 NeRF：两者都在积分同一个 radiance-field equation。

### 为什么它可微

每一步：projection、tile assignment、alpha compositing、SH evaluation，都对 Gaussian parameters 可微。给定一张 ground-truth image，计算 rendered pixel loss，反向传播穿过 rasteriser，再用 gradient descent 更新所有 `(mu, q, s, alpha, c_lm)`。经过约 30,000 次 iterations，Gaussians 会找到合适的位置、尺度和颜色。

### Densification 与 pruning

固定数量的 Gaussians 覆盖不了复杂场景。训练包含两个自适应机制：

- 当某个 Gaussian 的 gradient magnitude 很高但 scale 很小时，在当前位置 **Clone** 它：重建在这里需要更多细节。
- 当某个大 scale Gaussian 的 gradient 很高时，把它 **Split** 成两个更小的 Gaussians：一个大的 Gaussian 对这个区域来说太平滑。
- **Prune** opacity 低于阈值的 Gaussians：它们没有贡献。

Densification 每 N 次 iterations 运行一次。一个场景通常从约 100k 个初始 Gaussians（由 SfM points 播种）增长到训练结束时的 1-5M。

### 用一段话理解 spherical harmonics

View-dependent colour 是单位球面上的函数 `c(direction)`。Spherical harmonics 是球面上的 Fourier basis。截断到 degree `L`，每个 channel 会有 `(L+1)^2` 个 basis functions。为一个新视角评估颜色，就是把学到的 SH coefficients 与观察方向处的 basis 做 dot product。Degree 0 = 一个 coefficient = 常量颜色。Degree 3 = 16 个 coefficients = 足以捕捉 Lambertian shading、specular 和轻微 reflection。3D Gaussian Splatting 论文默认使用 degree 3。

### 2026 年生产栈

```text
1. Capture         smartphone / DJI drone / handheld scanner
2. SfM / MVS       COLMAP or GLOMAP derives camera poses + sparse points
3. Train 3DGS      nerfstudio / gsplat / inria official / PostShot (~10-30 min on RTX 4090)
4. Edit            SuperSplat / SplatForge (clean floaters, segment)
5. Export          .ply -> glTF KHR_gaussian_splatting or .usd (OpenUSD 26.03)
6. View            Cesium / Unreal / Babylon.js / Three.js / Vision Pro
```

### 4D 与生成式变体

- **4D Gaussian Splatting**：Gaussians 是时间的函数；用于 volumetric video（Superman 2026、A$AP Rocky 的 "Helicopter"）。
- **Generative splats**：text-to-splat models（World Labs 的 Marble），可以幻觉生成整个场景。
- **3D Gaussian Unscented Transform**：NVIDIA NuRec 用于 autonomous driving simulation 的变体。

## 动手实现

### Step 1: 一个 2D Gaussian

我们先构建一个 2D rasteriser。投影之后，3D 情况会归约到它。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def eval_2d_gaussian(means, covs, points):
    """
    means:  (G, 2)      centres
    covs:   (G, 2, 2)   covariance matrices
    points: (H, W, 2)   pixel coordinates
    returns: (G, H, W)  density at every pixel for every Gaussian
    """
    G = means.size(0)
    H, W, _ = points.shape
    flat = points.view(-1, 2)
    inv = torch.linalg.inv(covs)
    diff = flat[None, :, :] - means[:, None, :]
    d = torch.einsum("gpi,gij,gpj->gp", diff, inv, diff)
    density = torch.exp(-0.5 * d)
    return density.view(G, H, W)
```

`einsum` 为每个 (Gaussian, pixel) pair 计算 quadratic form `diff^T Sigma^-1 diff`。

### Step 2: 2D splatting rasteriser

Front-to-back alpha-compositing。Depth 在 2D 中没有意义，所以我们使用一个每 Gaussian 学到的 scalar 来表示顺序。

```python
def rasterise_2d(means, covs, colours, opacities, depths, image_size):
    """
    means:     (G, 2)
    covs:      (G, 2, 2)
    colours:   (G, 3)
    opacities: (G,)     in [0, 1]
    depths:    (G,)     per-Gaussian scalar used for ordering
    image_size: (H, W)
    returns:   (H, W, 3) rendered image
    """
    H, W = image_size
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=means.device),
        torch.arange(W, dtype=torch.float32, device=means.device),
        indexing="ij",
    )
    points = torch.stack([xx, yy], dim=-1)

    densities = eval_2d_gaussian(means, covs, points)
    alphas = opacities[:, None, None] * densities
    alphas = alphas.clamp(0.0, 0.99)

    order = torch.argsort(depths)
    alphas = alphas[order]
    colours_sorted = colours[order]

    T = torch.ones(H, W, device=means.device)
    out = torch.zeros(H, W, 3, device=means.device)
    for i in range(means.size(0)):
        a = alphas[i]
        out += (T * a)[..., None] * colours_sorted[i][None, None, :]
        T = T * (1.0 - a)
    return out
```

它不快：真实实现会使用 tile-based CUDA kernels，但数学完全正确，而且全程可微。

### Step 3: 可训练的 2D splat scene

```python
class Splats2D(nn.Module):
    def __init__(self, num_splats=128, image_size=64, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        H, W = image_size, image_size
        self.means = nn.Parameter(torch.rand(num_splats, 2, generator=g) * torch.tensor([W, H]))
        self.log_scale = nn.Parameter(torch.ones(num_splats, 2) * math.log(2.0))
        self.rot = nn.Parameter(torch.zeros(num_splats))  # single angle in 2D
        self.colour_logits = nn.Parameter(torch.randn(num_splats, 3, generator=g) * 0.5)
        self.opacity_logit = nn.Parameter(torch.zeros(num_splats))
        self.depth = nn.Parameter(torch.rand(num_splats, generator=g))

    def covs(self):
        s = torch.exp(self.log_scale)
        c, si = torch.cos(self.rot), torch.sin(self.rot)
        R = torch.stack([
            torch.stack([c, -si], dim=-1),
            torch.stack([si, c], dim=-1),
        ], dim=-2)
        S = torch.diag_embed(s ** 2)
        return R @ S @ R.transpose(-1, -2)

    def forward(self, image_size):
        covs = self.covs()
        colours = torch.sigmoid(self.colour_logits)
        opacities = torch.sigmoid(self.opacity_logit)
        return rasterise_2d(self.means, covs, colours, opacities, self.depth, image_size)
```

`log_scale`、`opacity_logit` 和 `colour_logits` 都是 unconstrained parameters，会在 render time 映射到正确的 activation。这是每个 3DGS 实现里的标准模式。

### Step 4: 将 2D Gaussians 拟合到 target image

```python
import math
import numpy as np

def make_target(size=64):
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    img = np.zeros((size, size, 3), dtype=np.float32)
    # Red circle
    mask = (xx - 20) ** 2 + (yy - 20) ** 2 < 10 ** 2
    img[mask] = [1.0, 0.2, 0.2]
    # Blue square
    mask = (np.abs(xx - 45) < 8) & (np.abs(yy - 40) < 8)
    img[mask] = [0.2, 0.3, 1.0]
    return torch.from_numpy(img)


target = make_target(64)
model = Splats2D(num_splats=64, image_size=64)
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for step in range(200):
    pred = model((64, 64))
    loss = F.mse_loss(pred, target)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 40 == 0:
        print(f"step {step:3d}  mse {loss.item():.4f}")
```

经过 200 steps，64 个 Gaussians 会落到这两个形状里。这就是整个思想：对显式几何 primitives 做 gradient descent。

### Step 5: 从 2D 到 3D

3D 扩展保留同一个循环。新增内容是：

1. 每个 Gaussian 的 rotation 是 quaternion，而不是一个 angle。
2. Covariance 是 `R S S^T R^T`，其中 `R` 由 quaternion 构建，`S = diag(exp(log_scale))`。
3. Projection `(mu, Sigma) -> (mu', Sigma')` 使用 camera extrinsics，以及 `mu` 处 perspective projection 的 Jacobian。
4. Colour 变成 spherical-harmonics expansion；在 viewing direction 上评估它。
5. Depth-sort 来自真实 camera-space z，而不是学到的 scalar。

每个生产实现（`gsplat`、`inria/gaussian-splatting`、`nerfstudio`）都会在 GPU 上用 tile-based CUDA kernels 做同样的事。

### Step 6: Spherical harmonics evaluation

SH basis 到 degree 3 时，每个 channel 有 16 terms。Evaluation：

```python
def eval_sh_degree_3(sh_coeffs, dirs):
    """
    sh_coeffs: (..., 16, 3)   last dim is RGB channels
    dirs:      (..., 3)       unit vectors
    returns:   (..., 3)
    """
    C0 = 0.282094791773878
    C1 = 0.488602511902920
    C2 = [1.092548430592079, 1.092548430592079,
          0.315391565252520, 1.092548430592079,
          0.546274215296039]
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    x2, y2, z2 = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    result = C0 * sh_coeffs[..., 0, :]
    result = result - C1 * y[..., None] * sh_coeffs[..., 1, :]
    result = result + C1 * z[..., None] * sh_coeffs[..., 2, :]
    result = result - C1 * x[..., None] * sh_coeffs[..., 3, :]

    result = result + C2[0] * xy[..., None] * sh_coeffs[..., 4, :]
    result = result + C2[1] * yz[..., None] * sh_coeffs[..., 5, :]
    result = result + C2[2] * (2.0 * z2 - x2 - y2)[..., None] * sh_coeffs[..., 6, :]
    result = result + C2[3] * xz[..., None] * sh_coeffs[..., 7, :]
    result = result + C2[4] * (x2 - y2)[..., None] * sh_coeffs[..., 8, :]

    # degree 3 terms omitted here for brevity; full 16-coefficient version in the code file
    return result
```

学到的 `sh_coeffs` 会存储该 Gaussian “每个方向上的颜色”。Render time 时，把它拿来对当前 view direction 求值，就得到一个 3-vector RGB。

## 实际使用

真实 3DGS 工作请使用 `gsplat`（Meta）或 `nerfstudio`：

```bash
pip install nerfstudio gsplat
ns-download-data example
ns-train splatfacto --data path/to/data
```

`splatfacto` 是 nerfstudio 的 3DGS trainer。对典型场景来说，一次 run 在 RTX 4090 上需要 10-30 minutes。

2026 年真正重要的 export options：

- `.ply`：raw Gaussian cloud（portable，文件最大）。
- `.splat`：PlayCanvas / SuperSplat quantised format。
- glTF `KHR_gaussian_splatting`：Khronos standard，可跨 viewers 移植（Feb 2026 RC）。
- OpenUSD `UsdVolParticleField3DGaussianSplat`：USD-native，用于 NVIDIA Omniverse 和 Vision Pro pipelines。

对 4D / dynamic scenes，`4DGS` 和 `Deformable-3DGS` 会用 time-varying means 和 opacities 扩展同一套机制。

## 交付成果

本课产出：

- `outputs/prompt-3dgs-capture-planner.md`：一个 prompt，用来为给定 scene type 规划 capture session（照片数量、camera path、lighting）。
- `outputs/skill-3dgs-export-router.md`：一个 skill，会根据 downstream viewer 或 engine 选择正确的 export format（`.ply` / `.splat` / glTF / USD）。

## 练习

1. **(Easy)** 在另一个 synthetic image 上运行上面的 2D splat trainer。把 `num_splats` 在 `[16, 64, 256]` 中变化，并绘制每种设置下的 MSE vs step。找出收益递减的拐点。
2. **(Medium)** 扩展 2D rasteriser，让每个 Gaussian 的 RGB colours 可以通过 degree-2 harmonic 依赖一个 scalar "view angle"。在一对 target images 上训练，并验证模型能重建两张图。
3. **(Hard)** Clone `nerfstudio`，在你拥有的任意场景（desk、plant、face、room）20-photo capture 上训练 `splatfacto`。导出到 glTF `KHR_gaussian_splatting`，并在一个 viewer（Three.js `GaussianSplats3D`、SuperSplat、Babylon.js V9）中打开。报告 training time、Gaussians 数量和 rendered fps。

## 关键术语

| Term | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| 3DGS | "Gaussian splats" | 用数百万个 3D Gaussians 表示的显式场景，每个 Gaussian 有 position、rotation、scale、opacity、SH colour |
| Covariance | "Shape of the Gaussian" | `Sigma = R S S^T R^T`；一个 Gaussian 的 orientation 和 anisotropic scale |
| Alpha compositing | "Back-to-front blend" | 与 NeRF volumetric render 相同的方程，但现在作用在显式稀疏集合上 |
| Densification | "Clone and split" | 在重建 under-fit 的地方自适应加入新的 Gaussians |
| Pruning | "Delete low-opacity" | 移除训练期间已经塌缩到 near-zero opacity 的 Gaussians |
| Spherical harmonics | "View-dependent colour" | 球面上的 Fourier basis；把颜色存成 viewing direction 的函数 |
| Splatfacto | "nerfstudio's 3DGS" | 2026 年训练 3DGS 最容易的路径 |
| `KHR_gaussian_splatting` | "glTF standard" | Khronos 2026 extension，让 3DGS 能跨 viewers 和 engines 移植 |

## 延伸阅读

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering (Kerbl et al., SIGGRAPH 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)：原始论文
- [gsplat (Meta/nerfstudio)](https://github.com/nerfstudio-project/gsplat)：production-quality CUDA rasteriser
- [nerfstudio Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html)：reference training recipe
- [Khronos KHR_gaussian_splatting extension](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_gaussian_splatting/README.md)：2026 年 portable format
- [OpenUSD 26.03 release notes](https://openusd.org/release/)：`UsdVolParticleField3DGaussianSplat` schema
- [THE FUTURE 3D State of Gaussian Splatting 2026](https://www.thefuture3d.com/blog-0/2026/4/4/state-of-gaussian-splatting-2026)：industry overview
