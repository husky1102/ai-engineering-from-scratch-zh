# 开放词表视觉——CLIP

> 一起训练 image encoder 和 text encoder，让匹配的 (image, caption) pairs 落到 shared space 中同一点附近。这就是全部技巧。

**类型:** Build + Use
**语言:** Python
**先修:** Phase 4 Lesson 14 (ViT), Phase 4 Lesson 17 (Self-Supervised)
**时间:** ~45 minutes

## 学习目标

- 解释 CLIP 的 two-tower architecture 和 contrastive training objective
- 使用 pretrained CLIP（或 SigLIP）做 zero-shot classification，不需要任何 task-specific training
- 从零实现 zero-shot classification：encode class prompts、计算 cosine similarity、取 argmax
- 区分 CLIP、SigLIP、OpenCLIP 和 LLaVA/LLaMA-vision models——2026 年各自适合什么

## 要解决的问题

传统 classifiers 是 closed-vocabulary：一个 1000-class ImageNet model 只能预测 1000 个 labels。每个新 category 都需要 labelled data 和 retrained head。

CLIP（Radford et al., OpenAI 2021）展示了：在从 web 抓取的 400M (image, caption) pairs 上训练，可以产生一个 inference 时能分类到任意 category set 的模型，而 categories 只需用 natural language 描述。你通过写一个 sentence 就能给它一个新 class。

这种能力——zero-shot transfer——就是每个现代 vision system 都从 CLIP-family checkpoint 开始的原因。Detection（Grounding DINO、OWL-ViT）、segmentation（CLIPSeg、SAM）、retrieval、content moderation、VLMs 和 text-to-image generation 都建立在 CLIP-style joint embeddings 之上。

## 核心概念

### Two towers

```mermaid
flowchart LR
    IMG["Image"] --> IENC["Image encoder<br/>(ViT-L/14)"] --> IEMB["Image embedding<br/>(1024,)"]
    TXT["Caption"] --> TENC["Text encoder<br/>(transformer)"] --> TEMB["Text embedding<br/>(1024,)"]
    IEMB --> SIM["Cosine similarity"]
    TEMB --> SIM

    style IENC fill:#dbeafe,stroke:#2563eb
    style TENC fill:#fef3c7,stroke:#d97706
    style SIM fill:#dcfce7,stroke:#16a34a
```

两个 encoders 末尾都有一个 linear projection，投到相同 embedding dimension（CLIP-B/32 为 512，CLIP-L/14 为 1024）。L2-normalise 后计算 cosine similarity。

### Objective

给定一批 N 个 (image, caption) pairs，构建 NxN similarity matrix。训练两个 encoders，使 diagonal（matching pairs）有高 similarity，off-diagonals（non-matching）有低 similarity。

```text
sim_matrix = image_embeddings @ text_embeddings.T / tau

loss_i2t = cross_entropy(sim_matrix,       targets=arange(N))
loss_t2i = cross_entropy(sim_matrix.T,     targets=arange(N))
loss = (loss_i2t + loss_t2i) / 2
```

它是 symmetric，因为 image-to-text 和 text-to-image retrieval 都应该有效。`tau`（temperature）通常作为 scalar parameter 学习，初始化为 0.07。

### SigLIP：更好的 loss

SigLIP（Zhai et al., 2023）用 per-pair sigmoid 替换了 softmax：

```text
loss = mean over pairs of log(1 + exp(-y_ij * sim_ij))
y_ij = +1 if matching, -1 otherwise
```

Per-pair loss 移除了 CLIP 要求的 batch-level normalisation。SigLIP 在 small batch sizes 下训练更好，并且在相同数据规模下达到或超过 CLIP。

### Zero-shot classification

给定 trained CLIP：

1. 为每个 class 组合一个 prompt："a photo of a {class}"。
2. 用 text encoder encode 所有 class prompts -> `T` shape (C, d)。
3. encode test image -> `I` shape (1, d)。
4. Similarity = `I @ T.T` shape (1, C)。
5. Argmax -> predicted class。

Prompt engineering 很重要。OpenAI 为 ImageNet 发布了 80 个 prompt templates（"a photo of a {}"、"a blurry photo of a {}"、"a sketch of a {}"、...）。对每个 class 平均所有 templates 的 embeddings，可以额外提升 1-3% top-1 accuracy。

### 2026 年 CLIP-style models 用在哪里

- **Zero-shot classification**——直接使用。
- **Image retrieval**——一次性 encode 所有 images，inference 时 embed query。
- **Text-conditioned detection**——Grounding DINO、OWL-ViT 用 CLIP text tower 包住 detector。
- **Text-conditioned segmentation**——CLIPSeg；SAM 通过 CLIP 使用 text-prompt inputs。
- **VLMs**——LLaVA、Qwen-VL、InternVL 将 CLIP-family vision encoder 接入 LLM。
- **Text-to-image gen**——Stable Diffusion、DALL-E 3 以 CLIP text embeddings 为条件。

一旦你有了 shared embedding space，每个 vision+language task 都会变成 distance computation。

## 动手实现

### Step 1: Tiny two-tower model

真实 CLIP 是 ViT + transformer。本课为了让训练信号在 CPU 上可见，towers 是作用于 pre-extracted features 的 small MLPs。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTower(nn.Module):
    def __init__(self, img_in=128, txt_in=64, emb=64):
        super().__init__()
        self.image_proj = nn.Sequential(nn.Linear(img_in, 128), nn.ReLU(), nn.Linear(128, emb))
        self.text_proj = nn.Sequential(nn.Linear(txt_in, 128), nn.ReLU(), nn.Linear(128, emb))
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)  # ln(1/0.07)

    def forward(self, img_feats, txt_feats):
        i = F.normalize(self.image_proj(img_feats), dim=-1)
        t = F.normalize(self.text_proj(txt_feats), dim=-1)
        return i, t, self.logit_scale.exp()
```

两个 projections，shared-dim output，learned temperature。shape 与真实 CLIP API 相同。

### Step 2: Contrastive loss

```python
def clip_loss(image_emb, text_emb, logit_scale):
    N = image_emb.size(0)
    sim = logit_scale * image_emb @ text_emb.T
    targets = torch.arange(N, device=sim.device)
    l_i = F.cross_entropy(sim, targets)
    l_t = F.cross_entropy(sim.T, targets)
    return (l_i + l_t) / 2
```

Symmetric。更高 logit_scale = 更尖锐 softmax = 更 confident，但也有 instability 风险。

### Step 3: Zero-shot classifier

```python
@torch.no_grad()
def zero_shot_classify(model, image_feats, class_text_feats, class_names):
    """
    image_feats:      (N, img_in)
    class_text_feats: (C, txt_in)   one averaged embedding per class
    """
    i = F.normalize(model.image_proj(image_feats), dim=-1)
    t = F.normalize(model.text_proj(class_text_feats), dim=-1)
    sim = i @ t.T
    pred = sim.argmax(dim=-1)
    return [class_names[p] for p in pred.tolist()]
```

每一步一行。这正是 production CLIP checkpoint 使用的 zero-shot procedure。

### Step 4: Sanity check

```python
torch.manual_seed(0)
model = TwoTower()

img = torch.randn(8, 128)
txt = torch.randn(8, 64)
i, t, scale = model(img, txt)
loss = clip_loss(i, t, scale)
print(f"batch size: {i.size(0)}   loss: {loss.item():.3f}")
```

随机初始化模型的 loss 应接近 `log(N) = log(8) = 2.08`——这是尚未学到结构时的 symmetric cross-entropy target。

## 实际使用

OpenCLIP 是 2026 年的 community default：

```python
import open_clip
import torch
from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-B-32")

image = preprocess(Image.open("dog.jpg")).unsqueeze(0)
text = tokenizer(["a photo of a dog", "a photo of a cat", "a photo of a car"])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print(probs)
```

SigLIP 更新，在小规模训练上更好，并且是新工作的首选：`google/siglip-base-patch16-224`。Hugging Face 同时提供二者。

## 交付成果

本课产出：

- `outputs/prompt-zero-shot-class-picker.md`——一个 prompt，给定 classes 列表和 domain，为 zero-shot CLIP 设计 class templates。
- `outputs/skill-image-text-retriever.md`——一个 skill，用任意 CLIP checkpoint 构建 image embedding index，支持 query-by-text 和 query-by-image。

## 练习

1. **(Easy)** 使用 pretrained OpenCLIP ViT-B/32，在 CIFAR-10 上用 80-template prompt set 做 zero-shot classification。报告 top-1 accuracy；应该在 85-90% 左右。
2. **(Medium)** 在同一个 CIFAR-10 task 上比较 single-template（"a photo of a {}"）与 80-template averaged embeddings。量化差距并解释为什么 templates 有帮助。
3. **(Hard)** 构建 zero-shot image retrieval index：用 CLIP embed 1,000 images，构建 FAISS index，用 natural language description 查询。为你手写的 20 个 held-out queries 报告 retrieval recall@5。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Two-tower | “Dual encoder” | 分离的 image 和 text encoders，末尾接 shared-dim projection head |
| Zero-shot | “No task-specific training” | inference 时分类到只由 text 描述的 classes；不触碰 labels |
| Temperature / logit_scale | “tau” | softmax 前缩放 similarity matrix 的 learned scalar |
| Prompt template | “A photo of a {}” | class names 外面的 natural-language wrapper；平均多个 templates 会提升 zero-shot accuracy |
| CLIP | “Image+text model” | 2021 年 OpenAI 模型；2026 年这一领域的通用词汇 |
| SigLIP | “Sigmoid CLIP” | 用 per-pair sigmoid 替换 softmax；在小 batches 上训练更好 |
| OpenCLIP | “Open reproduction” | 在 LAION 上由社区训练的 CLIP variants；open-source pipelines 的 production default |
| VLM | “Vision-language model” | CLIP-family encoder 加 LLM，训练后回答关于 images 的问题 |

## 延伸阅读

- [CLIP: Learning Transferable Visual Models from Natural Language Supervision (Radford et al., 2021)](https://arxiv.org/abs/2103.00020)
- [SigLIP: Sigmoid Loss for Language-Image Pre-Training (Zhai et al., 2023)](https://arxiv.org/abs/2303.15343)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)——community codebase
- [DINOv2 vs CLIP vs MAE: a features comparison](https://huggingface.co/blog/dinov2)——HF guide，包含 side-by-side use cases
