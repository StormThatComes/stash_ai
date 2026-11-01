"""Inference-only PMPose-B, matching the pinned upstream state dictionary.

Architecture follows Miroslav Purkrabek's PMPose MultiHead (GPL-3.0) and
OpenMMLab's ViT/FFN (Apache-2.0). See THIRD_PARTY_NOTICES.md. Training,
OpenMMLab registries and segmentation are deliberately outside this adapter.
"""
import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(768, 2304)
        self.proj = nn.Linear(768, 768)

    def forward(self, x):
        batch, count, _ = x.shape
        q, k, v = self.qkv(x).reshape(batch, count, 3, 12, 64).permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
        return self.proj(x.transpose(1, 2).reshape(batch, count, 768))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(768, eps=1e-6), nn.LayerNorm(768, eps=1e-6)
        self.attn = Attention()
        self.ffn = nn.Module()
        self.ffn.layers = nn.Sequential(nn.Sequential(nn.Linear(768, 3072), nn.GELU()), nn.Linear(3072, 768))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn.layers(self.ln2(x))


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.projection = nn.Conv2d(3, 768, kernel_size=16, stride=16, padding=2)
        self.pos_embed = nn.Parameter(torch.zeros(1, 192, 768))
        self.layers = nn.Sequential(*(Block() for _ in range(12)))
        self.ln1 = nn.LayerNorm(768, eps=1e-6)

    def forward(self, x):
        x = self.patch_embed.projection(x).flatten(2).transpose(1, 2)
        x = self.ln1(self.layers(x + self.pos_embed))
        return x.transpose(1, 2).reshape(-1, 768, 16, 12)


def sparsemax(x):
    """Euclidean projection onto the probability simplex, per heatmap."""
    x = x - x.amax(dim=-1, keepdim=True)
    ordered = x.sort(dim=-1, descending=True).values
    cumulative = ordered.cumsum(-1) - 1
    rank = torch.arange(1, x.shape[-1] + 1, device=x.device, dtype=x.dtype)
    count = (ordered * rank > cumulative).sum(-1, keepdim=True)
    threshold = cumulative.gather(-1, count - 1) / count
    return (x - threshold).clamp(0., 1.)


def scalar_head(*, error=False):
    layers = []
    for kernel in ((4, 3), (2, 2), (2, 2)):
        layers.extend((nn.Conv2d(768, 768, 3, padding=1), nn.BatchNorm2d(768),
                       nn.MaxPool2d(kernel), nn.ReLU()))
    layers.extend((nn.Conv2d(768, 23, 1), nn.ReLU() if error else nn.Sigmoid()))
    return nn.Sequential(*layers)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(768, 256, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU())
        self.final_layer = nn.Conv2d(256, 23, 1)
        self.probability_layers = scalar_head()
        self.visibility_layers = scalar_head()
        self.oks_layers = scalar_head()
        # Retain the frozen, unused error head for strict checkpoint validation.
        self.error_layers = scalar_head(error=True)

    def forward(self, x):
        heatmaps = self.final_layer(self.deconv_layers(x))[:, :17]
        heatmaps = sparsemax(heatmaps.flatten(2) / .5).reshape(-1, 17, 64, 48)
        return (heatmaps, self.oks_layers(x)[:, :17, 0, 0],
                self.probability_layers(x)[:, :17, 0, 0], self.visibility_layers(x)[:, :17, 0, 0])


class PMPoseB(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone, self.head = Backbone(), Head()

    def forward(self, x):
        return self.head(self.backbone(x))
