"""
Vision encoder used to turn each agent's egocentric 224x224x3 RGB observation
(the 7x7 grid patch rendered at 32px/cell) into a fixed-size feature vector
that CommNet consumes as its per-agent input h_i^0.

We just treat the encoder as a black box "f_enc: image -> R^d" as requested.
ViT-Tiny (patch16, 224 input) is the natural choice since 224x224 is exactly
its native input resolution. We use `timm` if it's installed (recommended,
`pip install timm`), and fall back to a hand-rolled ViT-Tiny-shaped module
with the same interface if timm isn't available, so this file has no hard
dependency beyond torch.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn

try:
    import timm
    _HAS_TIMM = True
except ImportError:
    _HAS_TIMM = False


class _MiniViTTiny(nn.Module):
    """
    Minimal fallback ViT-Tiny (patch16/224) implementation, used only if
    `timm` isn't installed. Same shape conventions as timm's
    vit_tiny_patch16_224: embed_dim=192, depth=12, heads=3, mlp_ratio=4.
    Not pretrained -- just here so the rest of the pipeline is runnable
    end-to-end without extra dependencies.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0,
                 drop=0.0):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=drop,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        # x: (B, 3, 224, 224)
        x = self.patch_embed(x)                       # (B, embed_dim, 14, 14)
        x = x.flatten(2).transpose(1, 2)               # (B, 196, embed_dim)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)                 # (B, 197, embed_dim)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]                                 # CLS token, (B, embed_dim)


class ViTTinyEncoder(nn.Module):
    """
    f_enc: (B, 3, 224, 224) uint8/float image -> (B, out_dim) feature vector.

    This is the encoder that turns each agent's local 7x7-cell RGB render
    into the vector CommNet calls h_i^0 in the paper. It is shared across
    agents (single set of weights, applied per-agent), matching CommNet's
    parameter-sharing assumption.

    Args:
        out_dim: dimensionality of the feature fed into CommNet (paper's
            hidden size). ViT-Tiny's native embedding is 192-d; we project
            it to `out_dim` with a small head.
        pretrained: only used if timm is available.
        freeze_backbone: if True, backbone weights are frozen and only the
            projection head is trained (useful for sample efficiency early
            in RL training, since ViT features from scratch + RL is
            notoriously sample-inefficient).
    """

    def __init__(self, out_dim: int = 128, pretrained: bool = False,
                 freeze_backbone: bool = False):
        super().__init__()
        if _HAS_TIMM:
            self.backbone = timm.create_model(
                "vit_tiny_patch16_224",
                pretrained=pretrained,
                num_classes=0,  # return pooled features, no classifier head
            )
            backbone_dim = self.backbone.num_features
        else:
            self.backbone = _MiniViTTiny()
            backbone_dim = self.backbone.embed_dim

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        self.proj = nn.Sequential(
            nn.LayerNorm(backbone_dim),
            nn.Linear(backbone_dim, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    @staticmethod
    def preprocess(obs_uint8: torch.Tensor) -> torch.Tensor:
        """
        obs_uint8: (..., 224, 224, 3) uint8 tensor as returned by the env.
        Returns: (..., 3, 224, 224) float tensor in [0, 1].
        Kept as a static method so the same normalization is used both
        during rollout collection and training.
        """
        x = obs_uint8.float() / 255.0
        x = x.movedim(-1, -3)  # ... H W C -> ... C H W
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, 224, 224) float image, already preprocessed.
        Returns: (B, out_dim)
        """
        feats = self.backbone(x)
        return self.proj(feats)
