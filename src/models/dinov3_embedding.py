# src/models/dinov3_embedding.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from transformers import DINOv3ViTModel


class DINOv3TeacherModel(nn.Module):
    """
    DINOv3-based teacher model for Pokemon card recognition.

    Architecture:
    - DINOv3-ViT-L/16 backbone (pretrained, partially frozen)
    - Projection head (trainable)
    - L2 normalization

    Fine-tuning strategy:
    - Epoch 1-5: Only projection head (backbone frozen)
    - Epoch 6+: Unfreeze last 4 transformer blocks
    """

    def __init__(
        self,
        model_name: str = 'dinov3_vitl16',
        embedding_dim: int = 768,
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 0,
    ):
        super().__init__()

        # Load DINOv3 backbone from HuggingFace
        self.backbone = DINOv3ViTModel.from_pretrained(f'facebook/{model_name}-pretrain-lvd1689m')

        backbone_dim = self.backbone.config.hidden_size  # 1024 for ViT-L

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(backbone_dim, embedding_dim),
        )

        # Freeze strategy
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if unfreeze_last_n_blocks > 0:
            # Unfreeze last N transformer blocks
            for block in self.backbone.encoder.layer[-unfreeze_last_n_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DINOv3 forward pass
        outputs = self.backbone(x)
        features = outputs.last_hidden_state[:, 0]  # CLS token, [B, 1024]

        # Project to embedding space
        embeddings = self.projection(features)  # [B, 768]

        # L2 normalize
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def unfreeze_backbone(self, last_n_blocks: int = 4):
        """Unfreeze last N transformer blocks for fine-tuning."""
        for block in self.backbone.encoder.layer[-last_n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
        print(f"Unfroze last {last_n_blocks} transformer blocks")


class ArcFaceLoss(nn.Module):
    """ArcFace loss for metric learning with 17,592 Pokemon card classes."""

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.5,
        scale: float = 64.0,
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.margin = margin
        self.scale = scale
        self.register_buffer('cos_m', torch.cos(torch.tensor(margin)))
        self.register_buffer('sin_m', torch.sin(torch.tensor(margin)))
        self.register_buffer('threshold', torch.cos(torch.tensor(torch.pi - margin)))
        self.register_buffer('mm', torch.sin(torch.tensor(torch.pi - margin)) * margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalize weight
        W = nn.functional.normalize(self.weight, p=2, dim=1)

        # Cosine similarity
        cosine = torch.mm(embeddings, W.t())
        sine = torch.sqrt(torch.clamp(1.0 - cosine ** 2, 1e-7, 1.0))

        # ArcFace formula
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(
            cosine > self.threshold,
            phi,
            cosine - self.mm
        )

        # One-hot
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        # Apply margin only to correct class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return nn.functional.cross_entropy(output, labels)
