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
        from_checkpoint: bool = False,  # Skip HF download when loading from checkpoint
    ):
        super().__init__()

        # Load DINO backbone from HuggingFace (skip if loading from checkpoint)
        # Support both DINOv2 (public) and DINOv3 (official Meta models)
        model_name_hf = model_name.replace('_', '-')

        # DINOv2 models: facebook/dinov2-{small,base,large,giant}
        # DINOv3 models: facebook/dinov3-{model}-pretrain-lvd1689m (official Meta)
        if model_name_hf.startswith('dinov2'):
            model_id = f'facebook/{model_name_hf}'  # e.g., facebook/dinov2-large
        else:
            # Use official Facebook/Meta DINOv3 models
            model_id = f'facebook/{model_name_hf}-pretrain-lvd1689m'  # DINOv3

        import logging
        logger = logging.getLogger(__name__)

        if not from_checkpoint:
            logger.info(f"Loading model from HuggingFace: {model_id}")
            self.backbone = DINOv3ViTModel.from_pretrained(model_id)
            logger.info(f"✅ Successfully loaded backbone from {model_id}")
            backbone_dim = self.backbone.config.hidden_size  # 1024 for ViT-L, 4096 for ViT-7B
        else:
            # When loading from checkpoint, download ONLY the config (not weights)
            # This ensures we get the EXACT model structure that matches the checkpoint
            logger.info(f"Loading config from HuggingFace for checkpoint loading (model: {model_name})")
            from transformers import DINOv3ViTConfig

            # Download the config from HuggingFace (small JSON file, ~1KB)
            # This gives us the COMPLETE config including all parameters like:
            # - num_register_tokens (4 for DINOv3-L)
            # - intermediate_size (4096 for MLP layers)
            # - and dozens of other parameters
            logger.info(f"Downloading config from HuggingFace: {model_id}")
            config = DINOv3ViTConfig.from_pretrained(model_id)
            logger.info(f"✅ Downloaded complete config (hidden_size={config.hidden_size}, "
                       f"num_register_tokens={getattr(config, 'num_register_tokens', 0)}, "
                       f"intermediate_size={getattr(config, 'intermediate_size', 'N/A')})")

            # Create model structure with complete config (no weights downloaded)
            self.backbone = DINOv3ViTModel(config)
            backbone_dim = config.hidden_size
            logger.info(f"✅ Created model structure matching checkpoint")

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
            for block in self.backbone.layer[-unfreeze_last_n_blocks:]:
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
        for block in self.backbone.layer[-last_n_blocks:]:
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

    def get_logits(self, embeddings: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """
        Compute ArcFace logits for knowledge distillation.

        Args:
            embeddings: [B, embedding_dim] normalized embeddings
            labels: [B] ground truth labels (optional, only needed if applying margin)

        Returns:
            logits: [B, num_classes] logits for distillation
        """
        # Normalize weight
        W = nn.functional.normalize(self.weight, p=2, dim=1)

        # Cosine similarity
        cosine = torch.mm(embeddings, W.t())

        if labels is not None:
            # Apply ArcFace margin to ground truth class
            sine = torch.sqrt(torch.clamp(1.0 - cosine ** 2, 1e-7, 1.0))
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
        else:
            # No labels - just use cosine similarity (inference mode)
            output = cosine

        output *= self.scale
        return output

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute ArcFace loss."""
        logits = self.get_logits(embeddings, labels)
        return nn.functional.cross_entropy(logits, labels)
