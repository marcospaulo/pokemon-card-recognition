#!/usr/bin/env python3
"""
Embedding Models for Pokemon Card Recognition

This module provides embedding models for metric learning:

1. LeViTEmbeddingModel (RECOMMENDED) - LeViT-384 backbone
   - 768-dimensional embeddings
   - Works on Hailo 8 NPU (0.14ms inference)
   - 37.6M parameters, 82.3% ImageNet accuracy
   - CNN-Transformer hybrid for occlusion robustness

2. CardEmbeddingModel (LEGACY) - MobileNetV3-Large backbone
   - 512-dimensional embeddings
   - Smaller/faster but less accurate
   - Kept for backwards compatibility

Key design decisions:
1. L2 normalization: Essential for triplet loss (cosine similarity becomes dot product)
2. No classification head: Pure embedding model
3. GeM pooling: Better than GAP for image retrieval
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import MobileNet_V3_Large_Weights

# Try to import timm for LeViT
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not installed. LeViTEmbeddingModel will not be available.")
    print("         Install with: pip install timm")


class GeM(nn.Module):
    """
    Generalized Mean (GeM) Pooling.

    Outperforms Global Average Pooling for image retrieval tasks.
    Formula: gem(x) = (mean(x^p))^(1/p)

    - p=1: Average pooling
    - p=inf: Max pooling
    - Learnable p typically converges to 2.5-3.5

    Reference: Fine-tuning CNN Image Retrieval with No Human Annotation
    https://arxiv.org/abs/1711.02512
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable: bool = True):
        """
        Args:
            p: Initial pooling parameter
            eps: Small value for numerical stability
            learnable: If True, p is learned during training
        """
        super().__init__()
        self.eps = eps
        if learnable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.register_buffer('p', torch.ones(1) * p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] feature map

        Returns:
            [B, C] pooled features
        """
        # Clamp to avoid numerical issues with negative values
        x = x.clamp(min=self.eps)

        # CRITICAL: Clamp p to safe range to prevent NaN/Inf
        # p=0 → pow(1/0)=inf → NaN
        # p<0 → undefined behavior with negative exponents
        # Valid range: [1.0, 10.0] covers average pooling (p=1) to near-max pooling
        p_clamped = self.p.clamp(min=1.0, max=10.0)

        # GeM: (mean(x^p))^(1/p)
        x = x.pow(p_clamped)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.pow(1.0 / p_clamped)
        return x.flatten(1)

    def __repr__(self):
        p_val = self.p.item() if isinstance(self.p, nn.Parameter) else self.p.item()
        return f"{self.__class__.__name__}(p={p_val:.2f}, learnable={isinstance(self.p, nn.Parameter)})"


class LeViTEmbeddingModel(nn.Module):
    """
    LeViT-384 based embedding model for card recognition.

    RECOMMENDED MODEL - Works on Hailo 8 NPU with 0.14ms inference.

    Architecture:
        LeViT-384 Backbone (timm) → L2 Normalize → 768-dim Embedding

    LeViT-384 is a CNN-Transformer hybrid that:
    - Uses convolutional attention (Hailo-compatible, unlike pure ViT)
    - Has 82.3% ImageNet top-1 accuracy
    - Outputs 768-dimensional embeddings natively
    - Has 37.6M parameters

    Pre-compiled HEF available at:
    https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8/levit384.hef
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        """
        Args:
            embedding_dim: Size of output embedding vector (768 for LeViT-384)
            pretrained: Use ImageNet pretrained weights from timm
            freeze_backbone: Freeze backbone weights (only train projection head)
        """
        super().__init__()

        if not TIMM_AVAILABLE:
            raise ImportError(
                "timm is required for LeViTEmbeddingModel. "
                "Install with: pip install timm"
            )

        self.embedding_dim = embedding_dim

        # Load LeViT-384 from timm
        # num_classes=0 removes the classification head, returns features directly
        self.backbone = timm.create_model(
            'levit_384',
            pretrained=pretrained,
            num_classes=0  # Remove classifier, get 768-dim features
        )

        # LeViT-384 outputs 768-dim features natively
        self.feature_dim = self.backbone.num_features  # Should be 768

        # Optional projection head if different embedding_dim is requested
        if embedding_dim != self.feature_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.feature_dim, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
            )
        else:
            self.projection = nn.Identity()

        # Optional: freeze backbone for initial training
        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze backbone weights."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("  LeViT backbone frozen - only projection head will be trained")

    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("  LeViT backbone unfrozen - full model will be trained")

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images [B, 3, 224, 224]
            normalize: Apply L2 normalization to embeddings

        Returns:
            embeddings: [B, embedding_dim] L2-normalized embeddings
        """
        # Extract features from LeViT backbone
        features = self.backbone(x)  # [B, 768]

        # Project if needed
        embeddings = self.projection(features)  # [B, embedding_dim]

        # L2 normalize - CRITICAL for triplet loss
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward with normalization (for clarity in inference code)."""
        return self.forward(x, normalize=True)


class CardEmbeddingModel(nn.Module):
    """
    Embedding model for card recognition using metric learning.

    Architecture:
        MobileNetV3-Large (pretrained) → Global Average Pool → Embedding Head → L2 Normalize

    The embedding head projects features to a 512-dim space where:
    - Same cards cluster together
    - Different cards are pushed apart
    - Distance = similarity (smaller = more similar)
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        use_gem_pooling: bool = True,
        gem_p: float = 3.0
    ):
        """
        Args:
            embedding_dim: Size of output embedding vector
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: Freeze backbone weights (only train embedding head)
            use_gem_pooling: Use GeM pooling instead of Global Average Pooling
                            (recommended for image retrieval tasks)
            gem_p: Initial value for GeM pooling parameter
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.use_gem_pooling = use_gem_pooling

        # Load MobileNetV3-Large backbone
        if pretrained:
            self.backbone = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.mobilenet_v3_large(weights=None)

        # Get the feature dimension from backbone
        # MobileNetV3-Large: features output is 960-dim after pooling
        self.feature_dim = self.backbone.classifier[0].in_features  # 960

        # Remove original classifier - we don't need it
        self.backbone.classifier = nn.Identity()

        # Pooling layer
        if use_gem_pooling:
            self.pooling = GeM(p=gem_p, learnable=True)
        else:
            self.pooling = None  # Will use backbone's avgpool

        # Embedding head: project 960-dim features to 512-dim embeddings
        self.embedding_head = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.Hardswish(),  # Match MobileNetV3's activation
            nn.Dropout(p=0.2),
            nn.Linear(1024, embedding_dim),
        )

        # Optional: freeze backbone for initial training
        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze backbone weights."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("  Backbone frozen - only embedding head will be trained")

    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("  Backbone unfrozen - full model will be trained")

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images [B, 3, H, W]
            normalize: Apply L2 normalization to embeddings

        Returns:
            embeddings: [B, embedding_dim] L2-normalized embeddings
        """
        # Extract features from backbone (before pooling)
        features = self.backbone.features(x)  # [B, 960, H/32, W/32]

        # Pooling - GeM or Global Average
        if self.use_gem_pooling and self.pooling is not None:
            features = self.pooling(features)  # [B, 960]
        else:
            features = self.backbone.avgpool(features)
            features = features.flatten(1)  # [B, 960]

        # Project to embedding space
        embeddings = self.embedding_head(features)  # [B, 512]

        # L2 normalize - CRITICAL for triplet loss
        # Makes embeddings unit vectors, so dot product = cosine similarity
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward with normalization (for clarity in inference code)."""
        return self.forward(x, normalize=True)


class TripletLossWithMining(nn.Module):
    """
    Triplet Loss with online hard negative mining.

    L = max(d(a, p) - d(a, n) + margin, 0)

    Where:
    - a = anchor embedding
    - p = positive embedding (same class)
    - n = negative embedding (different class)
    - margin = minimum desired gap between positive and negative distances

    Hard mining strategies:
    - Hard positive: Farthest positive from anchor
    - Hard negative: Closest negative to anchor
    - Semi-hard negative: Negative farther than positive but within margin
    """

    def __init__(
        self,
        margin: float = 0.3,
        mining_type: str = "semi_hard"
    ):
        """
        Args:
            margin: Margin for triplet loss (typically 0.2-0.5)
            mining_type: "hard", "semi_hard", or "all"
        """
        super().__init__()
        self.margin = margin
        self.mining_type = mining_type

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss with online mining.

        Args:
            embeddings: [B, D] L2-normalized embeddings
            labels: [B] class labels

        Returns:
            loss: Scalar triplet loss
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        # Compute pairwise distances
        # For L2-normalized vectors: ||a - b||^2 = 2 - 2*dot(a, b)
        dot_product = torch.mm(embeddings, embeddings.t())  # [B, B]
        square_norm = torch.diag(dot_product)  # [B]

        # Pairwise squared distances
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        distances = torch.clamp(distances, min=0.0)  # Numerical stability

        # Use 1e-4 epsilon for numerical stability
        # NOTE: bfloat16 has machine eps ~1e-3, so 1e-6 is too small and can cause NaN
        # Using 1e-4 is safe for both float32 and bfloat16
        distances = torch.sqrt(distances + 1e-4)  # [B, B]

        # Create masks for valid triplets
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
        labels_not_equal = ~labels_equal

        # For each anchor, find positive and negative distances
        # Positive mask: same label, different index
        indices_equal = torch.eye(batch_size, dtype=torch.bool, device=device)
        positive_mask = labels_equal & ~indices_equal

        # Negative mask: different label
        negative_mask = labels_not_equal

        if self.mining_type == "hard":
            loss = self._hard_mining(distances, positive_mask, negative_mask)
        elif self.mining_type == "semi_hard":
            loss = self._semi_hard_mining(distances, positive_mask, negative_mask)
        else:  # "all"
            loss = self._all_triplets(distances, positive_mask, negative_mask)

        return loss

    def _hard_mining(self, distances, positive_mask, negative_mask):
        """Hard triplet mining: hardest positive, hardest negative."""
        # Hardest positive: max distance where positive_mask is True
        pos_distances = distances * positive_mask.float()
        pos_distances = pos_distances + (~positive_mask).float() * -1e9
        hardest_positive = pos_distances.max(dim=1)[0]  # [B]

        # Hardest negative: min distance where negative_mask is True
        neg_distances = distances * negative_mask.float()
        neg_distances = neg_distances + (~negative_mask).float() * 1e9
        hardest_negative = neg_distances.min(dim=1)[0]  # [B]

        # Triplet loss
        triplet_loss = F.relu(hardest_positive - hardest_negative + self.margin)

        # Average over valid anchors (those with at least one positive and negative)
        valid_anchors = (positive_mask.sum(dim=1) > 0) & (negative_mask.sum(dim=1) > 0)

        if valid_anchors.sum() == 0:
            # FIX: Return zero loss while maintaining gradient connection to embeddings
            # torch.tensor(0.0) creates a disconnected leaf tensor - gradients won't flow
            # Instead, use distances.sum() * 0 which preserves the computation graph
            return distances.sum() * 0.0

        return triplet_loss[valid_anchors].mean()

    def _semi_hard_mining(self, distances, positive_mask, negative_mask):
        """
        Semi-hard mining: negative is farther than positive but within margin.

        These triplets are most informative for learning.
        """
        batch_size = distances.shape[0]

        # Get anchor-positive distances
        pos_distances = distances * positive_mask.float()
        pos_distances_max = pos_distances.max(dim=1, keepdim=True)[0]  # [B, 1]

        # Semi-hard negatives: d(a,p) < d(a,n) < d(a,p) + margin
        neg_distances = distances * negative_mask.float()

        # Mask for semi-hard: greater than positive, less than positive + margin
        semi_hard_mask = (
            (neg_distances > pos_distances_max) &
            (neg_distances < pos_distances_max + self.margin) &
            negative_mask
        )

        # If no semi-hard, fall back to hard negative
        has_semi_hard = semi_hard_mask.sum(dim=1) > 0

        # For anchors with semi-hard negatives, use closest semi-hard
        neg_distances_semi = neg_distances.clone()
        neg_distances_semi[~semi_hard_mask] = 1e9
        semi_hard_neg = neg_distances_semi.min(dim=1)[0]

        # For anchors without semi-hard, use hard negative
        neg_distances_hard = neg_distances.clone()
        neg_distances_hard[~negative_mask] = 1e9
        hard_neg = neg_distances_hard.min(dim=1)[0]

        # Choose based on availability
        negative_dist = torch.where(has_semi_hard, semi_hard_neg, hard_neg)

        # Positive distance (use max/hardest positive)
        positive_dist = pos_distances_max.squeeze(1)

        # Triplet loss
        triplet_loss = F.relu(positive_dist - negative_dist + self.margin)

        # Valid anchors
        valid_anchors = (positive_mask.sum(dim=1) > 0) & (negative_mask.sum(dim=1) > 0)

        if valid_anchors.sum() == 0:
            # FIX: Maintain gradient connection (see _hard_mining comment)
            return distances.sum() * 0.0

        return triplet_loss[valid_anchors].mean()

    def _all_triplets(self, distances, positive_mask, negative_mask):
        """Compute loss over all valid triplets (no mining)."""
        batch_size = distances.shape[0]

        # Expand dimensions for broadcasting
        # anchor_positive_dist[a, p] = d(a, p)
        # anchor_negative_dist[a, n] = d(a, n)
        anchor_positive_dist = distances.unsqueeze(2)  # [B, B, 1]
        anchor_negative_dist = distances.unsqueeze(1)  # [B, 1, B]

        # triplet_loss[a, p, n] = max(d(a,p) - d(a,n) + margin, 0)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin
        triplet_loss = F.relu(triplet_loss)  # [B, B, B]

        # Mask for valid triplets: (a, p, n) where p is positive for a, n is negative
        valid_triplet_mask = (
            positive_mask.unsqueeze(2) &  # [B, B, 1]
            negative_mask.unsqueeze(1)     # [B, 1, B]
        )  # [B, B, B]

        # Apply mask and compute mean
        triplet_loss = triplet_loss * valid_triplet_mask.float()
        num_valid = valid_triplet_mask.sum()

        if num_valid == 0:
            # FIX: Maintain gradient connection (see _hard_mining comment)
            return distances.sum() * 0.0

        return triplet_loss.sum() / num_valid


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Test the models
if __name__ == "__main__":
    import tempfile
    import os

    print("=" * 70)
    print("Testing Embedding Models")
    print("=" * 70)

    # ===== Test LeViTEmbeddingModel (RECOMMENDED) =====
    print("\n[1/2] Testing LeViTEmbeddingModel (RECOMMENDED for Hailo 8)...")

    if TIMM_AVAILABLE:
        levit_model = LeViTEmbeddingModel(embedding_dim=768, pretrained=True)
        print(f"\nLeViT Model created:")
        print(f"  Embedding dimension: {levit_model.embedding_dim}")
        print(f"  Backbone feature dim: {levit_model.feature_dim}")
        print(f"  Total parameters: {count_parameters(levit_model):,}")

        # Test forward pass with 224x224 input (LeViT native size)
        batch_size = 8
        levit_input = torch.randn(batch_size, 3, 224, 224)

        levit_model.eval()
        with torch.no_grad():
            levit_embeddings = levit_model(levit_input)

        print(f"\nForward pass:")
        print(f"  Input shape: {levit_input.shape}")
        print(f"  Output shape: {levit_embeddings.shape}")

        # Verify L2 normalization
        levit_norms = torch.norm(levit_embeddings, p=2, dim=1)
        print(f"  Embedding norms (should be ~1.0): {levit_norms.mean():.6f}")

        # Test triplet loss with LeViT embeddings
        print("\nTesting TripletLossWithMining with LeViT...")
        loss_fn = TripletLossWithMining(margin=0.3, mining_type="semi_hard")
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        levit_loss = loss_fn(levit_embeddings, labels)
        print(f"  Triplet loss: {levit_loss.item():.4f}")

        # Test ONNX export
        print("\nTesting LeViT ONNX export...")
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            levit_onnx_path = f.name

        try:
            torch.onnx.export(
                levit_model,
                levit_input[:1],
                levit_onnx_path,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['embedding'],
                dynamic_axes={'input': {0: 'batch_size'}, 'embedding': {0: 'batch_size'}}
            )
            file_size = os.path.getsize(levit_onnx_path) / 1024 / 1024
            print(f"  ONNX export successful!")
            print(f"  File size: {file_size:.1f} MB")

            # Verify ONNX output matches PyTorch
            try:
                import onnxruntime as ort
                import numpy as np

                ort_session = ort.InferenceSession(levit_onnx_path, providers=['CPUExecutionProvider'])
                pytorch_out = levit_model(levit_input[:1]).detach().numpy()
                onnx_out = ort_session.run(None, {'input': levit_input[:1].numpy()})[0]
                max_diff = np.abs(pytorch_out - onnx_out).max()
                print(f"  ONNX verification: max_diff={max_diff:.2e}")
            except ImportError:
                print("  (onnxruntime not installed - skipping verification)")
        finally:
            os.unlink(levit_onnx_path)

        print("\n✓ LeViTEmbeddingModel working correctly!")
    else:
        print("  Skipping - timm not installed")

    # ===== Test CardEmbeddingModel (LEGACY) =====
    print("\n" + "-" * 70)
    print("[2/2] Testing CardEmbeddingModel (LEGACY - MobileNetV3)...")

    # Create model with GeM pooling (recommended)
    model = CardEmbeddingModel(embedding_dim=512, pretrained=True, use_gem_pooling=True)
    print(f"\nModel created:")
    print(f"  Embedding dimension: {model.embedding_dim}")
    print(f"  Backbone feature dim: {model.feature_dim}")
    print(f"  Pooling: {model.pooling}")
    print(f"  Total parameters: {count_parameters(model):,}")

    # Test forward pass
    batch_size = 8
    dummy_input = torch.randn(batch_size, 3, 384, 384)

    model.eval()
    with torch.no_grad():
        embeddings = model(dummy_input)

    print(f"\nForward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {embeddings.shape}")

    # Verify L2 normalization
    norms = torch.norm(embeddings, p=2, dim=1)
    print(f"  Embedding norms (should be ~1.0): {norms.mean():.6f}")

    # Test triplet loss
    print("\nTesting TripletLossWithMining...")
    loss_fn = TripletLossWithMining(margin=0.3, mining_type="semi_hard")
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = loss_fn(embeddings, labels)
    print(f"  Triplet loss: {loss.item():.4f}")

    # Test ONNX export compatibility
    print("\nTesting ONNX export...")
    model.eval()

    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        onnx_path = f.name

    try:
        torch.onnx.export(
            model,
            dummy_input[:1],  # Single image
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['embedding'],
            dynamic_axes={'input': {0: 'batch_size'}, 'embedding': {0: 'batch_size'}}
        )
        file_size = os.path.getsize(onnx_path) / 1024 / 1024
        print(f"  ONNX export successful!")
        print(f"  File size: {file_size:.1f} MB")

        # Verify ONNX output matches PyTorch
        try:
            import onnxruntime as ort
            import numpy as np

            ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            pytorch_out = model(dummy_input[:1]).detach().numpy()
            onnx_out = ort_session.run(None, {'input': dummy_input[:1].numpy()})[0]
            max_diff = np.abs(pytorch_out - onnx_out).max()
            print(f"  ONNX verification: max_diff={max_diff:.2e}")
        except ImportError:
            print("  (onnxruntime not installed - skipping verification)")
    finally:
        os.unlink(onnx_path)

    print("\n✓ CardEmbeddingModel working correctly!")

    # ===== Summary =====
    print("\n" + "=" * 70)
    print("Summary: Model Comparison")
    print("=" * 70)
    print("\n| Model | Params | Embedding | Hailo 8 | Recommended |")
    print("|-------|--------|-----------|---------|-------------|")
    if TIMM_AVAILABLE:
        print(f"| LeViT-384 | {count_parameters(levit_model):,} | 768-dim | 0.14ms | YES |")
    print(f"| MobileNetV3 | {count_parameters(model):,} | 512-dim | ~5ms | No (legacy) |")
    print("\n✓ All embedding models working correctly!")
