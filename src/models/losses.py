#!/usr/bin/env python3
"""
ArcFace Loss Implementation for Card Embedding Training

ArcFace adds an angular margin to softmax loss:
- Creates tight embedding clusters (same cards cluster together)
- Better separation between classes
- Improves unknown card rejection (large distance from all clusters)

Formula: L = -log(e^(s*cos(theta+m)) / (e^(s*cos(theta+m)) + sum(e^(s*cos(theta_j)))))

Where:
- theta = angle between embedding and class center
- m = angular margin (typically 0.5)
- s = scale factor (typically 64)

Reference: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
https://arxiv.org/abs/1801.07698
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    """
    ArcFace classification head for metric learning.

    This head learns a weight matrix where each column represents
    a class center in the embedding space. During training, it pushes
    embeddings to be close to their class center with an angular margin.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float = 64.0,
        margin: float = 0.5,
        easy_margin: bool = False
    ):
        """
        Args:
            embedding_dim: Dimension of input embeddings (768 for LeViT-384)
            num_classes: Number of classes (cards)
            scale: Scale factor for logits (typically 30-64)
            margin: Angular margin in radians (typically 0.5)
            easy_margin: If True, use easier margin formulation
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin

        # Learnable class centers (normalized during forward pass)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Precompute margin values
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)  # Threshold for easy_margin
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute ArcFace logits.

        Args:
            embeddings: [B, embedding_dim] L2-normalized embeddings
            labels: [B] class labels

        Returns:
            logits: [B, num_classes] scaled logits with angular margin applied
        """
        # Normalize weight (class centers)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity between embeddings and class centers
        # embeddings: [B, D], weight_norm: [C, D] -> cosine: [B, C]
        cosine = F.linear(embeddings, weight_norm)

        # Clamp cosine to prevent numerical issues with acos
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)

        # Compute sin from cos: sin = sqrt(1 - cos^2)
        sine = torch.sqrt(1.0 - cosine.pow(2))

        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            # Easy margin: only apply margin when cos(theta) > 0
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Hard margin: apply margin with threshold
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Create one-hot encoding of labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        # Apply margin only to the correct class
        # output[i, j] = phi[i, j] if j == labels[i] else cosine[i, j]
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Scale logits
        output = output * self.scale

        return output


class ArcFaceLoss(nn.Module):
    """
    Combined ArcFace head + CrossEntropy loss.

    This is the complete loss function for ArcFace training.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float = 64.0,
        margin: float = 0.5,
        label_smoothing: float = 0.1
    ):
        """
        Args:
            embedding_dim: Dimension of embeddings
            num_classes: Number of classes
            scale: Scale factor (typically 64)
            margin: Angular margin (typically 0.5)
            label_smoothing: Label smoothing factor
        """
        super().__init__()

        self.arcface_head = ArcFaceHead(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            scale=scale,
            margin=margin
        )

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ArcFace loss.

        Args:
            embeddings: [B, embedding_dim] L2-normalized embeddings
            labels: [B] class labels

        Returns:
            loss: Scalar ArcFace loss
        """
        logits = self.arcface_head(embeddings, labels)
        loss = self.criterion(logits, labels)
        return loss

    def get_logits(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Get logits for accuracy computation."""
        return self.arcface_head(embeddings, labels)


class CombinedLoss(nn.Module):
    """
    Combined ArcFace + Triplet loss for best of both worlds.

    - ArcFace: Creates tight clusters, good for classification
    - Triplet: Fine-grained discrimination between similar cards

    This can be used in a single training phase instead of two separate phases.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        arcface_scale: float = 64.0,
        arcface_margin: float = 0.5,
        triplet_margin: float = 0.3,
        arcface_weight: float = 1.0,
        triplet_weight: float = 0.5,
        label_smoothing: float = 0.1
    ):
        super().__init__()

        self.arcface_loss = ArcFaceLoss(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            scale=arcface_scale,
            margin=arcface_margin,
            label_smoothing=label_smoothing
        )

        # Import triplet loss from embedding_model
        from embedding_model import TripletLossWithMining
        self.triplet_loss = TripletLossWithMining(
            margin=triplet_margin,
            mining_type="semi_hard"
        )

        self.arcface_weight = arcface_weight
        self.triplet_weight = triplet_weight

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> tuple:
        """
        Compute combined loss.

        Returns:
            (total_loss, arcface_loss, triplet_loss)
        """
        arc_loss = self.arcface_loss(embeddings, labels)
        trip_loss = self.triplet_loss(embeddings, labels)

        total = self.arcface_weight * arc_loss + self.triplet_weight * trip_loss

        return total, arc_loss, trip_loss


# Test the implementation
if __name__ == "__main__":
    print("=" * 60)
    print("Testing ArcFace Loss Implementation")
    print("=" * 60)

    # Configuration matching FINAL_PLAN.md
    embedding_dim = 768  # LeViT-384
    num_classes = 17592  # Exact count from PokeTCG_downloader/assets/card_images/
    batch_size = 64

    # Create random embeddings and labels
    embeddings = torch.randn(batch_size, embedding_dim)
    embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 normalize
    labels = torch.randint(0, num_classes, (batch_size,))

    # Test ArcFaceHead
    print("\n1. Testing ArcFaceHead...")
    head = ArcFaceHead(embedding_dim, num_classes, scale=64.0, margin=0.5)
    logits = head(embeddings, labels)
    print(f"   Input: embeddings {embeddings.shape}, labels {labels.shape}")
    print(f"   Output logits: {logits.shape}")
    print(f"   Logits range: [{logits.min():.2f}, {logits.max():.2f}]")

    # Test ArcFaceLoss
    print("\n2. Testing ArcFaceLoss...")
    loss_fn = ArcFaceLoss(embedding_dim, num_classes, scale=64.0, margin=0.5)
    loss = loss_fn(embeddings, labels)
    print(f"   Loss: {loss.item():.4f}")

    # Test backward pass
    print("\n3. Testing backward pass...")
    loss.backward()
    print("   Gradient computed successfully!")

    # Test CombinedLoss
    print("\n4. Testing CombinedLoss...")
    embeddings2 = torch.randn(batch_size, embedding_dim)
    embeddings2 = F.normalize(embeddings2, p=2, dim=1)
    labels2 = torch.randint(0, 100, (batch_size,))  # Smaller for triplet mining

    combined_loss = CombinedLoss(
        embedding_dim=embedding_dim,
        num_classes=100,
        arcface_weight=1.0,
        triplet_weight=0.5
    )
    total, arc, trip = combined_loss(embeddings2, labels2)
    print(f"   Total loss: {total.item():.4f}")
    print(f"   ArcFace loss: {arc.item():.4f}")
    print(f"   Triplet loss: {trip.item():.4f}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
