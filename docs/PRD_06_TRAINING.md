# Phase 5: SageMaker Training Guide
## PRD_06_TRAINING.md

**Parent Document:** PRD_01_OVERVIEW.md
**Phase:** Training (Parallel to Development)
**Platform:** Amazon SageMaker
**Last Updated:** January 2026

---

## Overview

This document provides complete training configurations for both models:
1. **Detection Model:** YOLO11n on SageMaker (unchanged)
2. **Embedding Model:** DINOv3-based Teacher-Student Architecture

### New Strategy: DINOv3 + Knowledge Distillation

Instead of training a small model from scratch, we use a **two-phase approach**:

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: Fine-tune "Teacher" Model (DINOv3-ViT-L/16)           │
│                                                                 │
│  - DINOv3 is pretrained on 1.7 BILLION images                   │
│  - Trained with iBOT (MAE-like masked reconstruction)           │
│  - Already robust to occlusion, glare, partial views            │
│  - Already understands textures, colors, shapes, text           │
│  - We only teach it: "these 17,592 things are Pokemon cards"    │
│  - Fine-tuning = CHEAP (~3 hours, ~$1.80)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                    Knowledge Distillation
                              │
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: Distill to "Student" Models                           │
│                                                                 │
│  Teacher's knowledge → Smaller, deployment-ready models         │
│  - Hailo-8L: ConvNeXt-Tiny (29M params) → .hef (CNN-optimized) │
│  - iOS: ConvNeXt-Base (89M params) → .mlmodel                   │
│  - Android: ConvNeXt-Small (50M params) → .onnx                 │
│  - Cost per student: ~$0.25 (~1 hour each)                      │
└─────────────────────────────────────────────────────────────────┘
```

**Why DINOv3 is Perfect for Pokemon Cards:**
- **iBOT Training:** Uses masked patch prediction (like MAE) for occlusion robustness
- **Self-Distillation:** DINO method learns view-invariant features
- **Massive Pre-training:** 1.7B images = understands visual patterns deeply
- **Proven Results:** State-of-the-art on fine-grained recognition tasks

### Why This is Better

| Approach | Training Cost | Accuracy | Flexibility |
|----------|--------------|----------|-------------|
| Train small model from scratch | $$$ (days) | Good | One target only |
| **Fine-tune DINOv3 + Distill** | $ (hours) | Excellent | Multiple targets |

**Key Insight:** DINOv3 was trained by Meta on 256 GPUs for weeks. We get all that knowledge for FREE and just add Pokemon card expertise.

---

## Model Architecture Overview

### Teacher Model: DINOv3-ViT-L/16

```
┌─────────────────────────────────────────┐
│  DINOv3-ViT-L/16 (Frozen/Partial)       │
│  - 304M parameters                       │
│  - Pretrained on 1.7B images            │
│  - Output: 1024-dim features            │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Projection Head (Trainable)            │
│  - Linear 1024 → 768                    │
│  - GELU + Dropout                        │
│  - Linear 768 → 768                      │
│  - L2 Normalize                          │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  ArcFace Head (Trainable)               │
│  - 17,592 classes                        │
│  - margin=0.5, scale=64                  │
└─────────────────────────────────────────┘
```

### Student Models (Distilled)

| Target | Architecture | Params | Input Size | Output |
|--------|--------------|--------|------------|--------|
| Hailo-8L | ConvNeXt-Tiny | 29M | 224×224 | 768-dim |
| iOS/CoreML | ConvNeXt-Base | 89M | 384×384 | 768-dim |
| Android/ONNX | ViT-Small | 22M | 224×224 | 768-dim |
| Server/API | DINOv3-ViT-L | 304M | 224×224 | 768-dim |

---

## AWS Infrastructure Setup

### S3 Bucket Structure (Updated)

```
s3://pokemon-card-recognition/
├── datasets/
│   ├── detection/                    # YOLO format (unchanged)
│   │   ├── images/{train,val,test}/
│   │   ├── labels/{train,val,test}/
│   │   └── pokemon_cards.yaml
│   │
│   └── embedding/                    # Classification format
│       ├── train/
│       │   └── [class_name]/
│       │       └── *.png
│       ├── val/
│       │   └── [class_name]/
│       │       └── *.png
│       ├── class_index.json
│       └── card_metadata.json
│
├── models/
│   ├── detection/
│   │   └── yolo11n-obb.pt
│   │
│   └── embedding/
│       ├── teacher/                  # DINOv3 fine-tuned
│       │   ├── dinov3_teacher.pt
│       │   └── dinov3_teacher.onnx
│       │
│       └── students/                 # Distilled models
│           ├── hailo/
│           │   ├── convnext_tiny.pt
│           │   ├── convnext_tiny.onnx
│           │   └── convnext_tiny.hef
│           ├── ios/
│           │   ├── convnext_base.pt
│           │   └── convnext_base.mlmodel
│           └── android/
│               ├── vit_small.pt
│               └── vit_small.onnx
│
└── artifacts/
    ├── embeddings/
    │   ├── teacher_embeddings.npy    # [17592, 768] from teacher
    │   ├── hailo_embeddings.npy      # [17592, 768] from student
    │   └── ios_embeddings.npy
    ├── card_metadata.json
    └── usearch.index
```

### IAM Role (unchanged)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::pokemon-card-recognition",
        "arn:aws:s3:::pokemon-card-recognition/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
      "Resource": "arn:aws:logs:*:*:*"
    }
  ]
}
```

---

## Detection Model Training (Unchanged)

Detection training remains the same - YOLO11n for card detection.
See original script in `src/training/train_detection.py`.

**Cost:** ~$0.25 per training run (spot instance)

---

## Embedding Model Training: Phase 1 - Teacher Fine-tuning

### Why Fine-tuning is Cheap

| What We're NOT Doing | What We ARE Doing |
|---------------------|-------------------|
| Training 304M params from random init | Loading pretrained DINOv3 weights |
| Teaching model to "see" | Model already sees perfectly |
| Learning image features | Learning card-specific features |
| Weeks of training | Hours of fine-tuning |

### requirements.txt

```
# Latest versions as of January 2026
torch>=2.9.0                   # Stable release (2.10 RC available)
torchvision>=0.20.0
timm>=1.0.12                   # Latest release (Jan 7, 2026)
transformers>=5.0.0            # Version 5 with DINOv3 support
albumentations>=1.4.0
opencv-python>=4.11.0
numpy>=2.0.0
Pillow>=11.0.0
tqdm>=4.67.0
```

### Model Definition

```python
# src/models/dinov3_embedding.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from transformers import Dinov3Model


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
        self.backbone = Dinov3Model.from_pretrained(f'facebook/{model_name}-pretrain-lvd1689m')

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
```

### Training Script: Phase 1 (Teacher)

```python
# src/training/train_dinov3_teacher.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import argparse
import logging
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm

from models.dinov3_embedding import DINOv3TeacherModel, ArcFaceLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # SageMaker paths with local fallbacks
    parser.add_argument('--model-dir', type=str,
                       default=os.environ.get('SM_MODEL_DIR', './outputs'))
    parser.add_argument('--train-dir', type=str,
                       default=os.environ.get('SM_CHANNEL_TRAIN', './data/train'))
    parser.add_argument('--val-dir', type=str,
                       default=os.environ.get('SM_CHANNEL_VAL', './data/val'))

    # Model
    parser.add_argument('--dinov3-model', type=str, default='dinov3_vitl16')
    parser.add_argument('--embedding-dim', type=int, default=768)

    # Training - Phase 1 (frozen backbone)
    parser.add_argument('--epochs-frozen', type=int, default=5)
    parser.add_argument('--epochs-unfrozen', type=int, default=15)
    parser.add_argument('--unfreeze-blocks', type=int, default=4)

    parser.add_argument('--batch-size', type=int, default=32)  # Smaller due to large model
    parser.add_argument('--lr-frozen', type=float, default=1e-3)  # Higher LR for projection
    parser.add_argument('--lr-unfrozen', type=float, default=1e-5)  # Lower LR for backbone
    parser.add_argument('--weight-decay', type=float, default=0.05)

    # Loss
    parser.add_argument('--arcface-margin', type=float, default=0.5)
    parser.add_argument('--arcface-scale', type=float, default=64.0)

    return parser.parse_args()


def get_transforms(image_size: int = 224):
    """DINOv3 uses 224x224 input for ViT-L/16."""

    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


def train_epoch(model, loss_fn, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc='Training'):
        images = images.to(device)
        labels = labels.to(device)

        embeddings = model(images)
        loss = loss_fn(embeddings, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Track accuracy (for monitoring)
        with torch.no_grad():
            W = nn.functional.normalize(loss_fn.weight, p=2, dim=1)
            cosine = torch.mm(embeddings, W.t())
            pred = cosine.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(dataloader), correct / total


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate using nearest neighbor retrieval."""
    model.eval()

    all_embeddings = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc='Evaluating'):
        images = images.to(device)
        embeddings = model(images)

        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels)

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # Compute pairwise distances
    distances = torch.cdist(embeddings, embeddings)
    distances.fill_diagonal_(float('inf'))

    # Top-1 accuracy (nearest neighbor)
    nn_indices = distances.argmin(dim=1)
    nn_labels = labels[nn_indices]
    top1_acc = (nn_labels == labels).float().mean().item()

    # Top-5 accuracy
    _, top5_indices = distances.topk(5, dim=1, largest=False)
    top5_labels = labels[top5_indices]
    top5_acc = (top5_labels == labels.unsqueeze(1)).any(dim=1).float().mean().item()

    return top1_acc, top5_acc


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"DINOv3 model: {args.dinov3_model}")

    # Ensure model directory exists
    os.makedirs(args.model_dir, exist_ok=True)

    # Load datasets
    train_transform, val_transform = get_transforms(224)

    train_dataset = datasets.ImageFolder(args.train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(args.val_dir, transform=val_transform)

    num_classes = len(train_dataset.classes)
    logger.info(f"Number of classes: {num_classes}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Create model (backbone frozen initially)
    model = DINOv3TeacherModel(
        model_name=args.dinov3_model,
        embedding_dim=args.embedding_dim,
        freeze_backbone=True,
    ).to(device)

    loss_fn = ArcFaceLoss(
        embedding_dim=args.embedding_dim,
        num_classes=num_classes,
        margin=args.arcface_margin,
        scale=args.arcface_scale,
    ).to(device)

    # ========== PHASE 1: Train projection head only ==========
    logger.info("=" * 60)
    logger.info("PHASE 1: Training projection head (backbone frozen)")
    logger.info("=" * 60)

    optimizer = torch.optim.AdamW(
        list(model.projection.parameters()) + list(loss_fn.parameters()),
        lr=args.lr_frozen,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_frozen
    )

    best_acc = 0

    for epoch in range(args.epochs_frozen):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs_frozen} (Frozen)")

        train_loss, train_acc = train_epoch(model, loss_fn, train_loader, optimizer, device)
        top1_acc, top5_acc = evaluate(model, val_loader, device)
        scheduler.step()

        logger.info(f"Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}")
        logger.info(f"Val Top1: {top1_acc:.2%}, Top5: {top5_acc:.2%}")

        if top1_acc > best_acc:
            best_acc = top1_acc
            torch.save({
                'model': model.state_dict(),
                'loss_fn': loss_fn.state_dict(),
                'epoch': epoch,
                'top1_acc': top1_acc,
            }, f'{args.model_dir}/best_teacher_frozen.pt')

    # ========== PHASE 2: Fine-tune backbone ==========
    logger.info("\n" + "=" * 60)
    logger.info(f"PHASE 2: Fine-tuning last {args.unfreeze_blocks} transformer blocks")
    logger.info("=" * 60)

    model.unfreeze_backbone(last_n_blocks=args.unfreeze_blocks)

    optimizer = torch.optim.AdamW(
        [
            {'params': model.projection.parameters(), 'lr': args.lr_unfrozen * 10},
            {'params': model.backbone.encoder.layer[-args.unfreeze_blocks:].parameters(), 'lr': args.lr_unfrozen},
            {'params': loss_fn.parameters(), 'lr': args.lr_unfrozen * 10},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_unfrozen
    )

    for epoch in range(args.epochs_unfrozen):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs_unfrozen} (Fine-tuning)")

        train_loss, train_acc = train_epoch(model, loss_fn, train_loader, optimizer, device)
        top1_acc, top5_acc = evaluate(model, val_loader, device)
        scheduler.step()

        logger.info(f"Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}")
        logger.info(f"Val Top1: {top1_acc:.2%}, Top5: {top5_acc:.2%}")

        if top1_acc > best_acc:
            best_acc = top1_acc
            torch.save({
                'model': model.state_dict(),
                'loss_fn': loss_fn.state_dict(),
                'epoch': args.epochs_frozen + epoch,
                'top1_acc': top1_acc,
            }, f'{args.model_dir}/best_teacher.pt')
            logger.info(f"New best model! Top1: {top1_acc:.2%}")

    # ========== Export ==========
    logger.info("\nExporting teacher model...")

    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        f'{args.model_dir}/dinov3_teacher.onnx',
        input_names=['input'],
        output_names=['embedding'],
        dynamic_axes={'input': {0: 'batch'}, 'embedding': {0: 'batch'}},
        opset_version=17,
    )

    # Save metrics
    metrics = {
        'model': args.dinov3_model,
        'num_classes': num_classes,
        'embedding_dim': args.embedding_dim,
        'best_top1_accuracy': best_acc,
        'epochs_frozen': args.epochs_frozen,
        'epochs_unfrozen': args.epochs_unfrozen,
    }

    with open(f'{args.model_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"\nTraining complete! Best Top1: {best_acc:.2%}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
```

### SageMaker Job: Teacher Training

```python
# scripts/launch_teacher_training.py

import sagemaker
from sagemaker.pytorch import PyTorch

session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = 'pokemon-card-recognition'

estimator = PyTorch(
    entry_point='train_dinov3_teacher.py',
    source_dir='./src/training',
    role=role,
    instance_count=1,
    instance_type='ml.g5.4xlarge',  # A10G 24GB - enough for DINOv3-ViT-L
    framework_version='2.1',
    py_version='py310',

    hyperparameters={
        'dinov3-model': 'dinov3_vitl16',
        'embedding-dim': 768,
        'epochs-frozen': 5,
        'epochs-unfrozen': 15,
        'unfreeze-blocks': 4,
        'batch-size': 32,
        'lr-frozen': 1e-3,
        'lr-unfrozen': 1e-5,
        'arcface-margin': 0.5,
        'arcface-scale': 64,
    },

    output_path=f's3://{bucket}/models/embedding/teacher/',

    use_spot_instances=True,
    max_wait=14400,  # 4 hours max
    max_run=10800,   # 3 hours training

    checkpoint_s3_uri=f's3://{bucket}/checkpoints/teacher/',
)

estimator.fit({
    'train': f's3://{bucket}/datasets/embedding/train/',
    'val': f's3://{bucket}/datasets/embedding/val/',
})
```

---

## Embedding Model Training: Phase 2 - Knowledge Distillation

### How Distillation Works

```
Teacher (DINOv3-ViT-L)          Student (ConvNeXt-T)
       │                              │
   [Image] ──────────────────────► [Image]
       │                              │
       ▼                              ▼
  Teacher                         Student
  Embedding ◄───── MSE Loss ─────► Embedding
  (768-dim)                        (768-dim)
       │
       └──── The student learns to MIMIC the teacher's output
```

The student doesn't see the 17,592 class labels - it just learns to produce the same embeddings as the teacher. This transfers ALL the teacher's knowledge.

### Distillation Training Script

```python
# src/training/distill_student.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import argparse
import logging
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm

from models.dinov3_embedding import DINOv3TeacherModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StudentModel(nn.Module):
    """Lightweight student model for edge deployment."""

    def __init__(self, model_name: str, embedding_dim: int = 768):
        super().__init__()

        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        backbone_dim = self.backbone.num_features

        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.projection(features)
        return nn.functional.normalize(embeddings, p=2, dim=1)


def distill(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ensure model directory exists
    os.makedirs(args.model_dir, exist_ok=True)

    # Load teacher - full DINOv3TeacherModel with projection
    logger.info("Loading teacher model...")

    # Validate checkpoint exists
    if not os.path.exists(args.teacher_checkpoint):
        raise FileNotFoundError(f"Teacher checkpoint not found: {args.teacher_checkpoint}")

    teacher = DINOv3TeacherModel(
        model_name='dinov3_vitl16',
        embedding_dim=args.embedding_dim,
        freeze_backbone=True
    )
    teacher_checkpoint = torch.load(args.teacher_checkpoint, map_location=device)
    teacher.load_state_dict(teacher_checkpoint['model'])
    teacher.eval()
    teacher.to(device)

    for param in teacher.parameters():
        param.requires_grad = False

    # Create student
    logger.info(f"Creating student: {args.student_model}")
    student = StudentModel(args.student_model, args.embedding_dim).to(device)

    # Dataset (train + validation)
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(args.train_dir, transform=train_transform)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )

    # Validation dataset
    if args.val_dir:
        val_dataset = datasets.ImageFolder(args.val_dir, transform=val_transform)
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )
    else:
        val_loader = None

    # Optimizer
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Distillation loss (MSE between embeddings)
    mse_loss = nn.MSELoss()

    # Best model tracking
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(args.epochs):
        # Training phase
        student.train()
        total_train_loss = 0

        for images, _ in tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]'):
            images = images.to(device)

            # Teacher uses 224x224 input (same as student for DINOv3 ViT-L/16)
            teacher_images = nn.functional.interpolate(images, size=(224, 224), mode='bilinear')

            with torch.no_grad():
                teacher_embeddings = teacher(teacher_images)

            student_embeddings = student(images)

            loss = mse_loss(student_embeddings, teacher_embeddings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        if val_loader:
            student.eval()
            total_val_loss = 0

            with torch.no_grad():
                for images, _ in tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]'):
                    images = images.to(device)
                    teacher_images = nn.functional.interpolate(images, size=(224, 224), mode='bilinear')

                    teacher_embeddings = teacher(teacher_images)
                    student_embeddings = student(images)

                    loss = mse_loss(student_embeddings, teacher_embeddings)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            logger.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

            # Save best model based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(student.state_dict(), f'{args.model_dir}/{args.student_model}_best.pt')
                logger.info(f"  → New best model! Val Loss = {avg_val_loss:.6f}")
        else:
            logger.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}")

        scheduler.step()

        # Save latest checkpoint
        torch.save(student.state_dict(), f'{args.model_dir}/{args.student_model}_latest.pt')

    # Load best model for export
    if val_loader and os.path.exists(f'{args.model_dir}/{args.student_model}_best.pt'):
        logger.info("Loading best model for export...")
        student.load_state_dict(torch.load(f'{args.model_dir}/{args.student_model}_best.pt'))

    # Final export
    logger.info("Exporting student model...")
    student.eval()
    dummy = torch.randn(1, 3, args.image_size, args.image_size).to(device)

    torch.onnx.export(
        student, dummy,
        f'{args.model_dir}/{args.student_model}.onnx',
        input_names=['input'],
        output_names=['embedding'],
        opset_version=17,
    )

    logger.info(f"Distillation complete! Best val loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher-checkpoint', type=str, required=True)
    parser.add_argument('--student-model', type=str, default='convnext_tiny')
    parser.add_argument('--embedding-dim', type=int, default=768)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train-dir', type=str, required=True)
    parser.add_argument('--val-dir', type=str, default=None, help='Validation directory (optional, but recommended)')
    parser.add_argument('--model-dir', type=str, default='./outputs')

    args = parser.parse_args()
    distill(args)
```

### Student Model Configurations

| Target | Student Model | Image Size | Command |
|--------|--------------|------------|---------|
| Hailo-8L | `convnext_tiny` | 224 | `--student-model convnext_tiny --image-size 224` |
| iOS | `convnext_base` | 384 | `--student-model convnext_base --image-size 384` |
| Android | `vit_small_patch16_224` | 224 | `--student-model vit_small_patch16_224 --image-size 224` |

---

## Cost Estimation (Updated)

### Phase 1: Teacher Fine-tuning
- Instance: `ml.g5.4xlarge` ($2.03/hr)
- Duration: ~3 hours (20 epochs)
- Spot discount: ~70%
- **Estimated cost: ~$1.80 per training run**

### Phase 2: Student Distillation (per student)
- Instance: `ml.g4dn.xlarge` ($0.736/hr)
- Duration: ~1 hour (30 epochs)
- Spot discount: ~70%
- **Estimated cost: ~$0.25 per student**

### Total Training Cost

| Phase | Cost |
|-------|------|
| Teacher fine-tuning (1x) | $1.80 |
| Hailo student distillation | $0.25 |
| iOS student distillation | $0.25 |
| Android student distillation | $0.25 |
| Detection model (unchanged) | $0.25 |
| **Total** | **~$2.80** |

**Compare to training from scratch:** Would cost $50+ and take days.

---

## Training Pipeline Summary

```
┌──────────────────────────────────────────────────────────────┐
│  STEP 1: Upload data to S3                                   │
│  aws s3 sync ./data s3://pokemon-card-recognition/datasets/  │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 2: Fine-tune DINOv3 Teacher (~3 hours)                 │
│  python scripts/launch_teacher_training.py                   │
│  Output: dinov3_teacher.pt, dinov3_teacher.onnx              │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 3: Distill to Students (~1 hour each)                  │
│  python src/training/distill_student.py \                    │
│    --teacher-checkpoint dinov3_teacher.pt \                  │
│    --student-model convnext_tiny                             │
│  Repeat for: convnext_base, vit_small_patch16_224            │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 4: Generate Reference Embeddings                       │
│  python scripts/build_reference_db.py \                      │
│    --model models/embedding/students/hailo/convnext_tiny.pt  │
│  Output: embeddings.npy, usearch.index                       │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 5: Export for Deployment                               │
│  - Hailo: ONNX → HEF (Hailo compiler)                        │
│  - iOS: ONNX → CoreML (coremltools)                          │
│  - Android: ONNX ready                                       │
└──────────────────────────────────────────────────────────────┘
```

---

## Key Benefits of DINOv3 Approach

1. **Cheaper Training:** Fine-tuning costs ~$2 vs $50+ for training from scratch

2. **Better Accuracy:** DINOv3 understands fine-grained visual details from 1.7B image pretraining

3. **Multiple Targets from One Training:**
   - Train teacher once
   - Distill to unlimited student architectures
   - iOS, Android, Hailo, web - all from same teacher

4. **Future-Proof:**
   - New card sets? Just update the teacher
   - New deployment target? Distill another student
   - Better base model released? Swap DINOv3 for newer model

5. **Research-Backed:** Meta's DINOv3 represents state-of-the-art self-supervised learning

---

## Next Steps

1. [ ] Download DINOv3 pretrained weights via HuggingFace (test locally first)
2. [ ] Upload classification dataset to S3
3. [ ] Run teacher fine-tuning on SageMaker
4. [ ] Evaluate teacher accuracy on test set
5. [ ] Distill to ConvNeXt-Tiny (Hailo target)
6. [ ] Distill to ConvNeXt-Base (iOS target)
7. [ ] Export and compile for each deployment target
8. [ ] Generate reference embeddings with each student
9. [ ] Benchmark inference speed on target hardware

---

## References

- [DINOv3 on HuggingFace](https://huggingface.co/facebook/dinov3_vitl16-pretrain-lvd1689m) - Pretrained model weights
- [DINOv3 Paper](https://arxiv.org/abs/2508.10104)
- [DINOv3 Blog Post](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/)
- [Knowledge Distillation Survey](https://arxiv.org/abs/2006.05525)
