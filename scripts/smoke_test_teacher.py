#!/usr/bin/env python3
"""
Smoke Test for DINOv3 Teacher Training

Runs a minimal 1-epoch training with small batch size to verify:
- All dependencies installed correctly
- Data loading works
- Model initialization works
- Training loop executes without errors
- Checkpoint saving works
- Mixed precision training works
- Distributed training setup works (if available)

Run this before launching full training to catch configuration errors early.
Estimated time: 2-3 minutes
Estimated cost: ~$0.10-0.20
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from training.dinov3_embedding import DINOv3TeacherModel, ArcFaceLoss
from training.train_dinov3_teacher import get_transforms, WarmupCosineScheduler, ModelEMA

def smoke_test():
    """Run smoke test for teacher training."""
    print("=" * 60)
    print("DINOv3 Teacher Training - Smoke Test")
    print("=" * 60)
    print("\nVerifying environment...")

    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ CRITICAL: No CUDA devices available")
        sys.exit(1)

    print(f"✅ CUDA available: {torch.cuda.device_count()} GPU(s)")
    print(f"   Device 0: {torch.cuda.get_device_name(0)}")

    # Check distributed training
    is_distributed = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1

    if is_distributed:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        print(f"✅ Distributed training: rank {rank}/{world_size}")
    else:
        rank = 0
        world_size = 1
        device = torch.device('cuda:0')
        print("✅ Single GPU mode")

    # Test imports
    print("\nTesting imports...")
    try:
        from transformers import DINOv3ViTModel
        print("✅ transformers (DINOv3)")
    except ImportError as e:
        print(f"❌ transformers import failed: {e}")
        sys.exit(1)

    try:
        import albumentations as A
        print("✅ albumentations")
    except ImportError as e:
        print(f"❌ albumentations import failed: {e}")
        sys.exit(1)

    try:
        from scipy.ndimage import gaussian_filter
        print("✅ scipy")
    except ImportError as e:
        print(f"❌ scipy import failed: {e}")
        sys.exit(1)

    # Test data loading
    print("\nTesting data loading...")
    train_dir = os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train')

    if not os.path.exists(train_dir):
        print(f"❌ Training data not found: {train_dir}")
        sys.exit(1)

    train_transform, val_transform = get_transforms(224)
    full_dataset = datasets.ImageFolder(train_dir, transform=train_transform)

    if len(full_dataset) == 0:
        print("❌ No training samples found")
        sys.exit(1)

    num_classes = len(full_dataset.classes)
    print(f"✅ Dataset loaded: {len(full_dataset)} images, {num_classes} classes")

    # Use tiny subset for smoke test (100 samples)
    subset_size = min(100, len(full_dataset))
    indices = list(range(subset_size))
    train_dataset = Subset(full_dataset, indices)

    # Create dataloaders
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=8,  # Small batch for smoke test
            sampler=train_sampler,
            num_workers=2,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

    print(f"✅ DataLoader created: {len(train_loader)} batches")

    # Test model initialization
    print("\nTesting model initialization...")
    try:
        model = DINOv3TeacherModel(
            model_name='dinov3_vit7b16',  # 7B parameter model
            embedding_dim=768,
            freeze_backbone=True,
        ).to(device)
        print("✅ DINOv3 model created")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        sys.exit(1)

    # Test torch.compile
    try:
        model = torch.compile(model, mode='reduce-overhead')
        print("✅ torch.compile succeeded")
    except Exception as e:
        print(f"⚠️  torch.compile failed (non-critical): {e}")

    # Test DDP
    if is_distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )
        print("✅ Model wrapped in DDP")

    # Test EMA
    model_for_inspection = model.module if is_distributed else model
    ema_model = ModelEMA(model_for_inspection, decay=0.999).to(device)
    print("✅ EMA initialized")

    # Test loss function
    try:
        loss_fn = ArcFaceLoss(
            embedding_dim=768,
            num_classes=num_classes,
            margin=0.5,
            scale=64.0,
        ).to(device)
        print("✅ ArcFace loss created")
    except Exception as e:
        print(f"❌ Loss function creation failed: {e}")
        sys.exit(1)

    # Test optimizer & scheduler
    optimizer = torch.optim.AdamW(
        list(model_for_inspection.projection.parameters()) + list(loss_fn.parameters()),
        lr=1e-3,
        weight_decay=1e-2,
    )
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=1, total_epochs=1, min_lr=1e-7)
    print("✅ Optimizer & scheduler created")

    # Test mixed precision
    scaler = GradScaler()
    print("✅ GradScaler created")

    # Test training loop
    print("\nTesting training loop (1 epoch)...")
    model.train()
    total_loss = 0
    num_batches = 0

    try:
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast():
                embeddings = model(images)
                loss = loss_fn(embeddings, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # Update EMA
            ema_model.update(model)

            total_loss += loss.item()
            num_batches += 1

            if rank == 0 and batch_idx % 5 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}: loss={loss.item():.4f}")

        avg_loss = total_loss / num_batches
        scheduler.step()

        if rank == 0:
            print(f"✅ Training loop complete: avg_loss={avg_loss:.4f}")

    except Exception as e:
        print(f"❌ Training loop failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Test checkpoint saving
    if rank == 0:
        print("\nTesting checkpoint saving...")
        try:
            checkpoint_path = '/tmp/smoke_test_checkpoint.pt'
            model_state = model.module.state_dict() if is_distributed else model.state_dict()
            torch.save({
                'model': model_state,
                'ema_model': ema_model.module.state_dict(),
                'loss_fn': loss_fn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler,
                'epoch': 0,
                'top1_acc': 0.0,
                'num_classes': num_classes,
                'phase': 'phase1',
            }, checkpoint_path)
            print(f"✅ Checkpoint saved: {checkpoint_path}")

            # Test loading
            checkpoint = torch.load(checkpoint_path, map_location=device)
            print("✅ Checkpoint loaded successfully")

            # Cleanup
            os.remove(checkpoint_path)
            print("✅ Checkpoint cleanup complete")

        except Exception as e:
            print(f"❌ Checkpoint save/load failed: {e}")
            sys.exit(1)

    # Cleanup distributed
    if is_distributed:
        dist.destroy_process_group()

    if rank == 0:
        print("\n" + "=" * 60)
        print("✅ ALL SMOKE TESTS PASSED!")
        print("=" * 60)
        print("\nEnvironment is ready for full training.")
        print("You can now launch the main training job with confidence.")
        print("=" * 60)

if __name__ == '__main__':
    smoke_test()
