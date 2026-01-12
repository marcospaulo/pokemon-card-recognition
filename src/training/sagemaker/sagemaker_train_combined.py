#!/usr/bin/env python3
"""
SageMaker Combined Training: ArcFace + Triplet Loss

FASTEST + BEST QUALITY approach:
- Single training phase with combined loss
- ArcFace creates tight clusters (classification-style)
- Triplet provides fine-grained discrimination
- LeViT-384 backbone with strong ImageNet pretrained weights

Performance on ml.p4d.24xlarge (8x A100):
- Estimated training time: 1-2 hours
- Effective batch size: 128 * 8 = 1024 per GPU batch

Key optimizations:
- DDP with cross-GPU triplet mining (GatherWithGradient)
- bfloat16 mixed precision (A100 native)
- OneCycleLR scheduler for fast convergence
- Combined loss avoids two-phase overhead
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Import our modules - handle both local and SageMaker environments
import sys
from pathlib import Path

# Add src to path for local development
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    # Try local imports first (for development)
    from src.data.dataset import TripletCardDataset, get_heavy_augmentation, get_val_transform, collate_triplets
    from src.models.embedding import LeViTEmbeddingModel, TripletLossWithMining
    from src.models.losses import ArcFaceLoss, CombinedLoss
except ImportError:
    # Fall back to SageMaker flat structure
    from triplet_dataset import TripletCardDataset, get_heavy_augmentation, get_val_transform, collate_triplets
    from embedding_model import LeViTEmbeddingModel, TripletLossWithMining
    from arcface_loss import ArcFaceLoss, CombinedLoss


def setup_distributed():
    """Initialize distributed training."""
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        return True, rank, local_rank, world_size
    return False, 0, 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def print_main(msg: str, rank: int):
    if is_main_process(rank):
        print(msg)


class GatherWithGradient(torch.autograd.Function):
    """Gather embeddings across GPUs while preserving gradients."""

    @staticmethod
    def forward(ctx, embeddings: torch.Tensor) -> torch.Tensor:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        batch_size, embedding_dim = embeddings.shape

        gathered_list = [
            torch.zeros(batch_size, embedding_dim, device=embeddings.device, dtype=embeddings.dtype)
            for _ in range(world_size)
        ]
        dist.all_gather(gathered_list, embeddings.contiguous())
        gathered_list[rank] = embeddings  # Preserve gradient for local
        all_embeddings = torch.cat(gathered_list, dim=0)

        ctx.rank = rank
        ctx.batch_size = batch_size
        ctx.world_size = world_size
        return all_embeddings

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        start_idx = ctx.rank * ctx.batch_size
        end_idx = start_idx + ctx.batch_size
        return grad_output[start_idx:end_idx].contiguous()


def gather_embeddings_distributed(embeddings: torch.Tensor, labels: torch.Tensor):
    """Gather embeddings and labels from all GPUs."""
    if not dist.is_initialized():
        return embeddings, labels

    all_embeddings = GatherWithGradient.apply(embeddings)

    batch_size = labels.shape[0]
    gathered_labels = [
        torch.zeros(batch_size, device=labels.device, dtype=labels.dtype)
        for _ in range(dist.get_world_size())
    ]
    dist.all_gather(gathered_labels, labels.contiguous())
    all_labels = torch.cat(gathered_labels, dim=0)

    return all_embeddings, all_labels


@torch.no_grad()
def compute_all_embeddings(model, dataloader, device):
    """Compute embeddings for all samples."""
    model.eval()
    all_embeddings, all_labels = [], []

    for images, labels in dataloader:
        images = images.to(device)
        embeddings = model(images)
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels)

    return torch.cat(all_embeddings, 0), torch.cat(all_labels, 0)


@torch.no_grad()
def compute_recall_at_k(query_emb, query_lab, gallery_emb, gallery_lab, k_values=[1, 5, 10]):
    """Compute Recall@K metrics with cross-set evaluation."""
    num_gallery = gallery_emb.shape[0]
    max_k = min(max(k_values), num_gallery)
    k_values = [min(k, num_gallery) for k in k_values]

    similarity = torch.mm(query_emb, gallery_emb.t())
    _, topk_indices = similarity.topk(max_k, dim=1)
    topk_labels = gallery_lab[topk_indices]
    correct = topk_labels == query_lab.unsqueeze(1)

    results = {}
    for k in k_values:
        recall = correct[:, :k].any(dim=1).float().mean().item() * 100
        results[f"recall@{k}"] = recall
    return results


def train_epoch(
    model, dataloader, optimizer, scheduler,
    arcface_loss_fn, triplet_loss_fn,
    device, epoch, rank,
    arcface_weight=1.0, triplet_weight=0.5,
    use_amp=True, use_cross_gpu_mining=True
):
    """Train one epoch with combined ArcFace + Triplet loss."""
    model.train()
    total_loss, total_arc, total_trip = 0.0, 0.0, 0.0
    num_batches = 0
    dtype = torch.bfloat16 if use_amp else torch.float32

    for batch_idx, (anchors, positives, labels) in enumerate(dataloader):
        anchors = anchors.to(device)
        positives = positives.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
            # Get embeddings
            anchor_emb = model(anchors)
            positive_emb = model(positives)

            # === ArcFace Loss (on anchor embeddings only) ===
            arc_loss = arcface_loss_fn(anchor_emb, labels)

            # === Triplet Loss (on combined embeddings, cross-GPU) ===
            local_emb = torch.cat([anchor_emb, positive_emb], dim=0)
            local_lab = torch.cat([labels, labels], dim=0)

            if use_cross_gpu_mining and dist.is_initialized():
                all_emb, all_lab = gather_embeddings_distributed(local_emb, local_lab)
            else:
                all_emb, all_lab = local_emb, local_lab

            trip_loss = triplet_loss_fn(all_emb, all_lab)

            # === Combined Loss ===
            loss = arcface_weight * arc_loss + triplet_weight * trip_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_arc += arc_loss.item()
        total_trip += trip_loss.item()
        num_batches += 1

        if batch_idx % 50 == 0:
            lr = optimizer.param_groups[0]['lr']
            print_main(
                f"  Epoch {epoch+1} [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} (Arc: {arc_loss.item():.4f}, Trip: {trip_loss.item():.4f}) "
                f"LR: {lr:.2e}",
                rank
            )

    return total_loss / num_batches, total_arc / num_batches, total_trip / num_batches


@torch.no_grad()
def validate(model, val_loader, device, rank, train_gallery_loader=None):
    """Validate using cross-set Recall@K."""
    val_emb, val_lab = compute_all_embeddings(model, val_loader, device)
    val_emb, val_lab = val_emb.cpu(), val_lab.cpu()

    if train_gallery_loader:
        train_emb, train_lab = compute_all_embeddings(model, train_gallery_loader, device)
        train_emb, train_lab = train_emb.cpu(), train_lab.cpu()
        metrics = compute_recall_at_k(val_emb, val_lab, train_emb, train_lab)
    else:
        metrics = compute_recall_at_k(val_emb, val_lab, val_emb, val_lab)

    return metrics


def train():
    """Main training function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--embedding-dim', type=int, default=768)
    # ArcFace params
    parser.add_argument('--arcface-scale', type=float, default=64.0)
    parser.add_argument('--arcface-margin', type=float, default=0.5)
    parser.add_argument('--arcface-weight', type=float, default=1.0)
    # Triplet params
    parser.add_argument('--triplet-margin', type=float, default=0.3)
    parser.add_argument('--triplet-weight', type=float, default=0.5)
    parser.add_argument('--mining-type', type=str, default='semi_hard')
    # Training params
    parser.add_argument('--early-stop-patience', type=int, default=10)
    parser.add_argument('--freeze-backbone-epochs', type=int, default=3)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    # Paths
    parser.add_argument('--model-dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--data-dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))

    args = parser.parse_args()

    # Setup distributed
    distributed, rank, local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    print_main("=" * 70, rank)
    print_main("COMBINED TRAINING: ArcFace + Triplet Loss", rank)
    print_main("Fastest + Best Quality Training for LeViT-384", rank)
    print_main("=" * 70, rank)

    if distributed:
        print_main(f"\nDistributed: {world_size} GPUs", rank)
    print_main(f"Effective batch: {args.batch_size * world_size}", rank)

    # Load class info
    with open(os.path.join(args.data_dir, 'class_index.json')) as f:
        class_info = json.load(f)
    num_classes = class_info['num_classes']
    class_to_idx = class_info['class_to_idx']
    print_main(f"\nClasses: {num_classes}", rank)

    # Create datasets (224x224 for LeViT-384)
    train_transform = get_heavy_augmentation(224)
    val_transform = get_val_transform(224)

    train_dataset = TripletCardDataset(
        root_dir=os.path.join(args.data_dir, 'train'),
        class_to_idx=class_to_idx,
        transform=train_transform,
        return_triplets=True
    )

    val_dataset = TripletCardDataset(
        root_dir=os.path.join(args.data_dir, 'val'),
        class_to_idx=class_to_idx,
        transform=val_transform,
        return_triplets=False
    )

    print_main(f"Train samples: {len(train_dataset)}", rank)
    print_main(f"Val samples: {len(val_dataset)}", rank)

    # Data loaders
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler,
            collate_fn=collate_triplets, num_workers=4, pin_memory=True, drop_last=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_triplets, num_workers=4, pin_memory=True, drop_last=True
        )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=4, pin_memory=True
    )

    # Gallery loader for cross-set validation
    train_gallery = TripletCardDataset(
        root_dir=os.path.join(args.data_dir, 'train'),
        class_to_idx=class_to_idx,
        transform=val_transform,
        return_triplets=False
    )
    train_gallery_loader = DataLoader(
        train_gallery, batch_size=args.batch_size * 2, shuffle=False, num_workers=4, pin_memory=True
    )

    # Create model
    print_main(f"\nModel: LeViT-384 ({args.embedding_dim}-dim)", rank)
    freeze_backbone = args.freeze_backbone_epochs > 0

    model = LeViTEmbeddingModel(
        embedding_dim=args.embedding_dim,
        pretrained=True,
        freeze_backbone=freeze_backbone
    )
    model = model.to(device)

    if distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank])

    # Loss functions
    arcface_loss = ArcFaceLoss(
        embedding_dim=args.embedding_dim,
        num_classes=num_classes,
        scale=args.arcface_scale,
        margin=args.arcface_margin,
        label_smoothing=args.label_smoothing
    ).to(device)

    triplet_loss = TripletLossWithMining(
        margin=args.triplet_margin,
        mining_type=args.mining_type
    )

    print_main(f"\nLoss: ArcFace (s={args.arcface_scale}, m={args.arcface_margin}) x {args.arcface_weight}", rank)
    print_main(f"    + Triplet (margin={args.triplet_margin}) x {args.triplet_weight}", rank)

    # Optimizer - include ArcFace head parameters
    all_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    all_params += list(arcface_loss.parameters())

    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=0.01)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr * 10,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )

    print_main(f"\nTraining:", rank)
    print_main(f"  Epochs: {args.epochs}", rank)
    print_main(f"  Freeze backbone: {args.freeze_backbone_epochs} epochs", rank)
    print_main(f"  Early stop patience: {args.early_stop_patience}", rank)

    # Training loop
    best_recall1 = -1.0
    patience_counter = 0
    best_model_path = os.path.join(args.model_dir, 'best_embedding.pth')

    # Save initial model as baseline
    if is_main_process(rank):
        state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save(state, best_model_path)

    print_main("\n" + "=" * 70, rank)
    print_main("Starting combined ArcFace + Triplet training...", rank)
    print_main("=" * 70, rank)

    for epoch in range(args.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)

        # Unfreeze backbone after warmup
        if epoch == args.freeze_backbone_epochs and args.freeze_backbone_epochs > 0:
            print_main(f"\nEpoch {epoch+1}: Unfreezing backbone", rank)
            if hasattr(model, 'module'):
                model.module.unfreeze_backbone()
            else:
                model.unfreeze_backbone()

            # Recreate optimizer with all parameters
            all_params = list(model.parameters()) + list(arcface_loss.parameters())
            optimizer = torch.optim.AdamW(all_params, lr=args.lr * 0.1, weight_decay=0.01)

            remaining_steps = (args.epochs - epoch) * len(train_loader)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining_steps, eta_min=args.lr * 0.001
            )

        # Train
        start = time.time()
        avg_loss, avg_arc, avg_trip = train_epoch(
            model, train_loader, optimizer, scheduler,
            arcface_loss, triplet_loss, device, epoch, rank,
            arcface_weight=args.arcface_weight,
            triplet_weight=args.triplet_weight
        )
        train_time = time.time() - start

        # Validate
        if is_main_process(rank):
            metrics = validate(model, val_loader, device, rank, train_gallery_loader)
            recall1 = metrics['recall@1']
            recall5 = metrics['recall@5']

            print(f"\nEpoch [{epoch+1}/{args.epochs}]:")
            print(f"  Loss: {avg_loss:.4f} (Arc: {avg_arc:.4f}, Trip: {avg_trip:.4f}) [{train_time:.1f}s]")
            print(f"  Recall@1: {recall1:.2f}%, Recall@5: {recall5:.2f}%")

            if recall1 > best_recall1:
                best_recall1 = recall1
                patience_counter = 0
                state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save(state, best_model_path)
                print(f"  Saved best model (Recall@1: {recall1:.2f}%)")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{args.early_stop_patience})")

            if patience_counter >= args.early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        if distributed:
            dist.barrier()

    # Export to ONNX
    if is_main_process(rank):
        print_main("\n" + "=" * 70, rank)
        print_main("Exporting to ONNX...", rank)

        export_model = LeViTEmbeddingModel(embedding_dim=args.embedding_dim, pretrained=False)
        export_model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
        export_model.eval()

        dummy = torch.randn(1, 3, 224, 224)
        onnx_path = os.path.join(args.model_dir, 'card_embedding.onnx')

        torch.onnx.export(
            export_model, dummy, onnx_path,
            export_params=True, opset_version=17, do_constant_folding=True,
            input_names=['input'], output_names=['embedding'],
            dynamic_axes={'input': {0: 'batch_size'}, 'embedding': {0: 'batch_size'}}
        )

        size_mb = os.path.getsize(onnx_path) / 1024 / 1024
        print_main(f"  ONNX: {onnx_path} ({size_mb:.1f} MB)", rank)

        # Verify ONNX
        try:
            import onnxruntime as ort
            import numpy as np
            sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            pytorch_out = export_model(dummy).detach().numpy()
            onnx_out = sess.run(None, {'input': dummy.numpy()})[0]
            max_diff = np.abs(pytorch_out - onnx_out).max()
            print_main(f"  ONNX verified: max_diff={max_diff:.2e}", rank)
        except Exception as e:
            print_main(f"  ONNX verification skipped: {e}", rank)

        # Save config
        with open(os.path.join(args.model_dir, 'training_config.json'), 'w') as f:
            json.dump({
                'embedding_dim': args.embedding_dim,
                'arcface_scale': args.arcface_scale,
                'arcface_margin': args.arcface_margin,
                'triplet_margin': args.triplet_margin,
                'best_recall1': best_recall1,
                'num_classes': num_classes,
                'image_size': 224,
                'training_type': 'combined_arcface_triplet'
            }, f, indent=2)

        print_main("\n" + "=" * 70, rank)
        print_main("Training complete!", rank)
        print_main(f"  Best Recall@1: {best_recall1:.2f}%", rank)
        print_main("=" * 70, rank)

    cleanup_distributed()


if __name__ == '__main__':
    train()
