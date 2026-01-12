#!/usr/bin/env python3
"""
SageMaker training script for Card Embedding Model using Triplet Loss

METRIC LEARNING APPROACH - NOT CLASSIFICATION
- Uses LeViT-384 (CNN-Transformer hybrid) - Hailo 8 NPU optimized
- Trains a 768-dim embedding model
- Uses triplet loss with online hard negative mining
- Evaluated using Recall@K metrics (not accuracy)
- Exports ONNX model for Hailo 8 deployment (0.14ms inference)

Designed for: 8x NVIDIA A100 80GB GPUs (ml.p4d.24xlarge)
Uses 2026 PyTorch best practices: DDP, bfloat16, torch.compile
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
import torchvision.transforms as transforms

# Import our modules - handle both local and SageMaker environments
from pathlib import Path

# Add src to path for local development
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    # Try local imports first (for development)
    from src.data.dataset import TripletCardDataset, get_heavy_augmentation, get_val_transform, collate_triplets
    from src.models.embedding import LeViTEmbeddingModel, TripletLossWithMining
except ImportError:
    # Fall back to SageMaker flat structure
    from triplet_dataset import TripletCardDataset, get_heavy_augmentation, get_val_transform, collate_triplets
    from embedding_model import LeViTEmbeddingModel, TripletLossWithMining


def setup_distributed():
    """Initialize distributed training if available."""
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])

        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

        return True, rank, local_rank, world_size
    elif torch.cuda.device_count() > 1:
        # Fallback for non-distributed multi-GPU
        return False, 0, 0, 1
    else:
        return False, 0, 0, 1


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """Check if this is the main process (for logging/saving)."""
    return rank == 0


def print_main(msg: str, rank: int):
    """Print only from main process."""
    if is_main_process(rank):
        print(msg)


class GatherWithGradient(torch.autograd.Function):
    """
    Custom autograd function for gathering embeddings across GPUs while preserving gradients.

    CRITICAL: Standard dist.all_gather() breaks gradient flow because gathered tensors
    are detached copies. This function preserves gradients for the local rank's embeddings
    by replacing the detached copy with the original tensor in the forward pass.

    During backward:
    - Gradients for local embeddings flow normally
    - Gradients for remote embeddings are discarded (they'll be computed on their respective GPUs)
    """

    @staticmethod
    def forward(ctx, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Gather embeddings from all GPUs, preserving gradient for local rank.

        Args:
            embeddings: [B, D] local embeddings with gradient history

        Returns:
            all_embeddings: [B*world_size, D] gathered embeddings where local
                           rank's portion is connected to computation graph
        """
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        batch_size = embeddings.shape[0]
        embedding_dim = embeddings.shape[1]

        # Pre-allocate tensors for gathering (these will be detached copies)
        gathered_list = [
            torch.zeros(batch_size, embedding_dim, device=embeddings.device, dtype=embeddings.dtype)
            for _ in range(world_size)
        ]

        # Gather embeddings from all GPUs
        dist.all_gather(gathered_list, embeddings.contiguous())

        # CRITICAL: Replace the detached copy at our rank with the original tensor
        # This preserves the gradient computation graph for local embeddings
        gathered_list[rank] = embeddings

        # Concatenate all embeddings
        all_embeddings = torch.cat(gathered_list, dim=0)

        # Save info for backward
        ctx.rank = rank
        ctx.batch_size = batch_size
        ctx.world_size = world_size

        return all_embeddings

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Extract gradients for local embeddings only.

        Each GPU only needs gradients for its own embeddings - gradients for
        other GPUs' embeddings will be computed during their backward pass.

        Args:
            grad_output: [B*world_size, D] gradients for all embeddings

        Returns:
            grad_embeddings: [B, D] gradients for local embeddings only
        """
        rank = ctx.rank
        batch_size = ctx.batch_size

        # Extract gradients for local rank's embeddings
        start_idx = rank * batch_size
        end_idx = start_idx + batch_size

        grad_embeddings = grad_output[start_idx:end_idx].contiguous()

        return grad_embeddings


def gather_embeddings_distributed(
    embeddings: torch.Tensor,
    labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gather embeddings and labels from all GPUs for proper triplet mining.

    CRITICAL for DDP: Without this, triplet mining only sees local batch,
    severely limiting negative mining effectiveness.

    Uses GatherWithGradient to preserve gradient flow through local embeddings.

    Args:
        embeddings: [B, D] local embeddings
        labels: [B] local labels

    Returns:
        gathered_embeddings: [B*world_size, D] all embeddings (gradient preserved for local)
        gathered_labels: [B*world_size] all labels
    """
    if not dist.is_initialized():
        return embeddings, labels

    world_size = dist.get_world_size()

    # Use gradient-preserving gather for embeddings
    all_embeddings = GatherWithGradient.apply(embeddings)

    # Labels don't need gradients - standard gather is fine
    batch_size = labels.shape[0]
    gathered_labels = [
        torch.zeros(batch_size, device=labels.device, dtype=labels.dtype)
        for _ in range(world_size)
    ]
    dist.all_gather(gathered_labels, labels.contiguous())
    all_labels = torch.cat(gathered_labels, dim=0)

    return all_embeddings, all_labels


@torch.no_grad()
def compute_all_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute embeddings for all samples in dataloader.

    Returns:
        embeddings: [N, D] tensor of embeddings
        labels: [N] tensor of labels
    """
    model.eval()

    all_embeddings = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)

        embeddings = model(images)
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels)

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return embeddings, labels


@torch.no_grad()
def compute_recall_at_k(
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_labels: torch.Tensor,
    k_values: list = [1, 5, 10],
    exclude_self: bool = True
) -> Dict[str, float]:
    """
    Compute Recall@K metrics.

    Recall@K = % of queries where the correct class appears in top-K retrievals

    Args:
        query_embeddings: [Q, D] query embeddings
        query_labels: [Q] query labels
        gallery_embeddings: [G, D] gallery embeddings (database to search)
        gallery_labels: [G] gallery labels
        k_values: List of K values to compute
        exclude_self: If True and Q==G (same-set evaluation), exclude diagonal
            to prevent self-matching. CRITICAL for unbiased evaluation.

    Returns:
        Dictionary of {f"recall@{k}": value} for each k
    """
    num_queries = query_embeddings.shape[0]
    num_gallery = gallery_embeddings.shape[0]

    # BOUNDS CHECK: Ensure k doesn't exceed gallery size
    max_k = max(k_values)
    if max_k > num_gallery:
        print(f"  Warning: max k={max_k} exceeds gallery size {num_gallery}, clamping k values")
        k_values = [min(k, num_gallery) for k in k_values]
        max_k = max(k_values)

    # Compute similarity matrix (dot product for L2-normalized embeddings)
    similarity = torch.mm(query_embeddings, gallery_embeddings.t())  # [Q, G]

    # CRITICAL FIX: Exclude self-matching when query == gallery
    # Without this, Recall@1 is always ~100% (query matches itself)
    is_same_set = False
    if exclude_self and num_queries == num_gallery:
        # Check if this is same-set evaluation (query == gallery)
        # Use embedding comparison to be safe
        is_same_set = torch.allclose(query_embeddings, gallery_embeddings, atol=1e-6)
        if is_same_set:
            # Mask diagonal with -inf so self never appears in top-K
            mask = torch.eye(num_queries, device=similarity.device, dtype=torch.bool)
            similarity = similarity.masked_fill(mask, float('-inf'))

    # Get top-K indices for each query
    _, topk_indices = similarity.topk(max_k, dim=1)  # [Q, max_k]

    # Get labels of top-K retrievals
    topk_labels = gallery_labels[topk_indices]  # [Q, max_k]

    # Check if correct label is in top-K
    correct = topk_labels == query_labels.unsqueeze(1)  # [Q, max_k]

    # Compute Recall@K for each K
    results = {}
    for k in k_values:
        recall_at_k = correct[:, :k].any(dim=1).float().mean().item()
        results[f"recall@{k}"] = recall_at_k * 100  # Convert to percentage

    return results


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    rank: int,
    use_amp: bool = True,
    use_cross_gpu_mining: bool = True
) -> float:
    """
    Train for one epoch with optional cross-GPU triplet mining.

    Args:
        use_cross_gpu_mining: If True, gather embeddings across all GPUs before
            triplet mining. CRITICAL for effective training with DDP.

    Returns:
        Average loss for the epoch
    """
    model.train()

    total_loss = 0.0
    num_batches = 0

    # Mixed precision training with bfloat16 (A100 native)
    dtype = torch.bfloat16 if use_amp else torch.float32

    for batch_idx, (anchors, positives, labels) in enumerate(dataloader):
        anchors = anchors.to(device)
        positives = positives.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
            # Get embeddings for both anchor and positive
            anchor_embeddings = model(anchors)
            positive_embeddings = model(positives)

            # Combine embeddings and labels for triplet mining
            # Each batch has: [anchor_0, anchor_1, ..., pos_0, pos_1, ...]
            local_embeddings = torch.cat([anchor_embeddings, positive_embeddings], dim=0)
            local_labels = torch.cat([labels, labels], dim=0)  # Same labels

            # CRITICAL: Gather embeddings from all GPUs for effective triplet mining
            # Without this, mining is limited to local batch only
            if use_cross_gpu_mining and dist.is_initialized():
                all_embeddings, all_labels = gather_embeddings_distributed(
                    local_embeddings, local_labels
                )
            else:
                all_embeddings, all_labels = local_embeddings, local_labels

            # Compute triplet loss with online mining across all gathered embeddings
            loss = loss_fn(all_embeddings, all_labels)

        # Backward pass (no GradScaler needed for bfloat16)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Step scheduler per-batch (OneCycleLR expects this)
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        # Log progress
        if batch_idx % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            effective_batch = all_embeddings.shape[0] if use_cross_gpu_mining else local_embeddings.shape[0]
            print_main(
                f"  Epoch {epoch+1} [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} LR: {current_lr:.2e} Mining batch: {effective_batch}",
                rank
            )

    return total_loss / num_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    rank: int,
    train_loader: DataLoader = None
) -> Dict[str, float]:
    """
    Validate model using Recall@K metrics.

    CRITICAL: For 1-image-per-class datasets, same-set evaluation is meaningless
    because after excluding self-match, there are no other same-class samples.

    Strategy: Cross-set evaluation (val → train)
    - Query: val embeddings
    - Gallery: train embeddings
    - Task: Can a val image find its class's train image?
    - This matches real deployment (gallery of known cards, query with photos)
    """
    # Compute embeddings for all validation samples
    val_embeddings, val_labels = compute_all_embeddings(model, val_loader, device)

    # Move to CPU for metric computation (save GPU memory)
    val_embeddings = val_embeddings.cpu()
    val_labels = val_labels.cpu()

    # Use cross-set evaluation if train_loader provided (meaningful for 1-image-per-class)
    if train_loader is not None:
        train_embeddings, train_labels = compute_all_embeddings(model, train_loader, device)
        train_embeddings = train_embeddings.cpu()
        train_labels = train_labels.cpu()

        # Cross-set: val queries find train gallery images
        metrics = compute_recall_at_k(
            query_embeddings=val_embeddings,
            query_labels=val_labels,
            gallery_embeddings=train_embeddings,
            gallery_labels=train_labels,
            k_values=[1, 5, 10],
            exclude_self=False  # Different sets, no self to exclude
        )
    else:
        # Fallback to same-set (will have low Recall@K with 1-image-per-class)
        # This is kept for backwards compatibility but not recommended
        metrics = compute_recall_at_k(
            query_embeddings=val_embeddings,
            query_labels=val_labels,
            gallery_embeddings=val_embeddings,
            gallery_labels=val_labels,
            k_values=[1, 5, 10]
        )

        # Warn if this looks like 1-image-per-class scenario
        unique_labels = len(torch.unique(val_labels))
        if unique_labels == len(val_labels):
            print("  ⚠ Warning: 1 sample per class detected - same-set Recall@K will be ~0%")
            print("    Consider using cross-set evaluation (val→train) for meaningful metrics")

    return metrics


def train():
    """Main training function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)  # Per GPU
    parser.add_argument('--lr', type=float, default=1e-5)  # Lower LR for triplet fine-tuning per FINAL_PLAN.md
    parser.add_argument('--embedding-dim', type=int, default=768)  # 768 for LeViT-384 per FINAL_PLAN.md
    parser.add_argument('--margin', type=float, default=0.3)
    parser.add_argument('--mining-type', type=str, default='semi_hard',
                        choices=['hard', 'semi_hard', 'all'])
    parser.add_argument('--early-stop-patience', type=int, default=15)
    parser.add_argument('--model-dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--data-dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    parser.add_argument('--use-compile', action='store_true',
                        help='Use torch.compile() for speedup')
    parser.add_argument('--freeze-backbone-epochs', type=int, default=5,
                        help='Freeze backbone for first N epochs (warm-up embedding head)')

    args = parser.parse_args()

    # Setup distributed training
    distributed, rank, local_rank, world_size = setup_distributed()

    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    print_main("=" * 70, rank)
    print_main("Training Card Embedding Model with Triplet Loss", rank)
    print_main("=" * 70, rank)

    if distributed:
        print_main(f"\nDistributed training: {world_size} GPUs", rank)
    else:
        print_main(f"\nSingle GPU training", rank)

    # Load class information
    class_index_file = os.path.join(args.data_dir, 'class_index.json')
    if not os.path.exists(class_index_file):
        raise FileNotFoundError(f"Required file not found: {class_index_file}")

    with open(class_index_file) as f:
        class_info = json.load(f)

    num_classes = class_info['num_classes']
    class_to_idx = class_info['class_to_idx']

    print_main(f"\nDataset:", rank)
    print_main(f"  Total classes: {num_classes}", rank)

    # Create datasets (224x224 for LeViT-384 per FINAL_PLAN.md)
    train_transform = get_heavy_augmentation(224)
    val_transform = get_val_transform(224)

    train_dataset = TripletCardDataset(
        root_dir=os.path.join(args.data_dir, 'train'),
        class_to_idx=class_to_idx,
        transform=train_transform,
        return_triplets=True
    )

    # For validation, we don't need triplets - just embeddings
    val_dataset = TripletCardDataset(
        root_dir=os.path.join(args.data_dir, 'val'),
        class_to_idx=class_to_idx,
        transform=val_transform,
        return_triplets=False  # Return (image, label) for embedding extraction
    )

    print_main(f"  Train samples: {len(train_dataset)}", rank)
    print_main(f"  Val samples: {len(val_dataset)}", rank)

    # Create data loaders
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            collate_fn=collate_triplets,
            num_workers=4,
            pin_memory=True,
            drop_last=True  # Important for triplet mining
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_triplets,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,  # Can use larger batch for inference
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create a gallery loader from train set (for cross-set validation)
    # CRITICAL: Uses val_transform (no augmentation) for stable embeddings
    train_gallery_dataset = TripletCardDataset(
        root_dir=os.path.join(args.data_dir, 'train'),
        class_to_idx=class_to_idx,
        transform=val_transform,  # No augmentation for gallery
        return_triplets=False  # Return (image, label) for embedding extraction
    )
    train_gallery_loader = DataLoader(
        train_gallery_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create model
    print_main(f"\nModel:", rank)
    print_main(f"  Architecture: LeViT-384 (CNN-Transformer hybrid, Hailo-optimized)", rank)
    print_main(f"  Embedding dimension: {args.embedding_dim}", rank)

    # Start with frozen backbone for first few epochs
    freeze_backbone = args.freeze_backbone_epochs > 0

    model = LeViTEmbeddingModel(
        embedding_dim=args.embedding_dim,
        pretrained=True,
        freeze_backbone=freeze_backbone
    )
    model = model.to(device)

    # Wrap with DDP if distributed
    if distributed:
        # CRITICAL: Convert BatchNorm to SyncBatchNorm BEFORE DDP wrapping
        # This prevents in-place operation errors with autocast and cross-GPU gradients
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        print_main("  Converted BatchNorm to SyncBatchNorm for DDP", rank)
        model = DDP(model, device_ids=[local_rank])

    # Optional: use torch.compile for speedup (PyTorch 2.0+)
    if args.use_compile:
        print_main("  Using torch.compile() for optimization", rank)
        model = torch.compile(model)

    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print_main(f"  Trainable parameters: {trainable_params:,} / {total_params:,}", rank)

    # Loss function
    loss_fn = TripletLossWithMining(margin=args.margin, mining_type=args.mining_type)
    print_main(f"\nLoss:", rank)
    print_main(f"  Triplet Loss with margin={args.margin}", rank)
    print_main(f"  Mining type: {args.mining_type}", rank)

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01
    )

    # Learning rate scheduler - OneCycleLR (2026 best practice)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr * 10,  # Peak at 10x base LR
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos'
    )

    print_main(f"\nTraining:", rank)
    print_main(f"  Epochs: {args.epochs}", rank)
    print_main(f"  Batch size per GPU: {args.batch_size}", rank)
    print_main(f"  Effective batch size: {args.batch_size * world_size}", rank)
    print_main(f"  Learning rate: {args.lr}", rank)
    print_main(f"  Freeze backbone epochs: {args.freeze_backbone_epochs}", rank)
    print_main(f"  Early stop patience: {args.early_stop_patience}", rank)

    # Training loop
    best_recall1 = -1.0  # Start at -1 so first validation (even 0%) saves as "best"
    patience_counter = 0
    best_model_path = os.path.join(args.model_dir, 'best_embedding.pth')

    # CRITICAL: Save initial model as baseline so we always have a "best" model
    # This prevents FileNotFoundError if recall never improves
    if is_main_process(rank):
        if hasattr(model, 'module'):
            initial_state = model.module.state_dict()
        else:
            initial_state = model.state_dict()
        torch.save(initial_state, best_model_path)
        print("  Saved initial model as baseline")

    print_main("\n" + "=" * 70, rank)
    print_main("Starting training...", rank)
    print_main("=" * 70, rank)

    for epoch in range(args.epochs):
        # Set epoch for distributed sampler
        if distributed:
            train_sampler.set_epoch(epoch)

        # Unfreeze backbone after warmup
        if epoch == args.freeze_backbone_epochs and args.freeze_backbone_epochs > 0:
            print_main(f"\nEpoch {epoch+1}: Unfreezing backbone for fine-tuning", rank)
            if hasattr(model, 'module'):
                model.module.unfreeze_backbone()
            else:
                model.unfreeze_backbone()

            # Re-create optimizer with all parameters
            # Use lower LR for fine-tuning pretrained backbone
            finetune_lr = args.lr * 0.1
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=finetune_lr,
                weight_decay=0.01
            )

            # FIX: Use CosineAnnealingLR instead of OneCycleLR to avoid warmup discontinuity
            # OneCycleLR with pct_start>0 causes sudden LR drop then spike
            # CosineAnnealingLR provides smooth decay from current LR
            remaining_epochs = args.epochs - epoch
            total_remaining_steps = remaining_epochs * len(train_loader)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_remaining_steps,
                eta_min=finetune_lr * 0.01  # Decay to 1% of finetune LR
            )
            print_main(f"  Recreated CosineAnnealingLR for {remaining_epochs} remaining epochs", rank)
            print_main(f"  Fine-tuning LR: {finetune_lr:.2e} → {finetune_lr * 0.01:.2e}", rank)

        # Train
        start_time = time.time()
        avg_loss = train_epoch(
            model, train_loader, optimizer, scheduler, loss_fn,
            device, epoch, rank, use_amp=True
        )
        train_time = time.time() - start_time

        # Validate using cross-set (val→train) for meaningful 1-image-per-class metrics
        if is_main_process(rank):
            metrics = validate(model, val_loader, device, rank, train_loader=train_gallery_loader)
            recall1 = metrics['recall@1']
            recall5 = metrics['recall@5']
            recall10 = metrics['recall@10']

            print(f"\nEpoch [{epoch+1}/{args.epochs}]:")
            print(f"  Train Loss: {avg_loss:.4f} ({train_time:.1f}s)")
            print(f"  Val Recall@1: {recall1:.2f}%")
            print(f"  Val Recall@5: {recall5:.2f}%")
            print(f"  Val Recall@10: {recall10:.2f}%")

            # Save best model
            if recall1 > best_recall1:
                best_recall1 = recall1
                patience_counter = 0

                # Save model state (handle DDP)
                if hasattr(model, 'module'):
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()

                torch.save(state_dict, best_model_path)
                print(f"  ✓ Saved best model (Recall@1: {recall1:.2f}%)")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{args.early_stop_patience})")

            # Save checkpoint every epoch
            checkpoint_path = os.path.join(args.model_dir, f'checkpoint_epoch_{epoch+1}.pth')
            checkpoint_dict = {
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_recall1': best_recall1,
                'metrics': metrics,
            }
            if hasattr(model, 'module'):
                checkpoint_dict['model_state_dict'] = model.module.state_dict()
            else:
                checkpoint_dict['model_state_dict'] = model.state_dict()

            torch.save(checkpoint_dict, checkpoint_path)

            # Early stopping
            if patience_counter >= args.early_stop_patience:
                print(f"\nEarly stopping: no improvement for {args.early_stop_patience} epochs")
                break

        # Synchronize processes
        if distributed:
            dist.barrier()

    # Export to ONNX (main process only)
    if is_main_process(rank):
        print_main("\n" + "=" * 70, rank)
        print_main("Exporting best model to ONNX...", rank)
        print_main("=" * 70, rank)

        # Load best model
        export_model = LeViTEmbeddingModel(
            embedding_dim=args.embedding_dim,
            pretrained=False
        )
        export_model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
        export_model.eval()

        # Export (224x224 for LeViT-384 per FINAL_PLAN.md)
        dummy_input = torch.randn(1, 3, 224, 224)
        onnx_path = os.path.join(args.model_dir, 'card_embedding.onnx')

        torch.onnx.export(
            export_model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=17,  # Updated from 11 to 17 for better operator support
            do_constant_folding=True,
            input_names=['input'],
            output_names=['embedding'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'embedding': {0: 'batch_size'}
            }
        )

        onnx_size = os.path.getsize(onnx_path) / 1024 / 1024
        print_main(f"  ONNX model: {onnx_path} ({onnx_size:.1f} MB)", rank)

        # CRITICAL: Verify ONNX export matches PyTorch output
        print_main("  Verifying ONNX export...", rank)
        onnx_verified = False
        try:
            import onnxruntime as ort
            import numpy as np

            # Create ONNX session
            ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

            # Get PyTorch output
            with torch.no_grad():
                pytorch_output = export_model(dummy_input).numpy()

            # Get ONNX output
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
            onnx_output = ort_session.run(None, ort_inputs)[0]

            # Check for NaN/Inf in outputs (critical failure)
            if np.any(np.isnan(onnx_output)) or np.any(np.isinf(onnx_output)):
                print_main(f"  ✗ CRITICAL: ONNX output contains NaN or Inf values!", rank)
                print_main(f"    NaN count: {np.sum(np.isnan(onnx_output))}", rank)
                print_main(f"    Inf count: {np.sum(np.isinf(onnx_output))}", rank)
            elif np.any(np.isnan(pytorch_output)) or np.any(np.isinf(pytorch_output)):
                print_main(f"  ✗ CRITICAL: PyTorch output contains NaN or Inf values!", rank)
            else:
                # Compare outputs
                max_diff = np.abs(pytorch_output - onnx_output).max()
                mean_diff = np.abs(pytorch_output - onnx_output).mean()

                if max_diff < 1e-4:
                    print_main(f"  ✓ ONNX verification passed (max_diff={max_diff:.2e})", rank)
                    onnx_verified = True
                elif max_diff < 1e-2:
                    print_main(f"  ⚠ ONNX verification WARNING: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}", rank)
                    print_main(f"    This may be acceptable numerical precision difference", rank)
                    onnx_verified = True
                else:
                    print_main(f"  ✗ ONNX verification FAILED: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}", rank)
                    print_main(f"    Large difference suggests export problem - check ONNX opset compatibility", rank)

                # Verify L2 normalization preserved
                onnx_norms = np.linalg.norm(onnx_output, axis=1)
                if np.allclose(onnx_norms, 1.0, atol=1e-4):
                    print_main(f"  ✓ L2 normalization preserved (norms={onnx_norms.mean():.6f})", rank)
                else:
                    print_main(f"  ⚠ L2 normalization issue: norms={onnx_norms.mean():.4f} (expected ~1.0)", rank)

        except ImportError:
            print_main("  ⚠ ONNX Runtime not available - skipping verification", rank)
            print_main("    Install with: pip install onnxruntime", rank)
        except Exception as e:
            # Log specific exception type for debugging
            exc_name = type(e).__name__
            if 'InvalidGraph' in exc_name:
                print_main(f"  ✗ ONNX model has invalid graph structure: {e}", rank)
            elif 'Fail' in exc_name or 'Runtime' in exc_name:
                print_main(f"  ✗ ONNX Runtime execution failed: {e}", rank)
            else:
                print_main(f"  ✗ ONNX verification error ({exc_name}): {e}", rank)
            import traceback
            print_main(f"    Full traceback:\n{traceback.format_exc()}", rank)

        if not onnx_verified:
            print_main("  ⚠ ONNX model may not work correctly on Raspberry Pi - verify before deployment!", rank)

        # Save class mapping for inference
        class_mapping_path = os.path.join(args.model_dir, 'class_to_idx.json')
        with open(class_mapping_path, 'w') as f:
            json.dump(class_to_idx, f, indent=2)
        print_main(f"  Class mapping: {class_mapping_path}", rank)

        # Save training config
        config_path = os.path.join(args.model_dir, 'training_config.json')
        config = {
            'embedding_dim': args.embedding_dim,
            'margin': args.margin,
            'mining_type': args.mining_type,
            'num_classes': num_classes,
            'best_recall1': best_recall1,
            'image_size': 224  # LeViT-384 input size per FINAL_PLAN.md
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print_main(f"  Training config: {config_path}", rank)

        print_main("\n" + "=" * 70, rank)
        print_main("✓ Training complete!", rank)
        print_main("=" * 70, rank)
        print_main(f"  Best Recall@1: {best_recall1:.2f}%", rank)
        print_main(f"  Model files in: {args.model_dir}", rank)
        print_main("=" * 70, rank)

    # Cleanup
    cleanup_distributed()


if __name__ == '__main__':
    train()
