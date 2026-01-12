# src/training/train_dinov3_teacher.py

import os
import argparse
import logging
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from PIL import Image, ImageFilter
from scipy.ndimage import rotate, convolve
from torch.utils.tensorboard import SummaryWriter

# Mixed precision training for 2-3x speedup on A100
from torch.cuda.amp import autocast, GradScaler

# Import from same directory (SageMaker compatibility)
from dinov3_embedding import DINOv3TeacherModel, ArcFaceLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WarmupCosineScheduler:
    """
    Learning rate scheduler with linear warmup + cosine annealing.

    Critical for distributed training with large batch sizes - prevents early instability.
    2025 best practice: warmup for 5-10% of total training.
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0

    def step(self):
        """Update learning rate based on current epoch."""
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (self.current_epoch + 1) / self.warmup_epochs
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group['lr'] = base_lr * warmup_factor
        else:
            # Cosine annealing after warmup
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group['lr'] = self.min_lr + (base_lr - self.min_lr) * cosine_factor

        self.current_epoch += 1

    def get_last_lr(self):
        """Return current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]


class ModelEMA:
    """
    Exponential Moving Average for model weights.

    Maintains a moving average of model parameters during training.
    Provides 1-2% accuracy boost and more stable inference.

    2025 best practice: EMA with decay=0.999 for large models (DINOv3).
    """
    def __init__(self, model, decay=0.999):
        # Handle torch.compile() wrapped models (OptimizedModule)
        # Extract the original model if it's wrapped
        if hasattr(model, '_orig_mod'):
            # torch.compile() wrapper - get the original model
            original_model = model._orig_mod
        else:
            original_model = model

        # Now extract model name and embedding dim from the original model
        model_name_raw = original_model.backbone.config._name_or_path.split('/')[-1]
        # Remove -pretrain-* suffix if present (DINOv3 models have this in HF repo name)
        # e.g., dinov3-vit7b16-pretrain-lvd1689m -> dinov3-vit7b16
        if '-pretrain-' in model_name_raw:
            model_name = model_name_raw.split('-pretrain-')[0]
            logger.info(f"EMA: Extracted model name '{model_name}' from '{model_name_raw}'")
        else:
            model_name = model_name_raw
            logger.info(f"EMA: Using model name '{model_name}' (no suffix to strip)")
        embedding_dim = original_model.projection[-1].out_features

        # Create a fresh instance of DINOv3TeacherModel (not compiled)
        from dinov3_embedding import DINOv3TeacherModel
        logger.info(f"EMA: Creating fresh model instance with name='{model_name}', embedding_dim={embedding_dim}")
        self.module = DINOv3TeacherModel(
            model_name=model_name,
            embedding_dim=embedding_dim,
            freeze_backbone=False,
            unfreeze_last_n_blocks=0
        )
        logger.info(f"EMA: Model instance created successfully")
        self.module.eval()
        self.decay = decay
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model):
        """Update EMA weights."""
        self.num_updates += 1
        # Adaptive decay: start slow, increase over time
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        # Handle torch.compile() + DDP wrappers (unwrap to get actual model)
        # Possible combinations:
        # 1. model (no wrapping)
        # 2. OptimizedModule(model) - torch.compile()
        # 3. DDP(model)
        # 4. DDP(OptimizedModule(model)) - both DDP and compile

        unwrapped = model
        # Unwrap DDP if present
        if hasattr(unwrapped, 'module'):
            unwrapped = unwrapped.module
        # Unwrap torch.compile() if present
        if hasattr(unwrapped, '_orig_mod'):
            unwrapped = unwrapped._orig_mod

        model_state = unwrapped.state_dict()
        ema_state = self.module.state_dict()

        for key in ema_state.keys():
            if model_state[key].dtype.is_floating_point:
                ema_state[key].mul_(decay).add_(model_state[key], alpha=1 - decay)

    def to(self, device):
        """Move EMA model to device."""
        self.module = self.module.to(device)
        return self


class EarlyStopping:
    """
    Early stopping to prevent overfitting and save compute time.

    Monitors validation metric and stops training if no improvement for N epochs.
    2025 best practice: patience=5-10 for fine-tuning large models.
    """
    def __init__(self, patience=7, min_delta=0.0, mode='max'):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for accuracy (higher is better), 'min' for loss (lower is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        """
        Call after each epoch with validation metric.
        Returns True if should stop training.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


@torch.no_grad()
def visualize_embeddings(model, dataloader, device, writer, epoch, num_samples=1000):
    """
    Visualize embeddings using TensorBoard's built-in projector.

    Extracts embeddings from validation set and logs to TensorBoard for visualization.
    TensorBoard will automatically compute PCA/t-SNE/UMAP.
    """
    model.eval()

    embeddings_list = []
    labels_list = []
    images_list = []

    # Collect samples
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Extract embeddings
        with autocast():
            embeddings = model(images)

        embeddings_list.append(embeddings.cpu())
        labels_list.append(labels.cpu())
        images_list.append(images.cpu())

        # Stop after num_samples
        if sum(e.size(0) for e in embeddings_list) >= num_samples:
            break

    # Concatenate
    embeddings_tensor = torch.cat(embeddings_list, dim=0)[:num_samples]
    labels_tensor = torch.cat(labels_list, dim=0)[:num_samples]
    images_tensor = torch.cat(images_list, dim=0)[:num_samples]

    # Log to TensorBoard (TensorBoard will compute PCA/t-SNE/UMAP automatically)
    writer.add_embedding(
        embeddings_tensor,
        metadata=labels_tensor.tolist(),
        label_img=images_tensor,
        global_step=epoch,
        tag='embeddings'
    )


def log_augmentation_samples(dataloader, writer, num_samples=16):
    """
    Log sample images with augmentations to TensorBoard.

    Visualizes training augmentations to verify they're appropriate for the task.
    """
    import torchvision.utils as vutils

    # Get one batch
    images, labels = next(iter(dataloader))

    # Select first num_samples
    images = images[:num_samples]
    labels = labels[:num_samples]

    # Denormalize for visualization (ImageNet stats)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)

    # Create grid
    grid = vutils.make_grid(images, nrow=4, padding=2, normalize=False)

    # Log to TensorBoard
    writer.add_image('Augmentation_Samples', grid, 0)
    writer.add_text('Augmentation_Labels', f'Sample labels: {labels.tolist()}', 0)


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


class GlareAugmentation:
    """Add realistic glare/reflection to simulate glossy card surfaces."""
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img):
        if np.random.random() > self.p:
            return img

        # Convert to numpy
        img_np = np.array(img)
        h, w = img_np.shape[:2]

        # Random glare parameters
        glare_x = np.random.randint(0, w)
        glare_y = np.random.randint(0, h)
        glare_radius = np.random.randint(min(h, w) // 8, min(h, w) // 4)
        glare_intensity = np.random.uniform(0.3, 0.7)

        # Create glare mask (Gaussian)
        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((x_grid - glare_x)**2 + (y_grid - glare_y)**2)
        glare_mask = np.exp(-(dist**2) / (2 * glare_radius**2))
        glare_mask = (glare_mask * glare_intensity * 255).astype(np.uint8)

        # Apply glare
        if len(img_np.shape) == 2:  # Grayscale
            img_np = np.clip(img_np + glare_mask, 0, 255).astype(np.uint8)
        else:  # RGB
            for c in range(3):
                img_np[:, :, c] = np.clip(img_np[:, :, c] + glare_mask, 0, 255).astype(np.uint8)

        return Image.fromarray(img_np)


class SleeveAugmentation:
    """
    Simulate card sleeves - critical for real-world Pokemon card recognition.
    Research shows this is essential for 95%+ accuracy through sleeves.
    """
    def __init__(self, p=0.4):
        self.p = p

    def __call__(self, img):
        if np.random.random() > self.p:
            return img

        img_np = np.array(img)
        h, w = img_np.shape[:2]

        sleeve_type = np.random.choice(['gloss', 'matte', 'scuffed'])

        if sleeve_type == 'gloss':
            # Glossy sleeve: slight glare + color tint
            glare = np.random.uniform(0.05, 0.15)
            img_np = np.clip(img_np * (1 + glare), 0, 255).astype(np.uint8)
            # Slight blue tint from plastic
            img_np[:, :, 2] = np.clip(img_np[:, :, 2] * 1.02, 0, 255).astype(np.uint8)

        elif sleeve_type == 'matte':
            # Matte sleeve: reduces contrast slightly
            img_np = np.clip(img_np * 0.95 + 10, 0, 255).astype(np.uint8)

        elif sleeve_type == 'scuffed':
            # Scuffed sleeve: random scratches/lines
            num_scratches = np.random.randint(3, 8)
            for _ in range(num_scratches):
                x1 = np.random.randint(0, w)
                y1 = np.random.randint(0, h)
                length = np.random.randint(10, 50)
                angle = np.random.uniform(0, 2 * np.pi)
                x2 = int(x1 + length * np.cos(angle))
                y2 = int(y1 + length * np.sin(angle))
                # Draw thin line (scratch)
                rr, cc = np.linspace(y1, y2, length).astype(int), np.linspace(x1, x2, length).astype(int)
                valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
                rr, cc = rr[valid], cc[valid]
                img_np[rr, cc] = np.clip(img_np[rr, cc] * 0.7, 0, 255).astype(np.uint8)

        return Image.fromarray(img_np)


class MotionBlurAugmentation:
    """Simulate camera shake/motion blur - different from static out-of-focus blur."""
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if np.random.random() > self.p:
            return img

        # Random motion blur direction and intensity
        size = np.random.choice([3, 5, 7])
        angle = np.random.uniform(0, 360)

        # Create motion blur kernel
        kernel = np.zeros((size, size))
        kernel[size // 2, :] = 1
        kernel = kernel / kernel.sum()

        # Rotate kernel to create directional blur
        kernel = rotate(kernel, angle, reshape=False)
        kernel = kernel / kernel.sum()

        # Apply motion blur kernel using convolution
        img_np = np.array(img)
        if len(img_np.shape) == 2:  # Grayscale
            img_np = convolve(img_np, kernel, mode='reflect')
        else:  # RGB
            for c in range(3):
                img_np[:, :, c] = convolve(img_np[:, :, c], kernel, mode='reflect')

        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np)


class ShadowAugmentation:
    """Add hand/object shadows when photographing cards."""
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img):
        if np.random.random() > self.p:
            return img

        img_np = np.array(img)
        h, w = img_np.shape[:2]

        # Random shadow from edge
        shadow_side = np.random.choice(['top', 'left', 'right', 'bottom'])
        shadow_depth = np.random.uniform(0.3, 0.7)
        shadow_fade = np.random.randint(h // 4, h // 2)

        # Create gradient mask
        if shadow_side == 'top':
            mask = np.linspace(shadow_depth, 1.0, shadow_fade)
            mask = np.pad(mask, (0, h - shadow_fade), constant_values=1.0)
            mask = np.tile(mask[:, np.newaxis], (1, w))
        elif shadow_side == 'bottom':
            mask = np.linspace(1.0, shadow_depth, shadow_fade)
            mask = np.pad(mask, (h - shadow_fade, 0), constant_values=1.0)
            mask = np.tile(mask[:, np.newaxis], (1, w))
        elif shadow_side == 'left':
            mask = np.linspace(shadow_depth, 1.0, shadow_fade)
            mask = np.pad(mask, (0, w - shadow_fade), constant_values=1.0)
            mask = np.tile(mask[np.newaxis, :], (h, 1))
        else:  # right
            mask = np.linspace(1.0, shadow_depth, shadow_fade)
            mask = np.pad(mask, (w - shadow_fade, 0), constant_values=1.0)
            mask = np.tile(mask[np.newaxis, :], (h, 1))

        # Apply shadow
        for c in range(3):
            img_np[:, :, c] = np.clip(img_np[:, :, c] * mask, 0, 255).astype(np.uint8)

        return Image.fromarray(img_np)


class HolographicAugmentation:
    """Simulate holographic/foil card patterns - critical for special Pokemon cards."""
    def __init__(self, p=0.15):
        self.p = p

    def __call__(self, img):
        if np.random.random() > self.p:
            return img

        img_np = np.array(img)
        h, w = img_np.shape[:2]

        # Create rainbow-like pattern
        y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        # Sine wave pattern for holographic effect
        pattern = np.sin(x_grid * 0.1 + y_grid * 0.05) * 0.5 + 0.5
        pattern = (pattern * 100).astype(np.uint8)

        # Apply to random channel(s) with low intensity
        channels = np.random.choice([0, 1, 2], size=np.random.randint(1, 3), replace=False)
        for c in channels:
            img_np[:, :, c] = np.clip(img_np[:, :, c] + pattern * 0.3, 0, 255).astype(np.uint8)

        return Image.fromarray(img_np)


def get_transforms(image_size: int = 224):
    """
    State-of-the-art augmentations for Pokemon card recognition (2026 research):

    GEOMETRIC:
    - Rotation (±15°) - cards at angles
    - Perspective (0.2) - viewing angles
    - Random crop - positional variation

    LIGHTING & COLOR:
    - ColorJitter - different lighting conditions
    - Exposure variation - over/underexposed images
    - Shadow simulation - hand/object shadows

    REAL-WORLD CONDITIONS:
    - Sleeve simulation (gloss/matte/scuffed) - CRITICAL for 95%+ accuracy
    - Glare - glossy card reflections
    - Holographic patterns - foil/special cards

    QUALITY DEGRADATION:
    - GaussianBlur - out-of-focus
    - MotionBlur - camera shake
    - RandomErasing - occlusion/partial coverage

    Research: PokéScope achieved 95%+ accuracy by training on 10,000+ sleeve variations
    """
    from PIL import Image, ImageFilter

    train_transform = transforms.Compose([
        # Geometric augmentations
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomRotation(degrees=15, fill=255),  # Card at angle
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3, fill=255),  # Perspective
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),

        # Lighting & color augmentations
        transforms.ColorJitter(
            brightness=0.5,  # Enhanced for exposure variation (0.4 -> 0.5)
            contrast=0.4,
            saturation=0.3,
            hue=0.1
        ),
        transforms.RandomGrayscale(p=0.05),

        # Real-world condition augmentations (CRITICAL for Pokemon cards)
        SleeveAugmentation(p=0.4),        # Gloss/matte/scuffed sleeves - MOST IMPORTANT
        ShadowAugmentation(p=0.3),        # Hand shadows when photographing
        GlareAugmentation(p=0.25),        # Glossy card reflections
        HolographicAugmentation(p=0.15),  # Foil/special card patterns

        # Blur augmentations
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        ], p=0.3),                         # Out-of-focus
        MotionBlurAugmentation(p=0.2),    # Camera shake

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

        # Occlusion (partial coverage by hand/other cards)
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


def train_epoch(model, loss_fn, dataloader, optimizer, device, scaler, ema_model=None, rank=0):
    """
    Training epoch with MIXED PRECISION for 2-3x speedup.

    Args:
        scaler: GradScaler for automatic mixed precision
        ema_model: Exponential moving average model (optional)
        rank: GPU rank (only show tqdm on rank 0)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # Only show tqdm on rank 0 to avoid I/O contention
    iterator = tqdm(dataloader, desc='Training') if rank == 0 else dataloader

    for images, labels in iterator:
        images = images.to(device, non_blocking=True)  # non_blocking for async transfer
        labels = labels.to(device, non_blocking=True)

        # MIXED PRECISION: Forward pass in FP16
        with autocast():
            embeddings = model(images)
            loss = loss_fn(embeddings, labels)

        # MIXED PRECISION: Backward pass with gradient scaling + clipping
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # Gradient clipping (prevents exploding gradients, critical for distributed training)
        scaler.unscale_(optimizer)  # Unscale before clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        # Update EMA weights after optimizer step
        if ema_model is not None:
            ema_model.update(model)

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
def evaluate(model, dataloader, device, rank=0):
    """
    Evaluate using nearest neighbor retrieval.

    OPTIMIZED: Keep everything on GPU (avoid slow CPU transfers).

    Args:
        rank: GPU rank (only show tqdm on rank 0)
    """
    model.eval()

    all_embeddings = []
    all_labels = []

    # Only show tqdm on rank 0
    iterator = tqdm(dataloader, desc='Evaluating') if rank == 0 else dataloader

    for images, labels in iterator:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)  # Keep on GPU

        # Mixed precision for evaluation too
        with autocast():
            embeddings = model(images)

        all_embeddings.append(embeddings)  # Keep on GPU!
        all_labels.append(labels)           # Keep on GPU!

    embeddings = torch.cat(all_embeddings, dim=0)  # All on GPU
    labels = torch.cat(all_labels, dim=0)          # All on GPU

    # Compute pairwise distances (ON GPU - much faster!)
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
    # ========== Distributed Training Setup ==========
    # Check if running in distributed mode (SageMaker sets these environment variables)
    is_distributed = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1

    if is_distributed:
        import torch.distributed as dist

        # Initialize process group (required for DDP)
        dist.init_process_group(backend='nccl')

        # Get rank and local_rank from environment
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        # Set device to local GPU
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')

        logger.info(f"Distributed training: rank {rank}/{world_size}, local_rank {local_rank}")
        logger.info(f"Using device: {device}")

        # Verify we have the expected number of GPUs (8 for p4d.24xlarge)
        if world_size != 8:
            logger.warning(f"Expected 8 GPUs for p4d.24xlarge, found {world_size}")
    else:
        # Single GPU mode
        if not torch.cuda.is_available():
            logger.error("CRITICAL: No CUDA devices available")
            logger.error("Training requires GPU. Running on CPU would be extremely slow.")
            raise RuntimeError("No GPU available. Check instance configuration.")

        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device('cuda:0')
        logger.info(f"Single GPU mode: {device}")

    logger.info(f"DINOv3 model: {args.dinov3_model}")

    # Ensure model directory exists (only on rank 0 to avoid race condition)
    if rank == 0:
        os.makedirs(args.model_dir, exist_ok=True)

    # ========== MLflow & TensorBoard Setup ==========
    # Set tracking URI (SageMaker default or local)
    mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'file:///opt/ml/output/mlflow')
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Initialize TensorBoard (only on rank 0)
    tensorboard_dir = '/opt/ml/output/tensorboard' if rank == 0 else None
    writer = SummaryWriter(log_dir=tensorboard_dir) if rank == 0 else None

    # Start MLflow run
    experiment_name = "dinov3-pokemon-teacher"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"dinov3_{args.dinov3_model}"):
        # Log all hyperparameters
        mlflow.log_params({
            'model': args.dinov3_model,
            'embedding_dim': args.embedding_dim,
            'epochs_frozen': args.epochs_frozen,
            'epochs_unfrozen': args.epochs_unfrozen,
            'unfreeze_blocks': args.unfreeze_blocks,
            'batch_size': args.batch_size,
            'lr_frozen': args.lr_frozen,
            'lr_unfrozen': args.lr_unfrozen,
            'weight_decay': args.weight_decay,
            'arcface_margin': args.arcface_margin,
            'arcface_scale': args.arcface_scale,
            'device': str(device),
        })

        logger.info(f"MLflow tracking URI: {mlflow_tracking_uri}")
        logger.info(f"MLflow experiment: {experiment_name}")
        logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")

        # Load datasets
        train_transform, val_transform = get_transforms(224)

        train_dataset = datasets.ImageFolder(args.train_dir, transform=train_transform)
        num_classes = len(train_dataset.classes)

        # Validate dataset
        if num_classes == 0:
            logger.error(f"No classes found in training directory: {args.train_dir}")
            raise ValueError("Training directory contains no valid class subdirectories")
        if len(train_dataset) == 0:
            logger.error(f"No training samples found in {args.train_dir}")
            raise ValueError("Training directory contains no valid images")
        if num_classes < 1000:
            logger.warning(f"Dataset may be incomplete: only {num_classes} classes (expected ~17,592)")

        logger.info(f"Number of classes: {num_classes}")
        logger.info(f"Training samples: {len(train_dataset)}")

        # Verify dataset is actually accessible (defensive check)
        if rank == 0:
            try:
                test_sample = train_dataset[0]
                logger.info(f"✅ Dataset verification passed - sample loaded successfully")
            except Exception as e:
                logger.error(f"❌ Dataset verification FAILED: {e}")
                logger.error(f"   Train directory: {args.train_dir}")
                logger.error(f"   Directory exists: {Path(args.train_dir).exists()}")
                if Path(args.train_dir).exists():
                    logger.error(f"   Contents: {list(Path(args.train_dir).iterdir())[:5]}")
                raise RuntimeError(f"Cannot load training data: {e}")

        mlflow.log_param('num_classes', num_classes)

        # Validation is optional (use 10% of training data if no val_dir)
        val_dir_path = Path(args.val_dir)
        if val_dir_path.exists() and len(list(val_dir_path.iterdir())) > 0:
            val_dataset = datasets.ImageFolder(args.val_dir, transform=val_transform)
            logger.info(f"Using validation set: {len(val_dataset)} images")
        else:
            # Use 10% of training data for validation (deterministic split with fixed seed)
            # Load dataset once to get total size
            full_dataset_for_count = datasets.ImageFolder(args.train_dir, transform=None)
            total_size = len(full_dataset_for_count)
            train_size = int(0.9 * total_size)
            val_size = total_size - train_size

            # Generate split indices
            generator = torch.Generator().manual_seed(42)
            indices = torch.randperm(total_size, generator=generator).tolist()
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            # Create datasets with correct transforms, then subset
            from torch.utils.data import Subset
            full_train_dataset = datasets.ImageFolder(args.train_dir, transform=train_transform)
            full_val_dataset = datasets.ImageFolder(args.train_dir, transform=val_transform)

            train_dataset = Subset(full_train_dataset, train_indices)
            val_dataset = Subset(full_val_dataset, val_indices)

            logger.info(f"No validation set found, using 10% of training data with seed=42: {val_size} images")
            logger.info(f"  Train subset: {len(train_dataset)} images with augmentations")
            logger.info(f"  Val subset: {len(val_dataset)} images without augmentations")

        mlflow.log_param('train_size', len(train_dataset))
        mlflow.log_param('val_size', len(val_dataset))

        # Create DataLoaders with OPTIMIZED settings for 8xA100
        # num_workers=4: Reduced from 8 to avoid file handle issues with persistent workers
        # persistent_workers=False: Disabled to avoid race conditions in distributed training
        # prefetch_factor=2: Pipeline 2 batches ahead
        if is_distributed:
            from torch.utils.data.distributed import DistributedSampler
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True  # Drop last incomplete batch for consistent batch sizes
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                sampler=train_sampler,  # Use sampler instead of shuffle
                num_workers=4,  # Reduced to avoid file handle issues
                pin_memory=True,
                persistent_workers=False,  # Disabled for distributed training reliability
                prefetch_factor=2  # Pre-fetch 2 batches per worker
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=4, pin_memory=True,
                persistent_workers=False, prefetch_factor=2
            )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
            persistent_workers=False, prefetch_factor=2
        )

        # Create model (backbone frozen initially)
        if rank == 0:
            # Verify HuggingFace token is available
            hf_token = os.environ.get('HUGGING_FACE_HUB_TOKEN', '')
            if hf_token:
                logger.info(f"✅ HuggingFace token available ({len(hf_token)} chars)")
            else:
                logger.warning("⚠️  No HuggingFace token - gated models will fail!")
            logger.info(f"Loading DINOv3 model: {args.dinov3_model}")

        model = DINOv3TeacherModel(
            model_name=args.dinov3_model,
            embedding_dim=args.embedding_dim,
            freeze_backbone=True,
        ).to(device)

        if rank == 0:
            logger.info(f"✅ Model loaded successfully")

        # OPTIMIZATION: torch.compile for 10-20% speedup (PyTorch 2.0+)
        # Compile BEFORE DDP wrapping for best performance
        try:
            if rank == 0:
                logger.info("Compiling model with torch.compile...")
            model = torch.compile(model, mode='reduce-overhead')
            if rank == 0:
                logger.info("Model compiled successfully")
        except Exception as e:
            if rank == 0:
                logger.warning(f"torch.compile failed (PyTorch < 2.0?): {e}")

        # Wrap model in DistributedDataParallel if using multiple GPUs
        if is_distributed:
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False  # All parameters should be used
            )
            logger.info(f"Model wrapped in DistributedDataParallel")

        # Log model architecture info
        # Access model.module when using DDP
        model_for_inspection = model.module if is_distributed else model
        total_params = sum(p.numel() for p in model_for_inspection.parameters())
        trainable_params = sum(p.numel() for p in model_for_inspection.parameters() if p.requires_grad)
        mlflow.log_param('total_params', total_params)
        mlflow.log_param('trainable_params_phase1', trainable_params)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters (Phase 1): {trainable_params:,}")

        # EMA: Exponential Moving Average for 1-2% accuracy boost
        ema_model = ModelEMA(model_for_inspection, decay=0.999).to(device)
        if rank == 0:
            logger.info("EMA enabled (decay=0.999)")

        loss_fn = ArcFaceLoss(
            embedding_dim=args.embedding_dim,
            num_classes=num_classes,
            margin=args.arcface_margin,
            scale=args.arcface_scale,
        ).to(device)

        # ========== MIXED PRECISION TRAINING SETUP ==========
        # GradScaler for automatic mixed precision (2-3x speedup on A100)
        scaler = GradScaler()
        if rank == 0:
            logger.info("Mixed precision training enabled (FP16)")

        # ========== AUTOMATIC RESUME FROM CHECKPOINT ==========
        resume_phase1_checkpoint = None
        resume_phase2_checkpoint = None
        start_epoch_phase1 = 0
        start_epoch_phase2 = 0

        # Check for existing checkpoints
        phase1_ckpt_path = f'{args.model_dir}/phase1_checkpoint.pt'
        phase2_ckpt_path = f'{args.model_dir}/phase2_checkpoint.pt'

        if os.path.exists(phase2_ckpt_path) and rank == 0:
            logger.info(f"Found Phase 2 checkpoint: {phase2_ckpt_path}")
            resume_phase2_checkpoint = torch.load(phase2_ckpt_path, map_location=device)
            logger.info(f"  Resuming from epoch {resume_phase2_checkpoint['epoch'] + 1}")
        elif os.path.exists(phase1_ckpt_path) and rank == 0:
            logger.info(f"Found Phase 1 checkpoint: {phase1_ckpt_path}")
            resume_phase1_checkpoint = torch.load(phase1_ckpt_path, map_location=device)
            logger.info(f"  Resuming from epoch {resume_phase1_checkpoint['epoch'] + 1}")

        # Broadcast resume decision to all ranks
        if is_distributed:
            resume_flags = torch.tensor(
                [resume_phase1_checkpoint is not None, resume_phase2_checkpoint is not None],
                dtype=torch.bool, device=device
            )
            dist.broadcast(resume_flags, src=0)
            if rank != 0:
                if resume_flags[1]:
                    resume_phase2_checkpoint = torch.load(phase2_ckpt_path, map_location=device)
                elif resume_flags[0]:
                    resume_phase1_checkpoint = torch.load(phase1_ckpt_path, map_location=device)

        # ========== PHASE 1: Train projection head only ==========
        # Skip Phase 1 if resuming from Phase 2
        if resume_phase2_checkpoint is None:
            logger.info("=" * 60)
            logger.info("PHASE 1: Training projection head (backbone frozen)")
            logger.info("=" * 60)

            # Access model.module for parameters when using DDP
            model_params = model.module if is_distributed else model
            optimizer = torch.optim.AdamW(
                list(model_params.projection.parameters()) + list(loss_fn.parameters()),
                lr=args.lr_frozen,
                weight_decay=args.weight_decay,
            )

            # LR warmup + cosine annealing (prevents early instability with large batch sizes)
            warmup_epochs_phase1 = max(1, int(args.epochs_frozen * 0.1))  # 10% warmup
            scheduler = WarmupCosineScheduler(
                optimizer,
                warmup_epochs=warmup_epochs_phase1,
                total_epochs=args.epochs_frozen,
                min_lr=1e-7
            )
            if rank == 0:
                logger.info(f"Phase 1: LR warmup for {warmup_epochs_phase1} epoch(s), then cosine annealing")

            best_acc = 0
            early_stopping_phase1 = EarlyStopping(patience=5, mode='max')
            if rank == 0:
                logger.info("Early stopping enabled (patience=5 epochs)")

            # Resume from Phase 1 checkpoint if available
            if resume_phase1_checkpoint is not None and resume_phase2_checkpoint is None:
                if rank == 0:
                    logger.info("Restoring Phase 1 checkpoint state...")
                model_for_inspection.load_state_dict(resume_phase1_checkpoint['model'])
                ema_model.module.load_state_dict(resume_phase1_checkpoint['ema_model'])
                loss_fn.load_state_dict(resume_phase1_checkpoint['loss_fn'])
                optimizer.load_state_dict(resume_phase1_checkpoint['optimizer'])
                scheduler = resume_phase1_checkpoint['scheduler']
                start_epoch_phase1 = resume_phase1_checkpoint['epoch'] + 1
                best_acc = resume_phase1_checkpoint['top1_acc']
                if rank == 0:
                    logger.info(f"  Restored: epoch={start_epoch_phase1}, best_acc={best_acc:.2%}")

            # Log augmentation samples at start of training (only on rank 0)
            if writer and rank == 0 and start_epoch_phase1 == 0:
                logger.info("Logging augmentation samples...")
                log_augmentation_samples(train_loader, writer, num_samples=16)

            for epoch in range(start_epoch_phase1, args.epochs_frozen):
                # Set epoch for distributed sampler (ensures different shuffle each epoch)
                if is_distributed:
                    train_sampler.set_epoch(epoch)

                if rank == 0:
                    logger.info(f"\nEpoch {epoch + 1}/{args.epochs_frozen} (Frozen)")

                train_loss, train_acc = train_epoch(model, loss_fn, train_loader, optimizer, device, scaler, ema_model, rank)
                top1_acc, top5_acc = evaluate(ema_model.module, val_loader, device, rank)  # Use EMA for validation
                current_lr = optimizer.param_groups[0]['lr']
                scheduler.step()

                logger.info(f"Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}")
                logger.info(f"Val Top1: {top1_acc:.2%}, Top5: {top5_acc:.2%}")

                # MLflow: Log metrics for this epoch
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_top1_accuracy': top1_acc,
                    'val_top5_accuracy': top5_acc,
                    'learning_rate': current_lr,
                }, step=epoch)

                # TensorBoard: Log metrics (only on rank 0)
                if writer:
                    writer.add_scalar('Phase1/Loss/train', train_loss, epoch)
                    writer.add_scalar('Phase1/Accuracy/train', train_acc, epoch)
                    writer.add_scalar('Phase1/Accuracy/val_top1', top1_acc, epoch)
                    writer.add_scalar('Phase1/Accuracy/val_top5', top5_acc, epoch)
                    writer.add_scalar('Phase1/LearningRate', current_lr, epoch)

                # Early stopping check
                if early_stopping_phase1(top1_acc):
                    if rank == 0:
                        logger.info(f"\nEarly stopping triggered! No improvement for {early_stopping_phase1.patience} epochs.")
                        logger.info(f"Best val top1: {best_acc:.2%}")
                    break

                if top1_acc > best_acc:
                    best_acc = top1_acc
                    # Only save checkpoint on rank 0 to avoid conflicts
                    if rank == 0:
                        checkpoint_path = f'{args.model_dir}/best_teacher_frozen.pt'
                        # Use model.module.state_dict() when using DDP
                        model_state = model.module.state_dict() if is_distributed else model.state_dict()
                        torch.save({
                            'model': model_state,
                            'ema_model': ema_model.module.state_dict(),  # Save EMA weights
                            'loss_fn': loss_fn.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler,
                            'epoch': epoch,
                            'top1_acc': top1_acc,
                            'num_classes': num_classes,  # Add num_classes for student distillation
                            'phase': 'phase1',
                        }, checkpoint_path)
                        logger.info(f"Saved best model (Phase 1): {checkpoint_path} (with EMA)")

                        # MLflow: Log best model checkpoint
                        mlflow.log_artifact(checkpoint_path, artifact_path='checkpoints')
                        mlflow.log_metric('best_val_top1_phase1', best_acc, step=epoch)

                # Save resume checkpoint every epoch (for auto-resume on crash)
                if rank == 0:
                    resume_checkpoint_path = f'{args.model_dir}/phase1_checkpoint.pt'
                    model_state = model.module.state_dict() if is_distributed else model.state_dict()
                    torch.save({
                        'model': model_state,
                        'ema_model': ema_model.module.state_dict(),
                        'loss_fn': loss_fn.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler,
                        'epoch': epoch,
                        'top1_acc': top1_acc,
                        'num_classes': num_classes,
                        'phase': 'phase1',
                    }, resume_checkpoint_path)

            # Visualize embeddings at end of Phase 1 (only on rank 0)
            if writer and rank == 0:
                logger.info("Generating embedding visualization for Phase 1...")
                visualize_embeddings(ema_model.module, val_loader, device, writer, epoch=args.epochs_frozen, num_samples=500)

        # ========== PHASE 2: Fine-tune backbone ==========
        logger.info("\n" + "=" * 60)
        logger.info(f"PHASE 2: Fine-tuning last {args.unfreeze_blocks} transformer blocks")
        logger.info("=" * 60)

        # Unfreeze backbone (use model.module when DDP)
        model_to_unfreeze = model.module if is_distributed else model
        model_to_unfreeze.unfreeze_backbone(last_n_blocks=args.unfreeze_blocks)

        # Log updated trainable params
        trainable_params = sum(p.numel() for p in model_to_unfreeze.parameters() if p.requires_grad)
        mlflow.log_param('trainable_params_phase2', trainable_params)
        logger.info(f"Trainable parameters (Phase 2): {trainable_params:,}")

        # Use model_to_unfreeze for accessing parameters
        optimizer = torch.optim.AdamW(
            [
                {'params': model_to_unfreeze.projection.parameters(), 'lr': args.lr_unfrozen * 10},
                {'params': model_to_unfreeze.backbone.layer[-args.unfreeze_blocks:].parameters(), 'lr': args.lr_unfrozen},
                {'params': loss_fn.parameters(), 'lr': args.lr_unfrozen * 10},
            ],
            weight_decay=args.weight_decay,
        )

        # LR warmup + cosine annealing (critical for fine-tuning pretrained transformers)
        warmup_epochs_phase2 = max(1, int(args.epochs_unfrozen * 0.1))  # 10% warmup
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=warmup_epochs_phase2,
            total_epochs=args.epochs_unfrozen,
            min_lr=1e-7
        )
        if rank == 0:
            logger.info(f"Phase 2: LR warmup for {warmup_epochs_phase2} epoch(s), then cosine annealing")

        early_stopping_phase2 = EarlyStopping(patience=7, mode='max')
        if rank == 0:
            logger.info("Early stopping enabled (patience=7 epochs)")

        # Resume from Phase 2 checkpoint if available (skip Phase 1 entirely)
        if resume_phase2_checkpoint is not None:
            if rank == 0:
                logger.info("Restoring Phase 2 checkpoint state...")
            model_for_inspection.load_state_dict(resume_phase2_checkpoint['model'])
            ema_model.module.load_state_dict(resume_phase2_checkpoint['ema_model'])
            loss_fn.load_state_dict(resume_phase2_checkpoint['loss_fn'])
            optimizer.load_state_dict(resume_phase2_checkpoint['optimizer'])
            scheduler = resume_phase2_checkpoint['scheduler']
            start_epoch_phase2 = resume_phase2_checkpoint['epoch'] - args.epochs_frozen + 1
            best_acc = resume_phase2_checkpoint['top1_acc']
            if rank == 0:
                logger.info(f"  Restored: epoch={start_epoch_phase2}, best_acc={best_acc:.2%}")
                logger.info("  Skipping Phase 1 (already completed)")

        for epoch in range(start_epoch_phase2, args.epochs_unfrozen):
            # Set epoch for distributed sampler
            if is_distributed:
                train_sampler.set_epoch(args.epochs_frozen + epoch)

            global_epoch = args.epochs_frozen + epoch
            if rank == 0:
                logger.info(f"\nEpoch {epoch + 1}/{args.epochs_unfrozen} (Fine-tuning)")

            train_loss, train_acc = train_epoch(model, loss_fn, train_loader, optimizer, device, scaler, ema_model, rank)
            top1_acc, top5_acc = evaluate(ema_model.module, val_loader, device, rank)  # Use EMA for validation
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()

            logger.info(f"Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}")
            logger.info(f"Val Top1: {top1_acc:.2%}, Top5: {top5_acc:.2%}")

            # MLflow: Log metrics for this epoch
            mlflow.log_metrics({
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_top1_accuracy': top1_acc,
                'val_top5_accuracy': top5_acc,
                'learning_rate': current_lr,
            }, step=global_epoch)

            # TensorBoard: Log metrics (only on rank 0)
            if writer:
                writer.add_scalar('Phase2/Loss/train', train_loss, global_epoch)
                writer.add_scalar('Phase2/Accuracy/train', train_acc, global_epoch)
                writer.add_scalar('Phase2/Accuracy/val_top1', top1_acc, global_epoch)
                writer.add_scalar('Phase2/Accuracy/val_top5', top5_acc, global_epoch)
                writer.add_scalar('Phase2/LearningRate', current_lr, global_epoch)

            # Early stopping check
            if early_stopping_phase2(top1_acc):
                if rank == 0:
                    logger.info(f"\nEarly stopping triggered! No improvement for {early_stopping_phase2.patience} epochs.")
                    logger.info(f"Best val top1: {best_acc:.2%}")
                break

            if top1_acc > best_acc:
                best_acc = top1_acc
                # Only save checkpoint on rank 0 to avoid conflicts
                if rank == 0:
                    checkpoint_path = f'{args.model_dir}/best_teacher.pt'
                    # Use model.module.state_dict() when using DDP
                    model_state = model.module.state_dict() if is_distributed else model.state_dict()
                    torch.save({
                        'model': model_state,
                        'ema_model': ema_model.module.state_dict(),  # Save EMA weights
                        'loss_fn': loss_fn.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler,
                        'epoch': global_epoch,
                        'top1_acc': top1_acc,
                        'num_classes': num_classes,  # Add num_classes for student distillation
                        'phase': 'phase2',
                    }, checkpoint_path)
                    logger.info(f"New best model! Top1: {top1_acc:.2%} (with EMA)")

                    # MLflow: Log best model checkpoint
                    mlflow.log_artifact(checkpoint_path, artifact_path='checkpoints')
                    mlflow.log_metric('best_val_top1_overall', best_acc, step=global_epoch)

            # Save resume checkpoint every epoch (for auto-resume on crash)
            if rank == 0:
                resume_checkpoint_path = f'{args.model_dir}/phase2_checkpoint.pt'
                model_state = model.module.state_dict() if is_distributed else model.state_dict()
                torch.save({
                    'model': model_state,
                    'ema_model': ema_model.module.state_dict(),
                    'loss_fn': loss_fn.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler,
                    'epoch': global_epoch,
                    'top1_acc': top1_acc,
                    'num_classes': num_classes,
                    'phase': 'phase2',
                }, resume_checkpoint_path)

        # Visualize embeddings at end of Phase 2 (only on rank 0)
        if writer and rank == 0:
            logger.info("Generating embedding visualization for Phase 2...")
            visualize_embeddings(ema_model.module, val_loader, device, writer, epoch=args.epochs_frozen + args.epochs_unfrozen, num_samples=500)

        # ========== Export (only on rank 0) ==========
        if rank == 0:
            logger.info("\nExporting teacher model...")

            model.eval()
            dummy_input = torch.randn(1, 3, 224, 224).to(device)

            # Use model.module when DDP for export
            export_model = model.module if is_distributed else model

            # Unwrap torch.compile() wrapper if present (ONNX export incompatible with compiled models)
            if hasattr(export_model, '_orig_mod'):
                export_model = export_model._orig_mod

            onnx_path = f'{args.model_dir}/dinov3_teacher.onnx'
            torch.onnx.export(
                export_model,
                dummy_input,
                onnx_path,
                input_names=['input'],
                output_names=['embedding'],
                dynamic_axes={'input': {0: 'batch'}, 'embedding': {0: 'batch'}},
                opset_version=17,
            )
            logger.info(f"ONNX export successful: {onnx_path}")

            # MLflow: Log ONNX model
            mlflow.log_artifact(onnx_path, artifact_path='models')

        # Save metrics (only on rank 0)
        if rank == 0:
            metrics = {
                'model': args.dinov3_model,
                'num_classes': num_classes,
                'embedding_dim': args.embedding_dim,
                'best_top1_accuracy': best_acc,
                'epochs_frozen': args.epochs_frozen,
                'epochs_unfrozen': args.epochs_unfrozen,
                'total_epochs': args.epochs_frozen + args.epochs_unfrozen,
            }

            metrics_path = f'{args.model_dir}/metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)

            # MLflow: Log final metrics
            mlflow.log_artifact(metrics_path, artifact_path='results')
            mlflow.log_metric('final_best_top1_accuracy', best_acc)

            # MLflow: Log PyTorch model (for model registry)
            mlflow.pytorch.log_model(export_model, "teacher_model")

        logger.info(f"\nTraining complete! Best Top1: {best_acc:.2%}")
        if rank == 0:
            logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")
        logger.info(f"View in MLflow UI: {mlflow_tracking_uri}")

        # Close TensorBoard writer
        if writer:
            writer.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
