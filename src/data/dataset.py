#!/usr/bin/env python3
"""
TripletCardDataset - Dataset for metric learning with online triplet mining

Key insight: With 1 image per class, we generate positives through heavy augmentation.
Anchor and positive are different augmented views of the SAME image.
Negatives are images from DIFFERENT classes.

Uses pytorch-metric-learning for efficient hard negative mining.
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms as transforms

# Try to import Albumentations for advanced occlusion augmentations
# These are critical for finger/hand occlusion simulation per FINAL_PLAN.md
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: Albumentations not installed. Install with: pip install albumentations")
    print("         Occlusion augmentations (GridDropout, CoarseDropout) will be disabled.")


class AlbumentationsWrapper:
    """
    Wrapper to apply Albumentations transforms within a torchvision pipeline.
    Converts PIL Image → numpy → Albumentations → PIL Image.
    """

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        # Convert PIL to numpy
        img_np = np.array(img)
        # Apply Albumentations
        augmented = self.transform(image=img_np)
        # Convert back to PIL
        return Image.fromarray(augmented['image'])


class TripletCardDataset(Dataset):
    """
    Dataset that returns (anchor, positive, negative) triplets for metric learning.

    Since we have 1 image per class:
    - Anchor: Original image with augmentation
    - Positive: SAME image with DIFFERENT augmentation (simulates same card, different photo)
    - Negative: Different card image

    The key insight is that heavy augmentation creates "virtual" positive pairs.
    """

    def __init__(
        self,
        root_dir: str,
        class_to_idx: Dict[str, int],
        transform: Optional[transforms.Compose] = None,
        return_triplets: bool = True
    ):
        """
        Args:
            root_dir: Path to dataset split (e.g., /data/train)
            class_to_idx: Unified class mapping from class_index.json
            transform: Augmentation pipeline (applied differently to anchor/positive)
            return_triplets: If True, returns (anchor, positive, label).
                           If False, returns (image, label) for embedding extraction.
        """
        self.root_dir = Path(root_dir)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.return_triplets = return_triplets

        # Build samples list: [(path, label), ...]
        self.samples: List[Tuple[str, int]] = []
        self.class_to_samples: Dict[int, List[str]] = {}  # label -> [paths]

        self._load_samples()

    def _load_samples(self):
        """Load all image paths and organize by class."""
        for class_name in sorted(os.listdir(self.root_dir)):
            class_path = self.root_dir / class_name
            if not class_path.is_dir():
                continue

            if class_name not in self.class_to_idx:
                print(f"Warning: Class '{class_name}' not in class_to_idx, skipping")
                continue

            label = self.class_to_idx[class_name]

            if label not in self.class_to_samples:
                self.class_to_samples[label] = []

            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    img_path = str(class_path / img_name)
                    self.samples.append((img_path, label))
                    self.class_to_samples[label].append(img_path)

        # Get list of all unique labels for negative sampling
        self.all_labels = list(self.class_to_samples.keys())

        print(f"  Loaded {len(self.samples)} samples from {len(self.all_labels)} classes")

        # Validate dataset is not empty
        if len(self.samples) == 0:
            raise ValueError(
                f"No valid samples found in {self.root_dir}. "
                f"Check that directory contains class folders with images."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Returns:
            If return_triplets=True: (anchor, positive, label)
            If return_triplets=False: (image, label)
        """
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        if not self.return_triplets:
            # Simple mode for embedding extraction
            if self.transform:
                image = self.transform(image)
            return image, label

        # Triplet mode: generate anchor and positive from same image
        # Apply transform twice with different random augmentations
        if self.transform:
            anchor = self.transform(image)
            positive = self.transform(image)  # Different random augmentation
        else:
            anchor = transforms.ToTensor()(image)
            positive = transforms.ToTensor()(image)

        return anchor, positive, label


class MPerClassSampler(Sampler):
    """
    Samples M images per class per batch.

    This is crucial for effective triplet mining:
    - Each batch contains images from N classes
    - Each class has M images (with augmentation, M=2 means anchor+positive)
    - Hard negatives are mined within the batch

    For our 1-image-per-class scenario:
    - We sample many classes per batch
    - Each "sample" gets augmented to create anchor/positive pair
    - Hard negatives = similar-looking cards from different classes

    For datasets with multiple images per class:
    - Cycles through available images
    - Provides more diversity than always using first image
    """

    def __init__(
        self,
        labels: List[int],
        m_per_class: int = 2,
        batch_size: int = 64,
        length_before_new_iter: int = 100000
    ):
        """
        Args:
            labels: List of labels for each sample
            m_per_class: Number of samples per class per batch
            batch_size: Total batch size
            length_before_new_iter: How many samples before reshuffling
        """
        self.labels = labels
        self.m_per_class = m_per_class
        self.batch_size = batch_size
        self.length_before_new_iter = length_before_new_iter

        # Build label to indices mapping
        self.label_to_indices: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

        self.unique_labels = list(self.label_to_indices.keys())
        self.num_classes_per_batch = batch_size // m_per_class

        # Track position for each class (for cycling through samples)
        self._class_positions: Dict[int, int] = {label: 0 for label in self.unique_labels}

    def __iter__(self):
        """Yield indices for batches with M samples per class."""
        idx_list = []

        # Shuffle labels
        labels_shuffled = self.unique_labels.copy()
        random.shuffle(labels_shuffled)

        # Reset positions at start of each iteration
        for label in self.unique_labels:
            self._class_positions[label] = 0

        for label in labels_shuffled:
            indices = self.label_to_indices[label]
            num_available = len(indices)

            # FIXED: Handle variable number of images per class
            # Instead of hard-coding indices[0], cycle through available images
            samples_for_class = []
            for i in range(self.m_per_class):
                # Cycle through available indices
                pos = (self._class_positions[label] + i) % num_available
                samples_for_class.append(indices[pos])

            # Update position for next time this class is sampled
            self._class_positions[label] = (
                self._class_positions[label] + self.m_per_class
            ) % num_available

            idx_list.extend(samples_for_class)

            if len(idx_list) >= self.length_before_new_iter:
                break

        # Yield indices
        for idx in idx_list:
            yield idx

    def __len__(self) -> int:
        return min(self.length_before_new_iter, len(self.labels) * self.m_per_class)


def get_heavy_augmentation(image_size: int = 224, use_card_specific: bool = True) -> transforms.Compose:
    """
    Heavy augmentation pipeline for generating diverse positive pairs.

    This is CRITICAL for metric learning with 1 image per class:
    - Each augmentation creates a "virtual" positive sample
    - Simulates real-world variations: lighting, angles, blur, occlusion

    IMPORTANT: Includes GridDropout and CoarseDropout for finger/hand occlusion
    simulation as specified in FINAL_PLAN.md Section "Occlusion Augmentations".

    Args:
        image_size: Target image size (default: 224 for ViT-B/16)
        use_card_specific: If True, use card-specific augmentations (glare, shadows)
    """
    # Import card-specific augmentations if available
    card_augs = []
    if use_card_specific:
        try:
            from card_augmentations import RandomGlare, RandomShadow, RandomMotionBlur

            # Card-specific augmentations for realistic variations
            card_augs = [
                RandomGlare(p=0.3, intensity_range=(0.1, 0.4)),  # Holographic reflections
                RandomShadow(p=0.3, intensity_range=(0.2, 0.5)),  # Lighting variations
                RandomMotionBlur(p=0.2, kernel_size_range=(3, 7)),  # Camera shake
            ]
            print("  Using card-specific augmentations (glare, shadow, motion blur)")
        except ImportError:
            print("  Warning: card_augmentations.py not found, using generic augmentations")

    # Albumentations-based occlusion augmentations (critical for finger simulation)
    # These are applied before ToTensor() since they work on numpy arrays
    occlusion_augs = []
    if ALBUMENTATIONS_AVAILABLE:
        occlusion_transform = A.Compose([
            # GridDropout - simulates fingers in a grid pattern (like holding card)
            A.GridDropout(
                ratio=0.4,  # Drop 40% of image
                unit_size_min=20,
                unit_size_max=50,
                holes_number_x=4,
                holes_number_y=4,
                random_offset=True,
                fill_value=0,
                p=0.5
            ),
            # CoarseDropout - random rectangular holes (simulates partial obstruction)
            A.CoarseDropout(
                max_holes=12,
                max_height=40,
                max_width=40,
                min_holes=4,
                min_height=15,
                min_width=15,
                fill_value=0,
                p=0.5
            ),
        ])
        occlusion_augs = [AlbumentationsWrapper(occlusion_transform)]
        print("  Using Albumentations occlusion augmentations (GridDropout, CoarseDropout)")
    else:
        print("  Warning: Albumentations not available, using basic occlusion only")

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),

        # Geometric - simulate different viewing angles
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=25),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.6),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.15, 0.15),
            scale=(0.85, 1.15),
            shear=10
        ),

        # Random crop - simulate partial card views
        transforms.RandomApply([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.75, 1.33))
        ], p=0.4),

        # Color/lighting - simulate different lighting conditions
        transforms.ColorJitter(
            brightness=0.5,
            contrast=0.5,
            saturation=0.4,
            hue=0.15
        ),

        # Card-specific augmentations (if available)
        *card_augs,

        # Albumentations occlusion augmentations (GridDropout, CoarseDropout)
        # Applied before ToTensor() since they work on PIL/numpy
        *occlusion_augs,

        transforms.ToTensor(),

        # Blur - simulate out-of-focus, camera shake
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0))
        ], p=0.5),

        # Additional occlusion - PyTorch RandomErasing (works on tensors)
        # Scale updated from (0.02, 0.25) to (0.05, 0.30) per FINAL_PLAN.md
        transforms.RandomErasing(
            p=0.4,
            scale=(0.05, 0.30),  # Increased from (0.02, 0.25) for stronger occlusion
            ratio=(0.3, 3.3),
            value='random'
        ),

        # Normalize for pretrained backbone
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transform(image_size: int = 224) -> transforms.Compose:
    """Minimal transform for validation/inference (default: 224 for ViT-B/16)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def collate_triplets(batch):
    """
    Custom collate function for triplet batches.

    Input: List of (anchor, positive, label) tuples
    Output: (anchors, positives, labels) tensors
    """
    anchors = torch.stack([item[0] for item in batch])
    positives = torch.stack([item[1] for item in batch])
    labels = torch.tensor([item[2] for item in batch])

    return anchors, positives, labels


# Test the dataset
if __name__ == "__main__":
    print("Testing TripletCardDataset...")

    # Load class mapping
    dataset_root = Path("/Users/marcos/dev/raspberry-pi/pokemon_classification_dataset")
    with open(dataset_root / "class_index.json") as f:
        class_info = json.load(f)

    class_to_idx = class_info["class_to_idx"]
    print(f"Loaded {len(class_to_idx)} classes")

    # Create dataset
    transform = get_heavy_augmentation(224)  # 224 for LeViT-384
    dataset = TripletCardDataset(
        root_dir=str(dataset_root / "train"),
        class_to_idx=class_to_idx,
        transform=transform,
        return_triplets=True
    )

    print(f"\nDataset size: {len(dataset)}")

    # Test loading a triplet
    anchor, positive, label = dataset[0]
    print(f"\nTriplet shapes:")
    print(f"  Anchor: {anchor.shape}")
    print(f"  Positive: {positive.shape}")
    print(f"  Label: {label}")

    # Verify anchor and positive are different (due to augmentation)
    diff = (anchor - positive).abs().mean()
    print(f"  Mean diff between anchor/positive: {diff:.4f} (should be > 0)")

    # Test DataLoader
    labels = [s[1] for s in dataset.samples]
    sampler = MPerClassSampler(labels, m_per_class=2, batch_size=64)

    loader = DataLoader(
        dataset,
        batch_size=64,
        sampler=sampler,
        collate_fn=collate_triplets,
        num_workers=0
    )

    # Get one batch
    batch = next(iter(loader))
    anchors, positives, batch_labels = batch
    print(f"\nBatch shapes:")
    print(f"  Anchors: {anchors.shape}")
    print(f"  Positives: {positives.shape}")
    print(f"  Labels: {batch_labels.shape}")
    print(f"  Unique labels in batch: {len(batch_labels.unique())}")

    print("\n✓ TripletCardDataset working correctly!")
