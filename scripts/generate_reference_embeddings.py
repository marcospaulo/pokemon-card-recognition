#!/usr/bin/env python3
"""
Generate reference embeddings for all 17,592 Pokemon cards using trained EfficientNet-Lite0 student model.

This script:
1. Loads the trained student model weights (student_stage2_final.pt)
2. Processes all card images from data/raw/card_images/
3. Generates 768-dim L2-normalized embeddings
4. Builds uSearch index for fast similarity search
5. Saves everything to data/reference/

Output files:
- embeddings.npy: [17592, 768] numpy array
- usearch.index: Vector search index
- index.json: Row → card_id mapping
- metadata.json: Card metadata
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm

# uSearch for vector search
try:
    from usearch.index import Index
    USEARCH_AVAILABLE = True
except ImportError:
    USEARCH_AVAILABLE = False
    print("Warning: usearch not installed. Install with: pip install usearch")

# timm for EfficientNet-Lite0
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("ERROR: timm is required. Install with: pip install timm")
    sys.exit(1)


class StudentModel(nn.Module):
    """EfficientNet-Lite0 student model for Pokemon card recognition"""

    def __init__(self, model_name='efficientnet_lite0', embedding_dim=768, num_classes=17592):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        backbone_dim = self.backbone.num_features  # 1280 for efficientnet_lite0

        # Projection head (matches training architecture exactly)
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(backbone_dim, embedding_dim),
        )

        # Classification head (not used in inference, but needed for loading checkpoint)
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)

    def forward(self, x):
        """Forward pass - returns normalized embeddings"""
        features = self.backbone(x)
        embeddings = self.projection(features)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings


def load_model(weights_path: str, device: str = 'cpu') -> StudentModel:
    """Load trained EfficientNet-Lite0 student model."""
    print(f"\nLoading model from: {weights_path}")

    model = StudentModel(
        model_name='efficientnet_lite0',
        embedding_dim=768,
        num_classes=17592
    )

    # Load state dict with weights_only=True for security
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)

    # Remove prefixes from state dict keys
    # '_orig_mod.' is from torch.compile()
    # 'module.' is from DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove '_orig_mod.' prefix if present
        if k.startswith('_orig_mod.'):
            k = k.replace('_orig_mod.', '')
        # Remove 'module.' prefix if present
        if k.startswith('module.'):
            k = k.replace('module.', '')
        new_state_dict[k] = v

    state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    print(f"✓ Model loaded successfully")
    print(f"  Architecture: EfficientNet-Lite0")
    print(f"  Embedding dimension: 768")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


def get_transform():
    """Get inference transform (224x224 for EfficientNet-Lite0)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def collect_card_images(images_dir: Path) -> tuple[list, list]:
    """
    Collect all card images from raw data directory.

    Returns:
        image_paths: List of image file paths
        card_ids: List of card IDs (filenames without extension)
    """
    image_paths = []
    card_ids = []

    # Supported image formats
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.webp']:
        for img_path in sorted(images_dir.glob(ext)):
            image_paths.append(str(img_path))
            card_ids.append(img_path.stem)  # Filename without extension

    return image_paths, card_ids


def compute_embeddings(
    model: StudentModel,
    image_paths: list,
    device: str,
    batch_size: int = 32
) -> np.ndarray:
    """Compute embeddings for all images using EfficientNet-Lite0."""
    transform = get_transform()
    embeddings = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Computing embeddings"):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = transform(img)
                batch_images.append(img_tensor)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                batch_images.append(torch.zeros(3, 224, 224))

        batch_tensor = torch.stack(batch_images).to(device)

        with torch.no_grad():
            batch_embeddings = model(batch_tensor).cpu().numpy()

        embeddings.append(batch_embeddings)

    return np.vstack(embeddings)


def build_usearch_index(embeddings: np.ndarray, metric: str = 'cos') -> Index:
    """
    Build uSearch index for fast similarity search.

    Args:
        embeddings: [N, D] L2-normalized embeddings
        metric: 'cos' for cosine similarity, 'l2sq' for L2 distance

    Returns:
        uSearch Index
    """
    if not USEARCH_AVAILABLE:
        raise ImportError("usearch is required. Install with: pip install usearch")

    dimension = embeddings.shape[1]
    n_vectors = embeddings.shape[0]

    # Create uSearch index with cosine similarity
    # For L2-normalized embeddings, cosine similarity = dot product
    index = Index(
        ndim=dimension,
        metric=metric,
        dtype='f32'  # float32
    )

    # Add vectors with sequential keys
    keys = np.arange(n_vectors, dtype=np.uint64)
    index.add(keys, embeddings.astype(np.float32))

    print(f"  Built uSearch index: {n_vectors} vectors, {dimension} dimensions")
    print(f"  Metric: {metric}")

    return index


def main():
    parser = argparse.ArgumentParser(description='Generate reference embeddings for Pokemon card recognition')
    parser.add_argument('--weights', type=str,
                        default='models/embedding/pytorch_weights/student_stage2_final.pt',
                        help='Path to trained model weights')
    parser.add_argument('--images', type=str, default='data/raw/card_images',
                        help='Path to card images directory')
    parser.add_argument('--output', type=str, default='data/reference',
                        help='Output directory for embeddings and index')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for embedding computation')
    args = parser.parse_args()

    print("=" * 70)
    print("Reference Embeddings Generator - Pokemon Card Recognition")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Model: EfficientNet-Lite0 (768-dim embeddings)")
    print(f"  Index: uSearch (optimized for ARM)")
    print(f"  Weights: {args.weights}")
    print(f"  Images: {args.images}")
    print(f"  Output: {args.output}")

    # Paths
    weights_path = Path(args.weights)
    images_dir = Path(args.images)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Verify weights exist
    if not weights_path.exists():
        print(f"\nERROR: Model weights not found at {weights_path}")
        print("Please ensure student_stage2_final.pt is extracted from model.tar.gz")
        return

    # Verify images directory
    if not images_dir.exists():
        print(f"\nERROR: Images directory not found at {images_dir}")
        return

    # Use MPS on Mac if available
    if torch.backends.mps.is_available():
        device = 'mps'
        print(f"\n✓ Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = 'cuda'
        print(f"\n✓ Using CUDA GPU")
    else:
        device = 'cpu'
        print(f"\n✓ Using CPU")

    # Load model
    print("\n[1/5] Loading EfficientNet-Lite0 student model...")
    model = load_model(str(weights_path), device)

    # Collect card images
    print("\n[2/5] Collecting card images...")
    image_paths, card_ids = collect_card_images(images_dir)
    print(f"  Found {len(image_paths)} card images")

    if len(image_paths) == 0:
        print(f"  ERROR: No images found in {images_dir}")
        return

    # Compute embeddings
    print(f"\n[3/5] Computing embeddings (batch_size={args.batch_size})...")
    embeddings = compute_embeddings(model, image_paths, device, batch_size=args.batch_size)
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Memory size: {embeddings.nbytes / 1024 / 1024:.1f} MB")

    # Build uSearch index
    print("\n[4/5] Building uSearch index...")
    if not USEARCH_AVAILABLE:
        print("  ERROR: usearch not installed!")
        print("  Install with: pip install usearch")
        return

    index = build_usearch_index(embeddings)

    # Save everything
    print("\n[5/5] Saving embeddings and index...")

    # Save uSearch index
    index_path = output_dir / "usearch.index"
    index.save(str(index_path))
    index_size = index_path.stat().st_size / 1024 / 1024
    print(f"  ✓ uSearch index saved: {index_path} ({index_size:.1f} MB)")

    # Save embeddings as numpy array (for backup/debugging)
    embeddings_path = output_dir / "embeddings.npy"
    np.save(str(embeddings_path), embeddings)
    embeddings_size = embeddings_path.stat().st_size / 1024 / 1024
    print(f"  ✓ Embeddings saved: {embeddings_path} ({embeddings_size:.1f} MB)")

    # Save index mapping (row → card_id)
    index_map = {str(i): card_ids[i] for i in range(len(card_ids))}
    index_map_path = output_dir / "index.json"
    with open(index_map_path, 'w') as f:
        json.dump(index_map, f, indent=2)
    print(f"  ✓ Index mapping saved: {index_map_path}")

    # Save metadata
    metadata = {
        'num_cards': len(card_ids),
        'embedding_dim': 768,
        'model': 'EfficientNet-Lite0 (Stage 2)',
        'image_size': 224,
        'index_type': 'usearch',
        'metric': 'cosine',
        'card_ids': card_ids,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    metadata_size = metadata_path.stat().st_size / 1024 / 1024
    print(f"  ✓ Metadata saved: {metadata_path} ({metadata_size:.1f} MB)")

    # Test the index
    print("\n[6/6] Testing uSearch index...")
    test_embedding = embeddings[0:1].astype(np.float32)
    matches = index.search(test_embedding, 5)

    print(f"  Top 5 matches for first card ({card_ids[0]}):")
    for i, match in enumerate(matches):
        matched_card = card_ids[match.key]
        print(f"    {i+1}. {matched_card} (distance: {match.distance:.4f})")

    # Summary
    print("\n" + "=" * 70)
    print("✅ REFERENCE DATABASE BUILD COMPLETE")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  {output_dir}/")
    print(f"  ├── usearch.index      ({index_size:.1f} MB)")
    print(f"  ├── embeddings.npy     ({embeddings_size:.1f} MB)")
    print(f"  ├── index.json         (card ID mapping)")
    print(f"  └── metadata.json      ({metadata_size:.1f} MB)")
    print(f"\n  Total cards indexed: {len(card_ids)}")
    print(f"  Embedding dimension: 768")
    print(f"  Total size: {(index_size + embeddings_size + metadata_size):.1f} MB")

    print("\nReady for deployment to Raspberry Pi!")
    print(f"  1. Copy pokemon_student_efficientnet_lite0_stage2.hef")
    print(f"  2. Copy {output_dir}/ directory")
    print(f"  3. Update inference pipeline to use these files")


if __name__ == "__main__":
    main()
