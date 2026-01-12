#!/usr/bin/env python3
"""Build uSearch index from reference card images for Raspberry Pi deployment.

Uses uSearch instead of FAISS per FINAL_PLAN.md:
- 2-5x faster ARM SIMD optimization
- 37% smaller memory footprint (40-bit vs 64-bit refs)
- Zero dependencies
- ~2-5ms for 17k search vs ~10ms for FAISS

Updated for LeViT-384 (768-dim embeddings, 224x224 input).
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

# uSearch for vector search
try:
    from usearch.index import Index
    USEARCH_AVAILABLE = True
except ImportError:
    USEARCH_AVAILABLE = False
    print("Warning: usearch not installed. Install with: pip install usearch")

# Add parent directory for embedding_model import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedding_model import LeViTEmbeddingModel


def load_model(weights_path: str, device: str = 'cpu') -> LeViTEmbeddingModel:
    """Load trained LeViT-384 embedding model."""
    model = LeViTEmbeddingModel(
        embedding_dim=768,  # LeViT-384 native dimension
        pretrained=False
    )
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def get_transform():
    """Get inference transform (224x224 for LeViT-384)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),  # LeViT-384 input size
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def compute_embeddings(
    model: LeViTEmbeddingModel,
    image_paths: list,
    device: str,
    batch_size: int = 32
) -> np.ndarray:
    """Compute embeddings for all images using LeViT-384."""
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
                batch_images.append(torch.zeros(3, 224, 224))  # LeViT-384 input size

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
    parser = argparse.ArgumentParser(description='Build uSearch index for Pokemon card recognition')
    parser.add_argument('--weights', type=str, default='trained_model/best_embedding.pth',
                        help='Path to trained model weights')
    parser.add_argument('--dataset', type=str, default='../pokemon_classification_dataset',
                        help='Path to classification dataset')
    parser.add_argument('--output', type=str, default='trained_model',
                        help='Output directory for index files')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for embedding computation')
    parser.add_argument('--use-all-splits', action='store_true', default=True,
                        help='Use images from train+val+test for reference index')
    args = parser.parse_args()

    print("=" * 60)
    print("uSearch Index Builder for Pokemon Card Recognition")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  Model: LeViT-384 (768-dim embeddings)")
    print(f"  Index: uSearch (optimized for ARM)")
    print(f"  Weights: {args.weights}")
    print(f"  Dataset: {args.dataset}")

    # Paths
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # Use MPS on Mac if available
    if torch.backends.mps.is_available():
        device = 'mps'
        print(f"\nUsing Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = 'cuda'
        print(f"\nUsing CUDA GPU")
    else:
        device = 'cpu'
        print(f"\nUsing CPU")

    # Load model
    print("\n1. Loading LeViT-384 embedding model...")
    model = load_model(args.weights, device)
    print(f"   Model loaded successfully")

    # Load class index
    print("\n2. Loading class index...")
    with open(dataset_path / "class_index.json") as f:
        class_index = json.load(f)

    class_to_idx = class_index['class_to_idx']
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    print(f"   Found {len(class_to_idx)} classes")

    # Collect ALL image paths (train + val + test for complete reference database)
    print("\n3. Collecting reference images...")

    splits = ['train', 'val', 'test'] if args.use_all_splits else ['train']
    print(f"   Using splits: {splits}")

    image_paths = []
    class_ids = []
    card_ids = []  # Original class names for metadata

    for split in splits:
        split_dir = dataset_path / split
        if not split_dir.exists():
            print(f"   Warning: {split} directory not found, skipping")
            continue

        for class_name, class_idx in tqdm(class_to_idx.items(), desc=f"Scanning {split}"):
            class_dir = split_dir / class_name
            if class_dir.exists():
                images = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.webp"))
                if images:
                    image_paths.append(str(images[0]))
                    class_ids.append(class_idx)
                    card_ids.append(class_name)

    print(f"   Found {len(image_paths)} reference images across {len(splits)} splits")

    # Compute embeddings
    print("\n4. Computing embeddings with LeViT-384...")
    embeddings = compute_embeddings(model, image_paths, device, batch_size=args.batch_size)
    print(f"   Embeddings shape: {embeddings.shape}")

    # Build uSearch index
    print("\n5. Building uSearch index...")
    if not USEARCH_AVAILABLE:
        print("   ERROR: usearch not installed!")
        print("   Install with: pip install usearch")
        return

    index = build_usearch_index(embeddings)

    # Save everything per FINAL_PLAN.md format
    print("\n6. Saving index and metadata...")

    # Save uSearch index
    index_path = output_dir / "usearch.index"
    index.save(str(index_path))
    index_size = index_path.stat().st_size / 1024 / 1024
    print(f"   uSearch index saved: {index_path} ({index_size:.1f} MB)")

    # Save embeddings as numpy array (for backup/debugging)
    embeddings_path = output_dir / "embeddings.npy"
    np.save(str(embeddings_path), embeddings)
    embeddings_size = embeddings_path.stat().st_size / 1024 / 1024
    print(f"   Embeddings saved: {embeddings_path} ({embeddings_size:.1f} MB)")

    # Save index mapping (key -> card_id)
    index_map = {str(i): card_ids[i] for i in range(len(card_ids))}
    index_map_path = output_dir / "index.json"
    with open(index_map_path, 'w') as f:
        json.dump(index_map, f)
    print(f"   Index mapping saved: {index_map_path}")

    # Save metadata
    metadata = {
        'num_cards': len(card_ids),
        'embedding_dim': 768,
        'model': 'LeViT-384',
        'image_size': 224,
        'index_type': 'usearch',
        'metric': 'cosine',
        'class_ids': class_ids,
        'idx_to_class': idx_to_class,
    }

    # Include card metadata if available
    try:
        with open(dataset_path / "card_metadata.json") as f:
            card_metadata = json.load(f)
        metadata['card_metadata'] = card_metadata
        print(f"   Included card metadata for {len(card_metadata)} cards")
    except Exception as e:
        print(f"   Warning: Could not load card metadata: {e}")

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    metadata_size = metadata_path.stat().st_size / 1024 / 1024
    print(f"   Metadata saved: {metadata_path} ({metadata_size:.1f} MB)")

    # Test the index
    print("\n7. Testing uSearch index...")
    test_embedding = embeddings[0:1].astype(np.float32)
    matches = index.search(test_embedding, 5)

    print(f"   Top 5 matches for first card ({card_ids[0]}):")
    for i, match in enumerate(matches):
        matched_card = card_ids[match.key]
        print(f"     {i+1}. {matched_card} (distance: {match.distance:.4f})")

    # Summary
    print("\n" + "=" * 60)
    print("USEARCH INDEX BUILD COMPLETE")
    print("=" * 60)
    print(f"\nOutput files (per FINAL_PLAN.md):")
    print(f"  {output_dir}/")
    print(f"  ├── usearch.index      ({index_size:.1f} MB) - uSearch index")
    print(f"  ├── embeddings.npy     ({embeddings_size:.1f} MB) - Raw embeddings")
    print(f"  ├── index.json         - Key to card_id mapping")
    print(f"  └── metadata.json      ({metadata_size:.1f} MB) - Card metadata")
    print(f"\n  Total cards indexed: {len(card_ids)}")
    print(f"  Embedding dimension: 768")

    print("\nFiles to copy to Raspberry Pi:")
    print(f"  1. card_embedding.onnx (LeViT-384 ONNX) OR levit384.hef (pre-compiled)")
    print(f"  2. usearch.index ({index_size:.1f} MB)")
    print(f"  3. embeddings.npy ({embeddings_size:.1f} MB)")
    print(f"  4. index.json + metadata.json")


if __name__ == "__main__":
    main()
