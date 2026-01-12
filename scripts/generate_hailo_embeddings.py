#!/usr/bin/env python3
"""
Generate reference embeddings using Hailo HEF model.

Downloads card images from S3 and generates embeddings using the
Hailo-8(L) NPU for consistent matching with runtime inference.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm
import tempfile
import subprocess

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from usearch.index import Index
    USEARCH_AVAILABLE = True
except ImportError:
    USEARCH_AVAILABLE = False
    print("Error: usearch required. Install with: pip install usearch")
    sys.exit(1)

try:
    from hailo_platform import HEF, VDevice, FormatType
    from hailo_platform import InputVStreamParams, OutputVStreamParams, InferVStreams
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    print("Error: Hailo platform required")
    sys.exit(1)


class HailoEmbeddingGenerator:
    """Generate embeddings using Hailo NPU"""

    def __init__(self, hef_path: str):
        print(f"Loading HEF model: {hef_path}")
        self.hef = HEF(hef_path)
        self.device = VDevice()
        self.network_group = self.device.configure(self.hef)[0]
        self.ng_params = self.network_group.create_params()

        # Get input/output vstream params using the correct API
        self.input_vstream_params = InputVStreamParams.make_from_network_group(
            self.network_group, quantized=True, format_type=FormatType.UINT8
        )
        self.output_vstream_params = OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32
        )

        # Get input name from params dict
        self.input_name = list(self.input_vstream_params.keys())[0]
        print(f"  Input: {self.input_name}")

        # Get output name
        self.output_name = list(self.output_vstream_params.keys())[0]
        print(f"  Output: {self.output_name}")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for Hailo inference (224x224 RGB uint8)"""
        # Resize to 224x224 (EfficientNet-Lite0 input)
        img = cv2.resize(image, (224, 224))

        # Convert BGR to RGB
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 255] uint8 (Hailo expects quantized input)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        return img

    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """Extract embedding from image using Hailo"""
        # Preprocess
        processed = self.preprocess(image)

        # Add batch dimension if needed (for NHWC format)
        if processed.ndim == 3:
            processed = np.expand_dims(processed, axis=0)

        # Run inference with network group activation
        with self.network_group.activate(self.ng_params):
            with InferVStreams(
                self.network_group,
                self.input_vstream_params,
                self.output_vstream_params
            ) as pipeline:
                input_data = {self.input_name: processed}
                output = pipeline.infer(input_data)
                embedding = list(output.values())[0]

        # Flatten and L2 normalize
        embedding = embedding.flatten()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)


def download_image_from_s3(s3_path: str, local_path: str) -> bool:
    """Download a single image from S3"""
    try:
        result = subprocess.run(
            ['aws', 's3', 'cp', s3_path, local_path, '--quiet'],
            capture_output=True, timeout=30
        )
        return result.returncode == 0
    except Exception:
        return False


def get_card_folders_from_s3(bucket: str, prefix: str) -> list:
    """List all card folders in S3"""
    result = subprocess.run(
        ['aws', 's3', 'ls', f's3://{bucket}/{prefix}'],
        capture_output=True, text=True
    )

    folders = []
    for line in result.stdout.strip().split('\n'):
        if 'PRE ' in line:
            folder = line.split('PRE ')[1].strip('/')
            folders.append(folder)
    return folders


def folder_to_card_id(folder_name: str) -> str:
    """Convert folder name to card ID

    Folder format: {name}-{lang}_{series}_{set}_{number}_{quality}
    Card ID format: {set}-{number}

    Example: Abra-en_base_base1_43_high -> base1-43
    """
    # Split by underscore
    parts = folder_name.split('_')

    # Find set and number (usually 3rd and 4th parts)
    # Format: name-lang_series_set_number_quality
    if len(parts) >= 4:
        set_name = parts[2]  # e.g., "base1"
        number = parts[3]    # e.g., "43"
        return f"{set_name}-{number}"

    return folder_name  # Fallback


def main():
    parser = argparse.ArgumentParser(description='Generate Hailo embeddings for Pokemon cards')
    parser.add_argument('--hef', type=str,
                       default='models/embedding/pokemon_student_efficientnet_lite0_stage2.hef',
                       help='Path to Hailo HEF model')
    parser.add_argument('--bucket', type=str,
                       default='pokemon-card-training-us-east-2',
                       help='S3 bucket name')
    parser.add_argument('--prefix', type=str,
                       default='classification_dataset/train/',
                       help='S3 prefix for card images')
    parser.add_argument('--output', type=str,
                       default='data/reference_hailo',
                       help='Output directory')
    parser.add_argument('--metadata', type=str,
                       default='data/reference/metadata.json',
                       help='Card metadata file')
    parser.add_argument('--limit', type=int, default=0,
                       help='Limit number of cards (0 = all)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')
    args = parser.parse_args()

    print("=" * 70)
    print("Hailo Embedding Generator - Pokemon Card Recognition")
    print("=" * 70)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load existing metadata
    if Path(args.metadata).exists():
        with open(args.metadata) as f:
            metadata = json.load(f)
        print(f"Loaded metadata from {args.metadata}")
    else:
        metadata = {}
        print("No existing metadata found")

    # Initialize Hailo
    print("\n[1/4] Initializing Hailo NPU...")
    generator = HailoEmbeddingGenerator(args.hef)

    # Get card folders from S3
    print("\n[2/4] Listing card folders from S3...")
    folders = get_card_folders_from_s3(args.bucket, args.prefix)
    print(f"  Found {len(folders)} card folders")

    if args.limit > 0:
        folders = folders[:args.limit]
        print(f"  Limited to {len(folders)} cards")

    # Check for resume checkpoint
    checkpoint_path = output_dir / "checkpoint.json"
    processed_cards = set()
    embeddings_list = []
    card_ids_list = []

    if args.resume and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        processed_cards = set(checkpoint.get('processed', []))

        # Load existing embeddings
        emb_path = output_dir / "embeddings_partial.npy"
        if emb_path.exists():
            embeddings_list = list(np.load(emb_path))
            card_ids_list = checkpoint.get('card_ids', [])

        print(f"  Resuming from checkpoint: {len(processed_cards)} cards already processed")

    # Process each card
    print("\n[3/4] Generating embeddings...")

    with tempfile.TemporaryDirectory() as tmpdir:
        for folder in tqdm(folders, desc="Processing cards"):
            card_id = folder_to_card_id(folder)

            # Skip if already processed
            if card_id in processed_cards:
                continue

            # Download image
            s3_path = f"s3://{args.bucket}/{args.prefix}{folder}/{folder}.png"
            local_path = os.path.join(tmpdir, f"{folder}.png")

            if not download_image_from_s3(s3_path, local_path):
                print(f"\n  Warning: Failed to download {folder}")
                continue

            # Load and process image
            try:
                image = cv2.imread(local_path)
                if image is None:
                    print(f"\n  Warning: Failed to load {folder}")
                    continue

                # Generate embedding
                embedding = generator.extract_embedding(image)

                embeddings_list.append(embedding)
                card_ids_list.append(card_id)
                processed_cards.add(card_id)

                # Save checkpoint every 100 cards
                if len(processed_cards) % 100 == 0:
                    np.save(output_dir / "embeddings_partial.npy", np.array(embeddings_list))
                    with open(checkpoint_path, 'w') as f:
                        json.dump({
                            'processed': list(processed_cards),
                            'card_ids': card_ids_list
                        }, f)

            except Exception as e:
                print(f"\n  Error processing {folder}: {e}")
                continue

            # Cleanup temp file
            try:
                os.remove(local_path)
            except:
                pass

    # Convert to numpy array
    if len(embeddings_list) == 0:
        print("\nERROR: No embeddings generated!")
        return

    embeddings = np.array(embeddings_list, dtype=np.float32)
    print(f"\n  Generated {len(embeddings)} embeddings")
    print(f"  Shape: {embeddings.shape}")

    # Build uSearch index
    print("\n[4/4] Building uSearch index...")

    index = Index(
        ndim=embeddings.shape[1],
        metric='cos',
        dtype='f32'
    )

    keys = np.arange(len(embeddings), dtype=np.uint64)
    index.add(keys, embeddings)

    # Save outputs
    print("\nSaving outputs...")

    # Save embeddings
    emb_path = output_dir / "embeddings.npy"
    np.save(emb_path, embeddings)
    print(f"  Embeddings: {emb_path} ({emb_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Save index
    index_path = output_dir / "usearch.index"
    index.save(str(index_path))
    print(f"  uSearch index: {index_path} ({index_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Save row -> card_id mapping
    index_map = {str(i): card_ids_list[i] for i in range(len(card_ids_list))}
    index_map_path = output_dir / "index.json"
    with open(index_map_path, 'w') as f:
        json.dump(index_map, f, indent=2)
    print(f"  Index mapping: {index_map_path}")

    # Copy metadata
    if metadata:
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Metadata: {metadata_path}")

    # Cleanup checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    partial_path = output_dir / "embeddings_partial.npy"
    if partial_path.exists():
        partial_path.unlink()

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"\nOutput: {output_dir}/")
    print(f"  Total cards: {len(embeddings)}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    print(f"\nTo use these embeddings, update demo_app.py:")
    print(f"  --reference {output_dir}")


if __name__ == "__main__":
    main()
