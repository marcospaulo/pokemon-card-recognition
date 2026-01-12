#!/usr/bin/env python3
"""
Generate embeddings for all Pokemon cards using Hailo 8 NPU.

This script:
1. Downloads all card images from S3 (if not present)
2. Generates embeddings using Hailo NPU (15.2ms per card)
3. Builds uSearch index for fast similarity search
4. Creates metadata mappings

Designed to run in background with nohup - resilient to disconnection.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from datetime import datetime
import subprocess

# Setup logging
LOG_FILE = Path.home() / "pokemon-card-recognition" / "embedding_generation.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path.home() / "pokemon-card-recognition"
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from hailo_platform import (
        HEF,
        VDevice,
        InferVStreams,
        InputVStreamParams,
        OutputVStreamParams,
        FormatType,
    )
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    logger.error("Hailo platform not available! Cannot proceed.")
    sys.exit(1)

import cv2
from usearch.index import Index


class HailoEmbeddingGenerator:
    """Generate embeddings using Hailo 8 NPU"""

    def __init__(self, hef_path: str):
        """Initialize Hailo device and load HEF model

        Args:
            hef_path: Path to .hef model file
        """
        logger.info(f"Loading HEF model: {hef_path}")

        if not os.path.exists(hef_path):
            raise FileNotFoundError(f"HEF model not found: {hef_path}")

        # Load HEF
        self.hef = HEF(hef_path)

        # Configure device
        self.device = VDevice()
        self.network_group = self.device.configure(self.hef)[0]

        # Setup input/output streams
        self.input_vstream_params = InputVStreamParams.make_from_network_group(
            self.network_group,
            quantized=True,
            format_type=FormatType.UINT8
        )

        self.output_vstream_params = OutputVStreamParams.make_from_network_group(
            self.network_group,
            quantized=False,
            format_type=FormatType.FLOAT32
        )

        logger.info("✅ Hailo device initialized successfully")

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess card image for Hailo inference

        Args:
            image_path: Path to card image

        Returns:
            Preprocessed image ready for inference (224x224x3 UINT8)
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to 224x224
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 255] UINT8 (Hailo expects UINT8 input)
        img = img.astype(np.uint8)

        # Add batch dimension: [224, 224, 3] -> [1, 224, 224, 3]
        img = np.expand_dims(img, axis=0)

        return img

    def generate_embedding(self, image_path: str) -> np.ndarray:
        """Generate 768-dim embedding for a single card image

        Args:
            image_path: Path to card image

        Returns:
            768-dimensional embedding vector
        """
        # Preprocess
        input_data = self.preprocess_image(image_path)

        # Run inference on Hailo NPU
        with InferVStreams(self.network_group, self.input_vstream_params, self.output_vstream_params) as infer_pipeline:
            input_dict = {list(self.network_group.get_input_vstream_infos())[0].name: input_data}
            output_dict = infer_pipeline.infer(input_dict)

            # Extract embedding from output
            embedding = list(output_dict.values())[0][0]  # Remove batch dimension

        return embedding

    def __del__(self):
        """Cleanup Hailo resources"""
        try:
            if hasattr(self, 'network_group'):
                self.network_group = None
            if hasattr(self, 'device'):
                self.device = None
        except:
            pass


def download_images_from_s3(output_dir: Path) -> bool:
    """Download all card images from S3

    Args:
        output_dir: Directory to download images to

    Returns:
        True if successful, False otherwise
    """
    logger.info("📥 Starting S3 download...")
    logger.info(f"Destination: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # AWS S3 sync command
    s3_bucket = "s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/data/raw/card_images/"

    cmd = [
        "aws", "s3", "sync",
        s3_bucket,
        str(output_dir),
        "--no-progress"
    ]

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

        # Log output in real-time
        for line in process.stdout:
            logger.info(f"S3: {line.strip()}")

        process.wait()

        if process.returncode == 0:
            logger.info("✅ S3 download completed successfully")
            return True
        else:
            logger.error(f"❌ S3 download failed with code {process.returncode}")
            return False

    except Exception as e:
        logger.error(f"❌ S3 download error: {e}")
        return False


def extract_card_id_from_filename(filename: str) -> str:
    """Extract card ID from filename

    Examples:
        'base1-1_Alakazam.png' -> 'base1-1'
        'sv4-162_Technical_Machine_Evolution.png' -> 'sv4-162'

    Args:
        filename: Image filename

    Returns:
        Card ID (set-number)
    """
    # Remove extension
    name = filename.replace('.png', '').replace('.jpg', '')

    # Split on underscore and take first part
    parts = name.split('_')
    card_id = parts[0]

    return card_id


def load_existing_metadata(metadata_path: Path) -> Dict:
    """Load existing metadata if available

    Args:
        metadata_path: Path to metadata.json

    Returns:
        Dictionary of card metadata
    """
    if metadata_path.exists():
        logger.info(f"Loading existing metadata from {metadata_path}")
        with open(metadata_path, 'r') as f:
            return json.load(f)
    else:
        logger.warning(f"No existing metadata found at {metadata_path}")
        return {}


def generate_all_embeddings(
    image_dir: Path,
    model_path: Path,
    output_dir: Path,
    batch_log_interval: int = 100
) -> Tuple[np.ndarray, Dict[int, str], Dict[str, dict]]:
    """Generate embeddings for all card images using Hailo NPU

    Args:
        image_dir: Directory containing card images
        model_path: Path to Hailo HEF model
        output_dir: Directory to save outputs
        batch_log_interval: Log progress every N images

    Returns:
        Tuple of (embeddings array, index mapping, metadata dict)
    """
    logger.info("=" * 80)
    logger.info("🚀 Starting embedding generation with Hailo 8 NPU")
    logger.info("=" * 80)

    # Initialize Hailo
    generator = HailoEmbeddingGenerator(str(model_path))

    # Get all image files
    image_files = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
    total_images = len(image_files)

    logger.info(f"Found {total_images} images to process")

    if total_images == 0:
        logger.error("❌ No images found!")
        sys.exit(1)

    # Load existing metadata
    metadata = load_existing_metadata(output_dir / "metadata.json")

    # Storage
    embeddings_list = []
    index_mapping = {}  # row -> card_id

    # Process each image
    start_time = time.time()

    for idx, image_path in enumerate(image_files):
        try:
            # Generate embedding
            embedding = generator.generate_embedding(str(image_path))
            embeddings_list.append(embedding)

            # Extract card ID from filename
            card_id = extract_card_id_from_filename(image_path.name)
            index_mapping[idx] = card_id

            # Create basic metadata if not exists
            if card_id not in metadata:
                # Parse card info from filename
                set_code = card_id.split('-')[0]
                card_number = card_id.split('-')[1] if '-' in card_id else '?'
                card_name = image_path.stem.split('_', 1)[1] if '_' in image_path.stem else 'Unknown'

                metadata[card_id] = {
                    "id": card_id,
                    "name": card_name.replace('_', ' '),
                    "set": set_code,
                    "number": card_number,
                    "image_filename": image_path.name
                }

            # Log progress
            if (idx + 1) % batch_log_interval == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (idx + 1)
                remaining = avg_time * (total_images - idx - 1)

                logger.info(
                    f"Progress: {idx + 1}/{total_images} ({(idx + 1) / total_images * 100:.1f}%) | "
                    f"Avg: {avg_time * 1000:.1f}ms/card | "
                    f"ETA: {remaining / 60:.1f} min"
                )

        except Exception as e:
            logger.error(f"❌ Failed to process {image_path.name}: {e}")
            continue

    # Convert to numpy array
    embeddings_array = np.array(embeddings_list, dtype=np.float32)

    total_time = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"✅ Embedding generation complete!")
    logger.info(f"Total images: {len(embeddings_array)}")
    logger.info(f"Total time: {total_time / 60:.1f} minutes")
    logger.info(f"Average: {total_time / len(embeddings_array) * 1000:.1f}ms per card")
    logger.info("=" * 80)

    return embeddings_array, index_mapping, metadata


def build_usearch_index(embeddings: np.ndarray, output_path: Path):
    """Build uSearch HNSW index for fast similarity search

    Args:
        embeddings: Embedding vectors (N x 768)
        output_path: Path to save index file
    """
    logger.info("🔨 Building uSearch index...")

    # Create index
    index = Index(
        ndim=embeddings.shape[1],
        metric='cos',  # Cosine similarity
        dtype='f32'    # Float32
    )

    # Add all embeddings
    for i in range(len(embeddings)):
        index.add(i, embeddings[i])

    # Save index
    index.save(str(output_path))

    logger.info(f"✅ Index saved to {output_path}")
    logger.info(f"   Vectors: {len(embeddings)}")
    logger.info(f"   Dimensions: {embeddings.shape[1]}")


def save_outputs(
    embeddings: np.ndarray,
    index_mapping: Dict[int, str],
    metadata: Dict[str, dict],
    output_dir: Path
):
    """Save all outputs to disk

    Args:
        embeddings: Embedding array
        index_mapping: Row to card ID mapping
        metadata: Card metadata dictionary
        output_dir: Output directory
    """
    logger.info("💾 Saving outputs...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    embeddings_path = output_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    logger.info(f"✅ Saved embeddings: {embeddings_path} ({embeddings.nbytes / 1024 / 1024:.1f} MB)")

    # Save index mapping
    index_path = output_dir / "index.json"
    # Convert int keys to strings for JSON
    index_mapping_str = {str(k): v for k, v in index_mapping.items()}
    with open(index_path, 'w') as f:
        json.dump(index_mapping_str, f, indent=2)
    logger.info(f"✅ Saved index mapping: {index_path}")

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✅ Saved metadata: {metadata_path}")

    # Build and save uSearch index
    usearch_path = output_dir / "usearch.index"
    build_usearch_index(embeddings, usearch_path)
    logger.info(f"✅ Saved uSearch index: {usearch_path}")


def main():
    """Main execution"""

    logger.info("=" * 80)
    logger.info("Pokemon Card Embedding Generation (Hailo 8 NPU)")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    # Paths
    project_root = PROJECT_ROOT
    image_dir = project_root / "data" / "raw" / "card_images"
    model_path = project_root / "models" / "embedding" / "pokemon_student_efficientnet_lite0_stage2.hef"
    output_dir = project_root / "data" / "reference"

    logger.info(f"Project root: {project_root}")
    logger.info(f"Image directory: {image_dir}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Output directory: {output_dir}")

    # Check if Hailo is available
    if not HAILO_AVAILABLE:
        logger.error("❌ Hailo platform not available!")
        sys.exit(1)

    # Check if model exists
    if not model_path.exists():
        logger.error(f"❌ Model not found: {model_path}")
        sys.exit(1)

    # Step 1: Download images if not present
    if not image_dir.exists() or len(list(image_dir.glob("*.png"))) < 1000:
        logger.info("📥 Images not found or incomplete. Starting download from S3...")
        success = download_images_from_s3(image_dir)
        if not success:
            logger.error("❌ Failed to download images from S3")
            sys.exit(1)
    else:
        logger.info(f"✅ Images already present: {len(list(image_dir.glob('*.png')))} files")

    # Step 2: Generate embeddings
    try:
        embeddings, index_mapping, metadata = generate_all_embeddings(
            image_dir=image_dir,
            model_path=model_path,
            output_dir=output_dir,
            batch_log_interval=100
        )
    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

    # Step 3: Save all outputs
    try:
        save_outputs(embeddings, index_mapping, metadata, output_dir)
    except Exception as e:
        logger.error(f"❌ Failed to save outputs: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

    # Step 4: Cleanup - Delete raw images to free space
    logger.info("🗑️  Cleaning up raw images to free space...")
    try:
        import shutil
        shutil.rmtree(image_dir)
        logger.info(f"✅ Deleted {image_dir} (~12.6 GB freed)")
    except Exception as e:
        logger.warning(f"⚠️  Failed to delete images: {e}")

    logger.info("=" * 80)
    logger.info("🎉 ALL DONE!")
    logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
