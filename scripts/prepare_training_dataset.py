#!/usr/bin/env python3
"""
Prepare Pokemon card training dataset for YOLO distillation

Options:
1. Download from Ultralytics Hub (requires API key)
2. Generate synthetic data using trainCreator.py
3. Use existing card images with generated annotations
"""

import sys
import yaml
from pathlib import Path
import subprocess
import os

def create_dataset_yaml(
    data_root: Path,
    output_path: Path,
    train_images: str = "images/train",
    val_images: str = "images/val",
):
    """Create YOLO dataset YAML file"""

    dataset_config = {
        'path': str(data_root.absolute()),
        'train': train_images,
        'val': val_images,
        'names': {
            0: 'pokemon_card'
        },
    }

    with open(output_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)

    print(f"✅ Created dataset YAML: {output_path}")
    return output_path

def download_from_ultralytics_hub(dataset_id: str = "8awcqoIQP0jIXIMDOCsC"):
    """
    Download dataset from Ultralytics Hub

    Requires: ULTRALYTICS_API_KEY environment variable
    Dataset: https://hub.ultralytics.com/datasets/8awcqoIQP0jIXIMDOCsC
    """

    print(f"\n[Option 1] Downloading from Ultralytics Hub...")
    print(f"Dataset ID: {dataset_id}")

    api_key = os.environ.get('ULTRALYTICS_API_KEY')
    if not api_key:
        print(f"❌ ULTRALYTICS_API_KEY not set")
        print(f"   Get your API key from: https://hub.ultralytics.com/settings")
        print(f"   Set it: export ULTRALYTICS_API_KEY=your_key_here")
        return None

    try:
        # Ultralytics Hub download (requires ultralytics package)
        from ultralytics import hub

        print(f"   Downloading dataset (this may be large ~1-2GB)...")

        # This will download to ~/.ultralytics/datasets/
        # Unfortunately, ultralytics doesn't have a direct download API
        # User needs to manually download or use the hub.download() if available

        print(f"   Note: Ultralytics Hub datasets are typically accessed during training")
        print(f"   The model will auto-download when you use this dataset ID in training")
        print(f"\n   To manually download:")
        print(f"   1. Visit: https://hub.ultralytics.com/datasets/{dataset_id}")
        print(f"   2. Click 'Download' button")
        print(f"   3. Extract to: data/processed/detection/")

        return None

    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

def generate_synthetic_data(num_images: int = 1000, output_dir: Path = None):
    """
    Generate synthetic training data using trainCreator.py

    This creates random scenes with Pokemon cards
    """

    print(f"\n[Option 2] Generating synthetic training data...")
    print(f"Images to generate: {num_images}")

    if output_dir is None:
        output_dir = Path("data/processed/detection/synthetic")

    output_dir.mkdir(parents=True, exist_ok=True)

    train_creator = Path("/Users/marcos/dev/raspberry-pi/pokemon-card-recognition/references/Pokemon-TCGP-Card-Scanner/trainCreator.py")

    if not train_creator.exists():
        print(f"❌ trainCreator.py not found: {train_creator}")
        return None

    # Check requirements
    card_images_dir = Path("/Users/marcos/dev/raspberry-pi/pokemon-card-recognition/data/raw/card_images")
    if not card_images_dir.exists():
        print(f"❌ Card images not found: {card_images_dir}")
        print(f"   Need source Pokemon card images for synthetic generation")
        return None

    print(f"✅ Found {len(list(card_images_dir.glob('*.png')))} card images")
    print(f"   This script would generate synthetic training scenes")
    print(f"   Each scene: 1-50 cards with random transformations")
    print(f"\n   To run manually:")
    print(f"   python {train_creator} --num-images {num_images} --output {output_dir}")

    return output_dir

def use_existing_dataset():
    """Check if dataset already exists from previous training"""

    print(f"\n[Option 3] Checking for existing dataset...")

    possible_locations = [
        Path("data/processed/detection"),
        Path("pokemon_yolo_dataset"),
        Path.home() / ".ultralytics" / "datasets",
    ]

    for location in possible_locations:
        if location.exists():
            # Check if it has proper YOLO structure
            has_images = (location / "images").exists()
            has_labels = (location / "labels").exists()

            if has_images and has_labels:
                print(f"✅ Found existing dataset: {location}")
                print(f"   Images: {location / 'images'}")
                print(f"   Labels: {location / 'labels'}")
                return location

    print(f"   No existing dataset found")
    return None

def main():
    """Main dataset preparation"""

    print("="*70)
    print("Pokemon Card Training Dataset Preparation")
    print("="*70)

    # Check for existing dataset first
    existing = use_existing_dataset()

    if existing:
        print(f"\n✅ Using existing dataset: {existing}")

        # Create dataset YAML
        yaml_path = Path("pokemon_cards_obb.yaml")
        create_dataset_yaml(
            data_root=existing,
            output_path=yaml_path,
            train_images="images/train",
            val_images="images/val",
        )

        print(f"\n✅ Dataset ready!")
        print(f"   YAML: {yaml_path.absolute()}")
        print(f"\nNext step:")
        print(f"   python scripts/distill_yolo_for_imx500.py {yaml_path}")
        return 0

    # Otherwise, guide user on options
    print(f"\n📋 Dataset Options:")
    print(f"\n1. Download from Ultralytics Hub (10k images, used for original training)")
    print(f"   - Visit: https://hub.ultralytics.com/datasets/8awcqoIQP0jIXIMDOCsC")
    print(f"   - Download and extract to: data/processed/detection/")
    print(f"   - Or set ULTRALYTICS_API_KEY and model will auto-download during training")

    print(f"\n2. Generate synthetic data (quick, but may be lower quality)")
    print(f"   - Uses trainCreator.py to create random scenes")
    print(f"   - Requires card images in: data/raw/card_images/")

    print(f"\n3. Use existing trained model as-is (if you just want to test)")
    print(f"   - Skip distillation, try exporting smaller YOLO variant")

    print(f"\n" + "="*70)
    print(f"Recommendation for best results:")
    print(f"="*70)
    print(f"\nFor distillation to match teacher accuracy, use the same training data:")
    print(f"1. Download dataset from Ultralytics Hub")
    print(f"2. Extract to: data/processed/detection/")
    print(f"3. Run: python {__file__}")
    print(f"4. Then run: python scripts/distill_yolo_for_imx500.py pokemon_cards_obb.yaml")

    return 1

if __name__ == '__main__':
    sys.exit(main())
