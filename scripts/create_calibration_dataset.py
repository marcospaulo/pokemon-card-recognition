#!/usr/bin/env python3
"""
Create calibration dataset from random Pokemon card images.

Samples 1024 random images from the full card image dataset
for use in Hailo quantization calibration.
"""
import os
import shutil
import random
from pathlib import Path

def main():
    # Paths
    project_root = Path("/Users/marcos/dev/raspberry-pi/pokemon-card-recognition")
    card_images_dir = project_root / "data" / "raw" / "card_images"
    calib_dir = project_root / "data" / "calibration"

    print("=" * 70)
    print("CREATE CALIBRATION DATASET")
    print("=" * 70)
    print(f"Source: {card_images_dir}")
    print(f"Destination: {calib_dir}")
    print()

    # Check source directory
    if not card_images_dir.exists():
        print(f"❌ ERROR: Card images directory not found: {card_images_dir}")
        return 1

    # Get all PNG images
    image_files = list(card_images_dir.glob("*.png"))
    total_images = len(image_files)

    print(f"Found {total_images:,} Pokemon card images")

    if total_images == 0:
        print("❌ ERROR: No PNG images found")
        return 1

    # Determine calibration count
    calib_count = min(1024, total_images)
    print(f"Sampling {calib_count} random images for Hailo calibration...")
    print()

    # Create calibration directory
    calib_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing calibration images
    for f in calib_dir.glob("*.png"):
        f.unlink()

    # Sample random images
    sampled_images = random.sample(image_files, calib_count)

    # Copy images
    for i, img_path in enumerate(sampled_images, 1):
        dest_path = calib_dir / img_path.name
        shutil.copy2(img_path, dest_path)

        if i % 100 == 0:
            print(f"  Copied {i}/{calib_count} images...")

    # Verify
    final_count = len(list(calib_dir.glob("*.png")))

    print()
    print("=" * 70)
    print("✅ CALIBRATION DATASET CREATED")
    print("=" * 70)
    print(f"Images: {final_count:,} real Pokemon card images")
    print(f"Location: {calib_dir}")
    print(f"Total size: {sum(f.stat().st_size for f in calib_dir.glob('*.png')) / 1024**2:.1f} MB")
    print()
    print("These images will be used for Hailo quantization calibration")
    print("to ensure optimal performance on real Pokemon cards.")
    print()

    return 0

if __name__ == '__main__':
    exit(main())
