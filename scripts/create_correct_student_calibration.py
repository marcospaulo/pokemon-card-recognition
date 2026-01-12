#!/usr/bin/env python3
"""
Create CORRECT calibration data for EfficientNet-Lite0 student model

This fixes the preprocessing issue where the original calibration data
was missing ImageNet normalization (mean/std adjustment).

Preprocessing pipeline (MUST match training/inference):
1. Resize to 224x224
2. Normalize to [0, 1] by dividing by 255
3. Apply ImageNet normalization:
   mean = [0.485, 0.456, 0.406]
   std  = [0.229, 0.224, 0.225]
4. Keep HWC format (Hailo expects height, width, channels)
"""

import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm

def preprocess_image_for_hailo(image_path: Path, size=(224, 224)):
    """
    Preprocess image for EfficientNet-Lite0 with CORRECT ImageNet normalization

    Args:
        image_path: Path to image file
        size: Target size (224, 224) for EfficientNet-Lite0

    Returns:
        numpy array in HWC format with ImageNet normalization applied
    """
    # Load and resize
    img = Image.open(image_path).convert('RGB')
    img = img.resize(size, Image.BILINEAR)

    # Convert to numpy and normalize to [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Apply ImageNet normalization (CRITICAL for model accuracy)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    img_array = (img_array - mean) / std

    # Keep in HWC format (Hailo expects this)
    # Shape: (224, 224, 3)

    return img_array


def create_calibration_set(
    calibration_images_dir: Path,
    output_path: Path,
    num_images: int = 500,
    size: tuple = (224, 224)
):
    """
    Create calibration dataset with correct preprocessing

    Args:
        calibration_images_dir: Directory containing calibration images
        output_path: Where to save the .npy file
        num_images: Number of images to use (500-1000 recommended)
        size: Image size (224, 224) for EfficientNet-Lite0
    """

    print("=" * 70)
    print("Creating Correct Calibration Data for EfficientNet-Lite0")
    print("=" * 70)

    # Find calibration images
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp']
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(calibration_images_dir.glob(ext))

    image_paths = sorted(image_paths)[:num_images]

    print(f"\nConfiguration:")
    print(f"  Input directory: {calibration_images_dir}")
    print(f"  Found images: {len(image_paths)}")
    print(f"  Using images: {min(num_images, len(image_paths))}")
    print(f"  Target size: {size}")
    print(f"  Output: {output_path}")

    if len(image_paths) == 0:
        print(f"\n❌ Error: No images found in {calibration_images_dir}")
        return False

    if len(image_paths) < num_images:
        print(f"\n⚠️  Warning: Only {len(image_paths)} images available (requested {num_images})")

    # Process images
    print(f"\nProcessing images with CORRECT preprocessing...")
    print(f"  1. Resize to {size}")
    print(f"  2. Normalize to [0, 1]")
    print(f"  3. Apply ImageNet mean/std normalization")
    print(f"  4. Keep HWC format")

    calibration_data = []
    failed_images = []

    for img_path in tqdm(image_paths, desc="Processing"):
        try:
            img_array = preprocess_image_for_hailo(img_path, size=size)
            calibration_data.append(img_array)
        except Exception as e:
            failed_images.append((img_path.name, str(e)))

    if failed_images:
        print(f"\n⚠️  Failed to process {len(failed_images)} images:")
        for name, error in failed_images[:5]:
            print(f"     {name}: {error}")

    # Convert to numpy array
    calibration_array = np.array(calibration_data, dtype=np.float32)

    print(f"\n✅ Processed {len(calibration_data)} images successfully")
    print(f"\nCalibration array:")
    print(f"  Shape: {calibration_array.shape}")
    print(f"  Dtype: {calibration_array.dtype}")
    print(f"  Min:   {calibration_array.min():.4f}")
    print(f"  Max:   {calibration_array.max():.4f}")
    print(f"  Mean:  {calibration_array.mean():.4f}")
    print(f"  Std:   {calibration_array.std():.4f}")

    # Verify normalization is correct
    print(f"\nValidation:")
    if calibration_array.min() < -2.0 and calibration_array.max() > 2.0:
        print(f"  ✅ ImageNet normalization applied correctly")
        print(f"     (values in range [-2.5, 2.5] as expected)")
    else:
        print(f"  ❌ WARNING: Normalization may be incorrect!")
        print(f"     Expected range: [-2.5, 2.5]")
        print(f"     Actual range: [{calibration_array.min():.2f}, {calibration_array.max():.2f}]")

    # Save to .npy file
    print(f"\nSaving to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), calibration_array)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")

    print(f"\n" + "=" * 70)
    print("✅ Calibration data created successfully!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Upload to EC2:")
    print(f"     scp {output_path} ubuntu@18.118.102.134:~/hailo_workspace/")
    print(f"  2. Recompile HEF with correct calibration")
    print(f"  3. Test new HEF vs ONNX to verify accuracy")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Create correct calibration data for EfficientNet-Lite0 Hailo compilation'
    )
    parser.add_argument(
        '--images',
        type=str,
        default='data/raw/card_images',
        help='Directory containing Pokemon card images for calibration'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/hailo_calibration/student_pokemon_calib_correct.npy',
        help='Output path for calibration .npy file'
    )
    parser.add_argument(
        '--num-images',
        type=int,
        default=500,
        help='Number of images to use for calibration (500-1000 recommended)'
    )

    args = parser.parse_args()

    # Convert to Path objects
    images_dir = Path(args.images)
    output_path = Path(args.output)

    # Verify input directory exists
    if not images_dir.exists():
        print(f"❌ Error: Input directory not found: {images_dir}")
        print(f"\n💡 Tip: Download calibration images from S3:")
        print(f"   aws s3 sync s3://pokemon-card-training-us-east-2/project/pokemon-card-recognition/models/efficientnet-hailo/calibration/ {images_dir}")
        return 1

    # Create calibration set
    success = create_calibration_set(
        calibration_images_dir=images_dir,
        output_path=output_path,
        num_images=args.num_images,
        size=(224, 224)  # EfficientNet-Lite0 input size
    )

    return 0 if success else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
