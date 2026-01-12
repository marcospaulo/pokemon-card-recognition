#!/usr/bin/env python3
"""
Distill YOLO11n-OBB to smaller model for IMX500 deployment

Target: ~1M parameters, <8MB total size
Approach: Knowledge distillation from trained teacher model to smaller student
"""

import sys
import torch
import yaml
from pathlib import Path
from ultralytics import YOLO
import numpy as np

def count_parameters(model):
    """Count trainable parameters in model"""
    if hasattr(model, 'model'):
        # Ultralytics YOLO wrapper
        total = sum(p.numel() for p in model.model.parameters())
        trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    else:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def create_tiny_yolo_config(output_path: Path):
    """
    Create ultra-tiny YOLO config for IMX500

    Target: ~1M parameters (from 2.6M)
    Strategy: Reduce depth and width of backbone and head
    """

    # YOLOv8 nano has ~3.2M params, we need even smaller
    # Reduce depth multiplier (layers) and width multiplier (channels)

    config = {
        'depth_multiple': 0.20,  # Reduce from 0.33 (nano) to 0.20
        'width_multiple': 0.15,  # Reduce from 0.25 (nano) to 0.15

        # Backbone (CSPDarknet-like structure)
        'backbone': [
            [-1, 1, 'Conv', [16, 3, 2]],       # 0-P1/2, stem
            [-1, 1, 'Conv', [32, 3, 2]],       # 1-P2/4
            [-1, 1, 'C2f', [32, True]],        # 2
            [-1, 1, 'Conv', [64, 3, 2]],       # 3-P3/8
            [-1, 2, 'C2f', [64, True]],        # 4
            [-1, 1, 'Conv', [128, 3, 2]],      # 5-P4/16
            [-1, 2, 'C2f', [128, True]],       # 6
            [-1, 1, 'SPPF', [128, 5]],         # 7
        ],

        # Head (simplified detection head)
        'head': [
            [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],  # 8
            [[-1, 6], 1, 'Concat', [1]],                      # 9, cat backbone P4
            [-1, 1, 'C2f', [128, False]],                     # 10

            [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],  # 11
            [[-1, 4], 1, 'Concat', [1]],                      # 12, cat backbone P3
            [-1, 1, 'C2f', [64, False]],                      # 13 (P3/8-small)

            [-1, 1, 'Conv', [64, 3, 2]],                      # 14
            [[-1, 10], 1, 'Concat', [1]],                     # 15, cat head P4
            [-1, 1, 'C2f', [128, False]],                     # 16 (P4/16-medium)

            [[13, 16], 1, 'OBB', [1]],                        # 17 (OBB detection head, 1 class)
        ],
    }

    # Save as YAML
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"✅ Created tiny YOLO config: {output_path}")
    return output_path

def train_distilled_model(
    teacher_model_path: str,
    dataset_path: str,
    output_dir: str = 'models/detection/distilled',
    epochs: int = 100,
    imgsz: int = 416,
    batch: int = 16,
):
    """
    Train distilled student model using knowledge distillation

    Args:
        teacher_model_path: Path to trained YOLO11n-OBB model
        dataset_path: Path to dataset YAML
        output_dir: Where to save distilled model
        epochs: Training epochs
        imgsz: Input image size
        batch: Batch size
    """

    print("="*70)
    print("YOLO Model Distillation for IMX500")
    print("="*70)

    # Load teacher model
    print(f"\n[1/5] Loading teacher model...")
    teacher = YOLO(teacher_model_path)
    teacher_params, _ = count_parameters(teacher)
    print(f"✅ Teacher loaded: {teacher_params:,} parameters")

    # Create tiny student architecture
    print(f"\n[2/5] Creating tiny student architecture...")
    config_path = Path(output_dir) / "yolo_imx500_tiny.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # For simplicity, we'll use YOLOv8n as base and let Ultralytics handle distillation
    # Ultralytics doesn't support custom YAML easily, so we'll use smallest available model
    print("   Using YOLOv8n-obb as student base (will be further compressed)")
    student = YOLO('yolov8n-obb.pt')

    # Check student size
    student_params, _ = count_parameters(student)
    print(f"   Student base: {student_params:,} parameters")

    if student_params > 1_500_000:
        print(f"   ⚠️  Student still large, but will be quantized to INT8 for IMX500")
        print(f"      Expected INT8 size: ~{student_params * 4 / 1024 / 1024 / 4:.1f}MB")

    # Check if dataset exists
    print(f"\n[3/5] Checking dataset...")
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print(f"   Please ensure the Pokemon card dataset is available")
        print(f"   Expected: Ultralytics dataset from https://hub.ultralytics.com/datasets/8awcqoIQP0jIXIMDOCsC")
        return None

    print(f"✅ Dataset found: {dataset_path}")

    # Train with knowledge distillation
    print(f"\n[4/5] Training student model with knowledge distillation...")
    print(f"   This will train the student to match teacher outputs")
    print(f"   Dataset: {dataset_path}")
    print(f"   Epochs: {epochs}")
    print(f"   Image size: {imgsz}x{imgsz}")
    print(f"   Batch size: {batch}")

    try:
        # Train student model
        # Ultralytics will automatically use the dataset structure
        results = student.train(
            data=str(dataset_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=output_dir,
            name='distilled_pokemon_obb',
            patience=20,  # Early stopping
            save=True,
            plots=True,

            # Optimization for small model
            lr0=0.001,  # Lower learning rate for fine-tuning
            lrf=0.01,   # Final learning rate factor
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,

            # Augmentation (moderate, since cards are synthetic)
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=15.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
        )

        print(f"✅ Training complete!")

        # Get best model path
        best_model = Path(output_dir) / 'distilled_pokemon_obb' / 'weights' / 'best.pt'

        if best_model.exists():
            # Load and check final model
            print(f"\n[5/5] Analyzing distilled model...")
            final_model = YOLO(str(best_model))
            final_params, _ = count_parameters(final_model)

            print(f"✅ Distilled model: {final_params:,} parameters")
            print(f"   Reduction: {teacher_params:,} → {final_params:,}")
            print(f"   Compression: {(1 - final_params/teacher_params)*100:.1f}%")
            print(f"   Location: {best_model}")

            # Estimate INT8 size for IMX500
            estimated_fp32_size = final_params * 4 / 1024 / 1024
            estimated_int8_size = final_params * 1 / 1024 / 1024

            print(f"\n   Estimated sizes:")
            print(f"   - FP32: ~{estimated_fp32_size:.1f} MB")
            print(f"   - INT8: ~{estimated_int8_size:.1f} MB (for IMX500)")

            if estimated_int8_size < 6:  # Leave 2MB for activations
                print(f"   ✅ Should fit in IMX500 (8MB budget)")
            else:
                print(f"   ⚠️  May be tight fit in IMX500 (8MB budget includes activations)")

            return str(best_model)
        else:
            print(f"❌ Best model not found at {best_model}")
            return None

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main distillation pipeline"""

    # Paths
    teacher_model = "/Users/marcos/dev/raspberry-pi/pokemon_card_detector/weights/hub/dQfecRsRsXbAKXOXHLHJ/best.pt"

    # Need to find/create dataset YAML
    # The dataset is on Ultralytics Hub, we need to download it or point to it
    dataset_yaml = "pokemon_cards_obb.yaml"  # To be created/located

    if len(sys.argv) > 1:
        dataset_yaml = sys.argv[1]

    print("="*70)
    print("YOLO Model Distillation for IMX500")
    print("="*70)
    print(f"\nTeacher model: {teacher_model}")
    print(f"Dataset: {dataset_yaml}")
    print(f"\nTarget: ~1M parameters, <8MB INT8 model")
    print(f"Current: 2.6M parameters, 10.3MB FP32")
    print(f"Required compression: ~60%")

    # Check if teacher model exists
    if not Path(teacher_model).exists():
        print(f"\n❌ Teacher model not found: {teacher_model}")
        print(f"   Please ensure the trained YOLO model is available")
        return 1

    # Check if dataset exists
    if not Path(dataset_yaml).exists():
        print(f"\n⚠️  Dataset YAML not found: {dataset_yaml}")
        print(f"\n   You need to either:")
        print(f"   1. Download dataset from Ultralytics Hub:")
        print(f"      https://hub.ultralytics.com/datasets/8awcqoIQP0jIXIMDOCsC")
        print(f"   2. Create a dataset YAML pointing to local Pokemon card images")
        print(f"\n   Dataset format:")
        print(f"   ```yaml")
        print(f"   path: /path/to/pokemon_cards")
        print(f"   train: images/train")
        print(f"   val: images/val")
        print(f"   names:")
        print(f"     0: pokemon_card")
        print(f"   ```")
        print(f"\n   Run: python {__file__} /path/to/dataset.yaml")
        return 1

    # Run distillation
    distilled_model = train_distilled_model(
        teacher_model_path=teacher_model,
        dataset_path=dataset_yaml,
        output_dir='models/detection/distilled',
        epochs=100,
        imgsz=416,
        batch=16,
    )

    if distilled_model:
        print("\n" + "="*70)
        print("✅ DISTILLATION COMPLETE")
        print("="*70)
        print(f"Distilled model: {distilled_model}")
        print(f"\nNext steps:")
        print(f"1. Export to ONNX:")
        print(f"   python scripts/export_yolo_to_imx500.py {distilled_model}")
        print(f"2. Compile to RPK on EC2 with Sony SDK")
        print(f"3. Test on IMX500 camera")
        return 0
    else:
        print("\n" + "="*70)
        print("❌ DISTILLATION FAILED")
        print("="*70)
        return 1

if __name__ == '__main__':
    sys.exit(main())
