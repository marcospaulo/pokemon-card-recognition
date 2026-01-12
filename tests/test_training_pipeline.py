#!/usr/bin/env python3
"""
Integration tests for DINOv3 training pipeline.

Run these BEFORE launching expensive SageMaker jobs to catch issues early.

Usage:
    python tests/test_training_pipeline.py
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets
from PIL import Image
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'training'))

from dinov3_embedding import DINOv3TeacherModel, ArcFaceLoss
from train_dinov3_teacher import (
    get_transforms,
    train_epoch,
    evaluate,
    ModelEMA,
    WarmupCosineScheduler,
)


def create_fake_dataset(num_classes=10, samples_per_class=5, image_size=224):
    """Create a fake dataset for testing."""
    tmp_dir = tempfile.mkdtemp()
    dataset_dir = Path(tmp_dir) / 'train'
    dataset_dir.mkdir()

    for class_id in range(num_classes):
        class_dir = dataset_dir / f'class_{class_id:04d}'
        class_dir.mkdir()

        for sample_id in range(samples_per_class):
            # Create random RGB image
            img = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
            img_pil = Image.fromarray(img)
            img_path = class_dir / f'sample_{sample_id:03d}.png'
            img_pil.save(img_path)

    return tmp_dir, dataset_dir


def test_model_instantiation():
    """Test that DINOv3 models can be instantiated without HuggingFace token issues."""
    print("\n[TEST] Model Instantiation")

    # Test ViT-Large (current model)
    try:
        model = DINOv3TeacherModel(
            model_name='dinov3_vitl16',
            embedding_dim=768,
            freeze_backbone=True,
        )
        print(f"  ✅ DINOv3-ViT-Large instantiated successfully")
        print(f"     Backbone dim: {model.backbone.config.hidden_size}")
        print(f"     Embedding dim: {model.projection[-1].out_features}")

        # Check dimensions
        assert model.backbone.config.hidden_size == 1024, "ViT-L should have 1024 hidden dim"
        assert model.projection[-1].out_features == 768, "Projection should output 768"

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False

    return True


def test_forward_pass():
    """Test forward pass with dummy input."""
    print("\n[TEST] Forward Pass")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DINOv3TeacherModel(
        model_name='dinov3_vitl16',
        embedding_dim=768,
        freeze_backbone=True,
    ).to(device)

    # Create dummy batch
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)

    try:
        with torch.no_grad():
            embeddings = model(dummy_input)

        assert embeddings.shape == (batch_size, 768), f"Expected (4, 768), got {embeddings.shape}"
        assert torch.allclose(torch.norm(embeddings, p=2, dim=1), torch.ones(batch_size).to(device), atol=1e-5), \
            "Embeddings should be L2-normalized"

        print(f"  ✅ Forward pass successful")
        print(f"     Input shape: {dummy_input.shape}")
        print(f"     Output shape: {embeddings.shape}")
        print(f"     L2 norm: {torch.norm(embeddings, p=2, dim=1)}")

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False
    finally:
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return True


def test_ema_with_compile():
    """Test that EMA works with torch.compile() wrapped models."""
    print("\n[TEST] EMA with torch.compile()")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DINOv3TeacherModel(
        model_name='dinov3_vitl16',
        embedding_dim=768,
        freeze_backbone=True,
    ).to(device)

    # Compile model (this wraps it in OptimizedModule)
    try:
        compiled_model = torch.compile(model, mode='reduce-overhead')
        print(f"  ✅ Model compiled successfully")
    except Exception as e:
        print(f"  ⚠️  torch.compile() not available (PyTorch < 2.0): {e}")
        compiled_model = model

    # Test EMA instantiation
    try:
        ema_model = ModelEMA(compiled_model, decay=0.999)
        print(f"  ✅ EMA created from compiled model")

        # Test EMA update
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        with torch.no_grad():
            _ = compiled_model(dummy_input)

        ema_model.update(compiled_model)
        print(f"  ✅ EMA update successful")

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False
    finally:
        del model, compiled_model, ema_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return True


def test_backbone_unfreezing():
    """Test that backbone unfreezing works correctly."""
    print("\n[TEST] Backbone Unfreezing")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DINOv3TeacherModel(
        model_name='dinov3_vitl16',
        embedding_dim=768,
        freeze_backbone=True,
    ).to(device)

    try:
        # Check that backbone is frozen initially
        frozen_count = sum(1 for p in model.backbone.parameters() if not p.requires_grad)
        total_count = sum(1 for p in model.backbone.parameters())
        print(f"  ✅ Backbone frozen: {frozen_count}/{total_count} parameters")

        # Unfreeze last 4 blocks
        model.unfreeze_backbone(last_n_blocks=4)

        # Check that some parameters are now unfrozen
        unfrozen_count = sum(1 for p in model.backbone.parameters() if p.requires_grad)
        print(f"  ✅ Unfroze last 4 blocks: {unfrozen_count}/{total_count} parameters trainable")

        assert unfrozen_count > 0, "No parameters were unfrozen"
        assert unfrozen_count < total_count, "All parameters were unfrozen (expected partial)"

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return True


def test_dataloader_with_transforms():
    """Test that DataLoader works with custom transforms."""
    print("\n[TEST] DataLoader with Transforms")

    tmp_dir, dataset_dir = create_fake_dataset(num_classes=5, samples_per_class=3)

    try:
        train_transform, val_transform = get_transforms(224)

        # Test train dataset
        train_dataset = datasets.ImageFolder(str(dataset_dir), transform=train_transform)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,  # Use 0 workers to avoid multiprocessing issues in tests
        )

        # Load one batch
        images, labels = next(iter(train_loader))
        assert images.shape[1:] == (3, 224, 224), f"Expected (3, 224, 224), got {images.shape[1:]}"
        print(f"  ✅ Train DataLoader works")
        print(f"     Batch shape: {images.shape}")
        print(f"     Labels: {labels}")

        # Test val dataset
        val_dataset = datasets.ImageFolder(str(dataset_dir), transform=val_transform)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )

        images, labels = next(iter(val_loader))
        print(f"  ✅ Val DataLoader works")

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False
    finally:
        shutil.rmtree(tmp_dir)

    return True


def test_training_step():
    """Test a single training step end-to-end."""
    print("\n[TEST] Single Training Step")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tmp_dir, dataset_dir = create_fake_dataset(num_classes=5, samples_per_class=3)

    try:
        # Create model and loss
        model = DINOv3TeacherModel(
            model_name='dinov3_vitl16',
            embedding_dim=768,
            freeze_backbone=True,
        ).to(device)

        num_classes = 5
        loss_fn = ArcFaceLoss(
            embedding_dim=768,
            num_classes=num_classes,
            margin=0.5,
            scale=64.0,
        ).to(device)

        # Create optimizer
        optimizer = torch.optim.AdamW(
            list(model.projection.parameters()) + list(loss_fn.parameters()),
            lr=1e-3,
            weight_decay=0.05,
        )

        # Create dataloader
        train_transform, _ = get_transforms(224)
        train_dataset = datasets.ImageFolder(str(dataset_dir), transform=train_transform)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
        )

        # Create EMA
        ema_model = ModelEMA(model, decay=0.999)

        # Create scaler for mixed precision
        scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

        # Run one training step
        model.train()
        images, labels = next(iter(train_loader))
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass with mixed precision
        from torch.cuda.amp import autocast
        with autocast():
            embeddings = model(images)
            loss = loss_fn(embeddings, labels)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Update EMA
        ema_model.update(model)

        print(f"  ✅ Training step successful")
        print(f"     Loss: {loss.item():.4f}")
        print(f"     Embeddings shape: {embeddings.shape}")

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(tmp_dir)
        del model, loss_fn, ema_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return True


def test_memory_usage():
    """Test GPU memory usage for batch size 256 on 8 GPUs (32 per GPU)."""
    print("\n[TEST] Memory Usage Estimation")

    if not torch.cuda.is_available():
        print("  ⚠️  CUDA not available, skipping memory test")
        return True

    device = torch.device('cuda:0')

    try:
        # Simulate training setup
        model = DINOv3TeacherModel(
            model_name='dinov3_vitl16',
            embedding_dim=768,
            freeze_backbone=True,
        ).to(device)

        num_classes = 17592
        loss_fn = ArcFaceLoss(
            embedding_dim=768,
            num_classes=num_classes,
            margin=0.5,
            scale=64.0,
        ).to(device)

        optimizer = torch.optim.AdamW(
            list(model.projection.parameters()) + list(loss_fn.parameters()),
            lr=1e-3,
            weight_decay=0.05,
        )

        # Create EMA (doubles memory)
        ema_model = ModelEMA(model, decay=0.999).to(device)

        # Measure memory after model loading
        torch.cuda.synchronize()
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3

        print(f"  Model + Optimizer + EMA memory:")
        print(f"    Allocated: {memory_allocated:.2f} GB")
        print(f"    Reserved: {memory_reserved:.2f} GB")

        # Test forward pass with batch size 32 (per GPU)
        batch_size = 32
        dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
        dummy_labels = torch.randint(0, num_classes, (batch_size,)).to(device)

        from torch.cuda.amp import autocast
        with autocast():
            embeddings = model(dummy_input)
            loss = loss_fn(embeddings, dummy_labels)

        loss.backward()

        torch.cuda.synchronize()
        memory_with_batch = torch.cuda.memory_allocated(device) / 1024**3

        print(f"  With batch size {batch_size}:")
        print(f"    Total memory: {memory_with_batch:.2f} GB")
        print(f"    Activations: {memory_with_batch - memory_allocated:.2f} GB")

        # Check if fits in 40GB A100
        if memory_with_batch < 40:
            print(f"  ✅ Fits in 40GB A100 (headroom: {40 - memory_with_batch:.2f} GB)")
        else:
            print(f"  ❌ EXCEEDS 40GB limit! ({memory_with_batch:.2f} GB)")
            return False

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        del model, loss_fn, optimizer, ema_model
        torch.cuda.empty_cache()

    return True


def test_onnx_export_with_compile():
    """Test that ONNX export works with torch.compile() wrapped models."""
    print("\n[TEST] ONNX Export with torch.compile()")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DINOv3TeacherModel(
        model_name='dinov3_vitl16',
        embedding_dim=768,
        freeze_backbone=True,
    ).to(device)

    # Compile model (this wraps it in OptimizedModule)
    try:
        compiled_model = torch.compile(model, mode='reduce-overhead')
        print(f"  ✅ Model compiled successfully")
    except Exception as e:
        print(f"  ⚠️  torch.compile() not available (PyTorch < 2.0): {e}")
        compiled_model = model

    # Test ONNX export
    try:
        import tempfile
        import os

        # Unwrap torch.compile() wrapper if present (ONNX export incompatible with compiled models)
        export_model = compiled_model
        if hasattr(export_model, '_orig_mod'):
            export_model = export_model._orig_mod
            print(f"  ✅ Unwrapped torch.compile() wrapper")

        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = os.path.join(tmp_dir, 'test_model.onnx')
            dummy_input = torch.randn(1, 3, 224, 224).to(device)

            model.eval()
            torch.onnx.export(
                export_model,
                dummy_input,
                onnx_path,
                input_names=['input'],
                output_names=['embedding'],
                dynamic_axes={'input': {0: 'batch'}, 'embedding': {0: 'batch'}},
                opset_version=17,
            )

            # Verify ONNX file was created
            assert os.path.exists(onnx_path), "ONNX file was not created"
            file_size = os.path.getsize(onnx_path) / (1024 ** 2)  # MB
            print(f"  ✅ ONNX export successful")
            print(f"     File size: {file_size:.2f} MB")

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        del model, compiled_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return True


def main():
    """Run all tests."""
    print("="*60)
    print("DINOv3 Training Pipeline Tests")
    print("="*60)

    tests = [
        ("Model Instantiation", test_model_instantiation),
        ("Forward Pass", test_forward_pass),
        ("EMA with torch.compile", test_ema_with_compile),
        ("Backbone Unfreezing", test_backbone_unfreezing),
        ("DataLoader with Transforms", test_dataloader_with_transforms),
        ("Single Training Step", test_training_step),
        ("ONNX Export with torch.compile", test_onnx_export_with_compile),
        ("Memory Usage", test_memory_usage),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n[TEST] {name}")
            print(f"  ❌ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "="*60)
    print("Test Results")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status:12} {name}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("✅ ALL TESTS PASSED - Safe to launch SageMaker training")
        return 0
    else:
        print("❌ SOME TESTS FAILED - Fix issues before launching")
        return 1


if __name__ == '__main__':
    sys.exit(main())
