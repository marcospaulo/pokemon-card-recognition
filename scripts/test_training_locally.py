#!/usr/bin/env python3
"""
Local training validation script
Tests all dependencies and runs a few training iterations with small data
to catch bugs before submitting expensive SageMaker jobs
"""

import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

print("="*60)
print("Testing Training Setup Locally")
print("="*60)

# Test 1: Check imports
print("\n1. Testing imports...")
try:
    from transformers import DINOv3ViTModel, DINOv3ViTConfig
    print("   ✓ transformers imported successfully")
    from models.dinov3_embedding import DINOv3TeacherModel, ArcFaceLoss
    print("   ✓ Custom models imported successfully")
    import albumentations as A
    print("   ✓ albumentations imported successfully")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    print("\n   Install dependencies:")
    print("   pip install transformers>=4.30.0 albumentations>=1.3.0")
    sys.exit(1)

# Test 2: Check CUDA
print("\n2. Checking GPU availability...")
if torch.cuda.is_available():
    print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"   ✓ CUDA version: {torch.version.cuda}")
    device = torch.device('cuda')
else:
    print("   ⚠ CUDA not available, using CPU (will be slow)")
    device = torch.device('cpu')

# Test 3: Load DINOv3 model
print("\n3. Loading DINOv3 model...")
try:
    model = DINOv3TeacherModel(
        model_name='dinov3_vitl16',
        embedding_dim=768,
        freeze_backbone=True,
    )
    print(f"   ✓ Model loaded successfully")
    print(f"   ✓ Backbone parameters: {sum(p.numel() for p in model.backbone.parameters()):,}")
    print(f"   ✓ Projection parameters: {sum(p.numel() for p in model.projection.parameters()):,}")

    # Move to device
    model = model.to(device)
    print(f"   ✓ Model moved to {device}")
except Exception as e:
    print(f"   ✗ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test forward pass
print("\n4. Testing forward pass...")
try:
    # Create dummy batch
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 224, 224).to(device)

    with torch.no_grad():
        embeddings = model(dummy_images)

    print(f"   ✓ Forward pass successful")
    print(f"   ✓ Input shape: {dummy_images.shape}")
    print(f"   ✓ Output shape: {embeddings.shape}")
    print(f"   ✓ Output normalized: {torch.allclose(torch.norm(embeddings, dim=1), torch.ones(batch_size).to(device), atol=1e-5)}")
except Exception as e:
    print(f"   ✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test ArcFace loss
print("\n5. Testing ArcFace loss...")
try:
    num_classes = 17592  # Full Pokemon card dataset
    loss_fn = ArcFaceLoss(
        embedding_dim=768,
        num_classes=num_classes,
        margin=0.5,
        scale=64.0,
    ).to(device)

    print(f"   ✓ ArcFace loss created")
    print(f"   ✓ Weight matrix shape: {loss_fn.weight.shape}")
    print(f"   ✓ Number of classes: {num_classes:,}")

    # Test loss computation
    dummy_labels = torch.randint(0, num_classes, (batch_size,)).to(device)
    loss = loss_fn(embeddings, dummy_labels)

    print(f"   ✓ Loss computed successfully: {loss.item():.4f}")
except Exception as e:
    print(f"   ✗ ArcFace loss failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test backward pass
print("\n6. Testing backward pass...")
try:
    optimizer = torch.optim.AdamW(
        list(model.projection.parameters()) + list(loss_fn.parameters()),
        lr=1e-3
    )

    # Forward + backward
    model.train()
    embeddings = model(dummy_images)
    loss = loss_fn(embeddings, dummy_labels)
    loss.backward()
    optimizer.step()

    print(f"   ✓ Backward pass successful")
    print(f"   ✓ Gradients computed and optimizer step completed")
except Exception as e:
    print(f"   ✗ Backward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test unfreezing backbone
print("\n7. Testing backbone unfreezing...")
try:
    model.unfreeze_backbone(last_n_blocks=4)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"   ✓ Backbone unfrozen (last 4 blocks)")
    print(f"   ✓ Trainable params: {trainable_params:,} / {total_params:,}")
    print(f"   ✓ Trainable ratio: {100 * trainable_params / total_params:.1f}%")
except Exception as e:
    print(f"   ✗ Unfreezing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Check dataset structure (if exists)
print("\n8. Checking dataset structure...")
data_dir = Path(__file__).parent.parent / 'data' / 'processed' / 'classification'
train_dir = data_dir / 'train'

if train_dir.exists():
    # Count classes and images
    classes = [d for d in train_dir.iterdir() if d.is_dir()]
    total_images = sum(len(list(c.glob('*.png'))) + len(list(c.glob('*.jpg'))) for c in classes)

    print(f"   ✓ Dataset found at {train_dir}")
    print(f"   ✓ Number of classes: {len(classes):,}")
    print(f"   ✓ Total images: {total_images:,}")

    # Test loading a single image
    try:
        from torchvision.datasets import ImageFolder
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        dataset = ImageFolder(str(train_dir), transform=transform)
        img, label = dataset[0]

        print(f"   ✓ Successfully loaded sample image")
        print(f"   ✓ Image shape: {img.shape}")
        print(f"   ✓ Label: {label}")
    except Exception as e:
        print(f"   ⚠ Could not test image loading: {e}")
else:
    print(f"   ⚠ Dataset not found at {train_dir}")
    print(f"   (This is OK for SageMaker - data will be loaded from S3)")

print("\n" + "="*60)
print("✅ All tests passed! Ready for SageMaker training")
print("="*60)
print("\nNext steps:")
print("1. Run: cd scripts && python3 launch_teacher_training_8xA100.py")
print("2. Monitor: SageMaker console")
print("="*60)
