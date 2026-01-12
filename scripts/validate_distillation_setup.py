#!/usr/bin/env python3
"""
Validate student distillation setup BEFORE launching expensive GPU jobs.
Catches parameter errors, import issues, and model creation problems locally.

Run this FIRST to avoid wasting money on SageMaker failures.
"""

import sys
import os
import tempfile
import tarfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'training'))

def test_imports():
    """Test that all required imports work."""
    print("=" * 70)
    print("TEST 1: Import Validation")
    print("=" * 70)

    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")

        import transformers
        print(f"✅ Transformers {transformers.__version__}")

        from dinov3_embedding import DINOv3TeacherModel, ArcFaceLoss
        print(f"✅ dinov3_embedding imports")

        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_teacher_model_creation():
    """Test creating teacher model with correct parameters."""
    print("\n" + "=" * 70)
    print("TEST 2: Teacher Model Creation (CPU, no weights)")
    print("=" * 70)

    try:
        from dinov3_embedding import DINOv3TeacherModel
        import torch

        # Test the EXACT call we use in training
        print("Creating DINOv3TeacherModel with from_checkpoint=True...")
        print("  Parameters: model_name='dinov3_vitl16', embedding_dim=768")
        print("  This will download config from HuggingFace (~1KB JSON)")

        model = DINOv3TeacherModel(
            model_name='dinov3_vitl16',
            embedding_dim=768,
            from_checkpoint=True,  # Downloads config only, creates structure
        )

        print(f"✅ Model created successfully")
        print(f"   Backbone: {type(model.backbone).__name__}")
        print(f"   Projection: {model.projection}")

        # Test forward pass with dummy input
        print("\nTesting forward pass with dummy input (1x3x224x224)...")
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✅ Forward pass successful, output shape: {output.shape}")

        return True

    except TypeError as e:
        print(f"❌ Parameter error (this is what we want to catch!): {e}")
        print("\n🔍 Checking DINOv3TeacherModel.__init__ signature:")
        import inspect
        sig = inspect.signature(DINOv3TeacherModel.__init__)
        print(f"   Signature: {sig}")
        return False

    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint_loading_simulation():
    """Simulate checkpoint loading to catch state_dict issues."""
    print("\n" + "=" * 70)
    print("TEST 3: Checkpoint Loading Simulation")
    print("=" * 70)

    try:
        from dinov3_embedding import DINOv3TeacherModel, ArcFaceLoss
        import torch

        # Create model (downloads config, creates structure with exact dimensions)
        print("Creating model with from_checkpoint=True (downloads HF config)...")
        model = DINOv3TeacherModel(
            model_name='dinov3_vitl16',
            embedding_dim=768,
            from_checkpoint=True,  # Downloads config to get exact model structure
        )

        # Save its state_dict (simulate torch.compile() by adding _orig_mod. prefix)
        print("Saving state_dict with _orig_mod. prefix (simulating torch.compile())...")
        state_dict = model.state_dict()
        # Add _orig_mod. prefix to simulate what torch.compile() does
        state_dict_with_prefix = {f'_orig_mod.{key}': value for key, value in state_dict.items()}

        # Create a new model and try loading
        print("Creating fresh model (downloads HF config) and loading state_dict...")
        model2 = DINOv3TeacherModel(
            model_name='dinov3_vitl16',
            embedding_dim=768,
            from_checkpoint=True,  # Downloads same config for exact structure match
        )

        # Strip _orig_mod. prefix (same as training code)
        if any(key.startswith('_orig_mod.') for key in state_dict_with_prefix.keys()):
            print("  Detected _orig_mod. prefix - stripping...")
            state_dict_clean = {key.replace('_orig_mod.', ''): value
                               for key, value in state_dict_with_prefix.items()}
            model2.load_state_dict(state_dict_clean)
        else:
            model2.load_state_dict(state_dict_with_prefix)

        print(f"✅ State dict loads successfully")
        print(f"   Keys: {len(state_dict)} parameters")

        return True

    except Exception as e:
        print(f"❌ State dict loading failed: {e}")
        return False


def test_arcface_creation():
    """Test ArcFace loss creation."""
    print("\n" + "=" * 70)
    print("TEST 4: ArcFace Loss Creation")
    print("=" * 70)

    try:
        from dinov3_embedding import ArcFaceLoss

        loss_fn = ArcFaceLoss(
            embedding_dim=768,
            num_classes=17592,
            margin=0.5,
            scale=64.0,
        )

        print(f"✅ ArcFace loss created successfully")
        print(f"   Embedding dim: 768, Num classes: 17592")

        return True

    except Exception as e:
        print(f"❌ ArcFace creation failed: {e}")
        return False


def test_student_model_imports():
    """Test student model imports."""
    print("\n" + "=" * 70)
    print("TEST 5: Student Model Imports")
    print("=" * 70)

    try:
        import timm
        print(f"✅ timm {timm.__version__}")

        # Test creating ConvNeXt-Tiny
        print("Creating ConvNeXt-Tiny (student model)...")
        student = timm.create_model('convnext_tiny', pretrained=False, num_classes=0)
        print(f"✅ ConvNeXt-Tiny created")

        import torch
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = student(dummy_input)
        print(f"✅ Student forward pass successful, output shape: {output.shape}")

        return True

    except Exception as e:
        print(f"❌ Student model test failed: {e}")
        return False


def main():
    print("\n" + "=" * 70)
    print("Student Distillation Setup Validation")
    print("This catches errors BEFORE launching expensive GPU jobs")
    print("=" * 70)

    results = []

    # Run tests
    results.append(("Imports", test_imports()))

    if results[0][1]:  # Only continue if imports work
        results.append(("Teacher Model Creation", test_teacher_model_creation()))
        results.append(("Checkpoint Loading", test_checkpoint_loading_simulation()))
        results.append(("ArcFace Loss", test_arcface_creation()))
        results.append(("Student Model", test_student_model_imports()))

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("✅ ALL TESTS PASSED - Safe to launch SageMaker training!")
        print("\nNext step:")
        print("  python scripts/launch_student_distillation_8xA100.py")
        return 0
    else:
        print("❌ VALIDATION FAILED - DO NOT launch SageMaker!")
        print("\nFix the errors above before launching expensive GPU jobs.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
