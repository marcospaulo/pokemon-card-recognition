#!/usr/bin/env python3
"""
COMPREHENSIVE PRE-FLIGHT CHECK
Run this BEFORE submitting to SageMaker to catch all errors locally
"""

import sys
import subprocess
from pathlib import Path
import json

print("="*70)
print("PRE-FLIGHT CHECK - Validating training setup")
print("="*70)

errors = []
warnings = []

# Test 1: Check training script imports
print("\n[1/7] Testing training script imports...")
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'training'))
    from dinov3_embedding import DINOv3TeacherModel, ArcFaceLoss
    print("  ✓ dinov3_embedding.py imports successfully")
except Exception as e:
    errors.append(f"Import error: {e}")
    print(f"  ✗ Import failed: {e}")

# Test 2: Check all required files exist
print("\n[2/7] Checking required files...")
required_files = [
    'src/training/train_dinov3_teacher.py',
    'src/training/dinov3_embedding.py',
    'src/training/requirements.txt',
    'scripts/launch_teacher_training_8xA100.py',
]

for file_path in required_files:
    full_path = Path(__file__).parent.parent / file_path
    if full_path.exists():
        print(f"  ✓ {file_path}")
    else:
        errors.append(f"Missing file: {file_path}")
        print(f"  ✗ Missing: {file_path}")

# Test 3: Check dataset structure
print("\n[3/7] Checking dataset structure...")
data_dir = Path(__file__).parent.parent / 'data' / 'processed' / 'classification'
train_dir = data_dir / 'train'

if not train_dir.exists():
    errors.append("Train directory doesn't exist")
    print(f"  ✗ Train directory not found: {train_dir}")
else:
    train_cards = [d for d in train_dir.iterdir() if d.is_dir()]
    num_train_cards = len(train_cards)

    print(f"  ✓ Train directory found")
    print(f"  ✓ Number of card classes: {num_train_cards:,}")

    if num_train_cards < 17500:
        warnings.append(f"Only {num_train_cards} cards in training (expected ~17,592)")
        print(f"  ⚠ Warning: Expected ~17,592 cards, found {num_train_cards}")

    # Check for images
    sample_card = train_cards[0] if train_cards else None
    if sample_card:
        images = list(sample_card.glob('*.png')) + list(sample_card.glob('*.jpg'))
        if images:
            print(f"  ✓ Sample card has {len(images)} image(s)")
        else:
            errors.append(f"No images found in {sample_card.name}")

# Test 4: Check S3 data exists
print("\n[4/7] Checking S3 dataset...")
try:
    result = subprocess.run(
        ['aws', 's3', 'ls', 's3://pokemon-card-training-us-east-2/classification_dataset/'],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode == 0 and 'train/' in result.stdout:
        print("  ✓ S3 dataset exists")

        # Check metadata
        meta_result = subprocess.run(
            ['aws', 's3', 'ls', 's3://pokemon-card-training-us-east-2/classification_dataset/card_metadata.json'],
            capture_output=True, text=True, timeout=10
        )
        if meta_result.returncode == 0:
            print("  ✓ card_metadata.json exists in S3")
        else:
            warnings.append("card_metadata.json not found in S3")
    else:
        errors.append("S3 dataset not found or incomplete")
        print("  ✗ S3 dataset missing or incomplete")
except Exception as e:
    warnings.append(f"Couldn't check S3: {e}")
    print(f"  ⚠ Couldn't check S3: {e}")

# Test 5: Validate requirements.txt
print("\n[5/7] Checking requirements.txt...")
req_file = Path(__file__).parent.parent / 'src' / 'training' / 'requirements.txt'
if req_file.exists():
    requirements = req_file.read_text()
    if 'transformers>=4.56.0' in requirements or 'transformers>=4.30.0' in requirements:
        print("  ✓ transformers version specified correctly")
    else:
        errors.append("transformers version in requirements.txt is incorrect")
        print("  ✗ transformers version incorrect")

    if 'albumentations' in requirements:
        print("  ✓ albumentations included")
    else:
        warnings.append("albumentations not in requirements")
else:
    errors.append("requirements.txt not found")

# Test 6: Check AWS credentials
print("\n[6/7] Checking AWS credentials...")
try:
    result = subprocess.run(
        ['aws', 'sts', 'get-caller-identity'],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode == 0:
        identity = json.loads(result.stdout)
        print(f"  ✓ AWS credentials valid")
        print(f"    Account: {identity.get('Account')}")
    else:
        errors.append("AWS credentials invalid")
        print("  ✗ AWS credentials invalid")
except Exception as e:
    errors.append(f"AWS check failed: {e}")
    print(f"  ✗ AWS check failed: {e}")

# Test 7: Check SageMaker role
print("\n[7/7] Checking SageMaker role...")
try:
    result = subprocess.run(
        ['aws', 'iam', 'get-role', '--role-name', 'SageMaker-ExecutionRole'],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode == 0:
        print("  ✓ SageMaker-ExecutionRole exists")
    else:
        errors.append("SageMaker-ExecutionRole not found")
        print("  ✗ SageMaker-ExecutionRole not found")
except Exception as e:
    warnings.append(f"Couldn't check SageMaker role: {e}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

if errors:
    print(f"\n❌ {len(errors)} ERROR(S) FOUND:")
    for i, error in enumerate(errors, 1):
        print(f"  {i}. {error}")
    print("\n⛔ DO NOT SUBMIT TO SAGEMAKER - FIX ERRORS FIRST")
    sys.exit(1)
elif warnings:
    print(f"\n⚠️  {len(warnings)} WARNING(S):")
    for i, warning in enumerate(warnings, 1):
        print(f"  {i}. {warning}")
    print("\n✅ No critical errors, but review warnings")
    sys.exit(0)
else:
    print("\n✅ ALL CHECKS PASSED")
    print("Ready to submit to SageMaker!")
    sys.exit(0)
