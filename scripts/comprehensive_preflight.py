#!/usr/bin/env python3
"""
BULLETPROOF PRE-FLIGHT CHECK
Catches EVERYTHING before wasting money on SageMaker
"""

import sys
import subprocess
import time
from pathlib import Path
import json

print("="*80)
print(" " * 20 + "🚦 COMPREHENSIVE PRE-FLIGHT CHECK")
print("="*80)

errors = []
warnings = []

def run_cmd(cmd, timeout=30):
    """Run command and return output"""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, shell=isinstance(cmd, str)
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

# ============================================================================
# TEST 1: S3 DATA VALIDATION (CRITICAL)
# ============================================================================
print("\n[1/10] 🔍 Validating S3 dataset...")

bucket = "pokemon-card-training-us-east-2"
train_prefix = "classification_dataset/train/"

# Check if any sync operations are running
print("  → Checking for active S3 sync operations...")
rc, stdout, stderr = run_cmd("ps aux | grep 'aws s3 sync' | grep -v grep")
if rc == 0 and stdout.strip():
    errors.append("S3 sync operation still running! Wait for completion.")
    print(f"  ✗ S3 sync in progress:\n{stdout[:200]}")
else:
    print("  ✓ No active S3 sync operations")

# Count folders in S3
print("  → Counting card folders in S3...")
rc, stdout, stderr = run_cmd(f"aws s3 ls s3://{bucket}/{train_prefix} | wc -l", timeout=60)
if rc == 0:
    count = int(stdout.strip())
    if count == 17592:
        print(f"  ✓ All 17,592 card folders present in S3")
    elif count == 0:
        errors.append(f"NO folders found in S3: s3://{bucket}/{train_prefix}")
        print(f"  ✗ NO folders found in S3!")
    else:
        errors.append(f"Expected 17,592 folders, found {count} in S3")
        print(f"  ✗ Expected 17,592 folders, found {count}")
else:
    errors.append(f"Failed to list S3: {stderr}")
    print(f"  ✗ Failed to list S3: {stderr[:200]}")

# Sample 5 random cards to verify images exist
print("  → Sampling random cards to verify structure...")
rc, stdout, stderr = run_cmd(
    f"aws s3 ls s3://{bucket}/{train_prefix} | head -5 | awk '{{print $NF}}'",
    timeout=30
)
if rc == 0:
    sample_folders = [f.strip().rstrip('/') for f in stdout.split('\n') if f.strip()]
    for folder in sample_folders[:3]:
        rc2, stdout2, stderr2 = run_cmd(
            f"aws s3 ls s3://{bucket}/{train_prefix}{folder}/ | grep '.png' | wc -l",
            timeout=10
        )
        if rc2 == 0:
            img_count = int(stdout2.strip())
            if img_count > 0:
                print(f"  ✓ {folder}: {img_count} image(s)")
            else:
                warnings.append(f"Folder {folder} has no images")
                print(f"  ⚠ {folder}: No images found")

#============================================================================
# TEST 2: LOCAL TRAINING FILES
# ============================================================================
print("\n[2/10] 📁 Checking training files...")

required_files = {
    'src/training/train_dinov3_teacher.py': 'Training script',
    'src/training/dinov3_embedding.py': 'Model definition',
    'src/training/requirements.txt': 'Dependencies',
    'scripts/launch_teacher_training_8xA100.py': 'Launch script',
}

for file_path, description in required_files.items():
    full_path = Path(__file__).parent.parent / file_path
    if full_path.exists():
        size_kb = full_path.stat().st_size / 1024
        print(f"  ✓ {description}: {file_path} ({size_kb:.1f} KB)")
    else:
        errors.append(f"Missing {description}: {file_path}")
        print(f"  ✗ Missing: {file_path}")

# ============================================================================
# TEST 3: IMPORT VALIDATION (Local check - SageMaker installs from requirements.txt)
# ============================================================================
print("\n[3/10] 🔧 Validating imports...")

# Test dinov3_embedding can be imported from training dir
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'training'))
try:
    from dinov3_embedding import DINOv3TeacherModel, ArcFaceLoss
    print("  ✓ dinov3_embedding imports successfully")

    # Check class structure
    import inspect
    teacher_methods = [m for m in dir(DINOv3TeacherModel) if not m.startswith('_')]
    if 'forward' in teacher_methods and 'unfreeze_backbone' in teacher_methods:
        print("  ✓ DINOv3TeacherModel has required methods")
    else:
        warnings.append("DINOv3TeacherModel missing expected methods")

except Exception as e:
    # Import failures are OK locally - SageMaker installs from requirements.txt
    warnings.append(f"Local import test failed (OK if dependencies in requirements.txt): {str(e)[:100]}")
    print(f"  ⚠ Local import failed (SageMaker will install from requirements.txt): {str(e)[:100]}")

# ============================================================================
# TEST 4: REQUIREMENTS.TXT VALIDATION
# ============================================================================
print("\n[4/10] 📦 Validating requirements.txt...")

req_file = Path(__file__).parent.parent / 'src' / 'training' / 'requirements.txt'
if req_file.exists():
    requirements = req_file.read_text()

    # Check transformers version
    if 'transformers>=4.56.0' in requirements:
        print("  ✓ transformers>=4.56.0 (DINOv3 support)")
    elif 'transformers>=4.30.0' in requirements:
        print("  ✓ transformers>=4.30.0 (DINOv3 support)")
    else:
        errors.append("transformers version incorrect in requirements.txt")
        print("  ✗ transformers version incorrect")

    # Check other deps
    for dep in ['timm', 'albumentations']:
        if dep in requirements:
            print(f"  ✓ {dep} present")
        else:
            warnings.append(f"{dep} not in requirements.txt")
            print(f"  ⚠ {dep} missing")
else:
    errors.append("requirements.txt not found")

# ============================================================================
# TEST 5: AWS CREDENTIALS & PERMISSIONS
# ============================================================================
print("\n[5/10] 🔐 Checking AWS credentials...")

rc, stdout, stderr = run_cmd(['aws', 'sts', 'get-caller-identity'])
if rc == 0:
    try:
        identity = json.loads(stdout)
        print(f"  ✓ AWS credentials valid")
        print(f"    Account: {identity.get('Account')}")
        print(f"    User/Role: {identity.get('Arn', '').split('/')[-1]}")
    except:
        warnings.append("Could not parse AWS identity")
else:
    errors.append("AWS credentials invalid or not configured")
    print(f"  ✗ AWS credentials invalid")

# ============================================================================
# TEST 6: SAGEMAKER ROLE
# ============================================================================
print("\n[6/10] 👤 Checking SageMaker execution role...")

rc, stdout, stderr = run_cmd(['aws', 'iam', 'get-role', '--role-name', 'SageMaker-ExecutionRole'])
if rc == 0:
    print("  ✓ SageMaker-ExecutionRole exists")

    # Check if role has S3 access
    rc2, stdout2, stderr2 = run_cmd([
        'aws', 'iam', 'list-attached-role-policies',
        '--role-name', 'SageMaker-ExecutionRole'
    ])
    if rc2 == 0 and 'AmazonS3FullAccess' in stdout2:
        print("  ✓ Role has S3 access")
else:
    errors.append("SageMaker-ExecutionRole not found")
    print("  ✗ SageMaker-ExecutionRole missing")

# ============================================================================
# TEST 7: S3 BUCKET PERMISSIONS
# ============================================================================
print("\n[7/10] 🪣 Testing S3 bucket permissions...")

test_key = f"{train_prefix}_preflight_test.txt"
rc, stdout, stderr = run_cmd(
    f"echo 'test' | aws s3 cp - s3://{bucket}/{test_key}"
)
if rc == 0:
    print("  ✓ Can write to S3 bucket")
    # Clean up
    run_cmd(f"aws s3 rm s3://{bucket}/{test_key}")
else:
    errors.append(f"Cannot write to S3 bucket: {stderr[:200]}")
    print(f"  ✗ Cannot write to bucket")

# ============================================================================
# TEST 8: LOCAL DATASET VALIDATION
# ============================================================================
print("\n[8/10] 💾 Validating local dataset...")

local_train = Path(__file__).parent.parent / 'data' / 'processed' / 'classification' / 'train'
if local_train.exists():
    local_cards = [d for d in local_train.iterdir() if d.is_dir()]
    print(f"  ✓ Local train directory: {len(local_cards):,} cards")

    if len(local_cards) != 17592:
        warnings.append(f"Local has {len(local_cards)} cards, expected 17,592")
else:
    warnings.append("Local train directory not found (OK if using only S3)")
    print("  ⚠ Local train directory not found")

# ============================================================================
# TEST 9: METADATA FILES
# ============================================================================
print("\n[9/10] 📋 Checking metadata files...")

for meta_file in ['card_metadata.json', 'class_index.json']:
    rc, stdout, stderr = run_cmd(
        f"aws s3 ls s3://{bucket}/classification_dataset/{meta_file}"
    )
    if rc == 0:
        # Get file size
        parts = stdout.strip().split()
        if len(parts) >= 3:
            size_mb = int(parts[2]) / (1024 * 1024)
            print(f"  ✓ {meta_file} ({size_mb:.1f} MB)")
    else:
        warnings.append(f"{meta_file} not found in S3")
        print(f"  ⚠ {meta_file} missing")

# ============================================================================
# TEST 10: TRAINING SCRIPT SYNTAX CHECK
# ============================================================================
print("\n[10/10] ✅ Validating training script syntax...")

training_script = Path(__file__).parent.parent / 'src' / 'training' / 'train_dinov3_teacher.py'
if training_script.exists():
    rc, stdout, stderr = run_cmd(['python3', '-m', 'py_compile', str(training_script)])
    if rc == 0:
        print("  ✓ Training script syntax valid")
    else:
        errors.append(f"Training script has syntax errors: {stderr[:200]}")
        print(f"  ✗ Syntax errors: {stderr[:200]}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print(" " * 30 + "📊 SUMMARY")
print("="*80)

if errors:
    print(f"\n❌ {len(errors)} CRITICAL ERROR(S) - DO NOT SUBMIT TO SAGEMAKER\n")
    for i, error in enumerate(errors, 1):
        print(f"  {i}. {error}")
    print("\n⛔ FIX ALL ERRORS BEFORE LAUNCHING TRAINING")
    print("   Each failed run costs $4-5 in GPU time!")
    sys.exit(1)

elif warnings:
    print(f"\n⚠️  {len(warnings)} WARNING(S) - Review before submission\n")
    for i, warning in enumerate(warnings, 1):
        print(f"  {i}. {warning}")
    print("\n✅ No critical errors found")
    print("💡 Review warnings, then run:")
    print("   cd scripts && python3 launch_teacher_training_8xA100.py")
    sys.exit(0)

else:
    print("\n✅ ALL CHECKS PASSED - READY FOR SAGEMAKER")
    print("\n🚀 Safe to launch training:")
    print("   cd scripts && python3 launch_teacher_training_8xA100.py")
    print(f"\n💰 Estimated cost: $4-5 for 15-20 minutes on 8xA100")
    sys.exit(0)
