#!/usr/bin/env python3
"""
Smoke tests that can run locally without heavy dependencies.
These verify basic code structure and catch syntax errors.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all training files can be imported (syntax check)."""
    print("\n[TEST] Import Syntax Check")

    training_dir = Path(__file__).parent.parent / 'src' / 'training'
    sys.path.insert(0, str(training_dir))

    try:
        # Check syntax by compiling
        train_file = training_dir / 'train_dinov3_teacher.py'
        with open(train_file) as f:
            compile(f.read(), str(train_file), 'exec')
        print("  ✅ train_dinov3_teacher.py syntax valid")

        model_file = training_dir / 'dinov3_embedding.py'
        with open(model_file) as f:
            compile(f.read(), str(model_file), 'exec')
        print("  ✅ dinov3_embedding.py syntax valid")

    except SyntaxError as e:
        print(f"  ❌ SYNTAX ERROR: {e}")
        return False

    return True


def test_dataset_structure():
    """Verify local file structure is correct."""
    print("\n[TEST] File Structure")

    repo_root = Path(__file__).parent.parent

    required_files = [
        'src/training/train_dinov3_teacher.py',
        'src/training/dinov3_embedding.py',
        'src/training/requirements.txt',
        'scripts/launch_teacher_training_8xA100.py',
    ]

    all_exist = True
    for file_path in required_files:
        full_path = repo_root / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ MISSING: {file_path}")
            all_exist = False

    return all_exist


def test_s3_dataset():
    """Verify S3 dataset structure."""
    print("\n[TEST] S3 Dataset Structure")

    try:
        import subprocess
        result = subprocess.run(
            ['aws', 's3', 'ls', 's3://pokemon-card-training-us-east-2/classification_dataset/train/',
             '--region', 'us-east-2'],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            print(f"  ❌ Cannot access S3: {result.stderr}")
            return False

        num_dirs = len(result.stdout.strip().split('\n'))
        print(f"  ✅ Found {num_dirs} class directories in S3")

        if num_dirs != 17592:
            print(f"  ⚠️  Expected 17,592 classes, found {num_dirs}")

        return True

    except Exception as e:
        print(f"  ⚠️  S3 check skipped: {e}")
        return True  # Don't fail if AWS not configured locally


def test_hyperparameters():
    """Verify hyperparameters are consistent."""
    print("\n[TEST] Hyperparameter Consistency")

    launch_script = Path(__file__).parent.parent / 'scripts' / 'launch_teacher_training_8xA100.py'

    with open(launch_script) as f:
        content = f.read()

    checks = [
        ("'dinov3-model': 'dinov3_vitl16'", "Model is ViT-Large"),
        ("'embedding-dim': 768", "Embedding dimension is 768"),
        ("'batch-size': 256", "Batch size is 256"),
        ("'epochs-frozen': 3", "Frozen epochs is 3"),
        ("'epochs-unfrozen': 10", "Unfrozen epochs is 10"),
    ]

    all_correct = True
    for check_str, description in checks:
        if check_str in content:
            print(f"  ✅ {description}")
        else:
            print(f"  ❌ MISSING: {description}")
            all_correct = False

    return all_correct


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("Smoke Tests (No Dependencies Required)")
    print("=" * 60)

    tests = [
        ("Import Syntax", test_imports),
        ("File Structure", test_dataset_structure),
        ("S3 Dataset", test_s3_dataset),
        ("Hyperparameters", test_hyperparameters),
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

    print("\n" + "=" * 60)
    print("Smoke Test Results")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status:12} {name}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("✅ ALL SMOKE TESTS PASSED")
        print("   Ready to launch SageMaker training")
        print("   (Full integration tests will run on SageMaker)")
        return 0
    else:
        print("❌ SOME SMOKE TESTS FAILED")
        print("   Fix these issues before launching")
        return 1


if __name__ == '__main__':
    sys.exit(main())
