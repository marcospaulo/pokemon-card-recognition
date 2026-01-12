#!/usr/bin/env python3
"""
Re-save teacher model checkpoint with architecture config included.

This eliminates the need for HuggingFace access during future inference.
The checkpoint will be self-contained with both weights and architecture.
"""

import sys
import os
import tempfile
import tarfile
from pathlib import Path

# Add src/training to path BEFORE importing torch
# This ensures WarmupCosineScheduler and ModelEMA classes are available during unpickling
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'training'))

import torch
import boto3
from botocore.exceptions import NoCredentialsError

# Import custom classes that are referenced in the checkpoint pickle
# These MUST be imported before torch.load() or unpickling will fail
from train_dinov3_teacher import WarmupCosineScheduler, ModelEMA


def download_checkpoint_from_s3():
    """Download teacher checkpoint from S3."""
    bucket = 'pokemon-card-training-us-east-2'
    key = 'models/embedding/teacher/pokemon-card-dinov3-teacher-2026-01-10-13-31-34-937/output/model.tar.gz'

    print(f"Downloading checkpoint from S3...")
    print(f"  Bucket: {bucket}")
    print(f"  Key: {key}")

    s3 = boto3.client('s3')

    # Download to scratchpad directory (not /tmp)
    scratchpad = Path('/var/folders/_q/76yzpcn13zq62pgyh1dr6yqh0000gn/T/claude/-Users-marcos-dev-raspberry-pi/6320e65a-7ac4-430b-a17d-d866aec3de15/scratchpad')
    scratchpad.mkdir(parents=True, exist_ok=True)

    local_path = scratchpad / 'model.tar.gz'

    print(f"Downloading to: {local_path}")
    s3.download_file(bucket, key, str(local_path))

    file_size_mb = local_path.stat().st_size / (1024 * 1024)
    print(f"✅ Downloaded {file_size_mb:.1f} MB")

    return local_path


def extract_checkpoint(tar_path):
    """Extract phase2_checkpoint.pt from tar.gz."""
    print(f"\nExtracting checkpoint from {tar_path}...")

    extract_dir = tar_path.parent / 'extracted'
    extract_dir.mkdir(exist_ok=True)

    with tarfile.open(tar_path, 'r:gz') as tar:
        # List contents
        print("Archive contents:")
        for member in tar.getmembers():
            print(f"  - {member.name} ({member.size / (1024*1024):.1f} MB)")

        # Extract only phase2_checkpoint.pt
        checkpoint_member = None
        for member in tar.getmembers():
            if 'phase2_checkpoint.pt' in member.name:
                checkpoint_member = member
                break

        if not checkpoint_member:
            raise FileNotFoundError("phase2_checkpoint.pt not found in archive")

        print(f"\nExtracting {checkpoint_member.name}...")
        tar.extract(checkpoint_member, path=extract_dir)

    checkpoint_path = extract_dir / checkpoint_member.name
    print(f"✅ Extracted to: {checkpoint_path}")

    return checkpoint_path


def load_and_resave_checkpoint(checkpoint_path, tar_path):
    """Load checkpoint and re-save with architecture config."""
    print(f"\nLoading checkpoint with custom classes in scope...")
    print(f"  WarmupCosineScheduler: {WarmupCosineScheduler}")
    print(f"  ModelEMA: {ModelEMA}")

    # Load checkpoint with weights_only=False (required for custom classes)
    # The custom classes are now in scope, so unpickling will succeed
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Immediately delete tar.gz to free up space (5.7GB)
    print(f"\n🗑️  Cleaning up tar.gz to free disk space...")
    if tar_path.exists():
        tar_path.unlink()
        print(f"  Deleted {tar_path.name} (freed ~5.7GB)")

    print(f"✅ Checkpoint loaded successfully")
    print(f"\nCheckpoint keys:")
    for key in checkpoint.keys():
        value = checkpoint[key]
        if isinstance(value, dict):
            print(f"  - {key}: dict with {len(value)} items")
        elif isinstance(value, torch.Tensor):
            print(f"  - {key}: tensor {value.shape}")
        else:
            print(f"  - {key}: {type(value).__name__}")

    # Create a new checkpoint with architecture config included
    print(f"\nCreating complete checkpoint with architecture config...")

    # Use EMA model if available (better quality), otherwise use regular model
    model_state = checkpoint.get('ema_model', checkpoint['model'])

    complete_checkpoint = {
        # Only include what's needed for inference (no optimizer/scheduler)
        'model': model_state,                   # Model state_dict (use EMA if available)
        'loss_fn': checkpoint['loss_fn'],       # ArcFace state_dict

        # Training metadata (for reference only)
        'epoch': checkpoint.get('epoch'),
        'top1_acc': checkpoint.get('top1_acc'),
        'num_classes': checkpoint.get('num_classes'),

        # NEW: Architecture configuration (the missing piece!)
        'model_config': {
            'model_name': 'dinov3_vitl16',       # DINOv3-ViT-Large
            'embedding_dim': 768,                # Embedding dimension
            'freeze_backbone': False,            # Backbone was fine-tuned
            'from_checkpoint': False,            # Not needed with config
        },

        # NEW: Loss configuration
        'loss_config': {
            'num_classes': 17592,                # Pokemon card classes
            'embedding_dim': 768,
            'margin': 0.5,                       # ArcFace margin
            'scale': 64.0,                       # ArcFace scale
        },

        # Metadata
        'training_metadata': {
            'original_job': 'pokemon-card-dinov3-teacher-2026-01-10-13-31-34-937',
            'model_type': 'DINOv3TeacherModel',
            'purpose': 'Fine-tuned for Pokemon card recognition',
            'resaved_with_config': True,
            'used_ema_weights': 'ema_model' in checkpoint,
        }
    }

    print(f"  Using {'EMA' if 'ema_model' in checkpoint else 'regular'} model weights")
    print(f"  Excluded optimizer and scheduler (not needed for inference)")

    # Save the complete checkpoint
    output_path = checkpoint_path.parent / 'dinov3_teacher_complete.pt'
    print(f"\nSaving complete checkpoint to: {output_path}")
    torch.save(complete_checkpoint, output_path)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Saved {file_size_mb:.1f} MB")

    return output_path


def verify_checkpoint(checkpoint_path):
    """Verify the checkpoint can be loaded without HuggingFace access."""
    print(f"\n" + "="*70)
    print("VERIFICATION: Load checkpoint without HuggingFace")
    print("="*70)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    print("✅ Checkpoint loaded successfully")
    print(f"\nModel config:")
    for key, value in checkpoint['model_config'].items():
        print(f"  - {key}: {value}")

    print(f"\nLoss config:")
    for key, value in checkpoint['loss_config'].items():
        print(f"  - {key}: {value}")

    print(f"\nState dict keys:")
    print(f"  - model: {len(checkpoint['model'])} parameters")
    print(f"  - loss_fn: {len(checkpoint['loss_fn'])} parameters")

    # Test creating model and loading weights
    print(f"\nTesting model creation and weight loading...")
    from dinov3_embedding import DINOv3TeacherModel, ArcFaceLoss

    # Create model using config from checkpoint
    config = checkpoint['model_config']
    model = DINOv3TeacherModel(
        model_name=config['model_name'],
        embedding_dim=config['embedding_dim'],
        freeze_backbone=config['freeze_backbone'],
        from_checkpoint=False,
    )

    # Load weights
    model.load_state_dict(checkpoint['model'])
    print(f"✅ Model created and weights loaded successfully")

    # Create loss function
    loss_config = checkpoint['loss_config']
    loss_fn = ArcFaceLoss(
        embedding_dim=loss_config['embedding_dim'],
        num_classes=loss_config['num_classes'],
        margin=loss_config['margin'],
        scale=loss_config['scale'],
    )
    loss_fn.load_state_dict(checkpoint['loss_fn'])
    print(f"✅ Loss function created and weights loaded successfully")

    print(f"\n" + "="*70)
    print("VERIFICATION PASSED - Checkpoint is self-contained!")
    print("="*70)


def upload_to_s3(local_path):
    """Upload complete checkpoint back to S3."""
    bucket = 'pokemon-card-training-us-east-2'
    key = 'models/embedding/teacher/dinov3_teacher_complete.pt'

    print(f"\nUploading to S3...")
    print(f"  Bucket: {bucket}")
    print(f"  Key: {key}")

    s3 = boto3.client('s3')
    s3.upload_file(str(local_path), bucket, key)

    s3_url = f"s3://{bucket}/{key}"
    print(f"✅ Uploaded to: {s3_url}")

    return s3_url


def main():
    print("="*70)
    print("Re-save Teacher Model with Architecture Config")
    print("="*70)
    print("\nThis will:")
    print("1. Download teacher checkpoint from S3")
    print("2. Load checkpoint with custom classes in scope")
    print("3. Add architecture config to checkpoint")
    print("4. Save complete self-contained checkpoint")
    print("5. Verify it can be loaded without HuggingFace")
    print("6. Upload back to S3")
    print("="*70 + "\n")

    try:
        # Step 1: Download
        tar_path = download_checkpoint_from_s3()

        # Step 2: Extract
        checkpoint_path = extract_checkpoint(tar_path)

        # Step 3: Load and re-save with config (will delete tar_path to free space)
        complete_path = load_and_resave_checkpoint(checkpoint_path, tar_path)

        # Step 4: Verify
        verify_checkpoint(complete_path)

        # Step 5: Upload
        s3_url = upload_to_s3(complete_path)

        print("\n" + "="*70)
        print("SUCCESS!")
        print("="*70)
        print(f"\nComplete checkpoint available at:")
        print(f"  {s3_url}")
        print(f"\nTo use in training:")
        print(f"  checkpoint = torch.load('dinov3_teacher_complete.pt')")
        print(f"  config = checkpoint['model_config']")
        print(f"  model = DINOv3TeacherModel(**config)")
        print(f"  model.load_state_dict(checkpoint['model'])")
        print("="*70)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
